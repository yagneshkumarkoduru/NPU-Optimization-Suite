#!/usr/bin/env python3
"""CCE-QOS head-to-head: OR-Tools CP-SAT versus bSBA on one Ising objective.

Replicates the NPU unified pipeline 32-var instance (seed 42).
Measures final energy and wall time on the same quantized objective.
The comparison is refused if solver outputs cannot be independently recomputed.
"""

from __future__ import annotations

import json
import hashlib
import os
import sys
import time
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

# ortools
from ortools.sat.python import cp_model

# Reuse NPU bSBA / SA code
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from unified_pipeline.unified_npu_compiler import UnifiedNPUCompiler


HERE = Path(__file__).resolve().parent
J_PATH = HERE / "J_32var_seed42.npy"
RESULT_PATH = HERE / "cpsat_vs_bsba_result.json"
OBJECTIVE_SCALE = 1000


def load_or_create_J() -> np.ndarray:
    if J_PATH.exists():
        J = np.load(J_PATH)
    else:
        np.random.seed(42)
        n = 32
        J = np.random.randn(n, n)
        J = 0.5 * (J + J.T)
        J = J - np.diag(np.diag(J))
        J_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.save(J_PATH, J)
    return J


def quantize_objective(J: np.ndarray) -> np.ndarray:
    """Use one explicitly quantized matrix for every solver in the comparison."""
    return np.rint(J * OBJECTIVE_SCALE).astype(np.int64) / OBJECTIVE_SCALE


def ising_energy(J: np.ndarray, spins: np.ndarray) -> float:
    """Return the declared E(s) = -0.5 * s.T @ J @ s objective."""
    return -0.5 * float(spins @ J @ spins)


def expanded_integer_objective(J_int: np.ndarray, spins: np.ndarray) -> int:
    """Evaluate the exact integer Boolean expansion used by CP-SAT."""
    bits = ((np.asarray(spins, dtype=np.int64) + 1) // 2).astype(np.int64)
    pair_term = sum(
        -4 * int(J_int[i, j]) * int(bits[i]) * int(bits[j])
        for i in range(len(bits))
        for j in range(i + 1, len(bits))
    )
    linear_term = sum(
        2 * int(np.sum(J_int[i, :])) * int(bits[i])
        for i in range(len(bits))
    )
    constant_term = -sum(
        int(J_int[i, j])
        for i in range(len(bits))
        for j in range(i + 1, len(bits))
    )
    return pair_term + linear_term + constant_term


def validate_solution(J: np.ndarray, spins: np.ndarray, reported_energy: float) -> None:
    if spins.shape != (J.shape[0],) or not np.all(np.isin(spins, (-1.0, 1.0))):
        raise RuntimeError("solver returned an invalid Ising spin vector")
    recomputed = ising_energy(J, spins)
    if not np.isclose(recomputed, reported_energy, rtol=0.0, atol=1e-10):
        raise RuntimeError(
            f"reported energy {reported_energy} does not match recomputed energy {recomputed}"
        )


def run_ortools_cpsat(J: np.ndarray, time_limit_s: float = 30.0) -> dict[str, Any]:
    n = J.shape[0]
    model = cp_model.CpModel()
    s = [model.NewBoolVar(f"s_{i}") for i in range(n)]

    # Objective: minimize E(s) = -0.5 * s.T @ J @ s, where s = 2*x - 1.
    # With symmetric zero-diagonal J, E = -sum(i<j) J[i,j] * s[i] * s[j].
    # Expanding s[i] * s[j] gives pair coefficient -4*J[i,j] and
    # linear coefficient 2*sum(j != i) J[i,j], plus a constant.
    J_int = np.rint(J * OBJECTIVE_SCALE).astype(np.int64)

    # Linearised quadratic objective (exact for comparison)
    terms: list[Any] = []
    for i in range(n):
        for j in range(i + 1, n):
            prod = model.NewIntVar(0, 1, f"prod_{i}_{j}")
            model.AddMultiplicationEquality(prod, [s[i], s[j]])
            terms.append((-4 * int(J_int[i, j])) * prod)
    for i in range(n):
        linear = 2 * int(np.sum(J_int[i, :]))
        terms.append(linear * s[i])
    constant = -int(sum(J_int[i, j] for i in range(n) for j in range(i + 1, n)))

    model.Minimize(sum(terms) + constant)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.num_search_workers = 8
    solver.parameters.log_search_progress = False

    t0 = time.perf_counter()
    status = solver.Solve(model)
    elapsed = time.perf_counter() - t0

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        spins = np.array([solver.Value(s[i]) * 2 - 1 for i in range(n)], dtype=np.float64)
        energy = ising_energy(J, spins)
        validate_solution(J, spins, energy)
        integer_objective = expanded_integer_objective(J_int, spins)
        if abs(solver.ObjectiveValue() - integer_objective) > 0.5:
            raise RuntimeError("CP-SAT objective does not match recomputed quantized energy")
        return {
            "status": "optimal" if status == cp_model.OPTIMAL else "feasible",
            "energy": energy,
            "spins": spins.tolist(),
            "objective_integer": integer_objective,
            "time_s": elapsed,
            "solver_status": solver.StatusName(status),
        }
    return {
        "status": "infeasible_or_timeout",
        "energy": float("inf"),
        "time_s": elapsed,
        "solver_status": solver.StatusName(status),
    }


def run_bsba_vs_sa(J: np.ndarray, num_restarts: int = 8, num_steps: int = 150) -> dict[str, Any]:
    c = UnifiedNPUCompiler()
    np.random.seed(42)
    bsba_start = time.perf_counter()

    # Single raw trajectory
    single_rng = np.random.default_rng(42)
    single_spins = c._bsba_core(J, num_steps, single_rng)
    single_energy = ising_energy(J, single_spins)
    validate_solution(J, single_spins, single_energy)
    single_ms = (time.perf_counter() - bsba_start) * 1000.0

    # Best-of-restarts + polish
    best_energy = float("inf")
    best_spins = None
    for i in range(num_restarts):
        rng = np.random.default_rng(1000 + i)
        spins = c._bsba_core(J, num_steps, rng)
        energy = ising_energy(J, spins)
        validate_solution(J, spins, energy)
        if energy < best_energy:
            best_energy = energy
            best_spins = spins
    polished_spins, polished_energy, polish_flips = c._ising_local_polish(J, best_spins, max_flips=400)
    validate_solution(J, polished_spins, polished_energy)
    bsba_total_ms = (time.perf_counter() - bsba_start) * 1000.0

    # SA baseline on same instance (timed)
    sa_start = time.perf_counter()
    sa_spins, sa_energy = c._simulated_annealing_baseline(J, num_sweeps=400)
    validate_solution(J, sa_spins, sa_energy)
    sa_time_ms = (time.perf_counter() - sa_start) * 1000.0

    return {
        "bSBA_single_energy": single_energy,
        "bSBA_single_ms": single_ms,
        "bSBA_best_of_restarts_energy": best_energy,
        "bSBA_polished_energy": polished_energy,
        "bSBA_polished_spins": polished_spins.tolist(),
        "bSBA_polish_flips": polish_flips,
        "bSBA_total_ms": bsba_total_ms,
        "SA_energy": sa_energy,
        "SA_spins": sa_spins.tolist(),
        "SA_ms": sa_time_ms,
        "speedup_x": sa_time_ms / bsba_total_ms if bsba_total_ms > 0 else 0.0,
    }


def main() -> int:
    raw_J = load_or_create_J()
    J = quantize_objective(raw_J)
    print("[1/3] ortools CPSAT exact solver (30s limit)")
    cpsat = run_ortools_cpsat(J, time_limit_s=30.0)
    print(f"      CPSAT energy: {cpsat['energy']:.4f} | {cpsat['time_s']:.3f}s | {cpsat['status']}")

    print("[2/3] bSBA (8 restarts + 1-opt polish) vs SA baseline")
    bsba = run_bsba_vs_sa(J)
    print(f"      bSBA polished: {bsba['bSBA_polished_energy']:.4f} | {bsba['bSBA_total_ms']:.1f}ms")
    print(f"      SA: {bsba['SA_energy']:.4f} | {bsba['SA_ms']:.1f}ms")

    if cpsat["status"] == "optimal":
        if not np.isclose(cpsat["energy"], bsba["bSBA_polished_energy"], rtol=0.0, atol=1e-10):
            comparison_status = "NO_QUALITY_TIE"
        else:
            comparison_status = "VALID_QUALITY_TIE"
    else:
        comparison_status = "NO_EXACT_REFERENCE"

    result = {
        "schema_version": 2,
        "instance": "32-var Ising (seed 42, symmetric zero-diagonal)",
        "objective": "E(s) = -0.5 * s.T @ J_quantized @ s; spins in {-1, +1}",
        "objective_scale": OBJECTIVE_SCALE,
        "raw_J_sha256": hashlib.sha256(raw_J.tobytes()).hexdigest(),
        "J_sha256": hashlib.sha256(J.tobytes()).hexdigest(),
        "comparison_status": comparison_status,
        "cpsat": cpsat,
        "bsba_sa": bsba,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"[3/3] saved -> {RESULT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
