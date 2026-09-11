"""Tests for the ballistic bifurcation solver and its SA baseline.

Covers:
  1. The bSBA loop converges to a spin configuration whose Ising energy is
     lower (more negative) than the average energy of random configurations.
  2. The simulated annealing baseline runs and its measured runtime is
     strictly positive.
  3. The reported speedup equals the measured SA runtime over the measured
     bSBA runtime.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conftest import load_module

unified = load_module(
    "unified_pipeline/unified_npu_compiler.py",
    "unified_npu_compiler",
)


def _build_instance(n):
    np.random.seed(42)
    J = np.random.randn(n, n) * 0.5
    J = (J + J.T) / 2.0
    np.fill_diagonal(J, 0.0)
    return J


def test_bsba_energy_beats_random_configurations():
    n = 16
    J = _build_instance(n)
    compiler = unified.UnifiedNPUCompiler()
    result = compiler.run_ballistic_bifurcation_solver(
        num_variables=n, num_steps=150, run_qaoa=False)

    rng = np.random.default_rng(0)
    random_energies = []
    for _ in range(200):
        s = rng.choice([-1.0, 1.0], size=n)
        random_energies.append(-0.5 * float(s @ J @ s))
    mean_random_energy = float(np.mean(random_energies))
    assert result["final_energy"] < mean_random_energy
    assert result["final_energy"] < 0.0


def test_sa_baseline_runs_with_positive_runtime():
    n = 16
    J = _build_instance(n)
    compiler = unified.UnifiedNPUCompiler()
    start = time.perf_counter()
    spins, energy = compiler._simulated_annealing_baseline(J, num_sweeps=50)
    runtime = time.perf_counter() - start
    assert runtime > 0.0
    assert len(spins) == n
    assert energy == pytest.approx(-0.5 * float(spins @ J @ spins))


def test_measured_speedup_is_real_ratio_of_runtimes():
    compiler = unified.UnifiedNPUCompiler()
    result = compiler.run_ballistic_bifurcation_solver(
        num_variables=16, num_steps=100, run_qaoa=False)
    assert result["sba_runtime_ms"] > 0.0
    assert result["sa_runtime_ms"] > 0.0
    assert result["speedup_x"] == pytest.approx(
        result["sa_runtime_ms"] / result["sba_runtime_ms"], rel=1e-9
    )
    # The SA baseline does 50x more work per variable than a single bSBA step
    # sweep on this instance; it must measurably take longer than bSBA.
    assert result["sa_runtime_ms"] > result["sba_runtime_ms"]


def test_bsba_restarts_and_polish_match_sa_quality():
    """Best-of-N restarts plus the 1-opt polish must not lose to the SA
    baseline on the same instance, and the polish must be monotone."""
    n = 16
    J = _build_instance(n)
    compiler = unified.UnifiedNPUCompiler()
    result = compiler.run_ballistic_bifurcation_solver(
        num_variables=n, num_steps=150, num_restarts=8, run_qaoa=False)
    assert result["bsba_polish_energy"] <= result["sa_best_energy"] + 1e-9
    assert result["bsba_polish_energy"] <= result["bsba_raw_energy"] + 1e-12
    assert result["bsba_polish_flips"] >= 0
    assert result["bsba_polish_ms"] >= 0.0
    assert result["bsba_restarts_ms"] > 0.0


def test_qaoa_ratio_against_exhaustive_ground_state():
    compiler = unified.UnifiedNPUCompiler()
    J = _build_instance(16)
    qaoa = compiler._qaoa_approximation_ratio(J, sub_n=10)
    # Independent exhaustive ground state for the 10-spin subinstance
    energies = compiler._ising_energies(J[:10, :10])
    ground = float(np.min(energies))
    assert qaoa["ground_energy"] == pytest.approx(ground)
    # p=1 QAOA is a relaxation: its expectation cannot beat the exact ground state
    assert qaoa["qaoa_energy"] >= ground - 1e-6
    assert 0.0 < qaoa["approx_ratio"] <= 1.0 + 1e-6
