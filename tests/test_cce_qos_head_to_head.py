"""Objective-equivalence tests for the CP-SAT versus bSBA comparison."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("ortools")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baselines.cce_qos_head_to_head.cpsat_vs_bsba import (  # noqa: E402
    OBJECTIVE_SCALE,
    expanded_integer_objective,
    ising_energy,
    quantize_objective,
    run_ortools_cpsat,
)


def test_boolean_expansion_matches_spin_energy_for_all_small_states() -> None:
    raw = np.array(
        [
            [0.0, 0.731, -1.204, 0.319],
            [0.731, 0.0, 0.447, -0.812],
            [-1.204, 0.447, 0.0, 0.225],
            [0.319, -0.812, 0.225, 0.0],
        ],
        dtype=np.float64,
    )
    objective = quantize_objective(raw)
    integer = np.rint(objective * OBJECTIVE_SCALE).astype(np.int64)

    for state in range(1 << objective.shape[0]):
        spins = np.array(
            [1.0 if state & (1 << index) else -1.0 for index in range(objective.shape[0])]
        )
        assert expanded_integer_objective(integer, spins) == pytest.approx(
            ising_energy(objective, spins) * OBJECTIVE_SCALE,
            abs=1e-9,
        )


def test_cp_sat_solution_recomputes_against_the_same_objective() -> None:
    raw = np.array(
        [[0.0, 0.5, -0.25], [0.5, 0.0, 0.75], [-0.25, 0.75, 0.0]],
        dtype=np.float64,
    )
    result = run_ortools_cpsat(quantize_objective(raw), time_limit_s=5.0)

    assert result["status"] == "optimal"
    assert result["energy"] == pytest.approx(
        ising_energy(quantize_objective(raw), np.asarray(result["spins"])),
        abs=1e-10,
    )
