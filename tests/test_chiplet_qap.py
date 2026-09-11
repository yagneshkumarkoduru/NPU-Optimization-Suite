"""Tests for the chiplet placement QAP (exhaustive permutation search).

Covers:
  1. The QAP search returns a permutation whose cost is at most the identity
     permutation cost.
  2. The unified pipeline pass and the standalone chiplet engine produce the
      same optimal cost for the same traffic matrix.
"""

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conftest import load_module

v3_module = load_module(
    "implementations/v3_ucie_chiplet_speculative_decoding/chiplet_speculative_compiler.py",
    "chiplet_speculative_compiler_v3",
)
pass_module = load_module(
    "unified_pipeline/chiplet_speculative_compiler_pass.py",
    "chiplet_speculative_compiler_pass",
)

# The shared traffic matrix used by both the standalone engine and the
# unified pipeline pass.
SHARED_TRAFFIC = np.array([
    [0, 320, 25, 10],
    [320, 0, 15, 290],
    [25, 15, 0, 310],
    [10, 290, 310, 0]
])
SHARED_HOPS = np.array([
    [0, 1, 1, 2],
    [1, 0, 2, 1],
    [1, 2, 0, 1],
    [2, 1, 1, 0]
])


def identity_cost(traffic, hops, n=4):
    return float(sum(traffic[u, v] * hops[u, v] for u in range(n) for v in range(n)))


def test_qap_cost_never_exceeds_identity_cost():
    best_perm, best_cost, id_cost, _ = (
        pass_module.ChipletSpeculativeCompiler.solve_placement_qap(SHARED_TRAFFIC, SHARED_HOPS)
    )
    assert best_cost <= id_cost + 1e-9
    assert len(set(best_perm)) == 4  # a valid bijection


def test_qap_optimal_cost_matches_standalone_engine():
    compiler = v3_module.ChipletSpeculativeCompiler(num_chiplets=4, ucie_bw_gbs=64.0)
    qap_result = compiler.optimize_chiplet_placement()
    best_perm, best_cost, _, _ = (
        pass_module.ChipletSpeculativeCompiler.solve_placement_qap(SHARED_TRAFFIC, SHARED_HOPS)
    )
    assert qap_result["opt_d2d_mb_hops"] == int(best_cost)
    assert qap_result["base_d2d_mb_hops"] == 3200
    assert int(best_cost) == 1990
    assert qap_result["traffic_reduction_pct"] == (
        (1.0 - 1990.0 / 3200.0) * 100.0
    )


def test_qap_bruteforce_agrees_on_random_instances():
    rng = np.random.default_rng(123)
    for _ in range(5):
        traffic = rng.integers(0, 50, size=(4, 4)).astype(float)
        traffic = (traffic + traffic.T) / 2.0
        np.fill_diagonal(traffic, 0.0)
        hops = rng.integers(0, 3, size=(4, 4)).astype(float)
        hops = (hops + hops.T) / 2.0
        np.fill_diagonal(hops, 0.0)
        best_perm, best_cost, _, _ = (
            pass_module.ChipletSpeculativeCompiler.solve_placement_qap(traffic, hops)
        )
        # Independent reference: recompute all 24 permutation costs
        all_costs = []
        for perm in itertools.permutations(range(4)):
            cost = sum(traffic[u, v] * hops[perm[u], perm[v]]
                       for u in range(4) for v in range(4))
            all_costs.append(cost)
        assert best_cost == pytest.approx(min(all_costs))
