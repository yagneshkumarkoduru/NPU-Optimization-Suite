"""Tests for the in-repo classical tiling baselines (baselines/).

Covers:
  1. The naive fixed-grid baseline moves strictly more DRAM traffic than the
     polyhedral engine on the same workload and byte model.
  2. The greedy power-of-two baseline's tile footprint fits the SRAM budget
     (80% of capacity, as configured).
  3. The engine's DRAM-traffic reduction is at least the naive baseline's.
  4. Every reported number is positive and finite.
"""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conftest import load_module

baselines = load_module(
    "baselines/classical_tiling_baselines.py",
    "classical_tiling_baselines",
)


@pytest.fixture(scope="module")
def table():
    return baselines.build_comparison_table(M=1024, N=1024, K=1024, sram_capacity_kb=64)


def _by_name(rows, fragment):
    matches = [r for r in rows if fragment in r["name"]]
    assert len(matches) == 1
    return matches[0]


def test_all_numbers_positive_and_finite(table):
    for row in table:
        assert row["footprint_kb"] > 0.0 and math.isfinite(row["footprint_kb"])
        assert row["dram_gb"] > 0.0 and math.isfinite(row["dram_gb"])
        assert row["reduction_vs_no_tiling_x"] > 0.0 and math.isfinite(
            row["reduction_vs_no_tiling_x"])


def test_naive_baseline_traffic_exceeds_engine_traffic(table):
    naive = _by_name(table, "naive fixed grid")
    engine = _by_name(table, "polyhedral engine")
    assert naive["dram_gb"] > engine["dram_gb"]


def test_greedy_baseline_footprint_within_sram_budget(table):
    greedy = _by_name(table, "greedy pow2")
    # The greedy baseline is configured for 80% of a 64 KB SRAM
    assert greedy["footprint_kb"] <= 0.8 * 64.0 + 1e-9


def test_engine_reduction_at_least_naive_reduction(table):
    naive = _by_name(table, "naive fixed grid")
    engine = _by_name(table, "polyhedral engine")
    assert engine["reduction_vs_no_tiling_x"] >= naive["reduction_vs_no_tiling_x"]


def test_no_tiling_baseline_matches_engine_untiled_model(table):
    """The no-tiling row must equal the engine's own un-tiled traffic model."""
    engine = baselines.PolyEngine.PolyhedralTilingEngine(sram_capacity_kb=64)
    res = engine.simulate_tiling_benefits(M=1024, N=1024, K=1024)
    no_tiling = _by_name(table, "no-tiling")
    assert no_tiling["dram_gb"] == pytest.approx(res["untiled_gb"], rel=1e-12)


def test_tvm_harness_exits_cleanly_without_tvm():
    """The optional TVM harness must exit 0 when TVM is absent."""
    import subprocess
    venv_python = sys.executable
    harness = baselines.REPO_ROOT + "/baselines/tvm_comparison_harness.py"
    proc = subprocess.run(
        [venv_python, harness],
        capture_output=True, text=True, timeout=120,
        cwd=baselines.REPO_ROOT,
    )
    assert proc.returncode == 0
    combined = (proc.stdout + proc.stderr).lower()
    try:
        import tvm  # noqa: F401
        tvm_available = True
    except ImportError:
        tvm_available = False
    if not tvm_available:
        assert "tvm not installed" in combined
