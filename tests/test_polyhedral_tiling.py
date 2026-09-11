"""Tests for the polyhedral loop tiling engine (implementations/v1).

Covers:
  1. The chosen tile fits within SRAM under the corrected mixed-precision
     byte model (INT8 A/B operands, INT32 C accumulator).
  2. The intensity objective matches the documented formula.
  3. The reported DRAM traffic reduction matches an independent naive
     no-tiling baseline calculation performed here in the test.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conftest import load_module

tiling_module = load_module(
    "implementations/v1_polyhedral_loop_tiling/polyhedral_tiling_engine.py",
    "polyhedral_tiling_engine",
)
PolyhedralTilingEngine = tiling_module.PolyhedralTilingEngine


def test_chosen_tile_fits_sram_capacity():
    engine = PolyhedralTilingEngine(sram_capacity_kb=64)
    tiles, intensity = engine.find_optimal_tile_sizes(M=1024, N=1024, K=1024)
    Ti, Tj, Tk = tiles
    footprint = engine.tile_footprint_bytes(Ti, Tj, Tk)
    # Corrected byte model: Ti*Tk*1 (INT8 A) + Tk*Tj*1 (INT8 B) + Ti*Tj*4 (INT32 C)
    expected_footprint = Ti * Tk * 1 + Tk * Tj * 1 + Ti * Tj * 4
    assert footprint == expected_footprint
    assert footprint <= 64 * 1024
    # A tile that would exceed SRAM under the corrected model must not be chosen
    assert (Ti, Tj, Tk) != (128, 128, 16), (
        "Tile (128,128,16) has a 68 KB footprint under the INT8/INT32 model "
        "and must exceed the 64 KB SRAM capacity"
    )


def test_intensity_objective_matches_documented_formula():
    engine = PolyhedralTilingEngine(sram_capacity_kb=64)
    Ti, Tj, Tk = 64, 128, 128
    documented_intensity = (2.0 * Ti * Tj * Tk) / (Ti * Tk + Tk * Tj + Ti * Tj)
    # The engine's search objective is the documented element-count formula
    best_tiles, best_intensity = engine.find_optimal_tile_sizes(M=1024, N=1024, K=1024)
    assert best_tiles == (Ti, Tj, Tk)
    assert best_intensity == pytest.approx(documented_intensity, rel=1e-9)
    assert best_intensity == pytest.approx(64.0, rel=1e-9)


def test_dram_traffic_reduction_matches_independent_baseline():
    engine = PolyhedralTilingEngine(sram_capacity_kb=64)
    res = engine.simulate_tiling_benefits(M=1024, N=1024, K=1024)
    Ti, Tj, Tk = res["optimal_tiles"]
    M = N = K = 1024

    # Independent naive no-tiling baseline:
    # A and B re-fetched for every (i, j) pair, C written once (INT32)
    untiled_independent = 2.0 * M * N * K + 4.0 * M * N
    # Independent tiled model: A re-read once per j-block, B once per i-block
    tiled_independent = M * N * K * (1.0 / Ti + 1.0 / Tj) + 4.0 * M * N

    assert res["untiled_gb"] == pytest.approx(untiled_independent / 1e9, rel=1e-9)
    assert res["tiled_gb"] == pytest.approx(tiled_independent / 1e9, rel=1e-9)
    assert res["traffic_reduction_x"] == pytest.approx(
        untiled_independent / tiled_independent, rel=1e-9
    )
    assert res["traffic_reduction_x"] > 10.0, "Tiling must substantially cut DRAM traffic"
    assert 0.0 <= res["l1_hit_rate_pct"] <= 100.0
