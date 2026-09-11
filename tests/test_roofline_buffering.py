"""Tests for the roofline & double-buffering engines.

Covers:
  1. The roofline knee equals peak_tflops / dram_bw (ridge point formula).
  2. The double-buffering speedup factor stays within its theoretical
     1.0x - 2.0x bound (perfect overlap can at most eliminate the smaller
     of the two per-tile times).
  3. Latency-hiding percentages fall within [0, 100].
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conftest import load_module

roofline_module = load_module(
    "implementations/v2_roofline_dma_double_buffering/roofline_double_buffering_engine.py",
    "roofline_double_buffering_engine",
)
memory_model = load_module(
    "memory_hierarchy_scheduling/roofline_and_double_buffering_model.py",
    "roofline_and_double_buffering_model",
)
RooflineDoubleBufferingEngine = roofline_module.RooflineDoubleBufferingEngine
DoubleBufferingSimulator = memory_model.DoubleBufferingSimulator
NPURooflineEngine = memory_model.NPURooflineEngine


def test_roofline_knee_formula():
    engine = RooflineDoubleBufferingEngine(peak_tflops=16.0, dram_bw_gbs=64.0)
    assert engine.ridge_dram == pytest.approx(16.0 * 1000.0 / 64.0)
    assert engine.ridge_dram == pytest.approx(250.0)

    analytical = NPURooflineEngine(peak_ops_tflops=16.0, dram_bw_gb_s=64.0)
    assert analytical.knee_intensity == pytest.approx(
        analytical.peak_ops / analytical.dram_bw
    )
    assert analytical.knee_intensity == pytest.approx(250.0)


def test_double_buffer_speedup_within_theoretical_bound():
    engine = RooflineDoubleBufferingEngine(peak_tflops=16.0, dram_bw_gbs=64.0)
    sim = engine.simulate_double_buffering(num_tiles=128, tile_flops=2.5e8, tile_bytes=1.2e6)
    assert 1.0 <= sim["speedup_factor"] <= 2.0
    assert 0.0 <= sim["stall_eliminated_pct"] <= 100.0
    assert 0.0 <= sim["pe_utilization_pct"] <= 100.0
    # Pipelined execution can never be slower than synchronous execution
    assert sim["t_async_total_ms"] < sim["t_sync_total_ms"]


def test_double_buffer_speedup_scales_with_dma_compute_ratio():
    engine = RooflineDoubleBufferingEngine(peak_tflops=16.0, dram_bw_gbs=64.0)
    # When compute and DMA times are balanced, speedup approaches 2.0x
    balanced = engine.simulate_double_buffering(num_tiles=64, tile_flops=1.0e9, tile_bytes=4.0e6)
    # When one stage dominates, speedup approaches 1.0x
    skewed = engine.simulate_double_buffering(num_tiles=64, tile_flops=1.0e6, tile_bytes=1.0e9)
    assert balanced["speedup_factor"] > skewed["speedup_factor"]
    assert skewed["speedup_factor"] < 1.2


def test_memory_model_latency_hiding_percentage_range():
    simulator = DoubleBufferingSimulator(dma_bandwidth_gb_s=64.0, pe_compute_throughput_gflops=16000.0)
    _, _, efficiencies = simulator.evaluate_latency_hiding(np.linspace(8.0, 256.0, 10))
    assert np.all(efficiencies >= 0.0)
    assert np.all(efficiencies <= 100.0)
    # Configuration matched to the standalone engine benchmark yields 45.10%
    assert float(np.mean(efficiencies)) == pytest.approx(45.10, abs=0.05)


def test_memory_model_bank_conflict_reduction_computed():
    reduction, uncoord_potential, coord_potential = (
        memory_model.SRAMBankContentionModel.compute_conflict_reduction()
    )
    assert uncoord_potential > coord_potential > 0.0
    assert 0.0 <= reduction <= 100.0
