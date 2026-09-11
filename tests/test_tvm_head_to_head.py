"""Tests for deterministic, dependency-light TVM benchmark helpers."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conftest import load_module


benchmark = load_module(
    "baselines/tvm_head_to_head/bench_tvm.py",
    "tvm_head_to_head_benchmark",
)


def test_shared_workload_is_seeded_and_exact(tmp_path):
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()

    _, _, first_reference, first_manifest = benchmark.create_shared_workload(
        np, first_dir, size=8, seed=17
    )
    _, _, second_reference, second_manifest = benchmark.create_shared_workload(
        np, second_dir, size=8, seed=17
    )

    assert np.array_equal(first_reference, second_reference)
    assert first_reference.dtype == np.int32
    assert first_manifest["sha256"] == second_manifest["sha256"]
    assert first_manifest["dimensions"] == {"m": 8, "n": 8, "k": 8}
    assert (first_dir / "inputs" / "manifest.json").is_file()


def test_native_measurement_requires_complete_median_samples():
    payload = {
        "mode": "tiled",
        "warmups": 1,
        "repetitions": 3,
        "samples_seconds": [0.1, 0.2, 0.3],
        "median_seconds": 0.2,
        "result_guard": 7,
    }

    parsed = benchmark.parse_native_measurement(json.dumps(payload), "tiled", 3, 1)

    assert parsed["median_kernel_seconds"] == pytest.approx(0.2)
    assert parsed["result_guard"] == 7


def test_native_measurement_rejects_mismatched_median():
    payload = {
        "mode": "naive",
        "warmups": 0,
        "repetitions": 3,
        "samples_seconds": [0.1, 0.2, 0.3],
        "median_seconds": 0.1,
    }

    with pytest.raises(benchmark.BenchmarkFailure, match="median"):
        benchmark.parse_native_measurement(json.dumps(payload), "naive", 3, 0)


def test_output_validation_rejects_an_incorrect_value():
    reference = np.array([[1, 2], [3, 4]], dtype=np.int32)
    actual = reference.copy()
    actual[1, 0] = 99

    with pytest.raises(benchmark.BenchmarkFailure, match=r"\(1, 0\)"):
        benchmark.assert_exact_output(np, "test", actual, reference)


def test_auto_tuning_is_explicitly_opt_in():
    arguments = benchmark.build_argument_parser().parse_args(["--run-id", "test-run"])

    assert arguments.auto_trials == 0


def test_aggregate_method_combines_rounds_correctly():
    rounds = [
        {
            "methods": {
                "native_tiled": {
                    "status": "complete",
                    "median_kernel_seconds": 0.1,
                    "kernel_seconds_samples": [0.09, 0.11],
                    "output_int32_row_major_sha256": "abc",
                }
            }
        },
        {
            "methods": {
                "native_tiled": {
                    "status": "complete",
                    "median_kernel_seconds": 0.12,
                    "kernel_seconds_samples": [0.11, 0.13],
                    "output_int32_row_major_sha256": "abc",
                }
            }
        },
    ]

    aggregated = benchmark.aggregate_method(rounds, "native_tiled")

    assert aggregated["round_count"] == 2
    assert aggregated["median_kernel_seconds"] == pytest.approx(0.11)
    assert len(aggregated["kernel_seconds_samples"]) == 4


def test_aggregate_method_rejects_inconsistent_hashes():
    rounds = [
        {"methods": {"m": {"status": "complete", "median_kernel_seconds": 0.1, "kernel_seconds_samples": [0.1], "output_int32_row_major_sha256": "x"}}},
        {"methods": {"m": {"status": "complete", "median_kernel_seconds": 0.1, "kernel_seconds_samples": [0.1], "output_int32_row_major_sha256": "y"}}},
    ]

    with pytest.raises(benchmark.BenchmarkFailure, match="hashes"):
        benchmark.aggregate_method(rounds, "m")
