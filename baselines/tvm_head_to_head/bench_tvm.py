#!/usr/bin/env python3
"""Run a reproducible native-C versus TVM CPU GEMM comparison.

Every measured method consumes the same generated INT8 inputs and must exactly
match the same INT32 reference. Timings include only the kernel invocation,
not generation, I/O, compilation, or TVM tuning. The native implementation is
required to be a 64-bit Windows executable so it can be fairly compared with
the 64-bit Python and TVM process.

The analytical tile engine selects the documented (64, 128, 128) tile. This
script measures a native implementation and external TVM schedules using that
workload; it does not turn the analytical model into a production compiler.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
NATIVE_SOURCE = HERE / "gemm_bench.c"
ARTIFACTS_ROOT = HERE / "artifacts"
ENGINE_TILE = (64, 128, 128)
EXPECTED_PE_MACHINE = 0x8664
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


class BenchmarkFailure(RuntimeError):
    """Raised when the benchmark cannot produce valid, comparable evidence."""


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be a finite value greater than zero")
    return parsed


def single_thread(value: str) -> int:
    parsed = positive_int(value)
    if parsed != 1:
        raise argparse.ArgumentTypeError(
            "only one thread is supported because the native kernels are serial"
        )
    return parsed


def run_id(value: str) -> str:
    if not RUN_ID_PATTERN.fullmatch(value):
        raise argparse.ArgumentTypeError(
            "must contain only letters, numbers, periods, underscores, and hyphens"
        )
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def command_output(command: list[str], cwd: Path) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return "unavailable"
    return (completed.stdout or completed.stderr).strip()


def git_revision() -> str:
    return command_output(["git", "rev-parse", "HEAD"], REPO_ROOT).splitlines()[0]


def configure_single_thread_environment() -> dict[str, str]:
    settings = {
        "TVM_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    for name, value in settings.items():
        os.environ[name] = value
    return settings


def create_artifact_directory(identifier: str) -> Path:
    artifact_dir = ARTIFACTS_ROOT / identifier
    if artifact_dir.exists():
        raise BenchmarkFailure(
            f"artifact directory already exists: {artifact_dir}. Choose a new --run-id."
        )
    artifact_dir.mkdir(parents=True)
    return artifact_dir


def create_shared_workload(
    np: Any,
    artifact_dir: Path,
    size: int,
    seed: int,
) -> tuple[Any, Any, Any, dict[str, Any]]:
    """Generate the sole input pair and exact reference used by every method."""
    inputs_dir = artifact_dir / "inputs"
    inputs_dir.mkdir()

    generator = np.random.default_rng(seed)
    a_np = generator.integers(-16, 17, size=(size, size), dtype=np.int8)
    b_np = generator.integers(-16, 17, size=(size, size), dtype=np.int8)
    reference = a_np.astype(np.int32) @ b_np.astype(np.int32)

    a_path = inputs_dir / "A.bin"
    b_path = inputs_dir / "B.bin"
    reference_path = inputs_dir / "C_ref.npy"
    a_np.tofile(a_path)
    b_np.tofile(b_path)
    np.save(reference_path, reference)

    manifest = {
        "rng": "numpy.random.PCG64",
        "seed": seed,
        "dimensions": {"m": size, "n": size, "k": size},
        "input_dtype": "int8",
        "accumulator_dtype": "int32",
        "input_value_range_inclusive": [-16, 16],
        "files": {
            "a": a_path.name,
            "b": b_path.name,
            "reference": reference_path.name,
        },
        "sha256": {
            "a": sha256_file(a_path),
            "b": sha256_file(b_path),
            "reference_npy": sha256_file(reference_path),
            "reference_int32_row_major": sha256_bytes(reference.tobytes(order="C")),
        },
    }
    write_json(inputs_dir / "manifest.json", manifest)
    return a_np, b_np, reference, manifest


def assert_exact_output(np: Any, label: str, actual: Any, reference: Any) -> str:
    if actual.shape != reference.shape:
        raise BenchmarkFailure(
            f"{label} produced shape {actual.shape}, expected {reference.shape}"
        )
    if actual.dtype != reference.dtype:
        raise BenchmarkFailure(
            f"{label} produced dtype {actual.dtype}, expected {reference.dtype}"
        )
    if not np.array_equal(actual, reference):
        mismatch = np.argwhere(actual != reference)[0]
        row, column = (int(mismatch[0]), int(mismatch[1]))
        raise BenchmarkFailure(
            f"{label} output mismatch at ({row}, {column}): "
            f"got {int(actual[row, column])}, expected {int(reference[row, column])}"
        )
    return sha256_bytes(actual.tobytes(order="C"))


def resolve_compiler(compiler: str) -> str:
    resolved = shutil.which(compiler)
    if resolved is None:
        candidate = Path(compiler)
        if candidate.is_file():
            resolved = str(candidate)
    if resolved is None:
        raise BenchmarkFailure(
            f"required 64-bit native compiler was not found: {compiler}. "
            "Set --compiler or add x86_64-w64-mingw32-gcc to PATH."
        )
    return resolved


def pe_machine(binary: Path) -> int:
    with binary.open("rb") as source:
        header = source.read(512)
    if header[:2] != b"MZ" or len(header) < 64:
        raise BenchmarkFailure(f"native binary is not a PE executable: {binary}")
    pe_offset = int.from_bytes(header[60:64], byteorder="little")
    if pe_offset + 6 > len(header):
        with binary.open("rb") as source:
            source.seek(pe_offset)
            header = source.read(6)
        if len(header) != 6:
            raise BenchmarkFailure(f"native binary has a truncated PE header: {binary}")
        signature, machine = header[:4], int.from_bytes(header[4:6], byteorder="little")
    else:
        signature = header[pe_offset:pe_offset + 4]
        machine = int.from_bytes(header[pe_offset + 4:pe_offset + 6], byteorder="little")
    if signature != b"PE\x00\x00":
        raise BenchmarkFailure(f"native binary has an invalid PE signature: {binary}")
    return machine


def compile_native_kernel(artifact_dir: Path, compiler: str) -> tuple[Path, dict[str, Any]]:
    binary_dir = artifact_dir / "native"
    binary_dir.mkdir()
    binary = binary_dir / "gemm_bench_x64.exe"
    compiler_path = resolve_compiler(compiler)
    flags = [
        "-O3",
        "-std=c11",
        "-Wall",
        "-Wextra",
        "-Werror",
    ]
    command = [
        compiler_path,
        *flags,
        "-o",
        str(binary),
        str(NATIVE_SOURCE),
    ]
    completed = subprocess.run(
        command,
        cwd=HERE,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise BenchmarkFailure(
            "native compilation failed:\n"
            + (completed.stdout + completed.stderr).strip()
        )

    machine = pe_machine(binary)
    if machine != EXPECTED_PE_MACHINE:
        raise BenchmarkFailure(
            f"native binary has PE machine 0x{machine:04x}, expected x86-64 "
            f"0x{EXPECTED_PE_MACHINE:04x}"
        )

    version = command_output([compiler_path, "--version"], HERE).splitlines()
    metadata = {
        "compiler_path": compiler_path,
        "compiler_version": version[0] if version else "unavailable",
        "flags": flags,
        "binary": str(binary.relative_to(artifact_dir)),
        "binary_sha256": sha256_file(binary),
        "pe_machine": "x86_64",
        "source_sha256": sha256_file(NATIVE_SOURCE),
    }
    return binary, metadata


def parse_native_measurement(
    stdout: str,
    expected_mode: str,
    expected_repetitions: int,
    expected_warmups: int,
) -> dict[str, Any]:
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise BenchmarkFailure(f"native benchmark did not emit JSON: {stdout!r}") from exc
    if payload.get("mode") != expected_mode:
        raise BenchmarkFailure(
            f"native benchmark reported mode {payload.get('mode')!r}, expected {expected_mode!r}"
        )
    if payload.get("repetitions") != expected_repetitions:
        raise BenchmarkFailure("native benchmark reported an unexpected repetition count")
    if payload.get("warmups") != expected_warmups:
        raise BenchmarkFailure("native benchmark reported an unexpected warm-up count")
    samples = payload.get("samples_seconds")
    if not isinstance(samples, list) or len(samples) != expected_repetitions:
        raise BenchmarkFailure("native benchmark emitted incomplete timing samples")
    parsed_samples = [float(sample) for sample in samples]
    if not all(math.isfinite(sample) and sample > 0.0 for sample in parsed_samples):
        raise BenchmarkFailure("native benchmark emitted invalid timing samples")
    reported_median = float(payload.get("median_seconds", "nan"))
    computed_median = float(median(parsed_samples))
    if not math.isclose(reported_median, computed_median, rel_tol=1e-8, abs_tol=1e-10):
        raise BenchmarkFailure("native benchmark median does not match its raw timing samples")
    return {
        "kernel_seconds_samples": parsed_samples,
        "median_kernel_seconds": computed_median,
        "result_guard": int(payload.get("result_guard", 0)),
    }


def run_native_mode(
    np: Any,
    binary: Path,
    artifact_dir: Path,
    mode: str,
    size: int,
    input_manifest: dict[str, Any],
    reference: Any,
    warmups: int,
    repetitions: int,
    timeout_seconds: float,
    round_index: int,
) -> dict[str, Any]:
    output_path = artifact_dir / "native" / f"C_{mode}_round_{round_index + 1}.bin"
    inputs_dir = artifact_dir / "inputs"
    command = [
        str(binary),
        mode,
        str(size),
        str(size),
        str(size),
        str(inputs_dir / input_manifest["files"]["a"]),
        str(inputs_dir / input_manifest["files"]["b"]),
        str(output_path),
        str(warmups),
        str(repetitions),
    ]
    completed = subprocess.run(
        command,
        cwd=HERE,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    if completed.returncode != 0:
        raise BenchmarkFailure(
            f"native {mode} benchmark failed:\n"
            + (completed.stdout + completed.stderr).strip()
        )
    measurement = parse_native_measurement(
        completed.stdout.strip(), mode, repetitions, warmups
    )
    expected_bytes = size * size * 4
    if not output_path.is_file() or output_path.stat().st_size != expected_bytes:
        raise BenchmarkFailure(f"native {mode} output is missing or has an unexpected size")
    actual = np.fromfile(output_path, dtype=np.int32).reshape((size, size))
    output_hash = assert_exact_output(np, f"native {mode}", actual, reference)
    if output_hash != input_manifest["sha256"]["reference_int32_row_major"]:
        raise BenchmarkFailure(f"native {mode} output hash differs from the shared reference")
    measurement.update(
        {
            "status": "complete",
            "correctness": "exact_int32_match",
            "output_sha256": sha256_file(output_path),
            "output_int32_row_major_sha256": output_hash,
        }
    )
    return measurement


def import_tvm_dependencies() -> tuple[Any, Any, Any, Any, Any]:
    try:
        import tvm
        import tvm.runtime as runtime
        from tvm import te
        from tvm.s_tir import Schedule
        from tvm.s_tir import meta_schedule
    except ImportError as exc:
        raise BenchmarkFailure(
            "TVM 0.26-compatible Python bindings are required for this benchmark"
        ) from exc
    return tvm, runtime, te, Schedule, meta_schedule


def make_prim_func(te: Any, size: int) -> Any:
    a = te.placeholder((size, size), name="A", dtype="int8")
    b = te.placeholder((size, size), name="B", dtype="int8")
    reduction = te.reduce_axis((0, size), name="k")
    c = te.compute(
        (size, size),
        lambda i, j: te.sum(
            a[i, reduction].astype("int32") * b[reduction, j].astype("int32"),
            axis=[reduction],
        ),
        name="C",
    )
    return te.create_prim_func([a, b, c])


def make_hand_schedule(Schedule: Any, module: Any) -> Any:
    """Apply the analytical engine's tile dimensions through TVM's Schedule API."""
    schedule = Schedule(module)
    block = schedule.get_sblock("C")
    i, j, k = schedule.get_loops(block)
    tile_i, tile_j, tile_k = ENGINE_TILE
    outer_k, inner_k = schedule.split(k, [None, tile_k])
    outer_i, inner_i = schedule.split(i, [None, tile_i])
    outer_j, inner_j = schedule.split(j, [None, tile_j])
    schedule.reorder(outer_i, outer_j, outer_k, inner_i, inner_j, inner_k)
    return schedule


def prepare_tvm_function(
    runtime: Any,
    schedule: Any,
    target: Any,
    a_np: Any,
    b_np: Any,
    reference: Any,
) -> tuple[Any, Any, Any, Any, Any]:
    """Build once outside timing and allocate stable buffers for every round."""
    import tvm

    function = tvm.build(schedule.mod, target=target)
    device = runtime.device("cpu", 0)
    a = runtime.empty(a_np.shape, dtype="int8", device=device)
    b = runtime.empty(b_np.shape, dtype="int8", device=device)
    c = runtime.empty(reference.shape, dtype="int32", device=device)
    a.copyfrom(a_np)
    b.copyfrom(b_np)
    return function, device, a, b, c


def time_tvm_function(
    np: Any,
    function: Any,
    device: Any,
    a: Any,
    b: Any,
    c: Any,
    reference: Any,
    warmups: int,
    repetitions: int,
    label: str,
) -> dict[str, Any]:
    for _ in range(warmups):
        function(a, b, c)
    assert_exact_output(np, f"{label} warm-up", c.numpy(), reference)

    evaluator = function.time_evaluator(
        function.entry_name,
        device,
        number=1,
        repeat=repetitions,
        min_repeat_ms=0,
    )
    raw_result = evaluator(a, b, c)
    samples = [float(sample) for sample in raw_result.results]
    if len(samples) != repetitions or not all(
        math.isfinite(sample) and sample > 0.0 for sample in samples
    ):
        raise BenchmarkFailure(f"{label} emitted incomplete or invalid timing samples")
    output_hash = assert_exact_output(np, label, c.numpy(), reference)
    return {
        "status": "complete",
        "correctness": "exact_int32_match",
        "kernel_seconds_samples": samples,
        "median_kernel_seconds": float(median(samples)),
        "output_int32_row_major_sha256": output_hash,
    }


def run_tvm_auto_schedule(
    meta_schedule: Any,
    module: Any,
    target: Any,
    artifact_dir: Path,
    trials: int,
    builder_timeout_seconds: float,
    seed: int,
) -> tuple[Any, dict[str, Any]]:
    work_dir = artifact_dir / "tvm_auto_database"
    builder = meta_schedule.builder.LocalBuilder(
        max_workers=1,
        timeout_sec=builder_timeout_seconds,
    )
    started = time.perf_counter()
    database = meta_schedule.tune_tir(
        mod=module,
        target=target,
        work_dir=str(work_dir),
        max_trials_global=trials,
        max_trials_per_task=trials,
        num_trials_per_iter=min(4, trials),
        builder=builder,
        runner="local",
        database="json",
        num_tuning_cores=1,
        seed=seed,
    )
    tuning_seconds = time.perf_counter() - started
    records = list(database.get_all_tuning_records())
    valid_records: list[list[float]] = []
    for record in records:
        run_seconds = [float(value) for value in record.run_secs]
        if run_seconds and all(
            math.isfinite(value) and 0.0 < value < 1.0e9 for value in run_seconds
        ):
            valid_records.append(run_seconds)
    if not valid_records:
        raise BenchmarkFailure(
            "TVM auto-tuning produced no valid tuning records. "
            "Do not use this run as an auto-tuned comparison."
        )
    schedule = meta_schedule.tir_integration.compile_tir(database, module, target)
    metadata = {
        "status": "complete",
        "trial_budget": trials,
        "builder_timeout_seconds": builder_timeout_seconds,
        "builder_max_workers": 1,
        "num_tuning_cores": 1,
        "seed": seed,
        "tuning_seconds": tuning_seconds,
        "database": str(work_dir.relative_to(artifact_dir)),
        "tuning_record_count": len(records),
        "valid_tuning_record_count": len(valid_records),
        "valid_record_run_seconds": valid_records,
    }
    return schedule, metadata


def aggregate_method(rounds: list[dict[str, Any]], method_name: str) -> dict[str, Any]:
    """Aggregate equally sized round measurements without hiding raw samples."""
    measurements = [round_result["methods"][method_name] for round_result in rounds]
    if not measurements or any(item.get("status") != "complete" for item in measurements):
        raise BenchmarkFailure(f"{method_name} has incomplete round measurements")
    round_medians = [float(item["median_kernel_seconds"]) for item in measurements]
    all_samples = [
        float(sample)
        for item in measurements
        for sample in item["kernel_seconds_samples"]
    ]
    output_hashes = {
        item["output_int32_row_major_sha256"]
        for item in measurements
    }
    if len(output_hashes) != 1:
        raise BenchmarkFailure(f"{method_name} produced inconsistent output hashes across rounds")
    return {
        "status": "complete",
        "correctness": "exact_int32_match_each_round",
        "round_count": len(measurements),
        "round_median_kernel_seconds": round_medians,
        "kernel_seconds_samples": all_samples,
        "median_kernel_seconds": float(median(round_medians)),
        "all_sample_median_kernel_seconds": float(median(all_samples)),
        "output_int32_row_major_sha256": output_hashes.pop(),
    }


def system_metadata(thread_settings: dict[str, str]) -> dict[str, Any]:
    return {
        "utc_started_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "logical_cpu_count": os.cpu_count(),
        "process_affinity": "not_pinned",
        "thread_environment": thread_settings,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, type=run_id)
    parser.add_argument("--size", type=positive_int, default=1024)
    parser.add_argument("--seed", type=nonnegative_int, default=20260911)
    parser.add_argument("--warmups", type=nonnegative_int, default=1)
    parser.add_argument("--repetitions", type=positive_int, default=7)
    parser.add_argument(
        "--rounds",
        type=positive_int,
        default=3,
        help="number of cyclically ordered method rounds",
    )
    parser.add_argument("--threads", type=single_thread, default=1)
    parser.add_argument("--compiler", default="x86_64-w64-mingw32-gcc")
    parser.add_argument("--native-timeout-seconds", type=positive_float, default=600.0)
    parser.add_argument(
        "--auto-trials",
        type=nonnegative_int,
        default=0,
        help=(
            "optional MetaSchedule trial budget; disabled by default because this host's "
            "LocalBuilder has not produced a valid record"
        ),
    )
    parser.add_argument("--auto-builder-timeout-seconds", type=positive_float, default=300.0)
    return parser


def run_benchmark(args: argparse.Namespace) -> Path:
    if args.threads != 1:
        raise BenchmarkFailure("only one thread is supported by the native kernels")
    thread_settings = configure_single_thread_environment()
    import numpy as np

    artifact_dir = create_artifact_directory(args.run_id)
    result_path = artifact_dir / "result.json"
    result: dict[str, Any] = {
        "status": "running",
        "benchmark": "native_c_vs_tvm_int8_gemm",
        "workload": "square INT8 x INT8 -> INT32 GEMM",
        "engine_tile": list(ENGINE_TILE),
        "measurement_scope": {
            "included": "kernel invocation only",
            "excluded": "input generation, disk I/O, compilation, and auto-tuning",
            "timing_statistic": "median of raw kernel samples",
            "method_order": "cyclic rotation once per round to reduce fixed-order bias",
        },
        "provenance": {
            "git_revision": git_revision(),
            "benchmark_source_sha256": sha256_file(Path(__file__).resolve()),
            "native_source_sha256": sha256_file(NATIVE_SOURCE),
            "system": system_metadata(thread_settings),
        },
        "parameters": {
            "size": args.size,
            "seed": args.seed,
            "warmups": args.warmups,
            "repetitions": args.repetitions,
            "rounds": args.rounds,
            "threads": args.threads,
            "auto_trials": args.auto_trials,
            "auto_builder_timeout_seconds": args.auto_builder_timeout_seconds,
        },
    }
    write_json(artifact_dir / "run_manifest.json", result)

    try:
        a_np, b_np, reference, input_manifest = create_shared_workload(
            np, artifact_dir, args.size, args.seed
        )
        result["inputs"] = input_manifest
        native_binary, native_metadata = compile_native_kernel(artifact_dir, args.compiler)
        result["native_build"] = native_metadata

        tvm, runtime, te, Schedule, meta_schedule = import_tvm_dependencies()
        target = tvm.target.Target({"kind": "llvm", "num-cores": args.threads})
        result["tvm"] = {
            "version": tvm.__version__,
            "module_path": str(Path(tvm.__file__).resolve()),
            "target": str(target),
        }
        module = make_prim_func(te, args.size)

        hand_schedule = make_hand_schedule(Schedule, module)
        prepared_tvm_methods: dict[str, tuple[Any, Any, Any, Any, Any]] = {
            "tvm_hand_schedule": prepare_tvm_function(
                runtime,
                hand_schedule,
                target,
                a_np,
                b_np,
                reference,
            )
        }
        method_names = ["native_naive", "native_tiled", "tvm_hand_schedule"]
        auto_metadata: dict[str, Any] | None = None

        if args.auto_trials > 0:
            print("Preparing TVM MetaSchedule auto-tuning")
            auto_schedule, auto_metadata = run_tvm_auto_schedule(
                meta_schedule,
                module,
                target,
                artifact_dir,
                args.auto_trials,
                args.auto_builder_timeout_seconds,
                args.seed,
            )
            prepared_tvm_methods["tvm_auto_schedule"] = prepare_tvm_function(
                runtime,
                auto_schedule,
                target,
                a_np,
                b_np,
                reference,
            )
            method_names.append("tvm_auto_schedule")

        result["rounds"] = []
        labels = {
            "native_naive": "native naive INT8 GEMM",
            "native_tiled": "native tiled INT8 GEMM",
            "tvm_hand_schedule": "TVM hand schedule using the analytical tile",
            "tvm_auto_schedule": "TVM MetaSchedule auto-tuned schedule",
        }
        for round_index in range(args.rounds):
            rotation = round_index % len(method_names)
            order = method_names[rotation:] + method_names[:rotation]
            round_result: dict[str, Any] = {
                "index": round_index + 1,
                "method_order": order,
                "methods": {},
            }
            for method_name in order:
                print(f"[round {round_index + 1}/{args.rounds}] {labels[method_name]}")
                if method_name == "native_naive":
                    measurement = run_native_mode(
                        np,
                        native_binary,
                        artifact_dir,
                        "naive",
                        args.size,
                        input_manifest,
                        reference,
                        args.warmups,
                        args.repetitions,
                        args.native_timeout_seconds,
                        round_index,
                    )
                elif method_name == "native_tiled":
                    measurement = run_native_mode(
                        np,
                        native_binary,
                        artifact_dir,
                        "tiled",
                        args.size,
                        input_manifest,
                        reference,
                        args.warmups,
                        args.repetitions,
                        args.native_timeout_seconds,
                        round_index,
                    )
                else:
                    function, device, a, b, c = prepared_tvm_methods[method_name]
                    measurement = time_tvm_function(
                        np,
                        function,
                        device,
                        a,
                        b,
                        c,
                        reference,
                        args.warmups,
                        args.repetitions,
                        f"{labels[method_name]} round {round_index + 1}",
                    )
                round_result["methods"][method_name] = measurement
            result["rounds"].append(round_result)

        result["native_naive"] = aggregate_method(result["rounds"], "native_naive")
        result["native_tiled"] = aggregate_method(result["rounds"], "native_tiled")
        result["tvm_hand_schedule"] = aggregate_method(
            result["rounds"], "tvm_hand_schedule"
        )

        if args.auto_trials == 0:
            result["tvm_auto_schedule"] = {
                "status": "not_run",
                "reason": "--auto-trials=0",
            }
        else:
            assert auto_metadata is not None
            result["tvm_auto_schedule"] = auto_metadata
            result["tvm_auto_schedule"].update(
                aggregate_method(result["rounds"], "tvm_auto_schedule")
            )

        native_naive = result["native_naive"]["median_kernel_seconds"]
        native_tiled = result["native_tiled"]["median_kernel_seconds"]
        hand_schedule_seconds = result["tvm_hand_schedule"]["median_kernel_seconds"]
        result["derived_ratios"] = {
            "native_tiled_speedup_over_native_naive": native_naive / native_tiled,
            "tvm_hand_runtime_relative_to_native_tiled": hand_schedule_seconds / native_tiled,
        }
        if result["tvm_auto_schedule"]["status"] == "complete":
            auto_seconds = result["tvm_auto_schedule"]["median_kernel_seconds"]
            result["derived_ratios"][
                "tvm_auto_runtime_relative_to_native_tiled"
            ] = auto_seconds / native_tiled

        result["status"] = "complete"
        write_json(result_path, result)
        return result_path
    except Exception as exc:
        result["status"] = "failed"
        result["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        write_json(result_path, result)
        raise


def main() -> int:
    args = build_argument_parser().parse_args()
    try:
        result_path = run_benchmark(args)
    except (BenchmarkFailure, subprocess.TimeoutExpired) as exc:
        print(f"benchmark failed: {exc}", file=sys.stderr)
        return 1
    print(f"saved -> {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
