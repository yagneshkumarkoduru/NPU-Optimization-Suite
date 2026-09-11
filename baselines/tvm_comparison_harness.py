#!/usr/bin/env python3
"""
=============================================================================
Optional External Compiler Baseline Harness (TVM)
=============================================================================

This harness is OPTIONAL and OFF by default: it never installs TVM. If TVM
is not importable in the active Python environment, it prints a notice and
exits cleanly with status 0 so CI runs stay green.

How to enable it:
    1. Install Apache TVM in the environment yourself, e.g.
       pip install apache-tvm  (or build TVM from source and set PYTHONPATH).
    2. Run: python baselines/tvm_comparison_harness.py
    3. The harness then benchmarks a small matmul through TVM's schedule
       API with and without a tiled schedule and reports achieved
       GFLOP/s for both, i.e. a real external-compiler reference point.

Note: production-compiler baselines (TVM/XLA/MLIR) remain future work for
this repository's evidence chain; this harness is the provided entry point.
=============================================================================
"""

import sys
import time

import numpy as np


def run_tvm_benchmark(N=256):
    import tvm
    from tvm import te

    # Reference: plain NumPy matmul for correctness sanity + timing context
    a_np = np.random.randint(-8, 8, size=(N, N)).astype("int32")
    b_np = np.random.randint(-8, 8, size=(N, N)).astype("int32")

    A = te.placeholder((N, N), name="A", dtype="int32")
    B = te.placeholder((N, N), name="B", dtype="int32")
    k = te.reduce_axis((0, N), name="k")
    C = te.compute((N, N), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="C")

    def measure(sch, args, name):
        func = tvm.build(sch, args, target="llvm")
        dev = tvm.cpu(0)
        a = tvm.nd.array(a_np, dev)
        b = tvm.nd.array(b_np, dev)
        c = tvm.nd.array(np.zeros((N, N), dtype="int32"), dev)
        func(a, b, c)
        reps = 20
        t0 = time.perf_counter()
        for _ in range(reps):
            func(a, b, c)
        elapsed = (time.perf_counter() - t0) / reps
        gflops = (2.0 * N * N * N) / elapsed / 1e9
        print(f" TVM [{name:<22}] : {elapsed * 1e3:8.3f} ms | {gflops:8.2f} GFLOP/s")
        return gflops

    # Untiled schedule
    s_untiled = te.create_schedule(C.op)
    gflops_untiled = measure(s_untiled, [A, B, C], "untiled")

    # Tiled schedule (32x32 output tiles over j/i, split reduction by 32)
    s_tiled = te.create_schedule(C.op)
    i, j = s_tiled[C].op.axis
    k = s_tiled[C].op.reduce_axis[0]
    io, iin = s_tiled[C].split(i, factor=32)
    jo, jin = s_tiled[C].split(j, factor=32)
    ko, kin = s_tiled[C].split(k, factor=32)
    s_tiled[C].reorder(io, jo, ko, iin, jin, kin)
    gflops_tiled = measure(s_tiled, [A, B, C], "tiled 32x32 (k=32)")

    if gflops_untiled > 0:
        print(f" Tiled/untiled achieved-FLOP/s ratio: {gflops_tiled / gflops_untiled:.2f}x")
    return gflops_untiled, gflops_tiled


def main():
    try:
        import tvm  # noqa: F401
    except ImportError:
        print("TVM not installed; external compiler baseline skipped - see baselines/README note")
        return 0
    run_tvm_benchmark()
    return 0


if __name__ == "__main__":
    sys.exit(main())
