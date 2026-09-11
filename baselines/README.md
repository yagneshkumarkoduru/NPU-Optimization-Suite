# In-Repo Classical Tiling Baselines

Real, computed classical reference points for the polyhedral tiling engine,
using the exact same DRAM-traffic / SRAM-footprint cost model as
`implementations/v1_polyhedral_loop_tiling/polyhedral_tiling_engine.py`
(INT8 A/B operands at 1 B/element, INT32 C accumulator at 4 B/element).

## Files

- `classical_tiling_baselines.py`: three in-repo baselines on the 1024x1024x1024
  INT8 matmul workload: no-tiling (full-tensor traffic), naive fixed (32,32,32)
  grid tiling, and greedy power-of-two blocking within 80% of the 64 KB SRAM.
  Run: `python baselines/classical_tiling_baselines.py`
- `tvm_comparison_harness.py`: OPTIONAL external-compiler harness. It does not
  install TVM. If TVM is missing it prints
  `TVM not installed; external compiler baseline skipped - see baselines/README note`
  and exits 0. To enable: install Apache TVM yourself in the environment
  (e.g. `pip install apache-tvm`) and rerun the harness.

## Status

Comparison numbers quoted in `EVIDENCE.md` and the papers are against these
in-repo classical tiling baselines. External production-compiler baselines
are now measured via the deterministic, correctness-gated head-to-head harness
in `tvm_head_to_head/`. The harness enforces identical INT8 inputs, exact
INT32 reference matching, 64-bit native compilation, round-robin ordering to
mitigate measurement bias, full provenance (git SHA, compiler flags, binary
hash, thread environment), and median-of-medians reporting. See
`bench_tvm.py --help` and the artifact `result.json` schema for details.
The old `tvm_comparison_harness.py` remains for optional quick sanity checks.
