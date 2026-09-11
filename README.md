# NPU Optimization Suite: Unified Hardware-Aware Compiler for Domain-Specific Neural Accelerators

[![CI](https://github.com/yagneshkumarkoduru/NPU-Optimization-Suite/actions/workflows/ci.yml/badge.svg)](https://github.com/yagneshkumarkoduru/NPU-Optimization-Suite/actions)
[![Target](https://img.shields.io/badge/Target-Heterogeneous%20NPUs%20%26%202.5D%20Chiplets-blue.svg)](#2-implementation-architecture)
[![Polyhedral Tiling](https://img.shields.io/badge/Component-Polyhedral%20Loop%20Tiling-059669.svg)](implementations/v1_polyhedral_loop_tiling/)
[![Roofline DMA](https://img.shields.io/badge/Component-Roofline%20DMA%20Double--Buffering-d97706.svg)](implementations/v2_roofline_dma_double_buffering/)
[![UCIe Chiplet](https://img.shields.io/badge/Component-UCIe%20Chiplet%20%26%20Speculative%20Decoding-512bd4.svg)](implementations/v3_ucie_chiplet_speculative_decoding/)
[![Theory](https://img.shields.io/badge/Theory-Polyhedral%20%26%20Chiplet%20Proofs-0284c7.svg)](docs/POLYHEDRAL_AND_CHIPLET_COMPILER_THEORY.md)
[![Paper](https://img.shields.io/badge/Manuscript-IEEE%20Micro%20%2F%20ACM%20TOCS-7c3aed.svg)](docs/paper/RESEARCH_PAPER.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Author:** [Yagnesh Kumar Koduru](https://github.com/yagneshkumarkoduru)  
**Affiliation:** Esthien Labs  
**Domain:** NPU Architecture, Hardware-Aware Compilers, Polyhedral Geometry, Chiplet Interconnects  
**Target Architecture:** Multi-level SRAM/DRAM Domain-Specific Neural Processing Units & 2.5D/3D Chiplet Systems  

---

## 1. Suite Overview & Problem Statement

Accelerating deep learning models on domain-specific Neural Processing Units (NPUs) requires joint optimization across physical hardware and compilation layers:
1. **Polyhedral Loop Fusion & Tiling:** Restructuring kernel loop nests to stream data through registers, eliminating off-chip DRAM traffic and maximizing operational intensity within L1 scratchpads.
2. **Memory Hierarchy & Latency Hiding:** Overlapping DMA transfers with arithmetic execution using asynchronous ping-pong double-buffering.
3. **Heterogeneous 2.5D/3D Chiplet Interconnects:** Mapping communicating subgraphs onto physical dies via Universal Chiplet Interconnect Express (UCIe) and amortizing memory bandwidth with tree speculative decoding.

Traditional compilers treat these layers as independent, decoupled optimization passes, introducing severe phase-ordering pathologies. 

The **NPU Optimization Suite** unifies these three subsystems into a cohesive, production-grade compiler pipeline:

> **Evidence status (2026-09-10):** All headline numbers in this README are traced to exact output files and live re-runs. See [**`EVIDENCE.md`**](EVIDENCE.md) for the verification log. All metrics are computed from the models in this repository; timing-based figures vary run to run.

```text
        Workload Neural Graph (ONNX / TVM TIR)
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  Polyhedral Loop Tiling & Affine TVM-TIR Emitter            │
│  - 73.29x Off-Chip DRAM Traffic Reduction                   │
│  - 98.64% Effective L1 Cache Hit Rate                       │
│  - Mixed-Precision Tile Footprint (56 KB <= 64 KB)          │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  NPU Roofline & Asynchronous Double-Buffering               │
│  - Operational intensity boundary analysis (I* = 250 FLOP/B)│
│  - 45.10% Memory Latency Hidden via Ping-Pong DMA Buffers    │
│  - 1.82x Throughput Acceleration                            │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  2.5D UCIe Chiplet Interconnect & Speculative Pass          │
│  - Exhaustive QAP Die-to-Die Routing (37.81% Traffic Relief)│
│  - 37.81% Interconnect Energy Cut (0.5 pJ/bit D2D)          │
│  - 1.91x LLM Speculative Tree Verification Acceleration     │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Implementation Architecture

The suite provides three implementation targets detailed in [`docs/IMPLEMENTATION_VERSIONS.md`](docs/IMPLEMENTATION_VERSIONS.md). Complete mathematical and physical proofs are documented in [`docs/POLYHEDRAL_AND_CHIPLET_COMPILER_THEORY.md`](docs/POLYHEDRAL_AND_CHIPLET_COMPILER_THEORY.md).

### 2.1 Implementation Matrix

| Component | Target Substrate | Core Algorithmic Formulation | Primary Memory Scope | Key Performance Breakthrough | Source Code |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Polyhedral Loop Tiling** | On-Chip Matrix Core | Iteration Space Convex Polyhedra, TVM-TIR | L1 Scratchpad ($64\text{ KB}$) | **$73.29\times$ DRAM Traffic Relief**, $98.64\%$ Hit Rate | [`implementations/v1_polyhedral_loop_tiling/`](implementations/v1_polyhedral_loop_tiling/) |
| **Roofline DMA Double-Buffering** | Systolic Array + DMA | Williams Roofline & Asynchronous Ping-Pong | L1 / L2 / DRAM Hierarchy | **$1.82\times$ Speedup**, $45.10\%$ Latency Hidden | [`implementations/v2_roofline_dma_double_buffering/`](implementations/v2_roofline_dma_double_buffering/) |
| **UCIe Chiplet Speculative Decoding** | Multi-Die Heterogeneous | QAP Interconnect Placement & Speculative Pass | UCIe D2D ($64\text{ GB/s}$, $<2\text{ ns}$) | **$1.91\times$ LLM Speedup**, $37.81\%$ Energy Cut | [`implementations/v3_ucie_chiplet_speculative_decoding/`](implementations/v3_ucie_chiplet_speculative_decoding/) |

---

## 3. Subsystem Directory Structure

- [**`implementations/`**](implementations/):
  - **`v1_polyhedral_loop_tiling/`**: Polyhedral loop nest optimizer and TVM-TIR code generator.
  - **`v2_roofline_dma_double_buffering/`**: Williams Roofline analyzer and asynchronous ping-pong DMA simulator.
  - **`v3_ucie_chiplet_speculative_decoding/`**: QAP chiplet placement optimizer and LLM speculative decoding pass.

- [**`memory_hierarchy_scheduling/`**](memory_hierarchy_scheduling/):
  - Multi-heuristic scheduling and 8-bank SRAM contention modeling (**26.47% scheduler cost cut**, **68.48% bank conflict cut** in the scheduling benchmark).

- [**`polyhedral_operator_fusion/`**](polyhedral_operator_fusion/):
  - In-register tensor streaming (**90.7% peak live activation compression**, computed by the fusion simulator).

- [**`quantum_bifurcation_qaoa/`**](quantum_bifurcation_qaoa/):
  - Ballistic Simulated Bifurcation Algorithm with a measured Simulated Annealing baseline. Measured on the 32-variable unified-pipeline instance (timings vary run to run, see [**`EVIDENCE.md`**](EVIDENCE.md)): single-run bSBA (150 steps) takes 1.6-2.1 ms but finds a worse energy (-33.6 vs SA's -42.76); bSBA with best-of-8 restarts (deterministic seed ramp) plus a feasibility-preserving 1-opt Ising-energy polish totals 12.9-14.4 ms and ties the SA baseline's best energy (-42.76) while the 400-sweep SA takes 28.2-41.0 ms, i.e. the same solution quality at 2.2-2.5x less wall time across paired runs. The QAOA depth sweep (p = 1, 2, 3, 2p-parameter multi-start COBYLA vs exhaustive 2^14 ground state) measures approximation ratios 0.433 / 0.595 / 0.694.

- [**`baselines/`**](baselines/):
  - In-repo classical tiling baselines (no-tiling, naive fixed grid, greedy power-of-two) computed with the tiling engine's own INT8/INT32 cost model, plus an optional external TVM comparison harness.

- [**`unified_pipeline/`**](unified_pipeline/):
  - End-to-end compiler pipeline (`unified_npu_compiler.py`) and chiplet speculative pass.

- [**`tests/`**](tests/):
  - Pytest suite covering SRAM feasibility, roofline bounds, QAP optimality, and solver convergence.

- [**`docs/`**](docs/):
  - [`docs/POLYHEDRAL_AND_CHIPLET_COMPILER_THEORY.md`](docs/POLYHEDRAL_AND_CHIPLET_COMPILER_THEORY.md): Mathematical derivations.
  - [`docs/IMPLEMENTATION_VERSIONS.md`](docs/IMPLEMENTATION_VERSIONS.md): Implementation matrix and comparisons.
  - [`docs/paper/RESEARCH_PAPER.md`](docs/paper/RESEARCH_PAPER.md): Full journal publication manuscript.
  - [`docs/paper/NPU_Optimization_Suite_IEEE_Micro.tex`](docs/paper/NPU_Optimization_Suite_IEEE_Micro.tex): Complete LaTeX manuscript.

---

## 4. Reproduction Commands

```bash
# Polyhedral Loop Tiling & TVM-TIR Emitter
python implementations/v1_polyhedral_loop_tiling/polyhedral_tiling_engine.py

# Williams Roofline & Asynchronous Double-Buffering
python implementations/v2_roofline_dma_double_buffering/roofline_double_buffering_engine.py

# 2.5D UCIe Chiplet Interconnect & Speculative Decoding Pass
python implementations/v3_ucie_chiplet_speculative_decoding/chiplet_speculative_compiler.py

# Master Unified Compiler Pipeline
python unified_pipeline/unified_npu_compiler.py

# In-Repo Classical Tiling Baselines (no-tiling / naive grid / greedy pow2 vs engine)
python baselines/classical_tiling_baselines.py

# Optional external TVM comparison harness (exits 0 with a notice if TVM is absent)
python baselines/tvm_comparison_harness.py

# Test suite
python -m pytest -q tests/
```

---

## 5. Key Quantitative Benchmarks

All values below are computed by the code in this repository (verification date 2026-09-10; see [`EVIDENCE.md`](EVIDENCE.md)).

| Metric | Baseline Architecture | NPU Optimization Suite | Quantitative Breakthrough |
| :--- | :---: | :---: | :---: |
| **DRAM Transfer Volume** | 2.15 GB | **0.03 GB** | **73.29x Memory Traffic Reduction** |
| **Effective L1 Hit Rate** | 62.4% | **98.64%** | **Near-Zero Cache Thrashing** |
| **Memory Latency Hidden** | 0.0% (Blocking) | **45.10%** | **Ping-Pong Double-Buffering** |
| **PE Utilization Under Load** | 45.4% | **82.79%** | **Sustained Compute Throughput** |
| **Cross-Die D2D Traffic** | 3,200 MB-hops | **1,990 MB-hops** | **37.81% Inter-Die Bandwidth Relief** |
| **D2D Interconnect Energy** | 12,800 uJ | **7,960 uJ** | **37.81% Interconnect Energy Cut** |
| **LLM Inference Speedup** | 1.00x (Autoregressive) | **1.91x** | **Speculative Tree Verification** |

---

## 6. Author & Citation

**Yagnesh Kumar Koduru**  
*Systems Architect & Compiler Researcher*  
Esthien Labs  
GitHub: [@yagneshkumarkoduru](https://github.com/yagneshkumarkoduru)  
Portfolio: [yagneshkumarkoduru.vercel.app](https://yagneshkumarkoduru.vercel.app/)  

```bibtex
@article{koduru2026npu,
  author = {Koduru, Yagnesh Kumar},
  title = {NPU Optimization Suite: Unified Hardware-Aware Compiler for Domain-Specific Neural Accelerators},
  journal = {IEEE Micro},
  year = {2026}
}
```
