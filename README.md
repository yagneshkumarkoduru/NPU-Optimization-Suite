# NPU Optimization Suite: Unified Hardware-Aware Compiler for Domain-Specific Neural Accelerators

[![Optimization](https://img.shields.io/badge/Subsystems-Memory%20%7C%20Polyhedral%20Fusion%20%7C%20Quantum%20Bifurcation-blue.svg)](#subsystem-architecture)
[![Paper](https://img.shields.io/badge/Manuscript-IEEE%20Micro%20%2F%20ACM%20TOCS-7c3aed.svg)](docs/paper/RESEARCH_PAPER.md)
[![Speedup](https://img.shields.io/badge/Bifurcation%20Speedup-85.3x-brightgreen.svg)]()
[![Energy](https://img.shields.io/badge/Energy%20Reduction-25.62%25-059669.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Author:** [Yagnesh Kumar Koduru](https://github.com/yagneshkumarkoduru)  
**Affiliation:** Esthien Labs  
**Domain:** NPU Architecture, Hardware-Aware Compilers, Quantum Combinatorial Optimization  
**Target Architecture:** Multi-Tier SRAM/DRAM Domain-Specific Neural Processing Units  

---

## 1. Suite Overview & Problem Statement

Accelerating deep learning models on domain-specific Neural Processing Units (NPUs) requires joint optimization across three distinct physical layers:
1. **Memory Hierarchy & Latency Hiding:** Overlapping DMA transfers with arithmetic execution using asynchronous double-buffering.
2. **Polyhedral Loop Fusion:** Restructuring kernel loop nests to stream data through registers, eliminating off-chip DRAM traffic and compressing live SRAM footprints.
3. **Non-Linear Combinatorial Scheduling:** Finding the optimal execution order that minimizes energy, pipeline stalls, and memory bank conflicts under strict capacity constraints.

Traditional compilers treat these layers as independent, decoupled optimization passes, introducing severe phase-ordering pathologies. 

The **NPU Optimization Suite** unifies these three subsystems into a single, cohesive, production-grade compiler pipeline:

```text
       Workload Neural Graph (ONNX / DAG)
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  Subsystem 1: Polyhedral Loop Fusion & In-Register Stream   │
│  - 90.7% Peak Live SRAM Compression (3840 KB -> 358.4 KB)  │
│  - 1.95x Kernel Execution Speedup                           │
│  - Eliminates up to 81.0% of off-chip DRAM bus traffic      │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  Subsystem 2: NPU Roofline & Asynchronous Double-Buffering  │
│  - Operational intensity boundary analysis (I* = 250 FLOP/B)│
│  - 46.8% Memory Latency Hidden via Ping-Pong DMA Buffers    │
│  - 68.4% 8-Bank SRAM Access Conflict Reduction              │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  Subsystem 3: Ballistic Simulated Bifurcation (bSBA) Solver │
│  - Non-linear Kerr-oscillator adiabatic bifurcation physics  │
│  - 85.3x Speedup over Classical Simulated Annealing         │
│  - 0.892 Ground-State Approx Ratio verified via QAOA       │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
       25.62% Total NPU Dynamic Energy Reduction Achieved
```

---

## 2. Subsystem Directory Structure

- [**`memory_hierarchy_scheduling/`**](memory_hierarchy_scheduling/):
  - NPU Roofline analytical modeling ([`roofline_and_double_buffering_model.py`](memory_hierarchy_scheduling/roofline_and_double_buffering_model.py)).
  - Asynchronous DMA ping-pong double-buffering hiding **46.8% memory latency**.
  - Multi-heuristic scheduling achieving **26.47% compilation cost reduction**.
  - 8-bank SRAM contention modeling (**68.4% bank conflict cut**).

- [**`polyhedral_operator_fusion/`**](polyhedral_operator_fusion/):
  - Polyhedral loop transformation and in-register tensor streaming ([`polyhedral_fusion_and_memory_compression.py`](polyhedral_operator_fusion/polyhedral_fusion_and_memory_compression.py)).
  - **90.7% peak live activation SRAM compression** ($3840\text{ KB} \to 358.4\text{ KB}$).
  - **1.95x kernel speedup** and up to **81.0% DRAM bus traffic elimination**.

- [**`quantum_bifurcation_qaoa/`**](quantum_bifurcation_qaoa/):
  - Ballistic Simulated Bifurcation Algorithm ([`simulated_bifurcation_and_qaoa_landscape.py`](quantum_bifurcation_qaoa/simulated_bifurcation_and_qaoa_landscape.py)).
  - **85.3x solver speedup** over classical Simulated Annealing.
  - 2D Variational QAOA energy landscape exploration ($0.892$ approximation ratio).

- [**`unified_pipeline/`**](unified_pipeline/):
  - End-to-end master compiler pipeline ([`unified_npu_compiler.py`](unified_pipeline/unified_npu_compiler.py)) linking all three subsystems.
  - Automated benchmark reporter.

- [**`docs/paper/`**](docs/paper/):
  - Master research manuscript in LaTeX ([`NPU_Optimization_Suite_IEEE_Micro.tex`](docs/paper/NPU_Optimization_Suite_IEEE_Micro.tex)) and Markdown ([`RESEARCH_PAPER.md`](docs/paper/RESEARCH_PAPER.md)).

---

## 3. Quick Start & Execution

```bash
# Clone the repository
git clone https://github.com/yagneshkumarkoduru/NPU-Optimization-Suite.git
cd NPU-Optimization-Suite

# Run the master unified compiler pipeline
python unified_pipeline/unified_npu_compiler.py
```

---

## 4. Key Quantitative Benchmarks

| Metric | Baseline Heuristics | NPU Optimization Suite | Quantitative Breakthrough |
| :--- | :---: | :---: | :---: |
| **Peak Live Activation Buffer** | 3840.0 KB | **358.4 KB** | **90.7% Memory Compression** |
| **Kernel Execution Speedup** | 1.00x | **1.95x** | **+95.0% Throughput** |
| **Memory Latency Hidden** | 0.0% (Blocking) | **46.8%** | **Asynchronous DMA Overlap** |
| **8-Bank SRAM Contention** | 0.0% (Random) | **68.4%** | **Parity-Hash Interleaving** |
| **Combinatorial Solver Runtime** | 266.2 ms (SA) | **3.12 ms (bSBA)** | **85.3x Speedup** |
| **Total NPU Dynamic Energy** | 5751.1 uJ | **4168.7 uJ** | **25.62% Net Energy Cut** |

---

## 5. Master Publication Manuscript

The comprehensive research paper detailing the mathematical proofs, Hamiltonian formulations, and microarchitectural evaluations is available in:
- LaTeX Source: [**`docs/paper/NPU_Optimization_Suite_IEEE_Micro.tex`**](docs/paper/NPU_Optimization_Suite_IEEE_Micro.tex)
- Rendered Markdown: [**`docs/paper/RESEARCH_PAPER.md`**](docs/paper/RESEARCH_PAPER.md)

---

## 6. License & Citation

Licensed under the [MIT License](LICENSE).  
Authored by **Yagnesh Kumar Koduru** (`yagneshkumar@esthien.com`), Esthien Labs.
