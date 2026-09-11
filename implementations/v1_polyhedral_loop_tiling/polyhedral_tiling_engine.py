#!/usr/bin/env python3
"""
 =============================================================================
 Polyhedral Loop Nest Tiling, Affine Transformations & TVM-TIR Emitter
 Project: NPU Optimization Suite (Polyhedral Loop Tiling component)
 Author: Yagnesh Kumar Koduru (Esthien Labs)
 Domain: Polyhedral Compilation, Loop Nest Optimization, On-Chip SRAM Locality

 Data type model (matches the project README specification):
   A, B operands: INT8  (1 byte per element)
   C accumulator: INT32 (4 bytes per element)
 ============================================================================= 
"""

import numpy as np
import matplotlib.pyplot as plt

class PolyhedralTilingEngine:
    """
    Formulates and solves polyhedral iteration space loop tiling under
    on-chip L1 SRAM capacity constraints.
    """
    # Byte widths per matrix element (INT8 A/B operands, INT32 C accumulator)
    BYTES_A = 1
    BYTES_B = 1
    BYTES_C = 4

    def __init__(self, sram_capacity_kb=64):
        self.sram_bytes = sram_capacity_kb * 1024

    def tile_footprint_bytes(self, Ti, Tj, Tk):
        """SRAM footprint of the A, B and C tiles under the mixed-precision model."""
        return (Ti * Tk * self.BYTES_A
                + Tk * Tj * self.BYTES_B
                + Ti * Tj * self.BYTES_C)

    def find_optimal_tile_sizes(self, M=1024, N=1024, K=1024):
        """
        Maximizes operational intensity:
          Objective: Maximize (2 * Ti * Tj * Tk) / (Ti*Tk + Tk*Tj + Ti*Tj)
          Constraint: (Ti*Tk*b_A + Tk*Tj*b_B + Ti*Tj*b_C) <= sram_bytes
          with b_A = b_B = 1 (INT8) and b_C = 4 (INT32 accumulator).
        """
        best_intensity = 0.0
        best_tiles = (16, 16, 16)

        # Grid search through valid power-of-two tile factors
        candidates = [16, 32, 64, 128, 256]
        for Ti in candidates:
            if Ti > M: continue
            for Tj in candidates:
                if Tj > N: continue
                for Tk in candidates:
                    if Tk > K: continue
                    footprint = self.tile_footprint_bytes(Ti, Tj, Tk)
                    if footprint <= self.sram_bytes:
                        # Arithmetic intensity (documented objective, element counts)
                        intensity = (2.0 * Ti * Tj * Tk) / (Ti * Tk + Tk * Tj + Ti * Tj)
                        if intensity > best_intensity:
                            best_intensity = intensity
                            best_tiles = (Ti, Tj, Tk)
        return best_tiles, best_intensity

    def simulate_tiling_benefits(self, M=1024, N=1024, K=1024):
        Ti, Tj, Tk = self.find_optimal_tile_sizes(M, N, K)[0]

        # Un-tiled DRAM traffic: A and B are re-fetched for every (i, j) pair
        # (naive cache thrashing); C is written once as INT32.
        untiled_dram_bytes = (M * N * K * (self.BYTES_A + self.BYTES_B)
                              + M * N * self.BYTES_C)
        # Polyhedrally tiled DRAM traffic: A tile rows are re-read once per
        # j-block, B tile columns once per i-block; C still written once.
        tiled_dram_bytes = (M * N * K * (1.0 / Ti + 1.0 / Tj)
                            + M * N * self.BYTES_C)

        dram_traffic_reduction = untiled_dram_bytes / tiled_dram_bytes
        # Fraction of element loads served from on-chip SRAM instead of DRAM
        l1_hit_rate = 1.0 - (tiled_dram_bytes / untiled_dram_bytes)
        
        return {
            "matrix_dims": (M, N, K),
            "optimal_tiles": (Ti, Tj, Tk),
            "footprint_kb": self.tile_footprint_bytes(Ti, Tj, Tk) / 1024.0,
            "untiled_gb": untiled_dram_bytes / 1e9,
            "tiled_gb": tiled_dram_bytes / 1e9,
            "traffic_reduction_x": dram_traffic_reduction,
            "l1_hit_rate_pct": l1_hit_rate * 100.0
        }

    def emit_tvm_tir_schedule(self, Ti, Tj, Tk):
        return f"""// TVM Tensor Intermediate Representation (TIR) Polyhedral Schedule
// Optimized for Esthien NPU Matrix Core with L1 Double-Buffer Pinning
@T.prim_func
def matmul_polyhedral_tiled(
    A: T.Buffer((1024, 1024), "int8"),
    B: T.Buffer((1024, 1024), "int8"),
    C: T.Buffer((1024, 1024), "int32")
):
    with T.block("root"):
        for i_outer, j_outer in T.grid(1024 // {Ti}, 1024 // {Tj}):
            A_local = T.alloc_buffer(({Ti}, {Tk}), "int8", scope="shared.l1")
            B_local = T.alloc_buffer(({Tk}, {Tj}), "int8", scope="shared.l1")
            C_local = T.alloc_buffer(({Ti}, {Tj}), "int32", scope="local.acc")
            for k_outer in range(1024 // {Tk}):
                // Affine double-buffered DMA async copy
                T.async_copy(A_local, A[i_outer*{Ti}:(i_outer+1)*{Ti}, k_outer*{Tk}:(k_outer+1)*{Tk}])
                T.async_copy(B_local, B[k_outer*{Tk}:(k_outer+1)*{Tk}, j_outer*{Tj}:(j_outer+1)*{Tj}])
                T.pipeline_barrier()
                for i_inner, j_inner, k_inner in T.grid({Ti}, {Tj}, {Tk}):
                    with T.block("mac"):
                        C_local[i_inner, j_inner] += T.Cast("int32", A_local[i_inner, k_inner]) * T.Cast("int32", B_local[k_inner, j_inner])
            // Writeback to L2 / DRAM
            T.copy(C[i_outer*{Ti}:(i_outer+1)*{Ti}, j_outer*{Tj}:(j_outer+1)*{Tj}], C_local)
"""

def run_benchmark():
    print("=" * 70)
    print("  NPU OPTIMIZATION SUITE: POLYHEDRAL LOOP TILING ENGINE")
    print("  Author: Yagnesh Kumar Koduru | Esthien Labs")
    print("=" * 70)
    
    engine = PolyhedralTilingEngine(sram_capacity_kb=64)
    res = engine.simulate_tiling_benefits(M=1024, N=1024, K=1024)
    
    print(f"[*] Target Matrix Dimensions : {res['matrix_dims']}")
    print(f"[*] Optimal Polyhedral Tiles  : Ti={res['optimal_tiles'][0]}, Tj={res['optimal_tiles'][1]}, Tk={res['optimal_tiles'][2]}")
    print(f"[*] On-Chip SRAM Footprint   : {res['footprint_kb']:.1f} KB / 64.0 KB (Safe Margin)")
    print(f"[*] Un-tiled DRAM Traffic    : {res['untiled_gb']:.2f} GB")
    print(f"[*] Tiled DRAM Traffic       : {res['tiled_gb']:.2f} GB")
    print(f"[*] Memory Traffic Reduction : {res['traffic_reduction_x']:.2f}x Bandwidth Relief")
    print(f"[*] Effective L1 Hit Rate    : {res['l1_hit_rate_pct']:.2f}%")
    
    print("\n--- [Synthesized TVM-TIR Schedule] ---")
    tir_code = engine.emit_tvm_tir_schedule(*res['optimal_tiles'])
    print(tir_code)
    print("=" * 70)

if __name__ == "__main__":
    run_benchmark()
