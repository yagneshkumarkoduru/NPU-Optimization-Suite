#!/usr/bin/env python3
"""
=============================================================================
Williams Roofline & Ping-Pong Double-Buffering DMA Engine
Project: NPU Optimization Suite (Tier 2 Implementation)
Author: Yagnesh Kumar Koduru (Esthien Labs)
Domain: Memory Hierarchy, Double-Buffering, Latency Hiding, Roofline Analysis
=============================================================================
"""

import os
import sys
import numpy as np

class RooflineDoubleBufferingEngine:
    def __init__(self, peak_tflops=16.0, dram_bw_gbs=64.0, l2_bw_gbs=256.0, l1_bw_gbs=1024.0):
        self.peak_tflops = peak_tflops
        self.dram_bw = dram_bw_gbs
        self.l2_bw = l2_bw_gbs
        self.l1_bw = l1_bw_gbs
        
        # Ridge points (Knee of the Roofline: FLOPs / Byte)
        self.ridge_dram = (peak_tflops * 1e12) / (dram_bw_gbs * 1e9)
        self.ridge_l2 = (peak_tflops * 1e12) / (l2_bw_gbs * 1e9)
        self.ridge_l1 = (peak_tflops * 1e12) / (l1_bw_gbs * 1e9)

    def evaluate_kernel_roofline(self, kernel_name, flops, dram_bytes, l2_bytes):
        intensity_dram = flops / max(1, dram_bytes)
        intensity_l2 = flops / max(1, l2_bytes)
        
        perf_dram_tflops = min(self.peak_tflops, (intensity_dram * self.dram_bw * 1e9) / 1e12)
        perf_l2_tflops = min(self.peak_tflops, (intensity_l2 * self.l2_bw * 1e9) / 1e12)
        
        return {
            "kernel": kernel_name,
            "flops": flops,
            "intensity_dram": intensity_dram,
            "intensity_l2": intensity_l2,
            "perf_dram": perf_dram_tflops,
            "perf_l2": perf_l2_tflops,
            "compute_bound": intensity_dram >= self.ridge_dram
        }

    def simulate_double_buffering(self, num_tiles=64, tile_flops=1.0e9, tile_bytes=2.0e7):
        """
        Simulates execution time comparing:
          1) Synchronous single-buffer: T_total = sum(T_dma + T_compute)
          2) Asynchronous double-buffer: T_total = T_dma_0 + sum(max(T_dma, T_compute)) + T_compute_N
        """
        # Time per tile in milliseconds
        t_compute_ms = (tile_flops / (self.peak_tflops * 1e12)) * 1000.0
        t_dma_ms = (tile_bytes / (self.dram_bw * 1e9)) * 1000.0
        
        # Synchronous execution
        t_sync_ms = num_tiles * (t_compute_ms + t_dma_ms)
        
        # Ping-pong double-buffering execution
        t_async_ms = t_dma_ms + (num_tiles - 1) * max(t_compute_ms, t_dma_ms) + t_compute_ms
        
        speedup = t_sync_ms / t_async_ms
        stall_elimination = (1.0 - (t_async_ms / t_sync_ms)) * 100.0
        effective_utilization = (num_tiles * t_compute_ms) / t_async_ms * 100.0
        
        return {
            "num_tiles": num_tiles,
            "t_compute_tile_ms": t_compute_ms,
            "t_dma_tile_ms": t_dma_ms,
            "t_sync_total_ms": t_sync_ms,
            "t_async_total_ms": t_async_ms,
            "speedup_factor": speedup,
            "stall_eliminated_pct": stall_elimination,
            "pe_utilization_pct": min(100.0, effective_utilization)
        }

def run_benchmark():
    print("=" * 70)
    print("  NPU OPTIMIZATION SUITE: TIER 2 ROOFLINE & DOUBLE-BUFFERING ENGINE")
    print("  Author: Yagnesh Kumar Koduru | Esthien Labs")
    print("=" * 70)
    
    engine = RooflineDoubleBufferingEngine(peak_tflops=16.0, dram_bw_gbs=64.0)
    print(f"[*] Peak NPU Compute Rate    : {engine.peak_tflops:.1f} TFLOPS")
    print(f"[*] Memory Subsystem BW      : DRAM={engine.dram_bw:.0f} GB/s | L2={engine.l2_bw:.0f} GB/s | L1={engine.l1_bw:.0f} GB/s")
    print(f"[*] Roofline Ridge Point     : {engine.ridge_dram:.1f} FLOP/Byte (DRAM Boundary)")
    
    # 1. Kernel Roofline Breakdown
    kernels = [
        ("Conv2D 3x3 (Batch=16)", 1.2e11, 2.5e8, 1.2e8),
        ("Pointwise 1x1 Conv", 8.4e10, 8.0e8, 2.1e8),
        ("LayerNorm / RMSNorm", 4.2e8, 1.5e8, 2.0e7),
        ("Multi-Head Attention", 6.8e10, 4.0e8, 9.5e7)
    ]
    print("\n--- [Kernel Roofline Evaluation] ---")
    for k in kernels:
        res = engine.evaluate_kernel_roofline(k[0], k[1], k[2], k[3])
        bound_type = "COMPUTE-BOUND" if res["compute_bound"] else "MEMORY-BOUND"
        print(f"  {res['kernel']:24s} | Intensity: {res['intensity_dram']:6.1f} FLOP/B | Perf: {res['perf_dram']:5.2f} TFLOPS [{bound_type}]")
        
    # 2. Ping-Pong Double-Buffering Simulation
    sim = engine.simulate_double_buffering(num_tiles=128, tile_flops=2.5e8, tile_bytes=1.2e6)
    print("\n--- [Ping-Pong Double-Buffering Simulation (128 Tiles)] ---")
    print(f"  Synchronous Execution Time  : {sim['t_sync_total_ms']:.2f} ms")
    print(f"  Asynchronous Pipelined Time : {sim['t_async_total_ms']:.2f} ms")
    print(f"  Net Throughput Speedup      : {sim['speedup_factor']:.2f}x Acceleration")
    print(f"  DRAM Stall Latency Hidden   : {sim['stall_eliminated_pct']:.2f}% Latency Eliminated")
    print(f"  Sustained PE Utilization    : {sim['pe_utilization_pct']:.2f}%")
    print("=" * 70)

if __name__ == "__main__":
    run_benchmark()
