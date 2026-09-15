"""
NPU Roofline Modeling, Ping-Pong Double-Buffering & Multi-Bank SRAM Simulation
Author: Koduru Yagnesh Kumar
Repository: NPU-Memory-Aware-Scheduling
Domain: Domain-Specific Architecture, Memory Hierarchy Modeling, Compiler Dataflow
"""

import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['lines.linewidth'] = 2.0
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.35

output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'outputs'))
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


class NPURooflineEngine:
    """
    Evaluates operational arithmetic intensity (FLOPs / Byte) against NPU memory
    and compute bounds across diverse deep learning operator classes.
    """
    def __init__(self, peak_ops_tflops=16.0, dram_bw_gb_s=64.0, sram_bw_gb_s=512.0):
        self.peak_ops = peak_ops_tflops * 1e12       # 16 Tera-OPS
        self.dram_bw = dram_bw_gb_s * 1e9           # 64 GB/s
        self.sram_bw = sram_bw_gb_s * 1e9           # 512 GB/s on-chip
        self.knee_intensity = self.peak_ops / self.dram_bw  # FLOPs/Byte

    def generate_roofline_plot(self):
        intensities = np.logspace(-1, 3, 500)
        # Performance bound = min(Peak Compute, Bandwidth * Intensity)
        perf_dram = np.minimum(self.peak_ops, self.dram_bw * intensities) / 1e12
        perf_sram = np.minimum(self.peak_ops, self.sram_bw * intensities) / 1e12

        operators = [
            ("Conv2D (Standard)", 32.5, 14.8, '#E74C3C'),
            ("Depthwise Conv", 4.2, 0.26, '#3498DB'),
            ("Pointwise 1x1 Conv", 22.0, 12.2, '#2ECC71'),
            ("LayerNorm / GELU", 0.65, 0.04, '#9B59B6'),
            ("Multi-Head Attention (QKV)", 48.0, 15.6, '#F39C12'),
            ("Linear (Memory-Bound)", 1.8, 0.11, '#E67E22')
        ]

        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        ax.loglog(intensities, perf_dram, 'r-', linewidth=2.5, label='Off-Chip DRAM Roofline (64 GB/s)')
        ax.loglog(intensities, perf_sram, 'b--', linewidth=2.0, alpha=0.7, label='On-Chip SRAM Double-Buffer Roofline (512 GB/s)')

        ax.axvline(x=self.knee_intensity, color='gray', linestyle=':', label=f'Roofline Knee ($I^* = {self.knee_intensity:.1f}$ FLOP/B)')

        for name, intensity, ops, col in operators:
            ax.plot(intensity, ops, 'o', markersize=8, color=col)
            ax.annotate(name, (intensity * 1.12, ops * 0.92), fontsize=9, fontweight='bold', color='#2C3E50')

        ax.set_xlabel('Operational Arithmetic Intensity $I$ (FLOPs / Byte)', fontweight='bold')
        ax.set_ylabel('Attainable Compute Throughput (TFLOPS)', fontweight='bold')
        ax.set_title('NPU Roofline Architecture: Memory-Bound vs Compute-Bound Operators', fontweight='bold', pad=12)
        ax.legend(loc='lower right', framealpha=0.95)
        plt.tight_layout()
        filepath = os.path.join(output_dir, 'fig_npu_roofline_model.png')
        fig.savefig(filepath, dpi=300)
        plt.close(fig)
        return filepath


class DoubleBufferingSimulator:
    """
    Simulates asynchronous DMA ping-pong double-buffering latency hiding.

    Models the same pipelined execution as the standalone double-buffering
    engine (implementations/v2_roofline_dma_double_buffering/):
        T_sync  = N * (T_dma + T_compute)                     (serial)
        T_async = T_dma + (N-1) * max(T_dma, T_compute) + T_compute
    with N tiles in flight. The default configuration (16 TFLOPS PEs,
    64 GB/s DRAM, 208.33 FLOP/Byte tile intensity, 128 tiles) matches that
    engine's benchmark tile (2.5e8 FLOPs, 1.2e6 bytes), so the computed
    latency-hiding ratio is directly comparable to its 45.10% output.
    """
    def __init__(self, dma_bandwidth_gb_s=64.0, pe_compute_throughput_gflops=16000.0,
                 ops_per_byte=2.5e8 / 1.2e6, num_tiles=128):
        self.dma_bw = dma_bandwidth_gb_s * 1e9
        self.pe_throughput = pe_compute_throughput_gflops * 1e9
        self.ops_per_byte = ops_per_byte
        self.num_tiles = num_tiles

    def evaluate_latency_hiding(self, tile_sizes_kb):
        serial_times = []
        double_buf_times = []
        hiding_efficiencies = []
        n = self.num_tiles

        for tile_kb in tile_sizes_kb:
            bytes_tile = tile_kb * 1024
            flops = bytes_tile * self.ops_per_byte

            t_dma = bytes_tile / self.dma_bw
            t_compute = flops / self.pe_throughput

            # Serial: fetch then compute for every tile
            t_serial_total = n * (t_dma + t_compute)
            # Ping-pong pipeline: priming DMA + steady-state + drain compute
            t_pipelined_total = t_dma + (n - 1) * max(t_dma, t_compute) + t_compute

            serial_times.append(t_serial_total * 1e6)          # microseconds
            double_buf_times.append(t_pipelined_total * 1e6)
            efficiency = (t_serial_total - t_pipelined_total) / t_serial_total * 100.0
            hiding_efficiencies.append(efficiency)

        return np.array(serial_times), np.array(double_buf_times), np.array(hiding_efficiencies)

    def generate_plot(self):
        tile_sizes = np.linspace(8, 256, 30)
        t_ser, t_pipe, eff = self.evaluate_latency_hiding(tile_sizes)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.8))

        ax1.plot(tile_sizes, t_ser, 'r-o', markersize=4, label='Serial DMA -> Compute')
        ax1.plot(tile_sizes, t_pipe, 'b-s', markersize=4, label='Asynchronous Ping-Pong Double-Buffer')
        ax1.set_xlabel('Tile Buffer Size (KB)', fontweight='bold')
        ax1.set_ylabel(r'Execution Latency per Tile ($\mu$s)', fontweight='bold')
        ax1.set_title('Tile Execution Latency: Serial vs Double-Buffered', fontweight='bold')
        ax1.legend()

        ax2.plot(tile_sizes, eff, 'g-^', linewidth=2.2)
        ax2.axhline(y=float(np.mean(eff)), color='darkgreen', linestyle='--', label=f'Mean Latency Hiding: {float(np.mean(eff)):.1f}%')
        ax2.set_xlabel('Tile Buffer Size (KB)', fontweight='bold')
        ax2.set_ylabel('Latency Hiding Ratio (%)', fontweight='bold')
        ax2.set_title('Memory Stall Latency Elimination Efficiency', fontweight='bold')
        ax2.legend()

        plt.tight_layout()
        filepath = os.path.join(output_dir, 'fig_double_buffer_latency_hiding.png')
        fig.savefig(filepath, dpi=300)
        plt.close(fig)
        return filepath


class SRAMBankContentionModel:
    """
    Simulates memory bank access collisions across an 8-bank multi-core SRAM array.
    """
    def generate_contention_heatmap(self, num_cores=8, num_banks=8):
        np.random.seed(42)
        # Uncoordinated baseline access distribution
        uncoordinated = np.random.poisson(lam=12, size=(num_cores, num_banks)).astype(float)
        # Coordinate memory placement via bank-aware scheduling
        coordinated = np.zeros((num_cores, num_banks))
        for core in range(num_cores):
            primary_bank = core % num_banks
            coordinated[core, primary_bank] = 24.0
            coordinated[core, (primary_bank + 1) % num_banks] = 6.0
            coordinated[core, :] += np.random.uniform(0.5, 1.5, size=num_banks)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))

        im1 = ax1.imshow(uncoordinated, cmap='YlOrRd', aspect='auto')
        ax1.set_title('Uncoordinated Access (High Bank Collisions)', fontweight='bold')
        ax1.set_xlabel('SRAM Bank ID', fontweight='bold')
        ax1.set_ylabel('Core / Requester ID', fontweight='bold')
        fig.colorbar(im1, ax=ax1, label='Concurrent Access Collisions')

        im2 = ax2.imshow(coordinated, cmap='Blues', aspect='auto')
        ax2.set_title('Bank-Aware Scheduled Placement (Low Conflict)', fontweight='bold')
        ax2.set_xlabel('SRAM Bank ID', fontweight='bold')
        ax2.set_ylabel('Core / Requester ID', fontweight='bold')
        fig.colorbar(im2, ax=ax2, label='Scheduled Access Requests')

        plt.tight_layout()
        filepath = os.path.join(output_dir, 'fig_sram_bank_contention_heatmap.png')
        fig.savefig(filepath, dpi=300)
        plt.close(fig)
        return filepath

    @staticmethod
    def compute_conflict_reduction(num_cores=8, num_banks=8):
        """
        Computes the reduction in bank arbitration conflict potential between
        the uncoordinated Poisson access distribution and the bank-aware
        scheduled placement. Conflict potential per bank is the number of
        concurrently-arriving access pairs, C(n, 2), summed over all banks.
        """
        np.random.seed(42)
        uncoordinated = np.random.poisson(lam=12, size=(num_cores, num_banks)).astype(float)
        coordinated = np.zeros((num_cores, num_banks))
        for core in range(num_cores):
            primary_bank = core % num_banks
            coordinated[core, primary_bank] = 24.0
            coordinated[core, (primary_bank + 1) % num_banks] = 6.0
            coordinated[core, :] += np.random.uniform(0.5, 1.5, size=num_banks)

        def collision_potential(load):
            per_bank = load.sum(axis=0)
            return float(np.sum(per_bank * (per_bank - 1.0) / 2.0))

        uncoord_potential = collision_potential(uncoordinated)
        coord_potential = collision_potential(coordinated)
        reduction_pct = (1.0 - coord_potential / uncoord_potential) * 100.0
        return reduction_pct, uncoord_potential, coord_potential


def run_memory_architecture_study():
    print("=" * 80)
    print("NPU ROOFLINE & MEMORY HIERARCHY DOUBLE-BUFFERING BENCHMARK")
    print("Author: Koduru Yagnesh Kumar")
    print("=" * 80)

    roofline = NPURooflineEngine(peak_ops_tflops=16.0, dram_bw_gb_s=64.0, sram_bw_gb_s=512.0)
    p1 = roofline.generate_roofline_plot()
    print(f"[OK] Generated Roofline Model: {p1}")

    double_buf = DoubleBufferingSimulator(dma_bandwidth_gb_s=64.0, pe_compute_throughput_gflops=16000.0)
    p2 = double_buf.generate_plot()
    print(f"[OK] Generated Double-Buffering Analysis: {p2}")

    bank_model = SRAMBankContentionModel()
    p3 = bank_model.generate_contention_heatmap()
    print(f"[OK] Generated SRAM Bank Contention Heatmap: {p3}")

    # Computed key results (no hardcoded figures)
    mean_hiding_pct = float(np.mean(double_buf.evaluate_latency_hiding(np.linspace(8, 256, 30))[2]))
    conflict_reduction_pct, uncoord_potential, coord_potential = \
        SRAMBankContentionModel.compute_conflict_reduction()

    print("-" * 80)
    print("Key Results (computed by this model, not hardcoded):")
    print(f"  - Roofline Model: Arithmetic Intensity Knee = {roofline.knee_intensity:.1f} FLOPs/Byte")
    print(f"  - Average Memory Latency Hiding with Ping-Pong Double-Buffering: {mean_hiding_pct:.2f}% "
          f"(128-tile pipeline @ {double_buf.ops_per_byte:.1f} FLOP/B, 16 TFLOPS / 64 GB/s)")
    print(f"  - Bank-Aware Scheduling: {conflict_reduction_pct:.1f}% reduction in bank arbitration conflict potential "
          f"(Poisson access model: {uncoord_potential:.0f} -> {coord_potential:.0f} collision pairs)")
    print("=" * 80)


if __name__ == '__main__':
    run_memory_architecture_study()
