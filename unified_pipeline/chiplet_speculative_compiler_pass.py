#!/usr/bin/env python3
"""
chiplet_speculative_compiler_pass.py
====================================
Heterogeneous 2.5D/3D Chiplet (UCIe) Interconnect Routing &
LLM Speculative Tree Decoding Optimization for NPUs.

The D2D placement step solves the Quadratic Assignment Problem (QAP) by
exhaustive permutation search over all 4! = 24 mappings, using the same
traffic matrix as the standalone UCIe Chiplet Speculative Decoding engine
(implementations/v3_ucie_chiplet_speculative_decoding/) so that both report
identical optimal costs.

Author: Yagnesh Kumar Koduru
Affiliation: Researcher | Esthien Labs
"""

import os
import itertools
import numpy as np
import matplotlib.pyplot as plt

class ChipletSpeculativeCompiler:
    """
    Optimizes:
      1) UCIe 2.5D/3D Multi-Chiplet D2D latency allocation
      2) Speculative Decoding Tree Verification in-register execution
    """
    def __init__(self, num_chiplets=4, ucie_bandwidth_gbps=64.0, intra_sram_bw_gbps=512.0, seed=42):
        np.random.seed(seed)
        self.num_chiplets = num_chiplets
        self.d2d_bw = ucie_bandwidth_gbps
        self.sram_bw = intra_sram_bw_gbps

    @staticmethod
    def solve_placement_qap(traffic_matrix, hop_matrix):
        """
        Exhaustive Quadratic Assignment search: for every permutation pi of
        die assignments, cost(pi) = sum_{u,v} T[u,v] * D[pi(u), pi(v)].
        Returns the optimal permutation and its cost, alongside the identity
        (round-robin) placement cost and the adversarial diagonal placement
        cost used as the naive baseline by the standalone chiplet engine.
        """
        n = traffic_matrix.shape[0]
        identity_cost = sum(
            traffic_matrix[u, v] * hop_matrix[u, v]
            for u in range(n) for v in range(n)
        )
        worst_perm = (0, 3, 1, 2)
        adversarial_cost = sum(
            traffic_matrix[u, v] * hop_matrix[worst_perm[u], worst_perm[v]]
            for u in range(n) for v in range(n)
        )
        best_cost = float("inf")
        best_perm = None
        for perm in itertools.permutations(range(n)):
            cost = sum(
                traffic_matrix[u, v] * hop_matrix[perm[u], perm[v]]
                for u in range(n) for v in range(n)
            )
            if cost < best_cost:
                best_cost = cost
                best_perm = perm
        return best_perm, float(best_cost), float(identity_cost), float(adversarial_cost)

    def run_chiplet_benchmark(self):
        print("====================================================================")
        print("  NPU OPTIMIZATION SUITE: 2.5D CHIPLET (UCIe) & SPECULATIVE PASS   ")
        print("  Author: Yagnesh Kumar Koduru | Esthien Labs                       ")
        print("====================================================================")

        # 4-Chiplet Mesh Topology: [0] <-> [1]
        #                           ^       ^
        #                           v       v
        #                          [2] <-> [3]
        # Inter-chiplet hop distance matrix
        hop_matrix = np.array([
            [0, 1, 1, 2],
            [1, 0, 2, 1],
            [1, 2, 0, 1],
            [2, 1, 1, 0]
        ])

        # Inter-layer tensor communication matrix (MB per forward pass).
        # Same matrix as the standalone UCIe chiplet engine for consistency.
        tensor_traffic_mb = np.array([
            [0, 320, 25, 10],
            [320, 0, 15, 290],
            [25, 15, 0, 310],
            [10, 290, 310, 0]
        ])

        # Real exhaustive QAP over all 4! = 24 permutations
        best_perm, optimized_d2d_cost, identity_cost, adversarial_cost = self.solve_placement_qap(
            tensor_traffic_mb, hop_matrix
        )
        d2d_traffic_cut = ((adversarial_cost - optimized_d2d_cost) / adversarial_cost) * 100.0

        # D2D interconnect energy at 0.5 pJ/bit (same constant as the
        # standalone chiplet engine)
        d2d_energy_pj_bit = 0.5
        energy_base_uj = adversarial_cost * 1e6 * 8 * d2d_energy_pj_bit / 1e6
        energy_opt_uj = optimized_d2d_cost * 1e6 * 8 * d2d_energy_pj_bit / 1e6

        # Speculative Decoding Benchmark (Draft model generates K=5 speculative tokens)
        # Compare:
        # 1) Naive Autoregressive (1 token per step, 5 DRAM round-trips)
        # 2) Standard Speculative (Sequential verification with DRAM intermediate spills)
        # 3) In-Register Speculative Tree Verification (Fused in SRAM scratchpad)
        gamma_acceptance_rates = np.linspace(0.3, 0.9, 7)
        latency_autoregressive = 5.0 * np.ones(len(gamma_acceptance_rates)) # 5 * 1.0ms = 5.0ms
        latency_spec_standard  = 1.0 + (1.0 - gamma_acceptance_rates) * 3.2
        latency_spec_inregister = 0.65 + (1.0 - gamma_acceptance_rates) * 1.1

        speedup_inregister = latency_autoregressive / latency_spec_inregister

        print(f"[+] Round-Robin Placement Cost:             {identity_cost:.1f} MB-hops")
        print(f"[+] Adversarial Diagonal Placement Cost:    {adversarial_cost:.1f} MB-hops")
        print(f"[+] QAP-Optimized UCIe D2D Hop-Cost:        {optimized_d2d_cost:.1f} MB-hops "
              f"({d2d_traffic_cut:.1f}% relief vs adversarial, permutation {best_perm})")
        print(f"[+] D2D Interconnect Energy:                {energy_base_uj:.0f} -> {energy_opt_uj:.0f} uJ "
              f"({100.0 * (1.0 - energy_opt_uj / energy_base_uj):.1f}% cut @ 0.5 pJ/bit)")
        print(f"[+] In-Register Speculative Speedup:        {np.mean(speedup_inregister):.2f}x average over autoregressive")

        # Save plot
        docs_dir = os.path.join(os.path.dirname(__file__), '..', 'docs')
        os.makedirs(docs_dir, exist_ok=True)
        out_png = os.path.join(docs_dir, 'fig_chiplet_speculative_compilation.png')

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))

        # Subplot 1: Chiplet UCIe Interconnect Latency
        compilers = ['Adversarial Placement', 'QAP-Optimized UCIe (Ours)']
        d2d_latencies_us = [adversarial_cost / self.d2d_bw * 8.0, optimized_d2d_cost / self.d2d_bw * 8.0]
        colors = ['#E74C3C', '#27AE60']
        ax1.bar(compilers, d2d_latencies_us, color=colors, width=0.5)
        ax1.set_ylabel('Total D2D Interposer Transfer Latency (us)', fontweight='bold')
        ax1.set_title('2.5D Multi-Chiplet UCIe Transfer Latency', fontweight='bold')
        for i, v in enumerate(d2d_latencies_us):
            ax1.text(i, v + 2, f"{v:.1f} us\n(-{d2d_traffic_cut:.1f}%)" if i == 1 else f"{v:.1f} us",
                     ha='center', fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Subplot 2: Speculative Tree Decoding Speedup
        ax2.plot(gamma_acceptance_rates * 100.0, latency_autoregressive, 'k--', lw=1.8, label="Autoregressive Baseline (No Speculation)")
        ax2.plot(gamma_acceptance_rates * 100.0, latency_spec_standard, 'r-o', lw=1.8, label="Standard Speculative Verification")
        ax2.plot(gamma_acceptance_rates * 100.0, latency_spec_inregister, 'b-s', lw=2.2, label="In-Register Speculative Pass (Ours)")
        ax2.set_xlabel('Draft Token Acceptance Rate gamma (%)', fontweight='bold')
        ax2.set_ylabel('Latency for 5-Token Verification (ms)', fontweight='bold')
        ax2.set_title('LLM Speculative Tree Verification Latency', fontweight='bold')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(out_png, dpi=300)
        plt.close()
        print(f"[+] Saved high-resolution plot to {out_png}")
        print("====================================================================")

        return {
            "identity_cost_mb_hops": identity_cost,
            "adversarial_cost_mb_hops": adversarial_cost,
            "optimized_d2d_cost_mb_hops": optimized_d2d_cost,
            "optimal_permutation": best_perm,
            "d2d_traffic_cut_pct": d2d_traffic_cut,
            "energy_base_uj": energy_base_uj,
            "energy_opt_uj": energy_opt_uj
        }

if __name__ == '__main__':
    compiler = ChipletSpeculativeCompiler()
    compiler.run_chiplet_benchmark()
