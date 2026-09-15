"""
unified_npu_compiler.py
=============================================================================
Unified Hardware-Aware NPU Compiler & Optimization Engine
Author: Koduru Yagnesh Kumar
Integrates:
  1. Polyhedral Loop Tiling & In-Register Tensor Streaming
  2. NPU Roofline Modeling & Asynchronous DMA Double-Buffering
  3. Ballistic Simulated Bifurcation (bSBA) & Variational QAOA Scheduling

All reported metrics are computed at runtime from the models below:
  - The kernel speedup is derived from a roofline (attainable-performance)
    calculation on the fused vs unfused DRAM traffic.
  - The SRAM bank-conflict reduction is computed from an explicit
    uncoordinated vs bank-aware access simulation.
  - The SA-vs-bSBA comparison times a real simulated-annealing baseline
    against the real symplectic bSBA loop on the same Ising instance, with
    best-of-N bSBA restarts (deterministic seed ramp) followed by a
    feasibility-preserving 1-opt Ising-energy polish.
  - The QAOA approximation ratio is computed against the exact ground state
    of a 14-spin subinstance (exhaustive over 2^14 states) using a real
    p-sweep (p = 1, 2, 3) statevector simulation with 2p-parameter
    multi-start COBYLA optimization at each depth.
=============================================================================
"""

import os
import json
import time
import numpy as np
from scipy.optimize import minimize


class UnifiedNPUCompiler:
    def __init__(self, sram_capacity_kb: float = 2048.0, dram_bw_gbps: float = 64.0, peak_gflops: float = 16000.0):
        self.sram_capacity_kb = sram_capacity_kb
        self.dram_bw_gbps = dram_bw_gbps
        self.peak_gflops = peak_gflops
        self.roofline_knee = self.peak_gflops / self.dram_bw_gbps # FLOP/Byte

    # ------------------------------------------------------------------
    # Phase 1: Polyhedral Loop Tiling & In-Register Streaming
    # ------------------------------------------------------------------
    def run_polyhedral_fusion(self, input_activation_kb: float, weight_kb: float, output_activation_kb: float, num_layers: int = 4):
        print("\n--- [Phase 1: Polyhedral Loop Tiling & In-Register Streaming] ---")
        unfused_dram_traffic = (input_activation_kb + 2 * output_activation_kb * (num_layers - 1) + weight_kb * num_layers)
        fused_dram_traffic = (input_activation_kb + output_activation_kb + weight_kb * num_layers)
        dram_traffic_eliminated_pct = ((unfused_dram_traffic - fused_dram_traffic) / unfused_dram_traffic) * 100.0

        # Unfused vs fused live buffer memory
        unfused_peak_live_kb = output_activation_kb * num_layers * 1.25
        tile_h, tile_w = 4, 14
        fused_peak_live_kb = (tile_h * tile_w * 64 * 4) / 1024.0 # Tiled live buffer
        sram_compression_pct = ((unfused_peak_live_kb - fused_peak_live_kb) / unfused_peak_live_kb) * 100.0

        # Kernel speedup from the roofline model: the kernel is memory-bound,
        # so attainable performance scales with arithmetic intensity
        # (FLOPs / DRAM bytes moved). FLOPs are identical fused and unfused;
        # only the traffic changes.
        output_elements = output_activation_kb * 1024.0 / 4.0  # INT32 outputs
        total_flops = 2.0 * output_elements * num_layers
        intensity_unfused = total_flops / (unfused_dram_traffic * 1024.0)
        intensity_fused = total_flops / (fused_dram_traffic * 1024.0)
        attainable_unfused = min(self.peak_gflops, intensity_unfused * self.dram_bw_gbps)
        attainable_fused = min(self.peak_gflops, intensity_fused * self.dram_bw_gbps)
        kernel_speedup = attainable_fused / attainable_unfused

        print(f" Unfused DRAM Traffic  : {unfused_dram_traffic:.2f} KB | Fused DRAM Traffic: {fused_dram_traffic:.2f} KB")
        print(f" DRAM Traffic Eliminated: {dram_traffic_eliminated_pct:.2f}%")
        print(f" Arithmetic Intensity   : {intensity_unfused:.2f} -> {intensity_fused:.2f} FLOP/Byte (Roofline Knee: {self.roofline_knee:.2f})")
        print(f" Peak Live Activation  : {unfused_peak_live_kb:.2f} KB -> {fused_peak_live_kb:.2f} KB")
        print(f" Live SRAM Compression : {sram_compression_pct:.2f}%")
        print(f" Roofline Kernel Speedup: {kernel_speedup:.2f}x (attainable-performance ratio)")

        return {
            "unfused_traffic_kb": unfused_dram_traffic,
            "fused_traffic_kb": fused_dram_traffic,
            "dram_traffic_reduction_pct": dram_traffic_eliminated_pct,
            "sram_compression_pct": sram_compression_pct,
            "intensity_unfused": intensity_unfused,
            "intensity_fused": intensity_fused,
            "kernel_speedup": kernel_speedup
        }

    # ------------------------------------------------------------------
    # Phase 2: Roofline & Double-Buffering (+ bank conflict model)
    # ------------------------------------------------------------------
    def run_roofline_and_double_buffering(self, total_flops: float, total_bytes: float, num_stages: int = 8,
                                          num_cores: int = 8, num_banks: int = 8):
        print("\n--- [Phase 2: NPU Roofline Analysis & Ping-Pong DMA Buffer Allocation] ---")
        intensity = total_flops / max(total_bytes, 1.0)
        is_compute_bound = intensity >= self.roofline_knee
        attainable_perf = min(self.peak_gflops, intensity * self.dram_bw_gbps)

        print(f" Operational Arithmetic Intensity: {intensity:.2f} FLOP/Byte (Roofline Knee: {self.roofline_knee:.2f} FLOP/B)")
        print(f" Execution Regime               : {'COMPUTE-BOUND' if is_compute_bound else 'MEMORY-BOUND'}")
        print(f" Attainable Performance         : {attainable_perf:.2f} GFLOP/s ({(attainable_perf/self.peak_gflops)*100:.1f}% peak)")

        # Ping-Pong Double Buffering Simulation
        t_compute_base = total_flops / (self.peak_gflops * 1e9)
        t_dma_base     = total_bytes / (self.dram_bw_gbps * 1e9)
        t_unbuffered   = (t_compute_base + t_dma_base) * 1e6 # in microseconds

        # With double buffering, compute and DMA overlap: T = max(T_comp, T_dma) + priming delay
        t_priming = (t_dma_base / num_stages) * 1e6
        t_buffered = (max(t_compute_base, t_dma_base) * 1e6) + t_priming
        latency_hiding_pct = ((t_unbuffered - t_buffered) / t_unbuffered) * 100.0

        print(f" Sequential Latency (Unbuffered): {t_unbuffered:.2f} us")
        print(f" Overlapped Latency (Double-Buf): {t_buffered:.2f} us")
        print(f" Memory Latency Hidden          : {latency_hiding_pct:.2f}%")

        # SRAM bank-conflict model: uncoordinated random bank selection vs
        # bank-aware (core i -> bank i mod B) placement. The conflict metric
        # is the peak per-bank concurrent access load, whose reduction
        # directly mirrors arbitration stall reduction.
        rng = np.random.default_rng(42)
        accesses_per_core = 64
        uncoordinated = np.zeros((num_cores, num_banks))
        for core in range(num_cores):
            banks = rng.integers(0, num_banks, size=accesses_per_core)
            for b in banks:
                uncoordinated[core, b] += 1.0
        coordinated = np.zeros((num_cores, num_banks))
        for core in range(num_cores):
            primary = core % num_banks
            coordinated[core, primary] += accesses_per_core * 0.75
            coordinated[core, (primary + 1) % num_banks] += accesses_per_core * 0.25

        peak_uncoordinated = float(np.max(uncoordinated.sum(axis=0)))
        peak_coordinated = float(np.max(coordinated.sum(axis=0)))
        bank_conflict_reduction_pct = (1.0 - peak_coordinated / peak_uncoordinated) * 100.0
        print(f" SRAM Bank Conflict Model       : peak per-bank load {peak_uncoordinated:.0f} -> {peak_coordinated:.0f} "
              f"({bank_conflict_reduction_pct:.1f}% peak-load reduction)")

        return {
            "intensity": intensity,
            "is_compute_bound": is_compute_bound,
            "latency_hiding_pct": latency_hiding_pct,
            "bank_peak_load_uncoordinated": peak_uncoordinated,
            "bank_peak_load_coordinated": peak_coordinated,
            "sram_conflict_reduction_pct": bank_conflict_reduction_pct
        }

    # ------------------------------------------------------------------
    # Phase 3 helpers: bSBA core, 1-opt polish, SA baseline
    # ------------------------------------------------------------------
    @staticmethod
    def _bsba_core(J, num_steps, rng, dt=0.5, c0=1.0):
        """One ballistic simulated bifurcation trajectory (symplectic Euler
        integration with the inelastic ballistic wall). Returns the sign
        spin configuration of the final oscillator amplitudes."""
        n = J.shape[0]
        x = rng.uniform(-0.1, 0.1, n)
        y = np.zeros(n)

        for s in range(num_steps):
            a_t = (s / float(num_steps)) * 1.5
            # Inelastic wall boundary condition for Ballistic SBA
            dx = y * dt
            x += dx
            # Ballistic wall condition: if |x| > 1, reflect position and reset velocity
            mask = np.abs(x) > 1.0
            x[mask] = np.sign(x[mask])
            y[mask] = 0.0

            dy = (-(a_t - c0) * x - c0 * (x**3) + np.dot(J, x)) * dt
            y += dy

        spins = np.sign(x)
        spins[spins == 0] = 1.0
        return spins

    @staticmethod
    def _ising_local_polish(J, spins, max_flips=400):
        """Feasibility-preserving 1-opt local search on the Ising energy.

        Single-spin flips are accepted only when the Ising energy strictly
        improves (delta E = 2 * s_i * (J s)_i < 0), sweeping until no
        improving flip exists or the flip budget is exhausted. The state
        stays a valid {+-1}^n configuration throughout.
        """
        s = spins.copy()
        n = J.shape[0]
        energy = -0.5 * float(s @ J @ s)
        flips = 0
        improved = True
        while improved and flips < max_flips:
            improved = False
            for i in range(n):
                if flips >= max_flips:
                    break
                delta = 2.0 * s[i] * float(J[i] @ s)
                if delta < 0.0:
                    s[i] = -s[i]
                    energy += delta
                    flips += 1
                    improved = True
        return s, energy, flips

    def run_ballistic_bifurcation_solver(self, num_variables: int = 32, num_steps: int = 150,
                                         num_restarts: int = 8, run_qaoa: bool = True):
        print("\n--- [Phase 3: Ballistic Simulated Bifurcation (bSBA) Combinatorial Optimization] ---")
        np.random.seed(42)
        # Construct synthetic NPU coupling matrix J
        J = np.random.randn(num_variables, num_variables) * 0.5
        J = (J + J.T) / 2.0
        np.fill_diagonal(J, 0.0)

        # Single-run raw bSBA (one trajectory, default schedule): the fast
        # baseline before restarts and polish are applied.
        single_start = time.perf_counter()
        single_rng = np.random.default_rng(1000)
        single_spins = self._bsba_core(J, num_steps, single_rng)
        single_energy = -0.5 * float(single_spins @ J @ single_spins)
        single_ms = (time.perf_counter() - single_start) * 1000.0

        # Best-of-N multi-restart bSBA: each restart uses a distinct
        # deterministic seed (base_seed + restart index), the best-energy
        # configuration is kept, then a 1-opt Ising-energy polish refines it.
        bsba_start = time.perf_counter()
        best_spins = None
        best_energy = np.inf
        for restart in range(num_restarts):
            rng = np.random.default_rng(1000 + restart)
            spins = self._bsba_core(J, num_steps, rng)
            energy = -0.5 * float(spins @ J @ spins)
            if energy < best_energy:
                best_energy = energy
                best_spins = spins
        raw_restart_ms = (time.perf_counter() - bsba_start) * 1000.0

        polish_start = time.perf_counter()
        polished_spins, polished_energy, polish_flips = self._ising_local_polish(
            J, best_spins, max_flips=400)
        polish_ms = (time.perf_counter() - polish_start) * 1000.0
        bsba_total_ms = (time.perf_counter() - bsba_start) * 1000.0

        # Real simulated annealing baseline on the SAME Ising instance
        sa_start = time.perf_counter()
        sa_spins, sa_energy = self._simulated_annealing_baseline(J, num_sweeps=400)
        sa_time_ms = (time.perf_counter() - sa_start) * 1000.0
        speedup = sa_time_ms / bsba_total_ms

        # Real QAOA approximation ratio sweep (p = 1, 2, 3) on a 14-spin subinstance
        qaoa = self._qaoa_approximation_ratio(J, sub_n=14, depths=(1, 2, 3)) if run_qaoa else None

        print(f" bSBA Single Run ({num_steps} steps)     : {single_ms:.3f} ms | energy {single_energy:.4f}")
        print(f" bSBA Best-of-{num_restarts} Restarts ({num_steps} steps each): {raw_restart_ms:.3f} ms")
        print(f" bSBA 1-opt Polish              : {polish_ms:.3f} ms ({polish_flips} improving flips)")
        print(f" bSBA Total Time (restarts+polish): {bsba_total_ms:.3f} ms")
        print(f" Classical Simulated Annealing  : {sa_time_ms:.3f} ms (400 sweeps, same J)")
        print(f" Measured Speedup over SA       : {speedup:.1f}x")
        print(f" bSBA Best Restart Energy       : {best_energy:.4f}")
        print(f" bSBA+Polish Hamiltonian Energy : {polished_energy:.4f}")
        print(f" SA Best Hamiltonian Energy     : {sa_energy:.4f}")
        print(" bSBA vs SA comparison table (measured on this run):")
        print("   +------------------------------------+-----------+-------------+")
        print("   | Method                             | Time (ms) | Energy      |")
        print("   +------------------------------------+-----------+-------------+")
        print(f"   | bSBA single run ({num_steps} steps)      | {single_ms:>9.3f} | {single_energy:>11.4f} |")
        print(f"   | bSBA best-of-{num_restarts} + 1-opt polish | {bsba_total_ms:>9.3f} | {polished_energy:>11.4f} |")
        print(f"   | Simulated annealing (400 sweeps)   | {sa_time_ms:>9.3f} | {sa_energy:>11.4f} |")
        print("   +------------------------------------+-----------+-------------+")
        if run_qaoa:
            print(f" QAOA Depth Sweep (p=1,2,3)     : ratios "
                  + ", ".join(f"p={p}: {d['approx_ratio']:.3f}" for p, d in
                              zip(qaoa['per_p_ratios'].keys(), qaoa['per_depth']))
                  + f" | best p={qaoa['best_p']} ratio {qaoa['approx_ratio']:.3f}")
            print(f"   (E_QAOA={qaoa['qaoa_energy']:.3f}, E_ground={qaoa['ground_energy']:.3f}, "
                  f"{qaoa['subinstance_spins']}-spin subinstance, exhaustive 2^{qaoa['subinstance_spins']} ground state, "
                  f"2p-parameter multi-start COBYLA)")

        result = {
            "sba_runtime_ms": bsba_total_ms,
            "bsba_single_run_ms": single_ms,
            "bsba_single_run_energy": single_energy,
            "bsba_restarts_ms": raw_restart_ms,
            "bsba_polish_ms": polish_ms,
            "bsba_polish_flips": polish_flips,
            "num_restarts": num_restarts,
            "num_steps": num_steps,
            "sa_runtime_ms": sa_time_ms,
            "speedup_x": speedup,
            "final_energy": polished_energy,
            "bsba_raw_energy": best_energy,
            "bsba_polish_energy": polished_energy,
            "sa_best_energy": sa_energy
        }
        if run_qaoa:
            result.update({
                "qaoa_approx_ratio": qaoa["approx_ratio"],
                "qaoa_best_p": qaoa["best_p"],
                "qaoa_per_p_ratios": qaoa["per_p_ratios"],
                "qaoa_energy": qaoa["qaoa_energy"],
                "qaoa_ground_energy": qaoa["ground_energy"],
                "qaoa_subinstance_spins": qaoa["subinstance_spins"]
            })
        return result

    @staticmethod
    def _simulated_annealing_baseline(J, num_sweeps=400, T_init=2.0, T_final=0.01, seed=7):
        """Classical bitwise simulated annealing on the same Ising instance.

        E(s) = -0.5 * s^T J s with s in {-1,+1}^n. Flip acceptance uses the
        exact incremental delta E(s_i -> -s_i) = 2 * s_i * (J s)_i.
        """
        rng = np.random.default_rng(seed)
        n = J.shape[0]
        s = rng.choice([-1.0, 1.0], size=n)
        energy = -0.5 * float(s @ J @ s)
        best_energy = energy
        best_s = s.copy()
        for sweep in range(num_sweeps):
            T = T_init * (T_final / T_init) ** (sweep / max(1, num_sweeps - 1))
            for i in range(n):
                delta = 2.0 * s[i] * float(J[i] @ s)
                if delta <= 0.0 or rng.random() < np.exp(-delta / T):
                    s[i] = -s[i]
                    energy += delta
                    if energy < best_energy:
                        best_energy = energy
                        best_s = s.copy()
        return best_s, best_energy

    @staticmethod
    def _ising_energies(J):
        """Exact energies of all 2^n bitstrings of the Ising Hamiltonian."""
        n = J.shape[0]
        states = np.arange(2 ** n, dtype=np.int64)
        bits = ((states[:, None] >> np.arange(n)[None, :]) & 1).astype(np.int8)
        spins = 1.0 - 2.0 * bits.astype(np.float64)
        # E = -0.5 * s^T J s for every state, vectorized over states
        energies = -0.5 * np.einsum('bi,ij,bj->b', spins, J, spins, optimize=True)
        return energies

    @staticmethod
    def _qaoa_expectation(params, energies, n):
        """Exact QAOA expectation <psi(gamma,beta)| H |psi(gamma,beta)>

        at any depth p (params = [gamma_1..gamma_p, beta_1..beta_p]) for a
        diagonal Ising problem Hamiltonian, evaluated via a full
        2^n-dimensional statevector simulation.
        """
        p = len(params) // 2
        dim = 2 ** n
        # Problem layer: H_P is diagonal -> elementwise phase
        state = np.full(dim, 1.0 / np.sqrt(dim), dtype=np.complex128)
        # Cache the mixer rotation per distinct beta to avoid rebuilding it
        rot_cache = {}

        for layer in range(p):
            gamma, beta = params[layer], params[p + layer]
            state = state * np.exp(-1j * gamma * energies)

            # Mixer layer: e^{-i beta H_M} with H_M = sum_i sigma^x_i, applied as
            # an identical 2x2 rotation on each qubit axis of the (2,)*n tensor.
            rot = rot_cache.get(beta)
            if rot is None:
                rot = np.array([[np.cos(beta), -1j * np.sin(beta)],
                                [-1j * np.sin(beta), np.cos(beta)]], dtype=np.complex128)
                rot_cache[beta] = rot
            state = state.reshape((2,) * n)
            for axis in range(n):
                state = np.moveaxis(state, axis, 0)
                shape = state.shape
                state = state.reshape(2, -1)
                state = rot @ state
                state = state.reshape(shape)
                state = np.moveaxis(state, 0, axis)
            state = state.reshape(-1)

        probs = np.abs(state) ** 2
        return float(np.dot(probs, energies))

    def _qaoa_approximation_ratio(self, J_full, sub_n=14, depths=(1, 2, 3),
                                  cobyla_maxiter=600):
        """Best approximation ratio of QAOA on a subinstance vs its exact
        ground state (exhaustive over 2^sub_n states), swept over depth
        p in `depths`. Each depth optimizes the full 2p-parameter vector
        (gamma_1..gamma_p, beta_1..beta_p) with multi-start COBYLA."""
        J_sub = J_full[:sub_n, :sub_n].copy()
        energies = self._ising_energies(J_sub)
        ground_energy = float(np.min(energies))

        per_depth = []
        for p in depths:
            def objective(params):
                return self._qaoa_expectation(params, energies, sub_n)

            # Multi-start COBYLA over the 2p-dimensional parameter space.
            # Interleaved initial gamma/beta values cover both shallow-angle
            # and deep-angle regimes.
            gamma_seeds = (0.05, 0.1, 0.2, 0.4, 0.7, 1.0)
            beta_seeds = (0.05, 0.15, 0.3, 0.6)
            starts = [(g, b) for g in gamma_seeds for b in beta_seeds]
            first_g, first_b = starts[0]
            best = (objective([first_g] * p + [first_b] * p),
                    [first_g] * p, [first_b] * p)
            for x0 in starts:
                gamma0 = [x0[0]] * p
                beta0 = [x0[1]] * p
                res = minimize(objective, x0=gamma0 + beta0, method="COBYLA",
                               options={"maxiter": cobyla_maxiter, "rhobeg": 0.3,
                                        "tol": 1e-8})
                if res.fun < best[0]:
                    best = (float(res.fun),
                            [float(v) for v in res.x[:p]],
                            [float(v) for v in res.x[p:]])
            qaoa_energy, opt_gammas, opt_betas = best
            per_depth.append({
                "p": p,
                "qaoa_energy": qaoa_energy,
                "approx_ratio": qaoa_energy / ground_energy,
                "optimal_gammas": opt_gammas,
                "optimal_betas": opt_betas
            })

        best_depth = max(per_depth, key=lambda d: d["approx_ratio"])
        return {
            "qaoa_energy": best_depth["qaoa_energy"],
            "ground_energy": ground_energy,
            "approx_ratio": best_depth["approx_ratio"],
            "best_p": best_depth["p"],
            "per_p_ratios": {str(d["p"]): d["approx_ratio"] for d in per_depth},
            "per_depth": per_depth,
            "subinstance_spins": sub_n
        }

    def compile(self):
        print("==================================================================")
        print("   UNIFIED NPU OPTIMIZATION SUITE -- FULL COMPILER PIPELINE")
        print("==================================================================")
        p1 = self.run_polyhedral_fusion(input_activation_kb=1024, weight_kb=2048, output_activation_kb=768, num_layers=4)
        p2 = self.run_roofline_and_double_buffering(total_flops=1.2e10, total_bytes=6.4e7, num_stages=8)
        p3 = self.run_ballistic_bifurcation_solver(num_variables=32, num_steps=150)

        # Unified Summary (all values computed above; the DRAM traffic
        # elimination in Phase 1 translates directly into off-chip DRAM
        # dynamic-energy savings since DRAM accesses dominate NPU energy)
        print("\n==================================================================")
        print("               UNIFIED COMPILER BENCHMARK SUMMARY                 ")
        print("==================================================================")
        print(f" 1. Polyhedral Fusion Live SRAM Compression : {p1['sram_compression_pct']:.2f}% ({p1['kernel_speedup']:.2f}x roofline kernel speedup)")
        print(f" 2. Asynchronous DMA Memory Latency Hidden  : {p2['latency_hiding_pct']:.2f}% ({p2['sram_conflict_reduction_pct']:.1f}% bank peak-load cut)")
        print(f" 3. Ballistic bSBA (8 restarts + 1-opt polish) vs SA: {p3['bsba_polish_energy']:.2f} vs {p3['sa_best_energy']:.2f} energy in {p3['sba_runtime_ms']:.1f} ms vs {p3['sa_runtime_ms']:.1f} ms ({p3['speedup_x']:.1f}x faster)")
        print(f" 4. Polyhedral DRAM Traffic Eliminated      : {p1['dram_traffic_reduction_pct']:.2f}% (off-chip DRAM dynamic-energy relief)")
        print(f" 5. QAOA Depth Sweep (14-spin exhaustive ground state): best ratio {p3['qaoa_approx_ratio']:.3f} at p={p3['qaoa_best_p']} (per-p: "
              + ", ".join(f"p={p}: {r:.3f}" for p, r in p3['qaoa_per_p_ratios'].items()) + ")")
        print("==================================================================\n")

        return {
            "polyhedral_fusion": p1,
            "roofline_double_buffering": p2,
            "quantum_bifurcation": p3
        }

if __name__ == "__main__":
    compiler = UnifiedNPUCompiler()
    res = compiler.compile()
    out_file = os.path.join(os.path.dirname(__file__), "unified_compiler_report.json")
    with open(out_file, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[SUCCESS] Unified compiler report written to: {out_file}")
