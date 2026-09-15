"""
Simulated Bifurcation Algorithm (SBA) & Variational QAOA Energy Landscape Benchmark
Author: Koduru Yagnesh Kumar
Repository: Quantum-QUBO-NPU-Optimization
Domain: Quantum Computing, Ising Solvers, Combinatorial Optimization

All reported metrics are computed at runtime:
  - The QAOA energy landscape is the exact p=1 QAOA expectation
    <psi(gamma, beta)| H |psi(gamma, beta)> over the full 2^n statevector of
    a 12-spin subinstance of the coupling matrix below (n=12 -> 4096 basis
    states, exact diagonal problem layer + per-qubit X-rotation mixer).
  - The COBYLA trajectory is a real scipy.optimize COBYLA run on that
    landscape.
  - The SA-vs-bSBA speedup times a real simulated annealing baseline against
    the real symplectic bSBA loop on the same Ising instance.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

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


class SimulatedBifurcationSimulator:
    """
    Simulates the ballistic Simulated Bifurcation Algorithm (bSBA) based on non-linear
    Kerr oscillator Hamiltonian dynamics for ultrafast combinatorial optimization.
    """
    def __init__(self, num_spins=16, steps=400, dt=0.05, seed=42):
        self.num_spins = num_spins
        self.steps = steps
        self.dt = dt
        self.seed = seed

        np.random.seed(seed)
        # Random NPU operator interaction coupling matrix J (symmetric, zero diagonal)
        J = np.random.randn(num_spins, num_spins)
        J = (J + J.T) / 2.0
        np.fill_diagonal(J, 0.0)
        self.J = J

    def run_bifurcation(self):
        # Position x and momentum y of Kerr oscillators
        np.random.seed(self.seed)
        x = np.random.uniform(-0.1, 0.1, self.num_spins)
        y = np.zeros(self.num_spins)

        c0 = 1.0
        delta_0 = -1.0
        delta_end = 1.0

        trajectory = [x.copy()]

        for step in range(self.steps):
            t_ratio = step / float(self.steps)
            delta = delta_0 + (delta_end - delta_0) * t_ratio

            # Derivative equations of motion:
            # dx/dt = c0 * y
            # dy/dt = - delta * x + xi * sum(J * x)
            xi = 0.5 * t_ratio
            dxdt = c0 * y
            dydt = -delta * x + xi * np.dot(self.J, x)

            # Inelastic wall boundary condition (bSBA)
            x_next = x + dxdt * self.dt
            y_next = y + dydt * self.dt

            # Clamp positions to [-1, 1]
            mask_clamp = np.abs(x_next) > 1.0
            x_next[mask_clamp] = np.sign(x_next[mask_clamp])
            y_next[mask_clamp] = 0.0

            x = x_next
            y = y_next
            trajectory.append(x.copy())

        return np.array(trajectory)

    def final_spin_energy(self):
        """Energy of the bSBA solution: E(s) = -0.5 * s^T J s."""
        traj = self.run_bifurcation()
        spins = np.sign(traj[-1])
        spins[spins == 0] = 1.0
        return -0.5 * float(spins @ self.J @ spins), spins

    def generate_bifurcation_plot(self):
        traj = self.run_bifurcation()
        steps = np.arange(traj.shape[0])

        fig, ax = plt.subplots(figsize=(8.5, 5.0))
        colors = plt.cm.tab20(np.linspace(0, 1, self.num_spins))

        for i in range(self.num_spins):
            ax.plot(steps, traj[:, i], color=colors[i], alpha=0.85, label=f'Spin {i+1}' if i < 6 else None)

        ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.axhline(y=-1.0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.axhline(y=0.0, color='gray', linestyle=':', linewidth=1.0, alpha=0.5)

        ax.set_xlabel('Bifurcation Evolution Step $t$', fontweight='bold')
        ax.set_ylabel('Continuous Oscillator State $x_i(t)$', fontweight='bold')
        ax.set_title('Ballistic Simulated Bifurcation: Quantum-Inspired Binary Phase Transition', fontweight='bold', pad=12)
        ax.legend(loc='upper left', ncol=2, framealpha=0.95)
        plt.tight_layout()
        filepath = os.path.join(output_dir, 'fig_simulated_bifurcation_convergence.png')
        fig.savefig(filepath, dpi=300)
        plt.close(fig)
        return filepath


def simulated_annealing_baseline(J, num_sweeps=400, T_init=2.0, T_final=0.01, seed=7):
    """Classical bitwise simulated annealing on the same Ising instance.

    E(s) = -0.5 * s^T J s. Incremental flip delta: 2 * s_i * (J s)_i.
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


class QAOALandscapeEngine:
    """
    Computes the exact 2D variational energy expectation surface
    <psi(gamma, beta)| H |psi(gamma, beta)> of a p=1 QAOA circuit via full
    2^n statevector simulation over the enumerated bitstring energies of a
    real Ising instance (a subinstance of the bSBA coupling matrix J).
    """
    def __init__(self, coupling_matrix, num_spins=12, grid_points=50):
        self.J = coupling_matrix[:num_spins, :num_spins].copy()
        self.n = num_spins
        self.grid_points = grid_points

        # Enumerate all 2^n bitstring energies E(s) = -0.5 s^T J s
        states = np.arange(2 ** self.n, dtype=np.int64)
        bits = ((states[:, None] >> np.arange(self.n)[None, :]) & 1).astype(np.int8)
        spins = 1.0 - 2.0 * bits.astype(np.float64)
        self.energies = -0.5 * np.einsum('bi,ij,bj->b', spins, self.J, spins, optimize=True)
        self.ground_energy = float(np.min(self.energies))

    def qaoa_expectation(self, gamma, beta):
        """Exact p=1 QAOA expectation via 2^n statevector simulation."""
        dim = 2 ** self.n
        # Problem layer (H_P diagonal): elementwise phase
        state = np.full(dim, 1.0 / np.sqrt(dim), dtype=np.complex128) * np.exp(-1j * gamma * self.energies)

        # Mixer layer e^{-i beta H_M}: identical 2x2 X-rotation per qubit axis
        state = state.reshape((2,) * self.n)
        rot = np.array([[np.cos(beta), -1j * np.sin(beta)],
                        [-1j * np.sin(beta), np.cos(beta)]], dtype=np.complex128)
        for axis in range(self.n):
            state = np.moveaxis(state, axis, 0)
            shape = state.shape
            state = state.reshape(2, -1)
            state = rot @ state
            state = state.reshape(shape)
            state = np.moveaxis(state, 0, axis)

        probs = np.abs(state.reshape(-1)) ** 2
        return float(np.dot(probs, self.energies))

    def generate_energy_surface(self):
        gamma = np.linspace(0, np.pi, self.grid_points)
        beta = np.linspace(0, np.pi / 2, self.grid_points)
        Z = np.zeros((len(beta), len(gamma)))
        for bi, b in enumerate(beta):
            for gi, g in enumerate(gamma):
                Z[bi, gi] = self.qaoa_expectation(g, b)
        G, B = np.meshgrid(gamma, beta)

        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        cp = ax.contourf(G, B, Z, levels=30, cmap='viridis')
        cbar = fig.colorbar(cp, ax=ax)
        cbar.set_label('QAOA Energy Expectation $\\langle \\mathcal{H} \\rangle$', fontweight='bold')

        # Mark optimal parameter minimum of the computed surface
        min_idx = np.unravel_index(np.argmin(Z), Z.shape)
        opt_gamma = gamma[min_idx[1]]
        opt_beta = beta[min_idx[0]]
        ax.plot(opt_gamma, opt_beta, 'r*', markersize=14, label=f'Global Minimum ($\\gamma^*={opt_gamma:.2f}, \\beta^*={opt_beta:.2f}$)')

        # Real multi-start COBYLA optimization trajectory on the exact
        # landscape: first start at the grid minimum, then diverse starts.
        grid_min_idx = np.unravel_index(np.argmin(Z), Z.shape)
        starts = [(gamma[grid_min_idx[1]], beta[grid_min_idx[0]]),
                  (0.6, 0.25), (1.4, 0.7), (0.3, 1.0), (2.2, 0.4)]
        best_result = None
        best_trajectory = None
        for x0 in starts:
            trajectory = [(x0[0], x0[1])]

            def objective(params):
                return self.qaoa_expectation(params[0], params[1])

            def record(xk, *args):
                trajectory.append((float(xk[0]), float(xk[1])))

            res = minimize(objective, x0=list(x0), method="COBYLA",
                           callback=record, options={"maxiter": 200, "rhobeg": 0.4})
            if best_result is None or res.fun < best_result.fun:
                best_result = res
                best_trajectory = np.array(trajectory)
        res = best_result
        traj = best_trajectory
        ax.plot(traj[:, 0], traj[:, 1], 'w-o', markersize=5, linewidth=1.8,
                label='Classical Optimizer Path (COBYLA)')

        ax.set_xlabel('Problem Hamiltonian Angle $\\gamma$', fontweight='bold')
        ax.set_ylabel('Mixer Hamiltonian Angle $\\beta$', fontweight='bold')
        ax.set_title('QAOA $p=1$ Energy Landscape: Exact Statevector Expectation ($n=12$ spins)', fontweight='bold', pad=12)
        ax.legend(loc='upper right', framealpha=0.95)
        plt.tight_layout()
        filepath = os.path.join(output_dir, 'fig_qaoa_energy_landscape_surface.png')
        fig.savefig(filepath, dpi=300)
        plt.close(fig)

        cobyla_energy = float(res.fun)
        approx_ratio = cobyla_energy / self.ground_energy
        return filepath, float(res.x[0]), float(res.x[1]), cobyla_energy, self.ground_energy, approx_ratio


def run_quantum_optimization_benchmark():
    print("=" * 80)
    print("SIMULATED BIFURCATION ALGORITHM & QAOA ENERGY LANDSCAPE BENCHMARK")
    print("Author: Koduru Yagnesh Kumar")
    print("=" * 80)

    sba = SimulatedBifurcationSimulator(num_spins=16, steps=400, dt=0.04)
    p1 = sba.generate_bifurcation_plot()
    print(f"[OK] Simulated Bifurcation Plot saved: {p1}")

    # Real solver comparison on the SAME 16-spin Ising instance
    sba_start = time.perf_counter()
    sba_energy, sba_spins = sba.final_spin_energy()
    sba_ms = (time.perf_counter() - sba_start) * 1000.0

    sa_start = time.perf_counter()
    sa_spins, sa_energy = simulated_annealing_baseline(sba.J, num_sweeps=400)
    sa_ms = (time.perf_counter() - sa_start) * 1000.0
    measured_speedup = sa_ms / sba_ms

    qaoa = QAOALandscapeEngine(coupling_matrix=sba.J, num_spins=12, grid_points=50)
    p2, opt_g, opt_b, cobyla_energy, ground_energy, approx_ratio = qaoa.generate_energy_surface()
    print(f"[OK] QAOA Energy Surface saved: {p2}")
    print(f"     Optimal Variational Parameters: gamma = {opt_g:.3f} rad, beta = {opt_b:.3f} rad")
    print(f"     COBYLA Minimum Energy Expectation: {cobyla_energy:.3f} "
          f"(exact ground state: {ground_energy:.3f} over 2^{qaoa.n} states)")
    print(f"     p=1 QAOA Approximation Ratio: {approx_ratio:.3f}")

    print("-" * 80)
    print("Solver Comparison Verdict (measured on this run):")
    print(f"  - Ballistic SBA: binary phase convergence across {sba.num_spins} coupled spins in {sba_ms:.2f} ms")
    print(f"  - Classical Simulated Annealing (400 sweeps, same instance): {sa_ms:.2f} ms")
    print(f"  - Measured Speedup of bSBA over SA: {measured_speedup:.1f}x")
    print(f"  - bSBA solution energy: {sba_energy:.3f} | SA best energy: {sa_energy:.3f}")
    print("=" * 80)
    return {
        "sba_runtime_ms": sba_ms,
        "sa_runtime_ms": sa_ms,
        "measured_speedup_x": measured_speedup,
        "sba_energy": sba_energy,
        "sa_energy": sa_energy,
        "qaoa_cobyla_energy": cobyla_energy,
        "qaoa_ground_energy": ground_energy,
        "qaoa_approx_ratio": approx_ratio
    }


if __name__ == '__main__':
    run_quantum_optimization_benchmark()
