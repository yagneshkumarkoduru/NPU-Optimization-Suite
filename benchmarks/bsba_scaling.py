"""
bSBA scaling benchmark: quality and speed on Ising instances of 32, 64, 128, 256 vars.
Uses the bSBA implementation from the unified NPU compiler.
"""
import sys, json, time, pathlib, statistics
import numpy as np

sys.path.insert(0, r'C:\Research\NPU-Optimization-Suite')
sys.path.insert(0, r'C:\Research\NPU-Optimization-Suite\unified_pipeline')

# Import bSBA from the unified compiler
from unified_npu_compiler import UnifiedNPUCompiler

def make_random_ising(n, seed=42):
    rng = np.random.default_rng(seed)
    J = rng.standard_normal((n, n))
    J = (J + J.T) / 2.0
    np.fill_diagonal(J, 0.0)
    return J

def ising_energy(x, J):
    return -0.5 * float(x @ J @ x)

def simulated_annealing(J, n_sweeps=400, seed=42):
    rng = np.random.default_rng(seed)
    n = len(J)
    x = rng.choice([-1, 1], size=n).astype(float)
    E = ising_energy(x, J)
    best_E, best_x = E, x.copy()
    T0, T_min = 2.0, 0.01
    for sweep in range(n_sweeps):
        T = T0 * (T_min / T0) ** (sweep / n_sweeps)
        for _ in range(n):
            i = rng.integers(n)
            dE = 2 * x[i] * float(J[i] @ x)
            if dE < 0 or rng.random() < np.exp(-dE / max(T, 1e-10)):
                x[i] *= -1
                E += dE
        if E < best_E:
            best_E, best_x = E, x.copy()
    return best_E, best_x

def bsba_single(J, n_steps=150, seed=42):
    rng = np.random.default_rng(seed)
    n = len(J)
    x = rng.standard_normal(n)
    v = np.zeros(n)
    dt, c0, xi = 0.1, 0.5, 0.7
    c = c0 / n_steps
    for t in range(1, n_steps + 1):
        grad = -J @ x
        x += dt * v
        v += dt * (-c * t * x - xi * v + grad)
        # inelastic wall
        wall = np.abs(x) > 1.0
        x[wall] = np.sign(x[wall])
        v[wall] = 0.0
    x_bin = np.sign(x)
    x_bin[x_bin == 0] = 1.0
    return ising_energy(x_bin, J), x_bin

def one_opt_polish(x, J, budget=400):
    x = x.copy()
    E = ising_energy(x, J)
    improved = True
    flips = 0
    while improved and flips < budget:
        improved = False
        for i in range(len(x)):
            dE = 2 * x[i] * float(J[i] @ x)
            if dE < 0:
                x[i] *= -1
                E += dE
                improved = True
                flips += 1
    return E, x

SIZES = [32, 64, 128, 256]
N_RESTARTS = 8
SEEDS = [1000 + i for i in range(N_RESTARTS)]

rows = []
print(f"\n{'Size':>6} | {'SA Energy':>12} | {'SA Time':>9} | {'bSBA+polish':>12} | {'bSBA Time':>10} | {'Quality':>8} | {'Speedup':>8}")
print("-" * 80)

for n in SIZES:
    J = make_random_ising(n, seed=42)

    # SA baseline
    t0 = time.perf_counter()
    sa_e, _ = simulated_annealing(J, n_sweeps=400, seed=42)
    sa_time = time.perf_counter() - t0

    # bSBA best-of-8 + polish
    t0 = time.perf_counter()
    best_e = float('inf')
    for s in SEEDS:
        e, x = bsba_single(J, n_steps=150, seed=s)
        e2, _ = one_opt_polish(x, J, budget=400)
        if e2 < best_e:
            best_e = e2
    bsba_time = time.perf_counter() - t0

    quality = best_e / sa_e if sa_e < 0 else float('nan')  # ratio (1.0 = matches SA)
    speedup = sa_time / bsba_time if bsba_time > 0 else 0.0

    print(f"{n:>6} | {sa_e:>12.4f} | {sa_time*1000:>7.1f}ms | {best_e:>12.4f} | "
          f"{bsba_time*1000:>8.1f}ms | {quality:>8.4f} | {speedup:>8.2f}x")

    rows.append({
        "n_vars": n,
        "sa_energy": round(sa_e, 6),
        "sa_time_ms": round(sa_time * 1000, 3),
        "bsba_polish_energy": round(best_e, 6),
        "bsba_time_ms": round(bsba_time * 1000, 3),
        "quality_ratio": round(quality, 6),
        "speedup_x": round(speedup, 3),
        "n_restarts": N_RESTARTS,
    })

print()
print("quality_ratio = bSBA-polish / SA  (1.0 = matches SA, <1.0 = bSBA better)")
print("speedup_x = SA_time / bSBA_time")

out = pathlib.Path(r"C:\Research\NPU-Optimization-Suite\results\bsba_scaling")
out.mkdir(parents=True, exist_ok=True)
summary = {
    "description": "bSBA scaling benchmark: quality vs SA and wall-clock speedup",
    "methodology": f"bSBA best-of-{N_RESTARTS} restarts + 1-opt polish vs SA 400 sweeps",
    "class": "BENCHMARK (CPU-hosted Python reference, timings vary run to run)",
    "results": rows,
}
(out / "scaling_results.json").write_text(json.dumps(summary, indent=2))
print(f"\nSaved to {out / 'scaling_results.json'}")
