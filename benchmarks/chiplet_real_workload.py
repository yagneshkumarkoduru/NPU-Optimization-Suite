"""
Chiplet QAP benchmark on real transformer (ViT-B), ResNet-50, and GPT-2 topologies.
Uses realistic non-sequential traffic matrices including residual connections.
Exhaustive 4! = 24 permutation search.
Class: BENCHMARK (analytical model, 0.5 pJ/bit UCIe D2D energy)
"""
import sys, json, itertools, pathlib, time
import numpy as np
sys.path.insert(0, r'C:\Research\NPU-Optimization-Suite')

def exhaustive_qap(traffic, dist):
    n = len(traffic)
    best_cost, worst_cost = float('inf'), 0.0
    best_perm, worst_perm = list(range(n)), list(range(n))
    naive_cost = sum(traffic[u][v] * dist[u][v] for u in range(n) for v in range(n))
    for perm in itertools.permutations(range(n)):
        cost = sum(traffic[u][v] * dist[perm[u]][perm[v]] for u in range(n) for v in range(n))
        if cost < best_cost:
            best_cost, best_perm = cost, list(perm)
        if cost > worst_cost:
            worst_cost, worst_perm = cost, list(perm)
    return best_perm, best_cost, worst_perm, worst_cost, naive_cost

# Linear 4-die topology distance matrix
dist = np.array([[0,1,2,3],[1,0,1,2],[2,1,0,1],[3,2,1,0]])

# --- ViT-B/16 Transformer Block with residual connections ---
# Chiplets: 0=QKV_proj, 1=Attn+Softmax, 2=FFN, 3=LayerNorm+Residual
# Residual path: QKV(0)->LN(3) and Attn(1)->LN(3) create non-trivial cross-traffic
vitb = np.array([
    [  0, 148,   0, 148],  # QKV -> Attn + LN(residual)
    [  0,   0, 148, 148],  # Attn -> FFN + LN(skip)
    [  0,   0,   0, 592],  # FFN -> LN (197*3072=592KB)
    [  0,   0,   0,   0],
])

# --- ResNet-50 Bottleneck Block with skip connection ---
# Chiplets: 0=Conv1x1_down, 1=Conv3x3, 2=Conv1x1_up, 3=BN+Add
# Skip connection: Conv1x1_down(0) directly to BN+Add(3) creates cross-traffic
resnet = np.array([
    [  0, 200,   0, 200],  # Conv1x1_down -> Conv3x3 + skip to BN+Add
    [  0,   0, 200,   0],  # Conv3x3 -> Conv1x1_up
    [  0,   0,   0, 800],  # Conv1x1_up -> BN+Add (4x channel expansion)
    [  0,   0,   0,   0],
])

# --- GPT-2 Multi-Head Attention with KV-Cache ---
# Chiplets: 0=QK_proj, 1=V_proj+KV_cache, 2=Attn_score, 3=Out_proj+Residual
# Bidirectional traffic between QK and Attn_score for incremental decoding
gpt2 = np.array([
    [  0,   0, 320,   0],  # QK -> Attn_score
    [  0,   0, 320,   0],  # V+KV -> Attn_score
    [320, 320,   0, 290],  # Attn_score -> QK (next step) + Out
    [290,   0,   0,   0],  # Out -> QK (next layer residual)
])

results = {}
print("=" * 75)
print("Chiplet QAP on Real Neural Network Topologies (4 chiplets, 24 perms)")
print("=" * 75)

for name, traffic in [
    ("ViT-B/16 Transformer Block (residual + skip connections)",  vitb),
    ("ResNet-50 Bottleneck Block (skip connection)",               resnet),
    ("GPT-2 Multi-Head Attention (KV-cache bidirectional flow)",  gpt2),
]:
    t0 = time.perf_counter()
    bp, bc, wp, wc, nc = exhaustive_qap(traffic, dist)
    elapsed = time.perf_counter() - t0
    vs_naive = (1 - bc/nc)*100 if nc>0 else 0.0
    vs_worst = (1 - bc/wc)*100 if wc>0 else 0.0
    e_naive = nc * 1e6 * 8 * 0.5e-12 * 1e6  # uJ (0.5 pJ/bit)
    e_opt   = bc * 1e6 * 8 * 0.5e-12 * 1e6

    print(f"\n{name}")
    print(f"  Naive (0,1,2,3):      {nc:6,} MB-hops | {e_naive:7.1f} uJ")
    print(f"  Optimal {tuple(bp)}: {bc:6,} MB-hops | {e_opt:7.1f} uJ | {vs_naive:.1f}% vs naive")
    print(f"  Worst   {tuple(wp)}: {wc:6,} MB-hops | {vs_worst:.1f}% gap (opt vs worst)")
    print(f"  Search:  {elapsed*1000:.2f}ms")

    results[name] = {
        "naive_mb_hops": int(nc), "optimal_mb_hops": int(bc),
        "optimal_perm": bp, "worst_mb_hops": int(wc), "worst_perm": wp,
        "reduction_vs_naive_pct": round(vs_naive, 2),
        "reduction_vs_worst_pct": round(vs_worst, 2),
        "energy_naive_uj": round(e_naive, 2), "energy_optimal_uj": round(e_opt, 2),
        "search_ms": round(elapsed*1000, 3),
    }

out = pathlib.Path(r"C:\Research\NPU-Optimization-Suite\results\chiplet_real")
out.mkdir(parents=True, exist_ok=True)
(out / "neural_topology_chiplet.json").write_text(json.dumps({
    "description": "Chiplet QAP on ViT-B, ResNet-50, GPT-2 with residual/skip connections",
    "class": "BENCHMARK (analytical, 0.5 pJ/bit UCIe, linear 4-die topology)",
    "results": results,
}, indent=2))
print(f"\nSaved: {out}/neural_topology_chiplet.json")
