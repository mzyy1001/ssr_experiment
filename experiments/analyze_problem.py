"""Diagnose why steering doesn't beat SSR-only."""
import json, numpy as np
from scipy.spatial.distance import jensenshannon

RESULTS_DIR = "/data/chenhongrui/business/results"
DATA_DIR = "/data/chenhongrui/business/data"

with open(f"{RESULTS_DIR}/qaw_expanded_L24_A0.1.json") as f:
    data = json.load(f)
with open(f"{RESULTS_DIR}/all_questions_expanded.json") as f:
    expanded = json.load(f)

questions = []
for sk, sv in expanded.items():
    for sub, qs in sv.items():
        for qid, qd in qs.items():
            questions.append({
                "key": f"{sub}/{qid}",
                "question": qd["question"],
                "options": qd["options"],
                "true_distribution": qd["true_distribution"],
                "cluster_weights": qd["cluster_weights"],
            })

per_cluster = {}
for qi_str, cpmfs in data["per_cluster_pmfs"].items():
    per_cluster[int(qi_str)] = {cid: np.array(pmf) for cid, pmf in cpmfs.items()}

# ===== Analysis 1: Do different clusters produce different answers? =====
print("=" * 70)
print("Analysis 1: Inter-cluster diversity")
print("  If clusters produce similar distributions, steering adds noise not signal.")
print("=" * 70)
for qi in range(len(questions)):
    qd = questions[qi]
    pmfs = list(per_cluster[qi].values())
    if len(pmfs) < 2:
        continue
    pairwise = []
    for i in range(len(pmfs)):
        for j in range(i+1, len(pmfs)):
            pairwise.append(jensenshannon(pmfs[i], pmfs[j])**2)
    entropies = [-np.sum(p * np.log(p + 1e-10)) for p in pmfs]
    print(f"  {qd['key']:<18} inter-JS={np.mean(pairwise):.4f} "
          f"cluster_entropy={np.mean(entropies):.2f} "
          f"max_entropy={np.log(len(qd['options'])):.2f}")

# ===== Analysis 2: Per-cluster argmax - do clusters agree? =====
print()
print("=" * 70)
print("Analysis 2: Do all clusters pick the same top answer?")
print("  If yes, steering doesn't differentiate clusters.")
print("=" * 70)
for qi in range(len(questions)):
    qd = questions[qi]
    argmax_counts = {}
    for cid, pmf in per_cluster[qi].items():
        am = int(np.argmax(pmf))
        argmax_counts[am] = argmax_counts.get(am, 0) + 1
    n = len(per_cluster[qi])
    dominant = max(argmax_counts.values())
    print(f"  {qd['key']:<18} clusters={n} argmax={argmax_counts} "
          f"agreement={dominant}/{n} ({100*dominant/n:.0f}%)")

# ===== Analysis 3: Entropy of steered outputs =====
print()
print("=" * 70)
print("Analysis 3: Are steered outputs near-uniform (high entropy)?")
print("  If cluster distributions are near-uniform, steering isn't producing")
print("  opinionated outputs — just noise around the prior.")
print("=" * 70)
for qi in range(len(questions)):
    qd = questions[qi]
    pmfs = list(per_cluster[qi].values())
    mean_ent = np.mean([-np.sum(p * np.log(p + 1e-10)) for p in pmfs])
    true = np.array(list(qd["true_distribution"].values()), dtype=float)
    true /= true.sum()
    true_ent = -np.sum(true * np.log(true + 1e-10))
    max_ent = np.log(len(qd["options"]))
    ratio = mean_ent / max_ent
    print(f"  {qd['key']:<18} cluster_ent={mean_ent:.2f}/{max_ent:.2f} ({ratio:.0%}) "
          f"true_ent={true_ent:.2f}")

# ===== Analysis 4: Prediction vs truth =====
print()
print("=" * 70)
print("Analysis 4: Aggregated prediction vs ground truth")
print("=" * 70)
for qi in range(len(questions)):
    qd = questions[qi]
    agg_pmf = np.zeros(len(qd["options"]))
    tw = 0
    for cid, pmf in per_cluster[qi].items():
        w = qd["cluster_weights"].get(cid, 0)
        agg_pmf += w * pmf
        tw += w
    agg_pmf /= tw
    true = np.array(list(qd["true_distribution"].values()), dtype=float)
    true /= true.sum()
    js_val = jensenshannon(true, agg_pmf)**2

    # Which option has the biggest error?
    diff = agg_pmf - true
    worst_idx = np.argmax(np.abs(diff))
    opts = qd["options"]
    print(f"  {qd['key']:<18} JS={js_val:.4f}")
    print(f"    true: [{', '.join(f'{t:.3f}' for t in true)}]")
    print(f"    pred: [{', '.join(f'{p:.3f}' for p in agg_pmf)}]")
    print(f"    worst: option[{worst_idx}]={opts[worst_idx][:20]} err={diff[worst_idx]:+.3f}")

# ===== Analysis 5: Variance across generations =====
print()
print("=" * 70)
print("Analysis 5: How stable are per-cluster predictions?")
print("  With only 3 generations/cluster, variance could dominate signal.")
print("=" * 70)
# We only have the MEAN of 3 generations, not individual ones.
# But we can estimate from inter-cluster variance vs intra-cluster variance
for qi in range(min(6, len(questions))):
    qd = questions[qi]
    pmfs = np.array(list(per_cluster[qi].values()))
    # Between-cluster variance
    between_var = np.mean(np.var(pmfs, axis=0))
    # Each pmf is mean of 3, so intra-cluster variance is unknown
    # But we can check if pmfs are "tight" around a common center
    center = pmfs.mean(axis=0)
    dist_from_center = [jensenshannon(p, center)**2 for p in pmfs]
    print(f"  {qd['key']:<18} between_var={between_var:.4f} "
          f"mean_dist_from_center={np.mean(dist_from_center):.4f} "
          f"max={np.max(dist_from_center):.4f}")

# ===== Analysis 6: The key question - does cluster identity predict anything? =====
print()
print("=" * 70)
print("Analysis 6: Correlation between cluster weight and prediction accuracy")
print("  If high-weight clusters produce worse predictions, weighting is the problem.")
print("=" * 70)
from scipy.stats import spearmanr
for qi in range(min(6, len(questions))):
    qd = questions[qi]
    true = np.array(list(qd["true_distribution"].values()), dtype=float)
    true /= true.sum()
    weights, per_cluster_js = [], []
    for cid, pmf in per_cluster[qi].items():
        w = qd["cluster_weights"].get(cid, 0)
        js_val = jensenshannon(true, pmf)**2
        weights.append(w)
        per_cluster_js.append(js_val)
    corr, pval = spearmanr(weights, per_cluster_js)
    print(f"  {qd['key']:<18} corr(weight, JS)={corr:+.3f} p={pval:.3f} "
          f"{'*** BAD: high-weight clusters predict worse' if corr > 0.3 else ''}")

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
