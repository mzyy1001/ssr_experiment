"""
Combined Demographic Post-Stratification + QAW experiment.

Tests the hypothesis that correcting for province-level selection bias
improves survey distribution reconstruction, and that combining demographic
correction with question-topic relevance (QAW) yields further gains.

Uses pre-computed per-cluster steered distributions from the expanded run
to avoid redundant generation.

Run on chen server:
  source /data/anaconda3/etc/profile.d/conda.sh && conda activate llama_qwen
  cd /data/chenhongrui/business/experiments
  python -u run_demo_qaw.py
"""
import json
import numpy as np
from pathlib import Path
from itertools import combinations
from scipy.spatial.distance import jensenshannon
from sentence_transformers import SentenceTransformer

from adaptive_weights import compute_question_topic_relevance, adaptive_weights
from demographic_reweight import (
    compute_province_distributions,
    compute_demographic_weights,
    combined_weights,
)

DATA_DIR = "/data/chenhongrui/business/data"
RESULTS_DIR = "/data/chenhongrui/business/results"
EMBEDDING_MODEL = "/data/chenhongrui/models/bge-base-zh-v1.5"

TAU_CANDIDATES = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0]
DEMO_STRENGTH_CANDIDATES = [0.0, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]


def load_precomputed():
    """Load per-cluster distributions from expanded experiment."""
    with open(f"{RESULTS_DIR}/qaw_expanded_L24_A0.1.json") as f:
        data = json.load(f)

    with open(f"{RESULTS_DIR}/all_questions_expanded.json") as f:
        expanded = json.load(f)

    with open(f"{DATA_DIR}/3_cluster_topics.json") as f:
        topics = json.load(f)

    # Reconstruct questions list
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

    # Load per-cluster pmfs
    per_cluster = {}
    for qi_str, cpmfs in data["per_cluster_pmfs"].items():
        per_cluster[int(qi_str)] = {
            cid: np.array(pmf) for cid, pmf in cpmfs.items()
        }

    return questions, topics, per_cluster


def agg(per_cluster_q, weights):
    n = len(next(iter(per_cluster_q.values())))
    a = np.zeros(n)
    tw = 0
    for cid, pmf in per_cluster_q.items():
        w = weights.get(cid, 0.0)
        a += w * pmf
        tw += w
    if tw > 0:
        a /= tw
    return a


def js(true_dist, pred):
    t = np.array(list(true_dist.values()), dtype=float)
    t /= t.sum()
    p = pred / (pred.sum() + 1e-10)
    return float(jensenshannon(t, p) ** 2)


def evaluate_config(questions, per_cluster, demo_weights, relevance_map,
                    cluster_ids, tau, demo_strength):
    """Evaluate a (tau, demo_strength) config on all questions."""
    js_vals = []
    for i, qd in enumerate(questions):
        r = relevance_map.get(qd["question"])
        cw = combined_weights(
            qd["cluster_weights"], demo_weights,
            relevance_scores=r, cluster_ids=cluster_ids,
            tau=tau, demo_strength=demo_strength,
        )
        pred = agg(per_cluster[i], cw)
        js_vals.append(js(qd["true_distribution"], pred))
    return js_vals


def loo_calibrate(questions, per_cluster, demo_weights, relevance_map, cluster_ids):
    """LOO calibration over (tau, demo_strength) grid."""
    n = len(questions)
    configs = [(t, d) for t in TAU_CANDIDATES for d in DEMO_STRENGTH_CANDIDATES]

    fold_results = []
    for ho in range(n):
        train = [i for i in range(n) if i != ho]
        best_cfg, best_js = configs[0], float("inf")

        for tau, ds in configs:
            js_vals = []
            for i in train:
                qd = questions[i]
                r = relevance_map.get(qd["question"])
                cw = combined_weights(
                    qd["cluster_weights"], demo_weights,
                    relevance_scores=r, cluster_ids=cluster_ids,
                    tau=tau, demo_strength=ds,
                )
                pred = agg(per_cluster[i], cw)
                js_vals.append(js(qd["true_distribution"], pred))
            m = np.mean(js_vals)
            if m < best_js:
                best_js, best_cfg = m, (tau, ds)

        # Evaluate on held-out
        tau, ds = best_cfg
        qd = questions[ho]
        r = relevance_map.get(qd["question"])
        cw = combined_weights(
            qd["cluster_weights"], demo_weights,
            relevance_scores=r, cluster_ids=cluster_ids,
            tau=tau, demo_strength=ds,
        )
        pred = agg(per_cluster[ho], cw)
        uni_pred = agg(per_cluster[ho], qd["cluster_weights"])

        fold_results.append({
            "key": qd["key"],
            "question": qd["question"][:40],
            "tau": tau,
            "demo_strength": ds,
            "combined_js": js(qd["true_distribution"], pred),
            "uniform_js": js(qd["true_distribution"], uni_pred),
        })

    return fold_results


def main():
    print("Loading precomputed data...", flush=True)
    questions, topics, per_cluster = load_precomputed()
    print(f"Loaded {len(questions)} questions with per-cluster distributions", flush=True)

    # Compute demographic weights
    print("\n=== Demographic Post-Stratification ===", flush=True)
    cluster_prov, sm_prov, survey_prov, survey_age = compute_province_distributions(
        f"{DATA_DIR}/2_meaningful_df.csv",
        f"{DATA_DIR}/2025-05-31-10-01-50_EXPORT_CSV_19470858_516_ds_survey_feedback_0.csv",
    )

    # Try all three methods
    for method in ["global_is", "cluster_is", "kl_penalty"]:
        dw = compute_demographic_weights(cluster_prov, sm_prov, survey_prov,
                                         method=method, smoothing=1.0)
        vals = list(dw.values())
        print(f"  {method}: mean={np.mean(vals):.3f} std={np.std(vals):.3f} "
              f"range=[{min(vals):.3f}, {max(vals):.3f}]", flush=True)

    # Use cluster_is as default
    demo_weights = compute_demographic_weights(
        cluster_prov, sm_prov, survey_prov, method="cluster_is", smoothing=1.0,
    )

    # Compute question-topic relevance
    print("\n=== Question-Topic Relevance ===", flush=True)
    encoder = SentenceTransformer(EMBEDDING_MODEL)
    q_texts = [qd["question"] for qd in questions]
    relevance_map, cluster_ids = compute_question_topic_relevance(q_texts, topics, encoder)

    # === Experiment 1: Demo-only reweighting (no QAW) ===
    print("\n" + "=" * 70, flush=True)
    print("Experiment 1: Demographic Post-Stratification Only", flush=True)
    print("=" * 70, flush=True)

    uniform_js = []
    for i, qd in enumerate(questions):
        pred = agg(per_cluster[i], qd["cluster_weights"])
        uniform_js.append(js(qd["true_distribution"], pred))
    print(f"  Baseline (uniform): mean JS = {np.mean(uniform_js):.4f}", flush=True)

    for ds in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
        demo_js = []
        for i, qd in enumerate(questions):
            cw = combined_weights(qd["cluster_weights"], demo_weights,
                                  demo_strength=ds)
            pred = agg(per_cluster[i], cw)
            demo_js.append(js(qd["true_distribution"], pred))
        imp = np.mean(uniform_js) - np.mean(demo_js)
        wins = sum(1 for d, u in zip(demo_js, uniform_js) if d < u)
        print(f"  Demo (strength={ds}): mean JS = {np.mean(demo_js):.4f} "
              f"Δ={imp:+.4f} wins={wins}/{len(questions)}", flush=True)

    # === Experiment 2: QAW-only (no demo) - tau sweep ===
    print("\n" + "=" * 70, flush=True)
    print("Experiment 2: QAW-Only (question relevance, no demo correction)", flush=True)
    print("=" * 70, flush=True)

    for tau in TAU_CANDIDATES:
        qaw_js = []
        for i, qd in enumerate(questions):
            r = relevance_map.get(qd["question"])
            cw = combined_weights(qd["cluster_weights"], demo_weights,
                                  relevance_scores=r, cluster_ids=cluster_ids,
                                  tau=tau, demo_strength=0.0)  # no demo
            pred = agg(per_cluster[i], cw)
            qaw_js.append(js(qd["true_distribution"], pred))
        imp = np.mean(uniform_js) - np.mean(qaw_js)
        print(f"  QAW tau={tau}: mean JS = {np.mean(qaw_js):.4f} Δ={imp:+.4f}", flush=True)

    # === Experiment 3: Combined Demo + QAW ===
    print("\n" + "=" * 70, flush=True)
    print("Experiment 3: Combined (Demo + QAW) Grid Search", flush=True)
    print("=" * 70, flush=True)

    best_cfg, best_js_val = None, float("inf")
    print(f"  {'tau':>6} {'demo_s':>6} {'mean_js':>8} {'Δ':>8} {'wins':>6}", flush=True)
    print("  " + "-" * 40, flush=True)

    for tau in [0.1, 0.3, 0.5, 1.0, 5.0]:
        for ds in [0.0, 0.5, 1.0, 1.5]:
            combo_js = evaluate_config(questions, per_cluster, demo_weights,
                                       relevance_map, cluster_ids, tau, ds)
            m = np.mean(combo_js)
            imp = np.mean(uniform_js) - m
            wins = sum(1 for c, u in zip(combo_js, uniform_js) if c < u)
            if m < best_js_val:
                best_js_val, best_cfg = m, (tau, ds)
            print(f"  {tau:>6} {ds:>6.1f} {m:>8.4f} {imp:>+8.4f} {wins:>4}/{len(questions)}", flush=True)

    print(f"\n  Best: tau={best_cfg[0]}, demo_strength={best_cfg[1]}, "
          f"JS={best_js_val:.4f} (uniform={np.mean(uniform_js):.4f})", flush=True)

    # === Experiment 4: LOO Calibration of (tau, demo_strength) ===
    print("\n" + "=" * 70, flush=True)
    print("Experiment 4: LOO Calibration (joint tau + demo_strength)", flush=True)
    print("=" * 70, flush=True)

    loo = loo_calibrate(questions, per_cluster, demo_weights, relevance_map, cluster_ids)

    loo_mean = np.mean([r["combined_js"] for r in loo])
    uni_mean = np.mean([r["uniform_js"] for r in loo])
    wins = sum(1 for r in loo if r["combined_js"] < r["uniform_js"])
    taus = [r["tau"] for r in loo]
    dss = [r["demo_strength"] for r in loo]

    print(f"  Combined LOO: {loo_mean:.4f}", flush=True)
    print(f"  Uniform:      {uni_mean:.4f}", flush=True)
    print(f"  Improvement:  {uni_mean - loo_mean:+.4f}", flush=True)
    print(f"  Win rate:     {wins}/{len(loo)} ({100*wins/len(loo):.0f}%)", flush=True)
    print(f"  Tau: mean={np.mean(taus):.2f} std={np.std(taus):.2f}", flush=True)
    print(f"  Demo_s: mean={np.mean(dss):.2f} std={np.std(dss):.2f}", flush=True)

    print("\n  Per-question:", flush=True)
    for r in loo:
        delta = r["uniform_js"] - r["combined_js"]
        print(f"    {r['key']:<18} tau={r['tau']:<5} ds={r['demo_strength']:<4} "
              f"JS={r['combined_js']:.4f} uni={r['uniform_js']:.4f} "
              f"{'+'if delta>0 else ''}{delta:.4f}", flush=True)

    # === Experiment 5: Demo methods comparison ===
    print("\n" + "=" * 70, flush=True)
    print("Experiment 5: Compare Demo Methods (global_is, cluster_is, kl_penalty)", flush=True)
    print("=" * 70, flush=True)

    for method in ["global_is", "cluster_is", "kl_penalty"]:
        for smoothing in [0.1, 1.0, 5.0]:
            dw = compute_demographic_weights(cluster_prov, sm_prov, survey_prov,
                                             method=method, smoothing=smoothing)
            demo_js = []
            for i, qd in enumerate(questions):
                cw = combined_weights(qd["cluster_weights"], dw, demo_strength=1.0)
                pred = agg(per_cluster[i], cw)
                demo_js.append(js(qd["true_distribution"], pred))
            imp = np.mean(uniform_js) - np.mean(demo_js)
            wins = sum(1 for d, u in zip(demo_js, uniform_js) if d < u)
            print(f"  {method} s={smoothing}: JS={np.mean(demo_js):.4f} "
                  f"Δ={imp:+.4f} wins={wins}/{len(questions)}", flush=True)

    # Save results
    results = {
        "n_questions": len(questions),
        "uniform_mean_js": float(np.mean(uniform_js)),
        "uniform_per_q": [float(j) for j in uniform_js],
        "best_global": {"tau": best_cfg[0], "demo_strength": best_cfg[1],
                        "js": float(best_js_val)},
        "loo": {
            "mean_js": float(loo_mean),
            "improvement": float(uni_mean - loo_mean),
            "win_rate": wins / len(loo),
            "per_question": loo,
        },
    }

    out = f"{RESULTS_DIR}/demo_qaw_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nResults saved to {out}", flush=True)


if __name__ == "__main__":
    main()
