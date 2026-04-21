"""
Deep dive into KL-penalty demographic reweighting.

KL-penalty was the best method (65% win rate). Explore:
1. Smoothing sweep
2. KL-penalty + QAW relevance combination
3. LOO calibration of (smoothing, tau, kl_exponent)
4. Per-question analysis: which questions benefit most?
5. Leave-k-out robustness

Uses precomputed per-cluster distributions (no GPU needed).
"""
import json
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations
from scipy.spatial.distance import jensenshannon
from scipy.special import rel_entr
from sentence_transformers import SentenceTransformer
import pandas as pd

from adaptive_weights import compute_question_topic_relevance
from demographic_reweight import compute_province_distributions

DATA_DIR = "/data/chenhongrui/business/data"
RESULTS_DIR = "/data/chenhongrui/business/results"
EMBEDDING_MODEL = "/data/chenhongrui/models/bge-base-zh-v1.5"


def load_precomputed():
    with open(f"{RESULTS_DIR}/qaw_expanded_L24_A0.1.json") as f:
        data = json.load(f)
    with open(f"{RESULTS_DIR}/all_questions_expanded.json") as f:
        expanded = json.load(f)
    with open(f"{DATA_DIR}/3_cluster_topics.json") as f:
        topics = json.load(f)

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

    return questions, topics, per_cluster


def kl_weights(cluster_prov, survey_prov, smoothing=1.0, exponent=1.0):
    """KL-penalty weights: penalize clusters whose province distribution
    diverges from survey."""
    all_provs = sorted(set().union(*[set(cp.keys()) for cp in cluster_prov.values()],
                                   set(survey_prov.keys())))
    sv_total = sum(survey_prov.values())
    p_s = np.array([(survey_prov.get(prov, 0) + smoothing) / (sv_total + smoothing * len(all_provs))
                    for prov in all_provs])
    p_s /= p_s.sum()

    weights = {}
    for cid, provs in cluster_prov.items():
        c_total = sum(provs.values())
        if c_total < 3:
            weights[str(cid)] = 1.0
            continue
        p_c = np.array([(provs.get(prov, 0) + smoothing) / (c_total + smoothing * len(all_provs))
                       for prov in all_provs])
        p_c /= p_c.sum()
        kl = np.sum(rel_entr(p_s, p_c))
        weights[str(cid)] = np.exp(-exponent * kl)

    mean_w = np.mean(list(weights.values()))
    if mean_w > 0:
        weights = {k: v / mean_w for k, v in weights.items()}
    return weights


def combined_w(base, kl_w, relevance=None, cluster_ids=None, tau=1.0, kl_strength=1.0):
    """Combine base weights, KL penalty, and optional relevance."""
    combined = {}
    for cid, bw in base.items():
        kw = kl_w.get(cid, 1.0) ** kl_strength
        combined[cid] = bw * kw

    if relevance is not None and cluster_ids is not None:
        r = relevance / max(tau, 1e-6)
        r = r - r.max()
        exp_r = np.exp(r)
        sm = exp_r / exp_r.sum()
        for i, cid in enumerate(cluster_ids):
            if cid in combined:
                combined[cid] *= sm[i]

    total = sum(combined.values())
    if total > 0:
        combined = {k: v / total for k, v in combined.items()}
    return combined


def agg(pcq, w):
    n = len(next(iter(pcq.values())))
    a = np.zeros(n)
    tw = 0
    for cid, pmf in pcq.items():
        wt = w.get(cid, 0.0)
        a += wt * pmf
        tw += wt
    return a / tw if tw > 0 else a


def js(td, pred):
    t = np.array(list(td.values()), dtype=float)
    t /= t.sum()
    p = pred / (pred.sum() + 1e-10)
    return float(jensenshannon(t, p) ** 2)


def main():
    print("Loading data...", flush=True)
    questions, topics, per_cluster = load_precomputed()
    cluster_prov, sm_prov, survey_prov, _ = compute_province_distributions(
        f"{DATA_DIR}/2_meaningful_df.csv",
        f"{DATA_DIR}/2025-05-31-10-01-50_EXPORT_CSV_19470858_516_ds_survey_feedback_0.csv",
    )
    encoder = SentenceTransformer(EMBEDDING_MODEL)
    q_texts = [qd["question"] for qd in questions]
    rel_map, cluster_ids = compute_question_topic_relevance(q_texts, topics, encoder)

    # Baseline
    uniform_js = [js(qd["true_distribution"], agg(per_cluster[i], qd["cluster_weights"]))
                  for i, qd in enumerate(questions)]
    print(f"Baseline (uniform): {np.mean(uniform_js):.4f}\n", flush=True)

    # === 1. KL-penalty sweep (smoothing × exponent) ===
    print("=" * 70, flush=True)
    print("1. KL-Penalty Parameter Sweep", flush=True)
    print("=" * 70, flush=True)
    print(f"  {'smooth':>7} {'exp':>5} {'mean_js':>8} {'Δ':>8} {'wins':>6} {'w_std':>7}", flush=True)
    print("  " + "-" * 50, flush=True)

    best_cfg, best_mean = None, float("inf")
    for sm in [0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 3.0, 5.0]:
        for exp in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]:
            kw = kl_weights(cluster_prov, survey_prov, smoothing=sm, exponent=exp)
            vals = list(kw.values())
            kl_js = []
            for i, qd in enumerate(questions):
                cw = combined_w(qd["cluster_weights"], kw, kl_strength=1.0)
                kl_js.append(js(qd["true_distribution"], agg(per_cluster[i], cw)))
            m = np.mean(kl_js)
            imp = np.mean(uniform_js) - m
            wins = sum(1 for k, u in zip(kl_js, uniform_js) if k < u)
            if m < best_mean:
                best_mean, best_cfg = m, (sm, exp)
            print(f"  {sm:>7.2f} {exp:>5.1f} {m:>8.4f} {imp:>+8.4f} {wins:>4}/23 {np.std(vals):>7.3f}",
                  flush=True)

    print(f"\n  Best: smoothing={best_cfg[0]}, exponent={best_cfg[1]}, JS={best_mean:.4f}", flush=True)

    # === 2. KL-penalty + QAW relevance ===
    print("\n" + "=" * 70, flush=True)
    print("2. KL-Penalty + QAW Combined Sweep", flush=True)
    print("=" * 70, flush=True)

    sm_opt, exp_opt = best_cfg
    kw_opt = kl_weights(cluster_prov, survey_prov, smoothing=sm_opt, exponent=exp_opt)

    print(f"  Using KL params: smoothing={sm_opt}, exponent={exp_opt}", flush=True)
    print(f"  {'tau':>6} {'kl_s':>5} {'mean_js':>8} {'Δ':>8} {'wins':>6}", flush=True)
    print("  " + "-" * 42, flush=True)

    best_combo, best_combo_js = None, float("inf")
    for tau in [0.1, 0.2, 0.3, 0.5, 1.0, 5.0]:
        for kl_s in [0.0, 0.5, 1.0, 1.5, 2.0]:
            combo_js = []
            for i, qd in enumerate(questions):
                r = rel_map.get(qd["question"])
                cw = combined_w(qd["cluster_weights"], kw_opt,
                                relevance=r, cluster_ids=cluster_ids,
                                tau=tau, kl_strength=kl_s)
                combo_js.append(js(qd["true_distribution"], agg(per_cluster[i], cw)))
            m = np.mean(combo_js)
            imp = np.mean(uniform_js) - m
            wins = sum(1 for c, u in zip(combo_js, uniform_js) if c < u)
            if m < best_combo_js:
                best_combo_js, best_combo = m, (tau, kl_s)
            print(f"  {tau:>6.1f} {kl_s:>5.1f} {m:>8.4f} {imp:>+8.4f} {wins:>4}/23", flush=True)

    print(f"\n  Best combined: tau={best_combo[0]}, kl_strength={best_combo[1]}, JS={best_combo_js:.4f}", flush=True)

    # === 3. LOO calibration ===
    print("\n" + "=" * 70, flush=True)
    print("3. LOO Calibration (tau, kl_strength)", flush=True)
    print("=" * 70, flush=True)

    n = len(questions)
    tau_cands = [0.1, 0.2, 0.3, 0.5, 1.0, 5.0]
    kl_cands = [0.0, 0.5, 1.0, 1.5, 2.0]
    configs = [(t, k) for t in tau_cands for k in kl_cands]

    folds = []
    for ho in range(n):
        train = [i for i in range(n) if i != ho]
        best, bjs = configs[0], float("inf")
        for tau, kl_s in configs:
            jv = []
            for i in train:
                qd = questions[i]
                r = rel_map.get(qd["question"])
                cw = combined_w(qd["cluster_weights"], kw_opt,
                                relevance=r, cluster_ids=cluster_ids,
                                tau=tau, kl_strength=kl_s)
                jv.append(js(qd["true_distribution"], agg(per_cluster[i], cw)))
            m = np.mean(jv)
            if m < bjs:
                bjs, best = m, (tau, kl_s)

        tau, kl_s = best
        qd = questions[ho]
        r = rel_map.get(qd["question"])
        cw = combined_w(qd["cluster_weights"], kw_opt,
                        relevance=r, cluster_ids=cluster_ids,
                        tau=tau, kl_strength=kl_s)
        pred = agg(per_cluster[ho], cw)
        uni = agg(per_cluster[ho], qd["cluster_weights"])
        folds.append({
            "key": qd["key"], "tau": tau, "kl_s": kl_s,
            "combined": js(qd["true_distribution"], pred),
            "uniform": js(qd["true_distribution"], uni),
        })

    loo_mean = np.mean([f["combined"] for f in folds])
    uni_mean = np.mean([f["uniform"] for f in folds])
    wins = sum(1 for f in folds if f["combined"] < f["uniform"])
    print(f"  LOO combined: {loo_mean:.4f}", flush=True)
    print(f"  Uniform:      {uni_mean:.4f}", flush=True)
    print(f"  Improvement:  {uni_mean - loo_mean:+.4f}", flush=True)
    print(f"  Win rate:     {wins}/{n} ({100*wins/n:.0f}%)", flush=True)
    taus = [f["tau"] for f in folds]
    kls = [f["kl_s"] for f in folds]
    print(f"  Tau: {np.mean(taus):.2f}±{np.std(taus):.2f}, KL_s: {np.mean(kls):.2f}±{np.std(kls):.2f}", flush=True)

    for f in folds:
        d = f["uniform"] - f["combined"]
        print(f"    {f['key']:<18} tau={f['tau']:<5} kl_s={f['kl_s']:<4} "
              f"JS={f['combined']:.4f} uni={f['uniform']:.4f} {'+'if d>0 else ''}{d:.4f}", flush=True)

    # === 4. Leave-k-out ===
    print("\n" + "=" * 70, flush=True)
    print("4. Leave-K-Out Robustness", flush=True)
    print("=" * 70, flush=True)

    for k in [2, 3, 5, 8, 11]:
        combos = list(combinations(range(n), k))
        rng = np.random.RandomState(42)
        if len(combos) > 30:
            idx = rng.choice(len(combos), 30, replace=False)
            combos = [combos[i] for i in idx]

        qaw_all, uni_all = [], []
        for ho_set in combos:
            ho_set = set(ho_set)
            train = [i for i in range(n) if i not in ho_set]
            best, bjs = configs[0], float("inf")
            for tau, kl_s in configs:
                jv = []
                for i in train:
                    qd = questions[i]
                    r = rel_map.get(qd["question"])
                    cw = combined_w(qd["cluster_weights"], kw_opt,
                                    relevance=r, cluster_ids=cluster_ids,
                                    tau=tau, kl_strength=kl_s)
                    jv.append(js(qd["true_distribution"], agg(per_cluster[i], cw)))
                m = np.mean(jv)
                if m < bjs:
                    bjs, best = m, (tau, kl_s)
            tau, kl_s = best
            for ho in ho_set:
                qd = questions[ho]
                r = rel_map.get(qd["question"])
                cw = combined_w(qd["cluster_weights"], kw_opt,
                                relevance=r, cluster_ids=cluster_ids,
                                tau=tau, kl_strength=kl_s)
                qaw_all.append(js(qd["true_distribution"], agg(per_cluster[ho], cw)))
                uni_all.append(js(qd["true_distribution"], agg(per_cluster[ho], qd["cluster_weights"])))

        imp = np.mean(uni_all) - np.mean(qaw_all)
        wr = np.mean([q < u for q, u in zip(qaw_all, uni_all)])
        print(f"  L{k}O: combined={np.mean(qaw_all):.4f} uni={np.mean(uni_all):.4f} "
              f"Δ={imp:+.4f} win={wr:.0%}", flush=True)

    # === 5. Per-question analysis ===
    print("\n" + "=" * 70, flush=True)
    print("5. Per-Question Analysis: Who Benefits?", flush=True)
    print("=" * 70, flush=True)

    tau_best, kl_best = best_combo
    for i, qd in enumerate(questions):
        r = rel_map.get(qd["question"])
        cw = combined_w(qd["cluster_weights"], kw_opt,
                        relevance=r, cluster_ids=cluster_ids,
                        tau=tau_best, kl_strength=kl_best)
        combo_j = js(qd["true_distribution"], agg(per_cluster[i], cw))
        uni_j = uniform_js[i]
        d = uni_j - combo_j
        # Also show which clusters changed weight most
        top_changes = sorted(
            [(cid, cw.get(cid, 0) - qd["cluster_weights"].get(cid, 0))
             for cid in set(list(cw.keys()) + list(qd["cluster_weights"].keys()))],
            key=lambda x: -abs(x[1])
        )[:3]
        changes_str = " ".join(f"[{c}]{d:+.3f}" for c, d in top_changes)
        print(f"  {qd['key']:<18} uni={uni_j:.4f} combo={combo_j:.4f} "
              f"{'+'if d>0 else ''}{d:.4f} | {changes_str}", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
