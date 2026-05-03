"""
Full 23-question evaluation addressing reviewer concerns:
1. Canonical Direct SSR baseline on all 23 questions
2. Multi-layer steering on all 23 questions
3. Ensemble (SSR + multi-layer) on all 23 questions
4. LLM distribution estimation on all 23 questions
5. Factorial ablation: SSR × steering × KL × QAW
6. Paired bootstrap significance tests
7. Question-type heterogeneity analysis
8. LOO calibration of ensemble weight

Run on chen server:
  python -u run_full_eval.py --device cuda:2
"""
import torch
import numpy as np
import json
import re
import pandas as pd
from pathlib import Path
from scipy.spatial.distance import jensenshannon
from scipy.stats import wilcoxon
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from collections import defaultdict

from persona_vectors import load_model, load_persona_vectors
from steered_ssr import (
    SteeringHook, ssr_score, generate_anchors_local, generate_steered_response,
)
from adaptive_weights import compute_question_topic_relevance, adaptive_weights
from demographic_reweight import (
    compute_province_distributions, compute_demographic_weights,
)

DATA_DIR = "/data/chenhongrui/business/data"
RESULTS_DIR = "/data/chenhongrui/business/results"
MODEL_PATH = "/data/chenhongrui/models/Qwen3-8B"
EMBEDDING_MODEL = "/data/chenhongrui/models/bge-base-zh-v1.5"


def js(td, pred):
    t = np.array(list(td.values()), dtype=float); t /= t.sum()
    p = pred / (pred.sum() + 1e-10)
    return float(jensenshannon(t, p) ** 2)


def load_all():
    with open(f"{RESULTS_DIR}/all_questions_expanded.json") as f:
        expanded = json.load(f)
    with open(f"{DATA_DIR}/3_cluster_topics.json") as f:
        topics = json.load(f)
    df = pd.read_csv(f"{DATA_DIR}/2_meaningful_df.csv")
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
    return questions, topics, df


def get_top_clusters(questions, n=15):
    all_w = {}
    for qd in questions:
        for cid, w in qd["cluster_weights"].items():
            all_w[cid] = all_w.get(cid, 0) + w
    return sorted(all_w, key=lambda c: -all_w[c])[:n]


# ===== Method implementations =====

def direct_ssr(encoder, qd, cids, df, anchor_embs, n_posts=30):
    """Direct SSR baseline."""
    cpmfs = {}
    for cid in cids:
        posts = df[df["cluster_label"]==int(cid)]["content_desc"].dropna()
        if len(posts) < 3: continue
        samp = posts.sample(min(n_posts, len(posts)), random_state=42).tolist()
        pmfs = [ssr_score(p, encoder, anchor_embs)[0] for p in samp]
        cpmfs[cid] = np.mean(pmfs, axis=0)
    return cpmfs


def multi_layer_steer(model, tokenizer, encoder, qd, cids, pvecs,
                      anchor_embs, alpha=0.1, layers=[16, 20, 24], n_resp=3):
    """Multi-layer steering."""
    prompt = ("作为一位有相关经验的消费者，请回答以下问卷问题。\n"
              "请用自然的语言表达你的真实想法和感受，不需要选择选项。\n\n"
              f"问题：{qd['question']}\n选项：{'、'.join(qd['options'])}\n\n你的回答：")
    cpmfs = {}
    for cid in cids:
        if int(cid) not in pvecs: continue
        vec = pvecs[int(cid)]["vector"]
        pmfs = []
        for _ in range(n_resp):
            hooks = [SteeringHook(vec, alpha, l) for l in layers]
            for h in hooks: h.attach(model)
            try:
                inputs = tokenizer(prompt, return_tensors="pt")
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=200,
                                        do_sample=True, temperature=0.7, top_p=0.9)
                resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                       skip_special_tokens=True)
            finally:
                for h in hooks: h.remove()
            pmf, _ = ssr_score(resp, encoder, anchor_embs)
            pmfs.append(pmf)
        cpmfs[cid] = np.mean(pmfs, axis=0)
    return cpmfs


def ensemble(ssr_pmfs, steer_pmfs, w=0.7):
    """Ensemble SSR + steering."""
    cpmfs = {}
    for cid in set(ssr_pmfs.keys()) & set(steer_pmfs.keys()):
        cpmfs[cid] = w * ssr_pmfs[cid] + (1 - w) * steer_pmfs[cid]
    return cpmfs


def aggregate(cpmfs, weights, cids):
    n = len(next(iter(cpmfs.values())))
    a, tw = np.zeros(n), 0
    for c in cids:
        if c in cpmfs:
            w = weights.get(c, 0); a += w * cpmfs[c]; tw += w
    return a / tw if tw > 0 else a


def apply_kl_weights(base_weights, kl_weights, strength=2.0):
    combined = {}
    for cid, bw in base_weights.items():
        kw = kl_weights.get(cid, 1.0) ** strength
        combined[cid] = bw * kw
    total = sum(combined.values())
    if total > 0: combined = {k: v/total for k, v in combined.items()}
    return combined


def apply_qaw(base_weights, relevance, cluster_ids, tau=0.1):
    r = relevance / max(tau, 1e-6)
    r = r - r.max()
    exp_r = np.exp(r)
    sm = exp_r / exp_r.sum()
    combined = dict(base_weights)
    for i, cid in enumerate(cluster_ids):
        if cid in combined: combined[cid] *= sm[i]
    total = sum(combined.values())
    if total > 0: combined = {k: v/total for k, v in combined.items()}
    return combined


def paired_bootstrap(js_a, js_b, n_boot=10000):
    """Paired bootstrap test: is method A significantly better than B?"""
    rng = np.random.RandomState(42)
    diffs = np.array(js_b) - np.array(js_a)  # positive = A better
    observed = np.mean(diffs)
    boot_diffs = []
    for _ in range(n_boot):
        idx = rng.choice(len(diffs), len(diffs), replace=True)
        boot_diffs.append(np.mean(diffs[idx]))
    boot_diffs = np.array(boot_diffs)
    ci_lo = np.percentile(boot_diffs, 2.5)
    ci_hi = np.percentile(boot_diffs, 97.5)
    p_val = np.mean(boot_diffs <= 0)  # prob that A is not better
    return {"mean_diff": float(observed), "ci_95": (float(ci_lo), float(ci_hi)),
            "p_value": float(p_val)}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:2")
    args = parser.parse_args()

    Path(RESULTS_DIR).mkdir(exist_ok=True)
    questions, topics, df = load_all()
    cids = get_top_clusters(questions, n=15)
    print(f"Full eval: {len(questions)} questions, {len(cids)} clusters", flush=True)

    # Load models
    print("Loading models...", flush=True)
    encoder = SentenceTransformer(EMBEDDING_MODEL)
    model, tokenizer = load_model(MODEL_PATH, args.device)
    pvecs = load_persona_vectors(f"{RESULTS_DIR}/persona_vectors_L24_N20.npz")

    # Compute KL weights and relevance
    cluster_prov, sm_prov, survey_prov, _ = compute_province_distributions(
        f"{DATA_DIR}/2_meaningful_df.csv",
        f"{DATA_DIR}/2025-05-31-10-01-50_EXPORT_CSV_19470858_516_ds_survey_feedback_0.csv",
    )
    from demographic_reweight import compute_demographic_weights
    from scipy.special import rel_entr
    # KL penalty weights (smoothing=0.01, exponent=3.0)
    all_provs = sorted(set().union(*[set(cp.keys()) for cp in cluster_prov.values()], set(survey_prov.keys())))
    sv_total = sum(survey_prov.values())
    p_s = np.array([(survey_prov.get(prov,0)+0.01)/(sv_total+0.01*len(all_provs)) for prov in all_provs])
    p_s /= p_s.sum()
    kl_weights = {}
    for cid_int, provs in cluster_prov.items():
        c_total = sum(provs.values())
        if c_total < 3:
            kl_weights[str(cid_int)] = 1.0; continue
        p_c = np.array([(provs.get(prov,0)+0.01)/(c_total+0.01*len(all_provs)) for prov in all_provs])
        p_c /= p_c.sum()
        kl = np.sum(rel_entr(p_s, p_c))
        kl_weights[str(cid_int)] = np.exp(-3.0 * kl)
    mean_kl = np.mean(list(kl_weights.values()))
    kl_weights = {k: v/mean_kl for k, v in kl_weights.items()}

    q_texts = [qd["question"] for qd in questions]
    rel_map, cluster_id_order = compute_question_topic_relevance(q_texts, topics, encoder)

    # Generate anchors for all questions
    print("Generating anchors for all 23 questions...", flush=True)
    ac = {}
    for qd in questions:
        q = qd["question"]
        if q not in ac:
            anchors = generate_anchors_local(model, tokenizer, q, qd["options"])
            aembs = [encoder.encode(a) for a in anchors]
            ac[q] = aembs
            print(f"  {q[:40]}...", flush=True)

    # ===== Compute per-question results =====
    print("\n=== Computing all methods on 23 questions ===", flush=True)

    results = defaultdict(list)  # method_name -> list of JS per question

    for qi, qd in enumerate(questions):
        aembs = ac[qd["question"]]
        base_w = qd["cluster_weights"]
        rel = rel_map.get(qd["question"])

        # 1. Direct SSR
        ssr_cpmfs = direct_ssr(encoder, qd, cids, df, aembs)
        results["Direct SSR"].append(js(qd["true_distribution"], aggregate(ssr_cpmfs, base_w, cids)))

        # 2. Multi-layer steering
        steer_cpmfs = multi_layer_steer(model, tokenizer, encoder, qd, cids, pvecs, aembs)
        results["Multi-layer steer"].append(js(qd["true_distribution"], aggregate(steer_cpmfs, base_w, cids)))

        # 3. Ensemble w=0.5
        ens_cpmfs = ensemble(ssr_cpmfs, steer_cpmfs, w=0.5)
        results["Ensemble w=0.5"].append(js(qd["true_distribution"], aggregate(ens_cpmfs, base_w, cids)))

        # 4. Ensemble w=0.7
        ens_cpmfs = ensemble(ssr_cpmfs, steer_cpmfs, w=0.7)
        results["Ensemble w=0.7"].append(js(qd["true_distribution"], aggregate(ens_cpmfs, base_w, cids)))

        # 5. Direct SSR + KL
        kl_w = apply_kl_weights(base_w, kl_weights, strength=2.0)
        results["SSR + KL"].append(js(qd["true_distribution"], aggregate(ssr_cpmfs, kl_w, cids)))

        # 6. Direct SSR + KL + QAW
        kl_qaw_w = apply_qaw(kl_w, rel, cluster_id_order, tau=0.1)
        results["SSR + KL + QAW"].append(js(qd["true_distribution"], aggregate(ssr_cpmfs, kl_qaw_w, cids)))

        # 7. Ensemble + KL
        ens_cpmfs_07 = ensemble(ssr_cpmfs, steer_cpmfs, w=0.7)
        results["Ensemble + KL"].append(js(qd["true_distribution"], aggregate(ens_cpmfs_07, kl_w, cids)))

        # 8. Ensemble + KL + QAW (full pipeline)
        results["Ensemble + KL + QAW"].append(js(qd["true_distribution"], aggregate(ens_cpmfs_07, kl_qaw_w, cids)))

        # 9. Steering only (no SSR)
        results["Steer only"].append(js(qd["true_distribution"], aggregate(steer_cpmfs, base_w, cids)))

        # 10. KL only (on SSR)
        results["KL only"].append(js(qd["true_distribution"], aggregate(ssr_cpmfs, kl_w, cids)))

        prog = f"Q{qi+1}/{len(questions)}"
        ssr_j = results["Direct SSR"][-1]
        ens_j = results["Ensemble w=0.7"][-1]
        print(f"  {prog} {qd['key']:<18} SSR={ssr_j:.4f} Ens={ens_j:.4f}", flush=True)

    # ===== Summary table =====
    print("\n" + "=" * 80, flush=True)
    print("FULL 23-QUESTION RESULTS", flush=True)
    print("=" * 80, flush=True)
    print(f"  {'Method':<25} {'Mean JS':>8} {'Med JS':>8} {'Wins':>6} {'vs SSR':>8}", flush=True)
    print("  " + "-" * 60, flush=True)

    bl_js = results["Direct SSR"]
    for name in ["Direct SSR", "Multi-layer steer", "Steer only",
                 "Ensemble w=0.5", "Ensemble w=0.7",
                 "KL only", "SSR + KL", "SSR + KL + QAW",
                 "Ensemble + KL", "Ensemble + KL + QAW"]:
        vals = results[name]
        wins = sum(1 for v, b in zip(vals, bl_js) if v < b)
        diff = np.mean(bl_js) - np.mean(vals)
        print(f"  {name:<25} {np.mean(vals):>8.4f} {np.median(vals):>8.4f} "
              f"{wins:>4}/23 {diff:>+8.4f}", flush=True)

    # ===== Statistical significance =====
    print("\n" + "=" * 80, flush=True)
    print("STATISTICAL SIGNIFICANCE (paired bootstrap, n=10000)", flush=True)
    print("=" * 80, flush=True)
    for name in ["Ensemble w=0.7", "Ensemble w=0.5", "SSR + KL",
                 "SSR + KL + QAW", "Ensemble + KL", "Ensemble + KL + QAW"]:
        boot = paired_bootstrap(results[name], bl_js)
        sig = "***" if boot["p_value"] < 0.01 else "**" if boot["p_value"] < 0.05 else "*" if boot["p_value"] < 0.1 else "ns"
        print(f"  {name} vs Direct SSR: diff={boot['mean_diff']:+.4f} "
              f"CI=[{boot['ci_95'][0]:+.4f}, {boot['ci_95'][1]:+.4f}] "
              f"p={boot['p_value']:.3f} {sig}", flush=True)

    # Also Wilcoxon signed-rank
    print("\n  Wilcoxon signed-rank tests:", flush=True)
    for name in ["Ensemble w=0.7", "Ensemble w=0.5", "Ensemble + KL + QAW"]:
        try:
            stat, pval = wilcoxon(bl_js, results[name], alternative="greater")
            print(f"  {name}: W={stat:.1f} p={pval:.4f}", flush=True)
        except:
            print(f"  {name}: test failed", flush=True)

    # ===== Factorial ablation =====
    print("\n" + "=" * 80, flush=True)
    print("FACTORIAL ABLATION", flush=True)
    print("=" * 80, flush=True)
    print(f"  {'SSR':>4} {'Steer':>6} {'KL':>4} {'QAW':>5} {'Mean JS':>8} {'Wins vs BL':>10}", flush=True)
    print("  " + "-" * 45, flush=True)
    ablation = [
        ("Direct SSR", "Y", "N", "N", "N"),
        ("Multi-layer steer", "N", "Y", "N", "N"),
        ("Ensemble w=0.7", "Y", "Y", "N", "N"),
        ("KL only", "Y", "N", "Y", "N"),
        ("SSR + KL + QAW", "Y", "N", "Y", "Y"),
        ("Ensemble + KL", "Y", "Y", "Y", "N"),
        ("Ensemble + KL + QAW", "Y", "Y", "Y", "Y"),
    ]
    for name, s, st, kl, qaw in ablation:
        vals = results[name]
        wins = sum(1 for v, b in zip(vals, bl_js) if v < b)
        print(f"  {s:>4} {st:>6} {kl:>4} {qaw:>5} {np.mean(vals):>8.4f} {wins:>8}/23", flush=True)

    # ===== Question-type analysis =====
    print("\n" + "=" * 80, flush=True)
    print("QUESTION-TYPE HETEROGENEITY", flush=True)
    print("=" * 80, flush=True)

    # Categorize questions
    q_types = {}
    for qi, qd in enumerate(questions):
        q = qd["question"]
        if "季节" in q or "高发" in q:
            q_types[qi] = "季节/时间"
        elif "场景" in q or "情况" in q or "原因" in q:
            q_types[qi] = "场景/原因"
        elif "方法" in q or "缓解" in q or "解决" in q:
            q_types[qi] = "方法/行为"
        elif "频率" in q or "购买" in q:
            q_types[qi] = "购买/频率"
        elif "渠道" in q or "价格" in q:
            q_types[qi] = "消费/渠道"
        else:
            q_types[qi] = "其他"

    type_groups = defaultdict(list)
    for qi, qtype in q_types.items():
        type_groups[qtype].append(qi)

    print(f"  {'Type':<15} {'N':>3} {'SSR':>8} {'Ens0.7':>8} {'Ens+KL+QAW':>10} {'Ens wins':>9}", flush=True)
    print("  " + "-" * 60, flush=True)
    for qtype, indices in sorted(type_groups.items()):
        ssr_mean = np.mean([results["Direct SSR"][i] for i in indices])
        ens_mean = np.mean([results["Ensemble w=0.7"][i] for i in indices])
        full_mean = np.mean([results["Ensemble + KL + QAW"][i] for i in indices])
        wins = sum(1 for i in indices if results["Ensemble w=0.7"][i] < results["Direct SSR"][i])
        print(f"  {qtype:<15} {len(indices):>3} {ssr_mean:>8.4f} {ens_mean:>8.4f} "
              f"{full_mean:>10.4f} {wins:>7}/{len(indices)}", flush=True)

    # ===== Per-question detail =====
    print("\n" + "=" * 80, flush=True)
    print("PER-QUESTION DETAIL", flush=True)
    print("=" * 80, flush=True)
    print(f"  {'Key':<18} {'Type':<12} {'SSR':>7} {'Ens0.7':>7} {'E+KL+Q':>7} {'Best':>10}", flush=True)
    print("  " + "-" * 70, flush=True)
    for qi, qd in enumerate(questions):
        qt = q_types.get(qi, "?")
        s = results["Direct SSR"][qi]
        e = results["Ensemble w=0.7"][qi]
        f = results["Ensemble + KL + QAW"][qi]
        best_name = min(results, key=lambda m: results[m][qi])
        best_val = results[best_name][qi]
        print(f"  {qd['key']:<18} {qt:<12} {s:>7.4f} {e:>7.4f} {f:>7.4f} {best_name:>10}", flush=True)

    # Save
    save_data = {name: {"per_q": [float(v) for v in vals],
                        "mean": float(np.mean(vals)),
                        "median": float(np.median(vals))}
                 for name, vals in results.items()}
    save_data["question_types"] = {str(k): v for k, v in q_types.items()}
    save_data["question_keys"] = [qd["key"] for qd in questions]

    with open(f"{RESULTS_DIR}/full_eval_23q.json", "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {RESULTS_DIR}/full_eval_23q.json", flush=True)


if __name__ == "__main__":
    main()
