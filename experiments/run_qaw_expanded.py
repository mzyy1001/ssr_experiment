"""
QAW experiment on expanded question set (23 questions).

With more questions, LOO calibration should be much more robust:
- Train on 22, test on 1 (vs train on 5, test on 1 before)
- Also test leave-k-out for k=2,3,5

Run on chen server:
  source /data/anaconda3/etc/profile.d/conda.sh && conda activate llama_qwen
  cd /data/chenhongrui/business/experiments
  python -u run_qaw_expanded.py --device cuda:2
"""
import argparse
import json
import numpy as np
from pathlib import Path
from itertools import combinations
from scipy.spatial.distance import jensenshannon
from sentence_transformers import SentenceTransformer

from adaptive_weights import (
    compute_question_topic_relevance,
    adaptive_weights,
)
from persona_vectors import load_model, load_persona_vectors
from steered_ssr import generate_steered_response, ssr_score, generate_anchors_local


MODEL_PATH = "/data/chenhongrui/models/Qwen3-8B"
DATA_DIR = "/data/chenhongrui/business/data"
RESULTS_DIR = "/data/chenhongrui/business/results"
EMBEDDING_MODEL = "/data/chenhongrui/models/bge-base-zh-v1.5"

TAU_CANDIDATES = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0]


def load_expanded():
    with open(f"{RESULTS_DIR}/all_questions_expanded.json") as f:
        data = json.load(f)
    with open(f"{DATA_DIR}/3_cluster_topics.json") as f:
        topics = json.load(f)
    questions = []
    for sk, sv in data.items():
        for sub, qs in sv.items():
            for qid, qd in qs.items():
                questions.append({
                    "key": f"{sub}/{qid}",
                    "question": qd["question"],
                    "options": qd["options"],
                    "true_distribution": qd["true_distribution"],
                    "cluster_weights": qd["cluster_weights"],
                })
    return questions, topics


def compute_per_cluster(model, tokenizer, encoder, persona_vectors,
                        questions, alpha, layer, n_responses=3):
    """One-time computation of per-cluster steered distributions."""
    prompt_tpl = (
        "作为一位有相关经验的消费者，请回答以下问卷问题。\n"
        "请用自然的语言表达你的真实想法和感受，不需要选择选项。\n\n"
        "问题：{question}\n选项：{options}\n\n你的回答："
    )

    # Anchors
    print("Generating anchors...", flush=True)
    anchors_cache = {}
    for qd in questions:
        q = qd["question"]
        if q not in anchors_cache:
            anchors = generate_anchors_local(model, tokenizer, q, qd["options"])
            anchor_embs = [encoder.encode(a) for a in anchors]
            anchors_cache[q] = (anchors, anchor_embs)
            print(f"  {q[:45]}... [{len(qd['options'])} opts]", flush=True)

    # Top-20 clusters
    all_cids = set()
    for qd in questions:
        for cid, w in sorted(qd["cluster_weights"].items(), key=lambda x: -x[1])[:20]:
            if int(cid) in persona_vectors and w > 1e-6:
                all_cids.add(cid)

    total = len(questions) * len(all_cids) * n_responses
    print(f"Generating steered responses: {len(questions)} questions × {len(all_cids)} clusters × {n_responses} = {total}", flush=True)

    per_cluster = {}
    count = 0
    for qi, qd in enumerate(questions):
        per_cluster[qi] = {}
        prompt = prompt_tpl.format(question=qd["question"], options="、".join(qd["options"]))
        _, anchor_embs = anchors_cache[qd["question"]]

        for cid in sorted(all_cids, key=lambda x: -qd["cluster_weights"].get(x, 0)):
            vec = persona_vectors[int(cid)]["vector"]
            pmfs = []
            for _ in range(n_responses):
                resp = generate_steered_response(model, tokenizer, prompt, vec, alpha, layer)
                pmf, _ = ssr_score(resp, encoder, anchor_embs)
                pmfs.append(pmf)
                count += 1
            per_cluster[qi][cid] = np.mean(pmfs, axis=0)
            if count % 120 == 0:
                print(f"  {count}/{total} ({100*count/total:.0f}%) - Q{qi+1}/{len(questions)}", flush=True)

    print(f"Done: {count} generations", flush=True)
    return per_cluster, anchors_cache


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
    p = pred / pred.sum()
    return float(jensenshannon(t, p) ** 2)


def calibrate_tau(train_indices, questions, per_cluster, relevance_map, cluster_ids):
    """Find best tau on training set."""
    best_tau, best_js = TAU_CANDIDATES[0], float("inf")
    for tau in TAU_CANDIDATES:
        js_vals = []
        for i in train_indices:
            qd = questions[i]
            r = relevance_map[qd["question"]]
            aw = adaptive_weights(qd["cluster_weights"], r, cluster_ids, tau)
            pred = agg(per_cluster[i], aw)
            js_vals.append(js(qd["true_distribution"], pred))
        m = np.mean(js_vals)
        if m < best_js:
            best_js, best_tau = m, tau
    return best_tau, best_js


def run_loo(questions, per_cluster, relevance_map, cluster_ids):
    n = len(questions)
    results = []
    for ho in range(n):
        train = [i for i in range(n) if i != ho]
        tau, _ = calibrate_tau(train, questions, per_cluster, relevance_map, cluster_ids)

        qd = questions[ho]
        r = relevance_map[qd["question"]]
        aw = adaptive_weights(qd["cluster_weights"], r, cluster_ids, tau)
        qaw_pred = agg(per_cluster[ho], aw)
        uni_pred = agg(per_cluster[ho], qd["cluster_weights"])

        results.append({
            "key": qd["key"],
            "question": qd["question"][:40],
            "tau": tau,
            "qaw_js": js(qd["true_distribution"], qaw_pred),
            "uniform_js": js(qd["true_distribution"], uni_pred),
        })
    return results


def run_lko(questions, per_cluster, relevance_map, cluster_ids, k, max_combos=30):
    n = len(questions)
    combos = list(combinations(range(n), k))
    if len(combos) > max_combos:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(combos), max_combos, replace=False)
        combos = [combos[i] for i in idx]

    qaw_js_all, uni_js_all, taus = [], [], []
    for ho_set in combos:
        ho_set = set(ho_set)
        train = [i for i in range(n) if i not in ho_set]
        tau, _ = calibrate_tau(train, questions, per_cluster, relevance_map, cluster_ids)
        taus.append(tau)

        for ho in ho_set:
            qd = questions[ho]
            r = relevance_map[qd["question"]]
            aw = adaptive_weights(qd["cluster_weights"], r, cluster_ids, tau)
            qaw_js_all.append(js(qd["true_distribution"], agg(per_cluster[ho], aw)))
            uni_js_all.append(js(qd["true_distribution"], agg(per_cluster[ho], qd["cluster_weights"])))

    return {
        "k": k, "n_combos": len(combos),
        "qaw_mean": float(np.mean(qaw_js_all)),
        "uniform_mean": float(np.mean(uni_js_all)),
        "improvement": float(np.mean(uni_js_all) - np.mean(qaw_js_all)),
        "win_rate": float(np.mean([q < u for q, u in zip(qaw_js_all, uni_js_all)])),
        "taus": taus,
        "tau_mean": float(np.mean(taus)),
        "tau_std": float(np.std(taus)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--layer", type=int, default=24)
    parser.add_argument("--n_responses", type=int, default=3)
    args = parser.parse_args()

    Path(RESULTS_DIR).mkdir(exist_ok=True)
    questions, topics = load_expanded()
    print(f"Loaded {len(questions)} questions", flush=True)

    # Models
    print("Loading models...", flush=True)
    encoder = SentenceTransformer(EMBEDDING_MODEL)
    model, tokenizer = load_model(MODEL_PATH, args.device)
    vec_path = f"{RESULTS_DIR}/persona_vectors_L{args.layer}_N20.npz"
    persona_vectors = load_persona_vectors(vec_path)

    # Relevance
    print("\n=== Question-Topic Relevance ===", flush=True)
    q_texts = [qd["question"] for qd in questions]
    relevance_map, cluster_ids = compute_question_topic_relevance(q_texts, topics, encoder)

    # Per-cluster distributions
    print(f"\n=== Per-Cluster Steered Distributions (alpha={args.alpha}, layer={args.layer}) ===", flush=True)
    per_cluster, _ = compute_per_cluster(
        model, tokenizer, encoder, persona_vectors,
        questions, args.alpha, args.layer, args.n_responses,
    )

    # Baseline uniform
    print("\n=== Baseline (uniform weights) ===", flush=True)
    for i, qd in enumerate(questions):
        pred = agg(per_cluster[i], qd["cluster_weights"])
        j = js(qd["true_distribution"], pred)
        print(f"  {qd['key']}: JS={j:.4f} - {qd['question'][:40]}", flush=True)
    uniform_mean = np.mean([
        js(qd["true_distribution"], agg(per_cluster[i], qd["cluster_weights"]))
        for i, qd in enumerate(questions)
    ])
    print(f"  Mean JS (uniform): {uniform_mean:.4f}", flush=True)

    # LOO
    print("\n=== Leave-One-Out Calibration ===", flush=True)
    loo = run_loo(questions, per_cluster, relevance_map, cluster_ids)
    loo_mean = np.mean([r["qaw_js"] for r in loo])
    loo_taus = [r["tau"] for r in loo]
    wins = sum(1 for r in loo if r["qaw_js"] < r["uniform_js"])
    for r in loo:
        delta = r["uniform_js"] - r["qaw_js"]
        marker = "+" if delta > 0 else "-"
        print(f"  {r['key']}: tau={r['tau']} QAW={r['qaw_js']:.4f} uni={r['uniform_js']:.4f} {marker}{abs(delta):.4f}", flush=True)
    print(f"  Mean QAW: {loo_mean:.4f}, Mean uniform: {uniform_mean:.4f}", flush=True)
    print(f"  Improvement: {uniform_mean - loo_mean:+.4f}", flush=True)
    print(f"  Win rate: {wins}/{len(loo)} ({100*wins/len(loo):.0f}%)", flush=True)
    print(f"  Tau mean={np.mean(loo_taus):.3f} std={np.std(loo_taus):.3f}", flush=True)

    # Leave-k-out
    print("\n=== Leave-K-Out Robustness ===", flush=True)
    lko_results = {}
    for k in [2, 3, 5, 8]:
        lko = run_lko(questions, per_cluster, relevance_map, cluster_ids, k)
        lko_results[k] = lko
        print(f"  L{k}O: QAW={lko['qaw_mean']:.4f} uni={lko['uniform_mean']:.4f} "
              f"Δ={lko['improvement']:+.4f} win={lko['win_rate']:.0%} "
              f"tau={lko['tau_mean']:.3f}±{lko['tau_std']:.3f}", flush=True)

    # Global tau sweep
    print("\n=== Global Tau Sweep ===", flush=True)
    for tau in TAU_CANDIDATES:
        js_vals = []
        for i, qd in enumerate(questions):
            r = relevance_map[qd["question"]]
            aw = adaptive_weights(qd["cluster_weights"], r, cluster_ids, tau)
            pred = agg(per_cluster[i], aw)
            js_vals.append(js(qd["true_distribution"], pred))
        print(f"  tau={tau}: mean_js={np.mean(js_vals):.4f}", flush=True)

    # Save
    results = {
        "n_questions": len(questions),
        "config": {"alpha": args.alpha, "layer": args.layer, "n_responses": args.n_responses},
        "uniform_mean_js": float(uniform_mean),
        "loo": {
            "mean_js": float(loo_mean),
            "improvement": float(uniform_mean - loo_mean),
            "win_rate": wins / len(loo),
            "taus": loo_taus,
            "tau_std": float(np.std(loo_taus)),
            "per_question": loo,
        },
        "lko": {str(k): v for k, v in lko_results.items()},
        "per_cluster_pmfs": {
            str(qi): {cid: pmf.tolist() for cid, pmf in cpmfs.items()}
            for qi, cpmfs in per_cluster.items()
        },
    }

    out = f"{RESULTS_DIR}/qaw_expanded_L{args.layer}_A{args.alpha}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {out}", flush=True)


if __name__ == "__main__":
    main()
