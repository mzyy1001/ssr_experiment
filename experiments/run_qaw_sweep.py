"""
QAW Robustness Sweep: vary (alpha, layer) and leave-k-out to show
adaptive weighting consistently improves over uniform weights.

Per-cluster distributions are computed ONCE per (alpha, layer) config,
then LOO/LKO calibration is instant (just array reweighting).

Run on chen server:
  source /data/anaconda3/etc/profile.d/conda.sh && conda activate llama_qwen
  cd /data/chenhongrui/business/experiments
  python -u run_qaw_sweep.py --device cuda:2
"""
import argparse
import json
import itertools
import numpy as np
from pathlib import Path
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


def load_questions_and_topics():
    with open(f"{RESULTS_DIR}/all_questions.json") as f:
        baseline = json.load(f)
    with open(f"{DATA_DIR}/3_cluster_topics.json") as f:
        topics = json.load(f)
    questions_data = []
    for survey_key, survey_data in baseline.items():
        for sub_id, questions in survey_data.items():
            for q_id, q_data in questions.items():
                questions_data.append({
                    "survey_key": survey_key,
                    "sub_id": sub_id,
                    "q_id": q_id,
                    "question": q_data["question"],
                    "options": q_data["options"],
                    "true_distribution": q_data["true_distribution"],
                    "cluster_weights": q_data["cluster_weights"],
                })
    return questions_data, topics


def compute_per_cluster_dists(model, tokenizer, encoder, persona_vectors,
                              questions_data, alpha, layer, anchors_cache,
                              n_responses=3):
    """Compute per-cluster steered distributions for one (alpha, layer) config."""
    prompt_template = (
        "作为一位有相关经验的消费者，请回答以下问卷问题。\n"
        "请用自然的语言表达你的真实想法和感受，不需要选择选项。\n\n"
        "问题：{question}\n选项：{options}\n\n你的回答："
    )

    # Clusters to use (top-20 by weight)
    all_cids = set()
    for qd in questions_data:
        for cid_str, w in sorted(qd["cluster_weights"].items(), key=lambda x: -x[1])[:20]:
            if int(cid_str) in persona_vectors and w > 1e-6:
                all_cids.add(cid_str)

    per_cluster_pmfs = {}
    total = len(questions_data) * len(all_cids) * n_responses
    count = 0

    for q_idx, qd in enumerate(questions_data):
        per_cluster_pmfs[q_idx] = {}
        prompt = prompt_template.format(question=qd["question"], options="、".join(qd["options"]))
        _, anchor_embs = anchors_cache[qd["question"]]

        for cid_str in sorted(all_cids, key=lambda x: -qd["cluster_weights"].get(x, 0)):
            vec = persona_vectors[int(cid_str)]["vector"]
            pmfs = []
            for _ in range(n_responses):
                resp = generate_steered_response(model, tokenizer, prompt, vec, alpha, layer)
                pmf, _ = ssr_score(resp, encoder, anchor_embs)
                pmfs.append(pmf)
                count += 1
            per_cluster_pmfs[q_idx][cid_str] = np.mean(pmfs, axis=0)

            if count % 60 == 0:
                print(f"    {count}/{total} ({100*count/total:.0f}%)", flush=True)

    print(f"    Done: {count} generations", flush=True)
    return per_cluster_pmfs


def aggregate(per_cluster_pmfs_q, weights):
    n = len(next(iter(per_cluster_pmfs_q.values())))
    agg = np.zeros(n)
    tw = 0
    for cid, pmf in per_cluster_pmfs_q.items():
        w = weights.get(cid, 0.0)
        agg += w * pmf
        tw += w
    if tw > 0:
        agg /= tw
    return agg


def run_loo(questions_data, per_cluster_pmfs, relevance_map, cluster_ids):
    """Standard leave-one-out calibration."""
    n = len(questions_data)
    results = []
    for held_out in range(n):
        best_tau, best_js = TAU_CANDIDATES[0], float("inf")
        for tau in TAU_CANDIDATES:
            js_vals = []
            for i in range(n):
                if i == held_out:
                    continue
                qd = questions_data[i]
                r = relevance_map[qd["question"]]
                aw = adaptive_weights(qd["cluster_weights"], r, cluster_ids, tau)
                pred = aggregate(per_cluster_pmfs[i], aw)
                true = np.array(list(qd["true_distribution"].values()), dtype=float)
                true /= true.sum()
                js_vals.append(jensenshannon(true, pred) ** 2)
            m = np.mean(js_vals)
            if m < best_js:
                best_js, best_tau = m, tau

        qd = questions_data[held_out]
        r = relevance_map[qd["question"]]
        aw = adaptive_weights(qd["cluster_weights"], r, cluster_ids, best_tau)
        pred = aggregate(per_cluster_pmfs[held_out], aw)
        true = np.array(list(qd["true_distribution"].values()), dtype=float)
        true /= true.sum()
        uniform_pred = aggregate(per_cluster_pmfs[held_out], qd["cluster_weights"])
        results.append({
            "tau": best_tau,
            "qaw_js": float(jensenshannon(true, pred) ** 2),
            "uniform_js": float(jensenshannon(true, uniform_pred) ** 2),
        })
    return results


def run_lko(questions_data, per_cluster_pmfs, relevance_map, cluster_ids, k):
    """Leave-k-out: train on n-k questions, test on k.
    Average over all C(n,k) combinations.
    """
    from itertools import combinations
    n = len(questions_data)
    all_combos = list(combinations(range(n), k))
    # Cap at 20 combos for tractability
    if len(all_combos) > 20:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(all_combos), 20, replace=False)
        all_combos = [all_combos[i] for i in indices]

    qaw_js_all, uniform_js_all, taus_all = [], [], []
    for held_out_set in all_combos:
        held_out_set = set(held_out_set)
        train_set = [i for i in range(n) if i not in held_out_set]

        best_tau, best_js = TAU_CANDIDATES[0], float("inf")
        for tau in TAU_CANDIDATES:
            js_vals = []
            for i in train_set:
                qd = questions_data[i]
                r = relevance_map[qd["question"]]
                aw = adaptive_weights(qd["cluster_weights"], r, cluster_ids, tau)
                pred = aggregate(per_cluster_pmfs[i], aw)
                true = np.array(list(qd["true_distribution"].values()), dtype=float)
                true /= true.sum()
                js_vals.append(jensenshannon(true, pred) ** 2)
            m = np.mean(js_vals)
            if m < best_js:
                best_js, best_tau = m, tau

        # Evaluate on held-out set
        for ho_idx in held_out_set:
            qd = questions_data[ho_idx]
            r = relevance_map[qd["question"]]
            aw = adaptive_weights(qd["cluster_weights"], r, cluster_ids, best_tau)
            pred = aggregate(per_cluster_pmfs[ho_idx], aw)
            true = np.array(list(qd["true_distribution"].values()), dtype=float)
            true /= true.sum()
            uniform_pred = aggregate(per_cluster_pmfs[ho_idx], qd["cluster_weights"])
            qaw_js_all.append(float(jensenshannon(true, pred) ** 2))
            uniform_js_all.append(float(jensenshannon(true, uniform_pred) ** 2))
        taus_all.append(best_tau)

    return {
        "k": k,
        "n_combos": len(all_combos),
        "mean_qaw_js": float(np.mean(qaw_js_all)),
        "mean_uniform_js": float(np.mean(uniform_js_all)),
        "improvement": float(np.mean(uniform_js_all) - np.mean(qaw_js_all)),
        "taus": taus_all,
        "tau_std": float(np.std(taus_all)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--n_responses", type=int, default=3)
    args = parser.parse_args()

    Path(RESULTS_DIR).mkdir(exist_ok=True)

    # Configs to sweep
    configs = [
        {"alpha": 0.05, "layer": 24},
        {"alpha": 0.1,  "layer": 16},
        {"alpha": 0.1,  "layer": 20},
        # alpha=0.1, layer=24 already done — load from cache
        {"alpha": 0.3,  "layer": 24},
        {"alpha": 0.5,  "layer": 24},
        {"alpha": 1.0,  "layer": 24},
    ]

    questions_data, topics = load_questions_and_topics()

    print("Loading models...", flush=True)
    encoder = SentenceTransformer(EMBEDDING_MODEL)
    model, tokenizer = load_model(MODEL_PATH, args.device)

    # Compute relevance (shared across all configs)
    question_texts = [qd["question"] for qd in questions_data]
    relevance_map, cluster_ids = compute_question_topic_relevance(
        question_texts, topics, encoder,
    )

    # Pre-generate anchors (shared)
    print("Generating anchors...", flush=True)
    anchors_cache = {}
    for qd in questions_data:
        q = qd["question"]
        if q not in anchors_cache:
            anchors = generate_anchors_local(model, tokenizer, q, qd["options"])
            anchor_embs = [encoder.encode(a) for a in anchors]
            anchors_cache[q] = (anchors, anchor_embs)
            print(f"  {q[:40]}...", flush=True)

    all_results = {}

    # Try to load cached per-cluster distributions for alpha=0.1, layer=24
    cached_path = f"{RESULTS_DIR}/qaw_ps_ssr_L24_A0.1.json"
    cached_per_cluster = None
    try:
        with open(cached_path) as f:
            cached = json.load(f)
        if "per_cluster_pmfs" in cached:
            cached_per_cluster = {}
            for q_idx_str, cluster_pmfs in cached["per_cluster_pmfs"].items():
                cached_per_cluster[int(q_idx_str)] = {
                    cid: np.array(pmf) for cid, pmf in cluster_pmfs.items()
                }
            print("Loaded cached per-cluster dists for alpha=0.1, layer=24", flush=True)
    except Exception:
        pass

    for cfg in configs:
        alpha, layer = cfg["alpha"], cfg["layer"]
        tag = f"A{alpha}_L{layer}"
        print(f"\n{'='*60}", flush=True)
        print(f"Config: alpha={alpha}, layer={layer}", flush=True)
        print(f"{'='*60}", flush=True)

        # Check if vectors exist
        vec_path = f"{RESULTS_DIR}/persona_vectors_L{layer}_N20.npz"
        if not Path(vec_path).exists():
            print(f"  SKIP: {vec_path} not found", flush=True)
            continue
        persona_vectors = load_persona_vectors(vec_path)

        # Use cache if available
        if alpha == 0.1 and layer == 24 and cached_per_cluster is not None:
            per_cluster_pmfs = cached_per_cluster
            print("  Using cached per-cluster distributions", flush=True)
        else:
            print(f"  Computing per-cluster distributions...", flush=True)
            per_cluster_pmfs = compute_per_cluster_dists(
                model, tokenizer, encoder, persona_vectors,
                questions_data, alpha, layer, anchors_cache,
                n_responses=args.n_responses,
            )

        # Uniform baseline
        uniform_js = []
        for i, qd in enumerate(questions_data):
            pred = aggregate(per_cluster_pmfs[i], qd["cluster_weights"])
            true = np.array(list(qd["true_distribution"].values()), dtype=float)
            true /= true.sum()
            uniform_js.append(float(jensenshannon(true, pred) ** 2))
        print(f"  Uniform mean JS: {np.mean(uniform_js):.4f}", flush=True)

        # LOO calibration
        loo_results = run_loo(questions_data, per_cluster_pmfs, relevance_map, cluster_ids)
        loo_mean = np.mean([r["qaw_js"] for r in loo_results])
        loo_taus = [r["tau"] for r in loo_results]
        print(f"  QAW LOO mean JS: {loo_mean:.4f}", flush=True)
        print(f"  Improvement:     {np.mean(uniform_js) - loo_mean:.4f}", flush=True)
        print(f"  Taus: {loo_taus}", flush=True)

        # Leave-2-out and leave-3-out
        lko_results = {}
        for k in [2, 3]:
            lko = run_lko(questions_data, per_cluster_pmfs, relevance_map, cluster_ids, k)
            lko_results[k] = lko
            print(f"  L{k}O: QAW={lko['mean_qaw_js']:.4f} uniform={lko['mean_uniform_js']:.4f} "
                  f"Δ={lko['improvement']:+.4f} tau_std={lko['tau_std']:.3f}", flush=True)

        # Global tau sweep
        global_sweep = {}
        for tau in TAU_CANDIDATES:
            js_vals = []
            for i, qd in enumerate(questions_data):
                r = relevance_map[qd["question"]]
                aw = adaptive_weights(qd["cluster_weights"], r, cluster_ids, tau)
                pred = aggregate(per_cluster_pmfs[i], aw)
                true = np.array(list(qd["true_distribution"].values()), dtype=float)
                true /= true.sum()
                js_vals.append(jensenshannon(true, pred) ** 2)
            global_sweep[str(tau)] = float(np.mean(js_vals))
        best_global_tau = min(global_sweep, key=global_sweep.get)
        print(f"  Global best: tau={best_global_tau} JS={global_sweep[best_global_tau]:.4f}", flush=True)

        all_results[tag] = {
            "alpha": alpha,
            "layer": layer,
            "uniform_mean_js": float(np.mean(uniform_js)),
            "uniform_per_q": uniform_js,
            "loo_mean_js": float(loo_mean),
            "loo_per_fold": loo_results,
            "loo_improvement": float(np.mean(uniform_js) - loo_mean),
            "loo_taus": loo_taus,
            "lko": {str(k): v for k, v in lko_results.items()},
            "global_tau_sweep": global_sweep,
            "global_best_tau": best_global_tau,
            "global_best_js": global_sweep[best_global_tau],
        }

    # Summary table
    print(f"\n{'='*80}", flush=True)
    print("ROBUSTNESS SUMMARY", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"{'Config':<18} {'Uniform':>8} {'QAW LOO':>8} {'QAW Glob':>8} {'LOO Δ':>8} {'Glob Δ':>8} {'τ*':>6} {'L2O Δ':>8} {'L3O Δ':>8}", flush=True)
    print("-" * 98, flush=True)
    for tag, r in sorted(all_results.items()):
        l2o_imp = r["lko"].get("2", {}).get("improvement", float("nan"))
        l3o_imp = r["lko"].get("3", {}).get("improvement", float("nan"))
        print(f"{tag:<18} {r['uniform_mean_js']:>8.4f} {r['loo_mean_js']:>8.4f} "
              f"{r['global_best_js']:>8.4f} {r['loo_improvement']:>+8.4f} "
              f"{r['uniform_mean_js']-r['global_best_js']:>+8.4f} "
              f"{r['global_best_tau']:>6} {l2o_imp:>+8.4f} {l3o_imp:>+8.4f}", flush=True)

    # Also show SSR-only baseline for reference
    print(f"\n  Reference: SSR-only (Project 1) = 0.0443", flush=True)

    out_path = f"{RESULTS_DIR}/qaw_robustness_sweep.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nResults saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
