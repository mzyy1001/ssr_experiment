"""
Runner for QAW (Questionnaire-level Adaptive Weighting) experiments.

Key optimization: per-cluster steered distributions are computed ONCE,
then reweighted analytically for different tau values. This avoids
redundant LLM generation during LOO calibration.

Run on chen server:
  source /data/anaconda3/etc/profile.d/conda.sh && conda activate llama_qwen
  cd /data/chenhongrui/business/experiments
  python -u run_qaw.py --device cuda:2
"""
import argparse
import json
import numpy as np
from pathlib import Path
from scipy.spatial.distance import jensenshannon
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from adaptive_weights import (
    compute_question_topic_relevance,
    adaptive_weights,
)


MODEL_PATH = "/data/chenhongrui/models/Qwen3-8B"
DATA_DIR = "/data/chenhongrui/business/data"
RESULTS_DIR = "/data/chenhongrui/business/results"
EMBEDDING_MODEL = "/data/chenhongrui/models/bge-base-zh-v1.5"


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


def compute_per_cluster_distributions(
    model, tokenizer, encoder, persona_vectors,
    questions_data, alpha, layer, n_responses=3,
):
    """Compute steered SSR distribution for each (question, cluster) pair.

    Returns:
        per_cluster_pmfs: dict[q_idx][cluster_id_str] -> np.ndarray (pmf)
        anchors_cache: dict[question_text] -> (anchors, anchor_embs)
    """
    from steered_ssr import generate_steered_response, ssr_score, generate_anchors_local
    import torch

    # Pre-generate anchors
    print("Generating anchors for all questions...")
    anchors_cache = {}
    for qd in questions_data:
        q = qd["question"]
        if q not in anchors_cache:
            anchors = generate_anchors_local(model, tokenizer, q, qd["options"])
            anchor_embs = [encoder.encode(a) for a in anchors]
            anchors_cache[q] = (anchors, anchor_embs)
            print(f"  Anchors for '{q[:40]}...': {anchors}")

    # Get the set of clusters we need (top-20 by base weight)
    all_cluster_ids = set()
    for qd in questions_data:
        sorted_clusters = sorted(qd["cluster_weights"].items(), key=lambda x: -x[1])[:20]
        for cid_str, w in sorted_clusters:
            if int(cid_str) in persona_vectors and w > 1e-6:
                all_cluster_ids.add(cid_str)
    print(f"Computing steered distributions for {len(all_cluster_ids)} clusters × {len(questions_data)} questions...")

    prompt_template = (
        "作为一位有相关经验的消费者，请回答以下问卷问题。\n"
        "请用自然的语言表达你的真实想法和感受，不需要选择选项。\n\n"
        "问题：{question}\n选项：{options}\n\n你的回答："
    )

    per_cluster_pmfs = {}
    total_gens = len(questions_data) * len(all_cluster_ids) * n_responses
    gen_count = 0

    for q_idx, qd in enumerate(questions_data):
        per_cluster_pmfs[q_idx] = {}
        prompt = prompt_template.format(
            question=qd["question"], options="、".join(qd["options"])
        )
        anchors, anchor_embs = anchors_cache[qd["question"]]

        for cid_str in sorted(all_cluster_ids, key=lambda x: -qd["cluster_weights"].get(x, 0)):
            cid = int(cid_str)
            vec = persona_vectors[cid]["vector"]

            cluster_pmfs = []
            for _ in range(n_responses):
                response = generate_steered_response(
                    model, tokenizer, prompt, vec, alpha, layer,
                )
                pmf, _ = ssr_score(response, encoder, anchor_embs)
                cluster_pmfs.append(pmf)
                gen_count += 1

            per_cluster_pmfs[q_idx][cid_str] = np.mean(cluster_pmfs, axis=0)

            if gen_count % 30 == 0:
                print(f"  Progress: {gen_count}/{total_gens} generations ({100*gen_count/total_gens:.0f}%)")

    print(f"  Done: {gen_count} total generations")
    return per_cluster_pmfs, anchors_cache


def aggregate_with_weights(per_cluster_pmfs_q, weights):
    """Fast reweighting: weighted sum of pre-computed per-cluster distributions."""
    n_options = len(next(iter(per_cluster_pmfs_q.values())))
    agg = np.zeros(n_options)
    total_w = 0
    for cid_str, pmf in per_cluster_pmfs_q.items():
        w = weights.get(cid_str, 0.0)
        agg += w * pmf
        total_w += w
    if total_w > 0:
        agg /= total_w
    return agg


def loo_calibrate(
    questions_data, per_cluster_pmfs, relevance_map, cluster_ids,
    tau_candidates=None,
):
    """LOO calibration using pre-computed per-cluster distributions.

    This is fast because it only does array operations, no LLM generation.
    """
    if tau_candidates is None:
        tau_candidates = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0]

    n = len(questions_data)
    fold_results = []

    for held_out_idx in range(n):
        # Find best tau on training set
        best_tau = tau_candidates[0]
        best_js = float("inf")

        for tau in tau_candidates:
            js_vals = []
            for i in range(n):
                if i == held_out_idx:
                    continue
                qd = questions_data[i]
                r = relevance_map[qd["question"]]
                aw = adaptive_weights(qd["cluster_weights"], r, cluster_ids, tau)
                pred = aggregate_with_weights(per_cluster_pmfs[i], aw)
                true = np.array(list(qd["true_distribution"].values()), dtype=float)
                true = true / true.sum()
                js_vals.append(jensenshannon(true, pred) ** 2)

            mean_js = np.mean(js_vals)
            if mean_js < best_js:
                best_js = mean_js
                best_tau = tau

        # Evaluate on held-out
        qd = questions_data[held_out_idx]
        r = relevance_map[qd["question"]]
        aw = adaptive_weights(qd["cluster_weights"], r, cluster_ids, best_tau)
        pred = aggregate_with_weights(per_cluster_pmfs[held_out_idx], aw)
        true = np.array(list(qd["true_distribution"].values()), dtype=float)
        true = true / true.sum()
        held_out_js = float(jensenshannon(true, pred) ** 2)

        # Also get uniform-weight baseline for this question
        uniform_pred = aggregate_with_weights(per_cluster_pmfs[held_out_idx], qd["cluster_weights"])
        uniform_js = float(jensenshannon(true, uniform_pred) ** 2)

        fold_results.append({
            "held_out_question": qd["question"],
            "calibrated_tau": best_tau,
            "train_js": best_js,
            "held_out_js": held_out_js,
            "uniform_js": uniform_js,
            "improvement": uniform_js - held_out_js,
            "top5_adaptive_weights": dict(sorted(aw.items(), key=lambda x: -x[1])[:5]),
            "pred_pmf": pred.tolist(),
            "true_pmf": true.tolist(),
        })

    # Global best tau (on all questions)
    global_results = {}
    for tau in tau_candidates:
        js_vals = []
        for i in range(n):
            qd = questions_data[i]
            r = relevance_map[qd["question"]]
            aw = adaptive_weights(qd["cluster_weights"], r, cluster_ids, tau)
            pred = aggregate_with_weights(per_cluster_pmfs[i], aw)
            true = np.array(list(qd["true_distribution"].values()), dtype=float)
            true = true / true.sum()
            js_vals.append(jensenshannon(true, pred) ** 2)
        global_results[tau] = {
            "mean_js": float(np.mean(js_vals)),
            "per_question_js": [float(j) for j in js_vals],
        }

    best_global_tau = min(global_results, key=lambda t: global_results[t]["mean_js"])

    return {
        "fold_results": fold_results,
        "mean_held_out_js": float(np.mean([r["held_out_js"] for r in fold_results])),
        "mean_uniform_js": float(np.mean([r["uniform_js"] for r in fold_results])),
        "mean_improvement": float(np.mean([r["improvement"] for r in fold_results])),
        "calibrated_taus": [r["calibrated_tau"] for r in fold_results],
        "tau_consistency_std": float(np.std([r["calibrated_tau"] for r in fold_results])),
        "global_tau_sweep": {str(k): v for k, v in global_results.items()},
        "global_best_tau": best_global_tau,
        "global_best_js": global_results[best_global_tau]["mean_js"],
    }


def main():
    parser = argparse.ArgumentParser(description="QAW Experiment Runner")
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--layer", type=int, default=24)
    parser.add_argument("--n_samples", type=int, default=20)
    parser.add_argument("--n_responses", type=int, default=3)
    args = parser.parse_args()

    Path(RESULTS_DIR).mkdir(exist_ok=True)

    # Load data
    questions_data, topics = load_questions_and_topics()

    # Load models
    print("Loading embedding model...")
    encoder = SentenceTransformer(EMBEDDING_MODEL)

    print("Loading LLM and persona vectors...")
    from persona_vectors import load_model, load_persona_vectors
    model, tokenizer = load_model(MODEL_PATH, args.device)
    vectors_path = f"{RESULTS_DIR}/persona_vectors_L{args.layer}_N{args.n_samples}.npz"
    persona_vectors = load_persona_vectors(vectors_path)

    # Step 1: Compute question-topic relevance
    print("\n" + "=" * 60)
    print("Step 1: Question-Topic Relevance")
    print("=" * 60)
    question_texts = [qd["question"] for qd in questions_data]
    relevance_map, cluster_ids = compute_question_topic_relevance(
        question_texts, topics, encoder,
    )

    # Step 2: Compute per-cluster steered distributions (ONCE)
    print("\n" + "=" * 60)
    print(f"Step 2: Per-Cluster Steered Distributions (alpha={args.alpha}, layer={args.layer})")
    print("=" * 60)
    per_cluster_pmfs, anchors_cache = compute_per_cluster_distributions(
        model, tokenizer, encoder, persona_vectors,
        questions_data, args.alpha, args.layer,
        n_responses=args.n_responses,
    )

    # Step 3: Baseline with uniform weights
    print("\n" + "=" * 60)
    print("Step 3: Baseline (uniform weights)")
    print("=" * 60)
    baseline_js = []
    for i, qd in enumerate(questions_data):
        pred = aggregate_with_weights(per_cluster_pmfs[i], qd["cluster_weights"])
        true = np.array(list(qd["true_distribution"].values()), dtype=float)
        true = true / true.sum()
        js = jensenshannon(true, pred) ** 2
        baseline_js.append(js)
        print(f"  {qd['question'][:40]}... JS={js:.4f}")
    print(f"  Mean JS (uniform): {np.mean(baseline_js):.4f}")

    # Step 4: LOO calibration
    print("\n" + "=" * 60)
    print("Step 4: LOO Calibration")
    print("=" * 60)
    results = loo_calibrate(
        questions_data, per_cluster_pmfs, relevance_map, cluster_ids,
    )

    print(f"\n  Mean JS (QAW, LOO): {results['mean_held_out_js']:.4f}")
    print(f"  Mean JS (uniform):  {results['mean_uniform_js']:.4f}")
    print(f"  Improvement:        {results['mean_improvement']:.4f}")
    print(f"  Calibrated taus:    {results['calibrated_taus']}")
    print(f"  Tau std:            {results['tau_consistency_std']:.4f}")
    print(f"  Global best tau:    {results['global_best_tau']}")
    print(f"  Global best JS:     {results['global_best_js']:.4f}")

    print("\n  Per-fold details:")
    for fold in results["fold_results"]:
        print(f"    {fold['held_out_question'][:40]}... "
              f"tau={fold['calibrated_tau']}, "
              f"JS={fold['held_out_js']:.4f} "
              f"(uniform={fold['uniform_js']:.4f}, "
              f"{'better' if fold['improvement'] > 0 else 'worse'} by {abs(fold['improvement']):.4f})")

    print("\n  Global tau sweep:")
    for tau_str, data in sorted(results["global_tau_sweep"].items(), key=lambda x: float(x[0])):
        print(f"    tau={tau_str}: mean_js={data['mean_js']:.4f}")

    # Add config and relevance analysis
    results["config"] = {
        "alpha": args.alpha,
        "layer": args.layer,
        "n_samples": args.n_samples,
        "n_responses": args.n_responses,
    }
    results["baseline_js"] = [float(j) for j in baseline_js]
    results["baseline_mean_js"] = float(np.mean(baseline_js))

    results["relevance_analysis"] = {}
    for qd in questions_data:
        q = qd["question"]
        r = relevance_map[q]
        top10 = sorted(zip(cluster_ids, r), key=lambda x: -x[1])[:10]
        results["relevance_analysis"][q] = [
            {"cluster": cid, "relevance": float(score), "topic": topics[cid][:60]}
            for cid, score in top10
        ]

    # Save per-cluster distributions for reuse
    per_cluster_serialized = {}
    for q_idx, cluster_pmfs in per_cluster_pmfs.items():
        per_cluster_serialized[str(q_idx)] = {
            cid: pmf.tolist() for cid, pmf in cluster_pmfs.items()
        }
    results["per_cluster_pmfs"] = per_cluster_serialized

    out_path = f"{RESULTS_DIR}/qaw_ps_ssr_L{args.layer}_A{args.alpha}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
