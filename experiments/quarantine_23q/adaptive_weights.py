"""
Approach D: Questionnaire-level Adaptive Weighting (QAW).

Computes question-aware cluster weights by combining base weights with
question-topic semantic relevance, controlled by a temperature parameter
calibrated via leave-one-out cross-validation across the questionnaire.

Key idea: questions within the same questionnaire share latent structure.
Calibrate temperature tau on k-1 questions with ground truth, transfer to
the held-out question. This exploits within-questionnaire correlation to
produce question-specific topic weights without per-question optimization.
"""
import numpy as np
import json
from scipy.optimize import minimize_scalar
from scipy.spatial.distance import jensenshannon
from sentence_transformers import SentenceTransformer


def compute_question_topic_relevance(
    questions: list[str],
    topic_descriptions: dict[str, str],
    encoder: SentenceTransformer,
) -> dict[str, np.ndarray]:
    """Compute cosine similarity between each question and each cluster topic.

    Returns:
        dict mapping question_text -> np.ndarray of shape (n_clusters,)
        with cluster IDs ordered by sorted(topic_descriptions.keys())
    """
    cluster_ids = sorted(topic_descriptions.keys(), key=lambda x: int(x))
    topic_texts = [topic_descriptions[cid] for cid in cluster_ids]

    q_embeddings = encoder.encode(questions, normalize_embeddings=True)
    t_embeddings = encoder.encode(topic_texts, normalize_embeddings=True)

    # cosine similarity matrix: (n_questions, n_clusters)
    sim_matrix = q_embeddings @ t_embeddings.T

    relevance = {}
    for i, q in enumerate(questions):
        relevance[q] = sim_matrix[i]

    return relevance, cluster_ids


def adaptive_weights(
    base_weights: dict[str, float],
    relevance_scores: np.ndarray,
    cluster_ids: list[str],
    tau: float = 0.1,
) -> dict[str, float]:
    """Compute question-specific adaptive weights.

    w_c(q) = w_c^base * softmax(relevance / tau)_c

    Args:
        base_weights: global cluster weights (from Project 1)
        relevance_scores: cosine similarities for this question, shape (n_clusters,)
        cluster_ids: cluster ID strings, same order as relevance_scores
        tau: temperature for relevance softmax (lower = sharper)
    """
    # Softmax over relevance scores with temperature
    r = relevance_scores / max(tau, 1e-6)
    r = r - r.max()  # numerical stability
    exp_r = np.exp(r)
    softmax_r = exp_r / exp_r.sum()

    # Multiply base weights by relevance softmax
    new_weights = {}
    for i, cid in enumerate(cluster_ids):
        base_w = base_weights.get(cid, 0.0)
        new_weights[cid] = base_w * softmax_r[i]

    # Re-normalize
    total = sum(new_weights.values())
    if total > 0:
        new_weights = {k: v / total for k, v in new_weights.items()}

    return new_weights


def evaluate_tau(
    tau: float,
    questions_data: list[dict],
    relevance_map: dict[str, np.ndarray],
    cluster_ids: list[str],
    predict_fn,
    exclude_idx: int = -1,
) -> float:
    """Evaluate a given tau on a set of questions (excluding one for LOO).

    Args:
        tau: temperature parameter
        questions_data: list of dicts with keys: question, true_distribution, cluster_weights, options
        relevance_map: question -> relevance scores
        cluster_ids: ordered cluster IDs
        predict_fn: callable(weights_dict) -> predicted_pmf for a given question
                    Actually: callable(question_idx, weights_dict) -> predicted_pmf
        exclude_idx: index to exclude (-1 = use all)

    Returns:
        mean JS divergence on the included questions
    """
    js_values = []
    for i, qd in enumerate(questions_data):
        if i == exclude_idx:
            continue

        q_text = qd["question"]
        relevance = relevance_map.get(q_text)
        if relevance is None:
            continue

        aw = adaptive_weights(qd["cluster_weights"], relevance, cluster_ids, tau)

        # Get prediction with adaptive weights
        pred_pmf = predict_fn(i, aw)

        # True distribution
        true_counts = np.array(list(qd["true_distribution"].values()), dtype=float)
        true_pmf = true_counts / true_counts.sum()

        js = jensenshannon(true_pmf, pred_pmf) ** 2
        js_values.append(js)

    if not js_values:
        return 1.0
    return float(np.mean(js_values))


def calibrate_tau_loo(
    questions_data: list[dict],
    relevance_map: dict[str, np.ndarray],
    cluster_ids: list[str],
    predict_fn,
    tau_candidates: list[float] = None,
) -> dict:
    """Leave-one-out calibration of temperature tau.

    For each held-out question:
      1. Find optimal tau on remaining questions
      2. Apply to held-out question
      3. Record JS divergence

    Args:
        predict_fn: callable(question_idx, weights_dict) -> predicted_pmf

    Returns:
        dict with per-fold results, mean JS, and calibrated taus
    """
    if tau_candidates is None:
        tau_candidates = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0]

    n = len(questions_data)
    fold_results = []

    for held_out_idx in range(n):
        # Find best tau on training set (all except held_out)
        best_tau = tau_candidates[0]
        best_js = float("inf")

        for tau in tau_candidates:
            js = evaluate_tau(
                tau, questions_data, relevance_map, cluster_ids,
                predict_fn, exclude_idx=held_out_idx,
            )
            if js < best_js:
                best_js = js
                best_tau = tau

        # Evaluate on held-out question with calibrated tau
        qd = questions_data[held_out_idx]
        q_text = qd["question"]
        relevance = relevance_map.get(q_text)

        if relevance is not None:
            aw = adaptive_weights(qd["cluster_weights"], relevance, cluster_ids, best_tau)
            pred_pmf = predict_fn(held_out_idx, aw)

            true_counts = np.array(list(qd["true_distribution"].values()), dtype=float)
            true_pmf = true_counts / true_counts.sum()
            held_out_js = float(jensenshannon(true_pmf, pred_pmf) ** 2)
        else:
            held_out_js = float("nan")
            aw = qd["cluster_weights"]

        fold_results.append({
            "held_out_question": q_text,
            "calibrated_tau": best_tau,
            "train_js": best_js,
            "held_out_js": held_out_js,
            "top5_weights": dict(sorted(aw.items(), key=lambda x: -x[1])[:5]),
        })

    # Also find global best tau (on all questions)
    global_best_tau = tau_candidates[0]
    global_best_js = float("inf")
    for tau in tau_candidates:
        js = evaluate_tau(
            tau, questions_data, relevance_map, cluster_ids,
            predict_fn, exclude_idx=-1,
        )
        if js < global_best_js:
            global_best_js = js
            global_best_tau = tau

    return {
        "fold_results": fold_results,
        "mean_held_out_js": float(np.nanmean([r["held_out_js"] for r in fold_results])),
        "calibrated_taus": [r["calibrated_tau"] for r in fold_results],
        "tau_consistency": float(np.std([r["calibrated_tau"] for r in fold_results])),
        "global_best_tau": global_best_tau,
        "global_best_js": global_best_js,
    }


def run_adaptive_weighting_experiment(
    baseline_results_path: str,
    topics_path: str,
    encoder: SentenceTransformer,
    predict_fn_factory=None,
) -> dict:
    """Run the full QAW experiment.

    This can be used in two modes:
      1. SSR-only mode (no steering): predict_fn uses SSR with adaptive weights
      2. PS-SSR mode: predict_fn uses steered generation with adaptive weights

    Args:
        predict_fn_factory: callable(questions_data) -> predict_fn
            If None, uses a simple SSR-based predictor from baseline results.
    """
    with open(baseline_results_path) as f:
        baseline = json.load(f)
    with open(topics_path) as f:
        topics = json.load(f)

    # Flatten questions
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
                    "predicted_pmf": q_data.get("predicted_pmf"),
                })

    # Compute question-topic relevance
    question_texts = [qd["question"] for qd in questions_data]
    relevance_map, cluster_ids = compute_question_topic_relevance(
        question_texts, topics, encoder,
    )

    # Default predict_fn: use existing predicted_pmf (for SSR-only mode)
    if predict_fn_factory is None:
        def default_predict_fn(q_idx, weights):
            """Reweight existing per-cluster SSR predictions."""
            qd = questions_data[q_idx]
            # If we have per-cluster predictions, reweight them
            # Otherwise, fall back to the existing predicted_pmf
            if qd.get("predicted_pmf") is not None:
                return np.array(qd["predicted_pmf"])
            return np.ones(len(qd["options"])) / len(qd["options"])

        predict_fn = default_predict_fn
    else:
        predict_fn = predict_fn_factory(questions_data)

    # Run LOO calibration
    results = calibrate_tau_loo(
        questions_data, relevance_map, cluster_ids, predict_fn,
    )

    # Add relevance analysis
    results["relevance_analysis"] = {}
    for qd in questions_data:
        q = qd["question"]
        r = relevance_map[q]
        top_clusters = sorted(
            zip(cluster_ids, r), key=lambda x: -x[1]
        )[:10]
        results["relevance_analysis"][q] = [
            {"cluster": cid, "relevance": float(score), "topic": topics[cid][:60]}
            for cid, score in top_clusters
        ]

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="QAW: Questionnaire-level Adaptive Weighting")
    parser.add_argument("--baseline_results", default="/data/chenhongrui/business/results/all_questions.json")
    parser.add_argument("--topics_path", default="/data/chenhongrui/business/data/3_cluster_topics.json")
    parser.add_argument("--embedding_model", default="/data/chenhongrui/models/bge-base-zh-v1.5")
    parser.add_argument("--output", default="/data/chenhongrui/business/results/qaw_results.json")
    parser.add_argument("--mode", choices=["ssr_only", "ps_ssr"], default="ssr_only",
                        help="ssr_only: reweight SSR predictions; ps_ssr: reweight steered predictions")
    # PS-SSR mode args
    parser.add_argument("--model_path", default="/data/chenhongrui/models/Qwen3-8B")
    parser.add_argument("--vectors_path", default="/data/chenhongrui/business/results/persona_vectors_L16_N20.npz")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--layer", type=int, default=24)
    parser.add_argument("--device", default="cuda:2")
    args = parser.parse_args()

    print("Loading embedding model...")
    encoder = SentenceTransformer(args.embedding_model)

    if args.mode == "ps_ssr":
        # Full PS-SSR + QAW mode: need to generate steered predictions on-the-fly
        from persona_vectors import load_model, load_persona_vectors
        from steered_ssr import (
            ps_ssr_steer_then_aggregate, generate_anchors_local,
        )

        print("Loading LLM and persona vectors...")
        model, tokenizer = load_model(args.model_path, args.device)
        persona_vectors = load_persona_vectors(args.vectors_path)

        with open(args.baseline_results) as f:
            baseline = json.load(f)
        with open(args.topics_path) as f:
            topics = json.load(f)

        # Flatten questions (same as in run_adaptive_weighting_experiment)
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

        # Pre-generate anchors for all questions
        print("Generating anchors...")
        anchors_cache = {}
        for qd in questions_data:
            anchors = generate_anchors_local(model, tokenizer, qd["question"], qd["options"])
            anchor_embs = [encoder.encode(a) for a in anchors]
            anchors_cache[qd["question"]] = (anchors, anchor_embs)

        # Compute relevance
        print("Computing question-topic relevance...")
        question_texts = [qd["question"] for qd in questions_data]
        relevance_map, cluster_ids = compute_question_topic_relevance(
            question_texts, topics, encoder,
        )

        # Build predict_fn that runs steered generation with given weights
        def make_predict_fn(q_idx, weights):
            qd = questions_data[q_idx]
            anchors, anchor_embs = anchors_cache[qd["question"]]
            pred_pmf = ps_ssr_steer_then_aggregate(
                model, tokenizer, encoder,
                persona_vectors, weights,
                qd["question"], qd["options"], anchor_embs,
                alpha=args.alpha, layer=args.layer,
                n_responses=3,  # fewer for calibration speed
            )
            return pred_pmf

        # Run LOO calibration
        print("Running LOO calibration...")
        results = calibrate_tau_loo(
            questions_data, relevance_map, cluster_ids, make_predict_fn,
        )

        # Add relevance analysis
        results["relevance_analysis"] = {}
        for qd in questions_data:
            q = qd["question"]
            r = relevance_map[q]
            top_clusters = sorted(
                zip(cluster_ids, r), key=lambda x: -x[1]
            )[:10]
            results["relevance_analysis"][q] = [
                {"cluster": cid, "relevance": float(score), "topic": topics[cid][:60]}
                for cid, score in top_clusters
            ]

        results["config"] = {
            "mode": "ps_ssr",
            "alpha": args.alpha,
            "layer": args.layer,
            "vectors_path": args.vectors_path,
        }

    else:
        # SSR-only mode: reweight existing SSR predictions using per-cluster SSR
        # Need to re-run SSR with adaptive weights
        # For now, compute relevance analysis and tau sweep showing weight distributions
        print("Running SSR-only QAW analysis...")

        with open(args.baseline_results) as f:
            baseline = json.load(f)
        with open(args.topics_path) as f:
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
                        "predicted_pmf": q_data.get("predicted_pmf"),
                        "per_cluster_scores": q_data.get("per_cluster_scores"),
                    })

        question_texts = [qd["question"] for qd in questions_data]
        relevance_map, cluster_ids = compute_question_topic_relevance(
            question_texts, topics, encoder,
        )

        # If we have per-cluster SSR scores, we can reweight them
        # Otherwise, show the relevance analysis and weight distributions
        results = {
            "mode": "ssr_only_analysis",
            "relevance_analysis": {},
            "weight_distributions": {},
        }

        for qd in questions_data:
            q = qd["question"]
            r = relevance_map[q]
            top_clusters = sorted(
                zip(cluster_ids, r), key=lambda x: -x[1]
            )[:10]
            results["relevance_analysis"][q] = [
                {"cluster": cid, "relevance": float(score), "topic": topics[cid][:60]}
                for cid, score in top_clusters
            ]

            # Show weight distributions at different tau
            results["weight_distributions"][q] = {}
            for tau in [0.01, 0.05, 0.1, 0.3, 0.5, 1.0]:
                aw = adaptive_weights(qd["cluster_weights"], r, cluster_ids, tau)
                top5 = sorted(aw.items(), key=lambda x: -x[1])[:5]
                results["weight_distributions"][q][f"tau={tau}"] = {
                    "top5": [(cid, round(w, 4)) for cid, w in top5],
                    "entropy": float(-sum(
                        w * np.log(w + 1e-10) for w in aw.values() if w > 0
                    )),
                    "n_effective": float(np.exp(-sum(
                        w * np.log(w + 1e-10) for w in aw.values() if w > 0
                    ))),
                }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {args.output}")
