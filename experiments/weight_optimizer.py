"""
Phase 2: Weight Determination for cluster combination.

Three approaches:
  A: Reuse Project 1's topic-level SSR weights (no ground truth needed)
  B: Oracle — optimize weights against real survey distributions (upper bound only)
  C: LLM-judged relevance (no ground truth needed)
"""
import numpy as np
import json
from scipy.optimize import minimize
from scipy.spatial.distance import jensenshannon


def weights_from_ssr(cluster_weights_dict: dict) -> dict:
    """Approach A: Use existing topic-level SSR weights from Project 1."""
    # Already computed in baseline results
    return cluster_weights_dict


def weights_from_llm_relevance(cluster_relevance: dict, cluster_weights_dict: dict) -> dict:
    """Approach C: Convert LLM-judged relevance to soft weights.

    Clusters marked '相关' get their SSR weight; '不相关' get downweighted by 0.1x.
    """
    adjusted = {}
    for cid, weight in cluster_weights_dict.items():
        rel = cluster_relevance.get(cid, "不相关")
        if rel == "相关":
            adjusted[cid] = weight
        else:
            adjusted[cid] = weight * 0.1  # heavily downweight irrelevant clusters

    # Re-normalize
    total = sum(adjusted.values())
    if total > 0:
        adjusted = {k: v / total for k, v in adjusted.items()}
    return adjusted


def weights_oracle_optimize(
    predict_fn,
    true_distribution: dict,
    cluster_ids: list,
    initial_weights: dict,
    max_iter: int = 100,
) -> dict:
    """Approach B (Oracle): Optimize weights to minimize JS divergence.

    WARNING: This uses ground truth and should only be reported as an upper bound.

    Args:
        predict_fn: callable(weights_dict) -> predicted_pmf (np.ndarray)
        true_distribution: dict of option -> count
        cluster_ids: list of cluster IDs to optimize over
        initial_weights: starting weights from Approach A
    """
    true_counts = np.array(list(true_distribution.values()), dtype=float)
    true_pmf = true_counts / true_counts.sum()

    # Initial weight vector
    w0 = np.array([initial_weights.get(str(cid), 0.0) for cid in cluster_ids])
    w0 = w0 / (w0.sum() + 1e-10)

    def objective(w):
        w = np.clip(w, 0, None)
        w = w / (w.sum() + 1e-10)
        weights_dict = {str(cid): float(wi) for cid, wi in zip(cluster_ids, w)}
        pred_pmf = predict_fn(weights_dict)
        return jensenshannon(true_pmf, pred_pmf) ** 2

    result = minimize(
        objective, w0,
        method="L-BFGS-B",
        bounds=[(0, 1)] * len(w0),
        options={"maxiter": max_iter},
    )

    optimized_w = np.clip(result.x, 0, None)
    optimized_w = optimized_w / (optimized_w.sum() + 1e-10)

    return {
        "weights": {str(cid): float(wi) for cid, wi in zip(cluster_ids, optimized_w)},
        "js_divergence": float(result.fun),
        "success": result.success,
    }


def leave_one_out_oracle(
    predict_fn,
    questions: list[dict],
    cluster_ids: list,
) -> dict:
    """Leave-one-question-out oracle weight optimization.

    For each question, optimize weights on all OTHER questions, then evaluate on held-out.
    This is still an oracle (uses ground truth) but with proper train/test split.
    """
    results = []
    for i, held_out in enumerate(questions):
        # Train on all except i
        train_qs = [q for j, q in enumerate(questions) if j != i]

        # Simple: average optimal weights across training questions
        # (More sophisticated: joint optimization, but 5 questions is too few)
        all_train_weights = []
        for tq in train_qs:
            initial_w = tq.get("cluster_weights", {})
            opt = weights_oracle_optimize(
                predict_fn, tq["true_distribution"],
                cluster_ids, initial_w,
            )
            all_train_weights.append(opt["weights"])

        # Average the optimized weights
        avg_weights = {}
        for cid in cluster_ids:
            vals = [w.get(str(cid), 0) for w in all_train_weights]
            avg_weights[str(cid)] = float(np.mean(vals))

        # Normalize
        total = sum(avg_weights.values())
        if total > 0:
            avg_weights = {k: v / total for k, v in avg_weights.items()}

        # Evaluate on held-out
        pred_pmf = predict_fn(avg_weights)
        true_counts = np.array(list(held_out["true_distribution"].values()), dtype=float)
        true_pmf = true_counts / true_counts.sum()
        js = jensenshannon(true_pmf, pred_pmf) ** 2

        results.append({
            "held_out_question": held_out.get("question", f"Q{i}"),
            "js_divergence": float(js),
            "optimized_weights_sample": dict(list(avg_weights.items())[:5]),
        })

    return {
        "mean_js": float(np.mean([r["js_divergence"] for r in results])),
        "per_question": results,
    }
