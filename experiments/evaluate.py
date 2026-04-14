"""
Evaluation metrics for survey distribution prediction.
Compares predicted PMFs against true survey distributions.
"""
import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import chi2_contingency, wasserstein_distance
import json
from collections import defaultdict


def normalize(counts: dict) -> np.ndarray:
    """Convert count dict to probability distribution."""
    vals = np.array(list(counts.values()), dtype=float)
    total = vals.sum()
    if total == 0:
        return np.ones(len(vals)) / len(vals)
    return vals / total


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence (0 = identical, 1 = maximally different)."""
    p = np.clip(p, 1e-10, None)
    q = np.clip(q, 1e-10, None)
    p = p / p.sum()
    q = q / q.sum()
    return float(jensenshannon(p, q) ** 2)  # squared JS = JS divergence


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q)."""
    p = np.clip(p, 1e-10, None)
    q = np.clip(q, 1e-10, None)
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def emd(p: np.ndarray, q: np.ndarray) -> float:
    """Earth Mover's Distance (ordinal-aware)."""
    positions = np.arange(len(p))
    return float(wasserstein_distance(positions, positions, p, q))


def mae_mean(p: np.ndarray, q: np.ndarray) -> float:
    """MAE between distribution means."""
    n = len(p)
    scores = np.arange(1, n + 1)
    return float(abs(np.dot(scores, p) - np.dot(scores, q)))


def evaluate_single(true_counts: dict, pred_pmf: list | np.ndarray) -> dict:
    """Evaluate a single question prediction."""
    true_pmf = normalize(true_counts)
    pred = np.array(pred_pmf)
    pred = pred / pred.sum()

    return {
        "js_divergence": js_divergence(true_pmf, pred),
        "kl_divergence": kl_divergence(true_pmf, pred),
        "emd": emd(true_pmf, pred),
        "mae_mean": mae_mean(true_pmf, pred),
        "true_pmf": true_pmf.tolist(),
        "pred_pmf": pred.tolist(),
    }


def evaluate_results(results_path: str) -> dict:
    """Evaluate all questions in a results file."""
    with open(results_path) as f:
        results = json.load(f)

    all_metrics = []
    per_question = {}

    for survey_key, survey_data in results.items():
        for sub_id, questions in survey_data.items():
            for q_id, q_data in questions.items():
                true_dist = q_data["true_distribution"]
                pred_pmf = q_data["predicted_pmf"]

                metrics = evaluate_single(true_dist, pred_pmf)
                qname = f"{survey_key}/{sub_id}/{q_id}"
                per_question[qname] = {
                    "question": q_data["question"],
                    **metrics,
                }
                all_metrics.append(metrics)

    # Aggregate
    summary = {}
    for metric in ["js_divergence", "kl_divergence", "emd", "mae_mean"]:
        vals = [m[metric] for m in all_metrics]
        summary[metric] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }

    return {"summary": summary, "per_question": per_question}


def bootstrap_ci(true_counts: dict, pred_pmf: list, n_bootstrap: int = 1000, ci: float = 0.95) -> dict:
    """Bootstrap confidence interval for JS divergence.

    Resamples from the true count distribution to estimate uncertainty.
    """
    true_pmf = normalize(true_counts)
    pred = np.array(pred_pmf)
    pred = pred / pred.sum()

    total_n = int(sum(true_counts.values()))
    rng = np.random.RandomState(42)
    js_samples = []

    for _ in range(n_bootstrap):
        # Resample from true distribution
        resampled = rng.multinomial(total_n, true_pmf)
        resampled_pmf = resampled / resampled.sum()
        js_samples.append(js_divergence(resampled_pmf, pred))

    alpha = (1 - ci) / 2
    return {
        "mean": float(np.mean(js_samples)),
        "ci_low": float(np.percentile(js_samples, 100 * alpha)),
        "ci_high": float(np.percentile(js_samples, 100 * (1 - alpha))),
        "std": float(np.std(js_samples)),
    }


def paired_permutation_test(
    true_dists: list[dict], pred_a: list[list], pred_b: list[list],
    n_permutations: int = 10000,
) -> float:
    """Paired permutation test: is method A significantly better than B?

    Returns p-value for H0: JS(A) >= JS(B), i.e. A is not better.
    """
    js_a = [js_divergence(normalize(t), np.array(p) / np.array(p).sum()) for t, p in zip(true_dists, pred_a)]
    js_b = [js_divergence(normalize(t), np.array(p) / np.array(p).sum()) for t, p in zip(true_dists, pred_b)]

    observed_diff = np.mean(js_a) - np.mean(js_b)  # negative if A is better
    diffs = np.array(js_a) - np.array(js_b)

    rng = np.random.RandomState(42)
    count = 0
    for _ in range(n_permutations):
        signs = rng.choice([-1, 1], size=len(diffs))
        perm_diff = np.mean(signs * diffs)
        if perm_diff <= observed_diff:
            count += 1

    return count / n_permutations


def entropy(pmf: np.ndarray) -> float:
    """Shannon entropy of a distribution."""
    p = np.clip(pmf, 1e-10, None)
    p = p / p.sum()
    return float(-np.sum(p * np.log2(p)))


def mixture_analysis(per_cluster_pmfs: dict[int, np.ndarray], weights: dict) -> dict:
    """Analyze steer-then-aggregate vs aggregate-then-steer via entropy.

    Shows whether per-cluster steering preserves multimodality while
    aggregate steering collapses it.
    """
    if not per_cluster_pmfs:
        return {}

    # Per-cluster entropies
    cluster_entropies = {
        cid: entropy(pmf) for cid, pmf in per_cluster_pmfs.items()
    }

    # Steer-then-aggregate: weighted mixture
    n_options = len(next(iter(per_cluster_pmfs.values())))
    mixture_pmf = np.zeros(n_options)
    total_w = 0
    for cid, pmf in per_cluster_pmfs.items():
        w = weights.get(str(cid), 0)
        mixture_pmf += w * pmf
        total_w += w
    if total_w > 0:
        mixture_pmf /= total_w

    mixture_entropy = entropy(mixture_pmf)
    mean_component_entropy = float(np.mean(list(cluster_entropies.values())))

    # Diversity: are different clusters giving different answers?
    if len(per_cluster_pmfs) > 1:
        pmf_matrix = np.stack(list(per_cluster_pmfs.values()))
        pairwise_js = []
        for i in range(len(pmf_matrix)):
            for j in range(i + 1, len(pmf_matrix)):
                pairwise_js.append(js_divergence(pmf_matrix[i], pmf_matrix[j]))
        inter_cluster_diversity = float(np.mean(pairwise_js))
    else:
        inter_cluster_diversity = 0.0

    return {
        "mixture_entropy": mixture_entropy,
        "mean_component_entropy": mean_component_entropy,
        "entropy_gap": mixture_entropy - mean_component_entropy,  # positive = mixture is more diverse
        "inter_cluster_diversity": inter_cluster_diversity,
        "n_clusters": len(per_cluster_pmfs),
        "mixture_pmf": mixture_pmf.tolist(),
    }


def entropy_outcome_correlation(
    per_question_diversity: dict[str, float],
    per_question_improvement: dict[str, float],
) -> dict:
    """Test whether questions with higher inter-cluster diversity benefit more from steer-then-aggregate.

    Connects the entropy/mixture analysis to prediction quality:
    If the correlation is positive, it mechanistically supports the mixture hypothesis.
    """
    from scipy.stats import spearmanr

    questions = sorted(set(per_question_diversity.keys()) & set(per_question_improvement.keys()))
    if len(questions) < 3:
        return {"correlation": None, "p_value": None, "n": len(questions), "note": "too few questions"}

    diversities = [per_question_diversity[q] for q in questions]
    improvements = [per_question_improvement[q] for q in questions]

    corr, p_val = spearmanr(diversities, improvements)
    return {
        "spearman_correlation": float(corr),
        "p_value": float(p_val),
        "n": len(questions),
        "interpretation": (
            "Higher inter-cluster diversity correlates with larger steer-then-aggregate advantage"
            if corr > 0 else
            "No positive correlation between diversity and steering advantage"
        ),
    }


def compare_methods(method_results: dict[str, str]) -> dict:
    """Compare multiple methods.

    Args:
        method_results: dict mapping method_name -> results_file_path
    """
    comparisons = {}
    for name, path in method_results.items():
        comparisons[name] = evaluate_results(path)

    # Build comparison table
    print(f"\n{'Method':<30} {'JS Div ↓':>10} {'KL Div ↓':>10} {'EMD ↓':>10} {'MAE Mean ↓':>12}")
    print("-" * 75)
    for name, result in comparisons.items():
        s = result["summary"]
        print(
            f"{name:<30} "
            f"{s['js_divergence']['mean']:>10.4f} "
            f"{s['kl_divergence']['mean']:>10.4f} "
            f"{s['emd']['mean']:>10.4f} "
            f"{s['mae_mean']['mean']:>12.4f}"
        )

    return comparisons


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python evaluate.py <results_file> [results_file2 ...]")
        sys.exit(1)

    if len(sys.argv) == 2:
        result = evaluate_results(sys.argv[1])
        print(json.dumps(result["summary"], indent=2))
    else:
        methods = {Path(p).stem: p for p in sys.argv[1:]}
        from pathlib import Path
        compare_methods(methods)
