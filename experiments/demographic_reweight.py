"""
Demographic Post-Stratification for Cluster Weights.

Inspired by:
- Giorgi et al. "Correcting Sociodemographic Selection Biases" (robust post-stratification)
- Hu et al. "Population-Aligned Persona Generation" (importance sampling + OT)
- Wang et al. "Forecasting elections with non-representative polls" (MRP)

Core idea: social media users are not representative of the survey population.
Clusters with demographic profiles closer to the survey population should get
higher weights. We implement this as importance reweighting at the cluster level.

Available demographics:
- Social media: pred_province (per-user)
- Survey: age (4 bins), province (~34 provinces)

Since we only have province overlap, we do province-based post-stratification:
  w_c^demo = sum_prov P_survey(prov) * I(cluster c has users from prov) / Q_sm(prov|cluster c)

Combined with QAW:
  w_c(q) = w_c^base * w_c^demo * relevance(q, c)^(1/tau)
"""
import numpy as np
import pandas as pd
import json
from collections import Counter, defaultdict


def compute_province_distributions(meaningful_df_path, survey_df_path):
    """Compute province distributions for social media clusters and survey.

    Returns:
        cluster_prov: dict[cluster_id] -> Counter(province -> count)
        sm_prov: Counter(province -> count) for all social media
        survey_prov: Counter(province -> count) for survey respondents
        survey_age: Counter(age_bin -> count)
    """
    sm = pd.read_csv(meaningful_df_path)
    cluster_prov = defaultdict(Counter)
    sm_prov = Counter()

    for _, row in sm.iterrows():
        cid = row['cluster_label']
        prov = row.get('pred_province')
        if pd.notna(prov):
            cluster_prov[cid][prov] += 1
            sm_prov[prov] += 1

    # Survey demographics
    df = pd.read_csv(survey_df_path, low_memory=False)
    survey_prov = Counter()
    survey_age = Counter()

    for _, row in df.iterrows():
        try:
            content = json.loads(row['content'])
            for item in content.get('list', []):
                q = item.get('question', '')
                if '常驻地' in q or '常住地' in q:
                    for ans in item.get('answers', []):
                        if ans.get('selected'):
                            survey_prov[ans['text']] += 1
                if '年龄' in q:
                    for ans in item.get('answers', []):
                        if ans.get('selected'):
                            survey_age[ans['text']] += 1
        except:
            pass

    return dict(cluster_prov), sm_prov, survey_prov, survey_age


def compute_demographic_weights(
    cluster_prov: dict,
    sm_prov: Counter,
    survey_prov: Counter,
    method: str = "cluster_is",
    smoothing: float = 1.0,
) -> dict:
    """Compute demographic correction weights for each cluster.

    Methods:
    - "global_is": importance sampling using global SM vs survey province ratios,
      then aggregate per cluster based on its province composition.
    - "cluster_is": directly compute per-cluster importance weight as
      sum_prov P_survey(prov) * P_cluster(prov) / P_sm(prov), with smoothing.
    - "kl_penalty": penalize clusters whose province distribution diverges
      from survey distribution.

    Args:
        smoothing: Laplace smoothing for sparse provinces (adaptive binning idea
                   from Giorgi et al.)
    Returns:
        dict mapping cluster_id (str) -> weight (float)
    """
    # Normalize to distributions
    sm_total = sum(sm_prov.values())
    sv_total = sum(survey_prov.values())
    all_provs = sorted(set(sm_prov.keys()) | set(survey_prov.keys()))

    # Province-level importance ratios: P_survey(prov) / P_sm(prov)
    prov_ratio = {}
    for prov in all_provs:
        p_sm = (sm_prov.get(prov, 0) + smoothing) / (sm_total + smoothing * len(all_provs))
        p_sv = (survey_prov.get(prov, 0) + smoothing) / (sv_total + smoothing * len(all_provs))
        prov_ratio[prov] = p_sv / p_sm

    demo_weights = {}

    if method == "global_is":
        # Per-user importance weight, averaged within cluster
        for cid, provs in cluster_prov.items():
            c_total = sum(provs.values())
            if c_total == 0:
                demo_weights[str(cid)] = 1.0
                continue
            # Average importance weight for users in this cluster
            w = sum(provs[prov] * prov_ratio.get(prov, 1.0) for prov in provs) / c_total
            demo_weights[str(cid)] = w

    elif method == "cluster_is":
        # Cluster-level IS: how well does this cluster's province profile
        # match the survey population?
        # w_c = sum_prov P_survey(prov) * P_cluster(prov|has_prov) / P_sm(prov)
        for cid, provs in cluster_prov.items():
            c_total = sum(provs.values())
            if c_total == 0:
                demo_weights[str(cid)] = 1.0
                continue
            w = 0
            for prov, cnt in provs.items():
                p_cluster_prov = cnt / c_total
                w += p_cluster_prov * prov_ratio.get(prov, 1.0)
            demo_weights[str(cid)] = w

    elif method == "kl_penalty":
        # Penalize clusters whose province distribution is far from survey
        from scipy.special import rel_entr
        for cid, provs in cluster_prov.items():
            c_total = sum(provs.values())
            if c_total < 5:
                demo_weights[str(cid)] = 1.0
                continue
            # Build province pmf for cluster and survey
            p_c = np.array([(provs.get(prov, 0) + smoothing) / (c_total + smoothing * len(all_provs))
                           for prov in all_provs])
            p_s = np.array([(survey_prov.get(prov, 0) + smoothing) / (sv_total + smoothing * len(all_provs))
                           for prov in all_provs])
            p_c /= p_c.sum()
            p_s /= p_s.sum()
            kl = np.sum(rel_entr(p_s, p_c))
            # Convert KL to weight: lower KL = higher weight
            demo_weights[str(cid)] = np.exp(-kl)

    # Normalize so mean weight = 1 (preserves scale of base weights)
    mean_w = np.mean(list(demo_weights.values()))
    if mean_w > 0:
        demo_weights = {k: v / mean_w for k, v in demo_weights.items()}

    return demo_weights


def combined_weights(
    base_weights: dict,
    demo_weights: dict,
    relevance_scores: np.ndarray = None,
    cluster_ids: list = None,
    tau: float = 0.3,
    demo_strength: float = 1.0,
) -> dict:
    """Combine base weights, demographic correction, and question relevance.

    w_c(q) = w_c^base * (w_c^demo)^demo_strength * softmax(relevance/tau)_c

    Args:
        base_weights: original cluster weights (cluster sizes)
        demo_weights: demographic post-stratification weights
        relevance_scores: per-question cosine similarities (optional)
        cluster_ids: cluster ID order for relevance_scores (optional)
        tau: temperature for relevance softmax
        demo_strength: exponent on demographic weights (0=ignore, 1=full)
    """
    combined = {}
    for cid, base_w in base_weights.items():
        dw = demo_weights.get(cid, 1.0) ** demo_strength
        combined[cid] = base_w * dw

    # Apply question relevance if provided
    if relevance_scores is not None and cluster_ids is not None:
        r = relevance_scores / max(tau, 1e-6)
        r = r - r.max()
        exp_r = np.exp(r)
        softmax_r = exp_r / exp_r.sum()

        for i, cid in enumerate(cluster_ids):
            if cid in combined:
                combined[cid] *= softmax_r[i]

    # Normalize
    total = sum(combined.values())
    if total > 0:
        combined = {k: v / total for k, v in combined.items()}

    return combined


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--meaningful_df", default="/data/chenhongrui/business/data/2_meaningful_df.csv")
    parser.add_argument("--survey_df", default="/data/chenhongrui/business/data/2025-05-31-10-01-50_EXPORT_CSV_19470858_516_ds_survey_feedback_0.csv")
    parser.add_argument("--method", choices=["global_is", "cluster_is", "kl_penalty"], default="cluster_is")
    parser.add_argument("--smoothing", type=float, default=1.0)
    args = parser.parse_args()

    cluster_prov, sm_prov, survey_prov, survey_age = compute_province_distributions(
        args.meaningful_df, args.survey_df
    )

    demo_weights = compute_demographic_weights(
        cluster_prov, sm_prov, survey_prov,
        method=args.method, smoothing=args.smoothing,
    )

    print(f"Method: {args.method}, smoothing={args.smoothing}")
    print(f"Demo weights (top 10 by deviation from 1.0):")
    sorted_w = sorted(demo_weights.items(), key=lambda x: -abs(x[1] - 1.0))
    for cid, w in sorted_w[:10]:
        print(f"  Cluster {cid}: {w:.3f}")

    print(f"\nWeight stats: mean={np.mean(list(demo_weights.values())):.3f}, "
          f"std={np.std(list(demo_weights.values())):.3f}, "
          f"min={min(demo_weights.values()):.3f}, max={max(demo_weights.values()):.3f}")
