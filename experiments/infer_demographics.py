"""
Infer demographics (age, gender) from social media text using LLM.

Inspired by Giorgi et al. who use classifiers to estimate demographics,
and Hu et al. who extract persona profiles from social media histories.

For each cluster, sample posts and ask Qwen3-8B to infer likely demographics.
Build per-cluster demographic profiles, then use for importance reweighting.
"""
import torch
import json
import re
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from tqdm import tqdm

from persona_vectors import load_model


INFER_PROMPT = """根据以下社交媒体帖子，推测作者最可能的人口统计特征。
只基于帖子内容推断，如果无法判断则回答"无法判断"。

帖子内容：
{post}

请严格按以下JSON格式回答：
{{"age": "24岁以下/24-35岁/35-50岁/50岁以上/无法判断", "gender": "男/女/无法判断"}}"""


def infer_demographics_for_cluster(
    model, tokenizer, posts: list[str], max_posts: int = 15
) -> dict:
    """Infer demographics for a sample of posts from one cluster.

    Returns:
        {"age": Counter, "gender": Counter, "n_inferred": int}
    """
    age_counter = Counter()
    gender_counter = Counter()
    n_inferred = 0

    for post in posts[:max_posts]:
        # Truncate long posts
        post_text = post[:300] if len(post) > 300 else post
        prompt = INFER_PROMPT.format(post=post_text)

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=100, do_sample=False,
            )
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:],
                                    skip_special_tokens=True)

        # Strip thinking tags
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()

        # Parse JSON
        try:
            match = re.search(r'\{[^}]+\}', response)
            if match:
                result = json.loads(match.group())
                age = result.get("age", "无法判断")
                gender = result.get("gender", "无法判断")
                age_counter[age] += 1
                gender_counter[gender] += 1
                n_inferred += 1
        except (json.JSONDecodeError, KeyError):
            pass

    return {
        "age": dict(age_counter),
        "gender": dict(gender_counter),
        "n_inferred": n_inferred,
    }


def build_cluster_demographics(
    model, tokenizer, meaningful_df: pd.DataFrame,
    n_samples_per_cluster: int = 15,
) -> dict:
    """Build demographic profiles for all clusters.

    Returns:
        dict mapping cluster_id -> {age: Counter, gender: Counter}
    """
    cluster_ids = sorted(meaningful_df["cluster_label"].unique())
    demographics = {}

    for cid in tqdm(cluster_ids, desc="Inferring demographics"):
        posts = (meaningful_df[meaningful_df["cluster_label"] == cid]["content_desc"]
                .dropna()
                .sample(min(n_samples_per_cluster,
                           len(meaningful_df[meaningful_df["cluster_label"] == cid])),
                       random_state=42)
                .tolist())

        if len(posts) < 3:
            continue

        demo = infer_demographics_for_cluster(model, tokenizer, posts)
        demographics[int(cid)] = demo

    return demographics


def compute_demographic_is_weights(
    cluster_demographics: dict,
    survey_age: Counter,
    survey_gender: Counter = None,
    smoothing: float = 1.0,
) -> dict:
    """Compute importance sampling weights based on inferred demographics.

    For each cluster, compute:
      w_c = sum_d P_survey(d) * P_cluster(d) / P_sm_overall(d)

    where d = (age, gender) or just age.
    """
    # Normalize survey age distribution
    age_bins = ["24岁以下", "24-35岁", "24岁-35岁", "24 岁(不含)以下", "24 岁-35 岁(不含)以下",
                "35岁-50岁", "35 岁-50 岁(不含)以下", "50岁以上", "无法判断",
                "24岁（不含）以下", "24 岁-35 岁（不含）以下", "35 岁-50 岁（不含）以下"]

    # Map survey age bins to standard bins
    def normalize_age(age_str):
        if any(k in age_str for k in ["24岁以下", "24 岁(不含)以下", "24岁（不含）以下", "20及以下"]):
            return "<24"
        elif any(k in age_str for k in ["24-35", "24 岁-35", "24岁-35", "21-30", "21-20"]):
            return "24-35"
        elif any(k in age_str for k in ["35-50", "35 岁-50"]):
            return "35-50"
        elif any(k in age_str for k in ["50岁以上", "50以上", "41-50"]):
            return "50+"
        return "unknown"

    # Survey age distribution (standardized)
    survey_age_std = Counter()
    for age_str, cnt in survey_age.items():
        survey_age_std[normalize_age(age_str)] += cnt

    # Remove unknown
    if "unknown" in survey_age_std:
        del survey_age_std["unknown"]
    sv_total = sum(survey_age_std.values())

    # Overall SM age distribution from all clusters
    sm_age_overall = Counter()
    for cid, demo in cluster_demographics.items():
        for age_str, cnt in demo["age"].items():
            sm_age_overall[normalize_age(age_str)] += cnt
    if "unknown" in sm_age_overall:
        del sm_age_overall["unknown"]
    sm_total = sum(sm_age_overall.values())

    all_ages = sorted(set(survey_age_std.keys()) | set(sm_age_overall.keys()) - {"unknown"})

    # Importance ratios per age bin
    age_ratio = {}
    for age in all_ages:
        p_sv = (survey_age_std.get(age, 0) + smoothing) / (sv_total + smoothing * len(all_ages))
        p_sm = (sm_age_overall.get(age, 0) + smoothing) / (sm_total + smoothing * len(all_ages))
        age_ratio[age] = p_sv / p_sm

    print(f"\n  Age importance ratios:", flush=True)
    for age in sorted(age_ratio.keys()):
        sv_pct = survey_age_std.get(age, 0) / sv_total * 100
        sm_pct = sm_age_overall.get(age, 0) / sm_total * 100
        print(f"    {age}: survey={sv_pct:.1f}% SM={sm_pct:.1f}% ratio={age_ratio[age]:.2f}", flush=True)

    # Per-cluster weights
    weights = {}
    for cid, demo in cluster_demographics.items():
        cluster_age = Counter()
        for age_str, cnt in demo["age"].items():
            cluster_age[normalize_age(age_str)] += cnt
        if "unknown" in cluster_age:
            del cluster_age["unknown"]
        c_total = sum(cluster_age.values())
        if c_total == 0:
            weights[str(cid)] = 1.0
            continue

        w = sum((cluster_age.get(age, 0) / c_total) * age_ratio.get(age, 1.0)
                for age in all_ages)
        weights[str(cid)] = w

    # Normalize mean to 1
    mean_w = np.mean(list(weights.values()))
    if mean_w > 0:
        weights = {k: v / mean_w for k, v in weights.items()}

    return weights


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="/data/chenhongrui/models/Qwen3-8B")
    parser.add_argument("--data_path", default="/data/chenhongrui/business/data/2_meaningful_df.csv")
    parser.add_argument("--survey_path", default="/data/chenhongrui/business/data/2025-05-31-10-01-50_EXPORT_CSV_19470858_516_ds_survey_feedback_0.csv")
    parser.add_argument("--output", default="/data/chenhongrui/business/results/cluster_demographics.json")
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--n_samples", type=int, default=15)
    args = parser.parse_args()

    # Load model
    model, tokenizer = load_model(args.model_path, args.device)

    # Load data
    df = pd.read_csv(args.data_path)
    print(f"Loaded {len(df)} posts, {df['cluster_label'].nunique()} clusters", flush=True)

    # Infer demographics
    demographics = build_cluster_demographics(model, tokenizer, df, args.n_samples)

    # Save
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(demographics, f, ensure_ascii=False, indent=2)
    print(f"\nSaved demographics to {args.output}", flush=True)

    # Show summary
    print("\n=== Cluster Demographics Summary ===", flush=True)
    all_age = Counter()
    all_gender = Counter()
    for cid, demo in demographics.items():
        for age, cnt in demo["age"].items():
            all_age[age] += cnt
        for g, cnt in demo["gender"].items():
            all_gender[g] += cnt

    total = sum(all_age.values())
    print("  Age distribution (inferred from SM):", flush=True)
    for age, cnt in all_age.most_common():
        print(f"    {age}: {cnt} ({100*cnt/total:.1f}%)", flush=True)
    print("  Gender distribution (inferred from SM):", flush=True)
    for g, cnt in all_gender.most_common():
        print(f"    {g}: {cnt} ({100*cnt/total:.1f}%)", flush=True)

    # Compute importance weights
    from demographic_reweight import compute_province_distributions
    _, _, survey_prov, survey_age = compute_province_distributions(
        args.data_path, args.survey_path,
    )
    demo_weights = compute_demographic_is_weights(demographics, survey_age)

    print("\n  Top clusters by demographic weight deviation:", flush=True)
    for cid, w in sorted(demo_weights.items(), key=lambda x: -abs(x[1]-1.0))[:10]:
        print(f"    Cluster {cid}: weight={w:.3f}", flush=True)
