"""
Baseline methods for comparison:
1. Direct LLM: Ask Qwen3-8B to directly answer (numeric rating)
2. LLM-as-persona: Prompt-based persona simulation (no steering, just description)
3. SSR-only: Already done in Project 1 (eps_0.1.json)
"""
import torch
import numpy as np
import json
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from collections import Counter

from persona_vectors import load_model
from steered_ssr import ssr_score, generate_anchors_local


# ─── Baseline 1: Direct LLM ───

DIRECT_PROMPT = """你是一位普通消费者。请直接回答以下问卷问题，只输出选项内容。

问题：{question}
选项：{options}

你的选择："""


def baseline_direct_llm(
    model, tokenizer,
    question: str, options: list[str],
    n_samples: int = 100,
) -> np.ndarray:
    """Ask LLM to directly choose an option, repeat n times to get distribution."""
    prompt = DIRECT_PROMPT.format(question=question, options="、".join(options))
    counts = Counter()

    for _ in range(n_samples):
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=50,
                do_sample=True, temperature=0.8, top_p=0.9,
            )
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

        # Match response to closest option
        best_match = None
        best_score = -1
        for opt in options:
            if opt in response:
                score = len(opt)
                if score > best_score:
                    best_match = opt
                    best_score = score
        if best_match is None:
            best_match = options[0]  # fallback

        counts[best_match] += 1

    pmf = np.array([counts.get(opt, 0) for opt in options], dtype=float)
    pmf = pmf / pmf.sum()
    return pmf


# ─── Baseline 3: LLM-as-persona (prompt-based, no steering) ───

PERSONA_PROMPT = """你是一位消费者，你的特征如下：
{persona_description}

请基于你的身份和经验，回答以下问卷问题。
请用自然的语言表达你的真实想法和感受。

问题：{question}
选项：{options}

你的回答："""


def baseline_persona_prompt(
    model, tokenizer, encoder,
    cluster_topics: dict,
    cluster_weights: dict,
    question: str, options: list[str],
    anchor_embeddings: list,
    n_responses: int = 5,
) -> np.ndarray:
    """Prompt-based persona simulation: describe cluster persona in prompt."""
    aggregated_pmf = np.zeros(len(options))
    total_weight = 0

    # Only use top-20 clusters by weight to keep runtime manageable
    sorted_clusters = sorted(cluster_weights.items(), key=lambda x: -x[1])[:20]
    for cid_str, weight in sorted_clusters:
        if weight < 1e-6:
            continue
        if cid_str not in cluster_topics:
            continue

        topic = cluster_topics[cid_str]
        prompt = PERSONA_PROMPT.format(
            persona_description=f"你主要关注的话题是：{topic}",
            question=question,
            options="、".join(options),
        )

        cluster_pmfs = []
        for _ in range(n_responses):
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=200,
                    do_sample=True, temperature=0.7, top_p=0.9,
                )
            response = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )
            pmf, _ = ssr_score(response, encoder, anchor_embeddings)
            cluster_pmfs.append(pmf)

        cluster_mean = np.mean(cluster_pmfs, axis=0)
        aggregated_pmf += weight * cluster_mean
        total_weight += weight

    if total_weight > 0:
        aggregated_pmf /= total_weight
    return aggregated_pmf


# ─── Baseline 4: Cluster-summary prompting (no steering, richer context) ───

CLUSTER_SUMMARY_PROMPT = """你是一位消费者调研分析专家。

以下是社交媒体上关于该产品的用户讨论主题及其权重：
{topic_list}

基于这些真实用户讨论的分布，请估计在问卷中用户会如何回答以下问题。
请用自然的语言描述你认为最可能的回答倾向。

问题：{question}
选项：{options}

你的分析："""


def baseline_cluster_summary(
    model, tokenizer, encoder,
    cluster_topics: dict,
    cluster_weights: dict,
    question: str, options: list[str],
    anchor_embeddings: list,
    n_responses: int = 10,
) -> np.ndarray:
    """Cluster-summary prompting: give LLM all topic summaries + weights, no steering."""
    # Build topic list string with weights
    sorted_topics = sorted(cluster_weights.items(), key=lambda x: -x[1])
    topic_lines = []
    for cid_str, weight in sorted_topics[:15]:  # top 15 topics
        if cid_str in cluster_topics and weight > 0.01:
            topic_lines.append(f"- [{weight:.1%}] {cluster_topics[cid_str]}")
    topic_list = "\n".join(topic_lines)

    prompt = CLUSTER_SUMMARY_PROMPT.format(
        topic_list=topic_list,
        question=question,
        options="、".join(options),
    )

    pmfs = []
    for _ in range(n_responses):
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=300,
                do_sample=True, temperature=0.7, top_p=0.9,
            )
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        pmf, _ = ssr_score(response, encoder, anchor_embeddings)
        pmfs.append(pmf)

    return np.mean(pmfs, axis=0)


# ─── Baseline 5: Retrieval-augmented prompting ───

RETRIEVAL_PROMPT = """你是一位消费者。以下是一些真实用户对该产品的评价：

{sample_comments}

基于这些真实用户的声音，请回答以下问卷问题。
请用自然的语言表达你的看法。

问题：{question}
选项：{options}

你的回答："""


def baseline_retrieval_augmented(
    model, tokenizer, encoder,
    meaningful_df,
    cluster_weights: dict,
    question: str, options: list[str],
    anchor_embeddings: list,
    n_responses: int = 10,
    n_comments: int = 10,
) -> np.ndarray:
    """Retrieval-augmented: sample representative comments as context."""
    # Sample comments weighted by cluster relevance
    all_comments = []
    for cid_str, weight in sorted(cluster_weights.items(), key=lambda x: -x[1])[:10]:
        cid = int(cid_str)
        cluster_comments = meaningful_df[meaningful_df["cluster_label"] == cid]["content_desc"].dropna()
        if len(cluster_comments) > 0:
            n_sample = max(1, int(n_comments * weight / sum(
                w for _, w in sorted(cluster_weights.items(), key=lambda x: -x[1])[:10]
            )))
            sampled = cluster_comments.sample(min(n_sample, len(cluster_comments)), random_state=42)
            all_comments.extend(sampled.tolist())

    comments_text = "\n".join(f"- {c[:100]}" for c in all_comments[:n_comments])

    prompt = RETRIEVAL_PROMPT.format(
        sample_comments=comments_text,
        question=question,
        options="、".join(options),
    )

    pmfs = []
    for _ in range(n_responses):
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=200,
                do_sample=True, temperature=0.7, top_p=0.9,
            )
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        pmf, _ = ssr_score(response, encoder, anchor_embeddings)
        pmfs.append(pmf)

    return np.mean(pmfs, axis=0)


# ─── Baseline 6: Direct comment-level SSR without generation ───

def baseline_direct_comment_ssr(
    encoder,
    meaningful_df,
    cluster_weights: dict,
    anchor_embeddings: list,
) -> np.ndarray:
    """Direct SSR on raw comments without LLM generation — pure embedding baseline."""
    aggregated_pmf = np.zeros(len(anchor_embeddings))
    total_weight = 0

    for cid_str, weight in cluster_weights.items():
        if weight < 1e-6:
            continue
        cid = int(cid_str)
        comments = meaningful_df[meaningful_df["cluster_label"] == cid]["content_desc"].dropna()
        if len(comments) == 0:
            continue

        sampled = comments.sample(min(30, len(comments)), random_state=42).tolist()
        cluster_pmfs = []
        for comment in sampled:
            pmf, _ = ssr_score(comment, encoder, anchor_embeddings)
            cluster_pmfs.append(pmf)

        cluster_mean = np.mean(cluster_pmfs, axis=0)
        aggregated_pmf += weight * cluster_mean
        total_weight += weight

    if total_weight > 0:
        aggregated_pmf /= total_weight
    return aggregated_pmf


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="/data/chenhongrui/models/Qwen3-8B")
    parser.add_argument("--baseline_results", default="/data/chenhongrui/business/results/all_questions.json")
    parser.add_argument("--topics_path", default="/data/chenhongrui/business/data/3_cluster_topics.json")
    parser.add_argument("--output_dir", default="/data/chenhongrui/business/results")
    parser.add_argument("--method", choices=["direct", "persona_prompt"], required=True)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--embedding_model", default="/data/chenhongrui/models/bge-base-zh-v1.5")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model_path, args.device)

    with open(args.baseline_results) as f:
        baseline = json.load(f)
    with open(args.topics_path) as f:
        cluster_topics = json.load(f)

    if args.method in ["persona_prompt"]:
        encoder = SentenceTransformer(args.embedding_model)

    results = {}
    for survey_key, survey_data in baseline.items():
        results[survey_key] = {}
        for sub_id, questions in survey_data.items():
            results[survey_key][sub_id] = {}
            for q_id, q_data in tqdm(questions.items(), desc=f"{args.method} {survey_key}/{sub_id}"):
                question = q_data["question"]
                options = q_data["options"]

                if args.method == "direct":
                    pred_pmf = baseline_direct_llm(model, tokenizer, question, options)
                elif args.method == "persona_prompt":
                    anchors = generate_anchors_local(model, tokenizer, question, options)
                    anchor_embs = [encoder.encode(a) for a in anchors]
                    pred_pmf = baseline_persona_prompt(
                        model, tokenizer, encoder,
                        cluster_topics, q_data["cluster_weights"],
                        question, options, anchor_embs,
                    )

                results[survey_key][sub_id][q_id] = {
                    "question": question,
                    "options": options,
                    "true_distribution": q_data["true_distribution"],
                    "predicted_pmf": pred_pmf.tolist(),
                    "method": args.method,
                }

    out_path = f"{args.output_dir}/baseline_{args.method}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved to {out_path}")
