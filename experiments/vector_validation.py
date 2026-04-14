"""
Persona Vector Validation.

Validates that extracted persona vectors encode meaningful cluster-discriminative
directions rather than generic topic/style/sentiment artifacts.

Methods:
1. Generate responses under each cluster's steering vector using neutral prompts
2. Use LLM as judge to label expressed attributes
3. Compare labeled attributes to cluster summaries
4. Include random/shuffled vectors as negative controls
"""
import torch
import numpy as np
import json
import pandas as pd
from tqdm import tqdm

from persona_vectors import load_model, load_persona_vectors
from steered_ssr import SteeringHook, create_control_vectors


NEUTRAL_PROMPTS = [
    "请你随便聊聊你最近的生活感受。",
    "你对健康养生有什么看法？",
    "描述一下你日常使用的产品。",
    "你觉得什么东西对你的生活很重要？",
    "谈谈你最近的消费体验。",
]

JUDGE_PROMPT = """你是一位用户研究专家。请分析以下文本，判断作者表达了什么样的消费者态度和关注点。

文本：
{text}

请输出JSON格式：
{{
  "topics": ["话题1", "话题2"],
  "sentiment": "正面/中性/负面",
  "product_relation": "直接使用/间接提及/无关",
  "key_attitude": "一句话总结作者的核心态度"
}}"""


def validate_persona_vectors(
    model, tokenizer,
    persona_vectors: dict,
    cluster_topics: dict,
    alpha: float = 2.0,
    layer: int = 16,
    n_prompts: int = 3,
) -> dict:
    """Validate persona vectors by generating steered responses and judging consistency.

    Returns validation report with:
    - Per-cluster: generated texts, judge labels, consistency score
    - Control comparison: random/shuffled vs real vectors
    """
    results = {"real_vectors": {}, "random_vectors": {}, "shuffled_vectors": {}}

    # Create control vectors
    random_vecs = create_control_vectors(persona_vectors, "random")
    shuffled_vecs = create_control_vectors(persona_vectors, "shuffled")

    for vector_type, vectors in [
        ("real_vectors", persona_vectors),
        ("random_vectors", random_vecs),
        ("shuffled_vectors", shuffled_vecs),
    ]:
        for cid in tqdm(list(vectors.keys())[:20], desc=f"Validating {vector_type}"):
            vec = vectors[cid]["vector"]
            generated_texts = []

            for prompt in NEUTRAL_PROMPTS[:n_prompts]:
                # Steer and generate
                inputs = tokenizer(prompt, return_tensors="pt")
                prompt_len = inputs["input_ids"].shape[1]
                hook = SteeringHook(vec, alpha, layer, mode="all", prompt_length=prompt_len)
                hook.attach(model)
                try:
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs, max_new_tokens=150,
                            do_sample=True, temperature=0.7, top_p=0.9,
                        )
                    response = tokenizer.decode(
                        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
                    )
                    generated_texts.append(response)
                finally:
                    hook.remove()

            # Use the same model as judge
            judge_results = []
            for text in generated_texts:
                judge_prompt = JUDGE_PROMPT.format(text=text)
                judge_inputs = tokenizer(judge_prompt, return_tensors="pt")
                judge_inputs = {k: v.to(model.device) for k, v in judge_inputs.items()}
                with torch.no_grad():
                    judge_outputs = model.generate(
                        **judge_inputs, max_new_tokens=300, do_sample=False,
                    )
                judge_response = tokenizer.decode(
                    judge_outputs[0][judge_inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                )
                judge_results.append(judge_response)

            results[vector_type][int(cid)] = {
                "cluster_topic": cluster_topics.get(str(cid), "unknown"),
                "generated_texts": generated_texts,
                "judge_labels": judge_results,
            }

    return results


def compute_consistency_scores(validation_results: dict, cluster_topics: dict) -> dict:
    """Compute how well steered outputs match their cluster topics.

    Uses BOTH:
    1. Embedding-based semantic similarity (independent of the generation model)
    2. Character-level keyword overlap (simple proxy)
    """
    from sentence_transformers import SentenceTransformer

    # Use an INDEPENDENT embedding model (not Qwen3-8B) for evaluation
    eval_encoder = SentenceTransformer("BAAI/bge-base-zh-v1.5")

    scores = {}
    for vector_type in ["real_vectors", "random_vectors", "shuffled_vectors"]:
        semantic_scores = []
        keyword_scores = []
        for cid, data in validation_results[vector_type].items():
            topic = data["cluster_topic"]

            # Semantic similarity (independent model)
            topic_emb = eval_encoder.encode(topic)
            for text in data["generated_texts"]:
                text_emb = eval_encoder.encode(text)
                sim = float(np.dot(topic_emb, text_emb) / (
                    np.linalg.norm(topic_emb) * np.linalg.norm(text_emb) + 1e-8
                ))
                semantic_scores.append(sim)

            # Keyword overlap (simple proxy)
            topic_keywords = set(topic)
            for text in data["generated_texts"]:
                text_chars = set(text)
                overlap = len(topic_keywords & text_chars) / max(len(topic_keywords), 1)
                keyword_scores.append(overlap)

        scores[vector_type] = {
            "semantic_similarity": {
                "mean": float(np.mean(semantic_scores)) if semantic_scores else 0,
                "std": float(np.std(semantic_scores)) if semantic_scores else 0,
            },
            "keyword_overlap": {
                "mean": float(np.mean(keyword_scores)) if keyword_scores else 0,
                "std": float(np.std(keyword_scores)) if keyword_scores else 0,
            },
            "n_samples": len(semantic_scores),
        }

    return scores


def validate_with_survey_prompts(
    model, tokenizer,
    persona_vectors: dict,
    questions: list[dict],
    alpha: float = 2.0,
    layer: int = 16,
) -> dict:
    """Targeted validation: test whether steered outputs shift in expected survey-option direction.

    For each question × cluster, generate a steered response and check if the
    SSR distribution moves toward the expected option given the cluster's topic.
    """
    from steered_ssr import ssr_score
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer("BAAI/bge-base-zh-v1.5")
    results = []

    for q_data in questions:
        question = q_data["question"]
        options = q_data["options"]
        # Generate anchors for this question
        anchor_embs = [encoder.encode(opt) for opt in options]

        for cid in list(persona_vectors.keys())[:10]:
            vec = persona_vectors[cid]["vector"]

            # Steered response
            prompt = f"请回答：{question}"
            inputs = tokenizer(prompt, return_tensors="pt")
            prompt_len = inputs["input_ids"].shape[1]

            hook = SteeringHook(vec, alpha, layer, mode="all", prompt_length=prompt_len)
            hook.attach(model)
            try:
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs, max_new_tokens=150,
                        do_sample=True, temperature=0.7, top_p=0.9,
                    )
                response = tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
                )
            finally:
                hook.remove()

            # Unsteered response (baseline)
            inputs2 = tokenizer(prompt, return_tensors="pt")
            inputs2 = {k: v.to(model.device) for k, v in inputs2.items()}
            with torch.no_grad():
                outputs2 = model.generate(
                    **inputs2, max_new_tokens=150,
                    do_sample=True, temperature=0.7, top_p=0.9,
                )
            baseline_response = tokenizer.decode(
                outputs2[0][inputs2["input_ids"].shape[1]:], skip_special_tokens=True
            )

            steered_pmf, _ = ssr_score(response, encoder, anchor_embs)
            baseline_pmf, _ = ssr_score(baseline_response, encoder, anchor_embs)

            results.append({
                "cluster_id": int(cid),
                "question": question,
                "steered_pmf": steered_pmf.tolist(),
                "baseline_pmf": baseline_pmf.tolist(),
                "pmf_shift": (steered_pmf - baseline_pmf).tolist(),
                "steered_text_sample": response[:200],
                "baseline_text_sample": baseline_response[:200],
            })

    return results


def analyze_pmf_shift_directionality(
    validation_results: list[dict],
    true_distributions: dict,
) -> dict:
    """Check if steering moves PMFs TOWARD ground truth, not just anywhere.

    For each steered response, compare:
    - JS(steered, true) vs JS(unsteered, true)
    - If steered < unsteered -> steering helps (moves toward truth)
    """
    from evaluate import js_divergence, normalize

    improvements = []
    for r in validation_results:
        q = r["question"]
        if q not in true_distributions:
            continue
        true_pmf = normalize(true_distributions[q])
        steered_pmf = np.array(r["steered_pmf"])
        baseline_pmf = np.array(r["baseline_pmf"])

        js_steered = js_divergence(steered_pmf, true_pmf)
        js_baseline = js_divergence(baseline_pmf, true_pmf)
        improvement = js_baseline - js_steered  # positive = steering helps

        improvements.append({
            "cluster_id": r["cluster_id"],
            "question": q,
            "js_steered": js_steered,
            "js_baseline": js_baseline,
            "improvement": improvement,
            "helps": improvement > 0,
        })

    n_helps = sum(1 for x in improvements if x["helps"])
    n_total = len(improvements)
    mean_improvement = float(np.mean([x["improvement"] for x in improvements])) if improvements else 0

    return {
        "n_helps": n_helps,
        "n_total": n_total,
        "help_rate": n_helps / max(n_total, 1),
        "mean_improvement": mean_improvement,
        "per_item": improvements,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="/data/chenhongrui/models/Qwen3-8B")
    parser.add_argument("--vectors_path", default="/home/mzyy1001/business/results/persona_vectors_L16_N20.npz")
    parser.add_argument("--topics_path", default="/home/mzyy1001/business/data/3_cluster_topics.json")
    parser.add_argument("--output", default="/home/mzyy1001/business/results/vector_validation.json")
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--layer", type=int, default=16)
    args = parser.parse_args()

    model, tokenizer = load_model(args.model_path, args.device)
    persona_vectors = load_persona_vectors(args.vectors_path)
    with open(args.topics_path) as f:
        cluster_topics = json.load(f)

    validation = validate_persona_vectors(
        model, tokenizer, persona_vectors, cluster_topics,
        alpha=args.alpha, layer=args.layer,
    )
    consistency = compute_consistency_scores(validation, cluster_topics)

    output = {"validation": {}, "consistency": consistency}
    # Serialize (skip raw texts for compactness in summary)
    for vtype in validation:
        output["validation"][vtype] = {}
        for cid, data in validation[vtype].items():
            output["validation"][vtype][str(cid)] = {
                "cluster_topic": data["cluster_topic"],
                "sample_text": data["generated_texts"][0] if data["generated_texts"] else "",
                "sample_judge": data["judge_labels"][0] if data["judge_labels"] else "",
            }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"Validation saved to {args.output}")
    print(f"\nConsistency scores:")
    for vtype, score in consistency.items():
        print(f"  {vtype}: {score['mean_consistency']:.3f} ± {score['std_consistency']:.3f}")
