"""
Phase 3: Persona-Steered SSR (PS-SSR).

Injects persona vectors into LLM hidden states via activation addition hooks,
then uses SSR to measure the steered model's output distribution.

Two strategies:
  - aggregate_then_steer: combine persona vectors first, then one steered pass
  - steer_then_aggregate: steer per-cluster, measure SSR, then combine
"""
import torch
import numpy as np
import json
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from persona_vectors import load_model, load_persona_vectors


# ─── SSR core (from Project 1) ───

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


def ssr_score(text: str, encoder, anchor_embeddings, normalization: str = "softmax"):
    """SSR: map text to distribution over anchors via cosine similarity.

    Args:
        normalization: "min_sub" (original), "softmax", "clipped", "rank"
    """
    v = encoder.encode(text)
    sims = np.array([cosine_sim(v, a) for a in anchor_embeddings], dtype=float)

    if normalization == "min_sub":
        sims = sims - sims.min()
        if sims.sum() == 0:
            sims += 1e-8
        pmf = sims / sims.sum()
    elif normalization == "softmax":
        # Temperature-scaled softmax
        temp = 0.1
        exp_sims = np.exp((sims - sims.max()) / temp)
        pmf = exp_sims / exp_sims.sum()
    elif normalization == "clipped":
        sims = np.clip(sims, 0, None)
        if sims.sum() == 0:
            sims += 1e-8
        pmf = sims / sims.sum()
    elif normalization == "rank":
        ranks = len(sims) - np.argsort(np.argsort(sims))  # higher sim → higher rank
        pmf = ranks.astype(float) / ranks.sum()
    else:
        raise ValueError(f"Unknown normalization: {normalization}")

    mean_score = np.dot(np.arange(1, len(anchor_embeddings) + 1), pmf)
    return pmf, mean_score


# ─── Steering hook ───

class SteeringHook:
    """Activation addition hook for persona vector injection.

    Supports two modes:
      - 'all': add steering vector to all token positions (default)
      - 'generated': only add to tokens beyond the prompt length
    """

    def __init__(self, steering_vector: np.ndarray, alpha: float = 2.0, layer: int = 16,
                 mode: str = "all", prompt_length: int = 0):
        self.steering_vector = torch.tensor(steering_vector, dtype=torch.bfloat16)
        self.alpha = alpha
        self.layer = layer
        self.mode = mode  # "all" or "generated"
        self.prompt_length = prompt_length
        self._handle = None

    def hook_fn(self, module, input, output):
        if isinstance(output, tuple):
            hs = output[0]
        else:
            hs = output

        device = hs.device
        sv = self.steering_vector.to(device)

        if self.mode == "generated" and hs.shape[1] > self.prompt_length:
            # Only steer generated token positions
            steering = torch.zeros_like(hs)
            steering[:, self.prompt_length:, :] = self.alpha * sv
            hs = hs + steering
        else:
            # Steer all token positions
            hs = hs + self.alpha * sv.unsqueeze(0).unsqueeze(0)

        if isinstance(output, tuple):
            return (hs,) + output[1:]
        return hs

    def attach(self, model):
        target_layer = model.model.layers[self.layer]
        self._handle = target_layer.register_forward_hook(self.hook_fn)
        return self

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


def create_control_vectors(persona_vectors: dict, control_type: str = "random") -> dict:
    """Create control vectors for ablation studies.

    Args:
        control_type: "random" (random direction, same norm),
                      "shuffled" (shuffled cluster assignments),
                      "zero" (zero vector)
    """
    control = {}
    example_vec = next(iter(persona_vectors.values()))["vector"]
    dim = example_vec.shape[0]

    if control_type == "zero":
        for cid, data in persona_vectors.items():
            control[cid] = {**data, "vector": np.zeros(dim)}

    elif control_type == "random":
        rng = np.random.RandomState(42)
        for cid, data in persona_vectors.items():
            rand_vec = rng.randn(dim).astype(np.float32)
            rand_vec = rand_vec / np.linalg.norm(rand_vec) * np.linalg.norm(data["vector"])
            control[cid] = {**data, "vector": rand_vec}

    elif control_type == "shuffled":
        cids = list(persona_vectors.keys())
        rng = np.random.RandomState(42)
        shuffled_cids = rng.permutation(cids).tolist()
        for orig_cid, shuffled_cid in zip(cids, shuffled_cids):
            control[orig_cid] = {
                **persona_vectors[orig_cid],
                "vector": persona_vectors[shuffled_cid]["vector"],
            }

    return control


def generate_steered_response(
    model, tokenizer, prompt: str,
    steering_vector: np.ndarray, alpha: float, layer: int,
    max_new_tokens: int = 200,
    steer_mode: str = "all",
) -> str:
    """Generate text from model with persona steering applied.

    Args:
        steer_mode: "all" (steer all tokens) or "generated" (steer only new tokens)
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    prompt_length = inputs["input_ids"].shape[1]
    hook = SteeringHook(steering_vector, alpha, layer, mode=steer_mode, prompt_length=prompt_length)
    hook.attach(model)
    try:
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    finally:
        hook.remove()
    return response


# ─── Anchor generation ───

ANCHOR_PROMPT = """你是一名用户调研分析专家。请为以下问卷题目的每个选项生成一个anchor sentence。
anchor sentence应该是一句自然、具体的中文表达，能够清晰地代表该选项的语义。

问题：{question}
选项：{options}

请严格按以下JSON格式输出：
{{"anchors": ["anchor1", "anchor2", ...]}}
注意：anchor的顺序必须与选项顺序一致。"""


def generate_anchors_local(model, tokenizer, question: str, options: list[str]) -> list[str]:
    """Generate anchor sentences using the local Qwen model.

    Robust JSON parsing with multiple fallback strategies.
    """
    prompt = ANCHOR_PROMPT.format(question=question, options=", ".join(options))
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=500,
            do_sample=False,
        )
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    # Strip thinking tags if present (Qwen3 may wrap in <think>...</think>)
    import re
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()

    # Try multiple JSON extraction strategies
    # Strategy 1: find {"anchors": [...]} pattern
    match = re.search(r'\{"anchors"\s*:\s*\[.*?\]\s*\}', response, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            if "anchors" in result and len(result["anchors"]) == len(options):
                return result["anchors"]
        except json.JSONDecodeError:
            pass

    # Strategy 2: find any JSON array
    match = re.search(r'\[.*?\]', response, re.DOTALL)
    if match:
        try:
            arr = json.loads(match.group())
            if isinstance(arr, list) and len(arr) == len(options):
                return arr
        except json.JSONDecodeError:
            pass

    # Strategy 3: fallback — use option text with simple expansion as anchors
    print(f"  Warning: Failed to parse anchors for '{question[:30]}...', using option text as fallback")
    return [f"关于{question}，我认为答案是{opt}" for opt in options]


# ─── Main PS-SSR pipeline ───

def ps_ssr_steer_then_aggregate(
    model, tokenizer, encoder,
    persona_vectors: dict,
    cluster_weights: dict,
    question: str, options: list[str],
    anchor_embeddings: list,
    alpha: float = 2.0,
    layer: int = 16,
    n_responses: int = 5,
) -> np.ndarray:
    """Strategy 2: Steer per-cluster, SSR each response, then aggregate.

    For each relevant cluster:
      1. Steer model with cluster's persona vector
      2. Generate response to the survey question
      3. SSR the response against anchors
      4. Weight and combine
    """
    prompt_template = (
        "作为一位有相关经验的消费者，请回答以下问卷问题。\n"
        "请用自然的语言表达你的真实想法和感受，不需要选择选项。\n\n"
        "问题：{question}\n选项：{options}\n\n你的回答："
    )
    prompt = prompt_template.format(question=question, options="、".join(options))

    aggregated_pmf = np.zeros(len(options))
    total_weight = 0

    # Only use top-20 clusters by weight to keep runtime manageable
    sorted_clusters = sorted(cluster_weights.items(), key=lambda x: -x[1])[:20]
    for cid_str, weight in sorted_clusters:
        cid = int(cid_str)
        if cid not in persona_vectors or weight < 1e-6:
            continue

        vec = persona_vectors[cid]["vector"]
        cluster_pmfs = []

        for _ in range(n_responses):
            response = generate_steered_response(
                model, tokenizer, prompt, vec, alpha, layer
            )
            pmf, _ = ssr_score(response, encoder, anchor_embeddings)
            cluster_pmfs.append(pmf)

        cluster_mean_pmf = np.mean(cluster_pmfs, axis=0)
        aggregated_pmf += weight * cluster_mean_pmf
        total_weight += weight

    if total_weight > 0:
        aggregated_pmf /= total_weight

    return aggregated_pmf


def ps_ssr_aggregate_then_steer(
    model, tokenizer, encoder,
    persona_vectors: dict,
    cluster_weights: dict,
    question: str, options: list[str],
    anchor_embeddings: list,
    alpha: float = 2.0,
    layer: int = 16,
    n_responses: int = 10,
) -> np.ndarray:
    """Strategy 1: Combine persona vectors first, then one steered pass.

    1. Weighted sum of persona vectors
    2. Steer model once
    3. Generate multiple responses
    4. SSR each and average
    """
    prompt_template = (
        "作为一位有相关经验的消费者，请回答以下问卷问题。\n"
        "请用自然的语言表达你的真实想法和感受，不需要选择选项。\n\n"
        "问题：{question}\n选项：{options}\n\n你的回答："
    )
    prompt = prompt_template.format(question=question, options="、".join(options))

    # Combine persona vectors
    combined_vec = np.zeros_like(next(iter(persona_vectors.values()))["vector"])
    total_weight = 0
    for cid_str, weight in cluster_weights.items():
        cid = int(cid_str)
        if cid not in persona_vectors or weight < 1e-6:
            continue
        combined_vec += weight * persona_vectors[cid]["vector"]
        total_weight += weight

    if total_weight > 0:
        combined_vec /= total_weight

    # Generate responses with combined steering
    pmfs = []
    for _ in range(n_responses):
        response = generate_steered_response(
            model, tokenizer, prompt, combined_vec, alpha, layer
        )
        pmf, _ = ssr_score(response, encoder, anchor_embeddings)
        pmfs.append(pmf)

    return np.mean(pmfs, axis=0)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="/data/chenhongrui/models/Qwen3-8B")
    parser.add_argument("--vectors_path", default="/data/chenhongrui/business/results/persona_vectors.npz")
    parser.add_argument("--baseline_results", default="/data/chenhongrui/business/results/all_questions.json")
    parser.add_argument("--output", default=None)
    parser.add_argument("--strategy", choices=["aggregate_then_steer", "steer_then_aggregate"], default="steer_then_aggregate")
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--embedding_model", default="/data/chenhongrui/models/bge-base-zh-v1.5")
    args = parser.parse_args()

    # Load models
    model, tokenizer = load_model(args.model_path, args.device)
    encoder = SentenceTransformer(args.embedding_model)

    # Load persona vectors
    persona_vectors = load_persona_vectors(args.vectors_path)

    # Load baseline results (contains questions, options, true distributions, cluster weights)
    with open(args.baseline_results) as f:
        baseline = json.load(f)

    results = {}
    strategy_fn = (
        ps_ssr_steer_then_aggregate if args.strategy == "steer_then_aggregate"
        else ps_ssr_aggregate_then_steer
    )

    for survey_key, survey_data in baseline.items():
        results[survey_key] = {}
        for sub_id, questions in survey_data.items():
            results[survey_key][sub_id] = {}
            for q_id, q_data in tqdm(questions.items(), desc=f"Survey {survey_key}/{sub_id}"):
                question = q_data["question"]
                options = q_data["options"]
                cluster_weights = q_data["cluster_weights"]

                # Generate anchors
                anchors = generate_anchors_local(model, tokenizer, question, options)
                anchor_embs = [encoder.encode(a) for a in anchors]

                # Run PS-SSR
                pred_pmf = strategy_fn(
                    model, tokenizer, encoder,
                    persona_vectors, cluster_weights,
                    question, options, anchor_embs,
                    alpha=args.alpha, layer=args.layer,
                )

                results[survey_key][sub_id][q_id] = {
                    "question": question,
                    "options": options,
                    "true_distribution": q_data["true_distribution"],
                    "predicted_pmf": pred_pmf.tolist(),
                    "anchors": anchors,
                    "strategy": args.strategy,
                    "alpha": args.alpha,
                    "layer": args.layer,
                }

    output = args.output or f"/data/chenhongrui/business/results/ps_ssr_{args.strategy}_L{args.layer}_A{args.alpha}.json"
    with open(output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {output}")
