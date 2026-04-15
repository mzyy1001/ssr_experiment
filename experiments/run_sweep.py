"""
Targeted sweep to find conditions where persona steering outperforms zero-vector.

Key hypotheses:
1. Alpha=2.0 is too aggressive — try much lower alphas (0.1, 0.3, 0.5, 1.0)
2. Steer-then-aggregate with few top clusters preserves per-cluster signal
3. Layer 16 may not be optimal — try early (8), late (24, 28)
4. Only relevant clusters (filtered by LLM relevance) should be steered
"""
import sys, os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch, json, numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from scipy.spatial.distance import jensenshannon

from persona_vectors import load_model, load_persona_vectors, extract_persona_vectors
from steered_ssr import (
    ssr_score, generate_anchors_local, create_control_vectors,
    SteeringHook, generate_steered_response,
)

MODEL_PATH = "/data/chenhongrui/models/Qwen3-8B"
VECTORS_BASE = "/data/chenhongrui/business/results/persona_vectors_L{layer}_N20.npz"
QUESTIONS_PATH = "/data/chenhongrui/business/results/all_questions.json"
DATA_PATH = "/data/chenhongrui/business/data/2_meaningful_df.csv"
TOPICS_PATH = "/data/chenhongrui/business/data/3_cluster_topics.json"
EMBEDDING_MODEL = "/data/chenhongrui/models/bge-base-zh-v1.5"
RESULTS_DIR = "/data/chenhongrui/business/results"
DEVICE = sys.argv[1] if len(sys.argv) > 1 else "cuda:2"

def js(p, q):
    p = np.clip(p, 1e-10, None); p = p / p.sum()
    q = np.clip(q, 1e-10, None); q = q / q.sum()
    return jensenshannon(p, q) ** 2

def norm(d):
    v = np.array(list(d.values()), dtype=float)
    return v / v.sum()

def run_steer_then_aggregate_topk(
    model, tokenizer, encoder,
    persona_vectors, cluster_weights,
    question, options, anchor_embs,
    alpha, layer, top_k=5, n_responses=5,
):
    """Steer-then-aggregate with only top-K clusters."""
    prompt = (
        "作为一位有相关经验的消费者，请回答以下问卷问题。\n"
        "请用自然的语言表达你的真实想法和感受，不需要选择选项。\n\n"
        f"问题：{question}\n选项：{'、'.join(options)}\n\n你的回答："
    )
    sorted_clusters = sorted(cluster_weights.items(), key=lambda x: -x[1])[:top_k]
    aggregated_pmf = np.zeros(len(options))
    total_weight = 0

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
            pmf, _ = ssr_score(response, encoder, anchor_embs)
            cluster_pmfs.append(pmf)
        cluster_mean = np.mean(cluster_pmfs, axis=0)
        aggregated_pmf += weight * cluster_mean
        total_weight += weight

    if total_weight > 0:
        aggregated_pmf /= total_weight
    return aggregated_pmf

def run_relevant_only(
    model, tokenizer, encoder,
    persona_vectors, cluster_weights, cluster_relevance,
    question, options, anchor_embs,
    alpha, layer, n_responses=5,
):
    """Only steer with clusters marked as relevant by LLM."""
    prompt = (
        "作为一位有相关经验的消费者，请回答以下问卷问题。\n"
        "请用自然的语言表达你的真实想法和感受，不需要选择选项。\n\n"
        f"问题：{question}\n选项：{'、'.join(options)}\n\n你的回答："
    )
    aggregated_pmf = np.zeros(len(options))
    total_weight = 0

    for cid_str, weight in sorted(cluster_weights.items(), key=lambda x: -x[1]):
        cid = int(cid_str)
        if cid not in persona_vectors or weight < 1e-6:
            continue
        # Only use relevant clusters
        rel = cluster_relevance.get(cid_str, "不相关")
        if rel != "相关":
            continue
        vec = persona_vectors[cid]["vector"]
        cluster_pmfs = []
        for _ in range(n_responses):
            response = generate_steered_response(
                model, tokenizer, prompt, vec, alpha, layer
            )
            pmf, _ = ssr_score(response, encoder, anchor_embs)
            cluster_pmfs.append(pmf)
        cluster_mean = np.mean(cluster_pmfs, axis=0)
        aggregated_pmf += weight * cluster_mean
        total_weight += weight

    if total_weight > 0:
        aggregated_pmf /= total_weight
    return aggregated_pmf


print(f"Loading models on {DEVICE}...")
model, tokenizer = load_model(MODEL_PATH, DEVICE)
encoder = SentenceTransformer(EMBEDDING_MODEL)

with open(QUESTIONS_PATH) as f:
    baseline = json.load(f)

import pandas as pd
meaningful_df = pd.read_csv(DATA_PATH)
with open(TOPICS_PATH) as f:
    cluster_topics = json.load(f)

# ─── Sweep 1: Alpha sweep at layer 16 with steer-then-aggregate top-5 ───
print("\n" + "=" * 70)
print("SWEEP 1: Alpha sweep (layer=16, steer-then-agg, top-5 clusters)")
print("=" * 70)

vectors_l16 = load_persona_vectors(VECTORS_BASE.format(layer=16))
zero_vectors = create_control_vectors(vectors_l16, "zero")

for alpha in [0.1, 0.3, 0.5, 1.0]:
    print(f"\n--- Alpha = {alpha} ---")
    results_real = []
    results_zero = []

    for sk, sv in baseline.items():
        for sid, questions in sv.items():
            for qid, qd in tqdm(questions.items(), desc=f"a={alpha}"):
                question = qd["question"]
                options = qd["options"]
                weights = qd["cluster_weights"]
                true_pmf = norm(qd["true_distribution"])

                anchors = generate_anchors_local(model, tokenizer, question, options)
                anchor_embs = [encoder.encode(a) for a in anchors]

                # Real vectors
                pred_real = run_steer_then_aggregate_topk(
                    model, tokenizer, encoder,
                    vectors_l16, weights, question, options, anchor_embs,
                    alpha=alpha, layer=16, top_k=5,
                )
                # Zero control
                pred_zero = run_steer_then_aggregate_topk(
                    model, tokenizer, encoder,
                    zero_vectors, weights, question, options, anchor_embs,
                    alpha=alpha, layer=16, top_k=5,
                )

                js_real = js(true_pmf, pred_real)
                js_zero = js(true_pmf, pred_zero)
                results_real.append(js_real)
                results_zero.append(js_zero)
                print(f"  {sid}/{qid}: real={js_real:.4f} zero={js_zero:.4f} {'✓' if js_real < js_zero else '✗'}")

    mean_real = np.mean(results_real)
    mean_zero = np.mean(results_zero)
    wins = sum(1 for r, z in zip(results_real, results_zero) if r < z)
    print(f"  Alpha={alpha}: real={mean_real:.4f} zero={mean_zero:.4f} wins={wins}/6")

# ─── Sweep 2: Layer sweep at alpha=0.5 with steer-then-agg top-5 ───
print("\n" + "=" * 70)
print("SWEEP 2: Layer sweep (alpha=0.5, steer-then-agg, top-5)")
print("=" * 70)

for layer in [8, 12, 20, 24, 28]:
    print(f"\n--- Layer = {layer} ---")
    # Extract vectors at this layer
    vec_path = VECTORS_BASE.format(layer=layer)
    if not os.path.exists(vec_path):
        print(f"  Extracting vectors at layer {layer}...")
        from persona_vectors import extract_persona_vectors, save_persona_vectors
        vecs = extract_persona_vectors(
            model, tokenizer, meaningful_df, cluster_topics,
            layer=layer, n_samples=20,
        )
        save_persona_vectors(vecs, vec_path)
    vectors = load_persona_vectors(vec_path)
    zero_vecs = create_control_vectors(vectors, "zero")

    results_real = []
    results_zero = []

    for sk, sv in baseline.items():
        for sid, questions in sv.items():
            for qid, qd in tqdm(questions.items(), desc=f"L={layer}"):
                question = qd["question"]
                options = qd["options"]
                weights = qd["cluster_weights"]
                true_pmf = norm(qd["true_distribution"])

                anchors = generate_anchors_local(model, tokenizer, question, options)
                anchor_embs = [encoder.encode(a) for a in anchors]

                pred_real = run_steer_then_aggregate_topk(
                    model, tokenizer, encoder,
                    vectors, weights, question, options, anchor_embs,
                    alpha=0.5, layer=layer, top_k=5,
                )
                pred_zero = run_steer_then_aggregate_topk(
                    model, tokenizer, encoder,
                    zero_vecs, weights, question, options, anchor_embs,
                    alpha=0.5, layer=layer, top_k=5,
                )

                js_real = js(true_pmf, pred_real)
                js_zero = js(true_pmf, pred_zero)
                results_real.append(js_real)
                results_zero.append(js_zero)
                print(f"  {sid}/{qid}: real={js_real:.4f} zero={js_zero:.4f} {'✓' if js_real < js_zero else '✗'}")

    mean_real = np.mean(results_real)
    mean_zero = np.mean(results_zero)
    wins = sum(1 for r, z in zip(results_real, results_zero) if r < z)
    print(f"  Layer={layer}: real={mean_real:.4f} zero={mean_zero:.4f} wins={wins}/6")

# ─── Sweep 3: Relevant-only clusters at best alpha/layer ───
print("\n" + "=" * 70)
print("SWEEP 3: Relevant-only clusters (alpha=0.5, layer=16)")
print("=" * 70)

results_rel = []
results_zero = []
vectors = load_persona_vectors(VECTORS_BASE.format(layer=16))
zero_vecs = create_control_vectors(vectors, "zero")

for sk, sv in baseline.items():
    for sid, questions in sv.items():
        for qid, qd in tqdm(questions.items(), desc="relevant-only"):
            question = qd["question"]
            options = qd["options"]
            weights = qd["cluster_weights"]
            relevance = qd.get("cluster_relevance", {})
            true_pmf = norm(qd["true_distribution"])

            anchors = generate_anchors_local(model, tokenizer, question, options)
            anchor_embs = [encoder.encode(a) for a in anchors]

            # Only relevant clusters
            pred_rel = run_relevant_only(
                model, tokenizer, encoder,
                vectors, weights, relevance,
                question, options, anchor_embs,
                alpha=0.5, layer=16,
            )
            # Zero baseline (same relevant filter)
            pred_zero_rel = run_relevant_only(
                model, tokenizer, encoder,
                zero_vecs, weights, relevance,
                question, options, anchor_embs,
                alpha=0.5, layer=16,
            )

            js_rel = js(true_pmf, pred_rel)
            js_zero = js(true_pmf, pred_zero_rel)
            results_rel.append(js_rel)
            results_zero.append(js_zero)
            n_rel = sum(1 for c in relevance.values() if c == "相关")
            print(f"  {sid}/{qid}: relevant={js_rel:.4f} zero={js_zero:.4f} {'✓' if js_rel < js_zero else '✗'} ({n_rel} relevant clusters)")

mean_rel = np.mean(results_rel)
mean_zero = np.mean(results_zero)
wins = sum(1 for r, z in zip(results_rel, results_zero) if r < z)
print(f"  Relevant-only: real={mean_rel:.4f} zero={mean_zero:.4f} wins={wins}/6")

print("\n====== SWEEP COMPLETE ======")
