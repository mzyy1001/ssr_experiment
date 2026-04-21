"""
Systematic fix attempts for the steering divergence problem.

Core problem: α=0.1 CAA steering produces near-identical outputs across clusters
(inter-cluster JS ≈ 0.003). We try multiple alternative approaches.

Fix 1: Multi-layer steering (inject at layers 16,20,24 simultaneously)
Fix 2: Larger α with norm-clipped vectors
Fix 3: In-context learning (ICL) — put actual posts in the prompt instead of steering
Fix 4: Direct post SSR — skip LLM, SSR raw posts against anchors
Fix 5: Hybrid ICL + steering
Fix 6: Contrastive prompt — tell the model what this cluster IS and IS NOT

For each fix, measure:
  (a) Inter-cluster divergence (steered JS between clusters)
  (b) JS vs ground truth (final reconstruction quality)

Uses 6 representative questions, top-10 clusters by weight.
"""
import torch
import numpy as np
import json
import pandas as pd
from pathlib import Path
from scipy.spatial.distance import jensenshannon
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from persona_vectors import load_model, load_persona_vectors, get_hidden_states
from steered_ssr import (
    SteeringHook, ssr_score, generate_anchors_local, generate_steered_response,
)

DATA_DIR = "/data/chenhongrui/business/data"
RESULTS_DIR = "/data/chenhongrui/business/results"
MODEL_PATH = "/data/chenhongrui/models/Qwen3-8B"
EMBEDDING_MODEL = "/data/chenhongrui/models/bge-base-zh-v1.5"


def load_setup():
    with open(f"{RESULTS_DIR}/all_questions_expanded.json") as f:
        expanded = json.load(f)
    with open(f"{DATA_DIR}/3_cluster_topics.json") as f:
        topics = json.load(f)

    questions = []
    for sk, sv in expanded.items():
        for sub, qs in sv.items():
            for qid, qd in qs.items():
                questions.append({
                    "key": f"{sub}/{qid}",
                    "question": qd["question"],
                    "options": qd["options"],
                    "true_distribution": qd["true_distribution"],
                    "cluster_weights": qd["cluster_weights"],
                })
    # Use 6 representative questions
    indices = [0, 2, 4, 8, 12, 16]
    questions = [questions[i] for i in indices]

    df = pd.read_csv(f"{DATA_DIR}/2_meaningful_df.csv")
    return questions, topics, df


def get_top_clusters(questions, n=10):
    """Get top-N clusters by weight."""
    all_weights = {}
    for qd in questions:
        for cid, w in qd["cluster_weights"].items():
            all_weights[cid] = all_weights.get(cid, 0) + w
    return sorted(all_weights, key=lambda c: -all_weights[c])[:n]


def js(true_dist, pred):
    t = np.array(list(true_dist.values()), dtype=float)
    t /= t.sum()
    p = pred / (pred.sum() + 1e-10)
    return float(jensenshannon(t, p) ** 2)


def inter_cluster_js(cluster_pmfs):
    """Mean pairwise JS between clusters."""
    cids = list(cluster_pmfs.keys())
    if len(cids) < 2:
        return 0.0
    vals = []
    for i in range(len(cids)):
        for j in range(i + 1, len(cids)):
            vals.append(jensenshannon(cluster_pmfs[cids[i]], cluster_pmfs[cids[j]]) ** 2)
    return float(np.mean(vals))


def aggregate(cluster_pmfs, weights, cluster_ids):
    n = len(next(iter(cluster_pmfs.values())))
    agg = np.zeros(n)
    tw = 0
    for cid in cluster_ids:
        if cid in cluster_pmfs:
            w = weights.get(cid, 0)
            agg += w * cluster_pmfs[cid]
            tw += w
    return agg / tw if tw > 0 else agg


def evaluate_method(name, all_cluster_pmfs, questions, cluster_ids):
    """Evaluate a method across all questions."""
    divergences = []
    js_vals = []
    for qi, qd in enumerate(questions):
        cpmfs = all_cluster_pmfs[qi]
        div = inter_cluster_js(cpmfs)
        divergences.append(div)
        pred = aggregate(cpmfs, qd["cluster_weights"], cluster_ids)
        js_vals.append(js(qd["true_distribution"], pred))
    return {
        "name": name,
        "mean_divergence": float(np.mean(divergences)),
        "max_divergence": float(np.max(divergences)),
        "mean_js": float(np.mean(js_vals)),
        "per_q_js": [float(j) for j in js_vals],
        "per_q_div": [float(d) for d in divergences],
    }


# ===== Fix implementations =====

def fix0_baseline(model, tokenizer, encoder, questions, cluster_ids, persona_vectors,
                  anchors_cache, alpha=0.1, layer=24):
    """Baseline: single-layer steering at α=0.1."""
    prompt_tpl = ("作为一位有相关经验的消费者，请回答以下问卷问题。\n"
                  "请用自然的语言表达你的真实想法和感受，不需要选择选项。\n\n"
                  "问题：{question}\n选项：{options}\n\n你的回答：")
    all_cpmfs = {}
    for qi, qd in enumerate(questions):
        prompt = prompt_tpl.format(question=qd["question"], options="、".join(qd["options"]))
        _, anchor_embs = anchors_cache[qd["question"]]
        cpmfs = {}
        for cid in cluster_ids:
            if int(cid) not in persona_vectors:
                continue
            vec = persona_vectors[int(cid)]["vector"]
            pmfs = []
            for _ in range(3):
                resp = generate_steered_response(model, tokenizer, prompt, vec, alpha, layer)
                pmf, _ = ssr_score(resp, encoder, anchor_embs)
                pmfs.append(pmf)
            cpmfs[cid] = np.mean(pmfs, axis=0)
        all_cpmfs[qi] = cpmfs
    return all_cpmfs


def fix1_multi_layer(model, tokenizer, encoder, questions, cluster_ids, persona_vectors,
                     anchors_cache, alpha=0.1, layers=[16, 20, 24]):
    """Fix 1: Inject steering vector at multiple layers simultaneously."""
    prompt_tpl = ("作为一位有相关经验的消费者，请回答以下问卷问题。\n"
                  "请用自然的语言表达你的真实想法和感受，不需要选择选项。\n\n"
                  "问题：{question}\n选项：{options}\n\n你的回答：")
    all_cpmfs = {}
    for qi, qd in enumerate(questions):
        prompt = prompt_tpl.format(question=qd["question"], options="、".join(qd["options"]))
        _, anchor_embs = anchors_cache[qd["question"]]
        cpmfs = {}
        for cid in cluster_ids:
            if int(cid) not in persona_vectors:
                continue
            vec = persona_vectors[int(cid)]["vector"]
            pmfs = []
            for _ in range(3):
                # Attach hooks at multiple layers
                hooks = []
                for l in layers:
                    h = SteeringHook(vec, alpha, l)
                    h.attach(model)
                    hooks.append(h)
                try:
                    inputs = tokenizer(prompt, return_tensors="pt")
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}
                    with torch.no_grad():
                        outputs = model.generate(**inputs, max_new_tokens=200,
                                                 do_sample=True, temperature=0.7, top_p=0.9)
                    resp = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:],
                                           skip_special_tokens=True)
                finally:
                    for h in hooks:
                        h.remove()
                pmf, _ = ssr_score(resp, encoder, anchor_embs)
                pmfs.append(pmf)
            cpmfs[cid] = np.mean(pmfs, axis=0)
        all_cpmfs[qi] = cpmfs
    return all_cpmfs


def fix2_large_alpha(model, tokenizer, encoder, questions, cluster_ids, persona_vectors,
                     anchors_cache, alpha=1.0, layer=24):
    """Fix 2: Larger alpha with unit-normed vectors."""
    prompt_tpl = ("作为一位有相关经验的消费者，请回答以下问卷问题。\n"
                  "请用自然的语言表达你的真实想法和感受，不需要选择选项。\n\n"
                  "问题：{question}\n选项：{options}\n\n你的回答：")
    all_cpmfs = {}
    for qi, qd in enumerate(questions):
        prompt = prompt_tpl.format(question=qd["question"], options="、".join(qd["options"]))
        _, anchor_embs = anchors_cache[qd["question"]]
        cpmfs = {}
        for cid in cluster_ids:
            if int(cid) not in persona_vectors:
                continue
            vec = persona_vectors[int(cid)]["vector"]
            # Normalize to unit norm, then scale by alpha
            vec_normed = vec / (np.linalg.norm(vec) + 1e-8)
            pmfs = []
            for _ in range(3):
                resp = generate_steered_response(model, tokenizer, prompt, vec_normed, alpha, layer)
                pmf, _ = ssr_score(resp, encoder, anchor_embs)
                pmfs.append(pmf)
            cpmfs[cid] = np.mean(pmfs, axis=0)
        all_cpmfs[qi] = cpmfs
    return all_cpmfs


def fix3_icl(model, tokenizer, encoder, questions, cluster_ids, df,
             anchors_cache, topics, n_examples=5):
    """Fix 3: In-context learning — include actual posts in the prompt."""
    all_cpmfs = {}
    for qi, qd in enumerate(questions):
        _, anchor_embs = anchors_cache[qd["question"]]
        cpmfs = {}
        for cid in cluster_ids:
            cid_int = int(cid)
            posts = df[df["cluster_label"] == cid_int]["content_desc"].dropna()
            if len(posts) < 3:
                continue
            sampled = posts.sample(min(n_examples, len(posts)), random_state=42).tolist()
            post_text = "\n".join(f"- {p[:100]}" for p in sampled)
            topic = topics.get(cid, "未知话题")

            prompt = (
                f"以下是一些真实消费者的社交媒体帖子，他们主要讨论的话题是：{topic[:60]}\n\n"
                f"帖子摘要：\n{post_text}\n\n"
                f"请你以这些消费者的视角，回答以下问卷问题。\n"
                f"请用自然的语言表达这类消费者可能的真实想法和感受。\n\n"
                f"问题：{qd['question']}\n选项：{'、'.join(qd['options'])}\n\n你的回答："
            )
            pmfs = []
            for _ in range(3):
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=200,
                                             do_sample=True, temperature=0.7, top_p=0.9)
                resp = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:],
                                       skip_special_tokens=True)
                pmf, _ = ssr_score(resp, encoder, anchor_embs)
                pmfs.append(pmf)
            cpmfs[cid] = np.mean(pmfs, axis=0)
        all_cpmfs[qi] = cpmfs
    return all_cpmfs


def fix4_direct_ssr(encoder, questions, cluster_ids, df, anchors_cache, n_posts=30):
    """Fix 4: Direct post SSR — skip LLM entirely."""
    all_cpmfs = {}
    for qi, qd in enumerate(questions):
        _, anchor_embs = anchors_cache[qd["question"]]
        cpmfs = {}
        for cid in cluster_ids:
            cid_int = int(cid)
            posts = df[df["cluster_label"] == cid_int]["content_desc"].dropna()
            if len(posts) < 3:
                continue
            sampled = posts.sample(min(n_posts, len(posts)), random_state=42).tolist()
            pmfs = []
            for post in sampled:
                pmf, _ = ssr_score(post, encoder, anchor_embs)
                pmfs.append(pmf)
            cpmfs[cid] = np.mean(pmfs, axis=0)
        all_cpmfs[qi] = cpmfs
    return all_cpmfs


def fix5_icl_plus_steering(model, tokenizer, encoder, questions, cluster_ids, df,
                           persona_vectors, anchors_cache, topics,
                           alpha=0.3, layer=24, n_examples=3):
    """Fix 5: Hybrid — ICL context + activation steering."""
    all_cpmfs = {}
    for qi, qd in enumerate(questions):
        _, anchor_embs = anchors_cache[qd["question"]]
        cpmfs = {}
        for cid in cluster_ids:
            cid_int = int(cid)
            if cid_int not in persona_vectors:
                continue
            posts = df[df["cluster_label"] == cid_int]["content_desc"].dropna()
            if len(posts) < 3:
                continue
            sampled = posts.sample(min(n_examples, len(posts)), random_state=42).tolist()
            post_text = "\n".join(f"- {p[:80]}" for p in sampled)
            topic = topics.get(cid, "未知话题")

            prompt = (
                f"参考以下消费者评论（话题：{topic[:40]}）：\n{post_text}\n\n"
                f"以这类消费者的视角回答：\n"
                f"问题：{qd['question']}\n选项：{'、'.join(qd['options'])}\n\n你的回答："
            )
            vec = persona_vectors[cid_int]["vector"]
            pmfs = []
            for _ in range(3):
                resp = generate_steered_response(model, tokenizer, prompt, vec, alpha, layer)
                pmf, _ = ssr_score(resp, encoder, anchor_embs)
                pmfs.append(pmf)
            cpmfs[cid] = np.mean(pmfs, axis=0)
        all_cpmfs[qi] = cpmfs
    return all_cpmfs


def fix6_contrastive_prompt(model, tokenizer, encoder, questions, cluster_ids, df,
                            anchors_cache, topics):
    """Fix 6: Contrastive prompt — tell model what this persona IS and IS NOT."""
    # For each cluster, find the most distant cluster
    cluster_embs = {}
    for cid in cluster_ids:
        cid_int = int(cid)
        posts = df[df["cluster_label"] == cid_int]["content_desc"].dropna()
        if len(posts) < 3:
            continue
        sampled = posts.sample(min(5, len(posts)), random_state=42).tolist()
        embs = encoder.encode(sampled)
        cluster_embs[cid] = embs.mean(axis=0)

    # Pairwise distances
    distances = {}
    cids = list(cluster_embs.keys())
    for i, ci in enumerate(cids):
        for j, cj in enumerate(cids):
            if i != j:
                dist = np.linalg.norm(cluster_embs[ci] - cluster_embs[cj])
                distances[(ci, cj)] = dist

    all_cpmfs = {}
    for qi, qd in enumerate(questions):
        _, anchor_embs = anchors_cache[qd["question"]]
        cpmfs = {}
        for cid in cluster_ids:
            if cid not in cluster_embs:
                continue
            # Find most distant cluster
            farthest = max([c for c in cids if c != cid],
                          key=lambda c: distances.get((cid, c), 0))
            topic_pos = topics.get(cid, "未知")[:50]
            topic_neg = topics.get(farthest, "未知")[:50]

            posts_pos = df[df["cluster_label"] == int(cid)]["content_desc"].dropna()
            sampled_pos = posts_pos.sample(min(3, len(posts_pos)), random_state=42).tolist()
            post_text = "\n".join(f"- {p[:80]}" for p in sampled_pos)

            prompt = (
                f"你是以下类型的消费者：\n"
                f"【你的特征】{topic_pos}\n"
                f"【你的真实评论】\n{post_text}\n\n"
                f"【你绝对不是】{topic_neg}\n\n"
                f"请以你的视角回答：\n"
                f"问题：{qd['question']}\n选项：{'、'.join(qd['options'])}\n\n你的回答："
            )
            pmfs = []
            for _ in range(3):
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=200,
                                             do_sample=True, temperature=0.7, top_p=0.9)
                resp = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:],
                                       skip_special_tokens=True)
                pmf, _ = ssr_score(resp, encoder, anchor_embs)
                pmfs.append(pmf)
            cpmfs[cid] = np.mean(pmfs, axis=0)
        all_cpmfs[qi] = cpmfs
    return all_cpmfs


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:2")
    args = parser.parse_args()

    Path(RESULTS_DIR).mkdir(exist_ok=True)
    questions, topics, df = load_setup()
    cluster_ids = get_top_clusters(questions, n=10)
    print(f"Testing {len(questions)} questions, {len(cluster_ids)} clusters", flush=True)

    print("Loading models...", flush=True)
    encoder = SentenceTransformer(EMBEDDING_MODEL)
    model, tokenizer = load_model(MODEL_PATH, args.device)
    persona_vectors = load_persona_vectors(f"{RESULTS_DIR}/persona_vectors_L24_N20.npz")

    # Pre-generate anchors
    print("Generating anchors...", flush=True)
    anchors_cache = {}
    for qd in questions:
        q = qd["question"]
        if q not in anchors_cache:
            anchors = generate_anchors_local(model, tokenizer, q, qd["options"])
            anchor_embs = [encoder.encode(a) for a in anchors]
            anchors_cache[q] = (anchors, anchor_embs)
            print(f"  {q[:40]}...", flush=True)

    results = []

    # Fix 0: Baseline
    print("\n" + "=" * 60, flush=True)
    print("Fix 0: Baseline (α=0.1, L24)", flush=True)
    print("=" * 60, flush=True)
    cpmfs = fix0_baseline(model, tokenizer, encoder, questions, cluster_ids,
                          persona_vectors, anchors_cache, alpha=0.1, layer=24)
    r = evaluate_method("Baseline α=0.1 L24", cpmfs, questions, cluster_ids)
    results.append(r)
    print(f"  Divergence: {r['mean_divergence']:.4f}  JS: {r['mean_js']:.4f}", flush=True)

    # Fix 1: Multi-layer
    print("\n" + "=" * 60, flush=True)
    print("Fix 1: Multi-layer steering (L16+L20+L24, α=0.1)", flush=True)
    print("=" * 60, flush=True)
    cpmfs = fix1_multi_layer(model, tokenizer, encoder, questions, cluster_ids,
                             persona_vectors, anchors_cache, alpha=0.1, layers=[16, 20, 24])
    r = evaluate_method("Multi-layer α=0.1", cpmfs, questions, cluster_ids)
    results.append(r)
    print(f"  Divergence: {r['mean_divergence']:.4f}  JS: {r['mean_js']:.4f}", flush=True)

    # Fix 2a: Large alpha, norm-clipped
    print("\n" + "=" * 60, flush=True)
    print("Fix 2a: Large α=1.0, unit-normed vectors", flush=True)
    print("=" * 60, flush=True)
    cpmfs = fix2_large_alpha(model, tokenizer, encoder, questions, cluster_ids,
                             persona_vectors, anchors_cache, alpha=1.0, layer=24)
    r = evaluate_method("Large α=1.0 normed", cpmfs, questions, cluster_ids)
    results.append(r)
    print(f"  Divergence: {r['mean_divergence']:.4f}  JS: {r['mean_js']:.4f}", flush=True)

    # Fix 2b: Even larger
    print("\n" + "=" * 60, flush=True)
    print("Fix 2b: Large α=3.0, unit-normed vectors", flush=True)
    print("=" * 60, flush=True)
    cpmfs = fix2_large_alpha(model, tokenizer, encoder, questions, cluster_ids,
                             persona_vectors, anchors_cache, alpha=3.0, layer=24)
    r = evaluate_method("Large α=3.0 normed", cpmfs, questions, cluster_ids)
    results.append(r)
    print(f"  Divergence: {r['mean_divergence']:.4f}  JS: {r['mean_js']:.4f}", flush=True)

    # Fix 3: ICL
    print("\n" + "=" * 60, flush=True)
    print("Fix 3: In-context learning (5 example posts)", flush=True)
    print("=" * 60, flush=True)
    cpmfs = fix3_icl(model, tokenizer, encoder, questions, cluster_ids, df,
                     anchors_cache, topics, n_examples=5)
    r = evaluate_method("ICL (5 posts)", cpmfs, questions, cluster_ids)
    results.append(r)
    print(f"  Divergence: {r['mean_divergence']:.4f}  JS: {r['mean_js']:.4f}", flush=True)

    # Fix 4: Direct SSR
    print("\n" + "=" * 60, flush=True)
    print("Fix 4: Direct post SSR (no LLM)", flush=True)
    print("=" * 60, flush=True)
    cpmfs = fix4_direct_ssr(encoder, questions, cluster_ids, df, anchors_cache, n_posts=30)
    r = evaluate_method("Direct SSR", cpmfs, questions, cluster_ids)
    results.append(r)
    print(f"  Divergence: {r['mean_divergence']:.4f}  JS: {r['mean_js']:.4f}", flush=True)

    # Fix 5: ICL + steering
    print("\n" + "=" * 60, flush=True)
    print("Fix 5: ICL + steering (α=0.3)", flush=True)
    print("=" * 60, flush=True)
    cpmfs = fix5_icl_plus_steering(model, tokenizer, encoder, questions, cluster_ids, df,
                                   persona_vectors, anchors_cache, topics,
                                   alpha=0.3, layer=24, n_examples=3)
    r = evaluate_method("ICL + steer α=0.3", cpmfs, questions, cluster_ids)
    results.append(r)
    print(f"  Divergence: {r['mean_divergence']:.4f}  JS: {r['mean_js']:.4f}", flush=True)

    # Fix 6: Contrastive prompt
    print("\n" + "=" * 60, flush=True)
    print("Fix 6: Contrastive prompt (IS vs IS NOT)", flush=True)
    print("=" * 60, flush=True)
    cpmfs = fix6_contrastive_prompt(model, tokenizer, encoder, questions, cluster_ids, df,
                                    anchors_cache, topics)
    r = evaluate_method("Contrastive prompt", cpmfs, questions, cluster_ids)
    results.append(r)
    print(f"  Divergence: {r['mean_divergence']:.4f}  JS: {r['mean_js']:.4f}", flush=True)

    # Summary
    print("\n" + "=" * 70, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 70, flush=True)
    print(f"  {'Method':<28} {'Divergence':>11} {'JS ↓':>8} {'Per-Q JS':>50}", flush=True)
    print("  " + "-" * 100, flush=True)
    for r in sorted(results, key=lambda x: x["mean_js"]):
        per_q = " ".join(f"{j:.4f}" for j in r["per_q_js"])
        print(f"  {r['name']:<28} {r['mean_divergence']:>11.4f} {r['mean_js']:>8.4f} {per_q}", flush=True)

    # Save
    out = f"{RESULTS_DIR}/fix_steering_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {out}", flush=True)


if __name__ == "__main__":
    main()
