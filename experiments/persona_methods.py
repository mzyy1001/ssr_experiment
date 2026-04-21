"""
Alternative persona methods to beat Direct SSR baseline (JS=0.0276).

Core insight: Direct SSR works because it measures post→anchor similarity
without information loss. Steering fails because generation is lossy.

New approaches use the LLM for REASONING about posts, not GENERATING text:

Method A: Per-post LLM voting
  For each post, ask LLM "which option would this author choose?" → aggregate votes.
  Uses LLM reasoning to understand implicit attitudes that embeddings miss.

Method B: Cluster-conditioned distribution estimation
  Show LLM sample posts + topic → ask to directly estimate answer distribution.
  LLM synthesizes multiple posts into a coherent distribution estimate.

Method C: SSR + LLM correction
  Start with Direct SSR distribution, then ask LLM to adjust based on
  qualitative cluster analysis. Hybrid: SSR provides base, LLM refines.

Method D: Persona-projected SSR
  Project each post's embedding through the persona vector direction,
  then SSR. Uses persona vector to reweight embedding dimensions rather
  than steer generation.

Method E: Multi-layer steering + Direct SSR ensemble
  Average Direct SSR and multi-layer steering predictions. Combines
  the stability of SSR with the differentiation of steering.

Run on chen server:
  python -u persona_methods.py --device cuda:2
"""
import torch
import numpy as np
import json
import re
import pandas as pd
from pathlib import Path
from scipy.spatial.distance import jensenshannon
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from persona_vectors import load_model, load_persona_vectors
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
    df = pd.read_csv(f"{DATA_DIR}/2_meaningful_df.csv")
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
    # 6 representative questions
    indices = [0, 2, 4, 8, 12, 16]
    return [questions[i] for i in indices], topics, df


def get_top_clusters(questions, n=10):
    all_w = {}
    for qd in questions:
        for cid, w in qd["cluster_weights"].items():
            all_w[cid] = all_w.get(cid, 0) + w
    return sorted(all_w, key=lambda c: -all_w[c])[:n]


def js(td, pred):
    t = np.array(list(td.values()), dtype=float); t /= t.sum()
    p = pred / (pred.sum() + 1e-10)
    return float(jensenshannon(t, p) ** 2)


def inter_js(cpmfs):
    cids = list(cpmfs.keys())
    if len(cids) < 2: return 0.0
    vals = [jensenshannon(cpmfs[cids[i]], cpmfs[cids[j]])**2
            for i in range(len(cids)) for j in range(i+1, len(cids))]
    return float(np.mean(vals))


def agg(cpmfs, weights, cids):
    n = len(next(iter(cpmfs.values())))
    a, tw = np.zeros(n), 0
    for c in cids:
        if c in cpmfs:
            w = weights.get(c, 0); a += w * cpmfs[c]; tw += w
    return a / tw if tw > 0 else a


def evaluate(name, all_cpmfs, questions, cids):
    divs, jss = [], []
    for qi, qd in enumerate(questions):
        divs.append(inter_js(all_cpmfs[qi]))
        jss.append(js(qd["true_distribution"], agg(all_cpmfs[qi], qd["cluster_weights"], cids)))
    return {"name": name, "div": float(np.mean(divs)), "js": float(np.mean(jss)),
            "per_q": [float(j) for j in jss]}


# ===== Baseline: Direct SSR =====
def method_direct_ssr(encoder, questions, cids, df, anchors_cache, n_posts=30):
    all_cpmfs = {}
    for qi, qd in enumerate(questions):
        _, aembs = anchors_cache[qd["question"]]
        cpmfs = {}
        for cid in cids:
            posts = df[df["cluster_label"]==int(cid)]["content_desc"].dropna()
            if len(posts) < 3: continue
            samp = posts.sample(min(n_posts, len(posts)), random_state=42).tolist()
            pmfs = [ssr_score(p, encoder, aembs)[0] for p in samp]
            cpmfs[cid] = np.mean(pmfs, axis=0)
        all_cpmfs[qi] = cpmfs
    return all_cpmfs


# ===== Method A: Per-post LLM voting =====
VOTE_PROMPT = """根据以下社交媒体帖子内容，推测这位消费者最可能选择哪个选项。

帖子：{post}

问题：{question}
选项：
{options_numbered}

请只输出选项编号（1-{n}），不需要解释。
选项编号："""

def method_a_voting(model, tokenizer, encoder, questions, cids, df, anchors_cache,
                    n_posts=20):
    """Per-post LLM voting: ask LLM to classify each post."""
    all_cpmfs = {}
    for qi, qd in enumerate(questions):
        opts = qd["options"]
        opts_str = "\n".join(f"{i+1}. {o}" for i, o in enumerate(opts))
        _, aembs = anchors_cache[qd["question"]]
        cpmfs = {}
        for cid in cids:
            posts = df[df["cluster_label"]==int(cid)]["content_desc"].dropna()
            if len(posts) < 3: continue
            samp = posts.sample(min(n_posts, len(posts)), random_state=42).tolist()
            votes = np.zeros(len(opts))
            for post in samp:
                prompt = VOTE_PROMPT.format(
                    post=post[:200], question=qd["question"],
                    options_numbered=opts_str, n=len(opts))
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=20, do_sample=False)
                resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                       skip_special_tokens=True)
                resp = re.sub(r'<think>.*?</think>', '', resp, flags=re.DOTALL).strip()
                # Parse vote
                match = re.search(r'(\d+)', resp)
                if match:
                    idx = int(match.group(1)) - 1
                    if 0 <= idx < len(opts):
                        votes[idx] += 1
                    else:
                        votes += 1.0 / len(opts)  # uniform fallback
                else:
                    votes += 1.0 / len(opts)
            pmf = votes / votes.sum()
            cpmfs[cid] = pmf
        all_cpmfs[qi] = cpmfs
    return all_cpmfs


# ===== Method B: Cluster-conditioned distribution estimation =====
DIST_PROMPT = """你是一位消费者调研分析专家。

以下是某类消费者群体的社交媒体帖子样本（话题：{topic}）：
{posts}

基于这些帖子反映的消费者特征和态度，请估计这个群体在以下问卷问题中各选项的选择比例。

问题：{question}
选项：
{options_numbered}

请严格按JSON格式输出各选项的比例（总和为1）：
{{"distribution": [选项1比例, 选项2比例, ...]}}"""

def method_b_dist_estimation(model, tokenizer, encoder, questions, cids, df,
                             anchors_cache, topics, n_examples=8):
    """Ask LLM to directly estimate the answer distribution for a cluster."""
    all_cpmfs = {}
    for qi, qd in enumerate(questions):
        opts = qd["options"]
        opts_str = "\n".join(f"{i+1}. {o}" for i, o in enumerate(opts))
        cpmfs = {}
        for cid in cids:
            posts = df[df["cluster_label"]==int(cid)]["content_desc"].dropna()
            if len(posts) < 3: continue
            samp = posts.sample(min(n_examples, len(posts)), random_state=42).tolist()
            post_text = "\n".join(f"- {p[:100]}" for p in samp)
            topic = topics.get(cid, "未知")[:60]

            prompt = DIST_PROMPT.format(
                topic=topic, posts=post_text, question=qd["question"],
                options_numbered=opts_str)
            pmfs_collected = []
            for _ in range(3):  # 3 samples for stability
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=200,
                                        do_sample=True, temperature=0.7, top_p=0.9)
                resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                       skip_special_tokens=True)
                resp = re.sub(r'<think>.*?</think>', '', resp, flags=re.DOTALL).strip()
                # Parse JSON
                match = re.search(r'\{[^}]*"distribution"\s*:\s*\[([^\]]+)\]', resp)
                if match:
                    try:
                        vals = [float(x.strip()) for x in match.group(1).split(",")]
                        if len(vals) == len(opts):
                            arr = np.array(vals)
                            arr = np.clip(arr, 0, None)
                            if arr.sum() > 0:
                                pmfs_collected.append(arr / arr.sum())
                                continue
                    except: pass
                # Fallback: try to find any list of numbers
                nums = re.findall(r'\d+\.?\d*', resp)
                try:
                    nums = [float(x) for x in nums if float(x) <= 1]
                except (ValueError, OverflowError):
                    nums = []
                if len(nums) >= len(opts):
                    arr = np.array(nums[:len(opts)])
                    if arr.sum() > 0:
                        pmfs_collected.append(arr / arr.sum())
                        continue
                # Ultimate fallback
                pmfs_collected.append(np.ones(len(opts)) / len(opts))

            cpmfs[cid] = np.mean(pmfs_collected, axis=0)
        all_cpmfs[qi] = cpmfs
    return all_cpmfs


# ===== Method C: SSR + LLM correction =====
CORRECT_PROMPT = """你是一位消费者调研分析专家。

一个基于语义相似度的模型预测了某消费者群体（话题：{topic}）对以下问题的回答分布：

问题：{question}
选项：{options_numbered}
模型预测：{prediction}

该群体的代表性帖子：
{posts}

请根据帖子内容判断模型预测是否合理。如果某些选项的比例明显偏高或偏低，请调整。
请严格按JSON格式输出调整后的分布：
{{"adjusted": [选项1比例, 选项2比例, ...]}}"""

def method_c_ssr_correction(model, tokenizer, encoder, questions, cids, df,
                            anchors_cache, topics, n_posts_ssr=30, n_posts_show=5):
    """Start with Direct SSR, then ask LLM to correct based on post content."""
    # First get Direct SSR predictions
    ssr_cpmfs = method_direct_ssr(encoder, questions, cids, df, anchors_cache, n_posts_ssr)

    all_cpmfs = {}
    for qi, qd in enumerate(questions):
        opts = qd["options"]
        opts_str = "\n".join(f"{i+1}. {o}" for i, o in enumerate(opts))
        cpmfs = {}
        for cid in cids:
            if cid not in ssr_cpmfs[qi]: continue
            ssr_pmf = ssr_cpmfs[qi][cid]
            pred_str = ", ".join(f"{o}: {p:.1%}" for o, p in zip(opts, ssr_pmf))

            posts = df[df["cluster_label"]==int(cid)]["content_desc"].dropna()
            if len(posts) < 3: continue
            samp = posts.sample(min(n_posts_show, len(posts)), random_state=42).tolist()
            post_text = "\n".join(f"- {p[:100]}" for p in samp)
            topic = topics.get(cid, "未知")[:60]

            prompt = CORRECT_PROMPT.format(
                topic=topic, question=qd["question"],
                options_numbered=opts_str, prediction=pred_str, posts=post_text)

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
            resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                   skip_special_tokens=True)
            resp = re.sub(r'<think>.*?</think>', '', resp, flags=re.DOTALL).strip()

            match = re.search(r'\{[^}]*"adjusted"\s*:\s*\[([^\]]+)\]', resp)
            if match:
                try:
                    vals = [float(x.strip()) for x in match.group(1).split(",")]
                    if len(vals) == len(opts):
                        arr = np.array(vals); arr = np.clip(arr, 0, None)
                        if arr.sum() > 0:
                            cpmfs[cid] = arr / arr.sum()
                            continue
                except: pass
            # Fallback: keep SSR
            cpmfs[cid] = ssr_pmf
        all_cpmfs[qi] = cpmfs
    return all_cpmfs


# ===== Method D: Persona-projected SSR =====
def method_d_projected_ssr(encoder, questions, cids, df, anchors_cache,
                           persona_vectors, n_posts=30, projection_strength=0.3):
    """Project post embeddings along persona direction, then SSR.

    For each post in cluster c:
      emb_adjusted = emb + strength * persona_direction_c (in embedding space)
    Then SSR with adjusted embedding.

    This uses the persona vector's direction to bias the SSR without generation.
    """
    all_cpmfs = {}
    for qi, qd in enumerate(questions):
        _, aembs = anchors_cache[qd["question"]]
        cpmfs = {}
        for cid in cids:
            cid_int = int(cid)
            if cid_int not in persona_vectors: continue
            posts = df[df["cluster_label"]==cid_int]["content_desc"].dropna()
            if len(posts) < 3: continue
            samp = posts.sample(min(n_posts, len(posts)), random_state=42).tolist()

            # Get persona direction in embedding space
            # Use mean of cluster post embeddings vs global mean as direction
            cluster_embs = encoder.encode(samp)
            global_sample = df["content_desc"].dropna().sample(min(100, len(df)), random_state=42).tolist()
            global_embs = encoder.encode(global_sample)
            direction = cluster_embs.mean(axis=0) - global_embs.mean(axis=0)
            direction = direction / (np.linalg.norm(direction) + 1e-8)

            pmfs = []
            for emb in cluster_embs:
                # Project: enhance the cluster-specific direction
                adjusted = emb + projection_strength * direction * np.linalg.norm(emb)
                # SSR with adjusted embedding
                sims = np.array([np.dot(adjusted, a) / (np.linalg.norm(adjusted) * np.linalg.norm(a) + 1e-8)
                                for a in aembs])
                sims = sims - sims.min()
                if sims.sum() == 0: sims += 1e-8
                pmf = sims / sims.sum()
                pmfs.append(pmf)
            cpmfs[cid] = np.mean(pmfs, axis=0)
        all_cpmfs[qi] = cpmfs
    return all_cpmfs


# ===== Method E: Ensemble SSR + Multi-layer steering =====
def method_e_ensemble(encoder, model, tokenizer, questions, cids, df,
                      persona_vectors, anchors_cache, n_posts=30,
                      alpha=0.1, layers=[16, 20, 24], ssr_weight=0.7):
    """Weighted ensemble of Direct SSR and multi-layer steering."""
    ssr_cpmfs = method_direct_ssr(encoder, questions, cids, df, anchors_cache, n_posts)

    prompt_tpl = ("作为一位有相关经验的消费者，请回答以下问卷问题。\n"
                  "请用自然的语言表达你的真实想法和感受，不需要选择选项。\n\n"
                  "问题：{question}\n选项：{options}\n\n你的回答：")
    all_cpmfs = {}
    for qi, qd in enumerate(questions):
        prompt = prompt_tpl.format(question=qd["question"], options="、".join(qd["options"]))
        _, aembs = anchors_cache[qd["question"]]
        cpmfs = {}
        for cid in cids:
            cid_int = int(cid)
            if cid_int not in persona_vectors or cid not in ssr_cpmfs[qi]: continue
            vec = persona_vectors[cid_int]["vector"]
            # Multi-layer steered
            steer_pmfs = []
            for _ in range(3):
                hooks = [SteeringHook(vec, alpha, l) for l in layers]
                for h in hooks: h.attach(model)
                try:
                    inputs = tokenizer(prompt, return_tensors="pt")
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}
                    with torch.no_grad():
                        out = model.generate(**inputs, max_new_tokens=200,
                                            do_sample=True, temperature=0.7, top_p=0.9)
                    resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                           skip_special_tokens=True)
                finally:
                    for h in hooks: h.remove()
                pmf, _ = ssr_score(resp, encoder, aembs)
                steer_pmfs.append(pmf)
            steer_avg = np.mean(steer_pmfs, axis=0)
            # Ensemble
            cpmfs[cid] = ssr_weight * ssr_cpmfs[qi][cid] + (1 - ssr_weight) * steer_avg
        all_cpmfs[qi] = cpmfs
    return all_cpmfs


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:2")
    args = parser.parse_args()

    Path(RESULTS_DIR).mkdir(exist_ok=True)
    questions, topics, df = load_setup()
    cids = get_top_clusters(questions, n=10)
    print(f"{len(questions)} questions, {len(cids)} clusters", flush=True)

    print("Loading models...", flush=True)
    encoder = SentenceTransformer(EMBEDDING_MODEL)
    model, tokenizer = load_model(MODEL_PATH, args.device)
    pvecs = load_persona_vectors(f"{RESULTS_DIR}/persona_vectors_L24_N20.npz")

    print("Generating anchors...", flush=True)
    ac = {}
    for qd in questions:
        q = qd["question"]
        if q not in ac:
            anchors = generate_anchors_local(model, tokenizer, q, qd["options"])
            aembs = [encoder.encode(a) for a in anchors]
            ac[q] = (anchors, aembs)
            print(f"  {q[:40]}...", flush=True)

    results = []

    # Baseline: Direct SSR
    print("\n" + "=" * 60, flush=True)
    print("BL: Direct SSR (no LLM)", flush=True)
    print("=" * 60, flush=True)
    cpmfs = method_direct_ssr(encoder, questions, cids, df, ac)
    r = evaluate("BL: Direct SSR", cpmfs, questions, cids)
    results.append(r)
    print(f"  Div={r['div']:.4f}  JS={r['js']:.4f}", flush=True)

    # Method A: Per-post voting
    print("\n" + "=" * 60, flush=True)
    print("Method A: Per-post LLM voting (20 posts/cluster)", flush=True)
    print("=" * 60, flush=True)
    cpmfs = method_a_voting(model, tokenizer, encoder, questions, cids, df, ac, n_posts=20)
    r = evaluate("A: LLM voting", cpmfs, questions, cids)
    results.append(r)
    print(f"  Div={r['div']:.4f}  JS={r['js']:.4f}", flush=True)

    # Method B: Distribution estimation
    print("\n" + "=" * 60, flush=True)
    print("Method B: LLM distribution estimation", flush=True)
    print("=" * 60, flush=True)
    cpmfs = method_b_dist_estimation(model, tokenizer, encoder, questions, cids, df, ac, topics)
    r = evaluate("B: LLM dist est", cpmfs, questions, cids)
    results.append(r)
    print(f"  Div={r['div']:.4f}  JS={r['js']:.4f}", flush=True)

    # Method C: SSR + LLM correction
    print("\n" + "=" * 60, flush=True)
    print("Method C: SSR + LLM correction", flush=True)
    print("=" * 60, flush=True)
    cpmfs = method_c_ssr_correction(model, tokenizer, encoder, questions, cids, df, ac, topics)
    r = evaluate("C: SSR+LLM correct", cpmfs, questions, cids)
    results.append(r)
    print(f"  Div={r['div']:.4f}  JS={r['js']:.4f}", flush=True)

    # Method D: Persona-projected SSR
    print("\n" + "=" * 60, flush=True)
    print("Method D: Persona-projected SSR", flush=True)
    print("=" * 60, flush=True)
    for strength in [0.1, 0.3, 0.5]:
        cpmfs = method_d_projected_ssr(encoder, questions, cids, df, ac, pvecs,
                                       projection_strength=strength)
        r = evaluate(f"D: Proj SSR s={strength}", cpmfs, questions, cids)
        results.append(r)
        print(f"  s={strength}: Div={r['div']:.4f}  JS={r['js']:.4f}", flush=True)

    # Method E: Ensemble
    print("\n" + "=" * 60, flush=True)
    print("Method E: SSR + Multi-layer ensemble", flush=True)
    print("=" * 60, flush=True)
    for sw in [0.5, 0.7, 0.9]:
        cpmfs = method_e_ensemble(encoder, model, tokenizer, questions, cids, df, pvecs, ac,
                                  ssr_weight=sw)
        r = evaluate(f"E: Ensemble w={sw}", cpmfs, questions, cids)
        results.append(r)
        print(f"  w={sw}: Div={r['div']:.4f}  JS={r['js']:.4f}", flush=True)

    # Summary
    print("\n" + "=" * 70, flush=True)
    print("SUMMARY (sorted by JS)", flush=True)
    print("=" * 70, flush=True)
    print(f"  {'Method':<28} {'Div':>8} {'JS ↓':>8}  Per-Q JS", flush=True)
    print("  " + "-" * 90, flush=True)
    for r in sorted(results, key=lambda x: x["js"]):
        pq = " ".join(f"{j:.4f}" for j in r["per_q"])
        beat = " *** BEATS BL" if r["js"] < results[0]["js"] and r["name"] != "BL: Direct SSR" else ""
        print(f"  {r['name']:<28} {r['div']:>8.4f} {r['js']:>8.4f}  {pq}{beat}", flush=True)

    with open(f"{RESULTS_DIR}/persona_methods.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {RESULTS_DIR}/persona_methods.json", flush=True)


if __name__ == "__main__":
    main()
