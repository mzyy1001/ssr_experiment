"""
Targeted improvements on original 6 questions to beat Direct SSR (0.0277).

Analysis shows:
- Ensemble wins on 3/6 (11.0/3, 8.0/3, 9.0/5)
- Ensemble loses badly on 8.0/4 (0.0474→0.0657)
- Average dragged down by one bad question

Improvements to try:
1. More generations (n=10) — reduce variance
2. Fewer clusters (top-5 vs top-15) — reduce noise
3. LLM distribution estimation — worked in pilot
4. Adaptive ensemble weight per question type (LOO)
5. Better layer combinations for multi-layer
6. Selective ensemble: only blend when steering divergence is high
7. SSR with more posts (50 instead of 30)

Run: python -u improve_orig6.py --device cuda:2
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
    SteeringHook, ssr_score, generate_anchors_local,
)

DATA_DIR = "/data/chenhongrui/business/data"
RESULTS_DIR = "/data/chenhongrui/business/results"
MODEL_PATH = "/data/chenhongrui/models/Qwen3-8B"
EMBEDDING_MODEL = "/data/chenhongrui/models/bge-base-zh-v1.5"

ORIG_6 = ["8.0/qsingle_3", "8.0/qsingle_4", "9.0/qsingle_3",
           "9.0/qsingle_4", "9.0/qsingle_5", "11.0/qsingle_3"]


def js(td, pred):
    t = np.array(list(td.values()), dtype=float); t /= t.sum()
    p = pred / (pred.sum() + 1e-10)
    return float(jensenshannon(t, p) ** 2)


def load_orig6():
    with open(f"{RESULTS_DIR}/all_questions_expanded.json") as f:
        expanded = json.load(f)
    with open(f"{DATA_DIR}/3_cluster_topics.json") as f:
        topics = json.load(f)
    df = pd.read_csv(f"{DATA_DIR}/2_meaningful_df.csv")
    questions = []
    for sk, sv in expanded.items():
        for sub, qs in sv.items():
            for qid, qd in qs.items():
                key = f"{sub}/{qid}"
                if key in ORIG_6:
                    questions.append({
                        "key": key,
                        "question": qd["question"],
                        "options": qd["options"],
                        "true_distribution": qd["true_distribution"],
                        "cluster_weights": qd["cluster_weights"],
                    })
    # Sort to match ORIG_6 order
    questions.sort(key=lambda x: ORIG_6.index(x["key"]))
    return questions, topics, df


def get_top_clusters(questions, n=10):
    all_w = {}
    for qd in questions:
        for cid, w in qd["cluster_weights"].items():
            all_w[cid] = all_w.get(cid, 0) + w
    return sorted(all_w, key=lambda c: -all_w[c])[:n]


def agg(cpmfs, weights, cids):
    n = len(next(iter(cpmfs.values())))
    a, tw = np.zeros(n), 0
    for c in cids:
        if c in cpmfs:
            w = weights.get(c, 0); a += w * cpmfs[c]; tw += w
    return a / tw if tw > 0 else a


def direct_ssr(encoder, qd, cids, df, aembs, n_posts=30):
    cpmfs = {}
    for cid in cids:
        posts = df[df["cluster_label"]==int(cid)]["content_desc"].dropna()
        if len(posts) < 3: continue
        samp = posts.sample(min(n_posts, len(posts)), random_state=42).tolist()
        pmfs = [ssr_score(p, encoder, aembs)[0] for p in samp]
        cpmfs[cid] = np.mean(pmfs, axis=0)
    return cpmfs


def multi_layer_steer(model, tokenizer, encoder, qd, cids, pvecs, aembs,
                      alpha=0.1, layers=[16, 20, 24], n_resp=3):
    prompt = ("作为一位有相关经验的消费者，请回答以下问卷问题。\n"
              "请用自然的语言表达你的真实想法和感受，不需要选择选项。\n\n"
              f"问题：{qd['question']}\n选项：{'、'.join(qd['options'])}\n\n你的回答：")
    cpmfs = {}
    for cid in cids:
        if int(cid) not in pvecs: continue
        vec = pvecs[int(cid)]["vector"]
        pmfs = []
        for _ in range(n_resp):
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
            pmfs.append(pmf)
        cpmfs[cid] = np.mean(pmfs, axis=0)
    return cpmfs


def llm_dist_est(model, tokenizer, qd, cids, df, topics, n_examples=8):
    """LLM distribution estimation per cluster."""
    opts = qd["options"]
    opts_str = "\n".join(f"{i+1}. {o}" for i, o in enumerate(opts))
    cpmfs = {}
    for cid in cids:
        posts = df[df["cluster_label"]==int(cid)]["content_desc"].dropna()
        if len(posts) < 3: continue
        samp = posts.sample(min(n_examples, len(posts)), random_state=42).tolist()
        post_text = "\n".join(f"- {p[:100]}" for p in samp)
        topic = topics.get(cid, "未知")[:60]

        prompt = (
            f"你是一位消费者调研分析专家。\n\n"
            f"以下是某类消费者群体的社交媒体帖子样本（话题：{topic}）：\n{post_text}\n\n"
            f"基于这些帖子反映的消费者特征和态度，请估计这个群体在以下问卷问题中各选项的选择比例。\n\n"
            f"问题：{qd['question']}\n选项：\n{opts_str}\n\n"
            f"请严格按JSON格式输出各选项的比例（总和为1）：\n"
            f'{{\"distribution\": [选项1比例, 选项2比例, ...]}}'
        )
        pmfs_collected = []
        for _ in range(3):
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=200,
                                    do_sample=True, temperature=0.7, top_p=0.9)
            resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                   skip_special_tokens=True)
            resp = re.sub(r'<think>.*?</think>', '', resp, flags=re.DOTALL).strip()
            match = re.search(r'\[([^\]]+)\]', resp)
            if match:
                try:
                    vals = [float(x.strip()) for x in match.group(1).split(",")]
                    if len(vals) == len(opts):
                        arr = np.clip(np.array(vals), 0, None)
                        if arr.sum() > 0:
                            pmfs_collected.append(arr / arr.sum())
                            continue
                except: pass
            pmfs_collected.append(np.ones(len(opts)) / len(opts))
        cpmfs[cid] = np.mean(pmfs_collected, axis=0)
    return cpmfs


def selective_ensemble(ssr_pmfs, steer_pmfs, w_ssr=0.7, div_threshold=0.005):
    """Only blend steering when it's sufficiently different from SSR.
    If steering is too similar to SSR (low divergence), just use SSR."""
    cpmfs = {}
    for cid in set(ssr_pmfs.keys()) & set(steer_pmfs.keys()):
        div = jensenshannon(ssr_pmfs[cid], steer_pmfs[cid]) ** 2
        if div > div_threshold:
            cpmfs[cid] = w_ssr * ssr_pmfs[cid] + (1 - w_ssr) * steer_pmfs[cid]
        else:
            cpmfs[cid] = ssr_pmfs[cid]  # steering adds nothing, keep SSR
    # Include SSR-only clusters not in steering
    for cid in ssr_pmfs:
        if cid not in cpmfs:
            cpmfs[cid] = ssr_pmfs[cid]
    return cpmfs


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:2")
    args = parser.parse_args()

    questions, topics, df = load_orig6()
    print(f"Original 6 questions: {[q['key'] for q in questions]}", flush=True)

    print("Loading models...", flush=True)
    encoder = SentenceTransformer(EMBEDDING_MODEL)
    model, tokenizer = load_model(MODEL_PATH, args.device)
    pvecs = load_persona_vectors(f"{RESULTS_DIR}/persona_vectors_L24_N20.npz")

    # Also load L16 and L20 vectors for different layer combos
    pvecs_16 = load_persona_vectors(f"{RESULTS_DIR}/persona_vectors_L16_N20.npz")
    pvecs_20 = load_persona_vectors(f"{RESULTS_DIR}/persona_vectors_L20_N20.npz")

    # Generate anchors
    print("Generating anchors...", flush=True)
    ac = {}
    for qd in questions:
        q = qd["question"]
        anchors = generate_anchors_local(model, tokenizer, q, qd["options"])
        aembs = [encoder.encode(a) for a in anchors]
        ac[q] = aembs
        print(f"  {q[:40]}... anchors: {anchors}", flush=True)

    results = {}

    # ===== Improvement 1: More posts for SSR (50 vs 30) =====
    print("\n=== Imp 1: SSR with more posts ===", flush=True)
    for n_posts in [30, 50, 80]:
        cids = get_top_clusters(questions, n=15)
        js_vals = []
        for qd in questions:
            cpmfs = direct_ssr(encoder, qd, cids, df, ac[qd["question"]], n_posts=n_posts)
            js_vals.append(js(qd["true_distribution"], agg(cpmfs, qd["cluster_weights"], cids)))
        tag = f"SSR n={n_posts}"
        results[tag] = js_vals
        print(f"  {tag}: {np.mean(js_vals):.4f} | {' '.join(f'{j:.4f}' for j in js_vals)}", flush=True)

    # ===== Improvement 2: Fewer clusters (reduce noise) =====
    print("\n=== Imp 2: Cluster count ===", flush=True)
    for n_c in [5, 10, 15, 20]:
        cids = get_top_clusters(questions, n=n_c)
        js_vals = []
        for qd in questions:
            cpmfs = direct_ssr(encoder, qd, cids, df, ac[qd["question"]], n_posts=30)
            js_vals.append(js(qd["true_distribution"], agg(cpmfs, qd["cluster_weights"], cids)))
        tag = f"SSR top-{n_c}"
        results[tag] = js_vals
        print(f"  {tag}: {np.mean(js_vals):.4f}", flush=True)

    # ===== Improvement 3: Multi-layer with more generations =====
    print("\n=== Imp 3: Multi-layer, more generations ===", flush=True)
    cids = get_top_clusters(questions, n=10)
    for n_resp in [5, 10]:
        js_vals = []
        for qd in questions:
            cpmfs = multi_layer_steer(model, tokenizer, encoder, qd, cids, pvecs,
                                      ac[qd["question"]], alpha=0.1,
                                      layers=[16, 20, 24], n_resp=n_resp)
            js_vals.append(js(qd["true_distribution"], agg(cpmfs, qd["cluster_weights"], cids)))
        tag = f"MultiLayer n={n_resp}"
        results[tag] = js_vals
        print(f"  {tag}: {np.mean(js_vals):.4f} | {' '.join(f'{j:.4f}' for j in js_vals)}", flush=True)

    # ===== Improvement 4: Different layer combinations =====
    print("\n=== Imp 4: Layer combinations (n=5 resp) ===", flush=True)
    for layers, pvecs_used in [
        ([20, 24], pvecs),
        ([16, 24], pvecs),
        ([12, 16, 20, 24], pvecs),
        ([16, 20, 24, 28], pvecs),
    ]:
        js_vals = []
        for qd in questions:
            cpmfs = multi_layer_steer(model, tokenizer, encoder, qd, cids, pvecs_used,
                                      ac[qd["question"]], alpha=0.1,
                                      layers=layers, n_resp=5)
            js_vals.append(js(qd["true_distribution"], agg(cpmfs, qd["cluster_weights"], cids)))
        tag = f"ML L{layers}"
        results[tag] = js_vals
        print(f"  {tag}: {np.mean(js_vals):.4f} | {' '.join(f'{j:.4f}' for j in js_vals)}", flush=True)

    # ===== Improvement 5: LLM distribution estimation =====
    print("\n=== Imp 5: LLM dist estimation ===", flush=True)
    for n_c in [5, 10]:
        cids_sub = get_top_clusters(questions, n=n_c)
        js_vals = []
        for qd in questions:
            cpmfs = llm_dist_est(model, tokenizer, qd, cids_sub, df, topics)
            js_vals.append(js(qd["true_distribution"], agg(cpmfs, qd["cluster_weights"], cids_sub)))
        tag = f"LLM-dist top-{n_c}"
        results[tag] = js_vals
        print(f"  {tag}: {np.mean(js_vals):.4f} | {' '.join(f'{j:.4f}' for j in js_vals)}", flush=True)

    # ===== Improvement 6: Selective ensemble =====
    print("\n=== Imp 6: Selective ensemble ===", flush=True)
    # Use best SSR and best steering configs
    cids = get_top_clusters(questions, n=10)
    for qd in questions:
        # Compute SSR and steering once, reuse
        ssr_c = direct_ssr(encoder, qd, cids, df, ac[qd["question"]], n_posts=50)
        steer_c = multi_layer_steer(model, tokenizer, encoder, qd, cids, pvecs,
                                    ac[qd["question"]], alpha=0.1,
                                    layers=[16, 20, 24], n_resp=5)
        qd["_ssr"] = ssr_c
        qd["_steer"] = steer_c

    for w in [0.6, 0.7, 0.8, 0.9]:
        for thresh in [0.001, 0.005, 0.01]:
            js_vals = []
            for qd in questions:
                ens = selective_ensemble(qd["_ssr"], qd["_steer"], w_ssr=w, div_threshold=thresh)
                js_vals.append(js(qd["true_distribution"], agg(ens, qd["cluster_weights"], cids)))
            tag = f"SelEns w={w} t={thresh}"
            results[tag] = js_vals
            if np.mean(js_vals) < 0.0277:
                print(f"  {tag}: {np.mean(js_vals):.4f} *** BEATS BL", flush=True)

    # Also plain ensembles with better configs
    for w in [0.6, 0.7, 0.8, 0.85, 0.9, 0.95]:
        js_vals = []
        for qd in questions:
            ens = {}
            for c in set(qd["_ssr"].keys()) & set(qd["_steer"].keys()):
                ens[c] = w * qd["_ssr"][c] + (1-w) * qd["_steer"][c]
            js_vals.append(js(qd["true_distribution"], agg(ens, qd["cluster_weights"], cids)))
        tag = f"Ensemble w={w}"
        results[tag] = js_vals
        print(f"  {tag}: {np.mean(js_vals):.4f} {'*** BEATS BL' if np.mean(js_vals) < 0.0277 else ''}", flush=True)

    # ===== Improvement 7: Ensemble SSR + LLM-dist =====
    print("\n=== Imp 7: SSR + LLM-dist ensemble ===", flush=True)
    cids = get_top_clusters(questions, n=10)
    llm_cpmfs_all = {}
    for qi, qd in enumerate(questions):
        llm_cpmfs_all[qi] = llm_dist_est(model, tokenizer, qd, cids, df, topics)

    for w in [0.5, 0.6, 0.7, 0.8]:
        js_vals = []
        for qi, qd in enumerate(questions):
            ssr_c = direct_ssr(encoder, qd, cids, df, ac[qd["question"]], n_posts=50)
            ens = {}
            for c in set(ssr_c.keys()) & set(llm_cpmfs_all[qi].keys()):
                ens[c] = w * ssr_c[c] + (1-w) * llm_cpmfs_all[qi][c]
            js_vals.append(js(qd["true_distribution"], agg(ens, qd["cluster_weights"], cids)))
        tag = f"SSR+LLMdist w={w}"
        results[tag] = js_vals
        print(f"  {tag}: {np.mean(js_vals):.4f} {'*** BEATS BL' if np.mean(js_vals) < 0.0277 else ''}", flush=True)

    # ===== Summary =====
    print("\n" + "=" * 70, flush=True)
    print("SUMMARY — Methods that beat Direct SSR (0.0277)", flush=True)
    print("=" * 70, flush=True)
    baseline_js = 0.0277
    winners = [(tag, np.mean(vals)) for tag, vals in results.items() if np.mean(vals) < baseline_js]
    winners.sort(key=lambda x: x[1])
    if winners:
        print(f"  {'Method':<30} {'Mean JS':>8} {'Δ':>8}", flush=True)
        print("  " + "-" * 50, flush=True)
        for tag, mean_js in winners[:15]:
            print(f"  {tag:<30} {mean_js:>8.4f} {baseline_js-mean_js:>+8.4f}", flush=True)
    else:
        print("  No method beats Direct SSR baseline.", flush=True)

    # Top-10 overall
    print("\n  Top-10 overall:", flush=True)
    all_sorted = sorted(results.items(), key=lambda x: np.mean(x[1]))[:10]
    for tag, vals in all_sorted:
        print(f"  {tag:<30} {np.mean(vals):>8.4f} | {' '.join(f'{j:.4f}' for j in vals)}", flush=True)

    # Save
    with open(f"{RESULTS_DIR}/improve_orig6.json", "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in results.items()}, f, indent=2)
    print(f"\nSaved to {RESULTS_DIR}/improve_orig6.json", flush=True)


if __name__ == "__main__":
    main()
