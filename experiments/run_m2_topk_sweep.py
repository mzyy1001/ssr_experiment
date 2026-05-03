"""
Optional: M2 top-K sweep over K ∈ {5, 10, 15, 20}.

Loads the model + encoder once, then runs M0 (per-cluster SSR n=50) and
M1 (per-cluster LLM-dist n_samples=3) over each K's top-K cluster list,
then ensembles M2 = 0.5 M0 + 0.5 M1 at the cluster level.

Output: results/topk_sweep_6q.json
"""
import argparse
import json
import os
import re
import time
import random

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import jensenshannon
from sentence_transformers import SentenceTransformer

from persona_vectors import load_model
from steered_ssr import ssr_score, generate_anchors_local


DATA_DIR = "/data/chenhongrui/business/data"
RESULTS_DIR = "/data/chenhongrui/business/results"
MODEL_PATH = "/data/chenhongrui/models/Qwen3-8B"
EMBED_PATH = "/data/chenhongrui/models/bge-base-zh-v1.5"

ORIG_6 = ["8.0/qsingle_3", "8.0/qsingle_4", "9.0/qsingle_3",
          "9.0/qsingle_4", "9.0/qsingle_5", "11.0/qsingle_3"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


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
                        "key": key, "question": qd["question"],
                        "options": qd["options"],
                        "true_distribution": qd["true_distribution"],
                        "cluster_weights": qd["cluster_weights"],
                    })
    questions.sort(key=lambda x: ORIG_6.index(x["key"]))
    return questions, topics, df


def get_top_clusters(questions, n=10):
    aw = {}
    for qd in questions:
        for cid, w in qd["cluster_weights"].items():
            aw[cid] = aw.get(cid, 0) + w
    return sorted(aw, key=lambda c: -aw[c])[:n]


def js_score(t, p):
    t = np.array(t, float); t /= t.sum()
    p = np.array(p, float); p /= (p.sum() + 1e-10)
    return float(jensenshannon(t, p) ** 2)


def k_xy(t, p):
    t = np.array(t, float); t /= t.sum()
    p = np.array(p, float); p /= (p.sum() + 1e-10)
    return 1 - float(np.max(np.abs(np.cumsum(t) - np.cumsum(p))))


def c_xy(t, p):
    t = np.array(t, float); t /= t.sum()
    p = np.array(p, float); p /= (p.sum() + 1e-10)
    return float(np.dot(t, p) / (np.linalg.norm(t) * np.linalg.norm(p) + 1e-10))


def agg(cpmfs, weights, cids):
    n = len(next(iter(cpmfs.values())))
    a, tw = np.zeros(n), 0.0
    for c in cids:
        if c in cpmfs:
            w = weights.get(c, 0); a += w * cpmfs[c]; tw += w
    return a / tw if tw > 0 else a


def compute_ssr_pmfs(encoder, qd, cids, df, aembs, n_posts=50, sample_seed=42):
    cpmfs = {}
    for cid in cids:
        posts = df[df["cluster_label"] == int(cid)]["content_desc"].dropna()
        if len(posts) < 3: continue
        samp = posts.sample(min(n_posts, len(posts)), random_state=sample_seed).tolist()
        pmfs = [ssr_score(p, encoder, aembs)[0] for p in samp]
        cpmfs[cid] = np.mean(pmfs, axis=0)
    return cpmfs


def compute_llmdist_pmfs(model, tokenizer, qd, cids, df, topics,
                         n_ex=8, n_samples=3, sample_seed=42):
    opts = qd["options"]
    opts_str = "\n".join(f"{i+1}. {o}" for i, o in enumerate(opts))
    cpmfs = {}
    for cid in cids:
        posts = df[df["cluster_label"] == int(cid)]["content_desc"].dropna()
        if len(posts) < 3: continue
        samp = posts.sample(min(n_ex, len(posts)), random_state=sample_seed).tolist()
        post_text = "\n".join(f"- {p[:100]}" for p in samp)
        topic = topics.get(cid, "未知")[:60]
        prompt = (
            f"你是一位消费者调研分析专家。\n\n"
            f"以下是某类消费者群体的社交媒体帖子样本（话题：{topic}）：\n{post_text}\n\n"
            f"基于这些帖子反映的消费者特征和态度，请估计这个群体在以下问卷问题中各选项的选择比例。\n\n"
            f"问题：{qd['question']}\n选项：\n{opts_str}\n\n"
            f"请严格按JSON格式输出各选项的比例（总和为1）：\n"
            f'{{"distribution": [选项1比例, 选项2比例, ...]}}'
        )
        pmfs_collected = []
        for _ in range(n_samples):
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=200,
                                    do_sample=True, temperature=0.7, top_p=0.9)
            resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                   skip_special_tokens=True)
            resp = re.sub(r'<think>.*?</think>', '', resp, flags=re.DOTALL).strip()
            m = re.search(r'\[([^\]]+)\]', resp)
            ok = False
            if m:
                try:
                    vals = [float(x.strip()) for x in m.group(1).split(",")]
                    if len(vals) == len(opts):
                        arr = np.clip(np.array(vals), 0, None)
                        if arr.sum() > 0:
                            pmfs_collected.append(arr / arr.sum())
                            ok = True
                except Exception:
                    pass
            if not ok:
                pmfs_collected.append(np.ones(len(opts)) / len(opts))
        cpmfs[cid] = np.mean(pmfs_collected, axis=0)
    return cpmfs


def evaluate_for_k(model, tokenizer, encoder, questions, topics, df,
                   anchors, K, n_posts, n_samples, ensemble_w, seed):
    cids = get_top_clusters(questions, n=K)
    log(f"  K={K}: clusters = {cids}")
    js_m0, js_m1, js_m2 = [], [], []
    k_m0, k_m1, k_m2 = [], [], []
    c_m0, c_m1, c_m2 = [], [], []
    per_q = {}
    for qi, qd in enumerate(questions):
        ssr_p = compute_ssr_pmfs(encoder, qd, cids, df,
                                  anchors[qd["key"]],
                                  n_posts=n_posts, sample_seed=seed)
        llm_p = compute_llmdist_pmfs(model, tokenizer, qd, cids,
                                      df, topics,
                                      n_ex=8, n_samples=n_samples,
                                      sample_seed=seed)
        true = list(qd["true_distribution"].values())
        pred_m0 = agg(ssr_p, qd["cluster_weights"], cids)
        pred_m1 = agg(llm_p, qd["cluster_weights"], cids)
        m2_cpmfs = {c: ensemble_w * ssr_p[c] + (1 - ensemble_w) * llm_p[c]
                    for c in cids if c in ssr_p and c in llm_p}
        pred_m2 = agg(m2_cpmfs, qd["cluster_weights"], cids)
        e = {"true": [float(x) for x in true],
             "m0": {"js": js_score(true, pred_m0), "k_xy": k_xy(true, pred_m0), "c_xy": c_xy(true, pred_m0)},
             "m1": {"js": js_score(true, pred_m1), "k_xy": k_xy(true, pred_m1), "c_xy": c_xy(true, pred_m1)},
             "m2": {"js": js_score(true, pred_m2), "k_xy": k_xy(true, pred_m2), "c_xy": c_xy(true, pred_m2)}}
        js_m0.append(e["m0"]["js"]); js_m1.append(e["m1"]["js"]); js_m2.append(e["m2"]["js"])
        k_m0.append(e["m0"]["k_xy"]); k_m1.append(e["m1"]["k_xy"]); k_m2.append(e["m2"]["k_xy"])
        c_m0.append(e["m0"]["c_xy"]); c_m1.append(e["m1"]["c_xy"]); c_m2.append(e["m2"]["c_xy"])
        per_q[qd["key"]] = e
        log(f"    {qd['key']}: M0={e['m0']['js']:.4f} M1={e['m1']['js']:.4f} M2={e['m2']['js']:.4f}")
    summary = {
        "M0": {"js": float(np.mean(js_m0)), "k_xy": float(np.mean(k_m0)), "c_xy": float(np.mean(c_m0))},
        "M1": {"js": float(np.mean(js_m1)), "k_xy": float(np.mean(k_m1)), "c_xy": float(np.mean(c_m1))},
        "M2": {"js": float(np.mean(js_m2)), "k_xy": float(np.mean(k_m2)), "c_xy": float(np.mean(c_m2))},
    }
    log(f"  K={K} mean: M0={summary['M0']['js']:.4f}  M1={summary['M1']['js']:.4f}  M2={summary['M2']['js']:.4f}")
    return {"K": K, "clusters": cids, "summary": summary, "per_q": per_q}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--ks", type=int, nargs="+", default=[5, 10, 15, 20])
    p.add_argument("--n_posts", type=int, default=50)
    p.add_argument("--n_samples", type=int, default=3)
    p.add_argument("--ensemble_w", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default=f"{RESULTS_DIR}/topk_sweep_6q.json")
    args = p.parse_args()

    log(f"=== top-K sweep on {args.device}, K={args.ks} ===")
    log("Loading data …")
    questions, topics, df = load_orig6()

    log("Loading model + encoder …")
    encoder = SentenceTransformer(EMBED_PATH, device=args.device)
    model, tokenizer = load_model(MODEL_PATH, args.device)

    log("Generating anchors (deterministic seed=42; shared across K) …")
    set_seed(42)
    anchors = {}
    for qd in questions:
        anc = generate_anchors_local(model, tokenizer, qd["question"], qd["options"])
        anchors[qd["key"]] = [encoder.encode(a) for a in anc]

    set_seed(args.seed)
    out = {"config": vars(args), "by_k": {}}
    for K in args.ks:
        t0 = time.time()
        out["by_k"][str(K)] = evaluate_for_k(
            model, tokenizer, encoder, questions, topics, df, anchors,
            K=K, n_posts=args.n_posts, n_samples=args.n_samples,
            ensemble_w=args.ensemble_w, seed=args.seed)
        log(f"  K={K} done in {time.time()-t0:.0f}s")

    log("=" * 60)
    log("SWEEP SUMMARY (mean JS over 6Q)")
    log("=" * 60)
    log(f"  {'K':>4} {'M0':>8} {'M1':>8} {'M2':>8}")
    for K in args.ks:
        s = out["by_k"][str(K)]["summary"]
        log(f"  {K:>4} {s['M0']['js']:.4f} {s['M1']['js']:.4f} {s['M2']['js']:.4f}")

    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    log(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
