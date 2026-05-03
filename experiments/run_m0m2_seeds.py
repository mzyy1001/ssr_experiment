"""
Priority 1: M0/M1/M2 stability across 5 random seeds.

Runs ONE seed of the M0/M1/M2 dump (matching dump_m0_m2_pmfs.py exactly),
saves per-question PMFs and JS/K/C metrics. Aggregation across seeds is
done by a separate post-process step.

Output: results/seeds/m0m2_seed{seed}.json
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True,
                   help="Single seed for THIS run (varies post sampling + torch RNG)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--n_posts", type=int, default=50)
    p.add_argument("--n_samples", type=int, default=3)
    p.add_argument("--ensemble_w", type=float, default=0.5)
    p.add_argument("--output_dir", default=f"{RESULTS_DIR}/seeds")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    log(f"=== seed={args.seed} on {args.device} ===")
    log("Loading data …")
    questions, topics, df = load_orig6()
    top_cids = get_top_clusters(questions, n=args.top_k)
    log(f"Top {args.top_k} clusters: {top_cids}")

    log("Loading model + encoder …")
    encoder = SentenceTransformer(EMBED_PATH, device=args.device)
    model, tokenizer = load_model(MODEL_PATH, args.device)

    # Anchors stay deterministic across seeds (don't confound with seed effect).
    log("Generating anchors (fixed seed=42 across all runs) …")
    set_seed(42)
    anchors = {}
    for qd in questions:
        anc = generate_anchors_local(model, tokenizer, qd["question"], qd["options"])
        anchors[qd["key"]] = [encoder.encode(a) for a in anc]

    # Now switch RNG to per-run seed, AND vary post-sampling seed.
    set_seed(args.seed)

    log("M0: SSR per-cluster …")
    per_q_ssr = {}
    for qi, qd in enumerate(questions):
        per_q_ssr[qi] = compute_ssr_pmfs(encoder, qd, top_cids, df,
                                          anchors[qd["key"]],
                                          n_posts=args.n_posts,
                                          sample_seed=args.seed)
        log(f"  Q{qi+1}/{len(questions)} M0 done")

    log("M1: LLM-dist per-cluster …")
    per_q_llm = {}
    for qi, qd in enumerate(questions):
        per_q_llm[qi] = compute_llmdist_pmfs(model, tokenizer, qd, top_cids,
                                              df, topics,
                                              n_ex=8, n_samples=args.n_samples,
                                              sample_seed=args.seed)
        log(f"  Q{qi+1}/{len(questions)} M1 done")

    log("Aggregating + scoring …")
    js_m0, js_m1, js_m2 = [], [], []
    out = {"config": vars(args), "top_clusters": [str(c) for c in top_cids],
           "per_question": {}, "summary": {}}
    for qi, qd in enumerate(questions):
        true = list(qd["true_distribution"].values())
        pred_m0 = agg(per_q_ssr[qi], qd["cluster_weights"], top_cids)
        pred_m1 = agg(per_q_llm[qi], qd["cluster_weights"], top_cids)
        m2_cpmfs = {c: args.ensemble_w * per_q_ssr[qi][c]
                       + (1 - args.ensemble_w) * per_q_llm[qi][c]
                    for c in top_cids
                    if c in per_q_ssr[qi] and c in per_q_llm[qi]}
        pred_m2 = agg(m2_cpmfs, qd["cluster_weights"], top_cids)
        e = {"question": qd["question"], "options": qd["options"],
             "true": [float(x) for x in true],
             "pred_m0": [float(x) for x in pred_m0],
             "pred_m1": [float(x) for x in pred_m1],
             "pred_m2": [float(x) for x in pred_m2],
             "m0": {"js": js_score(true, pred_m0), "k_xy": k_xy(true, pred_m0), "c_xy": c_xy(true, pred_m0)},
             "m1": {"js": js_score(true, pred_m1), "k_xy": k_xy(true, pred_m1), "c_xy": c_xy(true, pred_m1)},
             "m2": {"js": js_score(true, pred_m2), "k_xy": k_xy(true, pred_m2), "c_xy": c_xy(true, pred_m2)}}
        js_m0.append(e["m0"]["js"]); js_m1.append(e["m1"]["js"]); js_m2.append(e["m2"]["js"])
        out["per_question"][qd["key"]] = e
        log(f"  {qd['key']}: M0={e['m0']['js']:.4f} M1={e['m1']['js']:.4f} M2={e['m2']['js']:.4f}")

    out["summary"] = {
        "M0": float(np.mean(js_m0)),
        "M1": float(np.mean(js_m1)),
        "M2": float(np.mean(js_m2)),
    }
    log(f"seed={args.seed} | M0={out['summary']['M0']:.4f} M1={out['summary']['M1']:.4f} M2={out['summary']['M2']:.4f}")

    out_path = f"{args.output_dir}/m0m2_seed{args.seed}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    log(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
