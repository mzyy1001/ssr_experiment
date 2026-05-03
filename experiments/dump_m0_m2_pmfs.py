"""
Re-run M0 (hier-SSR n=50) and M2 (M0 + M1-vanilla ensemble w=0.5) on 6Q
benchmark and SAVE the aggregated predicted PMFs (the steered_m1_6q.json
only saved JS scalars, not PMFs needed for K_xy / C_xy).

Output: results/m0_m2_pmfs_6q.json with per-question {true, pred_m0, pred_m1,
pred_m2} + js/k_xy/c_xy summary.
"""
import argparse
import json
import re
import time
import random

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import jensenshannon
from sentence_transformers import SentenceTransformer

from persona_vectors import load_model, load_persona_vectors
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
                        "key": key,
                        "question": qd["question"],
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


def compute_ssr_pmfs(encoder, qd, cids, df, aembs, n_posts=50, seed=42):
    cpmfs = {}
    for cid in cids:
        posts = df[df["cluster_label"] == int(cid)]["content_desc"].dropna()
        if len(posts) < 3: continue
        samp = posts.sample(min(n_posts, len(posts)), random_state=seed).tolist()
        pmfs = [ssr_score(p, encoder, aembs)[0] for p in samp]
        cpmfs[cid] = np.mean(pmfs, axis=0)
    return cpmfs


def compute_llmdist_pmfs(model, tokenizer, qd, cids, df, topics,
                         n_ex=8, n_samples=3):
    """LLM distribution estimation per cluster (vanilla = no steering)."""
    opts = qd["options"]
    opts_str = "\n".join(f"{i+1}. {o}" for i, o in enumerate(opts))
    cpmfs = {}
    for cid in cids:
        posts = df[df["cluster_label"] == int(cid)]["content_desc"].dropna()
        if len(posts) < 3: continue
        samp = posts.sample(min(n_ex, len(posts)), random_state=42).tolist()
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
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                              max_length=1024)
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
    p.add_argument("--device", default="cuda:2")
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--n_posts", type=int, default=50)
    p.add_argument("--n_samples", type=int, default=3)
    p.add_argument("--ensemble_w", type=float, default=0.5,
                   help="M2 = w*M0 + (1-w)*M1 vanilla (paper memory says w=0.5 was the M2 reported)")
    p.add_argument("--output", default=f"{RESULTS_DIR}/m0_m2_pmfs_6q.json")
    args = p.parse_args()

    log("Loading data …")
    questions, topics, df = load_orig6()
    top_cids = get_top_clusters(questions, n=args.top_k)
    log(f"Top {args.top_k} clusters: {top_cids}")

    log("Loading model + encoder …")
    encoder = SentenceTransformer(EMBED_PATH)
    model, tokenizer = load_model(MODEL_PATH, args.device)
    log("Model loaded")

    log("Generating anchors (deterministic, seeded) …")
    set_seed(42)
    anchors = {}
    for qd in questions:
        anc = generate_anchors_local(model, tokenizer, qd["question"], qd["options"])
        anchors[qd["key"]] = [encoder.encode(a) for a in anc]
        log(f"  {qd['key']}: {anc[0][:40]}…")

    log("Computing M0 SSR per-cluster (n=50) …")
    per_q_ssr = {}
    for qi, qd in enumerate(questions):
        per_q_ssr[qi] = compute_ssr_pmfs(encoder, qd, top_cids, df,
                                          anchors[qd["key"]], n_posts=args.n_posts)
        log(f"  Q{qi+1}/{len(questions)} {qd['key']} SSR done ({len(per_q_ssr[qi])} clusters)")

    log("Computing M1 vanilla LLM-dist per-cluster …")
    set_seed(2026)
    per_q_llm = {}
    for qi, qd in enumerate(questions):
        per_q_llm[qi] = compute_llmdist_pmfs(model, tokenizer, qd, top_cids,
                                              df, topics,
                                              n_ex=8, n_samples=args.n_samples)
        log(f"  Q{qi+1}/{len(questions)} {qd['key']} LLM-dist done ({len(per_q_llm[qi])} clusters)")

    # Aggregate to question-level + compute metrics
    log("Aggregating + scoring …")
    out = {
        "config": vars(args),
        "top_clusters": [str(c) for c in top_cids],
        "per_question": {},
        "summary": {},
    }
    js_m0, js_m1, js_m2 = [], [], []
    k_m0, k_m1, k_m2 = [], [], []
    c_m0, c_m1, c_m2 = [], [], []
    for qi, qd in enumerate(questions):
        true = list(qd["true_distribution"].values())

        # M0 aggregate
        pred_m0 = agg(per_q_ssr[qi], qd["cluster_weights"], top_cids)
        # M1 aggregate
        pred_m1 = agg(per_q_llm[qi], qd["cluster_weights"], top_cids)
        # M2: ensemble at cluster level then aggregate
        m2_cpmfs = {}
        w = args.ensemble_w
        for c in top_cids:
            if c in per_q_ssr[qi] and c in per_q_llm[qi]:
                m2_cpmfs[c] = w * per_q_ssr[qi][c] + (1 - w) * per_q_llm[qi][c]
        pred_m2 = agg(m2_cpmfs, qd["cluster_weights"], top_cids)

        out["per_question"][qd["key"]] = {
            "question": qd["question"],
            "options": qd["options"],
            "true": [float(x) for x in true],
            "pred_m0": [float(x) for x in pred_m0],
            "pred_m1": [float(x) for x in pred_m1],
            "pred_m2": [float(x) for x in pred_m2],
            "m0": {"js": js_score(true, pred_m0), "k_xy": k_xy(true, pred_m0), "c_xy": c_xy(true, pred_m0)},
            "m1": {"js": js_score(true, pred_m1), "k_xy": k_xy(true, pred_m1), "c_xy": c_xy(true, pred_m1)},
            "m2": {"js": js_score(true, pred_m2), "k_xy": k_xy(true, pred_m2), "c_xy": c_xy(true, pred_m2)},
        }
        d = out["per_question"][qd["key"]]
        js_m0.append(d["m0"]["js"]); js_m1.append(d["m1"]["js"]); js_m2.append(d["m2"]["js"])
        k_m0.append(d["m0"]["k_xy"]); k_m1.append(d["m1"]["k_xy"]); k_m2.append(d["m2"]["k_xy"])
        c_m0.append(d["m0"]["c_xy"]); c_m1.append(d["m1"]["c_xy"]); c_m2.append(d["m2"]["c_xy"])
        log(f"  {qd['key']}: M0 JS={d['m0']['js']:.4f} K={d['m0']['k_xy']:.3f} C={d['m0']['c_xy']:.3f} | "
            f"M2 JS={d['m2']['js']:.4f} K={d['m2']['k_xy']:.3f} C={d['m2']['c_xy']:.3f}")

    out["summary"] = {
        "M0_hier_SSR": {"js": float(np.mean(js_m0)), "k_xy": float(np.mean(k_m0)), "c_xy": float(np.mean(c_m0))},
        "M1_LLM_vanilla": {"js": float(np.mean(js_m1)), "k_xy": float(np.mean(k_m1)), "c_xy": float(np.mean(c_m1))},
        "M2_ensemble_w0.5": {"js": float(np.mean(js_m2)), "k_xy": float(np.mean(k_m2)), "c_xy": float(np.mean(c_m2))},
    }

    log("=" * 60)
    log("SUMMARY (mean over 6Q)")
    log("=" * 60)
    for tag, s in out["summary"].items():
        log(f"  {tag:22s}  JS={s['js']:.4f}  K_xy={s['k_xy']:.3f}  C_xy={s['c_xy']:.3f}")

    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    log(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
