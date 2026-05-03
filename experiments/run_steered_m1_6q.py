"""
Steered M1 experiment on the 6Q real benchmark.

Design: same prompt as llm_dist_est (LLM reads sample posts + question, emits
JSON distribution over options), but with SteeringHook attached at layers
[16, 20, 24] using the cluster's persona vector during generation.

Compares:
  M0 — pure SSR baseline (reuse, fast)
  M1 vanilla — LLM-dist (n_samp=3), no steering
  M1 steered α=0.1 — steered LLM-dist
  M1 steered α=0.3
  M1 steered α=0.5
  M2 variants — SSR + (steered/vanilla) LLM-dist ensemble, weight sweep

Output: results/steered_m1_6q.json
"""
import argparse
import json
import re
import sys
import time
import random

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import jensenshannon
from sentence_transformers import SentenceTransformer

from persona_vectors import load_model, load_persona_vectors
from steered_ssr import SteeringHook, ssr_score, generate_anchors_local


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


def js_score(td, pred):
    t = np.array(list(td.values()), dtype=float); t /= t.sum()
    p = pred / (pred.sum() + 1e-10)
    return float(jensenshannon(t, p) ** 2)


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


def compute_llmdist_pmfs(model, tokenizer, qd, cids, df, topics, pvecs,
                         alpha=0.0, layers=(16, 20, 24),
                         n_ex=8, n_samples=3):
    """LLM distribution estimation per cluster.
    If alpha > 0, attach SteeringHooks on the given layers with each cluster's
    persona vector during generation.
    """
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

        # Prepare hooks if steering requested
        hooks = []
        if alpha > 0 and int(cid) in pvecs:
            vec = pvecs[int(cid)]["vector"]
            hooks = [SteeringHook(vec, alpha, l) for l in layers]

        pmfs_collected = []
        n_parse_fail = 0
        for _ in range(n_samples):
            for h in hooks: h.attach(model)
            try:
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                                  max_length=1024)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=200,
                                        do_sample=True, temperature=0.7, top_p=0.9)
                resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                       skip_special_tokens=True)
            finally:
                for h in hooks: h.remove()

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
                n_parse_fail += 1
        cpmfs[cid] = np.mean(pmfs_collected, axis=0)
    return cpmfs


def eval_method(per_q_pmfs, questions, top_cids):
    js_vals = []
    for qi, qd in enumerate(questions):
        pred = agg(per_q_pmfs[qi], qd["cluster_weights"], top_cids)
        js_vals.append(js_score(qd["true_distribution"], pred))
    return js_vals


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:2")
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--n_posts", type=int, default=50)
    p.add_argument("--n_samples", type=int, default=3)
    p.add_argument("--alphas", nargs="+", type=float,
                   default=[0.0, 0.05, 0.1, 0.2, 0.3])
    p.add_argument("--output", default=f"{RESULTS_DIR}/steered_m1_6q.json")
    args = p.parse_args()

    log("Loading data …")
    questions, topics, df = load_orig6()
    top_cids = get_top_clusters(questions, n=args.top_k)
    log(f"Top {args.top_k} clusters: {top_cids}")

    log("Loading model + encoder …")
    encoder = SentenceTransformer(EMBED_PATH)
    model, tokenizer = load_model(MODEL_PATH, args.device)
    pvecs = load_persona_vectors(f"{RESULTS_DIR}/persona_vectors_L24_N20.npz")
    log(f"Loaded persona vectors for {len(pvecs)} clusters")

    log("Generating anchors …")
    anchors = {}
    for qd in questions:
        anc = generate_anchors_local(model, tokenizer, qd["question"], qd["options"])
        anchors[qd["key"]] = [encoder.encode(a) for a in anc]
        log(f"  {qd['key']}: {anc[0][:40]}…")

    # SSR (deterministic)
    log("Computing SSR per-cluster (n=50) …")
    per_q_ssr = {}
    for qi, qd in enumerate(questions):
        per_q_ssr[qi] = compute_ssr_pmfs(encoder, qd, top_cids, df,
                                           anchors[qd["key"]], n_posts=args.n_posts)
    ssr_js = eval_method(per_q_ssr, questions, top_cids)
    log(f"  M0 SSR mean JS = {np.mean(ssr_js):.4f}")

    output = {
        "config": vars(args),
        "top_clusters": [str(c) for c in top_cids],
        "m0_ssr_js": ssr_js,
        "alpha_results": {},
        "weight_sweep": {},
    }

    # Compute LLM-dist (steered) for each alpha
    for alpha in args.alphas:
        set_seed(2026)
        tag = "vanilla" if alpha == 0.0 else f"alpha={alpha}"
        log(f"\n=== LLM-dist ({tag}) ===")
        per_q_llm = {}
        for qi, qd in enumerate(questions):
            per_q_llm[qi] = compute_llmdist_pmfs(
                model, tokenizer, qd, top_cids, df, topics, pvecs,
                alpha=alpha, layers=(16, 20, 24),
                n_ex=8, n_samples=args.n_samples
            )
            log(f"  Q{qi+1}/{len(questions)} done")

        # M1 alone
        m1_js = eval_method(per_q_llm, questions, top_cids)
        log(f"  M1 ({tag}): mean JS = {np.mean(m1_js):.4f} | "
            f"{' '.join(f'{v:.4f}' for v in m1_js)}")

        # M2 (SSR + steered LLM-dist) weight sweep
        weight_rows = {}
        for w in [0.3, 0.4, 0.5, 0.6, 0.7]:
            per_q_m2 = {}
            for qi in range(len(questions)):
                per_q_m2[qi] = {}
                for c in top_cids:
                    if c in per_q_ssr[qi] and c in per_q_llm[qi]:
                        per_q_m2[qi][c] = w * per_q_ssr[qi][c] + (1 - w) * per_q_llm[qi][c]
            m2_js = eval_method(per_q_m2, questions, top_cids)
            weight_rows[f"w={w}"] = m2_js
            log(f"    M2 w={w}: {np.mean(m2_js):.4f}")

        output["alpha_results"][tag] = {
            "m1_js": m1_js,
            "m1_mean": float(np.mean(m1_js)),
        }
        output["weight_sweep"][tag] = {k: [float(v) for v in vals]
                                         for k, vals in weight_rows.items()}

        # Save incrementally
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

    # Summary
    log("\n" + "=" * 70)
    log("SUMMARY")
    log("=" * 70)
    log(f"SSR baseline (M0): {np.mean(ssr_js):.4f}")
    log("")
    log(f"{'Config':<18} {'M1 alone':>12} {'M2 best w':>14} {'M2 best JS':>12}")
    log("-" * 60)
    for alpha in args.alphas:
        tag = "vanilla" if alpha == 0.0 else f"alpha={alpha}"
        m1 = output["alpha_results"][tag]["m1_mean"]
        best_w = min(output["weight_sweep"][tag],
                      key=lambda k: np.mean(output["weight_sweep"][tag][k]))
        best_js = np.mean(output["weight_sweep"][tag][best_w])
        log(f"{tag:<18} {m1:>12.4f} {best_w:>14} {best_js:>12.4f}")

    log(f"\nSaved → {args.output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        log(f"ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)
