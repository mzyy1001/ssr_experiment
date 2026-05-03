"""
Fill the two sparse cells of the 2x2 (per-post/cluster × SSR/LLM-vote):

  B1 — per-cluster CENTROID SSR
       for each cluster k:
         centroid_k = mean(bge.encode(p) for p in posts_k)
         pmf_k      = SSR(centroid_k, anchors)         # softmax(temp=0.1)
       aggregate via cluster_weights → final PMF
       (cf. M0 which does SSR-then-mean; B1 does mean-then-SSR)

  C2 — per-post LLM DISTRIBUTION (soft analog of Method A)
       for each cluster k, for each post p in cluster k:
         prompt LLM with post + question + options
         LLM outputs JSON distribution over options for THIS post
       average PMFs within cluster → cluster PMF
       aggregate via cluster_weights → final PMF
       (cf. Method A which uses hard-vote per post; C2 uses soft PMF per post)

Output: results/b1_c2_pmfs_6q.json
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

from persona_vectors import load_model
from steered_ssr import generate_anchors_local


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
    return questions, df


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


def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def ssr_softmax(text_or_emb, anchor_embeddings, temp=0.1, is_embedding=False):
    """Softmax-temp SSR. Identical to steered_ssr.ssr_score(normalization='softmax')."""
    v = text_or_emb if is_embedding else None
    sims = np.array([cosine_sim(v, a) for a in anchor_embeddings], dtype=float)
    exp_sims = np.exp((sims - sims.max()) / temp)
    pmf = exp_sims / exp_sims.sum()
    return pmf


# ─── B1: per-cluster centroid SSR ──────────────────────────────
def compute_b1_pmfs(encoder, qd, cids, df, aembs, n_posts=50, seed=42):
    cpmfs = {}
    for cid in cids:
        posts = df[df["cluster_label"] == int(cid)]["content_desc"].dropna()
        if len(posts) < 3:
            continue
        samp = posts.sample(min(n_posts, len(posts)), random_state=seed).tolist()
        # Embed each post, then MEAN-POOL into a centroid
        post_embs = encoder.encode(samp, batch_size=32, show_progress_bar=False)
        centroid = np.mean(post_embs, axis=0)
        # SSR-score the centroid (one PMF per cluster)
        cpmfs[cid] = ssr_softmax(centroid, aembs, is_embedding=True)
    return cpmfs


# ─── C2: per-post LLM distribution ─────────────────────────────
PER_POST_PROMPT = """根据以下社交媒体帖子内容，估计这位消费者在以下问卷问题中各选项的选择比例。

帖子：{post}

问题：{question}
选项：
{options_numbered}

请严格按JSON格式输出各选项的比例（总和为1，0到1的小数）：
{{"distribution": [选项1比例, 选项2比例, ...]}}

JSON："""


def parse_json_dist(resp, n_opts):
    """Robust parse: try JSON, then bracketed list, else None."""
    resp = re.sub(r'<think>.*?</think>', '', resp, flags=re.DOTALL).strip()
    # Look for [...] inside
    m = re.search(r'\[([^\]]+)\]', resp)
    if m:
        try:
            vals = [float(x.strip()) for x in m.group(1).split(',')]
            if len(vals) == n_opts:
                arr = np.clip(np.array(vals), 0, None)
                if arr.sum() > 0:
                    return arr / arr.sum()
        except Exception:
            pass
    return None


def compute_c2_pmfs(model, tokenizer, qd, cids, df, n_posts=50, seed=42, max_new=120):
    opts = qd["options"]
    n_opts = len(opts)
    opts_str = "\n".join(f"{i+1}. {o}" for i, o in enumerate(opts))
    cpmfs = {}
    n_parse_fail_total = 0
    n_total = 0
    for cid in cids:
        posts = df[df["cluster_label"] == int(cid)]["content_desc"].dropna()
        if len(posts) < 3:
            continue
        samp = posts.sample(min(n_posts, len(posts)), random_state=seed).tolist()
        pmfs = []
        for post in samp:
            n_total += 1
            prompt = PER_POST_PROMPT.format(
                post=post[:200], question=qd["question"], options_numbered=opts_str)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=max_new,
                                     do_sample=True, temperature=0.7, top_p=0.9)
            resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                    skip_special_tokens=True)
            pmf = parse_json_dist(resp, n_opts)
            if pmf is None:
                pmf = np.ones(n_opts) / n_opts
                n_parse_fail_total += 1
            pmfs.append(pmf)
        cpmfs[cid] = np.mean(pmfs, axis=0)
    return cpmfs, n_parse_fail_total, n_total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:2")
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--n_posts", type=int, default=50)
    p.add_argument("--output", default=f"{RESULTS_DIR}/b1_c2_pmfs_6q.json")
    p.add_argument("--skip_c2", action="store_true",
                   help="Skip the slow C2 (LLM-per-post) experiment for a quick B1-only run")
    args = p.parse_args()

    log("Loading data …")
    questions, df = load_orig6()
    top_cids = get_top_clusters(questions, n=args.top_k)
    log(f"Top {args.top_k} clusters: {top_cids}")

    log("Loading model + encoder …")
    encoder = SentenceTransformer(EMBED_PATH)
    model, tokenizer = load_model(MODEL_PATH, args.device)
    log("Model loaded")

    log("Generating anchors (deterministic, seeded 42 — matches M0) …")
    set_seed(42)
    anchors = {}
    for qd in questions:
        anc = generate_anchors_local(model, tokenizer, qd["question"], qd["options"])
        anchors[qd["key"]] = [encoder.encode(a) for a in anc]
        log(f"  {qd['key']}: {anc[0][:40]}…")

    out = {
        "config": vars(args),
        "top_clusters": [str(c) for c in top_cids],
        "per_question": {},
        "summary": {},
    }

    # ───── B1 ─────
    log("=== B1: per-cluster centroid SSR ===")
    per_q_b1 = {}
    for qi, qd in enumerate(questions):
        per_q_b1[qi] = compute_b1_pmfs(encoder, qd, top_cids, df,
                                        anchors[qd["key"]], n_posts=args.n_posts)
        log(f"  Q{qi+1}/{len(questions)} {qd['key']} B1 done ({len(per_q_b1[qi])} clusters)")

    # ───── C2 ─────
    per_q_c2 = None
    parse_stats = []
    if not args.skip_c2:
        log("=== C2: per-post LLM distribution ===")
        set_seed(2026)
        per_q_c2 = {}
        for qi, qd in enumerate(questions):
            t0 = time.time()
            per_q_c2[qi], nfail, ntot = compute_c2_pmfs(
                model, tokenizer, qd, top_cids, df,
                n_posts=args.n_posts)
            parse_stats.append((qd["key"], nfail, ntot))
            log(f"  Q{qi+1}/{len(questions)} {qd['key']} C2 done "
                f"({len(per_q_c2[qi])} clusters, {nfail}/{ntot} parse-fails, "
                f"elapsed={time.time()-t0:.0f}s)")

    # ───── Aggregate + score ─────
    log("Aggregating + scoring …")
    js_b1, k_b1, c_b1 = [], [], []
    js_c2, k_c2, c_c2 = [], [], []
    for qi, qd in enumerate(questions):
        true = list(qd["true_distribution"].values())

        pred_b1 = agg(per_q_b1[qi], qd["cluster_weights"], top_cids)
        b1d = {"js": js_score(true, pred_b1),
               "k_xy": k_xy(true, pred_b1),
               "c_xy": c_xy(true, pred_b1)}
        js_b1.append(b1d["js"]); k_b1.append(b1d["k_xy"]); c_b1.append(b1d["c_xy"])

        entry = {
            "question": qd["question"],
            "options": qd["options"],
            "true": [float(x) for x in true],
            "pred_b1": [float(x) for x in pred_b1],
            "b1": b1d,
        }
        log(f"  {qd['key']}: B1 JS={b1d['js']:.4f} K={b1d['k_xy']:.3f} C={b1d['c_xy']:.3f}", )

        if per_q_c2 is not None:
            pred_c2 = agg(per_q_c2[qi], qd["cluster_weights"], top_cids)
            c2d = {"js": js_score(true, pred_c2),
                   "k_xy": k_xy(true, pred_c2),
                   "c_xy": c_xy(true, pred_c2)}
            js_c2.append(c2d["js"]); k_c2.append(c2d["k_xy"]); c_c2.append(c2d["c_xy"])
            entry["pred_c2"] = [float(x) for x in pred_c2]
            entry["c2"] = c2d
            log(f"             C2 JS={c2d['js']:.4f} K={c2d['k_xy']:.3f} C={c2d['c_xy']:.3f}")

        out["per_question"][qd["key"]] = entry

    out["summary"]["B1_centroid_SSR"] = {
        "js": float(np.mean(js_b1)),
        "k_xy": float(np.mean(k_b1)),
        "c_xy": float(np.mean(c_b1)),
    }
    if per_q_c2 is not None:
        out["summary"]["C2_per_post_LLMdist"] = {
            "js": float(np.mean(js_c2)),
            "k_xy": float(np.mean(k_c2)),
            "c_xy": float(np.mean(c_c2)),
        }
        out["c2_parse_stats"] = [{"q": q, "n_fail": f, "n_total": t}
                                  for q, f, t in parse_stats]

    log("=" * 60)
    log("SUMMARY (mean over 6Q)")
    log("=" * 60)
    for tag, s in out["summary"].items():
        log(f"  {tag:24s}  JS={s['js']:.4f}  K_xy={s['k_xy']:.3f}  C_xy={s['c_xy']:.3f}")

    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    log(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
