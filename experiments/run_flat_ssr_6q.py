"""
Priority 2: Same-data flat (no-cluster) SSR baseline.

Pools the SAME posts that M0 sees (posts from the top-10 clusters used by
M0's pipeline) and computes SSR PMF per post, then averages flatly across
all posts — NO cluster_weights re-weighting. Isolates "do clusters matter,
or do real posts alone explain the gain over Maier paper-SSR transfer?"

Comparison ladder:
- Paper-SSR transfer (Maier): per-LLM-consumer flat (existing).
- Flat SSR (this script): real posts, per-post SSR, flat mean across posts.
- M0 hier cluster-weighted SSR (existing): per-cluster mean → cluster-mass aggregate.
- M2 ensemble (existing): cluster-level 0.5 SSR + 0.5 LLM-dist.

Output: results/flat_ssr_6q.json
"""
import argparse
import json
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--top_k", type=int, default=10,
                   help="Number of top clusters to draw posts from "
                        "(matches M0 setup; flat-pooling is across these clusters)")
    p.add_argument("--n_posts", type=int, default=50,
                   help="Posts per cluster, then pooled flat (matches M0 n_posts)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default=f"{RESULTS_DIR}/flat_ssr_6q.json")
    args = p.parse_args()

    set_seed(args.seed)
    log("Loading data …")
    questions, df = load_orig6()
    top_cids = get_top_clusters(questions, n=args.top_k)
    log(f"Top {args.top_k} clusters (drawing posts from): {top_cids}")

    log("Loading model + encoder …")
    encoder = SentenceTransformer(EMBED_PATH, device=args.device)
    model, tokenizer = load_model(MODEL_PATH, args.device)

    log("Generating anchors (fixed seed=42; matches M0) …")
    set_seed(42)
    anchors = {}
    for qd in questions:
        anc = generate_anchors_local(model, tokenizer, qd["question"], qd["options"])
        anchors[qd["key"]] = [encoder.encode(a) for a in anc]
        log(f"  {qd['key']}: anchor[0]={anc[0][:40]}…")

    out = {"config": vars(args), "top_clusters": [str(c) for c in top_cids],
           "per_question": {}, "summary": {}}
    js_all, k_all, c_all = [], [], []

    for qi, qd in enumerate(questions):
        # FLAT pool: take n_posts from each of top_k clusters, concatenate,
        # compute SSR PMF per post, mean WITHOUT cluster-mass weighting.
        all_posts = []
        for cid in top_cids:
            posts = df[df["cluster_label"] == int(cid)]["content_desc"].dropna()
            if len(posts) < 3:
                continue
            samp = posts.sample(min(args.n_posts, len(posts)), random_state=args.seed).tolist()
            all_posts.extend(samp)
        # Per-post SSR, flat mean
        pmfs = [ssr_score(p, encoder, anchors[qd["key"]])[0] for p in all_posts]
        pred = np.mean(pmfs, axis=0)
        true = list(qd["true_distribution"].values())
        js = js_score(true, pred); k = k_xy(true, pred); c = c_xy(true, pred)
        js_all.append(js); k_all.append(k); c_all.append(c)
        out["per_question"][qd["key"]] = {
            "question": qd["question"], "options": qd["options"],
            "true": [float(x) for x in true],
            "pred_flat": [float(x) for x in pred],
            "n_posts_total": len(all_posts),
            "flat": {"js": js, "k_xy": k, "c_xy": c},
        }
        log(f"  {qd['key']}: n_posts={len(all_posts):4d} JS={js:.4f} K={k:.3f} C={c:.3f}")

    out["summary"] = {
        "FlatSSR_no_cluster": {
            "js": float(np.mean(js_all)),
            "k_xy": float(np.mean(k_all)),
            "c_xy": float(np.mean(c_all)),
        }
    }
    log("=" * 60)
    log(f"FlatSSR_no_cluster: JS={out['summary']['FlatSSR_no_cluster']['js']:.4f}  "
        f"K={out['summary']['FlatSSR_no_cluster']['k_xy']:.3f}  "
        f"C={out['summary']['FlatSSR_no_cluster']['c_xy']:.3f}")
    log("Reference (existing): Paper-SSR=0.0430, M0=0.0269, M2=0.0254")

    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    log(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
