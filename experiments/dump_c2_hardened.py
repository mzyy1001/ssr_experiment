"""
Priority 3: Re-run C2 (per-post LLM-distribution) with a HARDENED JSON parser.

Original parser at dump_b1_c2_pmfs.py:154 used a single regex for the FIRST
[...] block. On Q5 it failed 266/500 = 53% of the time. This rewrite tries
a multi-strategy parse cascade and reports the per-strategy hit count.

Parser strategies (in order):
  1) json.loads on a {"distribution": [...]} block      (preferred)
  2) regex any [a, b, c, ...] block, exact n_opts
  3) regex extract last n_opts numeric tokens
  4) any percentage-style "选项i: x%" lines

Optionally splittable by --question_idx for parallel run across 6 GPUs.

Output: results/c2_hardened_6q.json (or per-Q shards if --question_idx given)
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


def agg(cpmfs, weights, cids):
    n = len(next(iter(cpmfs.values())))
    a, tw = np.zeros(n), 0.0
    for c in cids:
        if c in cpmfs:
            w = weights.get(c, 0); a += w * cpmfs[c]; tw += w
    return a / tw if tw > 0 else a


def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


# ─── Hardened parser cascade ─────────────────────────────────
def _normalize_arr(arr, n_opts):
    arr = np.clip(np.array(arr, float), 0, None)
    if len(arr) != n_opts or arr.sum() <= 0:
        return None
    return arr / arr.sum()


def parse_distribution_hardened(resp, n_opts):
    """Returns (pmf, strategy_name) or (None, 'fail').
    Tries strategies in order; returns the first that succeeds."""
    text = re.sub(r'<think>.*?</think>', '', resp, flags=re.DOTALL).strip()

    # Strategy 1: full JSON object {"distribution": [...]}
    for m in re.finditer(r'\{[^{}]*"distribution"[^{}]*\}', text, flags=re.DOTALL):
        try:
            obj = json.loads(m.group(0))
            arr = obj.get("distribution")
            if arr is not None:
                pmf = _normalize_arr(arr, n_opts)
                if pmf is not None:
                    return pmf, "s1_json_obj"
        except Exception:
            continue

    # Strategy 2: any bare [a, b, c, ...] with exact n_opts numeric entries
    for m in re.finditer(r'\[([^\[\]]+)\]', text):
        body = m.group(1)
        nums = re.findall(r'-?\d+(?:\.\d+)?', body)
        if len(nums) == n_opts:
            pmf = _normalize_arr([float(x) for x in nums], n_opts)
            if pmf is not None:
                return pmf, "s2_bracket_exact"

    # Strategy 3: last n_opts numeric tokens anywhere
    nums = re.findall(r'-?\d*\.\d+|-?\d+', text)
    if len(nums) >= n_opts:
        pmf = _normalize_arr([float(x) for x in nums[-n_opts:]], n_opts)
        if pmf is not None:
            return pmf, "s3_last_n_nums"

    # Strategy 4: percentage-style mentions like "选项1: 30%" or "1. 30%"
    pct_lines = re.findall(r'(?:选项)?\s*(\d+)[.:：]\s*(\d+(?:\.\d+)?)\s*%', text)
    if pct_lines:
        vals = [0.0] * n_opts
        for idx_str, pct_str in pct_lines:
            try:
                idx = int(idx_str) - 1
                if 0 <= idx < n_opts:
                    vals[idx] = float(pct_str)
            except Exception:
                continue
        pmf = _normalize_arr(vals, n_opts)
        if pmf is not None:
            return pmf, "s4_percent_lines"

    return None, "fail"


PER_POST_PROMPT = """根据以下社交媒体帖子内容，估计这位消费者在以下问卷问题中各选项的选择比例。

帖子：{post}

问题：{question}
选项：
{options_numbered}

请严格按JSON格式输出各选项的比例（总和为1，0到1的小数）：
{{"distribution": [选项1比例, 选项2比例, ...]}}

JSON："""


def compute_c2_pmfs_hardened(model, tokenizer, qd, cids, df,
                              n_posts=50, seed=42, max_new=120):
    opts = qd["options"]; n_opts = len(opts)
    opts_str = "\n".join(f"{i+1}. {o}" for i, o in enumerate(opts))
    cpmfs = {}
    parse_stats = {"s1_json_obj": 0, "s2_bracket_exact": 0,
                   "s3_last_n_nums": 0, "s4_percent_lines": 0, "fail": 0}
    n_total = 0
    for cid in cids:
        posts = df[df["cluster_label"] == int(cid)]["content_desc"].dropna()
        if len(posts) < 3: continue
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
            pmf, strategy = parse_distribution_hardened(resp, n_opts)
            parse_stats[strategy] += 1
            if pmf is None:
                pmf = np.ones(n_opts) / n_opts
            pmfs.append(pmf)
        cpmfs[cid] = np.mean(pmfs, axis=0)
    return cpmfs, parse_stats, n_total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--n_posts", type=int, default=50)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--question_idx", type=int, default=-1,
                   help="-1 = all 6 questions; 0-5 = single question (for parallel run)")
    p.add_argument("--output", default=f"{RESULTS_DIR}/c2_hardened_6q.json")
    args = p.parse_args()

    log(f"=== C2 hardened, qidx={args.question_idx} on {args.device} ===")
    questions, df = load_orig6()
    top_cids = get_top_clusters(questions, n=args.top_k)

    if args.question_idx >= 0:
        questions = [questions[args.question_idx]]
        log(f"Restricted to question {args.question_idx}: {questions[0]['key']}")

    log("Loading model + encoder …")
    encoder = SentenceTransformer(EMBED_PATH, device=args.device)
    model, tokenizer = load_model(MODEL_PATH, args.device)

    log("Generating anchors (fixed seed=42) …")
    set_seed(42)
    anchors = {}
    for qd in questions:
        anc = generate_anchors_local(model, tokenizer, qd["question"], qd["options"])
        anchors[qd["key"]] = [encoder.encode(a) for a in anc]

    out = {"config": vars(args), "top_clusters": [str(c) for c in top_cids],
           "per_question": {}, "parse_stats": {}}
    set_seed(args.seed)

    for qi, qd in enumerate(questions):
        t0 = time.time()
        cpmfs, stats, ntot = compute_c2_pmfs_hardened(
            model, tokenizer, qd, top_cids, df, n_posts=args.n_posts)
        pred = agg(cpmfs, qd["cluster_weights"], top_cids)
        true = list(qd["true_distribution"].values())
        js = js_score(true, pred); k = k_xy(true, pred); c = c_xy(true, pred)
        n_fail = stats["fail"]
        log(f"  Q{qi+1}/{len(questions)} {qd['key']}: JS={js:.4f} K={k:.3f} C={c:.3f}"
            f" | fails={n_fail}/{ntot} ({100*n_fail/ntot:.1f}%) elapsed={time.time()-t0:.0f}s")
        log(f"    parse strategies: {stats}")
        out["per_question"][qd["key"]] = {
            "question": qd["question"], "options": qd["options"],
            "true": [float(x) for x in true],
            "pred_c2_hardened": [float(x) for x in pred],
            "c2_hardened": {"js": js, "k_xy": k, "c_xy": c},
            "parse_stats": stats, "n_total": ntot,
        }
        out["parse_stats"][qd["key"]] = stats

    if args.question_idx < 0:
        # All-6Q summary
        js_all = [v["c2_hardened"]["js"] for v in out["per_question"].values()]
        k_all = [v["c2_hardened"]["k_xy"] for v in out["per_question"].values()]
        c_all = [v["c2_hardened"]["c_xy"] for v in out["per_question"].values()]
        out["summary"] = {
            "C2_hardened": {
                "js": float(np.mean(js_all)),
                "k_xy": float(np.mean(k_all)),
                "c_xy": float(np.mean(c_all)),
            }
        }
        log(f"C2_hardened mean JS={out['summary']['C2_hardened']['js']:.4f}")

    out_path = args.output
    if args.question_idx >= 0:
        base, ext = os.path.splitext(args.output)
        out_path = f"{base}_q{args.question_idx}{ext}"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    log(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
