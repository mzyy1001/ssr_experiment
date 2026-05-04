"""
Unified Table-1 evaluator: runs all 6 cells of the position-paper main
comparison (semantic-mapping × direct-LLM, no-social × post-level × structured)
on a configurable question set, with one model load. One seed per invocation;
seed-sweep by running multiple times.

Cells:
  paper_ssr   — Maier-style: LLM-as-consumer free text → SSR mean      (no social)
  direct_llm  — direct LLM hard voting per question                     (no social)
  flat_ssr    — real posts, no clusters, per-post SSR, flat mean        (post-level SSR)
  c2_hardened — per-post LLM JSON distribution, hardened parser         (post-level LLM)
  m0          — cluster SSR per-post, mass-weighted aggregate           (structured SSR)
  m1          — cluster LLM-distribution prompt, mass-weighted aggregate (structured LLM)

Output: results/table1/seed{S}_<question_set_tag>.json
"""
import argparse
import json
import os
import re
import time
import random
from collections import Counter

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


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


# ─── metrics ────────────────────────────────────────────────────────────
def _norm(t):
    t = np.array(t, float); return t / (t.sum() + 1e-10)


def js(true, pred):
    return float(jensenshannon(_norm(true), _norm(pred)) ** 2)


def kxy(true, pred):
    return 1 - float(np.max(np.abs(np.cumsum(_norm(true)) - np.cumsum(_norm(pred)))))


def cxy(true, pred):
    a, b = _norm(true), _norm(pred)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def metrics(pred, true):
    return {"js": js(true, pred), "k_xy": kxy(true, pred), "c_xy": cxy(true, pred)}


# ─── question loading ───────────────────────────────────────────────────
def load_questions(path, keys=None):
    """Loads from a nested {survey:{sub:{qid: {question, options, true_distribution, cluster_weights}}}} JSON."""
    with open(path) as f:
        d = json.load(f)
    out = []
    for sk, sv in d.items():
        for sub, qs in sv.items():
            for qid, qd in qs.items():
                key = f"{sub}/{qid}"
                if keys is not None and key not in keys:
                    continue
                if "options" not in qd or "true_distribution" not in qd:
                    continue
                out.append({
                    "key": key, "question": qd["question"], "options": qd["options"],
                    "true_distribution": qd["true_distribution"],
                    "cluster_weights": qd.get("cluster_weights", {}),
                })
    return out


def get_top_clusters(questions, n=10):
    agg = {}
    for q in questions:
        for cid, w in q["cluster_weights"].items():
            agg[cid] = agg.get(cid, 0) + w
    return sorted(agg, key=lambda c: -agg[c])[:n]


def agg_pmf(cpmfs, weights, cids):
    n = len(next(iter(cpmfs.values())))
    a, tw = np.zeros(n), 0.0
    for c in cids:
        if c in cpmfs:
            w = weights.get(c, 0); a += w * cpmfs[c]; tw += w
    return a / tw if tw > 0 else a


def strip_think(text):
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


def llm_gen(model, tokenizer, prompt, max_new=200, temperature=0.7, top_p=0.9, do_sample=True):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new,
                              do_sample=do_sample, temperature=temperature, top_p=top_p)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


# ─── Hardened parser cascade ────────────────────────────────────────────
def _norm_arr(arr, n_opts):
    arr = np.clip(np.array(arr, float), 0, None)
    if len(arr) != n_opts or arr.sum() <= 0:
        return None
    return arr / arr.sum()


def parse_dist(resp, n_opts):
    text = strip_think(resp)
    for m in re.finditer(r'\{[^{}]*"distribution"[^{}]*\}', text, flags=re.DOTALL):
        try:
            arr = json.loads(m.group(0)).get("distribution")
            if arr is not None:
                pmf = _norm_arr(arr, n_opts)
                if pmf is not None: return pmf, "s1"
        except Exception: continue
    for m in re.finditer(r'\[([^\[\]]+)\]', text):
        nums = re.findall(r'-?\d+(?:\.\d+)?', m.group(1))
        if len(nums) == n_opts:
            pmf = _norm_arr([float(x) for x in nums], n_opts)
            if pmf is not None: return pmf, "s2"
    nums = re.findall(r'-?\d*\.\d+|-?\d+', text)
    if len(nums) >= n_opts:
        pmf = _norm_arr([float(x) for x in nums[-n_opts:]], n_opts)
        if pmf is not None: return pmf, "s3"
    pct = re.findall(r'(?:选项)?\s*(\d+)[.:：]\s*(\d+(?:\.\d+)?)\s*%', text)
    if pct:
        vals = [0.0] * n_opts
        for idx_str, p in pct:
            try:
                idx = int(idx_str) - 1
                if 0 <= idx < n_opts: vals[idx] = float(p)
            except Exception: continue
        pmf = _norm_arr(vals, n_opts)
        if pmf is not None: return pmf, "s4"
    return None, "fail"


# ─── Method: paper-SSR (Maier-style) ────────────────────────────────────
PAPER_SSR_SYSTEM = (
    "你是一位中国消费者，可能使用过藿香正气水。"
    "请根据自己的真实想法用自然的中文回答问卷问题（2-3句话，直接表达观点，不要列选项）。"
)


def run_paper_ssr(model, tokenizer, encoder, qd, anchor_embs, n_samples=100, temperature=0.7):
    pmfs = []
    msgs_template = [
        {"role": "system", "content": PAPER_SSR_SYSTEM},
        {"role": "user", "content": ""},
    ]
    user = f"问题：{qd['question']}\n可能的选项：{'、'.join(qd['options'])}\n\n你的回答（用自然语言）："
    msgs_template[1]["content"] = user
    try:
        prompt = tokenizer.apply_chat_template(msgs_template, tokenize=False, add_generation_prompt=True)
    except Exception:
        prompt = f"{PAPER_SSR_SYSTEM}\n\n{user}"
    for _ in range(n_samples):
        resp = strip_think(llm_gen(model, tokenizer, prompt, max_new=120, temperature=temperature))
        if not resp:
            resp = "没意见"
        pmf, _ = ssr_score(resp, encoder, anchor_embs, normalization="softmax")
        pmfs.append(pmf)
    return np.mean(pmfs, axis=0)


# ─── Method: direct-LLM (hard voting) ───────────────────────────────────
DIRECT_PROMPT = "你是一位普通消费者。请直接回答以下问卷问题，只输出选项内容。\n\n问题：{q}\n选项：{o}\n\n你的选择："


def run_direct_llm(model, tokenizer, qd, n_samples=100, temperature=0.8):
    counts = Counter()
    prompt = DIRECT_PROMPT.format(q=qd["question"], o="、".join(qd["options"]))
    for _ in range(n_samples):
        resp = strip_think(llm_gen(model, tokenizer, prompt, max_new=50,
                                    temperature=temperature, top_p=0.9))
        # match longest option substring
        best, score = None, -1
        for opt in qd["options"]:
            if opt in resp and len(opt) > score:
                best, score = opt, len(opt)
        if best is None:
            best = qd["options"][0]
        counts[best] += 1
    return np.array([counts.get(o, 0) for o in qd["options"]], dtype=float) / max(1, sum(counts.values()))


# ─── Method: flat-SSR (real posts, no clusters) ─────────────────────────
def run_flat_ssr(encoder, qd, anchor_embs, posts_df, top_cids, n_posts=50, sample_seed=42):
    all_posts = []
    for cid in top_cids:
        ps = posts_df[posts_df["cluster_label"] == int(cid)]["content_desc"].dropna()
        if len(ps) < 3: continue
        all_posts.extend(ps.sample(min(n_posts, len(ps)), random_state=sample_seed).tolist())
    if not all_posts:
        return np.ones(len(qd["options"])) / len(qd["options"])
    pmfs = [ssr_score(p, encoder, anchor_embs, normalization="softmax")[0] for p in all_posts]
    return np.mean(pmfs, axis=0)


# ─── Methods: M0 (cluster SSR) + M1 (cluster LLM-dist) ──────────────────
def run_m0_pmfs(encoder, qd, top_cids, posts_df, anchor_embs, n_posts=50, sample_seed=42):
    cpmfs = {}
    for cid in top_cids:
        ps = posts_df[posts_df["cluster_label"] == int(cid)]["content_desc"].dropna()
        if len(ps) < 3: continue
        samp = ps.sample(min(n_posts, len(ps)), random_state=sample_seed).tolist()
        pmfs = [ssr_score(p, encoder, anchor_embs, normalization="softmax")[0] for p in samp]
        cpmfs[cid] = np.mean(pmfs, axis=0)
    return cpmfs


M1_PROMPT = (
    "你是一位消费者调研分析专家。\n\n"
    "以下是某类消费者群体的社交媒体帖子样本（话题：{topic}）：\n{posts}\n\n"
    "基于这些帖子反映的消费者特征和态度，请估计这个群体在以下问卷问题中各选项的选择比例。\n\n"
    "问题：{question}\n选项：\n{options_numbered}\n\n"
    "请严格按JSON格式输出各选项的比例（总和为1）：\n"
    '{{"distribution": [选项1比例, 选项2比例, ...]}}'
)


def run_m1_pmfs(model, tokenizer, qd, top_cids, posts_df, topics, n_ex=8, n_samples=3, sample_seed=42):
    n_opts = len(qd["options"])
    opts_str = "\n".join(f"{i+1}. {o}" for i, o in enumerate(qd["options"]))
    cpmfs = {}
    strats = Counter()
    for cid in top_cids:
        ps = posts_df[posts_df["cluster_label"] == int(cid)]["content_desc"].dropna()
        if len(ps) < 3: continue
        samp = ps.sample(min(n_ex, len(ps)), random_state=sample_seed).tolist()
        post_text = "\n".join(f"- {p[:100]}" for p in samp)
        topic = (topics.get(cid, "未知") or "未知")[:60]
        prompt = M1_PROMPT.format(topic=topic, posts=post_text,
                                    question=qd["question"], options_numbered=opts_str)
        pmfs = []
        for _ in range(n_samples):
            resp = llm_gen(model, tokenizer, prompt, max_new=200)
            pmf, strat = parse_dist(resp, n_opts)
            strats[strat] += 1
            if pmf is None:
                pmf = np.ones(n_opts) / n_opts
            pmfs.append(pmf)
        cpmfs[cid] = np.mean(pmfs, axis=0)
    return cpmfs, dict(strats)


# ─── Method: C2 hardened (per-post LLM-dist) ────────────────────────────
C2_PROMPT = (
    "根据以下社交媒体帖子内容，估计这位消费者在以下问卷问题中各选项的选择比例。\n\n"
    "帖子：{post}\n\n"
    "问题：{question}\n选项：\n{options_numbered}\n\n"
    "请严格按JSON格式输出各选项的比例（总和为1，0到1的小数）：\n"
    '{{"distribution": [选项1比例, 选项2比例, ...]}}\n\nJSON：'
)


def run_c2_pmfs(model, tokenizer, qd, top_cids, posts_df, n_posts=50, sample_seed=42, max_new=120):
    n_opts = len(qd["options"])
    opts_str = "\n".join(f"{i+1}. {o}" for i, o in enumerate(qd["options"]))
    cpmfs = {}
    strats = Counter()
    n_total = 0
    for cid in top_cids:
        ps = posts_df[posts_df["cluster_label"] == int(cid)]["content_desc"].dropna()
        if len(ps) < 3: continue
        samp = ps.sample(min(n_posts, len(ps)), random_state=sample_seed).tolist()
        pmfs = []
        for post in samp:
            n_total += 1
            prompt = C2_PROMPT.format(post=post[:200],
                                       question=qd["question"], options_numbered=opts_str)
            resp = llm_gen(model, tokenizer, prompt, max_new=max_new)
            pmf, strat = parse_dist(resp, n_opts)
            strats[strat] += 1
            if pmf is None:
                pmf = np.ones(n_opts) / n_opts
            pmfs.append(pmf)
        cpmfs[cid] = np.mean(pmfs, axis=0)
    return cpmfs, dict(strats), n_total


# ─── Main ───────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--questions", default=f"{RESULTS_DIR}/questions_expanded_13q.json")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--n_posts_cluster", type=int, default=50)
    p.add_argument("--n_posts_c2", type=int, default=50)
    p.add_argument("--n_samples_m1", type=int, default=3)
    p.add_argument("--n_samples_paper", type=int, default=100)
    p.add_argument("--n_samples_direct", type=int, default=100)
    p.add_argument("--methods", default="paper_ssr,direct_llm,flat_ssr,m0,m1,c2",
                   help="Comma-separated subset of methods to run.")
    p.add_argument("--question_keys", default="",
                   help="Comma-separated list of '<sub>/<qid>' keys; if set, restrict eval to these.")
    p.add_argument("--tag", default="13q")
    p.add_argument("--output_dir", default=f"{RESULTS_DIR}/table1")
    args = p.parse_args()

    methods = set(args.methods.split(","))
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = f"{args.output_dir}/seed{args.seed}_{args.tag}.json"
    log(f"=== Table1 eval — seed={args.seed} methods={methods} on {args.device} ===")

    log("Loading data …")
    questions_all = load_questions(args.questions)
    log(f"Loaded {len(questions_all)} questions from JSON.")
    # top_cids derived from FULL set so shards see consistent cluster selection
    top_cids = get_top_clusters(questions_all, n=args.top_k)
    log(f"Top {args.top_k} clusters (from full set): {top_cids}")
    if args.question_keys:
        wanted = set(s.strip() for s in args.question_keys.split(",") if s.strip())
        questions = [q for q in questions_all if q["key"] in wanted]
        log(f"Filtered to {len(questions)} questions: {[q['key'] for q in questions]}")
    else:
        questions = questions_all
    df_posts = pd.read_csv(f"{DATA_DIR}/2_meaningful_df.csv", usecols=["content_desc", "cluster_label"])
    with open(f"{DATA_DIR}/3_cluster_topics.json") as f:
        topics = json.load(f)

    log("Loading model + encoder …")
    encoder = SentenceTransformer(EMBED_PATH, device=args.device)
    model, tokenizer = load_model(MODEL_PATH, args.device)

    log("Generating anchors (deterministic seed=42; shared across methods/seeds) …")
    set_seed(42)
    anchors = {}
    for qd in questions:
        anc = generate_anchors_local(model, tokenizer, qd["question"], qd["options"])
        anchors[qd["key"]] = {"texts": anc,
                                "embs": [encoder.encode(a) for a in anc]}
        log(f"  {qd['key']}: {anc[0][:40]}…")

    set_seed(args.seed)
    out = {"config": vars(args), "top_clusters": top_cids,
           "per_question": {q["key"]: {"question": q["question"],
                                          "options": q["options"],
                                          "true": list(q["true_distribution"].values())}
                              for q in questions}}

    # ─── Method loop ────────────────────────────────────────────────────
    for qd in questions:
        out["per_question"][qd["key"]]["preds"] = {}
        out["per_question"][qd["key"]]["metrics"] = {}

    if "paper_ssr" in methods:
        log(f"\n=== paper_ssr (n_samples={args.n_samples_paper}) ===")
        for qi, qd in enumerate(questions):
            t0 = time.time()
            pred = run_paper_ssr(model, tokenizer, encoder, qd,
                                  anchors[qd["key"]]["embs"],
                                  n_samples=args.n_samples_paper)
            true = list(qd["true_distribution"].values())
            m = metrics(pred, true)
            out["per_question"][qd["key"]]["preds"]["paper_ssr"] = pred.tolist()
            out["per_question"][qd["key"]]["metrics"]["paper_ssr"] = m
            log(f"  Q{qi+1}/{len(questions)} {qd['key']:<22} JS={m['js']:.4f}  ({time.time()-t0:.0f}s)")

    if "direct_llm" in methods:
        log(f"\n=== direct_llm (n_samples={args.n_samples_direct}) ===")
        for qi, qd in enumerate(questions):
            t0 = time.time()
            pred = run_direct_llm(model, tokenizer, qd, n_samples=args.n_samples_direct)
            true = list(qd["true_distribution"].values())
            m = metrics(pred, true)
            out["per_question"][qd["key"]]["preds"]["direct_llm"] = pred.tolist()
            out["per_question"][qd["key"]]["metrics"]["direct_llm"] = m
            log(f"  Q{qi+1}/{len(questions)} {qd['key']:<22} JS={m['js']:.4f}  ({time.time()-t0:.0f}s)")

    if "flat_ssr" in methods:
        log("\n=== flat_ssr ===")
        for qi, qd in enumerate(questions):
            t0 = time.time()
            pred = run_flat_ssr(encoder, qd, anchors[qd["key"]]["embs"],
                                 df_posts, top_cids,
                                 n_posts=args.n_posts_cluster, sample_seed=args.seed)
            true = list(qd["true_distribution"].values())
            m = metrics(pred, true)
            out["per_question"][qd["key"]]["preds"]["flat_ssr"] = pred.tolist()
            out["per_question"][qd["key"]]["metrics"]["flat_ssr"] = m
            log(f"  Q{qi+1}/{len(questions)} {qd['key']:<22} JS={m['js']:.4f}  ({time.time()-t0:.0f}s)")

    cluster_pmfs_m0 = {}
    if "m0" in methods or "m1" in methods or "m2" in methods:
        log("\n=== m0 (cluster SSR) ===")
        for qi, qd in enumerate(questions):
            t0 = time.time()
            cpmfs = run_m0_pmfs(encoder, qd, top_cids, df_posts,
                                  anchors[qd["key"]]["embs"],
                                  n_posts=args.n_posts_cluster, sample_seed=args.seed)
            cluster_pmfs_m0[qd["key"]] = cpmfs
            pred = agg_pmf(cpmfs, qd["cluster_weights"], top_cids)
            true = list(qd["true_distribution"].values())
            m = metrics(pred, true)
            out["per_question"][qd["key"]]["preds"]["m0"] = pred.tolist()
            out["per_question"][qd["key"]]["metrics"]["m0"] = m
            log(f"  Q{qi+1}/{len(questions)} {qd['key']:<22} JS={m['js']:.4f}  ({time.time()-t0:.0f}s)")

    cluster_pmfs_m1 = {}
    if "m1" in methods or "m2" in methods:
        log(f"\n=== m1 (cluster LLM-dist, n_samples={args.n_samples_m1}) ===")
        for qi, qd in enumerate(questions):
            t0 = time.time()
            cpmfs, strats = run_m1_pmfs(model, tokenizer, qd, top_cids, df_posts, topics,
                                          n_ex=8, n_samples=args.n_samples_m1, sample_seed=args.seed)
            cluster_pmfs_m1[qd["key"]] = cpmfs
            pred = agg_pmf(cpmfs, qd["cluster_weights"], top_cids)
            true = list(qd["true_distribution"].values())
            m = metrics(pred, true)
            out["per_question"][qd["key"]]["preds"]["m1"] = pred.tolist()
            out["per_question"][qd["key"]]["metrics"]["m1"] = m
            out["per_question"][qd["key"]].setdefault("parse_strats", {})["m1"] = strats
            log(f"  Q{qi+1}/{len(questions)} {qd['key']:<22} JS={m['js']:.4f}  parse={strats}  ({time.time()-t0:.0f}s)")

    if "c2" in methods:
        log(f"\n=== c2 hardened (n_posts={args.n_posts_c2}) ===")
        for qi, qd in enumerate(questions):
            t0 = time.time()
            cpmfs, strats, ntot = run_c2_pmfs(model, tokenizer, qd, top_cids, df_posts,
                                                n_posts=args.n_posts_c2, sample_seed=args.seed)
            pred = agg_pmf(cpmfs, qd["cluster_weights"], top_cids)
            true = list(qd["true_distribution"].values())
            m = metrics(pred, true)
            out["per_question"][qd["key"]]["preds"]["c2"] = pred.tolist()
            out["per_question"][qd["key"]]["metrics"]["c2"] = m
            out["per_question"][qd["key"]].setdefault("parse_strats", {})["c2"] = strats
            out["per_question"][qd["key"]]["c2_n_total"] = ntot
            log(f"  Q{qi+1}/{len(questions)} {qd['key']:<22} JS={m['js']:.4f}  fails={strats.get('fail',0)}/{ntot}  ({time.time()-t0:.0f}s)")

    # ─── Summary across questions ──────────────────────────────────────
    out["summary"] = {}
    for method in ["paper_ssr", "direct_llm", "flat_ssr", "m0", "m1", "c2"]:
        vals = [out["per_question"][k]["metrics"].get(method) for k in out["per_question"]
                  if "metrics" in out["per_question"][k] and method in out["per_question"][k]["metrics"]]
        if not vals: continue
        js_arr = np.array([v["js"] for v in vals])
        k_arr = np.array([v["k_xy"] for v in vals])
        c_arr = np.array([v["c_xy"] for v in vals])
        out["summary"][method] = {
            "js_mean": float(js_arr.mean()),
            "js_std": float(js_arr.std(ddof=1)) if len(js_arr) > 1 else 0.0,
            "kxy_mean": float(k_arr.mean()),
            "cxy_mean": float(c_arr.mean()),
            "n_questions": len(js_arr),
        }
    log("\n=== Across-question summary (this seed only) ===")
    for m, s in out["summary"].items():
        log(f"  {m:<12} JS = {s['js_mean']:.4f} ± {s['js_std']:.4f}  (n={s['n_questions']})")

    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    log(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
