"""
Two follow-up experiments on the 6Q real benchmark:

A) M4 stability — rerun SSR + multi-layer-steer ensemble (w=0.5) three
   additional times with distinct torch seeds. Earlier runs showed M4 jumping
   between 0.0261 and 0.0315. We want mean and std across runs.

B) LLM-dist sample-count sweep — vary n_samples ∈ {5, 10, 20} in llm_dist_est
   and evaluate M1 (LLM-dist alone) and M2 (SSR + LLM-dist w=0.5). n=3 is the
   current baseline from improve_orig6.

Output: results/stability_6q.json
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
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


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
    t = np.array(list(td.values()), dtype=float)
    t /= t.sum()
    p = pred / (pred.sum() + 1e-10)
    return float(jensenshannon(t, p) ** 2)


def agg(cpmfs, weights, cids):
    n = len(next(iter(cpmfs.values())))
    a, tw = np.zeros(n), 0.0
    for c in cids:
        if c in cpmfs:
            w = weights.get(c, 0)
            a += w * cpmfs[c]
            tw += w
    return a / tw if tw > 0 else a


def compute_ssr_pmfs(encoder, qd, cids, df, aembs, n_posts=50, seed=42):
    cpmfs = {}
    for cid in cids:
        posts = df[df["cluster_label"] == int(cid)]["content_desc"].dropna()
        if len(posts) < 3:
            continue
        samp = posts.sample(min(n_posts, len(posts)), random_state=seed).tolist()
        pmfs = [ssr_score(p, encoder, aembs)[0] for p in samp]
        cpmfs[cid] = np.mean(pmfs, axis=0)
    return cpmfs


def compute_llmdist_pmfs(model, tokenizer, qd, cids, df, topics, n_ex=8,
                         n_samples=3, sample_seed=42):
    opts = qd["options"]
    opts_str = "\n".join(f"{i+1}. {o}" for i, o in enumerate(opts))
    cpmfs = {}
    for cid in cids:
        posts = df[df["cluster_label"] == int(cid)]["content_desc"].dropna()
        if len(posts) < 3:
            continue
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
            if m:
                try:
                    vals = [float(x.strip()) for x in m.group(1).split(",")]
                    if len(vals) == len(opts):
                        arr = np.clip(np.array(vals), 0, None)
                        if arr.sum() > 0:
                            pmfs_collected.append(arr / arr.sum())
                            continue
                except Exception:
                    pass
            pmfs_collected.append(np.ones(len(opts)) / len(opts))
        cpmfs[cid] = np.mean(pmfs_collected, axis=0)
    return cpmfs


def compute_mlsteer_pmfs(model, tokenizer, encoder, qd, cids, pvecs, aembs,
                          alpha=0.1, layers=(16, 20, 24), n_resp=5):
    prompt = (
        "作为一位有相关经验的消费者，请回答以下问卷问题。\n"
        "请用自然的语言表达你的真实想法和感受，不需要选择选项。\n\n"
        f"问题：{qd['question']}\n选项：{'、'.join(qd['options'])}\n\n你的回答："
    )
    cpmfs = {}
    for cid in cids:
        if int(cid) not in pvecs:
            continue
        vec = pvecs[int(cid)]["vector"]
        pmfs = []
        for _ in range(n_resp):
            hooks = [SteeringHook(vec, alpha, l) for l in layers]
            for h in hooks:
                h.attach(model)
            try:
                inputs = tokenizer(prompt, return_tensors="pt")
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=200,
                                        do_sample=True, temperature=0.7, top_p=0.9)
                resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                       skip_special_tokens=True)
            finally:
                for h in hooks:
                    h.remove()
            pmf, _ = ssr_score(resp, encoder, aembs)
            pmfs.append(pmf)
        cpmfs[cid] = np.mean(pmfs, axis=0)
    return cpmfs


def eval_method(per_q_pmfs, questions, top_cids):
    """Return list of per-question JS."""
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
    p.add_argument("--output", default=f"{RESULTS_DIR}/stability_6q.json")
    p.add_argument("--n_m4_repeats", type=int, default=3)
    args = p.parse_args()

    log("Loading data …")
    questions, topics, df = load_orig6()
    top_cids = get_top_clusters(questions, n=args.top_k)
    log(f"Top {args.top_k} clusters: {top_cids}")

    log("Loading model + encoder …")
    encoder = SentenceTransformer(EMBED_PATH)
    model, tokenizer = load_model(MODEL_PATH, args.device)
    pvecs = load_persona_vectors(f"{RESULTS_DIR}/persona_vectors_L24_N20.npz")

    log("Generating anchors …")
    anchors = {}
    for qd in questions:
        anc = generate_anchors_local(model, tokenizer, qd["question"], qd["options"])
        anchors[qd["key"]] = [encoder.encode(a) for a in anc]
        log(f"  {qd['key']}: {anc[0][:40]}…")

    # ─── SSR (deterministic, compute once) ───
    log("Computing SSR per-cluster distributions (n_posts=50) …")
    per_q_ssr = {}
    for qi, qd in enumerate(questions):
        per_q_ssr[qi] = compute_ssr_pmfs(encoder, qd, top_cids, df,
                                           anchors[qd["key"]],
                                           n_posts=args.n_posts)
    ssr_js = eval_method(per_q_ssr, questions, top_cids)
    log(f"  M0 SSR: mean JS = {np.mean(ssr_js):.4f} | "
        f"{' '.join(f'{v:.4f}' for v in ssr_js)}")

    output = {
        "ssr_js": ssr_js,
        "config": vars(args),
        "top_clusters": [str(c) for c in top_cids],
    }

    # ─── Experiment A: M4 stability ───
    log("\n=== Experiment A: M4 stability ===")
    m4_trials = []
    ml_trials = []
    for trial in range(args.n_m4_repeats):
        seed = 2024 + trial
        set_seed(seed)
        log(f"Trial {trial+1}/{args.n_m4_repeats} (seed={seed}) — ML-steer …")
        per_q_ml = {}
        for qi, qd in enumerate(questions):
            per_q_ml[qi] = compute_mlsteer_pmfs(model, tokenizer, encoder,
                                                  qd, top_cids, pvecs,
                                                  anchors[qd["key"]],
                                                  alpha=0.1, layers=(16, 20, 24),
                                                  n_resp=5)
            log(f"  Q{qi+1}/{len(questions)} done")

        # M3 (ML-steer alone)
        m3_js = eval_method(per_q_ml, questions, top_cids)
        ml_trials.append(m3_js)
        log(f"  M3 (ML-steer alone): mean JS = {np.mean(m3_js):.4f}")

        # M4 (SSR + ML-steer w=0.5)
        per_q_m4 = {}
        for qi in range(len(questions)):
            per_q_m4[qi] = {}
            for c in top_cids:
                if c in per_q_ssr[qi] and c in per_q_ml[qi]:
                    per_q_m4[qi][c] = 0.5 * per_q_ssr[qi][c] + 0.5 * per_q_ml[qi][c]
        m4_js = eval_method(per_q_m4, questions, top_cids)
        m4_trials.append(m4_js)
        log(f"  M4 (w=0.5): mean JS = {np.mean(m4_js):.4f}")

        # Save incrementally
        output["m4_trials"] = m4_trials
        output["m3_trials"] = ml_trials
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

    m4_means = [np.mean(t) for t in m4_trials]
    m3_means = [np.mean(t) for t in ml_trials]
    log(f"\n  M3 across trials: mean={np.mean(m3_means):.4f} "
        f"std={np.std(m3_means):.4f} values={[f'{m:.4f}' for m in m3_means]}")
    log(f"  M4 across trials: mean={np.mean(m4_means):.4f} "
        f"std={np.std(m4_means):.4f} values={[f'{m:.4f}' for m in m4_means]}")

    # ─── Experiment B: LLM-dist sample-count sweep ───
    log("\n=== Experiment B: LLM-dist sample-count sweep ===")
    llm_sweep = {}
    m1_sweep = {}
    m2_sweep = {}
    for n_samp in [5, 10]:
        set_seed(2026)
        log(f"n_samples={n_samp} — LLM-dist …")
        per_q_llm = {}
        for qi, qd in enumerate(questions):
            per_q_llm[qi] = compute_llmdist_pmfs(model, tokenizer, qd, top_cids,
                                                  df, topics, n_samples=n_samp)
            log(f"  Q{qi+1}/{len(questions)} done (n={n_samp})")

        # M1
        m1_js = eval_method(per_q_llm, questions, top_cids)
        m1_sweep[n_samp] = m1_js
        log(f"  M1 (LLM-dist n={n_samp}): mean JS = {np.mean(m1_js):.4f}")

        # M2 (SSR + LLM-dist w=0.5)
        per_q_m2 = {}
        for qi in range(len(questions)):
            per_q_m2[qi] = {}
            for c in top_cids:
                if c in per_q_ssr[qi] and c in per_q_llm[qi]:
                    per_q_m2[qi][c] = 0.5 * per_q_ssr[qi][c] + 0.5 * per_q_llm[qi][c]
        m2_js = eval_method(per_q_m2, questions, top_cids)
        m2_sweep[n_samp] = m2_js
        log(f"  M2 (SSR+LLMdist w=0.5, n={n_samp}): mean JS = {np.mean(m2_js):.4f}")

        # Sweep mixing weights at this n_samp
        log(f"  weight sweep at n_samp={n_samp}:")
        for w in [0.3, 0.4, 0.5, 0.6, 0.7]:
            per_q_wm = {}
            for qi in range(len(questions)):
                per_q_wm[qi] = {}
                for c in top_cids:
                    if c in per_q_ssr[qi] and c in per_q_llm[qi]:
                        per_q_wm[qi][c] = w * per_q_ssr[qi][c] + (1-w) * per_q_llm[qi][c]
            wj = eval_method(per_q_wm, questions, top_cids)
            llm_sweep[f"n={n_samp}_w={w}"] = wj
            log(f"    w={w}: mean JS = {np.mean(wj):.4f}")

        output["llm_sweep_m1"] = m1_sweep
        output["llm_sweep_m2"] = m2_sweep
        output["llm_weight_sweep"] = llm_sweep
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

    # ─── Summary ───
    log("\n" + "=" * 70)
    log("SUMMARY")
    log("=" * 70)
    log(f"\nSSR baseline (M0): {np.mean(ssr_js):.4f}")
    log("")
    log("Experiment A — M4 stability (across seeds):")
    log(f"  Trial means: {[f'{m:.4f}' for m in m4_means]}")
    log(f"  mean={np.mean(m4_means):.4f}  std={np.std(m4_means):.4f}  "
        f"range=[{min(m4_means):.4f}, {max(m4_means):.4f}]")
    log("")
    log("Experiment B — LLM-dist sample-count sweep:")
    log(f"  {'n_samp':<8} {'M1 (LLMdist)':<18} {'M2 (SSR+LLMdist w=0.5)':<24}")
    for n in [5, 10]:
        log(f"  {n:<8} {np.mean(m1_sweep[n]):<18.4f} "
            f"{np.mean(m2_sweep[n]):<24.4f}")
    log("")
    log("Weight sweep (best cell per n):")
    for n in [5, 10]:
        best_w = min([0.3, 0.4, 0.5, 0.6, 0.7],
                      key=lambda w: np.mean(llm_sweep[f"n={n}_w={w}"]))
        log(f"  n={n}: best w={best_w} -> "
            f"{np.mean(llm_sweep[f'n={n}_w={best_w}']):.4f}")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    log(f"\nSaved → {args.output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        log(f"ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)
