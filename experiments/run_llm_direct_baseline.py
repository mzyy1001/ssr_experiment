"""
LLM-only direct-distribution baseline (no social media, no SSR).

For each survey question, prompt Qwen3-8B with just the question + options
(no posts, no cluster topic, no examples) and ask it to output a distribution
over options. Sample N times per question, mean-pool PMFs.

This is the "pure LLM prior" — what the LLM thinks Chinese consumers' answer
distribution looks like, with zero social-media grounding. Mirrors the M1
prompt format but strips the cluster/post context.

Output: results/llm_direct_baseline_6q.json
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

from persona_vectors import load_model


DATA_DIR = "/data/chenhongrui/business/data"
RESULTS_DIR = "/data/chenhongrui/business/results"
MODEL_PATH = "/data/chenhongrui/models/Qwen3-8B"

ORIG_6 = ["8.0/qsingle_3", "8.0/qsingle_4", "9.0/qsingle_3",
          "9.0/qsingle_4", "9.0/qsingle_5", "11.0/qsingle_3"]


PROMPT = (
    "你是一位消费者调研分析专家。\n\n"
    "请估计中国消费者在以下问卷问题中各选项的选择比例。\n"
    "（仅基于你对该群体的常识和先验认知，不要请求更多上下文。）\n\n"
    "问题：{question}\n选项：\n{options_numbered}\n\n"
    "请严格按JSON格式输出各选项的比例（总和为1）：\n"
    '{{"distribution": [选项1比例, 选项2比例, ...]}}'
)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def load_orig6():
    with open(f"{RESULTS_DIR}/all_questions_expanded.json") as f:
        expanded = json.load(f)
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
                    })
    questions.sort(key=lambda x: ORIG_6.index(x["key"]))
    return questions


def js_score(t, p):
    t = np.array(list(t), float); t /= t.sum()
    p = np.array(p, float); p /= (p.sum() + 1e-10)
    return float(jensenshannon(t, p) ** 2)


def k_xy(t, p):
    t = np.array(list(t), float); t /= t.sum()
    p = np.array(p, float); p /= (p.sum() + 1e-10)
    return 1 - float(np.max(np.abs(np.cumsum(t) - np.cumsum(p))))


def c_xy(t, p):
    t = np.array(list(t), float); t /= t.sum()
    p = np.array(p, float); p /= (p.sum() + 1e-10)
    return float(np.dot(t, p) / (np.linalg.norm(t) * np.linalg.norm(p) + 1e-10))


def parse_dist(resp, n_opts):
    text = re.sub(r'<think>.*?</think>', '', resp, flags=re.DOTALL).strip()
    # try {...} JSON object first
    for m in re.finditer(r'\{[^{}]*"distribution"[^{}]*\}', text, flags=re.DOTALL):
        try:
            obj = json.loads(m.group(0))
            arr = obj.get("distribution")
            if arr and len(arr) == n_opts:
                arr = np.clip(np.array(arr, float), 0, None)
                if arr.sum() > 0:
                    return arr / arr.sum(), "s1_json"
        except Exception:
            continue
    # bare bracket list with exact n_opts numerics
    for m in re.finditer(r'\[([^\[\]]+)\]', text):
        nums = re.findall(r'-?\d+(?:\.\d+)?', m.group(1))
        if len(nums) == n_opts:
            arr = np.clip(np.array([float(x) for x in nums]), 0, None)
            if arr.sum() > 0:
                return arr / arr.sum(), "s2_bracket"
    # last-n numeric tokens
    nums = re.findall(r'-?\d*\.\d+|-?\d+', text)
    if len(nums) >= n_opts:
        arr = np.clip(np.array([float(x) for x in nums[-n_opts:]]), 0, None)
        if arr.sum() > 0:
            return arr / arr.sum(), "s3_lastn"
    return None, "fail"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--n_samples", type=int, default=10)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--output", default=f"{RESULTS_DIR}/llm_direct_baseline_6q.json")
    args = p.parse_args()

    set_seed(args.seed)
    log("Loading questions …")
    questions = load_orig6()

    log("Loading model …")
    model, tokenizer = load_model(MODEL_PATH, args.device)

    out = {"config": vars(args), "prompt_template": PROMPT,
           "per_question": {}, "summary": {}}
    js_all, k_all, c_all = [], [], []
    parse_total = {"s1_json": 0, "s2_bracket": 0, "s3_lastn": 0, "fail": 0}

    for qi, qd in enumerate(questions):
        opts = qd["options"]
        opts_str = "\n".join(f"{i+1}. {o}" for i, o in enumerate(opts))
        prompt = PROMPT.format(question=qd["question"], options_numbered=opts_str)
        pmfs = []
        sample_strats = {"s1_json": 0, "s2_bracket": 0, "s3_lastn": 0, "fail": 0}
        t0 = time.time()
        for s in range(args.n_samples):
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.no_grad():
                gen = model.generate(**inputs, max_new_tokens=200,
                                      do_sample=True, temperature=args.temperature,
                                      top_p=args.top_p)
            resp = tokenizer.decode(gen[0][inputs["input_ids"].shape[1]:],
                                     skip_special_tokens=True)
            pmf, strat = parse_dist(resp, len(opts))
            sample_strats[strat] += 1
            parse_total[strat] += 1
            if pmf is None:
                pmf = np.ones(len(opts)) / len(opts)
            pmfs.append(pmf)
        pred = np.mean(pmfs, axis=0)
        true = list(qd["true_distribution"].values())
        js = js_score(true, pred); k = k_xy(true, pred); c = c_xy(true, pred)
        js_all.append(js); k_all.append(k); c_all.append(c)
        out["per_question"][qd["key"]] = {
            "question": qd["question"], "options": opts,
            "true": [float(x) for x in true],
            "pred_llm_direct": [float(x) for x in pred],
            "metrics": {"js": js, "k_xy": k, "c_xy": c},
            "parse_strats": sample_strats,
            "n_samples": args.n_samples,
        }
        log(f"  Q{qi+1}/{len(questions)} {qd['key']}: JS={js:.4f} K={k:.3f} C={c:.3f} "
            f"strats={sample_strats} elapsed={time.time()-t0:.0f}s")

    out["summary"] = {
        "LLM_direct_baseline": {
            "js": float(np.mean(js_all)),
            "k_xy": float(np.mean(k_all)),
            "c_xy": float(np.mean(c_all)),
        },
        "parse_total": parse_total,
    }
    log("=" * 60)
    log(f"LLM-direct baseline | mean JS = {out['summary']['LLM_direct_baseline']['js']:.4f}")
    log(f"  K_xy = {out['summary']['LLM_direct_baseline']['k_xy']:.3f}")
    log(f"  C_xy = {out['summary']['LLM_direct_baseline']['c_xy']:.3f}")
    log(f"Reference: Paper-SSR=0.0430, Flat SSR=0.0295, M0=0.0269, M2=0.0254")

    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    log(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
