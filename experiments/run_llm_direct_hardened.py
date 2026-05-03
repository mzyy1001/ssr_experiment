"""
LLM-direct baseline (no social media) — hardened parser rerun.

Same prompt as run_llm_direct_baseline.py but:
  - Uses the exact 4-strategy parser from dump_c2_hardened.py (S1 JSON-obj,
    S2 bracket-exact, S3 last-n-nums, S4 percent-lines).
  - Bumped n_samples to 30 (vs 10) to wash out parse-fail variance.
  - Saves all raw responses for audit.
  - Reports two JS numbers: (A) with-uniform-fallback (apples-to-apples
    with original 0.0244), and (B) effective-only — average of just the
    successfully-parsed PMFs, ignoring fail calls.

Output: results/llm_direct_hardened_6q.json
"""
import argparse
import json
import re
import time
import random

import numpy as np
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


# ─── Hardened 4-strategy parser (exact copy from dump_c2_hardened.py) ──
def _normalize_arr(arr, n_opts):
    arr = np.clip(np.array(arr, float), 0, None)
    if len(arr) != n_opts or arr.sum() <= 0:
        return None
    return arr / arr.sum()


def parse_distribution_hardened(resp, n_opts):
    text = re.sub(r'<think>.*?</think>', '', resp, flags=re.DOTALL).strip()
    # S1: full JSON object containing "distribution"
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
    # S2: bare [a, b, c, ...] with exact n_opts numeric entries
    for m in re.finditer(r'\[([^\[\]]+)\]', text):
        body = m.group(1)
        nums = re.findall(r'-?\d+(?:\.\d+)?', body)
        if len(nums) == n_opts:
            pmf = _normalize_arr([float(x) for x in nums], n_opts)
            if pmf is not None:
                return pmf, "s2_bracket_exact"
    # S3: last n_opts numeric tokens anywhere
    nums = re.findall(r'-?\d*\.\d+|-?\d+', text)
    if len(nums) >= n_opts:
        pmf = _normalize_arr([float(x) for x in nums[-n_opts:]], n_opts)
        if pmf is not None:
            return pmf, "s3_last_n_nums"
    # S4: percent-style mentions
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--n_samples", type=int, default=30)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--save_responses", action="store_true", default=True)
    p.add_argument("--output", default=f"{RESULTS_DIR}/llm_direct_hardened_6q.json")
    args = p.parse_args()

    set_seed(args.seed)
    log("Loading questions …")
    questions = load_orig6()
    log("Loading model …")
    model, tokenizer = load_model(MODEL_PATH, args.device)

    out = {"config": vars(args), "prompt_template": PROMPT,
           "per_question": {}, "summary": {}}
    js_with_fb, js_eff_only = [], []
    parse_total = {"s1_json_obj": 0, "s2_bracket_exact": 0,
                   "s3_last_n_nums": 0, "s4_percent_lines": 0, "fail": 0}

    for qi, qd in enumerate(questions):
        opts = qd["options"]
        opts_str = "\n".join(f"{i+1}. {o}" for i, o in enumerate(opts))
        prompt = PROMPT.format(question=qd["question"], options_numbered=opts_str)
        valid_pmfs = []
        all_pmfs = []  # with uniform fallback for failed parses
        strats_q = {"s1_json_obj": 0, "s2_bracket_exact": 0,
                    "s3_last_n_nums": 0, "s4_percent_lines": 0, "fail": 0}
        failed_responses = []
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
            pmf, strat = parse_distribution_hardened(resp, len(opts))
            strats_q[strat] += 1
            parse_total[strat] += 1
            if pmf is not None:
                valid_pmfs.append(pmf)
                all_pmfs.append(pmf)
            else:
                all_pmfs.append(np.ones(len(opts)) / len(opts))
                if len(failed_responses) < 5:
                    failed_responses.append(resp.strip()[:300])

        true = list(qd["true_distribution"].values())
        # (A) with-fallback (apples-to-apples with original 0.0244 reporting)
        pred_fb = np.mean(all_pmfs, axis=0)
        # (B) effective-only — only successfully-parsed calls
        if valid_pmfs:
            pred_eff = np.mean(valid_pmfs, axis=0)
        else:
            pred_eff = np.ones(len(opts)) / len(opts)

        m_fb = {"js": js_score(true, pred_fb), "k_xy": k_xy(true, pred_fb), "c_xy": c_xy(true, pred_fb)}
        m_eff = {"js": js_score(true, pred_eff), "k_xy": k_xy(true, pred_eff), "c_xy": c_xy(true, pred_eff)}
        js_with_fb.append(m_fb["js"]); js_eff_only.append(m_eff["js"])

        out["per_question"][qd["key"]] = {
            "question": qd["question"], "options": opts,
            "true": [float(x) for x in true],
            "pred_with_fallback": [float(x) for x in pred_fb],
            "pred_effective_only": [float(x) for x in pred_eff],
            "metrics_with_fallback": m_fb,
            "metrics_effective_only": m_eff,
            "parse_strats": strats_q,
            "n_valid": len(valid_pmfs), "n_total": args.n_samples,
            "failed_responses_sample": failed_responses,
        }
        log(f"  Q{qi+1}/{len(questions)} {qd['key']}: "
            f"JS_fb={m_fb['js']:.4f}  JS_eff={m_eff['js']:.4f}  "
            f"valid={len(valid_pmfs)}/{args.n_samples}  "
            f"strats={strats_q}  elapsed={time.time()-t0:.0f}s")

    out["summary"] = {
        "with_fallback":   {"js": float(np.mean(js_with_fb))},
        "effective_only":  {"js": float(np.mean(js_eff_only))},
        "parse_total": parse_total,
        "fail_rate": parse_total["fail"] / sum(parse_total.values()),
    }
    log("=" * 60)
    log(f"Mean JS (with uniform fallback)  = {out['summary']['with_fallback']['js']:.4f}")
    log(f"Mean JS (effective only, no fb)  = {out['summary']['effective_only']['js']:.4f}")
    log(f"Total fail rate = {out['summary']['fail_rate']*100:.1f}%")
    log(f"Parse strategy total: {parse_total}")

    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    log(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
