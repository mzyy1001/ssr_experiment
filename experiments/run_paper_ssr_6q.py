"""
Paper-faithful SSR baseline on the 6Q real benchmark.

Method (Maier et al., "LLMs Reproduce Human Purchase Intent via Semantic
Similarity Elicitation of Likert Ratings"):
  1. LLM impersonates a generic consumer (no clustering, no real posts).
  2. Free-text response to the survey question.
  3. Embed response, cosine-similarity with anchor sentences, temp-0.1 softmax.
  4. Average PMFs across N samples → predicted distribution.

Differences from M0 (hierarchical SSR):
  - M0 embeds real cluster posts; paper-SSR embeds LLM-generated responses.
  - M0 aggregates via per-question `cluster_weights`; paper-SSR is flat.
  - Same embedder, anchors, softmax temperature, and JS metric for fair comparison.

Output: results/paper_ssr_6q.json
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

from persona_vectors import load_model
from steered_ssr import ssr_score, generate_anchors_local


DATA_DIR = "/data/chenhongrui/business/data"
RESULTS_DIR = "/data/chenhongrui/business/results"
MODEL_PATH = "/data/chenhongrui/models/Qwen3-8B"
EMBED_PATH = "/data/chenhongrui/models/bge-base-zh-v1.5"

ORIG_6 = ["8.0/qsingle_3", "8.0/qsingle_4", "9.0/qsingle_3",
          "9.0/qsingle_4", "9.0/qsingle_5", "11.0/qsingle_3"]


SYSTEM_PROMPT = (
    "你是一位中国消费者，可能使用过藿香正气水。"
    "请根据自己的真实想法用自然的中文回答问卷问题（2-3句话，直接表达观点，不要列选项）。"
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
                        "key": key,
                        "question": qd["question"],
                        "options": qd["options"],
                        "true_distribution": qd["true_distribution"],
                    })
    questions.sort(key=lambda x: ORIG_6.index(x["key"]))
    return questions


def js_score(td, pred):
    t = np.array(list(td.values()), dtype=float); t /= t.sum()
    p = pred / (pred.sum() + 1e-10)
    return float(jensenshannon(t, p) ** 2)


def generate_response(model, tokenizer, question, options, temperature=0.7):
    """Generate one free-text consumer response to the survey question."""
    opts_str = "、".join(options)
    user_prompt = f"问题：{question}\n可能的选项：{opts_str}\n\n你的回答（用自然语言）："
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    try:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        prompt = f"{SYSTEM_PROMPT}\n\n{user_prompt}"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
        )
    resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    resp = re.sub(r'<think>.*?</think>', '', resp, flags=re.DOTALL).strip()
    return resp


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:2")
    p.add_argument("--n_samples", type=int, default=100)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--output", default=f"{RESULTS_DIR}/paper_ssr_6q.json")
    p.add_argument("--save_responses", action="store_true",
                   help="Also save all generated responses for qualitative inspection")
    args = p.parse_args()

    set_seed(2026)

    log("Loading data …")
    questions = load_orig6()
    log(f"Loaded {len(questions)} questions")

    log("Loading model + encoder …")
    encoder = SentenceTransformer(EMBED_PATH)
    model, tokenizer = load_model(MODEL_PATH, args.device)

    log("Generating anchors (same procedure as M0) …")
    anchors = {}
    for qd in questions:
        anc = generate_anchors_local(model, tokenizer, qd["question"], qd["options"])
        anchors[qd["key"]] = {
            "texts": anc,
            "embeddings": [encoder.encode(a) for a in anc],
        }
        log(f"  {qd['key']}: {anc[0][:50]}…")

    js_per_q = []
    responses_per_q = {}
    predictions = {}
    for qi, qd in enumerate(questions):
        log(f"\n=== Q{qi+1}/{len(questions)} [{qd['key']}] ===")
        log(f"  question: {qd['question'][:80]}")
        aembs = anchors[qd["key"]]["embeddings"]

        pmfs = []
        responses = []
        t0 = time.time()
        for s in range(args.n_samples):
            resp = generate_response(model, tokenizer, qd["question"], qd["options"],
                                      temperature=args.temperature)
            if not resp:
                resp = "没意见"
            responses.append(resp)
            pmf, _ = ssr_score(resp, encoder, aembs, normalization="softmax")
            pmfs.append(pmf)
            if (s + 1) % 20 == 0:
                log(f"    {s+1}/{args.n_samples} samples, elapsed={time.time()-t0:.1f}s")

        mean_pmf = np.mean(pmfs, axis=0)
        mean_pmf = mean_pmf / (mean_pmf.sum() + 1e-10)
        js = js_score(qd["true_distribution"], mean_pmf)
        js_per_q.append(js)
        predictions[qd["key"]] = {
            "true": list(qd["true_distribution"].values()),
            "pred": mean_pmf.tolist(),
            "js": js,
        }
        if args.save_responses:
            responses_per_q[qd["key"]] = responses[:20]  # first 20 for inspection
        log(f"  JS = {js:.4f}   pred={np.round(mean_pmf, 3).tolist()}")

    output = {
        "config": vars(args),
        "method": "paper-faithful SSR (no cluster, no real posts; LLM-generate -> SSR)",
        "system_prompt": SYSTEM_PROMPT,
        "js_per_q": js_per_q,
        "js_mean": float(np.mean(js_per_q)),
        "predictions": predictions,
        "anchors": {k: v["texts"] for k, v in anchors.items()},
    }
    if args.save_responses:
        output["sample_responses"] = responses_per_q

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    log("\n" + "=" * 60)
    log("SUMMARY — paper-SSR baseline on 6Q")
    log("=" * 60)
    log(f"mean JS = {np.mean(js_per_q):.4f}")
    for qi, qd in enumerate(questions):
        log(f"  Q{qi+1} [{qd['key']}]: JS = {js_per_q[qi]:.4f}")
    log(f"\nSaved → {args.output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        log(f"ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)
