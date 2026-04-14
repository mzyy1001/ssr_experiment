"""Run control vector ablations: zero, random, shuffled vectors through PS-SSR."""
import sys, os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch, json, numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from persona_vectors import load_model, load_persona_vectors
from steered_ssr import (
    ssr_score, generate_anchors_local, create_control_vectors,
    ps_ssr_aggregate_then_steer,
)

MODEL_PATH = "/data/chenhongrui/models/Qwen3-8B"
VECTORS_PATH = "/data/chenhongrui/business/results/persona_vectors_L16_N20.npz"
QUESTIONS_PATH = "/data/chenhongrui/business/results/all_questions.json"
EMBEDDING_MODEL = "/data/chenhongrui/models/bge-base-zh-v1.5"
RESULTS_DIR = "/data/chenhongrui/business/results"
DEVICE = sys.argv[1] if len(sys.argv) > 1 else "cuda:3"

print(f"Loading models on {DEVICE}...")
model, tokenizer = load_model(MODEL_PATH, DEVICE)
encoder = SentenceTransformer(EMBEDDING_MODEL)
persona_vectors = load_persona_vectors(VECTORS_PATH)

with open(QUESTIONS_PATH) as f:
    baseline = json.load(f)

for control_type in ["zero", "random", "shuffled"]:
    print(f"\n{'='*60}")
    print(f"CONTROL: {control_type} vectors")
    print(f"{'='*60}")

    control_vectors = create_control_vectors(persona_vectors, control_type)
    results = {}

    for survey_key, survey_data in baseline.items():
        results[survey_key] = {}
        for sub_id, questions in survey_data.items():
            results[survey_key][sub_id] = {}
            for q_id, q_data in tqdm(questions.items(), desc=f"{control_type} {survey_key}/{sub_id}"):
                question = q_data["question"]
                options = q_data["options"]
                cluster_weights = q_data["cluster_weights"]

                anchors = generate_anchors_local(model, tokenizer, question, options)
                anchor_embs = [encoder.encode(a) for a in anchors]

                pred_pmf = ps_ssr_aggregate_then_steer(
                    model, tokenizer, encoder,
                    control_vectors, cluster_weights,
                    question, options, anchor_embs,
                    alpha=2.0, layer=16,
                )

                results[survey_key][sub_id][q_id] = {
                    "question": question,
                    "options": options,
                    "true_distribution": q_data["true_distribution"],
                    "predicted_pmf": pred_pmf.tolist(),
                    "control_type": control_type,
                }

    out_path = f"{RESULTS_DIR}/control_{control_type}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved to {out_path}")

print("\n====== ALL CONTROLS COMPLETE ======")
