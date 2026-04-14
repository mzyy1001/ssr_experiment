"""
Phase 1: Persona Vector Extraction via Contrastive Activation Addition (CAA).

For each cluster, extract a "persona vector" that captures the cluster's
attitude/opinion direction in the LLM's hidden state space.

Method:
  v_c = mean(hidden_states(positive_comments)) - mean(hidden_states(negative_comments))

Runs on chen server with Qwen3-8B.
"""
import torch
import numpy as np
import json
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


def load_model(model_path: str, device: str = "cuda:2"):
    """Load Qwen3-8B with hidden state output enabled."""
    print(f"Loading model from {model_path} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=device,
        output_hidden_states=True,
    )
    model.eval()
    return model, tokenizer


def get_hidden_states(model, tokenizer, texts: list[str], layer: int) -> np.ndarray:
    """Extract mean hidden states at a specific layer for a batch of texts.

    Returns:
        np.ndarray of shape (len(texts), hidden_dim)
    """
    hidden_states_list = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # outputs.hidden_states is a tuple of (n_layers+1,) tensors
        # each of shape (batch, seq_len, hidden_dim)
        hs = outputs.hidden_states[layer]  # (1, seq_len, hidden_dim)
        # Mean-pool over sequence length (excluding padding)
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        pooled = (hs * mask).sum(dim=1) / mask.sum(dim=1)  # (1, hidden_dim)
        hidden_states_list.append(pooled.squeeze(0).cpu().float().numpy())

    return np.stack(hidden_states_list)


def extract_persona_vectors(
    model,
    tokenizer,
    meaningful_df: pd.DataFrame,
    cluster_topics: dict,
    layer: int = 16,
    n_samples: int = 20,
    negation_strategy: str = "distant_cluster",
) -> dict:
    """Extract persona vector for each cluster.

    Args:
        negation_strategy:
            - "distant_cluster": comments from the most semantically distant cluster
            - "random_cluster": comments from a random other cluster
            - "global_mean": subtract the global mean of all comments (no contrast cluster)

    Returns:
        dict mapping cluster_id -> {
            "vector": np.ndarray,
            "pos_mean": np.ndarray,
            "neg_mean": np.ndarray,
            "n_pos": int,
            "n_neg": int,
        }
    """
    cluster_ids = sorted(meaningful_df["cluster_label"].unique())
    print(f"Extracting persona vectors for {len(cluster_ids)} clusters at layer {layer}...")

    # Pre-compute cluster centroids for distant-cluster negation
    cluster_centroids = {}
    if negation_strategy == "distant_cluster":
        print("Computing cluster centroids for negation selection...")
        for cid in tqdm(cluster_ids, desc="Centroids"):
            comments = meaningful_df[meaningful_df["cluster_label"] == cid]["content_desc"].dropna()
            sample = comments.sample(min(5, len(comments)), random_state=42).tolist()
            if sample:
                hs = get_hidden_states(model, tokenizer, sample, layer)
                cluster_centroids[cid] = hs.mean(axis=0)

    persona_vectors = {}
    for cid in tqdm(cluster_ids, desc="Persona vectors"):
        # Positive set: sample from this cluster
        pos_comments = (
            meaningful_df[meaningful_df["cluster_label"] == cid]["content_desc"]
            .dropna()
            .sample(min(n_samples, len(meaningful_df[meaningful_df["cluster_label"] == cid])), random_state=42)
            .tolist()
        )

        if len(pos_comments) < 3:
            print(f"  Cluster {cid}: too few comments ({len(pos_comments)}), skipping")
            continue

        # Negative set: from most distant cluster
        if negation_strategy == "distant_cluster":
            if cid not in cluster_centroids:
                continue
            distances = {
                oid: np.linalg.norm(cluster_centroids[cid] - cluster_centroids[oid])
                for oid in cluster_ids
                if oid != cid and oid in cluster_centroids
            }
            if not distances:
                continue
            farthest_cid = max(distances, key=distances.get)
            neg_comments = (
                meaningful_df[meaningful_df["cluster_label"] == farthest_cid]["content_desc"]
                .dropna()
                .sample(min(n_samples, len(meaningful_df[meaningful_df["cluster_label"] == farthest_cid])), random_state=42)
                .tolist()
            )
        elif negation_strategy == "random_cluster":
            rng = np.random.RandomState(cid)
            other_cids = [oid for oid in cluster_ids if oid != cid]
            random_cid = rng.choice(other_cids)
            neg_comments = (
                meaningful_df[meaningful_df["cluster_label"] == random_cid]["content_desc"]
                .dropna()
                .sample(min(n_samples, len(meaningful_df[meaningful_df["cluster_label"] == random_cid])), random_state=42)
                .tolist()
            )
        elif negation_strategy == "global_mean":
            # Use a global sample across all clusters as the "neutral" baseline
            neg_comments = (
                meaningful_df["content_desc"]
                .dropna()
                .sample(min(n_samples * 3, len(meaningful_df)), random_state=42)
                .tolist()
            )
        else:
            raise NotImplementedError(f"Strategy {negation_strategy} not yet implemented")

        # Extract hidden states
        pos_hs = get_hidden_states(model, tokenizer, pos_comments, layer)
        neg_hs = get_hidden_states(model, tokenizer, neg_comments, layer)

        pos_mean = pos_hs.mean(axis=0)
        neg_mean = neg_hs.mean(axis=0)
        vec = pos_mean - neg_mean

        persona_vectors[int(cid)] = {
            "vector": vec,
            "pos_mean": pos_mean,
            "neg_mean": neg_mean,
            "n_pos": len(pos_comments),
            "n_neg": len(neg_comments),
        }

    print(f"Extracted {len(persona_vectors)} persona vectors")
    return persona_vectors


def save_persona_vectors(persona_vectors: dict, save_path: str):
    """Save persona vectors to npz file."""
    arrays = {}
    metadata = {}
    for cid, data in persona_vectors.items():
        arrays[f"vec_{cid}"] = data["vector"]
        arrays[f"pos_{cid}"] = data["pos_mean"]
        arrays[f"neg_{cid}"] = data["neg_mean"]
        metadata[str(cid)] = {"n_pos": data["n_pos"], "n_neg": data["n_neg"]}

    np.savez(save_path, **arrays)
    with open(save_path.replace(".npz", "_meta.json"), "w") as f:
        json.dump(metadata, f)
    print(f"Saved to {save_path}")


def load_persona_vectors(load_path: str) -> dict:
    """Load persona vectors from npz file."""
    data = np.load(load_path)
    with open(load_path.replace(".npz", "_meta.json")) as f:
        metadata = json.load(f)

    persona_vectors = {}
    for cid_str, meta in metadata.items():
        cid = int(cid_str)
        persona_vectors[cid] = {
            "vector": data[f"vec_{cid}"],
            "pos_mean": data[f"pos_{cid}"],
            "neg_mean": data[f"neg_{cid}"],
            "n_pos": meta["n_pos"],
            "n_neg": meta["n_neg"],
        }
    return persona_vectors


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="/data/chenhongrui/models/Qwen3-8B")
    parser.add_argument("--data_path", default="/data/chenhongrui/business/data/2_meaningful_df.csv")
    parser.add_argument("--topics_path", default="/data/chenhongrui/business/data/3_cluster_topics.json")
    parser.add_argument("--output", default="/data/chenhongrui/business/results/persona_vectors.npz")
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--n_samples", type=int, default=20)
    parser.add_argument("--device", default="cuda:2")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model_path, args.device)
    meaningful_df = pd.read_csv(args.data_path)
    with open(args.topics_path) as f:
        cluster_topics = json.load(f)

    vectors = extract_persona_vectors(
        model, tokenizer, meaningful_df, cluster_topics,
        layer=args.layer, n_samples=args.n_samples,
    )
    save_persona_vectors(vectors, args.output)
