"""
Behavioral Clustering: recluster social media posts by survey-response
signature instead of topic similarity.

Current problem: 66 topic clusters produce near-identical steered outputs
(inter-cluster JS ≈ 0.002). Steering doesn't differentiate them.

Solution: cluster posts by how they steer the model's survey responses.
For each post, extract hidden state → project onto anchor directions →
get a "response signature" → cluster these signatures.

Posts that steer the model differently end up in different clusters.

Run on chen server:
  source /data/anaconda3/etc/profile.d/conda.sh && conda activate llama_qwen
  cd /data/chenhongrui/business/experiments
  python -u behavioral_cluster.py --device cuda:2
"""
import torch
import numpy as np
import json
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import jensenshannon
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from persona_vectors import load_model, get_hidden_states
from steered_ssr import ssr_score


DATA_DIR = "/data/chenhongrui/business/data"
RESULTS_DIR = "/data/chenhongrui/business/results"
MODEL_PATH = "/data/chenhongrui/models/Qwen3-8B"
EMBEDDING_MODEL = "/data/chenhongrui/models/bge-base-zh-v1.5"


def compute_post_signatures(model, tokenizer, encoder, posts, questions, layer=24):
    """Compute survey-response signature for each post.

    For each post:
      1. Extract hidden state at target layer (the "persona" of this post)
      2. For each survey question, steer the model with this hidden state
         and measure SSR → gets a distribution per question
      3. Concatenate all question distributions → response signature

    Optimization: Instead of full steered generation per post (too expensive
    for 12K posts), we use a cheaper proxy:
      - Compute SSR of the POST ITSELF against each question's anchors
      - This gives how the post content aligns with each answer option
      - Cluster by these alignments = cluster by response tendency

    This is O(n_posts × n_questions × n_anchors) embedding operations,
    NOT O(n_posts × n_questions × n_clusters) LLM generations.
    """
    print("Computing post response signatures...", flush=True)

    # Generate anchors for all questions
    print("  Generating anchors...", flush=True)
    from steered_ssr import generate_anchors_local
    anchor_cache = {}
    for qd in questions:
        q = qd["question"]
        if q not in anchor_cache:
            anchors = generate_anchors_local(model, tokenizer, q, qd["options"])
            anchor_embs = [encoder.encode(a) for a in anchors]
            anchor_cache[q] = anchor_embs
            print(f"    {q[:40]}... [{len(qd['options'])} opts]", flush=True)

    # For each post, compute SSR against all questions
    print(f"  Computing SSR signatures for {len(posts)} posts...", flush=True)
    signatures = []
    for idx, post in enumerate(tqdm(posts, desc="Signatures")):
        sig = []
        for qd in questions:
            anchor_embs = anchor_cache[qd["question"]]
            pmf, _ = ssr_score(post, encoder, anchor_embs)
            sig.extend(pmf.tolist())
        signatures.append(sig)

    return np.array(signatures), anchor_cache


def compute_hidden_state_signatures(model, tokenizer, posts, layer=24, batch_size=1):
    """Alternative: cluster by hidden state at steering layer.

    Posts with similar hidden states at layer L will produce similar
    steering effects. Clustering these directly gives "steering-similar" groups.
    """
    print(f"Extracting hidden states at layer {layer} for {len(posts)} posts...", flush=True)
    all_hs = []
    for i in tqdm(range(0, len(posts), batch_size), desc="Hidden states"):
        batch = posts[i:i+batch_size]
        hs = get_hidden_states(model, tokenizer, batch, layer)
        all_hs.append(hs)
    return np.vstack(all_hs)


def behavioral_kmeans(signatures, n_clusters_list=[5, 10, 15, 20, 30]):
    """Cluster posts by response signature using KMeans.

    Returns dict mapping n_clusters -> labels.
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(signatures)

    results = {}
    for k in n_clusters_list:
        print(f"  KMeans k={k}...", flush=True)
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        results[k] = labels
    return results


def evaluate_clustering(labels, posts_df, signatures, questions, encoder,
                        anchor_cache, model, tokenizer, persona_layer=24,
                        n_sample=20, alpha=0.1):
    """Evaluate a clustering by measuring inter-cluster divergence.

    For each cluster:
      1. Extract persona vector (CAA) from the cluster's posts
      2. Generate steered responses
      3. Measure inter-cluster JS divergence

    Also reports: cluster sizes, signature diversity.
    """
    from persona_vectors import get_hidden_states
    from steered_ssr import generate_steered_response

    unique_labels = sorted(set(labels))
    n_clusters = len(unique_labels)

    # Cluster sizes
    sizes = {l: np.sum(labels == l) for l in unique_labels}
    print(f"  Cluster sizes: min={min(sizes.values())}, max={max(sizes.values())}, "
          f"median={np.median(list(sizes.values())):.0f}", flush=True)

    # Signature-level diversity (cheap: just compare mean signatures)
    cluster_mean_sigs = {}
    for l in unique_labels:
        mask = labels == l
        cluster_mean_sigs[l] = signatures[mask].mean(axis=0)

    pairwise_js = []
    # Split signature into per-question chunks and measure JS
    q_lens = [len(qd["options"]) for qd in questions]
    for qi, qlen in enumerate(q_lens):
        start = sum(q_lens[:qi])
        for i in range(n_clusters):
            for j in range(i+1, n_clusters):
                p = cluster_mean_sigs[unique_labels[i]][start:start+qlen]
                q = cluster_mean_sigs[unique_labels[j]][start:start+qlen]
                p = p / (p.sum() + 1e-10)
                q = q / (q.sum() + 1e-10)
                pairwise_js.append(jensenshannon(p, q)**2)

    print(f"  Signature-level inter-cluster JS: mean={np.mean(pairwise_js):.4f} "
          f"max={np.max(pairwise_js):.4f}", flush=True)

    # Extract persona vectors for the new clusters
    print(f"  Extracting persona vectors for {n_clusters} behavioral clusters...", flush=True)
    posts_list = posts_df["content_desc"].tolist()
    persona_vectors = {}

    for l in unique_labels:
        mask = labels == l
        cluster_posts = [posts_list[i] for i in range(len(posts_list)) if mask[i]]
        if len(cluster_posts) < 3:
            continue

        # Sample for positive set
        rng = np.random.RandomState(l)
        pos_sample = rng.choice(cluster_posts, min(n_sample, len(cluster_posts)), replace=False).tolist()

        # Negative: sample from ALL other clusters
        other_posts = [posts_list[i] for i in range(len(posts_list)) if not mask[i]]
        neg_sample = rng.choice(other_posts, min(n_sample, len(other_posts)), replace=False).tolist()

        pos_hs = get_hidden_states(model, tokenizer, pos_sample, persona_layer)
        neg_hs = get_hidden_states(model, tokenizer, neg_sample, persona_layer)

        vec = pos_hs.mean(axis=0) - neg_hs.mean(axis=0)
        persona_vectors[l] = vec

    print(f"  Extracted {len(persona_vectors)} persona vectors", flush=True)

    # Steered generation divergence (expensive but definitive)
    # Sample 2 questions to keep it fast
    sample_qs = [questions[0], questions[2]]  # pick diverse questions
    prompt_tpl = (
        "作为一位有相关经验的消费者，请回答以下问卷问题。\n"
        "请用自然的语言表达你的真实想法和感受，不需要选择选项。\n\n"
        "问题：{question}\n选项：{options}\n\n你的回答："
    )

    steered_pmfs = {}  # (qi, cluster_label) -> pmf
    for qi, qd in enumerate(sample_qs):
        prompt = prompt_tpl.format(question=qd["question"], options="、".join(qd["options"]))
        anchor_embs = anchor_cache[qd["question"]]

        for l in list(persona_vectors.keys())[:min(10, len(persona_vectors))]:
            vec = persona_vectors[l]
            pmfs = []
            for _ in range(3):
                resp = generate_steered_response(model, tokenizer, prompt, vec, alpha, persona_layer)
                pmf, _ = ssr_score(resp, encoder, anchor_embs)
                pmfs.append(pmf)
            steered_pmfs[(qi, l)] = np.mean(pmfs, axis=0)

    # Measure steered inter-cluster JS
    steered_js = []
    for qi in range(len(sample_qs)):
        clusters_with_data = [l for l in persona_vectors if (qi, l) in steered_pmfs]
        for i in range(len(clusters_with_data)):
            for j in range(i+1, len(clusters_with_data)):
                p = steered_pmfs[(qi, clusters_with_data[i])]
                q = steered_pmfs[(qi, clusters_with_data[j])]
                steered_js.append(jensenshannon(p, q)**2)

    if steered_js:
        print(f"  Steered inter-cluster JS: mean={np.mean(steered_js):.4f} "
              f"max={np.max(steered_js):.4f}", flush=True)
    else:
        print(f"  No steered divergence computed", flush=True)

    return {
        "n_clusters": n_clusters,
        "sizes": sizes,
        "signature_js_mean": float(np.mean(pairwise_js)),
        "signature_js_max": float(np.max(pairwise_js)),
        "steered_js_mean": float(np.mean(steered_js)) if steered_js else None,
        "steered_js_max": float(np.max(steered_js)) if steered_js else None,
        "persona_vectors": {str(l): vec.tolist() for l, vec in persona_vectors.items()},
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--layer", type=int, default=24)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--method", choices=["ssr_signature", "hidden_state", "both"],
                        default="ssr_signature")
    args = parser.parse_args()

    Path(RESULTS_DIR).mkdir(exist_ok=True)

    # Load data
    df = pd.read_csv(f"{DATA_DIR}/2_meaningful_df.csv")
    posts = df["content_desc"].dropna().tolist()
    print(f"Loaded {len(posts)} posts", flush=True)

    with open(f"{RESULTS_DIR}/all_questions_expanded.json") as f:
        expanded = json.load(f)
    questions = []
    for sk, sv in expanded.items():
        for sub, qs in sv.items():
            for qid, qd in qs.items():
                questions.append({
                    "key": f"{sub}/{qid}",
                    "question": qd["question"],
                    "options": qd["options"],
                    "true_distribution": qd["true_distribution"],
                })

    # Load models
    print("Loading models...", flush=True)
    encoder = SentenceTransformer(EMBEDDING_MODEL)
    model, tokenizer = load_model(MODEL_PATH, args.device)

    # Use subset of questions for signatures (to keep dimension manageable)
    # Pick 6 diverse questions
    sig_questions = [questions[i] for i in [0, 2, 4, 8, 12, 16]]
    print(f"Using {len(sig_questions)} questions for response signatures", flush=True)

    # Compute signatures
    signatures, anchor_cache = compute_post_signatures(
        model, tokenizer, encoder, posts, sig_questions, layer=args.layer,
    )
    print(f"Signature matrix: {signatures.shape}", flush=True)

    # Cluster
    print("\n=== Behavioral Clustering ===", flush=True)
    cluster_results = behavioral_kmeans(signatures, n_clusters_list=[5, 10, 15, 20])

    # Compare with original topic clusters
    orig_labels = df["cluster_label"].values[:len(posts)]
    orig_n = len(set(orig_labels))

    print(f"\nOriginal topic clusters: {orig_n} clusters", flush=True)

    # Evaluate each clustering
    all_eval = {}
    for k, labels in sorted(cluster_results.items()):
        print(f"\n--- Behavioral k={k} ---", flush=True)
        ev = evaluate_clustering(
            labels, df.iloc[:len(posts)], signatures, sig_questions, encoder,
            anchor_cache, model, tokenizer,
            persona_layer=args.layer, alpha=args.alpha,
        )
        all_eval[k] = ev

    # Also evaluate original clusters (mapped to the same posts)
    print(f"\n--- Original topic clusters (k={orig_n}) ---", flush=True)
    ev_orig = evaluate_clustering(
        orig_labels, df.iloc[:len(posts)], signatures, sig_questions, encoder,
        anchor_cache, model, tokenizer,
        persona_layer=args.layer, alpha=args.alpha,
    )
    all_eval["original"] = ev_orig

    # Summary
    print("\n" + "=" * 70, flush=True)
    print("SUMMARY: Inter-cluster Divergence Comparison", flush=True)
    print("=" * 70, flush=True)
    print(f"  {'Clustering':<20} {'Sig JS':>8} {'Steered JS':>12} {'N clusters':>10}", flush=True)
    print("  " + "-" * 55, flush=True)
    for k, ev in sorted(all_eval.items(), key=lambda x: str(x[0])):
        label = f"Original ({orig_n})" if k == "original" else f"Behavioral (k={k})"
        sjs = f"{ev['signature_js_mean']:.4f}" if ev['signature_js_mean'] else "N/A"
        stjs = f"{ev['steered_js_mean']:.4f}" if ev.get('steered_js_mean') else "N/A"
        print(f"  {label:<20} {sjs:>8} {stjs:>12} {ev['n_clusters']:>10}", flush=True)

    # Save
    out = f"{RESULTS_DIR}/behavioral_clustering.json"
    # Don't save persona vectors (too large), just metadata
    save_data = {}
    for k, ev in all_eval.items():
        ev_copy = {key: val for key, val in ev.items() if key != "persona_vectors"}
        save_data[str(k)] = ev_copy
    save_data["cluster_labels"] = {str(k): labels.tolist() for k, labels in cluster_results.items()}

    with open(out, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved to {out}", flush=True)


if __name__ == "__main__":
    main()
