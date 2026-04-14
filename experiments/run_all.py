"""
Main experiment runner for PS-SSR research.
Orchestrates all phases: persona extraction → baselines → steering → evaluation.

Run on chen server:
  source /data/anaconda3/etc/profile.d/conda.sh && conda activate llama_qwen
  cd /home/mzyy1001/business/experiments
  python run_all.py --phase all --device cuda:2
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path


MODEL_PATH = "/data/chenhongrui/models/Qwen3-8B"
DATA_DIR = "/data/chenhongrui/business/data"
RESULTS_DIR = "/data/chenhongrui/business/results"
EMBEDDING_MODEL = "/data/chenhongrui/models/bge-base-zh-v1.5"


def run_phase(phase: str, device: str = "cuda:2", **kwargs):
    """Run a specific experiment phase."""

    if phase == "extract":
        print("\n" + "=" * 60)
        print("Phase 1: Extracting persona vectors")
        print("=" * 60)
        from persona_vectors import (
            load_model, extract_persona_vectors, save_persona_vectors,
        )
        import pandas as pd

        model, tokenizer = load_model(MODEL_PATH, device)
        df = pd.read_csv(f"{DATA_DIR}/2_meaningful_df.csv")
        with open(f"{DATA_DIR}/3_cluster_topics.json") as f:
            topics = json.load(f)

        layer = kwargs.get("layer", 16)
        n_samples = kwargs.get("n_samples", 20)

        vectors = extract_persona_vectors(
            model, tokenizer, df, topics,
            layer=layer, n_samples=n_samples,
        )
        save_persona_vectors(vectors, f"{RESULTS_DIR}/persona_vectors_L{layer}_N{n_samples}.npz")
        print("Phase 1 complete.")

    elif phase == "baseline_direct":
        print("\n" + "=" * 60)
        print("Baseline: Direct LLM")
        print("=" * 60)
        cmd = [
            sys.executable, "baselines.py",
            "--model_path", MODEL_PATH,
            "--method", "direct",
            "--device", device,
        ]
        subprocess.run(cmd, check=True)

    elif phase == "baseline_persona":
        print("\n" + "=" * 60)
        print("Baseline: Persona Prompt")
        print("=" * 60)
        cmd = [
            sys.executable, "baselines.py",
            "--model_path", MODEL_PATH,
            "--method", "persona_prompt",
            "--device", device,
            "--embedding_model", EMBEDDING_MODEL,
        ]
        subprocess.run(cmd, check=True)

    elif phase == "ps_ssr":
        print("\n" + "=" * 60)
        print("Phase 3: PS-SSR (Steer-then-aggregate)")
        print("=" * 60)
        layer = kwargs.get("layer", 16)
        alpha = kwargs.get("alpha", 2.0)
        n_samples = kwargs.get("n_samples", 20)
        strategy = kwargs.get("strategy", "steer_then_aggregate")

        cmd = [
            sys.executable, "steered_ssr.py",
            "--model_path", MODEL_PATH,
            "--vectors_path", f"{RESULTS_DIR}/persona_vectors_L{layer}_N{n_samples}.npz",
            "--strategy", strategy,
            "--alpha", str(alpha),
            "--layer", str(layer),
            "--device", device,
            "--embedding_model", EMBEDDING_MODEL,
            "--output", f"{RESULTS_DIR}/ps_ssr_{strategy}_L{layer}_A{alpha}.json",
        ]
        subprocess.run(cmd, check=True)

    elif phase == "evaluate":
        print("\n" + "=" * 60)
        print("Evaluation: Comparing all methods")
        print("=" * 60)
        from evaluate import compare_methods
        from pathlib import Path

        methods = {}
        # Baseline SSR (Project 1)
        if Path(f"{RESULTS_DIR}/eps_0.1.json").exists():
            methods["SSR-only (Project 1)"] = f"{RESULTS_DIR}/eps_0.1.json"
        # Direct LLM
        if Path(f"{RESULTS_DIR}/baseline_direct.json").exists():
            methods["Direct LLM"] = f"{RESULTS_DIR}/baseline_direct.json"
        # Persona prompt
        if Path(f"{RESULTS_DIR}/baseline_persona_prompt.json").exists():
            methods["Persona Prompt"] = f"{RESULTS_DIR}/baseline_persona_prompt.json"
        # PS-SSR variants
        for p in Path(RESULTS_DIR).glob("ps_ssr_*.json"):
            methods[f"PS-SSR ({p.stem})"] = str(p)

        if methods:
            comparisons = compare_methods(methods)
            with open(f"{RESULTS_DIR}/evaluation_comparison.json", "w") as f:
                json.dump(
                    {k: v["summary"] for k, v in comparisons.items()},
                    f, indent=2,
                )
            print(f"\nSaved comparison to {RESULTS_DIR}/evaluation_comparison.json")
        else:
            print("No result files found to evaluate.")

    elif phase == "ablation_layer":
        print("\n" + "=" * 60)
        print("Ablation: Layer sweep")
        print("=" * 60)
        for layer in [8, 12, 16, 20, 24]:
            print(f"\n--- Layer {layer} ---")
            run_phase("extract", device, layer=layer, n_samples=20)
            run_phase("ps_ssr", device, layer=layer, alpha=2.0, strategy="steer_then_aggregate")

    elif phase == "ablation_alpha":
        print("\n" + "=" * 60)
        print("Ablation: Steering strength sweep")
        print("=" * 60)
        for alpha in [0.5, 1.0, 2.0, 5.0, 10.0]:
            print(f"\n--- Alpha {alpha} ---")
            run_phase("ps_ssr", device, layer=16, alpha=alpha, strategy="steer_then_aggregate")

    elif phase == "all":
        run_phase("extract", device)
        run_phase("baseline_direct", device)
        run_phase("baseline_persona", device)
        run_phase("ps_ssr", device, strategy="steer_then_aggregate")
        run_phase("ps_ssr", device, strategy="aggregate_then_steer")
        run_phase("evaluate", device)

    else:
        print(f"Unknown phase: {phase}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PS-SSR Experiment Runner")
    parser.add_argument(
        "--phase",
        choices=[
            "extract", "baseline_direct", "baseline_persona",
            "ps_ssr", "evaluate",
            "ablation_layer", "ablation_alpha",
            "all",
        ],
        default="all",
    )
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--n_samples", type=int, default=20)
    parser.add_argument("--strategy", default="steer_then_aggregate")
    args = parser.parse_args()

    Path(RESULTS_DIR).mkdir(exist_ok=True)
    run_phase(
        args.phase, args.device,
        layer=args.layer, alpha=args.alpha,
        n_samples=args.n_samples, strategy=args.strategy,
    )
