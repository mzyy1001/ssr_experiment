"""
Post-process: aggregate per-seed M0/M1/M2 JSON files into a stability summary
with mean ± std for M0, M1, M2.

Reads:  results/seeds/m0m2_seed{seed}.json (one per seed)
Writes: results/stability_seeds_summary.json
"""
import argparse
import glob
import json
import os
import re

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds_dir", default="results/seeds")
    p.add_argument("--output", default="results/stability_seeds_summary.json")
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.seeds_dir, "m0m2_seed*.json")))
    if not files:
        raise SystemExit(f"No seed files in {args.seeds_dir}")

    runs = []
    for fp in files:
        with open(fp) as f:
            r = json.load(f)
        m = re.search(r"seed(\d+)\.json$", fp)
        seed = int(m.group(1)) if m else None
        runs.append({"seed": seed, "summary": r["summary"], "per_q": r["per_question"]})

    methods = ["M0", "M1", "M2"]
    seeds = [r["seed"] for r in runs]
    print(f"Aggregating {len(runs)} seed runs: seeds={seeds}")

    # Mean-JS over 6Q for each (method, seed)
    matrix = {m: [r["summary"][m] for r in runs] for m in methods}
    summary = {}
    for m in methods:
        vals = np.array(matrix[m])
        summary[m] = {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "min": float(vals.min()), "max": float(vals.max()),
            "per_seed": dict(zip([str(s) for s in seeds], [float(v) for v in vals])),
        }

    # Per-question stability for M2
    q_keys = list(runs[0]["per_q"].keys())
    per_q_m2 = {}
    for q in q_keys:
        vals = np.array([r["per_q"][q]["m2"]["js"] for r in runs])
        per_q_m2[q] = {"mean": float(vals.mean()),
                       "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                       "per_seed": dict(zip([str(s) for s in seeds],
                                             [float(v) for v in vals]))}

    out = {"n_seeds": len(runs), "seeds": seeds,
           "summary": summary, "per_q_m2": per_q_m2}
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved → {args.output}")
    print()
    print(f"{'Method':<8} {'mean':>8} {'± std':>8}  {'[min, max]':>20}")
    for m in methods:
        s = summary[m]
        print(f"{m:<8} {s['mean']:.4f} ± {s['std']:.4f}   "
              f"[{s['min']:.4f}, {s['max']:.4f}]")


if __name__ == "__main__":
    main()
