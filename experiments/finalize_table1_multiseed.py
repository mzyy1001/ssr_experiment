"""
Final aggregator: pull seeds 42/123/456/789 shards, compute per-cell mean
± seed-std on the 13Q expanded benchmark, and patch Table 2 of
paper_position_v2.tex with the multi-seed numbers.

Run as:
    python3 finalize_table1_multiseed.py [--seeds 42 123 456 789]

Reads:
  - results/table1_phaseA_full_13q.json (seed=42 full 6-method per-Q)
  - results/table1/seed{S}_s{S}_shard{0,2,7}.json for each S in --seeds (m1, c2)
Writes:
  - results/table1_multiseed_summary.json (full per-method × per-seed × per-Q)
  - patches nips_src/paper_position_v2.tex (Table 2 numbers)
"""
import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np


REPO = "/home/mzyy1001/business"
RESULTS_DIR = f"{REPO}/results"
TABLE1_DIR = f"{RESULTS_DIR}/table1"
PAPER = f"{REPO}/nips_src/paper_position_v2.tex"


def fmt(mean, std):
    return f"${mean:.4f} \\pm {std:.4f}$"


def fmt_bold(mean, std):
    return f"$\\mathbf{{{mean:.4f} \\pm {std:.4f}}}$"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789])
    args = ap.parse_args()

    # Per-method × per-seed × per-question JS values
    # Methods: paper_ssr / direct_llm / flat_ssr / m0 / m1 / c2
    data = defaultdict(lambda: defaultdict(dict))  # data[method][seed][qkey] = js

    # Seed 42 has all 6 methods (full Phase A)
    if 42 in args.seeds:
        full = json.load(open(f"{RESULTS_DIR}/table1_phaseA_full_13q.json"))
        for method, qd in full["per_question_js"].items():
            for k, v in qd.items():
                data[method][42][k] = float(v)

    # Other seeds: only m1 / c2 (and m0 since the runner saves it)
    for s in args.seeds:
        if s == 42:
            continue
        for fp in sorted(glob.glob(f"{TABLE1_DIR}/seed{s}_s{s}_shard*.json")):
            d = json.load(open(fp))
            for k, v in d.get("per_question", {}).items():
                for method in ["m0", "m1", "c2"]:
                    js = v.get("metrics", {}).get(method, {}).get("js")
                    if js is not None:
                        data[method][s][k] = float(js)

    # Sanity: which methods have which seeds?
    print(f"\n=== Seed coverage per method ===")
    for method in ["paper_ssr", "direct_llm", "flat_ssr", "m0", "m1", "c2"]:
        seeds_have = sorted(data[method].keys())
        print(f"  {method:<12}  seeds={seeds_have}")

    # Per-question JS averaged across seeds → mean ± across-question std
    # Plus across-seed std of the per-question-mean (cell-level instability)
    print(f"\n=== Cell-level summary ===")
    print(f"{'method':<12} {'seeds':<22} {'mean':>8} {'q-std':>8} {'seed-std':>10}")
    cell_summary = {}
    for method in ["paper_ssr", "direct_llm", "flat_ssr", "m0", "m1", "c2"]:
        seeds_have = sorted(data[method].keys())
        if not seeds_have:
            continue
        # Collect all (seed, q, js)
        keys = sorted(data[method][seeds_have[0]].keys())
        # Per-Q mean across seeds
        per_q_means = []
        per_seed_means = []
        for s in seeds_have:
            vals = [data[method][s][k] for k in keys if k in data[method][s]]
            per_seed_means.append(np.mean(vals))
        for k in keys:
            qvals = [data[method][s][k] for s in seeds_have if k in data[method][s]]
            per_q_means.append(np.mean(qvals))
        a_q = np.array(per_q_means)
        a_s = np.array(per_seed_means)
        cell_summary[method] = {
            "n_seeds": len(seeds_have),
            "seeds": seeds_have,
            "n_questions": len(keys),
            "mean": float(a_q.mean()),
            "std_across_questions": float(a_q.std(ddof=1)) if len(a_q) > 1 else 0.0,
            "std_across_seeds_of_means": float(a_s.std(ddof=1)) if len(a_s) > 1 else 0.0,
            "per_seed_mean": [float(x) for x in a_s],
            "per_q_mean_across_seeds": {k: float(v) for k, v in zip(keys, per_q_means)},
        }
        print(f"  {method:<12} {str(seeds_have):<22} "
              f"{a_q.mean():>8.4f} {a_q.std(ddof=1):>8.4f} "
              f"{a_s.std(ddof=1) if len(a_s) > 1 else 0:>10.4f}")

    # Save full summary
    with open(f"{RESULTS_DIR}/table1_multiseed_summary.json", "w") as f:
        json.dump(cell_summary, f, indent=2, ensure_ascii=False)
    print(f"\nSaved → {RESULTS_DIR}/table1_multiseed_summary.json")

    # Patch Table 2 in paper_position_v2.tex
    if os.path.exists(PAPER):
        with open(PAPER, encoding="utf-8") as f:
            tex = f.read()
        # Replace the row values inside Table 2 (the 13Q table). Table 2 is unique
        # because of the surrounding label `tab:2by2_13q`.
        # Compose the new rows.
        ms = cell_summary
        seed_set = sorted(set().union(*[set(ms[m]["seeds"]) for m in ms if "seeds" in ms[m]]))
        sem_struct_text = fmt_bold(ms["m0"]["mean"], ms["m0"]["std_across_questions"])
        llm_struct_text = fmt_bold(ms["m1"]["mean"], ms["m1"]["std_across_questions"])
        new_block = (
            f"Semantic mapping & "
            f"{fmt(ms['paper_ssr']['mean'], ms['paper_ssr']['std_across_questions'])} & "
            f"{fmt(ms['flat_ssr']['mean'], ms['flat_ssr']['std_across_questions'])} & "
            f"{sem_struct_text} \\\\\n"
            f"LLM simulation   & "
            f"{fmt(ms['direct_llm']['mean'], ms['direct_llm']['std_across_questions'])} & "
            f"{fmt(ms['c2']['mean'], ms['c2']['std_across_questions'])} & "
            f"{llm_struct_text} \\\\"
        )
        # Find the table block by its label and replace the two data rows
        pat = re.compile(
            r"(\\midrule\n)Semantic mapping & .+? \\\\\nLLM simulation\s+& .+? \\\\\n",
            re.DOTALL,
        )
        # Constrain to the 13Q table by anchoring on tab:2by2_13q label
        # Strategy: split the file at tab:2by2_13q's table block, patch only that block.
        m = re.search(r"\\label\{tab:2by2_13q\}", tex)
        if m:
            # Find begin/end of that table by walking backwards to \begin{table}
            start = tex.rfind("\\begin{table}", 0, m.start())
            end = tex.find("\\end{table}", m.start())
            if start != -1 and end != -1:
                block = tex[start:end]
                new_block_full = pat.sub(r"\1" + new_block + "\n", block, count=1)
                if new_block_full != block:
                    tex = tex[:start] + new_block_full + tex[end:]
                    # Also patch the caption to reflect seed count + range
                    seed_str = "/".join(str(s) for s in seed_set)
                    n_seeds_str = f"{len(seed_set)} seeds"
                    cap_pat = re.compile(
                        r"(Aggregated comparison on the expanded 13-question benchmark, reported)\s*\n?\s*as mean.+?\s*standard deviation across questions\.",
                        re.DOTALL,
                    )
                    new_cap = (
                        f"\\1 as mean$\\\\,\\\\pm\\\\,$standard deviation across the "
                        f"$n=13$ questions, with each cell averaged over {len(seed_set)} "
                        f"random seeds (\\\\{{{seed_str}\\\\})."
                    )
                    tex = cap_pat.sub(new_cap, tex, count=1)
                    with open(PAPER, "w", encoding="utf-8") as f:
                        f.write(tex)
                    print(f"Patched Table 2 in {PAPER}")
                else:
                    print("WARNING: could not match Table 2 row pattern in tex")
            else:
                print("WARNING: could not locate Table 2 begin/end in tex")
        else:
            print("WARNING: tab:2by2_13q label not found in tex")
    else:
        print(f"WARNING: {PAPER} not found")


if __name__ == "__main__":
    main()
