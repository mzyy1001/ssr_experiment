"""
Aggregate per-question C2-hardened shards into a single 6Q summary, and
diff against the original (un-hardened) C2 results in b1_c2_pmfs_6q.json.

Reads:  results/c2_hardened_6q_q{0..5}.json (per-Q shards)
        results/b1_c2_pmfs_6q.json          (original C2)
Writes: results/c2_hardened_summary.json
"""
import glob
import json
import os

import numpy as np


SHARD_GLOB = "results/c2_hardened_6q_q*.json"
ORIG_PATH = "results/b1_c2_pmfs_6q.json"
OUT = "results/c2_hardened_summary.json"


def main():
    shards = sorted(glob.glob(SHARD_GLOB))
    if not shards:
        raise SystemExit(f"No shards at {SHARD_GLOB}")
    print(f"Found {len(shards)} shards.")
    merged = {"per_question": {}, "parse_stats": {}}
    for fp in shards:
        with open(fp) as f:
            d = json.load(f)
        for k, v in d["per_question"].items():
            merged["per_question"][k] = v
            merged["parse_stats"][k] = d["parse_stats"][k]

    js_all = [v["c2_hardened"]["js"] for v in merged["per_question"].values()]
    k_all = [v["c2_hardened"]["k_xy"] for v in merged["per_question"].values()]
    c_all = [v["c2_hardened"]["c_xy"] for v in merged["per_question"].values()]
    merged["summary"] = {
        "C2_hardened": {
            "js": float(np.mean(js_all)),
            "k_xy": float(np.mean(k_all)),
            "c_xy": float(np.mean(c_all)),
        }
    }

    # Diff vs original
    diff_table = []
    if os.path.exists(ORIG_PATH):
        with open(ORIG_PATH) as f:
            orig = json.load(f)
        orig_pq = orig["per_question"]
        orig_stats = {s["q"]: s for s in orig.get("c2_parse_stats", [])}
        for k in merged["per_question"]:
            new = merged["per_question"][k]
            old = orig_pq.get(k, {})
            old_js = old.get("c2", {}).get("js")
            new_js = new["c2_hardened"]["js"]
            old_fail = orig_stats.get(k, {}).get("n_fail")
            new_fail = new["parse_stats"]["fail"]
            ntot = new["n_total"]
            diff_table.append({
                "question": k, "old_js": old_js, "new_js": new_js,
                "delta_js": new_js - old_js if old_js is not None else None,
                "old_fail": old_fail, "new_fail": new_fail, "n_total": ntot,
                "old_fail_rate": old_fail / 500 if old_fail is not None else None,
                "new_fail_rate": new_fail / ntot if ntot else None,
            })
        merged["diff_vs_original"] = diff_table

    with open(OUT, "w") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    print(f"Saved → {OUT}")
    print()
    print(f"C2 hardened mean JS = {merged['summary']['C2_hardened']['js']:.4f} "
          f"(orig was 0.0259)")
    print()
    print(f"{'Question':<22} {'old JS':>8} {'new JS':>8} {'Δ JS':>8} "
          f"{'old fail':>8} {'new fail':>8}")
    for r in diff_table:
        old_js_str = f"{r['old_js']:.4f}" if r['old_js'] is not None else "  --  "
        delta_str = f"{r['delta_js']:+.4f}" if r['delta_js'] is not None else "  --  "
        old_fail_str = f"{r['old_fail_rate']*100:5.1f}%" if r['old_fail_rate'] is not None else " --  "
        new_fail_str = f"{r['new_fail_rate']*100:5.1f}%" if r['new_fail_rate'] is not None else " --  "
        print(f"{r['question']:<22} {old_js_str:>8} {r['new_js']:.4f}   "
              f"{delta_str:>8}   {old_fail_str:>8} {new_fail_str:>8}")


if __name__ == "__main__":
    main()
