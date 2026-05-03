# CLAUDE.md — project context for Claude Code

This file is auto-loaded by Claude Code at the start of every session in this
repo, so the next Claude session on a new machine sees it automatically. The
local `~/.claude/projects/.../memory/` does NOT follow the user across PCs;
this file is the durable, cross-machine copy of project state.

A backup also lives on the chen server at `~/CLAUDE.md` (SSH key
`~/.ssh/id_server_chen`).

---

## Project state — PS-SSR (last update 2026-05-03)

**Repo:** `github.com:mzyy1001/ssr_experiment.git`, branch `main`, last commit
`f33edec` ("Consolidate PS-SSR pipeline + add baselines, ablations, and reports").

**Headline:** M2 (cluster-level ensemble of Direct-SSR + LLM-dist, w=0.5)
achieves **mean JS = 0.0254 ± 0.0004** over 5 seeds on the 6Q real benchmark
(6 Chinese consumer-health questions about TCM 藿香正气水).

**Baseline ladder (no-social-media → full method):**
| Method | Mean JS | Δ vs prev |
|---|---:|---|
| Paper-SSR (Maier-style transfer; LLM-as-consumer + SSR) | 0.0430 | — |
| LLM-direct distribution (no posts, hardened parser) | 0.0316 | LLM training prior alone |
| Flat SSR (real posts, no clusters) | 0.0295 | small grounding gain |
| M0 (cluster-mass-weighted SSR) | 0.0269 | clustering helps |
| **M2 (M0 + M1 ensemble, w=0.5)** | **0.0254** | headline |

**Stability:** σ(M2) = 0.0004 over 5 seeds. M2 vs M0 gap ≈ 4.7σ.

## Pipeline (use this for any new questionnaire data)

`pssr_pipeline.py` at the repo root — single-file end-to-end implementation.

```python
from pssr_pipeline import PSSRPipeline
pipe = PSSRPipeline(device="cuda:0")
results = pipe.run_questionnaire(questions, posts_df, cluster_topics)
# Auto-computes cluster_weights if missing.
```

Or CLI:
```bash
python pssr_pipeline.py predict \
  --questions new_questionnaire.json \
  --posts data/2_meaningful_df.csv \
  --topics data/3_cluster_topics.json \
  --output predictions.json
```

Validated hyperparameters (don't re-tune without reason): `top_k=10`,
`n_posts=50`, `n_samples=3`, `ensemble_w=0.5`, SSR `temp=0.1`, embedders
`bge-small-zh-v1.5` (cluster) + `bge-base-zh-v1.5` (SSR), LLM Qwen3-8B bf16.

## Setup on the new machine

1. **Clone:**
   ```
   git clone git@github.com:mzyy1001/ssr_experiment.git business
   ```
2. **Copy SSH key for chen server** (NOT in repo):
   ```
   # from old machine
   scp ~/.ssh/id_server_chen <new_machine>:~/.ssh/
   chmod 600 ~/.ssh/id_server_chen
   ```
3. **Copy `data/`** (gitignored — `taiji_all.csv` is ~4M rows, too big for git):
   ```
   rsync -av <old_machine>:/home/mzyy1001/business/data/ ./data/
   ```
   Or pull from chen: `scp -i ~/.ssh/id_server_chen -P 2222 -r chenhongrui@122.225.39.134:/data/chenhongrui/business/data ./data`
4. **Test locally** (no GPU needed for parsing):
   ```
   python -c "from pssr_pipeline import parse_distribution; print(parse_distribution('{\"distribution\": [0.2, 0.3, 0.5]}', 3))"
   ```

## GPU compute on chen server

- Host: `chenhongrui@122.225.39.134:2222`, key `~/.ssh/id_server_chen`.
- Conda env to activate: **`qiqi_rl_gpu`** (NOT `llama_qwen` — the older `experiments/config.py` says `llama_qwen` but every recent run actually used `qiqi_rl_gpu`).
- Activate with: `source /data/anaconda3/etc/profile.d/conda.sh && conda activate qiqi_rl_gpu`.
- Server-side workdir: `/data/chenhongrui/business/`. Mirror via rsync if you want files in sync.
- **GPU constraint heard 2026-04-29:** "use GPU 0; do not use GPU 2/5/6/7." Re-confirm with user before launching anything heavy. GPUs are shared with other users, check `nvidia-smi` first.

## Key files

| File | Role |
|---|---|
| `pssr_pipeline.py` | The reusable pipeline. Library + CLI. |
| `experiments/dump_m0_m2_pmfs.py` | Canonical M0/M1/M2 dump (matches headline 0.0254). |
| `experiments/dump_c2_hardened.py` | C2 with multi-strategy JSON parser. |
| `experiments/run_paper_ssr_6q.py` | Paper-SSR (Maier-style) baseline. |
| `experiments/run_flat_ssr_6q.py` | Flat-SSR control (no clusters). |
| `experiments/run_llm_direct_hardened.py` | LLM-only direct-distribution baseline. |
| `experiments/run_m0m2_seeds.py` + `aggregate_seeds.py` | 5-seed stability. |
| `experiments/run_m2_topk_sweep.py` | top-K sweep (K ∈ {5, 10, 15, 20}). |
| `experiments/quarantine_23q/` | 23Q work — IGNORE (LLM-generated questions, not real surveys). |
| `report_2x2.pdf` (+ `.tex`) | Diagnostic ablation report (4 pages). |
| `methods_results_summary.pdf` (+ `.tex`) | Comprehensive method catalog. |
| `data_methods_doc.pdf` (+ `.md`) | Data cleaning, clustering, prompts. |
| `STORY_v2.md` | Research narrative (Round 5 of GPT-5.4 review, 7/10). |
| `survey_ssr_exp.ipynb` | Original notebook (cells 1–8 cleaning, 12–22 clustering, 26–31 prompts + cluster_weights derivation). |
| `data/1_filter_df.csv` | After filter+dedup (27,838 posts). |
| `data/2_meaningful_df.csv` | After HDBSCAN noise removal (11,873 posts, 66 clusters). |
| `data/3_cluster_topics.json` | One-sentence Chinese topic per cluster. |
| `results/all_questions_expanded.json` | 6Q ground truth + per-question cluster_weights. |

## Known open threads

1. **New questionnaire data is coming.** User flagged this on 2026-05-03. The pipeline accepts new questions in either of two formats — flat list or the original nested `{survey_id: {sub: {qid: ...}}}` schema. Cluster weights are auto-computed if missing.
2. **Paper-SSR baseline reimplementation may understate Maier's true performance** — current implementation uses a single generic identity stub × 100 samples; Maier's paper samples from a demographic grid (age × gender × region). If the user revisits the baseline, this is the gap to close. Not currently flagged as blocking.
3. **Top-K sweep** showed K=10 ties K=15 on the mean (0.0250 vs 0.0249) but K=15 has a Q5-specific regression (+27%); K=10 is the safe default.

## Persistent user preferences

- Don't poll long-running experiments — use background monitors and wait for notifications. (Memorialized in `feedback_no_polling.md` originally.)
- 6Q is the canonical benchmark. Ignore the 23Q expansion (LLM-generated, not real survey data).
- Reports default to PDF. The local pdflatex doesn't have CJK; for Chinese-content docs, use `weasyprint` via the Markdown→HTML→PDF route (see `build_methods_pdf.py`).
- Match scope to what was asked. The user pushed back at least once on adding methods to the 2×2 ablation that weren't in the canonical method family — keep M2 as the headline, frame ablations as supporting evidence.
