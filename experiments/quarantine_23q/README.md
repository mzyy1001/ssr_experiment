# Quarantined 23Q experiments

These scripts evaluate on the **23-question expanded set**, which consists of
LLM-generated (synthetic) questions — not real survey items. They are kept
here for history but are **not part of the canonical PS-SSR benchmark** and
should not be used for paper claims.

The real benchmark is the 6 original survey questions. See
`../improve_orig6.py`, `../persona_methods.py`, `../fix_steering.py`,
`../behavioral_cluster.py` for the scripts still in active use, and
`../../EXPERIMENT_REPORT_QAW.md` / `../../report_qaw.pdf` for the current
results.

## Contents

| Script | Purpose |
|---|---|
| `run_full_eval.py` | 23Q full eval across SSR / steering / KL / QAW |
| `run_qaw.py` | 23Q QAW (adaptive topic weighting) |
| `run_qaw_expanded.py` | 23Q QAW on expanded question set |
| `run_qaw_sweep.py` | 23Q QAW robustness sweep across (α, layer) |
| `run_kl_deep.py` | 23Q KL-penalty deep sweep |
| `run_demo_qaw.py` | 23Q demographic QAW |
| `adaptive_weights.py` | QAW weight computation helpers |
| `demographic_reweight.py` | KL-penalty demographic correction |
| `infer_demographics.py` | Cluster demographic inference |
| `analyze_problem.py` | 23Q-specific diagnostics |

## Why quarantined (2026-04-22)

On the 23Q synthetic set no method significantly beats Direct SSR (all
bootstrap p > 0.12). The earlier reported gains from QAW and KL-penalty
(~3% relative) do not reflect performance on real survey reconstruction;
they reflect agreement with another LLM's prior over synthetic questions.
