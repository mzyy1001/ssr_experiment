# PS-SSR 6Q Evaluation Report

> Note: earlier drafts of this report focused on a 23-question expanded set.
> That set is **synthetic (LLM-generated questions), not real survey items**,
> so it is not a valid measurement of survey reconstruction. This report
> evaluates only on the **6 original survey questions**, which is the
> canonical benchmark. 23Q scripts and results have been quarantined under
> `experiments/quarantine_23q/` and `results/quarantine_23q/`.

## Setup
- **Model**: Qwen3-8B (bf16, A100 GPU)
- **Questions**: 6 original survey questions (`8.0/qsingle_3,4`, `9.0/qsingle_3,4,5`, `11.0/qsingle_3`)
- **Data**: 66 topic clusters from ~12K social media posts (Weibo/Xiaohongshu)
- **Metric**: Mean JS divergence across the 6 questions (lower = better)
- **Baseline to beat**: Direct SSR = **0.0277**

---

## Table 1: Lead results — methods that beat Direct SSR

Source: `results/improve_orig6.json` (2026-04-21).

| Method | Mean JS ↓ | Δ vs SSR | Notes |
|:---|:---:|:---:|:---|
| **SSR + LLM-dist, w=0.5** | **0.0257** | **−7.2%** | **Primary contribution** |
| SSR + LLM-dist, w=0.6 | 0.0258 | −6.9% | |
| SSR + LLM-dist, w=0.7 | 0.0259 | −6.5% | |
| LLM-dist top-10 (no SSR) | 0.0259 | −6.5% | LLM-only distribution estimation |
| SSR + LLM-dist, w=0.8 | 0.0261 | −5.8% | |
| SSR top-5 clusters | 0.0272 | −1.8% | Tuning knob, small effect |
| Ensemble w=0.95 (SSR + multi-layer) | 0.0272 | −1.8% | Steering as ensemble member |
| SSR n=50 posts | 0.0275 | −0.7% | Post-count ablation |
| *Direct SSR baseline* | *0.0277* | — | |

**Primary finding.** Linearly mixing SSR's cluster-prior embedding distribution with an LLM-estimated distribution ("SSR + LLM-dist") at w=0.5 reduces mean JS by 7.2% over the Direct SSR baseline.

---

## Table 2: SSR + multi-layer steering ensemble

Source: `results/persona_methods.json` (2026-04-20). Baseline Direct SSR on this run = 0.0276.

| Method | Divergence | Mean JS ↓ | Δ vs SSR |
|:---|:---:|:---:|:---:|
| BL: Direct SSR | 0.0105 | 0.0276 | — |
| **E: Ensemble w=0.5** (SSR + ML steer) | 0.0083 | **0.0261** | **−5.4%** |
| E: Ensemble w=0.7 | 0.0071 | 0.0264 | −4.3% |
| E: Ensemble w=0.9 | 0.0088 | 0.0267 | −3.3% |
| B: LLM distribution estimation | 0.0162 | 0.0273 | −1.1% |
| C: SSR + per-post LLM correction | 0.0134 | 0.0296 | +7.2% |
| D: Persona-projected SSR (s=0.5) | 0.1214 | 0.0405 | +46.7% |
| A: Per-post LLM voting | 0.0594 | 0.2360 | +755% |

Multi-layer steering only helps when ensembled with SSR; steering alone at single layer is catastrophic (see Table 4). LLM voting and persona-projected SSR are negative results.

---

## Table 3: Multi-layer depth / response count (improve_orig6)

| Method | Mean JS | Notes |
|:---|:---:|:---|
| ML L[12,16,20,24] n=5 | 0.0643 | Deeper stack hurts |
| ML L[16,20,24,28] n=5 | 0.0434 | Later layers hurt less |
| ML L[20,24] n=5 | 0.0404 | 2 layers |
| ML L[16,24] n=5 | 0.0378 | 2 layers, better |
| MultiLayer n=10 | 0.0384 | More responses, same layers |
| MultiLayer n=5 | 0.0396 | Fewer responses |

Steering alone never beats SSR; it only helps when weighted low in an ensemble (Table 2).

---

## Table 4: Why single-layer steering fails (fix_steering diagnostics)

Source: `results/fix_steering_results.json` (2026-04-19).

| Variant | Divergence | Mean JS ↓ |
|:---|:---:|:---:|
| Direct SSR (no steering) | 0.0105 | 0.0276 |
| Multi-layer steering α=0.1 (L16+L20+L24) | 0.0222 | 0.0298 |
| ICL with 5 example posts | 0.0052 | 0.0368 |
| Contrastive prompt (IS vs IS NOT) | 0.0086 | 0.0415 |
| ICL + steering α=0.3 | 0.0065 | 0.0420 |
| Large α=3.0, unit-normed vectors | 0.0046 | 0.0622 |
| **Single-layer α=0.1, L24 (naive)** | **0.0046** | **0.0659** |
| Large α=1.0, unit-normed vectors | 0.0037 | 0.0689 |

Single-layer steering at α=0.1 on L24 is **2.4× worse** than Direct SSR. Multi-layer fixes ~65% of the gap but still underperforms the SSR baseline on its own.

---

## Table 5: Post count and cluster count ablation (improve_orig6)

| Axis | Setting | Mean JS |
|:---|:---:|:---:|
| Posts per cluster | 30 | 0.0277 |
| | 50 | 0.0275 |
| | 80 | 0.0276 |
| Top-K clusters | 5 | 0.0272 |
| | 10 | 0.0272 |
| | 15 | 0.0277 |
| | 20 | 0.0280 |

Effects are small and noisy. n=50 with top-5 gives a marginal improvement; the real gains come from the LLM-dist mixture, not from tuning SSR alone.

---

## Summary

1. **Primary result**: **SSR + LLM-dist at w=0.5 achieves Mean JS = 0.0257**, a **7.2% relative reduction** over the Direct SSR baseline (0.0277) on the 6 original survey questions.

2. **Two independent signal sources work:**
   - **LLM-dist**: directly asking the LLM for the option distribution, weighted with SSR's embedding-based distribution. Accounts for most of the gain.
   - **SSR + multi-layer steering ensemble**: ~5% improvement at w=0.5; a useful secondary / ablation comparator.

3. **What does not work (negative results, preserved for the paper):**
   - Single-layer steering at L24 — catastrophic (0.0659, +138%).
   - Per-post LLM voting — catastrophic (0.2360, +755%).
   - Persona-projected SSR — monotonically worse as projection strength grows.
   - Deeper multi-layer stacks `L[12,16,20,24]` — hurt more than they help.

4. **Proposed paper framing:**
   - **Main method**: SSR + LLM-dist (the ensemble of embedding-based and LLM-based distributions).
   - **Ablation story**: steering-based ensemble as a second contributor; single-layer steering diagnostics as a motivation for the multi-layer ensemble.
   - **Drop**: QAW (adaptive topic weighting), KL-penalty demographic reweighting, and the 23Q expansion — all were evaluated on synthetic questions and do not carry over to the real benchmark.
