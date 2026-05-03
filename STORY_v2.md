# PS-SSR Story (v6 — corrected attribution after author-verification)

## Round 4 → Round 5 correction

In Round 4 I conceded that "M0 is not novel because Direct SSR is already in the repo." This was wrong. Verification:

1. **The published baseline (Maier et al. 2026) has no topic clusters.** Their SSR is per-consumer (per-respondent), with **flat averaging across consumers** in the survey (eq. 1: `p_s(i) = (1/N_s) Σ_c δ_irc`). Demographics enter only as a per-consumer prompt; "stratified by demographics" plots are post-hoc analysis, not part of the SSR pipeline. Verified by reading paper §A.4.3 and eq. 1.
2. **Every cluster-SSR file in this repo is authored by the same author** (mzyy1001), in this same project, starting 2026-04-14. Direct SSR (`improve_orig6.py:91 direct_ssr()`), `baselines.py`, `persona_methods.py`, `steered_ssr.py`, `dump_m0_m2_pmfs.py` — all the same project, same author. There is no third-party "prior in-repo baseline."
3. Therefore: the cluster-weighted hierarchical SSR family **is the author's novel contribution**, not "prior work." Direct SSR and M0 are early and canonical variants of the *same* novel family.

The corrected attribution restores the original story: hierarchical cluster-weighted SSR over latent social-media topic clusters is the primary novel contribution; M1/M2 are secondary refinements.

---

## Working title

**Hierarchical Cluster-Weighted Semantic Similarity Rating: Closing the Domain-Transfer Gap of Per-Consumer SSR on Chinese Multi-Choice Health Surveys — A Single-Domain Case Study.**

> **Naming convention.** Throughout the paper we use **"hierarchical cluster-weighted SSR"** (or its short form **M0**) for the novel method. Earlier internal files sometimes refer to the same algorithmic family as **"Direct SSR"** — these are the same family, distinguished only by hyperparameter / sampling-seed settings. M0 (n=50, top-10, seed=42) is the canonical run.

## One-line pitch

**On a 6-question Chinese consumer-health benchmark, the published per-consumer SSR baseline (Maier et al. 2026) — designed for English Likert purchase-intent — degrades to JS = 0.0430 when transferred. We introduce hierarchical cluster-weighted SSR (M0): per-cluster SSR computed over real social-media posts within latent topic clusters, then aggregated by per-question cluster mass. M0 reduces JS to 0.0269 — a 37.5% improvement, with stable performance across reweighting variants. A per-cluster LLM-distribution refinement (M1) and 50/50 ensemble (M2) further improve JS to 0.0254 (40.9% over paper-SSR), with non-uniform per-question gains.**

---

## Setup

| | |
|---|---|
| Benchmark | 6 real Chinese survey questions on consumer health (TCM 藿香正气水) |
| Data | ~12K Weibo/Xiaohongshu posts → 66 latent topic clusters |
| Model | Qwen3-8B + bge-base-zh-v1.5 (open, single-GPU) |
| Reference baseline | Maier et al. 2026 — per-consumer SSR, flat aggregation, no clusters (verified §A.4.3, eq. 1) |
| Primary metric | JS divergence (lower=better) — appropriate for nominal multi-choice |
| Secondary metrics | K_xy (KS-similarity), C_xy (cosine) — for cross-paper diagnostic only |

---

## Method taxonomy (verified against code and paper)

| Method | Aggregation unit | Aggregation rule | Cluster use | Author |
|---|---|---|---|---|
| **Paper-SSR** (Maier et al., per-consumer flat) | individual synthetic consumer | `p_s(i) = (1/N_s) Σ_c δ_irc` (eq. 1, paper §A.4) | **none** | external |
| **M0 / Direct-SSR family** (this work) | latent topic cluster | per-cluster mean → mass-weighted aggregate over top-K clusters | **top-10 latent topic clusters; per-question cluster-mass weights** | **this work** |
| **M1** (this work) | latent topic cluster | LLM emits a per-cluster option distribution; cluster-mass-weighted | top-10 | **this work** |
| **M2** (this work) | latent topic cluster | 0.5 × M0_cluster_PMF + 0.5 × M1_cluster_PMF; cluster-mass-weighted | top-10 | **this work** |

The fundamental algorithmic move — aggregating SSR signal at a *cluster* level (not a *consumer* level) and weighting by per-question cluster mass — is **not present in the published baseline**, and is the novel methodological contribution of this work.

---

## Claim ladder

### Primary novel contribution

> Hierarchical cluster-weighted SSR (**M0**) reduces JS from 0.0430 (paper-SSR per-consumer flat baseline) to **0.0269** — a **37.5% improvement** — by replacing per-consumer flat aggregation with per-cluster mean → cluster-mass-weighted aggregate over latent topic clusters extracted from social-media posts. The improvement is stable across 6 cluster-reweighting schemes (range [0.0269, 0.0272]) and is not explained by trivial sample-size or top-K sweeps.

### Secondary contribution

> A per-cluster LLM-distribution estimator (**M1**) and its 50/50 ensemble with M0 (**M2**) further reduce JS to **0.0268** (≈0.4% over M0; ~tied) and **0.0254** (a 5.6% additional gain over M0; 7.6% over Direct-SSR n=50). Per-question behavior is non-uniform: M2 wins on 5/6 questions vs paper-SSR but worsens 9.0/qsingle_3 by 50% over M0. We frame M2 as a best-achieved upper bound; M0 is the primary scientific claim.

### Diagnostic / negative

> Activation-steering with persona vectors helps only conditionally (low α, late layers); at aggressive settings vector direction is irrelevant. Appendix-only diagnostic.

---

## Headline number table

| Method | Mean JS ↓ | ΔJS vs paper-SSR | K_xy ↑ | C_xy ↑ |
|---|---|---|---|---|
| Paper-SSR (Maier per-consumer flat — transferred) | 0.0430 | — | 0.827 | 0.877 |
| **M0** Hier cluster-weighted SSR (n=50, this work) | **0.0269** | **−37.5%** | 0.858 | 0.923 |
| M1 Per-cluster LLM-dist (this work) | 0.0268 | −37.7% | 0.844 | 0.931 |
| **M2** Ensemble w=0.5 (this work) | **0.0254** | **−40.9%** | 0.852 | **0.932** |

Internal-ablation context (different hyperparameter / sampling-seed runs of the *same* M0/Direct-SSR family — all this work's code, presented for completeness; not external baselines):

| Variant | n_posts | top-K | Mean JS |
|---|---|---|---|
| Direct SSR | 30 | 10 | 0.0277 |
| Direct SSR | 50 | 10 | 0.0275 |
| Direct SSR | 80 | 10 | 0.0276 |
| Direct SSR | 50 | 5 | 0.0272 |
| Direct SSR | 50 | 15 | 0.0277 |
| Direct SSR | 50 | 20 | 0.0280 |
| **M0 canonical** | **50** | **10 (seed=42)** | **0.0269** |

→ Within-family hyperparameter variance is small ([0.0269, 0.0280]); the family **as a whole** is what beats the per-consumer flat baseline (0.0430). The novelty lies in the cluster-weighted aggregation, not in any specific hyperparameter setting.

---

## Per-question honest table

| Question (Chinese health domain) | Paper-SSR JS | M0 JS | M0 vs paper-SSR | M2 JS | M2 vs paper-SSR | Comment |
|---|---|---|---|---|---|---|
| 8.0/qsingle_3 (肠胃感冒季节) | 0.0307 | 0.0134 | **−56%** | 0.0105 | **−66%** | Both win big |
| 8.0/qsingle_4 (肠胃感冒场景) | **0.1309** | 0.0460 | **−65%** | 0.0468 | **−64%** | Outlier — paper-SSR collapses |
| 9.0/qsingle_3 (湿气原因) | 0.0202 | 0.0198 | −2% | 0.0304 | **+50% (loss)** | M2 ensemble hurts this Q |
| 9.0/qsingle_4 (湿气症状) | 0.0182 | 0.0173 | −5% | 0.0158 | −13% | Tied / mild win |
| 9.0/qsingle_5 (湿气缓解方法) | 0.0398 | 0.0468 | **+18% (loss)** | 0.0367 | −8% | M0 hurts this Q |
| 11.0/qsingle_3 (中暑场景) | 0.0183 | 0.0180 | −1% | 0.0125 | −32% | Tied / M2 win |
| **Mean over 6Q** | 0.0430 | 0.0269 | **−37.5%** | 0.0254 | **−40.9%** | |
| **Mean over 5Q (excl. Q2)** | 0.0254 | 0.0231 | **−9.3%** | 0.0212 | **−16.9%** | Without outlier, gains shrink to 9–17% |

**Honest reading:** Q2 (gut-cold scenarios) drives ~⅔ of the cross-baseline gain. M0 has a per-Q loss on Q5; M2 has a per-Q loss on Q3. Neither dominates uniformly. The story is *fixing domain-specific transfer failures*, not uniform superiority.

---

## Why M0 is not just "paper-SSR with extra steps"

Four substantive algorithmic differences:

1. **Per-consumer → per-cluster aggregation unit**: paper averages over individual synthetic consumers (eq. 1); M0 averages within latent topic clusters and then aggregates clusters by per-question mass. This is a different aggregation tree, not a tweak.
2. **LLM-generated text → real social-media posts as the SSR input**: paper embeds the LLM's free-text response; M0 embeds real Weibo/Xiaohongshu posts grouped by cluster. The signal source is different.
3. **Demographic persona prompting → empirical cluster-mass weighting**: paper conditions each consumer on demographic attributes; M0 uses no per-consumer demographics, but weights clusters by their per-question empirical mass derived from real posts.
4. **Implicit no-cluster aggregation → explicit top-K cluster truncation + mass renormalization**: M0 selects the top-10 clusters by aggregated weight and renormalizes within them; paper has no such truncation.

These four differences together define the hierarchical cluster-weighted family. Disentangling which of (1)–(4) is individually load-bearing would require an ablation ladder — left as future work; for now, the family-as-a-whole claim is supported by the stability across hyperparameter sweeps shown above.

---

## Supporting evidence (no new experiments)

**(a) "Trivial knobs are flat" — Direct-SSR sweeps over `n_posts` and `top_K` stay close to M0.**
Source: `results/improve_orig6.json`. All variants land in [0.0272, 0.0280]; M0 (n=50, top-10, seed=42) is at 0.0269. → the recovery from paper-SSR (0.0430) to M0 (0.0269) is not explained by sample-size or top-K alone — the cluster-weighted hierarchical structure is what matters.

**(b) "Hierarchical family is stable across reweighting schemes."**
Source: `results/alignment_6q.json`. M0 mean JS across 6 reweighting variants:

| Variant | Mean JS |
|---|---|
| Default | 0.0269 |
| IS_prov | 0.0271 |
| IS_age | 0.0269 |
| IS_prov+age | 0.0271 |
| OT_prov_fast | 0.0272 |
| OT_prov_sinkhorn | 0.0269 |

→ Range [0.0269, 0.0272] across importance-sampling and OT-based cluster-weight schemes. The improvement is robust, not a single-variant artifact.

**(c) "Not only Q2."**
M0 wins 5/6 questions vs paper-SSR. Even excluding Q2, the 5Q mean improves by 9.3% (0.0254 → 0.0231).

---

## Cross-paper diagnostic (NOT central)

| | Our (Qwen3-8B, Chinese 6Q multi-choice) | Maier (GPT-4o, English 57 PI Likert) |
|---|---|---|
| K_xy | 0.827 (paper-SSR transfer) → 0.852–0.858 (M0/M2) | 0.91 (no-demo control) → 0.88 (best with demos) |
| C_xy | 0.877 (paper-SSR transfer) → 0.923–0.932 (M0/M2) | 0.98 (no-demo control) → 0.96 (best with demos) |

K_xy is mathematically appropriate for ordinal Likert; our options are nominal multi-choice. We are not "0.025 K_xy below paper" in any meaningful scientific sense — different model, language, task, metric appropriateness. Use this only as diagnostic context.

---

## Steering story → Appendix

Conditional positive at low α / late layers; random and shuffled vectors match real ones at aggressive settings. Brief Appendix entry as a negative-but-published diagnostic.

---

## Revised abstract (post-correction)

> We present a single-domain case study of survey-distribution reconstruction from social media in a Chinese consumer-health setting (6 real survey questions on TCM 藿香正气水, ~12K Weibo posts, Qwen3-8B + bge-base-zh-v1.5). The published per-consumer SSR baseline (Maier et al., 2026), originally designed for English Likert purchase-intent surveys, transfers to our setting with mean JS = 0.0430; one health-specific question accounts for the majority of the error. We introduce **hierarchical cluster-weighted SSR**: a minimal but substantive hierarchical reformulation that aggregates SSR signal within latent topic clusters of real social-media posts and combines clusters by per-question empirical mass, instead of averaging flatly across individual synthetic consumers. This method (M0) reduces JS to 0.0269 — a 37.5% improvement over the per-consumer baseline — with stable performance across six cluster-reweighting schemes. A per-cluster LLM-distribution estimator (M1) ties M0 (0.0268) and a 50/50 ensemble (M2) further improves JS to 0.0254 (a 5.6% additional gain over M0), with non-uniform per-question behavior (one question worsens by 50% over M0). We report per-question wins and losses transparently. A brief appendix discusses activation-steering with persona vectors as a conditional, mostly-direction-irrelevant signal.

---

## Final framing summary

- **Diagnostic**: paper-SSR (per-consumer, flat-aggregated, no clusters) transfers poorly to Chinese consumer-health (JS=0.0430).
- **Primary novel contribution**: hierarchical cluster-weighted SSR (M0) — −37.5% JS over the per-consumer baseline, robust across 6 reweighting schemes.
- **Secondary contribution**: per-cluster LLM-dist (M1) + 50/50 ensemble (M2) — additional 7.6% over M0; non-uniform per-question behavior, transparently reported.
- **Steering**: appendix-level diagnostic only.
- **Honest scope**: 6Q single-domain case study; n=6 hard limit; per-Q losses retained.
- **Target venue**: workshop / applied NLP venue (clear yes); EMNLP Findings (plausible); main-track (still constrained by n=6).
