# Auto Review Log: Persona-Steered SSR (PS-SSR)

**Project**: Survey reconstruction from social media using LLM activation steering + SSR
**Started**: 2026-04-13
**Reviewer**: GPT-5.4 via Codex MCP (xhigh reasoning)
**Max Rounds**: 5
**Difficulty**: medium

---

## Round 1 (2026-04-13)

### Assessment (Summary)
- Score: 4/10
- Verdict: Almost, but not ready for top venue
- Key criticisms:
  - Only 6 survey questions — too small for main empirical claim
  - Weight optimization (Approach B) risks ground-truth leakage
  - Steering contribution not isolated — need random/shuffled/zero vector controls
  - Negative cluster construction encodes topic distance, not attitude
  - Claims about "consumer attitudes" under-validated
  - Social media ≠ survey population — representativeness bias
  - Baselines not strong enough — need cluster-summary, retrieval-augmented
  - Steering all tokens may be crude — test generated-only tokens
  - SSR normalization (min-subtraction) may distort — compare alternatives
  - No uncertainty/statistical testing — need bootstrap CIs
  - Mixture hypothesis needs mechanistic support
  - External validity weak — one product/domain

<details>
<summary>Click to expand full reviewer response</summary>

**Score: 4/10 for NeurIPS/ICML as currently described.**

The idea is interesting and potentially publishable, but not yet top-venue ready. The core weakness is that the method may look sophisticated while the causal contribution of activation steering remains under-identified. A reviewer would ask: does CAA actually improve reconstruction of survey distributions, or are you mostly reusing Project 1's topic weights plus LLM priors and anchor-similarity artifacts?

**Readiness: Almost, but not ready for top venue.**

## Critical Weaknesses

1. Only 6 survey questions is too small for the main empirical claim.
2. Weight optimization risks ground-truth leakage and overfitting.
3. The activation-steering contribution is not isolated.
4. Negative cluster construction may encode topic distance, not persona attitude.
5. Claims about "consumer attitudes" are under-validated.
6. The method may reconstruct social-media discourse, not the real survey population.
7. Baselines are not yet strong enough.
8. Steering all token positions may be crude and unstable.
9. Anchor-based SSR normalization may distort distributions.
10. Evaluation lacks uncertainty and statistical testing.
11. Steer-then-aggregate vs aggregate-then-steer claim needs mechanistic support.
12. External validity is weak.

</details>

### Actions Taken
1. **Added control vectors** (random, shuffled, zero) to `steered_ssr.py` — isolates steering contribution (weakness #3)
2. **Added 3 new baselines** to `baselines.py`: cluster-summary prompting, retrieval-augmented prompting, direct comment SSR — stronger comparison (weakness #7)
3. **Added bootstrap CI + paired permutation test** to `evaluate.py` — statistical rigor (weakness #10)
4. **Added alternative SSR normalizations** (softmax, clipped, rank) to `steered_ssr.py` — robustness check (weakness #9)
5. **Added generated-only token steering mode** to SteeringHook — addresses token steering concern (weakness #8)
6. **Added negation strategies** (random_cluster, global_mean) to `persona_vectors.py` — addresses negative construction concern (weakness #4)
7. **Reframed claims and scope** in EXPERIMENT_PLAN.md — case study positioning, oracle vs deployable weights separation, softened claims (weaknesses #1, #2, #5, #6)

### Status
- Continuing to Round 2

---

## Round 2 (2026-04-13)

### Assessment (Summary)
- Score: 5.5/10
- Verdict: Almost, but still not ready
- Key remaining:
  - No actual empirical results yet (controls implemented but not run)
  - Persona vector validity still unproven
  - Hard-negative for negation still missing
  - Multiple uncertainty sources not covered
  - Hyperparameter tuning risk with 6 questions
  - Direct comment SSR / retrieval may be competitive — need contingency plan
  - Mixture claim underdeveloped

### Actions Taken
1. **Vector validation module** (`vector_validation.py`): Generates responses under each cluster's steering vector using neutral prompts, uses LLM as judge to label attributes, compares to cluster summaries, includes random/shuffled controls. Computes consistency scores (weakness #3 from R2).
2. **Weight optimizer** (`weight_optimizer.py`): Implements all 3 approaches — SSR weights (A), LLM-judged relevance (C), and oracle JS-optimization (B) with leave-one-out CV. Oracle clearly labeled (weakness #6 from R2).
3. **Hyperparameter pre-registration protocol** (`HYPERPARAMETER_PROTOCOL.md`): Pre-registers primary config (layer=16, α=2.0, N=20, distant_cluster, min_sub, SSR weights, steer-then-aggregate) with rationale. Ablations reported as sensitivity, not selection (weakness #6 from R2).
4. **Contingency framing**: If direct comment SSR or retrieval is competitive, paper reframes around interpretability/controllability/sparse-comment advantage rather than accuracy superiority (weakness #7-8 from R2).

### Status
- Continuing to Round 3

---

## Round 3 (2026-04-13)

### Assessment (Summary)
- Score: 6/10
- Verdict: Almost, still not ready
- Key remaining:
  - Actual empirical results still missing (experiments not yet run)
  - Vector validation is circular (same model as judge)
  - Validation tests topic encoding, not attitude/persona
  - Study still structurally underpowered (6 questions)
  - Entropy analysis not yet implemented

### Actions Taken
1. **Independent validation model**: Updated `vector_validation.py` to use BGE embedding model (independent of Qwen3-8B) for semantic similarity scoring, replacing the circular same-model judging
2. **Survey-targeted validation**: Added `validate_with_survey_prompts()` — tests whether steered outputs shift SSR distribution in expected direction by comparing steered vs unsteered responses on actual survey questions
3. **Entropy/mixture analysis**: Added `mixture_analysis()` to `evaluate.py` — computes per-cluster entropy, mixture entropy, entropy gap, and inter-cluster diversity (pairwise JS between cluster outputs)
4. **Outcome-neutral framing**: Research question rewritten to be valid under either outcome ("study whether steering adds value" vs "steering improves accuracy")
5. **Validation framed as cluster-topic/behavioral consistency**, not persona/attitude validation

### Status
- Continuing to Round 4

---

## Round 4 (2026-04-13)

### Assessment (Summary)
- Score: 6.5/10
- Verdict: Almost, still not ready — design is credible, needs empirical results
- Key remaining:
  - No actual result table yet (the main blocker now)
  - BGE validation shares embedding space with SSR (partial coupling)
  - PMF shifts need to show movement toward ground truth, not just any movement
  - Entropy analysis needs connection to prediction quality
  - Statistical evidence fragile with n=6

### Actions Taken
1. **PMF shift directionality analysis**: Tests if steering moves toward ground truth
2. **Entropy-outcome correlation**: Spearman correlation connecting mixture diversity to steering advantage
3. **Validation coupling acknowledgment**: Documented BGE partial coupling, suggested secondary model

### Status
- Continuing to Round 5

---

## Round 5 — FINAL (2026-04-13)

### Assessment (Summary)
- Score: 7/10 (for experimental design)
- Verdict: **Ready to run experiments**; not ready for submission until results exist
- Reviewer says: "The design is now coherent, defensible, and much harder to dismiss."

### Reviewer's Final Recommendations (pre-experiment)
1. **Freeze primary success criterion**: PS-SSR must beat SSR-only, direct comment SSR, retrieval, and shuffled vectors on mean JS, improving at least 4/6 questions
2. **Do not use PMF directionality for model selection** (uses ground truth)
3. **Add one independent embedding model** (text2vec-chinese or m3e-base) for validation
4. **Multiple-comparison discipline**: pre-registered config is primary, ablations are exploratory
5. **Pre-write negative result interpretation**
6. **Track cost and variance** across methods
7. **Report absolute distributions** (predicted vs true PMFs), not only metrics
8. **Ensure anchors are not tuned to ground truth**

### Score Progression
| Round | Score | Key Improvement |
|-------|-------|-----------------|
| 1 | 4.0 | Initial design — under-controlled, overclaiming |
| 2 | 5.5 | Controls, baselines, statistical testing, reframed claims |
| 3 | 6.0 | Vector validation, weight optimizer, hyperparameter protocol |
| 4 | 6.5 | Independent validation model, entropy analysis, outcome-neutral framing |
| 5 | 7.0 | PMF directionality, entropy-outcome correlation, final polish |

### Venue Recommendation
- **Best fit**: Applied ML/NLP venue, computational social science, consumer analytics
- **Possible**: Top ML venue if results are strong and framing is careful
- **Safe**: Workshop (LLM steering, synthetic survey, social media measurement)

## Method Description

PS-SSR (Persona-Steered Semantic Similarity Rating) is a two-stage pipeline for approximating survey response distributions from social media data. Stage 1 extracts persona vectors from clustered social media comments via Contrastive Activation Addition: for each HDBSCAN cluster, the method computes the difference between mean hidden states of cluster comments and a contrastive set, producing a direction in LLM activation space that encodes the cluster's behavioral signature. Stage 2 injects these persona vectors into the LLM's transformer layers during generation, then applies Semantic Similarity Rating (SSR) to map the steered outputs to survey option distributions. A hierarchical weighting scheme combines per-cluster predictions into final survey distribution estimates. The framework supports both steer-then-aggregate (per-cluster steering followed by weighted combination) and aggregate-then-steer (combined vector injection) strategies.


---

## Post-Experiment Results (2026-04-13/14)

### Main Results Table

| Method | Mean JS ↓ | Q1 | Q2 | Q3 | Q4 | Q5 | Q6 |
|--------|-----------|-----|-----|-----|-----|-----|-----|
| **PS-SSR ATS (softmax)** | **0.069** | 0.041 | 0.217 | 0.059 | 0.039 | 0.047 | 0.013 |
| PS-SSR STA (softmax) | 0.070 | 0.042 | 0.219 | 0.061 | 0.039 | 0.047 | 0.014 |
| Persona Prompt | 0.086 | 0.109 | 0.020 | 0.027 | 0.112 | 0.152 | 0.095 |
| Direct LLM | 0.158 | 0.276 | 0.171 | 0.097 | 0.084 | 0.109 | 0.212 |
| PS-SSR ATS (min_sub) | 0.166 | 0.133 | 0.288 | 0.172 | 0.139 | 0.127 | 0.139 |

### Publishability Assessment (GPT-5.4)
- Score: 3/10 as-is, 5.5-6.5 with controls + second domain
- Best venue: ICWSM or ACL/EMNLP Findings
- Running: Control vector ablations (zero/random/shuffled)

### Control Vector Ablation Results (2026-04-14 00:30)

| Method | Mean JS ↓ | Q1 | Q2 | Q3 | Q4 | Q5 | Q6 |
|--------|-----------|-----|-----|-----|-----|-----|-----|
| **Control: zero vector** | **0.044** | 0.028 | 0.058 | 0.024 | 0.025 | 0.107 | 0.023 |
| PS-SSR real (softmax) | 0.069 | 0.041 | 0.217 | 0.059 | 0.039 | 0.047 | 0.013 |
| Control: shuffled vector | 0.070 | 0.040 | 0.222 | 0.061 | 0.039 | 0.046 | 0.013 |
| Control: random vector | 0.072 | 0.031 | 0.240 | 0.076 | 0.031 | 0.037 | 0.015 |
| Persona Prompt | 0.086 | 0.109 | 0.020 | 0.027 | 0.112 | 0.152 | 0.095 |
| Direct LLM | 0.158 | 0.276 | 0.171 | 0.097 | 0.084 | 0.109 | 0.212 |

### Critical Finding: Persona vectors do NOT outperform controls

- **Zero vector (no steering) is the BEST method** (JS=0.044)
- PS-SSR real vectors (0.069) ≈ shuffled vectors (0.070) ≈ random vectors (0.072)
- Real persona vectors win only 3/6 vs random, 3/6 vs shuffled, 2/6 vs zero
- **The steering direction does not matter** — the gains come from generation+SSR+softmax, not persona-specific steering

### Interpretation

The activation steering contribution is **not validated**. The improvement over Persona Prompt (0.086) comes from:
1. The generation+SSR pipeline with softmax normalization
2. NOT from cluster-specific persona directions

This matches the reviewer's predicted failure mode: "If random or shuffled vectors perform similarly to real vectors, the core contribution becomes much weaker."

### Revised Assessment
- The paper cannot claim persona steering improves survey reconstruction
- The contribution shifts to: SSR with softmax normalization + LLM generation outperforms prompt-based approaches
- Zero-vector steering (= unsteered generation + SSR) is actually the best, simplest method
- Need to investigate WHY zero vector beats everything (likely: unsteered generation is more natural/diverse)
