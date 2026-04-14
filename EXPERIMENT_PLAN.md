# Experiment Plan: Persona-Steered SSR for Survey Reconstruction

## Research Question

We study whether activation-steered personas can approximate survey distributions from social media data, and analyze when steering adds value beyond embedding-based and prompting-based baselines. Specifically: does injecting cluster-derived persona vectors into LLM hidden states improve or complement Hierarchical SSR for social-media-based survey distribution approximation?

## Core Idea

Survey Project 1 (baseline) uses pure embedding-space SSR — cosine similarity between comment embeddings and anchor sentence embeddings. This is effective but **model-agnostic**: it treats the LLM only as an anchor generator, not as a reasoning agent.

Survey Project 2 proposes: **steer the LLM itself** with persona vectors extracted from each cluster, then let the steered LLM generate responses that are measured via SSR. This combines:
- **Representation learning** (persona vectors capture cluster-level attitudes)
- **Causal intervention** (activation steering changes model behavior)
- **Distributional measurement** (SSR converts outputs to probability distributions)

## Method: Persona-Steered Hierarchical SSR (PS-SSR)

### Phase 1: Persona Vector Extraction (Contrastive Activation Addition)

For each cluster $c$:

1. Sample $N$ representative comments from cluster $c$ → positive set $P_c$
2. Construct negation set $N_c$ by either:
   - (a) Sampling from distant clusters (cosine distance in topic embedding space)
   - (b) Using LLM to generate negated versions of $P_c$ comments
3. Run both sets through target LLM, extract hidden states at layer $l$
4. Persona vector: $v_c = \text{mean}(h(P_c)) - \text{mean}(h(N_c))$

**Key design choices:**
- Which layer(s) to extract from → sweep layers [8, 12, 16, 20, 24] for Qwen3-8B (32 layers)
- How many samples N → [10, 20, 50]
- Negation strategy → compare (a) vs (b)

### Phase 2: Weight Determination (the open question from Survey Project 2)

Three approaches to determine cluster combination weights $w_c$:

**Approach A — First-layer SSR weights (reuse Project 1)**
Use the topic-level SSR from Project 1 as weights: $w_c = \text{SSR}_{\text{topic}}(q, t_c)$

**Approach B — Learned weights via JS-divergence minimization**
Optimize weights on a held-out portion of survey questions:
$$\min_{w} \text{JS}(\text{PS-SSR}(w), p_{\text{true}})$$
using scipy.optimize or simple grid search (few enough clusters to be tractable).

**Approach C — LLM-judged relevance weights**
Ask the LLM to rate each cluster's relevance to the question (already implemented in Project 1 as `cluster_relevance`), convert to soft weights.

### Phase 3: Steered Generation + SSR

For each survey question $q$:

1. Compute cluster weights $w_c$ for question $q$ (Phase 2)
2. For each relevant cluster $c$:
   - Construct combined steering vector: $v_q = \sum_c w_c \cdot v_c$
   - OR per-cluster steering: steer with $v_c$, generate response, measure SSR, then aggregate
3. Inject steering vector at layer $l$ with scaling factor $\alpha$
4. Prompt steered LLM: "As a consumer who has experience with 藿香正气水, answer: {question}"
5. Measure response via SSR against anchor sentences → per-cluster distribution
6. Aggregate: $p_{\text{pred}}(q) = \sum_c w_c \cdot \text{SSR}_c(q)$

### Two Steering Strategies

**Strategy 1: Aggregate-then-steer**
- $v_q = \sum_c w_c \cdot v_c$ → single steered LLM → SSR
- Faster (one forward pass per question)

**Strategy 2: Steer-then-aggregate**
- For each cluster: steer with $v_c$ → SSR → get $p_c(q)$
- Aggregate: $p_{\text{pred}} = \sum_c w_c \cdot p_c(q)$
- More faithful to mixture-of-personas idea

## Experimental Design

### Baselines
1. **Direct LLM**: Ask Qwen3-8B to directly answer survey questions (sample 100 times)
2. **SSR-only (Project 1)**: Embedding-based hierarchical SSR (current best)
3. **LLM-as-persona**: Prompt-based persona simulation (no steering, just persona description in prompt)
4. **Cluster-summary prompting**: Give LLM all topic summaries + weights as context
5. **Retrieval-augmented prompting**: Sample representative comments as in-context examples
6. **Direct comment SSR**: Apply SSR directly to raw comments without LLM generation

### Control Conditions
7. **PS-SSR + zero vector**: Steering with v=0 (generation + SSR only)
8. **PS-SSR + random vector**: Random direction, same norm (any perturbation?)
9. **PS-SSR + shuffled vector**: Wrong cluster assignments (cluster-specificity test)

### Proposed Methods
10. **PS-SSR (Aggregate-then-steer)**: Strategy 1 with weight approach A/C
11. **PS-SSR (Steer-then-aggregate)**: Strategy 2 with weight approach A/C
12. **PS-SSR (Oracle weights)**: Strategy 2 with Approach B (upper bound only)

### Evaluation Metrics
- **JS Divergence** (primary): between predicted and true survey distributions
- **KL Divergence**: asymmetric comparison
- **Earth Mover's Distance**: ordinal-aware metric
- **Mean Absolute Error** on distribution means
- **Chi-squared test**: statistical similarity of distributions

### Data Split
- Survey questions (8,3), (8,4), (9,3), (9,4), (9,5), (11,3) — 6 questions total
- Leave-one-out cross-validation for weight optimization (Approach B)
- All questions for reporting final metrics

### Ablations
1. Layer selection: which layer produces best persona vectors
2. Steering strength $\alpha$: [0.5, 1.0, 2.0, 5.0, 10.0]
3. Number of samples N for vector extraction
4. Negation strategy: distant-cluster vs LLM-generated
5. Weight approach: A vs B vs C
6. Steering strategy: aggregate-then-steer vs steer-then-aggregate

## Implementation Plan

### File Structure
```
business/
├── experiments/
│   ├── config.py              # Server URLs, model paths, hyperparams
│   ├── persona_vectors.py     # Phase 1: extract persona vectors
│   ├── weight_optimizer.py    # Phase 2: weight determination
│   ├── steered_ssr.py         # Phase 3: steering + SSR pipeline
│   ├── baselines.py           # Baseline methods
│   ├── evaluate.py            # Metrics computation
│   └── run_all.py             # Main experiment runner
├── results/
│   └── ...
└── EXPERIMENT_PLAN.md
```

### Dependencies
- `transformers` (Qwen3-8B model loading + hidden state extraction)
- `torch` (tensor ops, activation hooks)
- `sentence-transformers` (embedding model for SSR)
- `scipy` (optimization, JS divergence)
- `numpy`, `pandas`, `matplotlib`

### Hardware
- Chen server (chenhongrui@122.225.39.134:2222) with GPU for Qwen3-8B inference
- Or local vLLM server at 127.0.0.1:8071 if available

## Scope and Positioning

This work is positioned as a **case study / proof-of-concept** demonstrating that activation steering can improve social-media-based survey distribution approximation for a specific product domain (Chinese traditional medicine). We do NOT claim PS-SSR replaces real surveys or generalizes across arbitrary domains without further validation.

**Key limitation**: Social media users are not a representative survey sample. PS-SSR approximates survey-like distributions from social media signals, not true population distributions. Platform bias, demographic skew, and self-selection are inherent limitations.

## Weight Optimization Framing

- **Primary methods** (no ground-truth access): Approach A (SSR weights from Project 1) and Approach C (LLM-judged relevance)
- **Oracle upper bound** (uses ground truth): Approach B (JS-minimized weights) — reported separately as a ceiling, NOT as a deployable method

## Control Experiments (Steering Contribution Isolation)

To verify that gains come from meaningful persona steering, not artifacts:
1. **Zero vector control**: α=0 or v=0 — tests whether generation + SSR alone explains gains
2. **Random vector control**: random direction, same norm — tests whether any perturbation helps
3. **Shuffled vector control**: wrong cluster assignments — tests whether cluster-specific direction matters
4. **Wrong-cluster vector**: steer with semantically opposite cluster — tests directionality
5. **Generated-only steering**: steer only generated tokens, not prompt — tests sensitivity to steering mode

## Normalization Robustness Check

Compare SSR normalization methods to ensure conclusions aren't normalization artifacts:
- min-subtraction (original), softmax (temperature-scaled), clipped cosine, rank-based

## Statistical Rigor

- Bootstrap CIs (n=1000) for all JS divergence comparisons
- Paired permutation test for method comparisons
- Per-question breakdown (not just mean)
- Report failure/degeneration rates for steered generation

## Claims This Experiment Can Support (Reframed)

1. **Persona steering improves distributional approximation**: PS-SSR achieves lower JS divergence than embedding-only SSR, with statistical significance via paired permutation test, controlling for random/shuffled vectors
2. **Activation vectors encode cluster-discriminative directions**: Persona vectors steer generation in directions consistent with cluster topics (validated via control experiments, not overclaimed as "attitude capture")
3. **Steer-then-aggregate empirically outperforms aggregate-then-steer**: Observed on this case study; mechanistic support via per-cluster entropy analysis, framed as empirical finding not general law
4. **SSR weights transfer to steering context**: Topic-level SSR weights (Approach A) perform near oracle-optimized weights, suggesting no task-specific optimization is needed
5. **Interpretable pipeline**: The two-layer decomposition provides an audit trail; the SSR component is fully interpretable; the steering component adds a latent intervention that is less interpretable but empirically validated
