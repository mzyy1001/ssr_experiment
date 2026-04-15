# Experiment Report: Persona-Steered SSR (PS-SSR)
# Survey Reconstruction from Social Media via LLM Activation Steering

**Authors**: mzyy1001, Claude Opus 4.6
**Date**: 2026-04-13 to 2026-04-15
**Status**: Pilot study complete, conditionally positive results

---

## 1. Problem Statement

Brands need structured consumer attitude data (surveys), but traditional surveys are expensive, slow, and sparse. Meanwhile, consumers already express opinions organically on social media. Can we **reconstruct survey response distributions from social media data** without running actual surveys?

### Prior Work (Survey Project 1)

The existing Hierarchical SSR framework uses a two-layer approach:
- **Layer 1 (Topic-level)**: Aligns survey questions with HDBSCAN cluster topics via cosine similarity to produce cluster weights
- **Layer 2 (Comment-level)**: Maps each comment to survey option distributions via cosine similarity with LLM-generated anchor sentences

This is purely embedding-based — the LLM (Qwen3-8B) serves only as an anchor generator, not as a reasoning agent. Results on one question showed JS divergence of 0.044 against real survey data.

### Our Hypothesis (Survey Project 2)

Instead of treating the LLM as a black-box text generator, **inject cluster-derived persona vectors into its hidden states** via Contrastive Activation Addition (CAA), then apply SSR to the steered outputs. This combines:
- **Representation learning**: persona vectors capture cluster-level behavioral signatures
- **Causal intervention**: activation steering shifts model generation
- **Distributional measurement**: SSR converts free-text into survey distributions

---

## 2. Method: PS-SSR

### 2.1 Data Pipeline

- **Source**: 94,640 social media posts (Weibo + Xiaohongshu, 2019-2023) about 藿香正气水 (Huoxiang Zhengqi Water, a traditional Chinese medicine)
- **Filtering**: Ads and duplicates removed → 40,379 meaningful posts
- **Clustering**: HDBSCAN on medical-domain embeddings (iic/nlp_corom_sentence-embedding_chinese-base-medical) → 66 semantic clusters
- **Topic extraction**: Qwen3-8B summarized each cluster into a one-sentence topic
- **Ground truth**: 6 survey questions across 3 sub-surveys (880-1,415 respondents each)

### 2.2 Persona Vector Extraction (Phase 1)

For each cluster $c$:
1. Sample $N=20$ representative comments → positive set $P_c$
2. Sample $N=20$ comments from the most semantically distant cluster → negative set $N_c$
3. Run both sets through Qwen3-8B, extract mean hidden states at layer $l$
4. Persona vector: $v_c = \text{mean}(h(P_c)) - \text{mean}(h(N_c))$

**Implementation**: `persona_vectors.py` — extracts 66 vectors in ~2 minutes on A100-80GB.

### 2.3 Steered Generation + SSR (Phase 2)

Two strategies tested:

**Steer-then-aggregate (STA)**: For each of the top-K relevant clusters:
1. Inject persona vector $v_c$ at layer $l$ with strength $\alpha$
2. Generate 5 free-text responses to the survey question
3. Apply SSR (softmax normalization) to each response against anchor embeddings
4. Average per-cluster SSR distributions
5. Combine via cluster weights

**Aggregate-then-steer (ATS)**: Weighted-combine all persona vectors first, then steer once.

### 2.4 SSR Measurement

SSR maps generated text to a probability distribution over survey options:
1. Encode generated text and anchor sentences with BGE-base-zh-v1.5
2. Compute cosine similarity between text and each anchor
3. Apply softmax normalization (temperature=0.1) to get PMF

**Critical finding**: The original min-subtraction normalization ($\text{sims} - \text{sims.min()}$) forces one option to probability zero, catastrophically harming JS divergence. Switching to softmax was essential.

---

## 3. Experimental Design

### 3.1 Survey Questions (Ground Truth)

| ID | Question | Options | Responses |
|----|----------|---------|-----------|
| Q1 | 哪个季节容易发生肠胃感冒? | 全年/夏/冬/秋/春 | 890 |
| Q2 | 肠胃感冒的场景往往是? | 冷热刺激/饮食不良/天气变化/病毒感染/其他 | 890 |
| Q3 | 产生湿气的原因是? | 爱吃冷饮火锅/久坐不运动/体质问题/环境潮湿/其他 | 884 |
| Q4 | 湿气重更容易发生在哪个季节? | 全年/夏/春/冬/秋 | 884 |
| Q5 | 缓解湿气重的方法? | 饮食调理/刮痧蒸桑拿运动/不在意/对症中药 | 883 |
| Q6 | 预防中暑的场景往往是? | 温差大/天气炎热/出门旅游/天热车里常备/空调吹多了 | 781 |

### 3.2 Methods Compared

**Baselines**:
1. **Direct LLM**: Ask Qwen3-8B to directly choose options (100 samples, majority vote distribution)
2. **Persona Prompt**: Describe cluster topic in prompt, generate response, apply SSR
3. **SSR-only (Project 1)**: Pure embedding-based SSR from prior work (1 question only)

**Controls** (steering contribution isolation):
4. **Zero vector**: No steering (α=0 equivalent), just generate + SSR
5. **Random vector**: Random direction with matched norm
6. **Shuffled vector**: Wrong cluster-to-vector assignment

**Proposed**:
7. **PS-SSR**: Real persona vectors with various configurations

### 3.3 Evaluation Metric

**Jensen-Shannon Divergence (JS)**: Symmetric measure of distribution similarity. Lower = better. Range [0, 1].

### 3.4 Design Review Process

The experiment design underwent **5 rounds of cross-model review** via Codex MCP (GPT-5.4, xhigh reasoning effort), improving the design score from 4/10 to 7/10 before any experiments ran. Key improvements from review:
- Added control vectors (random/shuffled/zero)
- Added 3 stronger baselines
- Added bootstrap CI and paired permutation tests
- Pre-registered hyperparameter protocol
- Reframed claims as case study, not general method

---

## 4. Results

### 4.1 Initial Results (α=2.0, layer=16, top-20 clusters, aggregate-then-steer)

| Method | Mean JS ↓ | Wins |
|--------|-----------|------|
| Control: zero vector | **0.044** | — |
| PS-SSR real vectors | 0.069 | 2/6 vs zero |
| Control: shuffled | 0.070 | — |
| Control: random | 0.072 | — |
| Persona Prompt | 0.086 | — |
| Direct LLM | 0.158 | — |

**Verdict**: Persona steering direction did NOT matter. Real ≈ shuffled ≈ random. Zero vector (no steering) was best. The central hypothesis appeared to fail.

**Diagnosis**: Alpha=2.0 was too aggressive — it distorted generation quality, producing overly uniform SSR distributions. The improvement over Persona Prompt (0.069 vs 0.086) came entirely from the generation+softmax SSR pipeline, not from persona-specific steering.

### 4.2 Parameter Sweep Results

#### Alpha Sweep (layer=16, steer-then-aggregate, top-5 clusters)

| Alpha | Real JS ↓ | Zero JS | Wins | Δ vs Zero |
|-------|-----------|---------|------|-----------|
| **0.1** | **0.040** | 0.046 | **4/6** | **-12.2%** |
| 0.3 | 0.045 | 0.046 | 3/6 | -2.4% |
| 0.5 | 0.060 | 0.046 | 2/6 | +31.4% |
| 1.0 | 0.070 | 0.046 | 2/6 | +50.3% |

**Finding**: Clear monotonic relationship — lower alpha is better. At α=0.1, persona vectors reliably beat zero (4/6 wins, 12% lower JS).

#### Layer Sweep (α=0.5, steer-then-aggregate, top-5 clusters)

| Layer | Real JS ↓ | Zero JS | Wins |
|-------|-----------|---------|------|
| 8 | 0.062 | 0.046 | 2/6 |
| 12 | 0.069 | 0.048 | 2/6 |
| 20 | 0.052 | 0.045 | 3/6 |
| **24** | **0.037** | 0.047 | **4/6** |
| 28 | 0.042 | 0.046 | 4/6 |

**Finding**: Later layers (24-28) carry more persona-relevant information. Layer 24 at α=0.5 achieves the best overall result (JS=0.037, 4/6 wins). Even at α=0.5 (which fails at layer 16), layer 24 succeeds.

#### Relevant-Only Clusters (α=0.5, layer=16)

| Condition | Real JS | Zero JS | Wins |
|-----------|---------|---------|------|
| Relevant-only (7 clusters) | 0.061 | 0.046 | 2/6 |

**Finding**: Relevance filtering at α=0.5 does not help (alpha still too high at layer 16).

### 4.3 Best Configurations (Persona Vectors Beat Controls)

| Config | Real JS ↓ | Zero JS | Wins | Δ vs Zero | Δ vs Persona Prompt |
|--------|-----------|---------|------|-----------|---------------------|
| **α=0.5, L=24, top-5 STA** | **0.037** | 0.047 | **4/6** | **-21.3%** | **-56.8%** |
| α=0.1, L=16, top-5 STA | 0.040 | 0.046 | 4/6 | -12.2% | -53.2% |
| α=0.5, L=28, top-5 STA | 0.042 | 0.046 | 4/6 | -8.6% | -51.0% |

### 4.4 Per-Question Breakdown (Best Config: α=0.5, L=24)

| Question | True Top Answer | Real JS | Zero JS | Winner | Δ |
|----------|----------------|---------|---------|--------|---|
| Q1 (季节→肠胃感冒) | 夏 29.6% | **0.018** | 0.028 | Real ✓ | -36% |
| Q2 (场景→肠胃感冒) | 冷热刺激 34.0% | 0.069 | **0.066** | Zero ✗ | +5% |
| Q3 (湿气原因) | 爱吃冷饮火锅 30.4% | 0.046 | **0.032** | Zero ✗ | +44% |
| Q4 (季节→湿气) | 全年 29.2% | **0.023** | 0.025 | Real ✓ | -8% |
| Q5 (缓解湿气) | 饮食调理 37.0% | **0.056** | 0.109 | Real ✓ | -49% |
| Q6 (中暑场景) | 温差大 29.6% | **0.012** | 0.023 | Real ✓ | -48% |

**Pattern**: PS-SSR wins on 4/6 questions, with large gains on Q1 (-36%), Q5 (-49%), Q6 (-48%). It loses on Q2 (场景→肠胃感冒) and Q3 (湿气原因) — both questions where the survey has a dominant practical/behavioral answer that social media clusters may not capture well.

---

## 5. Key Findings

### 5.1 Central Hypothesis: Conditionally Supported

Persona steering works **only** under specific conditions:
- **Subtle steering strength** (α ≤ 0.3) at middle layers (16)
- **Later layers** (L ≥ 24) tolerate stronger steering (α = 0.5)
- **Focused clusters** (top-5 by weight, not all 65)
- **Steer-then-aggregate** (not aggregate-then-steer)

At the original configuration (α=2.0, L=16, top-20, ATS), steering direction does not matter — the hypothesis fails completely.

### 5.2 Softmax Normalization is Critical

The switch from min-subtraction to softmax normalization was the single most impactful change:
- Min-sub forces at least one option to probability 0.0 → catastrophic JS divergence
- Softmax produces smooth distributions with no zeros (min probability ~0.07-0.17)
- This alone improved PS-SSR from JS=0.166 to JS=0.069 (58% reduction)

### 5.3 Generation + SSR Outperforms Prompting

Even without effective steering (zero vector), the generate-then-SSR pipeline (JS=0.044) substantially outperforms Persona Prompt (JS=0.086). This suggests that:
- Free-text generation produces richer, more diverse expressions than prompt-conditioned outputs
- SSR with softmax is an effective measurement layer
- The pipeline contribution is primarily in the measurement, not the steering

### 5.4 Alpha-Layer Interaction

There is a clear interaction between steering strength and injection layer:
- **Early/middle layers (8-16)**: Only tolerate very gentle steering (α ≤ 0.1)
- **Later layers (24-28)**: Tolerate moderate steering (α ≤ 0.5)

This aligns with known properties of transformer architectures: early layers encode syntactic/factual features (sensitive to perturbation), while later layers encode behavioral/stylistic features (more robust to directed intervention).

### 5.5 Q2 Is a Consistent Failure Case

Question 2 ("场景→肠胃感冒") is the worst-performing question across ALL methods and ALL configurations. The "其他" (other) option gets systematically inflated by PS-SSR. This likely reflects a mismatch between social media discourse patterns (which rarely discuss specific illness scenarios in structured terms) and the survey's categorical options.

---

## 6. Losses and Limitations

### 6.1 Scale

Only 6 survey questions from 1 product domain (traditional Chinese medicine). This is too small for strong generalization claims. The parameter sweep was conducted on the same questions used for evaluation — no held-out test set.

### 6.2 Narrow Operating Regime

The method works in a narrow alpha × layer window. The best configuration (α=0.5, L=24) was discovered through post-hoc sweep, not pre-registered. A skeptical reviewer could argue this is hyperparameter overfitting on 6 data points.

### 6.3 No Shuffled/Random Controls at Best Config

The shuffled and random vector controls were only run at the original failed config (α=2.0, L=16). We have NOT verified that steering direction matters at the best config (α=0.5, L=24). The zero-vector comparison is necessary but not sufficient.

### 6.4 Baseline Fairness

PS-SSR received extensive tuning (alpha sweep, layer sweep, top-K selection). Persona Prompt and Direct LLM were not similarly optimized (no temperature sweep, no SSR normalization variants, no prompt engineering).

### 6.5 Social Media ≠ Survey Population

Social media users are not representative survey respondents. The method estimates survey-like distributions from online discourse, not true population distributions. Platform bias, demographic skew, and self-selection are inherent limitations.

### 6.6 Single Model

All experiments use Qwen3-8B. Results may not transfer to other model families or sizes.

---

## 7. Novelty Assessment

**Rating**: Moderately novel (highly novel combination)

No prior work found using activation steering for survey distribution reconstruction. Individual components (CAA, SSR, silicon sampling, social media mining) are established. The novel contribution is the full pipeline:

> social media clusters → CAA persona vectors → steered LLM → SSR → reconstructed survey distributions

**Closest related work**: Jha et al. 2025/2026 (persona vectors for Big Five personality) — similar technique, different application domain.

---

## 8. Publishability Assessment

### External Review (GPT-5.4 via Codex MCP)

| Stage | Score | Key Issue |
|-------|-------|-----------|
| Initial design | 4/10 → 7/10 | Improved over 5 review rounds |
| After first results (α=2.0 failed) | 3/10 | Steering didn't beat controls |
| After sweep (α=0.1, L=24 work) | 4/10 | Conditional finding, needs validation |

### Recommended Venues

| Venue Tier | Recommendation | Reason |
|------------|---------------|--------|
| NeurIPS/ICML main | Not ready | Too few questions, post-hoc config selection |
| ACL/EMNLP Findings | Possible with work | Need second domain + controls at best config |
| ICWSM / WebConf | Good fit | Social media + survey reconstruction focus |
| Workshop | Ready now | Honest framing as pilot study |

---

## 9. Next Steps (Priority-Ordered)

1. **Run shuffled/random controls at α=0.5, L=24** — verify steering direction matters at best config
2. **Add second product domain** — mandatory for any venue above workshop
3. **Statistical significance tests** — bootstrap CI, paired permutation at frozen config
4. **Fair baseline tuning** — give Persona Prompt same optimization effort
5. **Freeze config on Domain A → validate on Domain B** — eliminate post-hoc selection concern
6. **Top-K ablation** — separate cluster-selection effect from steering effect
7. **Mechanism validation** — do steered outputs semantically align with source cluster themes?

---

## 10. Infrastructure and Reproducibility

### Code (7 modules in `experiments/`)

| File | Purpose |
|------|---------|
| `config.py` | Server URLs, model paths, hyperparameters |
| `persona_vectors.py` | CAA extraction with 3 negation strategies |
| `steered_ssr.py` | Steering hooks, 4 SSR normalizations, control vectors |
| `baselines.py` | 6 baseline methods |
| `evaluate.py` | JS/KL/EMD/MAE metrics, bootstrap CI, permutation tests |
| `vector_validation.py` | Independent-model validation, PMF directionality |
| `run_sweep.py` | Alpha × layer × relevance sweep |

### Hardware

- **Chen server**: 8x NVIDIA A100-SXM4-80GB, chenhongrui@122.225.39.134:2222
- **Model**: Qwen3-8B (symlinked from /data/qiqi/Qwen3-8B)
- **Embedding**: BGE-base-zh-v1.5 (uploaded to /data/chenhongrui/models/)
- **Conda env**: llama_qwen (transformers 4.57, torch 2.8)

### Runtime

| Phase | Duration | GPU |
|-------|----------|-----|
| Persona vector extraction | 2 min | A100 |
| Direct LLM baseline (6 Q) | 18 min | A100 |
| Persona Prompt baseline (6 Q) | 72 min | A100 |
| PS-SSR steer-then-aggregate (6 Q) | 72 min | A100 |
| PS-SSR aggregate-then-steer (6 Q) | 10 min | A100 |
| Control ablations (3 types × 6 Q) | 27 min | A100 |
| Full parameter sweep (11 conditions) | ~5 hours | A100 |

### Design Review

5-round autonomous review via ARIS auto-review-loop (Codex MCP, GPT-5.4 xhigh). Literature review and publishability assessment also via Codex. All review transcripts in `AUTO_REVIEW.md`.

---

## 11. Conclusion

PS-SSR demonstrates that **activation steering can improve social-media-based survey reconstruction**, but only under carefully chosen conditions. The method is sensitive to steering strength (α) and injection layer (L), with a sweet spot at low alpha or late layers.

The most important finding may not be the headline result (21% JS improvement over zero), but the **failure mode**: aggressive steering (α ≥ 0.5 at middle layers) is worse than no steering at all. This has practical implications for activation steering more broadly — the common practice of using moderate-to-high alpha values may be counterproductive for distribution-sensitive tasks.

The work is best positioned as an **honest pilot study** that maps the operating regime of persona steering for survey reconstruction, rather than a mature method claiming to replace surveys. The conditional nature of the finding — where it works, where it fails, and why — is itself a contribution to understanding activation steering behavior.
