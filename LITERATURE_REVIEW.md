# Literature Review & Novelty Assessment: PS-SSR

**Reviewed by**: GPT-5.4 via Codex MCP (xhigh reasoning), 2026-04-13
**Thread ID**: 019d8772-009b-7032-8455-cce6a42b984e

---

## Novelty Rating: Moderately Novel (Highly Novel Combination)

Individual components are known; the full bridge is novel:
> `social media clusters → activation-space persona vectors → steered LLM responses → SSR option distributions → reconstructed survey marginals`

**No prior work found** using activation steering specifically for consumer population simulation or survey distribution reconstruction.

## Component Novelty Breakdown

| Component | Prior Art? | Assessment |
|-----------|-----------|------------|
| LLM synthetic respondents / silicon sampling | Yes | Argyle et al. 2023, Aher et al. 2023, Brand et al. 2024 |
| CAA / activation steering | Yes | Turner et al. 2023, Zou et al. 2023, Panickssery/Rimsky et al. 2024 |
| Persona activation vectors | Emerging | Jha et al. 2025/2026 (Big Five personality, not survey populations) |
| SSR | Yes | Maier et al. 2025 (Likert purchase-intent simulation) |
| Social media → survey/public opinion | Yes | Long history, mostly sentiment/topic models |
| **CAA + SSR + social-media clusters for survey reconstruction** | **No direct precedent found** | **This is the novel contribution** |

## Key Related Work

### A. LLM-Based Survey Simulation / Synthetic Respondents
- **Argyle et al. 2023** "Out of One, Many" — introduces "silicon samples" via GPT-3 + sociodemographic conditioning
- **Aher, Arriaga, Kalai 2023** (ICML) — "Turing Experiments" for replicating human-subject studies
- **Santurkar et al. 2023** (ICML) — OpinionQA; finds LMs misalign with demographic groups
- **Sun et al. 2024** — Random Silicon Sampling for subgroup opinion distributions
- **Brand, Israeli, Ngwe 2024** — GPT for market research, consumer preference elicitation
- **Sun, Pei, Choi, Jurgens 2025** (NAACL) — sociodemographic prompting often fails for subjective judgments

### B. Activation Steering / Representation Engineering
- **Turner et al. 2023/2024** — Activation Addition/ActAdd (steering vectors from prompt-pair diffs)
- **Zou et al. 2023/2025** — Representation Engineering (population-level concept directions)
- **Panickssery/Rimsky et al. 2024** — CAA for Llama 2 (contrastive behavioral steering)
- **Jha et al. 2025/2026** — Persona vectors for Big Five personality (closest to PS-SSR, but targets personality not survey populations)

### C. Social Media to Survey / Public Opinion
- **O'Connor et al. 2010** (ICWSM) — "From Tweets to Polls" (sentiment → polls)
- **Reveilhac et al. 2022** — systematic review of 187 papers combining social media + survey data
- **Skoric et al. 2020** — meta-analysis of electoral forecasts from social media

### D. Semantic Similarity for Distributional Measurement
- **Maier et al. 2025** — SSR for Likert purchase-intent (DIRECT prior art for SSR module)
- **Lehtonen et al. 2025** (IEEE Access) — Sentence-BERT similarity for synthetic Likert data
- **Reimers & Gurevych 2019** — Sentence-BERT (foundational embedding method)

## Positioning

PS-SSR fills the gap between three literatures:
1. **Silicon sampling** (can LLMs simulate respondents?) — uses prompts/demographics
2. **Social media opinion mining** (can digital traces approximate surveys?) — uses surface features
3. **Activation steering** (can internal representations control behavior?) — not applied to survey measurement

**Strongest competitor**: Prompt-based synthetic respondents with SSR (persona prompt baseline, JS=0.086)

## Suggested Contribution Statement

> We introduce Persona-Steered Semantic Similarity Rating (PS-SSR), a training-free framework for reconstructing survey response distributions from social media. PS-SSR is the first method, to our knowledge, to use activation-space persona vectors derived from organic discourse as the conditioning mechanism for LLM-based survey simulation, and to combine this with semantic-similarity distributional measurement over survey options.

## Papers to Cite (Priority)

1. Maier et al. 2025 — SSR (direct ancestor)
2. Argyle et al. 2023 — Silicon sampling (establishes the problem)
3. Turner et al. 2023/Panickssery et al. 2024 — CAA/activation steering (technical foundation)
4. Jha et al. 2025/2026 — Persona vectors (closest related)
5. Santurkar et al. 2023 / Sun et al. 2025 — Motivation (prompt-based fails for subjective tasks)
6. Brand et al. 2024 — Market research with LLMs (application domain)
