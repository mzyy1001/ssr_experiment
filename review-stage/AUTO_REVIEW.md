# Auto Review Loop — PS-SSR Research

## Round 1 (2026-04-20)

### Assessment (Summary)
- Score: 3/10
- Verdict: Not ready for NeurIPS/ICML main track
- Key criticisms:
  1. Final improvement evaluated on only 6 questions — need full 23-question eval
  2. Direct SSR baseline inconsistently evaluated — need canonical baseline on all 23 Qs
  3. Activation steering mostly failed — need honest reframing
  4. No statistical significance tests
  5. Possible tuning leakage (ensemble weight selected after observing results)
  6. Single-domain validation
  7. Thin methodological novelty
  8. Incomplete baseline comparisons (missing silicon sampling, RAG, etc.)
  9. Factorial ablation needed (SSR × steering × KL × QAW)
  10. Persona vector interpretability insufficient
  11. Anchor sensitivity unexplored
  12. Question heterogeneity analysis missing

<details>
<summary>Click to expand full reviewer response</summary>

**Top-Venue Score: 3/10**

As a NeurIPS/ICML submission, this is **not ready**. The idea is interesting, and the negative findings around activation steering are useful, but the current empirical case is too fragile. The main claimed gain is a **5.4% relative JS improvement on only 6 representative questions**, with no significance testing, no full 23-question evaluation, one product domain, and a final method that appears to be a simple ensemble where the strongest component is still direct SSR.

The work is closer to a promising workshop paper or an internal research report than a top-conference paper.

**Critical Weaknesses (ranked):**

1. Final claimed improvement evaluated on only 6 questions
2. Direct SSR baseline not consistently/fully evaluated
3. Activation steering method mostly failed — needs reframing
4. No statistical significance testing
5. Possible tuning leakage / calibration ambiguity
6. Single-domain validation
7. Methodological novelty is thin
8. Survey simulation baseline set incomplete
9. Role of demographic post-strat under-separated from steering
10. Persona vector interpretability insufficient
11. Metric/anchor sensitivity unexplored
12. Survey/question heterogeneity needs analysis

</details>

### Actions Planned
- Priority 1: Run full 23-question evaluation of Direct SSR, Multi-layer steering, Ensemble, and all ablations
- Priority 2: Statistical significance tests (paired bootstrap, permutation)
- Priority 3: Factorial ablation table
- Priority 4: Question-type heterogeneity analysis
- Priority 5: Anchor robustness check

### Status
- Continuing to Round 2
