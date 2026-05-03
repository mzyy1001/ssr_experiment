# Auto Review Loop — PS-SSR Story v2 (paper-SSR baseline integration)

**Loop session:** 2026-04-23
**Reviewer:** Codex MCP (gpt-5.4, xhigh reasoning)
**Thread ID:** `019dba95-6058-7e30-811a-2c2f624f6c60`
**Difficulty:** medium (default)
**Target document:** `STORY_v2.md`
**Stop condition met:** Round 5, score 7/10 (after author-flagged correction), verdict "Ready" for workshop / applied venue.
**Prior review log:** archived at `AUTO_REVIEW_2026-04-20.md.bak`

---

## Score progression

| Round | Score | Verdict | Key change |
|---|---|---|---|
| 1 | 4/10 | Not ready | Story too rosy; relies on weak transfer baseline |
| 2 | 5/10 | Almost (workshop) | Narrowed claims; added side-by-side baselines, per-Q honest table |
| 3 | 5/10 | Almost (taxonomy bug) | Reviewer caught Direct-SSR taxonomy mismatch with code |
| 4 | 6/10 | Ready (workshop) | Corrected taxonomy; M0 is in same family as Direct SSR; (incorrectly) attributed cluster-SSR family to "prior in-repo work" |
| 5 | **7/10** | **Ready (workshop / applied)** | Author-flagged correction: Maier paper has NO clusters (verified §A.4.3 + eq. 1); all cluster-SSR code authored by mzyy1001 in this same project. M0 is the genuine primary novel contribution, not "prior work" |

---

## Round 1 — Score 4/10

**Top criticisms:** weak transfer-baseline-only headline; over-strong "cluster prior is load-bearing" causal claim; K_xy cross-paper comparison unsound; M2 looks test-set tuned; Q2 dominates the gain; steering should be demoted; "personas" should be "latent clusters."

**Actions taken:** Rewrote STORY_v2.md narrowing all 7 weaknesses; added side-by-side paper-SSR + Direct SSR baselines; made M0 primary scientific claim and M2 secondary best-achieved; made JS the primary metric; added per-Q honest table including M0 (+18% Q5 loss) and M2 (+50% Q3 loss); added 5Q-without-Q2 mean; demoted steering; renamed "personas" → "latent topic clusters."

---

## Round 2 — Score 5/10

**Top criticisms:** needs (a) cheap-knobs-flat table from improve_orig6.json, (b) hierarchical-family-stable table from alignment_6q.json, (c) "not only Q2" sentence (5/6 wins), (d) explicit case-study framing in title, (e) 3-row method comparator, (f) keep causal language narrow.

**Actions taken:** Added "case study" framing to working title; added 3-row method comparator table; added (a) cheap-knobs-flat table from `improve_orig6.json`, (b) hierarchical-family-stable table from `alignment_6q.json`, (c) "M0 wins 5/6 questions; 5Q-no-Q2 mean improves 9.3%" sentence.

---

## Round 3 — Score 5/10 (taxonomy bug)

**Critical finding:** reviewer read `experiments/improve_orig6.py:91 direct_ssr()` directly and verified it is **already cluster-conditioned** (`for cid in cids` + aggregation via `cluster_weights`), not a "single pool / no clusters" baseline as my v4 taxonomy claimed. Same finding for `experiments/baselines.py:265` and `experiments/persona_methods.py:116`.

**Actions taken:**
- Verified the bug — reviewer is right
- Rewrote method taxonomy: Direct SSR and M0 are in the **same algorithmic family** (per-cluster SSR over real posts, mass-weighted by cluster). M0 is Direct SSR with `n_posts=50` and `random_state=42`.
- Reframed novel contribution: **M1 (per-cluster LLM-dist) and M2 (ensemble) are the novel methods, not M0.** M0 is reproducibility confirmation.
- Updated working title: "...the Limited Marginal Value of a Per-Cluster LLM-Distribution Ensemble"
- Updated abstract to make novel-vs-diagnostic split explicit

---

## Round 4 — Score 6/10 — STOP CONDITION MET

**Verdict:** Ready for workshop / applied venue. Almost for EMNLP Findings. Not ready for main-track.

**Reviewer raw response:**

> Yes. The corrected taxonomy in STORY_v2.md, checked against improve_orig6.py:88 and dump_m0_m2_pmfs.py:215, resolves the method-definition consistency problem. The novel-vs-diagnostic split is now scientifically defensible:
> - Paper-SSR = transfer diagnostic
> - Direct SSR / M0 = same hierarchical SSR family
> - M1 / M2 = actual novel methods
> - M2 = modest, non-uniform ~7.5–7.6% gain over the strong in-domain baseline
>
> Re-score: 6/10. Re-verdict: Ready for workshop/applied venue. Almost for EMNLP Findings. Not ready for main-track.
>
> Yes: the score is now >= 6. This clears my stop threshold for the auto-review loop.
>
> Non-blocking polish:
> - Update stale "Open questions" section that still says M0 as primary
> - Soften "do not approach M0" to "stay very close to M0"

**Actions taken:** Applied both non-blocking polish fixes; added Final framing summary section; loop terminated.

---

## Round 5 — Score 7/10 — STOP (after Round 4 → 5 author correction)

**Trigger:** Author flagged that the Round 4 conclusion ("M0 not novel because Direct SSR is prior in-repo baseline") was based on a wrong assumption — verified that Maier paper has no clusters and all cluster-SSR code in repo is authored by the same author in this same project.

**Verification:**
- Read Maier paper §A.4.3 + eq. 1 directly: `p_s(i) = (1/N_s) Σ_c δ_irc` — flat per-consumer aggregation, no clusters.
- `git log` on `improve_orig6.py`, `baselines.py`, `persona_methods.py`, `steered_ssr.py`, `dump_m0_m2_pmfs.py` — all author = mzyy1001, all commits within this same project (earliest 2026-04-14).
- Conclusion: cluster-weighted hierarchical SSR family is genuinely novel, not inherited prior work.

**Reviewer raw response:**

> I accept the corrected attribution. The Maier paper defines survey prediction as a flat aggregation over synthetic consumers in Appendix A.2 / A.4.3, and run_paper_ssr_6q.py matches that no-cluster setup. By contrast, improve_orig6.py and dump_m0_m2_pmfs.py are clearly cluster-level, mass-weighted methods. Repo history also shows those files were authored in this project by mzyy1001 on 2026-04-14 / 2026-04-21, not inherited from external prior work.
>
> 1. Yes. M0 is novel relative to Maier et al. The baseline paper is per-consumer and flat-averaged; your cluster-weighted hierarchical family is a genuine method change, not just internal renaming.
> 2. New score: 7/10. Above 6.
> 3. Defense against "trivial aggregation reformulation": (a) different unit of inference (consumers → clusters), (b) different evidence source (LLM free-text → real social posts), (c) different mixture model (flat avg → cluster-mass weighted), (d) large + stable empirical delta robust across reweighting. Frame as "minimal but substantive hierarchical reformulation of SSR."
> 4. Re-verdict: Ready for workshop/applied; Almost for EMNLP Findings; Not ready for main-track (n=6).
> 5. Polish: fix arithmetic (M1 vs M0 is 0.4% not 5.6%; M2 vs M0 is 5.6%; M2 vs Direct-SSR n=50 is 7.6%); use one naming convention (M0 / hierarchical cluster-weighted SSR, with note that earlier files used "Direct SSR"); keep "single-domain case study" visible in title and abstract.

**Actions taken:** Applied all three polish fixes (arithmetic corrected, naming convention block added under title, "Single-Domain Case Study" placed in title + abstract).

---

## Final state (post-Round 5)

- **Document:** `/home/mzyy1001/business/STORY_v2.md` (v6)
- **Primary novel contribution (corrected attribution):** **Hierarchical cluster-weighted SSR (M0)**. Aggregates SSR signal within latent topic clusters of real social-media posts and combines clusters by per-question empirical mass, instead of paper's flat averaging across individual synthetic consumers. **−37.5% JS over paper-SSR**, robust across 6 reweighting schemes.
- **Secondary contribution:** M1 (per-cluster LLM-dist) ties M0; M2 (M0+M1 ensemble) gives an additional 5.6% over M0 with non-uniform per-Q behavior.
- **Diagnostic:** paper-SSR transfer degrades on Chinese health domain (JS=0.0430), Q2 dominates the failure.
- **Steering story:** demoted to appendix.
- **Target venue:** workshop / applied NLP venue (ready); EMNLP Findings (almost); main-track (not ready, n=6 case-study scope).
- **Honest concessions retained:** non-uniform per-Q gains (M0 +18% Q5 loss, M2 +50% Q3 loss); Q2 dominates the cross-baseline delta; M0/Direct-SSR are the same family (different hyperparameter / sampling-seed runs).

---

## Method Description (for downstream paper-illustration)

Pipeline architecture, in data-flow order:

1. **Topic clustering**: ~12K Weibo/Xiaohongshu posts on consumer health are clustered into 66 latent topic clusters; the top-10 clusters by aggregated cluster-mass weight (across the 6 questions) are selected as inference units.
2. **Per-cluster SSR (M0)**: For each cluster, sample 50 real posts; embed each via bge-base-zh-v1.5; compute cosine similarity to question-specific anchor sentences (auto-generated by Qwen3-8B); apply softmax to obtain a per-cluster option-distribution PMF. Aggregate via per-question cluster-mass weights.
3. **Per-cluster LLM-distribution estimator (M1, this paper)**: For each cluster, sample 8 example posts; prompt Qwen3-8B with cluster topic + example posts + survey question + options; ask the model to emit a JSON option-distribution; average over 3 samples per cluster. Aggregate via per-question cluster-mass weights.
4. **Ensemble (M2, this paper)**: For each cluster, take 0.5×M0_PMF + 0.5×M1_PMF; aggregate via per-question cluster-mass weights.
5. **Evaluation**: Mean JS divergence (primary), K_xy / C_xy (cross-paper diagnostics) over 6 real Chinese consumer-health survey questions.

The hierarchical structure (per-cluster computation + cluster-mass aggregation) is shared across all M0/M1/M2 methods and is the **primary novel contribution** of this work — it is absent from the published per-consumer flat-aggregation baseline (Maier et al., paper §A.4.3 + eq. 1). M1 and M2 are secondary refinements built on top of the same hierarchical structure.
