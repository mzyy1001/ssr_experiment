# Hyperparameter Pre-Registration Protocol

To avoid hidden test-set tuning with only 6 questions, we pre-register
the primary configuration BEFORE examining per-question results.

## Primary Configuration (pre-registered)
- **Layer**: 16 (middle layer — standard choice for Qwen3-8B with 32 layers)
- **Alpha (steering strength)**: 2.0 (moderate — following CAA literature defaults)
- **N samples per cluster**: 20
- **Negation strategy**: distant_cluster (most principled contrastive approach)
- **Steering mode**: all tokens
- **SSR normalization**: min_sub (original from Project 1)
- **Weight approach**: A (SSR weights from Project 1)
- **Steering strategy**: steer_then_aggregate

## Rationale
All primary choices are justified a priori by:
- Literature precedents (middle layer for CAA, moderate alpha)
- Consistency with Project 1 (normalization, weights)
- Theoretical motivation (steer-then-aggregate preserves mixture structure)

## Ablation Results
All ablations (layer sweep, alpha sweep, normalization comparison, etc.)
are reported transparently as sensitivity analysis, NOT as configuration selection.

## Oracle Results
Approach B weights and any configuration selected by JS minimization
are clearly labeled "oracle" in all tables and figures.

## Reporting Order
1. Report primary configuration results first
2. Report ablations showing robustness (or lack thereof)
3. Report oracle upper bounds separately
4. Per-question breakdown for all conditions
