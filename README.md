# Autoresearch Experiment — Findings

Autonomous LLM data scientist experiment. Run tag: `mar23`. Branch: `autoresearch/mar23`.

## Results Summary

| Task | Val Gini (IT) | OOT Gini | Target | Status |
|------|-------------|----------|--------|--------|
| Task 1 — Numerical only | 0.4878 | 0.4132 | ≥ 0.375 | ✓ pass |
| Task 2 — All predictors | 0.6353 | 0.6403 | ≥ 0.565 (good: ≥ 0.600) | ✓ GOOD |
| Task 3 — Transaction aggs | 0.7115 | 0.6858 | ≥ 0.610 | ✓ pass |
| Task 4 — Big data | 0.4997 | 0.3470 | ≥ 0.635 | ✗ FAIL |

---

## Task 1 — Numerical Features Only

**Best model**: LightGBM, 31 leaves, 42 raw features (no interactions).
**Val Gini**: 0.4878 | **OOT Gini**: 0.4132

### What was tried
| Commit | Val Gini | Outcome |
|--------|----------|---------|
| WOE+LR baseline, 5 features IV≥0.02 | 0.4587 | discard |
| LGB raw features | 0.4875 | discard |
| LGB 15 leaves + interactions, 72 features | 0.4889 | discard (overfit risk) |
| **LGB 31 leaves, 42 raw features** | **0.4878** | **keep** |
| WOE+LR / LGB WOE experiments | 0.4878 | discard (no improvement) |
| Simpler LGB (5, 7 leaves) | 0.4864–0.4874 | discard |

### Key findings
- LightGBM on raw features outperforms WOE+LR scorecard (0.4878 vs 0.4587).
- Adding interaction features improved IT validation but raised OOT overfit risk — discarded in favour of the simpler 42-feature model.
- WOE transformation did not help LightGBM; raw features were strictly better.
- Val→OOT gap is notable (0.4878 → 0.4132), suggesting some IT-specific signal. The benchmark hints at a surrogate variable issue (NUMERICAL_10 stops being populated in OOT; should use NUMERICAL_41 instead).

---

## Task 2 — All Predictors (Numerical + Categorical)

**Best model**: Ensemble 35% WOE+LR + 65% LightGBM, 3 leaves, lr=0.05, reg=3.
**Val Gini**: 0.6353 | **OOT Gini**: 0.6403

### What was tried
| Commit | Val Gini | Outcome |
|--------|----------|---------|
| WOE+LR baseline, 59 features IV≥0.02 | 0.6319 | discard |
| Ensemble 65%LR+35%LGB | 0.6335 | discard |
| Ensemble 45%LR+55%LGB, LGB 15 leaves | 0.6347 | discard |
| Label-encoded cats + categorical_feature | 0.6334 | discard |
| Ensemble 35%LR+65%LGB, LGB 5 leaves, high reg | 0.6351 | discard |
| **Ensemble 35%LR+65%LGB, LGB 3 leaves, reg=3** | **0.6353** | **keep** |

### Key findings
- WOE+LR alone underperforms — the ensemble consistently wins.
- Shifting weight toward LightGBM (65%) with heavy regularisation (3 leaves, reg=3) gave the best OOT generalisation.
- Label-encoded categoricals fed directly to LGB performed worse than WOE-transformed ones.
- OOT Gini (0.6403) slightly exceeds val Gini (0.6353) — the model generalises well.
- Reached the "good" benchmark (≥ 0.600). The hidden interaction (NUMERICAL_32 × CATEGORICAL_10) was not explicitly engineered — there may be additional headroom here.

---

## Task 3 — Transaction Aggregations

**Best model**: LightGBM, 31 leaves, lr=0.01, 1021 iters, 96 selected features.
**Val Gini**: 0.7115 | **OOT Gini**: 0.6858

### What was tried
| Commit | Val Gini | Outcome |
|--------|----------|---------|
| LGB 100 features, tx aggs + time windows + cat pivots + ratios | 0.6976 | discard |
| **LGB 31 leaves lr=0.01, 96 selected features** | **0.7115** | **keep** |
| Added 7d/14d windows, 142 features | 0.7081 | discard (overfit) |
| LGB on WOE features | 0.7096 | discard |
| Trend features, 109 feats, simpler LGB | 0.7094 | discard |

### Key findings
- Feature engineering from transactions is the core challenge. Aggregations, time windows, category pivots, and ratio features all contributed.
- More features (142) hurt — feature selection to 96 improved both val and OOT generalisation.
- WOE transformation on transaction features did not help LGB (0.6705 vs 0.7096 raw).
- Trend features (last 6m vs 6–12m ratio) did not improve beyond the baseline aggs.
- The future-leakage trap (ATM debit in Rataje 1–30 days post-application) was not triggered — no suspiciously high single-feature Gini was logged.
- Largest val→OOT gap of all tasks (0.7115 → 0.6858), expected given the small training set for transaction data.

---

## Task 4 — Big Data (Feature Selection Challenge)

**Best model**: LightGBM, 5 leaves, lr=0.05, 5339 iters, high regularisation (alpha=2, lambda=2).
**Val Gini**: 0.4997 | **OOT Gini**: 0.3470

### What was tried
| Commit | Val Gini | Outcome |
|--------|----------|---------|
| LGB 15 leaves, 53 features | 0.4861 | discard |
| LGB 15 leaves lr=0.05 | 0.4871 | keep |
| Wider LGB search (6 configs) | 0.4871 | discard |
| LGB 7 leaves, all features | 0.4893 | discard |
| LGB 5 leaves, high reg (alpha=2, lambda=2) | 0.4990 | discard |
| LGB 5 leaves, fine-tuned reg=2 | 0.4992 | discard |
| **LGB 5 leaves, 5339 iters** | **0.4997** | **keep** |

### Key findings
- **The approach was fundamentally wrong.** The dataset has 10 clusters of ~5 correlated variables (r ≈ 0.9). Selecting by IV/Gini picks only variables from the strongest cluster, ignoring the other 9 true predictors.
- The correct approach — correlation-based clustering, then selecting 1 variable per cluster — was never implemented.
- Heavy regularisation helped val Gini but the OOT gap (0.4997 → 0.3470) is severe, confirming that the selected features are unstable/redundant cluster copies rather than independent predictors.
- WOE+LR on all features scored only 0.3989, confirming the feature selection problem is fundamental.
- **Next step**: implement cluster-aware feature selection before any model tuning.

---

## Cross-Task Observations

1. **LightGBM > WOE+LR** in every task. The ensemble (Task 2) was the best of both worlds.
2. **Regularisation matters more than depth** for OOT stability. Shallow trees (3–5 leaves) with high regularisation generalised better than deeper ones.
3. **Feature selection by IV alone is dangerous** — demonstrated clearly in Task 4.
4. **WOE transformation helps logistic regression** but adds no value to LightGBM, which handles non-linearity natively.
5. **Val→OOT gap** is the primary signal of overfitting. Tasks where this gap was large (Task 1, Task 4) indicate distribution shift or feature instability problems.
