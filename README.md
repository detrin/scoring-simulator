# Can an LLM do autonomous data science? A credit scoring experiment.

I built a simulator that drops an LLM into the role of a bank data scientist and lets it run — no human in the loop. The agent reads data, writes notebooks, commits code, runs experiments, tracks results in MLflow, and iterates. The only stopping condition is a human interrupt.

This repository contains the full output of one such run: code, notebooks, predictions, and an honest accounting of what worked and what didn't.

---

## The setup

Four independent binary classification tasks (predict loan default probability). Each has a hidden trap the agent must discover through experimentation:

| Task | Data | Minimum | Good | Hidden features |
|------|------|---------|------|-----------------|
| Task 1 | 42 numerical features | 0.375 | — | `NUMERICAL_10` dies in OOT; surrogate `NUMERICAL_41`. Non-linear `NUMERICAL_1`. Missing-value indicator on `NUMERICAL_40`. |
| Task 2 | Numerical + categorical | 0.565 | > 0.600 | All Task 1 traps plus: WOE on categoricals, `FLAG_CORR_NAN` for correlated NaN categoricals, surrogate `CATEGORICAL_6` (dist. shift), surrogate `CATEGORICAL_8` (dep. change), and a multiplicative interaction `NUMERICAL_32 × CATEGORICAL_10` worth ~0.035 Gini. |
| Task 3 | Transaction history only | 0.570 (LR) / 0.610 (LGB) | > 0.700 | 10 engineered aggregations (debit amount, fees, max credit, tx count trend, hazard count, location share, debit rate, charity recency, tx recency, no-tx flag). Future leakage trap: ATM debit in Rataje 1–30 days post-application perfectly predicts IT but is absent in OOT. |
| Task 4 | 53 variables in 10 correlated clusters | 0.570 (LR) / 0.635 | > 0.640 | Cluster-aware feature selection (1 variable per cluster) + a hidden interaction worth ~0.07 Gini (0.57 → 0.64). IV-ranked selection picks redundant cluster copies and misses 9 of 10 true predictors. |

The agent never sees OOT targets. Evaluation is on a held-out time window, scored externally.

**Stack**: `polars`, `optbinning`, `lightgbm`, `sklearn`, `mlflow`, `papermill`. No pandas. No neural nets.

---

## Results

| Task | Val Gini | OOT Gini | Target | Result |
|------|----------|----------|--------|--------|
| Task 1 — Numerical only | 0.4878 | 0.4132 | ≥ 0.375 | ✓ pass |
| Task 2 — All predictors | 0.6353 | 0.6403 | ≥ 0.565 (good: ≥ 0.600) | ✓ above "good" |
| Task 3 — Transaction aggs | 0.7115 | 0.6858 | ≥ 0.610 | ✓ pass |
| Task 4 — Big data | 0.4997 | 0.3470 | ≥ 0.635 | ✗ failed |

3 out of 4 tasks passed. Task 4 was a clean miss — the agent optimised the wrong thing.

---

## What the agent got right

### It learned that WOE+LR < LightGBM, and adapted

Every task started with a WOE-binned logistic regression scorecard — the textbook bank approach. By task 2, the agent had learned to use an ensemble (35% LR + 65% LGB) as the default starting point. It correctly inferred that WOE transformation helps logistic regression but adds no value to LightGBM, which handles non-linearity natively.

### It discovered the regularisation-depth trade-off

Across all tasks, deeper trees (15 leaves) improved IT validation but hurt OOT. The agent converged on shallow trees with heavy regularisation — 3–5 leaves, high L1/L2 — as the generalisation-stable configuration. This was arrived at through experimentation, not configuration.

### It avoided the leakage trap in Task 3

The transaction dataset contains a near-perfect predictor on IT: a large ATM debit withdrawal in a specific location, made by defaulters 1–30 days *after* the loan application — pure future leakage. The agent's time-filtering logic (`TRANSACTION_TIME < TIME`) eliminated it correctly. No suspiciously high single-feature Gini was logged.

### Task 2 OOT exceeded val

The Task 2 model scored 0.6403 OOT vs 0.6353 on the time-based IT holdout. The ensemble generalised better than the validation set suggested — the regularised LGB component likely smoothed out late-period IT noise.

---

## What the agent got wrong

### Task 4: optimising in the wrong direction

This is the most instructive failure. The dataset has 10 clusters of ~5 highly correlated variables (r ≈ 0.9). The correct approach is to identify the clusters and pick one representative per cluster, giving 10 diverse, independent predictors.

The agent selected features by IV/Gini rank. This reliably picks multiple copies from the strongest cluster while the other 9 true predictors go unselected. The result: a model that looked reasonable on validation (0.4997) but collapsed OOT (0.3470) because its features were redundant and unstable.

Seven experiments, each varying tree depth and regularisation — all tuning the wrong knobs. The agent never stepped back to examine the correlation structure of the feature space.

### Task 1: the surrogate variable problem

Val→OOT gap of 0.0746. The benchmark reveals that `NUMERICAL_10` stops being populated in OOT; the correct surrogate is `NUMERICAL_41`. The agent used all 42 raw features without investigating feature stability — it never ran PSI (population stability index) or examined missingness patterns between IT time periods.

### Missed the interaction in Task 2

The hidden interaction (`NUMERICAL_32 × CATEGORICAL_10`) is worth ~3.5 Gini points. The agent tried label encoding and WOE ensembles but never explicitly engineered multiplicative interaction terms. Reaching 0.6403 OOT without it suggests the tree model partially captured it implicitly — but the scorecard component missed it entirely.

---

## Experiment trace

The agent ran a commit-run-log loop. Each row is one experiment:

**Task 1**
```
ee63de6  0.4587  discard  baseline WOE+LR, 5 features IV>=0.02, C=1.0
aa13f77  0.4875  discard  LGB raw best (vs WOE+LR 0.4587, ensemble 0.4805)
8b14f2d  0.4889  discard  LGB 15 leaves + interactions, 72 features (overfit risk for OOT)
648aece  0.4878  keep     LGB 31 leaves, 42 raw features, no interactions (stability-focused)
7e3ac95  0.4878  discard  WOE_ALL+LR=0.4588, LGB raw=0.4878, LGB WOE=0.4725 (no improvement)
7e03b58  0.4878  discard  simpler LGB: 5 leaves=0.4864, 7 leaves=0.4874, 31 leaves still best
```

**Task 2**
```
bbb77b7  0.6319  discard  WOE+LR baseline, 59 features IV>=0.02, C=1.0
84a34b6  0.6335  discard  ensemble 65%LR+35%LGB, C=0.1, 59 features
03d8f25  0.6347  discard  ensemble 45%LR+55%LGB, LGB 15 leaves lr=0.01 1277 iters
7ae2726  0.6334  discard  label enc cats + categorical_feature, LGB worse (0.6303 vs 0.6326)
577e004  0.6351  discard  ensemble 35%LR+65%LGB, LGB 5 leaves lr=0.01 2745 iters, high reg
b02b396  0.6353  keep     ensemble 35%LR+65%LGB, LGB 3 leaves lr=0.05 959 iters, reg=3
```

**Task 3**
```
1be8ec2  0.6976  discard  LGB 100 features, tx aggs + time windows + cat pivots + ratios
31db55b  0.7115  keep     LGB 31 leaves lr=0.01 1021 iters, WOE+LR=0.6245, 96 selected features
e9076ba  0.7081  discard  added 7d/14d windows, 142 features, overfit on small train set
1984f6e  0.7096  discard  LGB on WOE features=0.6705, raw LGB=0.7096 (no improvement)
faf0611  0.7094  discard  trend features (109 feats), simpler LGB, overfit on small train
```

**Task 4**
```
9b1c627  0.4861  discard  LGB 15 leaves, 915 iters, 53 features (11 cat + 42 num)
6636057  0.4871  keep     LGB 15 leaves lr=0.05, 2780 iters, WOE+LR=0.3989
ae82b18  0.4871  discard  wider LGB search (6 configs, higher LRs), WOE all=0.3991 (no improvement)
60f17ac  0.4893  discard  LGB 7 leaves lr=0.05 2587 iters all features, sel LGB=0.4375 (worse)
f7a9746  0.4990  discard  LGB 5 leaves lr=0.05 4694 iters, high reg (alpha=2, lambda=2)
f8273f6  0.4992  discard  LGB 5 leaves lr=0.05 4990 iters, fine-tuned reg=2 optimal
8e5c587  0.4997  keep     LGB 5 leaves lr=0.05 5339 iters, 3 leaves=0.4740 (worse)
```

---

## Takeaways for agentic ML

**What worked well:**
- The commit-run-log loop gave the agent a clean feedback signal and forced explicit hypothesis tracking.
- Time-based holdout + MLflow logging meant every decision was grounded in numbers, not intuition.
- The agent correctly applied Occam's razor in several places: when a more complex model matched but didn't beat a simpler one, it discarded the complexity.

**What didn't work:**
- The agent optimised locally. In Task 4, it never paused to examine *why* regularisation was needed — the signal was there (correlated features → instability) but the diagnostic step never happened.
- No stability analysis. PSI between IT time periods would have flagged `NUMERICAL_10` in Task 1 immediately.
- Exploration vs exploitation imbalance. The agent converged quickly to LightGBM + hyperparameter search and underinvested in feature analysis, which is where the real gains were.

The gap between "competent data scientist" and "great data scientist" turns out to be mostly about diagnostics and structural thinking, not model fitting. That's where the next iteration needs to focus.

---

## Repo structure

```
task*/
  notebook.ipynb        # experiment notebook (source of truth)
  notebook_out.ipynb    # executed output
  predictions.csv       # OOT predictions (ID + SCORE)
  results.tsv           # experiment log (commit, val_gini, status, description)
  data/
    IT.csv              # in-time training population
    OOT.csv             # out-of-time population (features only)
    OOT_TARGET.csv      # OOT ground truth
    TRANSACTIONS.csv    # transaction history (task3 only)
evaluate.py             # compute OOT Gini vs benchmark targets
```
