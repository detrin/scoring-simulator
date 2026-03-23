# autoresearch — scoring simulator

This is an experiment to have an LLM act as an autonomous data scientist in a bank, building credit scoring models.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar23`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the data files**: inspect column names, shapes, and basic statistics for all in-scope files before writing any code.
4. **Initialize results.tsv**: Create `task*/results.tsv` with just the header row.
5. **Confirm and go**.

Once you get confirmation, kick off the experimentation.

## The problem

You are a data scientist in a bank. Each task is a binary classification problem: predict the probability of default (`TARGET = 1`) for loan applicants.

- **`IT.csv`** — in-time (training) population. Use this for model development and validation.
- **`OOT.csv`** — out-of-time population (a later time window). Features only, no target. Produce predictions on it.
- **`TRANSACTIONS.csv`** — transaction history (task3 only). All features must be engineered from here.

You will never see OOT targets. They are held externally.

Your working metric is **Gini coefficient** (= 2 × AUC − 1, higher is better).
Always validate on a **time-based holdout** of IT.csv — use the `TIME` column to split, with later months as validation. Never use random splits. Credit models must be validated chronologically.

## Stack

**Runtime**: `uv` for all execution. Run notebooks with:
```
uv run papermill task*/notebook.ipynb task*/notebook_out.ipynb
```
Redirect output: `uv run papermill ... > run.log 2>&1`

**Solutions must be notebooks** (`task*/notebook.ipynb`). Each experiment is a single notebook — data loading, feature engineering, model training, validation, and OOT predictions all in one place. Do not split logic across multiple scripts.

**Libraries:**
- `polars` for all data loading and manipulation — not pandas
- `optbinning` for feature binning and WOE transformation
- `lightgbm` or `sklearn` LogisticRegression as the final model
- `mlflow` for experiment tracking — log val_gini, model parameters, and feature list for every run

**MLflow**: use a local tracking URI (`mlflow.set_tracking_uri("mlruns")`). Log at minimum:
- `val_gini` metric
- all model hyperparameters
- number of features used
- task name and run tag as tags

**You CANNOT:**
- Modify any data files
- Access OOT targets
- Use neural networks, XGBoost, CatBoost, or other model families
- Use pandas

## Tasks

### Task 1 — Numerical features only
42 numerical features. Build a scorecard using WOE binning and logistic regression. Focus on feature quality over quantity.

### Task 2 — All predictors
Numerical and categorical features. Requires correct handling of both feature types.

### Task 3 — Transaction aggregations
`IT.csv` contains only `CUSTOMER_ID`, `TIME`, `TARGET`. All features must be engineered from `TRANSACTIONS.csv`.

**Critical rule**: only use transactions where `TRANSACTION_TIME < TIME` (no future leakage).
Time distances should be computed as `TIME − TRANSACTION_TIME`.

### Task 4 — Big data
~53 variables. Feature selection is the core challenge — think carefully about how to pick the right variables rather than using the most predictive ones naively.

## What to work on

Fair game:
- Feature engineering (interactions, ratios, aggregations, time-window features)
- Binning strategy (optbinning parameters, monotonicity constraints, bin granularity)
- WOE transformation and IV-based feature selection
- Stability analysis (PSI, WOE shift between IT time periods)
- Missing value treatment
- Model hyperparameters (regularization, depth, num_leaves)
- Ensemble of LogReg scorecard + LightGBM

**Simplicity criterion**: a bank scorecard must be explainable to regulators. All else equal, simpler is better. Removing a feature and keeping equal Gini is always a win.

## Output format

After each experiment, produce:

```
task*/predictions.csv
```

Two columns: `ID_APPLICATION` (tasks 1, 2, 4) or `CUSTOMER_ID` (task 3) and `SCORE` (predicted default probability, 0–1). Row order must match `OOT.csv`.

## Logging results

Log to `task*/results.tsv` (tab-separated):

```
commit	val_gini	status	description
```

- `commit`: 7-char git hash
- `val_gini`: Gini on time-based IT holdout (0.0000 for crashes)
- `status`: `keep`, `discard`, or `crash`
- `description`: short note on what was tried

## The experiment loop

LOOP FOREVER:

1. Check git state.
2. Pick an idea. Implement it in `task*/notebook.ipynb`.
3. `git commit`
4. Run: `uv run papermill task*/notebook.ipynb task*/notebook_out.ipynb > task*/run.log 2>&1`
5. Extract val_gini: `grep "val_gini" task*/run.log` or read from mlflow.
6. If the grep is empty, the run crashed — check `tail -n 50 task*/run.log` for the traceback.
7. Produce `task*/predictions.csv` on OOT.csv.
8. Log to `task*/results.tsv`.
9. val_gini improved → keep the commit, advance.
10. val_gini equal or worse → `git reset --hard`, discard.

**NEVER STOP**: Do not pause to ask the user if you should continue. The user may be away. You are autonomous. If you run out of ideas, go deeper — re-examine feature distributions, try different binning strategies, revisit feature selection, investigate stability, try the other model family. The loop runs until the human interrupts you.

**Crashes**: fix trivial bugs and retry. If the idea is fundamentally broken, log `crash`, reset, move on.

**Tasks are independent**: work on one task at a time or rotate. Each has its own `results.tsv` and `predictions.csv`.
