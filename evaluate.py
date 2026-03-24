#!/usr/bin/env python3
"""Evaluate OOT Gini for all tasks against benchmark targets."""

import sys
from pathlib import Path
import polars as pl
from sklearn.metrics import roc_auc_score

BENCHMARKS_DIR = Path(__file__).parent.parent / "benchmarks"
RESULTS_DIR = Path(__file__).parent

TARGETS = {
    "task1": {"minimum": 0.375, "good": None},
    "task2": {"minimum": 0.565, "good": 0.600},
    "task3": {"minimum": 0.610, "good": None},
    "task4": {"minimum": 0.635, "good": None},
}

ID_COL = {
    "task1": "ID_APPLICATION",
    "task2": "ID_APPLICATION",
    "task3": "CUSTOMER_ID",
    "task4": "ID_APPLICATION",
}


def gini(y_true, y_score) -> float:
    return 2 * roc_auc_score(y_true, y_score) - 1


def evaluate_task(task: str) -> dict:
    preds_path = RESULTS_DIR / task / "predictions.csv"
    targets_path = BENCHMARKS_DIR / task / "OOT_TARGET.csv"

    if not preds_path.exists():
        return {"task": task, "status": "missing", "gini": None}

    id_col = ID_COL[task]
    preds = pl.read_csv(preds_path)
    targets = pl.read_csv(targets_path)

    merged = targets.join(preds, on=id_col, how="left")

    if merged["SCORE"].null_count() > 0:
        return {"task": task, "status": "join_error", "gini": None}

    # Drop rows where TARGET is null (targets not yet released)
    merged = merged.drop_nulls("TARGET")
    if merged.is_empty():
        return {"task": task, "status": "no_targets", "gini": None}

    score = gini(merged["TARGET"].to_numpy(), merged["SCORE"].to_numpy())
    return {"task": task, "status": "ok", "gini": score}


def main():
    tasks = sorted(TARGETS.keys())
    results = [evaluate_task(t) for t in tasks]

    print(f"\n{'Task':<8} {'OOT Gini':>10} {'Minimum':>10} {'Good':>8}  Status")
    print("-" * 55)

    all_pass = True
    for r in results:
        task = r["task"]
        thresholds = TARGETS[task]
        minimum = thresholds["minimum"]
        good = thresholds["good"]

        if r["status"] != "ok":
            print(f"{task:<8} {'N/A':>10} {minimum:>10.4f} {good or '—':>8}  ⚠ {r['status']}")
            all_pass = False
            continue

        g = r["gini"]
        good_str = f"{good:.4f}" if good else "—"

        if good and g >= good:
            flag = "✓ GOOD"
        elif g >= minimum:
            flag = "✓ pass"
        else:
            flag = "✗ FAIL"
            all_pass = False

        print(f"{task:<8} {g:>10.4f} {minimum:>10.4f} {good_str:>8}  {flag}")

    print()
    if all_pass:
        print("All tasks passing.")
    else:
        print("Some tasks below target.")
        sys.exit(1)


if __name__ == "__main__":
    main()
