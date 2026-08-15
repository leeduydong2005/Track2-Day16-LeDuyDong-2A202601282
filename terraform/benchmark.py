#!/usr/bin/env python3
"""Train and benchmark LightGBM on the Kaggle credit-card fraud dataset."""

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split


DATA_PATH = Path.home() / "ml-benchmark" / "creditcard.csv"
RESULT_PATH = Path.home() / "ml-benchmark" / "benchmark_result.json"
RANDOM_STATE = 42


def seconds_since(start: float) -> float:
    return round(time.perf_counter() - start, 4)


def main() -> None:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}. Run the Kaggle download command from README_aws.md first."
        )

    started = time.perf_counter()
    frame = pd.read_csv(DATA_PATH)
    load_seconds = seconds_since(started)

    if "Class" not in frame.columns:
        raise ValueError("Expected the Kaggle fraud dataset to include a 'Class' label column.")

    features = frame.drop(columns=["Class"])
    labels = frame["Class"]
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, stratify=labels, random_state=RANDOM_STATE
    )

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )
    started = time.perf_counter()
    model.fit(x_train, y_train)
    training_seconds = seconds_since(started)

    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    one_row = x_test.iloc[[0]]
    started = time.perf_counter()
    model.predict_proba(one_row)
    single_inference_ms = round((time.perf_counter() - started) * 1000, 4)

    batch = x_test.iloc[:1000]
    started = time.perf_counter()
    model.predict_proba(batch)
    batch_seconds = time.perf_counter() - started

    result = {
        "dataset": str(DATA_PATH),
        "rows": int(len(frame)),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "load_data_seconds": load_seconds,
        "training_seconds": training_seconds,
        "best_iteration": int(model.best_iteration_ or model.n_estimators),
        "auc_roc": round(float(roc_auc_score(y_test, probabilities)), 6),
        "accuracy": round(float(accuracy_score(y_test, predictions)), 6),
        "f1_score": round(float(f1_score(y_test, predictions, zero_division=0)), 6),
        "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 6),
        "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 6),
        "inference_latency_one_row_ms": single_inference_ms,
        "inference_batch_1000_seconds": round(batch_seconds, 6),
        "inference_throughput_rows_per_second": round(len(batch) / batch_seconds, 2),
    }
    RESULT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"\nSaved results to {RESULT_PATH}")


if __name__ == "__main__":
    main()
