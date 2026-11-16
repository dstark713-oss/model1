from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    log_loss, roc_auc_score, average_precision_score,
    precision_score, recall_score, brier_score_loss,
)


def classification_metrics(y_true: pd.Series, p: pd.Series,
                           threshold: float = 0.5) -> dict:
    y = y_true.values
    pr = p.values
    if len(np.unique(y)) < 2:
        return {"n": int(len(y))}
    pred = (pr >= threshold).astype(int)
    return {
        "n": int(len(y)),
        "pos_rate": float(y.mean()),
        "logloss": float(log_loss(y, pr, labels=[0, 1])),
        "roc_auc": float(roc_auc_score(y, pr)),
        "pr_auc": float(average_precision_score(y, pr)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "brier": float(brier_score_loss(y, pr)),
    }


def calibration_table(y_true: pd.Series, p: pd.Series, bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"y": y_true.values, "p": p.values})
    df["bin"] = pd.qcut(df["p"], q=bins, duplicates="drop")
    out = df.groupby("bin", observed=True).agg(
        n=("y", "size"), mean_p=("p", "mean"), mean_y=("y", "mean"),
    ).reset_index()
    return out