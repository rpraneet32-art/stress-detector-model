"""
training/metrics.py — Evaluation metrics (Section 9.1).

Reports per-fold:
  - Accuracy
  - Balanced F1-score
  - Sensitivity (Recall)
  - Specificity
  - AUC-ROC
  - Inference latency per window on CPU (hard target: < 50 ms)
"""

import time
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score,
    roc_auc_score, confusion_matrix,
)
from typing import Dict, Any


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray = None,
) -> Dict[str, float]:
    """
    Compute all required metrics for one fold.

    Parameters
    ----------
    y_true : (N,) ground truth binary labels
    y_pred : (N,) predicted binary labels
    y_prob : (N,) predicted probability for class 1 (for AUC-ROC)

    Returns
    -------
    dict with: accuracy, f1, sensitivity, specificity, auc_roc
    """
    acc  = float(accuracy_score(y_true, y_pred))
    f1   = float(f1_score(y_true, y_pred, average="binary", zero_division=0))
    sens = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))

    # Specificity = TN / (TN + FP)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        spec = float(tn / (tn + fp + 1e-8))
    else:
        spec = float("nan")

    # AUC-ROC
    if y_prob is not None and len(np.unique(y_true)) > 1:
        auc = float(roc_auc_score(y_true, y_prob))
    else:
        auc = float("nan")

    return {
        "accuracy":    round(acc,  4),
        "f1":          round(f1,   4),
        "sensitivity": round(sens, 4),
        "specificity": round(spec, 4),
        "auc_roc":     round(auc,  4),
    }


def benchmark_latency(
    model: torch.nn.Module,
    window_shape: tuple = (1, 3, 384),
    n_runs: int = 200,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Measure inference latency per window on CPU.
    Hard target: < 50 ms (spec Section 9.1).

    Returns dict with mean_ms, std_ms, pass (bool against 50 ms target).
    """
    model = model.to(device)
    model.eval()
    dummy = torch.randn(*window_shape).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy)

    # Timed runs
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _  = model(dummy)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)  # ms

    mean_ms = float(np.mean(times))
    std_ms  = float(np.std(times))
    passes  = mean_ms < 50.0

    return {
        "latency_mean_ms": round(mean_ms, 3),
        "latency_std_ms":  round(std_ms,  3),
        "latency_pass":    passes,
        "latency_target_ms": 50.0,
    }


def aggregate_fold_results(fold_metrics: list) -> Dict[str, Any]:
    """
    Given a list of per-fold metric dicts, compute mean ± std across folds.
    """
    keys = [k for k in fold_metrics[0] if k not in ("latency_pass",)]
    summary = {}
    for k in keys:
        vals = [m[k] for m in fold_metrics if k in m]
        vals = [v for v in vals if v == v]  # drop NaN
        if vals:
            summary[f"{k}_mean"] = round(float(np.mean(vals)), 4)
            summary[f"{k}_std"]  = round(float(np.std(vals)),  4)
    return summary
