"""
training/classical.py — Classical ML baselines (Section 7).

Trains Random Forest, SVM, and XGBoost under the identical LOSO split
used for deep learning models. Uses the 15-feature vectors from Section 5.1.

Hyperparameter search is grid-search with a budget cap (HP_SEARCH_TRIALS_CLASSICAL).
NaN feature values (e.g. from windows with no R-peaks) are imputed with the
column median from the training fold before fitting.
"""

import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    ALL_SUBJECTS, HP_SEARCH_TRIALS_CLASSICAL, GLOBAL_SEED, seed_everything,
    CHECKPOINT_DIR,
)
from training.metrics import compute_metrics, aggregate_fold_results
from logger import run_logger

import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
import joblib


# ── Hyperparameter search spaces ────────────────────────────────────────────

RF_PARAM_GRID = {
    "clf__n_estimators":  [100, 200, 300],
    "clf__max_depth":     [None, 10, 20, 30],
    "clf__min_samples_leaf": [1, 2, 5],
    "clf__class_weight":  ["balanced"],
}

SVM_PARAM_GRID = {
    "clf__C":     [0.1, 1.0, 10.0, 100.0],
    "clf__gamma": ["scale", "auto"],
    "clf__kernel": ["rbf", "linear"],
    "clf__class_weight": ["balanced"],
}

XGB_PARAM_GRID = {
    "clf__n_estimators":  [100, 200, 300],
    "clf__max_depth":     [3, 5, 7],
    "clf__learning_rate": [0.05, 0.1, 0.2],
    "clf__subsample":     [0.8, 1.0],
    "clf__scale_pos_weight": [1, 2, 5],  # handles imbalance
}


def _make_pipeline(clf_name: str, seed: int) -> Pipeline:
    steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ]
    if clf_name == "RF":
        steps.append(("clf", RandomForestClassifier(random_state=seed, n_jobs=-1)))
    elif clf_name == "SVM":
        steps.append(("clf", SVC(probability=True, random_state=seed)))
    elif clf_name == "XGBoost":
        steps.append(("clf", xgb.XGBClassifier(
            random_state=seed, n_jobs=-1,
            use_label_encoder=False, eval_metric="logloss",
            verbosity=0,
        )))
    else:
        raise ValueError(f"Unknown classifier: {clf_name}")
    return Pipeline(steps)


def _get_param_grid(clf_name: str) -> dict:
    return {"RF": RF_PARAM_GRID, "SVM": SVM_PARAM_GRID, "XGBoost": XGB_PARAM_GRID}[clf_name]


def train_classical_loso(
    dataset_dict: dict,
    clf_name: str,
    seed: int = GLOBAL_SEED,
    n_trials: int = HP_SEARCH_TRIALS_CLASSICAL,
) -> list:
    """
    Run LOSO cross-validation for one classical model.

    Parameters
    ----------
    dataset_dict : full windowed dataset (all subjects)
    clf_name     : "RF", "SVM", or "XGBoost"
    seed         : random seed
    n_trials     : max RandomizedSearchCV trials (budget cap)

    Returns
    -------
    List of per-fold result dicts
    """
    seed_everything(seed)
    fold_results = []

    X_all = dataset_dict["features"]       # (N, 15)
    y_all = dataset_dict["labels"]         # (N,)
    s_all = dataset_dict["subject_ids"]    # (N,)

    config = {
        "model": clf_name,
        "n_trials": n_trials,
        "seed": seed,
    }

    for fold_idx, test_sub in enumerate(ALL_SUBJECTS):
        print(f"  [{clf_name}] Fold {fold_idx+1}/{len(ALL_SUBJECTS)} — test=S{test_sub}", end=" ")
        t0 = time.time()

        train_mask = s_all != test_sub
        test_mask  = s_all == test_sub

        X_train, y_train = X_all[train_mask], y_all[train_mask]
        X_test,  y_test  = X_all[test_mask],  y_all[test_mask]

        # Randomized hyperparameter search (inner CV = 3-fold stratified)
        pipe = _make_pipeline(clf_name, seed)
        search = RandomizedSearchCV(
            pipe,
            param_distributions=_get_param_grid(clf_name),
            n_iter=min(n_trials, 30),
            cv=3,
            scoring="f1",
            random_state=seed,
            n_jobs=-1,
            refit=True,
        )
        search.fit(X_train, y_train)
        best_pipe = search.best_estimator_

        # Evaluate on held-out test subject
        y_pred = best_pipe.predict(X_test)
        y_prob = best_pipe.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, y_pred, y_prob)

        wall_time = time.time() - t0
        print(f"→ F1={metrics['f1']:.3f} AUC={metrics['auc_roc']:.3f} ({wall_time:.1f}s)")

        # Save best pipeline for this fold
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"{clf_name}_fold{test_sub}_best.joblib")
        joblib.dump(best_pipe, ckpt_path)

        # Log
        run_logger.log(
            model_name=clf_name,
            fold=test_sub,
            config={**config, "best_params": search.best_params_},
            seed=seed,
            metrics=metrics,
            checkpoint_path=ckpt_path,
            wall_time_sec=round(wall_time, 2),
        )

        fold_results.append({
            "fold":        test_sub,
            "metrics":     metrics,
            "wall_time_sec": wall_time,
            "best_params": search.best_params_,
            "checkpoint":  ckpt_path,
        })

    return fold_results


def run_all_classical(dataset_dict: dict, seed: int = GLOBAL_SEED) -> dict:
    """Train all three classical baselines and return results keyed by model name."""
    results = {}
    for clf_name in ["RF", "SVM", "XGBoost"]:
        print(f"\n── {clf_name} ──────────────────────────────────────────")
        folds = train_classical_loso(dataset_dict, clf_name, seed=seed)
        fold_metrics = [f["metrics"] for f in folds]
        summary = aggregate_fold_results(fold_metrics)
        results[clf_name] = {"folds": folds, "summary": summary}
        print(f"  {clf_name} LOSO mean: {summary}")
    return results
