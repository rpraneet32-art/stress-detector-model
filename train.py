"""
train.py — Main training orchestrator (Sections 7–9).

Runs the complete pipeline end-to-end:
  1. Load preprocessed dataset (or build it from scratch if not cached)
  2. Run LOSO training for all 4 DL models across all 15 folds
  3. Run LOSO training for 3 classical baselines (RF, SVM, XGBoost)
  4. Compute aggregate metrics and print comparison table
  5. Export best model to ONNX and validate

Usage:
  python train.py [--skip-preprocessing] [--models cnn bilstm transformer stgcn]
                  [--classical-only] [--dl-only] [--fold N]

All arguments are optional. By default runs everything.
"""

import os
import sys
import json
import argparse
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    ALL_SUBJECTS, PROCESSED_DATA_DIR, OUTPUTS_DIR, CHECKPOINT_DIR,
    BATCH_SIZE, GLOBAL_SEED, seed_everything,
)
from data.windowing import build_dataset, save_dataset, load_dataset
from training.dataset import loso_split, compute_class_weights, make_dataloaders
from training.trainer import train_one_fold
from training.classical import run_all_classical
from training.metrics import aggregate_fold_results, benchmark_latency
from export.onnx_export import export_to_onnx, validate_onnx
import models as model_module


# ── Model registry ───────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "cnn":         model_module.CNN1D,
    "bilstm":      model_module.BiLSTM,
    "transformer": model_module.TemporalTransformer,
    "stgcn":       model_module.STGCN,
}


def parse_args():
    p = argparse.ArgumentParser(description="Stress Detector Training Pipeline")
    p.add_argument("--skip-preprocessing", action="store_true",
                   help="Load cached preprocessed dataset instead of rebuilding")
    p.add_argument("--models", nargs="+", default=list(MODEL_REGISTRY.keys()),
                   choices=list(MODEL_REGISTRY.keys()),
                   help="DL models to train (default: all four)")
    p.add_argument("--classical-only", action="store_true",
                   help="Only train classical baselines")
    p.add_argument("--dl-only", action="store_true",
                   help="Only train deep learning models")
    p.add_argument("--fold", type=int, default=None,
                   help="Run only this specific test subject fold (for debugging)")
    p.add_argument("--seed", type=int, default=GLOBAL_SEED)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    return p.parse_args()


def print_comparison_table(dl_results: dict, classical_results: dict):
    """Print consolidated comparison table (Section 9.2)."""
    header = f"\n{'Model':<16} {'Acc':>7} {'F1':>7} {'Sens':>7} {'Spec':>7} {'AUC':>7} {'Lat(ms)':>9} {'LatOK':>6}"
    sep    = "─" * len(header)
    print(sep)
    print(header)
    print(sep)

    rows = []

    for model_name, result in dl_results.items():
        fold_metrics = [f["metrics"] for f in result["folds"]]
        s = aggregate_fold_results(fold_metrics)
        lat_mean = np.mean([f["metrics"].get("latency_mean_ms", float("nan"))
                            for f in result["folds"]])
        lat_ok   = lat_mean < 50.0
        rows.append((model_name, s, lat_mean, lat_ok))

    for clf_name, result in classical_results.items():
        fold_metrics = [f["metrics"] for f in result["folds"]]
        s = aggregate_fold_results(fold_metrics)
        rows.append((clf_name, s, float("nan"), None))

    for name, s, lat, lat_ok in rows:
        lat_str = f"{lat:.1f}" if not np.isnan(lat) else "  N/A"
        ok_str  = "✅" if lat_ok else ("❌" if lat_ok is False else " --")
        print(
            f"{name:<16}"
            f" {s.get('accuracy_mean', float('nan')):>7.4f}"
            f" {s.get('f1_mean', float('nan')):>7.4f}"
            f" {s.get('sensitivity_mean', float('nan')):>7.4f}"
            f" {s.get('specificity_mean', float('nan')):>7.4f}"
            f" {s.get('auc_roc_mean', float('nan')):>7.4f}"
            f" {lat_str:>9}"
            f" {ok_str:>6}"
        )
    print(sep)


def recommend_best_model(dl_results: dict, classical_results: dict) -> str:
    """
    Recommend the best DL model to carry forward to hardware (Section 9.2).
    Decision criteria: must beat best classical baseline on F1 AND meet <50ms latency.
    Among qualifying DL models, pick by AUC-ROC.
    """
    # Best classical F1
    best_clf_f1 = 0.0
    for clf_name, result in classical_results.items():
        fold_metrics = [f["metrics"] for f in result["folds"]]
        s = aggregate_fold_results(fold_metrics)
        best_clf_f1 = max(best_clf_f1, s.get("f1_mean", 0.0))

    # Evaluate DL models
    candidates = []
    for model_name, result in dl_results.items():
        fold_metrics = [f["metrics"] for f in result["folds"]]
        s = aggregate_fold_results(fold_metrics)
        lat_mean = np.nanmean([f["metrics"].get("latency_mean_ms", float("nan"))
                                for f in result["folds"]])
        f1_mean  = s.get("f1_mean", 0.0)
        auc_mean = s.get("auc_roc_mean", 0.0)
        meets_lat = lat_mean < 50.0
        beats_clf = f1_mean > best_clf_f1

        candidates.append({
            "name": model_name, "f1": f1_mean, "auc": auc_mean,
            "latency_ms": lat_mean, "meets_latency": meets_lat,
            "beats_classical": beats_clf,
        })

    # Sort: qualifying models first (meets lat + beats clf), then by AUC
    qualified = [c for c in candidates if c["meets_latency"] and c["beats_classical"]]
    fallback  = [c for c in candidates if not (c["meets_latency"] and c["beats_classical"])]
    qualified.sort(key=lambda c: c["auc"], reverse=True)
    fallback.sort(key=lambda c: c["auc"], reverse=True)
    ranked = qualified + fallback

    best = ranked[0] if ranked else None
    print("\n── Model Recommendation (Section 9.2) ────────────────────")
    for c in ranked:
        flag = "⭐ RECOMMENDED" if c == best else ""
        print(
            f"  {c['name']:<16}  F1={c['f1']:.4f}  AUC={c['auc']:.4f}  "
            f"Lat={c['latency_ms']:.1f}ms  "
            f"{'✅lat' if c['meets_latency'] else '❌lat'}  "
            f"{'✅clf' if c['beats_classical'] else '❌clf'}  {flag}"
        )
    if best:
        print(f"\n  → Selected: {best['name']}")
        if not best["meets_latency"]:
            print("  ⚠️  WARNING: Selected model does NOT meet <50ms latency target")
        if not best["beats_classical"]:
            print("  ⚠️  WARNING: Selected model does NOT beat strongest classical baseline")
    return best["name"] if best else None


def main():
    args = parse_args()
    seed_everything(args.seed)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print("=" * 60)
    print("  STRESS DETECTOR — TRAINING PIPELINE")
    print("=" * 60)

    # ── Step 1: Load or build dataset ────────────────────────────────────
    processed_path = os.path.join(PROCESSED_DATA_DIR, "wesad_windows.npz")
    if args.skip_preprocessing and os.path.exists(processed_path):
        print("\n[1/5] Loading cached preprocessed dataset...")
        dataset = load_dataset(PROCESSED_DATA_DIR)
    else:
        print("\n[1/5] Building preprocessed dataset from raw WESAD files...")
        dataset = build_dataset()
        save_dataset(dataset)

    print(f"  Total windows: {len(dataset['labels'])}")
    print(f"  Stress fraction: {(dataset['labels']==1).mean():.3f}")
    print(f"  Subjects: {sorted(set(dataset['subject_ids'].tolist()))}")

    # ── Compute global class weights (for DL weighted CE) ─────────────
    class_weights = compute_class_weights(dataset["labels"])
    print(f"  Class weights: non-stress={class_weights[0]:.3f}, stress={class_weights[1]:.3f}")

    # ── Step 2: Deep learning LOSO training ──────────────────────────────
    dl_results = {}
    if not args.classical_only:
        print(f"\n[2/5] DL LOSO training — models: {args.models}")
        folds_to_run = [args.fold] if args.fold else ALL_SUBJECTS

        for model_name in args.models:
            print(f"\n  ── {model_name.upper()} ────────────────────────────────────────")
            model_class = MODEL_REGISTRY[model_name]
            fold_results = []

            for fold_sub in folds_to_run:
                print(f"  Fold S{fold_sub} ({folds_to_run.index(fold_sub)+1}/{len(folds_to_run)}):")
                train_d, val_d, test_d = loso_split(dataset, fold_sub, seed=args.seed)
                train_dl, val_dl, test_dl = make_dataloaders(
                    train_d, val_d, test_d, batch_size=args.batch_size
                )
                config = {
                    "model": model_name,
                    "batch_size": args.batch_size,
                    "seed": args.seed,
                }
                result = train_one_fold(
                    model=model_class(),
                    train_loader=train_dl,
                    val_loader=val_dl,
                    test_loader=test_dl,
                    class_weights=class_weights,
                    model_name=model_name,
                    fold=fold_sub,
                    config=config,
                    seed=args.seed,
                )
                fold_results.append(result)
                m = result["metrics"]
                print(
                    f"    → F1={m['f1']:.4f}  AUC={m['auc_roc']:.4f}  "
                    f"Acc={m['accuracy']:.4f}  "
                    f"Lat={m.get('latency_mean_ms', 0):.1f}ms"
                )

            dl_results[model_name] = {
                "folds":   fold_results,
                "summary": aggregate_fold_results([f["metrics"] for f in fold_results]),
            }

    # ── Step 3: Classical baselines ──────────────────────────────────────
    classical_results = {}
    if not args.dl_only:
        print("\n[3/5] Classical baselines (RF / SVM / XGBoost)...")
        classical_results = run_all_classical(dataset, seed=args.seed)

    # ── Step 4: Reporting ────────────────────────────────────────────────
    print("\n[4/5] Results comparison:")
    if dl_results or classical_results:
        print_comparison_table(dl_results, classical_results)

    # Save full results
    all_results = {"dl": {}, "classical": {}}
    for k, v in dl_results.items():
        all_results["dl"][k] = {
            "summary": v["summary"],
            "folds": [
                {
                    "fold": f["fold"],
                    "metrics": f["metrics"],
                    "wall_time_sec": f["wall_time_sec"],
                }
                for f in v["folds"]
            ],
        }
    for k, v in classical_results.items():
        all_results["classical"][k] = {
            "summary": v["summary"],
            "folds": [
                {
                    "fold": f["fold"],
                    "metrics": f["metrics"],
                }
                for f in v["folds"]
            ],
        }
    results_path = os.path.join(OUTPUTS_DIR, "all_results.json")
    with open(results_path, "w") as fp:
        json.dump(all_results, fp, indent=2)
    print(f"\nFull results saved → {results_path}")

    # ── Step 5: ONNX export ──────────────────────────────────────────────
    if dl_results and not args.classical_only:
        print("\n[5/5] ONNX export...")
        best_model_name = recommend_best_model(dl_results, classical_results)
        if best_model_name:
            model_class = MODEL_REGISTRY[best_model_name]

            # Load best checkpoint (use first fold as representative)
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"{best_model_name}_fold{ALL_SUBJECTS[0]}_best.pt")
            if os.path.exists(ckpt_path):
                ckpt  = torch.load(ckpt_path, map_location="cpu")
                model = model_class()
                model.load_state_dict(ckpt["model_state"])
                model.eval()

                onnx_path = export_to_onnx(model, best_model_name, out_dir=OUTPUTS_DIR)
                val_result = validate_onnx(model, onnx_path)
                print(f"ONNX validation: {'PASS ✅' if val_result['passed'] else 'FAIL ❌'}")

                # Save ONNX validation result
                with open(os.path.join(OUTPUTS_DIR, "onnx_validation.json"), "w") as fp:
                    json.dump({
                        "model": best_model_name,
                        "onnx_path": onnx_path,
                        **val_result,
                    }, fp, indent=2)
            else:
                print(f"  ⚠️  Checkpoint not found for {best_model_name}, skipping ONNX export")

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
