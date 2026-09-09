# ═══════════════════════════════════════════════════════════════
# CELL 3: TRAIN ALL 4 DEEP LEARNING MODELS (LOSO)
# ═══════════════════════════════════════════════════════════════
# ⏱️ ~2-5 hours on GPU T4 (all 4 models × 15 folds = 60 training runs)
#
# 💡 TIP: Set TEST_ONLY = True below to do a quick 1-fold smoke test first (~5 min)

import os, sys, json, time
import numpy as np
import torch

sys.path.insert(0, "/kaggle/working/repo")
os.chdir("/kaggle/working/repo")

from config import (
    seed_everything, GLOBAL_SEED, ALL_SUBJECTS, BATCH_SIZE,
    CHECKPOINT_DIR, OUTPUTS_DIR, PROCESSED_DATA_DIR,
)
seed_everything(GLOBAL_SEED)

from data.windowing import load_dataset
from training.dataset import loso_split, compute_class_weights, make_dataloaders
from training.trainer import train_one_fold
from training.metrics import aggregate_fold_results
import models as model_module

# ── Settings ─────────────────────────────────────────────────
TEST_ONLY = False    # ← Set True for quick test (1 model, 1 fold, ~5 min)
# ─────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "cnn":         model_module.CNN1D,
    "bilstm":      model_module.BiLSTM,
    "transformer": model_module.TemporalTransformer,
    "stgcn":       model_module.STGCN,
}

dataset = load_dataset()
class_weights = compute_class_weights(dataset["labels"])
print(f"Class weights: non-stress={class_weights[0]:.3f}, stress={class_weights[1]:.3f}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

folds_to_run  = [ALL_SUBJECTS[0]] if TEST_ONLY else ALL_SUBJECTS
models_to_run = ["cnn"] if TEST_ONLY else list(MODEL_REGISTRY.keys())

dl_results = {}
total_start = time.time()

for model_name in models_to_run:
    print(f"\n{'='*60}")
    print(f"  Training: {model_name.upper()}")
    print(f"{'='*60}")
    model_class = MODEL_REGISTRY[model_name]
    fold_results = []

    for fold_sub in folds_to_run:
        print(f"\n  Fold S{fold_sub} ({folds_to_run.index(fold_sub)+1}/{len(folds_to_run)}):")
        train_d, val_d, test_d = loso_split(dataset, fold_sub, seed=GLOBAL_SEED)
        train_dl, val_dl, test_dl = make_dataloaders(
            train_d, val_d, test_d, batch_size=BATCH_SIZE
        )
        config = {"model": model_name, "batch_size": BATCH_SIZE, "seed": GLOBAL_SEED}
        result = train_one_fold(
            model=model_class(),
            train_loader=train_dl,
            val_loader=val_dl,
            test_loader=test_dl,
            class_weights=class_weights,
            model_name=model_name,
            fold=fold_sub,
            config=config,
            seed=GLOBAL_SEED,
        )
        fold_results.append(result)
        m = result["metrics"]
        print(f"    → F1={m['f1']:.4f}  AUC={m['auc_roc']:.4f}  Acc={m['accuracy']:.4f}  Lat={m.get('latency_mean_ms',0):.1f}ms")

    dl_results[model_name] = {
        "folds":   fold_results,
        "summary": aggregate_fold_results([f["metrics"] for f in fold_results]),
    }
    s = dl_results[model_name]["summary"]
    print(f"\n  {model_name} LOSO mean: F1={s.get('f1_mean',0):.4f}  AUC={s.get('auc_roc_mean',0):.4f}")

total_time = time.time() - total_start
print(f"\n{'='*60}")
print(f"  DL TRAINING COMPLETE — {total_time/3600:.1f} hours")
print(f"{'='*60}")

# Save DL results
dl_out = {}
for k, v in dl_results.items():
    dl_out[k] = {
        "summary": v["summary"],
        "folds": [{"fold": f["fold"], "metrics": f["metrics"], "wall_time_sec": f["wall_time_sec"]} for f in v["folds"]],
    }
dl_path = os.path.join(OUTPUTS_DIR, "dl_results.json")
with open(dl_path, "w") as fp:
    json.dump(dl_out, fp, indent=2)
print(f"DL results saved → {dl_path}")
