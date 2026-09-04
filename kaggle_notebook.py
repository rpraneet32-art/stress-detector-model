# ============================================================
# STRESS DETECTOR — FULL TRAINING PIPELINE (Kaggle Notebook)
# ============================================================
# Dataset: WESAD (add via Kaggle → + Add Data → search "WESAD")
# GPU: Enable via Settings → Accelerator → GPU T4 x2
#
# This script runs the complete pipeline end-to-end:
#   1.  Install dependencies
#   2.  Clone / mount this repo's code
#   3.  Build preprocessed dataset from raw WESAD pkl files
#   4.  Train 4 DL models + 3 classical baselines (LOSO CV)
#   5.  Generate comparison report
#   6.  Export best model to ONNX
# ============================================================

# ── CELL 1: Install dependencies ─────────────────────────────────────────────
# Run this cell first and restart kernel if prompted

import subprocess, sys

packages = [
    "torch==2.2.2",
    "torchvision==0.17.2",
    "scikit-learn==1.4.2",
    "xgboost==2.0.3",
    "scipy==1.13.0",
    "PyWavelets==1.6.0",
    "numpy==1.26.4",
    "pandas==2.2.2",
    "onnx==1.16.1",
    "onnxruntime==1.18.1",
    "matplotlib==3.9.0",
    "tqdm==4.66.4",
    "joblib",
]

subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + packages, check=True)
print("✅ Dependencies installed")

# ── CELL 2: Verify GPU ───────────────────────────────────────────────────────

import torch
print(f"PyTorch version : {torch.__version__}")
print(f"CUDA available  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU             : {torch.cuda.get_device_name(0)}")
    print(f"VRAM            : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ── CELL 3: Mount repo code ───────────────────────────────────────────────────
# Option A: Code is committed to GitHub — clone it
# Option B: Code is uploaded as a Kaggle dataset — copy it
# Option C: Paste code inline (fallback)
#
# ✏️ EDIT THIS: set your GitHub repo URL, or use Option B if using a dataset

import os

REPO_URL = "https://github.com/YOUR_USERNAME/stress-detector-model.git"  # ← edit this

if os.path.exists("/kaggle/input/stress-detector-code"):
    # Option B: uploaded as Kaggle dataset
    import shutil
    shutil.copytree("/kaggle/input/stress-detector-code", "/kaggle/working/repo", dirs_exist_ok=True)
    print("✅ Code mounted from Kaggle dataset")
elif REPO_URL != "https://github.com/YOUR_USERNAME/stress-detector-model.git":
    # Option A: clone from GitHub
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, "/kaggle/working/repo"], check=True)
    print(f"✅ Code cloned from {REPO_URL}")
else:
    print("⚠️  Edit REPO_URL above, or upload code as a Kaggle dataset named 'stress-detector-code'")
    print("   Alternatively, the inline fallback below will write the code directly.")

REPO_ROOT = "/kaggle/working/repo"
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

# ── CELL 4: Verify WESAD dataset path ────────────────────────────────────────

import glob

# Try common Kaggle WESAD dataset slugs
WESAD_CANDIDATES = [
    "/kaggle/input/wesad",
    "/kaggle/input/wesad-dataset",
    "/kaggle/input/wearable-stress-and-affect-detection",
    "/kaggle/input/wesad-wearable-stress-and-affect-detection",
]

WESAD_ROOT = None
for candidate in WESAD_CANDIDATES:
    if os.path.exists(candidate):
        WESAD_ROOT = candidate
        break

if WESAD_ROOT:
    subjects_found = sorted(glob.glob(os.path.join(WESAD_ROOT, "S*", "S*.pkl")))
    print(f"✅ WESAD found at: {WESAD_ROOT}")
    print(f"   Subject pkl files: {len(subjects_found)}")
    for p in subjects_found:
        print(f"     {p}")
else:
    print("❌ WESAD dataset not found in /kaggle/input/")
    print("   → Go to your notebook → Add Data → search 'WESAD'")
    print("   → Or upload manually as a dataset")

# Update config to point to correct path
# (config.py auto-detects Kaggle and tries these candidates,
#  but if a non-standard slug is used, set it manually here)
os.environ["WESAD_ROOT_OVERRIDE"] = WESAD_ROOT or ""

# ── CELL 5: Seed everything & validate imports ────────────────────────────────

from config import seed_everything, GLOBAL_SEED, ALL_SUBJECTS, subject_pkl_path
seed_everything(GLOBAL_SEED)

print(f"Seed: {GLOBAL_SEED}")
print(f"Expected subjects: {ALL_SUBJECTS}")

missing = [s for s in ALL_SUBJECTS if not os.path.exists(subject_pkl_path(s))]
if missing:
    print(f"⚠️  Missing subject pkls: {missing}")
else:
    print(f"✅ All {len(ALL_SUBJECTS)} subject pkl files accessible")

# ── CELL 6: Quick model unit tests ───────────────────────────────────────────

from models import CNN1D, BiLSTM, TemporalTransformer, STGCN
from config import WINDOW_SAMPLES

dummy = torch.randn(4, 3, WINDOW_SAMPLES)
for name, cls in [("CNN1D", CNN1D), ("BiLSTM", BiLSTM),
                   ("Transformer", TemporalTransformer), ("STGCN", STGCN)]:
    m = cls()
    out = m(dummy)
    assert out.shape == (4, 2), f"{name}: expected (4,2), got {out.shape}"
    n = sum(p.numel() for p in m.parameters())
    print(f"✅ {name:<16}  output={tuple(out.shape)}  params={n:,}")

# ── CELL 7: Build preprocessed dataset ───────────────────────────────────────
# ⏱️ Expected time: ~5–15 minutes on Kaggle CPU

from data.windowing import build_dataset, save_dataset, load_dataset
from config import PROCESSED_DATA_DIR

processed_path = os.path.join(PROCESSED_DATA_DIR, "wesad_windows.npz")

if os.path.exists(processed_path):
    print("Loading cached preprocessed dataset...")
    dataset = load_dataset()
else:
    print("Building dataset from raw WESAD files (this takes ~5-15 min)...")
    dataset = build_dataset()
    save_dataset(dataset)

print(f"\n✅ Dataset ready:")
print(f"   Tensors  : {dataset['tensors'].shape}")
print(f"   Features : {dataset['features'].shape}")
print(f"   Labels   : {dataset['labels'].shape}  (stress={float((dataset['labels']==1).mean()):.1%})")
print(f"   Subjects : {sorted(set(dataset['subject_ids'].tolist()))}")

# ── CELL 8: Generate diagnostic plots ────────────────────────────────────────
# Run preprocessing on S2 and plot raw vs filtered (Section 10.4 review artifact)

from data.loader import load_subject
from data.preprocessing import preprocess, plot_raw_vs_filtered
from config import OUTPUTS_DIR
import matplotlib.pyplot as plt

sid = 2
raw_s, _ = load_subject(sid)
filt_s   = preprocess(raw_s)
plot_path = plot_raw_vs_filtered(raw_s, filt_s, sid, OUTPUTS_DIR)
img = plt.imread(plot_path)
plt.figure(figsize=(14, 6))
plt.imshow(img)
plt.axis("off")
plt.title(f"Diagnostic: Raw vs Filtered — S{sid}")
plt.tight_layout()
plt.show()
print(f"\n⚠️  SECTION 10.4 ACTION REQUIRED: Review plot above before continuing.")
print("   Check for: flat-lined channels, obvious artifacts, mislabeled subjects.")
print("   Type 'proceed' in the next cell when satisfied.\n")

# ── CELL 9: [HUMAN CHECKPOINT] Review plots before full training ─────────────
# 📋 ACTION: Look at the plot above. If it looks reasonable, run the next cell.
# ⚠️  If you see a flat-lined channel or obvious artifact on a subject,
#     stop and investigate before running full LOSO training.
print("When you are satisfied with the diagnostic plots, run the training cell below.")

# ── CELL 10: Train DL models — full LOSO ─────────────────────────────────────
# ⏱️ Expected time: ~2–6 hours on Kaggle GPU T4 (all 4 models × 15 folds)
# 💡 To test one model + one fold first: set TEST_ONLY = True

TEST_ONLY  = False   # ← set True to do a quick smoke-test (1 model, 1 fold)
DL_MODELS  = ["cnn", "bilstm", "transformer", "stgcn"]

import json
import numpy as np
from training.dataset import loso_split, compute_class_weights, make_dataloaders
from training.trainer import train_one_fold
from training.metrics import aggregate_fold_results
import models as model_module

MODEL_REGISTRY = {
    "cnn":         model_module.CNN1D,
    "bilstm":      model_module.BiLSTM,
    "transformer": model_module.TemporalTransformer,
    "stgcn":       model_module.STGCN,
}

from config import ALL_SUBJECTS, BATCH_SIZE, CHECKPOINT_DIR
seed_everything(GLOBAL_SEED)

class_weights = compute_class_weights(dataset["labels"])
folds_to_run  = [ALL_SUBJECTS[0]] if TEST_ONLY else ALL_SUBJECTS
models_to_run = ["cnn"] if TEST_ONLY else DL_MODELS

dl_results = {}

for model_name in models_to_run:
    print(f"\n{'='*60}")
    print(f"  Training: {model_name.upper()}")
    print(f"{'='*60}")
    model_class = MODEL_REGISTRY[model_name]
    fold_results = []

    for fold_sub in folds_to_run:
        print(f"\nFold S{fold_sub} ({folds_to_run.index(fold_sub)+1}/{len(folds_to_run)}):")
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
        print(f"  → F1={m['f1']:.4f}  AUC={m['auc_roc']:.4f}  Acc={m['accuracy']:.4f}  Lat={m.get('latency_mean_ms',0):.1f}ms")

    dl_results[model_name] = {
        "folds":   fold_results,
        "summary": aggregate_fold_results([f["metrics"] for f in fold_results]),
    }
    print(f"\n  {model_name} summary: {dl_results[model_name]['summary']}")

# ── CELL 11: Train classical baselines ───────────────────────────────────────
# ⏱️ Expected time: ~20–40 minutes

from training.classical import run_all_classical
seed_everything(GLOBAL_SEED)

print("Training classical baselines (RF / SVM / XGBoost) ...")
classical_results = run_all_classical(dataset, seed=GLOBAL_SEED)

# ── CELL 12: Consolidated comparison table ────────────────────────────────────

def fmt(s, key):
    m   = s.get(f"{key}_mean", float("nan"))
    std = s.get(f"{key}_std",  float("nan"))
    return f"{m:.4f}±{std:.4f}"

header = f"\n{'Model':<16} {'Acc':>14} {'F1':>14} {'Sens':>14} {'Spec':>14} {'AUC':>14} {'Lat(ms)':>9}"
print(header)
print("─" * len(header))

all_summaries = {}
for model_name, result in dl_results.items():
    s = result["summary"]
    all_summaries[model_name] = s
    lats = [f["metrics"].get("latency_mean_ms", float("nan")) for f in result["folds"]]
    lat  = float(np.nanmean(lats))
    ok   = "✅" if lat < 50 else "❌"
    print(f"{model_name:<16} {fmt(s,'accuracy'):>14} {fmt(s,'f1'):>14} {fmt(s,'sensitivity'):>14} "
          f"{fmt(s,'specificity'):>14} {fmt(s,'auc_roc'):>14} {lat:>8.1f}{ok}")

for clf, result in classical_results.items():
    s = result["summary"]
    all_summaries[clf] = s
    print(f"{clf:<16} {fmt(s,'accuracy'):>14} {fmt(s,'f1'):>14} {fmt(s,'sensitivity'):>14} "
          f"{fmt(s,'specificity'):>14} {fmt(s,'auc_roc'):>14} {'N/A':>9}")

# Save full results
all_results_out = {
    "dl":        {k: {"summary": v["summary"],
                      "folds": [{"fold": f["fold"], "metrics": f["metrics"]} for f in v["folds"]]}
                  for k, v in dl_results.items()},
    "classical": {k: {"summary": v["summary"],
                      "folds": [{"fold": f["fold"], "metrics": f["metrics"]} for f in v["folds"]]}
                  for k, v in classical_results.items()},
}
results_path = os.path.join(OUTPUTS_DIR, "all_results.json")
with open(results_path, "w") as fp:
    json.dump(all_results_out, fp, indent=2)
print(f"\nResults saved → {results_path}")

# ── CELL 13: ONNX export ──────────────────────────────────────────────────────

from export.onnx_export import export_to_onnx, validate_onnx

# ✏️ EDIT: Set this to the model you want to export (after Section 10.7 review)
EXPORT_MODEL_NAME = "cnn"   # ← change after reviewing the table above

model_class = MODEL_REGISTRY[EXPORT_MODEL_NAME]
ckpt_path   = os.path.join(CHECKPOINT_DIR, f"{EXPORT_MODEL_NAME}_fold{ALL_SUBJECTS[0]}_best.pt")

ckpt  = torch.load(ckpt_path, map_location="cpu")
model = model_class()
model.load_state_dict(ckpt["model_state"])
model.eval()

onnx_path  = export_to_onnx(model, EXPORT_MODEL_NAME, out_dir=OUTPUTS_DIR)
val_result = validate_onnx(model, onnx_path)
print(f"\nONNX export: {'✅ PASS' if val_result['passed'] else '❌ FAIL'}")

# ── CELL 14: Generate final report ────────────────────────────────────────────

from reporting.report import generate_report
report_path = generate_report(results_path)
print(f"\nFull report → {report_path}")

# List all outputs
print("\n── Output files ─────────────────────────────────────────────")
for root, dirs, files in os.walk(OUTPUTS_DIR):
    for f in files:
        full = os.path.join(root, f)
        size = os.path.getsize(full)
        print(f"  {full}  ({size:,} bytes)")

print("\n✅ Pipeline complete. Download /kaggle/working/outputs/ for all artifacts.")
