# ═══════════════════════════════════════════════════════════════
# CELL 6: ONNX EXPORT + FINAL REPORT
# ═══════════════════════════════════════════════════════════════
# ✏️  EDIT THE LINE BELOW — set your chosen model from the table above

EXPORT_MODEL = "cnn"   # ← CHANGE THIS to: "cnn", "bilstm", "transformer", or "stgcn"

# ═══════════════════════════════════════════════════════════════

import os, sys, json
import torch
sys.path.insert(0, "/kaggle/working/repo")
os.chdir("/kaggle/working/repo")

from config import seed_everything, GLOBAL_SEED, ALL_SUBJECTS, CHECKPOINT_DIR, OUTPUTS_DIR
seed_everything(GLOBAL_SEED)

import models as model_module
from export.onnx_export import export_to_onnx, validate_onnx

MODEL_REGISTRY = {
    "cnn":         model_module.CNN1D,
    "bilstm":      model_module.BiLSTM,
    "transformer": model_module.TemporalTransformer,
    "stgcn":       model_module.STGCN,
}

# 1. Load best checkpoint
model_class = MODEL_REGISTRY[EXPORT_MODEL]
ckpt_path = os.path.join(CHECKPOINT_DIR, f"{EXPORT_MODEL}_fold{ALL_SUBJECTS[0]}_best.pt")

print(f"Exporting model: {EXPORT_MODEL}")
print(f"Checkpoint: {ckpt_path}")

ckpt = torch.load(ckpt_path, map_location="cpu")
model = model_class()
model.load_state_dict(ckpt["model_state"])
model.eval()
print(f"Loaded from epoch {ckpt.get('epoch', '?')} | val_loss={ckpt.get('val_loss', '?'):.4f}")

# 2. Export to ONNX
onnx_path = export_to_onnx(model, EXPORT_MODEL, out_dir=OUTPUTS_DIR)

# 3. Validate ONNX
val_result = validate_onnx(model, onnx_path)
with open(os.path.join(OUTPUTS_DIR, "onnx_validation.json"), "w") as fp:
    json.dump({"model": EXPORT_MODEL, "onnx_path": onnx_path, **val_result}, fp, indent=2)

# 4. Generate final report
from reporting.report import generate_report
report_path = generate_report()

# 5. List all output files
print("\n" + "="*60)
print("  📦 ALL OUTPUT FILES")
print("="*60)
total_size = 0
for root, dirs, files in os.walk(OUTPUTS_DIR):
    for f in sorted(files):
        full = os.path.join(root, f)
        size = os.path.getsize(full)
        total_size += size
        print(f"  {f:<40} {size:>10,} bytes")

for root, dirs, files in os.walk(CHECKPOINT_DIR):
    for f in sorted(files):
        full = os.path.join(root, f)
        size = os.path.getsize(full)
        total_size += size
        print(f"  {f:<40} {size:>10,} bytes")

print(f"{'─'*60}")
print(f"  Total: {total_size/1e6:.1f} MB")

print(f"\n{'='*60}")
print(f"  ✅ PIPELINE COMPLETE!")
print(f"  📊 Report: {report_path}")
print(f"  🤖 ONNX:   {onnx_path}")
print(f"  📥 Download from /kaggle/working/outputs/")
print(f"{'='*60}")
