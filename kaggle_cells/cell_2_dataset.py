# ═══════════════════════════════════════════════════════════════
# CELL 2: BUILD DATASET + DIAGNOSTIC PLOT
# ═══════════════════════════════════════════════════════════════
# ⏱️ ~5-15 minutes
#
# ⚠️  HUMAN CHECKPOINT 10.4: After this cell runs, LOOK AT THE PLOT.
#     Check for flat-lined channels or obvious artifacts before continuing.

import os, sys
sys.path.insert(0, "/kaggle/working/repo")
os.chdir("/kaggle/working/repo")

from config import seed_everything, GLOBAL_SEED, PROCESSED_DATA_DIR, OUTPUTS_DIR, ALL_SUBJECTS
seed_everything(GLOBAL_SEED)

# Build or load preprocessed dataset
from data.windowing import build_dataset, save_dataset, load_dataset
import numpy as np

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

# Per-subject summary
print(f"\n{'Subject':>8} {'Windows':>8} {'Stress':>8} {'Non-Str':>8} {'Stress%':>8}")
print("─" * 44)
for sid in ALL_SUBJECTS:
    mask = dataset["subject_ids"] == sid
    total = mask.sum()
    stress = int((dataset["labels"][mask] == 1).sum())
    non_st = int((dataset["labels"][mask] == 0).sum())
    pct = f"{stress/total:.1%}" if total > 0 else "N/A"
    print(f"  S{sid:<5} {total:>8} {stress:>8} {non_st:>8} {pct:>8}")

# ── Diagnostic plot (Section 10.4) ──────────────────────────────────
from data.loader import load_subject
from data.preprocessing import preprocess, plot_raw_vs_filtered
import matplotlib.pyplot as plt

sid = 2
raw_s, _ = load_subject(sid)
filt_s   = preprocess(raw_s)
plot_path = plot_raw_vs_filtered(raw_s, filt_s, sid, OUTPUTS_DIR)

img = plt.imread(plot_path)
plt.figure(figsize=(16, 8))
plt.imshow(img)
plt.axis("off")
plt.title(f"Diagnostic: Raw vs Filtered — S{sid}", fontsize=14)
plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("  ⚠️  SECTION 10.4: REVIEW THE PLOT ABOVE")
print("  ✅ ECG should show clear R-peaks (orange trace)")
print("  ✅ PPG should be a smooth waveform")
print("  ✅ Temp should be slow-varying")
print("  🚩 STOP if any channel is flat or all-NaN")
print("="*60)
print("\n  If everything looks good → run the next cell to start training")
