# ═══════════════════════════════════════════════════════════════
# CELL 5: COMPARISON TABLE + RESULTS
# ═══════════════════════════════════════════════════════════════
# ⚠️  SECTION 10.7: Review the table below and decide which model
#     to export to ONNX for hardware deployment (next cell).

import os, sys, json
import numpy as np
sys.path.insert(0, "/kaggle/working/repo")
os.chdir("/kaggle/working/repo")

from config import OUTPUTS_DIR

# Load results from both training cells
dl_path = os.path.join(OUTPUTS_DIR, "dl_results.json")
cl_path = os.path.join(OUTPUTS_DIR, "classical_results.json")

with open(dl_path) as f:
    dl_results = json.load(f)
with open(cl_path) as f:
    classical_results = json.load(f)

# Merge into all_results.json (needed by report generator)
all_results = {"dl": dl_results, "classical": classical_results}
results_path = os.path.join(OUTPUTS_DIR, "all_results.json")
with open(results_path, "w") as fp:
    json.dump(all_results, fp, indent=2)

# ── Print comparison table ────────────────────────────────────
def fmt(s, key):
    m   = s.get(f"{key}_mean", float("nan"))
    std = s.get(f"{key}_std",  float("nan"))
    if np.isnan(m):
        return "    N/A    "
    return f"{m:.4f}±{std:.4f}"

print("\n" + "="*100)
print("  MODEL COMPARISON — 15-fold LOSO (mean ± std)")
print("="*100)
print(f"{'Model':<16} {'Accuracy':>14} {'F1':>14} {'Sensitivity':>14} {'Specificity':>14} {'AUC-ROC':>14} {'Lat(ms)':>9} {'<50ms':>6}")
print("─"*100)

for model_name, result in dl_results.items():
    s = result["summary"]
    lats = [f["metrics"].get("latency_mean_ms", float("nan")) for f in result["folds"]]
    lat = float(np.nanmean(lats))
    ok = "✅" if lat < 50 else "❌"
    print(f"{model_name:<16} {fmt(s,'accuracy'):>14} {fmt(s,'f1'):>14} {fmt(s,'sensitivity'):>14} "
          f"{fmt(s,'specificity'):>14} {fmt(s,'auc_roc'):>14} {lat:>8.1f} {ok:>6}")

print("─"*100)
for clf_name, result in classical_results.items():
    s = result["summary"]
    print(f"{clf_name:<16} {fmt(s,'accuracy'):>14} {fmt(s,'f1'):>14} {fmt(s,'sensitivity'):>14} "
          f"{fmt(s,'specificity'):>14} {fmt(s,'auc_roc'):>14} {'N/A':>9} {'--':>6}")
print("="*100)

# ── Find best model ──────────────────────────────────────────
best_name = None
best_auc = -1
best_clf_f1 = max(
    result["summary"].get("f1_mean", 0)
    for result in classical_results.values()
)

print(f"\nBest classical F1: {best_clf_f1:.4f}")
print("\nDL model analysis:")
for model_name, result in dl_results.items():
    s = result["summary"]
    f1 = s.get("f1_mean", 0)
    auc = s.get("auc_roc_mean", 0)
    lats = [f["metrics"].get("latency_mean_ms", float("nan")) for f in result["folds"]]
    lat = float(np.nanmean(lats))
    beats = "✅ beats" if f1 > best_clf_f1 else "❌ below"
    lat_ok = "✅" if lat < 50 else "❌"
    star = " ⭐" if f1 > best_clf_f1 and lat < 50 and auc > best_auc else ""
    if f1 > best_clf_f1 and lat < 50 and auc > best_auc:
        best_auc = auc
        best_name = model_name
    print(f"  {model_name:<16} F1={f1:.4f} {beats} classical  |  Lat={lat:.1f}ms {lat_ok}  |  AUC={auc:.4f}{star}")

print(f"\n{'='*60}")
if best_name:
    print(f"  🏆 RECOMMENDED MODEL: {best_name}")
else:
    print("  ⚠️  No DL model qualifies (beats baseline + <50ms latency)")
    print("  Pick the one with best AUC anyway:")
    best_name = max(dl_results, key=lambda k: dl_results[k]["summary"].get("auc_roc_mean", 0))
    print(f"  → Fallback: {best_name}")
print(f"{'='*60}")
print(f"\n  ✏️  In the NEXT CELL, set EXPORT_MODEL = \"{best_name}\"")
print(f"      (or change it to whichever model YOU prefer)")
