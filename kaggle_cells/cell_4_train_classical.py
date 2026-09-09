# ═══════════════════════════════════════════════════════════════
# CELL 4: TRAIN CLASSICAL BASELINES (RF / SVM / XGBoost)
# ═══════════════════════════════════════════════════════════════
# ⏱️ ~20-40 minutes
#
# ⚠️  SECTION 10.6: HP budget is set to 30 trials/model (default).
#     Change HP_BUDGET below if you want more/fewer trials.

import os, sys, json
sys.path.insert(0, "/kaggle/working/repo")
os.chdir("/kaggle/working/repo")

from config import seed_everything, GLOBAL_SEED, OUTPUTS_DIR
seed_everything(GLOBAL_SEED)

from data.windowing import load_dataset
from training.classical import run_all_classical

# ── HP Budget (Section 10.6) ─────────────────────────────────
HP_BUDGET = 30   # ← Change if needed: 10=quick, 30=normal, 50=thorough
# ─────────────────────────────────────────────────────────────

dataset = load_dataset()
print(f"HP search budget: {HP_BUDGET} trials per model")
print("Training classical baselines...\n")

classical_results = run_all_classical(dataset, seed=GLOBAL_SEED)

# Save classical results
cl_out = {}
for k, v in classical_results.items():
    cl_out[k] = {
        "summary": v["summary"],
        "folds": [{"fold": f["fold"], "metrics": f["metrics"]} for f in v["folds"]],
    }
cl_path = os.path.join(OUTPUTS_DIR, "classical_results.json")
with open(cl_path, "w") as fp:
    json.dump(cl_out, fp, indent=2)
print(f"\nClassical results saved → {cl_path}")
print("\n✅ Classical baselines complete!")
