"""
config.py — Central configuration for the stress detection pipeline.
Auto-detects environment (Kaggle / Local) and sets all path & hyperparameter
constants. Import this module at the top of every script.
"""

import os
import random
import numpy as np

# ---------------------------------------------------------------------------
# 1. Environment detection
# ---------------------------------------------------------------------------

def _detect_env() -> str:
    """Return 'kaggle' or 'local'."""
    if os.path.exists("/kaggle/input"):
        return "kaggle"
    return "local"

ENV = _detect_env()

# ---------------------------------------------------------------------------
# 2. Paths — resolved per environment
# ---------------------------------------------------------------------------

if ENV == "kaggle":
    # Kaggle mounts dataset at /kaggle/input/<dataset-slug>/
    # Common WESAD slugs on Kaggle: "wesad-dataset", "wearable-stress-and-affect-detection"
    # We try both and fall back gracefully.
    _KAGGLE_CANDIDATES = [
        "/kaggle/input/wesad",
        "/kaggle/input/wesad-dataset",
        "/kaggle/input/wearable-stress-and-affect-detection",
        "/kaggle/input/wesad-wearable-stress-and-affect-detection",
    ]
    RAW_DATA_ROOT = next(
        (p for p in _KAGGLE_CANDIDATES if os.path.exists(p)),
        "/kaggle/input/wesad",          # default (will error if wrong — intentional)
    )
    PROCESSED_DATA_DIR = "/kaggle/working/processed"
    CHECKPOINT_DIR     = "/kaggle/working/checkpoints"
    LOGS_DIR           = "/kaggle/working/logs"
    OUTPUTS_DIR        = "/kaggle/working/outputs"

else:  # local
    _REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
    _LOCAL_CANDIDATES = [
        r"D:\WESAD\WESAD",
        r"D:\WESAD",
        os.path.join(_REPO_ROOT, "data", "raw", "WESAD"),
    ]
    RAW_DATA_ROOT = next(
        (p for p in _LOCAL_CANDIDATES if os.path.exists(p) and any(os.path.exists(os.path.join(p, f"S{s}")) for s in [2, 3])),
        os.path.join(_REPO_ROOT, "data", "raw", "WESAD"),
    )
    PROCESSED_DATA_DIR = os.path.join(_REPO_ROOT, "data", "processed")
    CHECKPOINT_DIR     = os.path.join(_REPO_ROOT, "checkpoints")
    LOGS_DIR           = os.path.join(_REPO_ROOT, "logs")
    OUTPUTS_DIR        = os.path.join(_REPO_ROOT, "outputs")

# Create output dirs if they don't exist
for _d in [PROCESSED_DATA_DIR, CHECKPOINT_DIR, LOGS_DIR, OUTPUTS_DIR]:
    os.makedirs(_d, exist_ok=True)

# ---------------------------------------------------------------------------
# 3. WESAD subjects
# ---------------------------------------------------------------------------

# WESAD does not include S1 or S12 (excluded by dataset authors)
ALL_SUBJECTS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]

def subject_pkl_path(subject_id: int) -> str:
    """Return path to S{n}/S{n}.pkl for a given subject ID."""
    return os.path.join(RAW_DATA_ROOT, f"S{subject_id}", f"S{subject_id}.pkl")

# ---------------------------------------------------------------------------
# 4. Signal / windowing parameters
# ---------------------------------------------------------------------------

TARGET_FS       = 128           # Hz — common resampled rate for all channels
WINDOW_SEC      = 3.0           # seconds
WINDOW_SAMPLES  = int(TARGET_FS * WINDOW_SEC)   # 384
OVERLAP_FRAC    = 0.5
STEP_SAMPLES    = int(WINDOW_SAMPLES * (1 - OVERLAP_FRAC))  # 192

# Label mapping
LABEL_NON_STRESS = 0            # WESAD conditions 1 (baseline) + 3 (amusement)
LABEL_STRESS     = 1            # WESAD condition  2 (TSST)
WESAD_NON_STRESS_CONDITIONS = {1, 3}
WESAD_STRESS_CONDITIONS     = {2}

# Channel indices in the (T, 3) array
CH_ECG  = 0
CH_PPG  = 1
CH_TEMP = 2

# ---------------------------------------------------------------------------
# 5. Preprocessing parameters
# ---------------------------------------------------------------------------

NOTCH_FREQ  = 50.0      # Hz — powerline
NOTCH_Q     = 30.0

ECG_BP_LOW  = 0.5       # Hz
ECG_BP_HIGH = 40.0      # Hz
ECG_BP_ORDER = 4

PPG_BP_LOW  = 0.5       # Hz
PPG_BP_HIGH = 5.0       # Hz
PPG_BP_ORDER = 3

# ---------------------------------------------------------------------------
# 6. Training hyperparameters
# ---------------------------------------------------------------------------

LEARNING_RATE  = 1e-3
BETAS          = (0.9, 0.999)
WEIGHT_DECAY   = 1e-2
EARLY_STOP_PATIENCE = 12       # epochs
MAX_EPOCHS     = 100
BATCH_SIZE     = 64
VAL_SPLIT      = 0.2           # fraction of training subjects used for validation within each LOSO fold

# Hyperparameter search budget (Section 10.6 — default, awaiting human confirmation)
HP_SEARCH_TRIALS_CLASSICAL = 50   # per model (RF, SVM, XGBoost)

# ---------------------------------------------------------------------------
# 7. Reproducibility
# ---------------------------------------------------------------------------

GLOBAL_SEED = 42

def seed_everything(seed: int = GLOBAL_SEED) -> None:
    """Fix random seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

# ---------------------------------------------------------------------------
# 8. Logging
# ---------------------------------------------------------------------------

RUN_LOG_PATH = os.path.join(LOGS_DIR, "run_log.jsonl")   # structured per-run log

# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Environment  : {ENV}")
    print(f"RAW_DATA_ROOT: {RAW_DATA_ROOT}")
    print(f"PROCESSED_DIR: {PROCESSED_DATA_DIR}")
    print(f"CHECKPOINT_DIR: {CHECKPOINT_DIR}")
    print(f"Subjects     : {ALL_SUBJECTS}")
    print(f"Window       : {WINDOW_SAMPLES} samples @ {TARGET_FS} Hz = {WINDOW_SEC}s")
    print(f"Step         : {STEP_SAMPLES} samples")
    missing = [s for s in ALL_SUBJECTS if not os.path.exists(subject_pkl_path(s))]
    if missing:
        print(f"WARNING — Missing subject pkl files: {missing}")
    else:
        print("All 15 subject pkl files found.")
