# ═══════════════════════════════════════════════════════════════
# CELL 1: SETUP (OFFLINE — No internet needed)
# ═══════════════════════════════════════════════════════════════

import os, sys, glob, shutil

# 1. Verify pre-installed packages
print("Checking pre-installed packages...")
import torch, sklearn, scipy, numpy as np, xgboost
print(f"  ✅ torch     {torch.__version__}")
print(f"  ✅ sklearn   {sklearn.__version__}")
print(f"  ✅ scipy     {scipy.__version__}")
print(f"  ✅ numpy     {np.__version__}")
print(f"  ✅ xgboost   {xgboost.__version__}")
try:
    import onnx
    print(f"  ✅ onnx      {onnx.__version__}")
except ImportError:
    print("  ⚠️  onnx not found")
try:
    import onnxruntime
    print(f"  ✅ onnxrt    {onnxruntime.__version__}")
except ImportError:
    print("  ⚠️  onnxruntime not found — will handle at export time")

# 2. GPU check
print(f"\nCUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️  No GPU! Go to Settings → Accelerator → GPU T4 x2")

# 3. Find all datasets — print everything so we can see paths
print("\n── Everything in /kaggle/input/ ──")
for root, dirs, files in os.walk("/kaggle/input/"):
    level = root.replace("/kaggle/input/", "").count(os.sep)
    if level < 3:  # Don't go too deep
        indent = "  " * level
        print(f"  {indent}📁 {root}")
        if level < 2:
            for f in files[:10]:
                print(f"  {indent}  📄 {f}")
            if len(files) > 10:
                print(f"  {indent}  ... and {len(files)-10} more files")

# 4. Find the code — search for config.py anywhere in /kaggle/input/
print("\n── Searching for config.py ──")
config_files = glob.glob("/kaggle/input/**/config.py", recursive=True)
if config_files:
    print(f"  Found: {config_files}")
    CODE_SRC = os.path.dirname(config_files[0])
    print(f"  Code source: {CODE_SRC}")
else:
    print("  ❌ config.py not found anywhere in /kaggle/input/")
    print("     Make sure you uploaded config.py, data/, models/, training/, export/, reporting/")
    CODE_SRC = None

# 5. Copy code to working directory
REPO_DIR = "/kaggle/working/repo"
if CODE_SRC:
    if os.path.exists(REPO_DIR):
        shutil.rmtree(REPO_DIR)
    shutil.copytree(CODE_SRC, REPO_DIR)
    print(f"\n✅ Code copied: {CODE_SRC} → {REPO_DIR}")
else:
    print("\n❌ Cannot proceed without code. Upload it as a dataset first.")

sys.path.insert(0, REPO_DIR)
os.chdir(REPO_DIR)

# 6. Show what we have
print(f"\nRepo at {REPO_DIR}:")
for root, dirs, files in os.walk(REPO_DIR):
    level = root.replace(REPO_DIR, '').count(os.sep)
    indent = '  ' * level
    print(f"  {indent}{os.path.basename(root)}/")
    if level < 2:
        for f in sorted(files):
            print(f"  {indent}  {f}")

# 7. Find WESAD and fix paths
print("\n── Searching for WESAD pkl files ──")
pkl_files = glob.glob("/kaggle/input/**/S*.pkl", recursive=True)
if pkl_files:
    print(f"  Found {len(pkl_files)} pkl files")
    for p in sorted(pkl_files)[:3]:
        print(f"    {p}")
    if len(pkl_files) > 3:
        print(f"    ... and {len(pkl_files)-3} more")
else:
    print("  ❌ No S*.pkl files found!")

from config import seed_everything, ALL_SUBJECTS, WINDOW_SAMPLES, GLOBAL_SEED, RAW_DATA_ROOT, subject_pkl_path
seed_everything(GLOBAL_SEED)

missing = [s for s in ALL_SUBJECTS if not os.path.exists(subject_pkl_path(s))]
if missing and pkl_files:
    # Auto-fix: find where S2.pkl actually lives and symlink
    s2_candidates = [p for p in pkl_files if "S2.pkl" in p]
    if s2_candidates:
        actual_wesad_dir = os.path.dirname(os.path.dirname(s2_candidates[0]))
        print(f"\n  Auto-fixing paths:")
        print(f"    Expected: {RAW_DATA_ROOT}")
        print(f"    Actual:   {actual_wesad_dir}")
        os.makedirs(os.path.dirname(RAW_DATA_ROOT), exist_ok=True)
        if os.path.exists(RAW_DATA_ROOT):
            if os.path.islink(RAW_DATA_ROOT):
                os.unlink(RAW_DATA_ROOT)
            elif os.path.isdir(RAW_DATA_ROOT):
                shutil.rmtree(RAW_DATA_ROOT)
        os.symlink(actual_wesad_dir, RAW_DATA_ROOT)
        print(f"    ✅ Symlink created!")

    missing = [s for s in ALL_SUBJECTS if not os.path.exists(subject_pkl_path(s))]

if not missing:
    print(f"\n✅ All {len(ALL_SUBJECTS)} WESAD subjects accessible!")
else:
    print(f"\n❌ Still missing: {missing}")

# 8. Model unit tests
from models import CNN1D, BiLSTM, TemporalTransformer, STGCN
dummy = torch.randn(4, 3, WINDOW_SAMPLES)
print("\nModel unit tests:")
for name, cls in [("CNN1D", CNN1D), ("BiLSTM", BiLSTM),
                   ("Transformer", TemporalTransformer), ("STGCN", STGCN)]:
    m = cls()
    out = m(dummy)
    assert out.shape == (4, 2)
    n = sum(p.numel() for p in m.parameters())
    print(f"  ✅ {name:<16} params={n:,}")

print("\n" + "="*50)
print("  SETUP COMPLETE")
print("="*50)
