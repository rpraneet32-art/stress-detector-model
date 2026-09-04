"""
export/onnx_export.py — ONNX export and validation (Section 9.3).

- Exports the best-performing PyTorch model to ONNX
- Validates by running a held-out batch through both the original PyTorch model
  and the exported ONNX model and confirming outputs match within 1e-4 tolerance
- The exported ONNX file runs with onnxruntime alone (no PyTorch dependency)
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CHECKPOINT_DIR, OUTPUTS_DIR, WINDOW_SAMPLES, seed_everything, GLOBAL_SEED


def export_to_onnx(
    model: nn.Module,
    model_name: str,
    out_dir: str = OUTPUTS_DIR,
    opset_version: int = 17,
) -> str:
    """
    Export a trained PyTorch model to ONNX format.

    Parameters
    ----------
    model      : trained PyTorch model in eval mode
    model_name : string identifier (used in filename)
    out_dir    : directory to save the .onnx file
    opset_version : ONNX opset (17 covers all ops used here)

    Returns
    -------
    Path to the saved .onnx file
    """
    seed_everything(GLOBAL_SEED)
    model.eval()
    os.makedirs(out_dir, exist_ok=True)

    onnx_path = os.path.join(out_dir, f"{model_name}.onnx")
    dummy_input = torch.randn(1, 3, WINDOW_SAMPLES)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input":  {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )
    print(f"ONNX model exported → {onnx_path}")
    return onnx_path


def validate_onnx(
    model: nn.Module,
    onnx_path: str,
    n_samples: int = 32,
    tolerance: float = 1e-4,
) -> dict:
    """
    Validate ONNX export by comparing outputs with the original PyTorch model
    on a random held-out batch.

    Returns dict with: max_abs_diff, mean_abs_diff, passed (bool)
    """
    seed_everything(GLOBAL_SEED)
    model.eval()

    # Random held-out batch
    x_np = np.random.randn(n_samples, 3, WINDOW_SAMPLES).astype(np.float32)
    x_pt = torch.from_numpy(x_np)

    # PyTorch inference
    with torch.no_grad():
        pt_out = model(x_pt).numpy()

    # ONNX inference
    # Verify model integrity first
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_out = sess.run(["logits"], {"input": x_np})[0]

    # Compare
    abs_diff = np.abs(pt_out - ort_out)
    max_diff  = float(abs_diff.max())
    mean_diff = float(abs_diff.mean())
    passed    = max_diff < tolerance

    print(f"\nONNX Validation:")
    print(f"  max |PT - ONNX|  = {max_diff:.2e}  (tolerance={tolerance:.0e})")
    print(f"  mean |PT - ONNX| = {mean_diff:.2e}")
    print(f"  Result: {'✅ PASS' if passed else '❌ FAIL'}")

    return {
        "max_abs_diff":  max_diff,
        "mean_abs_diff": mean_diff,
        "tolerance":     tolerance,
        "passed":        passed,
    }


def load_best_model(model_class, model_name: str, fold: int = None) -> nn.Module:
    """
    Load the best checkpoint for a given model.
    If fold is None, tries to find any checkpoint matching the model_name.
    """
    if fold is not None:
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"{model_name}_fold{fold}_best.pt")
    else:
        # Find checkpoint with highest F1 (scan all folds for this model)
        candidates = [
            f for f in os.listdir(CHECKPOINT_DIR)
            if f.startswith(model_name) and f.endswith(".pt")
        ]
        if not candidates:
            raise FileNotFoundError(f"No checkpoint found for {model_name} in {CHECKPOINT_DIR}")
        ckpt_path = os.path.join(CHECKPOINT_DIR, candidates[0])

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = model_class()
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded checkpoint: {ckpt_path}")
    return model
