"""
data/loader.py — WESAD per-subject data loader (Section 2 & 3).

Loads one subject's .pkl file and returns:
  - signals: np.ndarray of shape (T, 3) at 128 Hz  [ECG, PPG, Temp]
  - labels:  np.ndarray of shape (T,) at 128 Hz    [0=non-stress, 1=stress, -1=discard]

Key design decisions (per spec):
  - ECG  : chest['ECG']  @ 700 Hz → decimated to 128 Hz
  - PPG  : wrist['BVP']  @  64 Hz → cubic-spline upsampled to 128 Hz
  - Temp : wrist['TEMP'] @   4 Hz → linearly interpolated to 128 Hz
  - label: chest label   @ 700 Hz → mode-aggregated at 128 Hz
  - Non-stress = WESAD conditions {1, 3}; Stress = condition {2}
  - All other condition codes → -1 (discarded at windowing stage)
"""

import os
import pickle
import numpy as np
from scipy import signal as sp_signal
from scipy.interpolate import CubicSpline
from typing import Tuple

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    TARGET_FS, ALL_SUBJECTS, subject_pkl_path,
    WESAD_NON_STRESS_CONDITIONS, WESAD_STRESS_CONDITIONS,
    LABEL_NON_STRESS, LABEL_STRESS,
    CH_ECG, CH_PPG, CH_TEMP,
)

# ── Source sampling rates (WESAD spec) ──────────────────────────────────────
FS_CHEST = 700      # ECG, label channel
FS_BVP   = 64       # wrist BVP (PPG proxy)
FS_TEMP  = 4        # wrist TEMP


# ── Resampling helpers ───────────────────────────────────────────────────────

def _decimate_to_target(x: np.ndarray, src_fs: int, tgt_fs: int) -> np.ndarray:
    """
    Anti-aliased downsampling via scipy.signal.resample_poly.
    Used for ECG: 700 → 128 Hz.
    """
    from math import gcd
    g = gcd(src_fs, tgt_fs)
    up, down = tgt_fs // g, src_fs // g
    return sp_signal.resample_poly(x, up, down)


def _cubic_upsample(x: np.ndarray, src_fs: int, tgt_fs: int) -> np.ndarray:
    """
    Cubic spline interpolation upsampling.
    Used for PPG (BVP): 64 → 128 Hz.
    """
    n_src = len(x)
    t_src = np.arange(n_src) / src_fs
    t_tgt = np.arange(0, t_src[-1], 1.0 / tgt_fs)
    cs = CubicSpline(t_src, x)
    return cs(t_tgt)


def _linear_upsample(x: np.ndarray, src_fs: int, tgt_fs: int) -> np.ndarray:
    """
    Linear interpolation upsampling.
    Used for Temperature: 4 → 128 Hz.
    """
    n_src = len(x)
    t_src = np.arange(n_src) / src_fs
    t_tgt = np.arange(0, t_src[-1], 1.0 / tgt_fs)
    return np.interp(t_tgt, t_src, x)


def _mode_aggregate_labels(raw_label: np.ndarray, src_fs: int, tgt_fs: int) -> np.ndarray:
    """
    Resample label channel (700 Hz) to 128 Hz via majority-vote in each block.
    Each output sample represents src_fs/tgt_fs input samples; we take the mode.
    """
    ratio = src_fs / tgt_fs                    # e.g. 700/128 ≈ 5.47
    n_out = int(len(raw_label) / ratio)
    out = np.zeros(n_out, dtype=np.int32)
    for i in range(n_out):
        i_start = int(i * ratio)
        i_end   = int((i + 1) * ratio)
        block   = raw_label[i_start:i_end]
        values, counts = np.unique(block, return_counts=True)
        out[i] = values[np.argmax(counts)]
    return out


def _map_labels(raw: np.ndarray) -> np.ndarray:
    """
    Map WESAD condition codes to binary labels.
      {1, 3} → 0 (non-stress)
      {2}    → 1 (stress)
      else   → -1 (discard)
    """
    out = np.full_like(raw, -1, dtype=np.int32)
    for c in WESAD_NON_STRESS_CONDITIONS:
        out[raw == c] = LABEL_NON_STRESS
    for c in WESAD_STRESS_CONDITIONS:
        out[raw == c] = LABEL_STRESS
    return out


# ── Main loader ─────────────────────────────────────────────────────────────

def load_subject(subject_id: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and align channels for one WESAD subject.

    Returns
    -------
    signals : np.ndarray, shape (T, 3)  — [ECG, PPG, Temp] at TARGET_FS Hz
    labels  : np.ndarray, shape (T,)    — {0, 1, -1} at TARGET_FS Hz
    """
    pkl_path = subject_pkl_path(subject_id)
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Subject pkl not found: {pkl_path}")

    import warnings
    with open(pkl_path, "rb") as f:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            warnings.simplefilter("ignore", FutureWarning)
            data = pickle.load(f, encoding="latin1")

    # ── Extract raw channels ────────────────────────────────────────────
    ecg_raw   = data["signal"]["chest"]["ECG"].squeeze().astype(np.float64)
    bvp_raw   = data["signal"]["wrist"]["BVP"].squeeze().astype(np.float64)
    temp_raw  = data["signal"]["wrist"]["TEMP"].squeeze().astype(np.float64)
    label_raw = data["label"].squeeze().astype(np.int32)

    # ── Resample to TARGET_FS ──────────────────────────────────────────
    ecg_rs   = _decimate_to_target(ecg_raw,  FS_CHEST, TARGET_FS)
    bvp_rs   = _cubic_upsample(bvp_raw,      FS_BVP,   TARGET_FS)
    temp_rs  = _linear_upsample(temp_raw,    FS_TEMP,  TARGET_FS)
    label_rs = _mode_aggregate_labels(label_raw, FS_CHEST, TARGET_FS)

    # ── Align lengths to the shortest channel ──────────────────────────
    min_len = min(len(ecg_rs), len(bvp_rs), len(temp_rs), len(label_rs))
    ecg_rs   = ecg_rs[:min_len]
    bvp_rs   = bvp_rs[:min_len]
    temp_rs  = temp_rs[:min_len]
    label_rs = label_rs[:min_len]

    # ── Stack and label-map ────────────────────────────────────────────
    signals = np.stack([ecg_rs, bvp_rs, temp_rs], axis=1)   # (T, 3)
    labels  = _map_labels(label_rs)                           # (T,)  {0,1,-1}

    return signals, labels


def channel_stats(signals: np.ndarray, labels: np.ndarray, subject_id: int) -> dict:
    """Return per-channel descriptive statistics and class balance for one subject."""
    ch_names = ["ECG", "PPG", "Temp"]
    stats = {"subject": subject_id}

    for i, name in enumerate(ch_names):
        ch = signals[:, i]
        stats[name] = {
            "min":  float(ch.min()),
            "max":  float(ch.max()),
            "mean": float(ch.mean()),
            "std":  float(ch.std()),
            "n_samples": int(len(ch)),
            "n_nan": int(np.isnan(ch).sum()),
            "n_inf": int(np.isinf(ch).sum()),
        }

    n_nonstress = int((labels == LABEL_NON_STRESS).sum())
    n_stress    = int((labels == LABEL_STRESS).sum())
    n_discard   = int((labels == -1).sum())
    total_valid = n_nonstress + n_stress

    stats["class_balance"] = {
        "n_non_stress": n_nonstress,
        "n_stress":     n_stress,
        "n_discard":    n_discard,
        "stress_frac":  round(n_stress / total_valid, 4) if total_valid > 0 else None,
    }
    return stats


# ── Sanity-check script ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    from config import OUTPUTS_DIR

    all_stats = []
    for sid in ALL_SUBJECTS:
        print(f"  Loading S{sid}...", end=" ", flush=True)
        try:
            sigs, labs = load_subject(sid)
            s = channel_stats(sigs, labs, sid)
            all_stats.append(s)
            cb = s["class_balance"]
            print(
                f"T={sigs.shape[0]} | non-stress={cb['n_non_stress']} "
                f"stress={cb['n_stress']} discard={cb['n_discard']} "
                f"stress_frac={cb['stress_frac']}"
            )
        except FileNotFoundError as e:
            print(f"MISSING: {e}")

    # Save as JSON summary
    out_path = os.path.join(OUTPUTS_DIR, "loader_sanity_check.json")
    with open(out_path, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"\nSaved channel stats to {out_path}")
