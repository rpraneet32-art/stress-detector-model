"""
data/windowing.py — Windowing, segmentation, and classical feature extraction (Section 5).

Produces two parallel datasets:
  1. Raw tensors   (N, 3, 384) — channels-first — for deep learning models
  2. Feature vectors (N, 36)   — statistical/frequency features — for classical baselines

Both datasets include subject IDs and binary labels and are saved to disk as .npz
so training scripts never repeat preprocessing on every run.

Classical feature set (Section 5.1):
  ECG   (8): Mean RR*, SDNN*, RMSSD*, pNN50*, VLF power, LF power, HF power, LF/HF ratio
  PPG   (4): Systolic peak amplitude, pulse interval variance, peak-to-peak std, SpO2 mean*
  Temp  (3): Mean, variance, linear slope (dT/dt)
  Total: up to 15 per channel-group, but see FEATURE_NAMES for exact count.

  *RR-interval features are approximated from the PPG/ECG at 128 Hz — a true R-peak
   detector is used for ECG; for subjects where no peaks are detected the values default
   to NaN (documented in the summary CSV).

Note on SpO2: WESAD's BVP channel is not a calibrated SpO2 trace. We use the BVP mean
amplitude as a proxy and document the deviation from the spec in the summary CSV.
"""

import os
import sys
import numpy as np
from scipy import signal as sp_signal
from scipy.stats import linregress
from typing import List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    TARGET_FS, WINDOW_SAMPLES, STEP_SAMPLES,
    LABEL_NON_STRESS, LABEL_STRESS,
    PROCESSED_DATA_DIR, ALL_SUBJECTS,
    CH_ECG, CH_PPG, CH_TEMP,
)

# ── Feature names (for documentation / column headers) ──────────────────────

FEATURE_NAMES = [
    # ECG time-domain HRV (from R-peak detection on the 128 Hz ECG window)
    "ecg_mean_rr",          # 0
    "ecg_sdnn",             # 1
    "ecg_rmssd",            # 2
    "ecg_pnn50",            # 3
    # ECG frequency-domain HRV (Welch PSD on RR series)
    "ecg_vlf_power",        # 4
    "ecg_lf_power",         # 5
    "ecg_hf_power",         # 6
    "ecg_lf_hf_ratio",      # 7
    # PPG morphological
    "ppg_systolic_amp",     # 8  (mean peak amplitude)
    "ppg_pulse_interval_var",  # 9
    "ppg_pp_std",           # 10 (peak-to-peak interval std)
    "ppg_spo2_proxy",       # 11 (BVP channel mean; true SpO2 unavailable — documented)
    # Temperature
    "temp_mean",            # 12
    "temp_var",             # 13
    "temp_slope",           # 14 (dT/dt via linear regression over window)
]

N_FEATURES = len(FEATURE_NAMES)   # 15 per window


# ── R-peak detector (Pan-Tompkins-inspired, simplified) ─────────────────────

def _detect_r_peaks(ecg: np.ndarray, fs: float = TARGET_FS) -> np.ndarray:
    """
    Lightweight R-peak detector using the squared derivative method.
    Returns sample indices of detected R-peaks.
    """
    # Differentiate and square
    diff_sq = np.diff(ecg, prepend=ecg[0]) ** 2
    # Smooth with a moving average (~50 ms window)
    win = max(1, int(0.05 * fs))
    kernel = np.ones(win) / win
    smoothed = np.convolve(diff_sq, kernel, mode="same")

    # Adaptive threshold: 40% of max
    thresh = 0.4 * smoothed.max()
    if thresh == 0:
        return np.array([], dtype=int)

    # Minimum distance between peaks: 250 ms (max ~240 BPM)
    min_dist = int(0.25 * fs)
    peaks, _ = sp_signal.find_peaks(smoothed, height=thresh, distance=min_dist)
    return peaks


def _hrv_features(ecg_window: np.ndarray) -> Tuple[float, ...]:
    """Compute 8 HRV features from a 3-second ECG window."""
    peaks = _detect_r_peaks(ecg_window)
    if len(peaks) < 2:
        return (np.nan,) * 8

    rr = np.diff(peaks) / TARGET_FS * 1000  # RR intervals in ms
    mean_rr = float(np.mean(rr))
    sdnn    = float(np.std(rr, ddof=1)) if len(rr) > 1 else np.nan
    rmssd   = float(np.sqrt(np.mean(np.diff(rr) ** 2))) if len(rr) > 1 else np.nan
    pnn50   = float(np.mean(np.abs(np.diff(rr)) > 50)) if len(rr) > 1 else np.nan

    # Frequency-domain: Welch PSD on RR series (uniform resample to 4 Hz)
    try:
        t_rr  = np.cumsum(rr) / 1000       # seconds
        t_uni = np.arange(t_rr[0], t_rr[-1], 1.0 / 4.0)
        if len(t_uni) < 4:
            raise ValueError("too short")
        rr_uni = np.interp(t_uni, t_rr, rr)
        freqs, psd = sp_signal.welch(rr_uni, fs=4.0, nperseg=min(len(rr_uni), 64))
        def _band(lo, hi):
            mask = (freqs >= lo) & (freqs < hi)
            return float(np.trapz(psd[mask], freqs[mask])) if mask.any() else np.nan
        vlf = _band(0.0033, 0.04)
        lf  = _band(0.04,   0.15)
        hf  = _band(0.15,   0.40)
        lf_hf = (lf / hf) if (hf and hf > 0) else np.nan
    except Exception:
        vlf = lf = hf = lf_hf = np.nan

    return mean_rr, sdnn, rmssd, pnn50, vlf, lf, hf, lf_hf


def _ppg_features(ppg_window: np.ndarray) -> Tuple[float, ...]:
    """Compute 4 PPG morphological features."""
    # Peak detection on PPG
    min_dist = int(0.3 * TARGET_FS)   # ≥ 300 ms between peaks
    peaks, props = sp_signal.find_peaks(
        ppg_window, distance=min_dist, prominence=0.01
    )
    if len(peaks) >= 2:
        amps     = ppg_window[peaks]
        sys_amp  = float(np.mean(amps))
        intervals = np.diff(peaks).astype(float)
        pi_var   = float(np.var(intervals))
        pp_std   = float(np.std(intervals))
    else:
        sys_amp = pi_var = pp_std = np.nan

    # SpO2 proxy: BVP channel mean (true SpO2 not derivable — documented)
    spo2_proxy = float(np.mean(ppg_window))

    return sys_amp, pi_var, pp_std, spo2_proxy


def _temp_features(temp_window: np.ndarray) -> Tuple[float, ...]:
    """Compute 3 temperature features."""
    t_mean = float(np.mean(temp_window))
    t_var  = float(np.var(temp_window))
    x = np.arange(len(temp_window))
    slope, *_ = linregress(x, temp_window)
    t_slope = float(slope)
    return t_mean, t_var, t_slope


def extract_features(window: np.ndarray) -> np.ndarray:
    """
    Extract the 15-feature vector from one (3, 384) window (channels-first).
    NaN values are allowed and will be imputed at training time.
    """
    assert window.shape == (3, WINDOW_SAMPLES), \
        f"Expected (3, {WINDOW_SAMPLES}), got {window.shape}"

    ecg_feats  = _hrv_features(window[CH_ECG])       # 8
    ppg_feats  = _ppg_features(window[CH_PPG])        # 4
    temp_feats = _temp_features(window[CH_TEMP])       # 3

    return np.array(ecg_feats + ppg_feats + temp_feats, dtype=np.float32)


# ── Windowing ────────────────────────────────────────────────────────────────

def window_subject(
    signals: np.ndarray,
    labels: np.ndarray,
    subject_id: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Slide windows over one subject's preprocessed signals.

    Returns
    -------
    tensors  : (N, 3, 384) float32   — raw windows for deep learning
    features : (N, 15)    float32   — classical feature vectors
    win_labels : (N,)     int32     — window labels {0, 1}
    subject_ids: (N,)     int32     — subject ID tag (required for LOSO split)
    """
    T, C = signals.shape
    assert C == 3

    tensors_list  = []
    features_list = []
    labels_list   = []

    start = 0
    while start + WINDOW_SAMPLES <= T:
        end = start + WINDOW_SAMPLES
        win_sig  = signals[start:end, :]          # (384, 3)
        win_lab  = labels[start:end]              # (384,)

        # --- Label: majority vote; discard if no clear majority
        valid_mask = win_lab >= 0
        if valid_mask.sum() == 0:
            start += STEP_SAMPLES
            continue
        valid_labels = win_lab[valid_mask]
        values, counts = np.unique(valid_labels, return_counts=True)
        winner_count = counts.max()
        if winner_count / len(valid_labels) <= 0.5:
            # No majority — discard this boundary window
            start += STEP_SAMPLES
            continue
        win_label = int(values[np.argmax(counts)])

        # Channels-first for deep learning: (3, 384)
        win_chfirst = win_sig.T.astype(np.float32)

        tensors_list.append(win_chfirst)
        features_list.append(extract_features(win_chfirst))
        labels_list.append(win_label)

        start += STEP_SAMPLES

    if not tensors_list:
        # Edge case: subject too short for any window
        empty = np.zeros((0,), dtype=np.int32)
        return (
            np.zeros((0, 3, WINDOW_SAMPLES), dtype=np.float32),
            np.zeros((0, N_FEATURES), dtype=np.float32),
            empty,
            empty,
        )

    tensors     = np.stack(tensors_list, axis=0)
    features    = np.stack(features_list, axis=0)
    win_labels  = np.array(labels_list, dtype=np.int32)
    subject_ids = np.full(len(win_labels), subject_id, dtype=np.int32)

    return tensors, features, win_labels, subject_ids


# ── Dataset builder (all subjects) ───────────────────────────────────────────

def build_dataset(subjects: List[int] = None) -> dict:
    """
    Run the full pipeline (load → preprocess → window) for all subjects and
    return a dict with the two parallel datasets.
    """
    from data.loader import load_subject
    from data.preprocessing import preprocess

    if subjects is None:
        subjects = ALL_SUBJECTS

    all_tensors   = []
    all_features  = []
    all_labels    = []
    all_sids      = []

    for sid in subjects:
        print(f"  Processing S{sid}...", end=" ", flush=True)
        raw, labels = load_subject(sid)
        filtered    = preprocess(raw)
        tens, feats, labs, sids = window_subject(filtered, labels, sid)
        all_tensors.append(tens)
        all_features.append(feats)
        all_labels.append(labs)
        all_sids.append(sids)
        print(f"-> {len(labs)} windows (stress={int((labs==1).sum())}, non-stress={int((labs==0).sum())})")

    dataset = {
        "tensors":     np.concatenate(all_tensors,  axis=0),   # (N, 3, 384)
        "features":    np.concatenate(all_features, axis=0),   # (N, 15)
        "labels":      np.concatenate(all_labels,   axis=0),   # (N,)
        "subject_ids": np.concatenate(all_sids,     axis=0),   # (N,)
    }
    return dataset


def save_dataset(dataset: dict, out_dir: str = PROCESSED_DATA_DIR) -> str:
    """Save dataset to disk as .npz. Returns the saved file path."""
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "wesad_windows.npz")
    np.savez_compressed(out_path, **dataset)
    print(f"Dataset saved -> {out_path}")
    print(f"  tensors:     {dataset['tensors'].shape}")
    print(f"  features:    {dataset['features'].shape}")
    print(f"  labels:      {dataset['labels'].shape}")
    print(f"  subject_ids: {dataset['subject_ids'].shape}")
    print(f"  stress frac: {(dataset['labels']==1).mean():.3f}")
    return out_path


def load_dataset(out_dir: str = PROCESSED_DATA_DIR) -> dict:
    """Load preprocessed dataset from disk."""
    path = os.path.join(out_dir, "wesad_windows.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Preprocessed dataset not found at {path}. Run build_dataset first.")
    data = np.load(path, allow_pickle=False)
    return {k: data[k] for k in data.files}


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    from config import OUTPUTS_DIR, seed_everything
    seed_everything()

    print("Building full WESAD dataset...")
    ds = build_dataset()
    save_dataset(ds)

    # Per-subject window count summary
    summary = []
    for sid in ALL_SUBJECTS:
        mask = ds["subject_ids"] == sid
        n_total    = mask.sum()
        n_stress   = int((ds["labels"][mask] == 1).sum())
        n_nonstress= int((ds["labels"][mask] == 0).sum())
        summary.append({
            "subject": sid,
            "n_windows": int(n_total),
            "n_stress": n_stress,
            "n_non_stress": n_nonstress,
        })

    out_path = os.path.join(OUTPUTS_DIR, "windowing_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWindowing summary saved -> {out_path}")
    print(f"Feature names ({N_FEATURES}): {FEATURE_NAMES}")
