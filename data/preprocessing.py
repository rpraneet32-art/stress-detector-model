"""
data/preprocessing.py — Signal filtering and normalization (Section 4).

Pipeline per subject, per channel (in this exact order):
  1. Powerline notch filter: IIR notch at 50 Hz, Q=30 — all channels
  2. Channel-specific bandpass:
       ECG:  4th-order Butterworth, 0.5–40 Hz
       PPG:  3rd-order Butterworth, 0.5–5.0 Hz
       Temp: notch only (no bandpass — slow-varying signal)
  3. Subject-wise z-score normalization: (x - μ_subject) / σ_subject
     Statistics computed from that subject's own data only (no global stats).

Input:  raw aligned (T, 3) array from loader.py
Output: filtered, normalized (T, 3) array
"""

import os
import sys
import numpy as np
from scipy import signal as sp_signal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    TARGET_FS,
    NOTCH_FREQ, NOTCH_Q,
    ECG_BP_LOW, ECG_BP_HIGH, ECG_BP_ORDER,
    PPG_BP_LOW, PPG_BP_HIGH, PPG_BP_ORDER,
    CH_ECG, CH_PPG, CH_TEMP,
)


# ── Filter design ────────────────────────────────────────────────────────────

def _notch_sos(fs: float = TARGET_FS) -> np.ndarray:
    """IIR notch filter at NOTCH_FREQ Hz, returned as SOS coefficients."""
    b, a = sp_signal.iirnotch(NOTCH_FREQ, NOTCH_Q, fs=fs)
    return sp_signal.tf2sos(b, a)


def _bandpass_sos(low: float, high: float, order: int, fs: float = TARGET_FS) -> np.ndarray:
    """Butterworth bandpass filter, returned as SOS coefficients."""
    nyq = fs / 2.0
    return sp_signal.butter(order, [low / nyq, high / nyq], btype="band", output="sos")


# Pre-compute filter coefficients (same for all subjects)
_notch    = _notch_sos(TARGET_FS)
_bp_ecg   = _bandpass_sos(ECG_BP_LOW, ECG_BP_HIGH, ECG_BP_ORDER, TARGET_FS)
_bp_ppg   = _bandpass_sos(PPG_BP_LOW, PPG_BP_HIGH, PPG_BP_ORDER, TARGET_FS)


# ── Core filtering ───────────────────────────────────────────────────────────

def _apply_notch(x: np.ndarray) -> np.ndarray:
    return sp_signal.sosfiltfilt(_notch, x)


def _apply_ecg_bandpass(x: np.ndarray) -> np.ndarray:
    return sp_signal.sosfiltfilt(_bp_ecg, x)


def _apply_ppg_bandpass(x: np.ndarray) -> np.ndarray:
    return sp_signal.sosfiltfilt(_bp_ppg, x)


def _zscore_normalize(x: np.ndarray) -> np.ndarray:
    """Subject-wise z-score: (x - μ) / σ.  σ=1 if flat signal."""
    mu  = x.mean()
    std = x.std()
    if std < 1e-8:
        return x - mu          # flat signal — just center
    return (x - mu) / std


# ── Public API ───────────────────────────────────────────────────────────────

def preprocess(raw_signals: np.ndarray) -> np.ndarray:
    """
    Apply the full preprocessing pipeline to a subject's aligned signal array.

    Parameters
    ----------
    raw_signals : np.ndarray, shape (T, 3)
        Raw ECG, PPG, Temp channels at TARGET_FS Hz.

    Returns
    -------
    filtered : np.ndarray, shape (T, 3)
        Filtered and z-score normalized signals.
    """
    T, C = raw_signals.shape
    assert C == 3, f"Expected 3 channels, got {C}"
    out = raw_signals.copy().astype(np.float64)

    # Step 1 — Notch filter (all channels)
    for ch in range(C):
        out[:, ch] = _apply_notch(out[:, ch])

    # Step 2 — Channel-specific bandpass
    out[:, CH_ECG]  = _apply_ecg_bandpass(out[:, CH_ECG])
    out[:, CH_PPG]  = _apply_ppg_bandpass(out[:, CH_PPG])
    # Temp: notch only — no further bandpass

    # Step 3 — Subject-wise z-score normalization
    for ch in range(C):
        out[:, ch] = _zscore_normalize(out[:, ch])

    return out


# ── Diagnostic plot helper ───────────────────────────────────────────────────

def plot_raw_vs_filtered(
    raw: np.ndarray,
    filtered: np.ndarray,
    subject_id: int,
    save_dir: str,
    duration_sec: float = 10.0,
) -> str:
    """
    Save a diagnostic figure showing raw vs. filtered for 10 s of each channel.
    Returns the saved file path.
    """
    import matplotlib.pyplot as plt

    n_plot = int(duration_sec * TARGET_FS)
    ch_names = ["ECG", "PPG", "Temp"]
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    t = np.arange(n_plot) / TARGET_FS

    for i, (ax, name) in enumerate(zip(axes, ch_names)):
        ax.plot(t, raw[:n_plot, i],      alpha=0.6, label="Raw",      color="steelblue",  lw=0.8)
        ax.plot(t, filtered[:n_plot, i], alpha=0.9, label="Filtered", color="darkorange", lw=1.0)
        ax.set_ylabel(name, fontsize=10)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"Raw vs Filtered — Subject S{subject_id}", fontsize=12)
    fig.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"S{subject_id}_raw_vs_filtered.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ── Quick smoke-test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    from data.loader import load_subject
    from config import OUTPUTS_DIR

    # Test on first available subject
    sid = 2
    print(f"Loading S{sid}...")
    raw_sigs, _ = load_subject(sid)

    print("Preprocessing...")
    filt_sigs = preprocess(raw_sigs)

    # Stats before / after
    ch_names = ["ECG", "PPG", "Temp"]
    print(f"\n{'Channel':<8} {'Raw μ':>10} {'Raw σ':>10} {'Filt μ':>10} {'Filt σ':>10}")
    for i, name in enumerate(ch_names):
        print(
            f"{name:<8}"
            f" {raw_sigs[:, i].mean():>10.4f}"
            f" {raw_sigs[:, i].std():>10.4f}"
            f" {filt_sigs[:, i].mean():>10.4f}"
            f" {filt_sigs[:, i].std():>10.4f}"
        )

    plot_path = plot_raw_vs_filtered(raw_sigs, filt_sigs, sid, OUTPUTS_DIR)
    print(f"\nDiagnostic plot saved → {plot_path}")
    print("Preprocessing smoke-test PASSED.")
