"""
demo.py — Live Interactive Demonstration & Validation Script.

Demonstrates that the champion ST-GCN ONNX model works on REAL held-out test data
extracted from the WESAD dataset (Subject S2).

Usage:
  python demo.py [--n-samples 10]
"""

import os
import argparse
import numpy as np
from inference import StressDetector

PROCESSED_DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "processed", "wesad_windows.npz")


def run_demo(n_samples: int = 10, subject_id: int = 2):
    print("=" * 70)
    print("   MULTIMODAL STRESS DETECTOR - LIVE MODEL DEMONSTRATION")
    print("=" * 70)
    print(f"Loading champion model (ST-GCN ONNX)...")
    detector = StressDetector()
    print("Model loaded successfully via ONNX Runtime!\n")

    if not os.path.exists(PROCESSED_DATA_PATH):
        print(f"Error: Processed test dataset not found at {PROCESSED_DATA_PATH}")
        return

    print(f"Loading real held-out test data from {PROCESSED_DATA_PATH}...")
    data = np.load(PROCESSED_DATA_PATH)
    tensors = data["tensors"]       # (N, 3, 384)
    labels = data["labels"]         # (N,)
    subject_ids = data["subject_ids"] # (N,)

    # Filter for the target test subject (default S2)
    mask = subject_ids == subject_id
    sub_tensors = tensors[mask]
    sub_labels = labels[mask]

    stress_idx = np.where(sub_labels == 1)[0]
    non_stress_idx = np.where(sub_labels == 0)[0]

    print(f"Test Subject S{subject_id} available windows: {len(sub_labels)}")
    print(f"  - Stress windows (TSST Induction)  : {len(stress_idx)}")
    print(f"  - Non-Stress windows (Baseline/Amuse): {len(non_stress_idx)}\n")

    # Pick alternating samples of real stress and non-stress windows
    half = n_samples // 2
    np.random.seed(42)
    sample_stress = np.random.choice(stress_idx, size=half, replace=False)
    sample_non_stress = np.random.choice(non_stress_idx, size=n_samples - half, replace=False)
    selected_indices = np.empty((n_samples,), dtype=int)
    selected_indices[0::2] = sample_stress[:len(selected_indices[0::2])]
    selected_indices[1::2] = sample_non_stress[:len(selected_indices[1::2])]

    print(f"{'Sample':<7} | {'Ground Truth':<12} | {'Predicted':<12} | {'Stress Prob':<12} | {'Latency':<8} | {'Status'}")
    print("-" * 70)

    correct = 0
    latencies = []

    for i, idx in enumerate(selected_indices):
        window = sub_tensors[idx]
        true_label = sub_labels[idx]
        true_str = "Stress" if true_label == 1 else "Non-Stress"

        res = detector.predict(window)
        pred_str = res["label"]
        stress_prob = res["stress_probability"]
        lat = res["latency_ms"]
        latencies.append(lat)

        is_match = (res["prediction"] == true_label)
        if is_match:
            correct += 1
            status = "MATCH [OK]"
        else:
            status = "MISMATCH"

        bar_len = int(stress_prob * 10)
        prob_bar = "#" * bar_len + "-" * (10 - bar_len)
        print(f"#{i+1:<6} | {true_str:<12} | {pred_str:<12} | {stress_prob:>5.1%} [{prob_bar}] | {lat:>5.1f}ms | {status}")

    print("-" * 70)
    acc = (correct / n_samples) * 100.0
    avg_lat = np.mean(latencies)
    print(f"\nLive Demonstration Summary:")
    print(f"  - Accuracy on Sample Batch : {acc:.1f}% ({correct}/{n_samples} correct)")
    print(f"  - Average CPU Latency      : {avg_lat:.2f} ms per 3-second window (Target: < 50 ms)")
    print(f"  - Real-Time Capability     : {3000 / avg_lat:.0f}x faster than real-time incoming stream!")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live stress detection demo")
    parser.add_argument("--n-samples", type=int, default=10, help="Number of test windows to demonstrate")
    parser.add_argument("--subject", type=int, default=2, help="Held-out subject ID (e.g. 2)")
    args = parser.parse_args()
    run_demo(n_samples=args.n_samples, subject_id=args.subject)
