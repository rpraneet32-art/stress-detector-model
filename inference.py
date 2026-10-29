"""
inference.py — Standalone Real-Time ONNX Inference Engine.

Runs real-time stress detection on 3.0-second biosignal windows
(ECG, PPG, Skin Temp @ 128 Hz) using ONNX Runtime alone.
Zero PyTorch dependency — ready for embedded/backend streaming from ESP32.
"""

import os
import time
import numpy as np
import onnxruntime as ort

MODEL_PATH = os.path.join(os.path.dirname(__file__), "outputs", "stgcn.onnx")


class StressDetector:
    """Standalone stress inference engine using ONNX Runtime."""

    def __init__(self, model_path: str = MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}. Run training and ONNX export first.")
        
        # Optimize ONNX Runtime session for low-latency CPU inference
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, window: np.ndarray, is_preprocessed: bool = True) -> dict:
        """
        Run inference on a single 3-second window.

        Parameters
        ----------
        window : np.ndarray of shape (3, 384) or (1, 3, 384)
                 Channel 0: AD8232 ECG (128 Hz)
                 Channel 1: MAX30102 PPG (128 Hz)
                 Channel 2: DS18B20 Skin Temperature (128 Hz)
        is_preprocessed : bool, default True
                 If True, assumes window is already bandpass filtered and subject-normalized
                 (e.g., cached WESAD benchmark windows).
                 If False, applies per-window standardization for uncalibrated raw live input.

        Returns
        -------
        dict with prediction (0=Non-Stress, 1=Stress), probability, and latency_ms.
        """
        if window.ndim == 2:
            window = np.expand_dims(window, axis=0)  # (1, 3, 384)
        
        window = window.astype(np.float32)

        if not is_preprocessed:
            # Per-window z-score normalization for uncalibrated raw input
            mean = window.mean(axis=-1, keepdims=True)
            std = window.std(axis=-1, keepdims=True) + 1e-8
            norm_window = (window - mean) / std
        else:
            # Direct pass-through for pre-normalized benchmark inputs
            norm_window = window

        t0 = time.perf_counter()
        logits = self.session.run(None, {self.input_name: norm_window})[0]  # (1, 2)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Softmax
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
        stress_prob = float(probs[0, 1])
        prediction = int(np.argmax(probs, axis=-1)[0])

        return {
            "prediction": prediction,
            "label": "Stress" if prediction == 1 else "Non-Stress",
            "stress_probability": round(stress_prob, 4),
            "non_stress_probability": round(1.0 - stress_prob, 4),
            "latency_ms": round(latency_ms, 2),
        }


if __name__ == "__main__":
    detector = StressDetector()
    print("StressDetector initialized successfully.")
    
    # Test with a dummy 3-second window
    dummy_window = np.random.randn(3, 384).astype(np.float32)
    res = detector.predict(dummy_window)
    print("\nTest Window Inference Result:")
    for k, v in res.items():
        print(f"  {k:<24}: {v}")
