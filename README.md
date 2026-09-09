# Multimodal Wearable Stress Detection Model

An end-to-end deep learning and signal processing pipeline for real-time acute stress detection from multimodal wearable physiological signals, developed against the **WESAD (Wearable Stress and Affect Detection)** dataset.

---

## 📌 Architecture & Features

### 1. Physiological Biosignal Suite (128 Hz Synchronized)
* **AD8232 ECG**: Decimated and anti-aliased downsampled from 700 Hz to 128 Hz.
* **MAX30102 PPG**: Cubic spline upsampled from 64 Hz to 128 Hz.
* **DS18B20 Skin Temperature**: Linearly interpolated from 4 Hz to 128 Hz.
* **Ground Truth Labels**: Mode-aggregated over 3.0-second sliding windows (50% overlap, 384 samples per window) matching WESAD protocol (Condition 1 & 3: Non-Stress, Condition 2: Stress).

### 2. Preprocessing & Signal Pipeline
* **Powerline Notch Filter**: 50 Hz IIR notch filter ($Q=30$).
* **Bandpass Filtering**:
  * ECG: 4th-order Butterworth bandpass (0.5 – 40.0 Hz).
  * PPG: 3rd-order Butterworth bandpass (0.5 – 5.0 Hz).
* **Subject-wise Z-Score Normalization**: Removes cross-subject baseline drift while preserving dynamic stress reactivity.

### 3. Models Evaluated
* **ST-GCN (Spatio-Temporal Graph Convolutional Network)**: Dynamically computes Pearson cross-correlation adjacency matrices between channels per 3s window.
* **1D-CNN**: Multi-kernel temporal convolutional baseline with residual connections.
* **Temporal Transformer**: Multi-head self-attention across temporal biosignal tokens.
* **BiLSTM**: Bidirectional recurrent baseline.
* **Classical Baselines**: Random Forest, XGBoost, and Support Vector Machine trained on 15 time-domain, frequency-domain, and HRV features.

---

## 📊 15-Fold Leave-One-Subject-Out (LOSO) Results

All models were evaluated using rigorous Leave-One-Subject-Out cross-validation across all 15 WESAD subjects ($S2$ through $S17$):

| Category | Model | Accuracy | F1 Score | Sensitivity | Specificity | AUC-ROC | CPU Latency | <50ms Spec |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Deep Learning** | **ST-GCN** ⭐ | **76.50% ± 16.5%** | **0.6578 ± 0.216** | **74.79% ± 26.5%** | **76.92% ± 26.5%** | **0.8665 ± 0.135** | **2.6 ms** | **PASS ✅** |
| | **1D-CNN** | 72.24% ± 18.0% | 0.5190 ± 0.299 | 59.47% ± 38.8% | 77.38% ± 26.9% | 0.8221 ± 0.182 | **1.1 ms** | **PASS ✅** |
| **Classical** | **Random Forest** | 76.60% ± 15.2% | 0.6872 ± 0.162 | 80.10% ± 21.5% | 74.84% ± 23.9% | 0.9002 ± 0.107 | N/A | — |
| | **XGBoost** | 74.59% ± 13.7% | 0.6559 ± 0.153 | 78.53% ± 22.9% | 72.69% ± 22.7% | 0.8999 ± 0.097 | N/A | — |

> **Selected Champion**: **ST-GCN** achieved the best deep learning AUC-ROC (0.8665) and F1 score (0.6578) with an ultra-low inference latency of **2.6 ms** on CPU (well below the 50 ms hard threshold for ESP32 packet streaming).

---

## 📦 Deployment & ONNX Export

The champion **ST-GCN** model has been exported to ONNX (`outputs/stgcn.onnx`, 73.8 KB) and numerically validated against PyTorch:
* **Max Absolute Output Difference**: `4.77e-06` (Tolerance: `1e-04`) -> **PASS ✅**
* **Standalone Execution**: Runs via `onnxruntime` with zero PyTorch runtime dependency.

---

## 🛠️ Quickstart

### 1. Environment Setup
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Dataset Setup
Extract the WESAD dataset (`S2` .. `S17`) into `data/raw/WESAD/` or configure the path in `config.py`.

### 3. Pipeline Execution
```bash
# Full 15-fold LOSO training (Deep Learning + Classical):
python train.py --skip-preprocessing

# Quick single-fold test (Subject S2):
python train.py --skip-preprocessing --fold 2

# Train only ST-GCN:
python train.py --skip-preprocessing --models stgcn --dl-only

# Generate Evaluation Report & Visualizations:
python reporting/report.py
```