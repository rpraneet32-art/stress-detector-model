# Multimodal Stress Detection — Model & Training Pipeline
## Agent Execution Specification

**Project:** AI-Driven Multimodal Wearable Stress Monitoring & Telemedicine Platform
**Scope of this document:** Software/model track only — data pipeline through trained, benchmarked, exported models. No hardware, firmware, or dashboard work is in scope here.
**Target hardware context (for later phases, informational only):** ESP32 + AD8232 (ECG) + MAX30102 (PPG/SpO2) + DS18B20 (skin temp) + client webcam. GSR is explicitly out of scope for this phase.

---

## 0. How to Use This Document

This document is written to be handed to a coding agent as a task specification. Section 1–9 are fully agent-executable — the agent should be able to go from an empty repository to trained, benchmarked, exported models by following them in order, without needing further design decisions from the human.

**Section 10 is the exception** — it lists everything that requires the human (Praneet) to act, because it needs credentials, manual downloads, physical judgment calls, or an account the agent cannot access. The agent should stop and request these inputs when it reaches a dependency on Section 10, rather than attempting to work around them.

---

## 1. Environment Setup

**Directive:** Set up a reproducible Python environment before writing any pipeline code.

- Python 3.10+
- Core libraries: `torch`, `torchvision`, `torch-geometric` (for ST-GCN), `scikit-learn`, `xgboost`, `numpy`, `scipy`, `pandas`
- Signal processing: `scipy.signal` (filtering), `PyWavelets` (if wavelet features are added later)
- Utilities: `onnx`, `onnxruntime` (for export/validation), `matplotlib` (diagnostic plots), `tqdm`
- Experiment tracking: a simple structured logger (CSV or JSON lines) recording, per run: config hash, random seed, fold number, all metrics, checkpoint path. Do not skip this — every number in the final report must be traceable to a specific run.
- Pin all dependency versions in `requirements.txt` or `environment.yml`. Fix random seeds (Python, NumPy, PyTorch) at the top of every entry-point script.
- Set up version control (Git) with a `.gitignore` excluding raw data and large checkpoints.

**Deliverable:** A working environment where `import torch, torch_geometric, xgboost, onnx` succeeds without error, and a Git repository initialized with the environment file committed.

---

## 2. Dataset: WESAD

**Directive:** Build the data pipeline against the WESAD (Wearable Stress and Affect Detection) dataset. Do not substitute another dataset without explicit instruction — the channel mapping in Section 3 is specific to WESAD's file structure.

**Note:** Acquiring the raw WESAD archive requires a manual step by the human — see Section 10.1. The agent should build the loader assuming the raw `.pkl` subject archives will be placed at a path such as `data/raw/WESAD/S{n}/S{n}.pkl` for each subject, and should not attempt to download the dataset itself.

WESAD contains 15 subjects, each with chest-worn (RespiBAN) and wrist-worn (Empatica E4) recordings, labeled with a `label` channel marking experimental condition per timestep.

---

## 3. Channel Extraction & Alignment

**Directive:** Extract exactly three physiological channels per subject, matching the physical sensor suite this project targets. Ignore all other WESAD channels (respiration, EDA, full ACC, etc.) — they are out of scope for this phase.

| Physical Device (target hardware) | WESAD Source Channel | Original Sampling Rate | Resample Target | Resampling Method |
|---|---|---|---|---|
| AD8232 ECG | `chest['ECG']` | 700 Hz | 128 Hz | Decimation / anti-aliased downsampling |
| MAX30102 PPG | `wrist['BVP']` | 64 Hz | 128 Hz | Cubic spline interpolation (upsampling) |
| DS18B20 Temp | `wrist['TEMP']` or `chest['Temp']` | 4 Hz / 700 Hz | 128 Hz | Linear interpolation |
| Ground truth | `label` | 700 Hz | 128 Hz | Mode aggregation per window (majority vote over resampled label at 128 Hz within each window) |

**Directive:** All three channels and the label must end up temporally aligned on a common 128 Hz timebase per subject before windowing.

### Label mapping (binary classification)

- **Class 0 — Non-Stress:** WESAD condition `1` (Baseline) **and** condition `3` (Amusement), merged into a single class.
- **Class 1 — Acute Stress:** WESAD condition `2` (TSST / stress induction protocol).
- Discard all other WESAD condition codes (transient/meditation segments, etc.) from the training and evaluation set.
- Document the class balance (sample count per class, per subject) as a diagnostic output — this feeds directly into the weighted loss in Section 6.

**Deliverable:** A per-subject loader function that returns a `(T, 3)` array (ECG, PPG, Temp at 128 Hz) and an aligned `(T,)` binary label array, plus a script that runs this across all 15 subjects and reports channel statistics (min/max/mean/std, missing-data gaps) and class balance per subject as a sanity-check artifact (e.g. a markdown or CSV summary table) before proceeding.

---

## 4. Signal Preprocessing

**Directive:** Apply the following filters in this exact order, per channel, per subject, before windowing:

1. **Powerline notch filter:** IIR notch at 50 Hz, Q = 30, applied to all channels.
2. **Channel-specific bandpass:**
   - ECG: 4th-order Butterworth bandpass, 0.5–40 Hz.
   - PPG: 3rd-order Butterworth bandpass, 0.5–5.0 Hz.
   - Temperature: no bandpass (slow-varying signal); notch only.
3. **Subject-wise z-score normalization:** `(x - μ_subject) / σ_subject`, computed independently per subject per channel, using statistics from that subject's own recording only (never global statistics, to avoid leaking inter-subject scale differences into the model).

**Deliverable:** A preprocessing module that takes the raw aligned `(T, 3)` array from Section 3 and returns a filtered, normalized `(T, 3)` array. Include a diagnostic plot (raw vs. filtered) for at least one subject, saved as an artifact for human review (see Section 10.4).

---

## 5. Windowing / Segmentation

**Directive:**
- Window length: 3.0 seconds = 384 samples at 128 Hz.
- Overlap: 50% → 1.5 second step size = 192 samples.
- Output tensor shape per window: `(3, 384)` (channels-first) for deep learning models.
- For classical baselines, additionally extract the 36-feature statistical/frequency vector described in Section 5.1 per window instead of the raw tensor.
- Label per window: majority-vote label of the underlying 128 Hz label sequence within that window; discard windows that straddle a condition boundary with no clear majority (>50%) if such windows appear.
- Perform windowing **per subject**, and keep a subject-ID tag on every window — this is required for the Leave-One-Subject-Out split in Section 7 and must never be discarded before that split is made.

### 5.1 Classical feature set (for RF/SVM/XGBoost baselines)

Extract per window:
- **ECG (time & frequency domain):** Mean RR, SDNN, RMSSD, pNN50, VLF power (0.0033–0.04 Hz), LF power (0.04–0.15 Hz), HF power (0.15–0.4 Hz), LF/HF ratio.
- **PPG (morphological):** Systolic peak amplitude, pulse interval variance, peak-to-peak interval standard deviation, SpO2 baseline mean (if derivable from the channel; otherwise omit and document the omission).
- **Temperature:** Window mean, window variance, linear slope (`dT/dt`) over the window.

Total: 36 features per window (verify exact count once implemented; document any deviation).

**Deliverable:** A windowing module producing two parallel datasets — raw tensors `(N, 3, 384)` with subject IDs and labels for deep learning, and feature vectors `(N, 36)` with the same subject IDs and labels for classical baselines. Both must be saved to disk (e.g. as `.npz` or `.pt`) so training scripts don't repeat preprocessing on every run.

---

## 6. Deep Learning Architectures

Implement all four architectures below exactly as specified — these definitions are fixed, not starting points for agent-driven redesign. If a spec is ambiguous or fails to run, flag it rather than silently substituting a different architecture.

### 6.1 1D-CNN

```
Conv1D(in=3, out=32, k=7, pad=3) → BatchNorm1d → ReLU → MaxPool1d(2)
Conv1D(32, 64, k=5, pad=2)       → BatchNorm1d → ReLU → MaxPool1d(2)
Conv1D(64, 128, k=3, pad=1)      → BatchNorm1d → ReLU → AdaptiveAvgPool1d(1)
Linear(128 → 32) → Dropout(0.3) → Linear(32 → 2)
```

### 6.2 Bi-LSTM

```
Input transposed to (B, 384, 3)
Bi-LSTM(input_size=3, hidden_size=64, num_layers=2, dropout=0.3, batch_first=True)
Concatenate final forward + backward hidden states → (B, 128)
Linear(128 → 64) → ReLU → Linear(64 → 2)
```

### 6.3 Temporal Transformer

```
Linear(3 → 64)  [projects channel dim to d_model=64]
Add learnable/sinusoidal positional encoding, shape (384, 64)
3× TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128, dropout=0.1, norm_first=True)
Global average pool over the temporal dimension → (B, 64)
Linear(64 → 2)
```

### 6.4 ST-GCN (Spatio-Temporal Graph Convolutional Network)

```
Graph nodes (V=3): Node 0 = ECG, Node 1 = PPG, Node 2 = Skin Temperature
Adjacency matrix A ∈ R^(3×3): dynamic Pearson correlation between the 3 channels,
    computed fresh per 3-second window (not fixed/learned globally)
GCNConv layer(s): propagate features across the correlation-weighted graph
1D temporal convolution over node embeddings across the time axis
Global mean pooling over nodes → Linear → Softmax(2)
```

**Deliverable:** Four model classes, each independently unit-testable (forward pass on a dummy `(B, 3, 384)` batch produces `(B, 2)` logits without shape errors), committed as separate modules (e.g. `models/cnn.py`, `models/bilstm.py`, `models/transformer.py`, `models/stgcn.py`).

---

## 7. Classical Baselines

**Directive:** Train Random Forest, SVM, and XGBoost on the 36-feature vectors from Section 5.1, under the identical LOSO split used for the deep learning models (Section 8). These are not a discarded step — every deep learning result must be reported alongside these baselines in the final comparison, and any deep learning model that fails to beat the strongest baseline should be flagged explicitly rather than omitted from the report.

Use standard scikit-learn/XGBoost defaults as a starting point, then apply a light hyperparameter search (grid or random search) within a fixed, documented budget (see Section 10.6 for approval of that budget) — do not hand-tune indefinitely chasing marginal gains.

---

## 8. Training & Validation Protocol

**Directive — this protocol is identical for all four deep learning architectures and both classical and deep models, to keep comparisons fair:**

- **Cross-validation:** Leave-One-Subject-Out (LOSO) across all 15 WESAD subjects. For each fold *k* ∈ [1, 15]: subject *k* is held out entirely as the test set; the remaining 14 subjects form the training/validation pool (further split, e.g. 80/20, for validation within that pool).
- **Loss function:** Weighted Cross-Entropy, weights inversely proportional to class frequency (computed from the class balance diagnostic in Section 3) to correct for the baseline/amusement vs. stress imbalance.
- **Optimizer:** AdamW, learning rate = 1e-3, β1 = 0.9, β2 = 0.999, weight decay = 1e-2.
- **Schedule:** Cosine annealing learning rate schedule.
- **Early stopping:** On validation loss, with a documented patience (e.g. 10–15 epochs) to prevent both overfitting and premature stopping.
- **Per-run logging:** Every fold of every model must log training/validation loss curves, final test-fold metrics, wall-clock training time, and the exact config/seed used (per Section 1).

**Deliverable:** 15 fold results × 4 deep learning models × baselines, all logged in the structured format from Section 1, with checkpoints saved per fold (or at minimum the best fold + a representative mid-performing fold, if storage is a constraint — document whichever policy is used).

---

## 9. Evaluation, Reporting & Export

### 9.1 Metrics (report per model, averaged across the 15 LOSO folds, with variance/std across folds shown — not just the mean)

- Accuracy
- Balanced F1-score
- Sensitivity (Recall)
- Specificity
- AUC-ROC
- Inference latency per window on CPU (target: **< 50 ms** — this is a hard target from the spec, not aspirational; report pass/fail explicitly against it)

### 9.2 Consolidated comparison

Produce one summary table: all four deep learning models + RF + SVM + XGBoost, all metrics, side by side. Include a short written note on which architecture is the strongest candidate to carry forward to hardware integration, and why (e.g., accuracy vs. latency trade-off) — this informs the human decision in Section 10.7.

### 9.3 Model export

- Export the best-performing model (by the criteria settled in 9.2) to **ONNX**.
- Validate the ONNX export by running inference on a held-out batch through both the original PyTorch model and the exported ONNX model, and confirming outputs match within numerical tolerance (e.g. `1e-4`).
- This ONNX artifact is the handoff point to the hardware/firmware phase — it must run with `onnxruntime` alone, with no PyTorch dependency, since that is what will eventually receive real-time ESP32 sensor packets.

**Deliverable:** A final written report (markdown or notebook) containing: the comparison table, per-model loss/accuracy curves, confusion matrices per model (aggregated across folds), the latency benchmark, the exported ONNX file, and the ONNX-vs-PyTorch validation result.

---

## 10. Manual Steps — Human-Only (Praneet)

These cannot be completed by the agent. The agent should treat each of these as a hard dependency and pause/request the corresponding input when it reaches that point in the pipeline, rather than attempting a workaround.

### 10.1 WESAD dataset acquisition
The raw WESAD archive is hosted by the University of Siegen and is not available via a simple programmatic download (`pip install`, `wget` of an open URL, etc.) — it typically requires visiting the dataset's request/download page directly. **Action:** download the WESAD `.zip` (all 15 subject archives) and place the extracted `S{n}/S{n}.pkl` files at the path the agent's loader expects (Section 2). Confirm total size and subject count (15) after extraction before telling the agent to proceed.

### 10.2 Compute environment access
The agent can write training code but cannot create accounts or authenticate. **Action:** set up a GPU-backed environment (Google Colab Pro, Kaggle Notebooks, or a local/cloud GPU box), and if using Drive/cloud storage for checkpoints, handle the authentication/mount step yourself before handing control back to the agent.

### 10.3 Storage for checkpoints and datasets
Raw WESAD data and trained checkpoints are large and should not go into Git. **Action:** decide where these live (Google Drive, a cloud bucket, local disk) and confirm the path convention with the agent so its save/load code points to the right place.

### 10.4 Reviewing diagnostic artifacts
The agent will produce sanity-check outputs (raw-vs-filtered signal plots, class balance tables, per-subject channel statistics). **Action:** actually look at these before greenlighting full training — a silently corrupted channel (e.g. a flat-lined temperature sensor, a mislabeled subject) is a judgment call a human needs to make, not something the agent can self-certify.

### 10.5 Reproducibility spot-check
**Action:** once the pipeline runs end-to-end, rerun at least one fold of one model yourself (or have the agent rerun it in a fresh session) and confirm you get matching results within a small tolerance. This is your check that the pipeline is genuinely reproducible and not accidentally dependent on session state.

### 10.6 Hyperparameter search budget approval
The classical baselines (Section 7) and potentially the deep models can be tuned indefinitely for marginal gains. **Action:** set an explicit budget (e.g. "no more than N search trials per model" or "no more than X GPU-hours total") before training starts, so the agent has a clear stopping condition instead of open-ended tuning.

### 10.7 Final model selection sign-off
Section 9.2 asks the agent to recommend a model to carry forward to hardware integration. **Action:** review that recommendation and make the actual call yourself — this decision affects the ONNX export target and everything downstream in the hardware phase, so it should be an explicit human decision, not something the agent finalizes unilaterally.

### 10.8 License / data-use compliance check
WESAD has its own usage terms (typically research/non-commercial use, with attribution). **Action:** confirm your use case (coursework/project) complies with those terms before any results are shared or published outside the team.

---

## Summary Checklist

| # | Task | Owner |
|---|---|---|
| 1 | Environment & repo setup | Agent |
| 2 | WESAD raw data download | **Human (10.1)** |
| 3 | Channel extraction, alignment, label mapping | Agent |
| 4 | Signal preprocessing (notch, bandpass, normalization) | Agent |
| 5 | Windowing + classical feature extraction | Agent |
| 6 | Implement 4 deep learning architectures | Agent |
| 7 | Implement classical baselines (RF/SVM/XGBoost) | Agent |
| 8 | LOSO training & validation, all models | Agent |
| 9 | Evaluation, comparison report, ONNX export | Agent |
| — | GPU/cloud environment setup | **Human (10.2)** |
| — | Checkpoint/data storage decision | **Human (10.3)** |
| — | Review diagnostic artifacts before full training | **Human (10.4)** |
| — | Reproducibility spot-check | **Human (10.5)** |
| — | Approve hyperparameter search budget | **Human (10.6)** |
| — | Final model sign-off for hardware handoff | **Human (10.7)** |
| — | Dataset license/compliance check | **Human (10.8)** |
