"""
Real-Time Multimodal Stress Telemetry Server & Technical Review Gateway
Platform: FastAPI + WebSockets + ONNX Runtime (ST-GCN)
Authors: Bhavya Singh (25BCE2004), Soham Sharma (25BCE0403), Praneet Raj (25BCE2280)
Institution: School of Computer Science and Engineering, Vellore Institute of Technology

Features:
1. Full 15-Subject WESAD Clinical Benchmark Replay with Chronological Streaming
2. Bi-directional WebSocket Control: Play, Pause, Step, Seek, Speed, and Phase Jump
3. Real-Time Dynamic Physiological Coupling (Pearson Graph Adjacency Calculation)
4. Dual Mode: Auto-detects ESP32 Wearable Hardware or runs WESAD Replay
5. Built-in REST APIs for Benchmark Results, 15-Fold Table, and Static Figures
"""

import os
import sys
import time
import json
import asyncio
import threading
from collections import deque
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import serial
import serial.tools.list_ports

# Add repository root to Python path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from inference import StressDetector

app = FastAPI(title="Multimodal Stress Telemetry & Review Gateway — Chota Chatri")

# Mount outputs directory for benchmark figures
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
if os.path.exists(OUTPUTS_DIR):
    app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")

DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "wesad_windows.npz")
RESULTS_PATH = os.path.join(BASE_DIR, "outputs", "all_results.json")
HTML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")

# Initialize Champion Inference Engine (ST-GCN)
print("[GATEWAY] Initializing ST-GCN ONNX Inference Engine...")
detector = StressDetector(model_path=os.path.join(BASE_DIR, "outputs", "stgcn.onnx"))
print("[GATEWAY] ST-GCN ONNX Engine loaded successfully! (73.8 KB, ~2.57 ms CPU latency)")

# Pre-load full WESAD dataset for all 15 subjects
subjects_data = {}
all_subject_ids = []

if os.path.exists(DATA_PATH):
    print(f"[GATEWAY] Loading full WESAD benchmark from {DATA_PATH}...")
    npz = np.load(DATA_PATH)
    tensors = npz["tensors"]       # [N, 3, 384]
    labels = npz["labels"]         # [N]
    subject_ids = npz["subject_ids"]  # [N]
    
    unique_subjs = np.unique(subject_ids)
    all_subject_ids = [int(s) for s in unique_subjs]
    
    for s in unique_subjs:
        s_int = int(s)
        mask = (subject_ids == s)
        s_tensors = tensors[mask]
        s_labels = labels[mask]
        
        # Determine stress and non-stress transition indices
        stress_indices = np.where(s_labels == 1)[0]
        first_stress = int(stress_indices[0]) if len(stress_indices) > 0 else 0
        last_stress = int(stress_indices[-1]) if len(stress_indices) > 0 else len(s_labels) - 1
        
        diffs = np.where(np.diff(s_labels) != 0)[0].tolist()
        
        subjects_data[s_int] = {
            "tensors": s_tensors,
            "labels": s_labels,
            "total_windows": len(s_labels),
            "stress_count": int(np.sum(s_labels == 1)),
            "non_stress_count": int(np.sum(s_labels == 0)),
            "first_stress": first_stress,
            "last_stress": last_stress,
            "transitions": [int(d) for d in diffs]
        }
    print(f"[GATEWAY] Successfully loaded {len(tensors)} windows across {len(unique_subjs)} subjects: {all_subject_ids}")
else:
    print(f"[GATEWAY] Warning: Dataset not found at {DATA_PATH}")


# =========================================================================
# Physiological Metrics & Dynamic Graph Adjacency Extractors
# =========================================================================
def extract_ecg_metrics(ecg_samples, fs=128):
    """Detect real R-peaks and compute live Heart Rate (BPM) and R-R interval."""
    if len(ecg_samples) < 128:
        return "-- BPM", "Buffering...", False
    
    std_val = float(np.std(ecg_samples))
    if std_val < 25.0:
        return "-- BPM", "Electrode Idle", False
    
    mean_val = np.mean(ecg_samples)
    thresh = mean_val + 1.25 * std_val
    min_dist = int(0.28 * fs)  # ~36 samples min refractory
    
    peaks = []
    for i in range(1, len(ecg_samples) - 1):
        if ecg_samples[i] > thresh and ecg_samples[i] > ecg_samples[i-1] and ecg_samples[i] > ecg_samples[i+1]:
            if not peaks or (i - peaks[-1]) >= min_dist:
                peaks.append(i)
                
    if len(peaks) >= 2:
        rr_ms = np.diff(peaks) / fs * 1000.0
        mean_rr = float(np.mean(rr_ms))
        if 350 <= mean_rr <= 1500:
            hr_bpm = round(60000.0 / mean_rr)
            return f"{hr_bpm} BPM", f"{round(mean_rr)} ms", True
        else:
            return "-- BPM", f"{round(mean_rr)} ms", True
    return "-- BPM", "Detecting rhythm...", True


def extract_ppg_metrics(ppg_samples):
    """Assess real pulsatile AC amplitude and optical perfusion from MAX30102."""
    if len(ppg_samples) < 64:
        return "Buffering...", False
    
    ptp = float(np.ptp(ppg_samples))
    dc = float(np.mean(ppg_samples))
    
    if dc < 1000 or ptp < 40:
        return "Finger Off Sensor", False
    elif ptp > 120:
        return f"Live Pulse (AC: {int(ptp)})", True
    else:
        return "Weak Signal (Adjust placement)", True


def extract_temp_metrics(temp_val):
    """Evaluate DS18B20 peripheral skin temperature probe."""
    if temp_val < -40 or temp_val == -127:
        return "-- °C", "Probe Disconnected"
    elif 20.0 <= temp_val <= 42.0:
        return f"{temp_val:.2f} °C", "Skin Normothermia"
    else:
        return f"{temp_val:.2f} °C", "Ambient Reading"


def compute_pearson_correlations(window):
    """
    Compute dynamic Pearson cross-modal correlations and normalized adjacency
    for the 3-node graph (0=ECG, 1=PPG, 2=TEMP) across a 3.0s window.
    """
    # window shape: (3, 384)
    w_c = window - np.mean(window, axis=-1, keepdims=True)
    cov = np.dot(w_c, w_c.T)
    std = np.linalg.norm(w_c, axis=-1, keepdims=True) + 1e-8
    corr = cov / np.dot(std, std.T)
    
    abs_corr = np.abs(corr)
    eye = np.eye(3)
    A = abs_corr + eye
    deg = np.sum(A, axis=-1, keepdims=True) + 1e-8
    A_norm = A / deg
    
    return {
        "r_ecg_ppg": round(float(corr[0, 1]), 3),
        "r_ecg_temp": round(float(corr[0, 2]), 3),
        "r_ppg_temp": round(float(corr[1, 2]), 3),
        "adj_matrix": np.round(A_norm, 3).tolist(),
        "raw_corr_matrix": np.round(corr, 3).tolist()
    }


# =========================================================================
# Serial Telemetry Harvester (For Physical ESP32 Hardware If Attached)
# =========================================================================
class SerialTelemetryHarvester:
    def __init__(self, target_port="COM5", baudrate=115200):
        self.target_port = target_port
        self.baudrate = baudrate
        self.ecg_buf = deque(maxlen=384)
        self.ppg_buf = deque(maxlen=384)
        self.temp_buf = deque(maxlen=384)
        self.lead_status = 0
        self.is_connected = False
        self.active_port = None
        self.packets_count = 0
        self.latest_raw_ecg = 0
        self.latest_raw_ppg = 0
        self.latest_raw_temp = 0.0
        self.window_seq = 0
        self.lock = threading.Lock()
        self.running = True
        self._last_warn_time = 0
        self.worker_thread = threading.Thread(target=self._harvester_loop, daemon=True)
        self.worker_thread.start()

    def _harvester_loop(self):
        while self.running:
            ser = None
            try:
                ports = [p.device for p in serial.tools.list_ports.comports()]
                chosen_port = self.target_port if self.target_port in ports else (ports[0] if ports else None)
                if chosen_port:
                    ser = serial.Serial(chosen_port, self.baudrate, timeout=1.0)
                    self.active_port = chosen_port
                    self.is_connected = True
                    print(f"[SERIAL] Connected to ESP32 on {chosen_port}")

                    while self.running and ser.is_open:
                        raw_line = ser.readline().decode('utf-8', errors='ignore').strip()
                        if raw_line.startswith("ECG:"):
                            items = {}
                            for part in raw_line.split(","):
                                if ":" in part:
                                    k, v = part.split(":", 1)
                                    items[k.strip()] = v.strip()

                            try:
                                ecg_val = float(items.get("ECG", 2048))
                                ppg_val = float(items.get("PPG_RED", items.get("PPG_IR", 180000)))
                                temp_val = float(items.get("TEMP", -127.0))
                                lead_val = int(items.get("LEAD_STATUS", 0))

                                with self.lock:
                                    self.ecg_buf.append(ecg_val)
                                    self.ppg_buf.append(ppg_val)
                                    self.temp_buf.append(temp_val)
                                    self.lead_status = lead_val
                                    self.packets_count += 1
                                    self.latest_raw_ecg = int(ecg_val)
                                    self.latest_raw_ppg = int(ppg_val)
                                    self.latest_raw_temp = temp_val
                            except ValueError:
                                pass
                else:
                    self.is_connected = False
            except serial.SerialException:
                self.is_connected = False
            except Exception:
                self.is_connected = False
            finally:
                if ser and ser.is_open:
                    try:
                        ser.close()
                    except:
                        pass
                time.sleep(1.5)

    def is_streaming(self):
        with self.lock:
            return self.is_connected and len(self.ecg_buf) >= 128

    def get_live_data(self):
        with self.lock:
            ecg = np.array(self.ecg_buf, dtype=np.float32)
            ppg = np.array(self.ppg_buf, dtype=np.float32)
            temp = np.array(self.temp_buf, dtype=np.float32)
            lead = self.lead_status
            port = self.active_port
            pkts = self.packets_count
            raw_ecg = self.latest_raw_ecg
            raw_ppg = self.latest_raw_ppg
            raw_temp = self.latest_raw_temp
            self.window_seq += 1
            seq = self.window_seq
        return ecg, ppg, temp, lead, port, pkts, raw_ecg, raw_ppg, raw_temp, seq


harvester = SerialTelemetryHarvester(target_port="COM5", baudrate=115200)


# =========================================================================
# REST Endpoints
# =========================================================================
@app.get("/")
async def get_dashboard():
    with open(HTML_PATH, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)


@app.get("/api/status")
async def get_status():
    return {
        "status": "online",
        "data_mode": "ESP32 Live Hardware" if harvester.is_connected else "WESAD Clinical Benchmark Replay",
        "hardware_connected": harvester.is_connected,
        "hardware_port": harvester.active_port,
        "packets_ingested": harvester.packets_count,
        "model_binary": "outputs/stgcn.onnx",
        "sample_rate_hz": 128,
        "window_duration_sec": 3.0,
        "window_samples": 384,
        "subjects_available": all_subject_ids
    }


@app.get("/api/subjects")
async def get_subjects():
    metadata = {}
    for sid, sdata in subjects_data.items():
        metadata[f"S{sid}"] = {
            "id": sid,
            "total_windows": sdata["total_windows"],
            "stress_count": sdata["stress_count"],
            "non_stress_count": sdata["non_stress_count"],
            "first_stress": sdata["first_stress"],
            "last_stress": sdata["last_stress"],
            "transitions": sdata["transitions"],
            "duration_minutes": round(sdata["total_windows"] * 1.5 / 60, 1)
        }
    return JSONResponse(content={"subjects": metadata, "default_subject": 2})


@app.get("/api/benchmarks")
async def get_benchmarks():
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return JSONResponse(content=data)
    return JSONResponse(content={"error": "Benchmark results not found"})


@app.get("/api/model_info")
async def get_model_info():
    return {
        "architecture": "Spatio-Temporal Graph Convolutional Network (ST-GCN)",
        "graph_nodes": ["AD8232 ECG", "MAX30102 PPG", "DS18B20 Skin Temperature"],
        "node_features": 1,
        "spatial_layers": "2x GCNConv (Hidden Dim = 32)",
        "temporal_layers": "1D TCN (Channels = 64, Kernel = 7)",
        "pooling": "Global Node Mean Pooling",
        "parameters": 6466,
        "model_size_bytes": 73812,
        "model_size_kb": 73.8,
        "cpu_latency_mean_ms": 2.57,
        "edge_latency_budget_ms": 50.0,
        "deployment_format": "ONNX Runtime (CPUExecutionProvider)",
        "peak_ram_mb": 1.2
    }


# =========================================================================
# Bi-directional WebSocket Telemetry Stream
# =========================================================================
@app.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket):
    await websocket.accept()
    print("[WS] Web client connected to Multimodal Telemetry Gateway.")
    
    # Session state
    current_subject = 2
    window_idx = 0
    is_playing = True
    speed_mult = 1.0
    step_requested = False
    
    # Receiver task for client controls
    async def client_listener():
        nonlocal current_subject, window_idx, is_playing, speed_mult, step_requested
        try:
            while True:
                msg_text = await websocket.receive_text()
                cmd = json.loads(msg_text)
                action = cmd.get("action")
                
                if action == "play":
                    is_playing = True
                elif action == "pause":
                    is_playing = False
                elif action == "toggle_play":
                    is_playing = not is_playing
                elif action == "step":
                    is_playing = False
                    step_requested = True
                elif action == "step_back":
                    is_playing = False
                    window_idx = max(0, window_idx - 2)
                    step_requested = True
                elif action == "reset":
                    window_idx = 0
                elif action == "seek":
                    idx = int(cmd.get("index", 0))
                    total = subjects_data[current_subject]["total_windows"] if current_subject in subjects_data else 1000
                    window_idx = max(0, min(idx, total - 1))
                elif action == "set_subject":
                    new_subj = int(cmd.get("subject", 2))
                    if new_subj in subjects_data:
                        current_subject = new_subj
                        window_idx = 0
                elif action == "set_speed":
                    speed_mult = float(cmd.get("speed", 1.0))
                elif action == "jump_phase":
                    phase = cmd.get("phase")
                    if current_subject in subjects_data:
                        s_info = subjects_data[current_subject]
                        if phase == "baseline":
                            window_idx = 50  # Stable baseline window
                        elif phase == "stress":
                            window_idx = max(0, s_info["first_stress"] - 5)  # 5 windows before stress onset
                        elif phase == "recovery":
                            window_idx = min(s_info["last_stress"] + 2, s_info["total_windows"] - 1)
        except WebSocketDisconnect:
            pass
        except Exception as e:
            print(f"[WS Listener] Error: {e}")

    listener_task = asyncio.create_task(client_listener())

    try:
        while True:
            # Mode 1: Physical ESP32 Hardware (if attached)
            if harvester.is_streaming():
                ecg, ppg, temp, lead_status, port, pkts, raw_ecg, raw_ppg, raw_temp, seq = harvester.get_live_data()
                
                live_hr, live_rr, ecg_ok = extract_ecg_metrics(ecg)
                ppg_status, ppg_ok = extract_ppg_metrics(ppg)
                temp_str, temp_desc = extract_temp_metrics(raw_temp)

                if len(ecg) < 384:
                    pad_len = 384 - len(ecg)
                    ecg_in = np.pad(ecg, (pad_len, 0), mode='edge')
                    ppg_in = np.pad(ppg, (pad_len, 0), mode='edge')
                    temp_in = np.pad(temp, (pad_len, 0), mode='edge')
                else:
                    ecg_in = ecg[-384:]
                    ppg_in = ppg[-384:]
                    temp_in = temp[-384:]

                raw_window = np.stack([ecg_in, ppg_in, temp_in])  # [3, 384]
                corrs = compute_pearson_correlations(raw_window)

                if lead_status == 1:
                    prediction = "Leads Detached"
                    stress_prob = 0.0
                    latency = 0.5
                else:
                    res = detector.predict(raw_window, is_preprocessed=False)
                    prediction = res["label"]
                    stress_prob = round(res["stress_probability"] * 100, 1)
                    latency = round(res["latency_ms"], 2)

                step = max(1, len(ecg) // 96)
                ecg_display = ecg[::step][-96:].tolist()
                ppg_display = ppg[::step][-96:].tolist()
                temp_display = temp[::step][-96:].tolist()

                payload = {
                    "seq": seq,
                    "total_windows": "∞ (Live Stream)",
                    "subject_id": "ESP32",
                    "is_live_hardware": True,
                    "is_playing": True,
                    "playback_speed": 1.0,
                    "stream_mode": f"Physical Wearable Hardware ({port})",
                    "source": f"ESP32 Serial Stream ({port})",
                    "ground_truth": "Live Human Physiological Feed",
                    "phase": "Live Physical Acquisition",
                    "prediction": prediction,
                    "stress_probability": stress_prob,
                    "latency_ms": latency,
                    "live_hr": live_hr,
                    "live_rr": live_rr,
                    "ppg_status": ppg_status,
                    "current_temp_c": temp_str,
                    "temp_desc": temp_desc,
                    "packets_count": pkts,
                    "raw_ecg": raw_ecg,
                    "raw_ppg": raw_ppg,
                    "raw_temp": round(raw_temp, 2) if raw_temp > -40 else "--",
                    "lead_status": "CONNECTED" if lead_status == 0 else "DETACHED",
                    "ecg_points": ecg_display,
                    "ppg_points": ppg_display,
                    "temp_points": temp_display,
                    "correlations": corrs,
                    "timestamp": time.strftime("%H:%M:%S")
                }
                await websocket.send_text(json.dumps(payload))
                await asyncio.sleep(1.0)

            # Mode 2: WESAD Clinical Benchmark Replay
            else:
                if current_subject in subjects_data and subjects_data[current_subject]["total_windows"] > 0:
                    s_data = subjects_data[current_subject]
                    total_w = s_data["total_windows"]
                    
                    if not is_playing and not step_requested:
                        # Idle loop while paused
                        await asyncio.sleep(0.1)
                        continue
                        
                    step_requested = False
                    
                    cur_idx = window_idx % total_w
                    raw_window = s_data["tensors"][cur_idx]  # [3, 384]
                    true_label = int(s_data["labels"][cur_idx])
                    
                    # Phase determination
                    if cur_idx < s_data["first_stress"]:
                        phase = "Baseline / Amusement (Calm)"
                    elif s_data["first_stress"] <= cur_idx <= s_data["last_stress"]:
                        phase = "TSST Acute Stress Protocol"
                    else:
                        phase = "Meditation / Recovery"

                    ground_truth = "Acute Stress (TSST)" if true_label == 1 else "Non-Stress (Baseline/Amuse)"
                    
                    ecg = raw_window[0]
                    ppg = raw_window[1]
                    temp = raw_window[2]
                    raw_temp = float(np.mean(temp))

                    live_hr, live_rr, _ = extract_ecg_metrics(ecg)
                    ppg_status, _ = extract_ppg_metrics(ppg)
                    temp_str, temp_desc = extract_temp_metrics(raw_temp)

                    # Dynamic Pearson cross-modal graph adjacency
                    corrs = compute_pearson_correlations(raw_window)

                    # ONNX inference on benchmark window
                    res = detector.predict(raw_window, is_preprocessed=True)
                    prediction = res["label"]
                    stress_prob = round(res["stress_probability"] * 100, 1)
                    latency = round(res["latency_ms"], 2)

                    step = 4
                    ecg_display = ecg[::step].tolist()
                    ppg_display = ppg[::step].tolist()
                    temp_display = temp[::step].tolist()

                    payload = {
                        "seq": cur_idx + 1,
                        "window_index": cur_idx,
                        "total_windows": total_w,
                        "subject_id": f"S{current_subject}",
                        "is_live_hardware": False,
                        "is_playing": is_playing,
                        "playback_speed": speed_mult,
                        "stream_mode": f"WESAD Clinical Replay (Subject S{current_subject})",
                        "source": f"WESAD Dataset &bull; Subject S{current_subject} ({total_w} Windows @ 128 Hz)",
                        "ground_truth": ground_truth,
                        "ground_truth_label": true_label,
                        "phase": phase,
                        "phase_transitions": s_data["transitions"],
                        "first_stress": s_data["first_stress"],
                        "last_stress": s_data["last_stress"],
                        "prediction": prediction,
                        "predicted_label": res["prediction"],
                        "is_match": (res["prediction"] == true_label),
                        "stress_probability": stress_prob,
                        "non_stress_probability": round(100.0 - stress_prob, 1),
                        "latency_ms": latency,
                        "live_hr": live_hr,
                        "live_rr": live_rr,
                        "ppg_status": ppg_status,
                        "current_temp_c": temp_str,
                        "temp_desc": temp_desc,
                        "packets_count": (cur_idx + 1) * 384,
                        "raw_ecg": int(ecg[-1]),
                        "raw_ppg": int(ppg[-1]),
                        "raw_temp": round(raw_temp, 2),
                        "lead_status": "CONNECTED",
                        "ecg_points": ecg_display,
                        "ppg_points": ppg_display,
                        "temp_points": temp_display,
                        "correlations": corrs,
                        "timestamp": time.strftime("%H:%M:%S")
                    }

                    if is_playing:
                        window_idx = (window_idx + 1) % total_w

                    await websocket.send_text(json.dumps(payload))
                    
                    # Sleep duration scaled by playback speed (Base ~1.2s per 3-sec window)
                    sleep_time = max(0.08, 1.2 / speed_mult)
                    await asyncio.sleep(sleep_time)
                else:
                    await asyncio.sleep(1.0)

    except WebSocketDisconnect:
        print("[WS] Client disconnected.")
    except Exception as e:
        print(f"[WS] Streaming loop error: {e}")
    finally:
        listener_task.cancel()


if __name__ == "__main__":
    uvicorn.run("dashboard.server:app", host="127.0.0.1", port=8000, reload=False)
