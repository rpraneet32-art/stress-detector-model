"""
run_demo.py — One-Click Technical Review Launcher & Replay Demonstration Suite.
Starts the FastAPI + WebSocket backend and launches the browser review interface.

Usage:
    python run_demo.py
    python run_dashboard.py
"""

import sys
import os
import webbrowser
import threading
import time
import uvicorn

def open_browser():
    time.sleep(1.2)
    url = "http://127.0.0.1:8000"
    print(f"\n" + "=" * 65)
    print(f" [OK] Wearable Stress Review Dashboard active at: {url}")
    print(f" [OK] Opening in your default web browser...")
    print(f" [OK] Ready for live faculty demonstration!")
    print(f" Press Ctrl+C in this terminal to shutdown.")
    print("=" * 65 + "\n")
    webbrowser.open(url)

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    threading.Thread(target=open_browser, daemon=True).start()
    uvicorn.run("dashboard.server:app", host="127.0.0.1", port=8000, reload=False, log_level="info")
