"""
One-Click Launcher for Multimodal Stress Telemetry Dashboard
Runs the FastAPI + WebSocket backend and opens the browser interface.

Usage:
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
    print(f"\n=======================================================")
    print(f" [OK] Dashboard active at: {url}")
    print(f" [OK] Opening in your web browser...")
    print(f" Press Ctrl+C in this terminal to terminate the server.")
    print(f"=======================================================\n")
    webbrowser.open(url)

if __name__ == "__main__":
    # Ensure current directory is in Python path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    # Launch browser thread
    threading.Thread(target=open_browser, daemon=True).start()
    
    # Start FastAPI / WebSocket server
    uvicorn.run("dashboard.server:app", host="127.0.0.1", port=8000, reload=False, log_level="info")
