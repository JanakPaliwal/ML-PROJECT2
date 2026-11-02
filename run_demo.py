"""
run_demo.py
Single-command launcher for the Mandi Price Forecasting & Early-Warning Intelligence System.
Serves the web dashboard locally and displays research summary metrics.
"""

import os
import sys
import webbrowser
import http.server
import socketserver
import threading
import time

PORT = 8080
WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")

def start_server(port=PORT):
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=WEB_DIR, **kwargs)
        
        def log_message(self, format, *args):
            # Suppress noisy HTTP request logs
            pass

    # Find open port if 8080 is occupied
    actual_port = port
    for p in range(port, port + 10):
        try:
            httpd = socketserver.TCPServer(("", p), Handler)
            actual_port = p
            break
        except OSError:
            continue
    
    server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    server_thread.start()
    return actual_port

def main():
    print("=" * 75)
    print("  AGRIMANDI AI: SHORT-TERM MANDI PRICE FORECASTING FOR ONION & TOMATO")
    print("  Targeting SDG 2 (Zero Hunger) & SDG 12 (Responsible Consumption & Production)")
    print("=" * 75)
    
    data_json = os.path.join(WEB_DIR, "forecast_data.json")
    if not os.path.exists(data_json):
        print("\n[!] Pre-computed forecast_data.json not found. Running training pipeline first...")
        from src.train_pipeline import run_pipeline
        run_pipeline()
    else:
        print("\n[OK] Pre-computed forecast_data.json verified.")

    port = start_server(PORT)
    url = f"http://localhost:{port}"

    print(f"\n[OK] Web Dashboard server started at: {url}")
    print(f"[OK] Serving files from: {WEB_DIR}")
    print(f"[OK] Research Manuscript: file:///{os.path.join(os.path.dirname(os.path.abspath(__file__)), 'paper', 'manuscript_short_communication.md')}")
    print("\nOpening web browser...")
    webbrowser.open(url)

    print("\nPress Ctrl+C to terminate the dashboard server.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nDashboard server terminated. Thank you for using AgriMandi AI!")

if __name__ == "__main__":
    main()
