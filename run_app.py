"""
TrustMed Cloud Deployment Launcher
Starts both FastAPI on internal port 8000 and Streamlit on Render's dynamic $PORT.
Cross-platform, immune to CRLF line ending issues on Linux containers.
"""

import os
import sys
import subprocess
import time

def main():
    port = os.environ.get("PORT", "10000")
    print(f"==================================================", flush=True)
    print(f"  TrustMed Starting on Render (PORT: {port})", flush=True)
    print(f"==================================================", flush=True)

    # 1. Start FastAPI backend (uvicorn) on 127.0.0.1:8000
    print("[1/2] Starting FastAPI backend on 127.0.0.1:8000...", flush=True)
    backend_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", "8000"],
        stdout=sys.stdout,
        stderr=sys.stderr
    )

    # 2. Allow backend 2 seconds to initialize
    time.sleep(2)

    # 3. Start Streamlit on 0.0.0.0:$PORT
    print(f"[2/2] Starting Streamlit frontend on 0.0.0.0:{port}...", flush=True)
    streamlit_cmd = [
        sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
        "--server.port", str(port),
        "--server.address", "0.0.0.0",
        "--server.headless", "true",
        "--server.enableCORS", "false",
        "--server.enableXsrfProtection", "false",
        "--browser.gatherUsageStats", "false"
    ]

    try:
        subprocess.run(streamlit_cmd, check=True)
    except KeyboardInterrupt:
        print("Stopping services...", flush=True)
    finally:
        if backend_proc.poll() is None:
            backend_proc.terminate()

if __name__ == "__main__":
    main()
