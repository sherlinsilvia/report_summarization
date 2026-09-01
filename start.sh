#!/usr/bin/env bash
set -e

# 1. Start FastAPI backend on internal port 8000 in the background
echo "--- Starting FastAPI backend (Uvicorn) on 127.0.0.1:8000 ---"
python -m uvicorn app:app --host 127.0.0.1 --port 8000 &
FASTAPI_PID=$!

# 2. Wait for FastAPI to become healthy
echo "--- Waiting for FastAPI backend to become ready ---"
for i in {1..15}; do
  if python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=1)" >/dev/null 2>&1; then
    echo "--- FastAPI backend is healthy and responding! ---"
    break
  fi
  echo "Waiting for FastAPI... ($i/15)"
  sleep 1
done

# 3. Determine Port for Streamlit (Render sets $PORT dynamically)
STREAMLIT_PORT="${PORT:-10000}"
echo "--- Starting Streamlit frontend on 0.0.0.0:${STREAMLIT_PORT} ---"

# 4. Start Streamlit in the foreground bound to Render's dynamic $PORT
exec python -m streamlit run streamlit_app.py \
  --server.port "${STREAMLIT_PORT}" \
  --server.address 0.0.0.0 \
  --server.headless true \
  --server.enableCORS false \
  --server.enableXsrfProtection false \
  --browser.gatherUsageStats false
