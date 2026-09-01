#!/bin/bash
# 1. Start FastAPI backend on internal port 8000
python -m uvicorn app:app --host 127.0.0.1 --port 8000 &

# 2. Wait 2 seconds for FastAPI to initialize
sleep 2

# 3. Start Streamlit frontend on Render assigned public $PORT
python -m streamlit run ui/streamlit_app.py --server.port $PORT --server.address 0.0.0.0
