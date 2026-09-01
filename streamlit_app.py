"""
Root Streamlit Entrypoint for Cloud Deployments (Streamlit Community Cloud, HuggingFace Spaces, etc.)
Redirects directly to ui/streamlit_app.py
"""
import os
import runpy

current_dir = os.path.dirname(os.path.abspath(__file__))
ui_app_path = os.path.join(current_dir, "ui", "streamlit_app.py")

if __name__ == "__main__" or True:
    runpy.run_path(ui_app_path, run_name="__main__")
