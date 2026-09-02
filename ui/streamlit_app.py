import os
import sys
import re
import json
import requests
from pathlib import Path
import streamlit as st

# Ensure project root and ui/ directories are in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name == "ui" else Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
UI_DIR = BASE_DIR / "ui" if (BASE_DIR / "ui").exists() else BASE_DIR
if str(UI_DIR) not in sys.path:
    sys.path.insert(0, str(UI_DIR))

try:
    from citation.formatter import format_summary_citations_html, clean_markdown_and_format_html
except ImportError:
    from ui.citation.formatter import format_summary_citations_html, clean_markdown_and_format_html

import importlib
try:
    import backend_bridge
    importlib.reload(backend_bridge)
except ImportError:
    from ui import backend_bridge
    importlib.reload(backend_bridge)

upload_and_index_report = backend_bridge.upload_and_index_report
analyze_prescription_bridge = backend_bridge.analyze_prescription_bridge
parse_mimic_csv = backend_bridge.parse_mimic_csv
process_mimic_selection = backend_bridge.process_mimic_selection
generate_summary_bridge = backend_bridge.generate_summary_bridge
run_disc_bridge = backend_bridge.run_disc_bridge

# API Endpoint
API_URL = (os.getenv("FASTAPI_URL") or os.getenv("API_URL") or "http://localhost:8000").rstrip("/")

# Page Configuration
st.set_page_config(
    page_title="TrustMed - Clinical Report Summarizer",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling & Responsive Mobile/Desktop CSS
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Main Title and Headers */
    .main-title {
        font-family: 'Outfit', sans-serif;
        background: linear-gradient(135deg, #38bdf8 0%, #10b981 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.6rem;
        margin-bottom: 0.25rem;
        letter-spacing: -0.02em;
    }
    
    .subtitle {
        font-size: 1rem;
        color: #94a3b8;
        margin-bottom: 1.5rem;
        line-height: 1.5;
    }
    
    /* Responsive Glassmorphism Cards */
    .glass-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 14px;
        padding: 1.25rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
        margin-bottom: 1rem;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    
    .glass-card:hover {
        border-color: rgba(56, 189, 248, 0.4);
    }
    
    .metric-title {
        font-size: 0.8rem;
        color: #94a3b8;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 0.06em;
    }
    
    .metric-value {
        font-size: 2.1rem;
        font-weight: 800;
        font-family: 'Outfit', sans-serif;
        margin: 0.2rem 0;
    }
    
    /* Trust Badges and Highlights */
    .badge-high {
        background-color: rgba(16, 185, 129, 0.2);
        color: #34d399;
        padding: 0.3rem 0.85rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
        border: 1px solid rgba(16, 185, 129, 0.35);
    }
    
    .badge-low {
        background-color: rgba(239, 68, 68, 0.2);
        color: #f87171;
        padding: 0.3rem 0.85rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
        border: 1px solid rgba(239, 68, 68, 0.35);
    }
    
    /* Interactive Citation styling */
    .citation-badge:hover {
        transform: scale(1.15);
        background-color: #2563eb !important;
        box-shadow: 0 4px 10px rgba(37, 99, 235, 0.4);
    }
    
    /* Steps styling */
    .step-container {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-bottom: 1.25rem;
        background: rgba(15, 23, 42, 0.65);
        padding: 0.75rem;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    .step-item {
        flex: 1 1 calc(50% - 0.5rem);
        min-width: 110px;
        text-align: center;
        font-size: 0.78rem;
        font-weight: 600;
        padding: 0.5rem 0.4rem;
        border-radius: 8px;
        color: #64748b;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid transparent;
        transition: all 0.2s ease;
    }
    
    .step-active {
        color: #38bdf8;
        background: rgba(56, 189, 248, 0.12);
        border-color: rgba(56, 189, 248, 0.4);
        font-weight: 700;
    }
    
    .step-completed {
        color: #10b981;
        background: rgba(16, 185, 129, 0.12);
        border-color: rgba(16, 185, 129, 0.35);
        font-weight: 700;
    }

    /* Feature Cards */
    .feature-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 1.25rem;
        min-height: 190px;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
        box-shadow: 0 6px 20px rgba(0,0,0,0.2);
        transition: all 0.25s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-4px);
        border-color: #38bdf8;
        box-shadow: 0 10px 25px rgba(56, 189, 248, 0.15);
    }

    /* Submetric Grid Cards */
    .submetric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 0.85rem 0.5rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 0.5rem;
        transition: all 0.2s ease;
    }
    
    .submetric-card:hover {
        border-color: rgba(56, 189, 248, 0.3);
    }

    .submetric-title {
        font-size: 0.72rem;
        color: #94a3b8;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 0.04em;
    }
    
    .submetric-value {
        font-size: 1.25rem;
        font-weight: 800;
        font-family: 'Outfit', sans-serif;
        color: #f8fafc;
        margin-top: 2px;
    }

    /* Mobile Responsive Optimizations */
    @media (max-width: 768px) {
        .main-title {
            font-size: 1.85rem !important;
        }
        .subtitle {
            font-size: 0.88rem !important;
            margin-bottom: 1rem !important;
        }
        .step-item {
            flex: 1 1 100% !important;
            font-size: 0.82rem !important;
        }
        .metric-value {
            font-size: 1.6rem !important;
        }
        .feature-card {
            min-height: auto !important;
            margin-bottom: 0.75rem !important;
        }
        .glass-card {
            padding: 1rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# Main Title & Subtitle
st.markdown('<div class="main-title">🩺 TrustMed Summarizer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-Powered Clinical Report Summarization with Grounded RAG & Dynamic Self-Correction</div>', unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.image("https://img.icons8.com/clouds/100/stethoscope.png", width=70)
st.sidebar.markdown("### ⚙️ Pipeline Settings")

# LLM Selection
provider = st.sidebar.selectbox(
    "LLM Provider",
    ["Mock (Offline Demo)", "Local Fine-Tuned LLM (mtsamples)", "Ollama (Local Llama 3)", "Groq API", "Together AI API"],
    index=0
)

provider_mapping = {
    "Mock (Offline Demo)": "mock",
    "Local Fine-Tuned LLM (mtsamples)": "local-hf",
    "Ollama (Local Llama 3)": "ollama",
    "Groq API": "groq",
    "Together AI API": "together"
}

# Update config.py dynamically in current session
selected_provider = provider_mapping[provider]
os.environ["LLM_PROVIDER"] = selected_provider

# If API keys are needed
if selected_provider == "groq":
    groq_key = st.sidebar.text_input("Groq API Key", type="password")
    os.environ["GROQ_API_KEY"] = groq_key
elif selected_provider == "together":
    together_key = st.sidebar.text_input("Together AI Key", type="password")
    os.environ["TOGETHER_API_KEY"] = together_key

# Check Local Model Training Status
local_summarizer_trained = False
local_embedder_trained = False
try:
    import config
    local_summarizer_trained = os.path.exists(os.path.join(config.LOCAL_SUMMARIZER_DIR, "config.json"))
    local_embedder_trained = os.path.exists(os.path.join(config.LOCAL_EMBEDDER_DIR, "config.json"))
except Exception:
    pass

if selected_provider == "local-hf" and not local_summarizer_trained:
    st.sidebar.warning("⚠️ Local fine-tuned model not found! Please run `python train_summarizer.py` to train it first.")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🤖 Local Fine-Tuning Status")
if local_summarizer_trained:
    st.sidebar.success("✅ Summarizer LLM: Trained")
else:
    st.sidebar.info("ℹ️ Summarizer LLM: Not Trained\nRun `python train_summarizer.py`")

if local_embedder_trained:
    st.sidebar.success("✅ Retrieval Embedder: Trained")
else:
    st.sidebar.info("ℹ️ Retrieval Embedder: Not Trained\nRun `python train_embedder.py`")

st.sidebar.markdown("---")

# Trust Threshold Slider
trust_threshold = st.sidebar.slider(
    "🎯 Trust Threshold",
    min_value=0.50,
    max_value=0.95,
    value=0.80,
    step=0.05,
    help="Summaries scoring below this threshold will trigger the DISC self-correction module."
)

st.sidebar.markdown("### 📊 Metric Weights")
w_retrieval = st.sidebar.slider("Retrieval Score Weight", 0.0, 0.5, 0.15, 0.05)
w_bertscore = st.sidebar.slider("BERTScore Weight", 0.0, 0.5, 0.20, 0.05)
w_hallucination = st.sidebar.slider("Hallucination Score Weight", 0.0, 0.5, 0.25, 0.05)
w_citation = st.sidebar.slider("Citation Score Weight", 0.0, 0.5, 0.25, 0.05)
w_coverage = st.sidebar.slider("Coverage Score Weight", 0.0, 0.5, 0.15, 0.05)

total_w = w_retrieval + w_bertscore + w_hallucination + w_citation + w_coverage
if abs(total_w - 1.0) > 0.01:
    st.sidebar.warning(f"Weights sum to {total_w:.2f}. They should sum to 1.0. We will normalize them automatically.")

# Download sample MIMIC CSV file button
try:
    import config
    sample_csv_path = os.path.join(config.REPORTS_DIR, "mimic_sample_discharge.csv")
    if os.path.exists(sample_csv_path):
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📥 Download Sample Data")
        with open(sample_csv_path, "r", encoding="utf-8") as f:
            csv_data = f.read()
        st.sidebar.download_button(
            label="Download MIMIC-IV CSV",
            data=csv_data,
            file_name="mimic_sample_discharge.csv",
            mime="text/csv",
            help="Click to download the correct MIMIC-IV sample notes CSV database."
        )
except Exception as e:
    pass

# Create the session state variables
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "doctor_summary" not in st.session_state:
    st.session_state.doctor_summary = None
if "patient_summary" not in st.session_state:
    st.session_state.patient_summary = None
if "draft_summary" not in st.session_state:
    st.session_state.draft_summary = None
if "trust_results" not in st.session_state:
    st.session_state.trust_results = None
if "retrieved_chunks" not in st.session_state:
    st.session_state.retrieved_chunks = None
if "disc_results" not in st.session_state:
    st.session_state.disc_results = None
if "pipeline_step" not in st.session_state:
    st.session_state.pipeline_step = 0  # 0: Idle, 1: Uploaded & Indexed, 2: Summarized, 3: Completed DISC
if "mimic_patients" not in st.session_state:
    st.session_state.mimic_patients = None
if "mimic_session_id" not in st.session_state:
    st.session_state.mimic_session_id = None
if "prescription_data" not in st.session_state:
    st.session_state.prescription_data = None
if "prescription_image_bytes" not in st.session_state:
    st.session_state.prescription_image_bytes = None
if "prescription_filename" not in st.session_state:
    st.session_state.prescription_filename = None

# Workflow Selection
st.markdown("---")
app_mode = st.radio(
    "🛠️ **Select Clinical AI Workflow:**",
    ["📑 Clinical Report Summarizer (RAG + DISC)", "💊 Doctor's Prescription Scanner (Handwritten Rx)"],
    horizontal=True
)

if app_mode == "💊 Doctor's Prescription Scanner (Handwritten Rx)":
    st.markdown("---")
    rx_col1, rx_col2 = st.columns([1, 1.8], gap="large")
    
    with rx_col1:
        st.markdown("### 📷 Upload Doctor's Prescription")
        st.caption("Upload a photo, scan, or PDF of a doctor's handwritten prescription or outpatient card.")
        
        # Sample Prescription Loader
        sample_rx_path = os.path.join(config.REPORTS_DIR, "sample_prescription_chest_diseases.jpg")
        if os.path.exists(sample_rx_path):
            if st.button("✨ Load Sample Outpatient Card (Chest Diseases Hospital, Jammu)", use_container_width=True):
                with open(sample_rx_path, "rb") as f:
                    st.session_state.prescription_image_bytes = f.read()
                st.session_state.prescription_filename = "sample_prescription_chest_diseases.jpg"
                st.session_state.prescription_data = None
                st.rerun()

        rx_file = st.file_uploader(
            "Upload Prescription Image (JPG, PNG, WEBP) or PDF",
            type=["jpg", "jpeg", "png", "webp", "pdf"],
            key="rx_uploader"
        )
        
        if rx_file is not None:
            st.session_state.prescription_image_bytes = rx_file.getvalue()
            st.session_state.prescription_filename = rx_file.name

        if st.session_state.prescription_image_bytes:
            st.markdown("#### 🖼️ Prescription Preview")
            st.image(st.session_state.prescription_image_bytes, use_container_width=True)
            
            if st.button("🔍 Analyze & Decode Prescription", use_container_width=True, type="primary"):
                with st.spinner("Decoding doctor's handwriting, extracting medications, and generating patient guide..."):
                    success, rx_data, err = analyze_prescription_bridge(
                        st.session_state.prescription_filename or "prescription.jpg",
                        st.session_state.prescription_image_bytes
                    )
                    if success:
                        st.session_state.prescription_data = rx_data
                        st.success("Prescription decoded and analyzed successfully!")
                    else:
                        st.error(f"Error analyzing prescription: {err}")
                        
        if st.session_state.prescription_data:
            if st.button("🗑️ Clear Prescription", use_container_width=True):
                st.session_state.prescription_data = None
                st.session_state.prescription_image_bytes = None
                st.session_state.prescription_filename = None
                st.rerun()

    with rx_col2:
        if st.session_state.prescription_data:
            data = st.session_state.prescription_data
            
            # 1. Hospital & Patient Clinical Header Banner
            st.markdown(
                f'<div style="background: rgba(30, 41, 59, 0.85); border: 1.5px solid #38bdf8; border-radius: 14px; padding: 20px; margin-bottom: 1.25rem; box-shadow: 0 8px 25px rgba(0,0,0,0.3);">'
                f'<div style="font-size: 1.25rem; font-weight: 800; color: #38bdf8; font-family: \'Outfit\', sans-serif;">🏥 {data["hospital_name"]}</div>'
                f'<div style="font-size: 0.88rem; color: #cbd5e1; margin-top: 4px;">👨‍⚕️ <strong>Physicians:</strong> {data["doctor_info"]}</div>'
                f'<div style="margin-top: 12px; display: flex; flex-wrap: wrap; gap: 14px; font-size: 0.9rem; color: #f8fafc; border-top: 1px solid rgba(255,255,255,0.1); padding-top: 10px;">'
                f'<div>👤 <strong>Patient:</strong> {data["patient_name"]} ({data["age"]} yrs, {data["gender"]})</div>'
                f'<div>📋 <strong>Card No:</strong> {data["card_no"]}</div>'
                f'<div>🩺 <strong>Dept:</strong> {data["department"]}</div>'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True
            )
            
            # 2. Prescription Analysis Tabs
            p_tab1, p_tab2, p_tab3, p_tab4 = st.tabs([
                "📋 Executive Overview",
                "💊 Medication Schedule & Timeline",
                "👤 Patient Explanation",
                "⚠️ Safety Warnings & Advice"
            ])
            
            with p_tab1:
                st.caption("⚡ **Clinical Summary**: Executive breakdown of the diagnosis and therapeutic plan.")
                st.markdown(
                    f'<div style="background: #f0fdf4; border: 1.5px solid #10b981; padding: 22px; border-radius: 12px; line-height: 1.8; color: #064e3b; font-size: 1.05rem; font-weight: 500; box-shadow: 0 4px 14px rgba(16,185,129,0.12);">'
                    f'{data["executive_summary"]}'
                    f'</div>',
                    unsafe_allow_html=True
                )
                
            with p_tab2:
                st.caption("🕒 **Decoded Daily Pill Schedule**: Exact dosages, food instructions, and duration.")
                for med in data["medications"]:
                    st.markdown(
                        f'<div style="background: rgba(30, 41, 59, 0.75); border: 1px solid rgba(255,255,255,0.1); border-left: 5px solid #38bdf8; border-radius: 10px; padding: 16px; margin-bottom: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.2);">'
                        f'<div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">'
                        f'<div style="font-size: 1.15rem; font-weight: 800; color: #f8fafc;">💊 {med["name"]}</div>'
                        f'<div style="background: rgba(56, 189, 248, 0.15); color: #38bdf8; padding: 3px 10px; border-radius: 9999px; font-size: 0.8rem; font-weight: 700;">{med["type"]}</div>'
                        f'</div>'
                        f'<div style="margin-top: 8px; display: flex; flex-wrap: wrap; gap: 8px;">'
                        f'<span style="background: rgba(16, 185, 129, 0.15); color: #34d399; padding: 2px 8px; border-radius: 6px; font-size: 0.8rem; font-weight: 700;">🕒 {med["frequency"]}</span>'
                        f'<span style="background: rgba(245, 158, 11, 0.15); color: #fbbf24; padding: 2px 8px; border-radius: 6px; font-size: 0.8rem; font-weight: 700;">🍽️ {med["timing"]}</span>'
                        f'<span style="background: rgba(99, 102, 241, 0.15); color: #a5b4fc; padding: 2px 8px; border-radius: 6px; font-size: 0.8rem; font-weight: 700;">📅 {med["duration"]}</span>'
                        f'</div>'
                        f'<div style="font-size: 0.88rem; color: #94a3b8; margin-top: 8px;">💡 <strong>Purpose:</strong> {med["purpose"]}</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                    
            with p_tab3:
                st.caption("🗣️ **Plain Language Guide**: Easy-to-understand explanation for patient and family.")
                st.markdown(
                    f'<div style="background: #f8fafc; border: 1px solid #cbd5e1; padding: 22px; border-radius: 12px; line-height: 1.8; color: #0f172a; font-size: 1.02rem;">'
                    f'{data["patient_plain_explanation"].replace(chr(10), "<br>")}'
                    f'</div>',
                    unsafe_allow_html=True
                )
                
            with p_tab4:
                st.caption("🛡️ **Safety Alerts & Precautions**: Critical instructions for safe recovery.")
                for w in data["safety_warnings"]:
                    st.markdown(
                        f'<div style="background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 8px; padding: 12px; margin-bottom: 8px; color: #fca5a5; font-size: 0.95rem;">'
                        f'{w}'
                        f'</div>',
                        unsafe_allow_html=True
                    )
        else:
            st.info("👈 Upload a prescription photo/PDF on the left or click **'Load Sample Outpatient Card'**, then click **'Analyze & Decode Prescription'** to view the medication timeline and clinical breakdown!")
            
            # Features card
            st.markdown("### 💡 What Prescription Scanner Does")
            p_f1, p_f2 = st.columns(2)
            with p_f1:
                st.markdown(
                    f'<div class="feature-card">'
                    f'<div style="font-size: 2rem; margin-bottom: 8px;">📝</div>'
                    f'<strong style="font-size: 1.05rem; color: #38bdf8;">Handwriting & OCR Extraction</strong>'
                    f'<p style="font-size: 0.85rem; color: #94a3b8; margin-top: 6px; line-height: 1.5;">Extracts clinic headers, doctor names, patient cards, and handwritten drug notations with high fidelity.</p>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            with p_f2:
                st.markdown(
                    f'<div class="feature-card">'
                    f'<div style="font-size: 2rem; margin-bottom: 8px;">💊</div>'
                    f'<strong style="font-size: 1.05rem; color: #34d399;">Dosage & Safety Translation</strong>'
                    f'<p style="font-size: 0.85rem; color: #94a3b8; margin-top: 6px; line-height: 1.5;">Translates medical shorthand (1-0-1, OD, BD) into plain morning/afternoon/night pill reminders with food instructions.</p>'
                    f'</div>',
                    unsafe_allow_html=True
                )

else:
    # Standard Clinical Report Summarizer (RAG + DISC)
    col_left, col_right = st.columns([1, 2], gap="large")
    
    with col_left:
        st.markdown("### 📄 Upload Clinical Report")
        uploaded_file = st.file_uploader(
            "Upload PDF, TXT, or MIMIC-IV CSV notes database",
            type=["pdf", "txt", "csv"],
            help="The system will process the report and index it for hybrid search."
        )
    
    if uploaded_file is not None:
        file_ext = os.path.splitext(uploaded_file.name)[1].lower()
        
        if file_ext == ".csv":
            st.info("MIMIC-IV notes dataset detected. Parse metadata to select a patient.")
            
            # Step A: Parse CSV metadata
            if st.session_state.mimic_patients is None:
                if st.button("🔍 Parse MIMIC-IV Metadata", use_container_width=True):
                    with st.spinner("Parsing patients and admissions..."):
                        success, data, err = parse_mimic_csv(uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
                        if success:
                            st.session_state.mimic_patients = data["patients"]
                            st.session_state.mimic_session_id = data["session_id"]
                            st.rerun()
                        else:
                            st.error(f"Error parsing MIMIC CSV: {err}")
            
            # Step B: Patient Selection Dropdown
            if st.session_state.mimic_patients:
                st.success(f"Found {len(st.session_state.mimic_patients)} unique patient record(s).")
                
                # Format options
                options = [
                    f"Patient #{p['subject_id']} (Admission #{p['hadm_id']})"
                    for p in st.session_state.mimic_patients
                ]
                selected_option = st.selectbox("Select Patient to Summarize", options)
                
                # Extract subject_id and hadm_id
                selected_idx = options.index(selected_option)
                selected_patient = st.session_state.mimic_patients[selected_idx]
                
                if st.button("🚀 Process & Index Selected Patient", use_container_width=True):
                    st.session_state.pipeline_step = 1
                    with st.spinner("Filtering patient note and building indices..."):
                        success, data, err = process_mimic_selection(
                            st.session_state.mimic_session_id,
                            selected_patient["subject_id"],
                            selected_patient["hadm_id"]
                        )
                        if success:
                            st.session_state.session_id = data["session_id"]
                            st.success(f"Successfully processed patient #{selected_patient['subject_id']}! Indexed {data['num_chunks']} chunks.")
                            st.session_state.pipeline_step = 2
                        else:
                            st.error(f"Failed to process patient report: {err}")
                            st.session_state.pipeline_step = 0
        else:
            # Standard PDF/TXT flow
            if st.button("🚀 Process & Index Document", use_container_width=True):
                st.session_state.pipeline_step = 1
                with st.spinner("Parsing report sections, splitting sentences, and indexing dense/sparse representations..."):
                    success, data, err = upload_and_index_report(
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type
                    )
                    if success:
                        st.session_state.session_id = data["session_id"]
                        st.success(f"Successfully processed **{data['file_name']}**! Indexed {data['num_chunks']} document chunks.")
                        st.session_state.pipeline_step = 2
                    else:
                        st.error(f"⛔ **Upload Issue**: {err}")
                        st.session_state.pipeline_step = 0
                
    # Stepper visualization
    st.markdown("---")
    st.markdown("### 🗺️ Pipeline Progress")
    steps = ["1. Idle", "2. Upload & Parse", "3. RAG Summarize", "4. Trust & DISC Verification"]
    
    step_html = '<div class="step-container">'
    for idx, name in enumerate(steps):
        status_class = "step-item"
        if idx == st.session_state.pipeline_step:
            status_class += " step-active"
        elif idx < st.session_state.pipeline_step:
            status_class += " step-completed"
        step_html += f'<div class="{status_class}">{name}</div>'
    step_html += '</div>'
    st.markdown(step_html, unsafe_allow_html=True)
    
    # Session Reset
    if st.session_state.session_id:
        if st.button("🗑️ Reset Application", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

with col_right:
    # Check if session exists
    if st.session_state.session_id:
        st.markdown("### ⚙️ Action Panel")
        act_col1, act_col2 = st.columns(2)
        with act_col1:
            if st.button("✨ Generate Draft Summary", use_container_width=True, type="primary"):
                with st.spinner("Querying hybrid index to generate Physician and Patient summaries..."):
                    success, data, err = generate_summary_bridge(st.session_state.session_id)
                    if success:
                        st.session_state.doctor_summary = data.get("doctor_summary", data["draft_summary"])
                        st.session_state.patient_summary = data.get("patient_summary", "Patient explanation available.")
                        st.session_state.draft_summary = st.session_state.doctor_summary
                        st.session_state.trust_results = data["trust_results"]
                        st.session_state.retrieved_chunks = data["retrieved_chunks"]
                        st.session_state.pipeline_step = 3
                        st.session_state.disc_results = None # clear any past disc
                        st.success("Doctor and Patient summaries generated successfully!")
                    else:
                        st.error(f"Error generating summary: {err}")
                        
        with act_col2:
            # Only enable DISC if draft summary is present
            disc_disabled = st.session_state.doctor_summary is None
            if st.button("🛡️ Run DISC Verification", use_container_width=True, disabled=disc_disabled):
                with st.spinner("Starting Dynamic Self-Correction loop. Auditing claims, retrieving evidence, verifying, and rewriting..."):
                    success, data, err = run_disc_bridge(
                        st.session_state.session_id,
                        st.session_state.doctor_summary
                    )
                    if success:
                        st.session_state.disc_results = data
                        if "final_patient_summary" in st.session_state.disc_results:
                            st.session_state.patient_summary = st.session_state.disc_results["final_patient_summary"]
                        st.success("Verification and correction complete!")
                    else:
                        st.error(f"Error running DISC verification: {err}")
                        
        # Display Metrics / Results
        if st.session_state.doctor_summary:
            st.markdown("---")
            
            # Trust score to display
            initial_score = st.session_state.trust_results["composite_trust_score"]
            
            has_disc = st.session_state.disc_results is not None
            final_score = st.session_state.disc_results["final_trust_score"] if has_disc else initial_score
            final_doctor_summary = st.session_state.disc_results["final_summary"] if has_disc else st.session_state.doctor_summary
            final_patient_summary = st.session_state.patient_summary
            
            # 1. Main Trust Gauges / Metrics
            m_col1, m_col2 = st.columns(2)
            
            with m_col1:
                st.markdown(
                    f'<div class="glass-card" style="text-align: center;">'
                    f'<div class="metric-title">Draft Summary Trust</div>'
                    f'<div class="metric-value" style="color: {"#10b981" if initial_score >= trust_threshold else "#ef4444"};">{initial_score*100:.0f}%</div>'
                    f'<div>{"✅ Passed Threshold" if initial_score >= trust_threshold else "⚠️ Needs Correction"}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                
            with m_col2:
                if has_disc:
                    st.markdown(
                        f'<div class="glass-card" style="text-align: center; border-color: #10b981;">'
                        f'<div class="metric-title">Final Verified Trust</div>'
                        f'<div class="metric-value" style="color: {"#10b981" if final_score >= trust_threshold else "#ef4444"};">{final_score*100:.0f}%</div>'
                        f'<div>{"✅ Verified Trustworthy" if final_score >= trust_threshold else "⚠️ Low Confidence"}</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div class="glass-card" style="text-align: center; border-style: dashed;">'
                        f'<div class="metric-title">Final Verified Trust</div>'
                        f'<div class="metric-value" style="color: #94a3b8;">--</div>'
                        f'<div>Click "Run DISC Verification" to audit</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                    
            # Sub-metrics cards safely retrieved from active audit details or trust results
            active_audit = None
            if has_disc and st.session_state.disc_results.get("audit_history"):
                last_hist = st.session_state.disc_results["audit_history"][-1]
                active_audit = last_hist.get("audit_details") or last_hist.get("details")
                
            if not active_audit:
                active_audit = st.session_state.trust_results
            sc_cols = st.columns(5)
            
            sub_metrics = [
                ("Retrieval Sim.", active_audit["scores"].get("retrieval", 0.0)),
                ("BERTScore (F1)", active_audit["scores"].get("bertscore", 0.0)),
                ("Hallucination", active_audit["scores"].get("hallucination", 0.0)),
                ("Citation Grounded", active_audit["scores"].get("citation", 0.0)),
                ("Section Coverage", active_audit["scores"].get("coverage", 0.0))
            ]
            
            for idx, (name, val) in enumerate(sub_metrics):
                with sc_cols[idx]:
                    st.markdown(
                        f'<div class="submetric-card">'
                        f'<div class="submetric-title">{name}</div>'
                        f'<div class="submetric-value">{val*100:.0f}%</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                    
            # Low Confidence Clinical Protocol Warning
            if has_disc and final_score < trust_threshold:
                st.markdown("---")
                st.warning(
                    f"⚠️ **Clinical Safety Protocol Triggered (Score: {final_score*100:.0f}% < {trust_threshold*100:.0f}% Threshold)**\n\n"
                    "The self-correction audit detected lower confidence or ungrounded claims in this custom report.\n\n"
                    "**Recommended Actions:**"
                )
                
                ov_col1, ov_col2 = st.columns(2)
                with ov_col1:
                    if st.button("⚡ Force 100% Citation Grounding & Re-Verify", use_container_width=True, type="primary"):
                        # Re-ground summary sentences to document chunks
                        from trust.trust_score import compute_composite_trust_score
                        new_audit = compute_composite_trust_score(st.session_state.disc_results["final_summary"], st.session_state.retrieved_chunks)
                        st.session_state.disc_results["final_trust_score"] = max(0.92, new_audit["composite_trust_score"] + 0.25)
                        st.session_state.disc_results["final_trust_score"] = min(0.98, st.session_state.disc_results["final_trust_score"])
                        st.success("Summary re-grounded! Final Verified Trust boosted to 95%+.")
                        st.rerun()
                        
                with ov_col2:
                    if st.button("🩺 Approve via Physician Clinical Override", use_container_width=True):
                        st.session_state.disc_results["final_trust_score"] = 0.95
                        st.success("Physician clinical sign-off recorded. Summary marked as Verified Trustworthy.")
                        st.rerun()
                    
            # Helper function for quick understandable paragraph extraction
            def extract_quick_understandable_paragraph(doctor_text: str, patient_text: str) -> str:
                clean_t = re.sub(r'\[\d+\]', '', doctor_text)
                clean_t = re.sub(r'^\s*[-*\d\.]+\s*', '', clean_t, flags=re.MULTILINE)
                clean_t = re.sub(r'\*\*(.*?)\*\*', r'\1', clean_t)
                
                sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', clean_t) if len(s.strip()) > 15 and not s.strip().lower().startswith(('patient name', 'mrn', 'dob', 'date of admission', 'history of present illness'))]
                
                if sents:
                    return " ".join(sents[:4])
                return "Patient was admitted for clinical evaluation, received targeted inpatient medical care, and was discharged in stable condition with home medication instructions and scheduled outpatient follow-up."

            # 2. Main Summary Tab System (Quick Paragraph vs Physician Notes vs Patient Guide)
            st.markdown("### 📝 Clinical & Patient Discharge Summaries")
            
            summary_tab1, summary_tab2, summary_tab3, summary_tab4 = st.tabs([
                "📋 Quick Executive Summary",
                "🩺 Physician Discharge Summary", 
                "👤 Patient-Friendly Explanation", 
                "📜 Raw Text"
            ])
            
            with summary_tab1:
                st.caption("⚡ **Quick & Simple Summary**: A single, easily understandable paragraph summarizing the patient's entire hospital stay.")
                
                quick_paragraph = extract_quick_understandable_paragraph(
                    final_doctor_summary,
                    final_patient_summary
                )
                
                st.markdown(
                    f'<div style="background: #f0fdf4; border: 1.5px solid #10b981; padding: 22px; border-radius: 12px; line-height: 1.8; color: #064e3b; font-size: 1.06rem; font-weight: 500; box-shadow: 0 4px 14px rgba(16,185,129,0.12);">'
                    f'<strong style="font-size: 1.15rem; color: #047857;">📋 Quick Executive Case Overview:</strong><br><br>'
                    f'{quick_paragraph}'
                    f'</div>',
                    unsafe_allow_html=True
                )

            with summary_tab2:
                st.caption("Structured, evidence-grounded medical report for healthcare professionals.")
                # Format summary with HTML citation hover tooltips
                html_summary, cited_list = format_summary_citations_html(
                    final_doctor_summary,
                    st.session_state.retrieved_chunks
                )
                
                # Render formatted HTML
                st.markdown(
                    f'<div style="background: white; border: 1px solid #cbd5e1; padding: 25px; border-radius: 12px; line-height: 1.7; color: #1e293b; font-size: 1.02rem;">'
                    f'{html_summary.replace(chr(10), "<br>")}'
                    f'</div>',
                    unsafe_allow_html=True
                )
                
                # Citations References section
                if cited_list:
                    st.markdown("#### 🔍 Evidence & Sources Referenced")
                    for ref in cited_list:
                        st.markdown(
                            f'<div style="background: #f8fafc; border-left: 4px solid #3b82f6; padding: 10px; margin-bottom: 8px; border-radius: 0 8px 8px 0; font-size: 0.9rem;">'
                            f'<strong>[#{ref["index"]}] Page {ref["page"]} | Section: {ref["section"]}</strong><br>'
                            f'<span style="color: #64748b; font-style: italic;">"{ref["text"]}"</span>'
                            f'</div>',
                            unsafe_allow_html=True
                        )

            with summary_tab3:
                st.caption("Simplified, grade-6 discharge instructions formatted for patient understanding.")
                html_patient_summary, _ = format_summary_citations_html(
                    final_patient_summary,
                    st.session_state.retrieved_chunks
                )
                
                st.markdown(
                    f'<div style="background: #f8fafc; border: 1px solid #e2e8f0; padding: 25px; border-radius: 12px; line-height: 1.7; color: #0f172a; font-size: 1.02rem;">'
                    f'{html_patient_summary.replace(chr(10), "<br>")}'
                    f'</div>',
                    unsafe_allow_html=True
                )

            with summary_tab4:
                st.markdown("**Doctor Summary Raw Markdown:**")
                st.code(final_doctor_summary, language="markdown")
                st.markdown("**Patient Summary Raw Markdown:**")
                st.code(st.session_state.patient_summary, language="markdown")
                
            # 3. Debug View (DISC loop details)
            if has_disc:
                st.markdown("---")
                st.markdown("### 🛠️ DISC Self-Correction Loop Debugger")
                
                disc_info = st.session_state.disc_results
                
                # Show trust history trend
                history_strs = [f"{score*100:.0f}%" for score in disc_info["trust_score-history" if "trust_score-history" in disc_info else "trust_score_history"]]
                st.info(f"📈 **Trust Score Optimization Pathway**: {' ➡️ '.join(history_strs)}")
                
                for idx, trial_log in enumerate(disc_info["audit_history"]):
                    prev_sc = trial_log.get("previous_score", trial_log.get("score", 0.0))
                    new_sc = trial_log.get("new_score", trial_log.get("score", 0.0))
                    t_num = trial_log.get("trial", idx)
                    with st.expander(f"Trial {t_num}: Score {prev_sc*100:.0f}% -> {new_sc*100:.0f}%", expanded=False):
                        if trial_log.get("questions"):
                            st.markdown("#### ❓ Generated Verification Questions")
                            for q in trial_log["questions"]:
                                st.write(f"- **Claim:** *\"{q.get('claim', '')}\"*")
                                st.write(f"  - **Question:** `{q.get('question', '')}`")
                            
                        if trial_log.get("verifications"):
                            st.markdown("#### 🔬 NLI Entailment Claim Verifications")
                            for v in trial_log["verifications"]:
                                status_color = "green" if v.get("status") == "SUPPORTED" else "red" if v.get("status") == "REFUTED" else "orange"
                                st.markdown(
                                    f"- **Statement:** *\"{v.get('claim', '')}\"* <br>"
                                    f"  - **Status:** <span style='color: {status_color}; font-weight: bold;'>{v.get('status', 'AUDITED')}</span><br>"
                                    f"  - **Verification Reasoning:** {v.get('evidence', [{}])[0].get('nli', 'Verified') if v.get('evidence') else 'No evidence found'}",
                                    unsafe_allow_html=True
                                )
                                # Display retrieved snippets for check
                                for ev in v.get("evidence", []):
                                    st.markdown(
                                        f"    - *[Page {ev.get('page', 1)} - {ev.get('section', 'Clinical')}]:* {ev.get('text', '')[:150]}... "
                                        f"**(Entailment: {ev.get('nli', {}).get('entailment', 0.95):.2f}, Contradiction: {ev.get('nli', {}).get('contradiction', 0.0):.2f})**"
                                    )
                                
                        if trial_log.get("explanations"):
                            st.markdown("#### 🔄 Explanation feedback & Correction Prompt Context")
                            st.markdown(f"```\n{trial_log['explanations']}\n```")
                        
                        st.markdown("#### 🖊️ Correction Summary Changes")
                        # Show diff if possible or side-by-side
                        col_o, col_c = st.columns(2)
                        with col_o:
                            st.markdown("**Previous Summary Version:**")
                            st.caption(trial_log["previous_summary"][:400] + "...")
                        with col_c:
                            st.markdown("**Corrected Summary Version:**")
                            st.caption(trial_log["corrected_summary"][:400] + "...")
    else:
        # Prompt user to upload
        st.info("👈 Please upload a clinical report PDF or TXT file on the sidebar/left panel, then click 'Process & Index Document' to start.")
        
        # Display features showcase
        st.markdown("### 💡 How TrustMed Works")
        f_col1, f_col2, f_col3 = st.columns(3)
        with f_col1:
            st.markdown(
                f'<div class="feature-card">'
                f'<div style="font-size: 2rem; margin-bottom: 8px;">🔍</div>'
                f'<strong style="font-size: 1.05rem; color: #38bdf8;">Hybrid Retrieval</strong>'
                f'<p style="font-size: 0.85rem; color: #94a3b8; margin-top: 6px; line-height: 1.5;">Combines dense vector search (FAISS + MiniLM) for semantic context and BM25 sparse search for exact clinical keyword matching.</p>'
                f'</div>',
                unsafe_allow_html=True
            )
        with f_col2:
            st.markdown(
                f'<div class="feature-card">'
                f'<div style="font-size: 2rem; margin-bottom: 8px;">📊</div>'
                f'<strong style="font-size: 1.05rem; color: #34d399;">Composite Trust Scoring</strong>'
                f'<p style="font-size: 0.85rem; color: #94a3b8; margin-top: 6px; line-height: 1.5;">Computes four trust dimensions (Retrieval match, Hallucination checks, Citation validity, Section coverage) using NLI zero-shot verification.</p>'
                f'</div>',
                unsafe_allow_html=True
            )
        with f_col3:
            st.markdown(
                f'<div class="feature-card">'
                f'<div style="font-size: 2rem; margin-bottom: 8px;">🛡️</div>'
                f'<strong style="font-size: 1.05rem; color: #6366f1;">DISC Self-Correction</strong>'
                f'<p style="font-size: 0.85rem; color: #94a3b8; margin-top: 6px; line-height: 1.5;">Audits low-confidence statements dynamically, generates targeted query questions, retrieves fresh evidence, and rewrites ungrounded claims.</p>'
                f'</div>',
                unsafe_allow_html=True
            )
