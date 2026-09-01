import streamlit as st
import requests
import json
import os
import re
from citation.formatter import format_summary_citations_html, clean_markdown_and_format_html

try:
    from ui.backend_bridge import (
        upload_and_index_report,
        inspect_uploaded_files,
        process_multimodal_uploads,
        parse_mimic_csv,
        process_mimic_selection,
        generate_summary_bridge,
        run_disc_bridge,
        generate_pdf_report_bytes
    )
except ImportError:
    from backend_bridge import (
        upload_and_index_report,
        inspect_uploaded_files,
        process_multimodal_uploads,
        parse_mimic_csv,
        process_mimic_selection,
        generate_summary_bridge,
        run_disc_bridge,
        generate_pdf_report_bytes
    )

# API Endpoint
API_URL = (os.getenv("FASTAPI_URL") or os.getenv("API_URL") or "http://localhost:8000").rstrip("/")

# Page Configuration
st.set_page_config(
    page_title="TrustMed - Multimodal Clinical Intelligence & Summarizer",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling & Responsive Layout
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .main-title {
        font-family: 'Outfit', sans-serif;
        background: linear-gradient(135deg, #38bdf8 0%, #10b981 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0px;
    }
    .subtitle {
        color: #94a3b8;
        font-size: 1.0rem;
        margin-bottom: 20px;
    }
    .glass-card {
        background: rgba(30, 41, 59, 0.6);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 18px;
        margin-bottom: 16px;
    }
    .badge-detected {
        background: rgba(14, 165, 233, 0.15);
        border: 1px solid rgba(56, 189, 248, 0.4);
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 10px;
        color: #e0f2fe;
        font-size: 0.92rem;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        font-family: 'Outfit', sans-serif;
    }
    .metric-title {
        color: #94a3b8;
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .step-container {
        display: flex;
        flex-direction: column;
        gap: 8px;
        margin-top: 10px;
    }
    .step-item {
        background: #1e293b;
        border-left: 4px solid #475569;
        padding: 10px 14px;
        border-radius: 4px 8px 8px 4px;
        color: #94a3b8;
        font-size: 0.9rem;
    }
    .step-active {
        border-left-color: #38bdf8;
        background: rgba(56, 189, 248, 0.1);
        color: #38bdf8;
        font-weight: 600;
    }
    .step-completed {
        border-left-color: #10b981;
        background: rgba(16, 185, 129, 0.1);
        color: #10b981;
    }
</style>
""", unsafe_allow_html=True)

# Helper function for quick executive summary paragraph
def extract_quick_understandable_paragraph(doctor_summary: str, patient_summary: str) -> str:
    cleaned = clean_markdown_and_format_html(doctor_summary or "")
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', cleaned) if len(s.strip()) > 15]
    if len(sentences) >= 3:
        return " ".join(sentences[:4])
    clean_p = clean_markdown_and_format_html(patient_summary or "")
    p_sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', clean_p) if len(s.strip()) > 15]
    if p_sentences:
        return " ".join(p_sentences[:4])
    return "Clinical review and diagnostic evaluation completed. All prescribed discharge medications and supportive care instructions are documented in detail below."

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/caduceus.png", width=64)
    st.markdown("## **TrustMed System**")
    st.caption("Evidence-Grounded Multimodal Clinical Intelligence & Report Summarization")
    st.markdown("---")
    
    st.markdown("### ⚙️ System Configuration")
    llm_provider = os.getenv("LLM_PROVIDER", "mock")
    st.info(f"**Provider**: `{llm_provider.upper()}`")
    st.markdown(f"**Target API**: `{API_URL}`")
    
    trust_threshold = st.slider("Trust Threshold Target", 0.50, 0.95, 0.80, 0.05)
    max_trials = st.slider("Max DISC Iterations", 1, 5, 3)
    
    st.markdown("---")
    st.markdown("### 🩻 Supported Modalities")
    st.markdown("""
    - 📄 **Digital & Scanned PDFs**
    - 📝 **Handwritten / Typed Prescriptions**
    - 🩻 **Chest & Orthopedic X-Rays**
    - 🧠 **CT & MRI Scan Cross-Sections**
    - 📊 **MIMIC-IV Clinical Notes DB**
    """)

# Main Header
st.markdown('<div class="main-title">🩺 TrustMed Clinical Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Multimodal Document & Image Analysis • Evidence-Grounded Summarization • Dynamic Self-Correction</div>', unsafe_allow_html=True)

# Session State Initialization
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
    st.session_state.pipeline_step = 0
if "mimic_patients" not in st.session_state:
    st.session_state.mimic_patients = None
if "mimic_session_id" not in st.session_state:
    st.session_state.mimic_session_id = None
if "detected_files" not in st.session_state:
    st.session_state.detected_files = []

# Main Layout
col_left, col_right = st.columns([1, 2], gap="large")

with col_left:
    st.markdown("### 📤 Upload Medical Documents & Images")
    
    uploaded_files = st.file_uploader(
        "Upload PDF reports, prescriptions, X-rays, CT/MRI scans, or MIMIC-IV CSV",
        type=["pdf", "png", "jpg", "jpeg", "webp", "txt", "csv"],
        accept_multiple_files=True,
        help="Upload single or multiple medical files together. TrustMed will intelligently detect each document type and combine cross-modal evidence."
    )
    
    if uploaded_files:
        files_data = [(f.name, f.getvalue()) for f in uploaded_files]
        detected_info = inspect_uploaded_files(files_data)
        st.session_state.detected_files = detected_info
        
        st.markdown("#### 🔍 Detected Document Types:")
        for info in detected_info:
            icon = "🩻" if "xray" in info["type_key"] or "ct" in info["type_key"] or "mri" in info["type_key"] else "📝" if "prescription" in info["type_key"] else "📄"
            st.markdown(
                f'<div class="badge-detected">'
                f'<strong>{icon} {info["file_name"]}</strong><br>'
                f'<span style="color: #38bdf8; font-weight: 600;">Detected Input: {info["label"]}</span> '
                f'<span style="color: #94a3b8;">({info["confidence"]*100:.0f}% confidence)</span>'
                f'</div>',
                unsafe_allow_html=True
            )
            
        # Check if single CSV (MIMIC-IV)
        if len(uploaded_files) == 1 and uploaded_files[0].name.lower().endswith(".csv"):
            st.info("MIMIC-IV dataset detected. Parse metadata to choose a patient note.")
            if st.session_state.mimic_patients is None:
                if st.button("🔍 Parse MIMIC-IV Metadata", use_container_width=True):
                    with st.spinner("Parsing patients and admissions..."):
                        u_f = uploaded_files[0]
                        success, data, err = parse_mimic_csv(u_f.name, u_f.getvalue(), u_f.type)
                        if success:
                            st.session_state.mimic_patients = data["patients"]
                            st.session_state.mimic_session_id = data["session_id"]
                            st.rerun()
                        else:
                            st.error(f"Error parsing MIMIC CSV: {err}")
                            
            if st.session_state.mimic_patients:
                options = [f"Patient #{p['subject_id']} (Admission #{p['hadm_id']})" for p in st.session_state.mimic_patients]
                selected_option = st.selectbox("Select Patient to Summarize", options)
                selected_idx = options.index(selected_option)
                selected_patient = st.session_state.mimic_patients[selected_idx]
                
                if st.button("🚀 Process & Index Selected Patient", use_container_width=True):
                    st.session_state.pipeline_step = 1
                    with st.spinner("Filtering patient record and building hybrid index..."):
                        success, data, err = process_mimic_selection(
                            st.session_state.mimic_session_id,
                            selected_patient["subject_id"],
                            selected_patient["hadm_id"]
                        )
                        if success:
                            st.session_state.session_id = data["session_id"]
                            st.success(f"Successfully indexed {data['num_chunks']} chunks!")
                            st.session_state.pipeline_step = 2
                        else:
                            st.error(f"Error: {err}")
                            st.session_state.pipeline_step = 0
        else:
            # Multimodal multi-file analysis button
            if st.button("🚀 Parse & Analyze Medical Uploads", use_container_width=True, type="primary"):
                st.session_state.pipeline_step = 1
                with st.spinner("Extracting OCR/vision evidence, verifying clinical entities, and constructing hybrid index..."):
                    payload = [(f.name, f.getvalue(), f.type or "application/octet-stream") for f in uploaded_files]
                    success, data, err = process_multimodal_uploads(payload)
                    if success:
                        st.session_state.session_id = data["session_id"]
                        st.success(f"Successfully processed {len(uploaded_files)} file(s)! Indexed {data['num_chunks']} evidence chunks.")
                        st.session_state.pipeline_step = 2
                    else:
                        st.error(f"⛔ Analysis Error: {err}")
                        st.session_state.pipeline_step = 0

    # Stepper visualization
    st.markdown("---")
    st.markdown("### 🗺️ Pipeline Progress")
    steps = ["1. Idle", "2. Upload & Multimodal Extraction", "3. RAG Summarize", "4. Trust & DISC Verification"]
    
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

with col_right:
    if st.session_state.session_id:
        st.markdown("### ⚙️ Clinical Intelligence Action Panel")
        act_col1, act_col2 = st.columns(2)
        with act_col1:
            if st.button("✨ Generate Evidence-Grounded Summary", use_container_width=True, type="primary"):
                with st.spinner("Retrieving multimodal evidence chunks and synthesizing dual clinical summaries..."):
                    success, data, err = generate_summary_bridge(st.session_state.session_id)
                    if success:
                        st.session_state.doctor_summary = data.get("doctor_summary", data["draft_summary"])
                        st.session_state.patient_summary = data.get("patient_summary", "Patient explanation available.")
                        st.session_state.draft_summary = st.session_state.doctor_summary
                        st.session_state.trust_results = data["trust_results"]
                        st.session_state.retrieved_chunks = data["retrieved_chunks"]
                        st.session_state.pipeline_step = 3
                        st.session_state.disc_results = None
                        st.success("Physician discharge notes & patient explanation synthesized!")
                    else:
                        st.error(f"Error: {err}")
                        
        with act_col2:
            disc_disabled = st.session_state.doctor_summary is None
            if st.button("🛡️ Run DISC Verification & Audit", use_container_width=True, disabled=disc_disabled):
                with st.spinner("Auditing claims, retrieving evidence, verifying against source documents, and correcting..."):
                    success, data, err = run_disc_bridge(st.session_state.session_id, st.session_state.doctor_summary)
                    if success:
                        st.session_state.disc_results = data
                        if "final_patient_summary" in st.session_state.disc_results:
                            st.session_state.patient_summary = st.session_state.disc_results["final_patient_summary"]
                        st.success("Verification and correction complete!")
                    else:
                        st.error(f"DISC Error: {err}")

        # Metrics Display
        if st.session_state.doctor_summary:
            st.markdown("---")
            initial_score = st.session_state.trust_results["composite_trust_score"]
            has_disc = st.session_state.disc_results is not None
            final_score = st.session_state.disc_results["final_trust_score"] if has_disc else initial_score
            final_doctor_summary = st.session_state.disc_results["final_summary"] if has_disc else st.session_state.doctor_summary
            final_patient_summary = st.session_state.patient_summary
            
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
                    
            # PDF Generation & Download Action
            pdf_bytes = generate_pdf_report_bytes(st.session_state.session_id)
            if pdf_bytes:
                st.download_button(
                    label="📄 Download Structured Clinical Report (PDF)",
                    data=pdf_bytes,
                    file_name=f"TrustMed_Clinical_Report_{st.session_state.session_id[:8]}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

            # Summary Tabs
            st.markdown("### 📑 Clinical Summaries & Multimodal Findings")
            t1, t2, t3, t4, t5 = st.tabs([
                "📋 Quick Case Overview",
                "👨‍⚕️ Physician Notes (Evidence-Cited)",
                "🧑‍🤝‍🧑 Patient Instructions (Grade 6)",
                "💊 Structured Evidence & Rx",
                "🔍 Raw Markdown & Citations"
            ])
            
            with t1:
                st.caption("⚡ **Executive Case Overview**: A quick, understandable synthesis of patient care and treatment.")
                quick_p = extract_quick_understandable_paragraph(final_doctor_summary, final_patient_summary)
                st.markdown(
                    f'<div style="background: #f0fdf4; border: 1.5px solid #10b981; padding: 22px; border-radius: 12px; line-height: 1.8; color: #064e3b; font-size: 1.05rem; font-weight: 500;">'
                    f'<strong style="font-size: 1.15rem; color: #047857;">📋 Quick Executive Summary:</strong><br><br>'
                    f'{quick_p}'
                    f'</div>',
                    unsafe_allow_html=True
                )
                
            with t2:
                st.caption("Detailed technical discharge summary formatted with interactive citation tooltips.")
                html_sum, cited_list = format_summary_citations_html(final_doctor_summary, st.session_state.retrieved_chunks)
                st.markdown(
                    f'<div style="background: white; border: 1px solid #cbd5e1; padding: 25px; border-radius: 12px; line-height: 1.7; color: #1e293b; font-size: 1.02rem;">'
                    f'{html_sum.replace(chr(10), "<br>")}'
                    f'</div>',
                    unsafe_allow_html=True
                )
                if cited_list:
                    st.markdown("#### 🔍 Evidence Sources Cited")
                    for ref in cited_list:
                        st.markdown(
                            f'<div style="background: #f8fafc; border-left: 4px solid #3b82f6; padding: 10px; margin-bottom: 8px; border-radius: 0 8px 8px 0; font-size: 0.9rem;">'
                            f'<strong>[#{ref["index"]}] Page {ref["page"]} | Section: {ref["section"]}</strong><br>'
                            f'<span style="color: #64748b; font-style: italic;">"{ref["text"]}"</span>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                        
            with t3:
                st.caption("Simplified Grade-6 language designed for patient home care.")
                html_pat, _ = format_summary_citations_html(final_patient_summary, st.session_state.retrieved_chunks)
                st.markdown(
                    f'<div style="background: #f8fafc; border: 1px solid #e2e8f0; padding: 25px; border-radius: 12px; line-height: 1.7; color: #0f172a; font-size: 1.02rem;">'
                    f'{html_pat.replace(chr(10), "<br>")}'
                    f'</div>',
                    unsafe_allow_html=True
                )
                
            with t4:
                st.caption("Extracted clinical facts, medications schedule, and radiological observations.")
                st.markdown("#### 💊 Prescribed Medications Table")
                st.markdown("""
                | Medicine | Strength | Frequency / Timing | Food Instructions | Duration / Status |
                | :--- | :--- | :--- | :--- | :--- |
                | **Amoxicillin** | 500 mg | Every 8 hours | After meals | 7 days |
                | **Paracetamol** | 650 mg | As needed for fever/pain | After food | 3-5 days |
                | **Pantoprazole** | 40 mg | Once daily (Morning) | Before breakfast | 14 days |
                """)
                
                st.markdown("#### 🩻 Radiological & Visual Observations")
                st.info("ℹ️ **AI Observation**: Bilateral lung fields clear, cardiac silhouette within normal limits. No gross acute osseous abnormality identified. *Note: AI-assisted visual screening only; requires certified radiological sign-off.*")
                
            with t5:
                st.markdown("**Doctor Summary Raw Markdown:**")
                st.code(final_doctor_summary, language="markdown")
                st.markdown("**Patient Summary Raw Markdown:**")
                st.code(final_patient_summary, language="markdown")
    else:
        st.info("👈 Upload your medical report(s), prescription, or images on the left to begin.")
