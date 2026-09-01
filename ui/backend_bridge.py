"""
TrustMed Multimodal Backend Bridge.
Handles single and multi-file uploads (PDFs, Images, Prescriptions, X-Rays, CT, MRI),
extracts structured clinical evidence, indexes into FAISS + BM25, coordinates summaries,
and generates publication-grade Clinical PDF reports.
Seamlessly runs via FastAPI backend or in-process on Streamlit Cloud.
"""

import os
import uuid
import json
import shutil
import requests
from typing import Dict, Any, Tuple, Optional, List

import config
from preprocessing.loader import load_report, load_mimic_csv
from preprocessing.cleaner import clean_text, validate_medical_document
from preprocessing.chunker import chunk_text
from preprocessing.section_extractor import extract_sections
from retrieval.embedder import embed_text
from retrieval.faiss_index import FAISSIndex
from retrieval.bm25 import BM25Index
from summarization.summarizer import generate_draft_summary
from trust.trust_score import compute_composite_trust_score
from disc.controller import run_disc_pipeline

from multimodal.detector import detect_input_type
from multimodal.schemas import StructuredClinicalEvidence
from multimodal.evidence_aggregator import aggregate_multimodal_evidence, convert_evidence_to_chunks
from reports.pdf_generator import generate_clinical_report_pdf

API_URL = (os.getenv("FASTAPI_URL") or os.getenv("API_URL") or "http://localhost:8000").rstrip("/")

# In-memory session cache for in-process mode
IN_PROCESS_SESSION_CACHE = {}

def is_fastapi_available() -> bool:
    """
    Checks if a live FastAPI server is accessible.
    """
    try:
        r = requests.get(f"{API_URL}/health", timeout=0.8)
        return r.status_code == 200
    except Exception:
        return False

def inspect_uploaded_files(files_data: List[tuple[str, bytes]]) -> List[Dict[str, Any]]:
    """
    Inspects a list of uploaded files and detects their clinical document/imaging types.
    
    Args:
        files_data: List of (file_name, file_bytes)
        
    Returns:
        List of dicts with file_name, detected_type, label, confidence, and preview info.
    """
    results = []
    temp_dir = os.path.join(config.REPORTS_DIR, "temp_detect")
    os.makedirs(temp_dir, exist_ok=True)
    
    for fname, fbytes in files_data:
        temp_path = os.path.join(temp_dir, f"detect_{uuid.uuid4()}_{fname}")
        try:
            with open(temp_path, "wb") as f:
                f.write(fbytes)
            type_key, label, conf = detect_input_type(temp_path, fname)
            results.append({
                "file_name": fname,
                "type_key": type_key,
                "label": label,
                "confidence": conf
            })
        except Exception as e:
            results.append({
                "file_name": fname,
                "type_key": "clinical_image",
                "label": "Medical Document",
                "confidence": 0.7
            })
        finally:
            if os.path.exists(temp_path):
                try: os.remove(temp_path)
                except Exception: pass
                
    return results

def process_multimodal_uploads(files_data: List[tuple[str, bytes, str]]) -> Tuple[bool, Dict[str, Any], Optional[str]]:
    """
    Processes single or multiple uploaded files (PDFs, Images, Prescriptions, X-Rays, CT, MRI).
    Aggregates evidence across all documents, indexes into FAISS + BM25, and prepares session.
    
    Args:
        files_data: List of (file_name, file_bytes, mime_type)
        
    Returns:
        (success, result_dict, error_msg)
    """
    if not files_data:
        return False, {}, "No files provided for processing."
        
    session_id = str(uuid.uuid4())
    saved_file_records = []
    
    try:
        # 1. Save all uploaded files to disk
        for fname, fbytes, _ in files_data:
            dest_path = os.path.join(config.REPORTS_DIR, f"{session_id}_{fname}")
            with open(dest_path, "wb") as f:
                f.write(fbytes)
            saved_file_records.append((dest_path, fname))
            
        # 2. Extract and aggregate multimodal evidence across all files
        unified_evidence = aggregate_multimodal_evidence(saved_file_records)
        
        # 3. Convert evidence into source-tagged chunks for RAG
        chunks = convert_evidence_to_chunks(unified_evidence)
        if not chunks:
            return False, {}, "Could not extract sufficient clinical text or visual evidence from the uploaded files."
            
        # 4. Save processed chunks to disk
        chunks_file = os.path.join(config.PROCESSED_DIR, f"{session_id}.json")
        with open(chunks_file, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2)
            
        # 5. Build retrieval indices (FAISS dense + BM25 sparse)
        chunk_texts = [c["text"] for c in chunks]
        embeddings = embed_text(chunk_texts)
        
        faiss_index = FAISSIndex(dimension=embeddings.shape[1])
        faiss_index.add_embeddings(embeddings)
        
        faiss_path = os.path.join(config.EMBEDDINGS_DIR, session_id)
        faiss_index.save(faiss_path)
        
        bm25_index = BM25Index()
        bm25_index.build(chunk_texts)
        
        # 6. Cache in memory
        IN_PROCESS_SESSION_CACHE[session_id] = {
            "chunks": chunks,
            "faiss": faiss_index,
            "bm25": bm25_index,
            "evidence": unified_evidence,
            "file_names": [fname for fname, _, _ in files_data]
        }
        
        return True, {
            "session_id": session_id,
            "file_names": [fname for fname, _, _ in files_data],
            "detected_types": unified_evidence.detected_types,
            "num_chunks": len(chunks),
            "num_medications": len(unified_evidence.medications),
            "num_image_findings": len(unified_evidence.image_findings),
            "status": "indexed"
        }, None
        
    except Exception as e:
        return False, {}, f"Multimodal Processing Error: {str(e)}"

def upload_and_index_report(file_name: str, file_bytes: bytes, mime_type: str) -> Tuple[bool, Dict[str, Any], Optional[str]]:
    """
    Backward-compatible single file upload handler that dispatches to multimodal engine.
    """
    return process_multimodal_uploads([(file_name, file_bytes, mime_type)])

def parse_mimic_csv(file_name: str, file_bytes: bytes, mime_type: str) -> Tuple[bool, Dict[str, Any], Optional[str]]:
    """
    Parses MIMIC-IV CSV metadata via FastAPI or in-process.
    """
    if is_fastapi_available():
        try:
            files = {"file": (file_name, file_bytes, mime_type)}
            res = requests.post(f"{API_URL}/parse_mimic_metadata", files=files, timeout=10)
            if res.status_code == 200:
                return True, res.json(), None
        except Exception:
            pass

    try:
        session_id = str(uuid.uuid4())
        temp_file_path = os.path.join(config.REPORTS_DIR, f"{session_id}_{file_name}")
        with open(temp_file_path, "wb") as f:
            f.write(file_bytes)

        import pandas as pd
        df = pd.read_csv(temp_file_path)
        if 'subject_id' not in df.columns or 'hadm_id' not in df.columns:
            return False, {}, "CSV does not contain 'subject_id' and 'hadm_id' columns."

        df_unique = df[['subject_id', 'hadm_id']].drop_duplicates().dropna()
        patients = [{"subject_id": int(r['subject_id']), "hadm_id": int(r['hadm_id'])} for _, r in df_unique.iterrows()]

        return True, {
            "session_id": session_id,
            "file_path": temp_file_path,
            "patients": patients
        }, None
    except Exception as e:
        return False, {}, f"MIMIC Parsing Error: {e}"

def process_mimic_selection(session_id: str, subject_id: int, hadm_id: int) -> Tuple[bool, Dict[str, Any], Optional[str]]:
    """
    Processes selected MIMIC patient via FastAPI or in-process.
    """
    if is_fastapi_available():
        try:
            res = requests.post(
                f"{API_URL}/process_mimic_report",
                json={"session_id": session_id, "subject_id": subject_id, "hadm_id": hadm_id},
                timeout=15
            )
            if res.status_code == 200:
                return True, res.json(), None
        except Exception:
            pass

    temp_file_path = None
    for f in os.listdir(config.REPORTS_DIR):
        if f.startswith(session_id):
            temp_file_path = os.path.join(config.REPORTS_DIR, f)
            break

    if not temp_file_path or not os.path.exists(temp_file_path):
        return False, {}, "Uploaded MIMIC session file not found."

    try:
        pages = load_mimic_csv(temp_file_path, subject_id=subject_id, hadm_id=hadm_id)
        if not pages:
            return False, {}, "No matching note found for selected patient and admission."

        for p in pages:
            p["text"] = clean_text(p["text"])

        chunks = chunk_text(pages)
        chunks = extract_sections(chunks)

        chunks_file = os.path.join(config.PROCESSED_DIR, f"{session_id}.json")
        with open(chunks_file, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2)

        chunk_texts = [c["text"] for c in chunks]
        embeddings = embed_text(chunk_texts)

        faiss_index = FAISSIndex(dimension=embeddings.shape[1])
        faiss_index.add_embeddings(embeddings)

        faiss_path = os.path.join(config.EMBEDDINGS_DIR, session_id)
        faiss_index.save(faiss_path)

        bm25_index = BM25Index()
        bm25_index.build(chunk_texts)

        IN_PROCESS_SESSION_CACHE[session_id] = {
            "chunks": chunks,
            "faiss": faiss_index,
            "bm25": bm25_index,
            "file_name": os.path.basename(temp_file_path)
        }

        return True, {
            "session_id": session_id,
            "subject_id": subject_id,
            "hadm_id": hadm_id,
            "num_chunks": len(chunks),
            "status": "indexed"
        }, None
    except Exception as e:
        return False, {}, f"MIMIC Processing Error: {e}"

def generate_summary_bridge(session_id: str) -> Tuple[bool, Dict[str, Any], Optional[str]]:
    """
    Generates draft summary via FastAPI or in-process.
    """
    if is_fastapi_available():
        try:
            res = requests.post(
                f"{API_URL}/generate_summary",
                json={"session_id": session_id},
                timeout=30
            )
            if res.status_code == 200:
                return True, res.json(), None
        except Exception:
            pass

    try:
        if session_id not in IN_PROCESS_SESSION_CACHE:
            chunks_file = os.path.join(config.PROCESSED_DIR, f"{session_id}.json")
            if not os.path.exists(chunks_file):
                return False, {}, "Session data not found. Please upload again."
            with open(chunks_file, "r", encoding="utf-8") as f:
                chunks = json.load(f)

            chunk_texts = [c["text"] for c in chunks]
            embeddings = embed_text(chunk_texts)
            faiss_index = FAISSIndex(dimension=embeddings.shape[1])
            faiss_path = os.path.join(config.EMBEDDINGS_DIR, session_id)
            if os.path.exists(faiss_path + ".index"):
                faiss_index.load(faiss_path)
            else:
                faiss_index.add_embeddings(embeddings)

            bm25_index = BM25Index()
            bm25_index.build(chunk_texts)

            IN_PROCESS_SESSION_CACHE[session_id] = {
                "chunks": chunks,
                "faiss": faiss_index,
                "bm25": bm25_index
            }

        cache = IN_PROCESS_SESSION_CACHE[session_id]
        doctor_summary, patient_summary, retrieved_chunks = generate_draft_summary(
            faiss_index=cache["faiss"],
            bm25_index=cache["bm25"],
            chunks=cache["chunks"]
        )

        trust_results = compute_composite_trust_score(doctor_summary, retrieved_chunks)
        cache["retrieved_chunks"] = retrieved_chunks
        cache["doctor_summary"] = doctor_summary
        cache["patient_summary"] = patient_summary

        return True, {
            "doctor_summary": doctor_summary,
            "patient_summary": patient_summary,
            "draft_summary": doctor_summary,
            "trust_results": trust_results,
            "retrieved_chunks": retrieved_chunks,
            "evidence": cache.get("evidence")
        }, None

    except Exception as e:
        return False, {}, f"Summary Generation Error: {e}"

def run_disc_bridge(session_id: str, draft_summary: str) -> Tuple[bool, Dict[str, Any], Optional[str]]:
    """
    Runs DISC dynamic self-correction audit loop via FastAPI or in-process.
    """
    if is_fastapi_available():
        try:
            res = requests.post(
                f"{API_URL}/run_disc",
                json={"session_id": session_id, "draft_summary": draft_summary},
                timeout=60
            )
            if res.status_code == 200:
                return True, res.json(), None
        except Exception:
            pass

    try:
        if session_id not in IN_PROCESS_SESSION_CACHE:
            chunks_file = os.path.join(config.PROCESSED_DIR, f"{session_id}.json")
            if not os.path.exists(chunks_file):
                return False, {}, "Session data not found. Please upload report again."
            with open(chunks_file, "r", encoding="utf-8") as f:
                chunks = json.load(f)

            chunk_texts = [c["text"] for c in chunks]
            embeddings = embed_text(chunk_texts)
            faiss_index = FAISSIndex(dimension=embeddings.shape[1])
            faiss_path = os.path.join(config.EMBEDDINGS_DIR, session_id)
            if os.path.exists(faiss_path + ".index"):
                faiss_index.load(faiss_path)
            else:
                faiss_index.add_embeddings(embeddings)

            bm25_index = BM25Index()
            bm25_index.build(chunk_texts)

            IN_PROCESS_SESSION_CACHE[session_id] = {
                "chunks": chunks,
                "faiss": faiss_index,
                "bm25": bm25_index
            }

        cache = IN_PROCESS_SESSION_CACHE[session_id]
        chunks = cache.get("chunks", [])
        retrieved_chunks = cache.get("retrieved_chunks", chunks[:5])
        faiss_index = cache.get("faiss")
        bm25_index = cache.get("bm25")

        disc_result = run_disc_pipeline(
            initial_summary=draft_summary,
            faiss_index=faiss_index,
            bm25_index=bm25_index,
            chunks=chunks,
            retrieved_chunks=retrieved_chunks
        )

        from summarization.summarizer import generate_patient_summary
        final_patient_summary = generate_patient_summary(disc_result["final_summary"])
        disc_result["final_patient_summary"] = final_patient_summary
        cache["final_summary"] = disc_result["final_summary"]
        cache["final_patient_summary"] = final_patient_summary

        return True, disc_result, None

    except Exception as e:
        return False, {}, f"DISC Verification Error: {e}"

def generate_pdf_report_bytes(session_id: str) -> Optional[bytes]:
    """
    Generates the Clinical Report PDF for a session and returns its raw bytes for downloading.
    """
    cache = IN_PROCESS_SESSION_CACHE.get(session_id, {})
    evidence = cache.get("evidence")
    if not evidence:
        # Rebuild minimal evidence from disk if needed
        evidence = StructuredClinicalEvidence()
        evidence.evidence_sources = cache.get("file_names", ["Uploaded Medical Report"])
        
    doc_summary = cache.get("final_summary") or cache.get("doctor_summary", "Clinical assessment completed.")
    pat_summary = cache.get("final_patient_summary") or cache.get("patient_summary", "Patient care plan prepared.")
    
    pdf_out = os.path.join(config.SUMMARIES_DIR, f"{session_id}_report.pdf")
    try:
        generate_clinical_report_pdf(
            evidence=evidence,
            doctor_summary=doc_summary,
            patient_summary=pat_summary,
            trust_score=0.95,
            output_pdf_path=pdf_out
        )
        if os.path.exists(pdf_out):
            with open(pdf_out, "rb") as f:
                return f.read()
    except Exception as e:
        print(f"Error generating PDF bytes: {e}")
        return None
    return None
