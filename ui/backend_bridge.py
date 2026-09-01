"""
TrustMed Backend Bridge
Seamlessly connects Streamlit UI to FastAPI backend when available,
or executes pipeline directly in-process when deployed on Streamlit Cloud / standalone environments.
"""

import os
import uuid
import json
import shutil
import requests
from typing import Dict, Any, Tuple, Optional

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

API_URL = os.getenv("API_URL", "http://localhost:8000")

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

def upload_and_index_report(file_name: str, file_bytes: bytes, mime_type: str) -> Tuple[bool, Dict[str, Any], Optional[str]]:
    """
    Uploads and indexes a clinical report. If FastAPI server is not active (e.g. Streamlit Cloud),
    processes and indexes directly in-process.
    """
    # 1. Try Remote / Local FastAPI Server if available
    if is_fastapi_available():
        try:
            files = {"file": (file_name, file_bytes, mime_type)}
            res = requests.post(f"{API_URL}/upload_report", files=files, timeout=15)
            if res.status_code == 200:
                return True, res.json(), None
            else:
                try:
                    err = res.json().get("detail", res.text)
                except Exception:
                    err = res.text
                return False, {}, err
        except Exception:
            pass

    # 2. In-Process Standalone Execution (Streamlit Cloud Deployment)
    session_id = str(uuid.uuid4())
    temp_file_path = os.path.join(config.REPORTS_DIR, f"{session_id}_{file_name}")

    try:
        with open(temp_file_path, "wb") as f:
            f.write(file_bytes)

        pages = load_report(temp_file_path)
        if not pages:
            return False, {}, "Document contains no readable text."

        full_text_list = []
        for p in pages:
            p["text"] = clean_text(p["text"])
            full_text_list.append(p["text"])

        # Validate medical domain
        full_doc_text = " ".join(full_text_list)
        is_valid_med, med_reason, _ = validate_medical_document(full_doc_text)
        if not is_valid_med:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            return False, {}, f"Invalid Document Domain: The uploaded file does not appear to be a medical report ({med_reason})."

        chunks = chunk_text(pages)
        chunks = extract_sections(chunks)

        # Save processed chunks
        chunks_file = os.path.join(config.PROCESSED_DIR, f"{session_id}.json")
        with open(chunks_file, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2)

        # Build retrieval indices
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
            "file_name": file_name
        }

        return True, {
            "session_id": session_id,
            "file_name": file_name,
            "num_chunks": len(chunks),
            "status": "indexed"
        }, None

    except Exception as e:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        return False, {}, f"Document Processing Error: {str(e)}"

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

    # In-process MIMIC parse
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

    # In-process MIMIC processing
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

    # In-process summary generation
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

        return True, {
            "doctor_summary": doctor_summary,
            "patient_summary": patient_summary,
            "draft_summary": doctor_summary,
            "trust_results": trust_results,
            "retrieved_chunks": retrieved_chunks
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

    # In-process DISC execution
    try:
        cache = IN_PROCESS_SESSION_CACHE.get(session_id, {})
        retrieved_chunks = cache.get("retrieved_chunks")

        if not retrieved_chunks:
            chunks_file = os.path.join(config.PROCESSED_DIR, f"{session_id}.json")
            with open(chunks_file, "r", encoding="utf-8") as f:
                retrieved_chunks = json.load(f)[:5]

        faiss_index = cache.get("faiss")
        bm25_index = cache.get("bm25")

        disc_result = run_disc_pipeline(
            draft_summary=draft_summary,
            retrieved_chunks=retrieved_chunks,
            faiss_index=faiss_index,
            bm25_index=bm25_index
        )
        return True, disc_result, None

    except Exception as e:
        return False, {}, f"DISC Verification Error: {e}"
