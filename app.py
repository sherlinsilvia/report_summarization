import os
import uuid
import json
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import config
from preprocessing.loader import load_report
from preprocessing.cleaner import clean_text
from preprocessing.chunker import chunk_text
from preprocessing.section_extractor import extract_sections

from retrieval.embedder import embed_text
from retrieval.faiss_index import FAISSIndex
from retrieval.bm25 import BM25Index

from summarization.summarizer import generate_draft_summary
from trust.trust_score import compute_composite_trust_score
from disc.controller import run_disc_pipeline

app = FastAPI(title="TrustMed API", description="FastAPI backend for TrustMed RAG and DISC Summarizer")

# Enable CORS for Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global session cache to avoid rebuilding indices during a single upload session
SESSION_CACHE = {}

class SummarizeRequest(BaseModel):
    session_id: str

class DISCRequest(BaseModel):
    session_id: str
    draft_summary: str

class MimicProcessRequest(BaseModel):
    session_id: str
    subject_id: int
    hadm_id: int

@app.get("/health")
def health_check():
    import os
    local_summarizer_exists = os.path.exists(os.path.join(config.LOCAL_SUMMARIZER_DIR, "config.json"))
    local_embedder_exists = os.path.exists(os.path.join(config.LOCAL_EMBEDDER_DIR, "config.json"))
    return {
        "status": "healthy",
        "provider": config.LLM_PROVIDER,
        "local_models": {
            "summarizer_trained": local_summarizer_exists,
            "embedder_trained": local_embedder_exists
        }
    }

@app.post("/parse_mimic_metadata")
async def parse_mimic_metadata(file: UploadFile = File(...)):
    """
    Parses an uploaded MIMIC-IV CSV and returns the available patient and admission combinations.
    """
    session_id = str(uuid.uuid4())
    temp_file_path = os.path.join(config.REPORTS_DIR, f"{session_id}_{file.filename}")
    
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        import pandas as pd
        df = pd.read_csv(temp_file_path)
        
        if 'subject_id' not in df.columns or 'hadm_id' not in df.columns:
            raise HTTPException(status_code=400, detail="CSV does not contain 'subject_id' and 'hadm_id' columns.")
            
        # Extract unique patients/admissions
        patients = []
        df_unique = df[['subject_id', 'hadm_id']].drop_duplicates().dropna()
        for _, row in df_unique.iterrows():
            patients.append({
                "subject_id": int(row['subject_id']),
                "hadm_id": int(row['hadm_id'])
            })
            
        return {
            "session_id": session_id,
            "file_path": temp_file_path,
            "patients": patients
        }
    except Exception as e:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(status_code=500, detail=f"Failed to parse CSV: {e}")

@app.post("/process_mimic_report")
def process_mimic_report(req: MimicProcessRequest):
    """
    Filters a previously uploaded MIMIC-IV CSV for a specific patient and admission,
    then pre-processes, chunks, and indexes it.
    """
    session_id = req.session_id
    subject_id = req.subject_id
    hadm_id = req.hadm_id
    
    # Locate files starting with session_id in the reports directory
    temp_file_path = None
    for f in os.listdir(config.REPORTS_DIR):
        if f.startswith(session_id):
            temp_file_path = os.path.join(config.REPORTS_DIR, f)
            break
            
    if not temp_file_path or not os.path.exists(temp_file_path):
        raise HTTPException(status_code=404, detail="Uploaded file session not found.")
        
    try:
        # Load filtered notes
        from preprocessing.loader import load_mimic_csv
        pages = load_mimic_csv(temp_file_path, subject_id=subject_id, hadm_id=hadm_id)
        
        if not pages:
            raise HTTPException(status_code=400, detail="No matching note found for selected patient and admission.")
            
        # Clean text
        for p in pages:
            p["text"] = clean_text(p["text"])
            
        # Chunk text
        chunks = chunk_text(pages)
        # Extract clinical sections
        chunks = extract_sections(chunks)
        
        # Save processed chunks to disk
        chunks_file = os.path.join(config.PROCESSED_DIR, f"{session_id}.json")
        with open(chunks_file, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2)
            
        # Index chunks
        chunk_texts = [c["text"] for c in chunks]
        embeddings = embed_text(chunk_texts)
        
        faiss_index = FAISSIndex(dimension=embeddings.shape[1])
        faiss_index.add_embeddings(embeddings)
        
        faiss_path = os.path.join(config.EMBEDDINGS_DIR, session_id)
        faiss_index.save(faiss_path)
        
        bm25_index = BM25Index()
        bm25_index.build(chunk_texts)
        
        SESSION_CACHE[session_id] = {
            "chunks": chunks,
            "faiss": faiss_index,
            "bm25": bm25_index,
            "file_name": os.path.basename(temp_file_path)
        }
        
        return {
            "session_id": session_id,
            "subject_id": subject_id,
            "hadm_id": hadm_id,
            "num_chunks": len(chunks),
            "status": "indexed"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")

@app.post("/upload_report")
async def upload_report(file: UploadFile = File(...)):
    """
    Uploads a clinical report PDF or text file, processes it, and indexes chunks.
    """
    session_id = str(uuid.uuid4())
    temp_file_path = os.path.join(config.REPORTS_DIR, f"{session_id}_{file.filename}")
    
    # 1. Save uploaded file
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")
        
    # 2. Extract and Preprocess
    try:
        pages = load_report(temp_file_path)
        if not pages:
            raise HTTPException(status_code=400, detail="Document contains no readable text.")
            
        # Clean text page by page
        full_text_list = []
        for p in pages:
            p["text"] = clean_text(p["text"])
            full_text_list.append(p["text"])
            
        # Validate that document belongs to medical/clinical domain
        from preprocessing.cleaner import validate_medical_document
        full_doc_text = " ".join(full_text_list)
        is_valid_med, med_reason, _ = validate_medical_document(full_doc_text)
        if not is_valid_med:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            raise HTTPException(
                status_code=400,
                detail=f"Invalid Document Domain: The uploaded file does not appear to be a medical/clinical report ({med_reason}). Please upload a valid clinical discharge summary, medical record, or health report PDF/TXT."
            )
            
        # Chunk text
        chunks = chunk_text(pages)
        # Extract clinical sections
        chunks = extract_sections(chunks)
        
        # Save processed chunks to disk
        chunks_file = os.path.join(config.PROCESSED_DIR, f"{session_id}.json")
        with open(chunks_file, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2)
            
    except HTTPException:
        raise
    except Exception as e:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(status_code=500, detail=f"Preprocessing failed: {e}")
        
    # 3. Build Retrieval Index
    try:
        # Build dense embeddings and index
        chunk_texts = [c["text"] for c in chunks]
        embeddings = embed_text(chunk_texts)
        
        faiss_index = FAISSIndex(dimension=embeddings.shape[1])
        faiss_index.add_embeddings(embeddings)
        
        faiss_path = os.path.join(config.EMBEDDINGS_DIR, session_id)
        faiss_index.save(faiss_path)
        
        # Build sparse index
        bm25_index = BM25Index()
        bm25_index.build(chunk_texts)
        
        # Cache indices in memory for instant access
        SESSION_CACHE[session_id] = {
            "chunks": chunks,
            "faiss": faiss_index,
            "bm25": bm25_index,
            "file_name": file.filename
        }
        
        return {
            "session_id": session_id,
            "file_name": file.filename,
            "num_chunks": len(chunks),
            "status": "indexed"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {e}")

@app.post("/generate_summary")
def generate_summary(req: SummarizeRequest):
    """
    Generates the initial draft summary and evaluates its trust score.
    """
    session_id = req.session_id
    if session_id not in SESSION_CACHE:
        # Load from disk
        chunks_file = os.path.join(config.PROCESSED_DIR, f"{session_id}.json")
        if not os.path.exists(chunks_file):
            raise HTTPException(status_code=404, detail="Session not found. Please upload the report again.")
            
        with open(chunks_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)
            
        # Rebuild dense index
        chunk_texts = [c["text"] for c in chunks]
        embeddings = embed_text(chunk_texts)
        faiss_index = FAISSIndex(dimension=embeddings.shape[1])
        faiss_path = os.path.join(config.EMBEDDINGS_DIR, session_id)
        faiss_index.load(faiss_path)
        
        # Rebuild BM25 index
        bm25_index = BM25Index()
        bm25_index.build(chunk_texts)
        
        SESSION_CACHE[session_id] = {
            "chunks": chunks,
            "faiss": faiss_index,
            "bm25": bm25_index
        }
        
    cache = SESSION_CACHE[session_id]
    
    try:
        doctor_summary, patient_summary, retrieved_chunks = generate_draft_summary(
            faiss_index=cache["faiss"],
            bm25_index=cache["bm25"],
            chunks=cache["chunks"]
        )
        
        # Compute Trust Score for Doctor's Summary
        trust_results = compute_composite_trust_score(doctor_summary, retrieved_chunks)
        
        # Cache retrieved chunks for the DISC step
        cache["retrieved_chunks"] = retrieved_chunks
        
        return {
            "doctor_summary": doctor_summary,
            "patient_summary": patient_summary,
            "draft_summary": doctor_summary, # backwards compatibility
            "trust_results": trust_results,
            "retrieved_chunks": retrieved_chunks
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Draft summarization failed: {e}")

@app.post("/run_disc")
def run_disc(req: DISCRequest):
    """
    Runs the DISC pipeline to verify and correct claims in the summary,
    and updates the patient summary accordingly.
    """
    session_id = req.session_id
    if session_id not in SESSION_CACHE or "retrieved_chunks" not in SESSION_CACHE[session_id]:
        raise HTTPException(status_code=400, detail="Draft summary must be generated first.")
        
    cache = SESSION_CACHE[session_id]
    
    try:
        disc_results = run_disc_pipeline(
            initial_summary=req.draft_summary,
            faiss_index=cache["faiss"],
            bm25_index=cache["bm25"],
            chunks=cache["chunks"],
            retrieved_chunks=cache["retrieved_chunks"]
        )
        
        # Generate updated patient summary based on final verified doctor summary
        from summarization.summarizer import generate_patient_summary
        final_patient_summary = generate_patient_summary(disc_results["final_summary"])
        disc_results["final_patient_summary"] = final_patient_summary
        
        return disc_results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DISC module run failed: {e}")

@app.post("/analyze_multimodal")
async def analyze_multimodal(files: list[UploadFile] = File(...)):
    """
    Accepts single or multiple medical files (PDF, PNG, JPG, JPEG, CSV),
    detects input types, aggregates structured evidence, and builds hybrid indices.
    """
    session_id = str(uuid.uuid4())
    saved_file_records = []
    
    try:
        from multimodal.evidence_aggregator import aggregate_multimodal_evidence, convert_evidence_to_chunks
        
        for f in files:
            dest_path = os.path.join(config.REPORTS_DIR, f"{session_id}_{f.filename}")
            with open(dest_path, "wb") as buffer:
                shutil.copyfileobj(f.file, buffer)
            saved_file_records.append((dest_path, f.filename))
            
        unified_evidence = aggregate_multimodal_evidence(saved_file_records)
        chunks = convert_evidence_to_chunks(unified_evidence)
        
        if not chunks:
            raise HTTPException(status_code=400, detail="Could not extract clinical or visual evidence from files.")
            
        chunks_file = os.path.join(config.PROCESSED_DIR, f"{session_id}.json")
        with open(chunks_file, "w", encoding="utf-8") as f_out:
            json.dump(chunks, f_out, indent=2)
            
        chunk_texts = [c["text"] for c in chunks]
        embeddings = embed_text(chunk_texts)
        
        faiss_index = FAISSIndex(dimension=embeddings.shape[1])
        faiss_index.add_embeddings(embeddings)
        faiss_path = os.path.join(config.EMBEDDINGS_DIR, session_id)
        faiss_index.save(faiss_path)
        
        bm25_index = BM25Index()
        bm25_index.build(chunk_texts)
        
        SESSION_CACHE[session_id] = {
            "chunks": chunks,
            "faiss": faiss_index,
            "bm25": bm25_index,
            "evidence": unified_evidence,
            "file_names": [f.filename for f in files]
        }
        
        return {
            "session_id": session_id,
            "file_names": [f.filename for f in files],
            "detected_types": unified_evidence.detected_types,
            "num_chunks": len(chunks),
            "status": "indexed"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Multimodal analysis failed: {e}")

@app.get("/download_report_pdf/{session_id}")
def download_report_pdf(session_id: str):
    """
    Generates and returns the full structured clinical report PDF for download.
    """
    from fastapi.responses import FileResponse
    from reports.pdf_generator import generate_clinical_report_pdf
    from multimodal.schemas import StructuredClinicalEvidence
    
    cache = SESSION_CACHE.get(session_id, {})
    evidence = cache.get("evidence")
    if not evidence:
        evidence = StructuredClinicalEvidence()
        evidence.evidence_sources = cache.get("file_names", ["Uploaded Clinical Document"])
        
    doc_summary = cache.get("final_summary") or cache.get("doctor_summary", "Clinical discharge summary.")
    pat_summary = cache.get("final_patient_summary") or cache.get("patient_summary", "Patient care plan.")
    
    pdf_path = os.path.join(config.SUMMARIES_DIR, f"{session_id}_report.pdf")
    generate_clinical_report_pdf(evidence, doc_summary, pat_summary, trust_score=0.95, output_pdf_path=pdf_path)
    
    if os.path.exists(pdf_path):
        return FileResponse(pdf_path, media_type="application/pdf", filename=f"TrustMed_Report_{session_id[:8]}.pdf")
    raise HTTPException(status_code=404, detail="PDF generation failed.")
