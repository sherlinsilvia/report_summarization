import os
import json
import config

# Use mock LLM for testing
config.LLM_PROVIDER = "mock"

from preprocessing.loader import load_mimic_csv
from preprocessing.cleaner import clean_text
from preprocessing.chunker import chunk_text
from preprocessing.section_extractor import extract_sections

from retrieval.embedder import embed_text
from retrieval.faiss_index import FAISSIndex
from retrieval.bm25 import BM25Index

from summarization.summarizer import generate_draft_summary
from trust.trust_score import compute_composite_trust_score
from disc.controller import run_disc_pipeline

def run_mimic_test():
    print("==================================================")
    print("     MIMIC-IV DATASET PIPELINE VERIFICATION       ")
    print("==================================================")
    
    csv_path = os.path.join("data", "reports", "mimic_sample_discharge.csv")
    print(f"\nStep 1: Reading MIMIC CSV from {csv_path}...")
    
    # Filter for Patient 1000248 (NSTEMI patient)
    subject_id = 1000248
    hadm_id = 2100852
    
    pages = load_mimic_csv(csv_path, subject_id=subject_id, hadm_id=hadm_id)
    print(f"Loaded note with {len(pages)} page segment(s) for Patient #{subject_id}.")
    if not pages:
        print("Error: Could not load filtered patient notes.")
        return
        
    # Clean, chunk, extract sections
    print("\nStep 2: Cleaning and Chunking Note...")
    for p in pages:
        p["text"] = clean_text(p["text"])
        
    chunks = chunk_text(pages)
    chunks = extract_sections(chunks)
    print(f"Created {len(chunks)} text chunks.")
    
    # Build retrieval indices
    print("\nStep 3: Indexing (Dense & Sparse)...")
    chunk_texts = [c["text"] for c in chunks]
    embeddings = embed_text(chunk_texts)
    
    faiss_index = FAISSIndex(dimension=embeddings.shape[1])
    faiss_index.add_embeddings(embeddings)
    
    bm25_index = BM25Index()
    bm25_index.build(chunk_texts)
    print("FAISS/BM25 indices established.")
    
    # Generate summary
    print("\nStep 4: Generating Grounded Summary...")
    draft_summary, patient_summary, retrieved_chunks = generate_draft_summary(
        faiss_index=faiss_index,
        bm25_index=bm25_index,
        chunks=chunks
    )
    print("\n--- DRAFT DOCTOR SUMMARY ---")
    print(draft_summary)
    print("\n--- DRAFT PATIENT SUMMARY ---")
    print(patient_summary)
    
    # Calculate trust score
    print("\nStep 5: Auditing Trust Score...")
    trust_results = compute_composite_trust_score(draft_summary, retrieved_chunks)
    initial_score = trust_results["composite_trust_score"]
    print(f"Initial Trust Score: {initial_score*100:.2f}%")
    
    # Run DISC Self-Correction
    config.TRUST_THRESHOLD = 0.85
    print(f"\nStep 6: Running DISC self-correction (Threshold: {config.TRUST_THRESHOLD*100:.2f}%)...")
    disc_results = run_disc_pipeline(
        initial_summary=draft_summary,
        faiss_index=faiss_index,
        bm25_index=bm25_index,
        chunks=chunks,
        retrieved_chunks=retrieved_chunks
    )
    
    print("\n--- FINAL CORRECTED SUMMARY ---")
    print(disc_results["final_summary"])
    print(f"Final Trust Score: {disc_results['final_trust_score']*100:.2f}%")
    print(f"Correction Triggered: {disc_results['is_corrected']}")
    
    print("\n==================================================")
    print("           MIMIC PIPELINE TEST SUCCESS            ")
    print("==================================================")

if __name__ == "__main__":
    run_mimic_test()
