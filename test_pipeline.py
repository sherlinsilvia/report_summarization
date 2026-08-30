import os
import json
import config

# Set LLM provider to mock by default for standalone testing
config.LLM_PROVIDER = "mock"

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

def run_test():
    print("==================================================")
    print("      TRUSTMED END-TO-END PIPELINE DRY RUN        ")
    print("==================================================")
    
    # 1. Load sample report
    report_path = os.path.join("data", "reports", "sample_report.txt")
    print(f"\nStep 1: Loading clinical report from {report_path}...")
    pages = load_report(report_path)
    print(f"Loaded {len(pages)} pages.")
    
    # 2. Clean, chunk, and extract sections
    print("\nStep 2: Preprocessing and Chunking...")
    for p in pages:
        p["text"] = clean_text(p["text"])
    chunks = chunk_text(pages)
    chunks = extract_sections(chunks)
    print(f"Created {len(chunks)} text chunks.")
    for idx, c in enumerate(chunks[:3]):
        print(f"  Chunk {idx} (Page {c['page']}, Section: {c['section']}): {c['text'][:80]}...")
        
    # 3. Build Retrieval Index
    print("\nStep 3: Embedding and Indexing (Dense & Sparse)...")
    chunk_texts = [c["text"] for c in chunks]
    embeddings = embed_text(chunk_texts)
    
    faiss_index = FAISSIndex(dimension=embeddings.shape[1])
    faiss_index.add_embeddings(embeddings)
    
    bm25_index = BM25Index()
    bm25_index.build(chunk_texts)
    print("FAISS and BM25 indices built successfully.")
    
    # 4. Generate Draft Summary
    print("\nStep 4: Generating RAG Draft Summary...")
    draft_summary, patient_summary, retrieved_chunks = generate_draft_summary(
        faiss_index=faiss_index,
        bm25_index=bm25_index,
        chunks=chunks
    )
    print("\n--- DRAFT DOCTOR SUMMARY ---")
    print(draft_summary)
    print("\n--- DRAFT PATIENT SUMMARY ---")
    print(patient_summary)
    
    # 5. Evaluate Trust Score
    print("\nStep 5: Evaluating Composite Trust Score for Draft...")
    trust_results = compute_composite_trust_score(draft_summary, retrieved_chunks)
    initial_score = trust_results["composite_trust_score"]
    print(f"Initial Trust Score: {initial_score*100:.2f}%")
    print("Individual Metric Scores:")
    for metric, score in trust_results["scores"].items():
        print(f"  - {metric}: {score*100:.2f}%")
        
    # 6. Run DISC Correction
    # We set a high threshold in config to force DISC to run even if score is high
    config.TRUST_THRESHOLD = 0.95
    print(f"\nStep 6: Running DISC Module (Configured Threshold: {config.TRUST_THRESHOLD*100:.2f}%)...")
    
    disc_results = run_disc_pipeline(
        initial_summary=draft_summary,
        faiss_index=faiss_index,
        bm25_index=bm25_index,
        chunks=chunks,
        retrieved_chunks=retrieved_chunks
    )
    
    print("\n--- FINAL VERIFIED SUMMARY ---")
    print(disc_results["final_summary"])
    print(f"Final Trust Score: {disc_results['final_trust_score']*100:.2f}%")
    print(f"Correction Triggered: {disc_results['is_corrected']}")
    print(f"Score Progression: {' -> '.join(f'{s*100:.1f}%' for s in disc_results['trust_score_history'])}")
    
    print("\n==================================================")
    print("               DRY RUN SUCCESSFUL                 ")
    print("==================================================")

if __name__ == "__main__":
    run_test()
