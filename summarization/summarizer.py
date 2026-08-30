from summarization.prompt import DOCTOR_SUMMARY_PROMPT, PATIENT_SUMMARY_PROMPT
from summarization.generator import query_llm
from retrieval.hybrid import hybrid_retrieve
from retrieval.faiss_index import FAISSIndex
from retrieval.bm25 import BM25Index

def generate_patient_summary(doctor_summary: str) -> str:
    """
    Generates a patient-friendly explanation based on the doctor summary.
    """
    prompt = PATIENT_SUMMARY_PROMPT.format(clinical_summary=doctor_summary)
    patient_summary = query_llm(
        prompt, 
        system_message="You are a compassionate doctor explaining medical care to a patient in simple Grade 6 English."
    )
    return patient_summary

def generate_draft_summary(
    faiss_index: FAISSIndex,
    bm25_index: BM25Index,
    chunks: list[dict],
    document_name: str = "Clinical Report"
) -> tuple[str, str, list[dict]]:
    """
    Coordinates hybrid retrieval and prompts the LLM to generate the initial
    doctor discharge summary and patient-friendly explanation.
    
    Returns: (doctor_summary_text, patient_summary_text, retrieved_chunks_list)
    """
    # 1. Retrieve most critical chunks from the clinical report
    query = "Patient demographics, chief complaint, diagnosis, medical history, investigations, treatment given, hospital course, discharge medications, follow-up."
    
    retrieved_chunks = hybrid_retrieve(
        query=query,
        faiss_index=faiss_index,
        bm25_index=bm25_index,
        chunks=chunks,
        top_k=8
    )
    
    # Sort retrieved chunks by chunk_id to keep them in reading order
    retrieved_chunks.sort(key=lambda x: x["chunk_id"])
    
    # 2. Format context segments for the prompt
    context_lines = []
    for i, chunk in enumerate(retrieved_chunks):
        context_lines.append(f"[{i}] (Page {chunk['page']}, Section: {chunk['section']}): {chunk['text']}")
        
    context_str = "\n\n".join(context_lines)
    
    # 3. Generate Doctor Discharge Summary
    doctor_prompt = DOCTOR_SUMMARY_PROMPT.format(context_str=context_str)
    doctor_summary = query_llm(doctor_prompt, system_message="You are an experienced physician generating a structured discharge summary.")
    
    # 4. Generate Patient Explanation
    patient_summary = generate_patient_summary(doctor_summary)
    
    return doctor_summary, patient_summary, retrieved_chunks
