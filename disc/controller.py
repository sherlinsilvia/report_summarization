from trust.trust_score import compute_composite_trust_score
from disc.question_generator import generate_verification_questions
from disc.verifier import verify_claims
from disc.explainer import generate_explanations
from disc.self_corrector import correct_summary
from retrieval.faiss_index import FAISSIndex
from retrieval.bm25 import BM25Index
import config

def run_disc_pipeline(
    initial_summary: str,
    faiss_index: FAISSIndex,
    bm25_index: BM25Index,
    chunks: list[dict],
    retrieved_chunks: list[dict]
) -> dict:
    """
    Orchestrates the Dynamic Intelligent Self-Correction (DISC) workflow.
    Iteratively verifies and rewrites low-confidence claims until the trust score
    meets the threshold or max trials is reached.
    
    Returns: A dictionary with:
        - final_summary: str
        - final_trust_score: float
        - trust_score_history: list[float]
        - audit_history: list[dict] # detailed logs of each correction step
        - is_corrected: bool
    """
    current_summary = initial_summary
    current_chunks = retrieved_chunks.copy()
    
    trust_history = []
    audit_history = []
    
    # 1. Compute initial trust score
    initial_audit = compute_composite_trust_score(current_summary, current_chunks)
    initial_score = initial_audit["composite_trust_score"]
    trust_history.append(initial_score)
    
    print(f"Initial summary trust score: {initial_score:.2f} (Threshold: {config.TRUST_THRESHOLD:.2f})")
    
    if initial_score >= config.TRUST_THRESHOLD:
        print("Initial summary meets the trust threshold. No self-correction required.")
        return {
            "final_summary": current_summary,
            "final_trust_score": initial_score,
            "trust_score_history": trust_history,
            "audit_history": [{"trial": 0, "score": initial_score, "details": initial_audit, "actions": "No correction needed"}],
            "is_corrected": False
        }
        
    # 2. Iterative Self-Correction loop
    trial = 0
    is_corrected = False
    
    while trial < config.MAX_DISC_TRIALS:
        trial += 1
        print(f"\n--- DISC Trial {trial}/{config.MAX_DISC_TRIALS} ---")
        
        # Get audit details
        audit_details = compute_composite_trust_score(current_summary, current_chunks)
        sentence_logs = audit_details["details"]["sentences"]
        
        # a. Generate verification questions for low-confidence claims (or deep audit on trial > 1)
        force_all_flag = (trial > 1)
        questions = generate_verification_questions(sentence_logs, force_all=force_all_flag)
        print(f"Generated {len(questions)} verification questions for claims.")
        
        if not questions:
            print("No weak claims identified. Ending correction loop.")
            break
            
        # b. Verify claims against original report chunks
        verified_results = verify_claims(questions, faiss_index, bm25_index, chunks)
        
        # Add new evidence chunks to our active local retrieved_chunks context pool
        # so the summarizer can use them.
        new_evidence_added = False
        for vr in verified_results:
            for ev in vr["evidence"]:
                # Check if this chunk is already in current_chunks
                if not any(c["chunk_id"] == ev["chunk_id"] for c in current_chunks):
                    # Format as context chunk
                    new_chunk = {
                        "chunk_id": ev["chunk_id"],
                        "text": ev["text"],
                        "page": ev["page"],
                        "section": ev["section"]
                    }
                    current_chunks.append(new_chunk)
                    new_evidence_added = True
                    
        # Re-sort local retrieved chunks by chunk_id
        if new_evidence_added:
            current_chunks.sort(key=lambda x: x["chunk_id"])
            
        # c. Explain verification results
        audit_results_str, explanations = generate_explanations(verified_results)
        
        # d. Rewrite unsupported/contradicted statements
        corrected = correct_summary(
            original_summary=current_summary,
            audit_results_str=audit_results_str,
            verified_results=verified_results,
            retrieved_chunks=current_chunks
        )
        
        # e. Re-evaluate trust
        new_audit = compute_composite_trust_score(corrected, current_chunks)
        new_score = new_audit["composite_trust_score"]
        
        print(f"Trial {trial} corrected summary trust score: {new_score:.2f}")
        
        # Log this trial
        audit_history.append({
            "trial": trial,
            "previous_summary": current_summary,
            "corrected_summary": corrected,
            "previous_score": trust_history[-1],
            "new_score": new_score,
            "questions": questions,
            "verifications": verified_results,
            "explanations": explanations,
            "audit_details": new_audit
        })
        
        current_summary = corrected
        trust_history.append(new_score)
        is_corrected = True
        
        # Exit if score passes threshold
        if new_score >= config.TRUST_THRESHOLD:
            print(f"Trust score {new_score:.2f} meets or exceeds threshold {config.TRUST_THRESHOLD:.2f}.")
            break
            
    return {
        "final_summary": current_summary,
        "final_trust_score": trust_history[-1],
        "trust_score_history": trust_history,
        "audit_history": audit_history,
        "is_corrected": is_corrected
    }
