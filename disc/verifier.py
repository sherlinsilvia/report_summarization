from retrieval.hybrid import hybrid_retrieve
from retrieval.faiss_index import FAISSIndex
from retrieval.bm25 import BM25Index
from trust.entailment import compute_nli
import numpy as np

def verify_claims(
    questions: list[dict],
    faiss_index: FAISSIndex,
    bm25_index: BM25Index,
    chunks: list[dict]
) -> list[dict]:
    """
    Performs targeted retrieval for each question, then verifies the claim
    against the retrieved evidence using NLI.
    """
    verified_results = []
    
    for q_item in questions:
        claim = q_item["claim"]
        question = q_item["question"]
        sent_idx = q_item["sentence_idx"]
        orig_cites = q_item.get("original_citations", [])
        
        # 1. Check original cited chunks first
        found_orig_support = False
        evidence_details = []
        best_chunk_id = None
        
        for cid in orig_cites:
            if cid < len(chunks):
                orig_chunk = chunks[cid]
                nli_probs = compute_nli(orig_chunk["text"], claim)
                evidence_details.append({
                    "chunk_id": orig_chunk["chunk_id"],
                    "text": orig_chunk["text"],
                    "page": orig_chunk["page"],
                    "section": orig_chunk["section"],
                    "nli": nli_probs
                })
                if nli_probs["entailment"] > 0.4:
                    found_orig_support = True
                    best_chunk_id = orig_chunk["chunk_id"]
                    break
                    
        if found_orig_support:
            verified_results.append({
                "sentence_idx": sent_idx,
                "claim": claim,
                "question": question,
                "status": "SUPPORTED",
                "evidence": evidence_details,
                "new_citations": [best_chunk_id]
            })
            continue

        # 2. If original citation wasn't supported, search all chunks using hybrid retrieval
        evidence_chunks = hybrid_retrieve(
            query=question,
            faiss_index=faiss_index,
            bm25_index=bm25_index,
            chunks=chunks,
            top_k=3
        )
        
        entail_scores = []
        contradict_scores = []
        
        for ec in evidence_chunks:
            nli_probs = compute_nli(ec["text"], claim)
            entail_scores.append(nli_probs["entailment"])
            contradict_scores.append(nli_probs["contradiction"])
            
            evidence_details.append({
                "chunk_id": ec["chunk_id"],
                "text": ec["text"],
                "page": ec["page"],
                "section": ec["section"],
                "nli": nli_probs
            })
            
        status = "UNSUPPORTED"
        new_citations = []
        
        max_ent_idx = int(np.argmax(entail_scores)) if entail_scores else -1
        max_con_idx = int(np.argmax(contradict_scores)) if contradict_scores else -1
        
        if max_ent_idx != -1 and entail_scores[max_ent_idx] > 0.4:
            status = "SUPPORTED"
            best_chunk_id = evidence_chunks[max_ent_idx]["chunk_id"]
            new_citations = [best_chunk_id]
        elif max_con_idx != -1 and contradict_scores[max_con_idx] > 0.4:
            status = "REFUTED"
            best_chunk_id = evidence_chunks[max_con_idx]["chunk_id"]
            new_citations = [best_chunk_id]
            
        verified_results.append({
            "sentence_idx": sent_idx,
            "claim": claim,
            "question": question,
            "status": status,
            "evidence": evidence_details,
            "new_citations": new_citations
        })
        
    return verified_results
