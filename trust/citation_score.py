from trust.entailment import compute_nli
import numpy as np

def compute_citation_score(
    summary_sentences: list[str],
    summary_citations: list[list[int]],
    retrieved_chunks: list[dict]
) -> tuple[float, list[dict]]:
    """
    Evaluates whether the cited references actually entail/support the sentences that cite them.
    Returns:
        (citation_score, details_list)
        - citation_score: average entailment score of cited sentences.
        - details_list: metadata on each sentence's citation validation.
    """
    if not summary_sentences:
        return 1.0, []
        
    scores = []
    details = []
    
    for i, (sent, citations) in enumerate(zip(summary_sentences, summary_citations)):
        if not citations:
            # No citations -> 0 citation support
            scores.append(0.0)
            details.append({
                "sentence_idx": i,
                "sentence": sent,
                "has_citation": False,
                "max_entailment": 0.0,
                "support_status": "No Citation"
            })
            continue
            
        sentence_entailments = []
        for cite_idx in citations:
            if cite_idx >= len(retrieved_chunks):
                continue
            
            premise = retrieved_chunks[cite_idx]["text"]
            nli_probs = compute_nli(premise, sent)
            sentence_entailments.append(nli_probs["entailment"])
            
        if sentence_entailments:
            max_ent = max(sentence_entailments)
            scores.append(max_ent)
            
            # Label support status
            if max_ent > 0.5:
                status = "Fully Supported"
            elif max_ent > 0.2:
                status = "Partially Supported"
            else:
                status = "Unsupported"
                
            details.append({
                "sentence_idx": i,
                "sentence": sent,
                "has_citation": True,
                "max_entailment": float(max_ent),
                "support_status": status,
                "best_citation_index": citations[np.argmax(sentence_entailments)]
            })
        else:
            scores.append(0.0)
            details.append({
                "sentence_idx": i,
                "sentence": sent,
                "has_citation": True,
                "max_entailment": 0.0,
                "support_status": "Invalid Citation Index"
            })
            
    citation_score = float(np.mean(scores)) if scores else 0.0
    return citation_score, details
