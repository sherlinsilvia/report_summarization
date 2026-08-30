from trust.entailment import compute_nli

def evaluate_hallucinations(
    summary_sentences: list[str],
    summary_citations: list[list[int]],
    retrieved_chunks: list[dict]
) -> tuple[float, list[dict]]:
    """
    Evaluates whether each summary sentence is contradicted by its cited chunks.
    If no citation exists, checks against all retrieved chunks.
    
    Returns:
        (hallucination_score, details_list)
        - hallucination_score: 1.0 - (contradicted_sentences / total_sentences)
        - details_list: metadata on each sentence classification (NLI results)
    """
    if not summary_sentences:
        return 1.0, []
        
    contradicted_count = 0
    details = []
    
    for i, (sent, citations) in enumerate(zip(summary_sentences, summary_citations)):
        # Determine premises to check against
        premises = []
        if citations:
            for cite_idx in citations:
                if cite_idx < len(retrieved_chunks):
                    premises.append((cite_idx, retrieved_chunks[cite_idx]["text"]))
        else:
            # Check against all retrieved chunks if no citation
            premises = list(enumerate([c["text"] for c in retrieved_chunks]))
            
        is_contradicted = False
        max_contradiction = 0.0
        best_match_info = {}
        
        # Check against each premise
        for cite_idx, premise_text in premises:
            nli_probs = compute_nli(premise_text, sent)
            
            # Record NLI details
            if nli_probs["contradiction"] > max_contradiction:
                max_contradiction = nli_probs["contradiction"]
                best_match_info = {
                    "cited_index": cite_idx,
                    "entailment": nli_probs["entailment"],
                    "contradiction": nli_probs["contradiction"],
                    "neutral": nli_probs["neutral"]
                }
                
            # If any premise directly contradicts it, count as contradiction
            # Threshold of 0.5
            if nli_probs["contradiction"] > 0.5:
                is_contradicted = True
                break
                
        if is_contradicted:
            contradicted_count += 1
            
        details.append({
            "sentence_idx": i,
            "sentence": sent,
            "is_contradicted": is_contradicted,
            "max_contradiction": max_contradiction,
            "best_match": best_match_info
        })
        
    hallucination_score = 1.0 - (contradicted_count / len(summary_sentences))
    return float(hallucination_score), details
