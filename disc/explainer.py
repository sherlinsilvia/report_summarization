def generate_explanations(verified_results: list[dict]) -> tuple[str, list[dict]]:
    """
    Generates explainable feedback for each audited claim.
    
    Returns:
        - audit_results_str: A formatted string for prompt context.
        - explanations: A list of dicts with human-readable reasoning logs.
    """
    lines = []
    explanations = []
    
    for item in verified_results:
        sent_idx = item["sentence_idx"]
        claim = item["claim"]
        status = item["status"]
        evidence = item["evidence"]
        
        reason = ""
        evidence_text = ""
        
        if status == "SUPPORTED":
            # For supported claims, we don't need correction, but we note the verification
            best_ev = max(evidence, key=lambda x: x["nli"]["entailment"])
            reason = f"Verified: The clinical report supports this claim on Page {best_ev['page']} ({best_ev['section']})."
            evidence_text = best_ev["text"]
            
        elif status == "REFUTED":
            # For refuted, highlight contradiction
            best_ev = max(evidence, key=lambda x: x["nli"]["contradiction"])
            reason = f"Refuted: The clinical report contradicts this claim on Page {best_ev['page']} ({best_ev['section']}). Evidence text states: '{best_ev['text']}'"
            evidence_text = best_ev["text"]
            
        else: # UNSUPPORTED
            reason = "Unsupported: No backing evidence was found in the clinical report."
            # Collect the top text retrieved for context
            if evidence:
                evidence_text = " / ".join(e["text"][:150] + "..." for e in evidence)
            else:
                evidence_text = "No related text chunks found."
                
        lines.append(f"Claim ID {sent_idx}: '{claim}'\n  Audit Status: {status}\n  Reasoning: {reason}")
        
        explanations.append({
            "sentence_idx": sent_idx,
            "claim": claim,
            "status": status,
            "reasoning": reason,
            "evidence_snippet": evidence_text
        })
        
    audit_results_str = "\n\n".join(lines)
    return audit_results_str, explanations
