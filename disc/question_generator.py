import json
import re
from summarization.prompt import CLAIM_EXTRACTION_PROMPT
from summarization.generator import query_llm

def generate_verification_questions(
    sentence_logs: list[dict]
) -> list[dict]:
    """
    Identifies low-confidence claims (contradicted or unsupported) from the trust audit logs,
    and generates verification questions for them.
    
    Each item in the output has:
    {
        "sentence_idx": int,
        "claim": str,
        "question": str,
        "original_citations": list[int]
    }
    """
    verification_targets = []
    
    # 1. Filter out sentences that have low confidence (is_contradicted or low entailment)
    for log in sentence_logs:
        is_weak = False
        reason = ""
        
        if log["is_contradicted"]:
            is_weak = True
            reason = "Contradicted"
        elif log["citation_entailment"] < 0.4:
            is_weak = True
            reason = "Low Support"
            
        if is_weak:
            verification_targets.append({
                "sentence_idx": log["index"],
                "claim": log["clean_sentence"],
                "reason": reason,
                "original_citations": log["citations"]
            })
            
    if not verification_targets:
        return []
        
    # 2. For each weak claim, generate a specific verification question
    # We construct a batch prompt to save LLM roundtrips
    targets_summary = "\n".join(
        f"- [ID: {t['sentence_idx']}] Claim: {t['claim']} (Original citations: {t['original_citations']})"
        for t in verification_targets
    )
    
    prompt = CLAIM_EXTRACTION_PROMPT.format(summary_text=targets_summary)
    
    questions = []
    try:
        response = query_llm(prompt, system_message="You are a clinical auditor. Output raw JSON list only.")
        # Strip any markdown code fence if the LLM output it
        json_str = re.sub(r'^```json\s*|```\s*$', '', response.strip(), flags=re.IGNORECASE)
        parsed_claims = json.loads(json_str)
        
        # Map questions back to targets by matching order or index
        for idx, item in enumerate(parsed_claims):
            if idx < len(verification_targets):
                target = verification_targets[idx]
                questions.append({
                    "sentence_idx": target["sentence_idx"],
                    "claim": target["claim"],
                    "question": item.get("question", f"Is it correct that {target['claim']}?"),
                    "original_citations": target["original_citations"]
                })
    except Exception as e:
        print(f"Failed to generate questions using LLM: {e}. Falling back to template questions.")
        # Rule-based fallback: construct standard verification questions
        for target in verification_targets:
            claim = target["claim"]
            # Basic question construction
            q = f"Is the clinical statement correct: '{claim}'?"
            questions.append({
                "sentence_idx": target["sentence_idx"],
                "claim": claim,
                "question": q,
                "original_citations": target["original_citations"]
            })
            
    return questions
