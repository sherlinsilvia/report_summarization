from summarization.prompt import CORRECTION_PROMPT
from summarization.generator import query_llm

def correct_summary(
    original_summary: str,
    audit_results_str: str,
    verified_results: list[dict],
    retrieved_chunks: list[dict]
) -> str:
    """
    Prompts the LLM to rewrite the summary, correcting only the refuted/unsupported claims
    by referencing the verified audit results and correct evidence.
    """
    # 1. Compile the correct context snippets from verified results to feed into the prompt
    correct_evidence_lines = []
    
    for item in verified_results:
        # Only provide evidence for claims that need correction (REFUTED/UNSUPPORTED)
        # or that have new citations
        if item["status"] in ["REFUTED", "UNSUPPORTED", "SUPPORTED"]:
            for i, ev in enumerate(item["evidence"]):
                # Mark which claim this evidence relates to
                correct_evidence_lines.append(
                    f"Evidence for Claim ID {item['sentence_idx']}: (Page {ev['page']}, Section: {ev['section']}): {ev['text']}"
                )
                
    correct_context_str = "\n\n".join(correct_evidence_lines)
    
    # 2. Build the correction prompt
    prompt = CORRECTION_PROMPT.format(
        original_summary=original_summary,
        audit_results_str=audit_results_str,
        correct_context_str=correct_context_str
    )
    
    # 3. Query LLM to correct the summary
    corrected_summary = query_llm(
        prompt=prompt,
        system_message="You are a meticulous clinical reviewer. Edit only what is wrong. Preserve correct facts and style."
    )
    
    return corrected_summary
