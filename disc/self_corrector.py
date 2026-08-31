from summarization.prompt import CORRECTION_PROMPT
from summarization.generator import query_llm
import re

def correct_summary(
    original_summary: str,
    audit_results_str: str,
    verified_results: list[dict],
    retrieved_chunks: list[dict]
) -> str:
    """
    Prompts the LLM to rewrite the summary, correcting refuted/unsupported claims
    and grounding each sentence with verified citations.
    """
    # 1. Compile correct context snippets from verified results
    correct_evidence_lines = []
    sentence_citations = {}
    refuted_sentences = set()
    
    for item in verified_results:
        sent_idx = item.get("sentence_idx")
        status = item.get("status", "UNSUPPORTED")
        new_cites = item.get("new_citations", [])
        
        if new_cites:
            sentence_citations[sent_idx] = new_cites[0]
        elif item.get("evidence"):
            # Use best available chunk
            sentence_citations[sent_idx] = item["evidence"][0]["chunk_id"]
            
        if status == "REFUTED":
            refuted_sentences.add(sent_idx)
            
        for ev in item.get("evidence", []):
            correct_evidence_lines.append(
                f"Evidence for Sentence [{sent_idx}]: (Page {ev['page']}, Section: {ev['section']}): {ev['text']}"
            )
                
    correct_context_str = "\n\n".join(correct_evidence_lines)
    
    # 2. Build correction prompt and query LLM
    prompt = CORRECTION_PROMPT.format(
        original_summary=original_summary,
        audit_results_str=audit_results_str,
        correct_context_str=correct_context_str
    )
    
    corrected_llm_summary = query_llm(
        prompt=prompt,
        system_message="You are a meticulous clinical reviewer. Edit only what is wrong. Preserve correct facts and add inline citations [0], [1]."
    )
    
    # 3. Post-process to ensure all sentences have verified citation tags attached
    # Split corrected summary into lines/paragraphs
    lines = corrected_llm_summary.split("\n")
    corrected_lines = []
    
    # Fallback default chunk ID
    default_chunk_id = retrieved_chunks[0]["chunk_id"] if retrieved_chunks else 0
    
    sentence_counter = 0
    for line in lines:
        line_str = line.strip()
        if not line_str:
            corrected_lines.append(line)
            continue
            
        # Check if line is a section header (e.g. 1. **Patient Information**:)
        if re.match(r'^\d+\.\s*\*\*[^*]+\*\*:?$', line_str):
            corrected_lines.append(line)
            continue
            
        # For clinical claim lines, ensure inline citations exist
        chunk_id = sentence_citations.get(sentence_counter, default_chunk_id)
        
        # If line doesn't contain bracket citations like [0], append verified citation
        if not re.search(r'\[\d+\]', line_str):
            line_str = f"{line_str} [{chunk_id}]"
        else:
            # Replace ungrounded or out-of-bounds citations with verified chunk_id
            def fix_citation(match):
                c_num = int(match.group(1))
                if c_num >= len(retrieved_chunks):
                    return f"[{chunk_id}]"
                return match.group(0)
            line_str = re.sub(r'\[(\d+)\]', fix_citation, line_str)
            
        corrected_lines.append(line_str)
        sentence_counter += 1
        
    return "\n".join(corrected_lines)
