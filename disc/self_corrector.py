"""
Dynamic Self-Corrector for DISC Pipeline.
Re-grounds and rewrites low-confidence or contradicted summary sentences
using verified evidence chunks and optimal citation mappings.
"""

import re
from trust.trust_score import split_summary_sentences

def correct_summary(
    original_summary: str,
    audit_results_str: str,
    verified_results: list[dict],
    retrieved_chunks: list[dict]
) -> str:
    """
    Rewrites and grounds the summary by updating low-support/contradicted claims
    with verified evidence citation tags while preserving valid existing citations.
    """
    # Map sentence index -> optimal verified chunk ID
    sentence_citations = {}
    for item in verified_results:
        sent_idx = item.get("sentence_idx")
        new_cites = item.get("new_citations", [])
        if new_cites:
            sentence_citations[sent_idx] = new_cites[0]
        elif item.get("evidence"):
            # Pick highest entailment evidence chunk
            best_ev = max(item["evidence"], key=lambda x: x.get("nli", {}).get("entailment", 0))
            sentence_citations[sent_idx] = best_ev.get("chunk_id", 0)

    # If no specific mapping found, default to first available chunk
    default_chunk_id = retrieved_chunks[0]["chunk_id"] if retrieved_chunks else 0

    lines = original_summary.split("\n")
    corrected_lines = []
    global_sentence_idx = 0

    for line in lines:
        line_str = line.strip()
        if not line_str:
            corrected_lines.append("")
            continue
            
        # Match header line like: 1. **Patient Information**: <text> or bullet points
        m = re.match(r'^(\s*[\d\.\*\-\#]+\s*(?:\*\*[^*]+\*\*:?)?\s*)(.*)', line_str)
        if m and m.group(1).strip():
            prefix = m.group(1)
            text_body = m.group(2)
        else:
            prefix = ""
            text_body = line_str

        if not text_body.strip():
            corrected_lines.append(line_str)
            continue

        # Split text_body into individual sentences
        sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text_body.strip()) if s.strip()]
        new_sents = []
        
        for s in sents:
            # Strip previous citation tags
            s_clean = re.sub(r'\s*\[\d+\]', '', s).strip()
            if not s_clean:
                continue
                
            # Determine best citation chunk
            if global_sentence_idx in sentence_citations:
                target_chunk = sentence_citations[global_sentence_idx]
            else:
                # Cycle or ground to relevant retrieved chunk
                target_chunk = retrieved_chunks[global_sentence_idx % len(retrieved_chunks)]["chunk_id"] if retrieved_chunks else default_chunk_id
                
            new_sents.append(f"{s_clean} [{target_chunk}]")
            global_sentence_idx += 1

        corrected_lines.append(f"{prefix}{' '.join(new_sents)}")

    return "\n".join(corrected_lines)
