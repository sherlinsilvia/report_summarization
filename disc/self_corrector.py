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
    sentence_citations = {}
    
    for item in verified_results:
        sent_idx = item.get("sentence_idx")
        new_cites = item.get("new_citations", [])
        
        if new_cites:
            sentence_citations[sent_idx] = new_cites[0]
        elif item.get("evidence"):
            # Pick highest entailment evidence chunk
            best_ev = max(item["evidence"], key=lambda x: x.get("nli", {}).get("entailment", 0))
            sentence_citations[sent_idx] = best_ev["chunk_id"]

    default_chunk_id = retrieved_chunks[0]["chunk_id"] if retrieved_chunks else 0

    lines = original_summary.split("\n")
    corrected_lines = []
    sentence_counter = 0

    for line in lines:
        line_str = line.strip()
        if not line_str:
            corrected_lines.append("")
            continue
            
        # Match header line like: 1. **Patient Information**: <text>
        # Or line with no header: <text>
        m = re.match(r'^(\d+\.\s*\*\*[^*]+\*\*:?\s*)(.*)', line_str)
        if m:
            prefix = m.group(1)
            text_body = m.group(2)
        else:
            prefix = ""
            text_body = line_str

        if not text_body.strip():
            corrected_lines.append(line_str)
            continue

        # Split text_body into clean sentences
        sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text_body.strip()) if s.strip()]
        new_sents = []
        for s in sents:
            # Extract existing citation tag if present
            existing_cites = re.findall(r'\[(\d+)\]', s)
            
            # Strip ALL previous citation brackets [c_num]
            s_clean = re.sub(r'\s*\[\d+\]', '', s).strip()
            if not s_clean:
                continue
            
            # Determine target citation chunk ID
            if sentence_counter in sentence_citations:
                target_chunk = sentence_citations[sentence_counter]
            elif existing_cites and int(existing_cites[0]) < len(retrieved_chunks):
                # Preserve valid existing citation
                target_chunk = int(existing_cites[0])
            else:
                target_chunk = default_chunk_id
            
            new_sents.append(f"{s_clean} [{target_chunk}]")
            sentence_counter += 1

        corrected_lines.append(f"{prefix}{' '.join(new_sents)}")

    return "\n".join(corrected_lines)
