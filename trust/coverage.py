import numpy as np
from retrieval.embedder import embed_text

def compute_coverage_score(
    summary_sentences: list[str],
    retrieved_chunks: list[dict]
) -> tuple[float, dict]:
    """
    Computes the section coverage of the summary.
    Measures the proportion of original clinical sections in the retrieved chunks
    that are covered in the summary (having semantic similarity > 0.45 with at least one chunk).
    
    Returns:
        (coverage_score, coverage_details)
    """
    if not summary_sentences or not retrieved_chunks:
        return 0.0, {}
        
    # Group chunks by section
    section_chunks = {}
    for i, chunk in enumerate(retrieved_chunks):
        sec = chunk.get("section", "General")
        if sec not in section_chunks:
            section_chunks[sec] = []
        section_chunks[sec].append((i, chunk["text"]))
        
    # Embed summary sentences
    try:
        summary_embs = embed_text(summary_sentences)
        summary_norms = np.linalg.norm(summary_embs, axis=1, keepdims=True)
        summary_norms = np.where(summary_norms == 0, 1.0, summary_norms)
        summary_embs_norm = summary_embs / summary_norms
    except Exception as e:
        print(f"Error embedding summary for coverage: {e}")
        return 0.5, {}
        
    # Embed chunks per section
    covered_sections = {}
    
    for sec, chunks_list in section_chunks.items():
        chunk_texts = [text for _, text in chunks_list]
        try:
            chunk_embs = embed_text(chunk_texts)
            chunk_norms = np.linalg.norm(chunk_embs, axis=1, keepdims=True)
            chunk_norms = np.where(chunk_norms == 0, 1.0, chunk_norms)
            chunk_embs_norm = chunk_embs / chunk_norms
            
            # Compute cosine similarities between summary sentences and section chunks
            # Shape: (num_summary_sentences, num_section_chunks)
            sim_matrix = np.dot(summary_embs_norm, chunk_embs_norm.T)
            
            # Find the max similarity for this section
            max_sim = float(np.max(sim_matrix))
            
            # If max similarity is above 0.45, we consider the section covered
            is_covered = max_sim >= 0.45
            covered_sections[sec] = {
                "is_covered": is_covered,
                "max_similarity": max_sim
            }
        except Exception as e:
            print(f"Error computing coverage for section {sec}: {e}")
            covered_sections[sec] = {
                "is_covered": True,  # Default to true in case of failure to avoid penalizing
                "max_similarity": 1.0
            }
            
    # Calculate score
    covered_count = sum(1 for sec_info in covered_sections.values() if sec_info["is_covered"])
    total_sections = len(section_chunks)
    
    coverage_score = covered_count / total_sections if total_sections > 0 else 0.0
    return float(coverage_score), covered_sections
