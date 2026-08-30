import re
import numpy as np
from retrieval.embedder import embed_text

def compute_retrieval_similarity(
    summary_sentences: list[str],
    summary_citations: list[list[int]],
    retrieved_chunks: list[dict]
) -> float:
    """
    Computes the average semantic cosine similarity between each summary sentence
    and the text of the source chunks it cites.
    """
    if not summary_sentences or not retrieved_chunks:
        return 0.0
        
    scores = []
    
    # 1. Pre-embed retrieved chunks to speed up lookup
    chunk_texts = [c["text"] for c in retrieved_chunks]
    try:
        chunk_embeddings = embed_text(chunk_texts)
    except Exception as e:
        print(f"Error embedding chunks: {e}")
        return 0.5
        
    # 2. Iterate through sentences and compute similarity with cited chunks
    for sent, citations in zip(summary_sentences, summary_citations):
        if not citations:
            # Sentence does not cite anything, assign similarity 0.0 (or default threshold)
            scores.append(0.0)
            continue
            
        try:
            sent_emb = embed_text([sent])[0]
            sent_norm = np.linalg.norm(sent_emb)
            sent_norm = 1.0 if sent_norm == 0 else sent_norm
            
            sentence_scores = []
            for cite_idx in citations:
                # Map the local cited index (e.g. [0], [1]) to the retrieved_chunks item
                if cite_idx >= len(retrieved_chunks):
                    continue
                    
                chunk_emb = chunk_embeddings[cite_idx]
                chunk_norm = np.linalg.norm(chunk_emb)
                chunk_norm = 1.0 if chunk_norm == 0 else chunk_norm
                
                similarity = np.dot(sent_emb, chunk_emb) / (sent_norm * chunk_norm)
                sentence_scores.append(similarity)
                
            if sentence_scores:
                scores.append(max(sentence_scores))  # Use maximum match if multiple chunks cited
            else:
                scores.append(0.0)
        except Exception as e:
            print(f"Error computing similarity for sentence: {sent}. {e}")
            scores.append(0.5)
            
    return float(np.mean(scores)) if scores else 0.0
