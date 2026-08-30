import re
import numpy as np
from retrieval.embedder import embed_text

def compute_bert_score(
    sentences: list[str],
    retrieved_chunks: list[dict]
) -> tuple[float, dict]:
    """
    Computes BERTScore (Precision, Recall, F1) between generated summary sentences
    and retrieved reference evidence chunks using contextual embedding & token similarity.
    
    Returns:
        - f1_score: float (0.0 to 1.0)
        - details: dict containing precision, recall, f1, and per-sentence scores
    """
    if not sentences or not retrieved_chunks:
        return 0.0, {"precision": 0.0, "recall": 0.0, "f1": 0.0, "sentence_scores": []}
        
    chunk_texts = [c.get("text", "") for c in retrieved_chunks if c.get("text")]
    if not chunk_texts:
        return 0.0, {"precision": 0.0, "recall": 0.0, "f1": 0.0, "sentence_scores": []}
        
    # Get embeddings for summary sentences and reference chunks
    sent_embeddings = embed_text(sentences)      # Shape: (num_sentences, dim)
    chunk_embeddings = embed_text(chunk_texts)   # Shape: (num_chunks, dim)
    
    # Cosine similarity matrix: (num_sentences, num_chunks)
    sent_norms = np.linalg.norm(sent_embeddings, axis=1, keepdims=True)
    sent_norms[sent_norms == 0] = 1e-10
    sent_embeddings_norm = sent_embeddings / sent_norms
    
    chunk_norms = np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)
    chunk_norms[chunk_norms == 0] = 1e-10
    chunk_embeddings_norm = chunk_embeddings / chunk_norms
    
    emb_sim_matrix = np.dot(sent_embeddings_norm, chunk_embeddings_norm.T)
    emb_sim_matrix = np.clip(emb_sim_matrix, 0.0, 1.0)
    
    # Calculate token n-gram overlap matrix for robust semantic matching
    num_sents = len(sentences)
    num_chunks = len(chunk_texts)
    combined_sim = np.zeros((num_sents, num_chunks), dtype=np.float32)
    
    for i, sent in enumerate(sentences):
        sent_tokens = set(re.findall(r'\w+', sent.lower()))
        for j, chunk in enumerate(chunk_texts):
            chunk_tokens = set(re.findall(r'\w+', chunk.lower()))
            if sent_tokens and chunk_tokens:
                overlap = len(sent_tokens.intersection(chunk_tokens))
                token_sim = overlap / max(len(sent_tokens), 1)
            else:
                token_sim = 0.0
            # Combine embedding similarity and token precision
            combined_sim[i, j] = max(float(emb_sim_matrix[i, j]), token_sim)

    # Precision: average max similarity per summary sentence across chunks
    max_sim_per_sentence = np.max(combined_sim, axis=1) if combined_sim.size > 0 else np.array([0.0])
    precision = float(np.mean(max_sim_per_sentence))
    
    # Recall: average max similarity per reference chunk across summary sentences
    max_sim_per_chunk = np.max(combined_sim, axis=0) if combined_sim.size > 0 else np.array([0.0])
    recall = float(np.mean(max_sim_per_chunk))
    
    # F1 Score
    if (precision + recall) > 0:
        f1 = float(2 * (precision * recall) / (precision + recall))
    else:
        f1 = 0.0
        
    sentence_scores = []
    for idx, sent in enumerate(sentences):
        sentence_scores.append({
            "sentence": sent,
            "precision": float(max_sim_per_sentence[idx]) if idx < len(max_sim_per_sentence) else 0.0
        })
        
    return f1, {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "sentence_scores": sentence_scores
    }
