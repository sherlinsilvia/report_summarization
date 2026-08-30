from retrieval.embedder import embed_text
from retrieval.faiss_index import FAISSIndex
from retrieval.bm25 import BM25Index

def hybrid_retrieve(
    query: str,
    faiss_index: FAISSIndex,
    bm25_index: BM25Index,
    chunks: list[dict],
    top_k: int = 5,
    rrf_k: int = 60
) -> list[dict]:
    """
    Performs hybrid retrieval using Reciprocal Rank Fusion (RRF).
    Combines dense FAISS cosine search with sparse BM25 keyword search.
    
    Returns: List of top_k chunk dictionaries, with retrieved metadata.
    """
    if not chunks:
        return []
        
    # 1. Dense FAISS retrieval
    try:
        query_embedding = embed_text([query])[0]
        # Query FAISS for a larger candidate pool (e.g., top_k * 3)
        faiss_sims, faiss_indices = faiss_index.search(query_embedding, top_k=top_k * 3)
    except Exception as e:
        print(f"FAISS search failed: {e}")
        faiss_indices = []
        
    # 2. Sparse BM25 retrieval
    try:
        # Query BM25 for a larger candidate pool (e.g., top_k * 3)
        bm25_results = bm25_index.search(query, top_k=top_k * 3)
        bm25_indices = [idx for _, idx in bm25_results]
    except Exception as e:
        print(f"BM25 search failed: {e}")
        bm25_indices = []
        
    # 3. Reciprocal Rank Fusion
    rrf_scores = {}
    
    # Process FAISS rank
    for rank, idx in enumerate(faiss_indices):
        # FAISS search returns -1 for empty or invalid indices
        if idx == -1 or idx >= len(chunks):
            continue
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (rrf_k + (rank + 1)))
        
    # Process BM25 rank
    for rank, idx in enumerate(bm25_indices):
        if idx >= len(chunks):
            continue
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (rrf_k + (rank + 1)))
        
    # Sort indices based on RRF scores
    sorted_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    top_rrf_indices = sorted_rrf[:top_k]
    
    # 4. Compile retrieved chunks
    retrieved_chunks = []
    for idx, rrf_score in top_rrf_indices:
        chunk = chunks[idx].copy()
        # Add retrieval score & method metadata
        chunk["rrf_score"] = rrf_score
        chunk["is_dense"] = idx in faiss_indices
        chunk["is_sparse"] = idx in bm25_indices
        retrieved_chunks.append(chunk)
        
    return retrieved_chunks
