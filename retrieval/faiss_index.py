import os
import numpy as np
from config import EMBEDDINGS_DIR

# Global flags and imports
HAS_FAISS = False
try:
    import faiss
    HAS_FAISS = True
except ImportError:
    print("Warning: FAISS not found. Falling back to Pure NumPy vector search.")

class FAISSIndex:
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = None
        self.vectors = []
        
        if HAS_FAISS:
            self.index = faiss.IndexFlatIP(dimension) # Cosine similarity (IP) after normalization
            
    def add_embeddings(self, embeddings: np.ndarray):
        """
        Adds normalized embeddings to the index.
        """
        if len(embeddings) == 0:
            return
            
        # L2 normalize for Inner Product to compute Cosine Similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1.0, norms)
        normalized_embeddings = embeddings / norms
        
        self.vectors.extend(normalized_embeddings.tolist())
        
        if HAS_FAISS and self.index is not None:
            self.index.add(normalized_embeddings)
            
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> tuple[np.ndarray, np.ndarray]:
        """
        Queries the vector index.
        Returns: (similarities, indices)
        """
        if len(self.vectors) == 0:
            return np.array([]), np.array([])
            
        # L2 normalize the query
        norm = np.linalg.norm(query_embedding)
        norm = 1.0 if norm == 0 else norm
        normalized_query = query_embedding / norm
        
        # Ensure correct shape (1, dim)
        if len(normalized_query.shape) == 1:
            normalized_query = np.expand_dims(normalized_query, axis=0)
            
        if HAS_FAISS and self.index is not None:
            similarities, indices = self.index.search(normalized_query, min(top_k, len(self.vectors)))
            return similarities[0], indices[0]
        else:
            # Fallback pure NumPy cosine search
            all_vectors = np.array(self.vectors, dtype=np.float32)
            # Dot products between query (1, dim) and all vectors (N, dim)
            similarities = np.dot(all_vectors, normalized_query.T).flatten()
            # Sort descending
            sorted_indices = np.argsort(similarities)[::-1]
            top_indices = sorted_indices[:top_k]
            top_similarities = similarities[top_indices]
            return top_similarities, top_indices

    def save(self, filepath: str):
        """
        Saves index and raw vectors.
        """
        if HAS_FAISS and self.index is not None:
            try:
                faiss.write_index(self.index, filepath + ".faiss")
            except Exception as e:
                print(f"Error saving FAISS index: {e}")
        
        # Save vectors as npy regardless
        try:
            np.save(filepath + ".npy", np.array(self.vectors, dtype=np.float32))
        except Exception as e:
            print(f"Error saving raw vectors: {e}")

    def load(self, filepath: str) -> bool:
        """
        Loads index and raw vectors.
        """
        npy_path = filepath + ".npy"
        faiss_path = filepath + ".faiss"
        
        if not os.path.exists(npy_path):
            return False
            
        try:
            self.vectors = np.load(npy_path).tolist()
        except Exception as e:
            print(f"Error loading raw vectors: {e}")
            return False
            
        if HAS_FAISS and os.path.exists(faiss_path):
            try:
                self.index = faiss.read_index(faiss_path)
                return True
            except Exception as e:
                print(f"Error loading FAISS index: {e}, recreating from raw vectors.")
                
        # Recreate index if faiss load failed or faiss not available
        if HAS_FAISS:
            self.index = faiss.IndexFlatIP(self.dimension)
            if self.vectors:
                self.index.add(np.array(self.vectors, dtype=np.float32))
        return True
