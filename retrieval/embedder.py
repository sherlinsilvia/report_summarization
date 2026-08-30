import numpy as np
from sentence_transformers import SentenceTransformer
import config

_model = None

def get_embedding_model():
    global _model
    if getattr(config, "USE_FALLBACK_EMBEDDER", False):
        return None
        
    if _model is None:
        try:
            import os
            # Check if a fine-tuned model exists locally
            local_model_path = str(config.LOCAL_EMBEDDER_DIR)
            if os.path.exists(local_model_path) and os.path.exists(os.path.join(local_model_path, "config.json")):
                print(f"Loading local fine-tuned embedding model from {local_model_path}...")
                _model = SentenceTransformer(local_model_path)
            else:
                print(f"Loading default embedding model {config.EMBEDDING_MODEL_NAME}...")
                _model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
        except Exception as e:
            try:
                print(f"Failed loading local model, falling back to default {config.EMBEDDING_MODEL_NAME}: {e}")
                _model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
            except Exception as ex:
                print(f"Error loading embedding model: {ex}")
                raise ex
    return _model

def embed_text(texts: list[str]) -> np.ndarray:
    """
    Generates sentence embeddings for a list of texts.
    Returns a numpy array of shape (num_texts, embedding_dim).
    """
    if not texts:
        return np.empty((0, 384), dtype=np.float32)
        
    if getattr(config, "USE_FALLBACK_EMBEDDER", False):
        # Deterministic dummy embedding for repeatability based on text length and checksum
        embeddings = []
        for text in texts:
            # Create a simple pseudo-embedding vector
            val = (len(text) % 100) / 100.0
            vec = np.zeros(384, dtype=np.float32)
            vec[0] = val
            vec[1] = 1.0 - val
            # Add some minor noise
            vec += np.random.RandomState(len(text) % 1000).randn(384) * 0.01
            embeddings.append(vec)
        return np.array(embeddings, dtype=np.float32)
    
    try:
        model = get_embedding_model()
        embeddings = model.encode(texts, convert_to_numpy=True)
        return embeddings.astype(np.float32)
    except Exception as e:
        print(f"Embedding failed, returning random dummy embeddings for testing: {e}")
        # Return random/zero embeddings as a defensive fallback for testing
        return np.random.randn(len(texts), 384).astype(np.float32)
