import re
from rank_bm25 import BM25Okapi

def simple_tokenize(text: str) -> list[str]:
    """
    Cleans and tokenizes text for keyword search.
    """
    text = text.lower()
    # Replace non-alphanumeric characters with spaces
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    # Split on whitespace and remove empty strings
    return [token for token in text.split() if token]

class BM25Index:
    def __init__(self):
        self.bm25 = None
        self.corpus = []
        
    def build(self, texts: list[str]):
        """
        Builds the BM25 index over a list of texts.
        """
        self.corpus = texts
        tokenized_corpus = [simple_tokenize(text) for text in texts]
        if tokenized_corpus and any(tokenized_corpus):
            self.bm25 = BM25Okapi(tokenized_corpus)
        else:
            self.bm25 = None
            
    def search(self, query: str, top_k: int = 5) -> list[tuple[float, int]]:
        """
        Queries the BM25 index.
        Returns a list of tuples (score, index) sorted by score descending.
        """
        if not self.bm25 or not self.corpus:
            return []
            
        tokenized_query = simple_tokenize(query)
        if not tokenized_query:
            return []
            
        scores = self.bm25.get_scores(tokenized_query)
        # Zip scores with index
        scored_indices = list(enumerate(scores))
        # Sort descending by score
        scored_indices.sort(key=lambda x: x[1], reverse=True)
        
        # Filter top-k
        top_results = [(score, idx) for idx, score in scored_indices[:top_k]]
        return top_results
