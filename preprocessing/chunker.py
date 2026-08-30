import re
import spacy

# Global spacy model holder
_nlp = None

def get_spacy_nlp():
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
            # Enable sentencizer only
            _nlp.add_pipe("sentencizer")
        except Exception:
            # Fallback to None if not loaded/downloaded yet
            _nlp = None
    return _nlp

def split_sentences_fallback(text: str) -> list[str]:
    """
    Fallback sentence splitter using regex if spaCy is not installed/downloaded.
    """
    # Splitting on periods, question marks, or exclamation marks followed by whitespace
    sentence_end = re.compile(r'(?<=[.!?])\s+')
    sentences = sentence_end.split(text)
    return [s.strip() for s in sentences if s.strip()]

def chunk_text(pages: list[dict], chunk_size: int = 600, overlap: int = 100) -> list[dict]:
    """
    Chunks text page-by-page while respecting sentence boundaries.
    Each chunk is a dictionary:
    {
        "chunk_id": int,
        "text": str,
        "page": int,
        "section": str (empty default, populated by section_extractor)
    }
    """
    nlp = get_spacy_nlp()
    chunks = []
    chunk_idx = 0
    
    for page_data in pages:
        page_num = page_data["page"]
        text = page_data["text"]
        
        # Split text into sentences
        if nlp:
            try:
                doc = nlp(text)
                sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
            except Exception:
                sentences = split_sentences_fallback(text)
        else:
            sentences = split_sentences_fallback(text)
            
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sent_len = len(sentence)
            # If a single sentence is longer than chunk_size, we just add it to avoid breaking it
            if current_length + sent_len > chunk_size and current_chunk:
                # Save current chunk
                chunk_text_str = " ".join(current_chunk)
                chunks.append({
                    "chunk_id": chunk_idx,
                    "text": chunk_text_str,
                    "page": page_num,
                    "section": "General"
                })
                chunk_idx += 1
                
                # Apply overlap (keep last few sentences if their length is within overlap)
                overlap_sentences = []
                overlap_len = 0
                for s in reversed(current_chunk):
                    if overlap_len + len(s) < overlap:
                        overlap_sentences.insert(0, s)
                        overlap_len += len(s)
                    else:
                        break
                
                current_chunk = overlap_sentences + [sentence]
                current_length = sum(len(s) for s in current_chunk) + len(current_chunk) - 1
            else:
                current_chunk.append(sentence)
                current_length += sent_len + 1 # +1 for space
                
        # Append residual sentences
        if current_chunk:
            chunk_text_str = " ".join(current_chunk)
            chunks.append({
                "chunk_id": chunk_idx,
                "text": chunk_text_str,
                "page": page_num,
                "section": "General"
            })
            chunk_idx += 1
            
    return chunks
