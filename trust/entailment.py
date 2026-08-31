import config

_pipeline = None
_fallback_warned = False

def get_nli_pipeline():
    global _pipeline, _fallback_warned
    if getattr(config, "USE_FALLBACK_NLI", False):
        if not _fallback_warned:
            print("USE_FALLBACK_NLI is enabled in config. Using fast rule-based fallback NLI.")
            _fallback_warned = True
        return None
        
    if _pipeline is None:
        try:
            from transformers import pipeline
            # Load the text-classification pipeline for NLI
            _pipeline = pipeline(
                "text-classification",
                model=config.NLI_MODEL_NAME,
                device=-1 # force CPU to avoid CUDA setup issues
            )
            print(f"Loaded NLI model: {config.NLI_MODEL_NAME}")
        except Exception as e:
            if not _fallback_warned:
                print(f"Unable to load HF NLI model ({e}). Using rule-based fallback.")
                _fallback_warned = True
            _pipeline = None
    return _pipeline

def compute_nli(premise: str, hypothesis: str) -> dict:
    """
    Computes entailment, contradiction, and neutral probabilities for a (premise, hypothesis) pair.
    Returns:
        {
            "entailment": float,
            "contradiction": float,
            "neutral": float
        }
    """
    pipe = get_nli_pipeline()
    if pipe is not None:
        try:
            # Prepare input in standard cross-encoder format
            # For cross-encoders, the input is often formatted as text + separator + text,
            # or passed as a tuple/list of pairs.
            result = pipe({"text": premise, "text_pair": hypothesis})
            # Let's inspect the outputs. Usually the pipeline outputs:
            # [{'label': 'LABEL_X', 'score': float}]
            # Or model label maps could be 'entailment', 'contradiction', 'neutral' or 'LABEL_0', 'LABEL_1', 'LABEL_2'
            # Let's map them.
            label = result["label"].lower()
            score = result["score"]
            
            # Map standard model label names or IDs
            # Typical DeBERTa MNLI mapping:
            # - contradiction (often label 0 or 'contradiction')
            # - entailment (often label 1 or 'entailment')
            # - neutral (often label 2 or 'neutral')
            
            label_map = {"entailment": 0.0, "contradiction": 0.0, "neutral": 0.0}
            
            # Read label2id if available to do a smart match
            model_labels = pipe.model.config.label2id
            canonical_labels = {v: k.lower() for k, v in model_labels.items()}
            
            # Set scores
            # Since pipeline only returns the top label, we can query raw model to get all logits,
            # but if using pipeline directly, we can do text classification with return_all_scores=True
            results = pipe({"text": premise, "text_pair": hypothesis}, return_all_scores=True)
            for res in results:
                lbl = res["label"].lower()
                sc = res["score"]
                # Match label substring
                if "entail" in lbl:
                    label_map["entailment"] = sc
                elif "contradict" in lbl or "refut" in lbl:
                    label_map["contradiction"] = sc
                else:
                    label_map["neutral"] = sc
            return label_map
        except Exception as e:
            print(f"NLI model inference error: {e}. Falling back to rule-based.")
            
    return compute_nli_fallback(premise, hypothesis)

def compute_nli_fallback(premise: str, hypothesis: str) -> dict:
    """
    Pure Python rule-based NLI fallback using medical content word overlap and contradiction indicators.
    """
    import re
    
    stop_words = {
        "the", "is", "are", "was", "were", "be", "been", "being", "a", "an", "and", "or", "in", "of",
        "to", "with", "for", "on", "at", "by", "as", "patient", "date", "mrn", "name", "summary",
        "has", "had", "have", "this", "that", "it", "from", "he", "she", "his", "her", "mg", "daily"
    }
    
    # Clean words stripping punctuation
    p_clean_words = set(re.findall(r'\b[a-zA-Z0-9]+\b', premise.lower()))
    h_clean_words = set(re.findall(r'\b[a-zA-Z0-9]+\b', hypothesis.lower()))
    
    if not h_clean_words:
        return {"entailment": 0.0, "contradiction": 0.0, "neutral": 1.0}
        
    # Check for direct contradictions (negations)
    negations = {"not", "no", "never", "denies", "denied", "without", "negative", "absent"}
    p_neg = p_clean_words.intersection(negations)
    h_neg = h_clean_words.intersection(negations)
    is_negation_mismatch = len(p_neg) != len(h_neg)
    
    # Content words (excluding stop words)
    p_content = p_clean_words - stop_words
    h_content = h_clean_words - stop_words
    
    if not h_content:
        h_content = h_clean_words
        
    overlap = p_content.intersection(h_content)
    overlap_ratio = len(overlap) / len(h_content) if h_content else 0.0
    
    if is_negation_mismatch and overlap_ratio > 0.3:
        return {"entailment": 0.1, "contradiction": 0.85, "neutral": 0.05}
    elif overlap_ratio >= 0.35 or len(overlap) >= 3:
        return {"entailment": 0.95, "contradiction": 0.02, "neutral": 0.03}
    elif overlap_ratio >= 0.15:
        return {"entailment": 0.50, "contradiction": 0.05, "neutral": 0.45}
    else:
        return {"entailment": 0.15, "contradiction": 0.10, "neutral": 0.75}
