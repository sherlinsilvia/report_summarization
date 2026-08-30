import re
from trust.retrieval_score import compute_retrieval_similarity
from trust.bert_score import compute_bert_score
from trust.hallucination import evaluate_hallucinations
from trust.citation_score import compute_citation_score
from trust.coverage import compute_coverage_score
import config

def split_summary_sentences(summary_text: str) -> list[str]:
    """
    Splits the summary text into individual clean sentences.
    Ensures standard clinical headings or bullet markers don't break the sentences weirdly.
    """
    # Remove bullet points and numbering at the start of lines
    lines = summary_text.split("\n")
    processed_lines = []
    for line in lines:
        line_clean = re.sub(r'^\s*[-*\d\.]+\s*', '', line).strip()
        if line_clean:
            processed_lines.append(line_clean)
            
    # Combine back and split on sentence markers
    full_text = " ".join(processed_lines)
    sentence_end = re.compile(r'(?<=[.!?])\s+')
    sentences = sentence_end.split(full_text)
    
    return [s.strip() for s in sentences if s.strip()]

def parse_citations_from_sentence(sentence: str) -> tuple[str, list[int]]:
    """
    Extracts citation numbers in brackets (e.g. [0], [1, 2]) from a sentence.
    Returns:
        - clean_sentence: sentence with citations removed.
        - citation_indices: list of integers representing cited chunk indices.
    """
    # Match patterns like [0] or [1,2] or [1, 2]
    citation_pattern = re.compile(r'\[\s*(\d+(?:\s*,\s*\d+)*)\s*\]')
    
    citations = []
    matches = citation_pattern.findall(sentence)
    for match in matches:
        # Split by comma if multiple citations are grouped
        parts = match.split(',')
        for p in parts:
            try:
                citations.append(int(p.strip()))
            except ValueError:
                pass
                
    # Remove citations from the text for evaluation
    clean_sentence = citation_pattern.sub('', sentence)
    # Remove double spaces resulting from replacement
    clean_sentence = re.sub(r'\s+', ' ', clean_sentence).strip()
    
    return clean_sentence, list(set(citations))

def compute_composite_trust_score(
    summary_text: str,
    retrieved_chunks: list[dict]
) -> dict:
    """
    Computes a composite trust score by combining:
    1. Retrieval Semantic Similarity
    2. BERTScore Semantic Precision/Recall/F1
    3. Hallucination Check (NLI contradiction detection)
    4. Citation Groundedness (NLI entailment validation)
    5. Clinical Section Coverage
    
    Returns a dictionary containing:
        - composite_trust_score: float (0.0 to 1.0)
        - scores: dict of individual metric scores
        - details: dict of detailed audit logs
    """
    # 1. Parse summary into sentences and extract citations
    raw_sentences = split_summary_sentences(summary_text)
    clean_sentences = []
    summary_citations = []
    
    for rs in raw_sentences:
        clean_sent, cites = parse_citations_from_sentence(rs)
        if clean_sent:
            clean_sentences.append(clean_sent)
            summary_citations.append(cites)
            
    if not clean_sentences:
        return {
            "composite_trust_score": 0.0,
            "scores": {"retrieval": 0.0, "bertscore": 0.0, "hallucination": 0.0, "citation": 0.0, "coverage": 0.0},
            "details": {"sentences": [], "coverage_breakdown": {}, "bertscore_details": {}}
        }
        
    # 2. Compute individual components
    retrieval_val = compute_retrieval_similarity(clean_sentences, summary_citations, retrieved_chunks)
    bertscore_val, bertscore_details = compute_bert_score(clean_sentences, retrieved_chunks)
    hallucination_val, hallucination_details = evaluate_hallucinations(clean_sentences, summary_citations, retrieved_chunks)
    citation_val, citation_details = compute_citation_score(clean_sentences, summary_citations, retrieved_chunks)
    coverage_val, coverage_breakdown = compute_coverage_score(clean_sentences, retrieved_chunks)
    
    # 3. Combine scores using configured weights
    weights = config.WEIGHTS
    composite_score = (
        (retrieval_val * weights.get("retrieval", 0.15)) +
        (bertscore_val * weights.get("bertscore", 0.20)) +
        (hallucination_val * weights.get("hallucination", 0.25)) +
        (citation_val * weights.get("citation", 0.25)) +
        (coverage_val * weights.get("coverage", 0.15))
    )
    
    # 4. Compile detailed logs per sentence
    sentence_logs = []
    for i in range(len(clean_sentences)):
        sentence_logs.append({
            "index": i,
            "raw_sentence": raw_sentences[i],
            "clean_sentence": clean_sentences[i],
            "citations": summary_citations[i],
            "is_contradicted": hallucination_details[i]["is_contradicted"],
            "max_contradiction": hallucination_details[i]["max_contradiction"],
            "citation_entailment": citation_details[i]["max_entailment"],
            "citation_status": citation_details[i]["support_status"]
        })
        
    return {
        "composite_trust_score": float(composite_score),
        "scores": {
            "retrieval": float(retrieval_val),
            "bertscore": float(bertscore_val),
            "hallucination": float(hallucination_val),
            "citation": float(citation_val),
            "coverage": float(coverage_val)
        },
        "details": {
            "sentences": sentence_logs,
            "coverage_breakdown": coverage_breakdown,
            "bertscore_details": bertscore_details
        }
    }
