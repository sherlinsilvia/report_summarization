import re

MEDICAL_KEYWORDS = {
    # Clinical headers & document structure
    "patient", "discharge", "admission", "admitted", "hospital", "clinic", "diagnosis", "history",
    "complaint", "medication", "medications", "treatment", "investigations", "impression", "findings",
    "vitals", "laboratory", "physician", "doctor", "nurse", "prescription", "symptoms", "examination",
    "outpatient", "inpatient", "attending", "consultant", "demographics", "progress", "note",
    # Medical terms, metrics & procedures
    "mrn", "dob", "mg", "po", "iv", "blood pressure", "heart rate", "pulse", "spo2", "temperature",
    "ekg", "ecg", "ct scan", "x-ray", "mri", "troponin", "wbc", "rbc", "hemoglobin", "creatinine",
    "crp", "inr", "pain", "fever", "cough", "infarction", "pneumonia", "fracture", "hypertension",
    "diabetes", "arrhythmia", "stroke", "sepsis", "surgery", "dose", "tablets", "capsule", "follow-up",
    "stent", "angiography", "carcinoma", "edema", "ischemia", "cardiovascular", "pulmonary", "renal"
}

def clean_text(text: str) -> str:
    """
    Cleans raw text extracted from clinical reports.
    - Standardizes line breaks and spacing.
    - Removes recurring footer/header page number patterns.
    - Sanitizes special unicode characters.
    """
    if not text:
        return ""
    
    # Remove page number headers/footers (e.g., "Page 1 of 5", "Page 2")
    text = re.sub(r'(?i)page\s+\d+(\s+of\s+\d+)?', '', text)
    
    # Replace multiple newlines with a single newline or space appropriately
    text = re.sub(r'\r\n', '\n', text)
    
    # Replace multiple spaces with a single space
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Normalize bullet points and lists
    text = re.sub(r'•', '-', text)
    
    # Normalize quotes and dashes
    text = text.replace('“', '"').replace('”', '"').replace('’', "'").replace('‘', "'")
    
    # Strip whitespace from each line and join
    lines = [line.strip() for line in text.split('\n')]
    # Remove empty lines
    lines = [line for line in lines if line]
    
    cleaned_text = '\n'.join(lines)
    return cleaned_text

def validate_medical_document(text: str) -> tuple[bool, str, float]:
    """
    Validates whether the uploaded document text belongs to the medical/clinical domain.
    Returns:
        (is_valid: bool, reason: str, confidence_score: float)
    """
    if not text or len(text.strip()) < 20:
        return False, "File is empty or contains insufficient readable text.", 0.0
        
    words = set(re.findall(r'\b[a-zA-Z]{2,}\b', text.lower()))
    if not words:
        return False, "No valid text tokens detected in document.", 0.0
        
    matches = words.intersection(MEDICAL_KEYWORDS)
    match_count = len(matches)
    density = match_count / min(len(words), 100)
    
    # Require at least 3 distinct clinical/medical keywords or >= 4% keyword density
    if match_count >= 3 or density >= 0.04:
        return True, f"Valid medical document verified ({match_count} clinical keywords detected).", min(1.0, match_count / 10.0)
    else:
        return False, f"Non-medical document detected (only {match_count} medical terms found).", min(1.0, match_count / 10.0)
