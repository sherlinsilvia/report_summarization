import re

# Regex patterns for clinical report section headers
SECTION_PATTERNS = {
    "Demographics": re.compile(
        r'^\s*(patient\s+info(rmation)?|demographics|patient\s+name|mrn|date\s+of\s+birth|dob|case\s+report|clinical\s+summary|subject)\b', 
        re.IGNORECASE
    ),
    "Chief Complaint": re.compile(
        r'^\s*(chief\s+complaint|cc|reason\s+for\s+visit|reason\s+for\s+consult(ation)?|presentation)\b', 
        re.IGNORECASE
    ),
    "History of Present Illness": re.compile(
        r'^\s*(history\s+of\s+present\s+illness|hpi|presenting\s+illness|history\s+of\s+illness)\b', 
        re.IGNORECASE
    ),
    "Past Medical History": re.compile(
        r'^\s*(past\s+medical\s+history|pmh|medical\s+history|past\s+history|surgical\s+history|social\s+history|family\s+history)\b', 
        re.IGNORECASE
    ),
    "Medications & Allergies": re.compile(
        r'^\s*(medications|meds|current\s+medications|active\s+medications|allergies|drug\s+allergies|intolerances)\b', 
        re.IGNORECASE
    ),
    "Physical Examination": re.compile(
        r'^\s*(physical\s+exam(ination)?|pe|objective|vital\s+signs|vitals|clinical\s+findings|review\s+of\s+systems|ros)\b', 
        re.IGNORECASE
    ),
    "Diagnostics & Lab Results": re.compile(
        r'^\s*(diagnostics|lab\s+results|laboratory|labs|imaging|x-ray|radiology|ct\s+scan|mri|tests|investigations|pathology)\b', 
        re.IGNORECASE
    ),
    "Assessment & Plan": re.compile(
        r'^\s*(assessment\s+&\s+plan|assessment|plan|assessment/plan|diagnosis|diagnostic\s+impression|impression|recommendations|follow-up|plan\s+of\s+care|discharge\s+plan)\b', 
        re.IGNORECASE
    )
}

def extract_sections(chunks: list[dict]) -> list[dict]:
    """
    Scans the text of chunks, determines section headers, and tags chunks
    with the current active section.
    """
    current_section = "General"
    
    for chunk in chunks:
        text = chunk["text"]
        lines = text.split("\n")
        
        # Check first few lines of the chunk to see if it starts with a section header
        for line in lines[:3]:  # usually headers appear at the start of a chunk
            line_stripped = line.strip()
            # If the line is short (like a header), check matches
            if len(line_stripped) < 60:
                # Remove common header decorations like dashes, asterisks, colons, numbers
                clean_line = re.sub(r'^[\d\.\-\*\#\s]+', '', line_stripped)
                clean_line = re.sub(r':\s*$', '', clean_line).strip()
                
                matched = False
                for section_name, pattern in SECTION_PATTERNS.items():
                    if pattern.match(clean_line) or pattern.match(line_stripped):
                        current_section = section_name
                        matched = True
                        break
                if matched:
                    break
        
        chunk["section"] = current_section
        
    return chunks
