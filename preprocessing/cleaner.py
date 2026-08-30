import re

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
