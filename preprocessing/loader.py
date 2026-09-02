import os
import fitz  # PyMuPDF
import pandas as pd

def load_pdf(file_path: str) -> list[dict]:
    """
    Loads a PDF file and extracts text page by page.
    Returns a list of dictionaries with page number and page text.
    """
    pages = []
    try:
        doc = fitz.open(file_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            pages.append({
                "page": page_num + 1,
                "text": text
            })
    except Exception as e:
        print(f"Error loading PDF {file_path}: {e}")
        # Fallback: if it's text-based or fitz fails, try reading as plain text
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            pages.append({
                "page": 1,
                "text": content
            })
        except Exception as ex:
            print(f"Fallback reading failed: {ex}")
    return pages

def load_text(file_path: str) -> list[dict]:
    """
    Loads a plain text file.
    Returns a list with a single dict containing the entire text.
    """
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        return [{"page": 1, "text": content}]
    except Exception as e:
        print(f"Error loading text file {file_path}: {e}")
        return []

def load_mimic_csv(file_path: str, subject_id: int = None, hadm_id: int = None) -> list[dict]:
    """
    Loads MIMIC-IV formatted notes from a CSV file.
    Filters by subject_id and hadm_id if specified.
    """
    try:
        df = pd.read_csv(file_path)
        
        # Filter columns if present
        if 'subject_id' in df.columns and subject_id is not None:
            df = df[df['subject_id'] == int(subject_id)]
        if 'hadm_id' in df.columns and hadm_id is not None:
            df = df[df['hadm_id'] == int(hadm_id)]
            
        if df.empty:
            return []
            
        pages = []
        for index, row in df.iterrows():
            text = str(row['text']) if 'text' in row else ""
            note_id = str(row['note_id']) if 'note_id' in row else f"note_{index}"
            subj = int(row['subject_id']) if 'subject_id' in row and not pd.isna(row['subject_id']) else 0
            hadm = int(row['hadm_id']) if 'hadm_id' in row and not pd.isna(row['hadm_id']) else 0
            
            pages.append({
                "page": len(pages) + 1,
                "text": text,
                "note_id": note_id,
                "subject_id": subj,
                "hadm_id": hadm
            })
        return pages
    except Exception as e:
        print(f"Error loading MIMIC CSV: {e}")
        return []

def load_image_document(file_path: str) -> list[dict]:
    """
    Loads an image document or prescription photo and returns structured clinical page text.
    """
    try:
        with open(file_path, "rb") as f:
            b = f.read()
        from prescription.analyzer import analyze_prescription_content
        res = analyze_prescription_content(b, os.path.basename(file_path))
        full_text = (
            f"HOSPITAL: {res['hospital_name']}\n"
            f"PHYSICIANS: {res['doctor_info']}\n"
            f"PATIENT: {res['patient_name']} (Age: {res['age']}, Gender: {res['gender']}, Card: {res['card_no']}, Department: {res['department']})\n"
            f"DIAGNOSIS: {res['diagnosis']}\n\n"
            f"PRESCRIBED MEDICATIONS:\n" +
            "\n".join([f"- {m['name']} ({m['type']}): {m['frequency']}, {m['timing']}, Duration: {m['duration']}. Purpose: {m['purpose']}" for m in res['medications']]) +
            "\n\nSAFETY PRECAUTIONS & ADVICE:\n" +
            "\n".join([f"- {w}" for w in res['safety_warnings']]) +
            f"\n\nEXECUTIVE SUMMARY: {res['executive_summary']}"
        )
        return [{
            "page": 1,
            "text": full_text
        }]
    except Exception as e:
        print(f"Error loading image document {file_path}: {e}")
        return load_text(file_path)

def load_report(file_path: str) -> list[dict]:
    """
    Orchestrates report loading based on file extension.
    Supports PDF, CSV, TXT, and all medical image formats (JPG, JPEG, PNG, WEBP, BMP).
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return load_pdf(file_path)
    elif ext == ".csv":
        return load_mimic_csv(file_path)
    elif ext in [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"]:
        return load_image_document(file_path)
    else:
        return load_text(file_path)
