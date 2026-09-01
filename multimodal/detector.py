"""
Input Type Detector for Multimodal Medical Documents & Images.
Classifies uploads into Text PDF, Scanned PDF, Handwritten Prescription,
Typed Prescription, X-Ray, CT Scan, MRI Scan, or Clinical Photo.
"""

import os
import re
import fitz  # PyMuPDF
from PIL import Image, ImageStat
from typing import Dict, Any, Tuple

# Detection category definitions
DOCUMENT_TYPES = {
    "text_pdf": "Text-Based Clinical PDF",
    "scanned_pdf": "Scanned Medical Report PDF",
    "typed_prescription": "Typed / Printed Prescription",
    "handwritten_prescription": "Handwritten Prescription",
    "xray_image": "X-Ray Radiograph",
    "ct_scan": "CT Scan Image",
    "mri_image": "MRI Scan Image",
    "clinical_image": "Clinical Photograph / Lab Slip",
    "clinical_report_text": "Clinical Discharge Text Note"
}

def analyze_image_properties(image_path: str) -> Dict[str, Any]:
    """
    Analyzes visual properties of an image file (contrast, brightness, aspect ratio, grayscale nature).
    """
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            mode = img.mode
            
            # Convert to grayscale for statistical analysis
            gray = img.convert('L')
            stat = ImageStat.Stat(gray)
            mean_brightness = stat.mean[0]
            std_contrast = stat.stddev[0]
            
            # Check if image is predominantly grayscale (typical of X-rays/CT/MRI)
            rgb = img.convert('RGB')
            r, g, b = rgb.split()
            diff_rg = ImageStat.Stat(r).mean[0] - ImageStat.Stat(g).mean[0]
            diff_gb = ImageStat.Stat(g).mean[0] - ImageStat.Stat(b).mean[0]
            is_grayscale_like = abs(diff_rg) < 8 and abs(diff_gb) < 8
            
            return {
                "width": width,
                "height": height,
                "aspect_ratio": round(width / max(1, height), 2),
                "brightness": round(mean_brightness, 1),
                "contrast": round(std_contrast, 1),
                "is_grayscale": is_grayscale_like,
                "mode": mode
            }
    except Exception as e:
        return {
            "width": 0, "height": 0, "aspect_ratio": 1.0,
            "brightness": 128, "contrast": 50, "is_grayscale": False, "error": str(e)
        }

def detect_input_type(file_path: str, file_name: str = "") -> Tuple[str, str, float]:
    """
    Detects the specific medical document or imaging modality of an uploaded file.
    
    Returns:
        (type_key: str, user_friendly_label: str, confidence: float)
    """
    fname_lower = (file_name or os.path.basename(file_path)).lower()
    ext = os.path.splitext(fname_lower)[1]
    
    # 1. PDF File Inspection
    if ext == ".pdf":
        try:
            doc = fitz.open(file_path)
            total_text_len = 0
            image_count = 0
            
            for page in doc:
                text = page.get_text()
                total_text_len += len(text.strip())
                image_count += len(page.get_images())
                
            # If text length is significant, it's a digital text PDF
            if total_text_len > 200:
                # Check for prescription keywords
                first_page_text = doc[0].get_text().lower()
                if any(k in first_page_text for k in ["rx", "prescription", "sig:", "dispense", "tablets", "take 1"]):
                    return "typed_prescription", DOCUMENT_TYPES["typed_prescription"], 0.92
                return "text_pdf", DOCUMENT_TYPES["text_pdf"], 0.95
            elif image_count > 0:
                return "scanned_pdf", DOCUMENT_TYPES["scanned_pdf"], 0.88
            else:
                return "text_pdf", DOCUMENT_TYPES["text_pdf"], 0.70
        except Exception:
            return "text_pdf", DOCUMENT_TYPES["text_pdf"], 0.60
            
    # 2. Image File Inspection (JPG, JPEG, PNG, WEBP)
    elif ext in [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"]:
        props = analyze_image_properties(file_path)
        
        # Check filename cues first
        if any(w in fname_lower for w in ["xray", "x-ray", "radiograph", "cxr", "chest_x"]):
            return "xray_image", DOCUMENT_TYPES["xray_image"], 0.95
        if any(w in fname_lower for w in ["ct", "computed_tomography", "ct_scan", "cat_scan"]):
            return "ct_scan", DOCUMENT_TYPES["ct_scan"], 0.95
        if any(w in fname_lower for w in ["mri", "magnetic_resonance", "mri_brain", "mri_spine"]):
            return "mri_image", DOCUMENT_TYPES["mri_image"], 0.95
        if any(w in fname_lower for w in ["rx", "prescription", "rx_slip", "meds"]):
            if "hand" in fname_lower or props["contrast"] < 45:
                return "handwritten_prescription", DOCUMENT_TYPES["handwritten_prescription"], 0.90
            return "typed_prescription", DOCUMENT_TYPES["typed_prescription"], 0.90

        # Visual property analysis:
        # X-Rays / CT / MRI are typically grayscale with dark backgrounds (brightness < 110) and high contrast (> 40)
        if props["is_grayscale"]:
            if props["brightness"] < 90 and props["contrast"] > 35:
                # Likely radiological scan (X-ray, CT, MRI)
                if props["aspect_ratio"] > 0.85 and props["aspect_ratio"] < 1.15 and props["brightness"] < 60:
                    return "ct_scan", DOCUMENT_TYPES["ct_scan"], 0.82
                return "xray_image", DOCUMENT_TYPES["xray_image"], 0.85
            elif props["brightness"] > 140:
                # Light background paper document (Scanned document or prescription)
                if props["contrast"] < 40:
                    return "handwritten_prescription", DOCUMENT_TYPES["handwritten_prescription"], 0.78
                return "typed_prescription", DOCUMENT_TYPES["typed_prescription"], 0.80

        # Non-grayscale or general document image
        if props["brightness"] > 130:
            return "handwritten_prescription", DOCUMENT_TYPES["handwritten_prescription"], 0.75
            
        return "clinical_image", DOCUMENT_TYPES["clinical_image"], 0.70
        
    # 3. CSV File Inspection
    elif ext == ".csv":
        return "clinical_report_text", "MIMIC-IV Clinical Notes Database", 0.98
        
    # 4. Default Text
    else:
        return "clinical_report_text", DOCUMENT_TYPES["clinical_report_text"], 0.85
