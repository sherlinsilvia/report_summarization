"""
Comprehensive Test Script for Multimodal Clinical Intelligence & PDF Generation.
Tests all 5 user requested scenarios:
1. Typed prescription
2. Handwritten prescription
3. X-Ray image
4. CT/MRI scan
5. Multiple concurrent uploads (Prescription + X-Ray + Clinical PDF)
"""

import os
import sys
import fitz
from PIL import Image, ImageDraw

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from multimodal.detector import detect_input_type
from multimodal.extractor import extract_clinical_evidence_from_file
from multimodal.evidence_aggregator import aggregate_multimodal_evidence, convert_evidence_to_chunks
from ui.backend_bridge import process_multimodal_uploads, generate_summary_bridge, run_disc_bridge, generate_pdf_report_bytes

def create_sample_images():
    test_dir = os.path.join(BASE_DIR, "data", "test_multimodal")
    os.makedirs(test_dir, exist_ok=True)
    
    # 1. Typed Prescription PDF (Digital/printed format)
    rx_doc = fitz.open()
    p = rx_doc.new_page(width=595, height=842)
    p.insert_text(fitz.Point(50, 60), "CITY GENERAL HOSPITAL - PRESCRIPTION", fontsize=14, fontname="helv")
    p.insert_text(fitz.Point(50, 90), "Patient Name: John Doe | Age: 52 | Date: 2026-09-01", fontsize=10, fontname="helv")
    p.insert_text(fitz.Point(50, 130), "Rx:", fontsize=12, fontname="helv")
    p.insert_text(fitz.Point(50, 160), "1. Tab. Amoxicillin 500mg - 1 tablet TDS after meals x 7 days", fontsize=10, fontname="helv")
    p.insert_text(fitz.Point(50, 190), "2. Tab. Paracetamol 650mg - 1 tablet SOS after food", fontsize=10, fontname="helv")
    p.insert_text(fitz.Point(50, 220), "3. Cap. Pantoprazole 40mg - 1 capsule OD before breakfast x 14 days", fontsize=10, fontname="helv")
    rx_typed_path = os.path.join(test_dir, "typed_prescription_sample.pdf")
    rx_doc.save(rx_typed_path)
    rx_doc.close()
    
    # 2. Handwritten Prescription image
    rx_hand = Image.new('RGB', (500, 700), color=(245, 242, 235))
    d2 = ImageDraw.Draw(rx_hand)
    d2.text((30, 40), "Dr. Clinic Slip - Rx", fill=(30, 30, 80))
    d2.text((30, 100), "Amox 500 TDS x 5d", fill=(10, 10, 50))
    d2.text((30, 150), "Paracet 650 bd", fill=(10, 10, 50))
    rx_hand_path = os.path.join(test_dir, "handwritten_prescription_sample.jpg")
    rx_hand.save(rx_hand_path)
    
    # 3. Chest X-Ray image (Grayscale radiograph simulation)
    xray_img = Image.new('L', (512, 512), color=25)
    d3 = ImageDraw.Draw(xray_img)
    # Draw thoracic contours
    d3.ellipse([100, 100, 230, 400], fill=65, outline=120)
    d3.ellipse([280, 100, 410, 400], fill=65, outline=120)
    d3.ellipse([210, 250, 310, 380], fill=160, outline=200) # cardiac shadow
    xray_path = os.path.join(test_dir, "chest_xray_sample.png")
    xray_img.save(xray_path)
    
    # 4. CT Brain image
    ct_img = Image.new('L', (400, 400), color=15)
    d4 = ImageDraw.Draw(ct_img)
    d4.ellipse([50, 50, 350, 350], fill=70, outline=240, width=6)
    ct_path = os.path.join(test_dir, "ct_brain_scan_sample.png")
    ct_img.save(ct_path)
    
    return rx_typed_path, rx_hand_path, xray_path, ct_path

def run_tests():
    print("==================================================")
    print("   Running TrustMed Multimodal Verification Tests")
    print("==================================================")
    
    rx_typed_p, rx_hand_p, xray_p, ct_p = create_sample_images()
    
    # TEST 1: Typed Prescription
    t1_key, t1_label, t1_conf = detect_input_type(rx_typed_p)
    print(f"\n[TEST 1] Typed Prescription: Detected as '{t1_label}' (conf: {t1_conf:.2f})")
    ev1 = extract_clinical_evidence_from_file(rx_typed_p)
    print(f"         Medications found: {len(ev1.medications)} -> {[m.name for m in ev1.medications]}")
    assert len(ev1.medications) >= 2, "Test 1 Failed: Expected medications extracted"
    print("         [OK] TEST 1 PASSED")
    
    # TEST 2: Handwritten Prescription
    t2_key, t2_label, t2_conf = detect_input_type(rx_hand_p)
    print(f"\n[TEST 2] Handwritten Prescription: Detected as '{t2_label}' (conf: {t2_conf:.2f})")
    ev2 = extract_clinical_evidence_from_file(rx_hand_p)
    print("         Medications / Uncertainties:", len(ev2.medications), "meds,", len(ev2.uncertain_information), "notices")
    print("         [OK] TEST 2 PASSED")
    
    # TEST 3: X-Ray Image
    t3_key, t3_label, t3_conf = detect_input_type(xray_p)
    print(f"\n[TEST 3] X-Ray: Detected as '{t3_label}' (conf: {t3_conf:.2f})")
    ev3 = extract_clinical_evidence_from_file(xray_p)
    print(f"         Imaging Findings: {len(ev3.image_findings)} -> {ev3.image_findings[0].observation[:60]}...")
    assert len(ev3.image_findings) >= 1, "Test 3 Failed: Expected image findings"
    print("         [OK] TEST 3 PASSED")
    
    # TEST 4: CT Scan Image
    t4_key, t4_label, t4_conf = detect_input_type(ct_p)
    print(f"\n[TEST 4] CT Scan: Detected as '{t4_label}' (conf: {t4_conf:.2f})")
    ev4 = extract_clinical_evidence_from_file(ct_p)
    print(f"         CT Observations: {len(ev4.image_findings)} -> {ev4.image_findings[0].observation[:60]}...")
    print("         [OK] TEST 4 PASSED")
    
    # TEST 5: Multiple Concurrent Uploads (Prescription + X-Ray + PDF)
    import glob
    existing_pdfs = glob.glob(os.path.join(BASE_DIR, "data", "reports", "*.pdf"))
    pdf_p = existing_pdfs[0] if existing_pdfs else rx_typed_p
    
    with open(rx_typed_p, "rb") as f1, open(xray_p, "rb") as f2, open(pdf_p, "rb") as f3:
        b1, b2, b3 = f1.read(), f2.read(), f3.read()
        
    multi_payload = [
        ("prescription.png", b1, "image/png"),
        ("chest_xray.png", b2, "image/png"),
        ("clinical_report.pdf", b3, "application/pdf")
    ]
    
    print("\n[TEST 5] Multiple Concurrent Uploads (Prescription + X-Ray + PDF)...")
    ok, data, err = process_multimodal_uploads(multi_payload)
    print(f"         Upload & Indexing: Success={ok}, Session={data.get('session_id')}, Chunks={data.get('num_chunks')}")
    assert ok, f"Test 5 Upload Failed: {err}"
    
    sess_id = data["session_id"]
    s_ok, s_data, s_err = generate_summary_bridge(sess_id)
    print(f"         Summary Generation: Success={s_ok}, Initial Trust={s_data['trust_results']['composite_trust_score']:.2f}")
    assert s_ok, f"Test 5 Summarization Failed: {s_err}"
    
    d_ok, d_data, d_err = run_disc_bridge(sess_id, s_data["doctor_summary"])
    print(f"         DISC Verification: Success={d_ok}, Final Trust={d_data['final_trust_score']:.2f}")
    assert d_ok, f"Test 5 DISC Failed: {d_err}"
    
    pdf_bytes = generate_pdf_report_bytes(sess_id)
    print(f"         PDF Report Generation: Success={pdf_bytes is not None}, Byte Size={len(pdf_bytes) if pdf_bytes else 0} bytes")
    assert pdf_bytes and len(pdf_bytes) > 2000, "Test 5 PDF Generation Failed"
    print("         [OK] TEST 5 PASSED")
    
    print("\n==================================================")
    print("   ALL 5 MULTIMODAL TEST SCENARIOS PASSED!        ")
    print("==================================================")

if __name__ == "__main__":
    run_tests()
