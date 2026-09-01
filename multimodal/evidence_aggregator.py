"""
Multimodal Evidence Aggregator.
Merges evidence from multiple uploaded documents (PDFs, Images, Prescriptions, X-Rays)
into a unified structured intermediate representation and creates source-tagged chunks for RAG indexing.
"""

from typing import List, Dict, Any
from multimodal.schemas import StructuredClinicalEvidence, MedicationItem, ImageFindingItem, PatientDemographics
from multimodal.extractor import extract_clinical_evidence_from_file

def aggregate_multimodal_evidence(file_paths_and_names: List[tuple[str, str]]) -> StructuredClinicalEvidence:
    """
    Combines extractions from multiple files into a single unified StructuredClinicalEvidence object.
    
    Args:
        file_paths_and_names: List of tuples (file_path, original_file_name)
        
    Returns:
        StructuredClinicalEvidence: Combined patient evidence across all uploaded sources.
    """
    unified = StructuredClinicalEvidence()
    
    for fpath, fname in file_paths_and_names:
        evidence = extract_clinical_evidence_from_file(fpath, fname)
        
        # Merge detected types and sources
        for dt in evidence.detected_types:
            if dt not in unified.detected_types:
                unified.detected_types.append(dt)
        if fname not in unified.evidence_sources:
            unified.evidence_sources.append(fname)
            
        # Merge demographics (first non-empty value takes priority)
        demo = evidence.patient_information
        if demo.name and not unified.patient_information.name:
            unified.patient_information.name = demo.name
        if demo.age and not unified.patient_information.age:
            unified.patient_information.age = demo.age
        if demo.gender and not unified.patient_information.gender:
            unified.patient_information.gender = demo.gender
        if demo.mrn and not unified.patient_information.mrn:
            unified.patient_information.mrn = demo.mrn
        if demo.date and not unified.patient_information.date:
            unified.patient_information.date = demo.date
            
        # Merge clinical lists
        unified.clinical_history.extend([h for h in evidence.clinical_history if h not in unified.clinical_history])
        unified.symptoms.extend([s for s in evidence.symptoms if s not in unified.symptoms])
        unified.diagnoses_mentioned.extend([d for d in evidence.diagnoses_mentioned if d not in unified.diagnoses_mentioned])
        unified.investigations.extend([inv for inv in evidence.investigations if inv not in unified.investigations])
        unified.recommendations.extend([r for r in evidence.recommendations if r not in unified.recommendations])
        unified.follow_up.extend([f for f in evidence.follow_up if f not in unified.follow_up])
        unified.uncertain_information.extend([u for u in evidence.uncertain_information if u not in unified.uncertain_information])
        
        # Merge medications
        for med in evidence.medications:
            if not any(m.name.lower() == med.name.lower() for m in unified.medications):
                unified.medications.append(med)
                
        # Merge image findings
        for img_f in evidence.image_findings:
            unified.image_findings.append(img_f)
            
        # Append raw text
        if evidence.raw_extracted_text:
            if unified.raw_extracted_text:
                unified.raw_extracted_text += "\n\n" + evidence.raw_extracted_text
            else:
                unified.raw_extracted_text = evidence.raw_extracted_text

    return unified

def convert_evidence_to_chunks(evidence: StructuredClinicalEvidence) -> List[Dict[str, Any]]:
    """
    Converts unified structured clinical evidence into source-tagged chunks
    suitable for dense FAISS and sparse BM25 indexing.
    """
    chunks = []
    chunk_idx = 0
    
    # 1. Patient Demographics Chunk
    demo = evidence.patient_information
    demo_parts = []
    if demo.name: demo_parts.append(f"Patient Name: {demo.name}")
    if demo.age: demo_parts.append(f"Age: {demo.age}")
    if demo.gender: demo_parts.append(f"Gender: {demo.gender}")
    if demo.mrn: demo_parts.append(f"MRN: {demo.mrn}")
    if demo.date: demo_parts.append(f"Record Date: {demo.date}")
    
    if demo_parts:
        chunks.append({
            "chunk_id": chunk_idx,
            "page": 1,
            "section": "Patient Information",
            "text": " | ".join(demo_parts) + f". Sources: {', '.join(evidence.evidence_sources)}"
        })
        chunk_idx += 1
        
    # 2. Symptoms & Chief Complaint Chunk
    if evidence.symptoms or evidence.diagnoses_mentioned:
        sym_text = "Symptoms: " + "; ".join(evidence.symptoms) if evidence.symptoms else ""
        diag_text = " Diagnoses: " + "; ".join(evidence.diagnoses_mentioned) if evidence.diagnoses_mentioned else ""
        chunks.append({
            "chunk_id": chunk_idx,
            "page": 1,
            "section": "Chief Complaint & Diagnosis",
            "text": (sym_text + diag_text).strip()
        })
        chunk_idx += 1
        
    # 3. Medications Chunks (Each medication gets precise source attribution)
    if evidence.medications:
        med_lines = []
        for m in evidence.medications:
            status = " [Uncertain / Verification Required]" if m.is_uncertain else ""
            med_lines.append(
                f"- {m.name} ({m.strength or 'Standard strength'}): {m.dosage or '1 dose'}, "
                f"Frequency: {m.frequency or 'As directed'}, Timing: {m.food_instruction or 'With water'}, "
                f"Duration: {m.duration or 'Per protocol'}{status} [Source: {m.source}]"
            )
        chunks.append({
            "chunk_id": chunk_idx,
            "page": 1,
            "section": "Discharge Medications",
            "text": "Prescribed Medications:\n" + "\n".join(med_lines)
        })
        chunk_idx += 1
        
    # 4. Medical Imaging & Vision Findings Chunks
    if evidence.image_findings:
        img_lines = []
        for img in evidence.image_findings:
            img_lines.append(
                f"[{img.modality} - {img.anatomical_region}]: {img.observation} "
                f"(Confidence: {img.confidence}). Safety Limitation: {img.limitations} [Source: {img.source}]"
            )
        chunks.append({
            "chunk_id": chunk_idx,
            "page": 1,
            "section": "Investigations & Imaging Findings",
            "text": "Radiological & Image Observations:\n" + "\n".join(img_lines)
        })
        chunk_idx += 1
        
    # 5. Recommendations & Follow-Up Chunk
    if evidence.follow_up or evidence.recommendations:
        fup_text = "; ".join(evidence.follow_up + evidence.recommendations)
        chunks.append({
            "chunk_id": chunk_idx,
            "page": 1,
            "section": "Follow-up & Instructions",
            "text": f"Outpatient Recommendations & Care Plan: {fup_text}"
        })
        chunk_idx += 1
        
    # 6. Uncertainties & Safety Notices Chunk
    if evidence.uncertain_information:
        chunks.append({
            "chunk_id": chunk_idx,
            "page": 1,
            "section": "Uncertain Information & Safety",
            "text": "Clinical Disclaimers & Uncertainties:\n" + "\n".join([f"- {u}" for u in evidence.uncertain_information])
        })
        chunk_idx += 1

    # If chunks list is empty (fallback to raw text split)
    if not chunks:
        raw = evidence.raw_extracted_text or "General medical record evaluated."
        from preprocessing.chunker import chunk_text
        chunks = chunk_text([{"page": 1, "text": raw}])
        from preprocessing.section_extractor import extract_sections
        chunks = extract_sections(chunks)

    return chunks
