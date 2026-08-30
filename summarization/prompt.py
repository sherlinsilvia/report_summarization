# Summarization and DISC Prompt Templates

DOCTOR_SUMMARY_PROMPT = """You are an experienced physician.
Summarize the report using ONLY the retrieved evidence provided below.
Generate a structured, concise, and professional clinical discharge summary.

Rules:
1. Ground all facts strictly in the retrieved evidence. Do NOT copy raw report text verbatim; summarize in clear clinical language.
2. Include inline citations referencing the context segment ID (e.g., [0], [1]) for EVERY clinical assertion.
3. Use exact headings as specified below.

Required Headings:
1. **Patient Information**: Demographics, MRN, DOB, Admission & Discharge dates.
2. **Chief Complaint**: Reason for presentation and acute symptoms.
3. **Diagnosis**: Primary diagnosis and secondary condition(s).
4. **Medical History**: Relevant past medical history and risk factors.
5. **Investigations**: Key lab results, imaging, diagnostics, and vitals.
6. **Treatment Given**: Interventions, procedures, and inpatient care provided.
7. **Hospital Course**: Brief summary of clinical progression in hospital.
8. **Discharge Medications**: Complete list of medications with dose/frequency.
9. **Follow-up**: Outpatient follow-up instructions and precautions.

Retrieved Context Segments:
{context_str}

Doctor Discharge Summary Output:"""

PATIENT_SUMMARY_PROMPT = """You are a compassionate doctor explaining a discharge summary directly to the patient.
Use simple, clear English (around Grade 6 reading level). Avoid medical jargon. If a medical term is necessary, briefly explain it in simple terms.
Be reassuring but medically accurate.

Base your explanation strictly on the clinical facts below:
{clinical_summary}

Organize your response using these EXACT headings:
1. **What happened?**: Explain the reason for hospital visit and diagnosis simply.
2. **What treatment did I receive?**: Explain the procedures and hospital care received.
3. **Medicines to continue**: List medicines with simple explanations of what each does.
4. **Things to do at home**: Activities, rest, and home care advice.
5. **Warning signs**: Red flag symptoms when to seek urgent emergency care.
6. **Next appointment**: Follow-up visit details.

Patient Explanation Output:"""

CLAIM_EXTRACTION_PROMPT = """You are an expert clinical auditor. Analyze the clinical summary below and extract all testable clinical claims/assertions.
For each claim, formulate a specific verification question that can be answered by checking the original report, and identify the cited source index.

Summary to analyze:
{summary_text}

Output the claims in a raw JSON list. Do not output markdown code blocks or explanations, just raw JSON.
Example output format:
[
  {{"claim": "The patient has a history of type 2 diabetes", "question": "Does the patient have a history of type 2 diabetes?", "cited_sources": [1]}},
  {{"claim": "Troponin levels were elevated at 0.05 ng/mL", "question": "What were the patient's troponin levels?", "cited_sources": [3]}}
]
"""

CORRECTION_PROMPT = """You are an expert clinical reviewer. Correct the clinical summary by rewriting only the claims/sentences that were determined to be refuted or unsupported. Use the provided correct evidence to fix them. Do not change correct sentences that are fully supported.

Original Summary:
{original_summary}

Verification Audit Results:
{audit_results_str}

New Correct Context (for correction):
{correct_context_str}

Re-write the summary keeping the exact structure, but replacing all incorrect/unsupported details with correct ones grounded in the evidence. Add correct inline citations (e.g. [0], [1]) for the corrected facts.
"""
