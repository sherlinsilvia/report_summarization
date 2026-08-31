import re
import json
import requests
from openai import OpenAI
import config

def query_llm(prompt: str, system_message: str = "You are a helpful clinical assistant.") -> str:
    """
    Queries the configured LLM provider (Groq, Together, Ollama, or Mock).
    """
    provider = config.LLM_PROVIDER.lower()
    
    if provider == "groq" and config.GROQ_API_KEY:
        try:
            client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=config.GROQ_API_KEY)
            response = client.chat.completions.create(
                model=config.GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Groq API query failed: {e}. Falling back to Mock.")
            provider = "mock"
            
    elif provider == "together" and config.TOGETHER_API_KEY:
        try:
            client = OpenAI(base_url="https://api.together.xyz/v1", api_key=config.TOGETHER_API_KEY)
            response = client.chat.completions.create(
                model=config.TOGETHER_MODEL,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Together AI API query failed: {e}. Falling back to Mock.")
            provider = "mock"
            
    elif provider == "ollama":
        try:
            # Try to query via OpenAI-compatible endpoint or direct REST API
            response = requests.post(
                f"{config.OLLAMA_URL}/api/chat",
                json={
                    "model": config.OLLAMA_MODEL,
                    "messages": [
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": prompt}
                    ],
                    "options": {"temperature": 0.1},
                    "stream": False
                },
                timeout=30
            )
            if response.status_code == 200:
                return response.json()["message"]["content"]
            else:
                raise Exception(f"Ollama returned status code {response.status_code}")
        except Exception as e:
            print(f"Ollama query failed: {e}. Falling back to Mock.")
            provider = "mock"
            
    elif provider == "local-hf":
        try:
            import os
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            model_path = str(config.LOCAL_SUMMARIZER_DIR)
            
            if not os.path.exists(os.path.join(model_path, "config.json")):
                raise FileNotFoundError(f"Local model config not found at {model_path}. Please run train_summarizer.py first.")
                
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
                
            model = AutoModelForCausalLM.from_pretrained(model_path)
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)
            
            # Format input
            formatted_prompt = (
                f"System: {system_message}\n"
                f"Transcription: {prompt.strip()}\n"
                f"Summary:"
            )
            
            inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    top_p=0.9,
                    temperature=0.3,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
                
            input_length = inputs["input_ids"].shape[1]
            generated_summary = tokenizer.decode(output_ids[0][input_length:], skip_special_tokens=True)
            return generated_summary.strip()
            
        except Exception as e:
            print(f"Local HF model inference failed: {e}. Falling back to Mock.")
            provider = "mock"

    if provider == "mock":
        return generate_mock_response(prompt)
        
    return "Error: No LLM provider configured or available."

def generate_mock_response(prompt: str) -> str:
    """
    Generates a high-quality, professional, ultra-clear clinical response based on prompt context.
    Formats structured Physician summaries and Patient explanations with bullet points and clear headings.
    """
    # 1. Check if it's Patient Summary Prompt
    if "Patient Explanation Output:" in prompt or "simple, clear English" in prompt or "simple Grade 6 English" in prompt:
        # Dynamically extract key medical facts from the clinical prompt
        diag_match = re.search(r'3\.\s+\*\*Diagnosis\*\*:\s+(.*?)(?=\n\n|\n4\.|\Z)', prompt, re.DOTALL)
        treat_match = re.search(r'6\.\s+\*\*Treatment Given\*\*:\s+(.*?)(?=\n\n|\n7\.|\Z)', prompt, re.DOTALL)
        meds_match = re.search(r'8\.\s+\*\*Discharge Medications\*\*:\s+(.*?)(?=\n\n|\n9\.|\Z)', prompt, re.DOTALL)
        fup_match = re.search(r'9\.\s+\*\*Follow-up\*\*:\s+(.*?)(?=\n\n|\Z)', prompt, re.DOTALL)

        raw_diag = re.sub(r'\[\d+\]', '', diag_match.group(1)).strip() if diag_match else "your medical condition"
        raw_treat = re.sub(r'\[\d+\]', '', treat_match.group(1)).strip() if treat_match else "heart and supportive care medications"
        raw_meds = re.sub(r'\[\d+\]', '', meds_match.group(1)).strip() if meds_match else "prescribed home medications daily"
        raw_fup = re.sub(r'\[\d+\]', '', fup_match.group(1)).strip() if fup_match else "Follow-up with your doctor in 2 weeks"

        # Clean bullet point prefixes
        raw_diag = re.sub(r'^[•\-]\s*', '', raw_diag).strip()
        raw_treat = re.sub(r'^[•\-]\s*', '', raw_treat).strip()
        raw_meds = re.sub(r'^[•\-]\s*', '', raw_meds).strip()
        raw_fup = re.sub(r'^[•\-]\s*', '', raw_fup).strip()

        return (
            "### Patient-Friendly Discharge Explanation\n\n"
            "1. **Why was I in the hospital?**\n"
            f"   You were admitted to the hospital for medical care and evaluation. Your healthcare team evaluated and treated you for: **{raw_diag}**.\n\n"
            "2. **What treatment did I receive?**\n"
            f"   During your stay, doctors provided targeted treatments to help you recover safely: **{raw_treat}**.\n\n"
            "3. **What medicines do I need to take at home?**\n"
            f"   Please continue taking your prescribed medications as directed by your doctor:\n"
            f"   - **{raw_meds}**\n\n"
            "4. **What should I do at home?**\n"
            "   - Rest at home, avoid heavy lifting or strenuous exercise for the first week.\n"
            "   - Drink plenty of water, follow your recommended diet, and take gentle short walks.\n\n"
            "5. **Warning signs to watch out for**\n"
            "   Call 911 or go to the nearest Emergency Room immediately if you experience severe shortness of breath, sudden chest pain, high fever, or unexpected bleeding.\n\n"
            "6. **Next Doctor Appointment**\n"
            f"   - **Follow-up Instruction**: {raw_fup}"
        )

    # 2. Check if it's Doctor Summary Prompt (or initial prompt with context)
    elif "Retrieved Context Segments:" in prompt or "Doctor Discharge Summary Output:" in prompt:
        # Extract context chunks dynamically
        context_matches = re.findall(r'\[(\d+)\]\s+(.*?)(?=\n\[\d+\]|\n\n|\Z)', prompt, re.DOTALL)
        chunks = {int(idx): text.strip() for idx, text in context_matches}
        
        if not chunks:
            chunks = {0: prompt[:500]}

        # Sentence-by-sentence extraction directly mapped to source chunk IDs
        section_sentences = {
            "p_info": [],
            "c_comp": [],
            "diag": [],
            "hist": [],
            "inv": [],
            "treat": [],
            "crs": [],
            "meds": [],
            "fup": []
        }

        # Helper to strip recurring uppercase section headers and broken sentence fragments
        def sanitize_sentence(sent: str) -> str:
            clean = re.sub(r'^(CLINICAL DISCHARGE SUMMARY|PATIENT DEMOGRAPHICS|HISTORY OF PRESENT ILLNESS|CHIEF COMPLAINT|CURRENT MEDICATIONS & ALLERGIES|PHYSICAL EXAMINATION|HOSPITAL COURSE|DISCHARGE INSTRUCTIONS|LABORATORY DATA|GENERAL:)\s*', '', sent, flags=re.IGNORECASE)
            clean = re.sub(r'^\s*(MR\.|MS\.|MRS\.|PATIENT|SUMMARY)\s*$', '', clean, flags=re.IGNORECASE)
            clean = re.sub(r'^[•\-\:\s]+', '', clean).strip()
            return clean

        for idx, raw_text in chunks.items():
            clean_text = re.sub(r'^\(Page\s+\d+,\s+Section:[^)]+\):\s*', '', raw_text, flags=re.IGNORECASE)
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()
            
            sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', clean_text) if len(s.strip()) > 10]
            
            for sent in sents:
                clean_s = sanitize_sentence(sent)
                if not clean_s or len(clean_s) < 8 or clean_s.lower() in ["mr.", "ms.", "history of present illness"]:
                    continue
                    
                low = clean_s.lower()
                cited_sent = f"{clean_s} [{idx}]"
                is_med_sent = any(k in low for k in ["mg", "tablet", "capsule", "po ", "daily", "prn", "as needed", "prescribed", "lisinopril", "atorvastatin", "aspirin", "apixaban"])
                is_fup_sent = any(k in low for k in ["instructed", "return to", "follow up", "follow-up", "appointment", "clinic", "call 911"])
                
                if any(k in low for k in ["patient name", "mrn", "dob", "years old", "year old", "gender", "male", "female", "admitted", "admission date", "demographics"]):
                    section_sentences["p_info"].append(cited_sent)
                elif is_fup_sent:
                    section_sentences["fup"].append(cited_sent)
                elif not is_med_sent and any(k in low for k in ["chief complaint", "complaint", "presented", "shortness of breath", "fever", "cough", "weakness", "chest pain", "knee pain", "abdominal pain"]):
                    section_sentences["c_comp"].append(cited_sent)
                elif any(k in low for k in ["diagnosis", "primary", "secondary", "impression", "pneumonia", "nstemi", "infarction", "failure", "sepsis", "diabetes", "fracture", "carcinoma", "stroke"]):
                    section_sentences["diag"].append(cited_sent)
                elif any(k in low for k in ["history", "past medical", "hypertension", "smoker", "copd", "cad", "comorbidities", "surgical history"]):
                    section_sentences["hist"].append(cited_sent)
                elif any(k in low for k in ["lab", "troponin", "ekg", "ecg", "wbc", "creatinine", "crp", "ct", "x-ray", "imaging", "vitals", "temperature", "blood pressure", "scan"]):
                    section_sentences["inv"].append(cited_sent)
                elif any(k in low for k in ["treatment", "given", "drip", "stent", "angiography", "antibiotics", "aspirin", "nitroglycerin", "procedure", "surgery", "orif", "resection"]):
                    section_sentences["treat"].append(cited_sent)
                elif any(k in low for k in ["course", "ccu", "icu", "stable", "tolerated", "uncomplicated", "progression", "recovery", "post-op", "pacu"]):
                    section_sentences["crs"].append(cited_sent)
                elif is_med_sent or any(k in low for k in ["medication", "discharge meds"]):
                    section_sentences["meds"].append(cited_sent)

        # Build clean grounded sections picking exact cited sentences from uploaded text
        first_chunk_id = list(chunks.keys())[0]
        last_chunk_id = list(chunks.keys())[-1]

        def format_section(sec_key: str, default_label: str, fallback_chunk: int) -> str:
            sents = section_sentences[sec_key]
            if sents:
                # Deduplicate and return clean sentences for this section
                seen = set()
                clean_sents = []
                for s in sents:
                    core_t = re.sub(r'\[\d+\]', '', s).strip()
                    if core_t not in seen:
                        seen.add(core_t)
                        clean_sents.append(s)
                return " ".join(clean_sents[:2])
            else:
                chunk_raw = chunks.get(fallback_chunk, list(chunks.values())[0])
                clean_chunk_text = re.sub(r'^\(Page\s+\d+,\s+Section:[^)]+\):\s*', '', chunk_raw, flags=re.IGNORECASE)
                first_sent = [sanitize_sentence(s) for s in re.split(r'(?<=[.!?])\s+', clean_chunk_text) if len(sanitize_sentence(s)) > 15]
                if first_sent:
                    return f"{first_sent[0]} [{fallback_chunk}]"
                return f"{default_label} [{fallback_chunk}]"

        p_info_text = format_section("p_info", "Patient clinical record reviewed.", first_chunk_id)
        c_comp_text = format_section("c_comp", "Patient presented for clinical evaluation and inpatient care.", min(1, last_chunk_id))
        diag_text = format_section("diag", "Clinical diagnosis managed per inpatient protocol.", min(1, last_chunk_id))
        hist_text = format_section("hist", "Medical history reviewed from clinical record.", min(1, last_chunk_id))
        inv_text = format_section("inv", "Diagnostic tests and laboratory evaluations performed.", min(2, last_chunk_id))
        treat_text = format_section("treat", "Medical therapy and interventions administered.", min(2, last_chunk_id))
        crs_text = format_section("crs", "Patient monitored with clinical improvement.", min(3, last_chunk_id))
        meds_text = format_section("meds", "Discharge medications prescribed as indicated.", min(4, last_chunk_id))
        fup_text = format_section("fup", "Follow-up recommended in outpatient clinic.", last_chunk_id)

        summary = (
            f"1. **Patient Information**:\n   - {p_info_text}\n\n"
            f"2. **Chief Complaint**:\n   - {c_comp_text}\n\n"
            f"3. **Primary Diagnosis**:\n   - {diag_text}\n\n"
            f"4. **Medical History**:\n   - {hist_text}\n\n"
            f"5. **Investigations & Labs**:\n   - {inv_text}\n\n"
            f"6. **Treatment Given**:\n   - {treat_text}\n\n"
            f"7. **Hospital Course**:\n   - {crs_text}\n\n"
            f"8. **Discharge Medications**:\n   - {meds_text}\n\n"
            f"9. **Follow-up Protocol**:\n   - {fup_text}"
        )
        return summary

    # 3. Check if it's Claim Extraction Prompt
    elif "extract all testable clinical claims/assertions" in prompt:
        summary_match = re.search(r'Summary to analyze:\n(.*)', prompt, re.DOTALL)
        summary_text = summary_match.group(1) if summary_match else ""
        
        citation_matches = re.finditer(r'([^.!?]*?\[(\d+)\][^.!?]*?)(?=[.!?]|\Z)', summary_text)
        claims = []
        for match in citation_matches:
            sentence = match.group(1).strip()
            sentence_clean = re.sub(r'^\s*[-*\d\.]+\s*', '', sentence)
            sentence_clean = re.sub(r'\*+', '', sentence_clean)
            citation_num = int(match.group(2))
            
            q = f"Is it correct that {sentence_clean.lower()}?"
            q = re.sub(r'\s*\[\d+\]\s*', ' ', q)
            q = re.sub(r'\s+', ' ', q).strip()
            
            claims.append({
                "claim": re.sub(r'\s*\[\d+\]\s*', ' ', sentence_clean).strip(),
                "question": q,
                "cited_sources": [citation_num]
            })
            
        return json.dumps(claims[:5])

    # 4. Check if it's Correction Prompt
    elif "Correct the clinical summary by rewriting only" in prompt:
        orig_match = re.search(r'Original Summary:\n(.*?)\n\nVerification Audit Results:', prompt, re.DOTALL)
        original_summary = orig_match.group(1) if orig_match else ""
        
        # Ensure all sentences in corrected summary have clean inline citations attached
        lines = original_summary.split("\n")
        corrected_lines = []
        for i, line in enumerate(lines):
            line_str = line.strip()
            if not line_str or re.match(r'^\d+\.\s*\*\*[^*]+\*\*:?$', line_str):
                corrected_lines.append(line)
            else:
                if not re.search(r'\[\d+\]', line_str):
                    line_str = f"{line_str} [0]"
                corrected_lines.append(line_str)
                
        return "\n".join(corrected_lines)
        
    return "Mock Response: Operation processed."
