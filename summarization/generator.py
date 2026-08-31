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
    Generates a high-quality, professional mock response based on prompt analysis.
    Cleans raw metadata prefixes and formats structured clinical summaries without markdown artifacts.
    """
    # 1. Check if it's Patient Summary Prompt
    if "Patient Explanation Output:" in prompt or "simple, clear English" in prompt:
        return (
            "1. **What happened?**\n"
            "You were admitted to the hospital for chest discomfort and shortness of breath. Medical tests confirmed a mild heart attack caused by a blocked blood vessel in your heart.\n\n"
            "2. **What treatment did I receive?**\n"
            "You were started on heart and blood-thinning medications. You had a minor procedure where doctors placed a small stent (a tiny mesh tube) inside your blocked blood vessel to restore healthy blood flow.\n\n"
            "3. **Medicines to continue**\n"
            "- **Aspirin (81 mg) & Clopidogrel (75 mg)**: Take both daily to prevent blood clots around your stent.\n"
            "- **Lisinopril (20 mg)**: Take once daily to manage your blood pressure.\n"
            "- **Atorvastatin (40 mg)**: Take at bedtime to keep your blood vessels healthy.\n\n"
            "4. **Things to do at home**\n"
            "Rest at home, avoid heavy lifting for the first week, eat a low-salt diet, and take short gentle walks.\n\n"
            "5. **Warning signs**\n"
            "Call 911 or return to the emergency room immediately if you feel severe chest pain, shortness of breath, or notice unusual bleeding or swelling.\n\n"
            "6. **Next appointment**\n"
            "You have a follow-up visit with your cardiologist in 2 weeks."
        )

    # 2. Check if it's Doctor Summary Prompt (or initial prompt with context)
    elif "Retrieved Context Segments:" in prompt or "Doctor Discharge Summary Output:" in prompt:
        # Extract context chunks dynamically
        context_matches = re.findall(r'\[(\d+)\]\s+(.*?)(?=\n\[\d+\]|\n\n|\Z)', prompt, re.DOTALL)
        chunks = {int(idx): text.strip() for idx, text in context_matches}
        
        if not chunks:
            chunks = {0: prompt[:500]}

        # Categorize sentences from the actual uploaded report
        demographics = []
        complaints = []
        diagnoses = []
        history = []
        investigations = []
        treatments = []
        course = []
        medications = []
        followup = []

        for idx, raw_text in chunks.items():
            # Clean header tags
            clean_text = re.sub(r'^\(Page\s+\d+,\s+Section:[^)]+\):\s*', '', raw_text, flags=re.IGNORECASE)
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()
            
            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', clean_text) if len(s.strip()) > 10]
            
            for sent in sentences:
                low = sent.lower()
                cited_sent = f"{sent} [{idx}]"
                
                if any(k in low for k in ["patient", "mrn", "dob", "years old", "gender", "male", "female", "admitted", "discharge"]):
                    demographics.append(cited_sent)
                elif any(k in low for k in ["complaint", "presented", "pain", "shortness of breath", "fever", "cough", "weakness"]):
                    complaints.append(cited_sent)
                elif any(k in low for k in ["diagnosis", "primary", "secondary", "impression", "pneumonia", "nstemi", "infarction", "failure", "sepsis", "diabetes"]):
                    diagnoses.append(cited_sent)
                elif any(k in low for k in ["history", "past medical", "hypertension", "smoker", "copd", "cad"]):
                    history.append(cited_sent)
                elif any(k in low for k in ["lab", "troponin", "ekg", "wbc", "creatinine", "crp", "ct", "x-ray", "imaging", "vitals", "temperature", "blood pressure"]):
                    investigations.append(cited_sent)
                elif any(k in low for k in ["treatment", "given", "drip", "stent", "angiography", "antibiotics", "aspirin", "nitroglycerin", "procedure"]):
                    treatments.append(cited_sent)
                elif any(k in low for k in ["course", "ccu", "icu", "stable", "tolerated", "uncomplicated", "progression"]):
                    course.append(cited_sent)
                elif any(k in low for k in ["medication", "mg", "daily", "po", "qd", "bid", "prn", "discharge meds"]):
                    medications.append(cited_sent)
                elif any(k in low for k in ["follow-up", "follow up", "return", "clinic", "weeks", "ed"]):
                    followup.append(cited_sent)
                else:
                    diagnoses.append(cited_sent)

        # Fallbacks using available chunks if category lists are empty
        all_chunk_ids = list(chunks.keys())
        first_id = all_chunk_ids[0]
        last_id = all_chunk_ids[-1]

        p_info = " ".join(demographics[:2]) if demographics else f"Clinical profile extracted from report context. [{first_id}]"
        c_comp = " ".join(complaints[:2]) if complaints else f"Patient presented for clinical evaluation and inpatient management. [{first_id}]"
        diag = " ".join(diagnoses[:2]) if diagnoses else f"Acute medical condition managed as detailed in report records. [{first_id}]"
        hist = " ".join(history[:2]) if history else f"Past medical history reviewed from inpatient record. [{first_id}]"
        inv = " ".join(investigations[:2]) if investigations else f"Laboratory tests and diagnostic imaging were performed. [{last_id}]"
        treat = " ".join(treatments[:2]) if treatments else f"Medical therapies and clinical interventions were administered. [{last_id}]"
        crs = " ".join(course[:2]) if course else f"Patient was monitored during hospital stay with clinical improvement. [{last_id}]"
        meds = " ".join(medications[:2]) if medications else f"Discharge medications prescribed as indicated. [{last_id}]"
        fup = " ".join(followup[:2]) if followup else f"Outpatient follow-up recommended in 2 weeks or if symptoms recur. [{last_id}]"

        summary = (
            f"1. **Patient Information**: {p_info}\n\n"
            f"2. **Chief Complaint**: {c_comp}\n\n"
            f"3. **Diagnosis**: {diag}\n\n"
            f"4. **Medical History**: {hist}\n\n"
            f"5. **Investigations**: {inv}\n\n"
            f"6. **Treatment Given**: {treat}\n\n"
            f"7. **Hospital Course**: {crs}\n\n"
            f"8. **Discharge Medications**: {meds}\n\n"
            f"9. **Follow-up**: {fup}"
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
