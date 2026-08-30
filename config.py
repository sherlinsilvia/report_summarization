import os
from pathlib import Path
import ssl
import requests
import urllib3

# Bypass SSL verification errors for HuggingFace downloads on local environment
try:
    import os
    import httpx
    # Disable SSL verification for huggingface_hub and datasets
    os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"
    os.environ["CURL_CA_BUNDLE"] = ""
    os.environ["REQUESTS_CA_BUNDLE"] = ""
    
    # Disable urllib3 warnings
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    # Monkeypatch requests to force verify=False
    orig_request = requests.Session.request
    def unverified_request(self, *args, **kwargs):
        kwargs['verify'] = False
        return orig_request(self, *args, **kwargs)
    requests.Session.request = unverified_request
    
    # Configure huggingface_hub httpx client with verify=False
    try:
        from huggingface_hub.utils._http import set_client_factory, hf_request_event_hook
        def custom_client_factory() -> httpx.Client:
            return httpx.Client(
                event_hooks={"request": [hf_request_event_hook]},
                follow_redirects=True,
                timeout=None,
                verify=False
            )
        set_client_factory(custom_client_factory)
    except Exception as hf_err:
        pass
    
    # Python standard SSL bypass
    ssl._create_default_https_context = ssl._create_unverified_context
except Exception:
    pass

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = DATA_DIR / "reports"
PROCESSED_DIR = DATA_DIR / "processed"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
OUTPUTS_DIR = BASE_DIR / "outputs"
SUMMARIES_DIR = OUTPUTS_DIR / "summaries"
LOGS_DIR = OUTPUTS_DIR / "logs"
LOCAL_SUMMARIZER_DIR = OUTPUTS_DIR / "fine_tuned_summarizer"
LOCAL_EMBEDDER_DIR = OUTPUTS_DIR / "fine_tuned_embedder"

# Ensure all directories exist
for directory in [REPORTS_DIR, PROCESSED_DIR, EMBEDDINGS_DIR, SUMMARIES_DIR, LOGS_DIR, LOCAL_SUMMARIZER_DIR, LOCAL_EMBEDDER_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# LLM Providers Configuration
# Can be 'ollama', 'groq', 'together', 'local-hf', or 'mock'
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "mock")

# Ollama Setup
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

# Groq Setup
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# Together AI Setup
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "")
TOGETHER_MODEL = os.getenv("TOGETHER_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct-Turbo")

# NLI model name
# "cross-encoder/nli-deberta-v3-base" or "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
# Or a smaller one if needed. Let's use a standard robust cross-encoder
NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-base"
USE_FALLBACK_NLI = True # Fallback to keyword/regex matching if Hugging Face or PyTorch is not available
USE_FALLBACK_EMBEDDER = True # Fallback to dummy embeddings to run in offline/sandboxed environments

# Embedding Model Setup
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Trust Threshold and Iterations
TRUST_THRESHOLD = 0.80
MAX_DISC_TRIALS = 3

# Weights for Composite Trust Score (must sum to 1.0)
WEIGHTS = {
    "retrieval": 0.15,
    "bertscore": 0.20,
    "hallucination": 0.25,
    "citation": 0.25,
    "coverage": 0.15
}
