"""Configuration — everything can be overridden with environment variables or a .env file."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.split(" #", 1)[0]  # allow inline comments
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()


def _get(name: str, default: str) -> str:
    return os.environ.get(name, default)


# --- LLM (any OpenAI-compatible endpoint: Ollama, OpenAI, OpenRouter, Groq...) ---
LLM_BASE_URL = _get("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_API_KEY = _get("LLM_API_KEY", "ollama")
CHAT_MODEL = _get("CHAT_MODEL", "qwen2.5:3b")
EMBED_MODEL = _get("EMBED_MODEL", "nomic-embed-text")
LLM_TIMEOUT = float(_get("LLM_TIMEOUT", "120"))

# --- Customer API (mock CRM from project 1) ---
CRM_BASE_URL = _get("CRM_BASE_URL", "http://localhost:4000")
CRM_API_KEY = _get("CRM_API_KEY", "dev-secret-key")

# --- RAG ---
DOCS_DIR = Path(_get("DOCS_DIR", str(ROOT / "data" / "docs")))
INDEX_DIR = Path(_get("INDEX_DIR", str(ROOT / "data" / "index")))
CHUNK_SIZE = int(_get("CHUNK_SIZE", "900"))        # characters
CHUNK_OVERLAP = int(_get("CHUNK_OVERLAP", "150"))  # characters
TOP_K = int(_get("TOP_K", "4"))
MIN_SCORE = float(_get("MIN_SCORE", "0.25"))       # below this a chunk is ignored
HYBRID_ALPHA = float(_get("HYBRID_ALPHA", "0.7"))  # weight of vector score vs keyword score

# --- Agent ---
MAX_AGENT_STEPS = int(_get("MAX_AGENT_STEPS", "5"))
LOG_DIR = Path(_get("LOG_DIR", str(ROOT / "logs")))
