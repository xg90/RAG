"""Global constants and configuration.

Convention: every .py file only exposes functions/constants, with no
execution entry points; entry points live in notebooks/.
"""

from __future__ import annotations

import os
from pathlib import Path

# Package root = code/
PKG_ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv(path: Path) -> None:
    """Load KEY=VALUE lines from a .env file into os.environ (existing vars win)."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and value and key not in os.environ:
            os.environ[key] = value


# Read secrets/overrides from code/.env (gitignored); real env vars take precedence.
_load_dotenv(PKG_ROOT / ".env")

# Data dir: defaults to PKG_ROOT/dataset, overridable via RAG_DATA_DIR.
DATA_DIR = Path(os.environ.get("RAG_DATA_DIR", PKG_ROOT / "dataset"))
RESULTS_DIR = Path(os.environ.get("RAG_RESULTS_DIR", PKG_ROOT / "results"))
CORPUS_PATH = DATA_DIR / "corpus.jsonl"
QA_RETRIEVAL_PATH = DATA_DIR / "qa_retrieval.jsonl"
QA_GENERATION_PATH = DATA_DIR / "qa_generation.jsonl"
QA_UNANSWERABLE_PATH = DATA_DIR / "qa_unanswerable.jsonl"

# Retrieval parameters
BM25_K1 = 1.5
BM25_B = 0.75
TOP_K_BM25 = 20
TOP_K_TFIDF = 20
TOP_K_EVAL = 50  # evaluation depth must cover recall@50

# Generation parameters
LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
TOP_N_CONTEXT = 20
MAX_CONTEXT_TOKENS = 4096

# DeepSeek API (OpenAI-compatible) generation
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_MAX_TOKENS = 512
DEEPSEEK_TEMPERATURE = 0.0

# RAGAS evaluation (LLM judge + local embeddings)
RAGAS_LLM_MODEL = os.environ.get("RAGAS_LLM_MODEL", DEEPSEEK_MODEL)
RAGAS_API_KEY = os.environ.get("RAGAS_API_KEY", DEEPSEEK_API_KEY)
RAGAS_BASE_URL = os.environ.get("RAGAS_BASE_URL", DEEPSEEK_BASE_URL)
RAGAS_MAX_TOKENS = 1024
RAGAS_SAMPLE_SIZE = 50  # 0 = evaluate every answerable record
RAGAS_MAX_CONTEXTS = 20
RAGAS_CONCURRENCY = 2
RAGAS_EMBEDDING_MODEL = os.environ.get(
    "RAGAS_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"
)
RAGAS_EMBEDDING_DEVICE = os.environ.get("RAGAS_EMBEDDING_DEVICE", "cpu")
# HuggingFace endpoint used to fetch embedding models. huggingface.co is often
# unreachable in some networks; the public mirror is used by default.
RAGAS_HF_ENDPOINT = os.environ.get(
    "RAGAS_HF_ENDPOINT",
    os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"),
)

SEED = 42
