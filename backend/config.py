# config.py
"""
Loads environment variables and configuration constants.
Edit .env or set environment variables before running.
"""

from dotenv import load_dotenv
import os

load_dotenv()

# -------------------------------
# Qdrant
# -------------------------------
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "mini_rag_gemini")
DISTANCE = "COSINE"

# -------------------------------
# Gemini (Google Generative AI)
# -------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")  # your Google GenAI API key
if not GEMINI_API_KEY:
    raise ValueError("❌ GEMINI_API_KEY not found. Please set it in .env")

# Embedding model
EMBED_MODEL = os.getenv("EMBED_MODEL", "embedding-001")

# Auto-detect embedding dimension based on model
if EMBED_MODEL == "embedding-001":         # Gemini base embedding model
    EMBED_DIM = 768
elif EMBED_MODEL == "text-embedding-004":  # Gemini high-dim embedding model
    EMBED_DIM = 3072
else:
    # fallback (allow override via env)
    EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))

# Chat model
CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-1.5-flash-latest")

# -------------------------------
# Cohere (reranker)
# -------------------------------
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
RERANK_MODEL = os.getenv("RERANK_MODEL", "rerank")

# -------------------------------
# Chunking parameters
# -------------------------------
TARGET_TOKENS = int(os.getenv("TARGET_TOKENS", "1000"))
OVERLAP_TOKENS = int(
    os.getenv("OVERLAP_TOKENS", str(int(int(os.getenv("TARGET_TOKENS", "1000")) * 0.15)))
)

# -------------------------------
# Retriever / reranker settings
# -------------------------------
TOPK_VECTOR = int(os.getenv("TOPK_VECTOR", "12"))
MMR_K = int(os.getenv("MMR_K", "8"))
RERANK_TOPK = int(os.getenv("RERANK_TOPK", "5"))

# -------------------------------
# Misc
# -------------------------------
MAX_SNIPPET_CHARS = int(os.getenv("MAX_SNIPPET_CHARS", "240"))
