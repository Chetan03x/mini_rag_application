# embeddings.py
"""
Chunking + embedding helpers using Google Gemini embeddings (via google.generativeai).
Also supports reading Streamlit UploadedFile objects and PDFs.
"""

from typing import List, Dict, Any, Optional
import hashlib
import io
import time

# Optional deps
try:
    import pdfplumber
except Exception:
    pdfplumber = None

try:
    import tiktoken
except Exception:
    tiktoken = None

import google.generativeai as genai
from config import (
    EMBED_MODEL,
    EMBED_DIM,
    TARGET_TOKENS,
    OVERLAP_TOKENS,
    GEMINI_API_KEY,
)

# --- Gemini client ---
genai.configure(api_key=GEMINI_API_KEY)

# --- Token encoder (best-effort) ---
ENC = None
if tiktoken is not None:
    try:
        ENC = tiktoken.get_encoding("cl100k_base")
    except Exception:
        ENC = None


def _make_chunk_id(source: str, position: int) -> str:
    return hashlib.sha1(f"{source}::{position}".encode()).hexdigest()[:32]


def extract_text_from_file(file) -> str:
    """
    Accepts a Streamlit UploadedFile, a bytes buffer, or a file-like object.
    Returns extracted text. Supports .pdf, .txt, .md.
    """
    if not file:
        return ""

    # Try to read bytes from file-like objects (e.g., Streamlit UploadedFile)
    data: Optional[bytes] = None
    filename = getattr(file, "name", None)

    if hasattr(file, "read"):
        # Streamlit UploadedFile returns bytes
        data = file.read()
    elif isinstance(file, (bytes, bytearray)):
        data = bytes(file)

    if data is None:
        # Fallback: assume it's already text
        return str(file)

    # If we know the name, branch by extension
    ext = (filename or "").lower()

    if ext.endswith(".pdf"):
        if pdfplumber is None:
            # pdfplumber not installed
            return ""
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            pages = [p.extract_text() or "" for p in pdf.pages]
        return "\n".join(pages)

    # Default: treat as utf-8 text
    try:
        return data.decode("utf-8", errors="ignore")
    except Exception:
        return ""


def _tokenize(text: str) -> List[int]:
    if ENC is not None:
        return ENC.encode(text)
    # Fallback: naive token approximation (1 token ~= 4 chars)
    approx_token_size = max(1, len(text) // 4)
    return list(range(approx_token_size))


def _detokenize(tokens: List[int], original_text: str) -> str:
    if ENC is not None:
        return ENC.decode(tokens)
    # Fallback: approximate slice of original text by proportion
    if not tokens:
        return ""
    # proportionally map token indices back to characters
    total_tokens = max(tokens[-1] + 1, 1)
    char_len = len(original_text)
    start_char = int((tokens[0] / total_tokens) * char_len)
    end_char = int((tokens[-1] + 1) / total_tokens * char_len)
    return original_text[start_char:end_char]


def chunk_text(
    raw: str,
    source: str = "user-paste",
    title: str = "Untitled Document",
    target_tokens: int = TARGET_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
) -> List[Dict[str, Any]]:
    """
    Chunk `raw` into token-sized chunks.
    Returns list of dicts: {id, text, meta}
    """
    if not raw:
        return []

    tokens = _tokenize(raw)
    chunks: List[Dict[str, Any]] = []

    # Step = window - overlap (guard against bad config)
    step = max(1, target_tokens - overlap_tokens)
    pos = 0

    for start in range(0, len(tokens), step):
        end = min(start + target_tokens, len(tokens))
        if ENC is not None:
            slice_tokens = tokens[start:end]
            text = ENC.decode(slice_tokens)
        else:
            # Fallback approximate detokenization
            text = _detokenize(tokens[start:end], raw)

        cid = _make_chunk_id(source, pos)
        meta = {
            "source": source,
            "title": title,
            "position": pos,
            "chunk_id": str(pos),
        }
        chunks.append({"id": cid, "text": text, "meta": meta})
        pos += 1

        if end == len(tokens):
            break

    return chunks


def embed_texts(texts: List[str], model: str = EMBED_MODEL) -> List[List[float]]:
    """
    Call Gemini embeddings API for a list of texts.
    Returns list of embedding vectors (length should match EMBED_DIM).
    """
    if not texts:
        return []

    vectors: List[List[float]] = []
    for text in texts:
        try:
            resp = genai.embed_content(model=model, content=text)
            # Handle both dict-like and object-like responses
            vec = None
            if isinstance(resp, dict) and "embedding" in resp:
                vec = resp["embedding"]
            elif hasattr(resp, "embedding"):
                vec = resp.embedding
            elif isinstance(resp, dict) and "data" in resp:
                # very defensive fallback
                data = resp.get("data") or []
                if data and "embedding" in data[0]:
                    vec = data[0]["embedding"]

            if not isinstance(vec, list):
                raise RuntimeError("Embedding response missing 'embedding' list.")

            # Optional: warn on unexpected dims; Qdrant will reject wrong dims
            if EMBED_DIM and len(vec) != EMBED_DIM:
                # You can log/print a warning, but still return what API gives.
                # print(f"⚠️ Embedding dim {len(vec)} != EMBED_DIM {EMBED_DIM}")
                pass

            vectors.append(vec)
        except Exception as e:
            # Fallback zero vector to avoid crashing the pipeline
            # (Upsert will fail if Qdrant collection has another dim.)
            # print(f"❌ Embedding error: {e}")
            vectors.append([0.0] * EMBED_DIM)
    return vectors
