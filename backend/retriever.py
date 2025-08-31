# retriever.py
"""
Retriever: searches Qdrant -> applies MMR -> optionally reranks with Cohere -> returns top chunks
"""
from typing import List, Dict, Any
import math
from vectorstore import search_vectors
from config import TOPK_VECTOR, MMR_K, RERANK_TOPK, RERANK_MODEL, COHERE_API_KEY
import cohere

# Initialize Cohere client only if API key is set
co = cohere.Client(api_key=COHERE_API_KEY) if COHERE_API_KEY else None

def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x*y for x,y in zip(a,b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(x*x for x in b))
    return dot / (na * nb + 1e-8)

def mmr(candidates: List[Dict[str, Any]], k: int = MMR_K, lamb: float = 0.5) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    remaining = candidates.copy()
    while len(selected) < k and remaining:
        best_idx = None
        best_val = -1e9
        for i, c in enumerate(remaining):
            sim_q = c.get("score", 0.0)
            sim_sel = max(_cosine(c["vector"], s["vector"]) for s in selected) if selected else 0.0
            val = lamb * sim_q - (1.0 - lamb) * sim_sel
            if val > best_val:
                best_val = val
                best_idx = i
        selected.append(remaining.pop(best_idx))
    return selected

def cohere_rerank(query: str, docs: List[str], model: str = RERANK_MODEL, top_k: int = RERANK_TOPK):
    """Calls Cohere rerank API. Returns (ranked_docs, raw_response)"""
    if not co or not docs:
        return [], None
    try:
        resp = co.rerank(model=model, query=query, documents=docs)
        ranked = sorted(resp.results, key=lambda r: r.score, reverse=True)
        top = ranked[:top_k]
        ranked_docs = [docs[r.index] for r in top]
        return ranked_docs, resp
    except Exception as e:
        print(f"⚠️ Cohere rerank failed: {e}")
        return [], None

def retrieve(query_vector: List[float], query_text: str,
             topk: int = TOPK_VECTOR, mmr_k: int = MMR_K, rerank_topk: int = RERANK_TOPK):
    """Full retrieval pipeline: vector search -> MMR -> optional rerank"""
    hits = search_vectors(query_vector, top_k=topk, with_payload=True)
    if not hits:
        return []

    candidates = [{"id": str(getattr(h, "id", None)),
                   "score": getattr(h, "score", 0.0),
                   "vector": getattr(h, "vector", None),
                   "payload": getattr(h, "payload", {})} for h in hits]

    mmr_sel = mmr(candidates, k=mmr_k, lamb=0.5)
    docs = [c["payload"].get("text", "") for c in mmr_sel]

    reranked_docs, rerank_meta = cohere_rerank(query_text, docs, top_k=rerank_topk)

    if rerank_meta:
        top_items = []
        try:
            ranked_results = sorted(rerank_meta.results, key=lambda r: r.score, reverse=True)
            for r in ranked_results[:rerank_topk]:
                idx = r.index
                top_items.append(mmr_sel[idx])
            return top_items
        except Exception:
            pass

    # fallback: just return MMR top-k
    return mmr_sel[:rerank_topk]
