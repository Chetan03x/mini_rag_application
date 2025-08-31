# vectorstore.py
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct
from config import QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION, EMBED_DIM

# Global client
_qdrant_client: Optional[QdrantClient] = None

def get_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY
        )
    return _qdrant_client

def init_vectorstore():
    client = get_client()
    client.recreate_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE)
    )
    return client

def upsert_points(ids: List[str], vectors: List[List[float]], payloads: List[Dict[str, Any]]):
    client = get_client()
    points = [
        PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i])
        for i in range(len(ids))
    ]
    return client.upsert(collection_name=QDRANT_COLLECTION, points=points)

def search_vectors(vector: List[float], top_k: int = 5, with_payload: bool = True):
    client = get_client()
    return client.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=vector,
        limit=top_k,
        with_payload=with_payload
    )
