"""
vector_store.py
---------------
All Qdrant Cloud interactions:
  - ensure_collection : create collection on startup if absent
  - upsert_chunks     : store document chunk vectors
  - search            : cosine similarity search
"""

import hashlib
from dataclasses import dataclass

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from app.config import settings
from app.services.embedder import EMBEDDING_DIMENSION

# Qdrant Cloud client — api_key required for cloud authentication
_client = AsyncQdrantClient(
    url=settings.qdrant_url,
    api_key=settings.qdrant_api_key,
)


@dataclass
class SearchResult:
    doc_id: str
    chunk_index: int
    text: str
    score: float


# ---------------------------------------------------------------------------
# Collection management
# ---------------------------------------------------------------------------


async def ensure_collection() -> None:
    """Create the Qdrant collection if it does not already exist."""
    existing = await _client.get_collections()
    names = {c.name for c in existing.collections}
    if settings.qdrant_collection not in names:
        await _client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(
                size=EMBEDDING_DIMENSION,
                distance=Distance.COSINE,
            ),
        )


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------


async def upsert_chunks(
    doc_id: str,
    texts: list[str],
    embeddings: list[list[float]],
) -> None:
    """
    Upsert chunk vectors into Qdrant Cloud.

    Each point payload stores the original text and metadata so retrieval
    returns human-readable content without a secondary DB lookup.

    Args:
        doc_id:     Unique document identifier.
        texts:      Chunk text strings (aligned with embeddings).
        embeddings: Embedding vectors (same order as texts).
    """
    points = [
        PointStruct(
            id=_point_id(doc_id, idx),
            vector=embedding,
            payload={
                "doc_id": doc_id,
                "chunk_index": idx,
                "text": text,
            },
        )
        for idx, (text, embedding) in enumerate(zip(texts, embeddings))
    ]
    await _client.upsert(
        collection_name=settings.qdrant_collection,
        points=points,
    )


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------


async def search(
    query_embedding: list[float],
    top_k: int = 5,
    doc_id_filter: str | None = None,
) -> list[SearchResult]:
    """
    Find the top-k most similar chunks for a query embedding.

    Args:
        query_embedding: Embedded query vector.
        top_k:           Number of results to return.
        doc_id_filter:   If provided, restricts search to a single document.

    Returns:
        SearchResult list ordered by descending cosine similarity.
    """
    query_filter: Filter | None = None
    if doc_id_filter:
        query_filter = Filter(
            must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id_filter))]
        )

    hits = await _client.search(
        collection_name=settings.qdrant_collection,
        query_vector=query_embedding,
        limit=top_k,
        query_filter=query_filter,
        with_payload=True,
    )

    return [
        SearchResult(
            doc_id=r.payload["doc_id"],
            chunk_index=r.payload["chunk_index"],
            text=r.payload["text"],
            score=r.score,
        )
        for r in hits
        if r.payload
    ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _point_id(doc_id: str, chunk_index: int) -> str:
    """Deterministic point ID derived from doc_id and chunk index."""
    return hashlib.md5(f"{doc_id}:{chunk_index}".encode()).hexdigest()
