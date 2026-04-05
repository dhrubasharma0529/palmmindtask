

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

import numpy as np
import tiktoken

ChunkingStrategy = Literal["semantic", "context_header"]

_TOKENIZER = tiktoken.get_encoding("cl100k_base")

# --- Semantic chunking knobs ---
SEMANTIC_SIMILARITY_THRESHOLD: float = 0.85   # cosine sim above which sentences merge
SEMANTIC_MAX_SENTENCES: int = 15              # hard cap on sentences per chunk

# --- Context-header chunking knobs ---
DEFAULT_MAX_TOKENS: int = 512
DEFAULT_OVERLAP_TOKENS: int = 50


@dataclass
class Chunk:
    text: str
    chunk_index: int
    strategy: ChunkingStrategy
    token_count: int


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def chunk_text(
    text: str,
    strategy: ChunkingStrategy,
    doc_title: str = "Document",
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[Chunk]:
    """
    Split *text* into chunks using the chosen strategy.

    Args:
        text:           Full extracted document text.
        strategy:       "semantic" or "context_header".
        doc_title:      Document title used in context-header prefix.
        max_tokens:     Token window size (context_header only).
        overlap_tokens: Token overlap between windows (context_header only).

    Returns:
        Ordered list of Chunk dataclasses.
    """
    if strategy == "semantic":
        return _semantic_chunks(text)
    if strategy == "context_header":
        return _context_header_chunks(text, doc_title, max_tokens, overlap_tokens)
    raise ValueError(f"Unknown strategy: {strategy!r}")


# ---------------------------------------------------------------------------
# Strategy 1 — Semantic chunking
# ---------------------------------------------------------------------------


def _semantic_chunks(text: str) -> list[Chunk]:
    """
    Group consecutive sentences whose token-frequency cosine similarity
    exceeds SEMANTIC_SIMILARITY_THRESHOLD into a single chunk.
    """
    sentences = _split_sentences(text)
    if not sentences:
        return []

    vectors = [_token_freq_vector(s) for s in sentences]
    groups: list[list[str]] = []
    current: list[str] = [sentences[0]]

    for i in range(1, len(sentences)):
        sim = _cosine_similarity(vectors[i - 1], vectors[i])
        if sim >= SEMANTIC_SIMILARITY_THRESHOLD and len(current) < SEMANTIC_MAX_SENTENCES:
            current.append(sentences[i])
        else:
            groups.append(current)
            current = [sentences[i]]
    groups.append(current)

    chunks: list[Chunk] = []
    for idx, group in enumerate(groups):
        chunk_text = " ".join(group).strip()
        if chunk_text:
            chunks.append(
                Chunk(
                    text=chunk_text,
                    chunk_index=idx,
                    strategy="semantic",
                    token_count=_count_tokens(chunk_text),
                )
            )
    return chunks


def _split_sentences(text: str) -> list[str]:
    """Split text on sentence-ending punctuation followed by whitespace."""
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in parts if s.strip()]


def _token_freq_vector(sentence: str) -> dict[str, int]:
    """Build a token-frequency dict for a sentence."""
    tokens = re.findall(r"\b\w+\b", sentence.lower())
    freq: dict[str, int] = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    return freq


def _cosine_similarity(a: dict[str, int], b: dict[str, int]) -> float:
    """Cosine similarity between two token-frequency dicts."""
    keys = set(a) | set(b)
    va = np.array([a.get(k, 0) for k in keys], dtype=float)
    vb = np.array([b.get(k, 0) for k in keys], dtype=float)
    norm_a, norm_b = np.linalg.norm(va), np.linalg.norm(vb)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(va, vb) / (norm_a * norm_b))


# ---------------------------------------------------------------------------
# Strategy 2 — Context-header chunking
# ---------------------------------------------------------------------------


def _context_header_chunks(
    text: str,
    doc_title: str,
    max_tokens: int,
    overlap_tokens: int,
) -> list[Chunk]:
    """
    Split text into fixed-token windows with overlap.
    Each chunk is prefixed with:
        [Document: <doc_title> | Section N]
    so the embedding captures document context alongside the content.
    """
    all_tokens = _TOKENIZER.encode(text)
    step = max(1, max_tokens - overlap_tokens)
    raw_chunks: list[str] = []

    start = 0
    while start < len(all_tokens):
        end = min(start + max_tokens, len(all_tokens))
        raw_chunks.append(_TOKENIZER.decode(all_tokens[start:end]))
        if end == len(all_tokens):
            break
        start += step

    chunks: list[Chunk] = []
    for idx, raw in enumerate(raw_chunks):
        header = f"[Document: {doc_title} | Section {idx + 1}]\n"
        full = header + raw.strip()
        chunks.append(
            Chunk(
                text=full,
                chunk_index=idx,
                strategy="context_header",
                token_count=_count_tokens(full),
            )
        )
    return chunks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _count_tokens(text: str) -> int:
    return len(_TOKENIZER.encode(text))
