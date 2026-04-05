
from dataclasses import dataclass
from groq import AsyncGroq

from app.config import settings
from app.services import memory
from app.services.embedder import embed_query
from app.services.vector_store import SearchResult, search

# Initialize Groq Client
_client = AsyncGroq(api_key=settings.groq_api_key)

_TOP_K: int = 5

_SYSTEM_PROMPT = """\
You are a helpful assistant that answers questions using the document context provided.

Rules:
- Base your answer ONLY on the context provided below.
- If the context does not contain enough information, say: "I'm sorry, I don't have enough information in the provided documents to answer that."
- Be concise, professional, and factual.
- Do not mention the context or document IDs to the user unless specifically asked.
"""

@dataclass
class RAGResult:
    answer: str
    sources: list[SearchResult]

async def query(session_id: str, user_query: str) -> RAGResult:
    """
    Run the full RAG pipeline for a single user turn using Groq.
    """
    # 1. Generate embedding (likely using Gemini as configured earlier)
    query_vec = await embed_query(user_query)
    
    # 2. Retrieve from Qdrant
    chunks = await search(query_embedding=query_vec, top_k=_TOP_K)
    
    # 3. Load conversation history from Redis
    history = await memory.get_history(session_id)

    # 4. Prepare the message stack
    messages = _build_messages(history, chunks, user_query)

    # 5. Call Groq
    response = await _client.chat.completions.create(
        model=settings.groq_chat_model, # e.g., "llama-3.3-70b-versatile"
        messages=messages,
        temperature=0.2,
        max_tokens=1024,
    )
    answer = response.choices[0].message.content or ""

    # 6. Persist turns to Redis
    await memory.append_turn(session_id, role="user", content=user_query)
    await memory.append_turn(session_id, role="assistant", content=answer)

    return RAGResult(answer=answer, sources=chunks)

# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

def _build_messages(
    history: list[dict[str, str]],
    chunks: list[SearchResult],
    user_query: str,
) -> list[dict[str, str]]:
    """
    Assemble the Groq messages list.
    """
    # Inject context into the system prompt
    context_block = _format_context(chunks)
    full_system_prompt = f"{_SYSTEM_PROMPT}\n\n## Relevant Context:\n{context_block}"
    
    messages: list[dict[str, str]] = [{"role": "system", "content": full_system_prompt}]
    
    # Add conversation history
    messages.extend(history)
    
    # Add the current user query
    messages.append({"role": "user", "content": user_query})
    
    return messages

def _format_context(chunks: list[SearchResult]) -> str:
    """Render retrieved chunks as a structured text block."""
    if not chunks:
        return "No relevant context found."
    
    parts = [
        f"--- DOCUMENT CHUNK {i} ---\n{c.text}"
        for i, c in enumerate(chunks, start=1)
    ]
    return "\n\n".join(parts)