"""
memory.py
---------
Per-session conversation history stored in Redis as a JSON list.
Uses manual SDK initialization (host, port, password) for better stability.
"""

import json
import redis.asyncio as aioredis

from app.config import settings

# Initialize the Redis client using specific arguments from your config
_redis: aioredis.Redis = aioredis.Redis(
    host=settings.redis_host,
    port=settings.redis_port,
    password=settings.redis_password,
    username="default",  # Default for Redis Cloud
    decode_responses=True
)

SESSION_TTL_SECONDS: int = 60 * 60 * 24   # 24 hours
MAX_TURNS: int = 20                        # rolling window

def _key(session_id: str) -> str:
    return f"chat:session:{session_id}"

async def get_history(session_id: str) -> list[dict[str, str]]:
    """
    Return conversation history for a session, oldest turn first.
    """
    try:
        raw = await _redis.get(_key(session_id))
        if not raw:
            return []
        return json.loads(raw)
    except Exception as e:
        # Log error here if needed
        return []

async def append_turn(session_id: str, role: str, content: str) -> None:
    """
    Append one turn to the session history and refresh the TTL.
    """
    history = await get_history(session_id)
    history.append({"role": role, "content": content})

    # Rolling window: keep only the last MAX_TURNS
    if len(history) > MAX_TURNS:
        history = history[-MAX_TURNS:]

    await _redis.set(
        _key(session_id), 
        json.dumps(history), 
        ex=SESSION_TTL_SECONDS
    )

async def clear_session(session_id: str) -> None:
    """Delete all stored history for a session."""
    await _redis.delete(_key(session_id))