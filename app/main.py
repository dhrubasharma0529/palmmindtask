

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from app.models.database import create_tables
from app.routers import chat, ingest
from app.services.vector_store import ensure_collection


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Run startup tasks before the app begins accepting requests."""
    create_tables()
    await ensure_collection()
    yield


app = FastAPI(
    title="RAG Document Intelligence API",
    description=(
        "Document ingestion with dual chunking strategies "
        "and conversational RAG with Redis memory and interview booking."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(ingest.router)
app.include_router(chat.router)


@app.get("/health", tags=["Health"], summary="Health check")
def health_check() -> dict[str, str]:
    """Returns 200 OK when the service is up."""
    return {"status": "ok"}
