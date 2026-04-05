
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.models.database import get_db
from app.models.sql import Document
from app.schemas.ingest import DocumentListResponse, DocumentMeta, IngestResponse
from app.services.chunker import ChunkingStrategy, chunk_text
from app.services.embedder import embed_chunks
from app.services.extractor import SUPPORTED_TYPES, extract_text
from app.services.vector_store import upsert_chunks

import uuid

router = APIRouter(prefix="/ingest", tags=["Ingestion"])


@router.post(
    "",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and ingest a PDF or TXT document",
)
async def ingest_document(
    file: UploadFile = File(..., description="PDF or TXT file"),
    strategy: ChunkingStrategy = Form(
        default="semantic",
        description="Chunking strategy: 'semantic' or 'context_header'",
    ),
    db: Session = Depends(get_db),
) -> IngestResponse:
    """
    Full ingestion pipeline:
    1. Validate file type
    2. Extract text via `unstructured`
    3. Chunk using the chosen strategy
    4. Embed chunks with text-embedding-3-small
    5. Upsert vectors to Qdrant
    6. Persist document metadata to SQL
    """
    filename = file.filename or "upload"
    suffix = Path(filename).suffix.lower().lstrip(".")

    if suffix not in SUPPORTED_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '.{suffix}'. Accepted: {sorted(SUPPORTED_TYPES)}",
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    # --- Extract ---
    try:
        raw_text = extract_text(file_bytes, filename)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Text extraction failed: {exc}",
        ) from exc

    if not raw_text.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No text could be extracted from the uploaded file.",
        )

    # --- Chunk ---
    doc_title = Path(filename).stem
    chunks = chunk_text(raw_text, strategy=strategy, doc_title=doc_title)

    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Chunking produced no output. The file may be too short.",
        )

    # --- Embed ---
    embeddings = await embed_chunks([c.text for c in chunks])

    # --- Store vectors ---
    doc_id = str(uuid.uuid4())
    await upsert_chunks(doc_id=doc_id, texts=[c.text for c in chunks], embeddings=embeddings)

    # --- Persist metadata ---
    record = Document(
        doc_id=doc_id,
        filename=filename,
        file_type=suffix,
        chunking_strategy=strategy,
        chunk_count=len(chunks),
    )
    db.add(record)
    db.commit()

    return IngestResponse(
        doc_id=doc_id,
        filename=filename,
        file_type=suffix,
        chunking_strategy=strategy,
        chunk_count=len(chunks),
        message="Document ingested successfully.",
    )


@router.get(
    "",
    response_model=DocumentListResponse,
    summary="List all ingested documents",
)
def list_documents(db: Session = Depends(get_db)) -> DocumentListResponse:
    """Return metadata for every ingested document, newest first."""
    records = db.query(Document).order_by(Document.created_at.desc()).all()
    return DocumentListResponse(
        documents=[DocumentMeta.model_validate(r) for r in records],
        total=len(records),
    )


@router.get(
    "/{doc_id}",
    response_model=DocumentMeta,
    summary="Fetch metadata for a single document",
)
def get_document(doc_id: str, db: Session = Depends(get_db)) -> DocumentMeta:
    """Return stored metadata for *doc_id*. Returns 404 if not found."""
    record = db.query(Document).filter(Document.doc_id == doc_id).first()
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{doc_id}' not found.",
        )
    return DocumentMeta.model_validate(record)
