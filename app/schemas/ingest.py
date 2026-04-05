from datetime import datetime

from pydantic import BaseModel


class IngestResponse(BaseModel):
    doc_id: str
    filename: str
    file_type: str
    chunking_strategy: str
    chunk_count: int
    message: str


class DocumentMeta(BaseModel):
    doc_id: str
    filename: str
    file_type: str
    chunking_strategy: str
    chunk_count: int
    created_at: datetime

    model_config = {"from_attributes": True}


class DocumentListResponse(BaseModel):
    documents: list[DocumentMeta]
    total: int
