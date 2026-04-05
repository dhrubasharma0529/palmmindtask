from datetime import datetime

from pydantic import BaseModel, EmailStr


class ChatRequest(BaseModel):
    session_id: str
    query: str


class SourceChunk(BaseModel):
    doc_id: str
    chunk_index: int
    text: str
    score: float


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: list[SourceChunk]
    booking_detected: bool


class BookingRequest(BaseModel):
    name: str
    email: EmailStr
    date: str           # "YYYY-MM-DD"
    time: str           # "HH:MM"
    session_id: str | None = None


class BookingResponse(BaseModel):
    id: int
    name: str
    email: str
    date: str
    time: str
    session_id: str | None
    created_at: datetime
    message: str

    model_config = {"from_attributes": True}


class BookingListResponse(BaseModel):
    bookings: list[BookingResponse]
    total: int


class ConversationTurn(BaseModel):
    role: str       # "user" | "assistant"
    content: str


class SessionHistoryResponse(BaseModel):
    session_id: str
    history: list[ConversationTurn]
