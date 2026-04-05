
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.models.database import get_db
from app.models.sql import InterviewBooking
from app.schemas.chat import (
    BookingListResponse,
    BookingRequest,
    BookingResponse,
    ChatRequest,
    ChatResponse,
    ConversationTurn,
    SessionHistoryResponse,
    SourceChunk,
)
from app.services import memory, rag
from app.services.booking import detect_booking_intent, save_booking

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post(
    "",
    response_model=ChatResponse,
    summary="Multi-turn RAG query with automatic booking detection",
)
async def chat(
    request: ChatRequest,
    db: Session = Depends(get_db),
) -> ChatResponse:
    """
    Main conversational endpoint.

    Flow:
    1. Run LLM booking-intent classification on the query.
    2. If booking intent detected:
       - All fields present → save booking, return confirmation.
       - Fields missing    → ask the user for the missing fields.
    3. Otherwise → run the standard RAG pipeline and return the answer
       with source chunk citations.

    Conversation history is persisted to Redis after every turn.
    """
    booking_info = await detect_booking_intent(request.query)
    booking_detected: bool = bool(booking_info.get("is_booking"))

    if booking_detected:
        missing: list[str] = booking_info.get("missing_fields", [])

        if missing:
            reply = (
                "I'd be happy to book an interview for you! "
                f"Could you please provide: {', '.join(missing)}?"
            )
        else:
            booking_req = BookingRequest(
                name=booking_info["name"],
                email=booking_info["email"],
                date=booking_info["date"],
                time=booking_info["time"],
                session_id=request.session_id,
            )
            save_booking(db, booking_req)
            reply = (
                "Your interview has been booked!\n"
                f"  Name:  {booking_info['name']}\n"
                f"  Email: {booking_info['email']}\n"
                f"  Date:  {booking_info['date']} at {booking_info['time']}"
            )

        await memory.append_turn(request.session_id, role="user", content=request.query)
        await memory.append_turn(request.session_id, role="assistant", content=reply)

        return ChatResponse(
            session_id=request.session_id,
            answer=reply,
            sources=[],
            booking_detected=True,
        )

    # Standard RAG flow
    result = await rag.query(session_id=request.session_id, user_query=request.query)

    sources = [
        SourceChunk(
            doc_id=s.doc_id,
            chunk_index=s.chunk_index,
            text=s.text,
            score=s.score,
        )
        for s in result.sources
    ]

    return ChatResponse(
        session_id=request.session_id,
        answer=result.answer,
        sources=sources,
        booking_detected=False,
    )


@router.post(
    "/book-interview",
    response_model=BookingResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Book an interview directly (no chat required)",
)
def book_interview(
    request: BookingRequest,
    db: Session = Depends(get_db),
) -> BookingResponse:
    """
    Persist an interview booking directly to SQL.
    Use this when all four fields (name, email, date, time) are already known.
    """
    record = save_booking(db, request)
    return BookingResponse(
        id=record.id,
        name=record.name,
        email=record.email,
        date=record.date,
        time=record.time,
        session_id=record.session_id,
        created_at=record.created_at,
        message="Interview booked successfully.",
    )


@router.get(
    "/sessions/{session_id}/history",
    response_model=SessionHistoryResponse,
    summary="Return conversation history for a session",
)
async def get_session_history(session_id: str) -> SessionHistoryResponse:
    """Fetch all stored turns for *session_id* from Redis, oldest first."""
    history = await memory.get_history(session_id)
    turns = [ConversationTurn(role=t["role"], content=t["content"]) for t in history]
    return SessionHistoryResponse(session_id=session_id, history=turns)


@router.get(
    "/bookings",
    response_model=BookingListResponse,
    summary="List all stored interview bookings",
)
def list_bookings(db: Session = Depends(get_db)) -> BookingListResponse:
    """Return all InterviewBooking records from SQL, newest first."""
    records = (
        db.query(InterviewBooking)
        .order_by(InterviewBooking.created_at.desc())
        .all()
    )
    bookings = [
        BookingResponse(
            id=r.id,
            name=r.name,
            email=r.email,
            date=r.date,
            time=r.time,
            session_id=r.session_id,
            created_at=r.created_at,
            message="",
        )
        for r in records
    ]
    return BookingListResponse(bookings=bookings, total=len(bookings))
