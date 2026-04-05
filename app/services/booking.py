
import json
import re

from groq import AsyncGroq
from sqlalchemy.orm import Session

from app.config import settings
from app.models.sql import InterviewBooking
from app.schemas.chat import BookingRequest

# Initialize the Groq client
# Ensure settings.groq_api_key is defined in your config
_client = AsyncGroq(api_key=settings.groq_api_key)

_SYSTEM_PROMPT = """
You are an assistant that detects interview booking requests.

Given a user message respond ONLY with a JSON object in this exact shape:
{
  "is_booking": true | false,
  "name":   "<string or null>",
  "email":  "<string or null>",
  "date":   "<YYYY-MM-DD or null>",
  "time":   "<HH:MM or null>",
  "missing_fields": ["name", "email", "date", "time"]
}

Rules:
- Set is_booking to true only when the user is clearly requesting to schedule an interview.
- Extract any fields already present in the message.
- List only the fields that are missing in missing_fields.
- Output valid JSON only — no prose, no markdown fences.
"""

async def detect_booking_intent(user_message: str) -> dict:
    """
    Classify whether *user_message* contains an interview booking request
    and extract any provided fields using Groq.
    """
    try:
        response = await _client.chat.completions.create(
            # Common Groq models: "llama-3.3-70b-versatile" or "mixtral-8x7b-32768"
            model=settings.groq_chat_model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0,
            max_tokens=256,
            # Groq supports JSON mode for more reliable parsing
            response_format={"type": "json_object"} 
        )

        raw = response.choices[0].message.content or "{}"
        return json.loads(raw)
        
    except (json.JSONDecodeError, Exception) as e:
        # Fallback on parse failure or API error
        return {
            "is_booking": False,
            "name": None,
            "email": None,
            "date": None,
            "time": None,
            "missing_fields": ["name", "email", "date", "time"],
        }

def save_booking(db: Session, booking: BookingRequest) -> InterviewBooking:
    """
    Persist a validated booking to the SQL database.
    (This logic remains unchanged as it is database-specific).
    """
    record = InterviewBooking(
        name=booking.name,
        email=str(booking.email),
        date=booking.date,
        time=booking.time,
        session_id=booking.session_id,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record