from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, String, func
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"

    id               = Column(Integer, primary_key=True, autoincrement=True)
    doc_id           = Column(String(36), unique=True, nullable=False, index=True)   # UUID length
    filename         = Column(String(255), nullable=False)
    file_type        = Column(String(10), nullable=False)          # "pdf" | "txt"
    chunking_strategy = Column(String(50), nullable=False)         # "semantic" | "context_header"
    chunk_count      = Column(Integer, nullable=False)
    created_at       = Column(DateTime, default=datetime.utcnow, server_default=func.now())


class InterviewBooking(Base):
    __tablename__ = "interview_bookings"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    name       = Column(String(255), nullable=False)
    email      = Column(String(255), nullable=False)
    date       = Column(String(20), nullable=False)    # "YYYY-MM-DD"
    time       = Column(String(10), nullable=False)    # "HH:MM"
    session_id = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, server_default=func.now())