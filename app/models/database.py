"""
database.py
-----------
SQLAlchemy engine and session setup for MySQL.
Tables are created automatically on app startup via create_tables().
"""

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import settings
from app.models.sql import Base

engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,      # reconnect automatically if the MySQL connection drops
    pool_recycle=3600,       # recycle connections every hour
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def create_tables() -> None:
    """Create all ORM-defined tables in MySQL if they don't already exist."""
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency — yields a DB session and closes it when the request ends."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
