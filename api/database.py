"""SQLAlchemy persistence for the identity gallery and analysis jobs."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import DateTime, Float, Integer, JSON, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


DATABASE_URL = "sqlite:///./database/irissecure.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class Identity(Base):
    __tablename__ = "identities"

    user_id: Mapped[str] = mapped_column(String(120), primary_key=True)
    eye: Mapped[str] = mapped_column(String(1), default="")
    person_id: Mapped[str] = mapped_column(String(120), default="")
    embedding: Mapped[list[float]] = mapped_column(JSON)
    num_samples: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class AnalysisJob(Base):
    __tablename__ = "analysis_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    status: Mapped[str] = mapped_column(String(20), default="queued")
    result: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    error: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


def initialize_database() -> None:
    if DATABASE_URL.startswith("sqlite"):
        os.makedirs("database", exist_ok=True)
    Base.metadata.create_all(bind=engine)
