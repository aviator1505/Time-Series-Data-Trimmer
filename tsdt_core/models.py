"""Validated persistence models shared across the application."""
from __future__ import annotations

import uuid

from pydantic import BaseModel, ConfigDict, Field


class AnnotationSegment(BaseModel):
    """A labeled time segment. Unknown fields from newer file formats are ignored."""

    model_config = ConfigDict(extra="ignore")

    start: float
    end: float
    label: str
    track: str = "default"
    color: str = "#4e79a7"
    id: int = Field(default_factory=lambda: uuid.uuid4().int & 0x7FFFFFFF)
    episode_index: int | None = None  # Manual episode index override for CSV export


class OperationRecord(BaseModel):
    """One entry in the operation history. Unknown fields are ignored."""

    model_config = ConfigDict(extra="ignore")

    description: str
    params: dict
    start: float
    end: float
