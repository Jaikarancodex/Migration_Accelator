"""Metadata pydantic model stored alongside each object in the migration repo."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field


class ObjectMetadata(BaseModel):
    """One entry's `metadata.json` in the migration repo."""

    name: str
    source_system: str
    source_file: str
    object_type: str = "workflow"
    input_tables: list[str] = Field(default_factory=list)
    output_tables: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(
        default_factory=list, description="Names of other migration-repo objects this reads from"
    )
    unsupported_tool_count: int = 0
    ingested_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
