"""Pydantic models for the subset of Databricks Asset Bundle config we generate.

Only what's needed for one job running one rendered pipeline module per
environment target — the full DAB schema is much larger; extend here as
more of it is needed.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

DABMode = Literal["development", "production"]


class DABTarget(BaseModel):
    mode: DABMode = "development"
    workspace_host: str
    catalog: str
    schema_: str = Field(alias="schema")

    model_config = {"populate_by_name": True}


class DABTask(BaseModel):
    task_key: str
    python_file: str
    parameters: list[str] = Field(default_factory=list)


class DABJob(BaseModel):
    name: str
    tasks: list[DABTask]


class DABBundle(BaseModel):
    """Top-level model that maps directly onto databricks.yml."""

    bundle_name: str
    job: DABJob
    targets: dict[str, DABTarget]
