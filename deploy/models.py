"""Pydantic models for the subset of Databricks Asset Bundle config we generate.

Only what's needed for one job running one rendered pipeline module per
environment target — the full DAB schema is much larger; extend here as
more of it is needed.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, model_validator

DABMode = Literal["development", "production"]


def dab_identifier(name: str) -> str:
    """Sanitize a name into a valid Terraform/Databricks resource identifier.

    Terraform resource names (and job/pipeline resource keys and task keys)
    must start with a letter or underscore and contain only letters, digits,
    underscores, and dashes — so a workflow named "Alteryx Use Case Workflow"
    cannot be used as a resource key verbatim.
    """
    ident = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_-")
    if not ident:
        return "workflow"
    if not re.match(r"[A-Za-z_]", ident):
        ident = f"_{ident}"
    return ident


class DABTarget(BaseModel):
    mode: DABMode = "development"
    workspace_host: str
    catalog: str
    schema_: str = Field(alias="schema")

    model_config = {"populate_by_name": True}


class DABTask(BaseModel):
    """One job task: a spark_python_task by default, a notebook_task if notebook_path is set."""

    task_key: str
    python_file: str | None = None
    notebook_path: str | None = None
    parameters: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _exactly_one_source(self) -> DABTask:
        if (self.python_file is None) == (self.notebook_path is None):
            raise ValueError("A DABTask needs exactly one of python_file or notebook_path")
        return self


class DABJob(BaseModel):
    name: str  # human-readable display name (may contain spaces)
    tasks: list[DABTask]
    key: str | None = None  # Terraform resource key; auto-derived from name if unset

    @model_validator(mode="after")
    def _default_key(self) -> DABJob:
        if self.key is None:
            self.key = dab_identifier(self.name)
        return self


class DABPipeline(BaseModel):
    """A Lakeflow/Spark Declarative Pipeline resource (for SDP-rendered specs)."""

    name: str  # human-readable display name (may contain spaces)
    catalog: str
    schema_: str = Field(alias="schema")
    library_path: str
    # Additional pipeline source files, e.g. a generated utility module that
    # library_path's main file imports (macro/cleanse helpers). Lakeflow
    # pipelines only execute files declared as libraries, so a sibling
    # helper module must be listed here too, not just deployed alongside.
    extra_library_paths: list[str] = Field(default_factory=list)
    serverless: bool = True
    key: str | None = None  # Terraform resource key; auto-derived from name if unset

    model_config = {"populate_by_name": True}

    @model_validator(mode="after")
    def _default_key(self) -> DABPipeline:
        if self.key is None:
            self.key = dab_identifier(self.name)
        return self


class DABBundle(BaseModel):
    """Top-level model that maps directly onto databricks.yml.

    A bundle carries a job (script/notebook deployments), a pipeline (SDP
    deployments), or both; it must not be empty.
    """

    bundle_name: str
    job: DABJob | None = None
    pipeline: DABPipeline | None = None
    targets: dict[str, DABTarget]

    @model_validator(mode="after")
    def _has_resources(self) -> DABBundle:
        if self.job is None and self.pipeline is None:
            raise ValueError("A DABBundle needs at least one of job or pipeline")
        return self
