"""Pydantic models for the subset of Databricks Asset Bundle config we generate.

Only what's needed for one job running one rendered pipeline module per
environment target — the full DAB schema is much larger; extend here as
more of it is needed.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

DABMode = Literal["development", "production"]


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
    name: str
    tasks: list[DABTask]


class DABPipeline(BaseModel):
    """A Lakeflow/Spark Declarative Pipeline resource (for SDP-rendered specs)."""

    name: str
    catalog: str
    schema_: str = Field(alias="schema")
    library_path: str
    serverless: bool = True

    model_config = {"populate_by_name": True}


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
