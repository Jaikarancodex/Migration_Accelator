"""Pydantic models for the YAML pipeline spec — the single source of truth.

The LLM's job is to produce a document that validates against `PipelineSpec`,
not to write PySpark/SQL directly. `convert/renderer.py` deterministically
turns a validated spec into runnable code, which is what makes the output
regenerable (re-render anytime) and testable (validate the spec, snapshot
the render) instead of trusting free-form LLM code generation.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field

Language = Literal["pyspark", "sql"]
MedallionLayer = Literal["bronze", "silver", "gold"]
JoinHow = Literal["inner", "left", "right", "full"]


class ColumnSelection(BaseModel):
    column: str
    rename: str | None = None
    drop: bool = False


class ComputedColumn(BaseModel):
    name: str
    expression: str = Field(description="Source-language expression, e.g. Alteryx formula syntax")


class Aggregation(BaseModel):
    column: str
    func: str  # "sum" | "count" | "avg" | "min" | "max" | ...
    alias: str


class ReadStep(BaseModel):
    op: Literal["read"] = "read"
    id: str
    source_table: str
    alias: str


class SelectStep(BaseModel):
    op: Literal["select"] = "select"
    id: str
    input: str
    columns: list[ColumnSelection]


class FilterStep(BaseModel):
    op: Literal["filter"] = "filter"
    id: str
    input: str
    condition: str


class WithColumnsStep(BaseModel):
    op: Literal["with_columns"] = "with_columns"
    id: str
    input: str
    columns: list[ComputedColumn]


class JoinStep(BaseModel):
    op: Literal["join"] = "join"
    id: str
    left: str
    right: str
    left_keys: list[str]
    right_keys: list[str]
    how: JoinHow = "inner"
    use_function: str | None = "safe_join"


class AggregateStep(BaseModel):
    op: Literal["aggregate"] = "aggregate"
    id: str
    input: str
    group_by: list[str]
    aggregations: list[Aggregation]


class CallFunctionStep(BaseModel):
    """Invoke a named function from the reusable library (functions/registry.py)."""

    op: Literal["call_function"] = "call_function"
    id: str
    input: str
    function: str
    args: dict[str, str] = Field(default_factory=dict)


class WriteStep(BaseModel):
    op: Literal["write"] = "write"
    id: str
    input: str
    target_table: str
    mode: Literal["overwrite", "append", "merge"] = "overwrite"


Step = Annotated[
    ReadStep
    | SelectStep
    | FilterStep
    | WithColumnsStep
    | JoinStep
    | AggregateStep
    | CallFunctionStep
    | WriteStep,
    Field(discriminator="op"),
]


class SourceRef(BaseModel):
    system: str  # e.g. "alteryx"
    object_name: str  # e.g. workflow name / source file stem


class TargetRef(BaseModel):
    catalog: str
    schema_: str = Field(alias="schema")
    layer: MedallionLayer

    model_config = {"populate_by_name": True}


class PipelineSpec(BaseModel):
    """The full spec for one converted pipeline object."""

    name: str
    language: Language
    source: SourceRef
    target: TargetRef
    steps: list[Step]
    functions_used: list[str] = Field(default_factory=list)

    def step_by_id(self, step_id: str) -> Step | None:
        return next((s for s in self.steps if s.id == step_id), None)
