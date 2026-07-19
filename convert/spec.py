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


class UnionStep(BaseModel):
    """Stack two or more inputs by column name (Alteryx Union tool)."""

    op: Literal["union"] = "union"
    id: str
    inputs: list[str] = Field(min_length=2)


class SortColumn(BaseModel):
    column: str
    descending: bool = False


class SortStep(BaseModel):
    op: Literal["sort"] = "sort"
    id: str
    input: str
    columns: list[SortColumn] = Field(min_length=1)


class DistinctStep(BaseModel):
    """Keep the first occurrence per key (Alteryx Unique tool's U output)."""

    op: Literal["distinct"] = "distinct"
    id: str
    input: str
    columns: list[str] = Field(default_factory=list)  # empty = all columns


class RecordIdStep(BaseModel):
    """Add a sequential identifier column (Alteryx Record ID tool)."""

    op: Literal["record_id"] = "record_id"
    id: str
    input: str
    column: str = "RecordID"


class PythonScriptStep(BaseModel):
    """Alteryx Python tool: the embedded notebook auto-adapted to run on Databricks.

    The renderer wraps the code in a function that converts the input to
    pandas, rewrites Alteryx.read/write calls, and returns a Spark dataframe.
    Runs on the driver — review for large data volumes.
    """

    op: Literal["python_script"] = "python_script"
    id: str
    input: str
    code: str


class FindReplaceStep(BaseModel):
    """Alteryx Find Replace: lookup-join `right` and replace values in `find_column`."""

    op: Literal["find_replace"] = "find_replace"
    id: str
    left: str  # data stream (F)
    right: str  # lookup stream (R)
    find_column: str
    search_column: str
    replace_column: str
    find_mode: str = "FindAny"


class AppendFieldsStep(BaseModel):
    """Alteryx Append Fields: cartesian-append `source`'s fields onto `target`."""

    op: Literal["append_fields"] = "append_fields"
    id: str
    target: str
    source: str


class MacroCallStep(BaseModel):
    """Invoke a converted .yxmc macro, emitted as a generated utility function."""

    op: Literal["macro_call"] = "macro_call"
    id: str
    input: str
    macro: str  # name of a MacroUtility in PipelineSpec.macros


class CleanseStep(BaseModel):
    """Data Cleansing macro converted to a generated cleanse_columns utility call."""

    op: Literal["cleanse"] = "cleanse"
    id: str
    input: str
    columns: list[str] | None = None  # None = all columns
    trim: bool = False
    collapse_whitespace: bool = False
    remove_all_whitespace: bool = False
    nulls_to_blank: bool = False
    numeric_nulls_to_zero: bool = False
    case: Literal["upper", "lower", "title"] | None = None


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
    | UnionStep
    | SortStep
    | DistinctStep
    | RecordIdStep
    | CleanseStep
    | MacroCallStep
    | PythonScriptStep
    | FindReplaceStep
    | AppendFieldsStep
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


class MacroUtility(BaseModel):
    """A converted .yxmc macro: rendered as one generated function per artifact.

    `steps` reference the special input id "macro_input" (the function's
    dataframe parameter); `returns` is the id of the step whose result the
    function returns.
    """

    name: str  # sanitized python identifier, e.g. macro_normalize_names
    returns: str
    steps: list[Step]


class PipelineSpec(BaseModel):
    """The full spec for one converted pipeline object."""

    name: str
    language: Language
    source: SourceRef
    target: TargetRef
    steps: list[Step]
    macros: list[MacroUtility] = Field(default_factory=list)
    functions_used: list[str] = Field(default_factory=list)

    def step_by_id(self, step_id: str) -> Step | None:
        return next((s for s in self.steps if s.id == step_id), None)
