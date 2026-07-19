"""Intermediate representation (IR) for a parsed Alteryx workflow.

The parser never hands raw XML to the LLM. It extracts an ordered set of
typed nodes plus their configuration into these pydantic models, which are
what the rest of the pipeline (repo layer, conversion, LLM prompt) consumes.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ToolType(StrEnum):
    INPUT = "input"
    SELECT = "select"
    FILTER = "filter"
    FORMULA = "formula"
    JOIN = "join"
    UNION = "union"
    SORT = "sort"
    UNIQUE = "unique"
    RECORD_ID = "record_id"
    CLEANSE = "cleanse"
    MACRO = "macro"  # references a .yxmc macro workflow by name
    MACRO_INPUT = "macro_input"  # placeholder input inside a .yxmc
    MACRO_OUTPUT = "macro_output"  # placeholder output inside a .yxmc
    PYTHON = "python"  # Alteryx Python tool (embedded Jupyter notebook)
    FIND_REPLACE = "find_replace"
    APPEND_FIELDS = "append_fields"
    SUMMARIZE = "summarize"
    OUTPUT = "output"
    UNSUPPORTED = "unsupported"


class FieldSelection(BaseModel):
    """A single field's treatment in a Select tool."""

    field: str
    selected: bool = True
    rename: str | None = None
    new_type: str | None = None


class FormulaExpression(BaseModel):
    """A single new/updated field produced by a Formula tool."""

    field: str
    expression: str
    data_type: str | None = None


class JoinInput(BaseModel):
    """One side of a Join tool (Alteryx joins have a Left and a Right input)."""

    side: str  # "left" | "right"
    keys: list[str]


class SortField(BaseModel):
    """One column's ordering in a Sort tool."""

    field: str
    descending: bool = False


class FindReplaceConfig(BaseModel):
    """Alteryx Find Replace tool: lookup-join a reference stream and replace values."""

    find_column: str  # column in the data (F) stream searched
    search_column: str  # column in the lookup (R) stream matched against
    replace_column: str  # column in the lookup (R) stream supplying replacements
    find_mode: str = "FindAny"  # "FindAny" (substring) | whole-field match


class CleanseConfig(BaseModel):
    """Options extracted from a Data Cleansing macro (Cleanse.yxmc or DataCleansePro)."""

    columns: list[str] | None = None  # None = all columns
    trim: bool = False
    collapse_whitespace: bool = False
    remove_all_whitespace: bool = False
    nulls_to_blank: bool = False
    numeric_nulls_to_zero: bool = False
    case: str | None = None  # "upper" | "lower" | "title"


class SummarizeAction(BaseModel):
    """One aggregation or group-by action in a Summarize tool."""

    field: str
    action: str  # e.g. "GroupBy", "Sum", "Count", "Avg", "Min", "Max"
    rename: str | None = None


class Node(BaseModel):
    """A single tool in the Alteryx workflow, extracted into typed config."""

    tool_id: str
    tool_type: ToolType
    raw_plugin: str = Field(description="Original Alteryx plugin name, for traceability")
    upstream_ids: list[str] = Field(default_factory=list)

    # tool-specific configuration; only the relevant fields are populated
    connection: dict[str, Any] | None = None  # INPUT: connection string/path/table
    table_name: str | None = None  # INPUT/OUTPUT
    fields: list[FieldSelection] = Field(default_factory=list)  # SELECT
    filter_expression: str | None = None  # FILTER (True-output predicate)
    formulas: list[FormulaExpression] = Field(default_factory=list)  # FORMULA
    join_inputs: list[JoinInput] = Field(default_factory=list)  # JOIN
    sort_fields: list[SortField] = Field(default_factory=list)  # SORT
    unique_fields: list[str] = Field(default_factory=list)  # UNIQUE
    record_id_field: str | None = None  # RECORD_ID
    cleanse: CleanseConfig | None = None  # CLEANSE
    macro_name: str | None = None  # MACRO: the referenced .yxmc name (stem, lowercase)
    python_code: str | None = None  # PYTHON: code cells of the embedded notebook
    find_replace: FindReplaceConfig | None = None  # FIND_REPLACE
    upstream_labels: dict[str, str] = Field(
        default_factory=dict, description="Connection label -> origin tool id (e.g. Left/Right)"
    )
    summarize_actions: list[SummarizeAction] = Field(default_factory=list)  # SUMMARIZE
    output_path: str | None = None  # OUTPUT

    annotation: str | None = None
    position: dict[str, float] = Field(default_factory=dict)


class UnsupportedTool(BaseModel):
    """A tool the parser encountered but does not know how to convert.

    Logged rather than raising, so the rest of the workflow still parses.
    `upstream_ids` is kept so converters can bridge over the gap — a
    downstream node whose input is unsupported falls back to this node's
    own upstream, with the skipped tool becoming a documented TODO.
    """

    tool_id: str
    plugin: str
    reason: str
    upstream_ids: list[str] = Field(default_factory=list)


class Workflow(BaseModel):
    """The full parsed representation of one .yxmd file."""

    source_file: str
    name: str
    nodes: list[Node] = Field(default_factory=list)
    unsupported: list[UnsupportedTool] = Field(default_factory=list)

    def node_by_id(self, tool_id: str) -> Node | None:
        return next((n for n in self.nodes if n.tool_id == tool_id), None)

    def referenced_macros(self) -> list[str]:
        """Names (stems, lowercase) of .yxmc macros this workflow invokes."""
        return sorted({n.macro_name for n in self.nodes if n.macro_name})

    def resolve_supported_upstream(self, tool_id: str) -> str | None:
        """Follow `tool_id` upstream through unsupported tools to the nearest supported node.

        Returns `tool_id` itself when it is a supported node; otherwise walks
        the unsupported chain upward so converters can bridge over skipped
        tools instead of referencing steps that don't exist.
        """
        supported = {n.tool_id for n in self.nodes}
        unsupported_by_id = {u.tool_id: u for u in self.unsupported}
        seen: set[str] = set()
        current = tool_id
        while current not in supported:
            if current in seen:
                return None
            seen.add(current)
            skipped = unsupported_by_id.get(current)
            if skipped is None or not skipped.upstream_ids:
                return None
            current = skipped.upstream_ids[0]
        return current

    def topological_order(self) -> list[Node]:
        """Return nodes ordered so every node's upstream nodes precede it."""
        visited: set[str] = set()
        ordered: list[Node] = []
        by_id = {n.tool_id: n for n in self.nodes}

        def visit(node: Node) -> None:
            if node.tool_id in visited:
                return
            visited.add(node.tool_id)
            for up_id in node.upstream_ids:
                up = by_id.get(up_id)
                if up is not None:
                    visit(up)
            ordered.append(node)

        for n in self.nodes:
            visit(n)
        return ordered
