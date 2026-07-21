"""Intermediate representation (IR) for a parsed Alteryx workflow.

The parser never hands raw XML to the LLM. It extracts an ordered set of
typed nodes plus their configuration into these pydantic models, which are
what the rest of the pipeline (repo layer, conversion, LLM prompt) consumes.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, PrivateAttr


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


class UpstreamEdge(BaseModel):
    """One incoming connection, with both endpoints' anchor names.

    Alteryx tools have multiple *output* anchors carrying different data —
    Filter emits True and False streams, Join emits L (unjoined left),
    J (joined), and R (unjoined right), Unique emits U and D. Which anchor a
    connection originates from decides what rows actually flow down it, so
    dropping it (as `upstream_ids` alone does) silently substitutes one
    stream for another.
    """

    origin_id: str
    origin_anchor: str = ""  # e.g. "True"/"False", "Left"/"Join"/"Right", "Unique"/"Dup"
    dest_label: str = ""  # e.g. "Left"/"Right", "Source"/"Targets", "F"/"R"


class Node(BaseModel):
    """A single tool in the Alteryx workflow, extracted into typed config."""

    tool_id: str
    tool_type: ToolType
    raw_plugin: str = Field(description="Original Alteryx plugin name, for traceability")
    upstream_ids: list[str] = Field(default_factory=list)
    upstream_edges: list[UpstreamEdge] = Field(
        default_factory=list,
        description="Incoming connections with origin/destination anchors; "
        "empty on IR persisted before anchors were captured",
    )

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
    output_mode: str | None = None  # OUTPUT: "overwrite" | "append" | "merge" (from Output Option)

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
    position: dict[str, float] = Field(default_factory=dict)


class Workflow(BaseModel):
    """The full parsed representation of one .yxmd file."""

    source_file: str
    name: str
    nodes: list[Node] = Field(default_factory=list)
    unsupported: list[UnsupportedTool] = Field(default_factory=list)

    # Lazy lookup maps for resolve_supported_upstream, rebuilt whenever the
    # node/unsupported lists are replaced (keyed on their identity+length).
    # Without this, each call rebuilds both maps — O(n) per call, O(n^2)
    # across a conversion — which big workflows turn into real wall time.
    _lookup_key: tuple[int, int, int, int] | None = PrivateAttr(default=None)
    _lookup_maps: tuple[set[str], dict[str, UnsupportedTool]] | None = PrivateAttr(default=None)

    def _upstream_maps(self) -> tuple[set[str], dict[str, UnsupportedTool]]:
        key = (id(self.nodes), len(self.nodes), id(self.unsupported), len(self.unsupported))
        if self._lookup_maps is None or self._lookup_key != key:
            self._lookup_maps = (
                {n.tool_id for n in self.nodes},
                {u.tool_id: u for u in self.unsupported},
            )
            self._lookup_key = key
        return self._lookup_maps

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
        supported, unsupported_by_id = self._upstream_maps()
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
        """Return nodes ordered so every node's upstream nodes precede it.

        Iterative DFS with an explicit stack: production workflows can chain
        hundreds of tools deep, and the recursive version blows Python's
        recursion limit (a hard crash on ingest) around ~500 chained tools.
        """
        visited: set[str] = set()
        ordered: list[Node] = []
        by_id = {n.tool_id: n for n in self.nodes}

        for root in self.nodes:
            if root.tool_id in visited:
                continue
            visited.add(root.tool_id)
            stack: list[tuple[Node, int]] = [(root, 0)]
            while stack:
                node, i = stack[-1]
                descended = False
                while i < len(node.upstream_ids):
                    up = by_id.get(node.upstream_ids[i])
                    i += 1
                    if up is not None and up.tool_id not in visited:
                        visited.add(up.tool_id)
                        stack[-1] = (node, i)
                        stack.append((up, 0))
                        descended = True
                        break
                if not descended:
                    ordered.append(node)
                    stack.pop()
        return ordered
