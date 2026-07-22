"""Rule-based Workflow -> PipelineSpec conversion, used only when no LLM is configured.

This exists so the review app is fully explorable without an Anthropic API
key: it does a direct 1:1 mapping of IR nodes to spec steps (the same
mapping an LLM is asked to produce), skipping the LLM call and YAML
round-trip entirely. It is demo/offline tooling for the app, not part of
the core conversion path in `llm/convert.py`.

Real-workflow behaviors:
- unsupported tools in the middle of a chain are bridged over (a step whose
  upstream was skipped connects to the nearest supported ancestor instead);
- file paths / database refs that aren't Unity Catalog identifiers are
  retargeted into the target catalog.schema with a sanitized table name.
"""

from __future__ import annotations

import re
from typing import Literal, cast

from convert.spec import (
    AggregateStep,
    Aggregation,
    AppendFieldsStep,
    CleanseStep,
    ColumnSelection,
    ComputedColumn,
    DistinctStep,
    FilterStep,
    FindReplaceStep,
    JoinStep,
    MacroCallStep,
    MacroUtility,
    PipelineSpec,
    PythonScriptStep,
    ReadStep,
    RecordIdStep,
    SelectStep,
    SortColumn,
    SortStep,
    SourceRef,
    Step,
    TargetRef,
    UnionStep,
    WithColumnsStep,
    WriteStep,
)
from ingest.alteryx.ir import Node, ToolType, Workflow

# Alteryx Select field types -> Spark SQL cast targets. Unknown/empty types
# map to None (no cast emitted) rather than a guessed type.
_ALTERYX_TYPE_MAP = {
    "byte": "tinyint", "int16": "smallint", "int32": "int", "int64": "bigint",
    "fixeddecimal": "decimal", "float": "float", "double": "double",
    "bool": "boolean", "string": "string", "v_string": "string",
    "wstring": "string", "v_wstring": "string", "date": "date",
    "datetime": "timestamp", "time": "string", "spatialobj": "binary",
}


def _spark_type(alteryx_type: str | None) -> str | None:
    if not alteryx_type:
        return None
    return _ALTERYX_TYPE_MAP.get(alteryx_type.strip().lower())


_AGG_FUNC_MAP = {
    "sum": "sum", "count": "count", "avg": "avg", "min": "min", "max": "max",
    "first": "first", "last": "last", "stddev": "stddev", "std": "stddev",
    "var": "variance", "variance": "variance", "countdistinct": "countDistinct",
}

_UC_IDENTIFIER = re.compile(r"[A-Za-z0-9_]+(\.[A-Za-z0-9_]+){1,2}")
_SQL_HINT = re.compile(r"\bselect\b.+\bfrom\b", re.IGNORECASE | re.DOTALL)
_DATA_FILE_EXT = re.compile(r"\.(csv|tsv|txt|xlsx?|yxdb|json|parquet|avro|orc)$", re.IGNORECASE)
_DOTTED_DB_REF = re.compile(r"[\w ]+(\.[\w ]+)+")
_SQL_FROM_REF = re.compile(r"\bfrom\s+((?:\[[^\]]+\]|[\w.#])+)", re.IGNORECASE)


def _looks_like_sql(raw: str) -> bool:
    """True when an Input tool's source is a custom SQL query, not a table/path.

    Alteryx lets a DbFileInput's "file" be a full query (e.g.
    'aka:<conn>|||select * from ##TempFinalResult'). That has no table name
    to derive — mangling the query text into an identifier (e.g.
    'select_from_tempfinalresult') produces something that looks like a
    real, converted table reference but silently isn't.
    """
    return bool(_SQL_HINT.search(raw.split("|||")[-1]))


def _sanitize_table_name(raw: str) -> str:
    """Derive a Unity Catalog table name from a file path or database reference.

    Keeps the *actual* table name where one exists: the last segment of a
    dotted database reference ('IONPMVIEW.dbo.project_cube_le' ->
    'project_cube_le') or of a bracketed one ('[DB].[dbo].[Table]' ->
    'table'), and a file path's basename minus extension. Only genuinely
    unstructured text collapses wholesale to underscores.
    """
    name = raw.split("|||")[-1].strip()
    brackets = re.findall(r"\[([^\]]+)\]", name)
    if brackets:
        name = brackets[-1]
    elif _DOTTED_DB_REF.fullmatch(name) and not _DATA_FILE_EXT.search(name):
        name = name.rsplit(".", 1)[-1]
    else:
        name = re.split(r"[\\/]", name)[-1]
        name = re.sub(r"\.[A-Za-z0-9]+$", "", name)
    name = re.sub(r"\W+", "_", name).strip("_").lower()
    return name or "unnamed"


def _sql_source_table(raw: str) -> str | None:
    """The actual table read by a trivially simple SQL source, else None.

    Only a single-FROM, join-free query with a plain (non-temp) table
    reference qualifies — anything more complex has no single \"actual
    table\" and keeps its todo_source_* placeholder.
    """
    query = raw.split("|||")[-1]
    if len(re.findall(r"\bfrom\b", query, re.IGNORECASE)) != 1:
        return None
    if re.search(r"\bjoin\b", query, re.IGNORECASE):
        return None
    match = _SQL_FROM_REF.search(query)
    if match is None:
        return None
    ref = match.group(1)
    if "#" in ref:  # SQL Server temp table: no durable source to point at
        return None
    brackets = re.findall(r"\[([^\]]+)\]", ref)
    last = brackets[-1] if brackets else ref.rsplit(".", 1)[-1]
    name = re.sub(r"\W+", "_", last).strip("_").lower()
    return name or None


def _source_table(raw: str, target: TargetRef, node_id: str) -> str:
    """Keep UC-style identifiers as-is; retarget file/DB refs into the target schema.

    A simple 'SELECT ... FROM one_table' source is landed under that
    table's real name; a complex query has no single real table, so it
    lands as an explicit todo_source_<node_id> placeholder — loud and
    reviewable — instead of a plausible-but-wrong name derived from the
    query text.
    """
    if _UC_IDENTIFIER.fullmatch(raw):
        return raw
    if _looks_like_sql(raw):
        derived = _sql_source_table(raw)
        if derived is not None:
            return f"{target.catalog}.{target.schema_}.{derived}"
        return f"{target.catalog}.{target.schema_}.todo_source_{node_id}"
    return f"{target.catalog}.{target.schema_}.{_sanitize_table_name(raw)}"


class _Converter:
    def __init__(
        self,
        workflow: Workflow,
        target: TargetRef,
        macros: dict[str, Workflow] | None = None,
    ) -> None:
        self.workflow = workflow
        self.target = target
        self.macros = macros or {}
        self.macro_utilities: dict[str, MacroUtility] = {}
        # Nodes converted to a pass-through (no step emitted): references to
        # them re-route to their own resolved input. Populated in topological
        # order, so downstream lookups always find their aliases.
        self._elided: dict[str, str] = {}
        # Placeholder reads synthesized when a chain's source is an
        # unsupported tool (e.g. a connector the parser can't convert):
        # the pipeline stays runnable once an engineer lands that data in
        # the todo_source_* table.
        self.synthetic_reads: dict[str, ReadStep] = {}
        # Steps synthesized for secondary output anchors (Filter's False
        # stream, Join's unjoined Left/Right streams): keyed by derived id,
        # created once no matter how many consumers tap the same anchor.
        self.derived_steps: dict[str, Step] = {}
        self._node_by_id = {n.tool_id: n for n in workflow.nodes}

    def _follow_aliases(self, tool_id: str) -> str:
        while tool_id in self._elided:
            tool_id = self._elided[tool_id]
        return tool_id

    def _macro_utility(self, key: str) -> MacroUtility | None:
        """Convert a registered .yxmc into a generated utility (cached per spec)."""
        key = key.lower()
        if key in self.macro_utilities:
            return self.macro_utilities[key]
        macro_wf = self.macros.get(key)
        if macro_wf is None:
            return None

        sub = _Converter(macro_wf, self.target, self.macros)
        steps: list[Step] = []
        for inner in macro_wf.topological_order():
            step = sub.step_for_node(inner)
            if step is None or isinstance(step, WriteStep):
                continue
            steps.append(step)
        outputs = [n for n in macro_wf.nodes if n.tool_type == ToolType.MACRO_OUTPUT]
        if not outputs or not steps:
            return None
        returns = sub._elided.get(outputs[0].tool_id, steps[-1].id)
        ordered = _topological_steps(
            list(sub.synthetic_reads.values()) + steps + list(sub.derived_steps.values())
        )
        utility = MacroUtility(
            name="macro_" + re.sub(r"\W+", "_", key).strip("_"), returns=returns, steps=ordered
        )
        self.macro_utilities[key] = utility
        return utility

    def _placeholder_read(self, for_tool_id: str) -> str:
        """Synthesize a TODO source table for a chain whose real source is unsupported."""
        step_id = f"src_{for_tool_id}"
        if step_id not in self.synthetic_reads:
            table = f"{self.target.catalog}.{self.target.schema_}.todo_source_{for_tool_id}"
            self.synthetic_reads[step_id] = ReadStep(
                id=step_id, source_table=table, alias=f"todo_source_{for_tool_id}"
            )
        return step_id

    def _input_id(self, node: Node) -> str:
        """Nearest emitted upstream step id, bridging unsupported and elided nodes."""
        for edge in node.upstream_edges:
            if edge.origin_id in self._node_by_id:
                return self._supply(edge.origin_id, edge.origin_anchor)
            resolved = self.workflow.resolve_supported_upstream(edge.origin_id)
            if resolved is not None:
                return self._follow_aliases(resolved)
        # IR persisted before anchors were captured: fall back to plain ids.
        for upstream in node.upstream_ids:
            resolved = self.workflow.resolve_supported_upstream(upstream)
            if resolved is not None:
                return self._follow_aliases(resolved)
        return self._placeholder_read(node.tool_id)

    def _resolve(self, tool_id: str) -> str:
        resolved = self.workflow.resolve_supported_upstream(tool_id)
        if resolved is None:
            return self._placeholder_read(tool_id)
        return self._follow_aliases(resolved)

    def _join_sides(self, join_node: Node) -> tuple[str, str, list[str], list[str]]:
        """The resolved (left_id, right_id, left_keys, right_keys) of a Join node.

        Shared by the Join conversion itself and the derived anti-join steps
        for its Left/Right anchors, so both agree on which stream is which.
        """
        labels = join_node.upstream_labels
        ids = join_node.upstream_ids
        left_raw = labels.get("Left") or (ids[0] if ids else join_node.tool_id)
        # Never let a missing label collapse both sides onto one node: that
        # renders as a self-join, silently corrupting the result.
        right_raw = labels.get("Right") or next(
            (uid for uid in ids if uid != left_raw), left_raw
        )
        left_keys = next((j.keys for j in join_node.join_inputs if j.side == "left"), [])
        right_keys = next((j.keys for j in join_node.join_inputs if j.side == "right"), [])
        left_id = self._supply(left_raw, self._anchor_of(join_node, left_raw, "Left"))
        right_id = self._supply(right_raw, self._anchor_of(join_node, right_raw, "Right"))
        return left_id, right_id, left_keys, right_keys

    def _anchor_of(self, node: Node, origin_id: str, dest_label: str) -> str:
        """The origin anchor of `node`'s edge from `origin_id` into `dest_label`."""
        for edge in node.upstream_edges:
            if edge.dest_label == dest_label and edge.origin_id == origin_id:
                return edge.origin_anchor
        for edge in node.upstream_edges:
            if edge.origin_id == origin_id:
                return edge.origin_anchor
        return ""

    def _supply(self, origin_id: str, anchor: str) -> str:
        """The step id supplying the data of `origin_id`'s `anchor` output.

        Most anchors are a tool's primary output and resolve as before. The
        secondary anchors carry *different rows* and get a derived step:
        - Filter "False": the negated filter (Alteryx's False stream);
        - Join "Left"/"Right": the unjoined rows of that side (anti-join);
        - Unique "Dup": no clean relational equivalent from the emitted
          steps, so it lands as an explicit todo_duplicates_* placeholder
          table rather than silently substituting the Unique stream.
        """
        origin = self._node_by_id.get(origin_id)
        if origin is None or not anchor:
            return self._resolve(origin_id)

        if origin.tool_type == ToolType.FILTER and anchor == "False":
            derived_id = f"{origin_id}_false"
            if derived_id not in self.derived_steps:
                condition = origin.filter_expression or "true"
                self.derived_steps[derived_id] = FilterStep(
                    id=derived_id,
                    input=self._input_id(origin),
                    condition=f"NOT ({condition})",
                )
            return derived_id

        if origin.tool_type == ToolType.JOIN and anchor in ("Left", "Right"):
            derived_id = f"{origin_id}_unjoined_{anchor.lower()}"
            if derived_id not in self.derived_steps:
                left_id, right_id, left_keys, right_keys = self._join_sides(origin)
                if anchor == "Left":
                    step = JoinStep(
                        id=derived_id, left=left_id, right=right_id,
                        left_keys=left_keys, right_keys=right_keys,
                        how="left_anti", use_function="safe_join",
                    )
                else:
                    step = JoinStep(
                        id=derived_id, left=right_id, right=left_id,
                        left_keys=right_keys, right_keys=left_keys,
                        how="left_anti", use_function="safe_join",
                    )
                self.derived_steps[derived_id] = step
            return derived_id

        if origin.tool_type == ToolType.UNIQUE and anchor.lower().startswith("dup"):
            step_id = f"dup_{origin_id}"
            if step_id not in self.synthetic_reads:
                table = (
                    f"{self.target.catalog}.{self.target.schema_}.todo_duplicates_{origin_id}"
                )
                self.synthetic_reads[step_id] = ReadStep(
                    id=step_id, source_table=table, alias=f"todo_duplicates_{origin_id}"
                )
            return step_id

        return self._resolve(origin_id)

    def step_for_node(self, node: Node) -> Step | None:
        target = self.target
        if node.tool_type == ToolType.INPUT:
            raw = node.table_name or node.tool_id
            table = _source_table(raw, target, node.tool_id)
            return ReadStep(id=node.tool_id, source_table=table, alias=table.rsplit(".", 1)[-1])

        if node.tool_type == ToolType.SELECT:
            listed = [f for f in node.fields if f.field != "*Unknown"]
            unknown_kept = any(f.field == "*Unknown" and f.selected for f in node.fields)
            if unknown_kept and not any(f.rename or not f.selected for f in listed):
                # "Keep all other columns" with no renames/drops on the listed
                # ones is a pass-through (its only effect would be type casts,
                # which ColumnSelection doesn't carry) — elide it.
                self._elided[node.tool_id] = self._input_id(node)
                return None
            columns = [
                ColumnSelection(
                    column=f.field, rename=f.rename, drop=not f.selected,
                    cast_type=_spark_type(f.new_type),
                )
                for f in listed
            ]
            return SelectStep(id=node.tool_id, input=self._input_id(node), columns=columns)

        if node.tool_type == ToolType.FILTER:
            return FilterStep(
                id=node.tool_id,
                input=self._input_id(node),
                condition=node.filter_expression or "true",
            )

        if node.tool_type == ToolType.FORMULA:
            computed = [ComputedColumn(name=f.field, expression=f.expression) for f in node.formulas]
            return WithColumnsStep(id=node.tool_id, input=self._input_id(node), columns=computed)

        if node.tool_type == ToolType.JOIN:
            left_id, right_id, left_keys, right_keys = self._join_sides(node)
            return JoinStep(
                id=node.tool_id,
                left=left_id,
                right=right_id,
                left_keys=left_keys,
                right_keys=right_keys,
                how="inner",
                use_function="safe_join",
            )

        if node.tool_type == ToolType.UNION:
            if node.upstream_edges:
                inputs = [
                    self._supply(e.origin_id, e.origin_anchor) for e in node.upstream_edges
                ]
            else:
                inputs = [self._resolve(u) for u in node.upstream_ids]
            if len(inputs) < 2:
                # A one-input Union is a pass-through in Alteryx. The old
                # fallback duplicated the single input, which self-unioned
                # the stream and doubled every row.
                self._elided[node.tool_id] = inputs[0] if inputs else self._input_id(node)
                return None
            return UnionStep(id=node.tool_id, inputs=inputs)

        if node.tool_type == ToolType.SORT:
            sort_columns = [
                SortColumn(column=f.field, descending=f.descending) for f in node.sort_fields
            ]
            if not sort_columns:
                return None
            return SortStep(id=node.tool_id, input=self._input_id(node), columns=sort_columns)

        if node.tool_type == ToolType.UNIQUE:
            return DistinctStep(
                id=node.tool_id, input=self._input_id(node), columns=node.unique_fields
            )

        if node.tool_type == ToolType.RECORD_ID:
            return RecordIdStep(
                id=node.tool_id,
                input=self._input_id(node),
                column=node.record_id_field or "RecordID",
            )

        if node.tool_type == ToolType.PYTHON:
            if not node.python_code:
                self._elided[node.tool_id] = self._input_id(node)
                return None
            return PythonScriptStep(
                id=node.tool_id, input=self._input_id(node), code=node.python_code
            )

        if node.tool_type == ToolType.FIND_REPLACE:
            if node.find_replace is None:
                self._elided[node.tool_id] = self._input_id(node)
                return None
            labels = node.upstream_labels
            ids = node.upstream_ids
            left = labels.get("Find") or labels.get("F") or (ids[0] if ids else node.tool_id)
            # Same self-collision guard as Join: with labels missing, pick a
            # *distinct* second id rather than replaying the first.
            right = labels.get("Replace") or labels.get("R") or next(
                (uid for uid in ids if uid != left), left
            )
            return FindReplaceStep(
                id=node.tool_id,
                left=self._supply(left, self._anchor_of(node, left, "F")),
                right=self._supply(right, self._anchor_of(node, right, "R")),
                find_column=node.find_replace.find_column,
                search_column=node.find_replace.search_column,
                replace_column=node.find_replace.replace_column,
                find_mode=node.find_replace.find_mode,
            )

        if node.tool_type == ToolType.APPEND_FIELDS:
            # Alteryx's actual destination-connection labels for this tool are
            # "Source" (the base/many-row stream) and "Targets" — plural, easy
            # to miss — for the single-row stream being appended onto it.
            labels = node.upstream_labels
            target_id = labels.get("Source")
            source_id = labels.get("Targets")
            if target_id is None or source_id is None:
                # Labels missing (older export?): fall back to positional
                # order, but never let both sides collide onto the same
                # node — that would silently render as a self cross-join
                # (every row paired with every row) instead of the intended
                # append, corrupting the row count without any error.
                ids = node.upstream_ids
                target_id = target_id or (ids[0] if ids else node.tool_id)
                source_id = source_id or next(
                    (uid for uid in ids if uid != target_id), target_id
                )
            return AppendFieldsStep(
                id=node.tool_id,
                target=self._supply(target_id, self._anchor_of(node, target_id, "Source")),
                source=self._supply(source_id, self._anchor_of(node, source_id, "Targets")),
            )

        if node.tool_type == ToolType.MACRO:
            utility = self._macro_utility(node.macro_name) if node.macro_name else None
            if utility is None:
                # Macro not uploaded (or not convertible): bridge over it so
                # the chain stays intact; it surfaces in the review warnings.
                self._elided[node.tool_id] = self._input_id(node)
                return None
            return MacroCallStep(id=node.tool_id, input=self._input_id(node), macro=utility.name)

        if node.tool_type == ToolType.MACRO_INPUT:
            self._elided[node.tool_id] = "macro_input"
            return None

        if node.tool_type == ToolType.MACRO_OUTPUT:
            self._elided[node.tool_id] = self._input_id(node)
            return None

        if node.tool_type == ToolType.CLEANSE:
            if node.cleanse is None:
                self._elided[node.tool_id] = self._input_id(node)
                return None
            return CleanseStep(
                id=node.tool_id,
                input=self._input_id(node),
                columns=node.cleanse.columns,
                trim=node.cleanse.trim,
                collapse_whitespace=node.cleanse.collapse_whitespace,
                remove_all_whitespace=node.cleanse.remove_all_whitespace,
                nulls_to_blank=node.cleanse.nulls_to_blank,
                numeric_nulls_to_zero=node.cleanse.numeric_nulls_to_zero,
                case=node.cleanse.case,  # type: ignore[arg-type]
            )

        if node.tool_type == ToolType.SUMMARIZE:
            group_by = [a.field for a in node.summarize_actions if a.action.lower() == "groupby"]
            aggregations = [
                Aggregation(
                    column=a.field,
                    func=_AGG_FUNC_MAP.get(a.action.lower(), "sum"),
                    alias=a.rename or f"{a.action.lower()}_{a.field}",
                )
                for a in node.summarize_actions
                if a.action.lower() != "groupby"
            ]
            return AggregateStep(
                id=node.tool_id, input=self._input_id(node), group_by=group_by, aggregations=aggregations
            )

        if node.tool_type == ToolType.OUTPUT:
            table = _sanitize_table_name(node.output_path or node.tool_id)
            mode = cast(
                "Literal['overwrite', 'append', 'merge']", node.output_mode or "overwrite"
            )
            return WriteStep(
                id=node.tool_id,
                input=self._input_id(node),
                target_table=f"{target.catalog}.{target.schema_}.{table}",
                mode=mode,
            )

        return None


def _step_inputs(step: Step) -> list[str]:
    if isinstance(step, JoinStep | FindReplaceStep):
        return [step.left, step.right]
    if isinstance(step, AppendFieldsStep):
        return [step.target, step.source]
    if isinstance(step, UnionStep):
        return list(step.inputs)
    input_id = getattr(step, "input", None)
    return [input_id] if input_id is not None else []


def _topological_steps(steps: list[Step]) -> list[Step]:
    """Order steps so every referenced input precedes its consumer.

    The workflow's own topological order can break when unsupported tools sit
    inside the chain (their edges are invisible to the IR traversal), so the
    emitted steps are re-sorted by their actual references.
    """
    by_id = {s.id: s for s in steps}
    ordered: list[Step] = []
    seen: set[str] = set()

    # Iterative DFS: step chains mirror the workflow's tool chains, which on
    # big workflows exceed Python's recursion limit (see topological_order).
    for root in steps:
        if root.id in seen:
            continue
        seen.add(root.id)
        stack: list[tuple[Step, list[str], int]] = [(root, _step_inputs(root), 0)]
        while stack:
            step, inputs, i = stack[-1]
            descended = False
            while i < len(inputs):
                upstream = by_id.get(inputs[i])
                i += 1
                if upstream is not None and upstream.id not in seen:
                    seen.add(upstream.id)
                    stack[-1] = (step, inputs, i)
                    stack.append((upstream, _step_inputs(upstream), 0))
                    descended = True
                    break
            if not descended:
                ordered.append(step)
                stack.pop()
    return ordered


def naive_spec_from_workflow(
    workflow: Workflow,
    target: TargetRef,
    macros: dict[str, Workflow] | None = None,
) -> PipelineSpec:
    """Directly map a Workflow's nodes onto PipelineSpec steps, no LLM involved."""
    converter = _Converter(workflow, target, macros)
    steps: list[Step] = []
    functions_used: set[str] = set()

    for node in workflow.topological_order():
        step = converter.step_for_node(node)
        if step is None:
            continue
        steps.append(step)

    steps = _topological_steps(
        list(converter.synthetic_reads.values())
        + steps
        + list(converter.derived_steps.values())
    )
    for step in steps:
        if isinstance(step, JoinStep) and step.use_function:
            functions_used.add(step.use_function)
    return PipelineSpec(
        name=workflow.name,
        language="pyspark",
        source=SourceRef(system="alteryx", object_name=workflow.name),
        target=target,
        steps=steps,
        macros=list(converter.macro_utilities.values()),
        functions_used=sorted(functions_used),
    )
