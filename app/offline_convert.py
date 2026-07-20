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

_AGG_FUNC_MAP = {"sum": "sum", "count": "count", "avg": "avg", "min": "min", "max": "max"}

_UC_IDENTIFIER = re.compile(r"[A-Za-z0-9_]+(\.[A-Za-z0-9_]+){1,2}")
_SQL_HINT = re.compile(r"\bselect\b.+\bfrom\b", re.IGNORECASE | re.DOTALL)


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

    Handles UNC/posix file paths (basename minus extension), Alteryx
    connection strings ('aka:...|||[DB].[dbo].[Table]' -> last bracket
    segment), and arbitrary text (non-word chars collapse to underscores).
    """
    name = raw.split("|||")[-1]
    brackets = re.findall(r"\[([^\]]+)\]", name)
    if brackets:
        name = brackets[-1]
    else:
        name = re.split(r"[\\/]", name)[-1]
        name = re.sub(r"\.[A-Za-z0-9]+$", "", name)
    name = re.sub(r"\W+", "_", name).strip("_").lower()
    return name or "unnamed"


def _source_table(raw: str, target: TargetRef, node_id: str) -> str:
    """Keep UC-style identifiers as-is; retarget file/DB refs into the target schema.

    A custom-SQL source has no real table name, so it's landed as an
    explicit todo_source_<node_id> placeholder — loud and reviewable —
    instead of a plausible-but-wrong name derived from the query text.
    """
    if _UC_IDENTIFIER.fullmatch(raw):
        return raw
    if _looks_like_sql(raw):
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
        ordered = _topological_steps(list(sub.synthetic_reads.values()) + steps)
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
                ColumnSelection(column=f.field, rename=f.rename, drop=not f.selected)
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
            labels = node.upstream_labels
            left_raw = labels.get("Left") or (
                node.upstream_ids[0] if node.upstream_ids else node.tool_id
            )
            right_raw = labels.get("Right") or (
                node.upstream_ids[1] if len(node.upstream_ids) > 1 else left_raw
            )
            left_id = self._resolve(left_raw)
            right_id = self._resolve(right_raw)
            left_keys = next((j.keys for j in node.join_inputs if j.side == "left"), [])
            right_keys = next((j.keys for j in node.join_inputs if j.side == "right"), [])
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
            resolved = [self._resolve(u) for u in node.upstream_ids]
            inputs = resolved if len(resolved) >= 2 else [self._input_id(node)] * 2
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
            left = labels.get("Find") or labels.get("F") or (
                node.upstream_ids[0] if node.upstream_ids else node.tool_id
            )
            right = labels.get("Replace") or labels.get("R") or (
                node.upstream_ids[1] if len(node.upstream_ids) > 1 else left
            )
            return FindReplaceStep(
                id=node.tool_id,
                left=self._resolve(left),
                right=self._resolve(right),
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
                id=node.tool_id, target=self._resolve(target_id), source=self._resolve(source_id)
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
            return WriteStep(
                id=node.tool_id,
                input=self._input_id(node),
                target_table=f"{target.catalog}.{target.schema_}.{table}",
                mode="overwrite",
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
    visiting: set[str] = set()
    done: set[str] = set()

    def visit(step: Step) -> None:
        if step.id in done or step.id in visiting:
            return
        visiting.add(step.id)
        for input_id in _step_inputs(step):
            upstream = by_id.get(input_id)
            if upstream is not None:
                visit(upstream)
        visiting.discard(step.id)
        done.add(step.id)
        ordered.append(step)

    for step in steps:
        visit(step)
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
        if isinstance(step, JoinStep) and step.use_function:
            functions_used.add(step.use_function)
        steps.append(step)

    steps = _topological_steps(list(converter.synthetic_reads.values()) + steps)
    return PipelineSpec(
        name=workflow.name,
        language="pyspark",
        source=SourceRef(system="alteryx", object_name=workflow.name),
        target=target,
        steps=steps,
        macros=list(converter.macro_utilities.values()),
        functions_used=sorted(functions_used),
    )
