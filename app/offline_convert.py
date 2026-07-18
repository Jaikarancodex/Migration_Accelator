"""Rule-based Workflow -> PipelineSpec conversion, used only when no LLM is configured.

This exists so the review app is fully explorable without an Anthropic API
key: it does a direct 1:1 mapping of IR nodes to spec steps (the same
mapping an LLM is asked to produce), skipping the LLM call and YAML
round-trip entirely. It is demo/offline tooling for the app, not part of
the core conversion path in `llm/convert.py`.
"""

from __future__ import annotations

from convert.spec import (
    AggregateStep,
    Aggregation,
    ColumnSelection,
    ComputedColumn,
    FilterStep,
    JoinStep,
    PipelineSpec,
    ReadStep,
    SelectStep,
    SourceRef,
    Step,
    TargetRef,
    WithColumnsStep,
    WriteStep,
)
from ingest.alteryx.ir import Node, ToolType, Workflow

_AGG_FUNC_MAP = {"sum": "sum", "count": "count", "avg": "avg", "min": "min", "max": "max"}


def _input_id(node: Node, fallback: str) -> str:
    return node.upstream_ids[0] if node.upstream_ids else fallback


def _table_ref(target: TargetRef, table_name: str) -> str:
    return f"{target.catalog}.{target.schema_}.{table_name}"


def _step_for_node(node: Node, target: TargetRef) -> Step | None:
    if node.tool_type == ToolType.INPUT:
        table = node.table_name or node.tool_id
        return ReadStep(id=node.tool_id, source_table=table, alias=table.rsplit(".", 1)[-1])

    if node.tool_type == ToolType.SELECT:
        columns = [
            ColumnSelection(column=f.field, rename=f.rename, drop=not f.selected) for f in node.fields
        ]
        return SelectStep(id=node.tool_id, input=_input_id(node, node.tool_id), columns=columns)

    if node.tool_type == ToolType.FILTER:
        return FilterStep(
            id=node.tool_id,
            input=_input_id(node, node.tool_id),
            condition=node.filter_expression or "true",
        )

    if node.tool_type == ToolType.FORMULA:
        computed_columns = [ComputedColumn(name=f.field, expression=f.expression) for f in node.formulas]
        return WithColumnsStep(
            id=node.tool_id, input=_input_id(node, node.tool_id), columns=computed_columns
        )

    if node.tool_type == ToolType.JOIN:
        left_id = node.upstream_ids[0] if len(node.upstream_ids) > 0 else node.tool_id
        right_id = node.upstream_ids[1] if len(node.upstream_ids) > 1 else left_id
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
            id=node.tool_id,
            input=_input_id(node, node.tool_id),
            group_by=group_by,
            aggregations=aggregations,
        )

    if node.tool_type == ToolType.OUTPUT:
        table = (node.output_path or node.tool_id).rsplit(".", 1)[-1]
        return WriteStep(
            id=node.tool_id,
            input=_input_id(node, node.tool_id),
            target_table=_table_ref(target, table),
            mode="overwrite",
        )

    return None


def naive_spec_from_workflow(workflow: Workflow, target: TargetRef) -> PipelineSpec:
    """Directly map a Workflow's nodes onto PipelineSpec steps, no LLM involved."""
    steps: list[Step] = []
    functions_used: set[str] = set()

    for node in workflow.topological_order():
        step = _step_for_node(node, target)
        if step is None:
            continue
        if isinstance(step, JoinStep) and step.use_function:
            functions_used.add(step.use_function)
        steps.append(step)

    return PipelineSpec(
        name=workflow.name,
        language="pyspark",
        source=SourceRef(system="alteryx", object_name=workflow.name),
        target=target,
        steps=steps,
        functions_used=sorted(functions_used),
    )
