"""Builds the Alteryx -> PySpark conversion prompt from a parsed Workflow.

Keeps prompt construction out of `llm/client.py` so the client stays a pure
transport layer and the prompt template can be iterated on independently.
"""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from convert.spec import TargetRef
from functions.registry import render_signatures_for_prompt
from ingest.alteryx.ir import Node, ToolType, Workflow
from knowledge.alteryx_tools import render_knowledge_for_prompt

_TEMPLATE_DIR = Path(__file__).parent / "prompts"
_ENV = Environment(
    loader=FileSystemLoader(str(_TEMPLATE_DIR)),
    autoescape=select_autoescape(enabled_extensions=()),
    trim_blocks=True,
    lstrip_blocks=True,
)

SYSTEM_PROMPT = (
    "You are a precise, conservative code-migration assistant for an internal "
    "Alteryx-to-Databricks accelerator. You output only valid YAML matching the "
    "requested schema, with no commentary, no markdown fences, and no invented "
    "fields. When a transformation is ambiguous, prefer the literal, most "
    "conservative reading of the source tool's semantics."
)


def _describe_node(node: Node) -> str:
    base = f"- [{node.tool_id}] {node.tool_type.value} (upstream: {node.upstream_ids or 'none'})"
    if node.tool_type == ToolType.INPUT:
        return f"{base} source_table={node.table_name}"
    if node.tool_type == ToolType.SORT:
        fields = ", ".join(f"{f.field} {'desc' if f.descending else 'asc'}" for f in node.sort_fields)
        return f"{base} order_by=[{fields}]"
    if node.tool_type == ToolType.UNIQUE:
        return f"{base} unique_keys={node.unique_fields}"
    if node.tool_type == ToolType.UNION:
        return f"{base} stacks all upstream inputs by column name"
    if node.tool_type == ToolType.RECORD_ID:
        return f"{base} adds sequential id column {node.record_id_field!r}"
    if node.tool_type == ToolType.SELECT:
        fields = ", ".join(
            f"{f.field}{'->' + f.rename if f.rename else ''}{' [DROP]' if not f.selected else ''}"
            for f in node.fields
        )
        return f"{base} fields=[{fields}]"
    if node.tool_type == ToolType.FILTER:
        return f"{base} keep_where={node.filter_expression!r}"
    if node.tool_type == ToolType.FORMULA:
        formulas = ", ".join(f"{f.field}={f.expression!r}" for f in node.formulas)
        return f"{base} formulas=[{formulas}]"
    if node.tool_type == ToolType.JOIN:
        joins = "; ".join(f"{j.side}_keys={j.keys}" for j in node.join_inputs)
        return f"{base} {joins}"
    if node.tool_type == ToolType.SUMMARIZE:
        actions = ", ".join(
            f"{a.action}({a.field}){' as ' + a.rename if a.rename else ''}"
            for a in node.summarize_actions
        )
        return f"{base} actions=[{actions}]"
    if node.tool_type == ToolType.OUTPUT:
        return f"{base} target_table={node.output_path}"
    return base


def build_alteryx_to_pyspark_prompt(workflow: Workflow, target: TargetRef) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for converting `workflow`."""
    nodes_description = "\n".join(_describe_node(n) for n in workflow.topological_order())
    unsupported = "\n".join(
        f"- [{u.tool_id}] {u.plugin}: {u.reason}" for u in workflow.unsupported
    )
    tool_knowledge = render_knowledge_for_prompt(
        {n.tool_type for n in workflow.nodes},
        [u.plugin for u in workflow.unsupported],
    )

    template = _ENV.get_template("alteryx_to_pyspark.j2")
    user_prompt = template.render(
        function_signatures=render_signatures_for_prompt(),
        target_catalog=target.catalog,
        target_schema=target.schema_,
        target_layer=target.layer,
        workflow_name=workflow.name,
        nodes_description=nodes_description,
        unsupported_tools=unsupported,
        tool_knowledge=tool_knowledge,
    )
    return SYSTEM_PROMPT, user_prompt
