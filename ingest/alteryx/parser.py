"""Parses Alteryx .yxmd workflow XML into the clean IR defined in ir.py.

Assumptions about the .yxmd schema (documented in the top-level README,
since no sample workflow was supplied for this session): tool configuration
follows the shape used by AlteryxBasePluginsGui tools (DbFileInput/Output,
AlteryxSelect, Filter, Formula, Join, Summarize). Real-world workflows vary
across Alteryx versions; unrecognized plugins or configuration shapes are
logged as unsupported rather than raising, so the rest of the workflow still
parses.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from xml.etree.ElementTree import Element  # noqa: S405 - parsing done via defusedxml below

import structlog
from defusedxml.ElementTree import parse as safe_parse

from ingest.alteryx.ir import (
    FieldSelection,
    FormulaExpression,
    JoinInput,
    Node,
    SummarizeAction,
    ToolType,
    UnsupportedTool,
    Workflow,
)

logger = structlog.get_logger(__name__)

# Maps the suffix of the Alteryx GuiSettings Plugin attribute to our ToolType.
_PLUGIN_TOOL_TYPES: dict[str, ToolType] = {
    "DbFileInput.DbFileInput": ToolType.INPUT,
    "TextInput.TextInput": ToolType.INPUT,
    "AlteryxSelect.AlteryxSelect": ToolType.SELECT,
    "Filter.Filter": ToolType.FILTER,
    "Formula.Formula": ToolType.FORMULA,
    "Join.Join": ToolType.JOIN,
    "Summarize.Summarize": ToolType.SUMMARIZE,
    "DbFileOutput.DbFileOutput": ToolType.OUTPUT,
}


def _plugin_name(gui_settings: Element | None) -> str:
    if gui_settings is None:
        return "Unknown"
    return gui_settings.get("Plugin", "Unknown")


def _tool_type_for_plugin(plugin: str) -> ToolType:
    for suffix, tool_type in _PLUGIN_TOOL_TYPES.items():
        if plugin.endswith(suffix):
            return tool_type
    return ToolType.UNSUPPORTED


def _parse_position(gui_settings: Element | None) -> dict[str, float]:
    if gui_settings is None:
        return {}
    pos = gui_settings.find("Position")
    if pos is None:
        return {}
    result = {}
    for attr in ("x", "y"):
        val = pos.get(attr)
        if val is not None:
            with contextlib.suppress(ValueError):
                result[attr] = float(val)
    return result


def _text(elem: Element | None) -> str | None:
    if elem is None or elem.text is None:
        return None
    return elem.text.strip() or None


def _parse_select(config: Element) -> list[FieldSelection]:
    fields: list[FieldSelection] = []
    select_fields = config.find("SelectFields")
    if select_fields is None:
        return fields
    for f in select_fields.findall("SelectField"):
        field_name = f.get("field")
        if not field_name:
            continue
        fields.append(
            FieldSelection(
                field=field_name,
                selected=f.get("selected", "True") != "False",
                rename=f.get("rename") or None,
                new_type=f.get("type") or None,
            )
        )
    return fields


def _parse_formula(config: Element) -> list[FormulaExpression]:
    formulas: list[FormulaExpression] = []
    formula_fields = config.find("FormulaFields")
    if formula_fields is None:
        return formulas
    for f in formula_fields.findall("FormulaField"):
        field_name = f.get("field")
        expression = f.get("expression")
        if not field_name or expression is None:
            continue
        formulas.append(
            FormulaExpression(field=field_name, expression=expression, data_type=f.get("type"))
        )
    return formulas


def _parse_join(config: Element) -> list[JoinInput]:
    joins: list[JoinInput] = []
    for join_info in config.findall("JoinInfo"):
        side = join_info.get("connection", "").lower()
        if side not in ("left", "right"):
            continue
        keys = [f.get("field", "") for f in join_info.findall("Field") if f.get("field")]
        if keys:
            joins.append(JoinInput(side=side, keys=keys))
    return joins


def _parse_summarize(config: Element) -> list[SummarizeAction]:
    actions: list[SummarizeAction] = []
    summarize_fields = config.find("SummarizeFields")
    if summarize_fields is None:
        return actions
    for f in summarize_fields.findall("SummarizeField"):
        field_name = f.get("field")
        action = f.get("action")
        if not field_name or not action:
            continue
        actions.append(SummarizeAction(field=field_name, action=action, rename=f.get("rename")))
    return actions


def _parse_node(elem: Element, connections: dict[str, list[str]]) -> Node | UnsupportedTool:
    tool_id = elem.get("ToolID", "")
    gui_settings = elem.find("GuiSettings")
    plugin = _plugin_name(gui_settings)
    tool_type = _tool_type_for_plugin(plugin)

    properties = elem.find("Properties")
    config = properties.find("Configuration") if properties is not None else None
    annotation_elem = properties.find("Annotation/AnnotationText") if properties is not None else None

    if tool_type == ToolType.UNSUPPORTED or config is None:
        return UnsupportedTool(
            tool_id=tool_id,
            plugin=plugin,
            reason="Unrecognized plugin type" if config is None else "No <Configuration> found",
        )

    node = Node(
        tool_id=tool_id,
        tool_type=tool_type,
        raw_plugin=plugin,
        upstream_ids=connections.get(tool_id, []),
        annotation=_text(annotation_elem),
        position=_parse_position(gui_settings),
    )

    if tool_type == ToolType.INPUT:
        node.table_name = _text(config.find("File"))
    elif tool_type == ToolType.SELECT:
        node.fields = _parse_select(config)
    elif tool_type == ToolType.FILTER:
        node.filter_expression = _text(config.find("Expression"))
    elif tool_type == ToolType.FORMULA:
        node.formulas = _parse_formula(config)
    elif tool_type == ToolType.JOIN:
        node.join_inputs = _parse_join(config)
    elif tool_type == ToolType.SUMMARIZE:
        node.summarize_actions = _parse_summarize(config)
    elif tool_type == ToolType.OUTPUT:
        node.output_path = _text(config.find("File"))

    return node


def _parse_connections(root: Element) -> dict[str, list[str]]:
    """Map destination ToolID -> list of origin ToolIDs (upstream nodes)."""
    upstream: dict[str, list[str]] = {}
    connections_elem = root.find("Connections")
    if connections_elem is None:
        return upstream
    for conn in connections_elem.findall("Connection"):
        origin = conn.find("Origin")
        destination = conn.find("Destination")
        if origin is None or destination is None:
            continue
        origin_id = origin.get("ToolID")
        dest_id = destination.get("ToolID")
        if not origin_id or not dest_id:
            continue
        upstream.setdefault(dest_id, []).append(origin_id)
    return upstream


def parse_yxmd(path: str | Path) -> Workflow:
    """Parse a single .yxmd file into a Workflow IR.

    Unsupported/unrecognized tools are collected in `Workflow.unsupported`
    rather than raising, so a workflow with a few unhandled tools still
    yields a usable partial IR for the tools we do understand.
    """
    path = Path(path)
    tree = safe_parse(str(path))
    root = tree.getroot()

    connections = _parse_connections(root)

    nodes: list[Node] = []
    unsupported: list[UnsupportedTool] = []

    nodes_elem = root.find("Nodes")
    if nodes_elem is None:
        logger.warning("no_nodes_element", file=str(path))
        nodes_elem = root

    for node_elem in nodes_elem.findall("Node"):
        parsed = _parse_node(node_elem, connections)
        if isinstance(parsed, UnsupportedTool):
            logger.warning(
                "unsupported_tool", file=str(path), tool_id=parsed.tool_id, plugin=parsed.plugin
            )
            unsupported.append(parsed)
        else:
            nodes.append(parsed)

    return Workflow(
        source_file=str(path),
        name=path.stem,
        nodes=nodes,
        unsupported=unsupported,
    )
