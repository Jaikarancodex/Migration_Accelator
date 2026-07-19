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
import re
from collections.abc import Iterator
from pathlib import Path
from xml.etree.ElementTree import Element  # noqa: S405 - parsing done via defusedxml below

import structlog
from defusedxml.ElementTree import parse as safe_parse

from ingest.alteryx.ir import (
    CleanseConfig,
    FieldSelection,
    FormulaExpression,
    JoinInput,
    Node,
    SortField,
    SummarizeAction,
    ToolType,
    UnsupportedTool,
    Workflow,
)

logger = structlog.get_logger(__name__)

# Maps the suffix of the Alteryx GuiSettings Plugin attribute to our ToolType.
# GUI-only elements: comments, previews, and container boxes. They carry no
# data semantics, so they are skipped silently instead of flagged unsupported.
_IGNORED_PLUGIN_SUFFIXES: tuple[str, ...] = (
    "TextBox.TextBox",
    "ToolContainer.ToolContainer",
    "Browse.Browse",
    "BrowseV2.BrowseV2",
)

_PLUGIN_TOOL_TYPES: dict[str, ToolType] = {
    "DbFileInput.DbFileInput": ToolType.INPUT,
    "TextInput.TextInput": ToolType.INPUT,
    "AlteryxSelect.AlteryxSelect": ToolType.SELECT,
    "Filter.Filter": ToolType.FILTER,
    "Formula.Formula": ToolType.FORMULA,
    "Join.Join": ToolType.JOIN,
    "Union.Union": ToolType.UNION,
    "Sort.Sort": ToolType.SORT,
    "Unique.Unique": ToolType.UNIQUE,
    "RecordID.RecordID": ToolType.RECORD_ID,
    "DataCleansePro.DataCleansePro": ToolType.CLEANSE,
    "MacroInput.MacroInput": ToolType.MACRO_INPUT,
    "MacroOutput.MacroOutput": ToolType.MACRO_OUTPUT,
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


def _parse_sort(config: Element) -> list[SortField]:
    fields: list[SortField] = []
    sort_info = config.find("SortInfo")
    if sort_info is None:
        return fields
    for f in sort_info.findall("Field"):
        field_name = f.get("field")
        if not field_name:
            continue
        fields.append(
            SortField(field=field_name, descending=f.get("order", "Ascending") == "Descending")
        )
    return fields


def _parse_unique(config: Element) -> list[str]:
    unique_fields = config.find("UniqueFields")
    if unique_fields is None:
        return []
    return [f.get("field", "") for f in unique_fields.findall("Field") if f.get("field")]


_VALID_CASES = ("upper", "lower", "title")


def _parse_cleanse_macro(config: Element) -> CleanseConfig:
    """Config of the classic Cleanse.yxmc macro (anonymous Value names)."""
    values = {v.get("name", ""): (v.text or "").strip() for v in config.findall("Value")}

    def flag(key: str) -> bool:
        return values.get(key) == "True"

    names = re.findall(r'"([^"]+)"', values.get("List Box (11)", ""))
    columns = None if not names or "*Unknown" in names else names
    case = values.get("Drop Down (81)", "none").lower()
    return CleanseConfig(
        columns=columns,
        trim=flag("Check Box (84)"),
        collapse_whitespace=flag("Check Box (117)"),
        remove_all_whitespace=flag("Check Box (122)"),
        nulls_to_blank=flag("Check Box (135)"),
        numeric_nulls_to_zero=flag("Check Box (136)"),
        case=case if flag("Check Box (132)") and case in _VALID_CASES else None,
    )


def _parse_cleanse_pro(config: Element) -> CleanseConfig:
    """Config of the modern Data Cleanse Pro tool (named elements)."""

    def flag(tag: str) -> bool:
        elem = config.find(tag)
        return elem is not None and elem.get("value") == "True"

    columns: list[str] | None = None
    fields_elem = config.find("Fields")
    if fields_elem is not None:
        names = [
            f.get("value", "")
            for f in fields_elem.findall("Field")
            if f.get("selected") == "True"
        ]
        columns = None if "*Unknown" in names or not names else names
    case = _text(config.find("ModifyCase")) or "none"
    return CleanseConfig(
        columns=columns,
        trim=flag("RemoveLeadingAndTrailingWhitespace"),
        collapse_whitespace=flag("RemoveTabsLineBreaksAndDuplicates"),
        remove_all_whitespace=flag("RemoveAllWhitespaces"),
        nulls_to_blank=flag("Checkbox_ReplaceStringColumns") and flag("radioButton_ReplaceNullwithBlanks"),
        numeric_nulls_to_zero=flag("Checkbox_ReplaceNumericColumns") and flag("radioButton_ReplaceNullwithZero"),
        case=case.lower() if flag("CheckBox_ModifyCase") and case.lower() in _VALID_CASES else None,
    )


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

    # Macro nodes reference a .yxmc instead of a plugin. The Data Cleansing
    # macro is understood and converted into a generated utility; other
    # macros stay unsupported until their inner workflow is converted.
    engine = elem.find("EngineSettings")
    macro = engine.get("Macro") if engine is not None else None
    if macro and tool_type == ToolType.UNSUPPORTED:
        if "cleanse" in macro.lower() and config is not None:
            tool_type = ToolType.CLEANSE
            plugin = f"Macro:{macro}"
        else:
            # A custom macro: kept as a typed node so the converter can
            # inline it when its .yxmc has been uploaded to the macro
            # registry; without a registered macro it is bridged over.
            tool_type = ToolType.MACRO
            plugin = f"Macro:{macro}"

    if tool_type == ToolType.UNSUPPORTED or config is None:
        return UnsupportedTool(
            tool_id=tool_id,
            plugin=plugin,
            reason="No <Configuration> found" if config is None else "Unrecognized plugin type",
            upstream_ids=connections.get(tool_id, []),
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
    elif tool_type == ToolType.SORT:
        node.sort_fields = _parse_sort(config)
    elif tool_type == ToolType.UNIQUE:
        node.unique_fields = _parse_unique(config)
    elif tool_type == ToolType.RECORD_ID:
        node.record_id_field = _text(config.find("FieldName")) or "RecordID"
    elif tool_type == ToolType.CLEANSE:
        node.cleanse = (
            _parse_cleanse_macro(config) if plugin.startswith("Macro:") else _parse_cleanse_pro(config)
        )
    elif tool_type == ToolType.MACRO:
        raw = plugin.removeprefix("Macro:")
        node.macro_name = Path(raw.replace("\\", "/")).stem.lower()
    elif tool_type == ToolType.SUMMARIZE:
        node.summarize_actions = _parse_summarize(config)
    elif tool_type == ToolType.OUTPUT:
        node.output_path = _text(config.find("File"))

    return node


def _iter_node_elements(parent: Element) -> Iterator[Element]:
    """Yield tool Node elements, descending into ToolContainer ChildNodes.

    Real workflows organize tools inside (possibly nested) container boxes;
    the container node itself is GUI-only, but its children are real tools.
    """
    for node_elem in parent.findall("Node"):
        plugin = _plugin_name(node_elem.find("GuiSettings"))
        if plugin.endswith("ToolContainer.ToolContainer"):
            child_nodes = node_elem.find("ChildNodes")
            if child_nodes is not None:
                yield from _iter_node_elements(child_nodes)
            continue
        yield node_elem


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

    for node_elem in _iter_node_elements(nodes_elem):
        plugin = _plugin_name(node_elem.find("GuiSettings"))
        # AlteryxGuiToolkit.* covers interface widgets (checkboxes, list
        # boxes, text boxes...) that parametrize macros but carry no data flow.
        if plugin.startswith("AlteryxGuiToolkit.") or any(
            plugin.endswith(suffix) for suffix in _IGNORED_PLUGIN_SUFFIXES
        ):
            logger.debug("ignored_gui_tool", file=str(path), plugin=plugin)
            continue
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
