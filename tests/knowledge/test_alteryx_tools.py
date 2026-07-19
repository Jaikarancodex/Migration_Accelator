from ingest.alteryx.ir import ToolType
from ingest.alteryx.parser import _PLUGIN_TOOL_TYPES
from knowledge.alteryx_tools import (
    CATALOG,
    lookup_by_plugin,
    mappings_for_tool_types,
    render_knowledge_for_prompt,
)


def test_every_parser_supported_plugin_has_a_catalog_entry() -> None:
    for suffix in _PLUGIN_TOOL_TYPES:
        mapping = lookup_by_plugin(f"AlteryxBasePluginsGui.{suffix}")
        assert mapping is not None, f"no knowledge entry for parser-supported tool {suffix}"
        assert mapping.parser_supported, f"{suffix} is parser-supported but not flagged as such"


def test_catalog_entries_have_substantive_guidance() -> None:
    for m in CATALOG:
        assert len(m.databricks_logic) > 20, f"{m.tool} guidance too thin"
        assert m.category


def test_lookup_matches_full_plugin_string() -> None:
    m = lookup_by_plugin("AlteryxSpatialPluginsGui.RegEx.RegEx")
    assert m is not None
    assert m.tool == "RegEx"
    assert lookup_by_plugin("Something.Unknown.Tool") is None


def test_mappings_for_tool_types_filters_to_present_tools() -> None:
    mappings = mappings_for_tool_types({ToolType.SORT, ToolType.UNION})
    tools = {m.tool for m in mappings}
    assert tools == {"Sort", "Union"}


def test_render_knowledge_includes_supported_and_unsupported_sections() -> None:
    text = render_knowledge_for_prompt(
        {ToolType.FILTER}, ["AlteryxSpatialPluginsGui.RegEx.RegEx"]
    )
    assert "Filter" in text
    assert "RegEx" in text
    assert "UNSUPPORTED" in text
    assert "regexp_extract" in text
