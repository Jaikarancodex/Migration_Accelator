from pathlib import Path

from ingest.alteryx.ir import ToolType
from ingest.alteryx.parser import parse_yxmd

FIXTURE = Path(__file__).parent.parent / "fixtures" / "alteryx" / "sales_summary.yxmd"
DEDUPE_FIXTURE = Path(__file__).parent.parent / "fixtures" / "alteryx" / "region_dedupe.yxmd"


def test_parses_all_supported_nodes() -> None:
    workflow = parse_yxmd(FIXTURE)
    assert workflow.name == "sales_summary"
    assert {n.tool_id for n in workflow.nodes} == {"1", "2", "3", "4", "5", "6", "7", "8"}


def test_logs_unsupported_tool_without_failing() -> None:
    workflow = parse_yxmd(FIXTURE)
    assert len(workflow.unsupported) == 1
    assert workflow.unsupported[0].tool_id == "9"
    assert "RegEx" in workflow.unsupported[0].plugin


def test_input_node_captures_table_name() -> None:
    workflow = parse_yxmd(FIXTURE)
    input_node = workflow.node_by_id("1")
    assert input_node is not None
    assert input_node.tool_type == ToolType.INPUT
    assert input_node.table_name == "legacy.sales.sales_raw"


def test_select_node_captures_fields_and_drops() -> None:
    workflow = parse_yxmd(FIXTURE)
    select_node = workflow.node_by_id("2")
    assert select_node is not None
    fields_by_name = {f.field: f for f in select_node.fields}
    assert fields_by_name["Notes"].selected is False
    assert fields_by_name["Amount"].selected is True


def test_filter_node_captures_expression() -> None:
    workflow = parse_yxmd(FIXTURE)
    filter_node = workflow.node_by_id("3")
    assert filter_node is not None
    assert filter_node.filter_expression == "[Amount] > 0"


def test_formula_node_captures_expression() -> None:
    workflow = parse_yxmd(FIXTURE)
    formula_node = workflow.node_by_id("4")
    assert formula_node is not None
    assert formula_node.formulas[0].field == "LineTotal"
    assert formula_node.formulas[0].expression == "[Amount] * [Quantity]"


def test_join_node_captures_both_sides() -> None:
    workflow = parse_yxmd(FIXTURE)
    join_node = workflow.node_by_id("6")
    assert join_node is not None
    sides = {j.side: j.keys for j in join_node.join_inputs}
    assert sides["left"] == ["CustomerID"]
    assert sides["right"] == ["CustomerID"]


def test_summarize_node_captures_actions() -> None:
    workflow = parse_yxmd(FIXTURE)
    summarize_node = workflow.node_by_id("7")
    assert summarize_node is not None
    actions = {(a.field, a.action) for a in summarize_node.summarize_actions}
    assert ("LineTotal", "Sum") in actions
    assert ("CustomerID", "GroupBy") in actions


def test_output_node_captures_target_table() -> None:
    workflow = parse_yxmd(FIXTURE)
    output_node = workflow.node_by_id("8")
    assert output_node is not None
    assert output_node.output_path == "legacy.sales.sales_summary"


def test_upstream_ids_wired_from_connections() -> None:
    workflow = parse_yxmd(FIXTURE)
    join_node = workflow.node_by_id("6")
    assert join_node is not None
    assert set(join_node.upstream_ids) == {"4", "5"}


def test_union_node_wires_both_upstream_inputs() -> None:
    workflow = parse_yxmd(DEDUPE_FIXTURE)
    union_node = workflow.node_by_id("3")
    assert union_node is not None
    assert union_node.tool_type == ToolType.UNION
    assert set(union_node.upstream_ids) == {"1", "2"}


def test_unique_node_captures_key_fields() -> None:
    workflow = parse_yxmd(DEDUPE_FIXTURE)
    unique_node = workflow.node_by_id("4")
    assert unique_node is not None
    assert unique_node.tool_type == ToolType.UNIQUE
    assert unique_node.unique_fields == ["OrderID"]


def test_sort_node_captures_fields_and_direction() -> None:
    workflow = parse_yxmd(DEDUPE_FIXTURE)
    sort_node = workflow.node_by_id("5")
    assert sort_node is not None
    assert sort_node.tool_type == ToolType.SORT
    assert [(f.field, f.descending) for f in sort_node.sort_fields] == [
        ("Region", False),
        ("Amount", True),
    ]


def test_dedupe_fixture_has_no_unsupported_tools() -> None:
    workflow = parse_yxmd(DEDUPE_FIXTURE)
    assert workflow.unsupported == []


NOKIA_LS = Path(
    r"C:\Users\JaikaranN\Desktop\Alteryx_sample(Nokia)\Alteryx Jobs_Shared(Nokia)"
    r"\LS_Certifications_DEV_Load\LS_Certifications_DEV_Load.yxmd"
)


def test_cleanse_macro_parses_when_real_workflow_available() -> None:
    if not NOKIA_LS.exists():
        import pytest

        pytest.skip("Nokia sample workflow not present on this machine")
    workflow = parse_yxmd(NOKIA_LS)
    cleanse_node = workflow.node_by_id("3")
    assert cleanse_node is not None
    assert cleanse_node.tool_type == ToolType.CLEANSE
    assert cleanse_node.cleanse is not None
    assert cleanse_node.cleanse.trim is True
    assert cleanse_node.cleanse.nulls_to_blank is True
    assert cleanse_node.cleanse.case is None  # dropdown set but enable-checkbox absent
    assert cleanse_node.cleanse.columns is not None
    assert "Candidate_Registry_ID" in cleanse_node.cleanse.columns


def test_cleanse_pro_parses_from_inline_xml(tmp_path: Path) -> None:
    xml = """<?xml version="1.0"?>
<AlteryxDocument yxmdVer="2023.1">
  <Nodes>
    <Node ToolID="1">
      <GuiSettings Plugin="AlteryxBasePluginsGui.DataCleansePro.DataCleansePro">
        <Position x="1" y="1" />
      </GuiSettings>
      <Properties>
        <Configuration>
          <RemoveLeadingAndTrailingWhitespace value="True" />
          <Checkbox_ReplaceStringColumns value="True" />
          <radioButton_ReplaceNullwithBlanks value="True" />
          <Fields>
            <Field value="name" selected="True" />
            <Field value="age" selected="True" />
          </Fields>
          <CheckBox_ModifyCase value="True" />
          <ModifyCase>upper</ModifyCase>
        </Configuration>
      </Properties>
    </Node>
  </Nodes>
  <Connections />
</AlteryxDocument>"""
    path = tmp_path / "cleanse.yxmd"
    path.write_text(xml, encoding="utf-8")
    workflow = parse_yxmd(path)
    node = workflow.node_by_id("1")
    assert node is not None and node.cleanse is not None
    assert node.cleanse.trim is True
    assert node.cleanse.nulls_to_blank is True
    assert node.cleanse.columns == ["name", "age"]
    assert node.cleanse.case == "upper"


def test_topological_order_respects_dependencies() -> None:
    workflow = parse_yxmd(FIXTURE)
    order = [n.tool_id for n in workflow.topological_order()]
    assert order.index("1") < order.index("2")
    assert order.index("4") < order.index("6")
    assert order.index("5") < order.index("6")
    assert order.index("7") < order.index("8")
