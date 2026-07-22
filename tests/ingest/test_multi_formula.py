"""Multi-Field and Multi-Row Formula: parse -> spec -> render."""

from pathlib import Path

from app.offline_convert import naive_spec_from_workflow
from convert.renderer import render_pyspark
from convert.spec import MultiFieldFormulaStep, MultiRowFormulaStep, TargetRef
from ingest.alteryx.parser import parse_yxmd

TARGET = TargetRef(catalog="main", schema="dev", layer="silver")


def _wf(tmp_path: Path, tool_xml: str, connect_from: str = "1") -> Path:
    xml = f"""<?xml version="1.0"?>
<AlteryxDocument yxmdVer="2023.1">
  <Nodes>
    <Node ToolID="1"><GuiSettings Plugin="AlteryxBasePluginsGui.DbFileInput.DbFileInput"/>
      <Properties><Configuration><File>main.dev.src</File></Configuration></Properties></Node>
    {tool_xml}
    <Node ToolID="9"><GuiSettings Plugin="AlteryxBasePluginsGui.DbFileOutput.DbFileOutput"/>
      <Properties><Configuration><File>main.dev.out</File></Configuration></Properties></Node>
  </Nodes>
  <Connections>
    <Connection><Origin ToolID="1" Connection="Output"/><Destination ToolID="2" Connection="Input"/></Connection>
    <Connection><Origin ToolID="{connect_from}" Connection="Output"/><Destination ToolID="9" Connection="Input"/></Connection>
  </Connections>
</AlteryxDocument>"""
    p = tmp_path / "wf.yxmd"
    p.write_text(xml, encoding="utf-8")
    return p


def test_multi_field_formula_parses_and_renders_per_field(tmp_path: Path) -> None:
    tool = """<Node ToolID="2"><GuiSettings Plugin="AlteryxBasePluginsGui.MultiFieldFormula.MultiFieldFormula"/>
      <Properties><Configuration>
        <Fields><Field name="A"/><Field name="B"/></Fields>
        <Expression>Trim([_CurrentField_])</Expression>
      </Configuration></Properties></Node>"""
    wf = parse_yxmd(_wf(tmp_path, tool, connect_from="2"))
    spec = naive_spec_from_workflow(wf, TARGET)
    step = next(s for s in spec.steps if isinstance(s, MultiFieldFormulaStep))
    assert step.fields == ["A", "B"]
    assert step.output_prefix is None  # replace in place
    source = render_pyspark(spec)
    assert 'withColumn("A", F.expr(\'Trim(`A`)\'))' in source
    assert 'withColumn("B", F.expr(\'Trim(`B`)\'))' in source


def test_multi_row_formula_renders_lag_window(tmp_path: Path) -> None:
    tool = """<Node ToolID="2"><GuiSettings Plugin="AlteryxBasePluginsGui.MultiRowFormula.MultiRowFormula"/>
      <Properties><Configuration>
        <UpdateField_Name>RunTotal</UpdateField_Name>
        <Expression>[Row-1:RunTotal] + [Amount]</Expression>
        <GroupByFields><Field name="Cust"/></GroupByFields>
        <NumRows value="1"/>
      </Configuration></Properties></Node>"""
    wf = parse_yxmd(_wf(tmp_path, tool, connect_from="2"))
    spec = naive_spec_from_workflow(wf, TARGET)
    step = next(s for s in spec.steps if isinstance(s, MultiRowFormulaStep))
    assert step.field == "RunTotal"
    assert step.group_by == ["Cust"]
    source = render_pyspark(spec)
    compile(source, "<g>", "exec")
    assert "LAG(`RunTotal`, 1) OVER (PARTITION BY `Cust` ORDER BY `__row_order`)" in source
    assert "monotonically_increasing_id()" in source
    assert "REVIEW: Alteryx used incoming record order" in source
    assert '.drop("__row_order")' in source


def test_multi_row_row_plus_becomes_lead(tmp_path: Path) -> None:
    tool = """<Node ToolID="2"><GuiSettings Plugin="AlteryxBasePluginsGui.MultiRowFormula.MultiRowFormula"/>
      <Properties><Configuration>
        <CreateField_Name>NextVal</CreateField_Name>
        <Expression>[Row+2:Amount]</Expression>
      </Configuration></Properties></Node>"""
    wf = parse_yxmd(_wf(tmp_path, tool, connect_from="2"))
    source = render_pyspark(naive_spec_from_workflow(wf, TARGET))
    assert "LEAD(`Amount`, 2) OVER (ORDER BY `__row_order`)" in source


def test_multi_formula_tools_no_longer_unsupported(tmp_path: Path) -> None:
    tool = """<Node ToolID="2"><GuiSettings Plugin="AlteryxBasePluginsGui.MultiFieldFormula.MultiFieldFormula"/>
      <Properties><Configuration>
        <Fields><Field name="A"/></Fields><Expression>[_CurrentField_]</Expression>
      </Configuration></Properties></Node>"""
    wf = parse_yxmd(_wf(tmp_path, tool, connect_from="2"))
    assert not wf.unsupported
    assert any(n.tool_type.value == "multi_field_formula" for n in wf.nodes)
