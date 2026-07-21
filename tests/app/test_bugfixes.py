"""Regression tests for three bugs found reviewing FNPRMT-PROD-NEBA-BOM-INFO_GCP's
generated output: an AppendFields self cross-join, duplicate `@dp.table` names in
SDP output (bronze reads and gold writes), and SQL-as-source producing a
plausible-but-wrong table name.
"""

from pathlib import Path

from app.offline_convert import naive_spec_from_workflow
from convert.renderer import render_sdp
from convert.spec import AppendFieldsStep, PipelineSpec, ReadStep, SourceRef, TargetRef, WriteStep
from ingest.alteryx.parser import parse_yxmd

TARGET = TargetRef(catalog="main", schema="migration_dev", layer="bronze")
SOURCE = SourceRef(system="alteryx", object_name="x")


def _append_fields_workflow(tmp_path: Path, source_first: bool) -> Path:
    """Mirrors the real bug: Alteryx labels are "Source" (base) and "Targets"
    (plural, appended values) — connection order in the XML varies, which is
    what broke the old "Target"/"Targe" guess-based lookup.
    """
    conn_a = '<Connection><Origin ToolID="1" Connection="Output" /><Destination ToolID="3" Connection="Source" /></Connection>'
    conn_b = '<Connection><Origin ToolID="2" Connection="Output" /><Destination ToolID="3" Connection="Targets" /></Connection>'
    conns = conn_a + conn_b if source_first else conn_b + conn_a
    xml = f"""<?xml version="1.0"?>
<AlteryxDocument yxmdVer="2023.1">
  <Nodes>
    <Node ToolID="1"><GuiSettings Plugin="AlteryxBasePluginsGui.DbFileInput.DbFileInput"/>
      <Properties><Configuration><File>main.x.base</File></Configuration></Properties></Node>
    <Node ToolID="2"><GuiSettings Plugin="AlteryxBasePluginsGui.DbFileInput.DbFileInput"/>
      <Properties><Configuration><File>main.x.values</File></Configuration></Properties></Node>
    <Node ToolID="3"><GuiSettings Plugin="AlteryxBasePluginsGui.AppendFields.AppendFields"/>
      <Properties><Configuration><CartesianMode>Allow</CartesianMode></Configuration></Properties></Node>
    <Node ToolID="4"><GuiSettings Plugin="AlteryxBasePluginsGui.DbFileOutput.DbFileOutput"/>
      <Properties><Configuration><File>main.x.out</File></Configuration></Properties></Node>
  </Nodes>
  <Connections>
    {conns}
    <Connection><Origin ToolID="3" Connection="Output" /><Destination ToolID="4" Connection="Input" /></Connection>
  </Connections>
</AlteryxDocument>"""
    path = tmp_path / "append.yxmd"
    path.write_text(xml, encoding="utf-8")
    return path


def test_append_fields_resolves_distinct_sides_regardless_of_connection_order(tmp_path: Path) -> None:
    for source_first in (True, False):
        wf = parse_yxmd(_append_fields_workflow(tmp_path, source_first))
        spec = naive_spec_from_workflow(wf, TARGET)
        append = next(s for s in spec.steps if isinstance(s, AppendFieldsStep))
        # target = the "Source"-labeled (base) stream, source = "Targets"-labeled
        assert append.target == "1"
        assert append.source == "2"
        assert append.target != append.source, "self cross-join bug regressed"


def test_append_fields_positional_fallback_never_collides() -> None:
    from ingest.alteryx.ir import Node, ToolType, Workflow

    nodes = [
        Node(tool_id="1", tool_type=ToolType.INPUT, raw_plugin="DbFileInput", table_name="main.x.a"),
        Node(tool_id="2", tool_type=ToolType.INPUT, raw_plugin="DbFileInput", table_name="main.x.b"),
        Node(
            tool_id="3", tool_type=ToolType.APPEND_FIELDS, raw_plugin="AppendFields",
            upstream_ids=["1", "2"], upstream_labels={},  # no labels at all
        ),
        Node(tool_id="4", tool_type=ToolType.OUTPUT, raw_plugin="DbFileOutput",
             upstream_ids=["3"], output_path="main.x.out"),
    ]
    wf = Workflow(source_file="x.yxmd", name="wf", nodes=nodes)
    spec = naive_spec_from_workflow(wf, TARGET)
    append = next(s for s in spec.steps if isinstance(s, AppendFieldsStep))
    assert append.target != append.source


def test_sdp_dedupes_bronze_reads_of_the_same_source_table() -> None:
    spec = PipelineSpec(
        name="dup_reads", language="pyspark", source=SOURCE, target=TARGET,
        steps=[
            ReadStep(id="a", source_table="main.x.shared", alias="shared"),
            ReadStep(id="b", source_table="main.x.shared", alias="shared"),
            WriteStep(id="w", input="a", target_table="main.x.out", mode="overwrite"),
        ],
    )
    source = render_sdp(spec)
    compile(source, "<generated>", "exec")
    assert source.count('@dp.table(name="bronze_shared"') == 1


def test_sdp_disambiguates_colliding_gold_write_names() -> None:
    spec = PipelineSpec(
        name="dup_writes", language="pyspark", source=SOURCE, target=TARGET,
        steps=[
            ReadStep(id="a", source_table="main.x.a", alias="a"),
            ReadStep(id="b", source_table="main.x.b", alias="b"),
            WriteStep(id="w1", input="a", target_table="main.x.sheet1", mode="overwrite"),
            WriteStep(id="w2", input="b", target_table="main.other.sheet1", mode="overwrite"),
        ],
    )
    source = render_sdp(spec)
    compile(source, "<generated>", "exec")
    assert '@dp.table(name="gold_sheet1"' in source
    assert '@dp.table(name="gold_sheet1_2"' in source
    assert "REVIEW: name collided" in source
    # both outputs are present, not silently dropped
    assert source.count("def gold_sheet1():") == 1
    assert source.count("def gold_sheet1_2():") == 1


def test_sql_query_source_becomes_flagged_placeholder_not_mangled_name(tmp_path: Path) -> None:
    xml = """<?xml version="1.0"?>
<AlteryxDocument yxmdVer="2023.1">
  <Nodes>
    <Node ToolID="1"><GuiSettings Plugin="AlteryxBasePluginsGui.DbFileInput.DbFileInput"/>
      <Properties><Configuration>
        <File>aka:6401126c3444ea25acc7deaf|||select * from ##TempFinalResult</File>
      </Configuration></Properties></Node>
    <Node ToolID="2"><GuiSettings Plugin="AlteryxBasePluginsGui.DbFileOutput.DbFileOutput"/>
      <Properties><Configuration><File>main.x.out</File></Configuration></Properties></Node>
  </Nodes>
  <Connections>
    <Connection><Origin ToolID="1" Connection="Output" /><Destination ToolID="2" Connection="Input" /></Connection>
  </Connections>
</AlteryxDocument>"""
    path = tmp_path / "sqlsrc.yxmd"
    path.write_text(xml, encoding="utf-8")
    wf = parse_yxmd(path)
    spec = naive_spec_from_workflow(wf, TARGET)
    read = next(s for s in spec.steps if s.op == "read")
    assert read.source_table == "main.migration_dev.todo_source_1"
    assert "select" not in read.source_table.lower()
    assert "tempfinalresult" not in read.source_table.lower()


def test_output_dotted_db_ref_keeps_actual_table_name() -> None:
    """'aka:<conn>|||IONPMVIEW.dbo.project_cube_le' wrote to the mangled
    'ionpmview_dbo_project_cube_le' in a real migrated workflow (W3). The
    actual table name is the last segment.
    """
    from app.offline_convert import _sanitize_table_name

    assert _sanitize_table_name("aka:65cf6443|||IONPMVIEW.dbo.project_cube_le") == "project_cube_le"
    assert _sanitize_table_name("[IONPMVIEW].[dbo].[Project_Cube_LE]") == "project_cube_le"
    assert _sanitize_table_name("dbo.StockList") == "stocklist"
    # file paths keep basename-minus-extension behavior
    assert _sanitize_table_name(r"C:\data\sales export.csv") == "sales_export"
    assert _sanitize_table_name("report.final.xlsx") == "report_final"


def test_simple_sql_source_lands_under_its_actual_table_name(tmp_path: Path) -> None:
    xml = """<?xml version="1.0"?>
<AlteryxDocument yxmdVer="2023.1">
  <Nodes>
    <Node ToolID="1"><GuiSettings Plugin="AlteryxBasePluginsGui.DbFileInput.DbFileInput"/>
      <Properties><Configuration>
        <File>aka:6401126c|||SELECT a, b FROM Stocklist.dbo.StockList WHERE VersionOrder = 1</File>
      </Configuration></Properties></Node>
    <Node ToolID="2"><GuiSettings Plugin="AlteryxBasePluginsGui.DbFileOutput.DbFileOutput"/>
      <Properties><Configuration><File>main.x.out</File></Configuration></Properties></Node>
  </Nodes>
  <Connections>
    <Connection><Origin ToolID="1" Connection="Output" /><Destination ToolID="2" Connection="Input" /></Connection>
  </Connections>
</AlteryxDocument>"""
    path = tmp_path / "simplesql.yxmd"
    path.write_text(xml, encoding="utf-8")
    wf = parse_yxmd(path)
    spec = naive_spec_from_workflow(wf, TARGET)
    read = next(s for s in spec.steps if s.op == "read")
    # single-table SELECT: the actual table, not an anonymous todo_source_1
    assert read.source_table == "main.migration_dev.stocklist"


def test_complex_sql_source_still_lands_as_todo_placeholder(tmp_path: Path) -> None:
    xml = """<?xml version="1.0"?>
<AlteryxDocument yxmdVer="2023.1">
  <Nodes>
    <Node ToolID="1"><GuiSettings Plugin="AlteryxBasePluginsGui.DbFileInput.DbFileInput"/>
      <Properties><Configuration>
        <File>aka:6401126c|||SELECT * FROM a.b JOIN c.d ON a.b.x = c.d.x</File>
      </Configuration></Properties></Node>
    <Node ToolID="2"><GuiSettings Plugin="AlteryxBasePluginsGui.DbFileOutput.DbFileOutput"/>
      <Properties><Configuration><File>main.x.out</File></Configuration></Properties></Node>
  </Nodes>
  <Connections>
    <Connection><Origin ToolID="1" Connection="Output" /><Destination ToolID="2" Connection="Input" /></Connection>
  </Connections>
</AlteryxDocument>"""
    path = tmp_path / "joinsql.yxmd"
    path.write_text(xml, encoding="utf-8")
    wf = parse_yxmd(path)
    spec = naive_spec_from_workflow(wf, TARGET)
    read = next(s for s in spec.steps if s.op == "read")
    assert read.source_table == "main.migration_dev.todo_source_1"
