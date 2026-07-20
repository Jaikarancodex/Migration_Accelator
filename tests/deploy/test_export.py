from pathlib import Path

import yaml

from app.offline_convert import naive_spec_from_workflow
from convert.spec import PipelineSpec, TargetRef
from deploy.export import export_bundle_from_spec
from ingest.alteryx.parser import parse_yxmd

FIXTURES = Path(__file__).parent.parent / "fixtures" / "alteryx"
TARGET = TargetRef(catalog="workspace", schema="default", layer="bronze")


def _spec_with_cleanse(tmp_path: Path) -> PipelineSpec:
    xml = """<?xml version="1.0"?>
<AlteryxDocument yxmdVer="2023.1">
  <Nodes>
    <Node ToolID="1"><GuiSettings Plugin="AlteryxBasePluginsGui.DbFileInput.DbFileInput"/>
      <Properties><Configuration><File>main.x.raw</File></Configuration></Properties></Node>
    <Node ToolID="2"><GuiSettings Plugin="AlteryxBasePluginsGui.DataCleansePro.DataCleansePro"/>
      <Properties><Configuration><RemoveLeadingAndTrailingWhitespace value="True" /></Configuration></Properties></Node>
    <Node ToolID="3"><GuiSettings Plugin="AlteryxBasePluginsGui.DbFileOutput.DbFileOutput"/>
      <Properties><Configuration><File>main.x.out</File></Configuration></Properties></Node>
  </Nodes>
  <Connections>
    <Connection><Origin ToolID="1" Connection="Output" /><Destination ToolID="2" Connection="Input" /></Connection>
    <Connection><Origin ToolID="2" Connection="Output" /><Destination ToolID="3" Connection="Input" /></Connection>
  </Connections>
</AlteryxDocument>"""
    path = tmp_path / "cleanse_flow.yxmd"
    path.write_text(xml, encoding="utf-8")
    wf = parse_yxmd(path)
    return naive_spec_from_workflow(wf, TARGET)


def test_export_writes_main_and_utility_file_for_job(tmp_path: Path) -> None:
    spec = _spec_with_cleanse(tmp_path)
    out = export_bundle_from_spec(
        spec, tmp_path / "bundle", workspace_host="https://x.cloud.databricks.com",
        artifact_format="job",
    )
    main_file = out / "src" / "cleanse_flow.py"
    util_file = out / "src" / "cleanse_flow_utils.py"
    assert main_file.exists()
    assert util_file.exists()
    assert "def cleanse_columns(" in util_file.read_text(encoding="utf-8")
    assert "from cleanse_flow_utils import cleanse_columns" in main_file.read_text(encoding="utf-8")
    assert "def cleanse_columns(" not in main_file.read_text(encoding="utf-8")


def test_export_sdp_declares_utility_as_pipeline_library(tmp_path: Path) -> None:
    spec = _spec_with_cleanse(tmp_path)
    out = export_bundle_from_spec(
        spec, tmp_path / "bundle_sdp", workspace_host="https://x.cloud.databricks.com",
        artifact_format="sdp",
    )
    doc = yaml.safe_load((out / "databricks.yml").read_text(encoding="utf-8"))
    pipeline = next(iter(doc["resources"]["pipelines"].values()))
    paths = [lib["file"]["path"] for lib in pipeline["libraries"]]
    assert paths == ["./src/cleanse_flow.py", "./src/cleanse_flow_utils.py"]


def test_export_without_macros_or_cleanse_writes_only_main_file(tmp_path: Path) -> None:
    wf = parse_yxmd(FIXTURES / "sales_summary.yxmd")
    spec = naive_spec_from_workflow(wf, TARGET)
    out = export_bundle_from_spec(
        spec, tmp_path / "bundle_plain", workspace_host="https://x.cloud.databricks.com",
        artifact_format="job",
    )
    src_files = sorted(p.name for p in (out / "src").iterdir())
    assert src_files == ["sales_summary.py"]
