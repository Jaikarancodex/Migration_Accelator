"""Alteryx Output-Option -> WriteStep refresh mode, and its downstream effect."""

from pathlib import Path

from app.offline_convert import naive_spec_from_workflow
from convert.io_map import spec_io
from convert.renderer import render_sdp
from convert.spec import TargetRef, WriteStep
from ingest.alteryx.parser import parse_yxmd

TARGET = TargetRef(catalog="main", schema="dev", layer="silver")


def _wf_with_output_option(tmp_path: Path, option_xml: str) -> Path:
    xml = f"""<?xml version="1.0"?>
<AlteryxDocument yxmdVer="2023.1">
  <Nodes>
    <Node ToolID="1"><GuiSettings Plugin="AlteryxBasePluginsGui.DbFileInput.DbFileInput"/>
      <Properties><Configuration><File>main.dev.src</File></Configuration></Properties></Node>
    <Node ToolID="2"><GuiSettings Plugin="AlteryxBasePluginsGui.DbFileOutput.DbFileOutput"/>
      <Properties><Configuration>
        <File>main.dev.out</File>
        {option_xml}
      </Configuration></Properties></Node>
  </Nodes>
  <Connections>
    <Connection><Origin ToolID="1" Connection="Output" /><Destination ToolID="2" Connection="Input" /></Connection>
  </Connections>
</AlteryxDocument>"""
    p = tmp_path / "wf.yxmd"
    p.write_text(xml, encoding="utf-8")
    return p


def _write_mode(tmp_path: Path, option_xml: str) -> str:
    wf = parse_yxmd(_wf_with_output_option(tmp_path, option_xml))
    spec = naive_spec_from_workflow(wf, TARGET)
    write = next(s for s in spec.steps if isinstance(s, WriteStep))
    return write.mode


def test_overwrite_is_default_when_no_option(tmp_path: Path) -> None:
    assert _write_mode(tmp_path, "") == "overwrite"


def test_append_option_detected(tmp_path: Path) -> None:
    assert _write_mode(tmp_path, "<OutputOption>Append to Existing</OutputOption>") == "append"


def test_update_option_maps_to_merge(tmp_path: Path) -> None:
    assert _write_mode(tmp_path, "<OutputMode>Update; Insert if New</OutputMode>") == "merge"


def test_option_read_from_attribute(tmp_path: Path) -> None:
    assert _write_mode(tmp_path, '<Table outputMode="Append" />') == "append"


def test_refresh_type_surfaced_in_spec_io(tmp_path: Path) -> None:
    wf = parse_yxmd(_wf_with_output_option(tmp_path, "<OutputOption>Append</OutputOption>"))
    _, targets = spec_io(naive_spec_from_workflow(wf, TARGET))
    assert targets[0].refresh_type == "incremental"


def test_sdp_flags_incremental_target_for_streaming_review(tmp_path: Path) -> None:
    wf = parse_yxmd(_wf_with_output_option(tmp_path, "<OutputOption>Append</OutputOption>"))
    sdp = render_sdp(naive_spec_from_workflow(wf, TARGET))
    assert "REVIEW" in sdp
    assert "streaming table" in sdp
    assert "materialized view" in sdp
