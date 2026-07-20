from pathlib import Path

from app.offline_convert import naive_spec_from_workflow
from convert.renderer import render_databricks_notebook, render_pyspark, render_sdp
from convert.spec import MacroCallStep, TargetRef
from ingest.alteryx.parser import parse_yxmd

FIXTURES = Path(__file__).parent.parent / "fixtures" / "alteryx"
TARGET = TargetRef(catalog="workspace", schema="default", layer="bronze")


def _main_workflow_with_macro(tmp_path: Path) -> Path:
    """A .yxmd whose middle node references normalize_names.yxmc."""
    xml = """<?xml version="1.0"?>
<AlteryxDocument yxmdVer="2023.1">
  <Nodes>
    <Node ToolID="1">
      <GuiSettings Plugin="AlteryxBasePluginsGui.DbFileInput.DbFileInput">
        <Position x="54" y="54" />
      </GuiSettings>
      <Properties>
        <Configuration>
          <File>legacy.hr.people</File>
        </Configuration>
      </Properties>
    </Node>
    <Node ToolID="2">
      <GuiSettings>
        <Position x="150" y="54" />
      </GuiSettings>
      <Properties>
        <Configuration>
          <Value name="Some Setting">x</Value>
        </Configuration>
      </Properties>
      <EngineSettings Macro="normalize_names.yxmc" />
    </Node>
    <Node ToolID="3">
      <GuiSettings Plugin="AlteryxBasePluginsGui.DbFileOutput.DbFileOutput">
        <Position x="246" y="54" />
      </GuiSettings>
      <Properties>
        <Configuration>
          <File>legacy.hr.people_clean</File>
        </Configuration>
      </Properties>
    </Node>
  </Nodes>
  <Connections>
    <Connection>
      <Origin ToolID="1" Connection="Output" />
      <Destination ToolID="2" Connection="Input" />
    </Connection>
    <Connection>
      <Origin ToolID="2" Connection="Output" />
      <Destination ToolID="3" Connection="Input" />
    </Connection>
  </Connections>
</AlteryxDocument>"""
    path = tmp_path / "people_pipeline.yxmd"
    path.write_text(xml, encoding="utf-8")
    return path


def test_registered_macro_becomes_utility_called_in_flow(tmp_path: Path) -> None:
    from convert.renderer import render_utility_module, utils_module_name

    macro_wf = parse_yxmd(FIXTURES / "normalize_names.yxmc")
    main_wf = parse_yxmd(_main_workflow_with_macro(tmp_path))

    spec = naive_spec_from_workflow(main_wf, TARGET, macros={"normalize_names": macro_wf})

    assert len(spec.macros) == 1
    macro = spec.macros[0]
    assert macro.name == "macro_normalize_names"
    assert any(isinstance(s, MacroCallStep) and s.macro == macro.name for s in spec.steps)

    # the macro body lives in a separate, shared utility module...
    module = utils_module_name(spec)
    util_code = render_utility_module(spec)
    assert util_code is not None
    assert "def macro_normalize_names(df_macro_input):" in util_code
    # macro internals: uppercase formula + filter, translated to Spark SQL
    assert "upper(`First`) || ' ' || upper(`Last`)" in util_code
    compile(util_code, "<generated>", "exec")

    # ...and every format imports it and calls it rather than inlining it.
    for renderer in (render_pyspark, render_databricks_notebook, render_sdp):
        code = renderer(spec)
        assert "def macro_normalize_names(df_macro_input):" not in code
        assert f"from {module} import macro_normalize_names" in code
        assert "macro_normalize_names(df_1)" in code

    job = render_pyspark(spec)
    compile(job, "<generated>", "exec")


def test_unregistered_macro_is_bridged_over(tmp_path: Path) -> None:
    main_wf = parse_yxmd(_main_workflow_with_macro(tmp_path))
    spec = naive_spec_from_workflow(main_wf, TARGET, macros={})

    assert spec.macros == []
    assert not any(isinstance(s, MacroCallStep) for s in spec.steps)
    # write connects straight through to the read (macro elided)
    write = next(s for s in spec.steps if s.op == "write")
    assert write.input == "1"
