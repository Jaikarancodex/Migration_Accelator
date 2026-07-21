"""Sources/targets extraction and source-path binding."""

from convert.io_map import (
    apply_source_overrides,
    spec_io,
    workflow_sources,
    workflow_targets,
)
from convert.spec import (
    FilterStep,
    JoinStep,
    PipelineSpec,
    ReadStep,
    SourceRef,
    TargetRef,
    WriteStep,
)
from ingest.alteryx.ir import Node, ToolType, Workflow

TARGET = TargetRef(catalog="main", schema="dev", layer="silver")
SOURCE = SourceRef(system="alteryx", object_name="wf")


def _spec() -> PipelineSpec:
    return PipelineSpec(
        name="wf", language="pyspark", source=SOURCE, target=TARGET,
        steps=[
            ReadStep(id="r1", source_table="main.dev.todo_source_1", alias="a"),
            ReadStep(id="r2", source_table="main.dev.customers", alias="customers"),
            FilterStep(id="f", input="r1", condition="[x] > 0"),
            JoinStep(id="j", left="f", right="r2", left_keys=["id"], right_keys=["id"]),
            WriteStep(id="w", input="j", target_table="main.dev.out", mode="overwrite"),
        ],
    )


def test_spec_io_counts_and_names() -> None:
    sources, targets = spec_io(_spec())
    assert [s.read_id for s in sources] == ["r1", "r2"]
    assert [t.target_table for t in targets] == ["main.dev.out"]
    assert targets[0].fed_by == "j"


def test_spec_io_first_consumer_and_details() -> None:
    sources, _ = spec_io(_spec())
    r1 = next(s for s in sources if s.read_id == "r1")
    assert r1.first_consumer is not None
    assert r1.first_consumer.step_id == "f"
    assert "filter" in r1.first_consumer.detail
    # r2 is consumed by the join (as its right input)
    r2 = next(s for s in sources if s.read_id == "r2")
    assert r2.first_consumer is not None
    assert r2.first_consumer.step_id == "j"
    assert "join" in r2.first_consumer.detail


def test_apply_source_overrides_rebinds_table_and_alias() -> None:
    updated = apply_source_overrides(
        _spec(), {"r1": "real.catalog.orders", "r2": "  "}  # blank ignored
    )
    sources, _ = spec_io(updated)
    r1 = next(s for s in sources if s.read_id == "r1")
    assert r1.source_table == "real.catalog.orders"
    assert r1.alias == "orders"  # refreshed from the new table's last segment
    r2 = next(s for s in sources if s.read_id == "r2")
    assert r2.source_table == "main.dev.customers"  # unchanged (blank override)


def test_apply_source_overrides_leaves_original_untouched() -> None:
    original = _spec()
    apply_source_overrides(original, {"r1": "x.y.z"})
    # the input spec must not be mutated in place
    assert original.steps[0].source_table == "main.dev.todo_source_1"  # type: ignore[union-attr]


def test_workflow_sources_and_targets_from_ir() -> None:
    wf = Workflow(
        source_file="x.yxmd", name="wf",
        nodes=[
            Node(tool_id="1", tool_type=ToolType.INPUT, raw_plugin="I", table_name="legacy.orders"),
            Node(tool_id="2", tool_type=ToolType.INPUT, raw_plugin="I", table_name=None),
            Node(
                tool_id="9", tool_type=ToolType.OUTPUT, raw_plugin="O",
                output_path="legacy.out", upstream_ids=["1"],
            ),
        ],
    )
    assert workflow_sources(wf) == [
        ("1", "legacy.orders"),
        ("2", "(custom SQL / no table name)"),
    ]
    assert workflow_targets(wf) == [("9", "legacy.out")]
