from pathlib import Path

import pytest

from ingest.alteryx.parser import parse_yxmd
from repo.graph import CyclicDependencyError, DependencyGraph, infer_dependencies
from repo.metadata import ObjectMetadata
from repo.store import MigrationRepo, ObjectNotFoundError

FIXTURE = Path(__file__).parent.parent / "fixtures" / "alteryx" / "sales_summary.yxmd"


def test_write_and_read_workflow_roundtrip(tmp_path: Path) -> None:
    workflow = parse_yxmd(FIXTURE)
    repo = MigrationRepo(tmp_path)

    metadata = repo.write_workflow(workflow)

    assert metadata.name == "sales_summary"
    assert metadata.input_tables == ["legacy.sales.customers", "legacy.sales.sales_raw"]
    assert metadata.output_tables == ["legacy.sales.sales_summary"]
    assert metadata.unsupported_tool_count == 1

    reloaded = repo.read_workflow("sales_summary")
    assert reloaded.name == workflow.name
    assert len(reloaded.nodes) == len(workflow.nodes)

    reloaded_metadata = repo.read_metadata("sales_summary")
    assert reloaded_metadata == metadata


def test_list_object_names(tmp_path: Path) -> None:
    workflow = parse_yxmd(FIXTURE)
    repo = MigrationRepo(tmp_path)
    repo.write_workflow(workflow)

    assert repo.list_object_names() == ["sales_summary"]


def test_read_missing_object_raises(tmp_path: Path) -> None:
    repo = MigrationRepo(tmp_path)
    with pytest.raises(ObjectNotFoundError):
        repo.read_workflow("does_not_exist")


def _meta(name: str, inputs: list[str], outputs: list[str]) -> ObjectMetadata:
    return ObjectMetadata(
        name=name, source_system="alteryx", source_file=f"{name}.yxmd",
        input_tables=inputs, output_tables=outputs,
    )


def test_infer_dependencies_from_matching_tables() -> None:
    metas = [
        _meta("bronze_customers", inputs=[], outputs=["raw.customers"]),
        _meta("silver_customers", inputs=["raw.customers"], outputs=["silver.customers"]),
        _meta("gold_summary", inputs=["silver.customers"], outputs=["gold.summary"]),
    ]
    deps = infer_dependencies(metas)
    assert deps["bronze_customers"] == []
    assert deps["silver_customers"] == ["bronze_customers"]
    assert deps["gold_summary"] == ["silver_customers"]


def test_dependency_graph_topological_order() -> None:
    metas = [
        _meta("gold_summary", inputs=["silver.customers"], outputs=["gold.summary"]),
        _meta("bronze_customers", inputs=[], outputs=["raw.customers"]),
        _meta("silver_customers", inputs=["raw.customers"], outputs=["silver.customers"]),
    ]
    graph = DependencyGraph(metas)
    order = graph.topological_order()

    assert order.index("bronze_customers") < order.index("silver_customers")
    assert order.index("silver_customers") < order.index("gold_summary")
    assert graph.dependencies_of("gold_summary") == ["silver_customers"]
    assert graph.dependents_of("bronze_customers") == ["silver_customers"]


def test_dependency_graph_detects_cycles() -> None:
    metas = [
        _meta("a", inputs=["table_b"], outputs=["table_a"]),
        _meta("b", inputs=["table_a"], outputs=["table_b"]),
    ]
    with pytest.raises(CyclicDependencyError):
        DependencyGraph(metas)
