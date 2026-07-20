"""app/repo_cache.py wrappers must never serve stale data: every read is keyed
on the underlying file's mtime, so editing a file through the repo (the only
way the app itself mutates state) must be reflected on the next read even if
an older, cached copy exists for the same object name.
"""

import time
from pathlib import Path

from app.repo_cache import (
    all_macros,
    list_macro_names,
    list_metadata,
    list_object_names,
    read_metadata,
    read_workflow,
)
from ingest.alteryx.ir import Node, ToolType, Workflow
from repo.store import MigrationRepo


def _workflow(name: str, table: str) -> Workflow:
    return Workflow(
        source_file=f"{name}.yxmd", name=name,
        nodes=[Node(tool_id="1", tool_type=ToolType.INPUT, raw_plugin="DbFileInput", table_name=table)],
    )


def test_read_workflow_reflects_rewrite_not_a_stale_cache(tmp_path: Path) -> None:
    repo = MigrationRepo(tmp_path)
    repo.write_workflow(_workflow("wf", "legacy.a"))
    first = read_workflow(repo, "wf")
    assert first.nodes[0].table_name == "legacy.a"

    time.sleep(0.01)  # ensure a distinct mtime on filesystems with coarse resolution
    repo.write_workflow(_workflow("wf", "legacy.b"))
    second = read_workflow(repo, "wf")
    assert second.nodes[0].table_name == "legacy.b"


def test_list_object_names_reflects_newly_ingested_workflow(tmp_path: Path) -> None:
    repo = MigrationRepo(tmp_path)
    repo.write_workflow(_workflow("first", "t1"))
    assert list_object_names(repo) == ["first"]

    time.sleep(0.01)
    repo.write_workflow(_workflow("second", "t2"))
    assert list_object_names(repo) == ["first", "second"]


def test_list_metadata_matches_object_count(tmp_path: Path) -> None:
    repo = MigrationRepo(tmp_path)
    repo.write_workflow(_workflow("a", "t1"))
    repo.write_workflow(_workflow("b", "t2"))
    names = {m.name for m in list_metadata(repo)}
    assert names == {"a", "b"}


def test_read_metadata_reflects_rewrite(tmp_path: Path) -> None:
    repo = MigrationRepo(tmp_path)
    repo.write_workflow(_workflow("wf", "t1"))
    assert read_metadata(repo, "wf").unsupported_tool_count == 0

    time.sleep(0.01)
    wf2 = _workflow("wf", "t1")
    from ingest.alteryx.ir import UnsupportedTool
    wf2.unsupported = [UnsupportedTool(tool_id="9", plugin="X", reason="r")]
    repo.write_workflow(wf2)
    assert read_metadata(repo, "wf").unsupported_tool_count == 1


def test_macros_reflect_newly_registered_macro(tmp_path: Path) -> None:
    repo = MigrationRepo(tmp_path)
    assert list_macro_names(repo) == []
    repo.write_macro(_workflow("MyMacro", "t1"))
    assert list_macro_names(repo) == ["mymacro"]
    assert set(all_macros(repo).keys()) == {"mymacro"}
