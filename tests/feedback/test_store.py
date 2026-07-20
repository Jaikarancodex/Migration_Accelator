from pathlib import Path

from feedback.store import (
    ConversionRecord,
    find_similar_corrections,
    log_conversion_triple,
    summarize_correction,
)


def test_log_conversion_triple_appends_and_is_readable(tmp_path: Path) -> None:
    path = tmp_path / "feedback.jsonl"
    log_conversion_triple(
        workflow_name="wf1", tool_types=["join", "append_fields"],
        generated_spec_yaml="steps:\n- op: join\n  target: a\n  source: a\n",
        corrected_spec_yaml="steps:\n- op: join\n  target: a\n  source: b\n",
        store_path=path,
    )
    assert path.exists()
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = ConversionRecord.model_validate_json(lines[0])
    assert record.workflow_name == "wf1"
    assert record.tool_types == ["join", "append_fields"]
    assert record.logged_at  # auto-populated


def test_log_conversion_triple_is_noop_when_unchanged(tmp_path: Path) -> None:
    path = tmp_path / "feedback.jsonl"
    same = "steps: []\n"
    log_conversion_triple("wf1", ["filter"], same, same, store_path=path)
    assert not path.exists()


def test_find_similar_corrections_ranks_by_tool_overlap(tmp_path: Path) -> None:
    path = tmp_path / "feedback.jsonl"
    log_conversion_triple("wf_append", ["append_fields", "join"], "a", "b", store_path=path)
    log_conversion_triple("wf_filter", ["filter", "select"], "c", "d", store_path=path)
    log_conversion_triple("wf_mixed", ["append_fields", "filter"], "e", "f", store_path=path)

    results = find_similar_corrections({"append_fields", "join"}, limit=2, store_path=path)
    assert [r.workflow_name for r in results] == ["wf_append", "wf_mixed"]


def test_find_similar_corrections_empty_when_no_overlap(tmp_path: Path) -> None:
    path = tmp_path / "feedback.jsonl"
    log_conversion_triple("wf1", ["filter"], "a", "b", store_path=path)
    assert find_similar_corrections({"summarize"}, store_path=path) == []


def test_find_similar_corrections_on_missing_store(tmp_path: Path) -> None:
    assert find_similar_corrections({"join"}, store_path=tmp_path / "nope.jsonl") == []


def test_summarize_correction_produces_diff_and_truncates() -> None:
    record = ConversionRecord(
        workflow_name="wf",
        tool_types=["join"],
        generated_spec_yaml="\n".join(f"line{i}" for i in range(30)),
        corrected_spec_yaml="\n".join(f"line{i}" for i in range(29)) + "\nCHANGED",
    )
    diff = summarize_correction(record, max_lines=5)
    assert diff.count("\n") <= 5
    assert "truncated" in diff
