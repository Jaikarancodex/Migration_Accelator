from pathlib import Path

import pytest

from convert.spec import PipelineSpec, TargetRef
from ingest.alteryx.parser import parse_yxmd
from llm.client import MockLLMClient
from llm.convert import SpecGenerationError, generate_pipeline_spec
from llm.prompt_builder import build_alteryx_to_pyspark_prompt

FIXTURE = Path(__file__).parent.parent / "fixtures" / "alteryx" / "sales_summary.yxmd"
TARGET = TargetRef(catalog="main", schema="migration_dev", layer="silver")

VALID_SPEC_YAML = """
name: sales_summary
language: pyspark
source:
  system: alteryx
  object_name: sales_summary
target:
  catalog: main
  schema: migration_dev
  layer: silver
steps:
  - op: read
    id: raw_sales
    source_table: legacy.sales.sales_raw
    alias: sales
  - op: filter
    id: positive_sales
    input: raw_sales
    condition: "[Amount] > 0"
  - op: write
    id: out
    input: positive_sales
    target_table: main.migration_dev.sales_summary
    mode: overwrite
functions_used: []
"""


def test_prompt_includes_function_signatures_and_nodes() -> None:
    workflow = parse_yxmd(FIXTURE)
    system, user = build_alteryx_to_pyspark_prompt(workflow, TARGET)

    assert "safe_join" in user
    assert "dedupe_by_key" in user
    assert "sales_summary" in user
    assert "[6] join" in user
    assert "RegEx" in user  # unsupported tool surfaced, not silently dropped
    assert "YAML" in system


def test_prompt_has_no_past_corrections_section_when_store_is_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import feedback.store as store_module

    monkeypatch.setattr(store_module, "_DEFAULT_STORE", tmp_path / "feedback.jsonl")
    workflow = parse_yxmd(FIXTURE)
    _, user = build_alteryx_to_pyspark_prompt(workflow, TARGET)
    assert "Past human corrections" not in user


def test_prompt_surfaces_similar_past_correction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import feedback.store as store_module

    monkeypatch.setattr(store_module, "_DEFAULT_STORE", tmp_path / "feedback.jsonl")
    # sales_summary.yxmd includes a join tool -- log a correction that overlaps on it.
    store_module.log_conversion_triple(
        workflow_name="past_join_fix",
        tool_types=["join", "input", "output"],
        generated_spec_yaml="steps:\n- op: join\n  left: a\n  right: a\n",
        corrected_spec_yaml="steps:\n- op: join\n  left: a\n  right: b\n",
    )
    workflow = parse_yxmd(FIXTURE)
    _, user = build_alteryx_to_pyspark_prompt(workflow, TARGET)
    assert "Past human corrections" in user
    assert "past_join_fix" in user
    assert "left: a" in user and "right: b" in user


def test_generate_pipeline_spec_success() -> None:
    workflow = parse_yxmd(FIXTURE)
    client = MockLLMClient(response=VALID_SPEC_YAML)

    spec = generate_pipeline_spec(client, workflow, TARGET)

    assert isinstance(spec, PipelineSpec)
    assert spec.name == "sales_summary"
    assert len(client.calls) == 1


def test_generate_pipeline_spec_strips_markdown_fences() -> None:
    workflow = parse_yxmd(FIXTURE)
    fenced = f"```yaml\n{VALID_SPEC_YAML}\n```"
    client = MockLLMClient(response=fenced)

    spec = generate_pipeline_spec(client, workflow, TARGET)

    assert spec.name == "sales_summary"


def test_generate_pipeline_spec_retries_on_invalid_yaml_then_succeeds() -> None:
    workflow = parse_yxmd(FIXTURE)
    responses = iter(["not: valid: yaml: at all: -", VALID_SPEC_YAML])

    def respond(_system: str, _user: str) -> str:
        return next(responses)

    client = MockLLMClient(response=respond)
    spec = generate_pipeline_spec(client, workflow, TARGET, max_retries=1)

    assert spec.name == "sales_summary"
    assert len(client.calls) == 2
    assert "previous attempt failed" in client.calls[1][1]


def test_generate_pipeline_spec_raises_after_exhausting_retries() -> None:
    workflow = parse_yxmd(FIXTURE)
    client = MockLLMClient(response="not valid yaml: [")

    with pytest.raises(SpecGenerationError):
        generate_pipeline_spec(client, workflow, TARGET, max_retries=1)

    assert len(client.calls) == 2
