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
