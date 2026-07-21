"""The auto-repair loop: a Databricks failure plus the current spec goes back
to the LLM, which must return a minimally-changed, schema-valid spec.
"""

import pytest
import yaml

from convert.spec import PipelineSpec, ReadStep, SourceRef, TargetRef, WriteStep
from llm.client import MockLLMClient
from llm.convert import SpecGenerationError
from llm.repair import repair_pipeline_spec


def _spec_yaml(table: str) -> str:
    spec = PipelineSpec(
        name="wf", language="pyspark",
        source=SourceRef(system="alteryx", object_name="wf"),
        target=TargetRef(catalog="main", schema="dev", layer="silver"),
        steps=[
            ReadStep(id="r", source_table=table, alias="t"),
            WriteStep(id="w", input="r", target_table="main.dev.out"),
        ],
    )
    return yaml.safe_dump(spec.model_dump(by_alias=True), sort_keys=False)


def test_repair_returns_validated_spec_and_prompts_with_error() -> None:
    fixed = _spec_yaml("main.dev.correct_table")
    client = MockLLMClient(fixed)

    repaired = repair_pipeline_spec(
        client, _spec_yaml("main.dev.wrong_table"),
        "AnalysisException: Table main.dev.wrong_table not found",
        workflow_name="wf", stage="run",
    )

    assert isinstance(repaired, PipelineSpec)
    read = repaired.steps[0]
    assert isinstance(read, ReadStep)
    assert read.source_table == "main.dev.correct_table"
    _, user = client.calls[0]
    # the prompt must carry both the broken spec and the real error text
    assert "main.dev.wrong_table" in user
    assert "AnalysisException" in user
    assert "run" in user


def test_repair_retries_on_invalid_yaml_then_succeeds() -> None:
    responses = iter(["not: [valid", _spec_yaml("main.dev.t")])
    client = MockLLMClient(lambda _s, _u: next(responses))

    repaired = repair_pipeline_spec(
        client, _spec_yaml("main.dev.x"), "some error", workflow_name="wf"
    )
    assert len(client.calls) == 2
    read = repaired.steps[0]
    assert isinstance(read, ReadStep)
    assert read.source_table == "main.dev.t"
    # the retry prompt tells the model its previous repair failed validation
    assert "previous repair failed validation" in client.calls[1][1]


def test_repair_raises_after_exhausting_retries() -> None:
    client = MockLLMClient("still not yaml: [")
    with pytest.raises(SpecGenerationError):
        repair_pipeline_spec(
            client, _spec_yaml("main.dev.x"), "boom", workflow_name="wf", max_retries=1
        )
