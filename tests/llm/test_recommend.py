import pytest

from ingest.alteryx.ir import Node, ToolType, UnsupportedTool, Workflow
from llm.client import MockLLMClient
from llm.recommend import (
    RecommendationError,
    heuristic_recommendation,
    recommend_deployment_format,
)


def _workflow(with_unsupported: bool = False, with_output: bool = True) -> Workflow:
    nodes = [
        Node(tool_id="1", tool_type=ToolType.INPUT, raw_plugin="DbFileInput", table_name="a.b.c"),
    ]
    if with_output:
        nodes.append(
            Node(tool_id="2", tool_type=ToolType.OUTPUT, raw_plugin="DbFileOutput",
                 upstream_ids=["1"], output_path="a.b.out")
        )
    unsupported = (
        [UnsupportedTool(tool_id="9", plugin="RegEx.RegEx", reason="Unrecognized plugin type")]
        if with_unsupported
        else []
    )
    return Workflow(source_file="x.yxmd", name="wf", nodes=nodes, unsupported=unsupported)


def test_recommend_parses_valid_yaml_response() -> None:
    client = MockLLMClient("format: sdp\nrationale: Declarative table-to-table shape.")
    rec = recommend_deployment_format(client, _workflow())
    assert rec.format == "sdp"
    assert "Declarative" in rec.rationale


def test_recommend_retries_with_error_context_then_succeeds() -> None:
    responses = iter(["format: spaceship\nrationale: nope", "format: job\nrationale: fine"])
    client = MockLLMClient(lambda system, user: next(responses))
    rec = recommend_deployment_format(client, _workflow())
    assert rec.format == "job"
    # the retry prompt must carry the validation failure back to the model
    assert "failed validation" in client.calls[1][1]


def test_recommend_raises_after_exhausting_retries() -> None:
    client = MockLLMClient("format: spaceship\nrationale: nope")
    with pytest.raises(RecommendationError):
        recommend_deployment_format(client, _workflow(), max_retries=1)


def test_recommend_prompt_includes_unsupported_tools() -> None:
    client = MockLLMClient("format: notebook\nrationale: needs iteration")
    recommend_deployment_format(client, _workflow(with_unsupported=True))
    _, user = client.calls[0]
    assert "RegEx.RegEx" in user


def test_heuristic_prefers_notebook_when_unsupported_tools_exist() -> None:
    assert heuristic_recommendation(_workflow(with_unsupported=True)).format == "notebook"


def test_heuristic_prefers_sdp_for_clean_table_writes() -> None:
    assert heuristic_recommendation(_workflow()).format == "sdp"


def test_heuristic_falls_back_to_job_without_outputs() -> None:
    assert heuristic_recommendation(_workflow(with_output=False)).format == "job"
