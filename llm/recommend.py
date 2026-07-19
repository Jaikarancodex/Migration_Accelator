"""LLM-based recommendation of the deployment format for a converted workflow.

Given a parsed Workflow, asks the LLM whether the converted pipeline should
ship as a Databricks job script, a notebook, or a Lakeflow/Spark Declarative
Pipeline (SDP), and why. Follows the same pattern as llm/convert.py: the
model emits YAML validated against a pydantic model, with the validation
error fed back on retry. `heuristic_recommendation` is the deterministic
fallback for offline use (no API key), mirroring app/offline_convert.py.
"""

from __future__ import annotations

import re
from typing import Literal

import structlog
import yaml
from pydantic import BaseModel, ValidationError

from ingest.alteryx.ir import ToolType, Workflow
from llm.client import LLMClient

logger = structlog.get_logger(__name__)

DeploymentFormat = Literal["job", "notebook", "sdp"]

_FENCE_RE = re.compile(r"^```[a-zA-Z]*\n|\n```$", re.MULTILINE)

_SYSTEM_PROMPT = (
    "You are a Databricks deployment advisor for an internal migration accelerator. "
    "Given a summary of a converted legacy pipeline, choose how it should be deployed:\n"
    "- job: a scheduled batch job running a plain Python script. Best for stable, "
    "fully-converted pipelines with simple table-to-table batch semantics.\n"
    "- notebook: a Databricks notebook run as a notebook task. Best when engineers "
    "still need to iterate interactively — e.g. the conversion is partial, has "
    "unsupported source tools, or needs cell-by-cell inspection before trusting it.\n"
    "- sdp: a Lakeflow/Spark Declarative Pipeline (dlt tables). Best for declarative "
    "table-to-table transformations feeding a medallion architecture, especially "
    "multi-output pipelines where Databricks should manage orchestration, retries, "
    "and data quality.\n"
    "Respond with only a YAML document with exactly two keys: `format` (one of "
    "job|notebook|sdp) and `rationale` (2-3 sentences). No markdown fences, no "
    "commentary."
)


class DeploymentRecommendation(BaseModel):
    """The validated shape of the LLM's (or heuristic's) answer."""

    format: DeploymentFormat
    rationale: str


class RecommendationError(Exception):
    """Raised when the LLM fails to produce a valid recommendation after retries."""


def _workflow_summary(workflow: Workflow) -> str:
    counts: dict[str, int] = {}
    for node in workflow.nodes:
        counts[node.tool_type.value] = counts.get(node.tool_type.value, 0) + 1
    outputs = [n.output_path for n in workflow.nodes if n.tool_type == ToolType.OUTPUT]
    lines = [
        f"Workflow name: {workflow.name}",
        f"Node counts by type: {counts}",
        f"Output tables ({len(outputs)}): {outputs}",
        f"Unsupported tools (need manual conversion): {len(workflow.unsupported)}",
    ]
    for u in workflow.unsupported:
        lines.append(f"  - [{u.tool_id}] {u.plugin}: {u.reason}")
    return "\n".join(lines)


def recommend_deployment_format(
    client: LLMClient,
    workflow: Workflow,
    max_retries: int = 2,
) -> DeploymentRecommendation:
    """Ask the LLM which deployment format fits this workflow, with retry-on-invalid."""
    user = f"## Pipeline summary\n{_workflow_summary(workflow)}\n\nChoose the deployment format."

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        prompt = user
        if last_error is not None:
            prompt = (
                f"{user}\n\n## Your previous attempt failed validation\n"
                f"Error: {last_error}\nFix the YAML and emit the corrected document only."
            )

        raw = client.complete(_SYSTEM_PROMPT, prompt)
        yaml_text = _FENCE_RE.sub("", raw.strip())

        try:
            data = yaml.safe_load(yaml_text)
            return DeploymentRecommendation.model_validate(data)
        except (yaml.YAMLError, ValidationError) as exc:
            last_error = exc
            logger.warning("recommendation_validation_failed", attempt=attempt, error=str(exc))

    raise RecommendationError(
        f"Failed to get a valid deployment recommendation for '{workflow.name}' after "
        f"{max_retries + 1} attempts: {last_error}"
    )


def heuristic_recommendation(workflow: Workflow) -> DeploymentRecommendation:
    """Deterministic fallback mirroring the prompt's decision rules, for offline use."""
    if workflow.unsupported:
        return DeploymentRecommendation(
            format="notebook",
            rationale=(
                f"{len(workflow.unsupported)} source tool(s) could not be converted "
                "automatically, so an engineer needs to iterate on this interactively "
                "before it can be trusted as a scheduled artifact."
            ),
        )

    output_count = sum(1 for n in workflow.nodes if n.tool_type == ToolType.OUTPUT)
    if output_count >= 1:
        return DeploymentRecommendation(
            format="sdp",
            rationale=(
                "The workflow is fully converted and writes to "
                f"{output_count} output table(s) — a declarative table-to-table shape "
                "where a Lakeflow/Spark Declarative Pipeline lets Databricks manage "
                "orchestration and data quality."
            ),
        )

    return DeploymentRecommendation(
        format="job",
        rationale=(
            "The workflow is fully converted but writes no managed output tables, "
            "so a plain scheduled job script is the simplest fit."
        ),
    )
