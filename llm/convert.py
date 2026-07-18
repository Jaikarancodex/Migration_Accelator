"""Orchestrates one LLM call + validation into a PipelineSpec.

Validation failures are fed back to the model as additional context and
retried, the same feedback pattern the eval harness uses for its gates —
applied here one level earlier, at spec-validation time.
"""

from __future__ import annotations

import re

import structlog
import yaml
from pydantic import ValidationError

from convert.spec import PipelineSpec, TargetRef
from ingest.alteryx.ir import Workflow
from llm.client import LLMClient
from llm.prompt_builder import build_alteryx_to_pyspark_prompt

logger = structlog.get_logger(__name__)

_FENCE_RE = re.compile(r"^```[a-zA-Z]*\n|\n```$", re.MULTILINE)


class SpecGenerationError(Exception):
    """Raised when the LLM fails to produce a valid PipelineSpec after retries."""


def _strip_fences(text: str) -> str:
    return _FENCE_RE.sub("", text.strip())


def generate_pipeline_spec(
    client: LLMClient,
    workflow: Workflow,
    target: TargetRef,
    max_retries: int = 2,
) -> PipelineSpec:
    """Convert a parsed Alteryx Workflow into a validated PipelineSpec."""
    system, user = build_alteryx_to_pyspark_prompt(workflow, target)

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        prompt = user
        if last_error is not None:
            prompt = (
                f"{user}\n\n## Your previous attempt failed validation\n"
                f"Error: {last_error}\nFix the YAML and emit the corrected document only."
            )

        raw = client.complete(system, prompt)
        yaml_text = _strip_fences(raw)

        try:
            data = yaml.safe_load(yaml_text)
            return PipelineSpec.model_validate(data)
        except (yaml.YAMLError, ValidationError) as exc:
            last_error = exc
            logger.warning("spec_validation_failed", attempt=attempt, error=str(exc))

    raise SpecGenerationError(
        f"Failed to generate a valid PipelineSpec for '{workflow.name}' after "
        f"{max_retries + 1} attempts: {last_error}"
    )
