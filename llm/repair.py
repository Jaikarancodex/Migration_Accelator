"""LLM-assisted repair of a PipelineSpec from a real Databricks failure.

Closes the loop that the deploy-error log opens: instead of only recording
what broke (feedback/store.py's deploy_errors.jsonl), the error output plus
the current spec go back to the LLM, which returns a minimally-changed spec
for a human to review and redeploy. Every applied repair is also logged as a
conversion correction, so the RAG loop learns from it like any human edit.
"""

from __future__ import annotations

import structlog
import yaml
from pydantic import ValidationError

from convert.spec import PipelineSpec
from llm.client import LLMClient
from llm.convert import SpecGenerationError, _strip_fences
from llm.prompt_builder import _ENV, SYSTEM_PROMPT

logger = structlog.get_logger(__name__)


def repair_pipeline_spec(
    client: LLMClient,
    spec_yaml: str,
    error_message: str,
    workflow_name: str,
    stage: str = "deploy",
    max_retries: int = 2,
) -> PipelineSpec:
    """Return a corrected PipelineSpec for a spec that failed on Databricks.

    Raises SpecGenerationError when the model cannot produce a valid spec —
    callers surface that instead of retrying forever.
    """
    template = _ENV.get_template("repair_spec.j2")
    user = template.render(
        workflow_name=workflow_name,
        spec_yaml=spec_yaml.strip(),
        error_message=error_message.strip()[:6000],
        stage=stage,
    )

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        prompt = user
        if last_error is not None:
            prompt = (
                f"{user}\n\n## Your previous repair failed validation\n"
                f"Error: {last_error}\nFix the YAML and emit the corrected document only."
            )
        raw = client.complete(SYSTEM_PROMPT, prompt)
        try:
            data = yaml.safe_load(_strip_fences(raw))
            return PipelineSpec.model_validate(data)
        except (yaml.YAMLError, ValidationError) as exc:
            last_error = exc
            logger.warning("repair_validation_failed", attempt=attempt, error=str(exc))

    raise SpecGenerationError(
        f"Could not produce a valid repaired spec for '{workflow_name}' after "
        f"{max_retries + 1} attempts: {last_error}"
    )
