"""Logs (source tool types, generated spec, human-corrected spec) triples and
retrieves the ones most similar to a new workflow, for injection into the LLM
conversion prompt as few-shot corrections.

This is the RAG-first half of the project's planned feedback loop (fine-tuning
is explicitly out of scope, per the original brief): every time a reviewer
edits a generated spec in the app, the before/after is logged here; the next
conversion of a workflow using similar tools gets those corrections surfaced
in its prompt, so the same mistake — like the AppendFields label mix-up caught
in review — doesn't have to be caught by a human twice.
"""

from __future__ import annotations

import difflib
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

_DEFAULT_STORE = Path(__file__).resolve().parent.parent / "migration_repo_output" / "feedback.jsonl"


class ConversionRecord(BaseModel):
    """One logged human correction to an LLM/offline-generated pipeline spec."""

    workflow_name: str
    tool_types: list[str]
    generated_spec_yaml: str
    corrected_spec_yaml: str
    logged_at: str = ""

    def model_post_init(self, __context: object) -> None:
        if not self.logged_at:
            self.logged_at = datetime.now(UTC).isoformat()


def log_conversion_triple(
    workflow_name: str,
    tool_types: list[str],
    generated_spec_yaml: str,
    corrected_spec_yaml: str,
    store_path: Path | None = None,
) -> None:
    """Append one correction record. No-ops if the correction is a no-op edit."""
    if corrected_spec_yaml.strip() == generated_spec_yaml.strip():
        return
    path = store_path or _DEFAULT_STORE
    path.parent.mkdir(parents=True, exist_ok=True)
    record = ConversionRecord(
        workflow_name=workflow_name,
        tool_types=tool_types,
        generated_spec_yaml=generated_spec_yaml,
        corrected_spec_yaml=corrected_spec_yaml,
    )
    with path.open("a", encoding="utf-8") as f:
        f.write(record.model_dump_json() + "\n")


def _load_records(store_path: Path | None = None) -> list[ConversionRecord]:
    path = store_path or _DEFAULT_STORE
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(ConversionRecord.model_validate_json(line))
    return records


def find_similar_corrections(
    tool_types: set[str],
    limit: int = 2,
    store_path: Path | None = None,
) -> list[ConversionRecord]:
    """Past corrections ranked by Jaccard overlap of tool types with `tool_types`.

    Simple keyword-set matching rather than embeddings — proportionate to the
    logged volume this app will realistically see, and needs no vector store.
    """
    records = _load_records(store_path)
    if not tool_types or not records:
        return []

    def overlap(record: ConversionRecord) -> float:
        record_types = set(record.tool_types)
        if not record_types:
            return 0.0
        union = tool_types | record_types
        return len(tool_types & record_types) / len(union) if union else 0.0

    scored = [(overlap(r), r) for r in records]
    scored = [(score, r) for score, r in scored if score > 0]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [r for _, r in scored[:limit]]


def summarize_correction(record: ConversionRecord, max_lines: int = 16) -> str:
    """A short unified diff between the generated and corrected spec, for prompts."""
    diff = list(
        difflib.unified_diff(
            record.generated_spec_yaml.splitlines(),
            record.corrected_spec_yaml.splitlines(),
            lineterm="",
            n=1,
        )
    )
    body = "\n".join(diff[:max_lines])
    if len(diff) > max_lines:
        body += "\n... (truncated)"
    return body
