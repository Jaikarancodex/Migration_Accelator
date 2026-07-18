"""Logs (source, generated, human-corrected, eval-result) triples.

TODO (not built this session): persist one record per conversion attempt so
few-shot/RAG retrieval (and later, fine-tuning) can be built on top of real
correction history. Should record: the ingest/alteryx IR, the LLM-generated
PipelineSpec, any human edits made in PR review, and the eval/parity.py
ParityReport for that attempt.
"""

from __future__ import annotations


def log_conversion_triple(*args: object, **kwargs: object) -> None:
    raise NotImplementedError("Feedback triple logging is not implemented in this session's slice.")
