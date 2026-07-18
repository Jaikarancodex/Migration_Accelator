"""Gate (c): medallion (bronze/silver/gold) compliance checks.

TODO (not built this session — see project non-goals): validate that a
PipelineSpec's `target.layer` matches structural expectations, e.g. bronze
writes are append-only from a single raw source, silver enforces
`add_audit_columns`/dedup, gold is aggregate-shaped and reads only from
silver/gold. Should consume `convert.spec.PipelineSpec` directly.
"""

from __future__ import annotations

from convert.spec import PipelineSpec


def check_medallion_compliance(spec: PipelineSpec) -> None:
    raise NotImplementedError("Medallion compliance gate is not implemented in this session's slice.")
