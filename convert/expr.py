"""Translates Alteryx-style `[Field]` expressions into Spark SQL expression text.

Known limitation (documented in README): this only rewrites field references
into backtick-quoted identifiers and passes the rest through to Spark's SQL
expression parser via `F.expr(...)`. Alteryx-specific functions with no
direct Spark SQL equivalent (e.g. `IIF`, `DateTimeAdd`) are NOT translated —
those still need either a manual mapping table or LLM-assisted rewriting,
which is out of scope for this session's slice.
"""

from __future__ import annotations

import re

_FIELD_REF = re.compile(r"\[([^\[\]]+)]")


def alteryx_expr_to_spark(expression: str) -> str:
    """Rewrite `[Field Name]` references to backtick-quoted Spark SQL identifiers."""
    return _FIELD_REF.sub(lambda m: f"`{m.group(1)}`", expression)
