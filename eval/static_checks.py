"""Gate (b): optimization / static-analysis checks on rendered PySpark code.

TODO (not built this session — see project non-goals): lint the rendered
module for anti-patterns (e.g. `.collect()` on large frames, missing
partition pruning, UDFs where a native function would do, cartesian joins),
likely via `ast` inspection of `convert/renderer.py` output plus a rules
table. Should return the same shape of pass/fail report as `eval/parity.py`
so `eval/` gates are composable.
"""

from __future__ import annotations


def run_static_checks(rendered_source: str) -> None:
    raise NotImplementedError("Static-analysis gate is not implemented in this session's slice.")
