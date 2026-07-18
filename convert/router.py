"""Routes a source object to Databricks SQL (set-based) or PySpark (procedural).

Per the mission: dataflow/procedural tools (Alteryx, Pentaho) convert to
PySpark; set-based SQL dialects (Synapse, Teradata, Oracle, Redshift,
BigQuery, MySQL, PostgreSQL stored procs) convert to Databricks SQL. Only the
Alteryx -> PySpark path is implemented this session; other source systems are
routed but conversion itself is a TODO (see convert/renderer.py).
"""

from __future__ import annotations

from convert.spec import Language

_PROCEDURAL_SOURCES: frozenset[str] = frozenset({"alteryx", "pentaho"})
_SET_BASED_SOURCES: frozenset[str] = frozenset(
    {"synapse", "teradata", "oracle", "redshift", "bigquery", "mysql", "postgresql"}
)


class UnknownSourceSystemError(ValueError):
    pass


def route_language(source_system: str) -> Language:
    """Return "pyspark" or "sql" for a given source system identifier."""
    system = source_system.lower()
    if system in _PROCEDURAL_SOURCES:
        return "pyspark"
    if system in _SET_BASED_SOURCES:
        return "sql"
    raise UnknownSourceSystemError(
        f"No routing rule for source system {source_system!r}. "
        f"Known procedural sources: {sorted(_PROCEDURAL_SOURCES)}; "
        f"known set-based sources: {sorted(_SET_BASED_SOURCES)}."
    )
