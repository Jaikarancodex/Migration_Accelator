"""Placeholder for the SQL-side reusable function/macro library.

This session's vertical slice is Alteryx -> PySpark only (see
convert/router.py); the set-based SQL conversion path routes here but has no
renderer yet. TODO: mirror functions/pyspark_lib as Jinja SQL macros
(e.g. audit-column CTEs, null-safe join snippets) once a SQL source dialect
is implemented.
"""
