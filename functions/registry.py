"""Registry of reusable custom functions, keyed by name.

This is deliberately decoupled from importing the actual implementation
modules (which pull in `pyspark`): the registry only needs to describe
*signatures* so they can be injected into the LLM prompt ("prefer calling
these over writing new logic") and referenced by the YAML spec / renderer.
The renderer resolves `import_path` to generate the real import statement.
"""

from __future__ import annotations

from pydantic import BaseModel


class FunctionSignature(BaseModel):
    """Describes one reusable function for prompt injection and rendering."""

    name: str
    language: str  # "pyspark" | "sql"
    import_path: str  # dotted path the renderer emits an import for
    signature: str  # human-readable signature, shown to the LLM verbatim
    description: str


PYSPARK_FUNCTIONS: list[FunctionSignature] = [
    FunctionSignature(
        name="add_audit_columns",
        language="pyspark",
        import_path="functions.pyspark_lib.common.add_audit_columns",
        signature="add_audit_columns(df: DataFrame, source_system: str) -> DataFrame",
        description="Adds _ingested_at and _source_system columns. Use on every bronze/silver write.",
    ),
    FunctionSignature(
        name="dedupe_by_key",
        language="pyspark",
        import_path="functions.pyspark_lib.common.dedupe_by_key",
        signature=(
            "dedupe_by_key(df: DataFrame, keys: list[str], order_by: str, "
            "descending: bool = True) -> DataFrame"
        ),
        description=(
            "Keeps one row per `keys`, the latest by `order_by`. Use for Alteryx Unique tools "
            "or dedup-before-Summarize patterns."
        ),
    ),
    FunctionSignature(
        name="safe_join",
        language="pyspark",
        import_path="functions.pyspark_lib.common.safe_join",
        signature=(
            "safe_join(left: DataFrame, right: DataFrame, left_keys: list[str], "
            "right_keys: list[str], how: str = 'inner') -> DataFrame"
        ),
        description=(
            "Null-safe join (NULL == NULL matches, like Alteryx's Join tool) that drops "
            "duplicate right-side key columns. Use for every Join tool conversion."
        ),
    ),
    FunctionSignature(
        name="standardize_column_names",
        language="pyspark",
        import_path="functions.pyspark_lib.common.standardize_column_names",
        signature="standardize_column_names(df: DataFrame) -> DataFrame",
        description="Renames all columns to lower_snake_case to match medallion conventions.",
    ),
]

FUNCTIONS_BY_NAME: dict[str, FunctionSignature] = {f.name: f for f in PYSPARK_FUNCTIONS}


def get_function(name: str) -> FunctionSignature | None:
    return FUNCTIONS_BY_NAME.get(name)


def render_signatures_for_prompt() -> str:
    """Render all known function signatures as a block for the LLM prompt."""
    lines = []
    for fn in PYSPARK_FUNCTIONS:
        lines.append(f"- {fn.signature}\n  {fn.description}")
    return "\n".join(lines)
