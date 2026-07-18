"""Starter reusable PySpark functions.

These exist so the LLM/renderer can call into a shared, tested library
instead of regenerating the same logic (dedup, joins, audit columns, name
standardization) in every pipeline. Keep signatures stable — they are quoted
verbatim in the LLM prompt (see llm/prompts) so the model can call them by
name.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime

from pyspark.sql import Column, DataFrame, Window
from pyspark.sql import functions as F  # noqa: N812


def add_audit_columns(df: DataFrame, source_system: str) -> DataFrame:
    """Add `_ingested_at` (UTC timestamp) and `_source_system` columns.

    Use on every bronze/silver write so lineage is queryable without
    re-deriving it from job metadata.
    """
    ingested_at = datetime.now(UTC).isoformat()
    return df.withColumn("_ingested_at", F.lit(ingested_at).cast("timestamp")).withColumn(
        "_source_system", F.lit(source_system)
    )


def dedupe_by_key(df: DataFrame, keys: list[str], order_by: str, descending: bool = True) -> DataFrame:
    """Keep one row per `keys` combination, the latest by `order_by`.

    Equivalent to Alteryx's Unique tool when paired with a sort, or to a
    dedup step feeding a Summarize tool. `order_by` is typically a
    watermark or last-modified column.
    """
    order_col: Column = F.col(order_by).desc() if descending else F.col(order_by).asc()
    window = Window.partitionBy(*keys).orderBy(order_col)
    return (
        df.withColumn("_rn", F.row_number().over(window))
        .filter(F.col("_rn") == 1)
        .drop("_rn")
    )


def safe_join(
    left: DataFrame,
    right: DataFrame,
    left_keys: list[str],
    right_keys: list[str],
    how: str = "inner",
) -> DataFrame:
    """Join on possibly-differently-named keys, treating NULLs as equal.

    Alteryx's Join tool treats NULL == NULL as a match; Spark's default `==`
    does not, so a naive translation silently drops NULL-keyed rows. This
    uses `eqNullSafe` to preserve Alteryx join semantics, and drops the
    right-side key columns to avoid ambiguous duplicate columns post-join.
    """
    if len(left_keys) != len(right_keys):
        raise ValueError("left_keys and right_keys must be the same length")

    condition = None
    for lk, rk in zip(left_keys, right_keys, strict=True):
        clause = left[lk].eqNullSafe(right[rk])
        condition = clause if condition is None else condition & clause

    joined = left.join(right, on=condition, how=how)
    drop_cols = [rk for rk, lk in zip(right_keys, left_keys, strict=True) if rk != lk]
    return joined.drop(*[right[c] for c in drop_cols]) if drop_cols else joined


_CAMEL_BOUNDARY = re.compile(r"(?<!^)(?=[A-Z])")
_NON_ALNUM = re.compile(r"[^0-9a-zA-Z]+")


def standardize_column_names(df: DataFrame) -> DataFrame:
    """Rename all columns to lower_snake_case.

    Legacy sources (Alteryx field names, Teradata/Oracle idents) are
    inconsistently cased; medallion conventions expect lower_snake_case.
    """
    renamed = df
    for col_name in df.columns:
        snake = _CAMEL_BOUNDARY.sub("_", col_name)
        snake = _NON_ALNUM.sub("_", snake).strip("_").lower()
        if snake != col_name:
            renamed = renamed.withColumnRenamed(col_name, snake)
    return renamed
