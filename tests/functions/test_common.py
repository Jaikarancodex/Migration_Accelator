"""Real-Spark tests for functions/pyspark_lib/common.py.

Skips automatically if no JVM is available (see conftest.py) — the logic is
also exercised indirectly by convert/renderer.py's output, but these confirm
actual DataFrame behavior wherever Spark can run.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pyspark")

from pyspark.sql import SparkSession  # noqa: E402

from functions.pyspark_lib.common import (  # noqa: E402
    add_audit_columns,
    dedupe_by_key,
    safe_join,
    standardize_column_names,
)


def test_add_audit_columns(spark: SparkSession) -> None:
    df = spark.createDataFrame([(1, "a")], ["id", "name"])
    result = add_audit_columns(df, source_system="alteryx")
    assert set(result.columns) == {"id", "name", "_ingested_at", "_source_system"}
    row = result.collect()[0]
    assert row["_source_system"] == "alteryx"


def test_dedupe_by_key_keeps_latest(spark: SparkSession) -> None:
    df = spark.createDataFrame(
        [(1, "old", 1), (1, "new", 2), (2, "only", 1)], ["id", "val", "version"]
    )
    result = dedupe_by_key(df, keys=["id"], order_by="version").orderBy("id")
    rows = result.collect()
    assert len(rows) == 2
    assert rows[0]["val"] == "new"
    assert rows[1]["val"] == "only"


def test_safe_join_matches_null_keys(spark: SparkSession) -> None:
    left = spark.createDataFrame([(1,), (None,)], ["key"])
    right = spark.createDataFrame([(1, "matched"), (None, "null_matched")], ["key", "label"])
    result = safe_join(left, right, left_keys=["key"], right_keys=["key"], how="inner")
    labels = {row["label"] for row in result.collect()}
    assert labels == {"matched", "null_matched"}


def test_standardize_column_names(spark: SparkSession) -> None:
    df = spark.createDataFrame([(1,)], ["CustomerID"])
    result = standardize_column_names(df)
    assert result.columns == ["customer_id"]
