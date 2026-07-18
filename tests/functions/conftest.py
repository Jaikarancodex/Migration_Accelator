"""Session-scoped local SparkSession, skipped where no JVM is available.

This environment has no Java runtime, so pyspark's local mode cannot start;
these tests are written to run for real wherever Java is present (e.g. CI),
and skip cleanly here rather than failing the whole suite.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest


@pytest.fixture(scope="session")
def spark() -> Iterator[object]:
    pytest.importorskip("pyspark")
    from pyspark.sql import SparkSession

    try:
        session = (
            SparkSession.builder.master("local[1]")
            .appName("migration-accelerator-tests")
            .config("spark.ui.enabled", "false")
            .getOrCreate()
        )
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"Could not start a local SparkSession (no JVM?): {exc}")
    yield session
    session.stop()
