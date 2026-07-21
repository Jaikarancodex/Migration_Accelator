"""Source-vs-target parity comparison: row counts, column checksums, key aggregates.

Deliberately engine-agnostic: it compares two `list[dict]` row sets, so it
can be fed by `spark_df.collect()`, a pandas `.to_dict("records")`, or a
legacy engine's own export — whichever produced the "source of truth" rows
for this synthetic dataset. This session's scaffold does not wire up a real
Spark execution (no Java runtime in this environment); wiring
`run_pyspark_pipeline(...) -> list[dict]` in front of this comparator is the
next step once a Databricks/Spark runtime is available.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

_AGG_FUNCS: dict[str, Callable[[list[Any]], Any]] = {
    "sum": sum,
    "count": len,
    "min": min,
    "max": max,
}


class ChecksumMismatch(BaseModel):
    column: str
    source_checksum: int
    target_checksum: int


class KeyAggregateMismatch(BaseModel):
    key: tuple[Any, ...]
    column: str
    func: str
    source_value: float | int | None
    target_value: float | int | None


class ParityReport(BaseModel):
    passed: bool
    row_count_source: int
    row_count_target: int
    row_count_match: bool
    checksum_mismatches: list[ChecksumMismatch] = Field(default_factory=list)
    key_aggregate_mismatches: list[KeyAggregateMismatch] = Field(default_factory=list)

    def summary(self) -> str:
        if self.passed:
            return f"PASS: {self.row_count_source} rows, all checksums and key aggregates match."
        lines = [f"FAIL: source_rows={self.row_count_source} target_rows={self.row_count_target}"]
        for cm in self.checksum_mismatches:
            lines.append(f"  checksum mismatch on {cm.column}: source={cm.source_checksum} target={cm.target_checksum}")
        for km in self.key_aggregate_mismatches:
            lines.append(
                f"  key aggregate mismatch key={km.key} {km.func}({km.column}): "
                f"source={km.source_value} target={km.target_value}"
            )
        return "\n".join(lines)


def _column_checksum(rows: list[dict[str, Any]], column: str) -> int:
    """Order-independent checksum: sum of a stable hash of each (present) value."""
    total = 0
    for row in rows:
        value = row.get(column)
        total += hash((column, value)) % (2**61 - 1)
    return total % (2**61 - 1)


def compare_row_counts(
    source_rows: list[dict[str, Any]], target_rows: list[dict[str, Any]]
) -> tuple[bool, int, int]:
    return len(source_rows) == len(target_rows), len(source_rows), len(target_rows)


def compare_checksums(
    source_rows: list[dict[str, Any]], target_rows: list[dict[str, Any]], columns: list[str]
) -> list[ChecksumMismatch]:
    mismatches: list[ChecksumMismatch] = []
    for column in columns:
        source_sum = _column_checksum(source_rows, column)
        target_sum = _column_checksum(target_rows, column)
        if source_sum != target_sum:
            mismatches.append(
                ChecksumMismatch(column=column, source_checksum=source_sum, target_checksum=target_sum)
            )
    return mismatches


def _group_aggregate(
    rows: list[dict[str, Any]], key_columns: list[str], column: str, func: str
) -> dict[tuple[Any, ...], float | int | None]:
    groups: dict[tuple[Any, ...], list[Any]] = {}
    for row in rows:
        key = tuple(row.get(k) for k in key_columns)
        groups.setdefault(key, []).append(row.get(column))

    agg_fn = _AGG_FUNCS[func]
    results: dict[tuple[Any, ...], float | int | None] = {}
    for key, values in groups.items():
        if func == "count":
            results[key] = agg_fn(values)
        else:
            non_null = [v for v in values if v is not None]
            results[key] = agg_fn(non_null) if non_null else None
    return results


def _aggregates_equal(a: float | int | None, b: float | int | None) -> bool:
    """Value equality with float tolerance.

    Floating-point sums depend on addition order (and on the Python version:
    3.12's `sum()` compensates rounding, 3.11's doesn't), so two row sets
    holding identical values can legitimately aggregate to 334.69999999999993
    vs 334.7. Real engine comparisons (Alteryx export vs Spark output) differ
    in the last bits the same way; bit-exact equality would report false
    mismatches on correct migrations.
    """
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)


def compare_key_aggregates(
    source_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
    key_columns: list[str],
    columns: list[str],
    func: str = "sum",
) -> list[KeyAggregateMismatch]:
    if func not in _AGG_FUNCS:
        raise ValueError(f"Unsupported aggregate function {func!r}; choose from {sorted(_AGG_FUNCS)}")

    mismatches: list[KeyAggregateMismatch] = []
    for column in columns:
        source_agg = _group_aggregate(source_rows, key_columns, column, func)
        target_agg = _group_aggregate(target_rows, key_columns, column, func)
        for key in sorted(set(source_agg) | set(target_agg), key=str):
            source_value = source_agg.get(key)
            target_value = target_agg.get(key)
            if not _aggregates_equal(source_value, target_value):
                mismatches.append(
                    KeyAggregateMismatch(
                        key=key, column=column, func=func, source_value=source_value, target_value=target_value
                    )
                )
    return mismatches


def run_parity_check(
    source_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
    key_columns: list[str],
    value_columns: list[str],
    aggregate_func: str = "sum",
) -> ParityReport:
    """Run the full parity gate: row counts, per-column checksums, key aggregates."""
    row_count_match, source_count, target_count = compare_row_counts(source_rows, target_rows)
    checksum_mismatches = compare_checksums(source_rows, target_rows, key_columns + value_columns)
    key_aggregate_mismatches = compare_key_aggregates(
        source_rows, target_rows, key_columns, value_columns, aggregate_func
    )

    passed = row_count_match and not checksum_mismatches and not key_aggregate_mismatches
    return ParityReport(
        passed=passed,
        row_count_source=source_count,
        row_count_target=target_count,
        row_count_match=row_count_match,
        checksum_mismatches=checksum_mismatches,
        key_aggregate_mismatches=key_aggregate_mismatches,
    )
