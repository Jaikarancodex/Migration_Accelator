"""Generates deterministic synthetic rows from a TableSchema.

Deterministic (seeded) generation matters here: the parity harness compares
source vs. target runs, so both sides must be fed the exact same synthetic
dataset for a comparison to mean anything.
"""

from __future__ import annotations

import random
from datetime import date, datetime, timedelta
from typing import Any

from eval.schema import ColumnSchema, TableSchema

_EPOCH = date(2020, 1, 1)


def _generate_value(column: ColumnSchema, row_index: int, rng: random.Random) -> Any:
    if column.nullable and rng.random() < 0.1:
        return None

    if column.key:
        # a handful of distinct keys, each shared by a few rows, so
        # group-by/aggregate comparisons have something to aggregate
        return row_index // 3 if column.data_type == "int" else f"K{row_index % 7}"

    if column.data_type == "int":
        return rng.randint(-1000, 1000)
    if column.data_type == "float":
        return round(rng.uniform(-1000.0, 1000.0), 2)
    if column.data_type == "bool":
        return rng.random() < 0.5
    if column.data_type == "date":
        return (_EPOCH + timedelta(days=rng.randint(0, 3650))).isoformat()
    if column.data_type == "timestamp":
        base = datetime.combine(_EPOCH, datetime.min.time())
        return (base + timedelta(seconds=rng.randint(0, 3650 * 86400))).isoformat()
    if column.data_type == "string":
        return f"val_{rng.randint(0, 9999)}"
    raise ValueError(f"Unsupported column type: {column.data_type}")


def generate_synthetic_rows(
    schema: TableSchema, num_rows: int, seed: int = 42
) -> list[dict[str, Any]]:
    """Generate `num_rows` synthetic rows matching `schema`, seeded for reproducibility."""
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for i in range(num_rows):
        row = {col.name: _generate_value(col, i, rng) for col in schema.columns}
        rows.append(row)
    return rows
