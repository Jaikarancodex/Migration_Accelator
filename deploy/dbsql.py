"""Databricks SQL Statement Execution helpers for the live parity gate.

Runs SQL against a workspace's serverless warehouse via the CLI's `api`
command so the app can seed expected data and compare migrated vs expected
tables with EXCEPT ALL row diffs. Auth is env-var only, mirroring
deploy/export.py.
"""

from __future__ import annotations

import csv
import io
import json
import os
import subprocess
import time
from typing import Any


class SqlError(RuntimeError):
    pass


def _cli_env(host: str, token: str | None) -> dict[str, str]:
    env = dict(os.environ)
    env["DATABRICKS_HOST"] = host
    env["DATABRICKS_CONFIG_FILE"] = os.devnull  # ignore stale CLI profiles
    if token:
        env["DATABRICKS_TOKEN"] = token
    return env


def _api(host: str, token: str | None, method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    cmd = ["databricks", "api", method, path]
    if body is not None:
        cmd += ["--json", json.dumps(body)]
    result = subprocess.run(
        cmd, env=_cli_env(host, token), capture_output=True, text=True, timeout=180
    )
    if result.returncode != 0:
        raise SqlError(result.stderr.strip() or result.stdout.strip())
    return json.loads(result.stdout) if result.stdout.strip() else {}


def first_warehouse_id(host: str, token: str | None) -> str:
    data = _api(host, token, "get", "/api/2.0/sql/warehouses")
    warehouses = data.get("warehouses", [])
    if not warehouses:
        raise SqlError("No SQL warehouse found in the workspace")
    return str(warehouses[0]["id"])


def run_sql(
    host: str, token: str | None, warehouse_id: str, statement: str, timeout_s: int = 150
) -> dict[str, Any]:
    """Execute one statement, waiting for completion; returns {columns, rows}."""
    resp = _api(
        host, token, "post", "/api/2.0/sql/statements",
        {"warehouse_id": warehouse_id, "statement": statement, "wait_timeout": "50s"},
    )
    deadline = time.monotonic() + timeout_s
    while resp.get("status", {}).get("state") in ("PENDING", "RUNNING"):
        if time.monotonic() > deadline:
            raise SqlError(f"Statement timed out: {statement[:120]}")
        time.sleep(3)
        resp = _api(host, token, "get", f"/api/2.0/sql/statements/{resp['statement_id']}")

    state = resp.get("status", {}).get("state")
    if state != "SUCCEEDED":
        message = resp.get("status", {}).get("error", {}).get("message", state)
        raise SqlError(f"{message}\n(statement: {statement[:300]})")
    columns = [c["name"] for c in resp.get("manifest", {}).get("schema", {}).get("columns", [])]
    rows = resp.get("result", {}).get("data_array") or []
    return {"columns": columns, "rows": rows}


def table_columns(host: str, token: str | None, warehouse_id: str, table: str) -> list[str]:
    return list(run_sql(host, token, warehouse_id, f"SELECT * FROM {table} LIMIT 0")["columns"])


def _sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def csv_to_table_statements(
    csv_bytes: bytes, table: str, max_rows: int = 5000, batch: int = 250
) -> list[str]:
    """CREATE + INSERT statements loading a CSV export into a Delta table.

    All columns are STRING (the parity diff casts both sides to STRING
    anyway); column mapping is enabled so headers with spaces survive.
    """
    reader = csv.reader(io.StringIO(csv_bytes.decode("utf-8-sig")))
    try:
        header = next(reader)
    except StopIteration as exc:
        raise SqlError("CSV file is empty") from exc

    col_ddl = ", ".join(f"`{c.strip()}` STRING" for c in header)
    col_list = ", ".join(f"`{c.strip()}`" for c in header)
    statements = [
        f"CREATE OR REPLACE TABLE {table} ({col_ddl}) "
        "TBLPROPERTIES ('delta.columnMapping.mode' = 'name', "
        "'delta.minReaderVersion' = '2', 'delta.minWriterVersion' = '5')"
    ]

    values: list[str] = []
    for i, row in enumerate(reader):
        if i >= max_rows:
            break
        padded = (row + [""] * len(header))[: len(header)]
        values.append("(" + ", ".join(_sql_literal(v) for v in padded) + ")")
    for start in range(0, len(values), batch):
        chunk = ", ".join(values[start : start + batch])
        statements.append(f"INSERT INTO {table} ({col_list}) VALUES {chunk}")
    return statements


def validation_report(
    host: str, token: str | None, warehouse_id: str, table: str
) -> dict[str, Any]:
    """Output validation when no Alteryx run is available to diff against.

    Structural checks on the migrated output: row count, schema, per-column
    null rates, and full-row duplicate count.
    """
    columns = table_columns(host, token, warehouse_id, table)
    checked = columns[:20]
    null_selects = ", ".join(
        f"SUM(CASE WHEN `{c}` IS NULL THEN 1 ELSE 0 END) AS `{c}`" for c in checked
    )
    stats = run_sql(
        host, token, warehouse_id,
        f"SELECT COUNT(*) AS __rows, {null_selects} FROM {table}",
    )
    row = stats["rows"][0]
    total = int(row[0])
    null_counts = dict(zip(checked, row[1:], strict=False))

    dupes = run_sql(
        host, token, warehouse_id,
        f"SELECT (SELECT COUNT(*) FROM {table}) - "
        f"(SELECT COUNT(*) FROM (SELECT DISTINCT * FROM {table}))",
    )["rows"][0][0]

    return {
        "table": table,
        "row_count": total,
        "columns": columns,
        "null_counts": null_counts,
        "duplicate_rows": int(dupes),
        "passed": total > 0,
    }


def parity_check(
    host: str,
    token: str | None,
    warehouse_id: str,
    migrated_table: str,
    expected_table: str,
    ignore_columns: list[str],
) -> dict[str, Any]:
    """Row-level parity: counts plus EXCEPT ALL in both directions.

    Compares the columns the two tables share (minus `ignore_columns`),
    casting every column to STRING so type differences don't mask value
    parity.
    """
    migrated_cols = table_columns(host, token, warehouse_id, migrated_table)
    expected_cols = table_columns(host, token, warehouse_id, expected_table)
    ignored_lower = {c.strip().lower() for c in ignore_columns if c.strip()}
    common = [
        c for c in migrated_cols
        if c in expected_cols and c.lower() not in ignored_lower
    ]
    if not common:
        raise SqlError(
            f"No comparable columns: migrated has {migrated_cols}, expected has {expected_cols}"
        )
    select_list = ", ".join(f"CAST(`{c}` AS STRING) AS `{c}`" for c in common)

    counts = run_sql(
        host, token, warehouse_id,
        f"SELECT (SELECT COUNT(*) FROM {migrated_table}) AS migrated, "
        f"(SELECT COUNT(*) FROM {expected_table}) AS expected",
    )["rows"][0]

    extra = run_sql(
        host, token, warehouse_id,
        f"SELECT {select_list} FROM {migrated_table} EXCEPT ALL "
        f"SELECT {select_list} FROM {expected_table} LIMIT 20",
    )
    missing = run_sql(
        host, token, warehouse_id,
        f"SELECT {select_list} FROM {expected_table} EXCEPT ALL "
        f"SELECT {select_list} FROM {migrated_table} LIMIT 20",
    )

    passed = (
        counts[0] == counts[1] and not extra["rows"] and not missing["rows"]
    )
    return {
        "passed": passed,
        "migrated_count": counts[0],
        "expected_count": counts[1],
        "compared_columns": common,
        "ignored_columns": sorted(ignored_lower),
        "extra_in_migrated": extra,
        "missing_from_migrated": missing,
    }
