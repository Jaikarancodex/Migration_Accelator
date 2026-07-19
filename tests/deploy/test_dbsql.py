import pytest

from deploy.dbsql import SqlError, csv_to_table_statements


def test_csv_to_table_statements_creates_and_batches() -> None:
    csv_bytes = b"Name,Amount\nAnna,10\nBen,20\nO'Neil,30\n"
    statements = csv_to_table_statements(csv_bytes, "workspace.default.expected_x", batch=2)

    assert statements[0].startswith("CREATE OR REPLACE TABLE workspace.default.expected_x")
    assert "`Name` STRING" in statements[0]
    assert "delta.columnMapping.mode" in statements[0]
    # 3 rows with batch=2 -> two INSERT statements
    inserts = [s for s in statements if s.startswith("INSERT")]
    assert len(inserts) == 2
    # single quotes escaped
    assert "O''Neil" in inserts[1]


def test_csv_to_table_statements_pads_short_rows() -> None:
    csv_bytes = b"A,B,C\n1,2\n"
    statements = csv_to_table_statements(csv_bytes, "t")
    assert "('1', '2', '')" in statements[1]


def test_empty_csv_raises() -> None:
    with pytest.raises(SqlError, match="empty"):
        csv_to_table_statements(b"", "t")
