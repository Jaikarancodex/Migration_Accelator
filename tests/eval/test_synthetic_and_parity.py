import copy

from eval.parity import run_parity_check
from eval.schema import ColumnSchema, TableSchema
from eval.synthetic import generate_synthetic_rows

SCHEMA = TableSchema(
    name="sales_summary",
    columns=[
        ColumnSchema(name="CustomerID", data_type="int", key=True, nullable=False),
        ColumnSchema(name="TotalSales", data_type="float"),
        ColumnSchema(name="TransactionCount", data_type="int"),
    ],
)


def test_generation_is_deterministic_for_same_seed() -> None:
    rows_a = generate_synthetic_rows(SCHEMA, num_rows=50, seed=7)
    rows_b = generate_synthetic_rows(SCHEMA, num_rows=50, seed=7)
    assert rows_a == rows_b


def test_generation_differs_for_different_seed() -> None:
    rows_a = generate_synthetic_rows(SCHEMA, num_rows=50, seed=7)
    rows_b = generate_synthetic_rows(SCHEMA, num_rows=50, seed=8)
    assert rows_a != rows_b


def test_generated_rows_have_all_columns() -> None:
    rows = generate_synthetic_rows(SCHEMA, num_rows=10, seed=1)
    assert len(rows) == 10
    for row in rows:
        assert set(row.keys()) == {"CustomerID", "TotalSales", "TransactionCount"}


def test_parity_check_passes_for_identical_data() -> None:
    rows = generate_synthetic_rows(SCHEMA, num_rows=100, seed=3)
    target_rows = copy.deepcopy(rows)

    report = run_parity_check(
        rows, target_rows, key_columns=["CustomerID"], value_columns=["TotalSales", "TransactionCount"]
    )

    assert report.passed
    assert report.row_count_match
    assert report.checksum_mismatches == []
    assert report.key_aggregate_mismatches == []


def test_parity_check_passes_when_row_order_differs() -> None:
    rows = generate_synthetic_rows(SCHEMA, num_rows=100, seed=3)
    shuffled = list(reversed(rows))

    report = run_parity_check(
        rows, shuffled, key_columns=["CustomerID"], value_columns=["TotalSales", "TransactionCount"]
    )

    assert report.passed


def test_parity_check_detects_row_count_mismatch() -> None:
    rows = generate_synthetic_rows(SCHEMA, num_rows=100, seed=3)
    target_rows = rows[:-1]

    report = run_parity_check(
        rows, target_rows, key_columns=["CustomerID"], value_columns=["TotalSales", "TransactionCount"]
    )

    assert not report.passed
    assert not report.row_count_match
    assert report.row_count_source == 100
    assert report.row_count_target == 99


def test_parity_check_detects_checksum_mismatch() -> None:
    rows = generate_synthetic_rows(SCHEMA, num_rows=20, seed=3)
    target_rows = copy.deepcopy(rows)
    target_rows[0]["TotalSales"] = (target_rows[0]["TotalSales"] or 0) + 999.0

    report = run_parity_check(
        rows, target_rows, key_columns=["CustomerID"], value_columns=["TotalSales", "TransactionCount"]
    )

    assert not report.passed
    assert any(m.column == "TotalSales" for m in report.checksum_mismatches)


def test_parity_check_detects_key_aggregate_mismatch() -> None:
    rows = [
        {"CustomerID": 1, "TotalSales": 10.0, "TransactionCount": 1},
        {"CustomerID": 1, "TotalSales": 20.0, "TransactionCount": 1},
        {"CustomerID": 2, "TotalSales": 5.0, "TransactionCount": 1},
    ]
    target_rows = [
        {"CustomerID": 1, "TotalSales": 10.0, "TransactionCount": 1},
        {"CustomerID": 1, "TotalSales": 25.0, "TransactionCount": 1},  # drifted
        {"CustomerID": 2, "TotalSales": 5.0, "TransactionCount": 1},
    ]

    report = run_parity_check(
        rows, target_rows, key_columns=["CustomerID"], value_columns=["TotalSales"]
    )

    assert not report.passed
    assert len(report.key_aggregate_mismatches) == 1
    mismatch = report.key_aggregate_mismatches[0]
    assert mismatch.key == (1,)
    assert mismatch.source_value == 30.0
    assert mismatch.target_value == 35.0


def test_summary_is_human_readable() -> None:
    rows = generate_synthetic_rows(SCHEMA, num_rows=10, seed=3)
    report = run_parity_check(
        rows, rows, key_columns=["CustomerID"], value_columns=["TotalSales", "TransactionCount"]
    )
    assert "PASS" in report.summary()
