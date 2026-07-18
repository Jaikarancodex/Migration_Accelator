import pytest

from convert.renderer import render_pyspark
from convert.spec import (
    AggregateStep,
    Aggregation,
    CallFunctionStep,
    ColumnSelection,
    ComputedColumn,
    FilterStep,
    JoinStep,
    PipelineSpec,
    ReadStep,
    SelectStep,
    SourceRef,
    TargetRef,
    WithColumnsStep,
    WriteStep,
)

TARGET = TargetRef(catalog="main", schema="migration_dev", layer="silver")
SOURCE = SourceRef(system="alteryx", object_name="sales_summary")


def _full_spec() -> PipelineSpec:
    return PipelineSpec(
        name="sales_summary",
        language="pyspark",
        source=SOURCE,
        target=TARGET,
        steps=[
            ReadStep(id="raw_sales", source_table="legacy.sales.sales_raw", alias="sales"),
            ReadStep(id="raw_customers", source_table="legacy.sales.customers", alias="customers"),
            SelectStep(
                id="selected",
                input="raw_sales",
                columns=[
                    ColumnSelection(column="CustomerID"),
                    ColumnSelection(column="Amount", rename="Amount"),
                    ColumnSelection(column="Notes", drop=True),
                ],
            ),
            FilterStep(id="positive", input="selected", condition="[Amount] > 0"),
            WithColumnsStep(
                id="with_total",
                input="positive",
                columns=[ComputedColumn(name="LineTotal", expression="[Amount] * [Quantity]")],
            ),
            JoinStep(
                id="joined",
                left="with_total",
                right="raw_customers",
                left_keys=["CustomerID"],
                right_keys=["CustomerID"],
                how="inner",
                use_function="safe_join",
            ),
            AggregateStep(
                id="agg",
                input="joined",
                group_by=["CustomerID", "Region"],
                aggregations=[
                    Aggregation(column="LineTotal", func="sum", alias="TotalSales"),
                    Aggregation(column="LineTotal", func="count", alias="TransactionCount"),
                ],
            ),
            CallFunctionStep(id="audited", input="agg", function="add_audit_columns", args={"source_system": "alteryx"}),
            WriteStep(id="out", input="audited", target_table="main.migration_dev.sales_summary", mode="overwrite"),
        ],
        functions_used=["safe_join", "add_audit_columns"],
    )


def test_render_produces_syntactically_valid_python() -> None:
    source = render_pyspark(_full_spec())
    compile(source, "<generated>", "exec")  # raises SyntaxError if malformed


def test_render_includes_function_imports() -> None:
    source = render_pyspark(_full_spec())
    assert "from functions.pyspark_lib.common import add_audit_columns" in source
    assert "from functions.pyspark_lib.common import safe_join" in source


def test_render_read_step() -> None:
    source = render_pyspark(_full_spec())
    assert 'df_raw_sales = spark.table("legacy.sales.sales_raw")' in source


def test_render_select_step_drops_and_renames() -> None:
    source = render_pyspark(_full_spec())
    assert "df_selected = df_raw_sales.select(" in source
    assert '"Notes"' not in source.split("df_selected = ")[1].split("\n")[0]


def test_render_filter_translates_field_refs() -> None:
    source = render_pyspark(_full_spec())
    assert "F.expr('`Amount` > 0')" in source


def test_render_with_columns_chains_withcolumn() -> None:
    source = render_pyspark(_full_spec())
    assert 'df_with_total.withColumn("LineTotal", F.expr(\'`Amount` * `Quantity`\'))' in source


def test_render_join_uses_safe_join_function() -> None:
    source = render_pyspark(_full_spec())
    assert "safe_join(df_with_total, df_raw_customers" in source


def test_render_aggregate_step() -> None:
    source = render_pyspark(_full_spec())
    assert 'F.sum("LineTotal").alias("TotalSales")' in source
    assert 'F.count("LineTotal").alias("TransactionCount")' in source


def test_render_call_function_step() -> None:
    source = render_pyspark(_full_spec())
    assert "add_audit_columns(df_agg, source_system='alteryx')" in source


def test_render_write_step() -> None:
    source = render_pyspark(_full_spec())
    assert 'df_audited.write.mode(\'overwrite\').saveAsTable("main.migration_dev.sales_summary")' in source
    # df_out is never assigned; the write must act on the input dataframe, not a var named after
    # the write step's own id, or the generated code raises NameError at runtime.
    assert "df_out." not in source


def test_render_merge_mode_emits_todo_not_crash() -> None:
    spec = PipelineSpec(
        name="merge_case",
        language="pyspark",
        source=SOURCE,
        target=TARGET,
        steps=[
            ReadStep(id="r", source_table="t", alias="t"),
            WriteStep(id="w", input="r", target_table="main.x.y", mode="merge"),
        ],
    )
    source = render_pyspark(spec)
    compile(source, "<generated>", "exec")
    assert "TODO" in source


def test_render_rejects_non_pyspark_spec() -> None:
    spec = PipelineSpec(
        name="sql_case", language="sql", source=SOURCE, target=TARGET,
        steps=[ReadStep(id="r", source_table="t", alias="t")],
    )
    with pytest.raises(ValueError, match="sql"):
        render_pyspark(spec)


def test_render_unknown_function_raises() -> None:
    spec = PipelineSpec(
        name="bad_fn", language="pyspark", source=SOURCE, target=TARGET,
        steps=[
            ReadStep(id="r", source_table="t", alias="t"),
            CallFunctionStep(id="c", input="r", function="does_not_exist"),
        ],
        functions_used=["does_not_exist"],
    )
    with pytest.raises(ValueError, match="does_not_exist"):
        render_pyspark(spec)
