import pytest

from convert.renderer import render_databricks_notebook, render_pyspark, render_sdp
from convert.spec import (
    AggregateStep,
    Aggregation,
    CallFunctionStep,
    ColumnSelection,
    ComputedColumn,
    DistinctStep,
    FilterStep,
    JoinStep,
    PipelineSpec,
    ReadStep,
    SelectStep,
    SortColumn,
    SortStep,
    SourceRef,
    TargetRef,
    UnionStep,
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


def test_render_union_sort_distinct_steps() -> None:
    spec = PipelineSpec(
        name="dedupe_case", language="pyspark", source=SOURCE, target=TARGET,
        steps=[
            ReadStep(id="east", source_table="legacy.sales.orders_east", alias="east"),
            ReadStep(id="west", source_table="legacy.sales.orders_west", alias="west"),
            UnionStep(id="stacked", inputs=["east", "west"]),
            DistinctStep(id="deduped", input="stacked", columns=["OrderID"]),
            SortStep(
                id="ordered", input="deduped",
                columns=[SortColumn(column="Region"), SortColumn(column="Amount", descending=True)],
            ),
            WriteStep(id="out", input="ordered", target_table="main.x.orders", mode="overwrite"),
        ],
    )
    source = render_pyspark(spec)
    compile(source, "<generated>", "exec")
    assert "df_stacked = df_east.unionByName(df_west, allowMissingColumns=True)" in source
    assert 'df_deduped = df_stacked.dropDuplicates(["OrderID"])' in source
    assert 'df_ordered = df_deduped.orderBy(F.col("Region").asc(), F.col("Amount").desc())' in source


def test_render_distinct_without_keys_uses_all_columns() -> None:
    spec = PipelineSpec(
        name="all_cols", language="pyspark", source=SOURCE, target=TARGET,
        steps=[
            ReadStep(id="r", source_table="t", alias="t"),
            DistinctStep(id="d", input="r"),
        ],
    )
    assert "df_d = df_r.dropDuplicates()" in render_pyspark(spec)


def test_render_notebook_has_cell_markers_and_no_run_wrapper() -> None:
    source = render_databricks_notebook(_full_spec())
    assert source.startswith("# Databricks notebook source")
    assert "# COMMAND ----------" in source
    assert "def run(spark)" not in source
    # notebook steps run at top level with the ambient spark session
    assert 'df_raw_sales = spark.table("legacy.sales.sales_raw")' in source


def test_render_notebook_rejects_sql_spec() -> None:
    spec = PipelineSpec(
        name="sql_case", language="sql", source=SOURCE, target=TARGET,
        steps=[ReadStep(id="r", source_table="t", alias="t")],
    )
    with pytest.raises(ValueError, match="sql"):
        render_databricks_notebook(spec)


def test_render_sdp_emits_dp_table_per_write() -> None:
    source = render_sdp(_full_spec())
    compile(source, "<generated>", "exec")
    assert "from pyspark import pipelines as dp" in source
    assert '@dp.table(name="sales_summary"' in source
    # the pipeline runtime owns the write: the table function returns the write step's input df
    assert "return df_audited" in source
    assert ".write.mode(" not in source
    # the legacy dlt module must not appear
    assert "dlt" not in source


def test_render_sdp_requires_a_write_step() -> None:
    spec = PipelineSpec(
        name="no_write", language="pyspark", source=SOURCE, target=TARGET,
        steps=[ReadStep(id="r", source_table="t", alias="t")],
    )
    with pytest.raises(ValueError, match="write"):
        render_sdp(spec)


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
