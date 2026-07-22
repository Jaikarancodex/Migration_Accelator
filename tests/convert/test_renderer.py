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


def test_render_filter_flags_unrecognized_function_for_review() -> None:
    spec = PipelineSpec(
        name="flagged", language="pyspark", source=SOURCE, target=TARGET,
        steps=[
            ReadStep(id="r", source_table="t", alias="t"),
            FilterStep(id="f", input="r", condition='GetWord([A], 2) = "x"'),
            WriteStep(id="w", input="f", target_table="main.x.flagged"),
        ],
    )
    source = render_pyspark(spec)
    filter_line = next(line for line in source.splitlines() if "df_f = df_r.filter" in line)
    assert "# REVIEW: verify" in filter_line
    assert "GetWord" in filter_line


def test_render_filter_no_review_comment_for_known_functions() -> None:
    source = render_pyspark(_full_spec())
    filter_line = next(line for line in source.splitlines() if "df_positive = " in line)
    assert "REVIEW" not in filter_line


def test_render_with_columns_flags_unrecognized_function_for_review() -> None:
    spec = PipelineSpec(
        name="flagged", language="pyspark", source=SOURCE, target=TARGET,
        steps=[
            ReadStep(id="r", source_table="t", alias="t"),
            WithColumnsStep(
                id="w1", input="r",
                columns=[ComputedColumn(name="Next", expression='DateTimeAdd([A],3,"FORTNIGHT")')],
            ),
            WriteStep(id="w", input="w1", target_table="main.x.flagged"),
        ],
    )
    source = render_pyspark(spec)
    with_columns_line = next(line for line in source.splitlines() if '"Next"' in line)
    assert "# REVIEW: verify" in with_columns_line
    assert "DateTimeAdd" in with_columns_line


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


def test_render_cleanse_imports_from_separate_utility_module_in_all_formats() -> None:
    from convert.renderer import render_utility_module, utils_module_name
    from convert.spec import CleanseStep

    spec = PipelineSpec(
        name="cleanse_case", language="pyspark", source=SOURCE, target=TARGET,
        steps=[
            ReadStep(id="r", source_table="t", alias="raw"),
            CleanseStep(id="c", input="r", columns=["a"], trim=True, nulls_to_blank=True),
            WriteStep(id="w", input="c", target_table="main.x.out", mode="overwrite"),
        ],
    )
    module = utils_module_name(spec)
    assert module == "cleanse_case_utils"

    util = render_utility_module(spec)
    assert util is not None
    compile(util, "<generated>", "exec")
    assert "def cleanse_columns(" in util

    job = render_pyspark(spec)
    compile(job, "<generated>", "exec")
    assert "def cleanse_columns(" not in job  # no longer inlined
    assert f"from {module} import cleanse_columns" in job
    assert "df_c = cleanse_columns(df_r, columns=['a'], trim=True, nulls_to_blank=True)" in job

    notebook = render_databricks_notebook(spec)
    assert "def cleanse_columns(" not in notebook
    assert f"from {module} import cleanse_columns" in notebook

    sdp = render_sdp(spec)
    compile(sdp, "<generated>", "exec")
    assert "def cleanse_columns(" not in sdp
    assert f"from {module} import cleanse_columns" in sdp
    # the utility is called from inside the silver layer
    silver = sdp.split('silver_cleanse_case"')[1].split("@dp.table")[0]
    assert "cleanse_columns(" in silver


def test_render_utility_module_is_none_when_unused() -> None:
    from convert.renderer import render_utility_module

    spec = PipelineSpec(
        name="plain_case", language="pyspark", source=SOURCE, target=TARGET,
        steps=[
            ReadStep(id="r", source_table="t", alias="raw"),
            WriteStep(id="w", input="r", target_table="main.x.out", mode="overwrite"),
        ],
    )
    assert render_utility_module(spec) is None


def test_adapt_python_code_rewrites_alteryx_calls() -> None:
    from convert.renderer import adapt_python_code

    code = (
        "from ayx import Alteryx\n"
        "import pandas as pd\n"
        'data = Alteryx.read("#1")\n'
        "df2 = data.dropna()\n"
        "Alteryx.write(df2, 1)\n"
    )
    adapted = adapt_python_code(code)
    assert "# [migrated] from ayx import Alteryx" in adapted
    assert "data = _input_pdf" in adapted
    assert "_output_pdf = df2" in adapted
    assert "Alteryx.write" not in adapted


def test_render_python_script_step_wraps_code() -> None:
    from convert.spec import PythonScriptStep

    spec = PipelineSpec(
        name="py_case", language="pyspark", source=SOURCE, target=TARGET,
        steps=[
            ReadStep(id="r", source_table="t", alias="raw"),
            PythonScriptStep(id="p", input="r", code='x = Alteryx.read("#1")\nAlteryx.write(x, 1)'),
            WriteStep(id="w", input="p", target_table="main.x.out", mode="overwrite"),
        ],
    )
    source = render_pyspark(spec)
    compile(source, "<generated>", "exec")
    assert "def _python_script_p(df):" in source
    assert "_input_pdf = df.toPandas()" in source
    assert "return spark.createDataFrame(_output_pdf)" in source
    assert "df_p = _python_script_p(df_r)" in source


def test_render_find_replace_and_append_fields() -> None:
    from convert.spec import AppendFieldsStep, FindReplaceStep

    spec = PipelineSpec(
        name="fr_case", language="pyspark", source=SOURCE, target=TARGET,
        steps=[
            ReadStep(id="data", source_table="t1", alias="data"),
            ReadStep(id="lookup", source_table="t2", alias="lookup"),
            FindReplaceStep(
                id="fr", left="data", right="lookup",
                find_column="Child", search_column="Flat Component", replace_column="Pcode",
            ),
            AppendFieldsStep(id="ap", target="fr", source="lookup"),
            WriteStep(id="w", input="ap", target_table="main.x.out", mode="overwrite"),
        ],
    )
    source = render_pyspark(spec)
    compile(source, "<generated>", "exec")
    assert '"__fr_key"' in source
    assert 'F.coalesce(F.col("__fr_val"), F.col("Child"))' in source
    assert "df_ap = df_fr.crossJoin(df_lookup)" in source


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


def test_render_sdp_emits_medallion_layers() -> None:
    source = render_sdp(_full_spec())
    compile(source, "<generated>", "exec")
    assert "from pyspark import pipelines as dp" in source
    # bronze: one raw landing table per read
    assert '@dp.table(name="bronze_sales"' in source
    assert '@dp.table(name="bronze_customers"' in source
    # silver: the transform chain reads from bronze
    assert '@dp.table(name="silver_sales_summary"' in source
    assert 'spark.read.table("bronze_sales")' in source
    # gold: aggregation happens here, not in silver
    assert '@dp.table(name="gold_sales_summary"' in source
    gold_body = source.split('gold_sales_summary"')[1]
    assert "groupBy" in gold_body
    silver_body = source.split('silver_sales_summary"')[1].split("@dp.table")[0]
    assert "groupBy" not in silver_body
    # the pipeline runtime owns the write
    assert ".write.mode(" not in source
    assert "dlt" not in source


def test_render_sdp_without_aggregates_gold_is_passthrough() -> None:
    spec = PipelineSpec(
        name="plain", language="pyspark", source=SOURCE, target=TARGET,
        steps=[
            ReadStep(id="r", source_table="t", alias="raw"),
            FilterStep(id="f", input="r", condition="[a] > 0"),
            WriteStep(id="w", input="f", target_table="main.x.plain_out", mode="overwrite"),
        ],
    )
    source = render_sdp(spec)
    compile(source, "<generated>", "exec")
    assert '@dp.table(name="gold_plain_out"' in source
    assert 'return spark.read.table("silver_plain")' in source


def test_render_sdp_requires_a_write_step() -> None:
    spec = PipelineSpec(
        name="no_write", language="pyspark", source=SOURCE, target=TARGET,
        steps=[ReadStep(id="r", source_table="t", alias="t")],
    )
    with pytest.raises(ValueError, match="write"):
        render_sdp(spec)


def _branching_spec() -> PipelineSpec:
    """A gold layer with two divergent output branches, a gold step that
    reads a bronze source directly (bypassing silver), a Union referencing a
    non-terminal silver-chain id, and one step no write ever consumes.
    """
    return PipelineSpec(
        name="branching", language="pyspark", source=SOURCE, target=TARGET,
        steps=[
            ReadStep(id="ra", source_table="legacy.a", alias="a"),
            ReadStep(id="rb", source_table="legacy.b", alias="b"),
            FilterStep(id="fa1", input="ra", condition="[x] > 0"),
            WithColumnsStep(
                id="fa2", input="fa1", columns=[ComputedColumn(name="y", expression="[x] * 2")]
            ),
            AggregateStep(
                id="agg1", input="fa2", group_by=["y"],
                aggregations=[Aggregation(column="y", func="sum", alias="total")],
            ),
            JoinStep(
                id="joinb", left="agg1", right="rb",
                left_keys=["y"], right_keys=["y"], how="inner", use_function=None,
            ),
            UnionStep(id="unioned", inputs=["joinb", "fa1"]),
            FilterStep(id="branch_a_only", input="unioned", condition="[total] > 0"),
            WithColumnsStep(
                id="branch_b_only", input="unioned",
                columns=[ComputedColumn(name="z", expression="[total] + 1")],
            ),
            FilterStep(id="orphan", input="unioned", condition="[total] < 100"),
            WriteStep(id="w1", input="branch_a_only", target_table="main.x.out_a"),
            WriteStep(id="w2", input="branch_b_only", target_table="main.x.out_b"),
        ],
    )


def test_render_sdp_gold_step_reads_own_bronze_not_silver() -> None:
    source = render_sdp(_branching_spec())
    compile(source, "<generated>", "exec")
    gold_a = source.split('gold_out_a"')[1].split("@dp.table")[0]
    # `joinb` reads `rb` directly (right="rb"), bypassing silver entirely --
    # it must resolve to rb's own bronze table, not silver's unrelated output.
    assert 'df_rb = spark.read.table("bronze_b")' in gold_a
    assert 'df_rb = spark.read.table("silver_branching")' not in source


def test_render_sdp_flags_nonterminal_silver_reference_for_review() -> None:
    source = render_sdp(_branching_spec())
    gold_a = source.split('gold_out_a"')[1].split("@dp.table")[0]
    # `unioned` references `fa1` directly, but silver's table only exposes
    # `fa2` (silver_last) -- the substitution is an approximation and must
    # be visibly flagged, while the true terminal id is not.
    assert 'df_fa1 = spark.read.table("silver_branching")  # REVIEW' in gold_a
    assert 'df_fa2 = spark.read.table("silver_branching")' in gold_a
    fa2_line = next(line for line in gold_a.splitlines() if "df_fa2 = spark.read.table" in line)
    assert "REVIEW" not in fa2_line


def test_render_sdp_per_write_slicing_avoids_cross_branch_duplication() -> None:
    source = render_sdp(_branching_spec())
    gold_a = source.split('gold_out_a"')[1].split("@dp.table")[0]
    gold_b = source.split('gold_out_b"')[1].split("@dp.table")[0]
    # Each gold table only computes its own branch, not its sibling's.
    assert "df_branch_a_only" in gold_a
    assert "df_branch_b_only" not in gold_a
    assert "df_branch_b_only" in gold_b
    assert "df_branch_a_only" not in gold_b
    # Both legitimately share the upstream join/union that feeds them.
    assert "unionByName" in gold_a
    assert "unionByName" in gold_b


def test_render_sdp_orphaned_gold_step_flagged_for_review() -> None:
    source = render_sdp(_branching_spec())
    assert "REVIEW" in source
    assert "'orphan'" in source
    assert "never reach a write step" in source
    # the orphaned step's own body must not silently appear in either output
    assert "df_orphan = " not in source


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


def _spec_with_special_cols() -> PipelineSpec:
    return PipelineSpec(
        name="spaced", language="pyspark", source=SOURCE, target=TARGET,
        steps=[
            ReadStep(id="r", source_table="legacy.src", alias="src"),
            SelectStep(
                id="s", input="r",
                columns=[
                    ColumnSelection(column="ProjectCube_Data[Redbox Customer]", rename="Redbox Customer"),
                    ColumnSelection(column="Amount"),
                ],
            ),
            WriteStep(id="w", input="s", target_table="main.dev.spaced_out", mode="overwrite"),
        ],
    )


def test_special_column_names_enable_delta_column_mapping_all_formats() -> None:
    spec = _spec_with_special_cols()
    sdp = render_sdp(spec)
    job = render_pyspark(spec)
    notebook = render_databricks_notebook(spec)
    compile(sdp, "<g>", "exec")
    compile(job, "<g>", "exec")
    # SDP: every @dp.table carries the column-mapping property (bronze+silver+gold)
    assert sdp.count('"delta.columnMapping.mode": "name"') == 3
    assert "table_properties=" in sdp
    # job / notebook: the write enables column mapping via writer options
    assert 'delta.columnMapping.mode", "name"' in job
    assert 'delta.minReaderVersion", "2"' in job
    assert 'delta.columnMapping.mode", "name"' in notebook


def test_clean_column_names_do_not_enable_column_mapping() -> None:
    # _full_spec uses only plain identifiers (Amount, Region, TotalSales...)
    sdp = render_sdp(_full_spec())
    job = render_pyspark(_full_spec())
    assert "columnMapping" not in sdp
    assert "columnMapping" not in job
    # the plain write is unchanged
    assert ".write.mode('overwrite').saveAsTable(" in job


def test_extended_summarize_aggregations_render() -> None:
    spec = PipelineSpec(
        name="agg2", language="pyspark", source=SOURCE, target=TARGET,
        steps=[
            ReadStep(id="r", source_table="t", alias="t"),
            AggregateStep(
                id="a", input="r", group_by=["region"],
                aggregations=[
                    Aggregation(column="amt", func="stddev", alias="sd"),
                    Aggregation(column="amt", func="countDistinct", alias="uniq"),
                    Aggregation(column="amt", func="first", alias="f"),
                ],
            ),
            WriteStep(id="w", input="a", target_table="main.x.agg2"),
        ],
    )
    source = render_pyspark(spec)
    compile(source, "<g>", "exec")
    assert 'F.stddev("amt").alias("sd")' in source
    assert 'F.countDistinct("amt").alias("uniq")' in source
    assert 'F.first("amt").alias("f")' in source


def test_select_renders_type_casts_preserving_names() -> None:
    spec = PipelineSpec(
        name="casts", language="pyspark", source=SOURCE, target=TARGET,
        steps=[
            ReadStep(id="r", source_table="t", alias="t"),
            SelectStep(
                id="s", input="r",
                columns=[
                    ColumnSelection(column="Amount", cast_type="double"),
                    ColumnSelection(column="Id", rename="CustId", cast_type="int"),
                    ColumnSelection(column="Name"),
                ],
            ),
            WriteStep(id="w", input="s", target_table="main.x.casts"),
        ],
    )
    source = render_pyspark(spec)
    compile(source, "<g>", "exec")
    # cast without rename keeps the original column name
    assert 'F.col("Amount").cast("double").alias("Amount")' in source
    # cast with rename keeps the rename
    assert 'F.col("Id").cast("int").alias("CustId")' in source
    # no cast -> untouched
    assert 'F.col("Name")' in source
    assert 'F.col("Name").cast' not in source
