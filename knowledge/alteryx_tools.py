"""Alteryx tool -> Databricks conversion knowledge base.

One entry per Alteryx Designer tool, describing what the tool does and the
equivalent Databricks/PySpark logic. This catalog is the accelerator's
"domain brain": it is injected into the LLM conversion prompt (so the model
converts from curated knowledge, not general recall) and surfaced in the
review app next to unsupported-tool warnings so engineers get concrete
manual-conversion guidance.

Compiled from Alteryx Designer documentation (help.alteryx.com tool list and
per-tool pages) covering the Favorites/Preparation/Join/Parse/Transform
categories that make up the vast majority of real workflows.
"""

from __future__ import annotations

from pydantic import BaseModel

from ingest.alteryx.ir import ToolType


class ToolMapping(BaseModel):
    """How one Alteryx tool maps onto Databricks/PySpark."""

    tool: str  # Alteryx Designer tool name
    plugin_suffix: str  # suffix of the .yxmd GuiSettings Plugin attribute
    category: str  # Alteryx palette category
    what_it_does: str
    databricks_logic: str  # concrete PySpark/SQL conversion guidance
    parser_supported: bool = False  # True when ingest/alteryx/parser.py handles it


CATALOG: list[ToolMapping] = [
    ToolMapping(
        tool="Input Data",
        plugin_suffix="DbFileInput.DbFileInput",
        category="In/Out",
        what_it_does="Reads data from files or database tables into the workflow.",
        databricks_logic='spark.table("catalog.schema.table") for tables; '
        'spark.read.format("csv"|"json"|...).load(path) for files landed in a Volume.',
        parser_supported=True,
    ),
    ToolMapping(
        tool="Output Data",
        plugin_suffix="DbFileOutput.DbFileOutput",
        category="In/Out",
        what_it_does="Writes the stream to a file or database table.",
        databricks_logic='df.write.mode("overwrite"|"append").saveAsTable("catalog.schema.table") '
        "into Unity Catalog; MERGE INTO for upsert semantics.",
        parser_supported=True,
    ),
    ToolMapping(
        tool="Select",
        plugin_suffix="AlteryxSelect.AlteryxSelect",
        category="Preparation",
        what_it_does="Keeps/drops columns, renames them, and changes data types.",
        databricks_logic='df.select(F.col("a").alias("b"), F.col("c").cast("double"), ...) — '
        "dropped fields are simply omitted from the select list.",
        parser_supported=True,
    ),
    ToolMapping(
        tool="Filter",
        plugin_suffix="Filter.Filter",
        category="Preparation",
        what_it_does="Splits rows into True/False streams from a boolean expression.",
        databricks_logic="df.filter(F.expr(condition)) for the True stream; "
        "df.filter(~F.expr(condition)) when the False output is also consumed downstream.",
        parser_supported=True,
    ),
    ToolMapping(
        tool="Formula",
        plugin_suffix="Formula.Formula",
        category="Preparation",
        what_it_does="Creates or updates columns from expressions.",
        databricks_logic='df.withColumn("name", F.expr(spark_sql_expression)) — Alteryx [Field] '
        "references become backtick-quoted identifiers; IIF(c,a,b) -> IF(c,a,b) / CASE WHEN.",
        parser_supported=True,
    ),
    ToolMapping(
        tool="Multi-Row Formula",
        plugin_suffix="MultiRowFormula.MultiRowFormula",
        category="Preparation",
        what_it_does="Row expressions referencing prior/subsequent rows (running totals, "
        "fill-down, row deltas).",
        databricks_logic="Window functions: F.lag/F.lead over Window.partitionBy(...).orderBy(...) "
        "for [Row-1:Field] references; F.sum(...).over(w.rowsBetween(Window.unboundedPreceding, 0)) "
        "for running totals; F.last(col, ignorenulls=True) over a window for fill-down.",
        parser_supported=True,
    ),
    ToolMapping(
        tool="Multi-Field Formula",
        plugin_suffix="MultiFieldFormula.MultiFieldFormula",
        category="Preparation",
        what_it_does="Applies one expression across many columns at once.",
        databricks_logic="df.select([F.expr(expression.replace('[_CurrentField_]', f'`{c}`'))"
        ".alias(c) if c in targets else F.col(c) for c in df.columns]) — expand the template "
        "expression per target column.",
        parser_supported=True,
    ),
    ToolMapping(
        tool="Sort",
        plugin_suffix="Sort.Sort",
        category="Preparation",
        what_it_does="Orders rows by one or more columns ascending/descending.",
        databricks_logic='df.orderBy(F.col("a").asc(), F.col("b").desc()) — note Spark sorts are '
        "only meaningful right before a write, limit, or window; intermediate sorts are dropped "
        "by the optimizer.",
        parser_supported=True,
    ),
    ToolMapping(
        tool="Sample",
        plugin_suffix="Sample.Sample",
        category="Preparation",
        what_it_does="Takes first/last/random N rows, optionally per group.",
        databricks_logic="df.limit(n) for first N; row_number() over "
        "Window.partitionBy(group).orderBy(...) <= n for per-group samples; "
        "df.sample(fraction) for random samples.",
    ),
    ToolMapping(
        tool="Unique",
        plugin_suffix="Unique.Unique",
        category="Preparation",
        what_it_does="Splits rows into unique (first occurrence per key) and duplicates.",
        databricks_logic="df.dropDuplicates([keys]) for the unique stream; the library function "
        "dedupe_by_key(df, keys, order_by) when 'which duplicate wins' matters.",
        parser_supported=True,
    ),
    ToolMapping(
        tool="Record ID",
        plugin_suffix="RecordID.RecordID",
        category="Preparation",
        what_it_does="Adds a sequential unique identifier column.",
        databricks_logic='df.withColumn("RecordID", F.row_number().over(Window.orderBy(...))) for '
        "true sequences; F.monotonically_increasing_id() when uniqueness (not density) is enough.",
        parser_supported=True,
    ),
    ToolMapping(
        tool="Data Cleansing",
        plugin_suffix="DataCleansePro.DataCleansePro",
        category="Preparation",
        what_it_does="Bulk-fixes nulls, whitespace, casing, and unwanted characters "
        "(both the classic Cleanse.yxmc macro and Data Cleanse Pro).",
        databricks_logic="Converted automatically into a generated cleanse_columns utility "
        "(F.trim / regexp_replace for whitespace, coalesce/fillna for nulls, "
        "upper/lower/initcap for casing) emitted into the artifact and called in the flow.",
        parser_supported=True,
    ),
    ToolMapping(
        tool="Join",
        plugin_suffix="Join.Join",
        category="Join",
        what_it_does="Joins two streams on keys; outputs J (matched), L, and R (unmatched) anchors.",
        databricks_logic="The J output is an inner join (library function safe_join, which is "
        "null-safe like Alteryx); L/R outputs are left_anti joins from each side when consumed.",
        parser_supported=True,
    ),
    ToolMapping(
        tool="Union",
        plugin_suffix="Union.Union",
        category="Join",
        what_it_does="Stacks two or more streams with similar schemas.",
        databricks_logic="df_a.unionByName(df_b, allowMissingColumns=True) chained across all "
        "inputs — unionByName (not union) so column alignment is by name, as in Alteryx's "
        "auto-config-by-name mode.",
        parser_supported=True,
    ),
    ToolMapping(
        tool="Append Fields",
        plugin_suffix="AppendFields.AppendFields",
        category="Join",
        what_it_does="Appends every source row's fields onto every target row (cartesian).",
        databricks_logic="Converted automatically to df_target.crossJoin(df_source) — usually "
        "the source side is a single-row stream of scalars/aggregates.",
        parser_supported=True,
    ),
    ToolMapping(
        tool="Find Replace",
        plugin_suffix="FindReplace.FindReplace",
        category="Join",
        what_it_does="Looks up values from a reference stream and replaces/appends fields.",
        databricks_logic="Converted automatically: a left join against the lookup dataframe "
        "plus F.coalesce(replacement, original). Substring (FindAny) mode renders as an "
        "exact-match join with a review note.",
        parser_supported=True,
    ),
    ToolMapping(
        tool="Text To Columns",
        plugin_suffix="TextToColumns.TextToColumns",
        category="Parse",
        what_it_does="Splits one delimited text column into multiple columns or rows.",
        databricks_logic='F.split(col, delimiter).getItem(i) per output column for split-to-columns; '
        "F.explode(F.split(col, delimiter)) for split-to-rows.",
    ),
    ToolMapping(
        tool="RegEx",
        plugin_suffix="RegEx.RegEx",
        category="Parse",
        what_it_does="Parses, matches, tokenizes, or replaces text with regular expressions.",
        databricks_logic="F.regexp_extract(col, pattern, group) for parse mode; "
        "F.regexp_replace for replace mode; col.rlike(pattern) for match mode; "
        "F.explode(F.split(...)) approximates tokenize-to-rows.",
    ),
    ToolMapping(
        tool="DateTime",
        plugin_suffix="DateTime.DateTime",
        category="Parse",
        what_it_does="Converts between date-time values and formatted strings.",
        databricks_logic="F.to_date/F.to_timestamp(col, fmt) for string->date; "
        "F.date_format(col, fmt) for date->string. Alteryx %Y-%m-%d style specifiers map to "
        "Spark's yyyy-MM-dd pattern letters.",
    ),
    ToolMapping(
        tool="Transpose",
        plugin_suffix="Transpose.Transpose",
        category="Transform",
        what_it_does="Pivots columns into Name/Value rows (wide -> long).",
        databricks_logic='The SQL stack() expression: df.selectExpr("key_cols", '
        '"stack(n, \'col1\', col1, \'col2\', col2, ...) as (Name, Value)") — or DataFrame.melt '
        "on recent Spark versions.",
    ),
    ToolMapping(
        tool="Cross Tab",
        plugin_suffix="CrossTab.CrossTab",
        category="Transform",
        what_it_does="Pivots rows into columns (long -> wide) with an aggregation.",
        databricks_logic="df.groupBy(keys).pivot(header_col).agg(F.sum/first/... (value_col)).",
    ),
    ToolMapping(
        tool="Summarize",
        plugin_suffix="Summarize.Summarize",
        category="Transform",
        what_it_does="Groups and aggregates (sum/count/min/max/avg/concat/first/last...).",
        databricks_logic="df.groupBy(group_cols).agg(F.sum/count/avg/min/max(...).alias(...)); "
        "string Concatenate -> F.concat_ws(sep, F.collect_list(col)).",
        parser_supported=True,
    ),
    ToolMapping(
        tool="Running Total",
        plugin_suffix="RunningTotal.RunningTotal",
        category="Transform",
        what_it_does="Cumulative sums per group over an ordered stream.",
        databricks_logic="F.sum(col).over(Window.partitionBy(group).orderBy(order)"
        ".rowsBetween(Window.unboundedPreceding, 0)).",
    ),
    ToolMapping(
        tool="Count Records",
        plugin_suffix="CountRecords.CountRecords",
        category="Transform",
        what_it_does="Outputs a single row with the stream's record count.",
        databricks_logic="df.agg(F.count(F.lit(1)).alias('Count')) — keep it as a dataframe so "
        "downstream steps can Append Fields / join it.",
    ),
    ToolMapping(
        tool="Python (Jupyter)",
        plugin_suffix="JupyterCode",
        category="Developer",
        what_it_does="Runs pandas/python code from an embedded Jupyter notebook.",
        databricks_logic="Converted automatically: the notebook's code cells are extracted and "
        "wrapped in a generated function (input.toPandas() in, spark.createDataFrame out) with "
        "Alteryx.read/write rewritten. Runs on the driver — review for large data volumes.",
        parser_supported=True,
    ),
    ToolMapping(
        tool="Directory",
        plugin_suffix="Directory.Directory",
        category="In/Out",
        what_it_does="Lists files in a folder as rows (name, path, size, dates).",
        databricks_logic="dbutils.fs.ls('/Volumes/...') collected into a dataframe, or "
        "spark.read with a glob path when the listing just feeds a reader.",
    ),
    ToolMapping(
        tool="Dynamic Input",
        plugin_suffix="DynamicInput.DynamicInput",
        category="Developer",
        what_it_does="Reads many files/queries driven by an incoming list of paths.",
        databricks_logic="spark.read.format(...).load(list_of_paths) — Spark readers accept "
        "multiple paths/globs natively, so the per-row read loop usually collapses into one read.",
    ),
    ToolMapping(
        tool="Sharepoint Input",
        plugin_suffix="SharepointInput.SharepointInput",
        category="Connector",
        what_it_does="Reads lists/files from Sharepoint.",
        databricks_logic="Land the Sharepoint data into a Volume or table first (Lakeflow "
        "connector, Azure Data Factory, or a scheduled copy), then spark.read from there; "
        "a todo_source_* placeholder table is generated for the read.",
    ),
    ToolMapping(
        tool="Macro Input",
        plugin_suffix="MacroInput.MacroInput",
        category="Interface",
        what_it_does="Placeholder input inside a .yxmc macro definition.",
        databricks_logic="Becomes the dataframe parameter of the generated utility function "
        "when the macro is registered and inlined.",
        parser_supported=True,
    ),
    ToolMapping(
        tool="Macro Output",
        plugin_suffix="MacroOutput.MacroOutput",
        category="Interface",
        what_it_does="Placeholder output inside a .yxmc macro definition.",
        databricks_logic="Becomes the return value of the generated utility function when "
        "the macro is registered and inlined.",
        parser_supported=True,
    ),
    ToolMapping(
        tool="Text Input",
        plugin_suffix="TextInput.TextInput",
        category="In/Out",
        what_it_does="Embeds a small hand-typed table directly in the workflow.",
        databricks_logic="spark.createDataFrame([...rows...], schema) inline, or promote the "
        "constants to a small seed table in Unity Catalog.",
        parser_supported=True,
    ),
]

_BY_SUFFIX: dict[str, ToolMapping] = {m.plugin_suffix: m for m in CATALOG}


def lookup_by_plugin(plugin: str) -> ToolMapping | None:
    """Match a .yxmd plugin attribute (e.g. AlteryxBasePluginsGui.Sort.Sort) to its mapping."""
    for suffix, mapping in _BY_SUFFIX.items():
        if plugin.endswith(suffix):
            return mapping
    return None


def mappings_for_tool_types(tool_types: set[ToolType]) -> list[ToolMapping]:
    """Mappings for the parser-supported tools present in a workflow."""
    wanted = {t.value for t in tool_types}
    type_by_suffix = {
        "DbFileInput.DbFileInput": "input",
        "TextInput.TextInput": "input",
        "DbFileOutput.DbFileOutput": "output",
        "AlteryxSelect.AlteryxSelect": "select",
        "Filter.Filter": "filter",
        "Formula.Formula": "formula",
        "Join.Join": "join",
        "Union.Union": "union",
        "Sort.Sort": "sort",
        "Unique.Unique": "unique",
        "RecordID.RecordID": "record_id",
        "Summarize.Summarize": "summarize",
    }
    return [m for m in CATALOG if type_by_suffix.get(m.plugin_suffix) in wanted]


def render_knowledge_for_prompt(workflow_tool_types: set[ToolType], unsupported_plugins: list[str]) -> str:
    """Render the knowledge-base section injected into the conversion prompt."""
    lines: list[str] = []
    for m in mappings_for_tool_types(workflow_tool_types):
        lines.append(f"- {m.tool} ({m.category}): {m.what_it_does}\n  Databricks: {m.databricks_logic}")
    for plugin in unsupported_plugins:
        unsupported_mapping = lookup_by_plugin(plugin)
        if unsupported_mapping is not None:
            lines.append(
                f"- {unsupported_mapping.tool} (UNSUPPORTED by parser — describe as a TODO, "
                f"do not convert): {unsupported_mapping.what_it_does}\n"
                f"  Manual Databricks conversion: {unsupported_mapping.databricks_logic}"
            )
    return "\n".join(lines)
