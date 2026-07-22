"""Generated utility functions for 'LS_Certifications_DEV_Load' (convert/renderer.py:render_utility_module).

Imported by the main pipeline file — not meant to be run directly.
DO NOT EDIT BY HAND — regenerate from the YAML spec instead.
"""

from pyspark.sql import functions as F  # noqa: N812

def cleanse_columns(df, columns=None, trim=False, collapse_whitespace=False,
                    remove_all_whitespace=False, nulls_to_blank=False,
                    numeric_nulls_to_zero=False, case=None):
    """Utility generated from an Alteryx Data Cleansing macro."""
    from pyspark.sql import types as _T
    numeric_types = (_T.IntegerType, _T.LongType, _T.FloatType, _T.DoubleType,
                     _T.DecimalType, _T.ShortType, _T.ByteType)
    string_cols = {f.name for f in df.schema.fields if isinstance(f.dataType, _T.StringType)}
    numeric_cols = {f.name for f in df.schema.fields if isinstance(f.dataType, numeric_types)}
    targets = list(columns) if columns is not None else list(df.columns)

    out = df
    for name in targets:
        if name not in string_cols:
            continue
        col = F.col(name)
        if trim:
            col = F.trim(col)
        if collapse_whitespace:
            col = F.regexp_replace(col, r"\s+", " ")
        if remove_all_whitespace:
            col = F.regexp_replace(col, r"\s", "")
        if case == "upper":
            col = F.upper(col)
        elif case == "lower":
            col = F.lower(col)
        elif case == "title":
            col = F.initcap(col)
        if nulls_to_blank:
            col = F.coalesce(col, F.lit(""))
        out = out.withColumn(name, col)
    if numeric_nulls_to_zero:
        subset = [name for name in targets if name in numeric_cols]
        if subset:
            out = out.fillna(0, subset=subset)
    return out

