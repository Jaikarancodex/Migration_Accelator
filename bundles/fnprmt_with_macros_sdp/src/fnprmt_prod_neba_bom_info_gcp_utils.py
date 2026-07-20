"""Generated utility functions for 'FNPRMT-PROD-NEBA-BOM-INFO_GCP' (convert/renderer.py:render_utility_module).

Imported by the main pipeline file — not meant to be run directly.
DO NOT EDIT BY HAND — regenerate from the YAML spec instead.
"""

from pyspark.sql import functions as F  # noqa: N812

def macro_read_neba(df_macro_input):  # noqa: ANN001, ANN201
    """Utility generated from the Alteryx macro 'macro_read_neba'."""
    df_1 = spark.table("main.migration_dev.todo_source_1")  # alias: todo_source_1
    return df_1

def macro_countrecords(df_macro_input):  # noqa: ANN001, ANN201
    """Utility generated from the Alteryx macro 'macro_countrecords'."""
    df_1 = spark.table("main.migration_dev.1")  # alias: 1
    df_3 = df_macro_input.groupBy().agg(F.count("Field1").alias("Count"))
    df_4 = df_1.unionByName(df_3, allowMissingColumns=True)
    return df_4
