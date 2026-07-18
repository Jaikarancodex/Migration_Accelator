# Migration Accelerator

LLM-based accelerator for converting legacy data pipelines into Databricks
SQL/PySpark: parse a source pipeline into a clean intermediate
representation, have an LLM emit a validated YAML pipeline spec (never raw
code), deterministically render that spec into runnable PySpark, verify it
with a synthetic-data parity harness, and package it as a Databricks Asset
Bundle (DAB).

This session built one vertical slice end-to-end: **Alteryx (.yxmd) -> PySpark**.
Everything else in the target architecture is scaffolded with `TODO`
docstrings — see [Non-goals / stubs](#non-goals--stubs-this-session).

## Architecture

```
.yxmd file
   |  ingest/alteryx/parser.py  (parse, don't dump: never feeds raw XML to the LLM)
   v
Workflow IR (ingest/alteryx/ir.py)
   |  repo/store.py
   v
migration repo: <root>/<object_name>/{ir.json, metadata.json}
   |  repo/graph.py  (infers dependencies from input/output table overlap)
   v
DependencyGraph.topological_order()  -- convert leaf objects first
   |  llm/prompt_builder.py + llm/client.py
   v
LLM call (Anthropic, or MockLLMClient in tests) -> YAML text
   |  llm/convert.py  (validates against convert/spec.py, retries with error context)
   v
PipelineSpec (pydantic, source of truth)
   |  convert/renderer.py  (deterministic, not LLM-driven)
   v
runnable PySpark module
   |  eval/parity.py  (synthetic data -> row counts, checksums, key aggregates)
   v
pass/fail ParityReport
   |  deploy/dab.py
   v
databricks.yml  (dev/staging/prod targets, one job)
```

## What's implemented this session

1. **Repo scaffold** — this layout, `pyproject.toml`, ruff + mypy (strict) config,
   pre-commit config.
2. **`ingest/alteryx/`** — `.yxmd` XML parser (via `defusedxml`, not the stdlib
   parser, to avoid XXE) into a typed IR (`ir.py`). Handles input, select,
   filter, formula, join, summarize, output; unrecognized tools are logged to
   `Workflow.unsupported` rather than raising.
3. **`repo/`** — `MigrationRepo` writes/reads `ir.json` + `metadata.json` per
   object; `DependencyGraph` (networkx) infers edges from input/output table
   overlap and raises `CyclicDependencyError` on cycles.
4. **`llm/`** — `LLMClient` ABC with `AnthropicLLMClient` (lazy-imports
   `anthropic`, never imported by anything except this file) and
   `MockLLMClient` (used by every test — no API key needed to run the suite).
   `prompt_builder.py` renders a Jinja2 template that injects the function
   library signatures and the workflow's nodes in topological order.
   `convert.py` validates the LLM's YAML output against `PipelineSpec` and
   retries (feeding back the validation error) up to `max_retries` times.
5. **`convert/`** — `spec.py` (the YAML pydantic model, discriminated union
   over step `op`), `router.py` (procedural sources -> PySpark, set-based SQL
   sources -> SQL), `renderer.py` (deterministic YAML -> PySpark, the only
   place that emits code), `expr.py` (`[Field]` -> Spark SQL identifier
   translation).
6. **`functions/pyspark_lib/common.py`** — four reusable functions
   (`add_audit_columns`, `dedupe_by_key`, `safe_join`, `standardize_column_names`)
   plus `functions/registry.py`, which describes their signatures for prompt
   injection without importing `pyspark` at prompt-build time.
7. **`eval/`** — `synthetic.py` (seeded synthetic row generation from a
   `TableSchema`) and `parity.py` (row-count, order-independent column
   checksum, and key-aggregate comparison -> `ParityReport`).
8. **`deploy/`** — `dab.py` generates a minimal `databricks.yml` (one job,
   dev/staging/prod targets) from a `DABBundle` model.
9. **Tests** — 60 passing (`pytest`), all against `MockLLMClient` /
   in-memory data, no network or API key required. 4 more (real-Spark
   `functions/pyspark_lib` tests) skip cleanly where no JVM is available (see
   [Known limitations](#known-limitations)) and run for real in an
   environment with Java.

Run everything:

```bash
python -m venv .venv
. .venv/Scripts/activate        # Windows Git Bash; use .venv\Scripts\Activate.ps1 in PowerShell
pip install -e ".[dev]"
pip install -e ".[spark]"       # optional: needed for the real-Spark function tests
pytest                          # 60 passed, 4 skipped without Java
ruff check .                    # clean
mypy .                          # clean (strict mode)
```

## Assumptions & defaults

No sample `.yxmd` file, target catalog/schema naming, or Databricks workspace
was provided for this session. Defaults chosen, all overridable:

- **Sample workflow**: `tests/fixtures/alteryx/sales_summary.yxmd` is a
  hand-written, synthetic workflow (two inputs -> select -> filter -> formula
  -> join -> summarize -> output, plus one deliberately-unsupported RegEx
  tool) built from the general Alteryx `AlteryxBasePluginsGui` XML schema. It
  is **not** exported from a real Alteryx instance — real-world `.yxmd` files
  vary by Alteryx version and may use configuration shapes the parser doesn't
  yet recognize (they'll land in `Workflow.unsupported` rather than crashing
  the parse).
- **Catalog/schema naming**: `configs/target.yaml` defaults to
  `catalog: main`, `schema_name: migration_dev`, `layer: bronze`, loaded via
  `configs/loader.py` into `configs/models.py:TargetDefaultsConfig`. Override
  per-object by constructing a different `TargetRef`.
- **Databricks workspace hosts**: `configs/deploy.yaml` has placeholder
  `https://adb-000000000000000.0.azuredatabricks.net` hosts for dev/staging/prod.
  Replace before actually running `databricks bundle deploy` (not done this
  session — see non-goals).
- **LLM model**: `claude-sonnet-5`, set in `llm/client.py:DEFAULT_MODEL`,
  overridable via the `model=` constructor arg on `AnthropicLLMClient`.
- **Dependency graph library**: `networkx`, chosen over hand-rolled DAG code
  since topological sort / cycle detection are exactly what it's for.

## Known limitations

- **Alteryx formula/filter expression translation is intentionally minimal**
  (`convert/expr.py`): `[Field]` references become backtick-quoted Spark SQL
  identifiers, and the rest of the expression is passed through to Spark's
  SQL parser via `F.expr(...)`. This works for arithmetic/comparison/boolean
  expressions but does **not** translate Alteryx-specific functions with no
  direct Spark SQL equivalent (`IIF`, `DateTimeAdd`, etc.) — those need a
  mapping table or LLM-assisted rewriting, which is future work.
- **No Java runtime in this dev environment**, so `pyspark`'s local mode
  can't start a `SparkSession`. `functions/pyspark_lib/common.py` is real,
  type-checked PySpark code and `eval/parity.py` is deliberately
  engine-agnostic (compares plain `list[dict]` rows, not Spark DataFrames
  directly) so the parity harness itself needs no Spark/Java to test. The 4
  tests in `tests/functions/` that do exercise real DataFrame behavior skip
  automatically via `pytest.importorskip` / a try/except around
  `SparkSession.builder...getOrCreate()` (`tests/functions/conftest.py`) and
  will run for real in any environment with Java installed (e.g. CI, an
  actual Databricks cluster).
- **`WriteStep(mode="merge")`** renders a `# TODO` comment instead of a
  `MERGE INTO` statement — Spark doesn't have a `.write.mode("merge")`
  equivalent to `overwrite`/`append`; a real MERGE needs target-table-aware
  SQL generation, deferred.
- **SQL-dialect conversion path is routed but not rendered.**
  `convert/router.py` correctly routes Teradata/Oracle/Synapse/etc. sources
  to `"sql"`, but there is no SQL renderer yet (`functions/sql_lib/` is a
  placeholder package) — only Alteryx -> PySpark is implemented.

## Non-goals / stubs this session

Per the session brief, these are stubbed with `TODO` docstrings, not built:

- Other source dialects (Synapse/Teradata/Oracle/Redshift/BigQuery SQL,
  Pentaho, MySQL/PostgreSQL stored procs) — only Alteryx is parsed.
- RAG retriever internals for few-shot example selection.
- Fine-tuning.
- `eval/static_checks.py` — optimization/static-analysis gate (b).
- `eval/medallion.py` — medallion compliance gate (c). Only gate (a), parity,
  is implemented.
- `feedback/store.py` — (source, generated, human-corrected, eval-result)
  triple logging.
- CI pipeline.
- Live Databricks deployment (`databricks bundle deploy` is never invoked;
  `deploy/dab.py` only generates the YAML).

## Repo structure

```
migration-accelerator/
  ingest/alteryx/    parser.py, ir.py — .yxmd -> Workflow IR
  repo/              store.py, graph.py, metadata.py — the migration repo layer
  llm/               client.py, prompt_builder.py, convert.py, prompts/*.j2
  convert/           spec.py, router.py, renderer.py, expr.py
  functions/         pyspark_lib/common.py, sql_lib/ (placeholder), registry.py
  eval/              schema.py, synthetic.py, parity.py, static_checks.py (stub), medallion.py (stub)
  deploy/            models.py, dab.py
  feedback/          store.py (stub)
  configs/           loader.py, models.py, target.yaml, deploy.yaml
  tests/             mirrors the package layout; tests/fixtures/alteryx has the sample .yxmd
```
