"""Review UI for the migration accelerator.

Walks an engineer through the same pipeline described in the README:
ingest -> migration repo / dependency graph -> LLM conversion (reviewable
and editable as YAML) -> deterministic PySpark render -> synthetic-data
preview -> Databricks Asset Bundle generation. Nothing here changes the
core library's behavior; this is a thin, human-in-the-loop front end over
it, matching the project's mission ("engineers review the LLM output
before it ships").

Run with: streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, cast

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st
import yaml
from pydantic import ValidationError

from app.offline_convert import naive_spec_from_workflow
from configs.loader import load_yaml_config
from configs.models import DeployDefaultsConfig, TargetDefaultsConfig
from convert.renderer import render_databricks_notebook, render_pyspark, render_sdp
from convert.spec import MedallionLayer, PipelineSpec, TargetRef
from deploy.dab import ArtifactFormat, build_databricks_yml, default_bundle, single_target_bundle
from deploy.dbsql import (
    SqlError,
    csv_to_table_statements,
    first_warehouse_id,
    parity_check,
    run_sql,
)
from deploy.export import deploy_bundle, export_bundle_from_spec, run_bundle_job
from ingest.alteryx.parser import parse_yxmd
from knowledge.alteryx_tools import lookup_by_plugin
from llm.client import AnthropicLLMClient
from llm.convert import SpecGenerationError, generate_pipeline_spec
from llm.recommend import (
    DeploymentRecommendation,
    RecommendationError,
    heuristic_recommendation,
    recommend_deployment_format,
)
from repo.graph import CyclicDependencyError, DependencyGraph
from repo.store import MigrationRepo

st.set_page_config(page_title="Migration Accelerator", layout="wide")

SAMPLE_FIXTURE = _ROOT / "tests" / "fixtures" / "alteryx" / "sales_summary.yxmd"
TARGET_DEFAULTS = load_yaml_config(_ROOT / "configs" / "target.yaml", TargetDefaultsConfig)
DEPLOY_DEFAULTS = load_yaml_config(_ROOT / "configs" / "deploy.yaml", DeployDefaultsConfig)


# Fixed project-local store (gitignored) so ingested workflows survive
# server restarts and page refreshes, unlike a per-session temp dir.
_REPO_DIR = _ROOT / "migration_repo_output"


def _repo() -> MigrationRepo:
    return MigrationRepo(_REPO_DIR)


def _spec_to_yaml(spec: PipelineSpec) -> str:
    return yaml.safe_dump(spec.model_dump(by_alias=True, mode="json"), sort_keys=False)


def _ingest_files(uploaded_files: list[Any]) -> list[str]:
    repo = _repo()
    ingested: list[str] = []
    for uploaded in uploaded_files:
        with tempfile.NamedTemporaryFile(suffix=".yxmd", delete=False) as tmp:
            tmp.write(uploaded.getvalue())
            tmp_path = tmp.name
        try:
            workflow = parse_yxmd(tmp_path)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user, not swallowed
            st.error(f"Failed to parse {uploaded.name}: {exc}")
            continue
        # Name the object after the uploaded file, not the temp copy it was
        # parsed from — otherwise every upload shows up as "tmpXXXX".
        original_name = Path(uploaded.name).stem
        workflow = workflow.model_copy(update={"name": original_name, "source_file": uploaded.name})
        repo.write_workflow(workflow)
        ingested.append(original_name)
    return ingested


st.title("Migration Accelerator — Review Console")
st.caption(
    "Alteryx (.yxmd) -> validated YAML spec -> deterministic PySpark -> parity preview -> Databricks Asset Bundle."
)

with st.sidebar:
    st.header("1. Ingest")
    use_sample = st.checkbox("Use bundled sample workflow", value=True)
    uploaded_files = st.file_uploader("Upload .yxmd file(s)", type=["yxmd"], accept_multiple_files=True)

    if st.button("Parse & store", type="primary"):
        ingested_names: list[str] = []
        if use_sample:
            repo = _repo()
            workflow = parse_yxmd(SAMPLE_FIXTURE)
            repo.write_workflow(workflow)
            ingested_names.append(workflow.name)
        if uploaded_files:
            ingested_names.extend(_ingest_files(list(uploaded_files)))
        if ingested_names:
            st.success(f"Ingested: {', '.join(ingested_names)}")
        else:
            st.warning("Nothing to ingest — upload a .yxmd or tick the sample checkbox.")

    st.divider()
    st.header("Macros (.yxmc)")
    st.caption(
        "Upload the macro files your workflows reference. Registered macros are "
        "converted into generated utility functions and called from the main pipeline."
    )
    macro_files = st.file_uploader(
        "Upload .yxmc macro(s)", type=["yxmc"], accept_multiple_files=True, key="macro_upload"
    )
    if st.button("Register macros") and macro_files:
        registered = []
        for mf in macro_files:
            with tempfile.NamedTemporaryFile(suffix=".yxmc", delete=False) as tmp:
                tmp.write(mf.getvalue())
                tmp_path = tmp.name
            try:
                macro_wf = parse_yxmd(tmp_path)
            except Exception as exc:  # noqa: BLE001 - surfaced to the user
                st.error(f"Failed to parse {mf.name}: {exc}")
                continue
            macro_wf = macro_wf.model_copy(
                update={"name": Path(mf.name).stem, "source_file": mf.name}
            )
            registered.append(_repo().write_macro(macro_wf))
        if registered:
            st.success(f"Registered macros: {', '.join(registered)}")
    known_macros = _repo().list_macro_names()
    if known_macros:
        st.caption("Registered: " + ", ".join(known_macros))

    st.divider()
    st.header("2. Target defaults")
    catalog = st.text_input("Catalog", value=TARGET_DEFAULTS.catalog)
    schema_name = st.text_input("Schema", value=TARGET_DEFAULTS.schema_name)
    layer_options = ["bronze", "silver", "gold"]
    layer = st.selectbox("Medallion layer", layer_options, index=layer_options.index(TARGET_DEFAULTS.layer))
    target = TargetRef(catalog=catalog, schema=schema_name, layer=cast(MedallionLayer, layer))

    st.divider()
    st.header("3. LLM backend")
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    backend = st.radio(
        "Conversion backend",
        ["Anthropic (needs ANTHROPIC_API_KEY)", "Offline demo (rule-based, no LLM)"],
        index=0 if has_key else 1,
    )
    if backend.startswith("Anthropic") and not has_key:
        st.warning("ANTHROPIC_API_KEY is not set in this environment — falls back to offline mode.")

repo = _repo()
object_names = repo.list_object_names()

FORMAT_LABELS: dict[ArtifactFormat, str] = {
    "job": "Job script (spark_python_task)",
    "notebook": "Notebook (notebook_task)",
    "sdp": "SDP / Declarative Pipeline (medallion)",
}
FORMAT_ORDER: list[ArtifactFormat] = ["job", "notebook", "sdp"]

tab_quick, tab_repo, tab_convert, tab_code, tab_parity, tab_deploy = st.tabs(
    [
        "Quick migrate",
        "Repo & dependency graph",
        "Convert",
        "Generated code",
        "Verify parity",
        "Deploy",
    ]
)

with tab_quick:
    st.markdown(
        "#### Migrate an Alteryx workflow to Databricks in one step\n"
        "1. Upload your `.yxmd` file &nbsp;→&nbsp; 2. Click **Migrate & deploy** &nbsp;→&nbsp; "
        "3. Open the job in your workspace.\n\n"
        "The app parses the workflow, converts it, picks the best deployment format, "
        "and deploys it. Use the other tabs only when you want to review or fine-tune."
    )
    quick_file = st.file_uploader("Alteryx workflow (.yxmd)", type=["yxmd"], key="quick_upload")
    qc1, qc2 = st.columns(2)
    quick_host = qc1.text_input(
        "Databricks workspace URL",
        value="https://dbc-922a9e09-b3e2.cloud.databricks.com",
        key="quick_host",
    )
    quick_env_token = os.environ.get("DATABRICKS_TOKEN", "")
    quick_token = quick_env_token or qc2.text_input(
        "Access token", type="password", key="quick_token",
        help="Databricks: Settings > Developer > Access tokens. Never stored.",
    )
    if quick_env_token:
        qc2.caption("Token found in environment — no need to paste one.")

    if st.button("Migrate & deploy", type="primary", key="quick_go"):
        if quick_file is None:
            st.error("Upload a .yxmd file first.")
        elif not quick_token:
            st.error("An access token is required to deploy.")
        else:
            progress = st.status("Migrating...", expanded=True)
            with progress:
                st.write("**1/4** Parsing the Alteryx workflow...")
                names = _ingest_files([quick_file])
                if not names:
                    progress.update(label="Parse failed", state="error")
                    st.stop()
                wf_name = names[0]
                workflow = _repo().read_workflow(wf_name)
                st.write(
                    f"Parsed **{wf_name}**: {len(workflow.nodes)} tools converted, "
                    f"{len(workflow.unsupported)} need manual attention."
                )

                st.write("**2/4** Converting to a Databricks pipeline spec...")
                quick_target = TargetRef(catalog="workspace", schema="default", layer="bronze")
                if has_key:
                    quick_spec = generate_pipeline_spec(AnthropicLLMClient(), workflow, quick_target)
                    rec = recommend_deployment_format(AnthropicLLMClient(), workflow)
                else:
                    quick_spec = naive_spec_from_workflow(
                        workflow, quick_target, macros=_repo().all_macros()
                    )
                    rec = heuristic_recommendation(workflow)
                st.write(f"Format: **{FORMAT_LABELS[rec.format]}** — {rec.rationale}")

                st.write("**3/4** Writing the asset bundle...")
                quick_safe = re.sub(r"\W+", "_", wf_name).strip("_").lower()
                quick_dir = _ROOT / "bundles" / quick_safe
                export_bundle_from_spec(
                    quick_spec, quick_dir, workspace_host=quick_host, artifact_format=rec.format
                )

                st.write(f"**4/4** Deploying to {quick_host} ...")
                ok, log = deploy_bundle(quick_dir, quick_host, quick_token)

            if ok:
                progress.update(label="Migration deployed!", state="complete")
                url_lines = [line.strip() for line in log.splitlines() if "URL:" in line]
                for line in url_lines:
                    st.success(f"Job created — {line}")
                st.session_state["quick_last_bundle"] = str(quick_dir)
                if workflow.unsupported:
                    with st.expander(
                        f"{len(workflow.unsupported)} tool(s) need manual follow-up"
                    ):
                        for u in workflow.unsupported:
                            m = lookup_by_plugin(u.plugin)
                            if m is not None:
                                st.caption(f"**{m.tool}** — {m.databricks_logic}")
                            else:
                                st.caption(
                                    f"[{u.tool_id}] {u.plugin} — likely an embedded macro or "
                                    "connector; a placeholder todo_source_* table was created "
                                    "where needed. Land that data, then re-run."
                                )
                st.caption(
                    "Before running the job, make sure its source tables exist in your "
                    "workspace (the generated code's spark.read/spark.table names)."
                )
            else:
                progress.update(label="Deploy failed", state="error")
                st.code(log)

    if st.session_state.get("quick_last_bundle") and st.button(
        "Run the deployed job now", key="quick_run"
    ):
        with st.spinner("Running job (waits for completion)..."):
            ok, log = run_bundle_job(
                st.session_state["quick_last_bundle"], quick_host, quick_token
            )
        (st.success if ok else st.error)(log[-1000:])

with tab_repo:
    st.subheader("Migration repo")
    if not object_names:
        st.info("No objects ingested yet — use the sidebar to parse a workflow.")
    else:
        metadatas = repo.list_metadata()
        st.dataframe(
            [
                {
                    "name": m.name,
                    "source_system": m.source_system,
                    "input_tables": ", ".join(m.input_tables),
                    "output_tables": ", ".join(m.output_tables),
                    "unsupported_tools": m.unsupported_tool_count,
                }
                for m in metadatas
            ],
            use_container_width=True,
        )
        try:
            graph = DependencyGraph(metadatas)
            order = graph.topological_order()
            st.write("**Topological conversion order** (dependencies first):")
            st.code(" -> ".join(order) or "(no objects)")
            if len(order) > 1:
                dot_lines = ["digraph {"]
                for name in order:
                    dot_lines.append(f'  "{name}";')
                for name in order:
                    for dep in graph.dependencies_of(name):
                        dot_lines.append(f'  "{dep}" -> "{name}";')
                dot_lines.append("}")
                st.graphviz_chart("\n".join(dot_lines))
        except CyclicDependencyError as exc:
            st.error(str(exc))

with tab_convert:
    if not object_names:
        st.info("Ingest an object first.")
    else:
        selected = st.selectbox("Object", object_names, key="convert_object")
        workflow = repo.read_workflow(selected)
        metadata = repo.read_metadata(selected)

        if metadata.unsupported_tool_count:
            st.warning(f"{metadata.unsupported_tool_count} tool(s) were not recognized and were skipped:")
            for u in workflow.unsupported:
                mapping = lookup_by_plugin(u.plugin)
                if mapping is not None:
                    st.caption(
                        f"[{u.tool_id}] **{mapping.tool}** ({mapping.category}) — {mapping.what_it_does} "
                        f"\n\nManual Databricks conversion: {mapping.databricks_logic}"
                    )
                else:
                    st.caption(
                        f"[{u.tool_id}] {u.plugin} — {u.reason}. If this is an embedded "
                        "macro, convert the macro's own workflow separately (or land its "
                        "output in the generated todo_source_* placeholder table)."
                    )

        with st.expander("Parsed workflow nodes (topological order)", expanded=False):
            for node in workflow.topological_order():
                st.text(f"[{node.tool_id}] {node.tool_type.value}  upstream={node.upstream_ids or 'none'}")

        spec_key = f"spec_yaml::{selected}"

        if st.button("Generate spec", key=f"generate::{selected}"):
            try:
                if backend.startswith("Anthropic") and has_key:
                    client = AnthropicLLMClient()
                    spec = generate_pipeline_spec(client, workflow, target)
                else:
                    if backend.startswith("Anthropic"):
                        st.toast("No API key — using offline rule-based conversion instead.")
                    spec = naive_spec_from_workflow(workflow, target, macros=_repo().all_macros())
                st.session_state[spec_key] = _spec_to_yaml(spec)
                # A freshly generated spec is already validated — store it so
                # the Generated code tab (format picker + render) works
                # immediately; re-validation is only needed after manual edits.
                st.session_state[f"validated_spec::{selected}"] = spec
            except SpecGenerationError as exc:
                st.error(str(exc))

        if spec_key in st.session_state:
            st.write("**Pipeline spec (YAML)** — review and edit before rendering:")
            edited = st.text_area("spec_yaml", value=st.session_state[spec_key], height=400, label_visibility="collapsed")
            st.session_state[spec_key] = edited

            if st.button("Validate edits", key=f"validate::{selected}"):
                try:
                    data = yaml.safe_load(edited)
                    validated_spec = PipelineSpec.model_validate(data)
                    st.session_state[f"validated_spec::{selected}"] = validated_spec
                    st.success("Spec is valid.")
                except (yaml.YAMLError, ValidationError) as exc:
                    st.error(f"Validation failed: {exc}")
            st.caption(
                "Spec is ready — open the **Generated code** tab to pick Job / Notebook / SDP "
                "(or ask the LLM to recommend one)."
            )

with tab_code:
    if not object_names:
        st.info("Ingest and convert an object first.")
    else:
        selected = st.selectbox("Object", object_names, key="code_object")
        stored_spec = cast(
            "PipelineSpec | None", st.session_state.get(f"validated_spec::{selected}")
        )
        if stored_spec is None:
            st.info(
                "No spec yet for this object — go to the **Convert** tab and click "
                "**Generate spec** first. The Job / Notebook / SDP options appear here after that."
            )
        else:
            rec_key = f"format_rec::{selected}"
            if st.button("Recommend format (LLM)", key=f"recommend::{selected}"):
                workflow = repo.read_workflow(selected)
                try:
                    if has_key:
                        rec = recommend_deployment_format(AnthropicLLMClient(), workflow)
                    else:
                        rec = heuristic_recommendation(workflow)
                        st.toast("No API key — using the rule-based recommendation instead.")
                    st.session_state[rec_key] = rec
                    st.session_state.pop(f"format_choice::{selected}", None)
                except RecommendationError as exc:
                    st.error(str(exc))

            stored_rec = cast("DeploymentRecommendation | None", st.session_state.get(rec_key))
            if stored_rec is not None:
                st.info(f"**Recommended: {FORMAT_LABELS[stored_rec.format]}** — {stored_rec.rationale}")

            default_index = FORMAT_ORDER.index(stored_rec.format) if stored_rec is not None else 0
            chosen_label = st.radio(
                "Output format",
                [FORMAT_LABELS[f] for f in FORMAT_ORDER],
                index=default_index,
                horizontal=True,
                key=f"format_choice::{selected}",
            )
            chosen_format = next(f for f, label in FORMAT_LABELS.items() if label == chosen_label)
            st.session_state[f"artifact_format::{selected}"] = chosen_format

            try:
                if chosen_format == "notebook":
                    code = render_databricks_notebook(stored_spec)
                elif chosen_format == "sdp":
                    code = render_sdp(stored_spec)
                else:
                    code = render_pyspark(stored_spec)
                st.code(code, language="python")
                st.download_button(
                    "Download .py", data=code, file_name=f"{selected}.py", mime="text/x-python"
                )
            except ValueError as exc:
                st.error(str(exc))

with tab_parity:
    st.subheader("Verify parity against the Alteryx run")
    st.markdown(
        "Prove the migration is correct: run the original Alteryx workflow once against a "
        "frozen source extract, export its output as CSV, and compare it row-for-row with "
        "the migrated pipeline's output table (`EXCEPT ALL` in both directions)."
    )
    pc1, pc2 = st.columns(2)
    parity_host = pc1.text_input(
        "Databricks workspace URL",
        value="https://dbc-922a9e09-b3e2.cloud.databricks.com",
        key="parity_host",
    )
    parity_env_token = os.environ.get("DATABRICKS_TOKEN", "")
    parity_token = parity_env_token or pc2.text_input(
        "Access token", type="password", key="parity_token"
    )
    if parity_env_token:
        pc2.caption("Token found in environment.")

    migrated_table = st.text_input(
        "Migrated output table", value="workspace.default.certifications_new"
    )
    expected_csv = st.file_uploader(
        "Alteryx output export (.csv)", type=["csv"], key="parity_csv",
        help="Export the original workflow's output from Alteryx as CSV and upload it here.",
    )
    expected_table = st.text_input(
        "...or an existing expected table",
        value="",
        help="Leave blank when uploading a CSV; the CSV is loaded into expected_<name>.",
    )
    ignore_raw = st.text_input(
        "Columns to exclude from the diff (comma-separated)",
        value="Load_Date",
        help="Non-deterministic columns: load timestamps, run ids, sequence columns "
        "whose order Alteryx and Spark assign differently.",
    )

    if st.button("Run parity check", type="primary"):
        if not parity_token:
            st.error("An access token is required.")
        else:
            try:
                with st.spinner("Finding SQL warehouse..."):
                    warehouse = first_warehouse_id(parity_host, parity_token)

                target_expected = expected_table.strip()
                if expected_csv is not None:
                    stem = re.sub(r"\W+", "_", Path(expected_csv.name).stem).strip("_").lower()
                    target_expected = f"workspace.default.expected_{stem}"
                    with st.spinner(f"Loading expected data into {target_expected}..."):
                        for stmt in csv_to_table_statements(expected_csv.getvalue(), target_expected):
                            run_sql(parity_host, parity_token, warehouse, stmt)
                if not target_expected:
                    st.error("Upload the Alteryx output CSV or name an expected table.")
                    st.stop()

                with st.spinner("Comparing tables (counts + EXCEPT ALL both directions)..."):
                    report = parity_check(
                        parity_host, parity_token, warehouse,
                        migrated_table.strip(), target_expected,
                        ignore_raw.split(","),
                    )
            except SqlError as exc:
                st.error(str(exc))
                st.stop()

            m, e = report["migrated_count"], report["expected_count"]
            if report["passed"]:
                st.success(
                    f"PARITY PASSED — {m} rows in both tables, no row-level differences "
                    f"across {len(report['compared_columns'])} compared columns."
                )
            else:
                st.error(f"PARITY FAILED — migrated: {m} rows, expected: {e} rows.")
                if report["extra_in_migrated"]["rows"]:
                    st.write("Rows in the migrated table but not in the Alteryx output (sample):")
                    st.dataframe(
                        [dict(zip(report["extra_in_migrated"]["columns"], r, strict=False))
                         for r in report["extra_in_migrated"]["rows"]],
                        use_container_width=True,
                    )
                if report["missing_from_migrated"]["rows"]:
                    st.write("Rows in the Alteryx output but missing from the migrated table (sample):")
                    st.dataframe(
                        [dict(zip(report["missing_from_migrated"]["columns"], r, strict=False))
                         for r in report["missing_from_migrated"]["rows"]],
                        use_container_width=True,
                    )
            st.caption(
                "Compared columns: " + ", ".join(report["compared_columns"]) +
                (" | Ignored: " + ", ".join(report["ignored_columns"])
                 if report["ignored_columns"] else "")
            )

with tab_deploy:
    st.subheader("Databricks Asset Bundle")
    if not object_names:
        st.info("Ingest an object first.")
    else:
        selected = st.selectbox("Object", object_names, key="deploy_object")
        # Bundle name is the deployment-state key: default to one per workflow
        # so deploys of different workflows never replace each other.
        default_bundle_name = re.sub(r"\W+", "_", selected).strip("_").lower()
        bundle_name = st.text_input("Bundle name", value=default_bundle_name)

        artifact_format = cast(
            ArtifactFormat, st.session_state.get(f"artifact_format::{selected}", "job")
        )
        st.caption(
            f"Deploying as: **{FORMAT_LABELS[artifact_format]}** "
            "(chosen in the Generated code tab)."
        )

        deploy_style = st.radio(
            "Target workspace",
            ["Azure Databricks (dev / staging / prod)", "Databricks Free Edition (single workspace)"],
        )

        if deploy_style.startswith("Azure"):
            dev_host = st.text_input("Dev host", value=DEPLOY_DEFAULTS.dev_host)
            staging_host = st.text_input("Staging host", value=DEPLOY_DEFAULTS.staging_host)
            prod_host = st.text_input("Prod host", value=DEPLOY_DEFAULTS.prod_host)
        else:
            free_host = st.text_input(
                "Workspace host",
                value="https://dbc-922a9e09-b3e2.cloud.databricks.com",
                help="Your Databricks Free Edition workspace URL.",
            )

        if st.button("Generate databricks.yml"):
            if deploy_style.startswith("Azure"):
                bundle = default_bundle(
                    bundle_name=bundle_name,
                    pipeline_name=selected,
                    python_file=f"{selected}.py",
                    dev_host=dev_host,
                    staging_host=staging_host,
                    prod_host=prod_host,
                    catalog=DEPLOY_DEFAULTS.catalog,
                    schema=DEPLOY_DEFAULTS.schema_name,
                    artifact_format=artifact_format,
                )
            else:
                bundle = single_target_bundle(
                    bundle_name=bundle_name,
                    pipeline_name=selected,
                    python_file=f"{selected}.py",
                    workspace_host=free_host,
                    catalog=DEPLOY_DEFAULTS.catalog,
                    schema=DEPLOY_DEFAULTS.schema_name,
                    artifact_format=artifact_format,
                )
            yml_text = build_databricks_yml(bundle)
            st.code(yml_text, language="yaml")
            st.download_button("Download databricks.yml", data=yml_text, file_name="databricks.yml", mime="text/yaml")

        st.divider()
        st.subheader("One-click migrate & deploy")
        deploy_spec = cast(
            "PipelineSpec | None", st.session_state.get(f"validated_spec::{selected}")
        )
        if deploy_spec is None:
            st.info("Generate a spec in the Convert tab first — then this deploys it in one click.")
        else:
            deploy_host = free_host if deploy_style.startswith("Databricks Free") else dev_host
            env_token = os.environ.get("DATABRICKS_TOKEN", "")
            token = env_token or st.text_input(
                "Databricks access token",
                type="password",
                help="Generate one under Settings > Developer > Access tokens. "
                "Used only for this deploy; never stored.",
            )
            if env_token:
                st.caption("Using the DATABRICKS_TOKEN from the environment.")

            if st.button("Migrate & deploy to Databricks", type="primary"):
                if not token:
                    st.error("A Databricks access token is required.")
                else:
                    safe_name = re.sub(r"\W+", "_", selected).strip("_").lower()
                    bundle_dir = _ROOT / "bundles" / safe_name
                    with st.spinner("Rendering artifact and writing bundle..."):
                        export_bundle_from_spec(
                            deploy_spec,
                            bundle_dir,
                            workspace_host=deploy_host,
                            artifact_format=artifact_format,
                            bundle_name=bundle_name,
                        )
                    with st.spinner(f"Deploying to {deploy_host} ..."):
                        ok, log = deploy_bundle(bundle_dir, deploy_host, token)
                    st.code(log)
                    if ok:
                        st.success(
                            f"Deployed. Bundle written to `bundles/{safe_name}/` — commit it to git "
                            "to keep the workflow versioned."
                        )
                    else:
                        st.error("Deploy failed — see the CLI output above.")

            if st.button("Run the deployed job now"):
                if not token:
                    st.error("A Databricks access token is required.")
                else:
                    safe_name = re.sub(r"\W+", "_", selected).strip("_").lower()
                    bundle_dir = _ROOT / "bundles" / safe_name
                    if not (bundle_dir / "databricks.yml").exists():
                        st.error("Deploy first — no bundle found for this object.")
                    else:
                        with st.spinner("Running job (waits for completion)..."):
                            ok, log = run_bundle_job(bundle_dir, deploy_host, token)
                        st.code(log)
                        (st.success if ok else st.error)(
                            "Job succeeded." if ok else "Job failed — see output above."
                        )
