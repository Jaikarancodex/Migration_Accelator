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
from convert.renderer import render_pyspark
from convert.spec import MedallionLayer, PipelineSpec, TargetRef
from deploy.dab import build_databricks_yml, default_bundle, single_target_bundle
from eval.schema import ColumnSchema, ColumnType, TableSchema
from eval.synthetic import generate_synthetic_rows
from ingest.alteryx.parser import parse_yxmd
from llm.client import AnthropicLLMClient
from llm.convert import SpecGenerationError, generate_pipeline_spec
from repo.graph import CyclicDependencyError, DependencyGraph
from repo.store import MigrationRepo

st.set_page_config(page_title="Migration Accelerator", layout="wide")

SAMPLE_FIXTURE = _ROOT / "tests" / "fixtures" / "alteryx" / "sales_summary.yxmd"
TARGET_DEFAULTS = load_yaml_config(_ROOT / "configs" / "target.yaml", TargetDefaultsConfig)
DEPLOY_DEFAULTS = load_yaml_config(_ROOT / "configs" / "deploy.yaml", DeployDefaultsConfig)


def _repo() -> MigrationRepo:
    if "repo_dir" not in st.session_state:
        st.session_state.repo_dir = tempfile.mkdtemp(prefix="migration_accelerator_")
    return MigrationRepo(st.session_state.repo_dir)


def _spec_to_yaml(spec: PipelineSpec) -> str:
    return yaml.safe_dump(spec.model_dump(by_alias=True, mode="json"), sort_keys=False)


def _ingest_files(uploaded_files: list[Any]) -> None:
    repo = _repo()
    for uploaded in uploaded_files:
        with tempfile.NamedTemporaryFile(suffix=".yxmd", delete=False) as tmp:
            tmp.write(uploaded.getvalue())
            tmp_path = tmp.name
        try:
            workflow = parse_yxmd(tmp_path)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user, not swallowed
            st.error(f"Failed to parse {uploaded.name}: {exc}")
            continue
        repo.write_workflow(workflow)
    st.session_state.pop("workflow_cache", None)


st.title("Migration Accelerator — Review Console")
st.caption(
    "Alteryx (.yxmd) -> validated YAML spec -> deterministic PySpark -> parity preview -> Databricks Asset Bundle."
)

with st.sidebar:
    st.header("1. Ingest")
    use_sample = st.checkbox("Use bundled sample workflow", value=True)
    uploaded_files = st.file_uploader("Upload .yxmd file(s)", type=["yxmd"], accept_multiple_files=True)

    if st.button("Parse & store", type="primary"):
        files_to_ingest = list(uploaded_files or [])
        if use_sample:
            repo = _repo()
            workflow = parse_yxmd(SAMPLE_FIXTURE)
            repo.write_workflow(workflow)
        if files_to_ingest:
            _ingest_files(files_to_ingest)
        st.success("Ingested.")

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

tab_repo, tab_convert, tab_code, tab_parity, tab_deploy = st.tabs(
    ["Repo & dependency graph", "Convert", "Generated code", "Parity preview", "Deploy"]
)

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
                st.caption(f"[{u.tool_id}] {u.plugin} — {u.reason}")

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
                    spec = naive_spec_from_workflow(workflow, target)
                st.session_state[spec_key] = _spec_to_yaml(spec)
            except SpecGenerationError as exc:
                st.error(str(exc))

        if spec_key in st.session_state:
            st.write("**Pipeline spec (YAML)** — review and edit before rendering:")
            edited = st.text_area("spec_yaml", value=st.session_state[spec_key], height=400, label_visibility="collapsed")
            st.session_state[spec_key] = edited

            if st.button("Validate", key=f"validate::{selected}"):
                try:
                    data = yaml.safe_load(edited)
                    validated_spec = PipelineSpec.model_validate(data)
                    st.session_state[f"validated_spec::{selected}"] = validated_spec
                    st.success("Spec is valid.")
                except (yaml.YAMLError, ValidationError) as exc:
                    st.error(f"Validation failed: {exc}")

with tab_code:
    if not object_names:
        st.info("Ingest and convert an object first.")
    else:
        selected = st.selectbox("Object", object_names, key="code_object")
        stored_spec = cast(
            "PipelineSpec | None", st.session_state.get(f"validated_spec::{selected}")
        )
        if stored_spec is None:
            st.info("Generate and validate a spec for this object in the Convert tab first.")
        else:
            code = render_pyspark(stored_spec)
            st.code(code, language="python")
            st.download_button("Download .py", data=code, file_name=f"{selected}.py", mime="text/x-python")

with tab_parity:
    st.subheader("Synthetic data preview")
    st.caption(
        "eval/parity.py compares source-vs-target row sets once both are produced by a real run. "
        "This environment has no Spark/Databricks runtime to execute the rendered pipeline against, "
        "so this tab only previews the synthetic input the harness would use — see README 'Known limitations'."
    )
    num_cols = st.number_input("Number of columns", min_value=1, max_value=10, value=3)
    schema_columns: list[ColumnSchema] = []
    type_options: list[ColumnType] = ["int", "float", "string", "bool", "date", "timestamp"]
    for i in range(int(num_cols)):
        c1, c2, c3 = st.columns(3)
        name = c1.text_input(f"Column {i + 1} name", value=f"col_{i + 1}", key=f"col_name_{i}")
        dtype = c2.selectbox(f"Column {i + 1} type", type_options, key=f"col_type_{i}")
        is_key = c3.checkbox("key", key=f"col_key_{i}")
        schema_columns.append(ColumnSchema(name=name, data_type=dtype, key=is_key))

    num_rows = st.slider("Rows to generate", min_value=5, max_value=200, value=20)
    if st.button("Generate synthetic rows"):
        schema = TableSchema(name="preview", columns=schema_columns)
        rows = generate_synthetic_rows(schema, num_rows)
        st.dataframe(rows, use_container_width=True)

with tab_deploy:
    st.subheader("Databricks Asset Bundle")
    if not object_names:
        st.info("Ingest an object first.")
    else:
        selected = st.selectbox("Object", object_names, key="deploy_object")
        bundle_name = st.text_input("Bundle name", value=DEPLOY_DEFAULTS.bundle_name)

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
                value="https://community.cloud.databricks.com",
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
                )
            else:
                bundle = single_target_bundle(
                    bundle_name=bundle_name,
                    pipeline_name=selected,
                    python_file=f"{selected}.py",
                    workspace_host=free_host,
                    catalog=DEPLOY_DEFAULTS.catalog,
                    schema=DEPLOY_DEFAULTS.schema_name,
                )
            yml_text = build_databricks_yml(bundle)
            st.code(yml_text, language="yaml")
            st.download_button("Download databricks.yml", data=yml_text, file_name="databricks.yml", mime="text/yaml")
            st.caption("This only generates the YAML — `databricks bundle deploy` is never invoked (see non-goals).")
