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

import difflib
import os
import re
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st
import yaml
from pydantic import ValidationError

from app import repo_cache
from app.flow_ui import (
    ICONS,
    Status,
    hero_html,
    pipeline_flow_html,
    stepper_html,
    workflow_canvas_html,
)
from app.offline_convert import naive_spec_from_workflow
from configs.loader import load_yaml_config
from configs.models import DeployDefaultsConfig, TargetDefaultsConfig
from convert.io_map import (
    apply_source_overrides,
    spec_io,
    workflow_sources,
    workflow_targets,
)
from convert.renderer import (
    render_databricks_notebook,
    render_pyspark,
    render_sdp,
    render_utility_module,
    utils_module_name,
)
from convert.spec import MedallionLayer, PipelineSpec, TargetRef
from deploy.dab import ArtifactFormat, build_databricks_yml, default_bundle, single_target_bundle
from deploy.dbsql import (
    SqlError,
    csv_to_table_statements,
    first_warehouse_id,
    parity_check,
    run_sql,
    validation_report,
)
from deploy.export import deploy_bundle, export_bundle_from_spec, run_bundle_job
from deploy.gitops import GitError, commit_and_push, repo_info, set_remote
from feedback.store import (
    code_correction_count,
    correction_count,
    correction_counts_by_tool,
    deploy_error_counts_by_stage,
    find_code_corrections,
    find_similar_corrections,
    latest_code_correction,
    log_code_correction,
    log_conversion_triple,
    log_deploy_error,
    recent_deploy_errors,
    summarize_code_correction,
    summarize_correction,
)
from ingest.alteryx.ir import ToolType
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
from llm.repair import repair_pipeline_spec
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


st.markdown(
    hero_html(
        "Migration Accelerator",
        "Turn Alteryx workflows into deployed, version-controlled Databricks pipelines — "
        "guided, one step at a time.",
    ),
    unsafe_allow_html=True,
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
    st.header("2. Macros (.yxmc)")
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
    known_macros = repo_cache.list_macro_names(_repo())
    if known_macros:
        st.caption("Registered: " + ", ".join(known_macros))
        macro_to_delete = st.selectbox(
            "Remove a registered macro", known_macros, key="macro_delete_select"
        )
        if st.button("Delete macro", key="macro_delete_btn"):
            _repo().delete_macro(macro_to_delete)
            st.success(f"Deleted macro: {macro_to_delete}")
            st.rerun()

    st.divider()
    st.header("3. Git repository")
    st.caption(
        "Version every deployed bundle in git. The bundle folder is committed and "
        "pushed here as part of Migrate & deploy."
    )
    _git_state = repo_info(_ROOT)
    if _git_state["is_repo"]:
        st.caption(
            f"Current: `{_git_state['remote_url'] or '(no remote)'}` on branch "
            f"`{_git_state['branch']}`"
        )
    else:
        st.caption("This project folder is not a git repository yet.")
    git_enabled = st.checkbox("Commit & push bundle on deploy", value=bool(_git_state["is_repo"]))
    git_remote_url = st.text_input(
        "Repository URL (leave blank to keep current remote)",
        value="",
        placeholder="https://github.com/you/your-repo.git",
    )
    git_branch = st.text_input(
        "Branch", value=str(_git_state["branch"] or "main")
    )
    git_token = st.text_input(
        "GitHub token (optional, for push auth)", type="password",
        help="Used only for a single push, never stored. Leave blank to use your "
        "existing git credentials.",
    )

    st.divider()
    st.header("4. Target defaults")
    catalog = st.text_input("Catalog", value=TARGET_DEFAULTS.catalog)
    schema_name = st.text_input("Schema", value=TARGET_DEFAULTS.schema_name)
    layer_options = ["bronze", "silver", "gold"]
    layer = st.selectbox("Medallion layer", layer_options, index=layer_options.index(TARGET_DEFAULTS.layer))
    target = TargetRef(catalog=catalog, schema=schema_name, layer=cast(MedallionLayer, layer))

    st.divider()
    st.header("5. LLM backend")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        typed_key = st.text_input(
            "Anthropic API key",
            type="password",
            key="anthropic_key_input",
            help="Create one at console.anthropic.com under API keys. Held in this "
            "app's process memory for the session only — never written to disk, "
            "git, or logs.",
        )
        if typed_key.strip():
            os.environ["ANTHROPIC_API_KEY"] = typed_key.strip()
            st.toast("Anthropic key set for this session.")
        with st.expander("Set it permanently instead"):
            st.code('setx ANTHROPIC_API_KEY "sk-ant-..."', language="powershell")
            st.caption(
                "Run once in PowerShell, then restart the app — the key loads "
                "from the environment automatically and this field disappears."
            )
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    backend = st.radio(
        "Conversion backend",
        ["Anthropic Claude (LLM)", "Offline demo (rule-based, no LLM)"],
        index=0 if has_key else 1,
    )
    if backend.startswith("Anthropic") and not has_key:
        st.warning(
            "No Anthropic key yet — paste one above to enable LLM conversion, "
            "AI repair, and format recommendations. Falling back to offline mode."
        )

repo = _repo()
object_names = repo_cache.list_object_names(repo)

_n_macros = len(repo_cache.list_macro_names(repo))
_n_learned = correction_count()
_llm_pill = (
    '<span class="ma-pill" style="border-color:#22c55e;color:#22c55e">LLM: Anthropic connected</span>'
    if has_key
    else '<span class="ma-pill" style="border-color:#f59e0b;color:#f59e0b">'
    "LLM: offline rule-based — add a key in the sidebar</span>"
)
st.markdown(
    '<div style="display:flex;flex-wrap:wrap;gap:8px;margin:-2px 0 14px">'
    f'<span class="ma-pill">{len(object_names)} workflow(s) ingested</span>'
    f'<span class="ma-pill">{_n_macros} macro(s) registered</span>'
    f"{_llm_pill}"
    f'<span class="ma-pill">{_n_learned} correction(s) learned</span>'
    "</div>",
    unsafe_allow_html=True,
)

FORMAT_LABELS: dict[ArtifactFormat, str] = {
    "job": "Job script (spark_python_task)",
    "notebook": "Notebook (notebook_task)",
    "sdp": "SDP / Declarative Pipeline (medallion)",
}
FORMAT_ORDER: list[ArtifactFormat] = ["job", "notebook", "sdp"]

_ARTIFACT_RENDERERS = {
    "job": render_pyspark,
    "notebook": render_databricks_notebook,
    "sdp": render_sdp,
}


def _clear_code_overrides(workflow_name: str) -> None:
    """Drop manual code edits when the spec they were made against changes.

    A regenerated/edited/repaired spec renders different code; silently
    deploying stale hand-edits over it would undo the spec change.
    """
    for fmt in FORMAT_ORDER:
        st.session_state.pop(f"code_override::{workflow_name}::{fmt}", None)


def _render_artifact_preview(spec: PipelineSpec, artifact_format: ArtifactFormat, key: str) -> None:
    """Show the generated main file, plus its utility file as a second tab if one exists.

    Macro/cleanse helpers render into a separate importable module shared by
    all three formats — this mirrors that two-file output in the preview so
    it's not hidden until deploy.
    """
    main_code = _ARTIFACT_RENDERERS[artifact_format](spec)
    utility_code = render_utility_module(spec)
    override_key = f"code_override::{spec.name}::{artifact_format}"
    override = st.session_state.get(override_key)
    reapplied = False
    # No session override (e.g. fresh re-upload)? Re-apply a saved manual edit
    # from disk, but only when the freshly-generated code is byte-identical to
    # what that edit was made against — otherwise the renderer/spec changed
    # underneath it and replaying a stale hand-fix would be wrong.
    if override is None:
        saved = latest_code_correction(spec.name, artifact_format)
        if saved is not None and saved.generated_code.strip() == main_code.strip():
            override = saved.edited_code
            st.session_state[override_key] = override
            reapplied = True
        elif saved is not None:
            st.info(
                "You previously hand-edited this format's code, but the generated "
                "version has since changed — review and re-save if the fix still applies."
            )
    deployed_code = override or main_code

    # The renderer marks anything it could not translate with certainty
    # (unknown Alteryx functions, approximated silver-chain reads, name
    # collisions) with a REVIEW comment. Surface those up front — they are
    # the exact lines a reviewer must check before trusting the code.
    review_lines = [
        line.strip()
        for line in (deployed_code + "\n" + (utility_code or "")).splitlines()
        if "REVIEW" in line
    ]
    if review_lines:
        st.warning(
            f"{len(review_lines)} line(s) flagged REVIEW — verify these before deploying, "
            "or fix them in the manual editor below."
        )
        with st.expander("Show flagged lines"):
            st.code("\n".join(review_lines), language="python")

    if override:
        st.success(
            "Re-applied your saved manual edit for this workflow — deploys use it."
            if reapplied else
            "Manual code edits are active for this format — deploys use your "
            "edited version, not the regenerated one."
        )
        if st.button("Discard manual edits", key=f"{key}::discard::{artifact_format}"):
            st.session_state.pop(override_key, None)
            st.rerun()

    if utility_code is None:
        st.code(deployed_code, language="python")
    else:
        module = utils_module_name(spec)
        tab_main, tab_util = st.tabs(
            [f"{module.rsplit('_utils', 1)[0]} (main)", f"{module}.py"]
        )
        with tab_main:
            st.code(deployed_code, language="python")
        with tab_util:
            st.caption(
                "Cleanse/macro helpers — imported by the main file, deployed alongside it."
            )
            st.code(utility_code, language="python")

    with st.expander("Edit code manually"):
        st.caption(
            "Fix anything directly in the code — deploys for this format use your "
            "version, and the edit is logged. What the log does: on the **LLM "
            "backend**, re-converting this workflow shows the LLM your edit so it "
            "targets the fix; on the **offline converter** (deterministic), it "
            "can't change output — it's a signal in the Learning log that a "
            "recurring edit belongs in the renderer. Prefer editing the **spec** "
            "when you can: spec fixes train across workflows and survive "
            "regeneration, code edits are per-format and cleared when the spec "
            "regenerates."
        )
        edited_code = st.text_area(
            "generated code",
            value=deployed_code,
            height=380,
            key=f"{key}::editor::{artifact_format}",
            label_visibility="collapsed",
        )
        if st.button(
            "Save code edits", type="primary",
            key=f"{key}::save_code::{artifact_format}",
        ):
            try:
                compile(edited_code.replace("# Databricks notebook source", ""), "<edited>", "exec")
            except SyntaxError as exc:
                st.error(f"Not valid Python — fix before saving: {exc}")
            else:
                if edited_code.strip() == main_code.strip():
                    st.session_state.pop(override_key, None)
                    st.success("Code matches the generated version — nothing to record.")
                else:
                    st.session_state[override_key] = edited_code
                    log_code_correction(spec.name, artifact_format, main_code, edited_code)
                    st.success(
                        "Code edits saved and logged — deploys now use your version."
                    )
                st.rerun()

    dl1, dl2 = st.columns(2)
    stem = utils_module_name(spec).rsplit("_utils", 1)[0]
    dl1.download_button(
        "Download main .py", data=deployed_code,
        file_name=f"{stem}.py", mime="text/x-python",
        key=f"{key}::dl_main::{artifact_format}",
    )
    if utility_code is not None:
        dl2.download_button(
            "Download utils .py", data=utility_code,
            file_name=f"{utils_module_name(spec)}.py", mime="text/x-python",
            key=f"{key}::dl_util::{artifact_format}",
        )

    past_edits = find_code_corrections(spec.name, artifact_format)
    if past_edits:
        with st.expander(f"Past manual code edits for this workflow ({len(past_edits)})"):
            for rec in past_edits:
                st.markdown(f"`{rec.logged_at[:16]}` — {rec.artifact_format}")
                st.code(summarize_code_correction(rec), language="diff")


def _render_io_panel(
    spec: PipelineSpec,
    key: str,
    on_apply: Callable[[PipelineSpec], None],
) -> None:
    """Sources & targets of the spec, with an editable real path per source.

    Shows how many sources/targets the workflow has and their names, lets the
    user bind each source to a real Databricks table, and shows where each
    source is first consumed downstream — so the placeholder/derived names
    become real, propagating through every generated artifact.
    """
    sources, targets = spec_io(spec)
    st.markdown(
        f"**{len(sources)} source(s)** feed this pipeline, writing **{len(targets)} target(s)**."
    )

    if targets:
        st.markdown("**Targets (outputs)**")
        st.dataframe(
            [
                {
                    "target table": t.target_table,
                    "refresh": t.refresh_type,
                    "mode": t.mode,
                    "from step": t.fed_by,
                }
                for t in targets
            ],
            use_container_width=True, hide_index=True,
        )
        if any(t.refresh_type == "incremental" for t in targets):
            st.caption(
                "⚠ Incremental target(s) detected — the generated code marks these with a "
                "`# REVIEW` note: they render as full-refresh materialized views, and become "
                "streaming tables / apply_changes only if their source is append-only."
            )

    if not sources:
        st.caption("This spec has no read steps.")
        return

    st.markdown("**Sources (inputs)** — set the real table each one should read:")
    overrides: dict[str, str] = {}
    placeholder_count = 0
    for src in sources:
        is_placeholder = "todo_source_" in src.source_table
        placeholder_count += is_placeholder
        label = f"`{src.read_id}` ({src.alias})" + ("  ⚠ placeholder" if is_placeholder else "")
        overrides[src.read_id] = st.text_input(
            label, value=src.source_table, key=f"{key}::iosrc::{src.read_id}"
        )
        if src.first_consumer is not None:
            fc = src.first_consumer
            st.caption(
                f"↳ first consumed by step `{fc.step_id}` — {fc.detail}; "
                f"read by {len(src.consumers)} step(s) total."
            )
        else:
            st.caption("↳ not consumed by any step (unused source).")

    if placeholder_count:
        st.warning(
            f"{placeholder_count} source(s) are placeholders (todo_source_*) — a custom "
            "SQL query or an unsupported connector had no real table name. Point them at "
            "the real table you land that data in."
        )

    if st.button("Apply source paths", type="primary", key=f"{key}::apply_io"):
        updated = apply_source_overrides(spec, overrides)
        on_apply(updated)
        st.success("Source paths applied — the spec and every generated artifact now use them.")
        st.rerun()


def _offer_auto_repair(
    state_key: str,
    workflow_name: str,
    spec_yaml: str,
    stage: str,
    tool_types: list[str],
    apply_repair: Callable[[PipelineSpec, str], None],
) -> None:
    """After a Databricks failure, offer a one-click LLM repair of the spec.

    The failure output is stored under `state_key`; the repaired spec is
    shown as a diff for human review, and applying it both swaps the spec in
    and logs the broken->repaired pair as a correction so the retrieval loop
    learns from it.
    """
    error_text = st.session_state.get(state_key)
    if not error_text or not spec_yaml:
        return
    if not has_key:
        st.caption(
            "Set ANTHROPIC_API_KEY to enable one-click AI repair of this failure."
        )
        return
    if st.button("Attempt AI repair", key=f"repair_btn::{state_key}"):
        try:
            with st.spinner("Asking the LLM for a minimal fix..."):
                repaired = repair_pipeline_spec(
                    AnthropicLLMClient(), spec_yaml, error_text, workflow_name, stage=stage
                )
            st.session_state[f"{state_key}::repaired"] = _spec_to_yaml(repaired)
        except SpecGenerationError as exc:
            st.error(str(exc))
    repaired_yaml = st.session_state.get(f"{state_key}::repaired")
    if repaired_yaml:
        diff_lines = list(
            difflib.unified_diff(
                spec_yaml.splitlines(), repaired_yaml.splitlines(), lineterm="", n=2
            )
        )
        st.code("\n".join(diff_lines[:80]) or "(no changes proposed)", language="diff")
        if st.button("Apply repaired spec", type="primary", key=f"repair_apply::{state_key}"):
            repaired_spec = PipelineSpec.model_validate(yaml.safe_load(repaired_yaml))
            apply_repair(repaired_spec, repaired_yaml)
            _clear_code_overrides(repaired_spec.name)
            log_conversion_triple(
                workflow_name=workflow_name,
                tool_types=tool_types,
                generated_spec_yaml=spec_yaml,
                corrected_spec_yaml=repaired_yaml,
            )
            st.session_state.pop(state_key, None)
            st.session_state.pop(f"{state_key}::repaired", None)
            st.success("Repaired spec applied and logged — redeploy to verify the fix.")
            st.rerun()


def _push_bundle_to_git(bundle_dir: Path, workflow_name: str) -> None:
    """Commit and push a generated bundle dir if git integration is enabled."""
    if not git_enabled:
        return
    if git_remote_url.strip():
        try:
            set_remote(_ROOT, git_remote_url.strip())
        except GitError as exc:
            st.warning(f"Could not set git remote: {exc}")
            return
    try:
        rel = str(bundle_dir.relative_to(_ROOT))
    except ValueError:
        rel = str(bundle_dir)
    ok, log = commit_and_push(
        _ROOT,
        [rel],
        f"Add migrated Databricks bundle for {workflow_name}",
        git_branch.strip() or "main",
        token=git_token or None,
    )
    (st.success if ok else st.error)("Git: " + log)


tab_quick, tab_repo, tab_convert, tab_code, tab_parity, tab_deploy = st.tabs(
    [
        "Guided migration",
        "Repo & flow",
        "Convert (advanced)",
        "Generated code (advanced)",
        "Verify parity",
        "Deploy (advanced)",
    ]
)

with tab_quick:
    WIZ_LABELS = ["Upload", "Convert", "Format", "Deploy", "Verify"]
    st.session_state.setdefault("wiz_step", 0)

    def _wiz_reset_from(step: int) -> None:
        """Invalidate later-step results when an earlier step changes."""
        if step <= 0:
            st.session_state.pop("wiz_spec", None)
            st.session_state.pop("wiz_spec_yaml", None)
        if step <= 1:
            st.session_state.pop("wiz_format", None)
        if step <= 2:
            st.session_state.pop("wiz_deployed", None)

    _done = [
        st.session_state.get("wiz_wf") is not None,
        st.session_state.get("wiz_spec") is not None,
        st.session_state.get("wiz_format") is not None,
        bool(st.session_state.get("wiz_deployed")),
    ]
    _completed = 0
    for flag in _done:
        if flag:
            _completed += 1
        else:
            break
    step = int(st.session_state["wiz_step"])
    st.markdown(stepper_html(WIZ_LABELS, step, _completed), unsafe_allow_html=True)

    gate_ok = False

    # ---- Step 0: Upload -------------------------------------------------
    if step == 0:
        with st.container(border=True):
            st.markdown("#### Step 1 — Upload your Alteryx workflow")
            st.caption(
                "Upload a `.yxmd` file. Macros it references (`.yxmc`) can be registered "
                "in the sidebar so they convert into utilities."
            )
            wiz_file = st.file_uploader(".yxmd workflow", type=["yxmd"], key="wiz_upload")
            if st.button("Parse workflow", type="primary", key="wiz_parse"):
                if wiz_file is None:
                    st.error("Choose a .yxmd file first.")
                else:
                    names = _ingest_files([wiz_file])
                    if names:
                        st.session_state["wiz_wf"] = names[0]
                        _wiz_reset_from(0)
                        st.rerun()

            wf_name = st.session_state.get("wiz_wf")
            if wf_name:
                workflow = repo_cache.read_workflow(_repo(), wf_name)
                st.success(
                    f"Parsed **{wf_name}** — {len(workflow.nodes)} tools ready, "
                    f"{len(workflow.unsupported)} need manual follow-up."
                )
                w_sources = workflow_sources(workflow)
                w_targets = workflow_targets(workflow)
                sc, tc = st.columns(2)
                sc.metric("Sources (Input tools)", len(w_sources))
                tc.metric("Targets (Output tools)", len(w_targets))
                with st.expander("Source & target names"):
                    st.markdown("**Reads from:**")
                    st.markdown(
                        "\n".join(f"- `{tid}` → {name}" for tid, name in w_sources)
                        or "- (no Input tools — sources appear as placeholders after conversion)"
                    )
                    st.markdown("**Writes to:**")
                    st.markdown(
                        "\n".join(f"- `{tid}` → {name}" for tid, name in w_targets)
                        or "- (no Output tools)"
                    )
                    st.caption(
                        "You'll bind each source to a real Databricks table in the "
                        "Convert step, after conversion."
                    )
                st.components.v1.html(
                    workflow_canvas_html(workflow), height=640, scrolling=False
                )
                missing = [
                    m for m in workflow.referenced_macros()
                    if m not in repo_cache.list_macro_names(_repo())
                ]
                if missing:
                    st.warning(
                        "Unregistered macros: " + ", ".join(f"{m}.yxmc" for m in missing)
                        + " — upload them in the sidebar to convert them fully."
                    )
                gate_ok = True

    # ---- Step 1: Convert ------------------------------------------------
    elif step == 1:
        with st.container(border=True):
            st.markdown("#### Step 2 — Convert to a Databricks pipeline")
            workflow = repo_cache.read_workflow(_repo(), st.session_state["wiz_wf"])
            c1, c2 = st.columns(2)
            wiz_catalog = c1.text_input("Target catalog", value="workspace", key="wiz_catalog")
            wiz_schema = c2.text_input("Target schema", value="default", key="wiz_schema")
            st.caption(
                "The converter maps every tool to Spark logic and points reads/writes at "
                f"`{wiz_catalog}.{wiz_schema}.*`. "
                + ("Using the Anthropic LLM." if has_key else "Using the offline rule-based converter.")
            )
            if st.button("Convert workflow", type="primary", key="wiz_convert"):
                wiz_target = TargetRef(catalog=wiz_catalog, schema=wiz_schema, layer="bronze")
                if has_key:
                    spec = generate_pipeline_spec(AnthropicLLMClient(), workflow, wiz_target)
                else:
                    spec = naive_spec_from_workflow(
                        workflow, wiz_target, macros=repo_cache.all_macros(_repo())
                    )
                st.session_state["wiz_spec"] = spec
                spec_yaml = _spec_to_yaml(spec)
                st.session_state["wiz_spec_yaml"] = spec_yaml
                # kept untouched by edits below, so "Apply edits" can diff
                # the human correction against what was actually generated.
                st.session_state["wiz_spec_yaml_generated"] = spec_yaml
                st.session_state["wiz_tool_types"] = [n.tool_type.value for n in workflow.nodes]
                _clear_code_overrides(spec.name)
                _wiz_reset_from(1)
                st.rerun()

            stored_spec = cast("PipelineSpec | None", st.session_state.get("wiz_spec"))
            if stored_spec is not None:
                st.success(
                    f"Converted into {len(stored_spec.steps)} pipeline steps"
                    + (
                        f" and {len(stored_spec.macros)} macro utilit(ies)."
                        if stored_spec.macros else "."
                    )
                )
                with st.expander("Review / edit the pipeline spec (YAML)", expanded=False):
                    st.caption(
                        "Fix anything the converter got wrong, then save — your "
                        "correction is stored and shown to the LLM on future "
                        "conversions of similar workflows."
                    )
                    edited = st.text_area(
                        "spec", value=st.session_state["wiz_spec_yaml"], height=320,
                        label_visibility="collapsed",
                    )
                    if st.button("Save edits & train", type="primary", key="wiz_apply"):
                        try:
                            revalidated = PipelineSpec.model_validate(yaml.safe_load(edited))
                            st.session_state["wiz_spec"] = revalidated
                            st.session_state["wiz_spec_yaml"] = edited
                            _clear_code_overrides(revalidated.name)
                            log_conversion_triple(
                                workflow_name=st.session_state["wiz_wf"],
                                tool_types=st.session_state.get("wiz_tool_types", []),
                                generated_spec_yaml=st.session_state["wiz_spec_yaml_generated"],
                                corrected_spec_yaml=edited,
                            )
                            st.success("Spec updated and correction recorded.")
                        except (yaml.YAMLError, ValidationError) as exc:
                            st.error(f"Invalid spec: {exc}")

                with st.expander("Sources & targets — set real table paths", expanded=True):
                    def _wiz_apply_io(updated: PipelineSpec) -> None:
                        st.session_state["wiz_spec"] = updated
                        st.session_state["wiz_spec_yaml"] = _spec_to_yaml(updated)
                        _clear_code_overrides(updated.name)

                    _render_io_panel(stored_spec, "wiz", _wiz_apply_io)
                gate_ok = True

    # ---- Step 2: Format -------------------------------------------------
    elif step == 2:
        with st.container(border=True):
            st.markdown("#### Step 3 — Choose the deployment format")
            workflow = repo_cache.read_workflow(_repo(), st.session_state["wiz_wf"])
            spec = st.session_state["wiz_spec"]
            if st.button("Recommend a format", key="wiz_reco"):
                st.session_state["wiz_rec"] = (
                    recommend_deployment_format(AnthropicLLMClient(), workflow)
                    if has_key else heuristic_recommendation(workflow)
                )
            rec = st.session_state.get("wiz_rec")
            if rec is not None:
                st.info(f"Recommended: **{FORMAT_LABELS[rec.format]}** — {rec.rationale}")
            default_idx = FORMAT_ORDER.index(rec.format) if rec is not None else 0
            chosen_label = st.radio(
                "Format", [FORMAT_LABELS[f] for f in FORMAT_ORDER], index=default_idx,
                key="wiz_format_radio",
            )
            chosen = next(f for f, label in FORMAT_LABELS.items() if label == chosen_label)
            st.session_state["wiz_format"] = chosen
            try:
                with st.expander("Preview generated code"):
                    _render_artifact_preview(spec, chosen, key="wiz_preview")
            except ValueError as exc:
                st.error(str(exc))
            gate_ok = True

    # ---- Step 3: Deploy -------------------------------------------------
    elif step == 3:
        with st.container(border=True):
            st.markdown("#### Step 4 — Deploy to Databricks")
            spec = st.session_state["wiz_spec"]
            fmt = cast(ArtifactFormat, st.session_state["wiz_format"])
            wiz_host = st.text_input(
                "Workspace URL", value=DEPLOY_DEFAULTS.wizard_host,
                key="wiz_host",
            )
            wiz_env_token = os.environ.get("DATABRICKS_TOKEN", "")
            wiz_token = wiz_env_token or st.text_input(
                "Access token", type="password", key="wiz_token"
            )
            if wiz_env_token:
                st.caption("Using DATABRICKS_TOKEN from the environment.")
            if git_enabled:
                st.caption(f"Git: bundle will be committed & pushed to branch `{git_branch}`.")

            if st.button("Deploy now", type="primary", key="wiz_deploy"):
                if not wiz_token:
                    st.error("An access token is required.")
                else:
                    safe = re.sub(r"\W+", "_", st.session_state["wiz_wf"]).strip("_").lower()
                    bundle_dir = _ROOT / "bundles" / safe
                    dstages: list[Status] = ["pending", "pending"]
                    dph = st.empty()

                    def _dflow() -> None:
                        dph.markdown(
                            pipeline_flow_html([
                                ("Build bundle", ICONS["bundle"], dstages[0]),
                                ("Deploy", ICONS["deploy"], dstages[1]),
                            ]),
                            unsafe_allow_html=True,
                        )

                    dstages[0] = "running"
                    _dflow()
                    export_bundle_from_spec(
                        spec, bundle_dir, workspace_host=wiz_host, artifact_format=fmt,
                        main_code_override=st.session_state.get(
                            f"code_override::{spec.name}::{fmt}"
                        ),
                    )
                    dstages[0] = "done"
                    dstages[1] = "running"
                    _dflow()
                    ok, log = deploy_bundle(bundle_dir, wiz_host, wiz_token)
                    dstages[1] = "done" if ok else "error"
                    _dflow()
                    if ok:
                        st.session_state["wiz_deployed"] = True
                        st.session_state["wiz_bundle"] = str(bundle_dir)
                        st.session_state["wiz_host"] = wiz_host
                        for line in (line.strip() for line in log.splitlines() if "URL:" in line):
                            st.success(f"Job created — {line}")
                        _push_bundle_to_git(bundle_dir, st.session_state["wiz_wf"])
                    else:
                        log_deploy_error(st.session_state["wiz_wf"], "deploy", log)
                        st.session_state["wiz_deploy_error"] = log
                        st.error("Deploy failed — details below.")
                        st.code(log)

            def _wiz_apply_repair(repaired: PipelineSpec, repaired_yaml: str) -> None:
                st.session_state["wiz_spec"] = repaired
                st.session_state["wiz_spec_yaml"] = repaired_yaml

            _offer_auto_repair(
                "wiz_deploy_error",
                st.session_state.get("wiz_wf", "unknown"),
                st.session_state.get("wiz_spec_yaml", ""),
                "deploy",
                st.session_state.get("wiz_tool_types", []),
                _wiz_apply_repair,
            )

            if st.session_state.get("wiz_deployed"):
                st.success("Deployed. Continue to verify the result.")
                gate_ok = True

    # ---- Step 4: Verify -------------------------------------------------
    elif step == 4:
        with st.container(border=True):
            st.markdown("#### Step 5 — Verify the migration")
            spec = st.session_state["wiz_spec"]
            default_table = next(
                (s.target_table for s in spec.steps if s.op == "write"),
                f"{spec.target.catalog}.{spec.target.schema_}.output",
            )
            wiz_host = st.session_state.get("wiz_host", DEPLOY_DEFAULTS.wizard_host)
            wiz_token = os.environ.get("DATABRICKS_TOKEN", "") or st.text_input(
                "Access token", type="password", key="wiz_verify_token"
            )
            table = st.text_input("Migrated output table", value=default_table, key="wiz_vtable")
            st.caption(
                "Run the job first (Deploy → Databricks Jobs UI, or the button below), then "
                "validate. Full row-level parity against an Alteryx export lives in the "
                "**Verify parity** tab."
            )
            vc1, vc2 = st.columns(2)
            if vc1.button("Run the deployed job", key="wiz_runjob") and st.session_state.get("wiz_bundle"):
                if not wiz_token:
                    st.error("Access token required.")
                else:
                    with st.spinner("Running job..."):
                        rok, rlog = run_bundle_job(st.session_state["wiz_bundle"], wiz_host, wiz_token)
                    if not rok:
                        log_deploy_error(st.session_state.get("wiz_wf", "unknown"), "run", rlog)
                        st.session_state["wiz_run_error"] = rlog
                    else:
                        st.session_state.pop("wiz_run_error", None)
                    (st.success if rok else st.error)(rlog[-800:])

            def _wiz_apply_run_repair(repaired: PipelineSpec, repaired_yaml: str) -> None:
                st.session_state["wiz_spec"] = repaired
                st.session_state["wiz_spec_yaml"] = repaired_yaml
                st.session_state["wiz_deployed"] = False  # must redeploy the fix

            _offer_auto_repair(
                "wiz_run_error",
                st.session_state.get("wiz_wf", "unknown"),
                st.session_state.get("wiz_spec_yaml", ""),
                "run",
                st.session_state.get("wiz_tool_types", []),
                _wiz_apply_run_repair,
            )
            if vc2.button("Validate output table", type="primary", key="wiz_validate"):
                if not wiz_token:
                    st.error("Access token required.")
                else:
                    try:
                        with st.spinner("Validating..."):
                            wh = first_warehouse_id(wiz_host, wiz_token)
                            report = validation_report(wiz_host, wiz_token, wh, table.strip())
                        if report["passed"]:
                            st.success(f"Table has {report['row_count']} rows, "
                                       f"{report['duplicate_rows']} duplicates.")
                        else:
                            st.error("Table is empty — did the job run?")
                        st.write(f"Columns: {len(report['columns'])}")
                    except SqlError as exc:
                        st.error(str(exc))
            st.balloons()
        gate_ok = True

    # ---- Navigation -----------------------------------------------------
    st.divider()
    nav_prev, _, nav_next = st.columns([1, 4, 1])
    if step > 0 and nav_prev.button("← Back", key="wiz_back"):
        st.session_state["wiz_step"] = step - 1
        st.rerun()
    if step < len(WIZ_LABELS) - 1:
        if nav_next.button("Continue →", type="primary", disabled=not gate_ok, key="wiz_cont"):
            st.session_state["wiz_step"] = step + 1
            st.rerun()
        if not gate_ok:
            nav_next.caption("Complete this step")

with tab_repo:
    st.subheader("Migration repo & flow")
    if not object_names:
        st.info("No objects ingested yet — use the sidebar to parse a workflow.")
    else:
        metadatas = repo_cache.list_metadata(repo)

        st.markdown("##### Workflow flow")
        canvas_obj = st.selectbox("Show flow for", object_names, key="canvas_object")
        canvas_wf = repo_cache.read_workflow(repo, canvas_obj)

        with st.expander("Delete this object"):
            st.caption(
                "Removes the ingested workflow (ir.json/metadata.json) from the repo. "
                "This does not affect anything already deployed to Databricks."
            )
            confirm_delete = st.checkbox(
                f"I want to permanently delete '{canvas_obj}' from the repo", key="confirm_delete_obj"
            )
            if st.button("Delete object", key="delete_obj_btn", disabled=not confirm_delete):
                repo.delete_object(canvas_obj)
                st.success(f"Deleted: {canvas_obj}")
                st.session_state.pop("confirm_delete_obj", None)
                st.rerun()

        converted = len(canvas_wf.nodes)
        manual = len(canvas_wf.unsupported)
        total = converted + manual
        out_tables = sum(1 for n in canvas_wf.nodes if n.tool_type == ToolType.OUTPUT)
        macros_used = len(canvas_wf.referenced_macros())
        t1, t2, t3, t4 = st.columns(4)
        t1.metric("Total tools", total)
        t2.metric(
            "Auto-converted", converted,
            delta=f"{round(100 * converted / total)}%" if total else None,
        )
        t3.metric("Manual follow-up", manual)
        t4.metric("Output tables", out_tables)
        if macros_used:
            st.caption(f"References {macros_used} macro(s): {', '.join(canvas_wf.referenced_macros())}")

        # components.html (iframe) rather than st.markdown: the canvas ships
        # its own pan/zoom script, and st.markdown never executes <script>.
        st.components.v1.html(workflow_canvas_html(canvas_wf), height=640, scrolling=False)

        with st.expander("Repo metadata"):
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
            st.markdown("##### Topological conversion order")
            st.caption("Dependencies first — leaf objects convert before the objects that read them.")
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

        with st.expander("Learning log — what the accelerator has learned"):
            st.caption(
                "Human spec corrections feed future LLM conversions (retrieval, "
                "not fine-tuning); Databricks failures are captured per stage. "
                "Tool types that keep appearing here are where the converter "
                "most needs work."
            )
            corr_counts = correction_counts_by_tool()
            stage_counts = deploy_error_counts_by_stage()
            lc1, lc2 = st.columns(2)
            with lc1:
                st.markdown("**Corrections by tool type**")
                if corr_counts:
                    st.dataframe(
                        [{"tool": t, "corrections": n} for t, n in corr_counts.items()],
                        use_container_width=True, hide_index=True,
                    )
                else:
                    st.caption("None yet — save an edited spec to start training.")
            with lc2:
                st.markdown("**Databricks failures by stage**")
                if stage_counts:
                    st.dataframe(
                        [{"stage": s, "failures": n} for s, n in stage_counts.items()],
                        use_container_width=True, hide_index=True,
                    )
                else:
                    st.caption("No failures recorded.")
            n_code_edits = code_correction_count()
            if n_code_edits:
                st.caption(
                    f"**{n_code_edits} manual code edit(s)** logged — recurring code-level "
                    "fixes point at renderer gaps worth closing in convert/renderer.py."
                )
            for err in recent_deploy_errors(limit=3):
                st.caption(
                    f"`{err.logged_at[:16]}` **{err.workflow_name}** ({err.stage}): "
                    f"{err.message[:160]}"
                )

with tab_convert:
    if not object_names:
        st.info("Ingest an object first.")
    else:
        selected = st.selectbox("Object", object_names, key="convert_object")
        workflow = repo_cache.read_workflow(repo, selected)
        metadata = repo_cache.read_metadata(repo, selected)

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
                    spec = naive_spec_from_workflow(workflow, target, macros=repo_cache.all_macros(_repo()))
                spec_yaml = _spec_to_yaml(spec)
                st.session_state[spec_key] = spec_yaml
                # Kept untouched by edits below, so "Validate edits" can diff
                # the human correction against what was actually generated.
                st.session_state[f"{spec_key}::generated"] = spec_yaml
                st.session_state[f"tool_types::{selected}"] = [
                    n.tool_type.value for n in workflow.nodes
                ]
                # A freshly generated spec is already validated — store it so
                # the Generated code tab (format picker + render) works
                # immediately; re-validation is only needed after manual edits.
                st.session_state[f"validated_spec::{selected}"] = spec
                _clear_code_overrides(spec.name)
            except SpecGenerationError as exc:
                st.error(str(exc))

        if spec_key in st.session_state:
            st.write("**Pipeline spec (YAML)** — review and edit before rendering:")
            edited = st.text_area("spec_yaml", value=st.session_state[spec_key], height=400, label_visibility="collapsed")
            st.session_state[spec_key] = edited

            if st.button("Save edits & train", type="primary", key=f"validate::{selected}"):
                try:
                    data = yaml.safe_load(edited)
                    validated_spec = PipelineSpec.model_validate(data)
                    st.session_state[f"validated_spec::{selected}"] = validated_spec
                    _clear_code_overrides(validated_spec.name)
                    generated_yaml = st.session_state.get(f"{spec_key}::generated")
                    trained = False
                    if generated_yaml is not None and generated_yaml.strip() != edited.strip():
                        log_conversion_triple(
                            workflow_name=selected,
                            tool_types=st.session_state.get(f"tool_types::{selected}", []),
                            generated_spec_yaml=generated_yaml,
                            corrected_spec_yaml=edited,
                        )
                        trained = True
                    st.success(
                        "Spec saved — correction recorded, future conversions of "
                        "similar workflows will see this fix."
                        if trained else "Spec is valid (no changes from the generated version)."
                    )
                except (yaml.YAMLError, ValidationError) as exc:
                    st.error(f"Validation failed: {exc}")
            st.caption(
                "Edits are compared against the generated spec and stored as a "
                "correction — the retrieval loop shows them to the LLM next time "
                "it converts a workflow using similar tools."
            )

            similar = find_similar_corrections(
                set(st.session_state.get(f"tool_types::{selected}", []))
            )
            if similar:
                with st.expander(
                    f"What the system learned from {len(similar)} similar past correction(s)"
                ):
                    for rec in similar:
                        st.markdown(f"**{rec.workflow_name}** — corrected on {rec.logged_at[:10]}")
                        st.code(summarize_correction(rec), language="diff")

            validated = cast(
                "PipelineSpec | None", st.session_state.get(f"validated_spec::{selected}")
            )
            if validated is not None:
                with st.expander("Sources & targets — set real table paths", expanded=True):
                    def _tab_apply_io(updated: PipelineSpec) -> None:
                        st.session_state[f"validated_spec::{selected}"] = updated
                        st.session_state[f"spec_yaml::{selected}"] = _spec_to_yaml(updated)
                        st.session_state[spec_key] = _spec_to_yaml(updated)
                        _clear_code_overrides(updated.name)

                    _render_io_panel(validated, f"conv::{selected}", _tab_apply_io)

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
                workflow = repo_cache.read_workflow(repo, selected)
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
                _render_artifact_preview(stored_spec, chosen_format, key=f"code::{selected}")
                code = _ARTIFACT_RENDERERS[chosen_format](stored_spec)
                dl1, dl2 = st.columns(2)
                dl1.download_button(
                    "Download main .py", data=code, file_name=f"{selected}.py",
                    mime="text/x-python", key=f"dl_main::{selected}",
                )
                utility_code = render_utility_module(stored_spec)
                if utility_code is not None:
                    dl2.download_button(
                        "Download utility .py", data=utility_code,
                        file_name=f"{utils_module_name(stored_spec)}.py",
                        mime="text/x-python", key=f"dl_util::{selected}",
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
        value=DEPLOY_DEFAULTS.wizard_host,
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

    parity_mode = st.radio(
        "Comparison source",
        [
            "Alteryx output CSV (best: row-level proof)",
            "Existing expected table",
            "No Alteryx access - structural validation only",
        ],
        help="No access to run the original Alteryx workflow? The third mode validates "
        "the migrated output on its own: row count, schema, null rates, duplicates.",
    )
    expected_csv = None
    expected_table = ""
    ignore_raw = ""
    if parity_mode.startswith("Alteryx output"):
        expected_csv = st.file_uploader(
            "Alteryx output export (.csv)", type=["csv"], key="parity_csv",
            help="Export the original workflow's output from Alteryx as CSV and upload it here.",
        )
    elif parity_mode.startswith("Existing"):
        expected_table = st.text_input("Expected table name", value="")
    if not parity_mode.startswith("No Alteryx"):
        ignore_raw = st.text_input(
            "Columns to exclude from the diff (comma-separated)",
            value="Load_Date",
            help="Non-deterministic columns: load timestamps, run ids, sequence columns "
            "whose order Alteryx and Spark assign differently.",
        )

    if st.button("Run parity check", type="primary"):
        if not parity_token:
            st.error("An access token is required.")
        elif parity_mode.startswith("No Alteryx"):
            try:
                with st.spinner("Validating migrated output..."):
                    warehouse = first_warehouse_id(parity_host, parity_token)
                    report = validation_report(
                        parity_host, parity_token, warehouse, migrated_table.strip()
                    )
            except SqlError as exc:
                st.error(str(exc))
                st.stop()
            if report["passed"]:
                st.success(f"Table exists with {report['row_count']} rows.")
            else:
                st.error("Table is empty — the migrated job may not have run.")
            st.write(f"Columns ({len(report['columns'])}): " + ", ".join(report["columns"]))
            st.write(f"Full-row duplicates: {report['duplicate_rows']}")
            nulls = {c: int(n) for c, n in report["null_counts"].items() if int(n) > 0}
            if nulls:
                st.write("Columns containing nulls:")
                st.dataframe([{"column": c, "nulls": n} for c, n in nulls.items()],
                             use_container_width=True)
            else:
                st.write("No nulls in the checked columns.")
            st.caption(
                "Structural validation only — for row-level proof, export the legacy "
                "system's historical output (even a one-off table dump) and use the "
                "CSV or expected-table mode."
            )
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
                # The strongest conversion-bug signal there is: capture it
                # alongside deploy/run failures rather than showing it once.
                sample = report["extra_in_migrated"]["rows"][:3]
                log_deploy_error(
                    migrated_table.strip(), "parity",
                    f"migrated={m} expected={e}; "
                    f"extra_in_migrated={len(report['extra_in_migrated']['rows'])} "
                    f"missing_from_migrated={len(report['missing_from_migrated']['rows'])}; "
                    f"sample_extra={sample!r}",
                )
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
            # Which of the three the "One-click migrate & deploy" button below
            # actually targets — previously it silently always used dev_host,
            # with no way to deploy the same job/SDP to staging or prod.
            deploy_env = st.selectbox("Deploy environment", ["dev", "staging", "prod"])
            _env_hosts = {"dev": dev_host, "staging": staging_host, "prod": prod_host}
            deploy_host = _env_hosts[deploy_env]
            # Mirrors default_bundle()'s convention: dev/staging get a
            # suffixed schema, prod uses the bare one.
            deploy_schema = (
                DEPLOY_DEFAULTS.schema_name
                if deploy_env == "prod"
                else f"{DEPLOY_DEFAULTS.schema_name}_{deploy_env}"
            )
        else:
            free_host = st.text_input(
                "Workspace host",
                value=DEPLOY_DEFAULTS.wizard_host,
                help="Your Databricks Free Edition workspace URL.",
            )
            deploy_env = "default"
            deploy_host = free_host
            deploy_schema = DEPLOY_DEFAULTS.schema_name

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
            st.caption(
                f"Target: **{deploy_env}** — `{deploy_host}`, schema `{deploy_schema}`."
                + (" Same as your Convert-step schema." if deploy_env == "default" else "")
            )
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
                    # A dev/staging/prod pick re-targets the spec's own schema too
                    # — for job/notebook the table names are literal in the
                    # generated code, so only re-pointing the DAB host would
                    # deploy correct code to the wrong environment's schema.
                    env_spec = (
                        deploy_spec
                        if deploy_env == "default"
                        else deploy_spec.model_copy(
                            update={
                                "target": deploy_spec.target.model_copy(
                                    update={"schema_": deploy_schema}
                                )
                            }
                        )
                    )
                    safe_name = re.sub(r"\W+", "_", selected).strip("_").lower()
                    env_suffix = "" if deploy_env == "default" else f"_{deploy_env}"
                    bundle_dir = _ROOT / "bundles" / f"{safe_name}{env_suffix}"
                    # A saved manual code edit was captured against one specific
                    # rendering; reusing it verbatim under a different schema
                    # would deploy the old environment's table names, so it only
                    # auto-applies to the single-workspace (default) target.
                    code_override = (
                        st.session_state.get(f"code_override::{deploy_spec.name}::{artifact_format}")
                        if deploy_env == "default"
                        else None
                    )
                    if deploy_env != "default" and st.session_state.get(
                        f"code_override::{deploy_spec.name}::{artifact_format}"
                    ):
                        st.warning(
                            "A saved manual code edit exists but targets a different schema — "
                            "not auto-applied here. Re-check the Generated code tab for this "
                            "environment if that fix still applies."
                        )
                    with st.spinner("Rendering artifact and writing bundle..."):
                        export_bundle_from_spec(
                            env_spec,
                            bundle_dir,
                            workspace_host=deploy_host,
                            artifact_format=artifact_format,
                            bundle_name=f"{bundle_name}{env_suffix}",
                            main_code_override=code_override,
                        )
                    with st.spinner(f"Deploying to {deploy_host} ..."):
                        ok, log = deploy_bundle(bundle_dir, deploy_host, token)
                    st.code(log)
                    if ok:
                        st.success(f"Deployed. Bundle written to `bundles/{safe_name}{env_suffix}/`.")
                        _push_bundle_to_git(bundle_dir, selected)
                    else:
                        log_deploy_error(selected, "deploy", log)
                        st.session_state[f"deploy_error::{selected}"] = log
                        st.error("Deploy failed — see the CLI output above.")

            def _tab_apply_repair(repaired: PipelineSpec, repaired_yaml: str) -> None:
                st.session_state[f"validated_spec::{selected}"] = repaired
                st.session_state[f"spec_yaml::{selected}"] = repaired_yaml

            _offer_auto_repair(
                f"deploy_error::{selected}",
                selected,
                st.session_state.get(f"spec_yaml::{selected}", ""),
                "deploy",
                st.session_state.get(f"tool_types::{selected}", []),
                _tab_apply_repair,
            )

            if st.button("Run the deployed job now"):
                if not token:
                    st.error("A Databricks access token is required.")
                else:
                    safe_name = re.sub(r"\W+", "_", selected).strip("_").lower()
                    env_suffix = "" if deploy_env == "default" else f"_{deploy_env}"
                    bundle_dir = _ROOT / "bundles" / f"{safe_name}{env_suffix}"
                    if not (bundle_dir / "databricks.yml").exists():
                        st.error(f"Deploy to {deploy_env} first — no bundle found for this object.")
                    else:
                        with st.spinner("Running job (waits for completion)..."):
                            ok, log = run_bundle_job(bundle_dir, deploy_host, token)
                        st.code(log)
                        if not ok:
                            log_deploy_error(selected, "run", log)
                            st.session_state[f"run_error::{selected}"] = log
                        else:
                            st.session_state.pop(f"run_error::{selected}", None)
                        (st.success if ok else st.error)(
                            "Job succeeded." if ok else "Job failed — see output above."
                        )

            def _tab_apply_run_repair(repaired: PipelineSpec, repaired_yaml: str) -> None:
                st.session_state[f"validated_spec::{selected}"] = repaired
                st.session_state[f"spec_yaml::{selected}"] = repaired_yaml

            _offer_auto_repair(
                f"run_error::{selected}",
                selected,
                st.session_state.get(f"spec_yaml::{selected}", ""),
                "run",
                st.session_state.get(f"tool_types::{selected}", []),
                _tab_apply_run_repair,
            )
