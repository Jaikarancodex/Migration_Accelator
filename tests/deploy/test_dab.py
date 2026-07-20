from pathlib import Path

import yaml

from deploy.dab import (
    build_databricks_yml,
    default_bundle,
    single_target_bundle,
    write_databricks_yml,
)
from deploy.models import DABBundle


def _bundle() -> DABBundle:
    return default_bundle(
        bundle_name="migration-accelerator",
        pipeline_name="sales_summary",
        python_file="./pipelines/sales_summary.py",
        dev_host="https://dev.databricks.example.com",
        staging_host="https://staging.databricks.example.com",
        prod_host="https://prod.databricks.example.com",
        catalog="main",
        schema="migration",
    )


def test_build_databricks_yml_is_valid_yaml() -> None:
    text = build_databricks_yml(_bundle())
    doc = yaml.safe_load(text)

    assert doc["bundle"]["name"] == "migration-accelerator"
    job = doc["resources"]["jobs"]["sales_summary_job"]
    assert job["tasks"][0]["spark_python_task"]["python_file"] == "./pipelines/sales_summary.py"


def test_build_databricks_yml_has_three_targets_with_correct_modes() -> None:
    doc = yaml.safe_load(build_databricks_yml(_bundle()))

    assert doc["targets"]["dev"]["mode"] == "development"
    assert doc["targets"]["staging"]["mode"] == "development"
    assert doc["targets"]["prod"]["mode"] == "production"
    assert doc["targets"]["prod"]["variables"]["schema"] == "migration"
    assert doc["targets"]["dev"]["variables"]["schema"] == "migration_dev"


def test_write_databricks_yml_creates_file(tmp_path: Path) -> None:
    output = write_databricks_yml(_bundle(), tmp_path / "databricks.yml")

    assert output.exists()
    doc = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert doc["bundle"]["name"] == "migration-accelerator"


def test_notebook_format_emits_notebook_task() -> None:
    bundle = single_target_bundle(
        bundle_name="migration-accelerator",
        pipeline_name="sales_summary",
        python_file="./pipelines/sales_summary.py",
        workspace_host="https://community.cloud.databricks.com",
        catalog="main",
        schema="migration",
        artifact_format="notebook",
    )
    doc = yaml.safe_load(build_databricks_yml(bundle))

    task = doc["resources"]["jobs"]["sales_summary_job"]["tasks"][0]
    assert task["notebook_task"]["notebook_path"] == "./pipelines/sales_summary.py"
    assert "spark_python_task" not in task


def test_sdp_format_emits_pipeline_resource_and_no_job() -> None:
    bundle = single_target_bundle(
        bundle_name="migration-accelerator",
        pipeline_name="sales_summary",
        python_file="./pipelines/sales_summary.py",
        workspace_host="https://community.cloud.databricks.com",
        catalog="main",
        schema="migration",
        artifact_format="sdp",
    )
    doc = yaml.safe_load(build_databricks_yml(bundle))

    assert "jobs" not in doc["resources"]
    pipeline = doc["resources"]["pipelines"]["sales_summary_pipeline"]
    assert pipeline["catalog"] == "main"
    assert pipeline["schema"] == "migration"
    assert pipeline["libraries"] == [{"file": {"path": "./pipelines/sales_summary.py"}}]


def test_sdp_includes_utility_module_as_extra_library() -> None:
    bundle = single_target_bundle(
        bundle_name="migration-accelerator",
        pipeline_name="sales_summary",
        python_file="./src/sales_summary.py",
        workspace_host="https://community.cloud.databricks.com",
        catalog="main",
        schema="migration",
        artifact_format="sdp",
        extra_artifact_paths=["./src/sales_summary_utils.py"],
    )
    doc = yaml.safe_load(build_databricks_yml(bundle))
    pipeline = doc["resources"]["pipelines"]["sales_summary_pipeline"]
    assert pipeline["libraries"] == [
        {"file": {"path": "./src/sales_summary.py"}},
        {"file": {"path": "./src/sales_summary_utils.py"}},
    ]


def test_space_containing_workflow_name_yields_valid_resource_keys() -> None:
    import re

    bundle = single_target_bundle(
        bundle_name="alteryx_use_case_workflow",
        pipeline_name="Alteryx Use Case Workflow",
        python_file="./src/Alteryx_Use_Case_Workflow.py",
        workspace_host="https://community.cloud.databricks.com",
        catalog="workspace",
        schema="default",
        artifact_format="notebook",
    )
    doc = yaml.safe_load(build_databricks_yml(bundle))

    (job_key,) = doc["resources"]["jobs"].keys()
    # Terraform resource-name rule: start with letter/underscore, then
    # letters/digits/underscores/dashes only.
    assert re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*", job_key), job_key
    task_key = doc["resources"]["jobs"][job_key]["tasks"][0]["task_key"]
    assert re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*", task_key), task_key
    # the display name keeps the original spaces
    assert doc["resources"]["jobs"][job_key]["name"] == "Alteryx Use Case Workflow_job"


def test_sdp_space_containing_name_yields_valid_pipeline_key() -> None:
    import re

    bundle = single_target_bundle(
        bundle_name="wf",
        pipeline_name="My Fancy Workflow",
        python_file="./src/My_Fancy_Workflow.py",
        workspace_host="https://community.cloud.databricks.com",
        catalog="workspace",
        schema="default",
        artifact_format="sdp",
    )
    doc = yaml.safe_load(build_databricks_yml(bundle))
    (pipe_key,) = doc["resources"]["pipelines"].keys()
    assert re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*", pipe_key), pipe_key


def test_single_target_bundle_has_exactly_one_development_target() -> None:
    bundle = single_target_bundle(
        bundle_name="migration-accelerator",
        pipeline_name="sales_summary",
        python_file="./pipelines/sales_summary.py",
        workspace_host="https://community.cloud.databricks.com",
        catalog="main",
        schema="migration",
    )
    doc = yaml.safe_load(build_databricks_yml(bundle))

    assert list(doc["targets"].keys()) == ["default"]
    assert doc["targets"]["default"]["mode"] == "development"
    assert doc["targets"]["default"]["workspace"]["host"] == "https://community.cloud.databricks.com"
    assert doc["targets"]["default"]["variables"]["schema"] == "migration"
