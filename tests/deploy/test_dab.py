from pathlib import Path

import yaml

from deploy.dab import build_databricks_yml, default_bundle, write_databricks_yml
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
