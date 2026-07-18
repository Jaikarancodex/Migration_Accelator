"""Renders a DABBundle into a `databricks.yml` document.

Only the dev/staging/prod deployment topology and job wiring live here —
this session does not run `databricks bundle deploy` against a live
workspace (see project non-goals); it only produces the YAML a human/CI step
would deploy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from deploy.models import DABBundle


def build_databricks_yml(bundle: DABBundle) -> str:
    doc: dict[str, Any] = {
        "bundle": {"name": bundle.bundle_name},
        "resources": {
            "jobs": {
                bundle.job.name: {
                    "name": bundle.job.name,
                    "tasks": [
                        {
                            "task_key": task.task_key,
                            "spark_python_task": {
                                "python_file": task.python_file,
                                **({"parameters": task.parameters} if task.parameters else {}),
                            },
                        }
                        for task in bundle.job.tasks
                    ],
                }
            }
        },
        "targets": {
            env_name: {
                "mode": target.mode,
                "workspace": {"host": target.workspace_host},
                "variables": {"catalog": target.catalog, "schema": target.schema_},
            }
            for env_name, target in bundle.targets.items()
        },
    }
    return yaml.safe_dump(doc, sort_keys=False, default_flow_style=False)


def write_databricks_yml(bundle: DABBundle, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.write_text(build_databricks_yml(bundle), encoding="utf-8")
    return path


def default_bundle(
    bundle_name: str,
    pipeline_name: str,
    python_file: str,
    dev_host: str,
    staging_host: str,
    prod_host: str,
    catalog: str,
    schema: str,
) -> DABBundle:
    """A minimal, sensible-default bundle: one job, one task, three env targets."""
    from deploy.models import DABJob, DABTarget, DABTask

    return DABBundle(
        bundle_name=bundle_name,
        job=DABJob(
            name=f"{pipeline_name}_job",
            tasks=[DABTask(task_key=pipeline_name, python_file=python_file)],
        ),
        targets={
            "dev": DABTarget(mode="development", workspace_host=dev_host, catalog=catalog, schema=f"{schema}_dev"),
            "staging": DABTarget(
                mode="development", workspace_host=staging_host, catalog=catalog, schema=f"{schema}_staging"
            ),
            "prod": DABTarget(mode="production", workspace_host=prod_host, catalog=catalog, schema=schema),
        },
    )
