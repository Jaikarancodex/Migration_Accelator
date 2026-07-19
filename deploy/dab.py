"""Renders a DABBundle into a `databricks.yml` document.

Only the dev/staging/prod deployment topology and job wiring live here —
this session does not run `databricks bundle deploy` against a live
workspace (see project non-goals); it only produces the YAML a human/CI step
would deploy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml

from deploy.models import DABBundle, DABTask

ArtifactFormat = Literal["job", "notebook", "sdp"]


def _task_doc(task: DABTask) -> dict[str, Any]:
    doc: dict[str, Any] = {"task_key": task.task_key}
    if task.notebook_path is not None:
        doc["notebook_task"] = {
            "notebook_path": task.notebook_path,
            **({"base_parameters": dict.fromkeys(task.parameters, "")} if task.parameters else {}),
        }
    else:
        doc["spark_python_task"] = {
            "python_file": task.python_file,
            **({"parameters": task.parameters} if task.parameters else {}),
        }
    return doc


def build_databricks_yml(bundle: DABBundle) -> str:
    resources: dict[str, Any] = {}
    if bundle.job is not None:
        resources["jobs"] = {
            bundle.job.name: {
                "name": bundle.job.name,
                "tasks": [_task_doc(task) for task in bundle.job.tasks],
            }
        }
    if bundle.pipeline is not None:
        resources["pipelines"] = {
            bundle.pipeline.name: {
                "name": bundle.pipeline.name,
                "catalog": bundle.pipeline.catalog,
                "schema": bundle.pipeline.schema_,
                "serverless": bundle.pipeline.serverless,
                "libraries": [{"file": {"path": bundle.pipeline.library_path}}],
            }
        }

    # Target-level variable overrides are only valid if the variables are
    # declared at the top level; default them from the first target.
    first_target = next(iter(bundle.targets.values()))
    doc: dict[str, Any] = {
        "bundle": {"name": bundle.bundle_name},
        "variables": {
            "catalog": {"description": "Target catalog", "default": first_target.catalog},
            "schema": {"description": "Target schema", "default": first_target.schema_},
        },
        "resources": resources,
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


def _bundle_resources(
    pipeline_name: str,
    artifact_path: str,
    artifact_format: ArtifactFormat,
    catalog: str,
    schema: str,
) -> dict[str, Any]:
    """Build the job/pipeline kwargs for a DABBundle from the chosen artifact format."""
    from deploy.models import DABJob, DABPipeline, DABTask

    if artifact_format == "sdp":
        return {
            "pipeline": DABPipeline(
                name=f"{pipeline_name}_pipeline",
                catalog=catalog,
                schema=schema,
                library_path=artifact_path,
            )
        }
    task = (
        DABTask(task_key=pipeline_name, notebook_path=artifact_path)
        if artifact_format == "notebook"
        else DABTask(task_key=pipeline_name, python_file=artifact_path)
    )
    return {"job": DABJob(name=f"{pipeline_name}_job", tasks=[task])}


def default_bundle(
    bundle_name: str,
    pipeline_name: str,
    python_file: str,
    dev_host: str,
    staging_host: str,
    prod_host: str,
    catalog: str,
    schema: str,
    artifact_format: ArtifactFormat = "job",
) -> DABBundle:
    """A minimal, sensible-default bundle: one job or pipeline, three env targets."""
    from deploy.models import DABTarget

    return DABBundle(
        bundle_name=bundle_name,
        **_bundle_resources(pipeline_name, python_file, artifact_format, catalog, schema),
        targets={
            "dev": DABTarget(mode="development", workspace_host=dev_host, catalog=catalog, schema=f"{schema}_dev"),
            "staging": DABTarget(
                mode="development", workspace_host=staging_host, catalog=catalog, schema=f"{schema}_staging"
            ),
            "prod": DABTarget(mode="production", workspace_host=prod_host, catalog=catalog, schema=schema),
        },
    )


def single_target_bundle(
    bundle_name: str,
    pipeline_name: str,
    python_file: str,
    workspace_host: str,
    catalog: str,
    schema: str,
    target_name: str = "default",
    artifact_format: ArtifactFormat = "job",
) -> DABBundle:
    """A one-resource, one-target bundle for a single free-tier workspace.

    Databricks Free Edition is one workspace with no dev/staging/prod
    promotion story, so `default_bundle`'s three-target model doesn't apply;
    this collapses it to a single "development"-mode target.
    """
    from deploy.models import DABTarget

    return DABBundle(
        bundle_name=bundle_name,
        **_bundle_resources(pipeline_name, python_file, artifact_format, catalog, schema),
        targets={
            target_name: DABTarget(
                mode="development", workspace_host=workspace_host, catalog=catalog, schema=schema
            ),
        },
    )
