"""Pydantic models for the YAML files in this directory."""

from __future__ import annotations

from pydantic import BaseModel

from convert.spec import MedallionLayer


class TargetDefaultsConfig(BaseModel):
    """Loaded from configs/target.yaml — default catalog/schema/layer naming."""

    catalog: str
    schema_name: str
    layer: MedallionLayer


class DeployDefaultsConfig(BaseModel):
    """Loaded from configs/deploy.yaml — DAB bundle defaults."""

    bundle_name: str
    dev_host: str
    staging_host: str
    prod_host: str
    catalog: str
    schema_name: str
    # Single-workspace deploy path (Databricks Free Edition, no dev/staging/
    # prod promotion): the one host used across the wizard and the advanced
    # tab's quick-deploy flow. Kept separate from dev_host/staging_host/
    # prod_host, which are for the multi-target Azure bundle path.
    wizard_host: str = ""
