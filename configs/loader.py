"""Generic YAML -> pydantic config loader, validated on load.

Every config in this project (target naming, DAB defaults, future eval
thresholds) should be a pydantic model loaded through this function rather
than read as a raw dict — that's what "config-driven, no hardcoded
paths/names" means in practice.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel

ModelT = TypeVar("ModelT", bound=BaseModel)


def load_yaml_config(path: str | Path, model: type[ModelT]) -> ModelT:
    text = Path(path).read_text(encoding="utf-8")
    data = yaml.safe_load(text) or {}
    return model.model_validate(data)
