"""The migration repo layer: a normalized store of extracted source artifacts.

One folder per object, `<root>/<name>/`, containing:
  - ir.json        the parsed Workflow IR (ingest/alteryx/ir.py)
  - metadata.json  ObjectMetadata (repo/metadata.py)

This is the boundary between ingestion (source-specific parsers) and
everything downstream (conversion, LLM, eval) — downstream code only ever
reads from this store, never from the original source files.
"""

from __future__ import annotations

from pathlib import Path

from ingest.alteryx.ir import ToolType, Workflow
from repo.metadata import ObjectMetadata


class ObjectNotFoundError(KeyError):
    pass


class MigrationRepo:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _object_dir(self, name: str) -> Path:
        return self.root / name

    def write_workflow(self, workflow: Workflow, source_system: str = "alteryx") -> ObjectMetadata:
        """Persist a parsed Workflow and derive its ObjectMetadata."""
        obj_dir = self._object_dir(workflow.name)
        obj_dir.mkdir(parents=True, exist_ok=True)

        (obj_dir / "ir.json").write_text(workflow.model_dump_json(indent=2), encoding="utf-8")

        input_tables = sorted(
            {n.table_name for n in workflow.nodes if n.tool_type == ToolType.INPUT and n.table_name}
        )
        output_tables = sorted(
            {n.output_path for n in workflow.nodes if n.tool_type == ToolType.OUTPUT and n.output_path}
        )
        metadata = ObjectMetadata(
            name=workflow.name,
            source_system=source_system,
            source_file=workflow.source_file,
            input_tables=input_tables,
            output_tables=output_tables,
            unsupported_tool_count=len(workflow.unsupported),
        )
        (obj_dir / "metadata.json").write_text(metadata.model_dump_json(indent=2), encoding="utf-8")
        return metadata

    def read_workflow(self, name: str) -> Workflow:
        path = self._object_dir(name) / "ir.json"
        if not path.exists():
            raise ObjectNotFoundError(name)
        return Workflow.model_validate_json(path.read_text(encoding="utf-8"))

    def read_metadata(self, name: str) -> ObjectMetadata:
        path = self._object_dir(name) / "metadata.json"
        if not path.exists():
            raise ObjectNotFoundError(name)
        return ObjectMetadata.model_validate_json(path.read_text(encoding="utf-8"))

    def write_metadata(self, metadata: ObjectMetadata) -> None:
        obj_dir = self._object_dir(metadata.name)
        obj_dir.mkdir(parents=True, exist_ok=True)
        (obj_dir / "metadata.json").write_text(metadata.model_dump_json(indent=2), encoding="utf-8")

    def list_object_names(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(p.name for p in self.root.iterdir() if p.is_dir() and p.name != "macros")

    # -- Macro registry (.yxmc workflows, keyed by lowercase stem) ----------

    def _macro_dir(self) -> Path:
        return self.root / "macros"

    @property
    def macro_dir(self) -> Path:
        """Public accessor for callers that need to key a cache on this path."""
        return self._macro_dir()

    def write_macro(self, workflow: Workflow) -> str:
        """Register a parsed .yxmc macro; returns the registry key."""
        key = workflow.name.lower()
        self._macro_dir().mkdir(parents=True, exist_ok=True)
        (self._macro_dir() / f"{key}.json").write_text(
            workflow.model_dump_json(indent=2), encoding="utf-8"
        )
        return key

    def read_macro(self, key: str) -> Workflow:
        path = self._macro_dir() / f"{key.lower()}.json"
        if not path.exists():
            raise ObjectNotFoundError(key)
        return Workflow.model_validate_json(path.read_text(encoding="utf-8"))

    def list_macro_names(self) -> list[str]:
        if not self._macro_dir().exists():
            return []
        return sorted(p.stem for p in self._macro_dir().glob("*.json"))

    def all_macros(self) -> dict[str, Workflow]:
        return {name: self.read_macro(name) for name in self.list_macro_names()}

    def list_metadata(self) -> list[ObjectMetadata]:
        return [self.read_metadata(name) for name in self.list_object_names()]
