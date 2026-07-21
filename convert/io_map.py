"""Sources and targets of a workflow / pipeline spec, and source-path binding.

Two views:

- workflow-level (`workflow_sources` / `workflow_targets`): the raw Input and
  Output tools straight from the parsed Alteryx IR, for a quick "what does
  this read and write" summary right after ingest.

- spec-level (`spec_io`): the ReadSteps and WriteSteps of a generated
  PipelineSpec, each source annotated with the steps that consume it (and the
  first one — where the data starts flowing). `apply_source_overrides` rebinds
  each source to a user-supplied table path, which then propagates through the
  renderer into every artifact format.
"""

from __future__ import annotations

from pydantic import BaseModel

from convert.spec import PipelineSpec, ReadStep, Step, WriteStep
from ingest.alteryx.ir import ToolType, Workflow


def workflow_sources(workflow: Workflow) -> list[tuple[str, str]]:
    """(tool_id, source name) for every Input tool, in tool order."""
    out: list[tuple[str, str]] = []
    for node in workflow.nodes:
        if node.tool_type == ToolType.INPUT:
            out.append((node.tool_id, node.table_name or "(custom SQL / no table name)"))
    return out


def workflow_targets(workflow: Workflow) -> list[tuple[str, str]]:
    """(tool_id, target name) for every Output tool, in tool order."""
    return [
        (node.tool_id, node.output_path or node.table_name or "(unnamed output)")
        for node in workflow.nodes
        if node.tool_type == ToolType.OUTPUT
    ]


def _step_input_ids(step: Step) -> list[str]:
    """Ids of the steps a step reads from, across every step shape."""
    ids: list[str] = []
    for attr in ("input", "left", "right", "target", "source"):
        value = getattr(step, attr, None)
        if isinstance(value, str):
            ids.append(value)
    inputs = getattr(step, "inputs", None)
    if isinstance(inputs, list):
        ids.extend(i for i in inputs if isinstance(i, str))
    return ids


def _consumer_detail(step: Step) -> str:
    """A short human description of what a consuming step does."""
    op = getattr(step, "op", "step")
    if op == "filter":
        return f"filter {getattr(step, 'condition', '')!r}"
    if op == "join":
        return f"join on {getattr(step, 'left_keys', [])} = {getattr(step, 'right_keys', [])}"
    if op == "aggregate":
        return f"aggregate group_by={getattr(step, 'group_by', [])}"
    if op == "union":
        return f"union of {len(getattr(step, 'inputs', []))} inputs"
    if op == "select":
        return "select / rename columns"
    if op == "write":
        return f"write to {getattr(step, 'target_table', '')}"
    return op


class Consumer(BaseModel):
    step_id: str
    op: str
    detail: str


class SourceBinding(BaseModel):
    """One ReadStep plus the steps that consume its data."""

    read_id: str
    alias: str
    source_table: str
    consumers: list[Consumer]

    @property
    def first_consumer(self) -> Consumer | None:
        return self.consumers[0] if self.consumers else None


class TargetBinding(BaseModel):
    write_id: str
    target_table: str
    mode: str
    fed_by: str


def spec_io(spec: PipelineSpec) -> tuple[list[SourceBinding], list[TargetBinding]]:
    """Sources (with consumers, in spec order) and targets of a spec."""
    sources: list[SourceBinding] = []
    for read in spec.steps:
        if not isinstance(read, ReadStep):
            continue
        consumers = [
            Consumer(step_id=s.id, op=getattr(s, "op", "step"), detail=_consumer_detail(s))
            for s in spec.steps
            if read.id in _step_input_ids(s)
        ]
        sources.append(
            SourceBinding(
                read_id=read.id, alias=read.alias,
                source_table=read.source_table, consumers=consumers,
            )
        )
    targets = [
        TargetBinding(
            write_id=w.id, target_table=w.target_table, mode=w.mode, fed_by=w.input
        )
        for w in spec.steps
        if isinstance(w, WriteStep)
    ]
    return sources, targets


def apply_source_overrides(spec: PipelineSpec, overrides: dict[str, str]) -> PipelineSpec:
    """Rebind ReadSteps to user-supplied table paths (blank values ignored).

    Returns a new spec; `overrides` maps a ReadStep id to its real table.
    The alias is refreshed from the new table's last segment so downstream
    variable names stay meaningful.
    """
    updated = spec.model_copy(deep=True)
    for step in updated.steps:
        if isinstance(step, ReadStep) and (new := overrides.get(step.id, "").strip()):
            step.source_table = new
            step.alias = new.rsplit(".", 1)[-1] or step.alias
    return updated
