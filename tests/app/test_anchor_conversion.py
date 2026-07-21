"""Anchor-aware conversion: Alteryx multi-output tools (Filter True/False,
Join L/J/R, Unique U/D) must route each consumer to the data its connection
actually originates from — not to the tool's primary output. Dropping the
origin anchor was the root cause of a real self-union (df.unionByName(df))
and of False-branch consumers silently receiving True-branch rows.
"""

from pathlib import Path

from app.offline_convert import naive_spec_from_workflow
from convert.spec import FilterStep, JoinStep, ReadStep, TargetRef, UnionStep
from ingest.alteryx.ir import Node, ToolType, UpstreamEdge, Workflow
from ingest.alteryx.parser import parse_yxmd

TARGET = TargetRef(catalog="main", schema="dev", layer="silver")


def _read(tool_id: str, table: str) -> Node:
    return Node(
        tool_id=tool_id, tool_type=ToolType.INPUT,
        raw_plugin="DbFileInput", table_name=table,
    )


def _edge(origin: str, anchor: str, dest_label: str = "Input") -> UpstreamEdge:
    return UpstreamEdge(origin_id=origin, origin_anchor=anchor, dest_label=dest_label)


def _workflow(nodes: list[Node]) -> Workflow:
    return Workflow(source_file="anchors.yxmd", name="anchors", nodes=nodes)


def _filter_node(tool_id: str, upstream: str, condition: str) -> Node:
    return Node(
        tool_id=tool_id, tool_type=ToolType.FILTER, raw_plugin="Filter",
        filter_expression=condition, upstream_ids=[upstream],
        upstream_edges=[_edge(upstream, "Output")],
    )


def _out(tool_id: str, upstream: str, anchor: str, table: str) -> Node:
    return Node(
        tool_id=tool_id, tool_type=ToolType.OUTPUT, raw_plugin="DbFileOutput",
        table_name=table, output_path=table, upstream_ids=[upstream],
        upstream_edges=[_edge(upstream, anchor)],
    )


def test_filter_false_anchor_gets_negated_condition() -> None:
    wf = _workflow([
        _read("1", "legacy.t"),
        _filter_node("2", "1", "[Amount] > 0"),
        _out("3", "2", "True", "kept"),
        _out("4", "2", "False", "rejected"),
    ])
    spec = naive_spec_from_workflow(wf, TARGET)

    derived = spec.step_by_id("2_false")
    assert isinstance(derived, FilterStep)
    assert derived.condition == "NOT ([Amount] > 0)"
    assert derived.input == "1"

    writes = {s.target_table: s.input for s in spec.steps if s.op == "write"}
    assert writes["main.dev.kept"] == "2"
    assert writes["main.dev.rejected"] == "2_false"


def _join_node(tool_id: str, left: str, right: str) -> Node:
    from ingest.alteryx.ir import JoinInput

    return Node(
        tool_id=tool_id, tool_type=ToolType.JOIN, raw_plugin="Join",
        upstream_ids=[left, right],
        upstream_labels={"Left": left, "Right": right},
        upstream_edges=[
            _edge(left, "Output", "Left"), _edge(right, "Output", "Right"),
        ],
        join_inputs=[
            JoinInput(side="left", keys=["id"]), JoinInput(side="right", keys=["cid"]),
        ],
    )


def test_join_left_anchor_becomes_left_anti_join() -> None:
    wf = _workflow([
        _read("1", "legacy.orders"),
        _read("2", "legacy.customers"),
        _join_node("3", "1", "2"),
        _out("4", "3", "Join", "matched"),
        _out("5", "3", "Left", "unmatched_orders"),
    ])
    spec = naive_spec_from_workflow(wf, TARGET)

    anti = spec.step_by_id("3_unjoined_left")
    assert isinstance(anti, JoinStep)
    assert anti.how == "left_anti"
    assert (anti.left, anti.right) == ("1", "2")
    assert (anti.left_keys, anti.right_keys) == (["id"], ["cid"])

    writes = {s.target_table: s.input for s in spec.steps if s.op == "write"}
    assert writes["main.dev.unmatched_orders"] == "3_unjoined_left"
    assert writes["main.dev.matched"] == "3"


def test_join_right_anchor_swaps_sides() -> None:
    wf = _workflow([
        _read("1", "legacy.orders"),
        _read("2", "legacy.customers"),
        _join_node("3", "1", "2"),
        _out("4", "3", "Right", "main.dev.unmatched_customers"),
    ])
    spec = naive_spec_from_workflow(wf, TARGET)

    anti = spec.step_by_id("3_unjoined_right")
    assert isinstance(anti, JoinStep)
    assert anti.how == "left_anti"
    assert (anti.left, anti.right) == ("2", "1")
    assert (anti.left_keys, anti.right_keys) == (["cid"], ["id"])


def test_union_of_true_and_false_anchors_is_not_a_self_union() -> None:
    """Two edges from the same Filter but different anchors must produce two
    distinct inputs — the collapsed form rendered df.unionByName(df) in a
    real converted workflow.
    """
    union = Node(
        tool_id="3", tool_type=ToolType.UNION, raw_plugin="Union",
        upstream_ids=["2", "2"],
        upstream_edges=[_edge("2", "True"), _edge("2", "False")],
    )
    wf = _workflow([
        _read("1", "legacy.t"),
        _filter_node("2", "1", "[x] = 1"),
        union,
        _out("4", "3", "Output", "main.dev.out"),
    ])
    spec = naive_spec_from_workflow(wf, TARGET)

    union_step = spec.step_by_id("3")
    assert isinstance(union_step, UnionStep)
    assert union_step.inputs == ["2", "2_false"]
    assert len(set(union_step.inputs)) == 2


def test_single_input_union_is_a_passthrough_not_a_row_doubler() -> None:
    union = Node(
        tool_id="3", tool_type=ToolType.UNION, raw_plugin="Union",
        upstream_ids=["2"], upstream_edges=[_edge("2", "True")],
    )
    wf = _workflow([
        _read("1", "legacy.t"),
        _filter_node("2", "1", "[x] = 1"),
        union,
        _out("4", "3", "Output", "main.dev.out"),
    ])
    spec = naive_spec_from_workflow(wf, TARGET)

    assert spec.step_by_id("3") is None  # elided, not a self-union
    write = next(s for s in spec.steps if s.op == "write")
    assert write.input == "2"


def test_unique_dup_anchor_lands_as_explicit_todo_placeholder() -> None:
    unique = Node(
        tool_id="2", tool_type=ToolType.UNIQUE, raw_plugin="Unique",
        unique_fields=["id"], upstream_ids=["1"],
        upstream_edges=[_edge("1", "Output")],
    )
    wf = _workflow([
        _read("1", "legacy.t"),
        unique,
        _out("3", "2", "Dup", "main.dev.dupes"),
    ])
    spec = naive_spec_from_workflow(wf, TARGET)

    placeholder = spec.step_by_id("dup_2")
    assert isinstance(placeholder, ReadStep)
    assert "todo_duplicates_2" in placeholder.source_table
    write = next(s for s in spec.steps if s.op == "write")
    assert write.input == "dup_2"


def test_parser_captures_origin_anchors_from_real_xml() -> None:
    fixture = Path(__file__).parent.parent / "fixtures" / "alteryx" / "sales_summary.yxmd"
    wf = parse_yxmd(fixture)
    formula = wf.node_by_id("4")
    assert formula is not None
    assert formula.upstream_edges, "parser must populate upstream_edges"
    assert formula.upstream_edges[0].origin_id == "3"
    assert formula.upstream_edges[0].origin_anchor == "True"


def test_join_missing_labels_never_self_joins() -> None:
    from ingest.alteryx.ir import JoinInput

    join = Node(
        tool_id="3", tool_type=ToolType.JOIN, raw_plugin="Join",
        upstream_ids=["1", "2"],  # no labels, no edges: worst-case old IR
        join_inputs=[JoinInput(side="left", keys=["id"]), JoinInput(side="right", keys=["id"])],
    )
    wf = _workflow([
        _read("1", "legacy.a"),
        _read("2", "legacy.b"),
        join,
        Node(
            tool_id="4", tool_type=ToolType.OUTPUT, raw_plugin="DbFileOutput",
            table_name="main.dev.out", output_path="main.dev.out", upstream_ids=["3"],
        ),
    ])
    spec = naive_spec_from_workflow(wf, TARGET)
    step = spec.step_by_id("3")
    assert isinstance(step, JoinStep)
    assert step.left != step.right


def test_deep_chain_does_not_hit_recursion_limit() -> None:
    """A 1500-tool chain crashed both topological sorts with RecursionError
    before they went iterative — the hard failure big workflows hit on ingest.
    """
    nodes: list[Node] = [_read("0", "legacy.deep")]
    for i in range(1, 1500):
        nodes.append(_filter_node(str(i), str(i - 1), f"[c] > {i}"))
    nodes.append(_out("1500", "1499", "True", "main.dev.deep_out"))
    wf = _workflow(nodes)

    ordered = wf.topological_order()
    assert len(ordered) == len(nodes)
    assert [n.tool_id for n in ordered][:2] == ["0", "1"]

    spec = naive_spec_from_workflow(wf, TARGET)
    assert len(spec.steps) == len(nodes)
