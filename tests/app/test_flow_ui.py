from app.flow_ui import hero_html, pipeline_flow_html, stepper_html, workflow_canvas_html
from ingest.alteryx.ir import Node, ToolType, UnsupportedTool, Workflow


def test_stepper_marks_done_current_locked() -> None:
    html = stepper_html(["Upload", "Convert", "Deploy"], current=1, completed=1)
    assert html.count("ma-step done") == 1  # step 0
    assert html.count("ma-step current") == 1  # step 1
    assert html.count("ma-step locked") == 1  # step 2
    assert "ma-connect done" in html  # connector into the completed step


def test_hero_html_escapes_and_includes_css() -> None:
    html = hero_html("Title", "Sub <b>tag</b>")
    assert "ma-hero" in html
    assert "&lt;b&gt;" in html  # subtitle is escaped


def test_pipeline_flow_marks_status_classes_and_edges() -> None:
    html = pipeline_flow_html(
        [("Parse", "P", "done"), ("Convert", "C", "running"), ("Deploy", "D", "pending")]
    )
    assert "ma-node done" in html
    assert "ma-node running" in html
    assert "ma-node pending" in html
    # a done stage produces a done (green) edge to the next node
    assert "ma-edge done" in html
    # two edges between three nodes
    assert html.count("ma-edge") >= 2


def test_workflow_canvas_renders_tool_nodes_and_unsupported() -> None:
    wf = Workflow(
        source_file="x.yxmd",
        name="wf",
        nodes=[
            Node(tool_id="1", tool_type=ToolType.INPUT, raw_plugin="DbFileInput"),
            Node(tool_id="2", tool_type=ToolType.FILTER, raw_plugin="Filter", upstream_ids=["1"]),
            Node(tool_id="3", tool_type=ToolType.OUTPUT, raw_plugin="DbFileOutput", upstream_ids=["2"]),
        ],
        unsupported=[
            UnsupportedTool(tool_id="9", plugin="Foo.SharepointInput", reason="unsupported"),
        ],
    )
    html = workflow_canvas_html(wf)
    assert "Input" in html
    assert "Filter" in html
    assert "Output" in html
    # unsupported node rendered as dashed/manual (label truncated for layout)
    assert "ma-tnode unsupported" in html
    assert "Sharepoint" in html
    # category-colored nodes: input=blue, output=green, plus a legend
    assert "--c:#3b82f6" in html  # input
    assert "--c:#22c55e" in html  # output
    assert "ma-legend2" in html
    assert "ma-tbar" in html
    # professional SVG icons (currentColor), not emoji
    assert "<svg" in html
    assert 'stroke="currentColor"' in html
    # real graph wires between connected nodes, and a pan/zoom viewport
    assert 'class="ma-wire"' in html
    assert 'id="ma-vp"' in html
    assert "<script>" in html


def test_workflow_canvas_uses_actual_alteryx_positions() -> None:
    """Cards must sit at (scaled) .yxmd coordinates — mirroring the layout the
    Alteryx developer drew — not in a single flex line.
    """
    wf = Workflow(
        source_file="x.yxmd", name="wf",
        nodes=[
            Node(tool_id="1", tool_type=ToolType.INPUT, raw_plugin="I",
                 position={"x": 54.0, "y": 54.0}),
            Node(tool_id="2", tool_type=ToolType.FORMULA, raw_plugin="F",
                 upstream_ids=["1"], position={"x": 150.0, "y": 54.0}),
            Node(tool_id="3", tool_type=ToolType.OUTPUT, raw_plugin="O",
                 upstream_ids=["2"], position={"x": 150.0, "y": 200.0}),
        ],
    )
    html = workflow_canvas_html(wf)
    # node 1 at origin margin; node 3 lower than node 2 (y offset preserved)
    assert "left:48px;top:48px" in html
    assert "top:230px" in html  # (200-54)*1.25 + 48 = 230.5 -> 230 (banker's rounding)


def test_workflow_canvas_badges_secondary_anchors() -> None:
    from ingest.alteryx.ir import UpstreamEdge

    wf = Workflow(
        source_file="x.yxmd", name="wf",
        nodes=[
            Node(tool_id="1", tool_type=ToolType.INPUT, raw_plugin="I"),
            Node(tool_id="2", tool_type=ToolType.FILTER, raw_plugin="F", upstream_ids=["1"]),
            Node(
                tool_id="3", tool_type=ToolType.OUTPUT, raw_plugin="O", upstream_ids=["2"],
                upstream_edges=[UpstreamEdge(origin_id="2", origin_anchor="False")],
            ),
        ],
    )
    html = workflow_canvas_html(wf)
    # the False-stream wire gets an amber style and an F badge
    assert "ma-wire warn" in html
    assert ">F</text>" in html


def test_workflow_canvas_layered_fallback_without_positions() -> None:
    """Old IR without coordinates must still fan out by graph depth, giving
    each chain level its own column instead of one long line.
    """
    wf = Workflow(
        source_file="x.yxmd", name="wf",
        nodes=[
            Node(tool_id="a", tool_type=ToolType.INPUT, raw_plugin="I"),
            Node(tool_id="b", tool_type=ToolType.INPUT, raw_plugin="I"),
            Node(tool_id="c", tool_type=ToolType.UNION, raw_plugin="U", upstream_ids=["a", "b"]),
        ],
    )
    html = workflow_canvas_html(wf)
    # a and b share level 0 (same x, stacked y); c is one level right
    assert "left:48px" in html
    assert "left:224px" in html  # 1 * (118 + 58) + 48


def test_no_emoji_in_rendered_flow_html() -> None:
    import re

    wf = Workflow(
        source_file="x.yxmd", name="wf",
        nodes=[Node(tool_id="1", tool_type=ToolType.INPUT, raw_plugin="DbFileInput")],
    )
    emoji = re.compile("[\U0001F000-\U0001FAFF☀-➿]")
    assert not emoji.search(workflow_canvas_html(wf))
    assert not emoji.search(pipeline_flow_html([("Deploy", "<svg/>", "running")]))


def test_workflow_canvas_truncates_large_workflows() -> None:
    nodes = [
        Node(tool_id=str(i), tool_type=ToolType.FORMULA, raw_plugin="Formula")
        for i in range(80)
    ]
    wf = Workflow(source_file="x.yxmd", name="big", nodes=nodes)
    html = workflow_canvas_html(wf, max_nodes=60)
    assert "Showing first 60 of 80" in html
