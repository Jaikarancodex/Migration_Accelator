from app.flow_ui import pipeline_flow_html, workflow_canvas_html
from ingest.alteryx.ir import Node, ToolType, UnsupportedTool, Workflow


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


def test_workflow_canvas_truncates_large_workflows() -> None:
    nodes = [
        Node(tool_id=str(i), tool_type=ToolType.FORMULA, raw_plugin="Formula")
        for i in range(80)
    ]
    wf = Workflow(source_file="x.yxmd", name="big", nodes=nodes)
    html = workflow_canvas_html(wf, max_nodes=60)
    assert "Showing first 60 of 80" in html
