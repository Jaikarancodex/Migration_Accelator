"""n8n-style flow visualization for the Streamlit app.

Renders the migration as a node-and-edge pipeline (like n8n's canvas): the
migration stages as a live status flow that lights up as each stage runs,
and a parsed Alteryx workflow as connected tool nodes. Pure HTML/CSS so it
works inside `st.markdown(..., unsafe_allow_html=True)` and updates live via
an `st.empty()` placeholder.
"""

from __future__ import annotations

import html
from typing import Literal

from ingest.alteryx.ir import ToolType, Workflow

Status = Literal["pending", "running", "done", "error"]

_STATUS_COLOR: dict[Status, str] = {
    "pending": "#9aa0a6",
    "running": "#f5a623",
    "done": "#2ecc71",
    "error": "#ff5c5c",
}

# Tool-type -> (emoji icon, short label) for the workflow canvas.
_TOOL_ICON: dict[ToolType, tuple[str, str]] = {
    ToolType.INPUT: ("\U0001f4e5", "Input"),
    ToolType.OUTPUT: ("\U0001f4e4", "Output"),
    ToolType.SELECT: ("\U0001f9ee", "Select"),
    ToolType.FILTER: ("\U0001f50d", "Filter"),
    ToolType.FORMULA: ("\U0001f9ee", "Formula"),
    ToolType.JOIN: ("\U0001f517", "Join"),
    ToolType.UNION: ("➕", "Union"),
    ToolType.SORT: ("↕️", "Sort"),
    ToolType.UNIQUE: ("\U0001f194", "Unique"),
    ToolType.RECORD_ID: ("\U0001f522", "Record ID"),
    ToolType.CLEANSE: ("\U0001f9f9", "Cleanse"),
    ToolType.SUMMARIZE: ("\U0001f4ca", "Summarize"),
    ToolType.MACRO: ("\U0001f9e9", "Macro"),
    ToolType.MACRO_INPUT: ("\U0001f4e5", "Macro In"),
    ToolType.MACRO_OUTPUT: ("\U0001f4e4", "Macro Out"),
    ToolType.PYTHON: ("\U0001f40d", "Python"),
    ToolType.FIND_REPLACE: ("\U0001f501", "Find/Replace"),
    ToolType.APPEND_FIELDS: ("\U0001f9f7", "Append"),
}

_STYLE = """
<style>
.ma-flow{display:flex;flex-wrap:wrap;align-items:stretch;gap:0;padding:14px 4px;font-family:
ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;}
.ma-node{display:flex;align-items:center;gap:10px;min-width:150px;padding:12px 14px;border-radius:12px;
background:rgba(127,127,127,0.08);border:1px solid rgba(127,127,127,0.28);position:relative;
transition:box-shadow .2s ease,border-color .2s ease;}
.ma-node .ma-ico{width:34px;height:34px;border-radius:9px;display:flex;align-items:center;
justify-content:center;font-size:18px;background:rgba(127,127,127,0.14);flex:none;}
.ma-node .ma-title{font-weight:600;font-size:0.92rem;line-height:1.1;}
.ma-node .ma-sub{font-size:0.72rem;opacity:0.7;text-transform:uppercase;letter-spacing:0.04em;margin-top:2px;}
.ma-node.pending{opacity:0.62;}
.ma-node.running{border-color:#f5a623;box-shadow:0 0 0 3px rgba(245,166,35,0.22);animation:ma-pulse 1.1s ease-in-out infinite;}
.ma-node.done{border-color:rgba(46,204,113,0.6);}
.ma-node.error{border-color:#ff5c5c;box-shadow:0 0 0 3px rgba(255,92,92,0.18);}
.ma-node .ma-dot{position:absolute;top:9px;right:9px;width:9px;height:9px;border-radius:50%;}
@keyframes ma-pulse{0%,100%{box-shadow:0 0 0 3px rgba(245,166,35,0.22);}50%{box-shadow:0 0 0 6px rgba(245,166,35,0.10);}}
.ma-edge{align-self:center;width:34px;height:3px;border-radius:2px;background:rgba(127,127,127,0.35);
position:relative;flex:none;margin:0 2px;}
.ma-edge.done{background:#2ecc71;}
.ma-edge.running{background:linear-gradient(90deg,#2ecc71 0%,#f5a623 100%);}
.ma-edge::after{content:"";position:absolute;right:-2px;top:-3px;border:4px solid transparent;
border-left-color:inherit;}
.ma-edge.done::after{border-left-color:#2ecc71;}
.ma-canvas{display:flex;flex-wrap:wrap;align-items:center;gap:0;padding:14px 4px;}
.ma-tnode{display:flex;flex-direction:column;align-items:center;gap:4px;min-width:74px;padding:10px 8px;
border-radius:11px;background:rgba(127,127,127,0.08);border:1px solid rgba(127,127,127,0.28);}
.ma-tnode.unsupported{border-style:dashed;border-color:#f5a623;opacity:0.85;}
.ma-tnode .ma-tico{font-size:20px;}
.ma-tnode .ma-tlabel{font-size:0.72rem;font-weight:600;text-align:center;line-height:1.05;}
.ma-tnode .ma-tid{font-size:0.62rem;opacity:0.55;}
.ma-legend{font-size:0.74rem;opacity:0.7;padding:2px 6px 10px;}
</style>
"""


ACCENT = "#6366f1"

GLOBAL_CSS = f"""
<style>
:root {{ --ma-accent: {ACCENT}; }}
/* Roomier main column, tighter default chrome */
.block-container {{ padding-top: 2.2rem; max-width: 1150px; }}
/* Primary buttons in the brand accent */
.stButton > button[kind="primary"], .stButton > button[data-testid="baseButton-primary"] {{
  background: var(--ma-accent); border-color: var(--ma-accent);
}}
.stButton > button {{ border-radius: 9px; font-weight: 600; }}
/* Tabs a touch larger and clearer */
.stTabs [data-baseweb="tab"] {{ font-weight: 600; }}
.ma-hero {{
  border-radius: 16px; padding: 22px 26px; margin-bottom: 8px;
  background: linear-gradient(120deg, rgba(99,102,241,0.16), rgba(99,102,241,0.03));
  border: 1px solid rgba(99,102,241,0.30);
}}
.ma-hero h1 {{ font-size: 1.5rem; margin: 0 0 4px; letter-spacing: -0.01em; }}
.ma-hero p {{ margin: 0; opacity: 0.78; font-size: 0.95rem; }}
.ma-pill {{ display:inline-flex; align-items:center; gap:6px; padding:3px 11px; border-radius:999px;
  font-size:0.76rem; font-weight:600; border:1px solid rgba(127,127,127,0.3); }}
.ma-steps {{ display:flex; align-items:flex-start; gap:0; padding:16px 2px 6px; font-family:
  ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; }}
.ma-step {{ display:flex; flex-direction:column; align-items:center; gap:7px; flex:1; min-width:0; }}
.ma-step .ma-circle {{ width:38px; height:38px; border-radius:50%; display:flex; align-items:center;
  justify-content:center; font-weight:700; font-size:0.95rem; border:2px solid rgba(127,127,127,0.35);
  background:rgba(127,127,127,0.06); color:inherit; transition:all .2s ease; }}
.ma-step .ma-caption {{ font-size:0.78rem; font-weight:600; text-align:center; opacity:0.7; }}
.ma-step.current .ma-circle {{ border-color:var(--ma-accent); color:var(--ma-accent);
  box-shadow:0 0 0 4px rgba(99,102,241,0.18); }}
.ma-step.current .ma-caption {{ opacity:1; color:var(--ma-accent); }}
.ma-step.done .ma-circle {{ background:var(--ma-accent); border-color:var(--ma-accent); color:#fff; }}
.ma-step.done .ma-caption {{ opacity:0.95; }}
.ma-step.locked {{ opacity:0.5; }}
.ma-connect {{ height:2px; flex:1; margin-top:19px; background:rgba(127,127,127,0.3); align-self:flex-start; }}
.ma-connect.done {{ background:var(--ma-accent); }}
</style>
"""


def hero_html(title: str, subtitle: str) -> str:
    return (
        GLOBAL_CSS
        + f'<div class="ma-hero"><h1>{html.escape(title)}</h1>'
        f"<p>{html.escape(subtitle)}</p></div>"
    )


def stepper_html(labels: list[str], current: int, completed: int) -> str:
    """Horizontal progress stepper. Steps [0, completed) are done; `current` is active."""
    parts = [GLOBAL_CSS, '<div class="ma-steps">']
    for i, label in enumerate(labels):
        state = "done" if i < completed else ("current" if i == current else "locked")
        inner = "✓" if state == "done" else str(i + 1)
        parts.append(
            f'<div class="ma-step {state}"><div class="ma-circle">{inner}</div>'
            f'<div class="ma-caption">{html.escape(label)}</div></div>'
        )
        if i < len(labels) - 1:
            parts.append(f'<div class="ma-connect {"done" if i < completed else ""}"></div>')
    parts.append("</div>")
    return "".join(parts)


def _stage_node(name: str, icon: str, status: Status) -> str:
    color = _STATUS_COLOR[status]
    sub = {"pending": "waiting", "running": "running…", "done": "done", "error": "failed"}[status]
    return (
        f'<div class="ma-node {status}">'
        f'<div class="ma-ico">{icon}</div>'
        f'<div><div class="ma-title">{html.escape(name)}</div>'
        f'<div class="ma-sub" style="color:{color}">{sub}</div></div>'
        f'<span class="ma-dot" style="background:{color}"></span></div>'
    )


def pipeline_flow_html(stages: list[tuple[str, str, Status]]) -> str:
    """Render the migration stage pipeline. `stages` = [(name, icon, status)]."""
    parts = [_STYLE, '<div class="ma-flow">']
    for i, (name, icon, status) in enumerate(stages):
        parts.append(_stage_node(name, icon, status))
        if i < len(stages) - 1:
            edge = "done" if status == "done" else ("running" if status == "running" else "")
            parts.append(f'<div class="ma-edge {edge}"></div>')
    parts.append("</div>")
    return "".join(parts)


def workflow_canvas_html(workflow: Workflow, max_nodes: int = 60) -> str:
    """Render the parsed Alteryx workflow as connected tool nodes (n8n-style)."""
    ordered = workflow.topological_order()
    truncated = len(ordered) > max_nodes
    ordered = ordered[:max_nodes]

    parts = [_STYLE, '<div class="ma-canvas">']
    for i, node in enumerate(ordered):
        icon, label = _TOOL_ICON.get(node.tool_type, ("⚙️", node.tool_type.value))
        parts.append(
            f'<div class="ma-tnode"><div class="ma-tico">{icon}</div>'
            f'<div class="ma-tlabel">{html.escape(label)}</div>'
            f'<div class="ma-tid">#{html.escape(node.tool_id)}</div></div>'
        )
        if i < len(ordered) - 1:
            parts.append('<div class="ma-edge done"></div>')
    for u in workflow.unsupported:
        short = u.plugin.split(".")[-1].split("\\")[-1]
        parts.append(
            f'<div class="ma-tnode unsupported"><div class="ma-tico">⚠️</div>'
            f'<div class="ma-tlabel">{html.escape(short[:12])}</div>'
            f'<div class="ma-tid">manual</div></div>'
        )
    parts.append("</div>")
    if truncated:
        parts.append(
            f'<div class="ma-legend">Showing first {max_nodes} of {len(workflow.nodes)} '
            "tools — the full flow deploys regardless.</div>"
        )
    parts.append(
        '<div class="ma-legend">Solid = converted automatically &nbsp;·&nbsp; '
        "Dashed amber = needs manual follow-up (connector / unknown tool).</div>"
    )
    return "".join(parts)
