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

def _svg(inner: str, size: int = 18) -> str:
    """Wrap SVG path content in a Feather/Lucide-style stroke icon (currentColor)."""
    return (
        f'<svg viewBox="0 0 24 24" width="{size}" height="{size}" fill="none" '
        'stroke="currentColor" stroke-width="2" stroke-linecap="round" '
        f'stroke-linejoin="round">{inner}</svg>'
    )


# Category SVG glyphs (monochrome, inherit the node's accent via currentColor).
_CATEGORY_SVG: dict[str, str] = {
    "input": _svg(
        '<ellipse cx="12" cy="5" rx="8" ry="3"/>'
        '<path d="M4 5v6c0 1.7 3.6 3 8 3s8-1.3 8-3V5"/>'
        '<path d="M4 11v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6"/>'
    ),
    "output": _svg(
        '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>'
        '<polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/>'
    ),
    "prep": _svg(
        '<line x1="4" y1="21" x2="4" y2="14"/><line x1="4" y1="10" x2="4" y2="3"/>'
        '<line x1="12" y1="21" x2="12" y2="12"/><line x1="12" y1="8" x2="12" y2="3"/>'
        '<line x1="20" y1="21" x2="20" y2="16"/><line x1="20" y1="12" x2="20" y2="3"/>'
        '<line x1="1" y1="14" x2="7" y2="14"/><line x1="9" y1="8" x2="15" y2="8"/>'
        '<line x1="17" y1="16" x2="23" y2="16"/>'
    ),
    "join": _svg(
        '<circle cx="18" cy="18" r="3"/><circle cx="6" cy="6" r="3"/>'
        '<path d="M6 21V9a9 9 0 0 0 9 9"/>'
    ),
    "transform": _svg(
        '<line x1="6" y1="20" x2="6" y2="14"/><line x1="12" y1="20" x2="12" y2="8"/>'
        '<line x1="18" y1="20" x2="18" y2="4"/>'
    ),
    "dev": _svg('<polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/>'),
    "manual": _svg(
        '<path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86'
        'a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/>'
        '<line x1="12" y1="17" x2="12.01" y2="17"/>'
    ),
}

# Migration stage glyphs, referenced by callers of pipeline_flow_html.
ICONS: dict[str, str] = {
    "upload": _svg(
        '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>'
        '<polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/>'
    ),
    "convert": _svg(
        '<polyline points="16 3 21 3 21 8"/><line x1="4" y1="20" x2="21" y2="3"/>'
        '<polyline points="21 16 21 21 16 21"/><line x1="15" y1="15" x2="21" y2="21"/>'
        '<line x1="4" y1="4" x2="9" y2="9"/>'
    ),
    "recommend": _svg(
        '<line x1="9" y1="18" x2="15" y2="18"/><line x1="10" y1="22" x2="14" y2="22"/>'
        '<path d="M15.09 14c.18-.98.65-1.74 1.41-2.5A4.65 4.65 0 0 0 18 8 6 6 0 0 0 6 8'
        'c0 1 .23 2.23 1.5 3.5A4.61 4.61 0 0 1 8.91 14"/>'
    ),
    "bundle": _svg(
        '<line x1="16.5" y1="9.4" x2="7.5" y2="4.21"/>'
        '<path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8'
        'a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/>'
        '<polyline points="3.27 6.96 12 12.01 20.73 6.96"/>'
        '<line x1="12" y1="22.08" x2="12" y2="12"/>'
    ),
    "deploy": _svg(
        '<polyline points="16 16 12 12 8 16"/><line x1="12" y1="12" x2="12" y2="21"/>'
        '<path d="M20.39 18.39A5 5 0 0 0 18 9h-1.26A8 8 0 1 0 3 16.3"/>'
    ),
    "verify": _svg(
        '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>'
        '<polyline points="22 4 12 14.01 9 11.01"/>'
    ),
}

_TOOL_LABEL: dict[ToolType, str] = {
    ToolType.INPUT: "Input",
    ToolType.OUTPUT: "Output",
    ToolType.SELECT: "Select",
    ToolType.FILTER: "Filter",
    ToolType.FORMULA: "Formula",
    ToolType.JOIN: "Join",
    ToolType.UNION: "Union",
    ToolType.SORT: "Sort",
    ToolType.UNIQUE: "Unique",
    ToolType.RECORD_ID: "Record ID",
    ToolType.CLEANSE: "Cleanse",
    ToolType.SUMMARIZE: "Summarize",
    ToolType.MACRO: "Macro",
    ToolType.MACRO_INPUT: "Macro In",
    ToolType.MACRO_OUTPUT: "Macro Out",
    ToolType.PYTHON: "Python",
    ToolType.FIND_REPLACE: "Find/Replace",
    ToolType.APPEND_FIELDS: "Append",
}

_TOOL_CATEGORY: dict[ToolType, str] = {
    ToolType.INPUT: "input",
    ToolType.MACRO_INPUT: "input",
    ToolType.OUTPUT: "output",
    ToolType.MACRO_OUTPUT: "output",
    ToolType.SELECT: "prep",
    ToolType.FILTER: "prep",
    ToolType.FORMULA: "prep",
    ToolType.SORT: "prep",
    ToolType.UNIQUE: "prep",
    ToolType.RECORD_ID: "prep",
    ToolType.CLEANSE: "prep",
    ToolType.JOIN: "join",
    ToolType.UNION: "join",
    ToolType.APPEND_FIELDS: "join",
    ToolType.FIND_REPLACE: "join",
    ToolType.SUMMARIZE: "transform",
    ToolType.PYTHON: "dev",
    ToolType.MACRO: "dev",
}
# category -> (accent color, label)
_CATEGORY_META: dict[str, tuple[str, str]] = {
    "input": ("#3b82f6", "Input"),
    "output": ("#22c55e", "Output"),
    "prep": ("#6366f1", "Prepare"),
    "join": ("#f59e0b", "Join / Blend"),
    "transform": ("#14b8a6", "Transform"),
    "dev": ("#a855f7", "Developer"),
    "manual": ("#ef4444", "Manual"),
}

_STYLE = """
<style>
.ma-flow{display:flex;flex-wrap:wrap;align-items:stretch;gap:0;padding:14px 4px;font-family:
ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;}
.ma-node{display:flex;align-items:center;gap:10px;min-width:150px;padding:12px 14px;border-radius:12px;
background:rgba(127,127,127,0.08);border:1px solid rgba(127,127,127,0.28);position:relative;
transition:box-shadow .2s ease,border-color .2s ease;}
.ma-node .ma-ico{width:34px;height:34px;border-radius:9px;display:flex;align-items:center;
justify-content:center;background:rgba(127,127,127,0.14);flex:none;color:inherit;}
.ma-node .ma-ico svg{display:block;}
.ma-node.running .ma-ico{color:#f5a623;}
.ma-node.done .ma-ico{color:#2ecc71;}
.ma-node.error .ma-ico{color:#ff5c5c;}
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
.ma-canvaswrap{border:1px solid rgba(127,127,127,0.2);border-radius:16px;padding:6px 12px 4px;
background:linear-gradient(180deg,rgba(127,127,127,0.04),transparent);}
.ma-canvas{display:flex;flex-wrap:wrap;align-items:center;gap:0;padding:16px 4px;}
.ma-tnode{position:relative;display:flex;flex-direction:column;align-items:center;gap:5px;
min-width:96px;padding:13px 11px 9px;border-radius:13px;background:rgba(127,127,127,0.05);
border:1px solid rgba(127,127,127,0.22);overflow:hidden;
transition:transform .14s ease,box-shadow .14s ease,border-color .14s ease;}
.ma-tnode:hover{transform:translateY(-3px);box-shadow:0 8px 20px rgba(0,0,0,0.14);
border-color:var(--c);}
.ma-tnode .ma-tbar{position:absolute;top:0;left:0;right:0;height:4px;background:var(--c);}
.ma-tnode .ma-tico{width:38px;height:38px;border-radius:50%;display:flex;align-items:center;
justify-content:center;color:var(--c);
background:color-mix(in srgb, var(--c) 15%, transparent);}
.ma-tnode .ma-tico svg{display:block;}
.ma-tnode .ma-tlabel{font-size:0.74rem;font-weight:700;text-align:center;line-height:1.05;}
.ma-tnode .ma-tid{font-size:0.6rem;opacity:0.5;font-weight:600;letter-spacing:0.02em;}
.ma-tnode.unsupported{border-style:dashed;border-color:var(--c);}
.ma-conn{align-self:center;flex:none;width:26px;height:2px;margin:0 -1px;
background:rgba(127,127,127,0.38);position:relative;}
.ma-conn::after{content:"";position:absolute;right:-1px;top:-3px;border:4px solid transparent;
border-left-color:rgba(127,127,127,0.55);}
.ma-legend2{display:flex;flex-wrap:wrap;gap:14px;padding:8px 8px 12px;font-size:0.76rem;opacity:0.85;}
.ma-key{display:inline-flex;align-items:center;gap:6px;font-weight:600;}
.ma-key i{width:11px;height:11px;border-radius:3px;display:inline-block;}
.ma-legend{font-size:0.74rem;opacity:0.65;padding:2px 8px 8px;}
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


def _tool_node_html(icon: str, label: str, tool_id: str, color: str, unsupported: bool) -> str:
    cls = "ma-tnode unsupported" if unsupported else "ma-tnode"
    tag = "manual" if unsupported else f"#{html.escape(tool_id)}"
    return (
        f'<div class="{cls}" style="--c:{color}"><span class="ma-tbar"></span>'
        f'<div class="ma-tico">{icon}</div>'
        f'<div class="ma-tlabel">{html.escape(label)}</div>'
        f'<div class="ma-tid">{tag}</div></div>'
    )


# Node card geometry (px, unscaled canvas units).
_CARD_W, _CARD_H = 118, 82
# Alteryx tools sit ~96px apart at ~60px wide; cards are bigger, so stretch.
_POS_SCALE_X, _POS_SCALE_Y = 1.55, 1.25
_MARGIN = 48

# Origin-anchor -> short wire badge. Anchors that mean "the one normal
# output" get no badge; multi-output anchors are the ones worth showing.
_ANCHOR_BADGE = {
    "true": "T", "false": "F",
    "left": "L", "join": "J", "right": "R",
    "unique": "U", "dup": "D",
}
_ANCHOR_WARN = {"false", "left", "right", "dup"}  # secondary streams: amber wire

_CANVAS_STYLE = """
<style>
*{box-sizing:border-box;margin:0;padding:0;}
body{font-family:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
background:transparent;}
.ma-canvaswrap{border:1px solid rgba(127,127,127,0.25);border-radius:16px;overflow:hidden;
background:
 radial-gradient(circle at 1px 1px, rgba(127,127,127,0.18) 1px, transparent 0) 0 0/22px 22px,
 linear-gradient(180deg, rgba(127,127,127,0.05), transparent);}
.ma-topbar{display:flex;align-items:center;gap:12px;padding:9px 14px;
border-bottom:1px solid rgba(127,127,127,0.18);font-size:0.78rem;
background:rgba(127,127,127,0.05);}
.ma-stats{display:flex;gap:14px;font-weight:600;opacity:0.85;}
.ma-tools{margin-left:auto;display:flex;gap:6px;}
.ma-tools button{border:1px solid rgba(127,127,127,0.35);background:rgba(127,127,127,0.08);
border-radius:7px;min-width:30px;height:26px;padding:0 8px;font-weight:700;cursor:pointer;
color:inherit;font-size:0.78rem;}
.ma-tools button:hover{border-color:#6366f1;color:#6366f1;}
.ma-viewport{position:relative;height:520px;overflow:hidden;cursor:grab;}
.ma-viewport.dragging{cursor:grabbing;}
.ma-world{position:absolute;top:0;left:0;transform-origin:0 0;}
.ma-edges{position:absolute;top:0;left:0;overflow:visible;pointer-events:none;}
.ma-wire{fill:none;stroke:rgba(127,127,127,0.55);stroke-width:2;}
.ma-wire.warn{stroke:#f59e0b;}
.ma-wire.manual{stroke:#ef4444;stroke-dasharray:5 4;}
.ma-badge{font-size:11px;font-weight:800;}
.ma-badge circle{fill:#f59e0b;}
.ma-badge.plain circle{fill:rgba(127,127,127,0.75);}
.ma-badge text{fill:#fff;}
.ma-tnode{position:absolute;display:flex;flex-direction:column;align-items:center;gap:4px;
width:118px;height:82px;padding:10px 8px 6px;border-radius:12px;
background:rgba(30,30,36,0.04);backdrop-filter:blur(2px);
border:1px solid rgba(127,127,127,0.35);overflow:hidden;
box-shadow:0 2px 8px rgba(0,0,0,0.10);}
@media (prefers-color-scheme: dark){.ma-tnode{background:rgba(30,30,36,0.85);}}
.ma-tnode:hover{border-color:var(--c);box-shadow:0 6px 18px rgba(0,0,0,0.22);z-index:5;}
.ma-tnode .ma-tbar{position:absolute;top:0;left:0;right:0;height:4px;background:var(--c);}
.ma-tnode .ma-tico{width:30px;height:30px;border-radius:50%;display:flex;align-items:center;
justify-content:center;color:var(--c);background:color-mix(in srgb, var(--c) 16%, transparent);}
.ma-tnode .ma-tico svg{display:block;width:17px;height:17px;}
.ma-tnode .ma-tlabel{font-size:0.68rem;font-weight:700;text-align:center;line-height:1.02;
white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:104px;}
.ma-tnode .ma-tid{font-size:0.58rem;opacity:0.55;font-weight:600;}
.ma-tnode.unsupported{border-style:dashed;border-color:var(--c);}
.ma-legend2{display:flex;flex-wrap:wrap;gap:13px;padding:8px 14px;font-size:0.74rem;
opacity:0.9;border-top:1px solid rgba(127,127,127,0.18);background:rgba(127,127,127,0.05);}
.ma-key{display:inline-flex;align-items:center;gap:6px;font-weight:600;}
.ma-key i{width:11px;height:11px;border-radius:3px;display:inline-block;}
.ma-legend{font-size:0.72rem;opacity:0.65;padding:6px 14px;}
</style>
"""

_CANVAS_SCRIPT = """
<script>
(function(){
  var vp=document.getElementById('ma-vp'),world=document.getElementById('ma-world');
  var W=parseFloat(world.dataset.w),H=parseFloat(world.dataset.h);
  var s=1,tx=0,ty=0,drag=null;
  function apply(){world.style.transform='translate('+tx+'px,'+ty+'px) scale('+s+')';}
  function fit(){
    var r=vp.getBoundingClientRect();
    s=Math.min(r.width/W,r.height/H,1);
    tx=(r.width-W*s)/2; ty=(r.height-H*s)/2;
    if(ty<0) ty=0;
    apply();
  }
  function zoomAt(f,cx,cy){
    var ns=Math.min(Math.max(s*f,0.08),2.5);
    tx=cx-(cx-tx)*(ns/s); ty=cy-(cy-ty)*(ns/s); s=ns; apply();
  }
  vp.addEventListener('wheel',function(e){
    e.preventDefault();
    var r=vp.getBoundingClientRect();
    zoomAt(e.deltaY<0?1.13:0.885,e.clientX-r.left,e.clientY-r.top);
  },{passive:false});
  vp.addEventListener('pointerdown',function(e){
    drag={x:e.clientX-tx,y:e.clientY-ty}; vp.classList.add('dragging');
    vp.setPointerCapture(e.pointerId);
  });
  vp.addEventListener('pointermove',function(e){
    if(drag){tx=e.clientX-drag.x; ty=e.clientY-drag.y; apply();}
  });
  vp.addEventListener('pointerup',function(){drag=null;vp.classList.remove('dragging');});
  document.getElementById('ma-zin').onclick=function(){var r=vp.getBoundingClientRect();zoomAt(1.25,r.width/2,r.height/2);};
  document.getElementById('ma-zout').onclick=function(){var r=vp.getBoundingClientRect();zoomAt(0.8,r.width/2,r.height/2);};
  document.getElementById('ma-fit').onclick=fit;
  document.getElementById('ma-100').onclick=function(){s=1;tx=12;ty=12;apply();};
  vp.addEventListener('dblclick',fit);
  fit();
})();
</script>
"""


def _canvas_positions(
    workflow: Workflow, elements: list[tuple[str, list[str], dict[str, float]]]
) -> dict[str, tuple[float, float]]:
    """Place every element, preferring the real Alteryx canvas coordinates.

    Nodes carry the .yxmd's own GuiSettings x/y — using them reproduces the
    layout the Alteryx developer actually drew. Elements without coordinates
    (older persisted IR) fall back to a layered left-to-right DAG layout, so
    big workflows still fan out instead of collapsing into a line.
    """
    positioned = [(eid, pos) for eid, _, pos in elements if pos.get("x") or pos.get("y")]
    coords: dict[str, tuple[float, float]] = {}

    if len(positioned) >= max(2, len(elements) // 2):
        min_x = min(p["x"] for _, p in positioned)
        min_y = min(p["y"] for _, p in positioned)
        taken: dict[tuple[int, int], int] = {}
        for eid, pos in positioned:
            x = (pos["x"] - min_x) * _POS_SCALE_X + _MARGIN
            y = (pos["y"] - min_y) * _POS_SCALE_Y + _MARGIN
            cell = (int(x // (_CARD_W + 10)), int(y // (_CARD_H + 10)))
            bump = taken.get(cell, 0)
            taken[cell] = bump + 1
            coords[eid] = (x, y + bump * (_CARD_H + 14))

    # Layered fallback for anything unplaced (level = longest path from a source).
    unplaced = [e for e in elements if e[0] not in coords]
    if unplaced:
        level: dict[str, int] = {}
        for eid, upstream, _ in elements:  # elements come topologically ordered
            level[eid] = max((level.get(u, -1) + 1 for u in upstream), default=0)
        base_y = max((y for _, y in coords.values()), default=0.0)
        base_y += (_CARD_H + 40) if coords else _MARGIN
        per_level: dict[int, int] = {}
        for eid, _upstream, _ in unplaced:
            lvl = level.get(eid, 0)
            row = per_level.get(lvl, 0)
            per_level[lvl] = row + 1
            coords[eid] = (
                lvl * (_CARD_W + 58) + _MARGIN,
                base_y + row * (_CARD_H + 22),
            )
    return coords


def _wire_path(x1: float, y1: float, x2: float, y2: float) -> str:
    dx = max(38.0, min(150.0, abs(x2 - x1) / 2))
    return f"M {x1:.0f} {y1:.0f} C {x1 + dx:.0f} {y1:.0f} {x2 - dx:.0f} {y2:.0f} {x2:.0f} {y2:.0f}"


def workflow_canvas_html(workflow: Workflow, max_nodes: int = 400) -> str:
    """Render the workflow as a positioned canvas mirroring the Alteryx layout.

    Node cards sit at the .yxmd's own coordinates, connections draw as SVG
    wires between them (badged with the origin anchor - T/F, L/J/R, U/D -
    when it is a secondary output), and the viewport pans and zooms, so a
    500-tool workflow reads like the original canvas instead of a flex line.
    Returns a self-contained document for st.components.v1.html.
    """
    ordered = workflow.topological_order()
    truncated = len(ordered) > max_nodes
    ordered = ordered[:max_nodes]
    shown_ids = {n.tool_id for n in ordered}

    elements: list[tuple[str, list[str], dict[str, float]]] = [
        (n.tool_id, n.upstream_ids, n.position) for n in ordered
    ]
    elements += [(u.tool_id, u.upstream_ids, u.position) for u in workflow.unsupported]
    coords = _canvas_positions(workflow, elements)

    world_w = max((x for x, _ in coords.values()), default=0.0) + _CARD_W + _MARGIN
    world_h = max((y for _, y in coords.values()), default=0.0) + _CARD_H + _MARGIN

    unsupported_ids = {u.tool_id for u in workflow.unsupported}
    anchor_of: dict[tuple[str, str], str] = {}
    for n in ordered:
        for e in n.upstream_edges:
            anchor_of[(e.origin_id, n.tool_id)] = e.origin_anchor

    wires: list[str] = []
    edge_count = 0
    for eid, upstream, _pos in elements:
        if eid not in coords:
            continue
        x2, y2 = coords[eid]
        for origin in upstream:
            if origin not in coords or (origin not in shown_ids and origin not in unsupported_ids):
                continue
            x1, y1 = coords[origin]
            path = _wire_path(x1 + _CARD_W, y1 + _CARD_H / 2, x2, y2 + _CARD_H / 2)
            anchor = anchor_of.get((origin, eid), "").lower()
            badge = _ANCHOR_BADGE.get(anchor, "")
            cls = "ma-wire"
            if origin in unsupported_ids or eid in unsupported_ids:
                cls += " manual"
            elif anchor in _ANCHOR_WARN:
                cls += " warn"
            wires.append(f'<path class="{cls}" d="{path}"/>')
            if badge:
                mx = (x1 + _CARD_W + x2) / 2
                my = (y1 + y2 + _CARD_H) / 2
                warn = "" if anchor in _ANCHOR_WARN else " plain"
                wires.append(
                    f'<g class="ma-badge{warn}"><circle cx="{mx:.0f}" cy="{my:.0f}" r="9"/>'
                    f'<text x="{mx:.0f}" y="{my + 4:.0f}" text-anchor="middle">{badge}</text></g>'
                )
            edge_count += 1

    cards: list[str] = []
    used_categories: set[str] = set()
    for node in ordered:
        if node.tool_id not in coords:
            continue
        x, y = coords[node.tool_id]
        label = _TOOL_LABEL.get(node.tool_type, node.tool_type.value)
        category = _TOOL_CATEGORY.get(node.tool_type, "prep")
        used_categories.add(category)
        color = _CATEGORY_META[category][0]
        tip = html.escape(
            (node.annotation or node.filter_expression or node.raw_plugin or "").strip()[:220]
        )
        cards.append(
            f'<div class="ma-tnode" style="--c:{color};left:{x:.0f}px;top:{y:.0f}px" '
            f'title="{tip}"><span class="ma-tbar"></span>'
            f'<div class="ma-tico">{_CATEGORY_SVG[category]}</div>'
            f'<div class="ma-tlabel">{html.escape(label)}</div>'
            f'<div class="ma-tid">#{html.escape(node.tool_id)}</div></div>'
        )
    for u in workflow.unsupported:
        if u.tool_id not in coords:
            continue
        x, y = coords[u.tool_id]
        used_categories.add("manual")
        color = _CATEGORY_META["manual"][0]
        short = u.plugin.split(".")[-1].split("\\")[-1]
        cards.append(
            f'<div class="ma-tnode unsupported" style="--c:{color};left:{x:.0f}px;top:{y:.0f}px" '
            f'title="{html.escape(u.plugin)} — {html.escape(u.reason)}">'
            f'<span class="ma-tbar"></span>'
            f'<div class="ma-tico">{_CATEGORY_SVG["manual"]}</div>'
            f'<div class="ma-tlabel">{html.escape(short[:16])}</div>'
            f'<div class="ma-tid">#{html.escape(u.tool_id)}</div></div>'
        )

    keys = [
        f'<span class="ma-key"><i style="background:{color}"></i>{label}</span>'
        for cat, (color, label) in _CATEGORY_META.items()
        if cat in used_categories
    ]
    stats = (
        f'<div class="ma-stats"><span>{len(ordered)} tools</span>'
        f"<span>{edge_count} connections</span>"
        f"<span>{len(workflow.unsupported)} manual</span></div>"
    )
    toolbar = (
        '<div class="ma-tools">'
        '<button id="ma-zout" title="Zoom out">&minus;</button>'
        '<button id="ma-zin" title="Zoom in">+</button>'
        '<button id="ma-fit" title="Fit to view">Fit</button>'
        '<button id="ma-100" title="Actual size">1:1</button></div>'
    )
    note = (
        f'<div class="ma-legend">Showing first {max_nodes} of {len(workflow.nodes)} '
        "tools — the full flow deploys regardless.</div>"
        if truncated
        else ""
    )

    return (
        _CANVAS_STYLE
        + '<div class="ma-canvaswrap">'
        + f'<div class="ma-topbar">{stats}{toolbar}</div>'
        + '<div class="ma-viewport" id="ma-vp">'
        + f'<div class="ma-world" id="ma-world" data-w="{world_w:.0f}" data-h="{world_h:.0f}" '
        + f'style="width:{world_w:.0f}px;height:{world_h:.0f}px">'
        + f'<svg class="ma-edges" width="{world_w:.0f}" height="{world_h:.0f}">'
        + "".join(wires)
        + "</svg>"
        + "".join(cards)
        + "</div></div>"
        + '<div class="ma-legend2">'
        + "".join(keys)
        + "</div>"
        + note
        + "</div>"
        + _CANVAS_SCRIPT
    )
