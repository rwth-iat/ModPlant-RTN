"""Notebook-facing helpers for ``ModPlant-RTN.ipynb``.

This module keeps reusable setup, plotting, and display code out of the
notebook so the notebook can focus on user-provided Module configuration,
recipe information, solving, and result inspection.
"""

from pathlib import Path
from html import escape
import importlib
import math
import sys
import pandas as pd
from IPython.display import HTML, display

_scripts_dir = Path(__file__).resolve().parent
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))
_rtn_root = _scripts_dir.parent
if str(_rtn_root) not in sys.path:
    sys.path.insert(0, str(_rtn_root))

import recipe_ir
import isa88_recipe
import cp_sat_planner
import schedule_validator
import recipe_to_rtn
import rtn
import modplant_rtn.visualization as app_visualization

build_recipe_ir = recipe_ir.build_recipe_ir
recipe_spec_from_legacy_order = recipe_ir.recipe_spec_from_legacy_order
auto_enrich_recipe_spec = recipe_ir.auto_enrich_recipe_spec
recipe_ir_to_dict = recipe_ir.recipe_ir_to_dict
save_general_recipe_xml_from_ir = isa88_recipe.save_general_recipe_xml_from_ir
save_recipe_ir_json = isa88_recipe.save_recipe_ir_json
parse_general_recipe_xml_to_ir = isa88_recipe.parse_general_recipe_xml_to_ir
PlannerConfig = cp_sat_planner.PlannerConfig
solve_recipe_ir_with_cp_sat = cp_sat_planner.solve_recipe_ir_with_cp_sat
solve_rtn_with_cp_sat = cp_sat_planner.solve_rtn_with_cp_sat
validate_schedule = schedule_validator.validate_schedule
# ``replay_plan_with_cpn`` is the legacy alias of ``validate_schedule``;
# accessing it emits a DeprecationWarning. Kept here only for notebooks that
# still import the old name.
replay_plan_with_cpn = schedule_validator.replay_plan_with_cpn
recipe_ir_to_rtn = recipe_to_rtn.recipe_ir_to_rtn
rtn_to_recipe_ir = recipe_to_rtn.rtn_to_recipe_ir
RTNModel = rtn.RTNModel

try:
    import ace_tools_open as tools
except Exception:
    tools = None


def display_df(name, df):
    if tools is not None:
        tools.display_dataframe_to_user(name=name, dataframe=df)
    else:
        print(f"\n{name}")
        display(df)


def plan_dataframe(session_or_result, cost_decimals=2):
    """Build the Notebook plan table from the App's public result rows.

    ``OptimizationSession.plan_rows`` is preferred because it also carries the
    App's OPC UA endpoint and namespace enrichment.  No fixed column list is
    maintained here, so future result fields appear in notebooks automatically.
    """
    rows = (
        session_or_result.plan_rows
        if hasattr(session_or_result, "plan_rows")
        else session_or_result.to_rows()
    )
    df = pd.DataFrame(rows)
    cost_columns = [column for column in df.columns if "Cost" in column]
    if cost_columns:
        df[cost_columns] = df[cost_columns].round(int(cost_decimals))
    return df


def _short_text(value, limit=34):
    text = "" if value is None else str(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _format_seconds(value):
    try:
        return f"{float(value):.0f}s"
    except Exception:
        return ""


def _recipe_layout(ir):
    ordered = ir.topological_nodes()
    successors = ir.successors()
    levels = {node.id: 0 for node in ordered}
    for node in ordered:
        for dst in successors.get(node.id, []):
            levels[dst] = max(levels.get(dst, 0), levels[node.id] + 1)

    by_level = {}
    topo_index = {node.id: idx for idx, node in enumerate(ordered)}
    for node in ordered:
        by_level.setdefault(levels[node.id], []).append(node.id)
    for node_ids in by_level.values():
        node_ids.sort(key=lambda node_id: topo_index[node_id])

    node_w, node_h, x_gap, y_gap = 220, 146, 86, 36
    margin, title_h = 28, 56
    positions = {}
    for level in sorted(by_level):
        for row, node_id in enumerate(by_level[level]):
            positions[node_id] = (margin + level * (node_w + x_gap), title_h + margin + row * (node_h + y_gap))

    max_level = max(by_level) if by_level else 0
    max_rows = max((len(nodes) for nodes in by_level.values()), default=1)
    width = margin * 2 + (max_level + 1) * node_w + max_level * x_gap
    height = title_h + margin * 2 + max_rows * node_h + (max_rows - 1) * y_gap
    return positions, width, height, node_w, node_h


def _node_style(node):
    styles = {
        "dose": ("#e8f7ef", "#2c9f67"),
        "mix": ("#eaf1ff", "#3c6fd1"),
        "usage": ("#fff3df", "#d28a22"),
        "settling": ("#f1f4f8", "#6c7a89"),
        "separation": ("#f9e8ee", "#c44f7d"),
        "and_split": ("#e8fbfb", "#138a8a"),
        "and_join": ("#e8fbfb", "#138a8a"),
        "xor_split": ("#f5ebff", "#8755c8"),
        "xor_join": ("#f5ebff", "#8755c8"),
        "or_split": ("#fff4d6", "#c47b00"),
        "or_join": ("#fff4d6", "#c47b00"),
        "loop_region": ("#e7f5ff", "#1971c2"),
        "start": ("#ecfdf3", "#16803c"),
        "end": ("#fff1f2", "#be123c"),
    }
    return styles.get(node.node_type, ("#f7f7f7", "#777777"))


def _node_lines(node, operation_meta=None):
    lines = [_short_text(node.name or node.id, 28), node.id]
    if node.control_node_type:
        lines.append(node.control_node_type)
    else:
        lines.append(node.node_type)
    if node.branch_group_id:
        branch = f" / {node.branch_id}" if node.branch_id else ""
        lines.append(f"Group: {node.branch_group_id}{branch}")
    if node.join_policy:
        lines.append(f"Join: {node.join_policy}")
    if node.params:
        params = ", ".join(f"{key}={value}" for key, value in node.params.items())
        lines.append(_short_text(params, 30))

    meta = operation_meta.get(node.id) if operation_meta else None
    if meta:
        if meta.get("time"):
            lines.append(meta["time"])
        if meta.get("module"):
            lines.append(f"Module: {meta['module']}")
        if meta.get("duration"):
            lines.append(meta["duration"])
        if meta.get("route"):
            lines.append(_short_text(f"Route: {meta['route']}", 30))
    return lines[:8]


def _edge_label(ir, src, dst):
    src_node = ir.nodes[src]
    dst_node = ir.nodes[dst]
    parts = []
    explicit_label = ir.metadata.get("edgeLabels", {}).get(f"{src}->{dst}")
    if explicit_label:
        parts.append(str(explicit_label))
    if dst_node.branch_id:
        parts.append(dst_node.branch_id)
    elif src_node.branch_id and dst_node.is_control:
        parts.append(src_node.branch_id)
    if dst_node.branch_group_id and dst_node.branch_group_id != src_node.branch_group_id:
        parts.append(dst_node.branch_group_id)
    return " / ".join(parts)


def _operation_meta_by_node(ir, planner_result):
    meta = {}
    for op in getattr(planner_result, "operations", []):
        if op.recipe_node_id not in ir.nodes:
            continue
        current = meta.setdefault(op.recipe_node_id, {"starts": [], "ends": [], "modules": [], "routes": [], "connects": [], "transfers": []})
        current["starts"].append(op.start_s)
        current["ends"].append(op.end_s)
        if op.module:
            current["modules"].append(op.module)
        if op.connection_path:
            current["routes"].append(op.connection_path)
        if op.connect_duration_s:
            current["connects"].append(op.connect_duration_s)
        if op.transfer_duration_s:
            current["transfers"].append(op.transfer_duration_s)

    compact = {}
    for node_id, item in meta.items():
        start = min(item["starts"]) if item["starts"] else None
        end = max(item["ends"]) if item["ends"] else None
        connect = max(item["connects"]) if item["connects"] else 0
        transfer = max(item["transfers"]) if item["transfers"] else 0
        compact[node_id] = {
            "time": f"{_format_seconds(start)} - {_format_seconds(end)}" if start is not None and end is not None else "",
            "module": ", ".join(sorted(set(item["modules"]))),
            "duration": f"Connect {_format_seconds(connect)} / transfer {_format_seconds(transfer)}" if connect or transfer else "",
            "route": "; ".join(dict.fromkeys(item["routes"])),
        }
    return compact


def _render_recipe_graph(ir, title, selected_branches=None, operation_meta=None, note=""):
    selected_branches = selected_branches or {}
    operation_meta = operation_meta or {}
    positions, width, height, node_w, node_h = _recipe_layout(ir)
    selected_nodes = set(operation_meta)

    def node_opacity(node):
        selected = selected_branches.get(node.branch_group_id)
        selected_set = (
            set(selected)
            if isinstance(selected, (list, tuple, set))
            else {selected} if selected else set()
        )
        if selected_set and node.branch_id and node.branch_id not in selected_set:
            return 0.22
        if selected_nodes and node.is_task and node.id not in selected_nodes and node.branch_group_id.startswith(("XOR_", "OR_")):
            return 0.22
        return 1.0

    node_opacities = {node_id: node_opacity(node) for node_id, node in ir.nodes.items()}
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#6b7280"/></marker></defs>',
        f'<text x="28" y="32" font-size="18" font-weight="700" fill="#1f2937">{escape(title)}</text>',
    ]
    if note:
        elements.append(f'<text x="28" y="52" font-size="12" fill="#b45309">{escape(note)}</text>')

    for src, dst in ir.edges:
        if src not in positions or dst not in positions:
            continue
        x1, y1 = positions[src]
        x2, y2 = positions[dst]
        start_x, start_y = x1 + node_w, y1 + node_h / 2
        end_x, end_y = x2, y2 + node_h / 2
        mid_x = (start_x + end_x) / 2
        opacity = min(node_opacities.get(src, 1.0), node_opacities.get(dst, 1.0))
        elements.append(f'<path d="M {start_x:.1f} {start_y:.1f} C {mid_x:.1f} {start_y:.1f}, {mid_x:.1f} {end_y:.1f}, {end_x:.1f} {end_y:.1f}" fill="none" stroke="#6b7280" stroke-width="1.8" marker-end="url(#arrow)" opacity="{opacity:.2f}"/>')
        label = _edge_label(ir, src, dst)
        if label:
            label_x, label_y = mid_x - 18, (start_y + end_y) / 2 - 6
            elements.append(f'<text x="{label_x:.1f}" y="{label_y:.1f}" font-size="11" fill="#4b5563" opacity="{opacity:.2f}">{escape(label)}</text>')

    for node in ir.topological_nodes():
        x, y = positions[node.id]
        fill, stroke = _node_style(node)
        opacity = node_opacities[node.id]
        dash = ' stroke-dasharray="6 4"' if node.is_control else ""
        elements.append(f'<g opacity="{opacity:.2f}">')
        stroke_width = 3 if node.id in operation_meta else 2
        elements.append(f'<rect x="{x}" y="{y}" width="{node_w}" height="{node_h}" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"{dash}/>' )
        for idx, line in enumerate(_node_lines(node, operation_meta)):
            weight = "700" if idx == 0 else "400"
            color = "#111827" if idx == 0 else "#374151"
            elements.append(f'<text x="{x + 12}" y="{y + 22 + idx * 15}" font-size="12" font-weight="{weight}" fill="{color}">{escape(line)}</text>')
        elements.append('</g>')

    elements.append('</svg>')
    legend = "<div style='display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:#374151;margin-top:6px'><span><b>Solid</b>: operation</span><span><b>Dashed</b>: split/join control</span><span><b>Dimmed</b>: unselected XOR/OR branch</span></div>"
    html = f"<div style='max-width:100%;overflow-x:auto;border:1px solid #d1d5db;border-radius:8px;padding:10px;background:white'>{''.join(elements)}{legend}</div>"
    display(HTML(html))


def display_recipe_graph(ir, title="General Recipe Control Flow", selected_branches=None):
    _render_recipe_graph(ir, title, selected_branches=selected_branches)


def _process_operation_kind(op):
    node_id = getattr(op, "recipe_node_id", "") or ""
    op_type = getattr(op, "operation_type", "") or ""
    text = f"{node_id} {op_type}".lower()
    if node_id.startswith("AUX_") and node_id.endswith("_CONNECT"):
        return "Auxiliary Connect"
    if node_id.startswith("AUX_"):
        return "Auxiliary Transfer"
    if node_id.endswith("_CONNECT") or "connect" in text:
        return "Connect"
    normalized = {
        "dose": "Dosing",
        "mix": "Mixing",
        "usage": "Usage",
        "settling": "Settling",
        "separation": "Separation",
    }.get(op_type.lower())
    if normalized:
        return normalized
    if getattr(op, "connection_path", ""):
        return "Recipe Transfer"
    return op_type or "Operation"


def _process_operation_color(kind):
    colors = {
        "Connect": ("#fde68a", "#b45309"),
        "Auxiliary Connect": ("#fde68a", "#b45309"),
        "Auxiliary Transfer": ("#ffedd5", "#ea580c"),
        "Recipe Transfer": ("#dbeafe", "#2563eb"),
        "Dosing": ("#dcfce7", "#16a34a"),
        "Mixing": ("#e0e7ff", "#4f46e5"),
        "Usage": ("#fef3c7", "#d97706"),
        "Settling": ("#f1f5f9", "#64748b"),
        "Separation": ("#fce7f3", "#db2777"),
    }
    return colors.get(kind, ("#e5e7eb", "#4b5563"))


def _process_lane_key(op):
    module = getattr(op, "module", "") or ""
    source = getattr(op, "source_module", "") or ""
    target = getattr(op, "target_module", "") or ""
    kind = _process_operation_kind(op)
    if kind in {"Dosing", "Recipe Transfer", "Auxiliary Transfer", "Separation"} and source:
        return source
    if "->" in module and source:
        return source
    if module:
        return module
    return source or target or "Unassigned"


def _layout_gantt_rows(operations):
    by_lane = {}
    for op in operations:
        by_lane.setdefault(_process_lane_key(op), []).append(op)

    rows = []
    def lane_sort_key(lane):
        if lane.startswith("HC"):
            digits = "".join(ch for ch in lane if ch.isdigit())
            return (0, int(digits) if digits else 999, lane)
        return (1, 999, lane)

    lane_order = sorted(by_lane, key=lane_sort_key)
    for lane in lane_order:
        sublane_ends = []
        for op in sorted(by_lane[lane], key=lambda item: (item.start_s, item.end_s, item.step_id)):
            placed = None
            for idx, end_s in enumerate(sublane_ends):
                if op.start_s >= end_s:
                    placed = idx
                    break
            if placed is None:
                placed = len(sublane_ends)
                sublane_ends.append(op.end_s)
            else:
                sublane_ends[placed] = op.end_s
            rows.append((lane, placed, op))
    return rows


def _operation_extra_label(op, kind):
    if kind in {"Dosing", "Separation", "Recipe Transfer", "Auxiliary Transfer"}:
        material = getattr(op, "material", {}) or {}
        if not material:
            params = (getattr(op, "trace", {}) or {}).get("params", {}) or {}
            ingredient = params.get("ingredient")
            amount = params.get("amount_L")
            if ingredient and amount is not None:
                material = {ingredient: float(amount)}
        if not material:
            return ""
        items = []
        for name, amount in material.items():
            try:
                items.append((name, float(amount)))
            except (TypeError, ValueError):
                continue
        if not items:
            return ""
        if len(items) == 1:
            name, amt = items[0]
            return f"{name}: {amt:.2f} L"
        total = sum(amt for _, amt in items)
        ratio_parts = [
            f"{name}: {amt / total * 100:.1f}%" if total > 0 else f"{name}: -"
            for name, amt in items
        ]
        return f"total {total:.2f} L ({', '.join(ratio_parts)})"
    if kind == "Mixing":
        params = (getattr(op, "trace", {}) or {}).get("params", {}) or {}
        rpm = params.get("rpm")
        if rpm is None:
            return ""
        try:
            return f"{int(float(rpm))} rpm"
        except (TypeError, ValueError):
            return f"{rpm} rpm"
    return ""


def _operation_tooltip(op, kind):
    parts = [
        f"Step {getattr(op, 'step_id', '')}",
        f"Recipe node: {getattr(op, 'recipe_node_id', '')}",
        f"Kind: {kind}",
        f"Operation: {getattr(op, 'operation', '')}",
        f"Module/lane: {_process_lane_key(op)}",
        f"Start: {_format_seconds(getattr(op, 'start_s', ''))}",
        f"End: {_format_seconds(getattr(op, 'end_s', ''))}",
    ]
    if getattr(op, "connection_path", ""):
        parts.append(f"Route: {op.connection_path}")
    if getattr(op, "connect_duration_s", 0):
        parts.append(f"Connect duration: {_format_seconds(op.connect_duration_s)}")
    if getattr(op, "transfer_duration_s", 0):
        parts.append(f"Transfer duration: {_format_seconds(op.transfer_duration_s)}")
    return " | ".join(parts)


def _adaptive_time_mapper(operations, min_start, max_end, plot_w):
    time_points = sorted({min_start, max_end} | {point for op in operations for point in (op.start_s, op.end_s)})
    gaps = [max(end - start, 0.0) for start, end in zip(time_points, time_points[1:])]
    positive_gaps = sorted(gap for gap in gaps if gap > 0)
    median_gap = positive_gaps[len(positive_gaps) // 2] if positive_gaps else 1.0

    # Compressed event scale: every start/end interval stays readable; long waits are only modestly wider.
    visual_gaps = []
    for gap in gaps:
        if gap <= 0:
            visual_gaps.append(0.0)
        else:
            visual_gaps.append(34.0 + 10.0 * math.log1p(gap / max(median_gap, 1.0)))
    total_visual = sum(visual_gaps) or 1.0
    cumulative = [0.0]
    for visual_gap in visual_gaps:
        cumulative.append(cumulative[-1] + visual_gap)

    def x_offset(value):
        value = min(max(float(value), min_start), max_end)
        for idx, (start, end) in enumerate(zip(time_points, time_points[1:])):
            if value <= end or idx == len(time_points) - 2:
                gap = max(end - start, 1e-9)
                fraction = (value - start) / gap
                return (cumulative[idx] + fraction * visual_gaps[idx]) / total_visual * plot_w
        return plot_w

    return x_offset, median_gap


def _parallel_windows(operations):
    points = sorted({point for op in operations for point in (op.start_s, op.end_s)})
    windows = []
    for start, end in zip(points, points[1:]):
        if end <= start:
            continue
        active = [op for op in operations if op.start_s < end and start < op.end_s]
        if len(active) > 1:
            windows.append((start, end, len(active)))
    return windows


def _render_process_gantt(planner_result, title):
    operations = list(getattr(planner_result, "operations", []) or [])
    feasible = getattr(planner_result, "status", "") in {"OPTIMAL", "FEASIBLE"}
    if not feasible or not operations:
        note = "No feasible timed operations were returned for the process plan."
        html = f"<div style='border:1px solid #d1d5db;border-radius:8px;padding:12px;background:white'><b>{escape(title)}</b><div style='color:#b45309;margin-top:4px'>{escape(note)}</div></div>"
        display(HTML(html))
        return

    operations.sort(key=lambda op: (op.start_s, op.end_s, op.step_id))
    min_start = min(op.start_s for op in operations)
    max_end = max(op.end_s for op in operations)
    rows = _layout_gantt_rows(operations)
    parallel_windows = _parallel_windows(operations)
    row_h, header_h, left_w, plot_w, right_pad = 42, 118, 210, 1600, 72
    max_label_chars = 0
    for op in operations:
        kind = _process_operation_kind(op)
        label = f"{getattr(op, 'step_id', '')}: {getattr(op, 'recipe_node_id', '')}"
        if kind in {"Connect", "Auxiliary Connect", "Auxiliary Transfer"}:
            label = f"{getattr(op, 'step_id', '')}: {kind}"
        route = getattr(op, "connection_path", "") or ""
        full_label = f"{label} | {_format_seconds(op.start_s)}-{_format_seconds(op.end_s)}"
        extra = _operation_extra_label(op, kind)
        if extra:
            full_label = f"{full_label} | {extra}"
        if route:
            full_label = f"{full_label} | {route}"
        max_label_chars = max(max_label_chars, len(full_label))
    label_pad = min(max(max_label_chars * 6 + 120, 360), 980)
    height = header_h + max(len(rows), 1) * row_h + 76
    width = left_w + plot_w + right_pad + label_pad
    x_offset, adaptive_scale = _adaptive_time_mapper(operations, min_start, max_end, plot_w)

    def x_for(value):
        return left_w + x_offset(value)

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<text x="24" y="30" font-size="18" font-weight="700" fill="#1f2937">{escape(title)}</text>',
        f'<text x="24" y="50" font-size="12" fill="#4b5563">Status: {escape(str(getattr(planner_result, "status", "")))} | Makespan: {_format_seconds(getattr(planner_result, "makespan_s", max_end))} | Adaptive time scale expands short operations and compresses long holds.</text>',
        f'<text x="24" y="68" font-size="11" fill="#6b7280">Blue background bands mark time windows where operations overlap.</text>',
        f'<text x="{left_w}" y="94" font-size="11" fill="#6b7280">Compressed visual time scale</text>',
    ]

    tick_values = sorted(set([min_start, max_end] + [op.start_s for op in operations] + [op.end_s for op in operations]))
    if len(tick_values) > 10:
        step = max(1, len(tick_values) // 9)
        tick_values = tick_values[::step]
        if max_end not in tick_values:
            tick_values.append(max_end)
    for t in tick_values:
        x = x_for(t)
        elements.append(f'<line x1="{x:.1f}" y1="100" x2="{x:.1f}" y2="{height - 52}" stroke="#e5e7eb" stroke-width="1"/>')
        elements.append(f'<text x="{x - 12:.1f}" y="108" font-size="10" fill="#6b7280">{_format_seconds(t)}</text>')

    seen_lane_labels = set()
    prev_lane = None
    for row_idx, (lane, sublane, op) in enumerate(rows):
        y = header_h + row_idx * row_h
        if row_idx % 2 == 0:
            elements.append(f'<rect x="0" y="{y - 8}" width="{width}" height="{row_h}" fill="#f9fafb"/>')
        if prev_lane is not None and lane != prev_lane:
            elements.append(f'<line x1="0" y1="{y - 8}" x2="{width}" y2="{y - 8}" stroke="#9ca3af" stroke-width="1.5"/>')
        prev_lane = lane
        if lane not in seen_lane_labels:
            seen_lane_labels.add(lane)
            elements.append(f'<text x="22" y="{y + 12}" font-size="12" font-weight="700" fill="#374151">{escape(_short_text(lane, 24))}</text>')

    for start, end, active_count in parallel_windows:
        x = x_for(start)
        band_w = max(x_for(end) - x, 2)
        elements.append(f'<rect x="{x:.1f}" y="{header_h - 10}" width="{band_w:.1f}" height="{height - header_h - 48}" fill="#93c5fd" opacity="0.42"><title>{active_count} operations overlap from {_format_seconds(start)} to {_format_seconds(end)}</title></rect>')

    for row_idx, (lane, sublane, op) in enumerate(rows):
        y = header_h + row_idx * row_h

        kind = _process_operation_kind(op)
        fill, stroke = _process_operation_color(kind)
        x = x_for(op.start_s)
        raw_w = x_for(op.end_s) - x
        bar_w = min(max(raw_w, 6), plot_w)
        label = f"{getattr(op, 'step_id', '')}: {getattr(op, 'recipe_node_id', '')}"
        if kind in {"Connect", "Auxiliary Connect", "Auxiliary Transfer"}:
            label = f"{getattr(op, 'step_id', '')}: {kind}"
        route = getattr(op, "connection_path", "") or ""
        tooltip = _operation_tooltip(op, kind)
        elements.append(f'<rect x="{x:.1f}" y="{y - 1}" width="{bar_w:.1f}" height="20" rx="4" fill="{fill}" stroke="{stroke}" stroke-width="1.5"><title>{escape(tooltip)}</title></rect>')
        label_x = x + bar_w + 7
        full_label = f"{label} | {_format_seconds(op.start_s)}-{_format_seconds(op.end_s)}"
        extra = _operation_extra_label(op, kind)
        if extra:
            full_label = f"{full_label} | {extra}"
        if route:
            full_label = f"{full_label} | {route}"
        elements.append(f'<text x="{label_x:.1f}" y="{y + 13}" font-size="10" fill="#374151"><title>{escape(full_label)}</title>{escape(full_label)}</text>')

    legend_items = ["Connect", "Auxiliary Connect", "Auxiliary Transfer", "Recipe Transfer", "Dosing", "Mixing", "Usage", "Settling", "Separation"]
    lx, ly = 24, height - 24
    elements.append(f'<rect x="{lx}" y="{ly - 10}" width="12" height="12" rx="2" fill="#93c5fd" opacity="0.60"/>')
    elements.append(f'<text x="{lx + 17}" y="{ly}" font-size="11" fill="#374151">Parallel window</text>')
    lx += 132
    for item in legend_items:
        fill, stroke = _process_operation_color(item)
        elements.append(f'<rect x="{lx}" y="{ly - 10}" width="12" height="12" rx="2" fill="{fill}" stroke="{stroke}"/>')
        elements.append(f'<text x="{lx + 17}" y="{ly}" font-size="11" fill="#374151">{escape(item)}</text>')
        lx += 17 + len(item) * 7 + 18

    elements.append('</svg>')
    html = f"<div style='max-width:100%;overflow-x:auto;border:1px solid #d1d5db;border-radius:8px;padding:10px;background:white'>{''.join(elements)}</div>"
    display(HTML(html))


def display_process_plan_graph(ir, planner_result, title="Selected Process Plan Graph"):
    # This is deliberately the exact SVG renderer used by the Flet desktop App.
    # ``ir`` remains in the signature for backwards-compatible notebook cells.
    # Reloading only this stateless display module lets a running notebook pick
    # up Gantt fixes without reloading RecipeIR/RTN/solver class definitions.
    visualization = importlib.reload(app_visualization)
    svg = visualization.gantt_svg(planner_result, title=title)
    display(HTML(
        "<div style='max-width:100%;overflow:auto;border:1px solid #293446;"
        "border-radius:16px;background:#11151d'>"
        f"{svg}</div>"
    ))


def display_connection_graph(plant, planner_result, title="Solved Module Connections"):
    # Same SVG renderer the Flet desktop App uses for its connection page: it
    # shows the port-to-port wiring the plan selects between modules, i.e. the
    # inter-module connection configuration that the Gantt lists as connect
    # operations. ``plant`` supplies the module capacities and capabilities.
    visualization = importlib.reload(app_visualization)
    svg = visualization.connection_graph_svg(plant, planner_result, title=title)
    display(HTML(
        "<div style='max-width:100%;overflow:auto;border:1px solid #293446;"
        "border-radius:16px;background:#11151d'>"
        f"{svg}</div>"
    ))
