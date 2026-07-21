from __future__ import annotations

import base64
from collections import defaultdict
from html import escape
import math
import re
from typing import Any


def svg_data_uri(svg: str) -> str:
    payload = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{payload}"


def _layers(ir) -> tuple[dict[str, int], dict[int, list]]:
    predecessors = ir.predecessors()
    by_layer: dict[int, list] = defaultdict(list)
    layer: dict[str, int] = {}
    for node in ir.topological_nodes():
        layer[node.id] = 0 if not predecessors.get(node.id) else max(layer[item] for item in predecessors[node.id]) + 1
        by_layer[layer[node.id]].append(node)
    return layer, dict(by_layer)


def recipe_sfc_layout(ir) -> dict[str, Any]:
    """Return the deterministic canvas geometry used by the SVG and UI hit targets."""
    _, levels = _layers(ir)
    node_w, node_h = 230, 82
    x_gap, y_gap = 54, 76
    margin_x, margin_y = 48, 72
    widest = max((len(items) for items in levels.values()), default=1)
    width = max(900, margin_x * 2 + widest * node_w + (widest - 1) * x_gap)
    height = max(420, margin_y * 2 + len(levels) * node_h + max(0, len(levels) - 1) * y_gap)
    positions: dict[str, tuple[float, float]] = {}
    for layer_index, nodes in levels.items():
        row_w = len(nodes) * node_w + max(0, len(nodes) - 1) * x_gap
        x = (width - row_w) / 2
        y = margin_y + layer_index * (node_h + y_gap)
        for node in nodes:
            positions[node.id] = (x, y)
            x += node_w + x_gap
    return {
        "width": width,
        "height": height,
        "node_width": node_w,
        "node_height": node_h,
        "positions": positions,
    }


def _selected_branch_set(selected_branches: dict[str, Any], node) -> set[str]:
    selected = selected_branches.get(node.branch_group_id)
    if isinstance(selected, (list, tuple, set)):
        return {str(item) for item in selected}
    return {str(selected)} if selected not in (None, "") else set()


def recipe_sfc_svg(ir, planner_result=None, title: str = "General Recipe SFC") -> str:
    layout = recipe_sfc_layout(ir)
    width, height = layout["width"], layout["height"]
    node_w, node_h = layout["node_width"], layout["node_height"]
    positions = layout["positions"]

    operation_by_node = {}
    selected_branches = {}
    if planner_result is not None:
        selected_branches = planner_result.selected_branches
        for operation in planner_result.operations:
            if operation.operation_type not in {"connect", "transfer"}:
                operation_by_node.setdefault(operation.recipe_node_id, operation)

    body = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10Z" fill="#7c8ba1"/></marker><filter id="shadow"><feDropShadow dx="0" dy="8" stdDeviation="10" flood-color="#000" flood-opacity=".25"/></filter></defs>',
        '<rect width="100%" height="100%" rx="24" fill="#11151d"/>',
        f'<text x="36" y="42" fill="#f5f7fb" font-family="Inter,Segoe UI,sans-serif" font-size="20" font-weight="700">{escape(title)}</text>',
    ]
    for source, target in ir.edges:
        if source not in positions or target not in positions:
            continue
        sx, sy = positions[source]
        tx, ty = positions[target]
        x1, y1 = sx + node_w / 2, sy + node_h
        x2, y2 = tx + node_w / 2, ty
        mid = (y1 + y2) / 2
        target_node = ir.nodes[target]
        selected_set = _selected_branch_set(selected_branches, target_node)
        inactive = bool(selected_set and target_node.branch_id and target_node.branch_id not in selected_set)
        active = planner_result is not None and not inactive
        edge_color = "#42e3a4" if active else "#536176" if inactive else "#7c8ba1"
        dash = ' stroke-dasharray="7 7"' if inactive else ""
        opacity = ".32" if inactive else "1"
        body.append(f'<path d="M{x1:.1f} {y1:.1f} V{mid:.1f} H{x2:.1f} V{y2:.1f}" fill="none" stroke="{edge_color}" stroke-width="2" opacity="{opacity}"{dash} marker-end="url(#arrow)"/>')

    for node in ir.topological_nodes():
        x, y = positions[node.id]
        is_control = node.is_control
        selected_set = _selected_branch_set(selected_branches, node)
        dimmed = bool(selected_set and node.branch_id and node.branch_id not in selected_set)
        fill = "#262e3d" if is_control else "#183650"
        stroke = "#a78bfa" if is_control else "#42b8ff"
        if node.id in operation_by_node:
            stroke = "#42e3a4"
        opacity = ".28" if dimmed else "1"
        radius = 12 if not is_control else 24
        body.append(f'<g opacity="{opacity}" filter="url(#shadow)"><rect x="{x}" y="{y}" width="{node_w}" height="{node_h}" rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="2"/></g>')
        label = node.control_node_type or node.name or node.id
        body.append(f'<text x="{x + 14}" y="{y + 28}" fill="#f5f7fb" font-family="Inter,Segoe UI,sans-serif" font-size="14" font-weight="700" opacity="{opacity}">{escape(str(label)[:30])}</text>')
        detail = node.id
        op = operation_by_node.get(node.id)
        if op:
            detail = f"{op.module}  ·  {op.start_s:g}–{op.end_s:g}s"
        elif node.branch_id:
            detail = f"Branch: {node.branch_id}"
        body.append(f'<text x="{x + 14}" y="{y + 54}" fill="#aeb9c9" font-family="Inter,Segoe UI,sans-serif" font-size="12" opacity="{opacity}">{escape(str(detail)[:34])}</text>')
        if planner_result is not None and node.branch_id:
            badge = "NOT SELECTED" if dimmed else "SELECTED"
            badge_color = "#7b8798" if dimmed else "#42e3a4"
            body.append(f'<text x="{x + node_w - 12}" y="{y + 18}" fill="{badge_color}" font-family="Inter,Segoe UI,sans-serif" font-size="9" font-weight="700" text-anchor="end" opacity="{opacity}">{badge}</text>')
    body.append("</svg>")
    return "".join(body)


def _adaptive_time_mapper(operations, plot_w: float):
    points = sorted({float(point) for op in operations for point in (op.start_s, op.end_s)})
    if len(points) < 2:
        return lambda value: 0.0, points or [0.0]
    gaps = [max(end - start, 0.0) for start, end in zip(points, points[1:])]
    positive = sorted(gap for gap in gaps if gap > 0)
    median = positive[len(positive) // 2] if positive else 1.0
    visual_gaps = [0.0 if gap <= 0 else 34.0 + 10.0 * math.log1p(gap / max(median, 1.0)) for gap in gaps]
    total = sum(visual_gaps) or 1.0
    cumulative = [0.0]
    for gap in visual_gaps:
        cumulative.append(cumulative[-1] + gap)

    def x_offset(value):
        value = min(max(float(value), points[0]), points[-1])
        for index, (start, end) in enumerate(zip(points, points[1:])):
            if value <= end or index == len(points) - 2:
                fraction = (value - start) / max(end - start, 1e-9)
                return (cumulative[index] + fraction * visual_gaps[index]) / total * plot_w
        return plot_w

    return x_offset, points


def _natural_lane_key(value: str) -> tuple:
    return tuple(int(part) if part.isdigit() else part.casefold() for part in re.split(r"(\d+)", value))


def _gantt_lane_group(lane: str) -> str:
    """Group a Module lane and all of its outgoing route lanes by source Module."""
    return lane.split("->", 1)[0].strip() or lane


def _gantt_operation_lane(operation) -> str:
    """Place physical flows on their directed route, grouped by source Module."""
    source = str(getattr(operation, "source_module", "") or "")
    target = str(getattr(operation, "target_module", "") or "")
    kind = str(getattr(operation, "operation_type", "") or "").casefold()
    if source and target and source != target and kind in {
        "dose", "separation", "transfer", "aux_transfer", "connect", "disconnect"
    }:
        return f"{source}->{target}"
    return str(getattr(operation, "module", "") or target or "Transfer")


def gantt_operation_tooltip(operation) -> str:
    """Return one detailed hover description shared by Flet and SVG Gantt."""
    material = getattr(operation, "material", {}) or {}
    material_text = ", ".join(
        f"{name}: {float(amount):g} L" for name, amount in material.items()
    ) or "—"
    route = str(getattr(operation, "connection_path", "") or "—")
    out_port = str(getattr(operation, "out_port", "") or "—")
    in_port = str(getattr(operation, "in_port", "") or "—")
    return "\n".join([
        f"Step {int(getattr(operation, 'step_id', 0))} · {getattr(operation, 'recipe_node_id', '')}",
        f"Operation: {getattr(operation, 'operation_type', '')} · {getattr(operation, 'operation', '')}",
        f"Module: {getattr(operation, 'module', '') or '—'}",
        f"Time: {float(getattr(operation, 'start_s', 0)):g}s → {float(getattr(operation, 'end_s', 0)):g}s · duration {float(getattr(operation, 'duration_s', 0)):g}s",
        f"Route: {route}",
        f"Ports: {out_port} → {in_port}",
        f"Material: {material_text}",
        f"Energy: {float(getattr(operation, 'energy_consumption_kwh', 0)):g} kWh · weighted cost €{float(getattr(operation, 'weighted_energy_cost', 0)):.2f}",
        f"CO₂e: {float(getattr(operation, 'co2_emissions_kg', 0)):g} kg · weighted cost €{float(getattr(operation, 'weighted_co2_cost', 0)):.2f}",
        f"Usage cost: €{float(getattr(operation, 'weighted_usage_cost', 0)):.2f} · total cost: €{float(getattr(operation, 'total_cost', 0)):.2f}",
    ])


def gantt_layout(planner_result) -> dict[str, Any]:
    """Return adaptive Gantt geometry for both SVG and interactive Flet views."""
    operations = [item for item in planner_result.operations if item.duration_s > 0]
    lane_names = {_gantt_operation_lane(item) for item in operations}
    lanes_by_group: dict[str, list[str]] = defaultdict(list)
    for lane in lane_names:
        lanes_by_group[_gantt_lane_group(lane)].append(lane)
    group_names = sorted(lanes_by_group, key=_natural_lane_key)
    for group in group_names:
        lanes_by_group[group].sort(key=lambda lane: (lane != group, _natural_lane_key(lane)))
    lanes = [lane for group in group_names for lane in lanes_by_group[group]]
    width = 1320
    left, right, top = 170, 36, 78
    lane_h, gap = 44, 12
    group_header_h, group_gap = 28, 18
    lane_y: dict[str, float] = {}
    groups = []
    cursor = float(top)
    for index, group in enumerate(group_names):
        group_top = cursor
        cursor += group_header_h
        group_lanes = lanes_by_group[group]
        for lane in group_lanes:
            lane_y[lane] = cursor
            cursor += lane_h + gap
        group_bottom = cursor - gap
        groups.append({
            "name": group,
            "lanes": list(group_lanes),
            "y": group_top,
            "height": max(group_header_h + lane_h, group_bottom - group_top),
            "index": index,
        })
        cursor = group_bottom + group_gap
    height = max(380, cursor + 52)
    plot_w = width - left - right
    x_offset, time_points = _adaptive_time_mapper(operations, plot_w)
    lane_index = {name: index for index, name in enumerate(lanes)}
    bars = []
    for operation in operations:
        lane = _gantt_operation_lane(operation)
        x = left + x_offset(operation.start_s)
        bar_w = max(42, x_offset(operation.end_s) - x_offset(operation.start_s))
        bars.append({
            "operation": operation,
            "lane": lane,
            "x": x,
            "y": lane_y[lane] + 5,
            "width": bar_w,
            "height": lane_h - 10,
        })
    return {
        "operations": operations,
        "lanes": lanes,
        "lane_index": lane_index,
        "lane_y": lane_y,
        "groups": groups,
        "width": width,
        "height": height,
        "left": left,
        "right": right,
        "top": top,
        "lane_height": lane_h,
        "gap": gap,
        "group_header_height": group_header_h,
        "group_gap": group_gap,
        "plot_width": plot_w,
        "time_points": time_points,
        "x_offset": x_offset,
        "bars": bars,
    }


def gantt_svg(
    planner_result,
    title: str = "Optimal RTN Schedule",
    *,
    include_tooltips: bool = True,
) -> str:
    layout = gantt_layout(planner_result)
    operations, lanes, lane_y = layout["operations"], layout["lanes"], layout["lane_y"]
    width, height = layout["width"], layout["height"]
    left, right, top = layout["left"], layout["right"], layout["top"]
    lane_h, gap, plot_w = layout["lane_height"], layout["gap"], layout["plot_width"]
    max_time = max([item.end_s for item in operations] + [1])
    x_offset, time_points = layout["x_offset"], layout["time_points"]
    colors = {"dose": "#42b8ff", "mix": "#a78bfa", "usage": "#42e3a4", "settling": "#f8c35e", "separation": "#ff7a9e", "connect": "#f59e0b", "transfer": "#fb7185"}
    body = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<defs><filter id="barShadow"><feDropShadow dx="0" dy="4" stdDeviation="5" flood-color="#000" flood-opacity=".25"/></filter></defs>',
        '<rect width="100%" height="100%" rx="24" fill="#11151d"/>',
        f'<text x="36" y="42" fill="#f5f7fb" font-family="Inter,Segoe UI,sans-serif" font-size="20" font-weight="700">{escape(title)}</text>',
        '<text x="36" y="61" fill="#8d9aae" font-family="Inter,Segoe UI,sans-serif" font-size="11">Adaptive time scale keeps short operations readable and compresses long holds.</text>',
    ]
    for group in layout["groups"]:
        fill = "#141b26" if group["index"] % 2 == 0 else "#121923"
        body.append(f'<rect x="16" y="{group["y"]:.1f}" width="{width - 32}" height="{group["height"]:.1f}" rx="14" fill="{fill}" stroke="#293446" stroke-width="1"/>')
        body.append(f'<text x="30" y="{group["y"] + 19:.1f}" fill="#42b8ff" font-family="Inter,Segoe UI,sans-serif" font-size="11" font-weight="700" letter-spacing=".6">{escape(group["name"])}</text>')
    ticks = time_points
    if len(ticks) > 9:
        stride = max(1, len(ticks) // 8)
        ticks = ticks[::stride]
        if max_time not in ticks:
            ticks.append(max_time)
    for value in ticks:
        x = left + x_offset(value)
        body.append(f'<line x1="{x:.1f}" y1="{top + layout["group_header_height"]}" x2="{x:.1f}" y2="{height - 44}" stroke="#2c3442" stroke-width="1"/>')
        body.append(f'<text x="{x:.1f}" y="{height - 20}" fill="#8d9aae" font-family="Inter,Segoe UI,sans-serif" font-size="11" text-anchor="middle">{value:g}s</text>')
    for lane in lanes:
        y = lane_y[lane]
        body.append(f'<text x="{left - 18}" y="{y + 27}" fill="#dce3ee" font-family="Inter,Segoe UI,sans-serif" font-size="13" font-weight="600" text-anchor="end">{escape(lane)}</text>')
        body.append(f'<rect x="{left}" y="{y}" width="{plot_w}" height="{lane_h}" rx="10" fill="#171d27"/>')
    tooltip_groups = []
    hover_rules = []
    tooltip_w = 500
    line_height = 15
    for bar_index, item in enumerate(layout["bars"]):
        operation, y, x, bar_w = item["operation"], item["y"], item["x"], item["width"]
        color = colors.get(operation.operation_type.casefold(), "#60a5fa")
        label = "Conn" if operation.operation_type.casefold() == "connect" and bar_w < 58 else operation.operation_type.title()
        if not include_tooltips:
            body.append(f'<rect x="{x:.1f}" y="{y}" width="{bar_w:.1f}" height="{lane_h - 10}" rx="8" fill="{color}" filter="url(#barShadow)"/>')
            body.append(f'<text x="{x + 6:.1f}" y="{y + 22}" fill="#081018" font-family="Inter,Segoe UI,sans-serif" font-size="{9 if bar_w < 58 else 11}" font-weight="700" pointer-events="none">{escape(label[:18])}</text>')
            continue
        tooltip_lines = gantt_operation_tooltip(operation).splitlines()
        tooltip_h = 20 + len(tooltip_lines) * line_height
        tooltip_x = min(max(x, 12), width - tooltip_w - 12)
        tooltip_y = y + lane_h + 4
        if tooltip_y + tooltip_h > height - 8:
            tooltip_y = max(8, y - tooltip_h - 8)
        tooltip = escape("\n".join(tooltip_lines))
        hover_class = f"gantt-hover-{bar_index}"
        tooltip_class = f"gantt-tooltip-{bar_index}"
        hover_rules.append(
            f'.{hover_class}:hover ~ .{tooltip_class}'
            '{opacity:1;visibility:visible}'
        )
        body.append(f'<g class="gantt-hover-target {hover_class}" style="cursor:help">')
        body.append(f'<rect x="{x:.1f}" y="{y}" width="{bar_w:.1f}" height="{lane_h - 10}" rx="8" fill="{color}" filter="url(#barShadow)"/>')
        body.append(f'<text x="{x + 6:.1f}" y="{y + 22}" fill="#081018" font-family="Inter,Segoe UI,sans-serif" font-size="{9 if bar_w < 58 else 11}" font-weight="700" pointer-events="none">{escape(label[:18])}</text>')
        # The transparent hitbox drives the custom SVG card. Keep a non-popup
        # <desc> for accessibility; a native <title> would create a second,
        # overlapping browser tooltip in Safari/Jupyter.
        body.append(f'<rect x="{x:.1f}" y="{y}" width="{bar_w:.1f}" height="{lane_h - 10}" rx="8" fill="transparent" pointer-events="all"><desc>{tooltip}</desc></rect>')
        body.append('</g>')
        text_elements = []
        for line_index, line in enumerate(tooltip_lines):
            compact = line if len(line) <= 82 else line[:79] + "..."
            text_elements.append(
                f'<text x="{tooltip_x + 14:.1f}" y="{tooltip_y + 21 + line_index * line_height:.1f}" '
                f'fill="{("#f5f7fb" if line_index == 0 else "#c7d1df")}" '
                f'font-family="Inter,Segoe UI,sans-serif" font-size="11" '
                f'font-weight="{("700" if line_index == 0 else "400")}">{escape(compact)}</text>'
            )
        tooltip_groups.append(
            f'<g class="gantt-tooltip {tooltip_class}" pointer-events="none">'
            f'<rect x="{tooltip_x:.1f}" y="{tooltip_y:.1f}" width="{tooltip_w}" height="{tooltip_h}" '
            'rx="12" fill="#080c13" fill-opacity=".97" stroke="#42b8ff" stroke-width="1.5" filter="url(#barShadow)"/>'
            + ''.join(text_elements)
            + '</g>'
        )
    if include_tooltips:
        body.insert(
            2,
            '<style>.gantt-tooltip{opacity:0;visibility:hidden;transition:opacity .12s ease}'
            + ''.join(hover_rules)
            + '</style>',
        )
        body.extend(tooltip_groups)
    body.append("</svg>")
    return "".join(body)


def _connection_endpoints(operation) -> tuple[str, str]:
    source = str(getattr(operation, "source_module", "") or "")
    target = str(getattr(operation, "target_module", "") or "")
    if (not source or not target) and "->" in str(getattr(operation, "module", "") or ""):
        fallback_source, fallback_target = str(operation.module).split("->", 1)
        source = source or fallback_source.strip()
        target = target or fallback_target.strip()
    return source, target


def _planned_event(operation, role: str = "") -> dict[str, Any]:
    material = {
        str(name): float(amount)
        for name, amount in (getattr(operation, "material", {}) or {}).items()
    }
    return {
        "step_id": int(getattr(operation, "step_id", 0)),
        "recipe_node_id": str(getattr(operation, "recipe_node_id", "") or ""),
        "operation_type": str(getattr(operation, "operation_type", "") or ""),
        "operation": str(getattr(operation, "operation", "") or ""),
        "role": role,
        "start_s": float(getattr(operation, "start_s", 0.0) or 0.0),
        "end_s": float(getattr(operation, "end_s", 0.0) or 0.0),
        "duration_s": float(getattr(operation, "duration_s", 0.0) or 0.0),
        "transfer_duration_s": float(getattr(operation, "transfer_duration_s", 0.0) or 0.0),
        "total_cost": float(getattr(operation, "total_cost", 0.0) or 0.0),
        "weighted_usage_cost": float(getattr(operation, "weighted_usage_cost", 0.0) or 0.0),
        "weighted_energy_cost": float(getattr(operation, "weighted_energy_cost", 0.0) or 0.0),
        "weighted_co2_cost": float(getattr(operation, "weighted_co2_cost", 0.0) or 0.0),
        "energy_consumption_kwh": float(getattr(operation, "energy_consumption_kwh", 0.0) or 0.0),
        "co2_emissions_kg": float(getattr(operation, "co2_emissions_kg", 0.0) or 0.0),
        "material": material,
        "volume_l": sum(material.values()),
        "connection_path": str(getattr(operation, "connection_path", "") or ""),
        "out_port": str(getattr(operation, "out_port", "") or ""),
        "in_port": str(getattr(operation, "in_port", "") or ""),
    }


def _ray_exit_distance(
    offset_x: float,
    offset_y: float,
    direction_x: float,
    direction_y: float,
    half_width: float,
    half_height: float,
) -> float:
    """Distance from an offset point inside a rectangle to its boundary along a ray."""
    candidates: list[float] = []
    if abs(direction_x) > 1e-9:
        boundary_x = half_width if direction_x > 0 else -half_width
        distance_x = (boundary_x - offset_x) / direction_x
        if distance_x >= 0:
            candidates.append(distance_x)
    if abs(direction_y) > 1e-9:
        boundary_y = half_height if direction_y > 0 else -half_height
        distance_y = (boundary_y - offset_y) / direction_y
        if distance_y >= 0:
            candidates.append(distance_y)
    return min(candidates) if candidates else 0.0


def connection_graph_layout(plant, planner_result) -> dict[str, Any]:
    """Aggregate the solved physical topology and lay Module on a regular polygon."""
    operations = sorted(planner_result.operations, key=lambda item: (item.start_s, item.end_s, item.step_id))
    assets_by_id = plant.by_id()
    module_ids = set(assets_by_id)
    for operation in operations:
        source, target = _connection_endpoints(operation)
        module_ids.update(item for item in (source, target) if item)
        module = str(getattr(operation, "module", "") or "")
        if module and "->" not in module:
            module_ids.add(module)
    module_ids = sorted(module_ids, key=_natural_lane_key)

    width, height = 1120.0, 720.0
    node_width, node_height = 176.0, 96.0
    center_x, center_y = width / 2.0, height / 2.0 + 18.0
    radius = min(width, height) * 0.34
    positions: dict[str, tuple[float, float]] = {}
    count = len(module_ids)
    for index, module_id in enumerate(module_ids):
        angle = -math.pi / 2 if count <= 1 else -math.pi / 2 + index * 2 * math.pi / count
        positions[module_id] = (
            center_x + radius * math.cos(angle) - node_width / 2.0,
            center_y + radius * math.sin(angle) - node_height / 2.0,
        )

    node_events: dict[str, list[dict[str, Any]]] = {module_id: [] for module_id in module_ids}
    edge_map: dict[tuple[str, str], dict[str, Any]] = {}
    for operation in operations:
        source, target = _connection_endpoints(operation)
        operation_type = str(getattr(operation, "operation_type", "") or "").casefold()
        for module_id in module_ids:
            role = ""
            if source == module_id and target and target != module_id:
                role = f"outgoing:{target}"
            elif target == module_id and source and source != module_id:
                role = f"incoming:{source}"
            elif str(getattr(operation, "module", "") or "") == module_id or (source == target == module_id):
                role = "local"
            if role:
                node_events[module_id].append(_planned_event(operation, role))

        if not source or not target or source == target:
            continue
        edge = edge_map.setdefault((source, target), {
            "id": f"{source}->{target}",
            "source": source,
            "target": target,
            "setup_events": [],
            "disconnect_events": [],
            "transfers": [],
        })
        event = _planned_event(operation)
        if operation_type == "connect":
            edge["setup_events"].append(event)
        elif operation_type == "disconnect":
            edge["disconnect_events"].append(event)
        elif event["transfer_duration_s"] > 0 or event["material"]:
            edge["transfers"].append(event)

    for edge in edge_map.values():
        setups = sorted(edge["setup_events"], key=lambda item: (item["end_s"], item["step_id"]))
        unused = set(range(len(setups)))
        transfers = sorted(edge["transfers"], key=lambda item: (item["start_s"], item["step_id"]))
        connection_events: list[dict[str, Any]] = []
        setup_to_transfer: dict[int, dict[str, Any]] = {}
        for transfer in transfers:
            candidates = [
                index for index in unused
                if setups[index]["end_s"] <= transfer["start_s"] + 1e-9
                and (
                    not transfer["connection_path"]
                    or not setups[index]["connection_path"]
                    or setups[index]["connection_path"] == transfer["connection_path"]
                )
            ]
            setup_index = max(candidates, key=lambda index: (setups[index]["end_s"], setups[index]["step_id"])) if candidates else None
            if setup_index is not None:
                unused.remove(setup_index)
                setup_to_transfer[setup_index] = transfer

        for index, setup in enumerate(setups):
            transfer = setup_to_transfer.get(index)
            connection = dict(setup)
            connection.update({
                "out_port": setup["out_port"] or (transfer["out_port"] if transfer else ""),
                "in_port": setup["in_port"] or (transfer["in_port"] if transfer else ""),
                "paired_transfer_step_id": transfer["step_id"] if transfer else None,
            })
            connection_events.append(connection)

        edge["connection_events"] = connection_events
        disconnect_events = sorted(
            edge["disconnect_events"],
            key=lambda item: (item["start_s"], item["end_s"], item["step_id"]),
        )
        edge["disconnect_events"] = disconnect_events
        edge["transfers"] = transfers
        edge.pop("setup_events", None)
        edge["connection_count"] = len(connection_events)
        edge["disconnect_count"] = len(disconnect_events)
        edge["usage_count"] = len(transfers)
        edge["total_volume_l"] = sum(item["volume_l"] for item in transfers)
        edge["total_connection_cost"] = sum(
            item["total_cost"] for item in [*connection_events, *disconnect_events]
        )
        edge["total_transfer_cost"] = sum(item["total_cost"] for item in transfers)
        edge["total_cost"] = edge["total_connection_cost"] + edge["total_transfer_cost"]

    nodes = []
    for module_id in module_ids:
        asset = assets_by_id.get(module_id)
        nodes.append({
            "id": module_id,
            "asset": asset,
            "position": positions[module_id],
            "plan_events": sorted(node_events[module_id], key=lambda item: (item["start_s"], item["end_s"], item["step_id"])),
        })

    edge_keys = set(edge_map)
    edges = []
    for key in sorted(edge_map, key=lambda item: (_natural_lane_key(item[0]), _natural_lane_key(item[1]))):
        edge = edge_map[key]
        source_x, source_y = positions[edge["source"]]
        target_x, target_y = positions[edge["target"]]
        source_cx, source_cy = source_x + node_width / 2.0, source_y + node_height / 2.0
        target_cx, target_cy = target_x + node_width / 2.0, target_y + node_height / 2.0
        dx, dy = target_cx - source_cx, target_cy - source_cy
        distance = max(math.hypot(dx, dy), 1.0)
        ux, uy = dx / distance, dy / distance
        offset = 11.0 if (edge["target"], edge["source"]) in edge_keys else 0.0
        px, py = -uy, ux
        offset_x, offset_y = px * offset, py * offset
        half_width, half_height = node_width / 2.0, node_height / 2.0
        source_exit = _ray_exit_distance(offset_x, offset_y, ux, uy, half_width, half_height)
        target_exit = _ray_exit_distance(offset_x, offset_y, -ux, -uy, half_width, half_height)
        arrow_gap = 12.0
        x1 = source_cx + offset_x + ux * (source_exit + arrow_gap)
        y1 = source_cy + offset_y + uy * (source_exit + arrow_gap)
        x2 = target_cx + offset_x - ux * (target_exit + arrow_gap)
        y2 = target_cy + offset_y - uy * (target_exit + arrow_gap)
        edge.update({
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "mid_x": (x1 + x2) / 2.0,
            "mid_y": (y1 + y2) / 2.0,
            "length": max(math.hypot(x2 - x1, y2 - y1), 1.0),
            "angle": math.atan2(y2 - y1, x2 - x1),
        })
        edges.append(edge)

    return {
        "width": width,
        "height": height,
        "node_width": node_width,
        "node_height": node_height,
        "nodes": nodes,
        "positions": positions,
        "edges": edges,
    }


def connection_graph_svg(plant, planner_result, title: str = "Solved Module Connections") -> str:
    layout = connection_graph_layout(plant, planner_result)
    width, height = layout["width"], layout["height"]
    body = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:g}" height="{height:g}" viewBox="0 0 {width:g} {height:g}">',
        '<defs><marker id="connection-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="8" markerHeight="8" orient="auto"><path d="M0 0L10 5L0 10Z" fill="#42e3a4"/></marker><filter id="node-shadow"><feDropShadow dx="0" dy="7" stdDeviation="8" flood-color="#000" flood-opacity=".28"/></filter></defs>',
        '<rect width="100%" height="100%" rx="24" fill="#11151d"/>',
        f'<text x="36" y="42" fill="#f5f7fb" font-family="Inter,Segoe UI,sans-serif" font-size="20" font-weight="700">{escape(title)}</text>',
        '<text x="36" y="62" fill="#8d9aae" font-family="Inter,Segoe UI,sans-serif" font-size="11">Directed connections and material transfers selected by the RTN plan.</text>',
    ]
    for edge in layout["edges"]:
        body.append(f'<path d="M{edge["x1"]:.1f} {edge["y1"]:.1f} L{edge["x2"]:.1f} {edge["y2"]:.1f}" fill="none" stroke="#42e3a4" stroke-width="4" marker-end="url(#connection-arrow)"/>')
    for node in layout["nodes"]:
        x, y = node["position"]
        asset = node["asset"]
        capacity = f'{asset.maximum_volume_l:g} L' if asset else "—"
        capability_count = len(asset.operations) if asset else 0
        body.append(f'<g filter="url(#node-shadow)"><rect x="{x:.1f}" y="{y:.1f}" width="{layout["node_width"]:.1f}" height="{layout["node_height"]:.1f}" rx="20" fill="#183650" stroke="#42b8ff" stroke-width="2"/></g>')
        body.append(f'<text x="{x + 16:.1f}" y="{y + 32:.1f}" fill="#f5f7fb" font-family="Inter,Segoe UI,sans-serif" font-size="17" font-weight="700">{escape(node["id"])}</text>')
        body.append(f'<text x="{x + 16:.1f}" y="{y + 56:.1f}" fill="#aeb9c9" font-family="Inter,Segoe UI,sans-serif" font-size="11">{escape(capacity)} · {capability_count} capabilities</text>')
        body.append(f'<text x="{x + 16:.1f}" y="{y + 77:.1f}" fill="#42e3a4" font-family="Inter,Segoe UI,sans-serif" font-size="10" font-weight="700">{len(node["plan_events"])} planned events</text>')
    body.append("</svg>")
    return "".join(body)


def operation_table_rows(planner_result) -> list[dict[str, Any]]:
    return planner_result.to_rows()
