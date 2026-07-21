from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Union

try:
    from material_composition import (
        MIXTURE,
        PURE_MATERIAL,
        add_composition,
        composition_signature,
        compositions_proportional,
        is_pure_composition,
        normalize_composition,
        subtract_composition,
        transfer_kind as classify_transfer_kind,
    )
except ImportError:  # pragma: no cover
    from .material_composition import (
        MIXTURE,
        PURE_MATERIAL,
        add_composition,
        composition_signature,
        compositions_proportional,
        is_pure_composition,
        normalize_composition,
        subtract_composition,
        transfer_kind as classify_transfer_kind,
    )

try:
    from cp_sat_planner import MaterialReservation, PlannerConfig, PlannerResult, PlannedOperation
    from recipe_ir import RecipeIR
    from rtn import RTNModel, TaskNodeView
except ImportError:  # pragma: no cover - notebook direct import compatibility
    from cp_sat_planner import MaterialReservation, PlannerConfig, PlannerResult, PlannedOperation
    from recipe_ir import RecipeIR
    from rtn import RTNModel, TaskNodeView


@dataclass
class MaterialEvent:
    batch_id: str
    material: str
    quantity_l: float
    location: str
    source_node_id: str
    components: Dict[str, float] = field(default_factory=dict)


@dataclass
class ControlEvent:
    node_id: str
    branch_group_id: str = ""
    branch_id: str = ""


@dataclass
class ResourceEvent:
    module_id: str
    capacity_l: float
    busy_intervals: List[Tuple[float, float, str]] = field(default_factory=list)


@dataclass
class InProgressEvent:
    operation_id: str
    recipe_node_id: str
    module_id: str
    start_s: float
    end_s: float


@dataclass
class ValidationResult:
    valid: bool
    errors: List[str]
    warnings: List[str]
    material_tokens: List[MaterialEvent]
    control_tokens: List[ControlEvent]
    resource_tokens: Dict[str, ResourceEvent]
    in_progress_history: List[InProgressEvent]
    profit_check: Dict[str, float]
    connection_errors: List[str] = field(default_factory=list)
    connection_log: List[str] = field(default_factory=list)
    terminal_connections: List[str] = field(default_factory=list)
    resource_errors: List[str] = field(default_factory=list)
    resource_log: List[str] = field(default_factory=list)
    composition_errors: List[str] = field(default_factory=list)
    composition_log: List[str] = field(default_factory=list)


def validate_schedule(
    rtn_or_ir: Union[RTNModel, RecipeIR],
    plan: PlannerResult,
    module_maximum_volume: Dict[str, List[float]] | None = None,
    module_resources: Dict[str, List[Any]] | None = None,
    config: PlannerConfig | None = None,
    tolerance: float = 1e-6,
) -> ValidationResult:
    """Validate a CP-SAT plan against RTN structural invariants.

    The first argument may be either an ``RTNModel`` (preferred) or a
    ``RecipeIR`` (legacy callers). RecipeIR is auto-lifted to RTN since the
    validator only reads structural data: tasks, precedence, choice groups,
    and ingredient totals — none of which require plant configuration.
    """
    if isinstance(rtn_or_ir, RecipeIR):
        try:
            from recipe_to_rtn import recipe_ir_to_rtn
        except ImportError:  # pragma: no cover
            from recipe_to_rtn import recipe_ir_to_rtn
        rtn = recipe_ir_to_rtn(rtn_or_ir)
    else:
        rtn = rtn_or_ir

    cfg = config or PlannerConfig()
    errors: List[str] = []
    warnings: List[str] = []
    connection_errors: List[str] = []
    connection_log: List[str] = []

    # Derive RTN-native lookup surfaces once.
    nodes_by_id: Dict[str, TaskNodeView] = {tid: TaskNodeView(t) for tid, t in rtn.tasks.items()}
    recipe_inputs: Dict[str, float] = rtn.derive_recipe_inputs()
    choice_groups: Dict[str, List[str]] = dict(rtn.choice_groups)
    precedence: List[Tuple[str, str]] = list(rtn.precedence)

    operations = sorted(plan.operations, key=lambda op: (op.start_s, op.end_s, op.step_id))
    op_by_node = {op.recipe_node_id: op for op in operations if op.recipe_node_id in nodes_by_id}
    selected = plan.selected_branches
    reservations = list(getattr(plan, "local_dose_reservations", []) or [])
    reservation_node_ids = {reservation.recipe_node_id for reservation in reservations}
    virtual_dose_nodes = (
        reservation_node_ids
        if reservations
        else _virtual_dose_node_ids(nodes_by_id, op_by_node, selected)
    )
    active_nodes = set(op_by_node) | virtual_dose_nodes
    process_compositions = _expected_process_compositions(
        nodes_by_id,
        precedence,
        active_nodes,
    )
    reservation_process_modules = _reservation_process_modules(
        reservations,
        nodes_by_id,
        precedence,
        op_by_node,
    )

    material_tokens = [
        MaterialEvent(
            batch_id=f"input_{ingr}",
            material=ingr,
            quantity_l=qty,
            location="Input",
            source_node_id="Formula.ProcessInputs",
            components={ingr: qty},
        )
        for ingr, qty in recipe_inputs.items()
    ]
    control_tokens: List[ControlEvent] = []
    resource_tokens = {
        wb: ResourceEvent(wb, float((vals or [0.0])[0]))
        for wb, vals in (module_maximum_volume or {}).items()
    }
    in_progress_history: List[InProgressEvent] = []

    _check_choice_groups(
        nodes_by_id,
        choice_groups,
        rtn.choice_group_policies,
        selected,
        active_nodes,
        errors,
    )
    _check_precedence(precedence, op_by_node, active_nodes, errors, tolerance)
    resource_errors: List[str] = []
    _check_resource_overlap(operations, resource_tokens, resource_errors, tolerance)
    errors.extend(resource_errors)
    resource_log = [
        f"{module} {start:g}–{end:g}s {label}"
        for module, resource in sorted(resource_tokens.items())
        for start, end, label in resource.busy_intervals
    ]
    if resource_errors:
        resource_log = [*resource_errors, "", *resource_log]
    terminal_connections = _check_connection_lifecycle(
        operations,
        connection_errors,
        connection_log,
        tolerance,
        require_final_disconnect=cfg.require_final_disconnect,
    )
    errors.extend(connection_errors)

    # Pre-compute virtual dose materials (self-transfers, unselected XOR doses)
    # so _check_transfer_inventory can use cumulative (not global) expected amounts
    # at each process step — matching the solver's per-stage cumulative semantics.
    if not reservations:
        reservations = [
            MaterialReservation(
                recipe_node_id=nid,
                process_module="",
                material=str(nodes_by_id[nid].params["ingredient"]),
                amount_l=float(nodes_by_id[nid].params["amount_L"]),
            )
            for nid in virtual_dose_nodes
        ]

    composition_errors: List[str] = []
    composition_log: List[str] = []
    _check_transfer_inventory(
        recipe_inputs,
        operations,
        resource_tokens,
        module_resources or {},
        reservations,
        process_compositions,
        reservation_process_modules,
        errors,
        composition_errors,
        composition_log,
        warnings,
        tolerance,
    )
    _check_capacity(operations, resource_tokens, errors, warnings, tolerance)

    current_mix_components: Dict[str, float] = {}
    for nid in virtual_dose_nodes:
        node = nodes_by_id[nid]
        ingr = str(node.params["ingredient"])
        qty = float(node.params["amount_L"])
        current_mix_components[ingr] = current_mix_components.get(ingr, 0.0) + qty
    for op in operations:
        in_progress_history.append(
            InProgressEvent(
                operation_id=f"op_{op.step_id:03d}",
                recipe_node_id=op.recipe_node_id,
                module_id=op.module,
                start_s=op.start_s,
                end_s=op.end_s,
            )
        )
        if op.operation_type == "aux_transfer":
            material = _single_material_name(op)
            qty = sum(op.material.values())
            material_tokens.append(
                MaterialEvent(
                    batch_id=f"{op.recipe_node_id}_{op.step_id}",
                    material=material or "auxiliary_transfer",
                    quantity_l=qty,
                    location=op.target_module,
                    source_node_id=op.recipe_node_id,
                    components=dict(op.material),
                )
            )
            continue
        if op.operation_type in {"connect", "disconnect"}:
            continue

        node = nodes_by_id[op.recipe_node_id]
        if node.node_type == "dose":
            ingr = str(node.params["ingredient"])
            qty = float(node.params["amount_L"])
            current_mix_components[ingr] = current_mix_components.get(ingr, 0.0) + qty
            material_tokens.append(
                MaterialEvent(
                    batch_id=f"{op.recipe_node_id}_{op.step_id}",
                    material=ingr,
                    quantity_l=qty,
                    location=op.module,
                    source_node_id=op.recipe_node_id,
                    components={ingr: qty},
                )
            )
        elif node.node_type == "mix":
            qty = sum(current_mix_components.values())
            material_tokens.append(
                MaterialEvent(
                    batch_id=f"{op.recipe_node_id}_{op.step_id}",
                    material="mixed_product",
                    quantity_l=qty,
                    location=op.module,
                    source_node_id=op.recipe_node_id,
                    components=dict(current_mix_components),
                )
            )
        elif node.node_type == "usage":
            material_tokens.append(
                MaterialEvent(
                    batch_id=f"{op.recipe_node_id}_{op.step_id}",
                    material="used_product",
                    quantity_l=sum(current_mix_components.values()),
                    location=op.module,
                    source_node_id=op.recipe_node_id,
                    components=dict(current_mix_components),
                )
            )
        elif node.node_type == "settling":
            material_tokens.append(
                MaterialEvent(
                    batch_id=f"{op.recipe_node_id}_{op.step_id}",
                    material="settled_product",
                    quantity_l=sum(current_mix_components.values()),
                    location=op.module,
                    source_node_id=op.recipe_node_id,
                    components=dict(current_mix_components),
                )
            )
        elif node.node_type == "heating":
            material_tokens.append(
                MaterialEvent(
                    batch_id=f"{op.recipe_node_id}_{op.step_id}",
                    material="heated_product",
                    quantity_l=sum(current_mix_components.values()),
                    location=op.module,
                    source_node_id=op.recipe_node_id,
                    components=dict(current_mix_components),
                )
            )
        elif node.node_type == "separation":
            ingr = str(node.params["ingredient"])
            # Use the effective material amount (may differ from node.params
            # for shared separations after XOR join where some branches
            # don't dose this material).
            qty = sum(float(v) for v in op.material.values()) if op.material is not None else float(node.params.get("amount_L", 0.0))
            available = current_mix_components.get(ingr, 0.0)
            if available + tolerance < qty:
                errors.append(f"Separation {node.id} removes {qty} L {ingr}, but only {available} L is available.")
            current_mix_components[ingr] = max(0.0, available - qty)
            material_tokens.append(
                MaterialEvent(
                    batch_id=f"{op.recipe_node_id}_{op.step_id}",
                    material=f"separated_{ingr}",
                    quantity_l=qty,
                    location=op.module,
                    source_node_id=op.recipe_node_id,
                    components={ingr: qty},
                )
            )
        control_tokens.append(ControlEvent(op.recipe_node_id, op.branch_group_id, op.branch_id))

    total_cost = sum(op.total_cost for op in operations)
    total_duration = sum(op.duration_s for op in operations)
    makespan = max((op.end_s for op in operations), default=0.0) - min((op.start_s for op in operations), default=0.0)
    expected_profit = cfg.base_profit + cfg.lambda_per_second * makespan - total_cost
    if abs(expected_profit - plan.objective_profit) > max(1e-4, tolerance):
        warnings.append(
            f"Profit mismatch: replay={expected_profit:.6g}, planner={plan.objective_profit:.6g}."
        )

    return ValidationResult(
        valid=not errors,
        errors=errors,
        warnings=warnings,
        material_tokens=material_tokens,
        control_tokens=control_tokens,
        resource_tokens=resource_tokens,
        in_progress_history=in_progress_history,
        profit_check={
            "base_profit": cfg.base_profit,
            "lambda_per_second": cfg.lambda_per_second,
            "total_duration_s": total_duration,
            "makespan_s": makespan,
            "total_cost": total_cost,
            "profit": expected_profit,
        },
        connection_errors=connection_errors,
        connection_log=connection_log,
        terminal_connections=terminal_connections,
        resource_errors=resource_errors,
        resource_log=resource_log,
        composition_errors=composition_errors,
        composition_log=composition_log,
    )


def _virtual_dose_node_ids(
    nodes_by_id: Dict[str, TaskNodeView],
    op_by_node: Dict[str, PlannedOperation],
    selected: Dict[str, Any],
) -> set[str]:
    virtual: set[str] = set()
    for node in nodes_by_id.values():
        if node.node_type != "dose" or node.id in op_by_node:
            continue
        if node.branch_group_id.startswith(("XOR_", "OR_")):
            group_selection = selected.get(node.branch_group_id)
            is_selected = (
                node.branch_id in group_selection
                if isinstance(group_selection, (list, tuple, set))
                else group_selection == node.branch_id
            )
            if not is_selected:
                continue
        virtual.add(node.id)
    return virtual


def _check_choice_groups(
    nodes_by_id: Dict[str, TaskNodeView],
    choice_groups: Dict[str, List[str]],
    choice_group_policies: Dict[str, Dict[str, Any]],
    selected: Dict[str, Any],
    active_nodes: set[str],
    errors: List[str],
) -> None:
    for group, branches in choice_groups.items():
        chosen_value = selected.get(group)
        if group.startswith("XOR_"):
            chosen = [chosen_value] if isinstance(chosen_value, str) else []
            if not chosen or chosen[0] not in branches:
                errors.append(f"XOR group {group} has no valid selected branch.")
                continue
        elif group.startswith("OR_"):
            chosen = list(chosen_value) if isinstance(chosen_value, (list, tuple, set)) else []
            policy = choice_group_policies.get(group, {})
            minimum = int(policy.get("minBranches", 1))
            maximum = int(policy.get("maxBranches", len(branches)))
            if any(branch not in branches for branch in chosen):
                errors.append(f"OR group {group} selected an unknown branch: {chosen}.")
                continue
            if not minimum <= len(chosen) <= maximum:
                errors.append(
                    f"OR group {group} selected {len(chosen)} branches; expected {minimum}..{maximum}."
                )
                continue
        else:
            continue
        for node in nodes_by_id.values():
            if node.branch_group_id == group and node.branch_id:
                should_be_active = node.branch_id in chosen and node.is_task
                if should_be_active and node.id not in active_nodes:
                    errors.append(f"Selected branch {group}/{node.branch_id} missing active node {node.id}.")
                if not should_be_active and node.id in active_nodes:
                    errors.append(f"Inactive branch node {node.id} appears in the plan.")


def _check_precedence(
    precedence: List[Tuple[str, str]],
    op_by_node: Dict[str, PlannedOperation],
    active_nodes: set[str],
    errors: List[str],
    tolerance: float,
) -> None:
    for src, dst in precedence:
        if src not in active_nodes or dst not in active_nodes:
            continue
        if src not in op_by_node or dst not in op_by_node:
            continue
        if op_by_node[dst].start_s + tolerance < op_by_node[src].end_s:
            errors.append(
                f"Precedence violation: {dst} starts at {op_by_node[dst].start_s}, "
                f"before {src} ends at {op_by_node[src].end_s}."
            )


def _expected_process_compositions(
    nodes_by_id: Dict[str, TaskNodeView],
    precedence: List[Tuple[str, str]],
    active_nodes: set[str],
) -> Dict[str, Dict[str, float]]:
    """Derive each process step's recipe-prefix composition from its ancestors.

    This is graph-based rather than operation-time based: a Local Dose has no
    physical row, and a later Local Dose must not be counted at an earlier Mix.
    """
    predecessors: Dict[str, List[str]] = {node_id: [] for node_id in nodes_by_id}
    for source, target in precedence:
        if source in nodes_by_id and target in nodes_by_id:
            predecessors[target].append(source)

    expected: Dict[str, Dict[str, float]] = {}
    for node_id, node in nodes_by_id.items():
        if node.node_type not in {"mix", "usage", "settling", "heating"}:
            continue
        ancestors: set[str] = set()
        frontier = list(predecessors.get(node_id, []))
        while frontier:
            predecessor = frontier.pop()
            if predecessor in ancestors:
                continue
            ancestors.add(predecessor)
            frontier.extend(predecessors.get(predecessor, []))
        composition: Dict[str, float] = {}
        for ancestor_id in ancestors & active_nodes:
            ancestor = nodes_by_id[ancestor_id]
            if ancestor.node_type != "dose":
                continue
            material = str(ancestor.params.get("ingredient", "")).upper()
            amount = float(ancestor.params.get("amount_L", 0.0))
            if material and amount > 0:
                composition[material] = composition.get(material, 0.0) + amount
        expected[node_id] = normalize_composition(composition)
    return expected


def _reservation_process_modules(
    reservations: List[MaterialReservation],
    nodes_by_id: Dict[str, TaskNodeView],
    precedence: List[Tuple[str, str]],
    op_by_node: Dict[str, PlannedOperation],
) -> Dict[str, set[str]]:
    """Map each Local Dose to the graph-nearest selected downstream Mix Module."""
    successors: Dict[str, List[str]] = {node_id: [] for node_id in nodes_by_id}
    for source, target in precedence:
        if source in nodes_by_id and target in nodes_by_id:
            successors[source].append(target)

    result: Dict[str, set[str]] = {}
    for reservation in reservations:
        frontier = list(successors.get(reservation.recipe_node_id, []))
        seen = {reservation.recipe_node_id}
        while frontier:
            mixes = {
                op_by_node[node_id].module
                for node_id in frontier
                if node_id in op_by_node and nodes_by_id[node_id].node_type == "mix"
            }
            if mixes:
                result[reservation.recipe_node_id] = mixes
                break
            next_frontier: List[str] = []
            for node_id in frontier:
                if node_id in seen:
                    continue
                seen.add(node_id)
                next_frontier.extend(successors.get(node_id, []))
            frontier = sorted(set(next_frontier) - seen)
    return result


def _check_resource_overlap(
    operations: List[PlannedOperation],
    resources: Dict[str, ResourceEvent],
    errors: List[str],
    tolerance: float,
) -> None:
    exclusive_by_module: Dict[str, List[Tuple[float, float, str]]] = {}
    incoming_by_module: Dict[str, List[Tuple[float, float, str]]] = {}
    by_port: Dict[str, List[Tuple[float, float, str]]] = {}
    by_global_connect: List[Tuple[float, float, str]] = []
    for op in operations:
        label = f"{op.recipe_node_id} ({op.operation_type})"
        if op.operation_type in {"connect", "disconnect"}:
            by_global_connect.append((op.start_s, op.end_s, op.recipe_node_id))
            if op.out_port and op.in_port:
                by_port.setdefault(f"{op.source_module}.{op.out_port}", []).append(
                    (op.start_s, op.end_s, op.recipe_node_id)
                )
                by_port.setdefault(f"{op.target_module}.{op.in_port}", []).append(
                    (op.start_s, op.end_s, op.recipe_node_id)
                )
        elif op.out_port and op.in_port:
            if op.source_module:
                exclusive_by_module.setdefault(op.source_module, []).append((op.start_s, op.end_s, f"{label} · Draining"))
            if op.target_module and op.target_module != op.source_module:
                incoming_by_module.setdefault(op.target_module, []).append((op.start_s, op.end_s, f"{label} · Filling"))
            by_port.setdefault(f"{op.source_module}.{op.out_port}", []).append((op.start_s, op.end_s, op.recipe_node_id))
            by_port.setdefault(f"{op.target_module}.{op.in_port}", []).append((op.start_s, op.end_s, op.recipe_node_id))
            connect_end = min(op.end_s, op.start_s + max(0.0, op.connect_duration_s))
            if connect_end > op.start_s + tolerance:
                by_global_connect.append((op.start_s, connect_end, op.recipe_node_id))
        elif op.module and "->" not in op.module:
            exclusive_by_module.setdefault(op.module, []).append((op.start_s, op.end_s, label))
    by_global_connect.sort(key=lambda interval: interval[0])
    for prev, curr in zip(by_global_connect, by_global_connect[1:]):
        if curr[0] + tolerance < prev[1]:
            errors.append(
                f"Global connect overlap: {prev[2]} [{prev[0]}, {prev[1]}] "
                f"and {curr[2]} [{curr[0]}, {curr[1]}]."
            )
    for wb in sorted(set(exclusive_by_module) | set(incoming_by_module)):
        intervals = sorted(exclusive_by_module.get(wb, []), key=lambda interval: interval[0])
        incoming = sorted(incoming_by_module.get(wb, []), key=lambda interval: interval[0])
        intervals.sort(key=lambda interval: interval[0])
        resources.setdefault(wb, ResourceEvent(wb, 0.0))
        for start, end, label in sorted([*intervals, *incoming], key=lambda interval: interval[0]):
            resources[wb].busy_intervals.append((start, end, label))
        for prev, curr in zip(intervals, intervals[1:]):
            if curr[0] + tolerance < prev[1]:
                errors.append(
                    f"Resource overlap on {wb}: {prev[2]} [{prev[0]}, {prev[1]}] "
                    f"and {curr[2]} [{curr[0]}, {curr[1]}]."
                )
        for exclusive in intervals:
            for filling in incoming:
                if filling[0] + tolerance < exclusive[1] and exclusive[0] + tolerance < filling[1]:
                    errors.append(
                        f"Resource overlap on {wb}: {exclusive[2]} [{exclusive[0]}, {exclusive[1]}] "
                        f"and {filling[2]} [{filling[0]}, {filling[1]}]."
                    )
    for port, intervals in by_port.items():
        intervals.sort(key=lambda interval: interval[0])
        for prev, curr in zip(intervals, intervals[1:]):
            if curr[0] + tolerance < prev[1]:
                errors.append(
                    f"Port overlap on {port}: {prev[2]} [{prev[0]}, {prev[1]}] "
                    f"and {curr[2]} [{curr[0]}, {curr[1]}]."
                )


def _check_connection_lifecycle(
    operations: List[PlannedOperation],
    errors: List[str],
    log: List[str],
    tolerance: float,
    require_final_disconnect: bool = True,
) -> List[str]:
    """Replay persistent Connect/Disconnect state and transfer eligibility."""
    events: List[Tuple[float, int, PlannedOperation]] = []
    for operation in operations:
        if operation.operation_type == "disconnect":
            events.append((operation.end_s, 0, operation))
        elif operation.operation_type == "connect":
            events.append((operation.end_s, 1, operation))
        elif operation.out_port and operation.in_port and operation.operation_type in {
            "dose", "separation", "transfer", "aux_transfer"
        }:
            events.append((operation.start_s, 2, operation))

    active: Dict[Tuple[str, str, str, str], str] = {}
    active_by_port: Dict[Tuple[str, str], Tuple[str, str, str, str]] = {}

    for event_time, _, operation in sorted(events, key=lambda item: (item[0], item[1], item[2].step_id)):
        pair = (operation.source_module, operation.out_port, operation.target_module, operation.in_port)
        source_port = (operation.source_module, operation.out_port)
        target_port = (operation.target_module, operation.in_port)
        signature = _connection_material_signature(operation)

        if operation.operation_type == "connect":
            occupying = {active_by_port.get(source_port), active_by_port.get(target_port)} - {None}
            if pair in active:
                errors.append(
                    f"Duplicate Connect on {operation.connection_path} at {event_time:g}s; "
                    "the port pair is already connected."
                )
                continue
            if occupying:
                occupied_text = ", ".join(_connection_pair_text(item) for item in sorted(occupying))
                errors.append(
                    f"Connect on {operation.connection_path} at {event_time:g}s reuses an occupied port; "
                    f"Disconnect required first ({occupied_text})."
                )
                continue
            active[pair] = signature
            active_by_port[source_port] = pair
            active_by_port[target_port] = pair
            log.append(
                f"{event_time:g}s CONNECTED {operation.connection_path} [{signature or 'legacy/unspecified'}]"
            )
            continue

        if operation.operation_type == "disconnect":
            current_signature = active.get(pair)
            if current_signature is None:
                errors.append(
                    f"Disconnect on {operation.connection_path} at {event_time:g}s has no matching active connection."
                )
                continue
            if signature and signature != "*" and current_signature not in {"*", signature}:
                errors.append(
                    f"Disconnect material signature mismatch on {operation.connection_path}: "
                    f"active={current_signature}, action={signature}."
                )
                continue
            active.pop(pair, None)
            active_by_port.pop(source_port, None)
            active_by_port.pop(target_port, None)
            log.append(f"{event_time:g}s DISCONNECTED {operation.connection_path}")
            continue

        current_signature = active.get(pair)
        if current_signature is None:
            competing = {active_by_port.get(source_port), active_by_port.get(target_port)} - {None}
            detail = (
                f"; occupied by {', '.join(_connection_pair_text(item) for item in sorted(competing))}"
                if competing else ""
            )
            errors.append(
                f"Transfer {operation.recipe_node_id} starts at {event_time:g}s without an active exact connection "
                f"on {operation.connection_path}{detail}."
            )
            continue
        if signature and current_signature not in {"*", signature}:
            errors.append(
                f"Transfer {operation.recipe_node_id} material signature mismatch on {operation.connection_path}: "
                f"active={current_signature}, transfer={signature}."
            )
            continue
        log.append(
            f"{event_time:g}s TRANSFER {operation.recipe_node_id} reused {operation.connection_path} "
            f"[{signature or current_signature}]"
        )

    terminal_connections = [
        f"{_connection_pair_text(pair)} [{signature}]"
        for pair, signature in sorted(active.items())
    ]
    for connection in terminal_connections:
        if require_final_disconnect:
            errors.append(
                f"Connection {connection} remains active at the end of the plan; "
                "an explicit Disconnect is required."
            )
        else:
            log.append(
                f"END ACTIVE {connection}; terminal teardown is disabled by planner policy."
            )
    return terminal_connections


def _connection_material_signature(operation: PlannedOperation) -> str:
    trace = operation.trace if isinstance(operation.trace, dict) else {}
    traced = str(trace.get("material_signature", "")).strip()
    if traced:
        return traced
    material = operation.material if isinstance(operation.material, dict) else {}
    signature = composition_signature(material)
    if signature:
        return signature
    return "*" if operation.operation_type in {"connect", "disconnect"} else ""


def _connection_pair_text(pair: Tuple[str, str, str, str]) -> str:
    return f"{pair[0]}.{pair[1]} -> {pair[2]}.{pair[3]}"


def _single_material_name(op: PlannedOperation) -> str:
    if len(op.material) == 1:
        return next(iter(op.material)).upper()
    physical_route = op.trace.get("physical_route") if isinstance(op.trace, dict) else None
    if isinstance(physical_route, dict) and physical_route.get("material"):
        return str(physical_route["material"]).upper()
    return ""


def _check_transfer_inventory(
    recipe_inputs: Dict[str, float],
    operations: List[PlannedOperation],
    resources: Dict[str, ResourceEvent],
    module_resources: Dict[str, List[Any]],
    reservations: List[MaterialReservation],
    process_compositions: Dict[str, Dict[str, float]],
    reservation_process_modules: Dict[str, set[str]],
    errors: List[str],
    composition_errors: List[str],
    composition_log: List[str],
    warnings: List[str],
    tolerance: float,
) -> None:
    """Replay physical transfers as start/remove and end/add events.

    This is intentionally independent from CP-SAT's inventory equations.  A
    plan is certified only when the replay proves source purity, proportional
    mixture movement, target compatibility, capacity, and process-location
    semantics from the emitted operations themselves.
    """
    if not module_resources:
        warnings.append("No initial Module resource metadata provided; transfer inventory replay check is partial.")

    inventory: Dict[str, Dict[str, float]] = {}
    for wb, raw in module_resources.items():
        material, qty = _resource_material_quantity(raw)
        if material and qty > tolerance:
            inventory.setdefault(wb, {})[material] = qty

    def report(message: str) -> None:
        composition_errors.append(message)
        errors.append(message)

    for reservation in reservations:
        material = str(reservation.material).upper()
        process_module = reservation.process_module
        if not process_module:
            # Backward-compatible validation of plans created before structured
            # reservations were added.  New planner results always provide the
            # exact Process Module.
            process_module = next(
                (
                    wb
                    for wb, stock in inventory.items()
                    if stock.get(material, 0.0) + tolerance >= reservation.amount_l
                    and is_pure_composition(stock, material)
                ),
                "",
            )
        stock = inventory.get(process_module, {})
        expected_process_modules = reservation_process_modules.get(reservation.recipe_node_id, set())
        if expected_process_modules and process_module not in expected_process_modules:
            report(
                f"Local Dose process binding violation: {reservation.recipe_node_id} reserves "
                f"material in {process_module or 'an unknown Module'}, but its graph-nearest "
                f"downstream Mix is assigned to {sorted(expected_process_modules)}."
            )
        if not process_module or not is_pure_composition(stock, material):
            report(
                f"Local Dose composition violation: {reservation.recipe_node_id} reserves "
                f"{reservation.amount_l:g} L {material} in {process_module or 'an unknown Module'}, "
                f"but its initial composition is {normalize_composition(stock)}."
            )
        elif stock.get(material, 0.0) + tolerance < reservation.amount_l:
            report(
                f"Local Dose inventory violation: {reservation.recipe_node_id} reserves "
                f"{reservation.amount_l:g} L {material} in {process_module}, but only "
                f"{stock.get(material, 0.0):g} L is available."
            )
        else:
            composition_log.append(
                f"RESERVE {reservation.recipe_node_id}: {reservation.amount_l:g} L {material} "
                f"in process Module {process_module}."
            )

    transfer_operations = [
        operation
        for operation in operations
        if operation.out_port
        and operation.in_port
        and operation.operation_type not in {"connect", "disconnect"}
        and normalize_composition(operation.material, tolerance=tolerance)
    ]
    process_operations = [
        operation
        for operation in operations
        if operation.operation_type in {"mix", "usage", "settling", "heating"}
    ]
    # priority: arrivals first, process-state checks second, departures third.
    # Thus a transfer ending exactly when its successor starts is available.
    events: List[Tuple[float, int, str, PlannedOperation]] = []
    for operation in transfer_operations:
        events.append((operation.end_s, 0, "arrival", operation))
        events.append((operation.start_s, 2, "departure", operation))
    for operation in process_operations:
        events.append((operation.start_s, 1, "process", operation))

    for event_time, _, event_kind, op in sorted(
        events,
        key=lambda item: (item[0], item[1], item[3].step_id),
    ):
        composition = normalize_composition(op.material, tolerance=tolerance)
        if event_kind == "process":
            expected = normalize_composition(
                process_compositions.get(op.recipe_node_id, {}), tolerance=tolerance
            )
            actual = normalize_composition(inventory.get(op.module, {}), tolerance=tolerance)
            for material in sorted(set(expected) | set(actual)):
                expected_qty = expected.get(material, 0.0)
                actual_qty = actual.get(material, 0.0)
                if abs(actual_qty - expected_qty) > tolerance:
                    report(
                        f"Material location violation: {op.recipe_node_id} starts on {op.module} "
                        f"at {event_time:g}s with {actual_qty:g} L {material}; expected "
                        f"{expected_qty:g} L of the current recipe composition."
                    )
            composition_log.append(
                f"PROCESS {event_time:g}s {op.recipe_node_id} @ {op.module}: "
                f"composition {actual}."
            )
            continue

        kind = op.transfer_kind or classify_transfer_kind(composition)
        source_inventory = normalize_composition(
            inventory.setdefault(op.source_module, {}), tolerance=tolerance
        )
        target_inventory = normalize_composition(
            inventory.setdefault(op.target_module, {}), tolerance=tolerance
        )

        if event_kind == "departure":
            enough = all(
                source_inventory.get(material, 0.0) + tolerance >= quantity
                for material, quantity in composition.items()
            )
            if not enough:
                report(
                    f"Material inventory violation: {op.recipe_node_id} drains {composition} "
                    f"from {op.source_module} at {event_time:g}s, but the source contains "
                    f"{source_inventory}."
                )

            if op.operation_type == "separation":
                # Separation capability is the sole legal selective extraction.
                pass
            elif kind == PURE_MATERIAL:
                material = next(iter(composition), "")
                if not is_pure_composition(source_inventory, material):
                    report(
                        f"Source composition violation: {op.recipe_node_id} attempts a pure "
                        f"{material} transfer from {op.source_module} at {event_time:g}s, but the "
                        f"source composition is {source_inventory}. Only Separation may "
                        "selectively extract a component from a mixture."
                    )
            elif kind == MIXTURE:
                if not compositions_proportional(source_inventory, composition, tolerance=tolerance):
                    report(
                        f"Mixture transfer ratio violation: {op.recipe_node_id} transfers "
                        f"{composition} from {op.source_module} at {event_time:g}s, but the source "
                        f"composition is {source_inventory}."
                    )

            inventory[op.source_module] = subtract_composition(
                source_inventory, composition, tolerance=tolerance
            )
            composition_log.append(
                f"DEPART {event_time:g}s {op.recipe_node_id}: {composition} from "
                f"{op.source_module}; remaining {inventory[op.source_module]}."
            )
            continue

        if op.operation_type == "dose":
            non_recipe = sorted(set(target_inventory) - {name.upper() for name in recipe_inputs})
            if non_recipe:
                report(
                    f"Dose target contamination: {op.recipe_node_id} fills {composition} into "
                    f"{op.target_module}, which contains non-recipe materials {non_recipe}."
                )
        elif op.operation_type == "separation":
            material = next(iter(composition), "")
            if target_inventory and not is_pure_composition(target_inventory, material):
                report(
                    f"Separation target material violation: {op.recipe_node_id} fills "
                    f"{material} into {op.target_module}, whose composition is {target_inventory}."
                )
        elif kind == PURE_MATERIAL:
            material = next(iter(composition), "")
            if target_inventory and not is_pure_composition(target_inventory, material):
                report(
                    f"Pure transfer target violation: {op.recipe_node_id} fills {material} into "
                    f"{op.target_module}, whose composition is {target_inventory}."
                )
        elif target_inventory and not compositions_proportional(
            target_inventory, composition, tolerance=tolerance
        ):
            report(
                f"Mixture transfer target ratio violation: {op.recipe_node_id} fills "
                f"{composition} into {op.target_module}, whose composition is {target_inventory}."
            )

        inventory[op.target_module] = add_composition(
            target_inventory, composition, tolerance=tolerance
        )
        cap = resources.get(op.target_module, ResourceEvent(op.target_module, 0.0)).capacity_l
        total = sum(inventory[op.target_module].values())
        if cap and total > cap + tolerance:
            message = f"Capacity violation on {op.target_module}: {total:g} L > {cap:g} L."
            if message not in errors:
                errors.append(message)
        composition_log.append(
            f"ARRIVE {event_time:g}s {op.recipe_node_id}: {composition} into "
            f"{op.target_module}; composition {inventory[op.target_module]}."
        )

    for op in operations:
        if (op.out_port or op.in_port) and op.target_module not in resources:
            warnings.append(f"No capacity metadata provided for {op.target_module}; transfer capacity replay check is partial.")

    for module, composition in sorted(inventory.items()):
        composition_log.append(f"END {module}: {normalize_composition(composition, tolerance=tolerance)}.")


def _check_recipe_product_inventory(
    cumulative_expected: Dict[str, float],
    inventory: Dict[str, Dict[str, float]],
    op: PlannedOperation,
    errors: List[str],
    tolerance: float,
) -> None:
    expected = {material.upper(): float(qty) for material, qty in cumulative_expected.items()}
    actual = inventory.setdefault(op.module, {})
    materials = sorted(set(expected) | {material for material, qty in actual.items() if qty > tolerance})
    for material in materials:
        actual_qty = actual.get(material, 0.0)
        expected_qty = expected.get(material, 0.0)
        if actual_qty + tolerance < expected_qty:
            errors.append(
                f"Material location violation: {op.recipe_node_id} runs on {op.module}, "
                f"but {op.module} has only {actual_qty} L {material}; expected {expected_qty} L of recipe product."
            )
        elif actual_qty > expected_qty + tolerance:
            errors.append(
                f"Material location violation: {op.recipe_node_id} runs on {op.module}, "
                f"but {op.module} has extra {actual_qty - expected_qty} L {material} before the recipe product operation."
            )


# --------------------------------------------------------------------------
# Deprecated legacy names.
#
# This module was historically tied to "CPN tokens" terminology, but it is a
# post-hoc constraint validator, not a Petri net simulator. The current public
# names are ``MaterialEvent`` / ``ControlEvent`` / ``ResourceEvent`` /
# ``InProgressEvent`` / ``ValidationResult`` / ``validate_schedule``. The old
# names below are still resolvable for backward compatibility but emit
# ``DeprecationWarning`` on access via the module-level ``__getattr__`` hook.
# --------------------------------------------------------------------------
import warnings as _warnings

_DEPRECATED_ALIASES: Dict[str, str] = {
    "MaterialToken": "MaterialEvent",
    "ControlToken": "ControlEvent",
    "ResourceToken": "ResourceEvent",
    "InProgressToken": "InProgressEvent",
    "ReplayResult": "ValidationResult",
}


def replay_plan_with_cpn(*args, **kwargs):
    """Deprecated. Use :func:`validate_schedule` instead."""
    _warnings.warn(
        "replay_plan_with_cpn is deprecated; use validate_schedule instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return validate_schedule(*args, **kwargs)


def __getattr__(name: str):
    target = _DEPRECATED_ALIASES.get(name)
    if target is not None:
        _warnings.warn(
            f"schedule_validator.{name} is deprecated; use {target} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return globals()[target]
    raise AttributeError(f"module 'schedule_validator' has no attribute {name!r}")


def _resource_material_quantity(raw: Any) -> Tuple[str, float]:
    if not raw:
        return "", 0.0
    if isinstance(raw, dict):
        return str(raw.get("material", "")).upper(), float(raw.get("quantity", 0.0))
    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        return str(raw[0]).upper(), float(raw[1])
    return "", 0.0


def _check_capacity(
    operations: List[PlannedOperation],
    resources: Dict[str, ResourceEvent],
    errors: List[str],
    warnings: List[str],
    tolerance: float,
) -> None:
    by_module_amount: Dict[str, float] = {}
    for op in operations:
        if op.operation_type == "dose":
            by_module_amount[op.module] = by_module_amount.get(op.module, 0.0) + sum(op.material.values())
            cap = resources.get(op.module, ResourceEvent(op.module, 0.0)).capacity_l
            if cap and by_module_amount[op.module] > cap + tolerance:
                errors.append(f"Capacity violation on {op.module}: {by_module_amount[op.module]} L > {cap} L.")
    for wb, qty in by_module_amount.items():
        if wb not in resources:
            warnings.append(f"No capacity metadata provided for {wb}; capacity replay check is partial.")
