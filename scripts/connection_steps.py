"""Rebuild the Master Recipe's control flow from the solved plan.

The General Recipe says what has to happen; the plan says in which order it
actually can. Exporting the General Recipe's structure unchanged produced a
Master Recipe that no longer matched its own Gantt chart: the physical
connect/disconnect actions were missing, auxiliary transfers were missing, and
activities the plan had serialised were still drawn as parallel branches.

So the control flow is rebuilt here from the precedence
:mod:`plan_precedence` recovers from the schedule:

* every planned operation becomes a node - including the connect/disconnect
  actions a person performs and the auxiliary transfers the planner inserts;
* an element with two or more predecessors gets an AND_JOIN, and one with two
  or more successors an AND_SPLIT. A linear stretch stays linear, which is what
  IEC 61131-3 SFC asks for and what makes the chart readable;
* choice groups (XOR/OR) and loops are preserved whole, because they are
  runtime decisions the plan does not get to flatten.

AND split/join map onto BatchML's ParallelDivergent/ParallelConvergent, so the
result stays interoperable. Ordering lives in the control flow alone - it used
to be duplicated as SynchronizationLinks, which let an activity start before
its join had fired.
"""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Iterable

_SHARED = Path(__file__).resolve().parents[1] / "shared"
if str(_SHARED) not in sys.path:
    sys.path.insert(0, str(_SHARED))

from plan_precedence import (
    CONNECT_TYPES,
    derive_precedence,
    plan_operations,
    successors_of,
    transitive_reduction,
)
from modplant_recipe import RecipeEdge, RecipeNode

# Operations the planner introduces that the General Recipe never mentions.
PLANNER_ONLY_TYPES = CONNECT_TYPES | {"aux_transfer"}
CHOICE_SPLITS = {"XOR_SPLIT", "OR_SPLIT"}
CHOICE_JOINS = {"XOR_JOIN", "OR_JOIN"}
# Links that carry a token, and therefore define the order things run in.
CONTROL_FLOW = "Control"


def _route(operation) -> tuple[str, str, str, str]:
    return (
        str(getattr(operation, "source_module", "") or ""),
        str(getattr(operation, "out_port", "") or ""),
        str(getattr(operation, "target_module", "") or ""),
        str(getattr(operation, "in_port", "") or ""),
    )


def _kind(operation) -> str:
    return str(getattr(operation, "operation_type", "")).casefold()


def connection_actions(planner_result) -> list:
    """Connect/disconnect actions of the solved plan, in planned order."""
    return [
        operation for operation in plan_operations(planner_result)
        if _kind(operation) in CONNECT_TYPES
    ]


def _slugify(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)


def action_step_id(action) -> str:
    return f"conn_{_slugify(str(action.recipe_node_id))}"


# ---------------------------------------------------------------------------
# 1. the plan's own operations become nodes
# ---------------------------------------------------------------------------

def _connection_node(action, equipment_bindings) -> RecipeNode:
    source, out_port, target, in_port = _route(action)
    kind = _kind(action)
    binding = (equipment_bindings or {}).get(source, {})
    return RecipeNode(
        action_step_id(action), "Activity",
        f"{'Connect' if kind == 'connect' else 'Disconnect'} {out_port} → {in_port}",
        activity_type=kind,
        parameters={
            "planned_start_s": float(action.start_s),
            "planned_end_s": float(action.end_s),
            "planned_duration_s": float(action.duration_s),
            "source_module": source, "target_module": target,
            "out_port": out_port, "in_port": in_port,
        },
        metadata={
            "activityType": kind,
            "executionMode": "manual",
            "manualAction": kind,
            "actualEquipmentId": binding.get("mtp_equipment_id") or source,
            "sourceModule": source, "targetModule": target,
            "outPort": out_port, "inPort": in_port,
            "connectionPath": str(getattr(action, "connection_path", "")
                                  or f"{source}.{out_port} -> {target}.{in_port}"),
            "plannedStartS": float(action.start_s),
            "plannedEndS": float(action.end_s),
            "plannedDurationS": float(action.duration_s),
            "processElementType": "Process Operation",
        },
    )


def _auxiliary_node(operation) -> RecipeNode:
    """An auxiliary transfer the planner added to free up a vessel."""
    source, out_port, target, in_port = _route(operation)
    return RecipeNode(
        str(operation.recipe_node_id), "Activity",
        f"Auxiliary transfer {source} → {target}",
        activity_type="aux_transfer",
        parameters={
            "planned_start_s": float(operation.start_s),
            "planned_end_s": float(operation.end_s),
            "planned_duration_s": float(operation.duration_s),
            "source_module": source, "target_module": target,
            "out_port": out_port, "in_port": in_port,
        },
        metadata={
            "activityType": "aux_transfer",
            "sourceModule": source, "targetModule": target,
            "outPort": out_port, "inPort": in_port,
            "connectionPath": str(getattr(operation, "connection_path", "")
                                  or f"{source}.{out_port} -> {target}.{in_port}"),
            "plannedStartS": float(operation.start_s),
            "plannedEndS": float(operation.end_s),
            "plannedDurationS": float(operation.duration_s),
            "auxiliaryTransfer": True,
            "processElementType": "Process Operation",
        },
    )


def _materialise_planned_nodes(graph, operations, equipment_bindings) -> dict[str, str]:
    """Add a node per planner-only operation. Returns operation id -> node id."""
    known = {node.id for node in graph.nodes}
    node_of: dict[str, str] = {}
    for operation in operations:
        key = str(operation.recipe_node_id)
        if key in known:
            node_of[key] = key
            continue
        kind = _kind(operation)
        if kind in CONNECT_TYPES:
            node = _connection_node(operation, equipment_bindings)
        elif kind in PLANNER_ONLY_TYPES:
            node = _auxiliary_node(operation)
        else:
            continue
        graph.nodes.append(node)
        known.add(node.id)
        node_of[key] = node.id
    return node_of


# ---------------------------------------------------------------------------
# 2. what the recipe itself already said about order
# ---------------------------------------------------------------------------

def _control_successors(graph) -> dict[str, list[str]]:
    successors: dict[str, list[str]] = {}
    for edge in graph.edges:
        if edge.flow_type == CONTROL_FLOW:
            successors.setdefault(edge.source, []).append(edge.target)
    return successors


def _nearest(start: str, successors: dict[str, list[str]], wanted: set[str]) -> set[str]:
    """The first members of *wanted* met when walking forward from *start*."""
    found: set[str] = set()
    seen = {start}
    frontier = list(successors.get(start, ()))
    while frontier:
        node = frontier.pop()
        if node in seen:
            continue
        seen.add(node)
        if node in wanted:
            found.add(node)
            continue
        frontier.extend(successors.get(node, ()))
    return found


def recipe_precedence(graph, wanted: set[str]) -> dict[str, set[str]]:
    """Order the recipe declares between *wanted* nodes, gateways skipped.

    The schedule cannot always show this: a dose that finishes early and then
    waits for its sibling leaves no trace in the timings, yet the mix after it
    still depends on both.
    """
    successors = _control_successors(graph)
    precedence: dict[str, set[str]] = {node_id: set() for node_id in wanted}
    for node_id in wanted:
        for reachable in _nearest(node_id, successors, wanted):
            precedence.setdefault(reachable, set()).add(node_id)
    return precedence


# ---------------------------------------------------------------------------
# 3. blocks - the parts of the recipe the plan may not flatten
# ---------------------------------------------------------------------------

def _strongly_connected(successors: dict[str, list[str]], nodes: Iterable[str]) -> list[set[str]]:
    """Tarjan's algorithm, iterative - loops must survive the rebuild."""
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    on_stack: set[str] = set()
    stack: list[str] = []
    components: list[set[str]] = []
    counter = 0

    for root in nodes:
        if root in index:
            continue
        work: list[tuple[str, int]] = [(root, 0)]
        while work:
            node, child = work[-1]
            if child == 0:
                index[node] = low[node] = counter
                counter += 1
                stack.append(node)
                on_stack.add(node)
            children = successors.get(node, [])
            if child < len(children):
                work[-1] = (node, child + 1)
                nxt = children[child]
                if nxt not in index:
                    work.append((nxt, 0))
                elif nxt in on_stack:
                    low[node] = min(low[node], index[nxt])
                continue
            work.pop()
            if work:
                low[work[-1][0]] = min(low[work[-1][0]], low[node])
            if low[node] == index[node]:
                component = set()
                while True:
                    member = stack.pop()
                    on_stack.discard(member)
                    component.add(member)
                    if member == node:
                        break
                components.append(component)
    return components


def _choice_members(graph, gateway, successors: dict[str, list[str]]) -> set[str] | None:
    """A choice group's split, branches and join - or None if it has no join."""
    group = str(gateway.metadata.get("gatewayGroupId", gateway.id))
    join = next(
        (node.id for node in graph.nodes
         if node.gateway_type in CHOICE_JOINS
         and str(node.metadata.get("gatewayGroupId", node.id)) == group),
        None,
    )
    if join is None:
        return None
    members = {gateway.id, join}
    frontier = list(successors.get(gateway.id, ()))
    while frontier:
        node = frontier.pop()
        if node in members:
            continue
        members.add(node)
        frontier.extend(successors.get(node, ()))
    return members


def runtime_guarded_groups(graph) -> set[str]:
    """Choice groups the running process decides, not the planner.

    A branch carrying a real guard is still open at execution time, so every
    branch of its group has to survive into the Master Recipe. A group the
    planner resolved carries literal conditions instead, and only the branch it
    chose ever runs.
    """
    guarded: set[str] = set()
    for gateway in graph.nodes:
        if gateway.gateway_type not in CHOICE_SPLITS:
            continue
        group = str(gateway.metadata.get("gatewayGroupId", gateway.id))
        for edge in graph.outgoing(gateway.id):
            if edge.condition is not None and edge.condition.get("op") != "literal":
                guarded.add(group)
    return guarded


def _blocks(graph) -> list[set[str]]:
    """Groups of nodes that must keep their internal wiring."""
    successors = _control_successors(graph)
    guarded = runtime_guarded_groups(graph)
    groups: list[set[str]] = [
        component for component in
        _strongly_connected(successors, [node.id for node in graph.nodes])
        if len(component) > 1
    ]
    for gateway in graph.nodes:
        if (gateway.gateway_type in CHOICE_SPLITS
                and str(gateway.metadata.get("gatewayGroupId", gateway.id)) in guarded):
            members = _choice_members(graph, gateway, successors)
            if members:
                groups.append(members)

    merged: list[set[str]] = []
    for group in groups:
        overlapping = [other for other in merged if other & group]
        for other in overlapping:
            merged.remove(other)
            group = group | other
        merged.append(group)
    return merged


def _entry_and_exit(graph, members: set[str]) -> tuple[str, str]:
    """Where control enters and leaves a block."""
    inside = [edge for edge in graph.edges
              if edge.flow_type == CONTROL_FLOW
              and edge.source in members and edge.target in members]
    fed = {edge.target for edge in inside}
    feeds = {edge.source for edge in inside}
    node_map = graph.node_map

    def pick(candidates: set[str], gateway_types: set[str]) -> str:
        ranked = sorted(candidates or members)
        gateways = [item for item in ranked
                    if node_map[item].gateway_type in gateway_types]
        return gateways[0] if gateways else ranked[0]

    return (pick(members - fed, CHOICE_SPLITS), pick(members - feeds, CHOICE_JOINS))


# ---------------------------------------------------------------------------
# 4. the rebuild
# ---------------------------------------------------------------------------

def _control_predecessors(graph) -> dict[str, list[str]]:
    predecessors: dict[str, list[str]] = {}
    for edge in graph.edges:
        if edge.flow_type == CONTROL_FLOW:
            predecessors.setdefault(edge.target, []).append(edge.source)
    return predecessors


def _gateway(graph, node_id: str, gateway_type: str) -> str:
    graph.nodes.append(RecipeNode(
        node_id, "Gateway", gateway_type, gateway_type=gateway_type,
        metadata={"gatewayGroupId": node_id},
    ))
    return node_id


def inject_connection_steps(graph, planner_result, equipment_bindings=None) -> int:
    """Rebuild *graph*'s control flow from the plan. Returns the node count added.

    Every planned operation ends up on the control path in the order the solver
    actually found, with AND_SPLIT/AND_JOIN only where the flow really does
    branch or converge.
    """
    operations = plan_operations(planner_result)
    if not operations:
        return 0

    added_before = len(graph.nodes)
    node_of = _materialise_planned_nodes(graph, operations, equipment_bindings)
    planned = set(node_of.values())
    if not planned:
        return 0

    blocks = _blocks(graph)
    in_block = {member: index for index, block in enumerate(blocks) for member in block}

    # The recipe's own order, over the nodes the plan scheduled. Only nodes the
    # recipe already contained can carry it - a connection step is brand new.
    declared = recipe_precedence(graph, {key for key in node_of if node_of[key] == key})
    precedence = derive_precedence(planner_result, declared)
    precedence = {node_of.get(key, key): {node_of.get(parent, parent) for parent in parents}
                  for key, parents in precedence.items()}

    # A recipe element the plan gave no operation is not a step: the dose whose
    # material is already in the vessel, and the branches of a choice the
    # planner resolved. Only a runtime-guarded branch stays without an
    # operation, because the live process may still select it.
    unplanned = {
        node.id for node in graph.nodes
        if node.kind == "Activity" and node.id not in planned and node.id not in in_block
    }

    # Lift onto blocks: a choice group or a loop moves as one element.
    def block_key(node_id: str) -> str:
        index = in_block.get(node_id)
        return f"__block_{index}__" if index is not None else node_id

    lifted: dict[str, set[str]] = {}
    for node_id, parents in precedence.items():
        key = block_key(node_id)
        bucket = lifted.setdefault(key, set())
        for parent in parents:
            parent_key = block_key(parent)
            if parent_key != key:
                bucket.add(parent_key)
    for key in list(lifted):
        for parent in lifted[key]:
            lifted.setdefault(parent, set())

    lifted = transitive_reduction(lifted)
    forward = successors_of(lifted)

    # Anchors: where each element is entered and left.
    anchors: dict[str, tuple[str, str]] = {}
    for key in lifted:
        if key.startswith("__block_"):
            members = blocks[int(key[len("__block_"):-2])]
            anchors[key] = _entry_and_exit(graph, members)
        else:
            anchors[key] = (key, key)

    start = next((node.id for node in graph.nodes if node.kind == "Start"), None)
    end = next((node.id for node in graph.nodes if node.kind == "End"), None)
    if start is not None:
        anchors[start] = (start, start)
        lifted[start] = set()
        for key, parents in lifted.items():
            if key != start and not parents:
                parents.add(start)
        forward = successors_of(lifted)
    if end is not None:
        anchors[end] = (end, end)
        lifted[end] = {key for key, targets in forward.items()
                       if not targets and key not in {end, start}}
        forward = successors_of(lifted)

    # Drop every control edge the rebuild owns; a block keeps its own wiring.
    graph.edges = [
        edge for edge in graph.edges
        if edge.flow_type != CONTROL_FLOW
        or (in_block.get(edge.source) is not None
            and in_block.get(edge.source) == in_block.get(edge.target))
    ]

    entry_anchor: dict[str, str] = {}
    exit_anchor: dict[str, str] = {}
    for key, parents in lifted.items():
        entry, leave = anchors[key]
        if len(parents) > 1:
            join = _gateway(graph, f"and_join_{_slugify(entry)}", "AND_JOIN")
            graph.edges.append(RecipeEdge(f"ctrl_{join}_to_{entry}", join, entry))
            entry = join
        if len(forward.get(key, ())) > 1:
            split = _gateway(graph, f"and_split_{_slugify(leave)}", "AND_SPLIT")
            graph.edges.append(RecipeEdge(f"ctrl_{leave}_to_{split}", leave, split))
            leave = split
        entry_anchor[key], exit_anchor[key] = entry, leave

    for key, parents in lifted.items():
        for parent in sorted(parents):
            source, target = exit_anchor[parent], entry_anchor[key]
            graph.edges.append(RecipeEdge(f"ctrl_{source}_to_{target}", source, target))

    _drop_nodes(graph, unplanned)
    _drop_orphan_gateways(graph)
    return len(graph.nodes) - added_before


def _drop_nodes(graph, dropped: set[str]) -> None:
    if not dropped:
        return
    graph.nodes = [node for node in graph.nodes if node.id not in dropped]
    graph.edges = [edge for edge in graph.edges
                   if edge.source not in dropped and edge.target not in dropped]


def _drop_orphan_gateways(graph) -> None:
    """The recipe's own AND gateways are replaced by the plan's, not kept."""
    wired = {edge.source for edge in graph.edges} | {edge.target for edge in graph.edges}
    orphans = {node.id for node in graph.nodes
               if node.kind == "Gateway" and node.id not in wired}
    if not orphans:
        return
    graph.nodes = [node for node in graph.nodes if node.id not in orphans]


def enriched_display_ir(ir, planner_result, equipment_bindings=None):
    """The recipe as the Master Recipe will contain it, for display.

    The planner's own SFC used to show the General Recipe, so the connection
    steps and the ordering the plan relies on were invisible even though the
    exported Master Recipe carries them. Building the same enriched graph here
    keeps what the operator will run and what the planner shows in step.
    """
    try:
        from recipe_graph_adapter import recipe_graph_to_display_ir, recipe_ir_to_graph
    except ImportError:  # pragma: no cover
        return ir
    try:
        graph = recipe_ir_to_graph(ir, recipe_level="Master")
        inject_connection_steps(graph, planner_result, equipment_bindings)
        return recipe_graph_to_display_ir(graph)
    except Exception:  # noqa: BLE001 - display must never break the results view
        return ir
