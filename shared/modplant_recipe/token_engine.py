from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .conditions import evaluate_condition
from .models import RecipeEdge, RecipeGraph, RecipeNode
from .validation import validate_graph


class ExecutionError(RuntimeError):
    pass


@dataclass
class ExecutionResult:
    completed: bool
    trace: list[dict[str, Any]] = field(default_factory=list)
    completed_activities: list[str] = field(default_factory=list)
    selected_branches: dict[str, list[str]] = field(default_factory=dict)


def execute_graph(
    graph: RecipeGraph,
    context: dict[str, Any] | None = None,
    activity_handler: Callable[[RecipeNode, dict[str, Any]], None] | None = None,
    *,
    max_events: int = 10000,
) -> ExecutionResult:
    """Execute RecipeGraph control flow with deterministic token semantics."""
    report = validate_graph(graph)
    if not report.valid:
        raise ExecutionError("Invalid RecipeGraph: " + "; ".join(report.errors))
    context = context or {}
    handler = activity_handler or (lambda _node, _context: None)
    starts = [node for node in graph.nodes if node.kind == "Start"]
    if len(starts) != 1:
        raise ExecutionError("Execution requires exactly one Start node")

    node_map = graph.node_map
    queue: list[tuple[str, str]] = [(starts[0].id, "__start__")]
    join_tokens: dict[str, set[str]] = {}
    active_or_branches: dict[str, set[str]] = {}
    trace: list[dict[str, Any]] = []
    completed: list[str] = []
    selected: dict[str, list[str]] = {}
    events = 0

    def emit(edge: RecipeEdge) -> None:
        if edge.condition is not None and edge.flow_type == "Jump" and not bool(evaluate_condition(edge.condition, context)):
            trace.append({"event": "jump-not-taken", "edge": edge.id})
            return
        if edge.flow_type == "Jump":
            trace.append({"event": "jump-taken", "edge": edge.id, "target": edge.target})
        elif edge.flow_type in {"Transfer", "Synchronization"}:
            trace.append({"event": edge.flow_type.lower(), "edge": edge.id, "target": edge.target})
        queue.append((edge.target, edge.id))

    while queue:
        events += 1
        if events > max_events:
            raise ExecutionError("Execution exceeded max_events")
        node_id, incoming_edge_id = queue.pop(0)
        node = node_map[node_id]
        incoming_edges = graph.incoming(node_id)
        outgoing = sorted(graph.outgoing(node_id), key=lambda edge: (edge.priority, edge.id))
        trace.append({"event": "enter", "node": node_id, "fromEdge": incoming_edge_id})

        if node.kind == "End":
            trace.append({"event": "complete", "node": node_id})
            continue
        if node.kind == "Activity":
            handler(node, context)
            completed.append(node.id)
            trace.append({"event": "activity-complete", "node": node.id})
            for edge in outgoing:
                emit(edge)
            continue
        if node.kind == "Transition":
            condition = node.metadata.get("condition")
            if condition is not None and not bool(evaluate_condition(condition, context)):
                trace.append({"event": "transition-false", "node": node.id})
                continue
            for edge in outgoing:
                emit(edge)
            continue
        if node.kind == "LoopRegion":
            loop = node.loop or {}
            bound = loop.get("maxIterations")
            if bound is None:
                bound = (context.get("loopBounds") or {}).get(node.id)
            if bound is None:
                raise ExecutionError(f"LoopRegion {node.id} is unbounded and needs context.loopBounds.{node.id}")
            body = RecipeGraph.from_dict(loop["body"])
            for iteration in range(1, int(bound) + 1):
                nested = execute_graph(body, context, handler, max_events=max_events - events)
                completed.extend(nested.completed_activities)
                trace.append({"event": "loop-iteration", "node": node.id, "iteration": iteration})
                if bool(evaluate_condition(loop["exitCondition"], context)):
                    break
            else:
                raise ExecutionError(f"LoopRegion {node.id} reached maxIterations without satisfying exitCondition")
            for edge in outgoing:
                emit(edge)
            continue
        if node.kind == "Start":
            for edge in outgoing:
                emit(edge)
            continue
        if node.kind != "Gateway":
            raise ExecutionError(f"Unsupported node kind: {node.kind}")

        gateway = node.gateway_type
        group_id = str(node.metadata.get("gatewayGroupId", node.id))
        if gateway == "AND_SPLIT":
            selected[group_id] = [edge.branch_id or edge.id for edge in outgoing]
            for edge in outgoing:
                emit(edge)
        elif gateway in {"AND_JOIN", "OR_JOIN"}:
            arrived = join_tokens.setdefault(node.id, set())
            arrived.add(incoming_edge_id)
            required = {edge.id for edge in incoming_edges}
            if gateway == "OR_JOIN" and group_id in active_or_branches:
                required = {
                    edge.id for edge in incoming_edges
                    if (edge.branch_id or edge.id) in active_or_branches[group_id]
                }
            if required.issubset(arrived):
                join_tokens.pop(node.id, None)
                for edge in outgoing:
                    emit(edge)
        elif gateway in {"XOR_SPLIT", "OR_SPLIT"}:
            operator_choice = (context.get("operatorChoices") or {}).get(group_id)
            passing = [
                edge for edge in outgoing
                if edge.condition is not None and bool(evaluate_condition(edge.condition, context))
            ]
            if operator_choice is not None:
                passing = [edge for edge in outgoing if (edge.branch_id or edge.id) == operator_choice]
                if not passing:
                    raise ExecutionError(f"Operator choice {operator_choice!r} is not a branch of {node.id}")
            if not passing:
                passing = [edge for edge in outgoing if edge.is_default]
            if gateway == "XOR_SPLIT":
                if not passing:
                    raise ExecutionError(f"XOR split {node.id} has no true or default branch")
                passing = [passing[0]]
            elif not passing:
                raise ExecutionError(f"OR split {node.id} has no active branch")
            if gateway == "OR_SPLIT":
                minimum = int(node.metadata.get("minBranches", 1))
                maximum = int(node.metadata.get("maxBranches", len(outgoing)))
                if not minimum <= len(passing) <= maximum:
                    raise ExecutionError(
                        f"OR split {node.id} selected {len(passing)} branches; expected {minimum}..{maximum}"
                    )
            branch_ids = [edge.branch_id or edge.id for edge in passing]
            selected[group_id] = branch_ids
            if gateway == "OR_SPLIT":
                active_or_branches[group_id] = set(branch_ids)
            trace.append({"event": "branch-selected", "node": node.id, "branches": branch_ids})
            for edge in passing:
                emit(edge)
        elif gateway == "XOR_JOIN":
            for edge in outgoing:
                emit(edge)
        else:
            raise ExecutionError(f"Unsupported gateway type: {gateway}")

    end_entered = any(item["event"] == "enter" and node_map[item["node"]].kind == "End" for item in trace)
    return ExecutionResult(end_entered, trace, completed, selected)
