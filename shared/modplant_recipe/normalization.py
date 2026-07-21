from __future__ import annotations

from copy import deepcopy
from typing import Any

from .conditions import evaluate_condition
from .models import RecipeEdge, RecipeGraph, RecipeNode
from .validation import validate_graph


def normalize_for_planning(
    graph: RecipeGraph,
    *,
    context: dict[str, Any] | None = None,
    loop_iterations: dict[str, int] | None = None,
    jump_decisions: dict[str, bool] | None = None,
) -> RecipeGraph:
    """Compile runtime control flow into a finite RTN planning DAG.

    Runtime loops must have a finite ``maxIterations`` planning envelope. A
    caller may select fewer iterations through ``loop_iterations``. Conditional
    jumps are resolved from ``jump_decisions`` first and then from the supplied
    condition ``context``. Missing jump inputs fail closed instead of silently
    selecting a path.
    """
    report = validate_graph(graph)
    if not report.valid:
        raise ValueError("Invalid RecipeGraph: " + "; ".join(report.errors))

    result = deepcopy(graph)
    result.recipe_level = graph.recipe_level
    result.conformance = sorted(set(result.conformance) | {"Module-Planning"})
    planning_metadata = result.metadata.setdefault("runtimePlanning", {})
    resolved_jumps = _resolve_jump_edges(
        result,
        context=context or {},
        decisions=jump_decisions or {},
    )
    if resolved_jumps:
        planning_metadata["jumpDecisions"] = resolved_jumps
        _prune_unreachable(result)

    selected_iterations: dict[str, int] = {}
    loop_nodes = [node for node in list(result.nodes) if node.kind == "LoopRegion"]
    for loop_node in loop_nodes:
        loop = loop_node.loop or {}
        bound = loop.get("maxIterations")
        if bound is None:
            raise ValueError(f"LoopRegion {loop_node.id} needs maxIterations for planning")
        minimum = int(loop.get("minIterations", 1))
        iterations = int((loop_iterations or {}).get(loop_node.id, loop.get("planningIterations", bound)))
        if not minimum <= iterations <= int(bound):
            raise ValueError(
                f"LoopRegion {loop_node.id} iterations={iterations} is outside {minimum}..{int(bound)}"
            )
        body = RecipeGraph.from_dict(loop["body"])
        body_report = validate_graph(body)
        if not body_report.valid:
            raise ValueError(f"LoopRegion {loop_node.id} body is invalid: {'; '.join(body_report.errors)}")
        _unroll_loop(result, loop_node, body, iterations)
        selected_iterations[loop_node.id] = iterations
    if selected_iterations:
        planning_metadata["loopIterations"] = selected_iterations
    _assert_dag(result)
    return result


def _resolve_jump_edges(
    graph: RecipeGraph,
    *,
    context: dict[str, Any],
    decisions: dict[str, bool],
) -> dict[str, bool]:
    jump_edges = [edge for edge in graph.edges if edge.flow_type == "Jump"]
    if not jump_edges:
        return {}
    resolved: dict[str, bool] = {}
    by_source: dict[str, list[RecipeEdge]] = {}
    for edge in jump_edges:
        if edge.id in decisions:
            take = bool(decisions[edge.id])
        elif edge.condition is None:
            raise ValueError(f"Conditional Jump {edge.id} needs a condition or an explicit jump decision")
        else:
            try:
                take = bool(evaluate_condition(edge.condition, context))
            except KeyError as exc:
                raise ValueError(
                    f"Conditional Jump {edge.id} cannot be planned without runtime context: {exc}"
                ) from exc
        resolved[edge.id] = take
        by_source.setdefault(edge.source, []).append(edge)

    removed_ids: set[str] = set()
    for source, source_jumps in by_source.items():
        taken = [edge for edge in source_jumps if resolved[edge.id]]
        if len(taken) > 1:
            raise ValueError(f"Multiple conditional jumps are true at {source}: {[edge.id for edge in taken]}")
        if taken:
            selected = taken[0]
            removed_ids.update(edge.id for edge in graph.outgoing(source) if edge.id != selected.id)
            selected.flow_type = "Control"
            selected.metadata = {**selected.metadata, "resolvedFromFlowType": "Jump"}
        else:
            removed_ids.update(edge.id for edge in source_jumps)
    graph.edges = [edge for edge in graph.edges if edge.id not in removed_ids]
    return resolved


def _prune_unreachable(graph: RecipeGraph) -> None:
    starts = [node.id for node in graph.nodes if node.kind == "Start"]
    ends = [node.id for node in graph.nodes if node.kind == "End"]
    successors: dict[str, list[str]] = {node.id: [] for node in graph.nodes}
    predecessors: dict[str, list[str]] = {node.id: [] for node in graph.nodes}
    for edge in graph.edges:
        successors.setdefault(edge.source, []).append(edge.target)
        predecessors.setdefault(edge.target, []).append(edge.source)

    def visit(seeds: list[str], adjacency: dict[str, list[str]]) -> set[str]:
        seen = set(seeds)
        queue = list(seeds)
        while queue:
            current = queue.pop()
            for other in adjacency.get(current, []):
                if other not in seen:
                    seen.add(other)
                    queue.append(other)
        return seen

    reachable = visit(starts, successors)
    productive = visit(ends, predecessors) if ends else set(successors)
    keep = reachable & productive
    graph.nodes = [node for node in graph.nodes if node.id in keep]
    graph.edges = [edge for edge in graph.edges if edge.source in keep and edge.target in keep]


def _unroll_loop(graph: RecipeGraph, loop_node: RecipeNode, body: RecipeGraph, bound: int) -> None:
    incoming = graph.incoming(loop_node.id)
    outgoing = graph.outgoing(loop_node.id)
    graph.nodes = [node for node in graph.nodes if node.id != loop_node.id]
    graph.edges = [edge for edge in graph.edges if edge.source != loop_node.id and edge.target != loop_node.id]

    body_starts = [node for node in body.nodes if node.kind == "Start"]
    body_ends = [node for node in body.nodes if node.kind == "End"]
    if len(body_starts) != 1 or len(body_ends) != 1:
        raise ValueError(f"LoopRegion {loop_node.id} body needs exactly one Start and one End")
    first_entries: list[str] = []
    previous_exits: list[str] = []
    if bound == 0:
        for incoming_edge in incoming:
            for outgoing_edge in outgoing:
                graph.edges.append(
                    RecipeEdge(
                        f"{incoming_edge.id}__skip__{outgoing_edge.id}",
                        incoming_edge.source,
                        outgoing_edge.target,
                    )
                )
        return
    for iteration in range(1, bound + 1):
        prefix = f"{loop_node.id}__i{iteration}__"
        internal_nodes = [node for node in body.nodes if node.kind not in {"Start", "End"}]
        for node in internal_nodes:
            copied = RecipeNode.from_dict({**node.to_dict(), "id": prefix + node.id})
            copied.metadata.update({"runtimeLoopId": loop_node.id, "runtimeLoopIteration": iteration})
            graph.nodes.append(copied)
        start_targets = [edge.target for edge in body.outgoing(body_starts[0].id)]
        end_sources = [edge.source for edge in body.incoming(body_ends[0].id)]
        entries = [prefix + node_id for node_id in start_targets if node_id != body_ends[0].id]
        exits = [prefix + node_id for node_id in end_sources if node_id != body_starts[0].id]
        if not first_entries:
            first_entries = entries
        for edge in body.edges:
            if edge.source in {body_starts[0].id, body_ends[0].id} or edge.target in {body_starts[0].id, body_ends[0].id}:
                continue
            copied = RecipeEdge.from_dict(edge.to_dict())
            copied.id = prefix + edge.id
            copied.source = prefix + edge.source
            copied.target = prefix + edge.target
            graph.edges.append(copied)
        for source in previous_exits:
            for target in entries:
                graph.edges.append(RecipeEdge(f"{source}__to__{target}", source, target))
        previous_exits = exits

    for edge in incoming:
        for target in first_entries:
            graph.edges.append(RecipeEdge(f"{edge.id}__{target}", edge.source, target, edge.flow_type))
    for source in previous_exits:
        for edge in outgoing:
            graph.edges.append(RecipeEdge(f"{source}__{edge.id}", source, edge.target, edge.flow_type))


def _assert_dag(graph: RecipeGraph) -> None:
    indegree = {node.id: 0 for node in graph.nodes}
    successors: dict[str, list[str]] = {node.id: [] for node in graph.nodes}
    for edge in graph.edges:
        if edge.flow_type == "Control":
            indegree[edge.target] += 1
            successors[edge.source].append(edge.target)
    ready = [node_id for node_id, degree in indegree.items() if degree == 0]
    seen = 0
    while ready:
        node_id = ready.pop()
        seen += 1
        for target in successors[node_id]:
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
    if seen != len(graph.nodes):
        raise ValueError("PlanningGraph contains a cycle; use a reducible LoopRegion")
