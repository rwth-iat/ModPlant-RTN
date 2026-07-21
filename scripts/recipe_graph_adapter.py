"""Adapters between the legacy RTN RecipeIR and canonical RecipeGraph v2."""
from __future__ import annotations

import sys
from pathlib import Path

_SHARED = Path(__file__).resolve().parents[1] / "shared"
if str(_SHARED) not in sys.path:
    sys.path.insert(0, str(_SHARED))

from modplant_recipe import RecipeEdge, RecipeGraph, RecipeNode, normalize_for_planning

try:
    from recipe_ir import RecipeIR, RecipeNode as IRNode
except ImportError:  # pragma: no cover
    from .recipe_ir import RecipeIR, RecipeNode as IRNode


_IR_TO_GATEWAY = {
    "and_split": "AND_SPLIT",
    "and_join": "AND_JOIN",
    "xor_split": "XOR_SPLIT",
    "xor_join": "XOR_JOIN",
    "or_split": "OR_SPLIT",
    "or_join": "OR_JOIN",
}
_GATEWAY_TO_IR = {value: key for key, value in _IR_TO_GATEWAY.items()}


def recipe_ir_to_graph(ir: RecipeIR, *, recipe_level: str = "General") -> RecipeGraph:
    predecessors = ir.predecessors()
    successors = ir.successors()
    graph = RecipeGraph(
        id=ir.id,
        name=str(ir.metadata.get("name", ir.id)),
        recipe_level=recipe_level,
        conformance=["Module-RoundTrip", "Module-Planning", "Module-Execution"],
        metadata={**ir.metadata, "volume": ir.volume, "inputs": ir.inputs, "outputs": ir.outputs},
    )
    starts = sorted(node_id for node_id in ir.nodes if not predecessors.get(node_id))
    ends = sorted(node_id for node_id in ir.nodes if not successors.get(node_id))
    graph.nodes.append(RecipeNode("__start__", "Start", "Start"))
    for node in ir.topological_nodes():
        if node.is_control:
            graph.nodes.append(
                RecipeNode(
                    node.id,
                    "Gateway",
                    node.name,
                    gateway_type=_IR_TO_GATEWAY[node.node_type],
                    metadata={
                        "gatewayGroupId": node.branch_group_id or node.id,
                        "joinPolicy": node.join_policy,
                        "semanticUri": node.semantic_uri,
                        **{
                            key: node.params[key]
                            for key in ("minBranches", "maxBranches")
                            if key in node.params
                        },
                    },
                )
            )
        else:
            graph.nodes.append(
                RecipeNode(
                    node.id,
                    "Activity",
                    node.name,
                    activity_type=node.node_type,
                    parameters=dict(node.params),
                    metadata={
                        "activityType": node.node_type,
                        "branchGroupId": node.branch_group_id,
                        "branchId": node.branch_id,
                        "semanticUri": node.semantic_uri,
                        "processElementType": "Process Operation",
                    },
                )
            )
    graph.nodes.append(RecipeNode("__end__", "End", "End"))
    edge_index = 1
    for node_id in starts:
        graph.edges.append(RecipeEdge(f"edge_{edge_index:04d}", "__start__", node_id))
        edge_index += 1
    for source, target in ir.edges:
        source_node = ir.nodes[source]
        target_node = ir.nodes[target]
        is_split = source_node.node_type.endswith("_split")
        branch_node = target_node if is_split else source_node
        # A split's outgoing edge carries its branch guard (if any). The guard
        # lives on the split node's ``branchConditions`` map; restore it onto the
        # edge, or a choice the running process makes looks like one the planner
        # already made, and its other branches are dropped from the recipe.
        branch_condition = None
        if is_split:
            branch_conditions = (source_node.params or {}).get("branchConditions") or {}
            branch_condition = branch_conditions.get(branch_node.branch_id)
        graph.edges.append(
            RecipeEdge(
                f"edge_{edge_index:04d}",
                source,
                target,
                branch_id=branch_node.branch_id,
                gateway_group_id=branch_node.branch_group_id,
                condition=branch_condition,
            )
        )
        edge_index += 1
    for node_id in ends:
        graph.edges.append(RecipeEdge(f"edge_{edge_index:04d}", node_id, "__end__"))
        edge_index += 1
    return graph


def recipe_graph_to_display_ir(graph: RecipeGraph) -> RecipeIR:
    """Create a display-only IR without resolving runtime loops or jumps."""
    ir = RecipeIR(
        id=graph.id,
        volume=float(graph.metadata.get("volume", 0.0)),
        inputs={str(key): float(value) for key, value in graph.metadata.get("inputs", {}).items()},
        outputs={str(key): float(value) for key, value in graph.metadata.get("outputs", {}).items()},
        metadata={key: value for key, value in graph.metadata.items() if key not in {"volume", "inputs", "outputs"}},
    )
    edge_labels: dict[str, str] = {}
    for node in graph.nodes:
        if node.kind == "Gateway":
            node_type = _GATEWAY_TO_IR.get(node.gateway_type)
            if node_type is None:
                raise ValueError(f"Cannot display unsupported gateway {node.gateway_type}")
            ir.nodes[node.id] = IRNode(
                id=node.id,
                node_type=node_type,
                name=node.name or node.id,
                semantic_uri=str(node.metadata.get("semanticUri", f"urn:modplant:control#{node.gateway_type}")),
                control_node_type=node.gateway_type,
                branch_group_id=str(node.metadata.get("gatewayGroupId", node.id)),
                join_policy=str(node.metadata.get("joinPolicy", "")),
                params={
                    key: int(node.metadata[key])
                    for key in ("minBranches", "maxBranches")
                    if key in node.metadata
                },
            )
        elif node.kind == "LoopRegion":
            loop = node.loop or {}
            ir.nodes[node.id] = IRNode(
                id=node.id,
                node_type="loop_region",
                name=node.name or node.id,
                params={
                    key: loop[key]
                    for key in ("minIterations", "maxIterations", "planningIterations")
                    if key in loop
                },
            )
        else:
            node_type = (
                node.activity_type
                if node.kind == "Activity" and node.activity_type
                else node.kind.lower()
            )
            ir.nodes[node.id] = IRNode(
                id=node.id,
                node_type=node_type,
                name=node.name or node.id,
                params=dict(node.parameters),
                branch_group_id=str(node.metadata.get("branchGroupId", "")),
                branch_id=str(node.metadata.get("branchId", "")),
            )

    for edge in graph.edges:
        if edge.source not in ir.nodes or edge.target not in ir.nodes:
            continue
        ir.edges.append((edge.source, edge.target))
        if edge.flow_type != "Control":
            edge_labels[f"{edge.source}->{edge.target}"] = edge.flow_type
        target = ir.nodes[edge.target]
        if edge.branch_id and not target.is_control:
            ir.nodes[edge.target] = IRNode(
                **{
                    **target.__dict__,
                    "branch_group_id": edge.gateway_group_id or target.branch_group_id,
                    "branch_id": edge.branch_id,
                }
            )
    if edge_labels:
        ir.metadata["edgeLabels"] = edge_labels
    ir.topological_nodes()
    return ir


def recipe_graph_to_ir(
    graph: RecipeGraph,
    *,
    planning_context: dict | None = None,
    loop_iterations: dict[str, int] | None = None,
    jump_decisions: dict[str, bool] | None = None,
) -> RecipeIR:
    if any(node.kind == "LoopRegion" for node in graph.nodes) or any(edge.flow_type == "Jump" for edge in graph.edges):
        graph = normalize_for_planning(
            graph,
            context=planning_context,
            loop_iterations=loop_iterations,
            jump_decisions=jump_decisions,
        )
    ir = RecipeIR(
        id=graph.id,
        volume=float(graph.metadata.get("volume", 0.0)),
        inputs={str(key): float(value) for key, value in graph.metadata.get("inputs", {}).items()},
        outputs={str(key): float(value) for key, value in graph.metadata.get("outputs", {}).items()},
        metadata={key: value for key, value in graph.metadata.items() if key not in {"volume", "inputs", "outputs"}},
    )
    for node in graph.nodes:
        if node.kind in {"Start", "End", "Transition"}:
            continue
        if node.kind == "Gateway":
            node_type = _GATEWAY_TO_IR.get(node.gateway_type)
            if node_type is None:
                raise ValueError(f"RTN PlanningGraph does not support {node.gateway_type}")
            ir.nodes[node.id] = IRNode(
                id=node.id,
                node_type=node_type,
                name=node.name or node.id,
                semantic_uri=str(node.metadata.get("semanticUri", f"urn:modplant:control#{node.gateway_type}")),
                control_node_type=node.gateway_type,
                branch_group_id=str(node.metadata.get("gatewayGroupId", node.id)),
                join_policy=str(
                    node.metadata.get(
                        "joinPolicy",
                        "wait_all"
                        if node.gateway_type.startswith("AND")
                        else "active_branches"
                        if node.gateway_type.startswith("OR")
                        else "exactly_one",
                    )
                ),
                params={
                    **{
                        key: int(node.metadata[key])
                        for key in ("minBranches", "maxBranches")
                        if key in node.metadata
                    },
                    "branchConditions": {
                        edge.branch_id or edge.id: edge.condition
                        for edge in graph.outgoing(node.id)
                        if edge.condition is not None
                    },
                },
            )
        else:
            parameters = dict(node.parameters)
            if "runtimeLoopId" in node.metadata:
                parameters["runtime_loop_id"] = node.metadata["runtimeLoopId"]
            if "runtimeLoopIteration" in node.metadata:
                parameters["runtime_loop_iteration"] = int(node.metadata["runtimeLoopIteration"])
            ir.nodes[node.id] = IRNode(
                id=node.id,
                node_type=node.activity_type or "operation",
                name=node.name or node.id,
                params=parameters,
                semantic_uri=str(node.metadata.get("semanticUri", "")),
                branch_group_id=str(node.metadata.get("branchGroupId", "")),
                branch_id=str(node.metadata.get("branchId", "")),
            )
    for edge in graph.edges:
        if edge.source in ir.nodes and edge.target in ir.nodes:
            ir.edges.append((edge.source, edge.target))
            target = ir.nodes[edge.target]
            if edge.branch_id and not target.is_control:
                ir.nodes[edge.target] = IRNode(
                    **{
                        **target.__dict__,
                        "branch_group_id": edge.gateway_group_id or target.branch_group_id,
                        "branch_id": edge.branch_id,
                    }
                )
    if not ir.inputs:
        totals: dict[str, float] = {}
        for node in ir.nodes.values():
            if node.node_type == "dose":
                ingredient = str(node.params.get("ingredient", ""))
                totals[ingredient] = totals.get(ingredient, 0.0) + float(node.params.get("amount_L", 0.0))
        ir.inputs = totals
        ir.volume = sum(totals.values())
        ir.outputs = {"Product": ir.volume}
    ir.topological_nodes()
    return ir
