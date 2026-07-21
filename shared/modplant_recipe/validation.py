from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .models import RecipeGraph


@dataclass
class ValidationReport:
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    conformance: dict[str, bool] = field(default_factory=dict)


def _schema() -> dict:
    path = Path(__file__).resolve().parents[1] / "schemas" / "recipe-graph.schema.json"
    return json.loads(path.read_text(encoding="utf-8"))


def validate_graph(graph: RecipeGraph) -> ValidationReport:
    errors: list[str] = []
    warnings: list[str] = []
    try:
        from jsonschema import Draft202012Validator
    except ModuleNotFoundError:
        Draft202012Validator = None
    if Draft202012Validator is not None:
        schema_validator = Draft202012Validator(_schema())
        for error in sorted(schema_validator.iter_errors(graph.to_dict()), key=lambda item: list(item.path)):
            location = "/".join(str(item) for item in error.path) or "$"
            errors.append(f"JSON Schema {location}: {error.message}")

    node_ids = [node.id for node in graph.nodes]
    edge_ids = [edge.id for edge in graph.edges]
    if len(node_ids) != len(set(node_ids)):
        errors.append("Node IDs must be unique")
    if len(edge_ids) != len(set(edge_ids)):
        errors.append("Edge IDs must be unique")
    nodes = graph.node_map
    for edge in graph.edges:
        if edge.source not in nodes:
            errors.append(f"Edge {edge.id} has unknown source {edge.source}")
        if edge.target not in nodes:
            errors.append(f"Edge {edge.id} has unknown target {edge.target}")
        if edge.source == edge.target:
            errors.append(f"Edge {edge.id} is a self edge; use LoopRegion instead")

    for node in graph.nodes:
        incoming = graph.incoming(node.id)
        outgoing = graph.outgoing(node.id)
        if node.kind == "Start" and incoming:
            errors.append(f"Start node {node.id} has incoming edges")
        if node.kind == "End" and outgoing:
            errors.append(f"End node {node.id} has outgoing edges")
        if node.kind == "Gateway":
            if node.gateway_type.endswith("_SPLIT") and len(outgoing) < 2:
                errors.append(f"Split gateway {node.id} needs at least two outgoing edges")
            if node.gateway_type.endswith("_JOIN") and len(incoming) < 2:
                errors.append(f"Join gateway {node.id} needs at least two incoming edges")
            if node.gateway_type == "XOR_SPLIT":
                defaults = [edge for edge in outgoing if edge.is_default]
                if len(defaults) > 1:
                    errors.append(f"XOR split {node.id} has more than one default branch")
                if any(edge.condition is None for edge in outgoing if not edge.is_default):
                    warnings.append(f"XOR split {node.id} has an unconditional non-default branch")
        if node.kind == "LoopRegion":
            bound = (node.loop or {}).get("maxIterations")
            if bound is None:
                warnings.append(f"LoopRegion {node.id} is execution-only because it is unbounded")

    has_jump = any(edge.flow_type == "Jump" for edge in graph.edges)
    has_unbounded_loop = any(
        node.kind == "LoopRegion" and (node.loop or {}).get("maxIterations") is None
        for node in graph.nodes
    )
    has_xor = any(node.gateway_type.startswith("XOR_") for node in graph.nodes)
    conformance = {
        "BatchML-Core": not has_jump,
        "Module-RoundTrip": not errors,
        "Module-Planning": not errors and not has_jump and not has_unbounded_loop,
        "Module-Execution": not errors,
    }
    if graph.recipe_level == "General" and has_xor:
        conformance["BatchML-Core"] = False
    return ValidationReport(not errors, errors, warnings, conformance)
