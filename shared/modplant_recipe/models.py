from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class RecipeNode:
    id: str
    kind: str
    name: str = ""
    activity_type: str = ""
    gateway_type: str = ""
    procedure_chart_element_type: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    loop: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {"id": self.id, "kind": self.kind}
        optional = {
            "name": self.name,
            "activityType": self.activity_type,
            "gatewayType": self.gateway_type,
            "procedureChartElementType": self.procedure_chart_element_type,
            "parameters": self.parameters,
            "loop": self.loop,
            "metadata": self.metadata,
        }
        value.update({key: item for key, item in optional.items() if item not in ("", {}, None)})
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RecipeNode":
        return cls(
            id=str(value["id"]),
            kind=str(value["kind"]),
            name=str(value.get("name", "")),
            activity_type=str(value.get("activityType", "")),
            gateway_type=str(value.get("gatewayType", "")),
            procedure_chart_element_type=str(value.get("procedureChartElementType", "")),
            parameters=dict(value.get("parameters", {})),
            loop=value.get("loop"),
            metadata=dict(value.get("metadata", {})),
        )


@dataclass
class RecipeEdge:
    id: str
    source: str
    target: str
    flow_type: str = "Control"
    branch_id: str = ""
    condition: dict[str, Any] | None = None
    priority: int = 0
    is_default: bool = False
    gateway_group_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "id": self.id,
            "source": self.source,
            "target": self.target,
            "flowType": self.flow_type,
        }
        optional = {
            "branchId": self.branch_id,
            "condition": self.condition,
            "priority": self.priority if self.priority else None,
            "isDefault": True if self.is_default else None,
            "gatewayGroupId": self.gateway_group_id,
            "metadata": self.metadata,
        }
        value.update({key: item for key, item in optional.items() if item not in ("", {}, None)})
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RecipeEdge":
        return cls(
            id=str(value["id"]),
            source=str(value["source"]),
            target=str(value["target"]),
            flow_type=str(value.get("flowType", "Control")),
            branch_id=str(value.get("branchId", "")),
            condition=value.get("condition"),
            priority=int(value.get("priority", 0)),
            is_default=bool(value.get("isDefault", False)),
            gateway_group_id=str(value.get("gatewayGroupId", "")),
            metadata=dict(value.get("metadata", {})),
        )


@dataclass
class RecipeGraph:
    id: str
    recipe_level: str = "General"
    name: str = ""
    nodes: list[RecipeNode] = field(default_factory=list)
    edges: list[RecipeEdge] = field(default_factory=list)
    condition_language: str = "urn:modplant:condition:ast:v1"
    conformance: list[str] = field(default_factory=lambda: ["Module-RoundTrip"])
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "2.0"

    @property
    def node_map(self) -> dict[str, RecipeNode]:
        return {node.id: node for node in self.nodes}

    def incoming(self, node_id: str) -> list[RecipeEdge]:
        return [edge for edge in self.edges if edge.target == node_id]

    def outgoing(self, node_id: str) -> list[RecipeEdge]:
        return [edge for edge in self.edges if edge.source == node_id]

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schemaVersion": self.schema_version,
            "id": self.id,
            "recipeLevel": self.recipe_level,
            "conditionLanguage": self.condition_language,
            "conformance": list(self.conformance),
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
        }
        if self.name:
            value["name"] = self.name
        if self.metadata:
            value["metadata"] = self.metadata
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RecipeGraph":
        return cls(
            id=str(value["id"]),
            recipe_level=str(value.get("recipeLevel", "General")),
            name=str(value.get("name", "")),
            nodes=[RecipeNode.from_dict(node) for node in value.get("nodes", [])],
            edges=[RecipeEdge.from_dict(edge) for edge in value.get("edges", [])],
            condition_language=str(value.get("conditionLanguage", "urn:modplant:condition:ast:v1")),
            conformance=[str(item) for item in value.get("conformance", [])],
            metadata=dict(value.get("metadata", {})),
            schema_version=str(value.get("schemaVersion", "2.0")),
        )

