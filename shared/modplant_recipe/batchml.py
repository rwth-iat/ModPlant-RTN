from __future__ import annotations

import base64
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterable

from .conditions import parse_condition, render_condition
from .models import RecipeEdge, RecipeGraph, RecipeNode
from .validation import validate_graph


NS = "http://www.mesa.org/xml/B2MML"
XSI = "http://www.w3.org/2001/XMLSchema-instance"
ET.register_namespace("b2mml", NS)
ET.register_namespace("xsi", XSI)


class ExportCapabilityError(ValueError):
    pass


def _q(name: str) -> str:
    return f"{{{NS}}}{name}"


def _child(parent: ET.Element, name: str, text: Any | None = None, **attributes: str) -> ET.Element:
    node = ET.SubElement(parent, _q(name), attributes)
    if text is not None:
        node.text = str(text)
    return node


def _text(parent: ET.Element | None, name: str, default: str = "") -> str:
    if parent is None:
        return default
    value = parent.findtext(_q(name))
    return value if value is not None else default


def _load_xml(source: str | Path | ET.Element) -> ET.Element:
    if isinstance(source, ET.Element):
        return source
    if isinstance(source, Path) or (isinstance(source, str) and "<" not in source):
        return ET.parse(source).getroot()
    return ET.fromstring(str(source))


def _serialize(root: ET.Element, destination: str | Path | None) -> str:
    ET.indent(root, space="  ")
    xml = ET.tostring(root, encoding="unicode", xml_declaration=True)
    if destination is not None:
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        Path(destination).write_text(xml, encoding="utf-8")
    return xml


_GATEWAY_TO_CHART = {
    "AND_SPLIT": "Start Parallel Indicator",
    "AND_JOIN": "End Parallel Indicator",
    "XOR_SPLIT": "Start Optional Parallel Indicator",
    "XOR_JOIN": "End Optional Parallel Indicator",
    "OR_SPLIT": "Start Optional Parallel Indicator",
    "OR_JOIN": "End Optional Parallel Indicator",
}
_CHART_TO_GATEWAY = {
    "Start Parallel Indicator": "AND_SPLIT",
    "End Parallel Indicator": "AND_JOIN",
    "Start Optional Parallel Indicator": "OR_SPLIT",
    "End Optional Parallel Indicator": "OR_JOIN",
}


def export_general_recipe(
    graph: RecipeGraph,
    destination: str | Path | None = None,
    *,
    mode: str = "strict",
) -> str:
    report = validate_graph(graph)
    if not report.valid:
        raise ExportCapabilityError("Invalid RecipeGraph: " + "; ".join(report.errors))
    if mode not in {"strict", "annotated", "lossy"}:
        raise ValueError("mode must be strict, annotated, or lossy")
    xor_nodes = [node.id for node in graph.nodes if node.gateway_type.startswith("XOR_")]
    loop_nodes = [node.id for node in graph.nodes if node.kind == "LoopRegion"]
    losses: list[str] = []
    if xor_nodes:
        losses.append("General Recipe optional parallel indicators cannot distinguish XOR from OR: " + ", ".join(xor_nodes))
    if loop_nodes:
        losses.append("General Recipe core has no structured LoopRegion: " + ", ".join(loop_nodes))
    if losses and mode == "strict":
        raise ExportCapabilityError("; ".join(losses))

    root = ET.Element(_q("GRecipe"), {f"{{{XSI}}}schemaLocation": f"{NS} BatchML-GeneralRecipe.xsd"})
    _child(root, "ID", graph.id)
    _child(root, "Description", graph.name or "Module nonlinear General Recipe")
    _child(root, "GRecipeType", "General")
    _child(root, "LifeCycleState", "Draft")
    procedure = _child(root, "ProcessProcedure")
    _child(procedure, "ID", f"{graph.id}_ProcessProcedure")
    _child(procedure, "Description", "Canonical RecipeGraph procedure")
    _child(procedure, "ProcessElementType", "Process")
    _child(procedure, "LifeCycleState", "Draft")

    for edge in graph.edges:
        link = _child(procedure, "DirectedLink")
        _child(link, "ID", edge.id)
        description_parts = []
        if edge.flow_type != "Control":
            description_parts.append(f"flow={edge.flow_type}")
        if edge.branch_id:
            description_parts.append(f"branch={edge.branch_id}")
        if edge.gateway_group_id:
            description_parts.append(f"gatewayGroup={edge.gateway_group_id}")
        if edge.condition is not None:
            description_parts.append(f"condition={render_condition(edge.condition)}")
        if edge.is_default:
            description_parts.append("default=true")
        if description_parts:
            _child(link, "Description", "MODPLANT:" + ";".join(description_parts))
        _child(link, "FromID", edge.source)
        _child(link, "ToID", edge.target)

    for node in graph.nodes:
        if node.kind in {"Gateway", "Start", "End"}:
            chart = _child(procedure, "ProcedureChartElement")
            _child(chart, "ID", node.id)
            label = node.name or node.id
            if node.kind == "Gateway":
                group = node.metadata.get("gatewayGroupId", node.id)
                label = f"MODPLANT:{node.gateway_type};gatewayGroup={group};label={label}"
                for key in ("minBranches", "maxBranches", "joinPolicy"):
                    if key in node.metadata:
                        label += f";{key}={node.metadata[key]}"
            _child(chart, "Label", label)
            _child(chart, "Description", node.name or node.id)
            chart_type = (
                _GATEWAY_TO_CHART[node.gateway_type]
                if node.kind == "Gateway"
                else ("Previous Operation Indicator" if node.kind == "Start" else "Next Operation Indicator")
            )
            _child(chart, "ProcedureChartElementType", chart_type)
    for node in graph.nodes:
        if node.kind in {"Gateway", "Start", "End"}:
            continue
        if node.kind == "LoopRegion":
            activity = RecipeNode(node.id, "Activity", node.name or node.id, "LoopRegion", metadata={"loop": node.loop or {}})
            _append_general_process_element(procedure, activity)
            continue
        _append_general_process_element(procedure, node)

    for index, loss in enumerate(losses, start=1):
        info = _child(procedure, "OtherInformation")
        _child(info, "OtherInfoID", f"ModuleCapabilityWarning{index}")
        _child(info, "Description", loss)
        value = _child(info, "OtherValue")
        _child(value, "ValueString", loss)
        _child(value, "DataType", "string")
    for key, raw_value in sorted(graph.metadata.items()):
        info = _child(procedure, "OtherInformation")
        _child(info, "OtherInfoID", f"ModuleGraph.{key}")
        _child(info, "Description", f"Canonical RecipeGraph metadata {key}")
        value = _child(info, "OtherValue")
        _child(
            value,
            "ValueString",
            json.dumps(raw_value, separators=(",", ":"))
            if isinstance(raw_value, (dict, list, bool, int, float))
            else raw_value,
        )
        _child(value, "DataType", "string")
    return _serialize(root, destination)


def _append_general_process_element(parent: ET.Element, node: RecipeNode) -> None:
    process = _child(parent, "ProcessElement")
    _child(process, "ID", node.id)
    _child(process, "Description", node.name or node.id)
    process_type = str(node.metadata.get("processElementType", "Process Operation"))
    if process_type not in {"Process", "Process Stage", "Process Operation", "Process Action", "Other"}:
        process_type = "Process Operation"
    _child(process, "ProcessElementType", process_type)
    for key, raw_value in sorted(node.parameters.items()):
        parameter = _child(process, "ProcessElementParameter")
        _child(parameter, "ID", key)
        _child(parameter, "Description", key)
        value = _child(parameter, "Value")
        _child(value, "ValueString", raw_value)
        _child(value, "DataType", _data_type(raw_value))
    for key, raw_value in sorted(node.metadata.items()):
        if key == "processElementType":
            continue
        info = _child(process, "OtherInformation")
        _child(info, "OtherInfoID", key)
        _child(info, "Description", f"RecipeGraph metadata {key}")
        value = _child(info, "OtherValue")
        _child(value, "ValueString", json.dumps(raw_value, separators=(",", ":")) if isinstance(raw_value, (dict, list)) else raw_value)
        _child(value, "DataType", "string")


def import_general_recipe(source: str | Path | ET.Element) -> RecipeGraph:
    root = _load_xml(source)
    if root.tag != _q("GRecipe"):
        candidate = root.find(f".//{_q('GRecipe')}")
        if candidate is None:
            raise ValueError("No BatchML GRecipe element found")
        root = candidate
    graph = RecipeGraph(
        id=_text(root, "ID", "ImportedGeneralRecipe"),
        name=_text(root, "Description"),
        recipe_level="General",
        conformance=["Module-RoundTrip"],
    )
    procedure = root.find(_q("ProcessProcedure"))
    if procedure is None:
        return graph
    for info in procedure.findall(_q("OtherInformation")):
        key = _text(info, "OtherInfoID")
        if not key.startswith("ModuleGraph."):
            continue
        value = _text(info.find(_q("OtherValue")), "ValueString")
        graph.metadata[key.removeprefix("ModuleGraph.")] = _maybe_json(value)
    for chart in procedure.findall(f".//{_q('ProcedureChartElement')}"):
        node_id = _text(chart, "ID")
        if not node_id:
            continue
        chart_type = _text(chart, "ProcedureChartElementType")
        label = _text(chart, "Label")
        if chart_type == "Previous Operation Indicator":
            graph.nodes.append(RecipeNode(node_id, "Start", _text(chart, "Description", node_id), procedure_chart_element_type=chart_type))
            continue
        if chart_type == "Next Operation Indicator":
            graph.nodes.append(RecipeNode(node_id, "End", _text(chart, "Description", node_id), procedure_chart_element_type=chart_type))
            continue
        gateway_type = _CHART_TO_GATEWAY.get(chart_type)
        metadata: dict[str, Any] = {}
        match = re.match(r"MODPLANT:(AND|XOR|OR)_(SPLIT|JOIN);gatewayGroup=([^;]+)", label)
        if match:
            gateway_type = f"{match.group(1)}_{match.group(2)}"
            metadata["gatewayGroupId"] = match.group(3)
            for part in label.split(";"):
                key, _, value = part.partition("=")
                if key in {"minBranches", "maxBranches"} and value:
                    metadata[key] = int(value)
                elif key == "joinPolicy" and value:
                    metadata[key] = value
        if gateway_type:
            graph.nodes.append(
                RecipeNode(node_id, "Gateway", _text(chart, "Description", node_id), gateway_type=gateway_type, procedure_chart_element_type=chart_type, metadata=metadata)
            )
    for process in procedure.findall(f".//{_q('ProcessElement')}"):
        node_id = _text(process, "ID")
        if not node_id:
            continue
        parameters = {}
        for parameter in process.findall(_q("ProcessElementParameter")):
            parameters[_text(parameter, "ID")] = _coerce(_text(parameter.find(_q("Value")), "ValueString"))
        metadata = {"processElementType": _text(process, "ProcessElementType", "Process Operation")}
        for info in process.findall(_q("OtherInformation")):
            key = _text(info, "OtherInfoID")
            value = _text(info.find(_q("OtherValue")), "ValueString")
            if key:
                metadata[key] = _maybe_json(value)
        activity_type = str(metadata.get("activityType") or _infer_activity_type(process, metadata))
        if activity_type == "LoopRegion" or "loop" in metadata:
            graph.nodes.append(
                RecipeNode(
                    node_id,
                    "LoopRegion",
                    _text(process, "Description", node_id),
                    loop=dict(metadata.get("loop") or {}),
                    metadata={key: value for key, value in metadata.items() if key != "loop"},
                )
            )
        else:
            graph.nodes.append(RecipeNode(node_id, "Activity", _text(process, "Description", node_id), activity_type, parameters=parameters, metadata=metadata))
    for link in procedure.findall(f".//{_q('DirectedLink')}"):
        edge = RecipeEdge(_text(link, "ID", f"edge_{len(graph.edges)+1}"), _text(link, "FromID"), _text(link, "ToID"))
        description = _text(link, "Description")
        if description.startswith("MODPLANT:"):
            for part in description.removeprefix("MODPLANT:").split(";"):
                key, _, value = part.partition("=")
                if key == "flow": edge.flow_type = value
                elif key == "branch": edge.branch_id = value
                elif key == "gatewayGroup": edge.gateway_group_id = value
                elif key == "condition": edge.condition = parse_condition(value)
                elif key == "default": edge.is_default = value.lower() == "true"
        if edge.source and edge.target:
            graph.edges.append(edge)
    return graph


_GATEWAY_TO_LINK = {
    "AND_SPLIT": "ParallelDivergent",
    "AND_JOIN": "ParallelConvergent",
    "XOR_SPLIT": "SerialDivergent",
    "XOR_JOIN": "SerialConvergent",
    "OR_SPLIT": "SerialDivergent",
    "OR_JOIN": "SerialConvergent",
}
_LINK_TO_GATEWAY = {
    "ParallelDivergent": "AND_SPLIT",
    "ParallelConvergent": "AND_JOIN",
    "SerialDivergent": "XOR_SPLIT",
    "SerialConvergent": "XOR_JOIN",
}


def export_master_recipe(
    graph: RecipeGraph,
    destination: str | Path | None = None,
    *,
    selected_branches: dict[str, Any] | None = None,
) -> str:
    report = validate_graph(graph)
    if not report.valid:
        raise ExportCapabilityError("Invalid RecipeGraph: " + "; ".join(report.errors))
    if any(node.kind == "LoopRegion" for node in graph.nodes):
        raise ExportCapabilityError("Normalize bounded LoopRegion nodes before BatchML Master Recipe export")

    root = ET.Element(_q("BatchInformation"), {f"{{{XSI}}}schemaLocation": f"{NS} BatchML-BatchInformation.xsd"})
    header = _child(root, "ListHeader")
    _child(header, "ID", f"{graph.id}_ListHeader")
    master = _child(root, "MasterRecipe")
    _child(master, "ID", f"MasterRecipe_{graph.id}")
    _child(master, "Version", "2.0")
    _child(master, "Description", graph.name or "RTN-selected nonlinear Master Recipe")
    logic = _child(master, "ProcedureLogic")

    node_map = graph.node_map
    transition_conditions: dict[str, dict[str, Any]] = {}
    synthetic_transitions: dict[str, tuple[RecipeEdge, str]] = {}
    links: list[dict[str, Any]] = []
    handled_edges: set[str] = set()

    for gateway in [node for node in graph.nodes if node.kind == "Gateway"]:
        incoming = graph.incoming(gateway.id)
        outgoing = graph.outgoing(gateway.id)
        from_endpoints = [(_endpoint_id(node_map[edge.source]), _endpoint_type(node_map[edge.source])) for edge in incoming]
        to_endpoints: list[tuple[str, str]] = []
        if gateway.gateway_type in {"XOR_SPLIT", "OR_SPLIT"}:
            for index, edge in enumerate(sorted(outgoing, key=lambda item: (item.priority, item.id)), start=1):
                transition_id = f"T_{gateway.id}_{index}"
                synthetic_transitions[transition_id] = (edge, gateway.id)
                to_endpoints.append((transition_id, "Transition"))
                links.append({
                    "id": f"L_{transition_id}_to_{edge.target}",
                    "from_endpoints": [(transition_id, "Transition")],
                    "to_endpoints": [(_endpoint_id(node_map[edge.target]), _endpoint_type(node_map[edge.target]))],
                    "type": "ControlLink",
                    "description": f"MODPLANT_BRANCH={edge.branch_id or edge.id}",
                })
        else:
            to_endpoints = [(_endpoint_id(node_map[edge.target]), _endpoint_type(node_map[edge.target])) for edge in outgoing]
        group_id = str(gateway.metadata.get("gatewayGroupId", gateway.id))
        gateway_description = f"MODPLANT_GATEWAY={gateway.gateway_type};GROUP={group_id}"
        if gateway.gateway_type in {"OR_SPLIT", "OR_JOIN"}:
            gateway_description += (
                f";MIN={int(gateway.metadata.get('minBranches', 1))}"
                f";MAX={int(gateway.metadata.get('maxBranches', max(1, len(outgoing))))}"
            )
        branch_map = {
            edge.source: edge.branch_id
            for edge in incoming
            if edge.branch_id
        }
        if branch_map:
            gateway_description += ";BRANCH_MAP=" + ",".join(
                f"{source}:{branch}" for source, branch in sorted(branch_map.items())
            )
        links.append({
            "id": f"L_{gateway.id}",
            "from_endpoints": from_endpoints,
            "to_endpoints": to_endpoints,
            "type": _GATEWAY_TO_LINK[gateway.gateway_type],
            "description": gateway_description,
        })
        handled_edges.update(edge.id for edge in incoming + outgoing)

    for edge in graph.edges:
        if edge.id in handled_edges:
            continue
        link_type = {"Transfer": "TransferLink", "Synchronization": "SynchronizationLink"}.get(edge.flow_type, "ControlLink")
        description = f"RecipeGraph edge {edge.id}"
        if edge.flow_type == "Jump":
            description = f"MODPLANT_FLOW=Jump;EDGE={edge.id}"
            if edge.condition is not None:
                encoded_condition = base64.urlsafe_b64encode(
                    json.dumps(edge.condition, separators=(",", ":")).encode("utf-8")
                ).decode("ascii")
                description += f";CONDITION_AST_B64={encoded_condition}"
        links.append({
            "id": f"L_{edge.id}",
            "from_endpoints": [(_endpoint_id(node_map[edge.source]), _endpoint_type(node_map[edge.source]))],
            "to_endpoints": [(_endpoint_id(node_map[edge.target]), _endpoint_type(node_map[edge.target]))],
            "type": link_type,
            "description": description,
        })

    for link_data in links:
        _append_master_link(logic, **link_data)
    for node in graph.nodes:
        if node.kind in {"Gateway"}:
            continue
        if node.kind == "Transition":
            condition = node.metadata.get("condition", {"op": "literal", "value": True})
            transition_conditions[node.id] = condition
            _append_transition(logic, node.id, condition, node.name or node.id)
        else:
            step = _child(logic, "Step")
            _child(step, "ID", node.id)
            _child(step, "RecipeElementID", node.metadata.get("recipeElementId", node.id))
            _child(step, "RecipeElementVersion", "2.0")
            _child(step, "Description", node.name or node.id)
    # Choice groups whose branches carry real (non-literal) guards are decided at
    # runtime from live process state (ISA-88 selection divergence). The planner's
    # nominal branch pick must NOT overwrite such a guard with a literal, or the
    # runtime engine loses the ability to select the branch from sensor values.
    runtime_guarded_groups: set[str] = set()
    for _transition_id, (edge, gateway_id) in synthetic_transitions.items():
        group = str(node_map[gateway_id].metadata.get("gatewayGroupId", gateway_id))
        condition = edge.condition
        if condition is not None and condition.get("op") != "literal":
            runtime_guarded_groups.add(group)

    for transition_id, (edge, gateway_id) in synthetic_transitions.items():
        group = str(node_map[gateway_id].metadata.get("gatewayGroupId", gateway_id))
        selected_branch = (selected_branches or {}).get(group)
        condition = edge.condition
        if selected_branch is not None and group not in runtime_guarded_groups:
            branch_id = edge.branch_id or edge.id
            is_selected = (
                branch_id in selected_branch
                if isinstance(selected_branch, (list, tuple, set))
                else selected_branch == branch_id
            )
            condition = {"op": "literal", "value": is_selected}
        if condition is None:
            condition = {"op": "literal", "value": bool(edge.is_default)}
        annotation = f"branch={edge.branch_id or edge.id};priority={edge.priority};default={str(edge.is_default).lower()}"
        _append_transition(logic, transition_id, condition, annotation)

    for node in graph.nodes:
        if node.kind in {"Gateway", "Transition"}:
            continue
        recipe_element = _child(master, "RecipeElement")
        _child(recipe_element, "ID", node.metadata.get("recipeElementId", node.id))
        _child(recipe_element, "Version", "2.0")
        _child(recipe_element, "Description", node.name or node.id)
        element_type = "Begin" if node.kind == "Start" else "End" if node.kind == "End" else "Operation"
        _child(recipe_element, "RecipeElementType", element_type)
        if node.metadata.get("actualEquipmentId"):
            _child(recipe_element, "ActualEquipmentID", node.metadata["actualEquipmentId"])
        for key, value in sorted(node.parameters.items()):
            _append_batch_parameter(recipe_element, key, value)
    runtime_metadata = dict(graph.metadata.get("runtimePlanning", {}))
    if selected_branches:
        runtime_metadata["selectedBranches"] = selected_branches
    if runtime_metadata:
        info = _child(master, "OtherInformation")
        _child(info, "ID", "ModuleRuntimePlanning")
        value = _child(info, "Value")
        _child(value, "ValueString", json.dumps(runtime_metadata, separators=(",", ":")))
        _child(value, "DataInterpretation", "Constant")
        _child(value, "DataType", "string")
        _child(value, "UnitOfMeasure", "1")
        _child(info, "Description", "Finite RTN scenario assumptions and selected branches")
    return _serialize(root, destination)


def _append_master_link(parent: ET.Element, id: str, from_endpoints: list[tuple[str, str]], to_endpoints: list[tuple[str, str]], type: str, description: str) -> None:
    link = _child(parent, "Link")
    _child(link, "ID", id)
    for endpoint_id, endpoint_type in from_endpoints:
        endpoint = _child(link, "FromID")
        _child(endpoint, "FromIDValue", endpoint_id)
        _child(endpoint, "FromType", endpoint_type)
        _child(endpoint, "IDScope", "External")
    for endpoint_id, endpoint_type in to_endpoints:
        endpoint = _child(link, "ToID")
        _child(endpoint, "ToIDValue", endpoint_id)
        _child(endpoint, "ToType", endpoint_type)
        _child(endpoint, "IDScope", "External")
    _child(link, "LinkType", type)
    _child(link, "Depiction", "LineAndArrow")
    _child(link, "EvaluationOrder", "1")
    _child(link, "Description", description)


def _append_transition(parent: ET.Element, id: str, condition: dict[str, Any], description: str) -> None:
    transition = _child(parent, "Transition")
    _child(transition, "ID", id)
    rendered = render_condition(condition)
    _child(transition, "Condition", rendered)
    _child(transition, "ConditionAnnotation", f"urn:modplant:condition:ast:v1 {json.dumps(condition, separators=(',', ':'))}")
    _child(transition, "Description", description)


def _append_batch_parameter(parent: ET.Element, key: str, raw_value: Any) -> None:
    parameter = _child(parent, "Parameter")
    _child(parameter, "ID", key)
    _child(parameter, "Description", key)
    _child(parameter, "ParameterType", "ProcessParameter")
    value = _child(parameter, "Value")
    _child(value, "ValueString", raw_value)
    _child(value, "DataInterpretation", "Constant")
    _child(value, "DataType", _data_type(raw_value))
    _child(value, "UnitOfMeasure", "1")


def import_master_recipe(source: str | Path | ET.Element) -> RecipeGraph:
    root = _load_xml(source)
    master = root if root.tag == _q("MasterRecipe") else root.find(f".//{_q('MasterRecipe')}")
    if master is None:
        raise ValueError("No BatchML MasterRecipe element found")
    graph = RecipeGraph(_text(master, "ID", "ImportedMasterRecipe").removeprefix("MasterRecipe_"), "Master", _text(master, "Description"), conformance=["BatchML-Core", "Module-RoundTrip", "Module-Execution"])
    for info in master.findall(_q("OtherInformation")):
        if _text(info, "ID") != "ModuleRuntimePlanning":
            continue
        raw = _text(info.find(_q("Value")), "ValueString")
        parsed = _maybe_json(raw)
        if isinstance(parsed, dict):
            graph.metadata["runtimePlanning"] = parsed
    recipe_elements: dict[str, ET.Element] = {
        _text(element, "ID"): element for element in master.findall(_q("RecipeElement"))
    }
    logic = master.find(_q("ProcedureLogic"))
    if logic is None:
        return graph

    transition_conditions: dict[str, dict[str, Any]] = {}
    transition_branches: dict[str, str] = {}
    for step in logic.findall(_q("Step")):
        node_id = _text(step, "ID")
        recipe_element_id = _text(step, "RecipeElementID", node_id)
        element = recipe_elements.get(recipe_element_id)
        element_type = _text(element, "RecipeElementType")
        kind = "Start" if element_type == "Begin" or recipe_element_id == "Init" else "End" if element_type == "End" or recipe_element_id == "End" else "Activity"
        parameters: dict[str, Any] = {}
        if element is not None:
            for parameter in element.findall(_q("Parameter")):
                parameters[_text(parameter, "ID")] = _coerce(_text(parameter.find(_q("Value")), "ValueString"))
        graph.nodes.append(RecipeNode(node_id, kind, _text(step, "Description", recipe_element_id), element_type.lower() or "operation", parameters=parameters, metadata={"recipeElementId": recipe_element_id, "recipeElementType": element_type}))
    for transition in logic.findall(_q("Transition")):
        node_id = _text(transition, "ID")
        condition = _condition_from_transition(transition)
        transition_conditions[node_id] = condition
        branch_match = re.search(r"(?:^|;)branch=([^;]+)", _text(transition, "Description"))
        if branch_match:
            transition_branches[node_id] = branch_match.group(1)
        graph.nodes.append(RecipeNode(node_id, "Transition", _text(transition, "Description", node_id), metadata={"condition": condition, "conditionText": _text(transition, "Condition")}))

    links = logic.findall(_q("Link"))
    gateway_by_link: dict[str, str] = {}
    gateway_specs: dict[str, dict[str, Any]] = {}
    for link in links:
        link_id = _text(link, "ID", f"L{len(gateway_by_link)+1}")
        link_type = _text(link, "LinkType", "ControlLink")
        if link_type not in _LINK_TO_GATEWAY:
            continue
        description = " ".join(link.findalltext(_q("Description"))) if hasattr(link, "findalltext") else _text(link, "Description")
        gateway_type = _LINK_TO_GATEWAY[link_type]
        match = re.search(r"MODPLANT_GATEWAY=(AND|XOR|OR)_(SPLIT|JOIN);GROUP=([^;\s]+)", description)
        group = link_id
        if match:
            gateway_type = f"{match.group(1)}_{match.group(2)}"
            group = match.group(3)
        cardinality_match = re.search(r";MIN=(\d+);MAX=(\d+)", description)
        gateway_metadata: dict[str, Any] = {"gatewayGroupId": group, "batchmlLinkId": link_id}
        if cardinality_match:
            gateway_metadata.update({
                "minBranches": int(cardinality_match.group(1)),
                "maxBranches": int(cardinality_match.group(2)),
            })
        branch_map: dict[str, str] = {}
        branch_map_match = re.search(r";BRANCH_MAP=([^;]+)", description)
        if branch_map_match:
            for entry in branch_map_match.group(1).split(","):
                endpoint, separator, branch = entry.partition(":")
                if separator and endpoint and branch:
                    branch_map[endpoint] = branch
        gateway_id = f"gateway_{link_id}"
        gateway_by_link[link_id] = gateway_id
        gateway_specs[link_id] = {
            "gatewayType": gateway_type,
            "group": group,
            "metadata": gateway_metadata,
            "branchMap": branch_map,
        }
        graph.nodes.append(
            RecipeNode(gateway_id, "Gateway", gateway_id, gateway_type=gateway_type, metadata=gateway_metadata)
        )

    node_ids = {node.id for node in graph.nodes}
    for link in links:
        link_id = _text(link, "ID", f"L{len(graph.edges)+1}")
        link_type = _text(link, "LinkType", "ControlLink")
        raw_from_ids = [_text(endpoint, "FromIDValue") for endpoint in link.findall(_q("FromID"))]
        raw_to_ids = [_text(endpoint, "ToIDValue") for endpoint in link.findall(_q("ToID"))]
        from_ids = [gateway_by_link.get(endpoint_id, endpoint_id) for endpoint_id in raw_from_ids]
        to_ids = [gateway_by_link.get(endpoint_id, endpoint_id) for endpoint_id in raw_to_ids]
        description = " ".join(link.findalltext(_q("Description"))) if hasattr(link, "findalltext") else _text(link, "Description")
        if link_type in _LINK_TO_GATEWAY:
            spec = gateway_specs[link_id]
            gateway_type = str(spec["gatewayType"])
            group = str(spec["group"])
            branch_map = dict(spec["branchMap"])
            gateway_id = gateway_by_link[link_id]
            for index, (raw_source_id, source_id) in enumerate(zip(raw_from_ids, from_ids), start=1):
                if source_id in node_ids:
                    graph.edges.append(RecipeEdge(f"{link_id}_in_{index}", source_id, gateway_id, branch_id=branch_map.get(raw_source_id, branch_map.get(source_id, "")), gateway_group_id=group))
            for index, target_id in enumerate(to_ids, start=1):
                if target_id in node_ids:
                    condition = transition_conditions.get(target_id) if gateway_type in {"XOR_SPLIT", "OR_SPLIT"} else None
                    graph.edges.append(RecipeEdge(f"{link_id}_out_{index}", gateway_id, target_id, branch_id=transition_branches.get(target_id, target_id), condition=condition, priority=index, gateway_group_id=group))
            continue
        flow_type = {"TransferLink": "Transfer", "SynchronizationLink": "Synchronization"}.get(link_type, "Control")
        condition = None
        if "MODPLANT_FLOW=Jump" in description:
            flow_type = "Jump"
            condition_match = re.search(r"(?:^|;)CONDITION_AST_B64=([A-Za-z0-9_=-]+)", description)
            if condition_match:
                try:
                    condition = json.loads(
                        base64.urlsafe_b64decode(condition_match.group(1)).decode("utf-8")
                    )
                except (ValueError, json.JSONDecodeError, UnicodeDecodeError):
                    condition = None
        for source_index, source_id in enumerate(from_ids, start=1):
            for target_index, target_id in enumerate(to_ids, start=1):
                if source_id in node_ids and target_id in node_ids:
                    graph.edges.append(
                        RecipeEdge(
                            f"{link_id}_{source_index}_{target_index}",
                            source_id,
                            target_id,
                            flow_type,
                            condition=condition,
                        )
                    )
    return graph


def _condition_from_transition(transition: ET.Element) -> dict[str, Any]:
    annotation = _text(transition, "ConditionAnnotation")
    prefix = "urn:modplant:condition:ast:v1 "
    if annotation.startswith(prefix):
        try:
            return json.loads(annotation[len(prefix):])
        except json.JSONDecodeError:
            pass
    return parse_condition(_text(transition, "Condition", "False"))


def _endpoint_type(node: RecipeNode) -> str:
    if node.kind == "Transition":
        return "Transition"
    if node.kind == "Gateway":
        return "Link"
    return "Step"


def _endpoint_id(node: RecipeNode) -> str:
    return f"L_{node.id}" if node.kind == "Gateway" else node.id


def _data_type(value: Any) -> str:
    if isinstance(value, bool): return "boolean"
    if isinstance(value, int): return "int"
    if isinstance(value, float): return "double"
    return "string"


def _coerce(value: str) -> Any:
    if value.lower() in {"true", "false"}: return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def _maybe_json(value: str) -> Any:
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value


def _infer_activity_type(process: ET.Element, metadata: dict[str, Any] | None = None) -> str:
    description = _text(process, "Description").lower()
    aliases = {
        "dosing": "dose",
        "dose": "dose",
        "mixing": "mix",
        "mix": "mix",
        "usage": "usage",
        "settling": "settling",
        "heating": "heating",
        "separation": "separation",
    }
    for marker, activity_type in aliases.items():
        if marker in description:
            return activity_type
    semantic_uri = str((metadata or {}).get("semanticUri", "")).lower()
    for marker, activity_type in aliases.items():
        if marker in semantic_uri:
            return activity_type
    return _text(process, "ProcessElementType", "process-operation").lower().replace(" ", "-")
