from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from recipe_ir import RecipeIR, RecipeNode, recipe_ir_from_dict, recipe_ir_to_dict
    from recipe_graph_adapter import recipe_graph_to_ir, recipe_ir_to_graph
except ImportError:  # pragma: no cover - notebook direct import compatibility
    from recipe_ir import RecipeIR, RecipeNode, recipe_ir_from_dict, recipe_ir_to_dict
    from recipe_graph_adapter import recipe_graph_to_ir, recipe_ir_to_graph

from modplant_recipe import export_general_recipe, export_master_recipe, import_general_recipe


NS_B2MML = "http://www.mesa.org/xml/B2MML"
NS_XSI = "http://www.w3.org/2001/XMLSchema-instance"
ET.register_namespace("b2mml", NS_B2MML)
ET.register_namespace("xsi", NS_XSI)


def _b(tag: str) -> str:
    return f"{{{NS_B2MML}}}{tag}"


def _ts_suffix() -> str:
    return "/" + datetime.now().strftime("%m.%d/%H:%M:%S.%f")[:-5] + "/"


def _slug(text: str, maxlen: int = 80) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")[:maxlen] or "Recipe"


def _amount_block(parent: ET.Element, qty: float) -> None:
    amt = ET.SubElement(parent, _b("Amount"))
    ET.SubElement(amt, _b("QuantityString")).text = f"{qty:.6g}"
    ET.SubElement(amt, _b("DataType")).text = "double"
    ET.SubElement(amt, _b("UnitOfMeasure")).text = "http://si-digital-framework.org/SI/units/litre"
    ET.SubElement(amt, _b("Key")).text = "http://qudt.org/vocab/quantitykind/LiquidVolume"


def _materials_block(parent: ET.Element, block_id: str, desc: str, mtype: str, material_ids: list[str]) -> None:
    mats = ET.SubElement(parent, _b("Materials"))
    ET.SubElement(mats, _b("ID")).text = block_id
    ET.SubElement(mats, _b("Description")).text = desc
    ET.SubElement(mats, _b("MaterialsType")).text = mtype
    for mid in material_ids:
        mat = ET.SubElement(mats, _b("Material"))
        ET.SubElement(mat, _b("ID")).text = mid


def _value_param(
    pe: ET.Element,
    pid: str,
    desc: str,
    value: str,
    data_type: str,
    unit: str = "",
    key: str = "",
) -> None:
    prm = ET.SubElement(pe, _b("ProcessElementParameter"))
    ET.SubElement(prm, _b("ID")).text = pid
    ET.SubElement(prm, _b("Description")).text = desc
    val = ET.SubElement(prm, _b("Value"))
    ET.SubElement(val, _b("ValueString")).text = value
    ET.SubElement(val, _b("DataType")).text = data_type
    if unit:
        ET.SubElement(val, _b("UnitOfMeasure")).text = unit
    if key:
        ET.SubElement(val, _b("Key")).text = key


def _other_info(pe: ET.Element, info_id: str, desc: str, value: str, data_type: str = "string", key: str = "") -> None:
    oi = ET.SubElement(pe, _b("OtherInformation"))
    ET.SubElement(oi, _b("OtherInfoID")).text = info_id
    ET.SubElement(oi, _b("Description")).text = desc
    ov = ET.SubElement(oi, _b("OtherValue"))
    ET.SubElement(ov, _b("ValueString")).text = value
    ET.SubElement(ov, _b("DataType")).text = data_type
    if key:
        ET.SubElement(ov, _b("Key")).text = key


def _node_xml_id(node: RecipeNode, suffix: str) -> str:
    return f"{_slug(node.id)}{suffix}"


def build_general_recipe_tree_from_ir(ir: RecipeIR) -> ET.ElementTree:
    """Build a schema-valid BatchML General Recipe via RecipeGraph v2.

    XOR metadata uses the documented annotated projection because BatchML core
    optional-parallel indicators cannot distinguish XOR from inclusive OR.
    """
    xml = export_general_recipe(recipe_ir_to_graph(ir), mode="annotated")
    return ET.ElementTree(ET.fromstring(xml))

    # Legacy builder retained below temporarily as a readable migration record.
    suffix = _ts_suffix()
    link_suffix = _ts_suffix()
    node_ids = {nid: _node_xml_id(node, suffix) for nid, node in ir.nodes.items()}
    input_ids = {ingr: f"Educt_{_slug(ingr)}{idx:03d}{suffix}" for idx, ingr in enumerate(ir.inputs, start=1)}
    product_id = f"Product001{suffix}"

    root = ET.Element(_b("GRecipe"), {f"{{{NS_XSI}}}schemaLocation": "http://www.mesa.org/xml/B2MML "})
    ET.SubElement(root, _b("ID")).text = f"GeneralRecipe_{_slug(ir.id)}"
    ET.SubElement(root, _b("Description")).text = "General Recipe auto-generated from RecipeIR"
    ET.SubElement(root, _b("GRecipeType")).text = "General"
    ET.SubElement(root, _b("LifeCycleState")).text = "Draft"

    formula = ET.SubElement(root, _b("Formula"))
    ET.SubElement(formula, _b("Description")).text = "The formula defines Inputs, Intermediates and Outputs of the Procedure"

    pin = ET.SubElement(formula, _b("ProcessInputs"))
    ET.SubElement(pin, _b("ID")).text = f"InputListID001{suffix}"
    ET.SubElement(pin, _b("Description")).text = "List of Process Inputs"
    ET.SubElement(pin, _b("MaterialsType")).text = "Input"
    for order, (ingr, qty) in enumerate(ir.inputs.items(), start=1):
        mat = ET.SubElement(pin, _b("Material"))
        ET.SubElement(mat, _b("ID")).text = input_ids[ingr]
        ET.SubElement(mat, _b("Description")).text = f"Ingredient {ingr}"
        ET.SubElement(mat, _b("Order")).text = str(order)
        _amount_block(mat, qty)

    pout = ET.SubElement(formula, _b("ProcessOutputs"))
    ET.SubElement(pout, _b("ID")).text = f"OutputListID001{suffix}"
    ET.SubElement(pout, _b("Description")).text = "List of Process Outputs"
    ET.SubElement(pout, _b("MaterialsType")).text = "Output"
    mat = ET.SubElement(pout, _b("Material"))
    ET.SubElement(mat, _b("ID")).text = product_id
    ET.SubElement(mat, _b("Description")).text = "Mixed product"
    ET.SubElement(mat, _b("Order")).text = "1"
    _amount_block(mat, sum(ir.inputs.values()))

    pinter = ET.SubElement(formula, _b("ProcessIntermediates"))
    ET.SubElement(pinter, _b("ID")).text = f"IntermediateListID001{suffix}"
    ET.SubElement(pinter, _b("Description")).text = "List of Process Intermediates"
    ET.SubElement(pinter, _b("MaterialsType")).text = "Intermediate"
    for order, node in enumerate(ir.task_nodes(), start=1):
        imat = ET.SubElement(pinter, _b("Material"))
        ET.SubElement(imat, _b("ID")).text = f"IntermediateAfter_{_slug(node.id)}{suffix}"
        ET.SubElement(imat, _b("Description")).text = f"Intermediate after {node.name}"
        ET.SubElement(imat, _b("Order")).text = str(order)
        _amount_block(imat, sum(ir.inputs.values()))

    pproc = ET.SubElement(root, _b("ProcessProcedure"))
    ET.SubElement(pproc, _b("ID")).text = f"ProcessProcedureID001{suffix}"
    ET.SubElement(pproc, _b("Description")).text = "Top level ProcessElement"
    ET.SubElement(pproc, _b("ProcessElementType")).text = "Process"
    ET.SubElement(pproc, _b("LifeCycleState")).text = "Draft"
    _materials_block(pproc, f"ProcedureInputMaterials{suffix}", "Input Materials of Procedure", "Input", list(input_ids.values()))
    _materials_block(pproc, f"ProcedureIntermediateMaterials{suffix}", "Intermediate Materials of Procedure", "Intermediate", [])
    _materials_block(pproc, f"ProcedureOutputMaterials{suffix}", "Output Materials of Procedure", "Output", [product_id])

    link_idx = 0
    ingredient_to_first_dose: Dict[str, str] = {}
    for node in ir.topological_nodes():
        if node.node_type == "dose":
            ingredient_to_first_dose.setdefault(str(node.params["ingredient"]), node.id)
    for ingr, nid in ingredient_to_first_dose.items():
        dl = ET.SubElement(pproc, _b("DirectedLink"))
        ET.SubElement(dl, _b("ID")).text = f"{link_idx}{link_suffix}"
        ET.SubElement(dl, _b("FromID")).text = input_ids[ingr]
        ET.SubElement(dl, _b("ToID")).text = node_ids[nid]
        link_idx += 1
    for src, dst in ir.edges:
        dl = ET.SubElement(pproc, _b("DirectedLink"))
        ET.SubElement(dl, _b("ID")).text = f"{link_idx}{link_suffix}"
        ET.SubElement(dl, _b("FromID")).text = node_ids[src]
        ET.SubElement(dl, _b("ToID")).text = node_ids[dst]
        link_idx += 1

    for node in ir.topological_nodes():
        pe = ET.SubElement(pproc, _b("ProcessElement"))
        ET.SubElement(pe, _b("ID")).text = node_ids[node.id]
        ET.SubElement(pe, _b("Description")).text = node.name
        ET.SubElement(pe, _b("ProcessElementType")).text = "Process" if not node.is_control else "Control"
        _materials_block(pe, f"{node_ids[node.id]}InputMaterials", f"Input Materials of {node.id}", "Input", [])
        _materials_block(pe, f"{node_ids[node.id]}IntermediateMaterials", f"Intermediate Materials of {node.id}", "Intermediate", [])
        _materials_block(pe, f"{node_ids[node.id]}OutputMaterials", f"Output Materials of {node.id}", "Output", [])
        _node_params_to_xml(pe, node)
        if node.semantic_uri:
            _other_info(
                pe,
                "SemanticDescription",
                "URI referencing the Ontology Class definition",
                node.semantic_uri,
                data_type="uriReference",
                key=f"Capability_with_Query.{node.node_type}",
            )
        for key, value in _control_metadata(node).items():
            _other_info(pe, key, f"Module control metadata: {key}", value)
        _other_info(pe, "RecipeIRNodeID", "Stable RecipeIR node id", node.id)

    return ET.ElementTree(root)


def _node_params_to_xml(pe: ET.Element, node: RecipeNode) -> None:
    if node.node_type == "dose":
        _value_param(
            pe,
            f"Dosing_Amount_{node.id}",
            "Amount of Dosing",
            f"{float(node.params.get('amount_L', 0.0)):.6g}",
            "double",
            "http://si-digital-framework.org/SI/units/litre",
            "http://qudt.org/vocab/quantitykind/LiquidVolume",
        )
        _value_param(pe, f"Dosing_Ingredient_{node.id}", "Ingredient of Dosing", str(node.params.get("ingredient", "")), "string")
    elif node.node_type == "mix":
        _value_param(
            pe,
            f"Revolutions_per_minute_{node.id}",
            "Revolutions per minute",
            str(int(node.params.get("rpm", 0))),
            "int",
            "http://qudt.org/vocab/unit/REV-PER-MIN",
            "http://qudt.org/vocab/quantitykind/RotationalVelocity",
        )
        _duration_param(pe, f"Mixing_Duration_{node.id}", "Duration of the process step mixing", int(node.params.get("duration_s", 0)))
    elif node.node_type == "usage":
        _duration_param(pe, f"Usage_Duration_{node.id}", "Duration of the process step usage", int(node.params.get("duration_s", 0)))
    elif node.node_type == "settling":
        _duration_param(pe, f"Settling_Duration_{node.id}", "Duration of the process step settling", int(node.params.get("duration_s", 0)))
    elif node.node_type == "separation":
        _value_param(
            pe,
            f"Separation_Volume_{node.id}",
            "Volume to separate",
            f"{float(node.params.get('amount_L', 0.0)):.6g}",
            "double",
            "http://si-digital-framework.org/SI/units/litre",
            "http://qudt.org/vocab/quantitykind/LiquidVolume",
        )
        _value_param(pe, f"Separation_Ingredient_{node.id}", "Ingredient to separate", str(node.params.get("ingredient", "")), "string")


def _duration_param(pe: ET.Element, pid: str, desc: str, seconds: int) -> None:
    _value_param(
        pe,
        pid,
        desc,
        str(int(seconds)),
        "int",
        "http://si-digital-framework.org/SI/units/second",
        "http://www.w3.org/2006/time#Duration",
    )


def _control_metadata(node: RecipeNode) -> Dict[str, str]:
    out = {}
    if node.control_node_type:
        out["ControlNodeType"] = node.control_node_type
    if node.branch_group_id:
        out["BranchGroupID"] = node.branch_group_id
    if node.branch_id:
        out["BranchID"] = node.branch_id
    if node.join_policy:
        out["JoinPolicy"] = node.join_policy
    return out


def save_general_recipe_xml_from_ir(ir: RecipeIR, out_path: Optional[str | Path] = None) -> str:
    tree = build_general_recipe_tree_from_ir(ir)
    if out_path is None:
        out_path = Path(__file__).resolve().parent / f"GeneralRecipe_{_slug(ir.id)}.xml"
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ", level=0)
    tree.write(out, encoding="utf-8", xml_declaration=True)
    return str(out)


def save_master_recipe_xml_from_ir(
    ir: RecipeIR,
    out_path: Optional[str | Path] = None,
    *,
    selected_branches: Optional[Dict[str, Any]] = None,
    planner_result=None,
    equipment_bindings: Optional[Dict[str, Dict[str, Any]]] = None,
) -> str:
    """Export a BatchML Master Recipe executable by the Recipol token engine."""
    if out_path is None:
        out_path = Path(__file__).resolve().parent / f"MasterRecipe_{_slug(ir.id)}.xml"
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    graph = recipe_ir_to_graph(ir, recipe_level="Master")
    if planner_result is not None:
        # The Master Recipe is the plan, so its control flow is rebuilt from the
        # order the solver actually found: the connect/disconnect actions a
        # person performs and the auxiliary transfers the planner inserts become
        # steps, and an element gets a gateway only where the flow really does
        # branch or converge. See connection_steps.
        from connection_steps import inject_connection_steps

        inject_connection_steps(graph, planner_result, equipment_bindings)
    if planner_result is not None:
        operations_by_node: Dict[str, list] = {}
        for operation in planner_result.operations:
            operations_by_node.setdefault(operation.recipe_node_id, []).append(operation)
        activity_index = 0
        for node in graph.nodes:
            candidates = sorted(
                operations_by_node.get(node.id, []),
                key=lambda operation: (operation.start_s, operation.step_id),
            )
            if not candidates or node.kind != "Activity":
                continue
            activity_index += 1
            primary = next(
                (operation for operation in candidates if operation.operation_type not in {"connect", "transfer"}),
                candidates[0],
            )
            procedure_id = _slug(primary.operation or primary.operation_type)
            node.metadata.update({
                "actualEquipmentId": primary.module or primary.target_module,
                "procedureId": procedure_id,
                "recipeElementId": f"{activity_index:03d}:{procedure_id}",
                "plannedRecipeNodeId": node.id,
            })
            node.parameters.update({
                "planned_start_s": primary.start_s,
                "planned_end_s": primary.end_s,
                "planned_duration_s": primary.duration_s,
                "planned_recipe_node_id": node.id,
            })
            binding = (equipment_bindings or {}).get(primary.module or primary.target_module, {})
            operation_methods = binding.get("operation_methods", {})
            method = operation_methods.get(str(primary.operation_type).casefold(), "")
            if not method:
                method = operation_methods.get(str(primary.operation).casefold(), "")
            if not method:
                capability_name = {
                    "dose": "filling",
                    "mix": "stirring",
                    "usage": "none",
                    "settling": "settling",
                    "separation": "draining",
                    "connect": "connect",
                    "transfer": "draining",
                }.get(str(primary.operation_type).casefold(), "")
                method = operation_methods.get(capability_name, "")
            node.parameters.update({
                "opcua_endpoint": binding.get("opcua_endpoint", ""),
                "opcua_namespace_uri": binding.get("opcua_namespace_uri", ""),
                "opcua_method": method,
                "planned_source_module": primary.source_module,
                "planned_target_module": primary.target_module,
                "planned_out_port": primary.out_port,
                "planned_in_port": primary.in_port,
            })
    export_master_recipe(
        graph,
        out,
        selected_branches=selected_branches,
    )
    return str(out)


def save_recipe_ir_json(ir: RecipeIR, out_path: str | Path) -> str:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(recipe_ir_to_dict(ir), indent=2, ensure_ascii=False), encoding="utf-8")
    return str(out)


def load_recipe_ir_json(path: str | Path) -> RecipeIR:
    return recipe_ir_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def parse_general_recipe_xml_to_ir(
    path: str | Path,
    *,
    planning_context: Optional[Dict[str, Any]] = None,
    loop_iterations: Optional[Dict[str, int]] = None,
    jump_decisions: Optional[Dict[str, bool]] = None,
) -> RecipeIR:
    return recipe_graph_to_ir(
        import_general_recipe(path),
        planning_context=planning_context,
        loop_iterations=loop_iterations,
        jump_decisions=jump_decisions,
    )

    # Legacy parser retained below temporarily as a readable migration record.
    tree = ET.parse(path)
    root = tree.getroot()
    ns = {"b2mml": NS_B2MML}

    def txt(parent: ET.Element, xpath: str) -> str:
        node = parent.find(xpath, ns)
        return node.text if node is not None and node.text is not None else ""

    recipe_id = txt(root, "b2mml:ID").replace("GeneralRecipe_", "") or "ParsedRecipe"
    ir = RecipeIR(id=recipe_id, volume=0.0)
    xml_to_node: Dict[str, str] = {}

    for pe in root.findall(".//b2mml:ProcessProcedure/b2mml:ProcessElement", ns):
        xml_id = txt(pe, "b2mml:ID")
        metadata = _read_other_info(pe, ns)
        node_id = metadata.get("RecipeIRNodeID") or _slug(xml_id)
        sem = metadata.get("SemanticDescription", "")
        params = _read_params(pe, ns)
        node_type = _infer_node_type(sem, metadata.get("ControlNodeType", ""), txt(pe, "b2mml:Description"))
        node = RecipeNode(
            id=node_id,
            node_type=node_type,
            name=txt(pe, "b2mml:Description") or node_id,
            params=params,
            semantic_uri=sem,
            control_node_type=metadata.get("ControlNodeType", ""),
            branch_group_id=metadata.get("BranchGroupID", ""),
            branch_id=metadata.get("BranchID", ""),
            join_policy=metadata.get("JoinPolicy", ""),
        )
        ir.nodes[node.id] = node
        xml_to_node[xml_id] = node.id

    for mat in root.findall(".//b2mml:ProcessInputs/b2mml:Material", ns):
        desc = txt(mat, "b2mml:Description")
        qty = txt(mat.find("b2mml:Amount", ns), "b2mml:QuantityString") if mat.find("b2mml:Amount", ns) is not None else "0"
        ingr = desc.replace("Ingredient", "").strip() or txt(mat, "b2mml:ID")
        try:
            ir.inputs[ingr] = float(qty)
        except ValueError:
            ir.inputs[ingr] = 0.0
    ir.volume = sum(ir.inputs.values())
    ir.outputs = {"Product": ir.volume}

    for dl in root.findall(".//b2mml:DirectedLink", ns):
        src = xml_to_node.get(txt(dl, "b2mml:FromID"))
        dst = xml_to_node.get(txt(dl, "b2mml:ToID"))
        if src and dst:
            ir.edges.append((src, dst))
    return ir


def _read_other_info(pe: ET.Element, ns: Dict[str, str]) -> Dict[str, str]:
    out = {}
    for oi in pe.findall("b2mml:OtherInformation", ns):
        key = oi.find("b2mml:OtherInfoID", ns)
        value = oi.find("b2mml:OtherValue/b2mml:ValueString", ns)
        if key is not None and value is not None:
            out[key.text or ""] = value.text or ""
    return out


def _read_params(pe: ET.Element, ns: Dict[str, str]) -> Dict[str, object]:
    params: Dict[str, object] = {}
    for prm in pe.findall("b2mml:ProcessElementParameter", ns):
        desc = prm.findtext("b2mml:Description", default="", namespaces=ns)
        val = prm.findtext("b2mml:Value/b2mml:ValueString", default="", namespaces=ns)
        if "Amount of Dosing" in desc:
            params["amount_L"] = float(val or 0)
        elif "Ingredient of Dosing" in desc:
            params["ingredient"] = val
        elif "Revolutions" in desc:
            params["rpm"] = int(float(val or 0))
        elif "Duration" in desc:
            params["duration_s"] = int(float(val or 0))
        elif "Volume to separate" in desc:
            params["amount_L"] = float(val or 0)
        elif "Ingredient to separate" in desc:
            params["ingredient"] = val
    return params


def _infer_node_type(semantic_uri: str, control_node_type: str, desc: str) -> str:
    if control_node_type:
        return control_node_type.lower()
    sem = semantic_uri.lower()
    if "dosing" in sem:
        return "dose"
    if "mixing" in sem:
        return "mix"
    if "usage" in sem:
        return "usage"
    if "settling" in sem:
        return "settling"
    if "separation" in sem:
        return "separation"
    return _slug(desc).lower()
