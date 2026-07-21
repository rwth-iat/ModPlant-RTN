from __future__ import annotations

import io
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from .models import (
    InterfacePort,
    OperationCapability,
    PlantModel,
    ModuleAsset,
    default_operations,
    default_plant_model,
)

AAS_NS = "https://admin-shell.io/aas/3/0"
MODPLANT_SM = "https://modplant.dev/idta/ModPlantOperationalData/1/0/Submodel"
CAPABILITY_SM = "https://admin-shell.io/idta/CapabilityDescription/1/0/Submodel"
AID_SM = "https://admin-shell.io/idta/AssetInterfacesDescription/1/1/Submodel"
NAMEPLATE_SM = "https://admin-shell.io/idta/Nameplate/3/0/Submodel"
ET.register_namespace("", AAS_NS)


def _q(tag: str) -> str:
    return f"{{{AAS_NS}}}{tag}"


def _text(parent: ET.Element, name: str, default: str = "") -> str:
    node = parent.find(_q(name))
    return (node.text or "").strip() if node is not None else default


def _semantic_value(element: ET.Element) -> str:
    node = element.find(f"{_q('semanticId')}/{_q('keys')}/{_q('key')}/{_q('value')}")
    return (node.text or "").strip() if node is not None else ""


def _property(parent: ET.Element, id_short: str, value, value_type: str = "xs:string", semantic: str = "") -> ET.Element:
    prop = ET.SubElement(parent, _q("property"))
    ET.SubElement(prop, _q("idShort")).text = id_short
    if semantic:
        _semantic_id(prop, semantic)
    ET.SubElement(prop, _q("valueType")).text = value_type
    ET.SubElement(prop, _q("value")).text = str(value)
    return prop


def _collection(parent: ET.Element, id_short: str, semantic: str = "") -> ET.Element:
    collection = ET.SubElement(parent, _q("submodelElementCollection"))
    ET.SubElement(collection, _q("idShort")).text = id_short
    if semantic:
        _semantic_id(collection, semantic)
    return ET.SubElement(collection, _q("value"))


def _semantic_id(parent: ET.Element, value: str) -> None:
    sid = ET.SubElement(parent, _q("semanticId"))
    ET.SubElement(sid, _q("type")).text = "ExternalReference"
    keys = ET.SubElement(sid, _q("keys"))
    key = ET.SubElement(keys, _q("key"))
    ET.SubElement(key, _q("type")).text = "GlobalReference"
    ET.SubElement(key, _q("value")).text = value


def _submodel(root: ET.Element, asset: ModuleAsset, id_short: str, semantic: str) -> ET.Element:
    submodel = ET.SubElement(root, _q("submodel"))
    ET.SubElement(submodel, _q("idShort")).text = id_short
    ET.SubElement(submodel, _q("id")).text = f"https://modplant.dev/assets/{asset.asset_id}/{id_short}"
    ET.SubElement(submodel, _q("kind")).text = "Instance"
    _semantic_id(submodel, semantic)
    return ET.SubElement(submodel, _q("submodelElements"))


def build_module_aas_environment(asset: ModuleAsset) -> ET.ElementTree:
    """Create an AAS 3.0 environment aligned with IDTA templates plus one extension.

    CapabilityDescription and AssetInterfacesDescription carry their published
    IDTA semantic IDs. Scheduling costs, physical fluid ports and initial
    inventory live in the versioned ModPlantOperationalData extension because
    no published IDTA template covers that complete planning state.
    """
    asset.validate()
    env = ET.Element(_q("environment"))
    shells = ET.SubElement(env, _q("assetAdministrationShells"))
    shell = ET.SubElement(shells, _q("assetAdministrationShell"))
    ET.SubElement(shell, _q("idShort")).text = f"{asset.asset_id}_AAS"
    ET.SubElement(shell, _q("id")).text = f"https://modplant.dev/assets/{asset.asset_id}/aas"
    info = ET.SubElement(shell, _q("assetInformation"))
    ET.SubElement(info, _q("assetKind")).text = "Instance"
    ET.SubElement(info, _q("globalAssetId")).text = f"https://modplant.dev/assets/{asset.asset_id}"
    refs = ET.SubElement(shell, _q("submodels"))
    submodels_root = ET.SubElement(env, _q("submodels"))

    submodel_specs = [
        ("Nameplate", NAMEPLATE_SM),
        ("CapabilityDescription", CAPABILITY_SM),
        ("AssetInterfacesDescription", AID_SM),
        ("ModPlantOperationalData", MODPLANT_SM),
    ]
    for id_short, _ in submodel_specs:
        ref = ET.SubElement(refs, _q("reference"))
        ET.SubElement(ref, _q("type")).text = "ModelReference"
        keys = ET.SubElement(ref, _q("keys"))
        key = ET.SubElement(keys, _q("key"))
        ET.SubElement(key, _q("type")).text = "Submodel"
        ET.SubElement(key, _q("value")).text = f"https://modplant.dev/assets/{asset.asset_id}/{id_short}"

    nameplate = _submodel(submodels_root, asset, "Nameplate", NAMEPLATE_SM)
    _property(nameplate, "ManufacturerName", asset.manufacturer)
    _property(nameplate, "ManufacturerProductDesignation", asset.asset_id)
    _property(nameplate, "ManufacturerProductDescription", asset.description)

    capabilities = _submodel(submodels_root, asset, "CapabilityDescription", CAPABILITY_SM)
    capability_set = _collection(
        capabilities,
        f"{asset.asset_id}CapabilitySet",
        "https://admin-shell.io/idta/CapabilityDescription/CapabilitySet/1/0",
    )
    for index, operation in enumerate(asset.operations, 1):
        container = _collection(
            capability_set,
            f"Capability{index:02d}_{_safe_id(operation.operation)}",
            "https://admin-shell.io/idta/CapabilityDescription/CapabilityContainer/1/0",
        )
        capability = ET.SubElement(container, _q("capability"))
        ET.SubElement(capability, _q("idShort")).text = operation.operation
        _semantic_id(capability, f"https://modplant.dev/capabilities/{_safe_id(operation.operation)}")
        props = _collection(
            container,
            "PropertySet",
            "https://admin-shell.io/idta/CapabilityDescription/PropertySet/1/0",
        )
        _property(props, "Parameter", operation.parameter)

    aid = _submodel(submodels_root, asset, "AssetInterfacesDescription", AID_SM)
    interface = _collection(aid, "OPCUAInterface", f"{AID_SM}/Interface")
    _property(interface, "Endpoint", asset.opcua_endpoint, semantic=f"{AID_SM}/Endpoint")
    _property(interface, "NamespaceUri", asset.opcua_namespace_uri, semantic=f"{AID_SM}/NamespaceUri")
    methods = _collection(interface, "InteractionMetadata", f"{AID_SM}/InteractionMetadata")
    for index, operation in enumerate(asset.operations, 1):
        if operation.opcua_method:
            method = _collection(methods, f"Method{index:02d}", f"{AID_SM}/Action")
            _property(method, "Operation", operation.operation)
            _property(method, "MethodNode", operation.opcua_method)

    operational = _submodel(submodels_root, asset, "ModPlantOperationalData", MODPLANT_SM)
    _property(operational, "ProfileVersion", "2.0.0")
    _property(operational, "MaximumVolumeL", asset.maximum_volume_l, "xs:double", f"{MODPLANT_SM}/MaximumVolume")
    inventory = _collection(operational, "InitialInventory", f"{MODPLANT_SM}/InitialInventory")
    _property(inventory, "Material", asset.initial_material)
    _property(inventory, "QuantityL", asset.initial_quantity_l, "xs:double")
    operation_set = _collection(operational, "OperationCapabilities", f"{MODPLANT_SM}/OperationCapabilities")
    for index, operation in enumerate(asset.operations, 1):
        item = _collection(operation_set, f"Operation{index:02d}", f"{MODPLANT_SM}/Operation")
        _property(item, "Name", operation.operation)
        _property(item, "Parameter", operation.parameter)
        _property(
            item,
            "UsageCostPerInvocationEUR",
            operation.usage_cost_per_invocation_eur,
            "xs:double",
            f"{MODPLANT_SM}/UsageCostPerInvocation",
        )
        _property(
            item,
            "EnergyConsumptionRateKWhPerSecond",
            operation.energy_consumption_rate_kwh_s,
            "xs:double",
            f"{MODPLANT_SM}/EnergyConsumptionRate",
        )
        _property(
            item,
            "CO2EmissionRateKgCO2ePerSecond",
            operation.co2_emission_rate_kg_s,
            "xs:double",
            f"{MODPLANT_SM}/OperationalCO2EmissionRate",
        )
        _property(item, "MetricBasis", operation.metric_basis, semantic=f"{MODPLANT_SM}/MetricBasis")
        if operation.has_fixed_duration:
            _property(item, "FixedDurationS", operation.fixed_duration_s, "xs:double")
        _property(item, "OPCUAMethod", operation.opcua_method)
    ports = _collection(operational, "PhysicalInterfaces", f"{MODPLANT_SM}/PhysicalInterfaces")
    for index, port in enumerate(asset.interfaces, 1):
        item = _collection(ports, f"Port{index:02d}", f"{MODPLANT_SM}/PhysicalInterface")
        _property(item, "Type", port.port_type)
        _property(item, "Port", port.port)
    return ET.ElementTree(env)


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", value).strip("_") or "Item"


def export_module_aasx(asset: ModuleAsset, output_path: str | Path) -> str:
    output = Path(output_path)
    if output.suffix.lower() != ".aasx":
        output = output.with_suffix(".aasx")
    output.parent.mkdir(parents=True, exist_ok=True)
    tree = build_module_aas_environment(asset)
    ET.indent(tree, space="  ")
    buffer = io.BytesIO()
    tree.write(buffer, encoding="utf-8", xml_declaration=True)
    aas_name = f"aasx/{asset.asset_id}/{asset.asset_id}.aas.xml"
    content_types = """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/aasx/aasx-origin" ContentType="application/asset-administration-shell-package"/>
</Types>"""
    root_rels = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="AASXOrigin" Type="http://admin-shell.io/aasx/relationships/aasx-origin" Target="/aasx/aasx-origin"/>
</Relationships>"""
    origin_rels = f"""<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="AAS" Type="http://admin-shell.io/aasx/relationships/aas-spec" Target="/{aas_name}"/>
</Relationships>"""
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", root_rels)
        archive.writestr("aasx/aasx-origin", b"")
        archive.writestr("aasx/_rels/aasx-origin.rels", origin_rels)
        archive.writestr(aas_name, buffer.getvalue())
    return str(output)


def _load_xml(path: Path) -> ET.ElementTree:
    if path.suffix.lower() == ".aasx":
        with ZipFile(path) as archive:
            candidates = [
                name for name in archive.namelist()
                if name.lower().endswith((".aas.xml", ".xml"))
                and not name.startswith(("_rels/", "docProps/"))
                and "[Content_Types]" not in name
            ]
            if not candidates:
                raise ValueError(f"No AAS XML environment found in {path}")
            preferred = next((name for name in candidates if name.lower().endswith(".aas.xml")), candidates[0])
            return ET.parse(archive.open(preferred))
    return ET.parse(path)


def _all_properties(parent: ET.Element) -> dict[str, str]:
    values = {}
    for prop in parent.iter(_q("property")):
        name = _text(prop, "idShort")
        if name:
            values[name] = _text(prop, "value")
    return values


def _children_collections(parent: ET.Element) -> list[ET.Element]:
    value = parent.find(_q("value"))
    return list(value.findall(_q("submodelElementCollection"))) if value is not None else []


def _find_collection(parent: ET.Element, id_short: str) -> ET.Element | None:
    return next((item for item in parent.iter(_q("submodelElementCollection")) if _text(item, "idShort") == id_short), None)


def _float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_module_asset(path: str | Path) -> ModuleAsset:
    source = Path(path)
    tree = _load_xml(source)
    root = tree.getroot()
    shell = root.find(f".//{_q('assetAdministrationShell')}")
    global_id = _text(shell.find(_q("assetInformation")), "globalAssetId") if shell is not None else ""
    shell_short = _text(shell, "idShort") if shell is not None else source.stem
    asset_id = (global_id.rstrip("/").split("/")[-1] or shell_short).replace("_AAS", "")
    warnings: list[str] = []
    submodels = list(root.findall(f".//{_q('submodels')}/{_q('submodel')}"))
    operational = next((item for item in submodels if _semantic_value(item) == MODPLANT_SM or _text(item, "idShort") == "ModPlantOperationalData"), None)
    aid = next((item for item in submodels if "AssetInterfaces" in _text(item, "idShort") or "AssetInterfaces" in _semantic_value(item)), None)

    if operational is not None:
        props = _all_properties(operational)
        maximum = _float(props.get("MaximumVolumeL"), 0)
        inventory = _find_collection(operational, "InitialInventory")
        inventory_props = _all_properties(inventory) if inventory is not None else {}
        operation_root = _find_collection(operational, "OperationCapabilities")
        operations = []
        for collection in _children_collections(operation_root) if operation_root is not None else []:
            values = _all_properties(collection)
            operation_name = values.get("Name", "None")
            operations.append(OperationCapability(
                operation=operation_name,
                parameter=_coerce(values.get("Parameter", "")),
                usage_cost_per_invocation_eur=_float(values.get("UsageCostPerInvocationEUR")),
                energy_consumption_rate_kwh_s=_float(values.get("EnergyConsumptionRateKWhPerSecond")),
                co2_emission_rate_kg_s=_float(values.get("CO2EmissionRateKgCO2ePerSecond")),
                fixed_duration_s=_float(values.get("FixedDurationS")),
                opcua_method=values.get("OPCUAMethod", ""),
                metric_basis=values.get("MetricBasis", "Unspecified"),
            ))
        port_root = _find_collection(operational, "PhysicalInterfaces")
        interfaces = []
        for collection in _children_collections(port_root) if port_root is not None else []:
            values = _all_properties(collection)
            interfaces.append(InterfacePort(values.get("Type", "Input").title(), values.get("Port", "")))
    else:
        defaults = default_plant_model().by_id().get(asset_id)
        maximum = defaults.maximum_volume_l if defaults else _extract_legacy_maximum_volume(root)
        operations = list(defaults.operations) if defaults else default_operations()
        interfaces = list(defaults.interfaces) if defaults else [InterfacePort("Input", f"{asset_id}_In1"), InterfacePort("Output", f"{asset_id}_Out1")]
        inventory_props = {
            "Material": defaults.initial_material if defaults else "",
            "QuantityL": str(defaults.initial_quantity_l if defaults else 0),
        }
        warnings.append(f"{asset_id}: ModPlantOperationalData is missing; RTN defaults were applied")

    aid_props = _all_properties(aid) if aid is not None else {}
    endpoint = aid_props.get("Endpoint", "") or _find_text_matching(root, r"^opc\.tcp://")
    namespace = aid_props.get("NamespaceUri", "") or f"urn:modplant:{asset_id.lower()}"
    if not endpoint:
        endpoint = f"opc.tcp://localhost:4840/{asset_id}"
        warnings.append(f"{asset_id}: OPC UA endpoint is missing; a local placeholder was applied")
    asset = ModuleAsset(
        asset_id=asset_id,
        maximum_volume_l=maximum or 10,
        operations=operations,
        interfaces=interfaces,
        initial_material=inventory_props.get("Material", ""),
        initial_quantity_l=_float(inventory_props.get("QuantityL")),
        opcua_endpoint=endpoint,
        opcua_namespace_uri=namespace,
        source_path=str(source),
        warnings=warnings,
    )
    asset.validate()
    return asset


def _coerce(value: str):
    value = (value or "").strip()
    if not value:
        return ""
    try:
        return float(value) if "." in value else int(value)
    except ValueError:
        return value


def _find_text_matching(root: ET.Element, pattern: str) -> str:
    regex = re.compile(pattern, re.IGNORECASE)
    for element in root.iter():
        text = (element.text or "").strip()
        if regex.search(text):
            return text
    return ""


def _extract_legacy_maximum_volume(root: ET.Element) -> float:
    for element in root.iter():
        if _text(element, "idShort").casefold() == "volume":
            maximum = _text(element, "max") or _text(element, "value")
            if _float(maximum) > 0:
                return _float(maximum)
    return 10.0


def load_plant_from_aas(paths: list[str | Path], *, include_defaults: bool = False) -> PlantModel:
    plant = default_plant_model() if include_defaults else PlantModel()
    for path in paths:
        plant.upsert(load_module_asset(path))
    plant.validate()
    return plant
