from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable


@dataclass
class OperationCapability:
    operation: str
    parameter: Any = ""
    usage_cost_per_invocation_eur: float = 0.0
    energy_consumption_rate_kwh_s: float = 0.0
    co2_emission_rate_kg_s: float = 0.0
    fixed_duration_s: float = 0.0
    opcua_method: str = ""
    metric_basis: str = "Engineering estimate"

    def __post_init__(self) -> None:
        self.fixed_duration_s = float(self.fixed_duration_s or 0)
        self.usage_cost_per_invocation_eur = float(self.usage_cost_per_invocation_eur or 0)
        self.energy_consumption_rate_kwh_s = float(self.energy_consumption_rate_kwh_s or 0)
        self.co2_emission_rate_kg_s = float(self.co2_emission_rate_kg_s or 0)
        if not self.has_fixed_duration:
            self.fixed_duration_s = 0.0

    @property
    def has_fixed_duration(self) -> bool:
        """Only equipment-intrinsic connection actions own a fixed duration."""
        return self.operation.strip().casefold() in {"connect", "disconnect"}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OperationCapability":
        operation = str(data.get("operation", "None"))
        return cls(
            operation=operation,
            parameter=data.get("parameter", ""),
            usage_cost_per_invocation_eur=float(data.get("usage_cost_per_invocation_eur", 0) or 0),
            energy_consumption_rate_kwh_s=float(data.get("energy_consumption_rate_kwh_s", 0) or 0),
            co2_emission_rate_kg_s=float(data.get("co2_emission_rate_kg_s", 0) or 0),
            fixed_duration_s=float(data.get("fixed_duration_s", 0) or 0),
            opcua_method=str(data.get("opcua_method", "")),
            metric_basis=str(data.get("metric_basis", "Engineering estimate")),
        )


@dataclass
class InterfacePort:
    port_type: str
    port: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InterfacePort":
        kind = str(data.get("port_type", data.get("type", "Input"))).title()
        if kind not in {"Input", "Output"}:
            raise ValueError(f"Unsupported physical interface type: {kind}")
        return cls(kind, str(data.get("port", "")))


@dataclass
class ModuleAsset:
    asset_id: str
    maximum_volume_l: float
    operations: list[OperationCapability] = field(default_factory=list)
    interfaces: list[InterfacePort] = field(default_factory=list)
    initial_material: str = ""
    initial_quantity_l: float = 0.0
    opcua_endpoint: str = ""
    opcua_namespace_uri: str = "urn:modplant:module"
    manufacturer: str = "IAT RWTH Aachen"
    description: str = "Modular process equipment"
    source_path: str = ""
    warnings: list[str] = field(default_factory=list)

    def validate(self) -> None:
        if not self.asset_id.strip():
            raise ValueError("Module ID must not be empty")
        if self.maximum_volume_l <= 0:
            raise ValueError(f"{self.asset_id}: maximum volume must be positive")
        if self.initial_quantity_l < 0 or self.initial_quantity_l > self.maximum_volume_l:
            raise ValueError(f"{self.asset_id}: initial quantity must be between 0 and maximum volume")
        if self.initial_quantity_l and not self.initial_material:
            raise ValueError(f"{self.asset_id}: initial material is required when initial quantity is positive")
        names = [item.port for item in self.interfaces]
        if any(not name for name in names) or len(names) != len(set(names)):
            raise ValueError(f"{self.asset_id}: physical port names must be unique and non-empty")
        if not self.operations:
            raise ValueError(f"{self.asset_id}: at least one operation capability is required")

    def operation(self, name: str) -> OperationCapability | None:
        wanted = name.casefold()
        return next((item for item in self.operations if item.operation.casefold() == wanted), None)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModuleAsset":
        asset = cls(
            asset_id=str(data.get("asset_id", "")),
            maximum_volume_l=float(data.get("maximum_volume_l", 0) or 0),
            operations=[OperationCapability.from_dict(item) for item in data.get("operations", [])],
            interfaces=[InterfacePort.from_dict(item) for item in data.get("interfaces", [])],
            initial_material=str(data.get("initial_material", "")),
            initial_quantity_l=float(data.get("initial_quantity_l", 0) or 0),
            opcua_endpoint=str(data.get("opcua_endpoint", "")),
            opcua_namespace_uri=str(data.get("opcua_namespace_uri", "urn:modplant:module")),
            manufacturer=str(data.get("manufacturer", "IAT RWTH Aachen")),
            description=str(data.get("description", "Modular process equipment")),
            source_path=str(data.get("source_path", "")),
            warnings=list(data.get("warnings", [])),
        )
        asset.validate()
        return asset


@dataclass
class PlantModel:
    assets: list[ModuleAsset] = field(default_factory=list)

    def validate(self) -> None:
        ids = [asset.asset_id for asset in self.assets]
        if len(ids) != len(set(ids)):
            raise ValueError("Plant contains duplicate Module IDs")
        for asset in self.assets:
            asset.validate()

    def upsert(self, asset: ModuleAsset) -> None:
        asset.validate()
        for index, current in enumerate(self.assets):
            if current.asset_id == asset.asset_id:
                self.assets[index] = asset
                return
        self.assets.append(asset)
        self.assets.sort(key=lambda item: item.asset_id)

    def by_id(self) -> dict[str, ModuleAsset]:
        return {asset.asset_id: asset for asset in self.assets}

    def to_rtn_inputs(self) -> tuple[dict, dict, dict, dict]:
        self.validate()
        operations = {
            asset.asset_id: [
                (
                    item.operation,
                    item.parameter,
                    float(item.usage_cost_per_invocation_eur),
                    float(item.energy_consumption_rate_kwh_s),
                    float(item.co2_emission_rate_kg_s),
                    float(item.fixed_duration_s),
                )
                for item in asset.operations
            ]
            for asset in self.assets
        }
        interfaces = {
            asset.asset_id: [(item.port_type, item.port) for item in asset.interfaces]
            for asset in self.assets
        }
        maximum_volumes = {asset.asset_id: [float(asset.maximum_volume_l)] for asset in self.assets}
        resources = {
            asset.asset_id: [asset.initial_material, float(asset.initial_quantity_l)]
            for asset in self.assets
            if asset.initial_material and asset.initial_quantity_l > 0
        }
        return operations, interfaces, maximum_volumes, resources

    def equipment_bindings(self) -> dict[str, dict[str, Any]]:
        return {
            asset.asset_id: {
                "opcua_endpoint": asset.opcua_endpoint,
                "opcua_namespace_uri": asset.opcua_namespace_uri,
                "operation_methods": {
                    operation.operation.casefold(): operation.opcua_method
                    for operation in asset.operations
                    if operation.opcua_method
                },
            }
            for asset in self.assets
        }

    @property
    def warnings(self) -> list[str]:
        return [warning for asset in self.assets for warning in asset.warnings]


def plant_model_from_rtn_inputs(
    module_ops: dict[str, list[tuple]],
    module_interfaces: dict[str, list[tuple[str, str]]],
    module_maximum_volume: dict[str, Any],
    module_resources: dict[str, Any] | None = None,
    *,
    equipment_bindings: dict[str, dict[str, Any]] | None = None,
) -> PlantModel:
    """Build the App plant model from editable Notebook-style RTN dictionaries.

    Operation records use the public RTN tuple contract::

        (operation, parameter, usage_cost, energy_rate_kwh_s,
         co2_rate_kg_s, fixed_duration_s)

    Historic three-item records remain readable; omitted energy, CO2 and fixed
    duration values become zero.  This adapter lets notebooks retain their
    convenient ``module_ops``/``_hc_data`` input cells while sharing the exact
    App optimization service afterwards.
    """

    resources = module_resources or {}
    bindings = equipment_bindings or {}
    asset_ids = sorted(
        set(module_ops)
        | set(module_interfaces)
        | set(module_maximum_volume)
        | set(resources)
    )
    assets: list[ModuleAsset] = []
    for asset_id in asset_ids:
        binding = bindings.get(asset_id, {})
        operation_methods = binding.get("operation_methods", {}) or {}
        raw_volume = module_maximum_volume.get(asset_id, 0)
        if isinstance(raw_volume, (list, tuple)):
            raw_volume = raw_volume[0] if raw_volume else 0

        operations: list[OperationCapability] = []
        for record in module_ops.get(asset_id, []):
            if not record:
                continue
            values = list(record) + ["", 0.0, 0.0, 0.0, 0.0]
            operations.append(OperationCapability(
                operation=str(values[0]),
                parameter=values[1],
                usage_cost_per_invocation_eur=float(values[2] or 0),
                energy_consumption_rate_kwh_s=float(values[3] or 0),
                co2_emission_rate_kg_s=float(values[4] or 0),
                fixed_duration_s=float(values[5] or 0),
                opcua_method=(
                    str(values[6] or "")
                    if len(record) > 6
                    else str(operation_methods.get(str(values[0]).casefold(), ""))
                ),
                metric_basis="Notebook-editable RTN capability input",
            ))

        raw_resource = resources.get(asset_id, ["", 0.0])
        material = str(raw_resource[0]) if raw_resource else ""
        quantity = float(raw_resource[1] or 0) if len(raw_resource) > 1 else 0.0
        assets.append(ModuleAsset(
            asset_id=asset_id,
            maximum_volume_l=float(raw_volume or 0),
            operations=operations,
            interfaces=[
                InterfacePort(str(port_type).title(), str(port))
                for port_type, port in module_interfaces.get(asset_id, [])
            ],
            initial_material=material,
            initial_quantity_l=quantity,
            opcua_endpoint=str(binding.get("opcua_endpoint", "")),
            opcua_namespace_uri=str(binding.get("opcua_namespace_uri", "urn:modplant:module")),
        ))

    plant = PlantModel(assets)
    plant.validate()
    return plant


def default_operations(stirring: Iterable[int] = (100, 200), power_scale: float = 1.0) -> list[OperationCapability]:
    """Engineering estimates for the pilot-scale demonstration equipment.

    Energy is stored as kWh/s and operational carbon as kg CO2e/s. The latter
    uses an explicit demo estimate of 0.4 kg CO2e/kWh; imported industrial AAS
    instances are expected to provide measured or supplier values instead.
    """
    def capability(
        operation: str,
        parameter: Any,
        usage_cost: float,
        power_kw: float,
        method: str,
        fixed_duration_s: float = 0.0,
    ) -> OperationCapability:
        # The bundled AAS is a didactic optimization fixture.  Its operational
        # energy/CO2 intensities are deliberately amplified so both terms stay
        # visible next to per-invocation equipment costs in a two-decimal plan.
        demo_impact_scale = 100.0
        energy_rate = power_kw * power_scale * demo_impact_scale / 3600.0
        return OperationCapability(
            operation=operation,
            parameter=parameter,
            usage_cost_per_invocation_eur=usage_cost,
            energy_consumption_rate_kwh_s=energy_rate,
            co2_emission_rate_kg_s=energy_rate * 0.4,
            fixed_duration_s=fixed_duration_s,
            opcua_method=method,
            metric_basis="Pilot-scale demo engineering estimate; didactic energy/CO2 impact scale x100; 0.4 kg CO2e/kWh",
        )

    items = [
        capability("Draining", 0.1, 3, 2.2, "Drain"),
        capability("Filling", 0.1, 0, 0.75, "Fill"),
        capability("Settling", "", 1, 0.12, "Settle"),
    ]
    items.extend(capability("Stirring", rpm, 3, 0.02 * float(rpm), "Stir") for rpm in stirring)
    items.extend([
        capability("Connect", "", 2, 0.5, "Connect", 3),
        capability("Disconnect", "", 2, 0.5, "Disconnect", 2),
        capability("None", "", 0, 0.08, ""),
    ])
    return items


def _plant_from_definitions(definitions: Iterable[tuple]) -> PlantModel:
    assets = []
    for asset_id, volume, inputs, outputs, material, quantity, stirring, power_scale in definitions:
        ports = [InterfacePort("Input", f"{asset_id}_In{i}") for i in range(1, inputs + 1)]
        ports += [InterfacePort("Output", f"{asset_id}_Out{i}") for i in range(1, outputs + 1)]
        assets.append(ModuleAsset(
            asset_id=asset_id,
            maximum_volume_l=volume,
            operations=default_operations(stirring, power_scale),
            interfaces=ports,
            initial_material=material,
            initial_quantity_l=quantity,
            opcua_endpoint=f"opc.tcp://localhost:4840/{asset_id}",
            opcua_namespace_uri=f"urn:modplant:{asset_id.lower()}",
        ))
    return PlantModel(assets)


def default_plant_model() -> PlantModel:
    """The four Module supplied for the ModPlant demonstration."""
    return _plant_from_definitions([
        ("HC10", 10, 3, 3, "A", 10, (100, 200), 1.0),
        ("HC20", 15, 3, 3, "B", 10, (150, 300), 1.25),
        ("HC30", 10, 4, 3, "C", 10, (100, 150), 1.0),
        ("HC40", 30, 1, 2, "", 0, (), 1.7),
    ])


def expanded_example_plant_model() -> PlantModel:
    """Optional eight-Module set used only by the AASX example generator."""
    return _plant_from_definitions([
        ("HC10", 10, 3, 3, "A", 10, (100, 200), 1.0),
        ("HC20", 15, 3, 3, "B", 10, (150, 300), 1.25),
        ("HC30", 10, 4, 3, "C", 10, (100, 150), 1.0),
        ("HC40", 30, 1, 2, "", 0, (), 1.7),
        ("HC50", 20, 2, 2, "", 0, (100, 250), 1.35),
        ("HC60", 25, 2, 3, "", 0, (150, 300), 1.5),
        ("HC70", 12, 3, 1, "", 0, (80, 160), 1.0),
        ("HC80", 40, 2, 2, "", 0, (), 2.0),
    ])
