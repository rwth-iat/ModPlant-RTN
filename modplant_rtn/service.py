from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from time import perf_counter
from typing import Any, Callable

RTN_ROOT = Path(__file__).resolve().parents[1]
MODPLANT_ROOT = RTN_ROOT.parent
for location in (RTN_ROOT / "scripts", MODPLANT_ROOT / "shared"):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

from cp_sat_planner import PlannerConfig
from isa88_recipe import parse_general_recipe_xml_to_ir, save_master_recipe_xml_from_ir
from pipeline import PipelineResult, build_isa88_recipe_from_setting, plan_general_recipe_xml
from recipe_ir import auto_enrich_recipe_spec, build_recipe_ir

from .aas import export_module_aasx, load_plant_from_aas
from .models import PlantModel, ModuleAsset, default_plant_model, expanded_example_plant_model


DEMO_RECIPE_SPEC = {
    "id": "Module_Parallel_Choice_Example",
    "volume": 6.0,
    "procedure": [
        {"dose": {"ingredient": "A", "amount_L": 1.0}},
        {"dose": {"ingredient": "B", "amount_L": 2.0}},
        {"dose": {"ingredient": "C", "amount_L": 3.0}},
        {"mix": {"rpm": 150, "duration_s": 30}},
        {"usage": {"duration_s": 3600}},
        {"settling": {"duration_s": 300}},
        {"separation": {"order": ["C", "B", "A"]}},
    ],
}


@dataclass
class RTNSettings:
    solver_time_limit_s: float = 180.0
    num_workers: int = 8
    solver_random_seed: int = 0
    time_penalty_per_second: float = 0.5
    profit_per_litre: float = 300.0
    usage_cost_weight: float = 1.0
    energy_cost_weight: float = 1.0
    co2_penalty: float = 1.0
    electricity_price_eur_per_kwh: float = 0.3
    connect_duration_s: float = 3.0
    disconnect_duration_s: float = 2.0
    require_final_disconnect: bool = False
    relative_gap_limit: float = 0.02
    enable_auxiliary_transfers: bool = True
    allow_process_transfers: bool = True
    auxiliary_transfer_mode: str = "lazy"

    def validate(self) -> None:
        if self.solver_time_limit_s <= 0:
            raise ValueError("Solver time limit must be positive")
        if not 1 <= self.num_workers <= 64:
            raise ValueError("CP-SAT workers must be between 1 and 64")
        if self.solver_random_seed < 0:
            raise ValueError("Solver random seed must not be negative")
        if self.time_penalty_per_second < 0:
            raise ValueError("Time penalty must not be negative")
        if self.profit_per_litre < 0:
            raise ValueError("Profit per litre must not be negative")
        if min(self.usage_cost_weight, self.energy_cost_weight, self.co2_penalty) < 0:
            raise ValueError("Cost weights and CO2 penalty must not be negative")
        if self.electricity_price_eur_per_kwh < 0:
            raise ValueError("Electricity price must not be negative")
        if self.connect_duration_s < 0:
            raise ValueError("Connect duration must not be negative")
        if self.disconnect_duration_s < 0:
            raise ValueError("Disconnect duration must not be negative")
        if not 0 <= self.relative_gap_limit <= 1:
            raise ValueError("Relative gap must be between 0 and 1")
        if self.auxiliary_transfer_mode not in {"lazy", "eager"}:
            raise ValueError("Auxiliary transfer mode must be lazy or eager")

    def to_dict(self) -> dict[str, Any]:
        return {
            "solver_time_limit_s": self.solver_time_limit_s,
            "num_workers": self.num_workers,
            "solver_random_seed": self.solver_random_seed,
            "time_penalty_per_second": self.time_penalty_per_second,
            "profit_per_litre": self.profit_per_litre,
            "usage_cost_weight": self.usage_cost_weight,
            "energy_cost_weight": self.energy_cost_weight,
            "co2_penalty": self.co2_penalty,
            "electricity_price_eur_per_kwh": self.electricity_price_eur_per_kwh,
            "connect_duration_s": self.connect_duration_s,
            "disconnect_duration_s": self.disconnect_duration_s,
            "require_final_disconnect": self.require_final_disconnect,
            "relative_gap_limit": self.relative_gap_limit,
            "enable_auxiliary_transfers": self.enable_auxiliary_transfers,
            "allow_process_transfers": self.allow_process_transfers,
            "auxiliary_transfer_mode": self.auxiliary_transfer_mode,
        }

    def to_planner_config(self, recipe_volume_l: float) -> PlannerConfig:
        """Translate UI settings once for every App and Notebook caller."""
        self.validate()
        return PlannerConfig(
            base_profit=float(recipe_volume_l) * self.profit_per_litre,
            lambda_per_second=-abs(self.time_penalty_per_second),
            usage_cost_weight=self.usage_cost_weight,
            energy_cost_weight=self.energy_cost_weight,
            co2_penalty=self.co2_penalty,
            electricity_price_eur_per_kwh=self.electricity_price_eur_per_kwh,
            connect_duration_s=self.connect_duration_s,
            disconnect_duration_s=self.disconnect_duration_s,
            require_final_disconnect=self.require_final_disconnect,
            enable_auxiliary_transfers=self.enable_auxiliary_transfers,
            allow_process_transfers=self.allow_process_transfers,
            auxiliary_transfer_mode=self.auxiliary_transfer_mode,
            solver_time_limit_s=self.solver_time_limit_s,
            num_workers=self.num_workers,
            solver_random_seed=self.solver_random_seed,
            relative_gap_limit=self.relative_gap_limit,
        )


@dataclass
class OptimizationSession:
    recipe_path: str
    plant: PlantModel
    pipeline: PipelineResult
    elapsed_s: float = 0.0
    settings: RTNSettings | None = None

    @property
    def plan_rows(self) -> list[dict[str, Any]]:
        assets = self.plant.by_id()
        rows = []
        for row in self.pipeline.planner_result.to_rows():
            equipment = row.get("Module") or row.get("Target Module") or row.get("Source Module")
            asset = assets.get(equipment)
            enriched = dict(row)
            enriched["OPC UA Endpoint"] = asset.opcua_endpoint if asset else ""
            enriched["Namespace URI"] = asset.opcua_namespace_uri if asset else ""
            rows.append(enriched)
        return rows

    @property
    def summary(self) -> dict[str, Any]:
        result = self.pipeline.planner_result
        validation = self.pipeline.validation_result
        return {
            "status": result.status,
            "objective_profit": result.objective_profit,
            "makespan_s": result.makespan_s,
            "total_cost": result.total_cost,
            "total_usage_cost": result.total_usage_cost,
            "total_energy_consumption_kwh": result.total_energy_consumption_kwh,
            "total_energy_cost": result.total_energy_cost,
            "total_co2_emissions_kg": result.total_co2_emissions_kg,
            "total_weighted_usage_cost": result.total_weighted_usage_cost,
            "total_weighted_energy_cost": result.total_weighted_energy_cost,
            "total_weighted_co2_cost": result.total_weighted_co2_cost,
            "selected_branches": result.selected_branches,
            "validation_valid": validation.valid,
            "validation_errors": list(validation.errors),
            "validation_warnings": list(getattr(validation, "warnings", [])),
            "master_recipe": self.pipeline.master_xml_path,
            "elapsed_s": self.elapsed_s,
            "settings": self.settings.to_dict() if self.settings else {},
        }


def preview_recipe(path: str | Path, **runtime_decisions):
    return parse_general_recipe_xml_to_ir(path, **runtime_decisions)


def ensure_demo_recipe(output_dir: str | Path | None = None) -> tuple[Any, str]:
    """Create the built-in, deterministic General Recipe used by the desktop UI."""
    output = Path(output_dir or RTN_ROOT / "generated" / "modplant-rtn" / "demo")
    ir, xml_path, _ = build_isa88_recipe_from_setting(DEMO_RECIPE_SPEC, out_dir=output)
    return ir, xml_path


def enrich_linear_recipe_for_planning(ir):
    """Lift a flat process sequence into the nonlinear planning form used by RTN.

    Existing AND/XOR/OR/loop/jump graphs are preserved.  A plain sequence is
    reconstructed as a recipe spec, then the established notebook enrichment
    adds an AND fork/join around consecutive dosing and an optimizer-resolved
    usage choice.  The imported General Recipe itself remains unchanged in the
    UI; only the PlanningGraph/solved projection uses this enriched IR.
    """
    if any(node.is_control for node in ir.nodes.values()):
        return ir
    nodes = ir.topological_nodes()
    if not any(node.node_type == "mix" for node in nodes) or sum(node.node_type == "dose" for node in nodes) < 2:
        return ir
    procedure: list[dict[str, Any]] = []
    separation_order: list[str] = []
    for node in nodes:
        if node.node_type == "separation":
            ingredient = str(node.params.get("ingredient", ""))
            if ingredient:
                separation_order.append(ingredient)
            continue
        if node.node_type in {"dose", "mix", "usage", "settling"}:
            procedure.append({node.node_type: dict(node.params)})
    if separation_order:
        procedure.append({"separation": {"order": separation_order}})
    spec = {
        "id": ir.id,
        "volume": ir.volume,
        "procedure": procedure,
        "metadata": {**dict(ir.metadata), "planning_enrichment": "parallel_dosing_and_usage_choice"},
    }
    return build_recipe_ir(auto_enrich_recipe_spec(spec, parallel_dosing=True, usage_alternatives=True))


def optimize_recipe(
    recipe_path: str | Path,
    plant: PlantModel,
    *,
    output_dir: str | Path | None = None,
    solver_time_limit_s: float | None = None,
    settings: RTNSettings | None = None,
    planning_context: dict[str, Any] | None = None,
    loop_iterations: dict[str, int] | None = None,
    jump_decisions: dict[str, bool] | None = None,
    progress_callback: Callable[[str, float, str], None] | None = None,
) -> OptimizationSession:
    started_at = perf_counter()
    if progress_callback:
        progress_callback("aas", 0.02, "Validating AAS equipment capabilities, ports, inventory and OPC UA bindings")
    plant.validate()
    settings = settings or RTNSettings()
    if solver_time_limit_s is not None:
        settings = RTNSettings(**{**settings.to_dict(), "solver_time_limit_s": float(solver_time_limit_s)})
    settings.validate()
    operations, interfaces, maximum_volumes, resources = plant.to_rtn_inputs()
    output = Path(output_dir or RTN_ROOT / "generated" / "modplant-rtn")
    output.mkdir(parents=True, exist_ok=True)
    if progress_callback:
        progress_callback("recipe", 0.08, "Parsing General Recipe and resolving OR, loops and conditional jumps")
    general_recipe_ir = preview_recipe(
        recipe_path,
        planning_context=planning_context,
        loop_iterations=loop_iterations,
        jump_decisions=jump_decisions,
    )
    recipe_ir = enrich_linear_recipe_for_planning(general_recipe_ir)
    config = settings.to_planner_config(recipe_ir.volume)
    result = plan_general_recipe_xml(
        recipe_path,
        operations,
        module_interfaces=interfaces,
        module_maximum_volume=maximum_volumes,
        module_resources=resources,
        planner_config=config,
        out_dir=output,
        planning_context=planning_context,
        loop_iterations=loop_iterations,
        jump_decisions=jump_decisions,
        equipment_bindings=plant.equipment_bindings(),
        preparsed_ir=recipe_ir,
        progress_callback=progress_callback,
        generate_master_recipe=False,
    )
    return OptimizationSession(str(recipe_path), plant, result, perf_counter() - started_at, settings)


def optimize_recipe_ir(
    recipe_ir,
    plant: PlantModel,
    *,
    recipe_path: str | Path,
    output_dir: str | Path | None = None,
    settings: RTNSettings | None = None,
    progress_callback: Callable[[str, float, str], None] | None = None,
) -> OptimizationSession:
    """Run a prepared RecipeIR through the same service used by the desktop App.

    This is the stable Notebook integration point.  It intentionally delegates
    RTN compilation, CP-SAT, Connect/Disconnect lifecycle construction, cost
    accounting and validation to ``plan_general_recipe_xml``. Master Recipe
    generation is deliberately deferred until an explicit export action.
    """

    started_at = perf_counter()
    if progress_callback:
        progress_callback("aas", 0.02, "Validating editable Module capabilities, ports and inventory")
    plant.validate()
    settings = settings or RTNSettings()
    settings.validate()
    operations, interfaces, maximum_volumes, resources = plant.to_rtn_inputs()
    output = Path(output_dir or RTN_ROOT / "generated" / "notebooks")
    output.mkdir(parents=True, exist_ok=True)
    result = plan_general_recipe_xml(
        recipe_path,
        operations,
        module_interfaces=interfaces,
        module_maximum_volume=maximum_volumes,
        module_resources=resources,
        planner_config=settings.to_planner_config(recipe_ir.volume),
        out_dir=output,
        equipment_bindings=plant.equipment_bindings(),
        preparsed_ir=recipe_ir,
        progress_callback=progress_callback,
        generate_master_recipe=False,
    )
    return OptimizationSession(
        str(recipe_path),
        plant,
        result,
        perf_counter() - started_at,
        settings,
    )


def load_or_default_plant(paths: list[str] | None = None) -> PlantModel:
    if paths:
        return load_plant_from_aas(paths)
    return default_plant_model()


def generate_example_aasx_set(output_dir: str | Path) -> list[str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    return [export_module_aasx(asset, output / f"{asset.asset_id}.aasx") for asset in expanded_example_plant_model().assets]


def export_master_recipe(
    session: OptimizationSession,
    target: str | Path,
    *,
    record_path: bool = True,
) -> str:
    """Generate a Master Recipe only in response to an explicit export."""
    if not session.pipeline.validation_result.valid:
        raise ValueError("Only a successfully validated plan can be exported as a Master Recipe")
    destination = Path(target)
    destination.parent.mkdir(parents=True, exist_ok=True)
    exported = save_master_recipe_xml_from_ir(
        session.pipeline.recipe_ir,
        destination,
        selected_branches=session.pipeline.planner_result.selected_branches,
        planner_result=session.pipeline.planner_result,
        equipment_bindings=session.plant.equipment_bindings(),
    )
    if record_path:
        session.pipeline.master_xml_path = str(exported)
    return str(exported)


def copy_master_recipe(session: OptimizationSession, target: str | Path) -> str:
    """Backward-compatible name for explicit, on-demand Master export."""
    return export_master_recipe(session, target)


def export_custom_module(asset: ModuleAsset, target: str | Path) -> str:
    return export_module_aasx(asset, target)
