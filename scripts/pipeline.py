from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
from pathlib import Path
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

_SHARED = Path(__file__).resolve().parents[1] / "shared"
if str(_SHARED) not in sys.path:
    sys.path.insert(0, str(_SHARED))

from modplant_recipe import import_general_recipe
from modplant_recipe.conditions import evaluate_condition

try:
    from schedule_validator import ValidationResult, validate_schedule
    from cp_sat_planner import PlannerConfig, PlannerResult, solve_recipe_ir_with_cp_sat, solve_rtn_with_cp_sat
    from isa88_recipe import parse_general_recipe_xml_to_ir, save_general_recipe_xml_from_ir, save_master_recipe_xml_from_ir, save_recipe_ir_json
    from recipe_ir import RecipeIR, build_recipe_ir, recipe_spec_from_legacy_order
    from recipe_to_rtn import recipe_ir_to_rtn
    from rtn import RTNModel
except ImportError:  # pragma: no cover - notebook direct import compatibility
    from schedule_validator import ValidationResult, validate_schedule
    from cp_sat_planner import PlannerConfig, PlannerResult, solve_recipe_ir_with_cp_sat, solve_rtn_with_cp_sat
    from isa88_recipe import parse_general_recipe_xml_to_ir, save_general_recipe_xml_from_ir, save_master_recipe_xml_from_ir, save_recipe_ir_json
    from recipe_ir import RecipeIR, build_recipe_ir, recipe_spec_from_legacy_order
    from recipe_to_rtn import recipe_ir_to_rtn
    from rtn import RTNModel


@dataclass
class PipelineResult:
    recipe_ir: RecipeIR
    xml_path: str
    ir_json_path: str
    planner_result: PlannerResult
    validation_result: ValidationResult
    master_xml_path: str = ""
    rtn_model: Optional[RTNModel] = None

    @property
    def replay_result(self) -> ValidationResult:
        """Deprecated alias for ``validation_result``."""
        import warnings
        warnings.warn(
            "PipelineResult.replay_result is deprecated; use validation_result.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.validation_result


@dataclass
class RuntimeScenarioResult:
    """One finite RTN plan in a runtime-contingent plan family."""

    scenario_id: str
    loop_iterations: Dict[str, int]
    jump_decisions: Dict[str, bool]
    result: PipelineResult


ProgressCallback = Callable[[str, float, str], None]


def _emit_progress(callback: Optional[ProgressCallback], stage: str, progress: float, detail: str) -> None:
    if callback is not None:
        callback(stage, max(0.0, min(1.0, float(progress))), detail)


def build_isa88_recipe_from_setting(
    setting: Dict[str, Any],
    *,
    out_dir: str | Path = "RTN/generated",
    parallel_dosing_before_mix: bool = False,
    usage_alternatives: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[RecipeIR, str, str]:
    """Build RecipeIR plus ISA-88/B2MML XML from either old order or new recipe_spec."""
    if "procedure" in setting:
        spec = setting
    elif "order" in setting:
        spec = recipe_spec_from_legacy_order(
            setting,
            parallel_dosing_before_mix=parallel_dosing_before_mix,
            usage_alternatives=usage_alternatives,
        )
    else:
        raise ValueError("Expected either a new recipe_spec with 'procedure' or a legacy order with 'order'.")

    ir = build_recipe_ir(spec)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    xml_path = save_general_recipe_xml_from_ir(ir, out / f"GeneralRecipe_{ir.id}.xml")
    ir_json_path = save_recipe_ir_json(ir, out / f"RecipeIR_{ir.id}.json")
    return ir, xml_path, ir_json_path


def plan_isa88_recipe_with_cpn_cp_sat(
    setting: Dict[str, Any],
    module_ops: Dict[str, List[Tuple[str, Any, float]]],
    *,
    module_interfaces: Optional[Dict[str, List[Tuple[str, str]]]] = None,
    module_maximum_volume: Optional[Dict[str, List[float]]] = None,
    module_resources: Optional[Dict[str, List[Any]]] = None,
    planner_config: Optional[PlannerConfig] = None,
    out_dir: str | Path = "RTN/generated",
    parallel_dosing_before_mix: bool = False,
    usage_alternatives: Optional[List[Dict[str, Any]]] = None,
    expose_rtn_model: bool = True,
    equipment_bindings: Optional[Dict[str, Dict[str, Any]]] = None,
) -> PipelineResult:
    ir, xml_path, ir_json_path = build_isa88_recipe_from_setting(
        setting,
        out_dir=out_dir,
        parallel_dosing_before_mix=parallel_dosing_before_mix,
        usage_alternatives=usage_alternatives,
    )
    cfg = planner_config or PlannerConfig()
    planner_result = solve_recipe_ir_with_cp_sat(
        ir,
        module_ops,
        module_interfaces=module_interfaces,
        module_maximum_volume=module_maximum_volume,
        module_resources=module_resources,
        config=cfg,
    )
    validation_result = validate_schedule(
        ir,
        planner_result,
        module_maximum_volume=module_maximum_volume,
        module_resources=module_resources,
        config=cfg,
    )
    if not validation_result.valid:
        raise ValueError("Schedule validation failed: " + "; ".join(validation_result.errors))
    master_xml_path = save_master_recipe_xml_from_ir(
        ir,
        Path(out_dir) / f"MasterRecipe_{ir.id}.xml",
        selected_branches=planner_result.selected_branches,
        planner_result=planner_result,
        equipment_bindings=equipment_bindings,
    )
    rtn_model = (
        recipe_ir_to_rtn(
            ir,
            module_ops=module_ops,
            module_interfaces=module_interfaces,
            module_maximum_volume=module_maximum_volume,
            module_resources=module_resources,
        )
        if expose_rtn_model
        else None
    )
    return PipelineResult(
        recipe_ir=ir,
        xml_path=xml_path,
        ir_json_path=ir_json_path,
        planner_result=planner_result,
        validation_result=validation_result,
        master_xml_path=master_xml_path,
        rtn_model=rtn_model,
    )


# Preferred public name (legacy alias retained for compatibility)
plan_recipe = plan_isa88_recipe_with_cpn_cp_sat


def plan_general_recipe_xml(
    xml_path: str | Path,
    module_ops: Dict[str, List[Tuple[str, Any, float]]],
    *,
    module_interfaces: Optional[Dict[str, List[Tuple[str, str]]]] = None,
    module_maximum_volume: Optional[Dict[str, List[float]]] = None,
    module_resources: Optional[Dict[str, List[Any]]] = None,
    planner_config: Optional[PlannerConfig] = None,
    out_dir: str | Path = "RTN/generated",
    planning_context: Optional[Dict[str, Any]] = None,
    loop_iterations: Optional[Dict[str, int]] = None,
    jump_decisions: Optional[Dict[str, bool]] = None,
    equipment_bindings: Optional[Dict[str, Dict[str, Any]]] = None,
    preparsed_ir: Optional[RecipeIR] = None,
    progress_callback: Optional[ProgressCallback] = None,
    generate_master_recipe: bool = True,
) -> PipelineResult:
    """Import and solve a runtime-aware BatchML General Recipe.

    Runtime LoopRegions are compiled with a finite iteration selection, and
    conditional jumps are resolved from explicit decisions or planning context.
    ``generate_master_recipe=False`` keeps optimization side-effect free until
    an explicit UI export action requests the Master Recipe projection.
    """
    if preparsed_ir is None:
        _emit_progress(progress_callback, "recipe", 0.08, "Parsing and normalizing General Recipe control flow")
        ir = parse_general_recipe_xml_to_ir(
            xml_path,
            planning_context=planning_context,
            loop_iterations=loop_iterations,
            jump_decisions=jump_decisions,
        )
    else:
        ir = preparsed_ir
    cfg = planner_config or PlannerConfig()
    if planning_context is not None:
        cfg = replace(cfg, condition_context=dict(planning_context))
    _emit_progress(progress_callback, "rtn", 0.22, "Compiling PlanningGraph and plant capabilities into the RTN model")
    rtn_model = recipe_ir_to_rtn(
        ir,
        module_ops=module_ops,
        module_interfaces=module_interfaces,
        module_maximum_volume=module_maximum_volume,
        module_resources=module_resources,
    )
    _emit_progress(progress_callback, "solve", 0.34, "CP-SAT is selecting Module, routes, timing and nonlinear branches")
    planner_result = solve_rtn_with_cp_sat(rtn_model, config=cfg)
    _emit_progress(progress_callback, "validation", 0.82, "Validating precedence, inventory, capacity, ports and branch semantics")
    validation_result = validate_schedule(
        ir,
        planner_result,
        module_maximum_volume=module_maximum_volume,
        module_resources=module_resources,
        config=cfg,
    )
    if not validation_result.valid:
        raise ValueError("Schedule validation failed: " + "; ".join(validation_result.errors))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ir_json_path = save_recipe_ir_json(ir, out / f"RecipeIR_{ir.id}.json")
    master_xml_path = ""
    if generate_master_recipe:
        _emit_progress(progress_callback, "master", 0.91, "Writing optimal plan, OPC UA bindings and ports into the Master Recipe")
        master_xml_path = save_master_recipe_xml_from_ir(
            ir,
            out / f"MasterRecipe_{ir.id}.xml",
            selected_branches=planner_result.selected_branches,
            planner_result=planner_result,
            equipment_bindings=equipment_bindings,
        )
    completion = (
        "Optimization, validation and Master Recipe generation completed"
        if generate_master_recipe
        else "Optimization and validation completed; Master Recipe export is available on request"
    )
    _emit_progress(progress_callback, "complete", 1.0, completion)
    return PipelineResult(
        recipe_ir=ir,
        xml_path=str(Path(xml_path)),
        ir_json_path=ir_json_path,
        planner_result=planner_result,
        validation_result=validation_result,
        master_xml_path=master_xml_path,
        rtn_model=rtn_model,
    )


def plan_general_recipe_runtime_scenarios(
    xml_path: str | Path,
    module_ops: Dict[str, List[Tuple[str, Any, float]]],
    *,
    module_interfaces: Optional[Dict[str, List[Tuple[str, str]]]] = None,
    module_maximum_volume: Optional[Dict[str, List[float]]] = None,
    module_resources: Optional[Dict[str, List[Any]]] = None,
    planner_config: Optional[PlannerConfig] = None,
    out_dir: str | Path = "RTN/generated/runtime-scenarios",
    planning_context: Optional[Dict[str, Any]] = None,
    loop_iterations: Optional[Dict[str, int]] = None,
    jump_decisions: Optional[Dict[str, bool]] = None,
    max_scenarios: int = 64,
) -> List[RuntimeScenarioResult]:
    """Plan every finite runtime loop/jump scenario not fixed by the caller.

    This produces a contingent plan family. At runtime, Recipol or a supervisory
    controller selects/replans the member matching observed loop counts and jump
    conditions; RTN never guesses an unknown sensor result.
    """
    graph = import_general_recipe(xml_path)
    fixed_loops = dict(loop_iterations or {})
    fixed_jumps = dict(jump_decisions or {})
    context = dict(planning_context or {})

    loop_dimensions: List[Tuple[str, List[int]]] = []
    for node in sorted((item for item in graph.nodes if item.kind == "LoopRegion"), key=lambda item: item.id):
        loop = node.loop or {}
        if node.id in fixed_loops:
            values = [int(fixed_loops[node.id])]
        else:
            bound = loop.get("maxIterations")
            if bound is None:
                raise ValueError(f"LoopRegion {node.id} needs maxIterations for runtime scenario planning")
            minimum = int(loop.get("minIterations", 1))
            values = list(range(minimum, int(bound) + 1))
        loop_dimensions.append((node.id, values))

    jump_dimensions: List[Tuple[str, List[bool]]] = []
    for edge in sorted((item for item in graph.edges if item.flow_type == "Jump"), key=lambda item: item.id):
        if edge.id in fixed_jumps:
            values = [bool(fixed_jumps[edge.id])]
        elif edge.condition is not None:
            try:
                values = [bool(evaluate_condition(edge.condition, context))]
            except KeyError:
                values = [False, True]
        else:
            raise ValueError(f"Conditional Jump {edge.id} needs a condition or an explicit decision")
        jump_dimensions.append((edge.id, values))

    dimensions: List[Tuple[str, str, List[Any]]] = [
        ("loop", key, values) for key, values in loop_dimensions
    ] + [
        ("jump", key, values) for key, values in jump_dimensions
    ]
    scenario_count = 1
    for _, _, values in dimensions:
        scenario_count *= len(values)
    if scenario_count > int(max_scenarios):
        raise ValueError(
            f"Runtime scenario count {scenario_count} exceeds max_scenarios={int(max_scenarios)}"
        )

    combinations = product(*(values for _, _, values in dimensions)) if dimensions else [()]
    output = Path(out_dir)
    scenarios: List[RuntimeScenarioResult] = []
    for index, combination in enumerate(combinations, start=1):
        scenario_loops = dict(fixed_loops)
        scenario_jumps = dict(fixed_jumps)
        labels = []
        for (kind, key, _), value in zip(dimensions, combination):
            if kind == "loop":
                scenario_loops[key] = int(value)
                labels.append(f"{key}={int(value)}")
            else:
                scenario_jumps[key] = bool(value)
                labels.append(f"{key}={'take' if value else 'continue'}")
        scenario_id = f"scenario_{index:03d}" + ("__" + "__".join(labels) if labels else "")
        result = plan_general_recipe_xml(
            xml_path,
            module_ops,
            module_interfaces=module_interfaces,
            module_maximum_volume=module_maximum_volume,
            module_resources=module_resources,
            planner_config=planner_config,
            out_dir=output / scenario_id,
            planning_context=context,
            loop_iterations=scenario_loops,
            jump_decisions=scenario_jumps,
        )
        scenarios.append(RuntimeScenarioResult(scenario_id, scenario_loops, scenario_jumps, result))
    return scenarios


# Runtime replanning is a context-updated invocation of the same finite compiler.
replan_runtime_recipe = plan_general_recipe_xml
