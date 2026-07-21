from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Tuple

try:
    from material_composition import (
        MIXTURE,
        PURE_MATERIAL,
        add_composition,
        composition_signature,
        composition_tuple,
        compositions_proportional,
        is_pure_composition,
        normalize_composition,
        transfer_kind as classify_transfer_kind,
    )
except ImportError:  # pragma: no cover - package import compatibility
    from .material_composition import (
        MIXTURE,
        PURE_MATERIAL,
        add_composition,
        composition_signature,
        composition_tuple,
        compositions_proportional,
        is_pure_composition,
        normalize_composition,
        transfer_kind as classify_transfer_kind,
    )

try:
    from recipe_ir import RecipeIR, RecipeNode
    from rtn import RTNModel, TaskNodeView
except ImportError:  # pragma: no cover - notebook direct import compatibility
    from recipe_ir import RecipeIR, RecipeNode
    from rtn import RTNModel, TaskNodeView


@dataclass
class PlannerConfig:
    base_profit: float = 120.0
    lambda_per_second: float = -0.5
    usage_cost_weight: float = 1.0
    energy_cost_weight: float = 1.0
    co2_penalty: float = 1.0
    electricity_price_eur_per_kwh: float = 0.3
    connect_duration_s: float = 3.0
    disconnect_duration_s: float = 2.0
    require_final_disconnect: bool = False
    enable_auxiliary_transfers: bool = True
    allow_process_transfers: bool = True
    volume_scale: int = 0   # 0 = auto-detect from recipe + plant config
    time_scale: int = 1
    cost_scale: int = 100_000
    max_horizon_s: Optional[int] = None
    solver_time_limit_s: float = 180.0
    num_workers: int = 8
    solver_random_seed: int = 0
    relative_gap_limit: float = 0.0
    auxiliary_transfer_mode: str = "lazy"
    lazy_auxiliary_max_iterations: int = 8
    lazy_auxiliary_relaxed_stop_after_first_solution: bool = True
    lazy_auxiliary_eager_audit: bool = False
    condition_context: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def resolve_volume_scale(
        volume_scale: int = 0,
        recipe_spec: Optional[dict] = None,
        module_ops: Optional[dict] = None,
    ) -> int:
        """Return volume_scale; if 0, auto-detect from recipe + plant precision."""
        if volume_scale > 0:
            return volume_scale
        max_dec = 0
        for step in (recipe_spec or {}).get("procedure", []):
            for params in step.values():
                if isinstance(params, dict):
                    for v in params.values():
                        if isinstance(v, float):
                            s = str(v)
                            dec = len(s.split(".")[1]) if "." in s else 0
                            max_dec = max(max_dec, dec)
        for ops in (module_ops or {}).values():
            for op in ops:
                param = op[1] if len(op) > 1 else None
                if isinstance(param, float):
                    s = str(param)
                    dec = len(s.split(".")[1]) if "." in s else 0
                    max_dec = max(max_dec, dec)
        return 10 ** max_dec if max_dec > 0 else 10


@dataclass
class PlannedOperation:
    step_id: int
    recipe_node_id: str
    branch_group_id: str
    branch_id: str
    operation_type: str
    operation: str
    module: str
    source_module: str = ""
    target_module: str = ""
    out_port: str = ""
    in_port: str = ""
    connection_path: str = ""
    start_s: float = 0.0
    end_s: float = 0.0
    duration_s: float = 0.0
    connect_duration_s: float = 0.0
    transfer_duration_s: float = 0.0
    operation_cost: float = 0.0
    connection_cost: float = 0.0
    usage_cost: float = 0.0
    energy_consumption_kwh: float = 0.0
    energy_cost: float = 0.0
    co2_emissions_kg: float = 0.0
    weighted_usage_cost: float = 0.0
    weighted_energy_cost: float = 0.0
    weighted_co2_cost: float = 0.0
    total_cost: float = 0.0
    material: Dict[str, float] = field(default_factory=dict)
    transfer_kind: str = ""
    trace: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MaterialReservation:
    recipe_node_id: str
    process_module: str
    material: str
    amount_l: float


@dataclass
class PlannerResult:
    status: str
    objective_profit: float
    base_profit: float
    total_duration_s: float
    makespan_s: float
    total_cost: float
    selected_branches: Dict[str, Any]
    operations: List[PlannedOperation]
    total_usage_cost: float = 0.0
    total_energy_consumption_kwh: float = 0.0
    total_energy_cost: float = 0.0
    total_co2_emissions_kg: float = 0.0
    total_weighted_usage_cost: float = 0.0
    total_weighted_energy_cost: float = 0.0
    total_weighted_co2_cost: float = 0.0
    local_dose_reservations: List[MaterialReservation] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_rows(self) -> List[Dict[str, Any]]:
        return [
            {
                "Step": op.step_id,
                "Recipe Node": op.recipe_node_id,
                "Branch Group": op.branch_group_id,
                "Branch": op.branch_id,
                "Operation Type": op.operation_type,
                "Transfer Kind": op.transfer_kind,
                "Operation": op.operation,
                "Module": op.module,
                "Source Module": op.source_module,
                "Target Module": op.target_module,
                "OutPort": op.out_port,
                "InPort": op.in_port,
                "Connection Path": op.connection_path,
                "Start (s)": op.start_s,
                "End (s)": op.end_s,
                "Duration (s)": op.duration_s,
                "Connect Duration (s)": op.connect_duration_s,
                "Transfer Duration (s)": op.transfer_duration_s,
                "Operation Cost": op.operation_cost,
                "Connection Cost": op.connection_cost,
                "Usage Cost": op.usage_cost,
                "Energy Consumption (kWh)": op.energy_consumption_kwh,
                "Energy Cost": op.energy_cost,
                "CO2 Emissions (kg)": op.co2_emissions_kg,
                "Weighted Usage Cost": op.weighted_usage_cost,
                "Weighted Energy Cost": op.weighted_energy_cost,
                "Weighted CO2 Cost": op.weighted_co2_cost,
                "Total Cost": op.total_cost,
                "Material": op.material,
            }
            for op in self.operations
        ]


@dataclass(frozen=True)
class CostBreakdown:
    usage_cost: float = 0.0
    energy_consumption_kwh: float = 0.0
    energy_cost: float = 0.0
    co2_emissions_kg: float = 0.0
    weighted_usage_cost: float = 0.0
    weighted_energy_cost: float = 0.0
    weighted_co2_cost: float = 0.0

    @property
    def total_weighted_cost(self) -> float:
        return self.weighted_usage_cost + self.weighted_energy_cost + self.weighted_co2_cost


@dataclass(frozen=True)
class TransferCandidate:
    source_module: str
    target_module: str
    out_port: str
    in_port: str
    material: str
    amount_l: float
    duration_s: float
    connect_duration_s: float
    transfer_duration_s: float
    flow_costs: CostBreakdown
    connect_costs: CostBreakdown
    is_virtual: bool = False

    @property
    def operation_cost(self) -> float:
        return self.flow_costs.usage_cost

    @property
    def connection_cost(self) -> float:
        return self.connect_costs.usage_cost

    @property
    def total_cost(self) -> float:
        return self.flow_costs.total_weighted_cost + self.connect_costs.total_weighted_cost

    @property
    def connection_path(self) -> str:
        return f"{self.source_module}.{self.out_port} -> {self.target_module}.{self.in_port}"


@dataclass(frozen=True)
class AuxiliaryTransferCandidate:
    boundary_index: int
    material: str
    source_module: str
    target_module: str
    out_port: str
    in_port: str
    max_amount_l: float
    speed_l_s: float
    operation_usage_cost: float
    connection_usage_cost: float
    flow_energy_rate_kwh_s: float
    flow_co2_rate_kg_s: float
    connect_energy_rate_kwh_s: float
    connect_co2_rate_kg_s: float
    transfer_kind: str = PURE_MATERIAL
    composition: Tuple[Tuple[str, float], ...] = ()

    @property
    def connection_path(self) -> str:
        return f"{self.source_module}.{self.out_port} -> {self.target_module}.{self.in_port}"

    @property
    def composition_dict(self) -> Dict[str, float]:
        if self.transfer_kind == MIXTURE:
            return dict(self.composition)
        return {self.material.upper(): self.max_amount_l}

    @property
    def connection_material(self) -> Any:
        return self.composition_dict if self.transfer_kind == MIXTURE else self.material


ConnectionPair = Tuple[str, str, str, str]
ConnectionStateKey = Tuple[str, str, str, str, str]


@dataclass
class _ConnectionDemand:
    """One optional physical transfer requiring a persistent port-pair link."""

    order_key: Tuple[int, int, int]
    event_key: Tuple[Any, ...]
    demand_id: str
    source_module: str
    target_module: str
    out_port: str
    in_port: str
    material: Any
    present: Any
    transfer_start: Any
    transfer_end: Any

    @property
    def pair(self) -> ConnectionPair:
        return self.source_module, self.out_port, self.target_module, self.in_port

    @property
    def material_signature(self) -> str:
        return _connection_material_signature(self.material)

    @property
    def state_key(self) -> ConnectionStateKey:
        return (*self.pair, self.material_signature)


@dataclass
class _ConnectionAction:
    """A solver-backed Connect/Disconnect state transition for extraction."""

    action_id: str
    action_type: str
    trigger_id: str
    pair: ConnectionPair
    material_signature: str
    present: Any
    start: Any
    end: Any
    duration_s: float
    costs: CostBreakdown


CAPABILITY_BY_NODE_TYPE = {
    "dose": ("Draining", "Filling"),
    "mix": ("Stirring",),
    "usage": ("None",),
    "settling": ("Settling",),
    "heating": ("Heating",),
    "separation": ("Draining", "Filling"),
}


def solve_rtn_with_cp_sat(
    model: RTNModel,
    config: Optional[PlannerConfig] = None,
) -> PlannerResult:
    """RTN-first entry point. Plant configuration is read from
    ``model.metadata['plant']`` (preserved by ``recipe_ir_to_rtn``).

    The inner solver consumes the RTN model directly: tasks, precedence, choice
    groups, and ingredient totals are derived from RTN structure rather than a
    RecipeIR proxy.
    """
    plant = (model.metadata or {}).get("plant", {}) or {}
    module_ops = plant.get("module_ops") or {}
    if not module_ops:
        raise ValueError(
            "RTNModel.metadata['plant']['module_ops'] is required for the CP-SAT planner. "
            "Build the model via recipe_ir_to_rtn(..., module_ops=...)."
        )
    return _solve_rtn_with_cp_sat_lazy(
        model,
        module_ops=module_ops,
        module_interfaces=plant.get("module_interfaces"),
        config=config,
    )


def solve_recipe_ir_with_cp_sat(
    ir: RecipeIR,
    module_ops: Dict[str, List[Tuple[str, Any, float]]],
    module_interfaces: Optional[Dict[str, List[Tuple[str, str]]]] = None,
    module_maximum_volume: Optional[Dict[str, List[float]]] = None,
    module_resources: Optional[Dict[str, List[Any]]] = None,
    config: Optional[PlannerConfig] = None,
) -> PlannerResult:
    """Legacy entry point: lifts RecipeIR + plant config into RTNModel and
    delegates to the RTN-native pipeline.
    """
    try:
        from recipe_to_rtn import recipe_ir_to_rtn
    except ImportError:  # pragma: no cover
        from recipe_to_rtn import recipe_ir_to_rtn

    rtn = recipe_ir_to_rtn(
        ir,
        module_ops=module_ops,
        module_interfaces=module_interfaces,
        module_maximum_volume=module_maximum_volume,
        module_resources=module_resources,
    )
    return _solve_rtn_with_cp_sat_lazy(
        rtn,
        module_ops,
        module_interfaces=module_interfaces,
        config=config,
    )


def _solve_rtn_with_cp_sat_lazy(
    rtn: RTNModel,
    module_ops: Dict[str, List[Tuple[str, Any, float]]],
    module_interfaces: Optional[Dict[str, List[Tuple[str, str]]]] = None,
    config: Optional[PlannerConfig] = None,
) -> PlannerResult:
    """RTN-native lazy auxiliary-transfer pipeline (was the outer body of
    solve_recipe_ir_with_cp_sat). Operates entirely on RTNModel.
    """
    cfg = config or PlannerConfig()
    mode = str(cfg.auxiliary_transfer_mode or "lazy").lower()
    if mode not in {"lazy", "eager"}:
        raise ValueError("PlannerConfig.auxiliary_transfer_mode must be either 'lazy' or 'eager'.")
    # Cache RTN-derived solver inputs once.
    task_views: List[TaskNodeView] = [TaskNodeView(t) for t in rtn.topological_tasks()]
    recipe_inputs: Dict[str, float] = rtn.derive_recipe_inputs()
    if not cfg.enable_auxiliary_transfers or mode == "eager":
        seed = _solve_rtn_with_cp_sat_once(
            rtn,
            module_ops,
            module_interfaces=module_interfaces,
            config=cfg,
            enforce_connection_lifecycle=False,
            diagnostics_extra={
                "auxiliary_transfer_mode": "disabled" if not cfg.enable_auxiliary_transfers else "eager",
                "lazy_fallback_used": False,
                "lazy_iterations": 0,
            },
        )
        if seed.status not in {"OPTIMAL", "FEASIBLE"}:
            return seed
        lifecycle_result = _solve_rtn_with_cp_sat_once(
            rtn,
            module_ops,
            module_interfaces=module_interfaces,
            config=cfg,
            solution_hint=seed,
            fix_routing_to_hint=True,
            diagnostics_extra={
                "auxiliary_transfer_mode": "disabled" if not cfg.enable_auxiliary_transfers else "eager",
                "lazy_fallback_used": False,
                "lazy_iterations": 0,
                "connection_seeded": True,
                "connection_port_search": "all_ports_for_seeded_route",
            },
        )
        if lifecycle_result.status in {"OPTIMAL", "FEASIBLE"}:
            return _certify_or_invalidate_result(rtn, lifecycle_result, cfg)
        exact_result = _solve_rtn_with_cp_sat_once(
            rtn,
            module_ops,
            module_interfaces=module_interfaces,
            config=cfg,
            solution_hint=seed,
            fix_routing_to_hint=True,
            exact_physical_hint=True,
            diagnostics_extra={
                "auxiliary_transfer_mode": "disabled" if not cfg.enable_auxiliary_transfers else "eager",
                "lazy_fallback_used": False,
                "lazy_iterations": 0,
                "connection_seeded": True,
                "connection_port_search": "exact_seed_fallback",
                "connection_flexible_status": lifecycle_result.status,
            },
        )
        return _certify_or_invalidate_result(rtn, exact_result, cfg)

    lazy_diagnostics: Dict[str, Any] = {
        "auxiliary_transfer_mode": "lazy",
        "lazy_fallback_used": False,
        "lazy_iterations": 0,
        "lazy_rounds": [],
    }
    aux_candidates: List[AuxiliaryTransferCandidate] = []
    aux_keys: set[Tuple[int, str, str, str, str, str]] = set()
    max_iterations = max(1, int(cfg.lazy_auxiliary_max_iterations))
    fallback_reason = ""

    for iteration in range(1, max_iterations + 1):
        lazy_diagnostics["lazy_iterations"] = iteration
        relaxed = _solve_rtn_with_cp_sat_once(
            rtn,
            module_ops,
            module_interfaces=module_interfaces,
            config=cfg,
            aux_candidates_override=aux_candidates,
            relaxed_auxiliary_inventory=True,
            diagnostics_extra={
                **lazy_diagnostics,
                "lazy_candidate_count": len(aux_candidates),
                "lazy_certification": "relaxed",
            },
        )
        if relaxed.status not in {"OPTIMAL", "FEASIBLE"}:
            fallback_reason = f"relaxed solve returned {relaxed.status}"
            break

        violations = _find_auxiliary_transfer_violations(
            relaxed,
            task_views,
            recipe_inputs,
            rtn,
        )
        new_candidates = _targeted_auxiliary_transfer_candidates(
            violations,
            module_ops,
            module_interfaces or {},
            rtn,
            cfg,
        )
        added = 0
        for aux in new_candidates:
            key = _auxiliary_candidate_key(aux)
            if key in aux_keys:
                continue
            aux_keys.add(key)
            aux_candidates.append(aux)
            added += 1

        lazy_diagnostics["lazy_rounds"].append(
            {
                "iteration": iteration,
                "relaxed_status": relaxed.status,
                "violation_count": len(violations),
                "new_candidate_count": added,
                "candidate_count": len(aux_candidates),
                "violations": violations[:10],
            }
        )

        if violations and added and iteration < max_iterations:
            continue

        certification_seed = _solve_rtn_with_cp_sat_once(
            rtn,
            module_ops,
            module_interfaces=module_interfaces,
            config=cfg,
            aux_candidates_override=aux_candidates,
            relaxed_auxiliary_inventory=False,
            enforce_connection_lifecycle=False,
            diagnostics_extra={
                **lazy_diagnostics,
                "lazy_candidate_count": len(aux_candidates),
                "lazy_certification": "hard_inventory_seed",
            },
        )
        certified = certification_seed
        if certification_seed.status in {"OPTIMAL", "FEASIBLE"}:
            certified = _solve_rtn_with_cp_sat_once(
                rtn,
                module_ops,
                module_interfaces=module_interfaces,
                config=cfg,
                aux_candidates_override=aux_candidates,
                relaxed_auxiliary_inventory=False,
                solution_hint=certification_seed,
                fix_routing_to_hint=True,
                diagnostics_extra={
                    **lazy_diagnostics,
                    "lazy_candidate_count": len(aux_candidates),
                    "lazy_certification": "hard_inventory",
                    "connection_seeded": True,
                    "connection_port_search": "all_ports_for_seeded_route",
                },
            )
            if (
                certified.status not in {"OPTIMAL", "FEASIBLE"}
                or not _replay_validates_result(rtn, certified, cfg)
            ):
                flexible_status = certified.status
                certified = _solve_rtn_with_cp_sat_once(
                    rtn,
                    module_ops,
                    module_interfaces=module_interfaces,
                    config=cfg,
                    aux_candidates_override=aux_candidates,
                    relaxed_auxiliary_inventory=False,
                    solution_hint=certification_seed,
                    fix_routing_to_hint=True,
                    exact_physical_hint=True,
                    diagnostics_extra={
                        **lazy_diagnostics,
                        "lazy_candidate_count": len(aux_candidates),
                        "lazy_certification": "hard_inventory",
                        "connection_seeded": True,
                        "connection_port_search": "exact_seed_fallback",
                        "connection_flexible_status": flexible_status,
                    },
                )
        if certified.status in {"OPTIMAL", "FEASIBLE"} and _replay_validates_result(
            rtn,
            certified,
            cfg,
        ):
            if not cfg.lazy_auxiliary_eager_audit:
                certified.diagnostics.update(lazy_diagnostics)
                certified.diagnostics.update(
                    {
                        "lazy_candidate_count": len(aux_candidates),
                        "lazy_certification": "hard_inventory",
                        "lazy_fallback_used": False,
                        "lazy_eager_audit_enabled": False,
                    }
                )
                return _certify_or_invalidate_result(rtn, certified, cfg)
            eager_audit = _solve_rtn_with_cp_sat_once(
                rtn,
                module_ops,
                module_interfaces=module_interfaces,
                config=cfg,
                diagnostics_extra={
                    **lazy_diagnostics,
                    "auxiliary_transfer_mode": "lazy",
                    "lazy_candidate_count": len(aux_candidates),
                    "lazy_certification": "eager_optimality_audit",
                },
            )
            if (
                eager_audit.status in {"OPTIMAL", "FEASIBLE"}
                and eager_audit.objective_profit > certified.objective_profit + 1e-6
            ):
                eager_audit.diagnostics.update(
                    {
                        **lazy_diagnostics,
                        "auxiliary_transfer_mode": "lazy",
                        "lazy_fallback_used": True,
                        "lazy_fallback_reason": "eager optimality audit improved objective",
                        "lazy_candidate_count": len(aux_candidates),
                        "lazy_eager_audit_enabled": True,
                        "eager_auxiliary_candidate_count": eager_audit.diagnostics.get("auxiliary_candidate_count"),
                    }
                )
                return _certify_or_invalidate_result(rtn, eager_audit, cfg)
            certified.diagnostics.update(lazy_diagnostics)
            certified.diagnostics.update(
                {
                    "lazy_candidate_count": len(aux_candidates),
                    "lazy_certification": "hard_inventory",
                    "lazy_fallback_used": False,
                    "lazy_eager_audit_enabled": True,
                    "eager_auxiliary_candidate_count": eager_audit.diagnostics.get("auxiliary_candidate_count"),
                }
            )
            return _certify_or_invalidate_result(rtn, certified, cfg)

        fallback_reason = (
            "hard inventory certification failed"
            if certified.status in {"OPTIMAL", "FEASIBLE"}
            else f"hard inventory solve returned {certified.status}"
        )
        break

    if not fallback_reason:
        fallback_reason = f"lazy iteration limit {max_iterations} reached"
    eager_seed = _solve_rtn_with_cp_sat_once(
        rtn,
        module_ops,
        module_interfaces=module_interfaces,
        config=cfg,
        enforce_connection_lifecycle=False,
        diagnostics_extra={
            **lazy_diagnostics,
            "auxiliary_transfer_mode": "lazy",
            "lazy_fallback_used": True,
            "lazy_fallback_reason": fallback_reason,
            "lazy_candidate_count": len(aux_candidates),
            "lazy_certification": "eager_inventory_seed",
        },
    )
    eager = eager_seed
    if eager_seed.status in {"OPTIMAL", "FEASIBLE"}:
        eager = _solve_rtn_with_cp_sat_once(
            rtn,
            module_ops,
            module_interfaces=module_interfaces,
            config=cfg,
            solution_hint=eager_seed,
            fix_routing_to_hint=True,
            diagnostics_extra={
                **lazy_diagnostics,
                "auxiliary_transfer_mode": "lazy",
                "lazy_fallback_used": True,
                "lazy_fallback_reason": fallback_reason,
                "lazy_candidate_count": len(aux_candidates),
                "lazy_certification": "eager_hard_inventory",
                "connection_seeded": True,
                "connection_port_search": "all_ports_for_seeded_route",
            },
        )
        if (
            eager.status not in {"OPTIMAL", "FEASIBLE"}
            or not _replay_validates_result(rtn, eager, cfg)
        ):
            flexible_status = eager.status
            eager = _solve_rtn_with_cp_sat_once(
                rtn,
                module_ops,
                module_interfaces=module_interfaces,
                config=cfg,
                solution_hint=eager_seed,
                fix_routing_to_hint=True,
                exact_physical_hint=True,
                diagnostics_extra={
                    **lazy_diagnostics,
                    "auxiliary_transfer_mode": "lazy",
                    "lazy_fallback_used": True,
                    "lazy_fallback_reason": fallback_reason,
                    "lazy_candidate_count": len(aux_candidates),
                    "lazy_certification": "eager_hard_inventory",
                    "connection_seeded": True,
                    "connection_port_search": "exact_seed_fallback",
                    "connection_flexible_status": flexible_status,
                },
            )
    eager.diagnostics.update(
        {
            **lazy_diagnostics,
            "auxiliary_transfer_mode": "lazy",
            "lazy_fallback_used": True,
            "lazy_fallback_reason": fallback_reason,
            "lazy_candidate_count": len(aux_candidates),
            "eager_auxiliary_candidate_count": eager.diagnostics.get("auxiliary_candidate_count"),
        }
    )
    return _certify_or_invalidate_result(rtn, eager, cfg)


def _infeasible_result(cfg: PlannerConfig, reason: str, **extra) -> PlannerResult:
    """Return a structured INFEASIBLE PlannerResult with diagnostics."""
    return PlannerResult(
        status="INFEASIBLE",
        objective_profit=float("-inf"),
        base_profit=cfg.base_profit,
        total_duration_s=0.0,
        makespan_s=0.0,
        total_cost=0.0,
        selected_branches={},
        operations=[],
        diagnostics={"infeasible_reason": reason, **extra},
    )


def _solve_rtn_with_cp_sat_once(
    rtn: RTNModel,
    module_ops: Dict[str, List[Tuple[str, Any, float]]],
    module_interfaces: Optional[Dict[str, List[Tuple[str, str]]]] = None,
    config: Optional[PlannerConfig] = None,
    aux_candidates_override: Optional[List[AuxiliaryTransferCandidate]] = None,
    relaxed_auxiliary_inventory: bool = False,
    enforce_connection_lifecycle: bool = True,
    solution_hint: Optional[PlannerResult] = None,
    fix_routing_to_hint: bool = False,
    exact_physical_hint: bool = False,
    diagnostics_extra: Optional[Dict[str, Any]] = None,
) -> PlannerResult:
    try:
        from ortools.sat.python import cp_model
    except ImportError as exc:  # pragma: no cover - depends on user env
        raise ImportError(
            "OR-Tools is required for CP-SAT planning. Install it in the Python environment used by this notebook/script."
        ) from exc

    cfg = config or PlannerConfig()
    tasks: List[TaskNodeView] = [TaskNodeView(t) for t in rtn.topological_tasks()]
    if cfg.volume_scale <= 0:
        cfg = replace(cfg, volume_scale=_auto_volume_scale(tasks, module_ops))
    if not tasks:
        return _infeasible_result(cfg, "RTNModel contains no executable tasks")
    recipe_inputs: Dict[str, float] = rtn.derive_recipe_inputs()

    # Horizon precedence: explicit config > explicit model value (non-default) >
    # solver estimate. The RTNModel default (86_400 s) is treated as "unspecified"
    # because it is almost always too loose for practical variable domains.
    _model_horizon = rtn.horizon_s if rtn.horizon_s and rtn.horizon_s != 86_400 else 0
    horizon = (
        cfg.max_horizon_s
        or _model_horizon
        or _estimate_horizon(
            tasks, module_ops, rtn, cfg
        )
    )
    model = cp_model.CpModel()
    horizon_i = int(horizon * cfg.time_scale)
    cost_scale = int(cfg.cost_scale)

    active: Dict[str, Any] = {}
    starts: Dict[str, Any] = {}
    ends: Dict[str, Any] = {}
    duration_vars: Dict[str, Any] = {}
    assign: Dict[Tuple[str, str], Any] = {}
    route_select: Dict[Tuple[str, int], Any] = {}
    route_candidates: Dict[str, List[TransferCandidate]] = {}
    is_local_dose: Dict[Tuple[str, str], Any] = {}
    aux_candidates: List[AuxiliaryTransferCandidate] = []
    aux_amount: Dict[int, Any] = {}
    aux_present: Dict[int, Any] = {}
    aux_starts: Dict[int, Any] = {}
    aux_ends: Dict[int, Any] = {}
    aux_durations: Dict[int, Any] = {}
    aux_cost_vars: Dict[int, Any] = {}
    connection_demands: List[_ConnectionDemand] = []
    # A Module can accept simultaneous Filling operations through distinct input
    # ports, but may perform only one exclusive/local or outgoing Draining
    # operation at a time. Incoming intervals are therefore kept separately:
    # they may overlap each other, but never an exclusive interval.
    incoming_intervals_by_module: Dict[str, List[Any]] = {wb: [] for wb in module_ops}
    intervals_by_module: Dict[str, List[Any]] = {wb: [] for wb in module_ops}
    assigned_costs: Dict[Tuple[str, str], int] = {}
    assigned_cost_breakdowns: Dict[Tuple[str, str], CostBreakdown] = {}
    chosen_resource_cost: Dict[str, Any] = {}
    task_position = {node.id: index for index, node in enumerate(tasks)}

    choice_groups = dict(rtn.choice_groups)
    selected_branch_vars: Dict[Tuple[str, str], Any] = {}
    for group, branches in choice_groups.items():
        if group.startswith(("XOR_", "OR_")):
            group_vars = []
            for branch in branches:
                var = model.NewBoolVar(f"branch_selected_{group}_{branch}")
                selected_branch_vars[(group, branch)] = var
                group_vars.append(var)
            if group.startswith("XOR_"):
                model.AddExactlyOne(group_vars)
            else:
                policy = rtn.choice_group_policies.get(group, {})
                minimum = int(policy.get("minBranches", 1))
                maximum = int(policy.get("maxBranches", len(branches)))
                model.Add(sum(group_vars) >= minimum)
                model.Add(sum(group_vars) <= maximum)
                for branch, condition in policy.get("branchConditions", {}).items():
                    if (group, branch) not in selected_branch_vars:
                        continue
                    enabled = _evaluate_planning_condition(condition, cfg.condition_context)
                    if enabled is False:
                        model.Add(selected_branch_vars[(group, branch)] == 0)

    for node in tasks:
        active[node.id] = _node_active_var(model, node, selected_branch_vars)
        starts[node.id] = model.NewIntVar(0, horizon_i, f"start_{node.id}")
        duration_vars[node.id] = model.NewIntVar(0, horizon_i, f"duration_{node.id}")
        ends[node.id] = model.NewIntVar(0, horizon_i, f"end_{node.id}")
        model.Add(ends[node.id] == starts[node.id] + duration_vars[node.id]).OnlyEnforceIf(active[node.id])
        model.Add(starts[node.id] == 0).OnlyEnforceIf(active[node.id].Not())
        model.Add(ends[node.id] == 0).OnlyEnforceIf(active[node.id].Not())
        model.Add(duration_vars[node.id] == 0).OnlyEnforceIf(active[node.id].Not())

        if node.node_type in {"dose", "separation"}:
            routes = _enumerate_transfer_candidates(
                node,
                module_ops,
                module_interfaces or {},
                rtn,
                cfg,
                allow_dynamic_inventory=cfg.enable_auxiliary_transfers,
            )
            if relaxed_auxiliary_inventory or not enforce_connection_lifecycle:
                routes = _canonical_transfer_candidates(routes)
            route_candidates[node.id] = routes
            if not routes:
                return _infeasible_result(
                    cfg,
                    f"No physical transfer route found for node {node.id} ({node.node_type})",
                    node_id=node.id,
                    node_type=node.node_type,
                )
            # ── Unified Flow Model for doses ──
            # Each route gets a flow_amount (0..Q). Self-transfer (source==target)
            # is zero-duration/cost. At most 1 physical + 1 self-transfer route.
            if node.node_type == "dose" and cfg.enable_auxiliary_transfers:
                _amt_raw = float((node.params or {}).get("amount_L", 0.0))
                _amt_i = max(1, int(round(_amt_raw * cfg.volume_scale)))
                _flow_terms = []
                _cost_terms = []
                _max_cost = 0
                _phys_used: List[Any] = []
                _self_used: List[Any] = []
                for _idx, _route in enumerate(routes):
                    _is_self = (_route.source_module == _route.target_module)
                    _fv = model.NewIntVar(0, _amt_i, f"flow_{node.id}_{_idx}")
                    _used = model.NewBoolVar(f"use_{node.id}_{_idx}")
                    model.Add(_fv >= _used)
                    model.Add(_fv <= _amt_i * _used)
                    _flow_terms.append(_fv)
                    route_select[(node.id, _idx)] = _used
                    if _is_self:
                        _self_used.append(_used)
                        model.AddImplication(_used, active[node.id])
                        model.Add(duration_vars[node.id] == 0).OnlyEnforceIf(_used)
                    else:
                        _phys_used.append(_used)
                        _flow_i = max(0, int(round(_route.transfer_duration_s * cfg.time_scale)))
                        model.AddImplication(_used, active[node.id])
                        model.Add(duration_vars[node.id] == _flow_i).OnlyEnforceIf(_used)
                        _si = model.NewOptionalIntervalVar(starts[node.id], _flow_i, ends[node.id], _used, f"si_{node.id}_{_idx}")
                        _di = model.NewOptionalIntervalVar(starts[node.id], _flow_i, ends[node.id], _used, f"di_{node.id}_{_idx}")
                        # Draining is exclusive on the source. Multiple source
                        # Module may Fill one target concurrently through
                        # distinct target ports.
                        intervals_by_module.setdefault(_route.source_module, []).append(_si)
                        incoming_intervals_by_module.setdefault(_route.target_module, []).append(_di)
                        intervals_by_module.setdefault(f"PORT:{_route.source_module}.{_route.out_port}", []).append(_si)
                        intervals_by_module.setdefault(f"PORT:{_route.target_module}.{_route.in_port}", []).append(_di)
                        connection_demands.append(_ConnectionDemand(
                            order_key=(2 * task_position[node.id] + 1, _idx, 0),
                            event_key=("recipe", node.id),
                            demand_id=node.id,
                            source_module=_route.source_module,
                            target_module=_route.target_module,
                            out_port=_route.out_port,
                            in_port=_route.in_port,
                            material=_route.material,
                            present=_used,
                            transfer_start=starts[node.id],
                            transfer_end=ends[node.id],
                        ))
                    _rcost_i = int(round(_route.flow_costs.total_weighted_cost * cost_scale))
                    _max_cost = max(_max_cost, _rcost_i)
                    _cost_terms.append(_used * _rcost_i)
                # At most one physical route (port interval model requires fixed duration)
                if _phys_used:
                    model.Add(sum(_phys_used) <= 1)
                if _self_used:
                    model.Add(sum(_self_used) <= 1)
                _dtarget = model.NewIntVar(0, _amt_i, f"dtarget_{node.id}")
                model.Add(_dtarget == _amt_i).OnlyEnforceIf(active[node.id])
                model.Add(_dtarget == 0).OnlyEnforceIf(active[node.id].Not())
                model.Add(sum(_flow_terms) == _dtarget)
                cvar = model.NewIntVar(0, max(1, _max_cost), f"chosen_cost_{node.id}")
                model.Add(cvar == sum(_cost_terms))
                chosen_resource_cost[node.id] = cvar
                continue

            # ── Separation / non-dose nodes: keep original route_select ──
            rvars = []
            max_node_cost = 0
            for idx, route in enumerate(routes):
                rvar = model.NewBoolVar(f"route_{node.id}_{idx}_{route.source_module}_{route.out_port}_to_{route.target_module}_{route.in_port}")
                route_select[(node.id, idx)] = rvar
                rvars.append(rvar)
                if route.source_module == route.target_module:
                    model.AddImplication(rvar, active[node.id])
                    model.Add(duration_vars[node.id] == 0).OnlyEnforceIf(rvar)
                    continue
                flow_duration = max(0, int(round(route.transfer_duration_s * cfg.time_scale)))
                route_cost = int(round(route.flow_costs.total_weighted_cost * cost_scale))
                max_node_cost = max(max_node_cost, route_cost)
                model.AddImplication(rvar, active[node.id])
                model.Add(duration_vars[node.id] == flow_duration).OnlyEnforceIf(rvar)
                src_interval = model.NewOptionalIntervalVar(
                    starts[node.id], flow_duration, ends[node.id], rvar,
                    f"interval_{node.id}_{idx}_{route.source_module}_{route.out_port}",
                )
                dst_interval = model.NewOptionalIntervalVar(
                    starts[node.id], flow_duration, ends[node.id], rvar,
                    f"interval_{node.id}_{idx}_{route.target_module}_{route.in_port}",
                )
                # Draining is source-exclusive; target Filling may overlap
                # other incoming Filling intervals on distinct input ports.
                intervals_by_module.setdefault(route.source_module, []).append(src_interval)
                incoming_intervals_by_module.setdefault(route.target_module, []).append(dst_interval)
                intervals_by_module.setdefault(f"PORT:{route.source_module}.{route.out_port}", []).append(src_interval)
                intervals_by_module.setdefault(f"PORT:{route.target_module}.{route.in_port}", []).append(dst_interval)
                connection_demands.append(_ConnectionDemand(
                    order_key=(2 * task_position[node.id] + 1, idx, 0),
                    event_key=("recipe", node.id),
                    demand_id=node.id,
                    source_module=route.source_module,
                    target_module=route.target_module,
                    out_port=route.out_port,
                    in_port=route.in_port,
                    material=route.material,
                    present=rvar,
                    transfer_start=starts[node.id],
                    transfer_end=ends[node.id],
                ))
            model.Add(sum(rvars) == active[node.id])
            cvar = model.NewIntVar(0, max_node_cost, f"chosen_cost_{node.id}")
            model.Add(cvar == sum(route_select[(node.id, idx)] * int(round(route.flow_costs.total_weighted_cost * cost_scale)) for idx, route in enumerate(routes)))
            chosen_resource_cost[node.id] = cvar
            continue

        feasible_module = _feasible_module_for_node(node, module_ops, rtn, nominal_volume=sum(recipe_inputs.values()))
        if not feasible_module:
            return _infeasible_result(
                cfg,
                f"No feasible Module found for node {node.id} ({node.node_type})",
                node_id=node.id,
                node_type=node.node_type,
            )

        assign_vars = []
        fixed_duration = max(0, int(round(_duration_for_node(node, module_ops) * cfg.time_scale)))
        fixed_duration_s = fixed_duration / cfg.time_scale
        for wb in feasible_module:
            avar = model.NewBoolVar(f"assign_{node.id}_{wb}")
            assign[(node.id, wb)] = avar
            assign_vars.append(avar)
            interval = model.NewOptionalIntervalVar(
                starts[node.id],
                fixed_duration,
                ends[node.id],
                avar,
                f"interval_{node.id}_{wb}",
            )
            intervals_by_module.setdefault(wb, []).append(interval)
            breakdown = _operation_costs(node, wb, module_ops, fixed_duration_s, cfg)
            assigned_cost_breakdowns[(node.id, wb)] = breakdown
            assigned_costs[(node.id, wb)] = int(round(breakdown.total_weighted_cost * cost_scale))
            model.AddImplication(avar, active[node.id])
        model.Add(sum(assign_vars) == active[node.id])
        model.Add(duration_vars[node.id] == fixed_duration).OnlyEnforceIf(active[node.id])

        max_node_cost = max(assigned_costs[(node.id, wb)] for wb in feasible_module)
        cvar = model.NewIntVar(0, max_node_cost, f"chosen_cost_{node.id}")
        model.Add(cvar == sum(assign[(node.id, wb)] * assigned_costs[(node.id, wb)] for wb in feasible_module))
        chosen_resource_cost[node.id] = cvar

    _link_transfer_targets_to_process_modules(
        tasks,
        model,
        assign,
        route_select,
        route_candidates,
        precedence=list(rtn.precedence),
        allow_process_transfers=cfg.allow_process_transfers,
    )

    # Only the deliberately relaxed discovery probe disables self-routes.
    # A hard-inventory seed without the connection automaton must retain them:
    # otherwise material already present in the selected process Module is
    # needlessly evacuated in full and then physically dosed back.
    if relaxed_auxiliary_inventory:
        for node_id, routes in route_candidates.items():
            for idx, route in enumerate(routes):
                if route.source_module == route.target_module:
                    model.Add(route_select[(node_id, idx)] == 0)

    if cfg.enable_auxiliary_transfers:
        if aux_candidates_override is None:
            aux_candidates = _enumerate_auxiliary_transfer_candidates(
                tasks,
                recipe_inputs,
                module_ops,
                module_interfaces or {},
                rtn,
                cfg,
            )
        else:
            aux_candidates = list(aux_candidates_override)
        if relaxed_auxiliary_inventory or not enforce_connection_lifecycle:
            aux_candidates = _canonical_auxiliary_candidates(aux_candidates)
        for idx, aux in enumerate(aux_candidates):
            resolved_candidate_composition = normalize_composition(aux.composition_dict)
            max_amount_i = max(
                0,
                int(round(sum(resolved_candidate_composition.values()) * cfg.volume_scale))
                if aux.transfer_kind == MIXTURE
                else int(round(aux.max_amount_l * cfg.volume_scale)),
            )
            if max_amount_i <= 0:
                continue
            present = model.NewBoolVar(
                f"aux_present_{idx}_{aux.boundary_index}_{aux.material}_{aux.source_module}_{aux.out_port}_to_{aux.target_module}_{aux.in_port}"
            )
            amount = model.NewIntVar(0, max_amount_i, f"aux_amount_{idx}")
            start = model.NewIntVar(0, horizon_i, f"aux_start_{idx}")
            end = model.NewIntVar(0, horizon_i, f"aux_end_{idx}")
            flow_duration = model.NewIntVar(0, horizon_i, f"aux_flow_duration_{idx}")
            duration = model.NewIntVar(0, horizon_i, f"aux_duration_{idx}")
            aux_present[idx] = present
            aux_amount[idx] = amount
            aux_starts[idx] = start
            aux_ends[idx] = end
            aux_durations[idx] = duration

            if aux.transfer_kind == MIXTURE:
                # A mixture candidate is one concrete proportional vector.  Lazy
                # discovery can add a smaller vector when only part of a batch
                # must move; keeping the vector fixed makes its exact connection
                # signature available to the persistent-link automaton.
                model.Add(amount == max_amount_i * present)
            else:
                model.Add(amount <= max_amount_i * present)
                model.Add(amount >= present)
            denom = max(1, int(round(cfg.volume_scale * aux.speed_l_s)))
            model.Add(flow_duration * denom >= amount * cfg.time_scale)
            model.Add(duration == flow_duration)
            model.Add(end == start + duration).OnlyEnforceIf(present)
            model.Add(start == 0).OnlyEnforceIf(present.Not())
            model.Add(end == 0).OnlyEnforceIf(present.Not())
            model.Add(duration == 0).OnlyEnforceIf(present.Not())
            interval = model.NewOptionalIntervalVar(start, duration, end, present, f"interval_aux_{idx}")
            intervals_by_module.setdefault(aux.source_module, []).append(interval)
            if aux.target_module != aux.source_module:
                incoming_intervals_by_module.setdefault(aux.target_module, []).append(interval)
            intervals_by_module.setdefault(f"PORT:{aux.source_module}.{aux.out_port}", []).append(interval)
            intervals_by_module.setdefault(f"PORT:{aux.target_module}.{aux.in_port}", []).append(interval)
            connection_demands.append(_ConnectionDemand(
                order_key=(2 * aux.boundary_index, idx, 1),
                event_key=(
                    "aux",
                    aux.boundary_index,
                    composition_signature(resolved_candidate_composition),
                    aux.source_module,
                    aux.target_module,
                ),
                demand_id=f"AUX_{idx:03d}",
                source_module=aux.source_module,
                target_module=aux.target_module,
                out_port=aux.out_port,
                in_port=aux.in_port,
                material=aux.connection_material,
                present=present,
                transfer_start=start,
                transfer_end=end,
            ))

            for task_before in tasks[: aux.boundary_index]:
                model.Add(start >= ends[task_before.id]).OnlyEnforceIf([present, active[task_before.id]])
            for task_after in tasks[aux.boundary_index :]:
                model.Add(end <= starts[task_after.id]).OnlyEnforceIf([present, active[task_after.id]])

            fixed_usage_weighted = aux.operation_usage_cost * cfg.usage_cost_weight
            fixed_cost_i = int(round(fixed_usage_weighted * cost_scale))
            flow_cost_per_second = (
                aux.flow_energy_rate_kwh_s * cfg.electricity_price_eur_per_kwh * cfg.energy_cost_weight
                + aux.flow_co2_rate_kg_s * cfg.co2_penalty
            )
            flow_cost_per_tick_i = int(round(flow_cost_per_second * cost_scale / cfg.time_scale))
            max_cost_i = fixed_cost_i + max(0, flow_cost_per_tick_i) * horizon_i + 1
            cost_var = model.NewIntVar(0, max_cost_i, f"aux_cost_{idx}")
            model.Add(cost_var == fixed_cost_i * present + flow_cost_per_tick_i * duration)
            aux_cost_vars[idx] = cost_var

        if not relaxed_auxiliary_inventory:
            _enforce_inventory_flow_with_auxiliary_transfers(
                model,
                tasks,
                recipe_inputs,
                assign,
                route_select,
                route_candidates,
                aux_candidates,
                aux_amount,
                aux_present,
                active,
                rtn,
                cfg,
            )
    else:
        _enforce_transfer_inventory_capacity(
            model,
            route_select,
            route_candidates,
            rtn,
            cfg,
        )

    if relaxed_auxiliary_inventory:
        # Lazy discovery solves are temporary inventory probes. Keeping the
        # full connection automaton out of these discarded intermediate models
        # preserves the solver budget; the certified solve below always uses it.
        connection_actions: List[_ConnectionAction] = []
        connection_action_cost_terms: List[Any] = []
        connection_action_ends: List[Any] = []
        connection_action_intervals: List[Any] = []
    else:
        connection_actions, connection_action_cost_terms, connection_action_ends, connection_action_intervals = (
            _add_connection_lifecycle_model(
                model,
                connection_demands,
                module_ops,
                cfg,
                horizon_i,
                cost_scale,
                intervals_by_module,
            )
        )

    input_port_capacity = {
        wb: max(1, sum(1 for raw in (module_interfaces or {}).get(wb, []) if str(raw[0]).casefold() == "input"))
        for wb in module_ops
    }
    for resource, exclusive_intervals in intervals_by_module.items():
        if resource.startswith("PORT:"):
            if exclusive_intervals:
                model.AddNoOverlap(exclusive_intervals)
            continue
        incoming_intervals = incoming_intervals_by_module.get(resource, [])
        if incoming_intervals:
            capacity = input_port_capacity.get(resource, 1)
            model.AddCumulative(
                [*exclusive_intervals, *incoming_intervals],
                [*[capacity] * len(exclusive_intervals), *[1] * len(incoming_intervals)],
                capacity,
            )
        elif exclusive_intervals:
            model.AddNoOverlap(exclusive_intervals)
    if connection_action_intervals:
        model.AddNoOverlap(connection_action_intervals)

    for src, dst in rtn.precedence:
        if src not in active or dst not in active:
            continue
        model.Add(starts[dst] >= ends[src]).OnlyEnforceIf([active[src], active[dst]])

    makespan = model.NewIntVar(0, horizon_i, "makespan")
    for node in tasks:
        model.Add(makespan >= ends[node.id]).OnlyEnforceIf(active[node.id])
    for idx, end in aux_ends.items():
        model.Add(makespan >= end).OnlyEnforceIf(aux_present[idx])
    for action_end in connection_action_ends:
        model.Add(makespan >= action_end)

    total_cost_i = model.NewIntVar(0, 10**12, "total_cost_i")
    model.Add(
        total_cost_i
        == sum(chosen_resource_cost[node.id] for node in tasks)
        + sum(aux_cost_vars.values())
        + sum(connection_action_cost_terms)
    )

    lambda_i = int(round(cfg.lambda_per_second * cost_scale / cfg.time_scale))
    base_profit_i = int(round(cfg.base_profit * cost_scale))
    profit_expr = base_profit_i + lambda_i * makespan - total_cost_i
    model.Maximize(profit_expr)

    if solution_hint is not None:
        _apply_solution_hints(
            model,
            solution_hint,
            tasks,
            active,
            starts,
            ends,
            duration_vars,
            assign,
            route_select,
            route_candidates,
            aux_candidates,
            aux_present,
            aux_starts,
            aux_ends,
            selected_branch_vars,
            cfg,
            fix_routing_to_hint,
            exact_physical_hint,
        )

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(cfg.solver_time_limit_s)
    solver.parameters.num_search_workers = int(cfg.num_workers)
    solver.parameters.random_seed = int(cfg.solver_random_seed)
    if cfg.relative_gap_limit and cfg.relative_gap_limit > 0:
        solver.parameters.relative_gap_limit = float(cfg.relative_gap_limit)
    if relaxed_auxiliary_inventory and cfg.lazy_auxiliary_relaxed_stop_after_first_solution:
        solver.parameters.stop_after_first_solution = True
    status_code = solver.Solve(model)
    status = solver.StatusName(status_code)
    if status not in {"OPTIMAL", "FEASIBLE"}:
        diagnostics = {
            "solver_status": status,
            "solver_wall_time_s": solver.WallTime(),
            "relative_gap_limit": cfg.relative_gap_limit,
            "solver_random_seed": cfg.solver_random_seed,
            "relaxed_auxiliary_inventory": relaxed_auxiliary_inventory,
            "stop_after_first_solution": bool(
                relaxed_auxiliary_inventory and cfg.lazy_auxiliary_relaxed_stop_after_first_solution
            ),
        }
        diagnostics.update(diagnostics_extra or {})
        return PlannerResult(
            status=status,
            objective_profit=float("-inf"),
            base_profit=cfg.base_profit,
            total_duration_s=0.0,
            makespan_s=0.0,
            total_cost=0.0,
            selected_branches={},
            operations=[],
            diagnostics=diagnostics,
        )

    selected_branches: Dict[str, Any] = {}
    for group, branches in choice_groups.items():
        chosen = [
            branch for branch in branches
            if (group, branch) in selected_branch_vars
            and solver.BooleanValue(selected_branch_vars[(group, branch)])
        ]
        if group.startswith("XOR_"):
            if chosen:
                selected_branches[group] = chosen[0]
        elif group.startswith("OR_"):
            selected_branches[group] = chosen

    operations: List[PlannedOperation] = []
    for action in connection_actions:
        if not solver.BooleanValue(action.present):
            continue
        source_module, out_port, target_module, in_port = action.pair
        duration_s = action.duration_s
        action_name = "Connect" if action.action_type == "connect" else "Disconnect"
        operations.append(
            PlannedOperation(
                step_id=0,
                recipe_node_id=action.action_id,
                branch_group_id="",
                branch_id="",
                operation_type=action.action_type,
                operation=f"{action_name}({out_port} -> {in_port}, {duration_s:.1f}s)",
                module=f"{source_module}->{target_module}",
                source_module=source_module,
                target_module=target_module,
                out_port=out_port,
                in_port=in_port,
                connection_path=f"{source_module}.{out_port} -> {target_module}.{in_port}",
                start_s=solver.Value(action.start) / cfg.time_scale,
                end_s=solver.Value(action.end) / cfg.time_scale,
                duration_s=duration_s,
                connect_duration_s=duration_s if action.action_type == "connect" else 0.0,
                transfer_duration_s=0.0,
                operation_cost=0.0,
                connection_cost=action.costs.usage_cost,
                usage_cost=action.costs.usage_cost,
                energy_consumption_kwh=action.costs.energy_consumption_kwh,
                energy_cost=action.costs.energy_cost,
                co2_emissions_kg=action.costs.co2_emissions_kg,
                weighted_usage_cost=action.costs.weighted_usage_cost,
                weighted_energy_cost=action.costs.weighted_energy_cost,
                weighted_co2_cost=action.costs.weighted_co2_cost,
                total_cost=action.costs.total_weighted_cost,
                material={},
                trace={
                    "active": True,
                    "is_recipe_step": False,
                    "connection_lifecycle": True,
                    "connection_action": action.action_type,
                    "material_signature": action.material_signature,
                    "trigger": action.trigger_id,
                },
            )
        )

    for idx, aux in enumerate(aux_candidates):
        if idx not in aux_present or not solver.BooleanValue(aux_present[idx]):
            continue
        amount_l = solver.Value(aux_amount[idx]) / cfg.volume_scale
        duration_s = solver.Value(aux_durations[idx]) / cfg.time_scale
        transfer_duration_s = duration_s
        resolved_composition = (
            normalize_composition(aux.composition_dict)
            if aux.transfer_kind == MIXTURE
            else {aux.material.upper(): amount_l}
        )
        material_text = ", ".join(
            f"{material}: {quantity:.3f} litre"
            for material, quantity in resolved_composition.items()
        )
        aux_flow_costs = _cost_breakdown(
            aux.operation_usage_cost,
            aux.flow_energy_rate_kwh_s,
            aux.flow_co2_rate_kg_s,
            transfer_duration_s,
            cfg,
        )
        operations.append(
            PlannedOperation(
                step_id=0,
                recipe_node_id=f"AUX_{idx:03d}",
                branch_group_id="",
                branch_id="",
                operation_type="aux_transfer",
                operation=(
                    f"{'Mixture' if aux.transfer_kind == MIXTURE else 'Pure material'} Transfer: "
                    f"Draining({aux.source_module}), Filling({aux.target_module}), ({material_text})"
                ),
                module=aux.target_module,
                source_module=aux.source_module,
                target_module=aux.target_module,
                out_port=aux.out_port,
                in_port=aux.in_port,
                connection_path=aux.connection_path,
                start_s=solver.Value(aux_starts[idx]) / cfg.time_scale,
                end_s=solver.Value(aux_ends[idx]) / cfg.time_scale,
                duration_s=transfer_duration_s,
                connect_duration_s=0.0,
                transfer_duration_s=transfer_duration_s,
                operation_cost=aux_flow_costs.usage_cost,
                connection_cost=0.0,
                usage_cost=aux_flow_costs.usage_cost,
                energy_consumption_kwh=aux_flow_costs.energy_consumption_kwh,
                energy_cost=aux_flow_costs.energy_cost,
                co2_emissions_kg=aux_flow_costs.co2_emissions_kg,
                weighted_usage_cost=aux_flow_costs.weighted_usage_cost,
                weighted_energy_cost=aux_flow_costs.weighted_energy_cost,
                weighted_co2_cost=aux_flow_costs.weighted_co2_cost,
                total_cost=aux_flow_costs.total_weighted_cost,
                material=resolved_composition,
                transfer_kind=aux.transfer_kind,
                trace={
                    "active": True,
                    "is_recipe_step": False,
                    "auxiliary_transfer": aux.__dict__,
                    "symbolic_amount_variable": f"aux_amount_{idx}",
                    "resolved_amount_l": amount_l,
                    "boundary_index": aux.boundary_index,
                    "material_signature": composition_signature(resolved_composition),
                },
            )
        )

    for node in tasks:
        if not solver.BooleanValue(active[node.id]):
            continue
        chosen_module = ""
        chosen_route: Optional[TransferCandidate] = None
        chosen_idx: Optional[int] = None
        if node.node_type in {"dose", "separation"}:
            chosen_idx = next(idx for idx, _ in enumerate(route_candidates[node.id]) if solver.BooleanValue(route_select[(node.id, idx)]))
            chosen_route = route_candidates[node.id][chosen_idx]
            if chosen_route.source_module == chosen_route.target_module:
                continue  # self-transfer: material already present, no physical operation
            chosen_module = chosen_route.target_module if node.node_type == "dose" else chosen_route.source_module
            operation_costs = chosen_route.flow_costs
            operation = _operation_label(node, chosen_module, chosen_route)
        else:
            chosen_module = next(wb for (nid, wb), var in assign.items() if nid == node.id and solver.BooleanValue(var))
            operation_costs = assigned_cost_breakdowns[(node.id, chosen_module)]
            operation = _operation_label(node, chosen_module)
        duration_s = solver.Value(duration_vars[node.id]) / cfg.time_scale
        start_s = solver.Value(starts[node.id]) / cfg.time_scale
        end_s = solver.Value(ends[node.id]) / cfg.time_scale
        resolved_material = _material_for_node(node, selected_branches, tasks)
        operations.append(
            PlannedOperation(
                step_id=0,
                recipe_node_id=node.id,
                branch_group_id=node.branch_group_id,
                branch_id=node.branch_id,
                operation_type=node.node_type,
                operation=operation,
                module=chosen_module,
                source_module=chosen_route.source_module if chosen_route else chosen_module,
                target_module=chosen_route.target_module if chosen_route else chosen_module,
                out_port=chosen_route.out_port if chosen_route else "",
                in_port=chosen_route.in_port if chosen_route else "",
                connection_path=chosen_route.connection_path if chosen_route else "",
                start_s=start_s,
                end_s=end_s,
                duration_s=duration_s,
                connect_duration_s=0.0,
                transfer_duration_s=chosen_route.transfer_duration_s if chosen_route else 0.0,
                operation_cost=operation_costs.usage_cost,
                connection_cost=0.0,
                usage_cost=operation_costs.usage_cost,
                energy_consumption_kwh=operation_costs.energy_consumption_kwh,
                energy_cost=operation_costs.energy_cost,
                co2_emissions_kg=operation_costs.co2_emissions_kg,
                weighted_usage_cost=operation_costs.weighted_usage_cost,
                weighted_energy_cost=operation_costs.weighted_energy_cost,
                weighted_co2_cost=operation_costs.weighted_co2_cost,
                total_cost=operation_costs.total_weighted_cost,
                material=resolved_material,
                transfer_kind=PURE_MATERIAL if chosen_route else "",
                trace={
                    "semantic_uri": node.semantic_uri,
                    "params": node.params,
                    "active": True,
                    "is_recipe_step": True,
                    "physical_route": chosen_route.__dict__ if chosen_route else {},
                    "material_signature": (
                        composition_signature(resolved_material) if chosen_route else ""
                    ),
                },
            )
        )
    operations.sort(key=lambda op: (op.start_s, op.end_s, op.recipe_node_id))
    for idx, op in enumerate(operations, start=1):
        op.step_id = idx

    total_cost = sum(op.total_cost for op in operations)
    total_usage_cost = sum(op.usage_cost for op in operations)
    total_energy_consumption_kwh = sum(op.energy_consumption_kwh for op in operations)
    total_energy_cost = sum(op.energy_cost for op in operations)
    total_co2_emissions_kg = sum(op.co2_emissions_kg for op in operations)
    total_weighted_usage_cost = sum(op.weighted_usage_cost for op in operations)
    total_weighted_energy_cost = sum(op.weighted_energy_cost for op in operations)
    total_weighted_co2_cost = sum(op.weighted_co2_cost for op in operations)
    total_duration = sum(op.duration_s for op in operations)
    makespan_s = solver.Value(makespan) / cfg.time_scale
    objective_profit = cfg.base_profit + cfg.lambda_per_second * makespan_s - total_cost
    local_dose_count = sum(
        1 for node_id, routes in route_candidates.items()
        for idx, route in enumerate(routes)
        if route.source_module == route.target_module
        and solver.BooleanValue(route_select.get((node_id, idx), 0))
    )
    local_dose_reservations = [
        MaterialReservation(
            recipe_node_id=node_id,
            process_module=route.target_module,
            material=route.material.upper(),
            amount_l=route.amount_l,
        )
        for node_id, routes in route_candidates.items()
        for idx, route in enumerate(routes)
        if route.source_module == route.target_module
        and solver.BooleanValue(route_select.get((node_id, idx), 0))
    ]
    selected_transfer_routes = {
        node_id: [
            {
                "source_module": route.source_module,
                "target_module": route.target_module,
                "out_port": route.out_port,
                "in_port": route.in_port,
                "material": route.material,
                "connection_path": route.connection_path,
                "is_self": route.source_module == route.target_module,
            }
            for idx, route in enumerate(routes)
            if solver.BooleanValue(route_select.get((node_id, idx), 0))
        ]
        for node_id, routes in route_candidates.items()
    }
    terminal_connections = _terminal_connections_from_operations(operations)
    diagnostics = {
        "solver_objective": solver.ObjectiveValue() / cost_scale,
        "horizon_s": horizon,
        "lambda_per_second": cfg.lambda_per_second,
        "usage_cost_weight": cfg.usage_cost_weight,
        "energy_cost_weight": cfg.energy_cost_weight,
        "co2_penalty": cfg.co2_penalty,
        "electricity_price_eur_per_kwh": cfg.electricity_price_eur_per_kwh,
        "relative_gap_limit": cfg.relative_gap_limit,
        "solver_random_seed": cfg.solver_random_seed,
        "enable_auxiliary_transfers": cfg.enable_auxiliary_transfers,
        "auxiliary_transfer_count": sum(1 for idx in aux_present if solver.BooleanValue(aux_present[idx])),
        "auxiliary_candidate_count": len(aux_candidates),
        "local_dose_count": local_dose_count,
        "selected_transfer_routes": selected_transfer_routes,
        "require_final_disconnect": cfg.require_final_disconnect,
        "terminal_connections": terminal_connections,
        "relaxed_auxiliary_inventory": relaxed_auxiliary_inventory,
        "solver_wall_time_s": solver.WallTime(),
        "stop_after_first_solution": bool(
            relaxed_auxiliary_inventory and cfg.lazy_auxiliary_relaxed_stop_after_first_solution
        ),
    }
    diagnostics.update(diagnostics_extra or {})
    return PlannerResult(
        status=status,
        objective_profit=objective_profit,
        base_profit=cfg.base_profit,
        total_duration_s=total_duration,
        makespan_s=makespan_s,
        total_cost=total_cost,
        selected_branches=selected_branches,
        operations=operations,
        total_usage_cost=total_usage_cost,
        total_energy_consumption_kwh=total_energy_consumption_kwh,
        total_energy_cost=total_energy_cost,
        total_co2_emissions_kg=total_co2_emissions_kg,
        total_weighted_usage_cost=total_weighted_usage_cost,
        total_weighted_energy_cost=total_weighted_energy_cost,
        total_weighted_co2_cost=total_weighted_co2_cost,
        local_dose_reservations=local_dose_reservations,
        diagnostics=diagnostics,
    )


def _selected_prefix_compositions(
    tasks: List[RecipeNode],
    selected_branches: Dict[str, Any],
) -> Dict[str, Dict[str, float]]:
    """Return the selected recipe composition visible at every task cut."""
    cumulative: Dict[str, float] = {}
    expected: Dict[str, Dict[str, float]] = {}
    for node in tasks:
        selected = True
        group = str(node.branch_group_id or "")
        if group.startswith(("XOR_", "OR_")) and node.branch_id:
            chosen = selected_branches.get(group)
            selected = (
                node.branch_id in chosen
                if isinstance(chosen, (list, tuple, set))
                else chosen == node.branch_id
            )
        if selected and node.node_type == "dose":
            material, amount, produces = _ingredient_effect(node)
            if produces and material and amount > 0:
                name = material.upper()
                cumulative[name] = cumulative.get(name, 0.0) + amount
        expected[node.id] = normalize_composition(cumulative)
    return expected


def _find_auxiliary_transfer_violations(
    result: PlannerResult,
    tasks: List[RecipeNode],
    recipe_inputs: Dict[str, float],
    rtn: RTNModel,
    tolerance: float = 1e-6,
) -> List[Dict[str, Any]]:
    task_index = {node.id: idx for idx, node in enumerate(tasks)}
    inventory: Dict[str, Dict[str, float]] = {}
    _init_inv = _init_inventory_from_rtn(rtn)
    for wb, raw in _init_inv.items():
        material, qty = _resource_material_quantity_raw(raw)
        material = material.upper()
        if material and qty > tolerance:
            inventory.setdefault(wb, {})[material] = qty

    violations: List[Dict[str, Any]] = []
    operations = sorted(result.operations, key=lambda op: (op.start_s, op.end_s, op.step_id))
    expected_by_task = _selected_prefix_compositions(tasks, result.selected_branches)
    last_product_process_op: Optional[PlannedOperation] = None
    dose_target_boundaries: Dict[str, int] = {}
    for op in operations:
        if op.operation_type == "dose" and op.target_module:
            b = task_index.get(op.recipe_node_id, 0)
            dose_target_boundaries[op.target_module] = min(
                b, dose_target_boundaries.get(op.target_module, b)
            )
    for op in operations:
        boundary = task_index.get(op.recipe_node_id, 0)
        if op.operation_type in {"mix", "usage", "settling", "heating"}:
            wb_inventory = inventory.setdefault(op.module, {})
            expected = expected_by_task.get(
                op.recipe_node_id,
                {material.upper(): float(qty) for material, qty in recipe_inputs.items()},
            )
            materials = sorted(set(expected) | {m for m, qty in wb_inventory.items() if qty > tolerance})
            process_source_module = (
                last_product_process_op.module
                if last_product_process_op is not None and op.operation_type in {"usage", "settling"}
                else ""
            )
            process_transfer_composition: Dict[str, float] = {}
            mixture_violation_added = False
            if process_source_module and process_source_module != op.module:
                process_transfer_composition = {
                    material: expected_qty - wb_inventory.get(material, 0.0)
                    for material, expected_qty in expected.items()
                    if expected_qty - wb_inventory.get(material, 0.0) > tolerance
                }
                source_stock = inventory.get(process_source_module, {})
                if (
                    len(process_transfer_composition) > 1
                    and compositions_proportional(process_transfer_composition, expected, tolerance=tolerance)
                    and all(
                        source_stock.get(material, 0.0) + tolerance >= amount
                        for material, amount in process_transfer_composition.items()
                    )
                ):
                    violations.append(
                        {
                            "kind": "mixture_shortage",
                            "boundary_index": boundary,
                            "composition": process_transfer_composition,
                            "target_module": op.module,
                            "needed_l": sum(process_transfer_composition.values()),
                            "operation": op.recipe_node_id,
                            "source_modules": [process_source_module],
                            "source_quantities": {
                                process_source_module: dict(source_stock),
                            },
                            "process_transfer": True,
                            "process_source_module": process_source_module,
                        }
                    )
                    mixture_violation_added = True
            for material in materials:
                actual_qty = wb_inventory.get(material, 0.0)
                expected_qty = expected.get(material, 0.0)
                if process_source_module and process_source_module != op.module:
                    source_modules = [process_source_module]
                    source_quantities = {
                        process_source_module: inventory.get(process_source_module, {}).get(material, 0.0)
                    }
                    process_transfer = True
                else:
                    source_modules = [
                        source_module
                        for source_module, stock in inventory.items()
                        if stock.get(material, 0.0) > tolerance and source_module != op.module
                    ]
                    source_quantities = {
                        source_module: stock.get(material, 0.0)
                        for source_module, stock in inventory.items()
                        if stock.get(material, 0.0) > tolerance and source_module != op.module
                    }
                    process_transfer = False
                if actual_qty + tolerance < expected_qty:
                    if mixture_violation_added:
                        continue
                    violations.append(
                        {
                            "kind": "shortage",
                            "boundary_index": boundary,
                            "material": material,
                            "target_module": op.module,
                            "needed_l": expected_qty - actual_qty,
                            "operation": op.recipe_node_id,
                            "source_modules": source_modules,
                            "source_quantities": source_quantities,
                            "process_transfer": process_transfer,
                            "process_source_module": process_source_module,
                        }
                    )
                elif actual_qty > expected_qty + tolerance:
                    violations.append(
                        {
                            "kind": "excess",
                            "boundary_index": boundary,
                            "material": material,
                            "source_module": op.module,
                            "needed_l": actual_qty - expected_qty,
                            "operation": op.recipe_node_id,
                            "min_dose_boundary": dose_target_boundaries.get(op.module, boundary),
                        }
                    )
            last_product_process_op = op
            continue

        if not (op.out_port and op.in_port):
            continue
        composition = normalize_composition(op.material, tolerance=tolerance)
        if not composition:
            continue
        boundary = int(
            (op.trace or {}).get("boundary_index", boundary)
            if isinstance(op.trace, dict)
            else boundary
        )
        kind = op.transfer_kind or classify_transfer_kind(composition)
        source_inventory = normalize_composition(
            inventory.setdefault(op.source_module, {}), tolerance=tolerance
        )
        target_inventory = normalize_composition(
            inventory.setdefault(op.target_module, {}), tolerance=tolerance
        )

        if op.operation_type != "separation":
            if kind == PURE_MATERIAL:
                material = next(iter(composition))
                if not is_pure_composition(source_inventory, material):
                    violations.append(
                        {
                            "kind": "source_composition",
                            "boundary_index": boundary,
                            "operation": op.recipe_node_id,
                            "source_module": op.source_module,
                            "requested_composition": composition,
                            "source_composition": source_inventory,
                        }
                    )
            elif not compositions_proportional(source_inventory, composition, tolerance=tolerance):
                violations.append(
                    {
                        "kind": "mixture_ratio",
                        "boundary_index": boundary,
                        "operation": op.recipe_node_id,
                        "source_module": op.source_module,
                        "requested_composition": composition,
                        "source_composition": source_inventory,
                    }
                )

        for material, quantity in composition.items():
            available = source_inventory.get(material, 0.0)
            if available + tolerance < quantity:
                process_source_module = (
                    last_product_process_op.module
                    if op.operation_type == "separation" and last_product_process_op is not None
                    else ""
                )
                violations.append(
                    {
                        "kind": "shortage",
                        "boundary_index": boundary,
                        "material": material,
                        "target_module": op.source_module,
                        "needed_l": quantity - available,
                        "operation": op.recipe_node_id,
                        "source_modules": (
                            [process_source_module]
                            if process_source_module and process_source_module != op.source_module
                            else []
                        ),
                        "source_quantities": (
                            {
                                process_source_module: inventory.get(process_source_module, {}).get(
                                    material, 0.0
                                )
                            }
                            if process_source_module and process_source_module != op.source_module
                            else {}
                        ),
                        "process_transfer": bool(
                            process_source_module and process_source_module != op.source_module
                        ),
                        "process_source_module": process_source_module,
                    }
                )
            source_inventory[material] = max(0.0, available - quantity)
            if source_inventory[material] <= tolerance:
                source_inventory.pop(material, None)
        inventory[op.source_module] = source_inventory

        if op.operation_type == "aux_transfer":
            if kind == PURE_MATERIAL:
                material = next(iter(composition))
                compatible_target = not target_inventory or is_pure_composition(
                    target_inventory, material
                )
            else:
                compatible_target = not target_inventory or compositions_proportional(
                    target_inventory, composition, tolerance=tolerance
                )
            if not compatible_target:
                violations.append(
                    {
                        "kind": "target_composition",
                        "boundary_index": boundary,
                        "operation": op.recipe_node_id,
                        "target_module": op.target_module,
                        "requested_composition": composition,
                        "target_composition": target_inventory,
                    }
                )

        inventory[op.target_module] = add_composition(
            target_inventory, composition, tolerance=tolerance
        )
        _cap_v = _capacity_from_rtn(rtn)
        cap = float((_cap_v.get(op.target_module) or [0.0])[0])
        total = sum(inventory[op.target_module].values())
        if cap and total > cap + tolerance:
            violations.append(
                {
                    "kind": "capacity",
                    "boundary_index": boundary,
                    "material": next(iter(composition), ""),
                    "source_module": op.target_module,
                    "needed_l": total - cap,
                    "operation": op.recipe_node_id,
                }
            )

    return violations


def _valid_material_boundaries(tasks: List[Any]) -> List[int]:
    """Return global inventory cuts that do not split an AND region.

    Recipe operations inside an AND region remain independently schedulable.
    Only extra inventory-rebalancing transfers treat the region as atomic, so a
    deterministic topological ordering cannot invent a material state between
    sibling branches.
    """
    groups: Dict[str, List[int]] = {}
    for index, task in enumerate(tasks):
        group = str(getattr(task, "branch_group_id", "") or "")
        if group.startswith("AND_"):
            groups.setdefault(group, []).append(index)
    valid: List[int] = []
    for boundary in range(len(tasks) + 1):
        if any(min(indices) < boundary <= max(indices) for indices in groups.values()):
            continue
        valid.append(boundary)
    return valid


def _nearest_valid_material_boundary(
    boundary: int,
    valid_boundaries: List[int],
    *,
    prefer_before: bool,
) -> int:
    if not valid_boundaries:
        return max(0, boundary)
    if boundary in valid_boundaries:
        return boundary
    before = [candidate for candidate in valid_boundaries if candidate < boundary]
    after = [candidate for candidate in valid_boundaries if candidate > boundary]
    if prefer_before and before:
        return max(before)
    if not prefer_before and after:
        return min(after)
    return max(before) if before else min(after)


def _targeted_auxiliary_transfer_candidates(
    violations: List[Dict[str, Any]],
    module_ops: Dict[str, List[Tuple[str, Any, float]]],
    module_interfaces: Dict[str, List[Tuple[str, str]]],
    rtn: RTNModel,
    cfg: PlannerConfig,
) -> List[AuxiliaryTransferCandidate]:
    candidates: List[AuxiliaryTransferCandidate] = []
    _init_inv_t = _init_inventory_from_rtn(rtn)
    _cap_t = _capacity_from_rtn(rtn)
    task_views = [TaskNodeView(task) for task in rtn.topological_tasks()]
    valid_boundaries = _valid_material_boundaries(task_views)
    for violation in violations:
        requested_composition = normalize_composition(violation.get("composition") or {})
        is_mixture = len(requested_composition) > 1
        material = str(violation.get("material", "")).upper()
        if is_mixture:
            material = ""
        if not material and not is_mixture:
            continue
        boundary_raw = int(violation.get("boundary_index", 0))
        min_dose_raw = int(violation.get("min_dose_boundary", boundary_raw))
        shortage_like = violation.get("kind") in {"shortage", "mixture_shortage"}
        if shortage_like:
            boundary = boundary_raw
        else:
            boundary = min(boundary_raw, min_dose_raw)
        boundary = _nearest_valid_material_boundary(
            boundary,
            valid_boundaries,
            prefer_before=not shortage_like,
        )
        if shortage_like:
            target_modules = [str(violation.get("target_module", ""))]
            source_modules = list(violation.get("source_modules") or [])
            if not source_modules and not violation.get("process_transfer"):
                source_modules = [
                    wb
                    for wb, raw in _init_inv_t.items()
                    if _resource_material_quantity_raw(raw)[0].upper() == material
                    and _resource_material_quantity_raw(raw)[1] > 0
                ]
        else:
            source_modules = [str(violation.get("source_module", ""))]
            target_modules = list(module_ops)

        for src in source_modules:
            src_material, src_qty = _resource_material_quantity_raw(_init_inv_t.get(src))
            source_quantities = violation.get("source_quantities") if isinstance(violation.get("source_quantities"), dict) else {}
            raw_dynamic_quantity = source_quantities.get(src, 0.0)
            if isinstance(raw_dynamic_quantity, dict):
                dynamic_src_qty = sum(float(value) for value in raw_dynamic_quantity.values())
            else:
                dynamic_src_qty = float(raw_dynamic_quantity or 0.0)
            if is_mixture:
                available_qty = sum(requested_composition.values())
            elif src_material.upper() == material and src_qty > 0:
                available_qty = src_qty
            elif dynamic_src_qty > 0:
                available_qty = dynamic_src_qty
            else:
                available_qty = float(violation.get("needed_l", 0.0) or 0.0)
            if not (_has_op(src, "Draining", module_ops) and _has_op(src, "Connect", module_ops)):
                continue
            for dst in target_modules:
                if not dst or src == dst:
                    continue
                if not (_has_op(dst, "Filling", module_ops) and _has_op(dst, "Connect", module_ops)):
                    continue
                dst_cap = float((_cap_t.get(dst) or [src_qty])[0] or src_qty)
                max_amount = (
                    sum(requested_composition.values())
                    if is_mixture
                    else min(available_qty, dst_cap)
                )
                if dst_cap and max_amount > dst_cap + 1e-9:
                    continue
                if max_amount <= 0:
                    continue
                for out_port, in_port in _auxiliary_port_pairs(module_interfaces, src, dst):
                    drain_speed = float(_operation_param(src, "Draining", module_ops) or 1.0)
                    fill_speed = float(_operation_param(dst, "Filling", module_ops) or 1.0)
                    speed = min(drain_speed, fill_speed) if drain_speed and fill_speed else 1.0
                    drain_usage, drain_energy, drain_co2 = _capability_metrics(src, "Draining", module_ops)
                    fill_usage, fill_energy, fill_co2 = _capability_metrics(dst, "Filling", module_ops)
                    src_connect_usage, src_connect_energy, src_connect_co2 = _capability_metrics(src, "Connect", module_ops)
                    dst_connect_usage, dst_connect_energy, dst_connect_co2 = _capability_metrics(dst, "Connect", module_ops)
                    candidates.append(
                        AuxiliaryTransferCandidate(
                            boundary_index=boundary,
                            material=material,
                            source_module=src,
                            target_module=dst,
                            out_port=out_port,
                            in_port=in_port,
                            max_amount_l=max_amount,
                            speed_l_s=speed,
                            operation_usage_cost=drain_usage + fill_usage,
                            connection_usage_cost=src_connect_usage + dst_connect_usage,
                            flow_energy_rate_kwh_s=drain_energy + fill_energy,
                            flow_co2_rate_kg_s=drain_co2 + fill_co2,
                            connect_energy_rate_kwh_s=src_connect_energy + dst_connect_energy,
                            connect_co2_rate_kg_s=src_connect_co2 + dst_connect_co2,
                            transfer_kind=MIXTURE if is_mixture else PURE_MATERIAL,
                            composition=composition_tuple(requested_composition) if is_mixture else (),
                        )
                    )
    return candidates


def _auxiliary_candidate_key(aux: AuxiliaryTransferCandidate) -> Tuple[int, str, str, str, str, str]:
    signature = composition_signature(aux.composition_dict)
    return (
        aux.boundary_index,
        f"{aux.transfer_kind}:{signature}",
        aux.source_module,
        aux.target_module,
        aux.out_port,
        aux.in_port,
    )


def _canonical_auxiliary_candidates(
    candidates: List[AuxiliaryTransferCandidate],
) -> List[AuxiliaryTransferCandidate]:
    """Remove port symmetry from inventory-only planning models.

    Those models decide boundary/material/source/target/amount. Physical port
    alternatives are restored in the lifecycle solve, where they can affect
    reuse, Connect/Disconnect cost, and makespan.
    """
    selected: List[AuxiliaryTransferCandidate] = []
    seen: set[Tuple[int, str, str, str]] = set()
    for candidate in candidates:
        logical_key = (
            candidate.boundary_index,
            f"{candidate.transfer_kind}:{composition_signature(candidate.composition_dict)}",
            candidate.source_module,
            candidate.target_module,
        )
        if logical_key in seen:
            continue
        seen.add(logical_key)
        selected.append(candidate)
    return selected


def _operation_material_name(op: PlannedOperation) -> str:
    if len(op.material) == 1:
        return next(iter(op.material)).upper()
    physical_route = op.trace.get("physical_route") if isinstance(op.trace, dict) else None
    if isinstance(physical_route, dict) and physical_route.get("material"):
        return str(physical_route["material"]).upper()
    auxiliary_transfer = op.trace.get("auxiliary_transfer") if isinstance(op.trace, dict) else None
    if isinstance(auxiliary_transfer, dict) and auxiliary_transfer.get("material"):
        return str(auxiliary_transfer["material"]).upper()
    return ""


def _replay_validates_result(
    rtn: RTNModel,
    result: PlannerResult,
    
    cfg: PlannerConfig,
) -> bool:
    """Validate a solver result via the RTN-native schedule validator."""
    try:
        try:
            from schedule_validator import validate_schedule
        except ImportError:  # pragma: no cover - notebook direct import compatibility
            from schedule_validator import validate_schedule

        validation = validate_schedule(
            rtn,
            result,
            module_maximum_volume=_capacity_from_rtn(rtn),
            module_resources=_init_inventory_from_rtn(rtn),
            config=cfg,
        )
        return bool(validation.valid)
    except Exception:
        return False


def _certify_or_invalidate_result(
    rtn: RTNModel,
    result: PlannerResult,
    cfg: PlannerConfig,
) -> PlannerResult:
    """Never expose a physically invalid CP-SAT solution as feasible."""
    if result.status not in {"OPTIMAL", "FEASIBLE"}:
        return result
    try:
        try:
            from schedule_validator import validate_schedule
        except ImportError:  # pragma: no cover
            from schedule_validator import validate_schedule
        validation = validate_schedule(
            rtn,
            result,
            module_maximum_volume=_capacity_from_rtn(rtn),
            module_resources=_init_inventory_from_rtn(rtn),
            config=cfg,
        )
    except Exception as exc:  # pragma: no cover - defensive public boundary
        result.status = "MODEL_INVALID"
        result.diagnostics.update(
            {
                "certification_valid": False,
                "certification_errors": [f"Validator failed: {exc}"],
            }
        )
        return result
    if validation.valid:
        result.diagnostics["certification_valid"] = True
        return result
    result.status = "MODEL_INVALID"
    result.diagnostics.update(
        {
            "certification_valid": False,
            "certification_errors": list(validation.errors),
            "composition_errors": list(validation.composition_errors),
        }
    )
    return result


def _apply_solution_hints(
    model: Any,
    seed: PlannerResult,
    tasks: List[TaskNodeView],
    active: Dict[str, Any],
    starts: Dict[str, Any],
    ends: Dict[str, Any],
    duration_vars: Dict[str, Any],
    assign: Dict[Tuple[str, str], Any],
    route_select: Dict[Tuple[str, int], Any],
    route_candidates: Dict[str, List[TransferCandidate]],
    aux_candidates: List[AuxiliaryTransferCandidate],
    aux_present: Dict[int, Any],
    aux_starts: Dict[int, Any],
    aux_ends: Dict[int, Any],
    selected_branch_vars: Dict[Tuple[str, str], Any],
    cfg: PlannerConfig,
    fix_routing: bool = False,
    exact_physical_hint: bool = False,
) -> None:
    """Seed the full connection model from a fast inventory/routing solve."""
    operations = [op for op in seed.operations if op.operation_type not in {"connect", "disconnect"}]
    by_node = {op.recipe_node_id: op for op in operations if not op.recipe_node_id.startswith("AUX_")}

    # The inventory seed fixes only the logical route.  The lifecycle solve
    # must retain every physical port pair for that route: active connections
    # created for other targets may occupy any one of the source ports, and a
    # single preselected "spare" can therefore be the wrong spare.  Pruning to
    # one alternate made a free Out3/Out4 invisible and forced an unnecessary
    # Disconnect + reconnect on Out2.  Unrelated source/target/material routes
    # are still fixed to zero below, so this does not reopen logical routing.

    for node in tasks:
        operation = by_node.get(node.id)
        if operation is None:
            continue
        model.AddHint(starts[node.id], int(round(operation.start_s * cfg.time_scale)))
        model.AddHint(ends[node.id], int(round(operation.end_s * cfg.time_scale)))
        model.AddHint(duration_vars[node.id], int(round(operation.duration_s * cfg.time_scale)))
        for (node_id, module), variable in assign.items():
            if node_id == node.id:
                value = int(module == operation.module)
                model.AddHint(variable, value)
                if fix_routing:
                    model.Add(variable == value)

    for node_id, routes in route_candidates.items():
        operation = by_node.get(node_id)
        if operation is None or not operation.connection_path:
            continue
        selected_material = _operation_material_name(operation).upper()
        for route_index, route in enumerate(routes):
            variable = route_select.get((node_id, route_index))
            if variable is not None:
                value = int(route.connection_path == operation.connection_path)
                model.AddHint(variable, value)
                if fix_routing:
                    if exact_physical_hint:
                        model.Add(variable == value)
                        continue
                    # The inventory seed fixes the logical material route, not
                    # the physical port pair.  Keep every equivalent physical
                    # pair free so the lifecycle objective can choose reuse,
                    # teardown/switch, or an additional free connection.
                    same_logical_route = (
                        route.source_module == operation.source_module
                        and route.target_module == operation.target_module
                        and route.material.upper() == selected_material
                    )
                    if not same_logical_route:
                        model.Add(variable == 0)

    selected_aux_keys: Dict[Tuple[int, str, str, str, str, str], PlannedOperation] = {}
    selected_aux_logical_keys: set[Tuple[int, str, str, str]] = set()
    for operation in operations:
        if operation.operation_type != "aux_transfer":
            continue
        auxiliary = operation.trace.get("auxiliary_transfer") if isinstance(operation.trace, dict) else None
        boundary = int((auxiliary or {}).get("boundary_index", operation.trace.get("boundary_index", 0)))
        operation_kind = operation.transfer_kind or classify_transfer_kind(operation.material)
        material_identity = f"{operation_kind}:{composition_signature(operation.material)}"
        key = (
            boundary,
            material_identity,
            operation.source_module,
            operation.target_module,
            operation.out_port,
            operation.in_port,
        )
        selected_aux_keys[key] = operation
        selected_aux_logical_keys.add(key[:4])
    for index, candidate in enumerate(aux_candidates):
        if index not in aux_present:
            continue
        key = _auxiliary_candidate_key(candidate)
        selected_operation = selected_aux_keys.get(key)
        value = int(selected_operation is not None)
        model.AddHint(aux_present[index], value)
        if fix_routing:
            if exact_physical_hint:
                model.Add(aux_present[index] == value)
            else:
                # As for recipe transfers, fix source/target/material while
                # leaving all physical port pairs to the lifecycle optimizer.
                if key[:4] not in selected_aux_logical_keys:
                    model.Add(aux_present[index] == 0)
        if selected_operation is not None:
            model.AddHint(aux_starts[index], int(round(selected_operation.start_s * cfg.time_scale)))
            model.AddHint(aux_ends[index], int(round(selected_operation.end_s * cfg.time_scale)))

    for (group, branch), variable in selected_branch_vars.items():
        chosen = seed.selected_branches.get(group)
        is_selected = branch in chosen if isinstance(chosen, (list, tuple, set)) else chosen == branch
        model.AddHint(variable, int(is_selected))
        if fix_routing:
            model.Add(variable == int(is_selected))


def _node_active_var(model: Any, node: RecipeNode, selected_branch_vars: Dict[Tuple[str, str], Any]) -> Any:
    if node.branch_group_id.startswith(("XOR_", "OR_")) and node.branch_id:
        return selected_branch_vars[(node.branch_group_id, node.branch_id)]
    var = model.NewBoolVar(f"active_{node.id}")
    model.Add(var == 1)
    return var


def _evaluate_planning_condition(condition: Dict[str, Any], context: Dict[str, Any]) -> Optional[bool]:
    """Return a known planning-time condition value, or None when runtime-only."""
    try:
        from modplant_recipe.conditions import evaluate_condition
        return bool(evaluate_condition(condition, context))
    except KeyError:
        return None


def _estimate_horizon(
    tasks: List[RecipeNode],
    module_ops: Dict[str, List[Tuple[str, Any, float]]],
    rtn: RTNModel,
    cfg: PlannerConfig,
) -> int:
    total = 0
    for node in tasks:
        total += max(1, int(_duration_for_node(node, module_ops, cfg)))
    aux_total = 0
    if cfg.enable_auxiliary_transfers:
        aux_total = _estimate_auxiliary_inventory_horizon(
            module_ops,
            rtn,
            cfg,
        )
    transfer_steps = sum(node.node_type in {"dose", "separation"} for node in tasks)
    # Every physical route may require both a Connect and a Disconnect. The
    # extra budget is deliberately conservative because selected auxiliary
    # relocation routes also participate in the same persistent state model.
    connection_actions = 2 * max(1, transfer_steps + len(_init_inventory_from_rtn(rtn)) + len(module_ops))
    connection_total = connection_actions * max(
        0.0,
        float(cfg.connect_duration_s),
        float(cfg.disconnect_duration_s),
    )
    return max(int((total + aux_total + connection_total) * 2), 1)


def _estimate_auxiliary_inventory_horizon(
    module_ops: Dict[str, List[Tuple[str, Any, float]]],
    rtn: RTNModel,
    cfg: PlannerConfig,
) -> int:
    """Conservative time budget for moving initial inventory out of the way.

    Some feasible schedules must first relocate surplus initial material from a
    future process vessel. The recipe task durations alone can be much shorter
    than this cleanup transfer, so include a one-pass upper bound over initial
    inventory whenever auxiliary transfers are enabled.
    """
    speeds = []
    for src in module_ops:
        drain = _operation_param(src, "Draining", module_ops)
        if not drain:
            continue
        for dst in module_ops:
            if src == dst:
                continue
            fill = _operation_param(dst, "Filling", module_ops)
            if fill:
                speeds.append(min(float(drain), float(fill)))
    positive_speeds = [speed for speed in speeds if speed > 0]
    if not positive_speeds:
        return 0

    slowest_speed = min(positive_speeds)

    _init_inv = _init_inventory_from_rtn(rtn)
    initial_qty = sum(
        max(0.0, _resource_material_quantity_raw(raw)[1])
        for raw in _init_inv.values()
    )
    if initial_qty <= 0:
        return 0

    connect_budget = (
        cfg.connect_duration_s + cfg.disconnect_duration_s
    ) * max(1, len(_init_inv))
    return int(initial_qty / slowest_speed + connect_budget + 1)


def _duration_for_node(
    node: RecipeNode,
    module_ops: Dict[str, List[Tuple[str, Any, float]]],
    cfg: Optional[PlannerConfig] = None,
) -> float:
    if node.node_type in {"mix", "usage", "settling", "heating"}:
        return float(node.params.get("duration_s", 0.0))
    if node.node_type in {"dose", "separation"}:
        _, amount, _ = _ingredient_effect(node)
        speeds = []
        for wb in module_ops:
            drain = _operation_param(wb, "Draining", module_ops)
            fill = _operation_param(wb, "Filling", module_ops)
            if drain and fill:
                speeds.append(min(float(drain), float(fill)))
        speed = max(speeds) if speeds else 1.0
        transfer_duration = amount / speed if speed > 0 else 0.0
        return (cfg.connect_duration_s if cfg else 0.0) + transfer_duration
    return 0.0


def _feasible_module_for_node(
    node: RecipeNode,
    module_ops: Dict[str, List[Tuple[str, Any, float]]],
    rtn: RTNModel,
    nominal_volume: float = 0.0,
) -> List[str]:
    feasible = []
    needed = CAPABILITY_BY_NODE_TYPE.get(node.node_type, ())
    _cap = _capacity_from_rtn(rtn)
    for wb, ops in module_ops.items():
        op_names = {str(op[0]) for op in ops}
        if node.node_type == "mix":
            rpm = str(node.params.get("rpm", ""))
            if not any(op[0] == "Stirring" and str(op[1]) == rpm for op in ops):
                continue
        elif not all(op in op_names for op in needed[:1]):
            continue
        if node.node_type in {"mix", "usage", "settling", "heating"}:
            capacity = float((_cap.get(wb) or [0])[0]) if _cap else 0.0
            if capacity and capacity < (nominal_volume or _nominal_volume(node)):
                continue
        feasible.append(wb)
    return feasible


def _enumerate_transfer_candidates(
    node: RecipeNode,
    module_ops: Dict[str, List[Tuple[str, Any, float]]],
    module_interfaces: Dict[str, List[Tuple[str, str]]],
    rtn: RTNModel,
    cfg: PlannerConfig,
    allow_dynamic_inventory: bool = False,
) -> List[TransferCandidate]:
    material, amount, _ = _ingredient_effect(node)
    # material and amount derived from RTN Task coefficients (single source of truth)
    _init_inv = _init_inventory_from_rtn(rtn)
    _cap = _capacity_from_rtn(rtn)
    candidates: List[TransferCandidate] = []

    for src in module_ops:
        if not _has_op(src, "Draining", module_ops):
            continue
        if not _has_op(src, "Connect", module_ops):
            continue
        if node.node_type == "dose" and not allow_dynamic_inventory and not _resource_contains(_init_inv, src, material, amount):
            continue
        for dst in module_ops:
            if src == dst:
                continue
            if not _has_op(dst, "Filling", module_ops):
                continue
            if not _has_op(dst, "Connect", module_ops):
                continue
            cap = float((_cap.get(dst) or [0.0])[0])
            if cap and cap < amount:
                continue
            if node.node_type == "dose" and not allow_dynamic_inventory and _resource_quantity(_init_inv, dst) > 1e-9:
                continue
            if node.node_type == "separation" and not allow_dynamic_inventory:
                dst_material, dst_qty = _resource_material_quantity(_init_inv, dst)
                if dst_qty > 1e-9 and dst_material.upper() != material.upper():
                    continue
            for out_port in _ports(module_interfaces, src, "Output"):
                for in_port in _ports(module_interfaces, dst, "Input"):
                    drain_speed = float(_operation_param(src, "Draining", module_ops) or 1.0)
                    fill_speed = float(_operation_param(dst, "Filling", module_ops) or 1.0)
                    speed = min(drain_speed, fill_speed) if drain_speed and fill_speed else 1.0
                    transfer_duration = amount / speed if speed > 0 else 0.0
                    connect_duration = float(cfg.connect_duration_s)
                    duration = connect_duration + transfer_duration
                    drain_usage, drain_energy, drain_co2 = _capability_metrics(src, "Draining", module_ops)
                    fill_usage, fill_energy, fill_co2 = _capability_metrics(dst, "Filling", module_ops)
                    src_connect_usage, src_connect_energy, src_connect_co2 = _capability_metrics(src, "Connect", module_ops)
                    dst_connect_usage, dst_connect_energy, dst_connect_co2 = _capability_metrics(dst, "Connect", module_ops)
                    candidates.append(
                        TransferCandidate(
                            source_module=src,
                            target_module=dst,
                            out_port=out_port,
                            in_port=in_port,
                            material=material,
                            amount_l=amount,
                            duration_s=duration,
                            connect_duration_s=connect_duration,
                            transfer_duration_s=transfer_duration,
                            flow_costs=_cost_breakdown(
                                drain_usage + fill_usage,
                                drain_energy + fill_energy,
                                drain_co2 + fill_co2,
                                transfer_duration,
                                cfg,
                            ),
                            connect_costs=_cost_breakdown(
                                src_connect_usage + dst_connect_usage,
                                src_connect_energy + dst_connect_energy,
                                src_connect_co2 + dst_connect_co2,
                                connect_duration,
                                cfg,
                            ),
                        )
                    )

    # Self-transfer route: material already present (source==target, 0 cost)
    if node.node_type == "dose" and cfg.enable_auxiliary_transfers:
        _mat = material.upper()
        for wb, raw in (_init_inv or {}).items():
            wb_material, wb_quantity = _resource_material_quantity_raw(raw)
            if wb_material.upper() != _mat or wb_quantity + 1e-9 < amount:
                continue
            if not _has_op(wb, "Draining", module_ops) or not _has_op(wb, "Filling", module_ops):
                continue
            candidates.append(
                TransferCandidate(
                    source_module=wb, target_module=wb,
                    out_port="", in_port="",
                    material=material, amount_l=amount,
                    duration_s=0.0, connect_duration_s=0.0,
                    transfer_duration_s=0.0,
                    flow_costs=CostBreakdown(), connect_costs=CostBreakdown(),
                )
            )
    return candidates


def _canonical_transfer_candidates(
    candidates: List[TransferCandidate],
) -> List[TransferCandidate]:
    """Remove physical-port symmetry from inventory-only seed models."""
    selected: List[TransferCandidate] = []
    seen: set[Tuple[str, str, str, bool]] = set()
    for candidate in candidates:
        logical_key = (
            candidate.material.upper(),
            candidate.source_module,
            candidate.target_module,
            candidate.source_module == candidate.target_module,
        )
        if logical_key in seen:
            continue
        seen.add(logical_key)
        selected.append(candidate)
    return selected


def _link_transfer_targets_to_process_modules(
    tasks: List[RecipeNode],
    model: Any,
    assign: Dict[Tuple[str, str], Any],
    route_select: Dict[Tuple[str, int], Any],
    route_candidates: Dict[str, List[TransferCandidate]],
    precedence: Optional[List[Tuple[str, str]]] = None,
    allow_process_transfers: bool = False,
) -> None:
    """Bind each material route to its graph-nearest process operation.

    The previous implementation used the last shared Mix in a topological list,
    which silently rebound early doses in multi-Mix/nonlinear recipes.  Reachable
    graph distance is deterministic and remains valid when sibling tasks are
    reordered by the topological sorter.
    """
    first_mix = next((node for node in tasks if node.node_type == "mix"), None)
    material_steps = [node for node in tasks if node.node_type in {"mix", "usage", "settling", "heating"}]
    if first_mix is None:
        return

    by_id = {node.id: node for node in tasks}
    successors: Dict[str, List[str]] = {node.id: [] for node in tasks}
    predecessors: Dict[str, List[str]] = {node.id: [] for node in tasks}
    for source, target in precedence or []:
        if source in by_id and target in by_id:
            successors[source].append(target)
            predecessors[target].append(source)

    def nearest(start_id: str, *, forward: bool, node_types: set[str]) -> List[Any]:
        adjacency = successors if forward else predecessors
        frontier = list(adjacency.get(start_id, []))
        seen = {start_id}
        while frontier:
            found = [by_id[node_id] for node_id in frontier if by_id[node_id].node_type in node_types]
            if found:
                return found
            next_frontier: List[str] = []
            for node_id in frontier:
                if node_id in seen:
                    continue
                seen.add(node_id)
                next_frontier.extend(adjacency.get(node_id, []))
            frontier = sorted(set(next_frontier) - seen)
        return []

    for dose in [node for node in tasks if node.node_type == "dose"]:
        target_mixes = nearest(dose.id, forward=True, node_types={"mix"})
        if not target_mixes:
            # Legacy RTNModel instances may not carry precedence.  In that case
            # preserve the historical first-Mix interpretation, never the last.
            target_mixes = [first_mix]
        if not target_mixes:
            continue
        for idx, route in enumerate(route_candidates.get(dose.id, [])):
            compatible_assignments = [
                assign[(target_mix.id, route.target_module)]
                for target_mix in target_mixes
                if (target_mix.id, route.target_module) in assign
            ]
            if not compatible_assignments:
                model.Add(route_select[(dose.id, idx)] == 0)
            elif len(compatible_assignments) == 1:
                model.AddImplication(route_select[(dose.id, idx)], compatible_assignments[0])
            else:
                model.Add(sum(compatible_assignments) >= route_select[(dose.id, idx)])
    if not allow_process_transfers:
        for step in material_steps:
            if step.id == first_mix.id:
                continue
            first_mix_modules = [wb for (nid, wb) in assign if nid == first_mix.id]
            step_modules = {wb for (nid, wb) in assign if nid == step.id}
            for wb in first_mix_modules:
                if not step.branch_group_id.startswith(("XOR_", "OR_")) and wb not in step_modules and (first_mix.id, wb) in assign:
                    model.Add(assign[(first_mix.id, wb)] == 0)
            for (nid, wb), avar in list(assign.items()):
                if nid != step.id:
                    continue
                first_mix_assign = assign.get((first_mix.id, wb))
                if first_mix_assign is not None:
                    model.AddImplication(avar, first_mix_assign)
                    if not step.branch_group_id.startswith(("XOR_", "OR_")):
                        model.AddImplication(first_mix_assign, avar)
                else:
                    model.Add(avar == 0)
    # Separations drain the post-process material. Bind each separation's source
    # to the module of its direct predecessor (the latest mix/usage/settling before
    # the separation), not globally to the first mix. This handles multi-mix and
    # staged recipes where intermediate product may move between modules.
    for sep in [node for node in tasks if node.node_type == "separation"]:
        process_predecessors = nearest(
            sep.id,
            forward=False,
            node_types={"mix", "usage", "settling", "heating"},
        ) or [first_mix]
        sep_source_modules = {route.source_module for route in route_candidates.get(sep.id, [])}
        for process_predecessor in process_predecessors:
            for wb in [wb for (nid, wb) in assign if nid == process_predecessor.id]:
                if wb not in sep_source_modules:
                    model.Add(assign[(process_predecessor.id, wb)] == 0)
        for idx, route in enumerate(route_candidates.get(sep.id, [])):
            compatible_assignments = [
                assign[(process_predecessor.id, route.source_module)]
                for process_predecessor in process_predecessors
                if (process_predecessor.id, route.source_module) in assign
            ]
            if not compatible_assignments:
                model.Add(route_select[(sep.id, idx)] == 0)
            elif len(compatible_assignments) == 1:
                model.AddImplication(route_select[(sep.id, idx)], compatible_assignments[0])
            else:
                model.Add(sum(compatible_assignments) >= route_select[(sep.id, idx)])


def _init_inventory_from_rtn(rtn: RTNModel) -> Dict[str, Any]:
    """Extract {module: [material, qty]} from RTN ``init.inventory.*`` resources."""
    out: Dict[str, Any] = {}
    for rid, r in rtn.resources.items():
        if not r.metadata.get("is_initial_inventory"):
            continue
        module = str(r.metadata.get("module") or "")
        ingr = str(r.metadata.get("ingredient") or "")
        if module and ingr and r.initial_level > 0:
            out[module] = [ingr, r.initial_level]
    return out


def _capacity_from_rtn(rtn: RTNModel) -> Dict[str, Any]:
    """Extract {module: [max_volume_L]} from equipment Resource metadata."""
    out: Dict[str, Any] = {}
    for r in rtn.equipment_resources():
        cap = r.metadata.get("max_volume_L")
        if cap is not None:
            out[r.id] = [float(cap)]
    return out


def _add_connection_lifecycle_model(
    model: Any,
    demands: List[_ConnectionDemand],
    module_ops: Dict[str, List[Tuple]],
    cfg: PlannerConfig,
    horizon_i: int,
    cost_scale: int,
    intervals_by_module: Dict[str, List[Any]],
) -> Tuple[List[_ConnectionAction], List[Any], List[Any], List[Any]]:
    """Model persistent one-to-one port connections and explicit teardown.

    A selected transfer either reuses its already-active exact port pair or
    performs the required Disconnect transition(s) followed by one Connect.
    The pair then remains active across later operations until a switch or the
    mandatory end-of-plan Disconnect. Connection actions reserve both physical
    ports, so a port cannot be reconfigured while material is flowing.
    """
    ordered = sorted(demands, key=lambda item: item.order_key)
    state_keys = sorted({demand.state_key for demand in ordered})
    if not state_keys:
        return [], [], [], []

    grouped: Dict[Tuple[int, Tuple[Any, ...]], List[_ConnectionDemand]] = {}
    group_order: Dict[Tuple[int, Tuple[Any, ...]], Tuple[int, int, int]] = {}
    for demand in ordered:
        group_key = (demand.order_key[0], demand.event_key)
        grouped.setdefault(group_key, []).append(demand)
        group_order[group_key] = min(group_order.get(group_key, demand.order_key), demand.order_key)
    events = sorted(grouped.values(), key=lambda items: group_order[(items[0].order_key[0], items[0].event_key)])

    actions: List[_ConnectionAction] = []
    cost_terms: List[Any] = []
    action_ends: List[Any] = []
    action_intervals: List[Any] = []
    state: Dict[ConnectionStateKey, Any] = {}
    for key_index, key in enumerate(state_keys):
        initial = model.NewBoolVar(f"connection_initial_{key_index}")
        model.Add(initial == 0)
        state[key] = initial

    phase_time: Any = model.NewIntVar(0, horizon_i, "connection_phase_0")
    model.Add(phase_time == 0)
    prior_demands: Dict[ConnectionStateKey, List[_ConnectionDemand]] = {key: [] for key in state_keys}
    action_counter = 0

    def add_action(
        action_type: str,
        trigger_id: str,
        state_key: ConnectionStateKey,
        present: Any,
        suffix: str,
    ) -> Tuple[_ConnectionAction, Any]:
        nonlocal action_counter
        action_counter += 1
        pair = state_key[:4]
        material_signature = state_key[4]
        duration_s, costs = _connection_action_profile(pair, action_type, module_ops, cfg)
        duration_i = max(0, int(round(duration_s * cfg.time_scale)))
        start = model.NewIntVar(0, horizon_i, f"connection_action_start_{action_counter}")
        end = model.NewIntVar(0, horizon_i, f"connection_action_end_{action_counter}")
        model.Add(start == 0).OnlyEnforceIf(present.Not())
        model.Add(end == 0).OnlyEnforceIf(present.Not())
        interval = model.NewOptionalIntervalVar(
            start,
            duration_i,
            end,
            present,
            f"connection_action_interval_{action_counter}",
        )
        source_module, out_port, target_module, in_port = pair
        # Connect/Disconnect reserves only the selected physical ports. Other
        # free ports may be configured while a Module is dosing or processing.
        # The matching transfer interval on these same PORT resources prevents
        # disconnecting a connection while it is carrying material.
        intervals_by_module.setdefault(f"PORT:{source_module}.{out_port}", []).append(interval)
        intervals_by_module.setdefault(f"PORT:{target_module}.{in_port}", []).append(interval)
        action_intervals.append(interval)
        action_ends.append(end)
        cost_i = int(round(costs.total_weighted_cost * cost_scale))
        cost_terms.append(present * cost_i)
        action = _ConnectionAction(
            action_id=f"{trigger_id}_{suffix}",
            action_type=action_type,
            trigger_id=trigger_id,
            pair=pair,
            material_signature=material_signature,
            present=present,
            start=start,
            end=end,
            duration_s=duration_s,
            costs=costs,
        )
        actions.append(action)
        return action, interval

    for event_index, event_demands in enumerate(events, start=1):
        model.Add(sum(demand.present for demand in event_demands) <= 1)
        selected_by_key: Dict[ConnectionStateKey, Any] = {}
        demands_by_key: Dict[ConnectionStateKey, List[_ConnectionDemand]] = {}
        for demand in event_demands:
            demands_by_key.setdefault(demand.state_key, []).append(demand)
        for key_index, (key, options) in enumerate(demands_by_key.items()):
            selected = model.NewBoolVar(f"connection_selected_{event_index}_{key_index}")
            model.Add(selected == sum(option.present for option in options))
            selected_by_key[key] = selected

        disconnect_actions: Dict[ConnectionStateKey, _ConnectionAction] = {}
        for key_index, current_key in enumerate(state_keys):
            conflicting_selected = [
                selected
                for selected_key, selected in selected_by_key.items()
                if selected_key != current_key
                and _connection_pairs_conflict(current_key[:4], selected_key[:4])
            ]
            if not conflicting_selected:
                continue
            switch = model.NewBoolVar(f"connection_switch_{event_index}_{key_index}")
            model.Add(switch == sum(conflicting_selected))
            disconnect_needed = model.NewBoolVar(f"connection_needs_disconnect_{event_index}_{key_index}")
            model.Add(disconnect_needed <= switch)
            model.Add(disconnect_needed <= state[current_key])
            model.Add(disconnect_needed >= switch + state[current_key] - 1)
            disconnect, _ = add_action(
                "disconnect",
                f"EVENT_{event_index:03d}",
                current_key,
                disconnect_needed,
                f"DISCONNECT_{key_index:03d}",
            )
            model.Add(disconnect.start >= phase_time).OnlyEnforceIf(disconnect_needed)
            for previous in prior_demands[current_key]:
                model.Add(disconnect.start >= previous.transfer_end).OnlyEnforceIf(
                    [disconnect_needed, previous.present]
                )
            disconnect_actions[current_key] = disconnect

        connect_actions: Dict[ConnectionStateKey, _ConnectionAction] = {}
        for key_index, (selected_key, selected) in enumerate(selected_by_key.items()):
            needs_connect = model.NewBoolVar(f"connection_needs_connect_{event_index}_{key_index}")
            model.Add(needs_connect <= selected)
            model.Add(needs_connect + state[selected_key] <= 1)
            model.Add(needs_connect >= selected - state[selected_key])
            trigger = demands_by_key[selected_key][0].demand_id
            connect, _ = add_action("connect", trigger, selected_key, needs_connect, "CONNECT")
            model.Add(connect.start >= phase_time).OnlyEnforceIf(needs_connect)
            for disconnected_key, disconnect in disconnect_actions.items():
                if _connection_pairs_conflict(selected_key[:4], disconnected_key[:4]):
                    model.Add(connect.start >= disconnect.end).OnlyEnforceIf(
                        [needs_connect, disconnect.present]
                    )
            for option in demands_by_key[selected_key]:
                model.Add(connect.end <= option.transfer_start).OnlyEnforceIf(option.present)
            connect_actions[selected_key] = connect

        phase_end = model.NewIntVar(0, horizon_i, f"connection_phase_{event_index}")
        model.AddMaxEquality(
            phase_end,
            [
                phase_time,
                *[action.end for action in disconnect_actions.values()],
                *[action.end for action in connect_actions.values()],
            ],
        )
        for demand in event_demands:
            model.Add(demand.transfer_start >= phase_end).OnlyEnforceIf(demand.present)

        next_state: Dict[ConnectionStateKey, Any] = {}
        for key_index, current_key in enumerate(state_keys):
            selected = selected_by_key.get(current_key)
            if selected is None:
                selected = model.NewBoolVar(f"connection_not_selected_{event_index}_{key_index}")
                model.Add(selected == 0)
            conflicting_selected = [
                candidate_selected
                for candidate_key, candidate_selected in selected_by_key.items()
                if candidate_key != current_key
                and _connection_pairs_conflict(current_key[:4], candidate_key[:4])
            ]
            switched = model.NewBoolVar(f"connection_state_switched_{event_index}_{key_index}")
            model.Add(switched == sum(conflicting_selected))
            kept = model.NewBoolVar(f"connection_state_kept_{event_index}_{key_index}")
            model.Add(kept <= state[current_key])
            model.Add(kept + switched <= 1)
            model.Add(kept >= state[current_key] - switched)
            updated = model.NewBoolVar(f"connection_state_{event_index}_{key_index}")
            model.Add(updated >= kept)
            model.Add(updated >= selected)
            model.Add(updated <= kept + selected)
            next_state[current_key] = updated

        _enforce_connection_state_port_exclusivity(model, next_state)
        state = next_state
        phase_time = phase_end
        for demand in event_demands:
            prior_demands[demand.state_key].append(demand)

    if cfg.require_final_disconnect:
        for key_index, state_key in enumerate(state_keys):
            final_disconnect, _ = add_action(
                "disconnect",
                "FINAL",
                state_key,
                state[state_key],
                f"DISCONNECT_{key_index:03d}",
            )
            model.Add(final_disconnect.start >= phase_time).OnlyEnforceIf(final_disconnect.present)
            for previous in prior_demands[state_key]:
                model.Add(final_disconnect.start >= previous.transfer_end).OnlyEnforceIf(
                    [final_disconnect.present, previous.present]
                )

    return actions, cost_terms, action_ends, action_intervals


def _terminal_connections_from_operations(
    operations: List[PlannedOperation],
) -> List[Dict[str, str]]:
    """Return port-pair state intentionally left active after the plan."""
    active: Dict[ConnectionPair, str] = {}
    actions = sorted(
        (operation for operation in operations if operation.operation_type in {"connect", "disconnect"}),
        key=lambda operation: (
            operation.end_s,
            0 if operation.operation_type == "disconnect" else 1,
            operation.step_id,
        ),
    )
    for operation in actions:
        pair = (
            operation.source_module,
            operation.out_port,
            operation.target_module,
            operation.in_port,
        )
        if operation.operation_type == "disconnect":
            active.pop(pair, None)
        else:
            active[pair] = str(operation.trace.get("material_signature", ""))
    return [
        {
            "source_module": pair[0],
            "out_port": pair[1],
            "target_module": pair[2],
            "in_port": pair[3],
            "connection_path": f"{pair[0]}.{pair[1]} -> {pair[2]}.{pair[3]}",
            "material_signature": signature,
        }
        for pair, signature in sorted(active.items())
    ]


def _connection_material_signature(material: Any) -> str:
    """Canonical reuse identity for the fluid carried by a connection.

    A single material deliberately ignores transfer volume. A mixture includes
    every component quantity, which also fixes its composition ratio.
    """
    if isinstance(material, dict):
        return composition_signature(material)
    name = str(material or "").strip().upper()
    return composition_signature({name: 1.0}) if name else ""


def _connection_pairs_conflict(left: ConnectionPair, right: ConnectionPair) -> bool:
    left_ports = {(left[0], left[1]), (left[2], left[3])}
    right_ports = {(right[0], right[1]), (right[2], right[3])}
    return bool(left_ports & right_ports)


def _enforce_connection_state_port_exclusivity(
    model: Any,
    state: Dict[ConnectionStateKey, Any],
) -> None:
    by_port: Dict[Tuple[str, str], List[Any]] = {}
    for state_key, active in state.items():
        by_port.setdefault((state_key[0], state_key[1]), []).append(active)
        by_port.setdefault((state_key[2], state_key[3]), []).append(active)
    for active_pairs in by_port.values():
        if len(active_pairs) > 1:
            model.Add(sum(active_pairs) <= 1)


def _connection_action_profile(
    pair: ConnectionPair,
    action_type: str,
    module_ops: Dict[str, List[Tuple]],
    cfg: PlannerConfig,
) -> Tuple[float, CostBreakdown]:
    operation = "Connect" if action_type == "connect" else "Disconnect"
    source_module, _, target_module, _ = pair
    endpoint_records = []
    duration_records = []
    for module in (source_module, target_module):
        record = _operation_record(module, operation, module_ops)
        duration_records.append(record)
        if record is None and operation == "Disconnect":
            # Migration fallback for legacy fixtures/AAS: teardown uses the
            # corresponding Connect profile, but remains an explicit action.
            record = _operation_record(module, "Connect", module_ops)
        endpoint_records.append(record)
    fallback_duration_s = (
        float(cfg.connect_duration_s)
        if action_type == "connect"
        else float(cfg.disconnect_duration_s)
    )
    durations = [
        value if (value := _record_metric(record, 5)) > 0 else fallback_duration_s
        for record in duration_records
    ]
    duration_s = max(durations or [fallback_duration_s])
    usage = sum(_record_metric(record, 2) for record in endpoint_records)
    energy_rate = sum(_record_metric(record, 3) for record in endpoint_records)
    co2_rate = sum(_record_metric(record, 4) for record in endpoint_records)
    return duration_s, _cost_breakdown(usage, energy_rate, co2_rate, duration_s, cfg)


def _enumerate_auxiliary_transfer_candidates(
    tasks: List[RecipeNode],
    recipe_inputs: Dict[str, float],
    module_ops: Dict[str, List[Tuple[str, Any, float]]],
    module_interfaces: Dict[str, List[Tuple[str, str]]],
    rtn: RTNModel,
    cfg: PlannerConfig,
) -> List[AuxiliaryTransferCandidate]:
    total_by_material: Dict[str, float] = {}
    _init_inv = _init_inventory_from_rtn(rtn)
    for raw in _init_inv.values():
        material, qty = _resource_material_quantity_raw(raw)
        if material and qty > 0:
            total_by_material[material.upper()] = total_by_material.get(material.upper(), 0.0) + qty
    for material, qty in recipe_inputs.items():
        total_by_material.setdefault(material.upper(), float(qty))
    for node in tasks:
        if node.node_type in {"dose", "separation"}:
            mat, amt, _ = _ingredient_effect(node)
            material = mat.upper() if mat else ""
            if material:
                total_by_material.setdefault(material, float(amt))

    candidates: List[AuxiliaryTransferCandidate] = []
    capacities = _capacity_from_rtn(rtn)

    def append_route_candidates(
        boundary: int,
        *,
        material: str,
        amount_l: float,
        kind: str = PURE_MATERIAL,
        composition: Optional[Dict[str, float]] = None,
    ) -> None:
        if amount_l <= 0:
            return
        for src in module_ops:
            if not (_has_op(src, "Draining", module_ops) and _has_op(src, "Connect", module_ops)):
                continue
            for dst in module_ops:
                if src == dst:
                    continue
                if not (_has_op(dst, "Filling", module_ops) and _has_op(dst, "Connect", module_ops)):
                    continue
                dst_cap = float((capacities.get(dst) or [amount_l])[0] or amount_l)
                max_amount = amount_l if kind == MIXTURE else min(amount_l, dst_cap)
                if max_amount <= 0 or (kind == MIXTURE and dst_cap + 1e-9 < max_amount):
                    continue
                for out_port, in_port in _auxiliary_port_pairs(module_interfaces, src, dst):
                    drain_speed = float(_operation_param(src, "Draining", module_ops) or 1.0)
                    fill_speed = float(_operation_param(dst, "Filling", module_ops) or 1.0)
                    speed = min(drain_speed, fill_speed) if drain_speed and fill_speed else 1.0
                    drain_usage, drain_energy, drain_co2 = _capability_metrics(src, "Draining", module_ops)
                    fill_usage, fill_energy, fill_co2 = _capability_metrics(dst, "Filling", module_ops)
                    src_connect_usage, src_connect_energy, src_connect_co2 = _capability_metrics(src, "Connect", module_ops)
                    dst_connect_usage, dst_connect_energy, dst_connect_co2 = _capability_metrics(dst, "Connect", module_ops)
                    candidates.append(
                        AuxiliaryTransferCandidate(
                            boundary_index=boundary,
                            material=material,
                            source_module=src,
                            target_module=dst,
                            out_port=out_port,
                            in_port=in_port,
                            max_amount_l=max_amount,
                            speed_l_s=speed,
                            operation_usage_cost=drain_usage + fill_usage,
                            connection_usage_cost=src_connect_usage + dst_connect_usage,
                            flow_energy_rate_kwh_s=drain_energy + fill_energy,
                            flow_co2_rate_kg_s=drain_co2 + fill_co2,
                            connect_energy_rate_kwh_s=src_connect_energy + dst_connect_energy,
                            connect_co2_rate_kg_s=src_connect_co2 + dst_connect_co2,
                            transfer_kind=kind,
                            composition=composition_tuple(composition) if composition else (),
                        )
                    )

    valid_boundaries = _valid_material_boundaries(tasks)
    for boundary in valid_boundaries:
        for material, total_qty in sorted(total_by_material.items()):
            append_route_candidates(boundary, material=material, amount_l=total_qty)

    # Process-Module changes move the whole current recipe composition.  Eager
    # planning includes the full vector; lazy planning may add a smaller but
    # proportional fixed vector when the destination already contains part of
    # the same batch.
    expected_mixture = normalize_composition(recipe_inputs)
    if cfg.allow_process_transfers and len(expected_mixture) > 1:
        seen_process = False
        for boundary, task in enumerate(tasks):
            if task.node_type not in {"mix", "usage", "settling", "heating"}:
                continue
            if seen_process and boundary in valid_boundaries:
                append_route_candidates(
                    boundary,
                    material="",
                    amount_l=sum(expected_mixture.values()),
                    kind=MIXTURE,
                    composition=expected_mixture,
                )
            seen_process = True
    return candidates


def _enforce_inventory_flow_with_auxiliary_transfers(
    model: Any,
    tasks: List[RecipeNode],
    recipe_inputs: Dict[str, float],
    assign: Dict[Tuple[str, str], Any],
    route_select: Dict[Tuple[str, int], Any],
    route_candidates: Dict[str, List[TransferCandidate]],
    aux_candidates: List[AuxiliaryTransferCandidate],
    aux_amount: Dict[int, Any],
    aux_present: Dict[int, Any],
    active: Dict[str, Any],
    rtn: RTNModel,
    cfg: PlannerConfig,
) -> None:
    scale = int(cfg.volume_scale)
    _init_inv_if = _init_inventory_from_rtn(rtn)
    _cap_if = _capacity_from_rtn(rtn)
    materials = _all_materials(tasks, recipe_inputs, _init_inv_if)
    modules = sorted(set(_cap_if) | set(_init_inv_if) | {wb for wb, _ in assign})
    for routes in route_candidates.values():
        for route in routes:
            modules.extend([route.source_module, route.target_module])
    for aux in aux_candidates:
        modules.extend([aux.source_module, aux.target_module])
    modules = sorted(set(modules))

    max_total = sum(max(0.0, _resource_material_quantity_raw(raw)[1]) for raw in _init_inv_if.values())
    max_total += sum(float(v) for v in recipe_inputs.values())
    max_bound = max(1, int(round(max_total * scale)))

    inv_before: Dict[Tuple[int, str, str], Any] = {}
    inv_after_aux: Dict[Tuple[int, str, str], Any] = {}
    for stage in range(len(tasks) + 1):
        for wb in modules:
            cap = float((_cap_if.get(wb) or [0.0])[0])
            bound = max_bound if cap <= 0 else max(max_bound, int(round(cap * scale)))
            for material in materials:
                inv_before[(stage, wb, material)] = model.NewIntVar(0, bound, f"inv_before_{stage}_{wb}_{material}")
                inv_after_aux[(stage, wb, material)] = model.NewIntVar(0, bound, f"inv_after_aux_{stage}_{wb}_{material}")

    for wb in modules:
        init_material, init_qty = _resource_material_quantity_raw(_init_inv_if.get(wb))
        for material in materials:
            initial_i = int(round(init_qty * scale)) if init_material.upper() == material else 0
            model.Add(inv_before[(0, wb, material)] == initial_i)

    aux_by_boundary: Dict[int, List[Tuple[int, AuxiliaryTransferCandidate]]] = {}
    for idx, aux in enumerate(aux_candidates):
        if idx in aux_amount:
            aux_by_boundary.setdefault(aux.boundary_index, []).append((idx, aux))

    recipe_materials = {material.upper() for material in recipe_inputs}
    required_recipe = {material.upper(): int(round(amount * scale)) for material, amount in recipe_inputs.items()}
    and_group_start: Dict[str, int] = {}
    for task_index, task in enumerate(tasks):
        group = str(task.branch_group_id or "")
        if group.startswith("AND_"):
            and_group_start.setdefault(group, task_index)

    def aux_component_expr(idx: int, aux: AuxiliaryTransferCandidate, material: str) -> Any:
        if aux.transfer_kind == MIXTURE:
            quantity = normalize_composition(aux.composition_dict).get(material, 0.0)
            return int(round(quantity * scale)) * aux_present[idx]
        if aux.material.upper() == material:
            return aux_amount[idx]
        return 0

    # Per-stage cumulative expected materials (prefix of recipe inputs up to each stage).
    # Derive per-stage expected material vector from RTN Task coefficients.
    # Optional-branch doses are tracked per branch (not added to global cumulative)
    # so that process steps in each branch only see their own branch's materials.
    # Shared steps (after XOR join) use conditional constraints keyed on the
    # branch-selection variable (= active[task] for XOR/OR tasks).
    stage_expected: Dict[int, Dict[str, int]] = {}
    stage_optional_expected: Dict[int, Dict[str, Dict[str, Dict[str, float]]]] = {}
    _cumulative: Dict[str, float] = {}
    _xor_cumulative: Dict[str, Dict[str, Dict[str, float]]] = {}  # group→branch→{mat:amt}
    # Build (group, branch) → branch-selection-variable lookup
    _branch_var_lookup: Dict[Tuple[str, str], Any] = {}
    for _n in tasks:
        if _n.branch_group_id.startswith(("XOR_", "OR_")) and _n.branch_id:
            _branch_var_lookup[(_n.branch_group_id, _n.branch_id)] = active[_n.id]
    for _stage, _node in enumerate(tasks):
        _mat, _amt, _is_prod = _ingredient_effect(_node)
        if _is_prod and _mat and _amt > 0:
            if _node.branch_group_id.startswith(("XOR_", "OR_")):
                _g = _node.branch_group_id
                _b = _node.branch_id
                _xor_cumulative.setdefault(_g, {}).setdefault(_b, {})
                _xor_cumulative[_g][_b][_mat] = _xor_cumulative[_g][_b].get(_mat, 0.0) + _amt
            else:
                _cumulative[_mat] = _cumulative.get(_mat, 0.0) + _amt
        stage_expected[_stage] = {m: int(round(a * scale)) for m, a in _cumulative.items()}
        stage_optional_expected[_stage] = {
            group: {
                branch: dict(amounts)
                for branch, amounts in branches.items()
            }
            for group, branches in _xor_cumulative.items()
        }
    for idx, aux in enumerate(aux_candidates):
        aux_recipe_materials = set(normalize_composition(aux.composition_dict))
        if idx not in aux_present or not (aux_recipe_materials & recipe_materials):
            continue
        previous_process = next(
            (node for node in reversed(tasks[: aux.boundary_index]) if node.node_type in {"mix", "usage", "settling", "heating"}),
            None,
        )
        if previous_process is None:
            continue
        source_assign = assign.get((previous_process.id, aux.source_module))
        if source_assign is None:
            model.Add(aux_present[idx] == 0)
        else:
            model.AddImplication(aux_present[idx], source_assign)
    for stage, boundary_aux in aux_by_boundary.items():
        for left_index, (left_id, left_aux) in enumerate(boundary_aux):
            left_composition = normalize_composition(left_aux.composition_dict)
            for right_id, right_aux in boundary_aux[left_index + 1 :]:
                if left_aux.target_module != right_aux.target_module:
                    continue
                right_composition = normalize_composition(right_aux.composition_dict)
                if not compositions_proportional(left_composition, right_composition):
                    model.Add(aux_present[left_id] + aux_present[right_id] <= 1)
        for idx, aux in boundary_aux:
            if idx not in aux_present:
                continue
            composition_i = {
                material: int(round(quantity * scale))
                for material, quantity in normalize_composition(aux.composition_dict).items()
            }
            if aux.transfer_kind == PURE_MATERIAL:
                pure_material = aux.material.upper()
                for material in materials:
                    if material == pure_material:
                        continue
                    # Ordinary Draining cannot peel one component from a mixed
                    # source, and Pure Aux is not a hidden recipe Dose into a
                    # mixed destination.
                    model.Add(inv_before[(stage, aux.source_module, material)] == 0).OnlyEnforceIf(aux_present[idx])
                    model.Add(inv_before[(stage, aux.target_module, material)] == 0).OnlyEnforceIf(aux_present[idx])
            else:
                component_names = sorted(composition_i)
                for material in materials:
                    if material not in composition_i:
                        model.Add(inv_before[(stage, aux.source_module, material)] == 0).OnlyEnforceIf(aux_present[idx])
                        model.Add(inv_before[(stage, aux.target_module, material)] == 0).OnlyEnforceIf(aux_present[idx])
                for material, required_amount in composition_i.items():
                    model.Add(
                        inv_before[(stage, aux.source_module, material)] >= required_amount
                    ).OnlyEnforceIf(aux_present[idx])
                for left_index, left in enumerate(component_names):
                    for right in component_names[left_index + 1 :]:
                        left_required = composition_i[left]
                        right_required = composition_i[right]
                        # Source and any existing target material must share the
                        # exact transfer ratio.  The all-zero target naturally
                        # satisfies these equations and is therefore allowed.
                        model.Add(
                            inv_before[(stage, aux.source_module, left)] * right_required
                            == inv_before[(stage, aux.source_module, right)] * left_required
                        ).OnlyEnforceIf(aux_present[idx])
                        model.Add(
                            inv_before[(stage, aux.target_module, left)] * right_required
                            == inv_before[(stage, aux.target_module, right)] * left_required
                        ).OnlyEnforceIf(aux_present[idx])

    for stage in range(len(tasks) + 1):
        for wb in modules:
            cap = float((_cap_if.get(wb) or [0.0])[0])
            total_after_aux_terms = []
            for material in materials:
                incoming = [
                    aux_component_expr(idx, aux, material)
                    for idx, aux in aux_by_boundary.get(stage, [])
                    if aux.target_module == wb
                ]
                outgoing = [
                    aux_component_expr(idx, aux, material)
                    for idx, aux in aux_by_boundary.get(stage, [])
                    if aux.source_module == wb
                ]
                if outgoing:
                    model.Add(sum(outgoing) <= inv_before[(stage, wb, material)])
                model.Add(
                    inv_after_aux[(stage, wb, material)]
                    == inv_before[(stage, wb, material)] + sum(incoming) - sum(outgoing)
                )
                total_after_aux_terms.append(inv_after_aux[(stage, wb, material)])
            if cap:
                model.Add(sum(total_after_aux_terms) <= int(round(cap * scale)))

        if stage == len(tasks):
            continue
        node = tasks[stage]
        for wb in modules:
            cap = float((_cap_if.get(wb) or [0.0])[0])
            total_after_task_terms = []
            for material in materials:
                delta_terms = []
                if node.node_type in {"dose", "separation"}:
                    _mat, _amt, _is_prod = _ingredient_effect(node)
                    # For shared separation (not in an XOR branch), amount may
                    # depend on which XOR branch was selected — different branches
                    # dose different materials. Use a variable amount keyed on
                    # branch selection so undosed materials are skipped.
                    if (node.node_type == "separation"
                            and not node.branch_group_id.startswith(("XOR_", "OR_"))
                            and _xor_cumulative):
                        _amt_var = model.NewIntVar(
                            0, int(round(_amt * scale)), f"sep_amt_{node.id}"
                        )
                        _amt_terms = [int(round(_cumulative.get(_mat, 0.0) * scale))]
                        for _g, _branches in _xor_cumulative.items():
                            for _b, _bdoses in _branches.items():
                                _bv = _branch_var_lookup.get((_g, _b))
                                if _bv is not None:
                                    _bd = int(round(_bdoses.get(_mat, 0.0) * scale))
                                    if _bd > 0:
                                        _amt_terms.append(_bd * _bv)
                        model.Add(_amt_var == sum(_amt_terms))
                    else:
                        _amt_var = None
                    amount_i = int(round(_amt * scale)) if _mat and _amt > 0 else 0
                    if material == _mat and (amount_i > 0 or _amt_var is not None):
                        for idx, route in enumerate(route_candidates.get(node.id, [])):
                            _rvar = route_select[(node.id, idx)]
                            if _amt_var is not None:
                                _eff = model.NewIntVar(0, max(1, int(round(_amt * scale))),
                                                      f"eff_sep_{node.id}_{idx}")
                                model.Add(_eff == 0).OnlyEnforceIf(_rvar.Not())
                                model.Add(_eff == _amt_var).OnlyEnforceIf(_rvar)
                                _src_term = -_eff
                                _dst_term = _eff
                            else:
                                _src_term = -amount_i * _rvar
                                _dst_term = amount_i * _rvar
                            if route.source_module == wb:
                                delta_terms.append(_src_term)
                            if route.target_module == wb:
                                delta_terms.append(_dst_term)
                model.Add(inv_before[(stage + 1, wb, material)] == inv_after_aux[(stage, wb, material)] + sum(delta_terms))
                total_after_task_terms.append(inv_before[(stage + 1, wb, material)])
            if cap:
                model.Add(sum(total_after_task_terms) <= int(round(cap * scale)))

        if node.node_type in {"mix", "usage", "settling", "heating"}:
            # Baseline is unconditional material. Optional contributions are a
            # linear sum of every selected XOR/OR branch available at this stage.
            _base = stage_expected.get(stage, {})
            _optional = stage_optional_expected.get(stage, {})
            own_group = (
                node.branch_group_id
                if node.branch_group_id.startswith(("XOR_", "OR_")) and node.branch_id
                else ""
            )
            own_doses = (_optional.get(own_group) or {}).get(node.branch_id, {}) if own_group else {}
            for (nid, wb), avar in assign.items():
                if nid != node.id:
                    continue
                for material in materials:
                    terms: List[Any] = [_base.get(material, 0)]
                    if own_group:
                        terms.append(int(round(own_doses.get(material, 0.0) * scale)))
                    for group, branches in _optional.items():
                        if group == own_group:
                            continue
                        for branch, branch_doses in branches.items():
                            branch_var = _branch_var_lookup.get((group, branch))
                            amount = int(round(branch_doses.get(material, 0.0) * scale))
                            if branch_var is not None and amount:
                                terms.append(amount * branch_var)
                    model.Add(inv_after_aux[(stage, wb, material)] == sum(terms)).OnlyEnforceIf(avar)
        if node.node_type == "separation":
            sep_material, _, _ = _ingredient_effect(node)
            for idx, route in enumerate(route_candidates.get(node.id, [])):
                rvar = route_select[(node.id, idx)]
                for material in materials:
                    if material != sep_material:
                        model.Add(inv_after_aux[(stage, route.target_module, material)] == 0).OnlyEnforceIf(rvar)

        if node.node_type == "dose":
            dose_material, dose_amount, _ = _ingredient_effect(node)
            if not dose_material:
                continue
            dose_amount_i = int(round(dose_amount * scale))
            source_stage = and_group_start.get(str(node.branch_group_id or ""), stage)
            for idx, route in enumerate(route_candidates.get(node.id, [])):
                rvar = route_select[(node.id, idx)]
                model.Add(
                    inv_after_aux[(source_stage, route.source_module, dose_material)] >= dose_amount_i
                ).OnlyEnforceIf(rvar)
                for material in materials:
                    if material != dose_material:
                        model.Add(
                            inv_after_aux[(source_stage, route.source_module, material)] == 0
                        ).OnlyEnforceIf(rvar)
                if route.source_module == route.target_module:
                    continue
                for material in materials:
                    if material != dose_material and material not in recipe_materials:
                        model.Add(
                            inv_after_aux[(stage, route.target_module, material)] == 0
                        ).OnlyEnforceIf(rvar)


def _ingredient_effect(node: Any) -> Tuple[str, float, bool]:
    """Extract (material_name, amount, is_production) from the task's RTN coefficients.

    Returns ("", 0.0, False) for tasks without ingredient effects (mix, usage,
    settling, heating). For dose: is_production=True. For separation:
    is_production=False.
    """
    task = getattr(node, 'task', None)
    if task is None:
        return "", 0.0, False
    for coef in task.coefficients:
        _idx = coef.resource_id.rfind("state.ingredient.")
        if _idx >= 0:
            material = coef.resource_id[_idx + len("state.ingredient."):]
            amount = abs(coef.coefficient)
            is_production = coef.coefficient > 0
            return material, amount, is_production
    return "", 0.0, False


def _all_materials(tasks: List[RecipeNode], recipe_inputs: Dict[str, float], init_inventory: Dict[str, Any]) -> List[str]:
    materials = {m.upper() for m in recipe_inputs}
    for raw in init_inventory.values():
        material, _ = _resource_material_quantity_raw(raw)
        if material:
            materials.add(material.upper())
    for node in tasks:
        if node.node_type in {"dose", "separation"}:
            material = str(node.params.get("ingredient", "")).upper()
            if material:
                materials.add(material)
    return sorted(materials)


def _enforce_transfer_inventory_capacity(
    model: Any,
    route_select: Dict[Tuple[str, int], Any],
    route_candidates: Dict[str, List[TransferCandidate]],
    rtn: RTNModel,
    cfg: PlannerConfig,
) -> None:
    scale = int(cfg.volume_scale)
    _init_inv_etc = _init_inventory_from_rtn(rtn)
    _cap_etc = _capacity_from_rtn(rtn)
    module_ids = set(_cap_etc) | set(_init_inv_etc)
    materials = {
        _resource_material_quantity(_init_inv_etc, wb)[0].upper()
        for wb in _init_inv_etc
        if _resource_material_quantity(_init_inv_etc, wb)[0]
    }
    for routes in route_candidates.values():
        for route in routes:
            module_ids.add(route.source_module)
            module_ids.add(route.target_module)
            materials.add(route.material.upper())

    for wb in sorted(module_ids):
        initial_total = int(round(_resource_quantity(_init_inv_etc, wb) * scale))
        total_terms = [initial_total]
        for node_id, routes in route_candidates.items():
            for idx, route in enumerate(routes):
                amount_i = int(round(route.amount_l * scale))
                if route.target_module == wb:
                    total_terms.append(route_select[(node_id, idx)] * amount_i)
                if route.source_module == wb:
                    total_terms.append(route_select[(node_id, idx)] * -amount_i)
        cap = float((_cap_etc.get(wb) or [0.0])[0])
        if cap:
            model.Add(sum(total_terms) <= int(round(cap * scale)))

        for material in sorted(m for m in materials if m):
            initial_material, initial_qty = _resource_material_quantity(_init_inv_etc, wb)
            initial_i = int(round(initial_qty * scale)) if initial_material.upper() == material else 0
            material_terms = [initial_i]
            for node_id, routes in route_candidates.items():
                for idx, route in enumerate(routes):
                    if route.material.upper() != material:
                        continue
                    amount_i = int(round(route.amount_l * scale))
                    if route.target_module == wb:
                        material_terms.append(route_select[(node_id, idx)] * amount_i)
                    if route.source_module == wb:
                        material_terms.append(route_select[(node_id, idx)] * -amount_i)
            model.Add(sum(material_terms) >= 0)


def _ports(module_interfaces: Dict[str, List[Tuple[str, str]]], wb: str, port_type: str) -> List[str]:
    if module_interfaces:
        return [name for typ, name in module_interfaces.get(wb, []) if typ == port_type]
    suffix = "Out1" if port_type == "Output" else "In1"
    return [f"{wb}_{suffix}"]


def _auxiliary_port_pairs(
    module_interfaces: Dict[str, List[Tuple[str, str]]],
    source_module: str,
    target_module: str,
) -> List[Tuple[str, str]]:
    """Return a compact, physically distinct auxiliary-route port set.

    Inventory feasibility depends on source/target/material and amount, not on
    every Cartesian port permutation. Keeping one canonical pair and one
    alternate whenever either endpoint has a spare port preserves routing
    flexibility without hundreds of symmetric candidates.
    """
    outputs = _ports(module_interfaces, source_module, "Output")
    inputs = _ports(module_interfaces, target_module, "Input")
    if not outputs or not inputs:
        return []
    pairs = [(outputs[0], inputs[0])]
    if len(outputs) > 1 or len(inputs) > 1:
        pairs.append((outputs[min(1, len(outputs) - 1)], inputs[min(1, len(inputs) - 1)]))
    return pairs


def _auto_volume_scale(tasks: List[TaskNodeView], module_ops: Dict[str, List[Tuple[str, Any, float]]]) -> int:
    """Return minimal volume_scale from dose amounts and operation parameters."""
    max_dec = 0
    for task in tasks:
        if task.node_type == "dose":
            amt = float((task.params or {}).get("amount_L", 0))
            s = f"{amt:.10g}"
            dec = len(s.split(".")[1]) if "." in s else 0
            max_dec = max(max_dec, dec)
    for ops in (module_ops or {}).values():
        for op in ops:
            param = op[1] if len(op) > 1 else None
            if isinstance(param, float):
                s = f"{param:.10g}"
                dec = len(s.split(".")[1]) if "." in s else 0
                max_dec = max(max_dec, dec)
    return 10 ** max_dec if max_dec > 0 else 10


def _resource_contains(resources: Dict[str, List[Any]], wb: str, material: str, amount: float) -> bool:
    raw_material, raw_qty = _resource_material_quantity(resources, wb)
    return raw_material.upper() == material.upper() and raw_qty >= amount


def _resource_quantity(resources: Dict[str, List[Any]], wb: str) -> float:
    return _resource_material_quantity(resources, wb)[1]


def _resource_material_quantity(resources: Dict[str, List[Any]], wb: str) -> Tuple[str, float]:
    return _resource_material_quantity_raw(resources.get(wb))


def _resource_material_quantity_raw(raw: Any) -> Tuple[str, float]:
    if not raw:
        return "", 0.0
    if isinstance(raw, dict):
        return str(raw.get("material", "")), float(raw.get("quantity", 0.0))
    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        return str(raw[0]), float(raw[1])
    return "", 0.0


def _operation_record(
    wb: str,
    operation: str,
    module_ops: Dict[str, List[Tuple]],
    parameter: Any = None,
) -> Tuple | None:
    matches = [record for record in module_ops.get(wb, []) if record and str(record[0]) == operation]
    if parameter is not None:
        exact = [record for record in matches if len(record) > 1 and str(record[1]) == str(parameter)]
        if exact:
            return exact[0]
    return matches[0] if matches else None


def _record_metric(record: Tuple | None, index: int) -> float:
    if record is None or len(record) <= index:
        return 0.0
    try:
        return float(record[index] or 0)
    except (TypeError, ValueError):
        return 0.0


def _capability_metrics(
    wb: str,
    operation: str,
    module_ops: Dict[str, List[Tuple]],
    parameter: Any = None,
) -> Tuple[float, float, float]:
    record = _operation_record(wb, operation, module_ops, parameter)
    return _record_metric(record, 2), _record_metric(record, 3), _record_metric(record, 4)


def _cost_breakdown(
    usage_cost: float,
    energy_rate_kwh_s: float,
    co2_rate_kg_s: float,
    duration_s: float,
    cfg: PlannerConfig,
) -> CostBreakdown:
    duration = max(0.0, float(duration_s))
    energy = max(0.0, float(energy_rate_kwh_s)) * duration
    energy_cost = energy * cfg.electricity_price_eur_per_kwh
    co2 = max(0.0, float(co2_rate_kg_s)) * duration
    usage = max(0.0, float(usage_cost))
    return CostBreakdown(
        usage_cost=usage,
        energy_consumption_kwh=energy,
        energy_cost=energy_cost,
        co2_emissions_kg=co2,
        weighted_usage_cost=usage * cfg.usage_cost_weight,
        weighted_energy_cost=energy_cost * cfg.energy_cost_weight,
        weighted_co2_cost=co2 * cfg.co2_penalty,
    )


def _combine_costs(*costs: CostBreakdown) -> CostBreakdown:
    return CostBreakdown(**{
        field_name: sum(getattr(item, field_name) for item in costs)
        for field_name in (
            "usage_cost",
            "energy_consumption_kwh",
            "energy_cost",
            "co2_emissions_kg",
            "weighted_usage_cost",
            "weighted_energy_cost",
            "weighted_co2_cost",
        )
    })


def _has_op(wb: str, operation: str, module_ops: Dict[str, List[Tuple]]) -> bool:
    return _operation_record(wb, operation, module_ops) is not None


def _nominal_volume(node: RecipeNode) -> float:
    if node.node_type in {"dose", "separation"}:
        _, amt, _ = _ingredient_effect(node)
        return amt
    return float(node.params.get("volume_L", 0.0))


def _operation_param(wb: str, operation: str, module_ops: Dict[str, List[Tuple]]) -> Any:
    record = _operation_record(wb, operation, module_ops)
    return record[1] if record is not None and len(record) > 1 else None


def _operation_capability(node: RecipeNode) -> Tuple[str, Any]:
    if node.node_type == "mix":
        return "Stirring", node.params.get("rpm", "")
    return {
        "settling": ("Settling", None),
        "heating": ("Heating", None),
        "usage": ("None", None),
    }.get(node.node_type, ("None", None))


def _operation_costs(
    node: RecipeNode,
    wb: str,
    module_ops: Dict[str, List[Tuple]],
    duration_s: float,
    cfg: PlannerConfig,
) -> CostBreakdown:
    operation, parameter = _operation_capability(node)
    usage, energy_rate, co2_rate = _capability_metrics(wb, operation, module_ops, parameter)
    return _cost_breakdown(usage, energy_rate, co2_rate, duration_s, cfg)


def _operation_cost(node: RecipeNode, wb: str, module_ops: Dict[str, List[Tuple]]) -> float:
    """Unweighted usage cost per invocation; retained for legacy callers."""
    if node.node_type == "dose":
        return _cost(wb, "Draining", module_ops) + _cost(wb, "Filling", module_ops)
    if node.node_type == "separation":
        return _cost(wb, "Draining", module_ops) + _cost(wb, "Filling", module_ops)
    if node.node_type == "mix":
        rpm = str(node.params.get("rpm", ""))
        return _capability_metrics(wb, "Stirring", module_ops, rpm)[0]
    if node.node_type == "settling":
        return _cost(wb, "Settling", module_ops)
    if node.node_type == "heating":
        return _cost(wb, "Heating", module_ops)
    if node.node_type == "usage":
        return _cost(wb, "None", module_ops)
    return 0.0


def _cost(wb: str, operation: str, module_ops: Dict[str, List[Tuple]]) -> float:
    return _capability_metrics(wb, operation, module_ops)[0]


def _operation_label(node: RecipeNode, wb: str, route: Optional[TransferCandidate] = None) -> str:
    if node.node_type == "dose":
        if route:
            return (
                f"Dosing: Open Valve of {route.out_port} only, "
                f"Draining({route.source_module}), Filling({route.target_module}), "
                f"({route.material}: {route.amount_l:.1f} litre)"
            )
        return f"Dosing ({wb}), {node.params['ingredient']}: {float(node.params['amount_L']):.1f} litre"
    if node.node_type == "mix":
        return f"Stirring ({wb}), {int(node.params['rpm'])}rpm for {float(node.params['duration_s']):.1f}s"
    if node.node_type == "usage":
        return f"Usage ({wb}), {float(node.params['duration_s']):.1f}s: None"
    if node.node_type == "settling":
        return f"Settling ({wb}), {float(node.params['duration_s']):.1f}s: Settling"
    if node.node_type == "heating":
        return f"Heating ({wb}), {float(node.params['duration_s']):.1f}s: Heating"
    if node.node_type == "separation":
        if route:
            return (
                f"Separation: Open Valve of {route.out_port} only, "
                f"Draining({route.source_module}), Filling({route.target_module}), "
                f"({route.material}: {route.amount_l:.1f} litre)"
            )
        return f"Separation ({wb}), {node.params['ingredient']}: {float(node.params['amount_L']):.1f} litre"
    return f"{node.name} ({wb})"


def _material_for_node(
    node: RecipeNode,
    selected_branches: Optional[Dict[str, Any]] = None,
    tasks: Optional[List[Any]] = None,
) -> Dict[str, float]:
    if node.node_type not in {"dose", "separation"}:
        return {}
    mat, amt, _ = _ingredient_effect(node)
    if not mat or amt <= 0:
        return {}
    # For shared separations (not in an XOR branch), the effective amount
    # depends on which XOR branches were selected. Subtract doses from
    # unselected branches so the validator sees the correct amount.
    if (node.node_type == "separation"
            and selected_branches
            and tasks
            and not node.branch_group_id.startswith(("XOR_", "OR_"))):
        effective = amt
        for _t in tasks:
            if (_t.node_type == "dose"
                    and _t.branch_group_id.startswith(("XOR_", "OR_"))
                    and _t.branch_id):
                _selected = selected_branches.get(_t.branch_group_id)
                _is_selected = (
                    _t.branch_id in _selected
                    if isinstance(_selected, (list, tuple, set))
                    else _selected == _t.branch_id
                )
                if _selected is not None and not _is_selected:
                    _t_mat, _t_amt, _ = _ingredient_effect(_t)
                    if _t_mat == mat:
                        effective -= _t_amt
        amt = max(0.0, effective)
    return {mat: amt} if mat and amt > 0 else {}
