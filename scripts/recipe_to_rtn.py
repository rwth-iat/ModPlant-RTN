"""Adapter: ``RecipeIR`` + plant configuration → ``RTNModel``.

The user-facing recipe layer (``RecipeIR``) describes *what* should happen.
The solver-facing layer (``RTNModel``) describes the abstract scheduling
problem. This module bridges the two by lifting recipe nodes into RTN tasks
and lifting plant equipment, ports, and ingredients into RTN resources.

Plant-level configuration that has no clean place in pure RTN (port routing
maps, per-port-pair connection paths, capability lists, etc.) is preserved
verbatim under ``RTNModel.metadata['plant']`` so the CP-SAT planner can read
it back without losing fidelity.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

try:
    from recipe_ir import RecipeIR, RecipeNode
    from rtn import (
        ProductionCoefficient,
        Resource,
        RTNModel,
        Task,
    )
except ImportError:  # pragma: no cover
    from recipe_ir import RecipeIR, RecipeNode
    from rtn import ProductionCoefficient, Resource, RTNModel, Task


# Synthetic state-resource name templates.
# BATCH_PREFIX is empty for single-batch mode. For multi-batch scheduling,
# set it to f"batch.{batch_id}." so that each batch's ingredient states are
# isolated (e.g. "batch.001.state.ingredient.A" vs "batch.002.state.ingredient.A").
BATCH_PREFIX = ""
INGREDIENT_STATE_PREFIX = f"{BATCH_PREFIX}state.ingredient."
MIXED_STATE = f"{BATCH_PREFIX}state.mixed"
PRODUCT_STATE = f"{BATCH_PREFIX}state.product"


def _ingredient_state_id(ingredient: str) -> str:
    """Return the RTN resource id for an ingredient state (batch-aware)."""
    return f"{BATCH_PREFIX}state.ingredient.{ingredient}"


def _is_ingredient_state(resource_id: str) -> bool:
    """Check whether *resource_id* names an ingredient state resource."""
    return "state.ingredient." in resource_id


def _task_duration_s(node: RecipeNode) -> int:
    """Return the structural duration in seconds for a recipe task node.

    Mix/usage/settling carry an explicit ``duration_s``; dose and separation
    have variable transfer-driven duration computed by the solver. We use 0
    here so the RTN ``duration_s`` represents only the *core* operation
    duration; transfer/connect time is added by the solver.
    """
    params = node.params or {}
    if "duration_s" in params:
        return int(params["duration_s"])
    return 0


def _coefficients_for(node: RecipeNode) -> Tuple[ProductionCoefficient, ...]:
    """Derive RTN production coefficients from recipe-node semantics."""
    params = node.params or {}
    if node.node_type == "dose":
        ingredient = str(params.get("ingredient", "")).upper()
        amount = float(params.get("amount_L", 0.0))
        if not ingredient or amount <= 0:
            return ()
        return (
            ProductionCoefficient(resource_id=_ingredient_state_id(ingredient),
                                  coefficient=+amount, time_offset="end"),
        )
    if node.node_type == "mix":
        # Mix consumes nothing explicit (ingredients persist as state until separation)
        # but marks the material as "mixed" so usage can run.
        return (
            ProductionCoefficient(resource_id=MIXED_STATE, coefficient=+1.0, time_offset="end"),
        )
    if node.node_type == "usage":
        return (
            ProductionCoefficient(resource_id=MIXED_STATE, coefficient=-1.0, time_offset="start"),
            ProductionCoefficient(resource_id=PRODUCT_STATE, coefficient=+1.0, time_offset="end"),
        )
    if node.node_type in {"settling", "heating"}:
        return ()
    if node.node_type == "separation":
        ingredient = str(params.get("ingredient", "")).upper()
        amount = float(params.get("amount_L", 0.0))
        if not ingredient or amount <= 0:
            return ()
        return (
            ProductionCoefficient(resource_id=_ingredient_state_id(ingredient),
                                  coefficient=-amount, time_offset="end"),
        )
    return ()


def _collect_ingredient_states(ir: RecipeIR) -> Dict[str, float]:
    """Sum dose volumes per ingredient → max state-resource capacity."""
    totals: Dict[str, float] = {}
    for node in ir.nodes.values():
        if node.node_type != "dose":
            continue
        params = node.params or {}
        ingredient = str(params.get("ingredient", "")).upper()
        amount = float(params.get("amount_L", 0.0))
        if not ingredient or amount <= 0:
            continue
        totals[ingredient] = totals.get(ingredient, 0.0) + amount
    return totals


def _filter_eligible_modules(
    node: RecipeNode,
    module_ids: List[str],
    module_ops: Optional[Mapping[str, Any]],
    module_maximum_volume: Optional[Mapping[str, Any]],
    nominal_volume: float,
) -> List[str]:
    """Return the subset of *module_ids* that can host *node*."""
    # Dose and separation use per-route source/target eligibility; the
    # task-level eligible_resources sets the universe. The solver narrows
    # further via route_candidates.
    if node.node_type in {"dose", "separation"}:
        return list(module_ids)

    needed_first = {
        "mix": "Stirring",
        "usage": "None",
        "settling": "Settling",
        "heating": "Heating",
    }.get(node.node_type)
    if needed_first is None:
        return list(module_ids)

    feasible: List[str] = []
    for wb in module_ids:
        ops = (module_ops or {}).get(wb, [])
        op_names = {str(op[0]) for op in ops}
        if node.node_type == "mix":
            rpm = str(node.params.get("rpm", ""))
            if not any(op[0] == "Stirring" and str(op[1]) == rpm for op in ops):
                continue
        elif needed_first not in op_names:
            continue
        if node.node_type in {"mix", "usage", "settling", "heating"} and module_maximum_volume:
            cap = float((module_maximum_volume.get(wb) or [0])[0])
            if cap and cap < nominal_volume:
                continue
        feasible.append(wb)
    return feasible


def recipe_ir_to_rtn(
    ir: RecipeIR,
    *,
    module_ops: Optional[Mapping[str, Any]] = None,
    module_interfaces: Optional[Mapping[str, Any]] = None,
    module_maximum_volume: Optional[Mapping[str, Any]] = None,
    module_resources: Optional[Mapping[str, Any]] = None,
    horizon_s: int = 86_400,
) -> RTNModel:
    """Lift a RecipeIR + plant config into an abstract RTNModel.

    Plant configuration is preserved under ``model.metadata['plant']`` for
    consumption by domain-aware solvers (the CP-SAT planner uses port-routing
    and capability info that does not encode cleanly into pure RTN).
    """
    model = RTNModel(horizon_s=int(horizon_s))

    # ----- Resources: equipment (modules) -----
    module_ids: List[str] = []
    if module_ops is not None:
        module_ids = sorted(module_ops.keys())
    elif module_resources is not None:
        module_ids = sorted(module_resources.keys())

    for module in module_ids:
        capacity = None
        if module_maximum_volume and module in module_maximum_volume:
            volumes = module_maximum_volume[module]
            if isinstance(volumes, (list, tuple)) and volumes:
                capacity = float(max(volumes))
            elif isinstance(volumes, (int, float)):
                capacity = float(volumes)
        model.add_resource(
            Resource(id=module, kind="equipment", capacity=1.0, metadata={"max_volume_L": capacity})
        )

    # ----- Resources: ports (collected from interfaces) -----
    # Interface entries are (direction, port_name), e.g. ("Input", "HC10_In1").
    port_ids = set()
    if module_interfaces:
        for module, interfaces in module_interfaces.items():
            for entry in interfaces or []:
                if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                    port_ids.add(f"{module}.{entry[1]}")
    for pid in sorted(port_ids):
        model.add_resource(Resource(id=pid, kind="port", capacity=1.0))

    # ----- Resources: material states -----
    ingredient_totals = _collect_ingredient_states(ir)
    for ingredient, total in ingredient_totals.items():
        model.add_resource(
            Resource(
                id=_ingredient_state_id(ingredient),
                kind="state",
                capacity=total,
                initial_level=0.0,
                metadata={"is_ingredient": True, "ingredient_name": ingredient,
                          "batch_id": "default"},
            )
        )

    # mixed / product synthetic states (binary-ish flags for usage/separation gating)
    if any(n.node_type == "mix" for n in ir.nodes.values()):
        model.add_resource(Resource(id=MIXED_STATE, kind="state", capacity=None, initial_level=0.0))
    if any(n.node_type == "usage" for n in ir.nodes.values()):
        model.add_resource(Resource(id=PRODUCT_STATE, kind="state", capacity=None, initial_level=0.0))

    # ----- Resources: per-module initial inventory -----
    # Each module × ingredient pair becomes a state resource whose initial_level
    # carries the starting inventory. The solver reads these directly instead of
    # the module_resources metadata dict — making Resource the single source of
    # truth for all material data.
    INIT_INVENTORY_PREFIX = "init.inventory."
    if module_resources:
        for module in module_ids:
            raw = module_resources.get(module)
            material, qty = "", 0.0
            if isinstance(raw, dict):
                material = str(raw.get("material", "")).upper()
                qty = float(raw.get("quantity", 0.0))
            elif isinstance(raw, (list, tuple)) and len(raw) >= 2:
                material = str(raw[0]).upper()
                qty = float(raw[1])
            if material and qty > 0:
                rid = f"{INIT_INVENTORY_PREFIX}{module}.{material}"
                model.add_resource(
                    Resource(
                        id=rid, kind="state", capacity=qty, initial_level=qty,
                        metadata={"module": module, "ingredient": material,
                                  "is_initial_inventory": True, "batch_id": "default"},
                    )
                )

    # ----- Tasks -----
    _nominal_vol = sum(_collect_ingredient_states(ir).values())
    for node in ir.nodes.values():
        if not node.is_task:
            continue
        eligible = frozenset(_filter_eligible_modules(node, module_ids, module_ops, module_maximum_volume, _nominal_vol))
        task = Task(
            id=node.id,
            task_type=node.node_type,
            duration_s=_task_duration_s(node),
            duration_mode="fixed" if node.node_type in {"mix", "usage", "settling", "heating"} else "min",
            coefficients=_coefficients_for(node),
            eligible_resources=eligible,
            branch_group_id=node.branch_group_id or None,
            branch_id=node.branch_id or None,
            metadata={
                "recipe_node_name": node.name,
                "semantic_uri": node.semantic_uri,
                "params": dict(node.params),
            },
        )
        model.add_task(task)

    # ----- Precedence (task→task, control nodes already skipped) -----
    model.precedence = list(ir.task_precedence_edges())

    # ----- Choice groups -----
    model.choice_groups = ir.choice_groups()
    for node in ir.nodes.values():
        if node.node_type != "or_split" or not node.branch_group_id:
            continue
        branches = model.choice_groups.get(node.branch_group_id, [])
        model.choice_group_policies[node.branch_group_id] = {
            "type": "OR",
            "minBranches": int(node.params.get("minBranches", 1)),
            "maxBranches": int(node.params.get("maxBranches", len(branches))),
            "branchConditions": dict(node.params.get("branchConditions", {})),
        }

    # ----- Plant config preserved as metadata -----
    plant: Dict[str, Any] = {}
    if module_ops is not None:
        plant["module_ops"] = {k: list(v) for k, v in module_ops.items()}
    if module_interfaces is not None:
        plant["module_interfaces"] = {k: list(v) for k, v in module_interfaces.items()}
    if module_maximum_volume is not None:
        plant["module_maximum_volume"] = {k: list(v) if isinstance(v, (list, tuple)) else v for k, v in module_maximum_volume.items()}
    if module_resources is not None:
        plant["module_resources"] = {k: list(v) for k, v in module_resources.items()}
    model.metadata["plant"] = plant
    model.metadata["recipe_id"] = ir.id
    model.metadata["recipe_volume_L"] = ir.volume
    model.metadata["recipe_inputs"] = dict(ir.inputs)
    model.metadata["recipe_outputs"] = dict(ir.outputs)
    # Preserve the complete control graph for graph-aware material cuts.  RTN
    # task precedence intentionally omits split/join nodes, but auxiliary
    # inventory transfers must still know that an AND region is atomic: a
    # topological list position between sibling branches is not a legal process
    # boundary.
    model.metadata["control_flow"] = {
        "nodes": {
            node.id: {
                "node_type": node.node_type,
                "branch_group_id": node.branch_group_id,
                "branch_id": node.branch_id,
                "control_node_type": node.control_node_type,
                "join_policy": node.join_policy,
            }
            for node in ir.nodes.values()
        },
        "edges": [list(edge) for edge in ir.edges],
    }

    return model


def rtn_to_recipe_ir(model: RTNModel) -> RecipeIR:
    """Reconstruct a ``RecipeIR`` view sufficient for the CP-SAT planner.

    The RecipeIR returned exposes exactly the four surfaces the solver reads:
    ``task_nodes()``, ``choice_groups()``, ``task_precedence_edges()``, and
    ``.inputs``. Control nodes are not reconstructed (the planner does not
    traverse them). Round-trip identity is therefore not guaranteed for any
    code that walks ``ir.nodes`` / ``ir.edges`` directly.

    Use this when an RTN-first call site needs to delegate into the existing
    CP-SAT solver implementation. The full RTN-native solver rewrite is a
    separate work item.
    """
    ir = RecipeIR(
        id=str(model.metadata.get("recipe_id", "RTN_Recipe")),
        volume=float(model.metadata.get("recipe_volume_L", 0.0)),
        inputs=dict(model.metadata.get("recipe_inputs", {})),
        outputs=dict(model.metadata.get("recipe_outputs", {})),
        metadata=dict(model.metadata),
    )
    ir.metadata["choiceGroupPolicies"] = {
        group: dict(policy) for group, policy in model.choice_group_policies.items()
    }
    for tid, task in model.tasks.items():
        meta = task.metadata or {}
        node = RecipeNode(
            id=tid,
            node_type=task.task_type,
            name=str(meta.get("recipe_node_name", tid)),
            params=dict(meta.get("params", {})),
            semantic_uri=str(meta.get("semantic_uri", "")),
            control_node_type="",
            branch_group_id=task.branch_group_id or "",
            branch_id=task.branch_id or "",
            join_policy="",
        )
        ir.nodes[tid] = node
    # Edges = task→task precedence (control nodes are not reconstructed).
    ir.edges = list(model.precedence)
    return ir
