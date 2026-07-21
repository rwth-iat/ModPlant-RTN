from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple


SEMANTIC_URIS = {
    "dose": "http://www.iat.rwth-aachen.de/capability-ontology#Dosing",
    "mix": "http://www.iat.rwth-aachen.de/capability-ontology#MixingOfLiquids",
    "usage": "http://www.iat.rwth-aachen.de/capability-ontology#Usage",
    "settling": "http://www.iat.rwth-aachen.de/capability-ontology#Settling",
    "separation": "http://www.iat.rwth-aachen.de/capability-ontology#Separation",
    "and_split": "urn:modplant:control#AND_SPLIT",
    "and_join": "urn:modplant:control#AND_JOIN",
    "xor_split": "urn:modplant:control#XOR_SPLIT",
    "xor_join": "urn:modplant:control#XOR_JOIN",
    "or_split": "urn:modplant:control#OR_SPLIT",
    "or_join": "urn:modplant:control#OR_JOIN",
}


@dataclass(frozen=True)
class RecipeNode:
    id: str
    node_type: str
    name: str
    params: Dict[str, Any] = field(default_factory=dict)
    semantic_uri: str = ""
    control_node_type: str = ""
    branch_group_id: str = ""
    branch_id: str = ""
    join_policy: str = ""

    @property
    def is_control(self) -> bool:
        return self.node_type in {"and_split", "and_join", "xor_split", "xor_join", "or_split", "or_join"}

    @property
    def is_task(self) -> bool:
        return not self.is_control


@dataclass
class RecipeIR:
    id: str
    volume: float
    nodes: Dict[str, RecipeNode] = field(default_factory=dict)
    edges: List[Tuple[str, str]] = field(default_factory=list)
    inputs: Dict[str, float] = field(default_factory=dict)
    outputs: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def successors(self) -> Dict[str, List[str]]:
        succ = defaultdict(list)
        for src, dst in self.edges:
            succ[src].append(dst)
        return dict(succ)

    def predecessors(self) -> Dict[str, List[str]]:
        pred = defaultdict(list)
        for src, dst in self.edges:
            pred[dst].append(src)
        return dict(pred)

    def topological_nodes(self) -> List[RecipeNode]:
        indeg = {nid: 0 for nid in self.nodes}
        succ = defaultdict(list)
        for src, dst in self.edges:
            succ[src].append(dst)
            indeg[dst] = indeg.get(dst, 0) + 1
        queue = deque(sorted(nid for nid, deg in indeg.items() if deg == 0))
        out: List[RecipeNode] = []
        while queue:
            nid = queue.popleft()
            out.append(self.nodes[nid])
            for dst in sorted(succ[nid]):
                indeg[dst] -= 1
                if indeg[dst] == 0:
                    queue.append(dst)
        if len(out) != len(self.nodes):
            raise ValueError("RecipeIR graph contains a cycle or unresolved node.")
        return out

    def task_nodes(self) -> List[RecipeNode]:
        return [node for node in self.topological_nodes() if node.is_task]

    def choice_groups(self) -> Dict[str, List[str]]:
        groups: Dict[str, set[str]] = defaultdict(set)
        for node in self.nodes.values():
            if node.branch_group_id and node.branch_id:
                groups[node.branch_group_id].add(node.branch_id)
        return {gid: sorted(branches) for gid, branches in groups.items()}

    def task_precedence_edges(self) -> List[Tuple[str, str]]:
        """Return immediate task-to-task precedence, skipping control nodes."""
        succ = self.successors()
        out: set[Tuple[str, str]] = set()
        for node in self.nodes.values():
            if not node.is_task:
                continue
            queue = deque(succ.get(node.id, []))
            seen = set()
            while queue:
                nid = queue.popleft()
                if nid in seen:
                    continue
                seen.add(nid)
                other = self.nodes[nid]
                if other.is_task:
                    out.add((node.id, other.id))
                    continue
                queue.extend(succ.get(nid, []))
        return sorted(out)


class _IRBuilder:
    def __init__(self, recipe_id: str, volume: float):
        self.ir = RecipeIR(id=recipe_id, volume=float(volume))
        self._counts: Dict[str, int] = defaultdict(int)

    def add_node(
        self,
        node_type: str,
        name: str,
        params: Optional[Dict[str, Any]] = None,
        *,
        branch_group_id: str = "",
        branch_id: str = "",
        control_node_type: str = "",
        join_policy: str = "",
    ) -> str:
        self._counts[node_type] += 1
        nid = f"{node_type}_{self._counts[node_type]:03d}"
        semantic_uri = SEMANTIC_URIS.get(node_type, "")
        node = RecipeNode(
            id=nid,
            node_type=node_type,
            name=name,
            params=params or {},
            semantic_uri=semantic_uri,
            control_node_type=control_node_type,
            branch_group_id=branch_group_id,
            branch_id=branch_id,
            join_policy=join_policy,
        )
        self.ir.nodes[nid] = node
        return nid

    def add_edge(self, src: str, dst: str) -> None:
        if src != dst and (src, dst) not in self.ir.edges:
            self.ir.edges.append((src, dst))


def recipe_spec_from_legacy_order(
    order: Dict[str, Any],
    *,
    parallel_dosing_before_mix: bool = False,
    usage_alternatives: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Convert the existing linear notebook order dict into the new recipe spec."""
    volume = float(order["volume"])
    order_list = order.get("order", [])
    ratio = order.get("ratio", {})

    dose_items = [x for x in order_list if isinstance(x, str)]
    weights: List[float] = []
    used = defaultdict(int)
    for ingr in dose_items:
        used[ingr] += 1
        weights.append(float(ratio[ingr][used[ingr] - 1]))
    total_weight = sum(weights) or 1.0

    occurrence_amounts: List[Tuple[str, float]] = []
    for ingr, weight in zip(dose_items, weights):
        occurrence_amounts.append((ingr, volume * weight / total_weight))

    procedure: List[Dict[str, Any]] = []
    pending_doses: List[Dict[str, Any]] = []
    dose_idx = 0
    for item in order_list:
        if isinstance(item, str):
            ingr, amount = occurrence_amounts[dose_idx]
            dose_idx += 1
            step = {"dose": {"ingredient": ingr, "amount_L": amount}}
            if parallel_dosing_before_mix:
                pending_doses.append(step)
            else:
                procedure.append(step)
            continue
        if isinstance(item, dict) and "mix" in item:
            if pending_doses:
                procedure.append({"parallel": pending_doses, "join": "wait_all"})
                pending_doses = []
            mix = item["mix"]
            procedure.append({"mix": {"rpm": int(mix["rpm"]), "duration_s": int(mix["duration"])}})

    if pending_doses:
        procedure.extend(pending_doses)

    usage_s, settling_s = (order.get("usage_and_settling") or [3600, 300])
    if usage_alternatives:
        procedure.append({"choice": usage_alternatives, "select": "optimizer"})
    else:
        procedure.append({"usage": {"duration_s": int(usage_s)}})
    procedure.append({"settling": {"duration_s": int(settling_s)}})
    procedure.append({"separation": {"order": order.get("separation_order", list(reversed(dose_items)))}})

    return {
        "id": order.get("id", "Recipe_From_Legacy_Order"),
        "volume": volume,
        "procedure": procedure,
        "metadata": {"source": "legacy_order", "legacy_order": order},
    }


def auto_enrich_recipe_spec(
    spec: Dict[str, Any],
    *,
    parallel_dosing: bool = True,
    usage_alternatives: bool = True,
) -> Dict[str, Any]:
    """Auto-enrich a flat recipe_spec with parallel dosing and usage alternatives.

    When ``parallel_dosing`` is enabled, consecutive ``dose`` steps immediately
    before a ``mix`` are wrapped in
    ``{"parallel": [...], "join": "wait_all"}``.  Every top-level ``usage``
    step is wrapped in a ``choice`` block with *fast* (1800 s) and *standard*
    (original duration) branches when ``usage_alternatives`` is enabled, so the
    CP-SAT planner can pick the best option.
    """
    import copy
    spec = copy.deepcopy(spec)
    procedure = list(spec.get("procedure", []))
    enriched = []
    i, n = 0, len(procedure)
    while i < n:
        step = procedure[i]
        if "dose" in step:
            dose_group = [step]
            j = i + 1
            while j < n and "dose" in procedure[j]:
                dose_group.append(procedure[j])
                j += 1
            if parallel_dosing and j < n and "mix" in procedure[j]:
                enriched.append({"parallel": list(dose_group), "join": "wait_all"})
                enriched.append(procedure[j])
                i = j + 1
            else:
                enriched.extend(dose_group)
                i = j
        elif "usage" in step and usage_alternatives:
            original_duration = step["usage"]["duration_s"]
            enriched.append({
                "choice": [
                    {"branch_id": "fast", "steps": [{"usage": {"duration_s": 1800}}]},
                    {"branch_id": "standard", "steps": [{"usage": {"duration_s": original_duration}}]},
                ],
                "select": "optimizer",
            })
            i += 1
        else:
            enriched.append(step)
            i += 1
    spec["procedure"] = enriched
    return spec


def build_recipe_ir(spec_or_order: Dict[str, Any]) -> RecipeIR:
    """Build a RecipeIR from the new procedure spec or the old notebook order format."""
    if "procedure" not in spec_or_order and "order" in spec_or_order:
        spec_or_order = recipe_spec_from_legacy_order(spec_or_order)

    recipe_id = str(spec_or_order.get("id") or "Recipe_001")
    builder = _IRBuilder(recipe_id, float(spec_or_order.get("volume", 0.0)))
    builder.ir.metadata.update(spec_or_order.get("metadata", {}))

    def connect_frontier(frontier: List[str], target: str) -> None:
        for src in frontier:
            builder.add_edge(src, target)

    def parse_steps(
        steps: Iterable[Dict[str, Any]],
        frontier: List[str],
        *,
        branch_group_id: str = "",
        branch_id: str = "",
    ) -> List[str]:
        current = list(frontier)
        for step in steps:
            if "dose" in step:
                dose = step["dose"]
                nid = builder.add_node(
                    "dose",
                    f"Dosing {dose['ingredient']}",
                    {"ingredient": str(dose["ingredient"]).upper(), "amount_L": float(dose["amount_L"])},
                    branch_group_id=branch_group_id,
                    branch_id=branch_id,
                )
                connect_frontier(current, nid)
                current = [nid]
            elif "mix" in step:
                mix = step["mix"]
                nid = builder.add_node(
                    "mix",
                    "Mixing_of_Liquids",
                    {"rpm": int(mix["rpm"]), "duration_s": int(mix["duration_s"])},
                    branch_group_id=branch_group_id,
                    branch_id=branch_id,
                )
                connect_frontier(current, nid)
                current = [nid]
            elif "usage" in step:
                usage = step["usage"]
                nid = builder.add_node(
                    "usage",
                    "Usage",
                    {"duration_s": int(usage["duration_s"])},
                    branch_group_id=branch_group_id,
                    branch_id=branch_id,
                )
                connect_frontier(current, nid)
                current = [nid]
            elif "settling" in step:
                settling = step["settling"]
                nid = builder.add_node(
                    "settling",
                    "Settling",
                    {"duration_s": int(settling["duration_s"])},
                    branch_group_id=branch_group_id,
                    branch_id=branch_id,
                )
                connect_frontier(current, nid)
                current = [nid]
            elif "heating" in step:
                heating = step["heating"]
                nid = builder.add_node(
                    "heating",
                    "Heating",
                    {"duration_s": int(heating["duration_s"])},
                    branch_group_id=branch_group_id,
                    branch_id=branch_id,
                )
                connect_frontier(current, nid)
                current = [nid]
            elif "separation" in step:
                sep_order = [str(x).upper() for x in step["separation"].get("order", [])]
                for ingr in sep_order:
                    # For XOR branches, only sum doses within the same branch
                    if branch_group_id.startswith("XOR_") and branch_id:
                        amount = _branch_ingredient_total(builder.ir, ingr, branch_group_id, branch_id)
                    else:
                        amount = _ingredient_total(builder.ir, ingr)
                    nid = builder.add_node(
                        "separation",
                        f"Separation of {ingr}",
                        {"ingredient": ingr, "amount_L": amount},
                        branch_group_id=branch_group_id,
                        branch_id=branch_id,
                    )
                    connect_frontier(current, nid)
                    current = [nid]
            elif "parallel" in step:
                group = f"AND_{len(builder.ir.choice_groups()) + len(builder.ir.nodes) + 1:03d}"
                split = builder.add_node(
                    "and_split",
                    "AND_SPLIT",
                    control_node_type="AND_SPLIT",
                    branch_group_id=group,
                    join_policy="wait_all",
                )
                connect_frontier(current, split)
                branch_ends: List[str] = []
                for idx, branch in enumerate(step["parallel"], start=1):
                    bid = f"b{idx}"
                    branch_steps = branch.get("steps", [branch]) if isinstance(branch, dict) else [branch]
                    ends = parse_steps(branch_steps, [split], branch_group_id=group, branch_id=bid)
                    branch_ends.extend(ends)
                join = builder.add_node(
                    "and_join",
                    "AND_JOIN",
                    control_node_type="AND_JOIN",
                    branch_group_id=group,
                    join_policy=step.get("join", "wait_all"),
                )
                connect_frontier(branch_ends, join)
                current = [join]
            elif "choice" in step:
                group = f"XOR_{len(builder.ir.choice_groups()) + len(builder.ir.nodes) + 1:03d}"
                split = builder.add_node(
                    "xor_split",
                    "XOR_SPLIT",
                    control_node_type="XOR_SPLIT",
                    branch_group_id=group,
                    join_policy="exactly_one",
                )
                connect_frontier(current, split)
                branch_ends = []
                for idx, branch in enumerate(step["choice"], start=1):
                    bid = str(branch.get("branch_id", f"b{idx}"))
                    ends = parse_steps(branch.get("steps", []), [split], branch_group_id=group, branch_id=bid)
                    branch_ends.extend(ends)
                join = builder.add_node(
                    "xor_join",
                    "XOR_JOIN",
                    control_node_type="XOR_JOIN",
                    branch_group_id=group,
                    join_policy="exactly_one",
                )
                connect_frontier(branch_ends, join)
                current = [join]
            elif "inclusive" in step or "or" in step:
                branches = step.get("inclusive", step.get("or", []))
                group = f"OR_{len(builder.ir.choice_groups()) + len(builder.ir.nodes) + 1:03d}"
                minimum = int(step.get("min_branches", step.get("minBranches", 1)))
                maximum = int(step.get("max_branches", step.get("maxBranches", len(branches))))
                if not 1 <= minimum <= maximum <= len(branches):
                    raise ValueError(
                        f"OR group {group} requires 1 <= min_branches <= max_branches <= {len(branches)}"
                    )
                split = builder.add_node(
                    "or_split",
                    "OR_SPLIT",
                    {"minBranches": minimum, "maxBranches": maximum},
                    control_node_type="OR_SPLIT",
                    branch_group_id=group,
                    join_policy="active_branches",
                )
                connect_frontier(current, split)
                branch_ends = []
                for idx, branch in enumerate(branches, start=1):
                    bid = str(branch.get("branch_id", f"b{idx}"))
                    ends = parse_steps(branch.get("steps", []), [split], branch_group_id=group, branch_id=bid)
                    branch_ends.extend(ends)
                join = builder.add_node(
                    "or_join",
                    "OR_JOIN",
                    {"minBranches": minimum, "maxBranches": maximum},
                    control_node_type="OR_JOIN",
                    branch_group_id=group,
                    join_policy="active_branches",
                )
                connect_frontier(branch_ends, join)
                current = [join]
            else:
                raise ValueError(f"Unsupported recipe step: {step!r}")
        return current

    parse_steps(spec_or_order.get("procedure", []), [])
    _refresh_io(builder.ir)
    return builder.ir


def _ingredient_total(ir: RecipeIR, ingredient: str) -> float:
    total = 0.0
    for node in ir.nodes.values():
        if node.node_type == "dose" and node.params.get("ingredient") == ingredient:
            total += float(node.params.get("amount_L", 0.0))
    return total


def _branch_ingredient_total(ir: RecipeIR, ingredient: str, branch_group: str, branch_id: str) -> float:
    """Sum dose amounts only for tasks in a specific XOR branch."""
    total = 0.0
    for node in ir.nodes.values():
        if (node.node_type == "dose"
                and node.params.get("ingredient") == ingredient
                and node.branch_group_id == branch_group
                and node.branch_id == branch_id):
            total += float(node.params.get("amount_L", 0.0))
    return total


def _refresh_io(ir: RecipeIR) -> None:
    totals: Dict[str, float] = defaultdict(float)
    for node in ir.nodes.values():
        if node.node_type == "dose":
            totals[str(node.params["ingredient"])] += float(node.params["amount_L"])
    ir.inputs = dict(sorted(totals.items()))
    ir.outputs = {"Product": sum(totals.values())}


def recipe_ir_to_dict(ir: RecipeIR) -> Dict[str, Any]:
    return {
        "id": ir.id,
        "volume": ir.volume,
        "inputs": ir.inputs,
        "outputs": ir.outputs,
        "metadata": ir.metadata,
        "nodes": [
            {
                "id": node.id,
                "node_type": node.node_type,
                "name": node.name,
                "params": node.params,
                "semantic_uri": node.semantic_uri,
                "control_node_type": node.control_node_type,
                "branch_group_id": node.branch_group_id,
                "branch_id": node.branch_id,
                "join_policy": node.join_policy,
            }
            for node in ir.topological_nodes()
        ],
        "edges": [{"from": src, "to": dst} for src, dst in ir.edges],
    }


def recipe_ir_from_dict(data: Dict[str, Any]) -> RecipeIR:
    ir = RecipeIR(
        id=str(data["id"]),
        volume=float(data.get("volume", 0.0)),
        inputs={str(k): float(v) for k, v in data.get("inputs", {}).items()},
        outputs={str(k): float(v) for k, v in data.get("outputs", {}).items()},
        metadata=dict(data.get("metadata", {})),
    )
    for item in data.get("nodes", []):
        node = RecipeNode(
            id=str(item["id"]),
            node_type=str(item["node_type"]),
            name=str(item.get("name", item["id"])),
            params=dict(item.get("params", {})),
            semantic_uri=str(item.get("semantic_uri", "")),
            control_node_type=str(item.get("control_node_type", "")),
            branch_group_id=str(item.get("branch_group_id", "")),
            branch_id=str(item.get("branch_id", "")),
            join_policy=str(item.get("join_policy", "")),
        )
        ir.nodes[node.id] = node
    ir.edges = [(str(e["from"]), str(e["to"])) for e in data.get("edges", [])]
    if not ir.inputs:
        _refresh_io(ir)
    return ir
