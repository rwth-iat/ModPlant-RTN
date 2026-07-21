"""Resource-Task Network (RTN) core data model.

Implements the formalism of Pantelides (1994) for batch process scheduling.
Tasks consume and produce resources via signed coefficients; equipment, ports,
utilities, and material states are uniformly represented as ``Resource``.

This module is solver-agnostic. ``cp_sat_planner.py`` (the production solver)
consumes ``RTNModel`` instances directly; ``schedule_validator.py`` reuses the
same model for post-hoc constraint checks.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, FrozenSet, List, Literal, Optional, Tuple, Union


ResourceKind = Literal["equipment", "port", "utility", "state"]
TaskType = Literal[
    "dose",
    "mix",
    "usage",
    "settling",
    "separation",
    "transfer",
]
TimeOffset = Union[Literal["start", "end"], int]


@dataclass(frozen=True)
class Resource:
    """Unified resource: equipment (module), port, utility, or material state.

    capacity=None means unbounded. initial_level applies at horizon t=0.
    """

    id: str
    kind: ResourceKind
    capacity: Optional[float] = None
    initial_level: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProductionCoefficient:
    """Signed effect of a task on a resource at a specific time offset.

    coefficient < 0 = consumption, > 0 = production. ``time_offset`` may be
    "start", "end", or an int (seconds relative to task start).
    """

    resource_id: str
    coefficient: float
    time_offset: TimeOffset = "start"


@dataclass(frozen=True)
class Task:
    """A schedulable activity with resource effects and timing."""

    id: str
    task_type: str
    duration_s: int
    duration_mode: Literal["fixed", "min"] = "fixed"
    coefficients: Tuple[ProductionCoefficient, ...] = ()
    eligible_resources: FrozenSet[str] = frozenset()
    branch_group_id: Optional[str] = None
    branch_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class TaskNodeView:
    """Adapts an RTN ``Task`` to the legacy ``RecipeNode`` field shape.

    The CP-SAT solver internal accesses tasks via field-style attributes
    (``id``, ``node_type``, ``name``, ``params``, ``semantic_uri``,
    ``branch_group_id``, ``branch_id``, ``is_task``). This view provides
    those attributes by reading the equivalent data from a ``Task``, so the
    inner solver can operate on RTN tasks without further refactoring.

    The underlying ``Task`` is exposed via ``.task`` for code that needs
    RTN-native access (e.g., coefficients, eligible_resources).
    """

    __slots__ = (
        "task",
        "id",
        "node_type",
        "name",
        "params",
        "semantic_uri",
        "branch_group_id",
        "branch_id",
        "join_policy",
        "control_node_type",
    )

    def __init__(self, task: "Task") -> None:
        meta = task.metadata or {}
        self.task = task
        self.id = task.id
        self.node_type = task.task_type
        self.name = str(meta.get("recipe_node_name") or task.id)
        self.params = dict(meta.get("params") or {})
        self.semantic_uri = str(meta.get("semantic_uri") or "")
        self.branch_group_id = task.branch_group_id or ""
        self.branch_id = task.branch_id or ""
        self.join_policy = ""
        self.control_node_type = ""

    @property
    def is_control(self) -> bool:
        return False

    @property
    def is_task(self) -> bool:
        return True

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"TaskNodeView(id={self.id!r}, node_type={self.node_type!r})"


@dataclass
class RTNModel:
    """A complete Resource-Task Network problem instance."""

    horizon_s: int
    resources: Dict[str, Resource] = field(default_factory=dict)
    tasks: Dict[str, Task] = field(default_factory=dict)
    precedence: List[Tuple[str, str]] = field(default_factory=list)
    choice_groups: Dict[str, List[str]] = field(default_factory=dict)
    choice_group_policies: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_resource(self, resource: Resource) -> None:
        if resource.id in self.resources:
            raise ValueError(f"duplicate resource id: {resource.id}")
        self.resources[resource.id] = resource

    def add_task(self, task: Task) -> None:
        if task.id in self.tasks:
            raise ValueError(f"duplicate task id: {task.id}")
        for coef in task.coefficients:
            if coef.resource_id not in self.resources:
                raise ValueError(
                    f"task {task.id} references unknown resource {coef.resource_id}"
                )
        self.tasks[task.id] = task

    def equipment_resources(self) -> List[Resource]:
        return [r for r in self.resources.values() if r.kind == "equipment"]

    def state_resources(self) -> List[Resource]:
        return [r for r in self.resources.values() if r.kind == "state"]

    def port_resources(self) -> List[Resource]:
        return [r for r in self.resources.values() if r.kind == "port"]

    def topological_tasks(self) -> List[Task]:
        """Return tasks in a deterministic topological order based on precedence.

        Mirrors ``RecipeIR.task_nodes()`` semantics. Raises ``ValueError`` if
        ``precedence`` contains a cycle.
        """
        indeg: Dict[str, int] = {tid: 0 for tid in self.tasks}
        succ: Dict[str, List[str]] = defaultdict(list)
        for src, dst in self.precedence:
            if src in self.tasks and dst in self.tasks:
                succ[src].append(dst)
                indeg[dst] = indeg.get(dst, 0) + 1
        queue: deque[str] = deque(sorted(tid for tid, deg in indeg.items() if deg == 0))
        out: List[Task] = []
        while queue:
            tid = queue.popleft()
            out.append(self.tasks[tid])
            for dst in sorted(succ[tid]):
                indeg[dst] -= 1
                if indeg[dst] == 0:
                    queue.append(dst)
        if len(out) != len(self.tasks):
            raise ValueError("RTNModel.precedence contains a cycle.")
        return out

    def derive_recipe_inputs(self) -> Dict[str, float]:
        """Derive {ingredient_name: total_amount_L} from ingredient state resources.

        Ingredient state resources are identified by ``metadata['is_ingredient']
        == True``; the ingredient name is read from ``metadata['ingredient_name']``.
        Capacity holds the total dose amount.
        """
        out: Dict[str, float] = {}
        for r in self.state_resources():
            if not r.metadata.get("is_ingredient"):
                continue
            name = str(r.metadata.get("ingredient_name") or "")
            if not name:
                continue
            out[name] = float(r.capacity or 0.0)
        return dict(sorted(out.items()))

    def producers_of(self, resource_id: str) -> List[Task]:
        return [
            t
            for t in self.tasks.values()
            if any(c.resource_id == resource_id and c.coefficient > 0 for c in t.coefficients)
        ]

    def consumers_of(self, resource_id: str) -> List[Task]:
        return [
            t
            for t in self.tasks.values()
            if any(c.resource_id == resource_id and c.coefficient < 0 for c in t.coefficients)
        ]

    def validate(self) -> List[str]:
        """Return list of structural errors. Empty list = valid."""
        errors: List[str] = []
        for src, dst in self.precedence:
            if src not in self.tasks:
                errors.append(f"precedence: unknown task {src}")
            if dst not in self.tasks:
                errors.append(f"precedence: unknown task {dst}")
        for group_id, branches in self.choice_groups.items():
            for branch in branches:
                tasks_in_branch = [
                    t for t in self.tasks.values()
                    if t.branch_group_id == group_id and t.branch_id == branch
                ]
                if not tasks_in_branch:
                    errors.append(
                        f"choice_group {group_id} declares branch {branch} but no task"
                    )
            policy = self.choice_group_policies.get(group_id, {})
            if group_id.startswith("OR_"):
                minimum = int(policy.get("minBranches", 1))
                maximum = int(policy.get("maxBranches", len(branches)))
                if not 1 <= minimum <= maximum <= len(branches):
                    errors.append(
                        f"choice_group {group_id} has invalid OR cardinality {minimum}..{maximum}"
                    )
        for task in self.tasks.values():
            for elig in task.eligible_resources:
                if elig not in self.resources:
                    errors.append(
                        f"task {task.id}: eligible_resource {elig} not in resources"
                    )
            for coef in task.coefficients:
                if coef.resource_id not in self.resources:
                    errors.append(
                        f"task {task.id}: coefficient references unknown resource "
                        f"{coef.resource_id}"
                    )
        return errors

    def to_dict(self) -> Dict[str, Any]:
        return {
            "horizon_s": self.horizon_s,
            "resources": {rid: asdict(r) for rid, r in self.resources.items()},
            "tasks": {
                tid: {
                    **asdict(t),
                    "eligible_resources": sorted(t.eligible_resources),
                    "coefficients": [asdict(c) for c in t.coefficients],
                }
                for tid, t in self.tasks.items()
            },
            "precedence": [list(p) for p in self.precedence],
            "choice_groups": {gid: list(b) for gid, b in self.choice_groups.items()},
            "choice_group_policies": {
                gid: dict(policy) for gid, policy in self.choice_group_policies.items()
            },
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RTNModel":
        model = cls(
            horizon_s=int(data["horizon_s"]),
            choice_group_policies={
                str(group): dict(policy)
                for group, policy in data.get("choice_group_policies", {}).items()
            },
        )
        for rid, rdata in data.get("resources", {}).items():
            model.resources[rid] = Resource(**rdata)
        for tid, tdata in data.get("tasks", {}).items():
            coefs = tuple(
                ProductionCoefficient(**c) for c in tdata.get("coefficients", ())
            )
            elig = frozenset(tdata.get("eligible_resources", ()))
            kwargs = {
                k: v
                for k, v in tdata.items()
                if k not in {"coefficients", "eligible_resources"}
            }
            model.tasks[tid] = Task(
                coefficients=coefs, eligible_resources=elig, **kwargs
            )
        model.precedence = [tuple(p) for p in data.get("precedence", [])]
        model.choice_groups = {
            gid: list(b) for gid, b in data.get("choice_groups", {}).items()
        }
        model.metadata = dict(data.get("metadata", {}))
        return model
