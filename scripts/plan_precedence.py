"""Derive the recipe's precedence graph from the solved plan.

The optimal plan says *when* everything happens; the Master Recipe has to say
*why*. This module reads the schedule back and recovers the ordering the solver
was honouring, so the exported SFC is the plan and can be checked by eye.

Three things make an operation wait:

* the operation that ends exactly when it starts - in an optimal schedule a
  tight edge is the constraint that was binding;
* the connection that opened the port pair it uses;
* the recipe's own order, which the schedule cannot always reveal: a dose that
  finishes early and waits for its sibling leaves no tight edge behind, yet the
  mix after it still depends on both.

The union is then transitively reduced, so only immediate predecessors remain:
a separation that follows settling does not also list the connection made at
the top of the batch, because settling already depends on it.
"""
from __future__ import annotations

from typing import Any, Iterable

CONNECT_TYPES = {"connect", "disconnect"}
TOLERANCE = 1e-6


def _route(operation) -> tuple[str, str, str, str]:
    return (
        str(getattr(operation, "source_module", "") or ""),
        str(getattr(operation, "out_port", "") or ""),
        str(getattr(operation, "target_module", "") or ""),
        str(getattr(operation, "in_port", "") or ""),
    )


def _has_route(operation) -> bool:
    source, out_port, target, in_port = _route(operation)
    return bool(out_port or in_port) and bool(source or target)


def _kind(operation) -> str:
    return str(getattr(operation, "operation_type", "")).casefold()


def _key(operation) -> str:
    return str(operation.recipe_node_id)


def plan_operations(planner_result) -> list:
    """Every scheduled operation, in planned order."""
    if planner_result is None:
        return []
    return sorted(
        planner_result.operations,
        key=lambda item: (item.start_s, item.end_s, str(item.recipe_node_id)),
    )


def _tight_predecessors(operation, operations: Iterable) -> list:
    """Operations that end exactly when this one starts."""
    return [
        other for other in operations
        if _key(other) != _key(operation)
        and abs(other.end_s - operation.start_s) <= TOLERANCE
    ]


def _opening_connection(operation, operations: Iterable):
    """The latest connect that opened this operation's port pair before it ran."""
    if not _has_route(operation) or _kind(operation) in CONNECT_TYPES:
        return None
    route = _route(operation)
    candidates = [
        other for other in operations
        if _kind(other) == "connect"
        and _route(other) == route
        and other.end_s <= operation.start_s + TOLERANCE
    ]
    return max(candidates, key=lambda item: item.end_s) if candidates else None


def transitive_reduction(edges: dict[str, set[str]]) -> dict[str, set[str]]:
    """Drop an edge whenever a longer path already implies it."""
    reachable: dict[str, set[str]] = {}

    def reach(node: str, seen: set[str]) -> set[str]:
        if node in reachable:
            return reachable[node]
        if node in seen:
            return set()
        seen = seen | {node}
        found: set[str] = set()
        for parent in edges.get(node, ()):  # parents = predecessors
            found.add(parent)
            found |= reach(parent, seen)
        reachable[node] = found
        return found

    reduced: dict[str, set[str]] = {}
    for node, parents in edges.items():
        keep = set()
        for parent in parents:
            # Implied when another parent already reaches it.
            others = parents - {parent}
            if any(parent in reach(other, set()) for other in others):
                continue
            keep.add(parent)
        reduced[node] = keep
    return reduced


def derive_precedence(planner_result, recipe_precedence: dict[str, set[str]] | None = None
                      ) -> dict[str, set[str]]:
    """Immediate predecessors per operation, keyed by recipe node id.

    ``recipe_precedence`` carries the order the recipe itself declares, which
    the schedule alone cannot recover for an operation that finished early and
    then waited.
    """
    operations = plan_operations(planner_result)
    if not operations:
        return {}
    by_key = {}
    for operation in operations:
        by_key.setdefault(_key(operation), operation)

    raw: dict[str, set[str]] = {key: set() for key in by_key}
    for operation in by_key.values():
        parents = {
            _key(other) for other in _tight_predecessors(operation, by_key.values())
            if other.end_s <= operation.start_s + TOLERANCE
        }
        opening = _opening_connection(operation, by_key.values())
        if opening is not None:
            parents.add(_key(opening))
        for declared in (recipe_precedence or {}).get(_key(operation), ()):  # recipe order
            if declared in by_key:
                parents.add(declared)
        parents.discard(_key(operation))
        raw[_key(operation)] = parents
    return transitive_reduction(raw)


def successors_of(precedence: dict[str, set[str]]) -> dict[str, set[str]]:
    successors: dict[str, set[str]] = {key: set() for key in precedence}
    for node, parents in precedence.items():
        for parent in parents:
            successors.setdefault(parent, set()).add(node)
    return successors


def describe(planner_result, recipe_precedence: dict[str, set[str]] | None = None
              ) -> list[dict[str, Any]]:
    """Flat description of the derived graph, for tests and diagnostics."""
    precedence = derive_precedence(planner_result, recipe_precedence)
    operations = {_key(op): op for op in plan_operations(planner_result)}
    successors = successors_of(precedence)
    rows = []
    for key, operation in operations.items():
        rows.append({
            "node": key,
            "type": _kind(operation),
            "start_s": operation.start_s,
            "end_s": operation.end_s,
            "predecessors": sorted(precedence.get(key, set())),
            "successors": sorted(successors.get(key, set())),
        })
    return sorted(rows, key=lambda row: (row["start_s"], row["node"]))
