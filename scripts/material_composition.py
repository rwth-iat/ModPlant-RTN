"""Canonical material-composition semantics shared by planning and validation.

The RTN planner represents one physical batch.  A composition is therefore a
mapping from material name to volume in litres.  All helpers in this module are
side-effect free so the CP-SAT model and the post-hoc validator cannot silently
drift to different interpretations of purity, proportional transfer, or
connection reuse.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isclose
from typing import Dict, Iterable, Mapping, Tuple


CompositionTuple = Tuple[Tuple[str, float], ...]
PURE_MATERIAL = "pure_material"
MIXTURE = "mixture"


def normalize_composition(
    composition: Mapping[str, float] | Iterable[Tuple[str, float]] | None,
    *,
    tolerance: float = 1e-12,
) -> Dict[str, float]:
    """Return a sorted, upper-case, zero-free composition mapping."""
    if composition is None:
        return {}
    items = composition.items() if isinstance(composition, Mapping) else composition
    normalized: Dict[str, float] = {}
    for raw_name, raw_quantity in items:
        name = str(raw_name).strip().upper()
        quantity = float(raw_quantity)
        if not name or abs(quantity) <= tolerance:
            continue
        normalized[name] = normalized.get(name, 0.0) + quantity
    return {
        name: quantity
        for name, quantity in sorted(normalized.items())
        if abs(quantity) > tolerance
    }


def composition_tuple(
    composition: Mapping[str, float] | Iterable[Tuple[str, float]] | None,
) -> CompositionTuple:
    return tuple(normalize_composition(composition).items())


def transfer_kind(composition: Mapping[str, float] | Iterable[Tuple[str, float]] | None) -> str:
    return MIXTURE if len(normalize_composition(composition)) > 1 else PURE_MATERIAL


def composition_signature(
    composition: Mapping[str, float] | Iterable[Tuple[str, float]] | None,
) -> str:
    """Return the persistent-connection identity requested by ModPlant.

    Pure-material connections deliberately ignore transferred volume.  Mixture
    connections include every resolved component quantity, so a different
    mixture amount requires an explicit Disconnect/Connect transition even when
    the ratio is unchanged.
    """
    normalized = normalize_composition(composition)
    if not normalized:
        return ""
    if len(normalized) == 1:
        return f"MATERIAL:{next(iter(normalized))}"
    return "MIX:" + "|".join(
        f"{name}={quantity:.12g}" for name, quantity in normalized.items()
    )


def is_pure_composition(
    composition: Mapping[str, float] | Iterable[Tuple[str, float]] | None,
    material: str | None = None,
) -> bool:
    normalized = normalize_composition(composition)
    if len(normalized) != 1:
        return False
    return material is None or next(iter(normalized)) == str(material).strip().upper()


def compositions_proportional(
    left: Mapping[str, float] | Iterable[Tuple[str, float]] | None,
    right: Mapping[str, float] | Iterable[Tuple[str, float]] | None,
    *,
    tolerance: float = 1e-6,
) -> bool:
    """Return True when two non-empty vectors have one common scale factor."""
    lhs = normalize_composition(left, tolerance=tolerance)
    rhs = normalize_composition(right, tolerance=tolerance)
    if not lhs or set(lhs) != set(rhs):
        return False
    pivot = next(iter(lhs))
    if abs(rhs[pivot]) <= tolerance:
        return False
    scale = lhs[pivot] / rhs[pivot]
    if scale <= 0:
        return False
    return all(
        isclose(lhs[name], rhs[name] * scale, rel_tol=tolerance, abs_tol=tolerance)
        for name in lhs
    )


def subtract_composition(
    inventory: Mapping[str, float],
    outgoing: Mapping[str, float],
    *,
    tolerance: float = 1e-6,
) -> Dict[str, float]:
    result = normalize_composition(inventory, tolerance=tolerance)
    for material, quantity in normalize_composition(outgoing, tolerance=tolerance).items():
        result[material] = result.get(material, 0.0) - quantity
    return normalize_composition(result, tolerance=tolerance)


def add_composition(
    inventory: Mapping[str, float],
    incoming: Mapping[str, float],
    *,
    tolerance: float = 1e-6,
) -> Dict[str, float]:
    result = normalize_composition(inventory, tolerance=tolerance)
    for material, quantity in normalize_composition(incoming, tolerance=tolerance).items():
        result[material] = result.get(material, 0.0) + quantity
    return normalize_composition(result, tolerance=tolerance)


@dataclass(frozen=True)
class PureMaterialTransfer:
    material: str
    amount_l: float

    @property
    def composition(self) -> Dict[str, float]:
        return normalize_composition({self.material: self.amount_l})

    @property
    def signature(self) -> str:
        return composition_signature(self.composition)


@dataclass(frozen=True)
class MixtureTransfer:
    components: CompositionTuple

    @classmethod
    def from_mapping(cls, composition: Mapping[str, float]) -> "MixtureTransfer":
        normalized = composition_tuple(composition)
        if len(normalized) < 2:
            raise ValueError("MixtureTransfer requires at least two non-zero components")
        return cls(normalized)

    @property
    def composition(self) -> Dict[str, float]:
        return dict(self.components)

    @property
    def amount_l(self) -> float:
        return sum(self.composition.values())

    @property
    def signature(self) -> str:
        return composition_signature(self.composition)
