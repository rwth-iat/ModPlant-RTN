"""Deterministic random recipe generation for the RTN notebook UI."""
from __future__ import annotations

import copy
import random
from typing import Any, Dict


SUPPORTED_TOPOLOGIES = ("nonlinear", "parallel", "linear")
AVAILABLE_INGREDIENTS = ("A", "B", "C")


def generate_random_recipe_spec(
    seed: int,
    *,
    topology: str = "nonlinear",
    ingredient_count: int = 3,
) -> Dict[str, Any]:
    """Create a deterministic, plant-compatible General Recipe.

    ``nonlinear`` and ``parallel`` contain an explicit AND split/join around
    dosing. ``linear`` keeps dosing sequential. ``nonlinear`` and ``linear``
    also contain a usage step that the normal RTN enrichment pass converts
    into an optimizer-selected XOR choice.
    """
    topology = str(topology).lower()
    if topology not in SUPPORTED_TOPOLOGIES:
        raise ValueError(
            f"Unsupported topology {topology!r}; expected one of {SUPPORTED_TOPOLOGIES}."
        )

    ingredient_count = int(ingredient_count)
    if not 1 <= ingredient_count <= len(AVAILABLE_INGREDIENTS):
        raise ValueError(
            f"ingredient_count must be between 1 and {len(AVAILABLE_INGREDIENTS)}."
        )

    rng = random.Random(int(seed))
    ingredients = list(AVAILABLE_INGREDIENTS[:ingredient_count])
    amounts = {
        ingredient: rng.choice((0.5, 1.0, 1.5, 2.0))
        for ingredient in ingredients
    }
    dose_steps = [
        {"dose": {"ingredient": ingredient, "amount_L": amounts[ingredient]}}
        for ingredient in ingredients
    ]

    procedure = []
    if topology in {"nonlinear", "parallel"} and len(dose_steps) > 1:
        procedure.append(
            {
                "parallel": copy.deepcopy(dose_steps),
                "join": "wait_all",
            }
        )
    else:
        procedure.extend(copy.deepcopy(dose_steps))

    procedure.append(
        {
            "mix": {
                "rpm": rng.choice((100, 150, 200)),
                "duration_s": rng.choice((15, 30, 45, 60)),
            }
        }
    )

    if topology in {"nonlinear", "linear"}:
        procedure.append(
            {"usage": {"duration_s": rng.choice((2400, 3000, 3600, 4200))}}
        )

    procedure.append(
        {"settling": {"duration_s": rng.choice((60, 120, 180, 300))}}
    )

    separation_order = list(ingredients)
    rng.shuffle(separation_order)
    procedure.append({"separation": {"order": separation_order}})

    volume = round(sum(amounts.values()), 3)
    return {
        "id": f"RTN_Random_{topology}_{int(seed)}",
        "volume": volume,
        "procedure": procedure,
        "metadata": {
            "source": "seeded_random_recipe",
            "random_seed": int(seed),
            "topology": topology,
            "ingredient_count": ingredient_count,
        },
    }
