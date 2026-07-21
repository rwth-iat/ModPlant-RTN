"""Demonstrate the legacy linear-order dict format being lifted into RecipeIR.

    $ python RTN/demos/run_legacy_order.py
"""
from __future__ import annotations

import _path  # noqa: F401

from . import build_recipe_ir, recipe_spec_from_legacy_order
from sample_data import sample_legacy_order


def main() -> None:
    legacy = sample_legacy_order()
    spec = recipe_spec_from_legacy_order(
        legacy,
        parallel_dosing_before_mix=True,
        usage_alternatives=[
            {"branch_id": "fast", "steps": [{"usage": {"duration_s": 1800}}]},
            {"branch_id": "standard", "steps": [{"usage": {"duration_s": 3600}}]},
        ],
    )
    ir = build_recipe_ir(spec)
    print(f"Legacy order: {legacy['order']}")
    print(f"Lifted RecipeIR: {len(ir.nodes)} nodes, {len(ir.edges)} edges")
    print(f"Choice / branch groups: {ir.choice_groups()}")


if __name__ == "__main__":
    main()
