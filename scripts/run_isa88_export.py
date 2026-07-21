"""Export the sample recipe to ISA-88 / B2MML General Recipe XML and re-parse.

    $ python RTN/demos/run_isa88_export.py
"""
from __future__ import annotations

from pathlib import Path

import _path  # noqa: F401

from RTN import (
    build_recipe_ir,
    parse_general_recipe_xml_to_ir,
    save_general_recipe_xml_from_ir,
    save_recipe_ir_json,
)
from sample_data import sample_recipe_spec


def main() -> None:
    ir = build_recipe_ir(sample_recipe_spec())
    out_dir = Path(__file__).resolve().parent.parent / "generated"
    out_dir.mkdir(parents=True, exist_ok=True)

    xml_path = save_general_recipe_xml_from_ir(ir, out_dir / f"GeneralRecipe_{ir.id}.xml")
    json_path = save_recipe_ir_json(ir, out_dir / f"RecipeIR_{ir.id}.json")
    print(f"XML      : {xml_path}")
    print(f"JSON     : {json_path}")

    parsed = parse_general_recipe_xml_to_ir(xml_path)
    print(f"\nRound-trip: original nodes={len(ir.nodes)} edges={len(ir.edges)} "
          f"-> parsed nodes={len(parsed.nodes)} edges={len(parsed.edges)}")
    assert len(ir.nodes) == len(parsed.nodes), "node count mismatch on roundtrip"
    print("OK")


if __name__ == "__main__":
    main()
