from __future__ import annotations

import unittest

try:
    from ..scripts.random_recipe import generate_random_recipe_spec
    from ..scripts.recipe_ir import auto_enrich_recipe_spec, build_recipe_ir
except ImportError:  # pragma: no cover - direct test execution
    from random_recipe import generate_random_recipe_spec
    from recipe_ir import auto_enrich_recipe_spec, build_recipe_ir


class RandomRecipeTests(unittest.TestCase):
    def test_same_seed_is_reproducible(self) -> None:
        first = generate_random_recipe_spec(1234, topology="nonlinear", ingredient_count=3)
        second = generate_random_recipe_spec(1234, topology="nonlinear", ingredient_count=3)
        self.assertEqual(first, second)

    def test_nonlinear_general_and_enriched_control_flow(self) -> None:
        spec = generate_random_recipe_spec(42, topology="nonlinear", ingredient_count=3)
        general_ir = build_recipe_ir(spec)
        enriched_ir = build_recipe_ir(auto_enrich_recipe_spec(spec))

        self.assertTrue(
            any(node.control_node_type == "AND_SPLIT" for node in general_ir.nodes.values())
        )
        self.assertFalse(
            any(node.control_node_type == "XOR_SPLIT" for node in general_ir.nodes.values())
        )
        self.assertTrue(
            any(node.control_node_type == "AND_SPLIT" for node in enriched_ir.nodes.values())
        )
        self.assertTrue(
            any(node.control_node_type == "XOR_SPLIT" for node in enriched_ir.nodes.values())
        )

    def test_flat_general_is_visibly_different_from_full_enrichment(self) -> None:
        spec = generate_random_recipe_spec(42, topology="linear", ingredient_count=3)
        general_ir = build_recipe_ir(spec)
        enriched_ir = build_recipe_ir(auto_enrich_recipe_spec(spec))

        self.assertFalse(any(node.is_control for node in general_ir.nodes.values()))
        self.assertTrue(
            any(node.control_node_type == "AND_SPLIT" for node in enriched_ir.nodes.values())
        )
        self.assertTrue(
            any(node.control_node_type == "XOR_SPLIT" for node in enriched_ir.nodes.values())
        )
        self.assertNotEqual(
            [(node.id, node.node_type) for node in general_ir.topological_nodes()],
            [(node.id, node.node_type) for node in enriched_ir.topological_nodes()],
        )

    def test_generated_volume_matches_doses(self) -> None:
        spec = generate_random_recipe_spec(99, topology="parallel", ingredient_count=2)
        ir = build_recipe_ir(spec)
        self.assertAlmostEqual(spec["volume"], sum(ir.inputs.values()))


if __name__ == "__main__":
    unittest.main()
