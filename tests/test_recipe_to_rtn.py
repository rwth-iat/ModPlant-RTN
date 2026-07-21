"""Unit tests for the RecipeIR → RTNModel adapter."""
from __future__ import annotations

import unittest

try:
    from ..recipe_ir import build_recipe_ir
    from ..recipe_to_rtn import (
        INGREDIENT_STATE_PREFIX,
        MIXED_STATE,
        PRODUCT_STATE,
        recipe_ir_to_rtn,
    )
    from ..rtn import RTNModel
except ImportError:  # pragma: no cover
    from recipe_ir import build_recipe_ir
    from recipe_to_rtn import (
        INGREDIENT_STATE_PREFIX,
        MIXED_STATE,
        PRODUCT_STATE,
        recipe_ir_to_rtn,
    )
    from rtn import RTNModel


def _simple_spec():
    return {
        "id": "RTN_Adapter_Test",
        "volume": 100.0,
        "procedure": [
            {"dose": {"ingredient": "A", "amount_L": 60.0}},
            {"dose": {"ingredient": "B", "amount_L": 40.0}},
            {"mix": {"rpm": 200, "duration_s": 120}},
            {"usage": {"duration_s": 600}},
            {"settling": {"duration_s": 60}},
            {"separation": {"order": ["A", "B"]}},
        ],
    }


class RecipeToRTNAdapterTests(unittest.TestCase):
    def test_basic_topology(self):
        ir = build_recipe_ir(_simple_spec())
        plant = {
            "module_ops": {"W1": [], "W2": []},
            "module_maximum_volume": {"W1": [200.0], "W2": [150.0]},
        }
        model = recipe_ir_to_rtn(
            ir,
            module_ops=plant["module_ops"],
            module_maximum_volume=plant["module_maximum_volume"],
        )

        # Equipment resources
        equipment = {r.id for r in model.equipment_resources()}
        self.assertEqual(equipment, {"W1", "W2"})
        self.assertEqual(model.resources["W1"].metadata.get("max_volume_L"), 200.0)

        # Ingredient state resources
        states = {r.id for r in model.state_resources()}
        self.assertIn(INGREDIENT_STATE_PREFIX + "A", states)
        self.assertIn(INGREDIENT_STATE_PREFIX + "B", states)
        self.assertIn(MIXED_STATE, states)
        self.assertIn(PRODUCT_STATE, states)

        # Capacity = total dose volume per ingredient
        self.assertEqual(model.resources[INGREDIENT_STATE_PREFIX + "A"].capacity, 60.0)
        self.assertEqual(model.resources[INGREDIENT_STATE_PREFIX + "B"].capacity, 40.0)

        # Tasks: dose×2, mix, usage, settling, separation×2 (one per ingredient)
        task_types = sorted(t.task_type for t in model.tasks.values())
        self.assertEqual(
            task_types,
            ["dose", "dose", "mix", "separation", "separation", "settling", "usage"],
        )

        # Precedence is acyclic
        self.assertTrue(model.precedence)
        self.assertEqual(model.validate(), [])

    def test_dose_produces_separation_consumes(self):
        ir = build_recipe_ir(_simple_spec())
        model = recipe_ir_to_rtn(ir, module_ops={"W1": []})

        a_state = INGREDIENT_STATE_PREFIX + "A"
        producers = {t.id for t in model.producers_of(a_state)}
        consumers = {t.id for t in model.consumers_of(a_state)}
        self.assertTrue(any(t.startswith("dose_") for t in producers))
        self.assertTrue(any(t.startswith("separation_") for t in consumers))

    def test_serialization_roundtrip(self):
        ir = build_recipe_ir(_simple_spec())
        model = recipe_ir_to_rtn(ir, module_ops={"W1": []})
        roundtripped = RTNModel.from_dict(model.to_dict())
        self.assertEqual(roundtripped.to_dict(), model.to_dict())

    def test_choice_group_passthrough(self):
        spec = {
            "id": "ChoiceTest",
            "volume": 50.0,
            "procedure": [
                {"dose": {"ingredient": "X", "amount_L": 50.0}},
                {"mix": {"rpm": 100, "duration_s": 60}},
                {
                    "choice": [
                        {"steps": [{"usage": {"duration_s": 300}}]},
                        {"steps": [{"usage": {"duration_s": 600}}]},
                    ],
                    "select": "optimizer",
                },
                {"settling": {"duration_s": 30}},
                {"separation": {"order": ["X"]}},
            ],
        }
        ir = build_recipe_ir(spec)
        model = recipe_ir_to_rtn(ir, module_ops={"W1": []})
        self.assertTrue(model.choice_groups, "choice groups should be carried over")

    def test_plant_config_in_metadata(self):
        ir = build_recipe_ir(_simple_spec())
        model = recipe_ir_to_rtn(
            ir,
            module_ops={"W1": [("op", "params", 1.0)]},
            module_interfaces={"W1": [("Input", "p1"), ("Output", "p2")]},
        )
        plant = model.metadata.get("plant", {})
        self.assertIn("module_ops", plant)
        self.assertIn("module_interfaces", plant)
        self.assertIn("W1.p1", model.resources)
        self.assertIn("W1.p2", model.resources)


if __name__ == "__main__":
    unittest.main()
