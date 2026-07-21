"""Independent unit tests for the RTN data model (rtn.py).

These tests exercise ``Resource``, ``Task``, ``ProductionCoefficient``,
``RTNModel``, and ``TaskNodeView`` without touching the adapter, the
solver, or any optional dependency. They give precise failure localization
when the data model changes shape; the adapter and solver tests cover the
same surfaces indirectly via end-to-end paths.
"""
from __future__ import annotations

import unittest

try:
    from ..rtn import (
        ProductionCoefficient,
        Resource,
        RTNModel,
        Task,
        TaskNodeView,
    )
except ImportError:  # pragma: no cover - direct script execution from RTN/
    from rtn import (
        ProductionCoefficient,
        Resource,
        RTNModel,
        Task,
        TaskNodeView,
    )


# --- Resource -------------------------------------------------------------

class ResourceTests(unittest.TestCase):
    def test_construction_defaults(self):
        r = Resource(id="W1", kind="equipment")
        self.assertEqual(r.id, "W1")
        self.assertEqual(r.kind, "equipment")
        self.assertIsNone(r.capacity)
        self.assertEqual(r.initial_level, 0.0)
        self.assertEqual(r.metadata, {})

    def test_immutable(self):
        r = Resource(id="W1", kind="equipment", capacity=1.0)
        with self.assertRaises(Exception):
            r.id = "W2"  # frozen dataclass

    def test_metadata_is_carried(self):
        r = Resource(id="A", kind="state", capacity=10.0,
                     metadata={"is_ingredient": True, "ingredient_name": "A"})
        self.assertTrue(r.metadata["is_ingredient"])
        self.assertEqual(r.metadata["ingredient_name"], "A")


# --- ProductionCoefficient ------------------------------------------------

class ProductionCoefficientTests(unittest.TestCase):
    def test_negative_means_consumption(self):
        c = ProductionCoefficient(resource_id="A", coefficient=-2.5)
        self.assertLess(c.coefficient, 0)

    def test_default_time_offset_is_start(self):
        c = ProductionCoefficient(resource_id="A", coefficient=1.0)
        self.assertEqual(c.time_offset, "start")

    def test_time_offset_can_be_int(self):
        c = ProductionCoefficient(resource_id="A", coefficient=1.0, time_offset=30)
        self.assertEqual(c.time_offset, 30)


# --- Task -----------------------------------------------------------------

class TaskTests(unittest.TestCase):
    def test_construction_defaults(self):
        t = Task(id="t1", task_type="dose", duration_s=0)
        self.assertEqual(t.id, "t1")
        self.assertEqual(t.task_type, "dose")
        self.assertEqual(t.duration_s, 0)
        self.assertEqual(t.duration_mode, "fixed")
        self.assertEqual(t.coefficients, ())
        self.assertEqual(t.eligible_resources, frozenset())
        self.assertIsNone(t.branch_group_id)
        self.assertIsNone(t.branch_id)
        self.assertEqual(t.metadata, {})

    def test_coefficients_are_tuple(self):
        t = Task(
            id="t1",
            task_type="dose",
            duration_s=0,
            coefficients=(ProductionCoefficient("A", 1.0),),
        )
        self.assertEqual(len(t.coefficients), 1)
        self.assertEqual(t.coefficients[0].resource_id, "A")


# --- RTNModel: add_resource / add_task ------------------------------------

class RTNModelMutationTests(unittest.TestCase):
    def test_add_resource(self):
        m = RTNModel(horizon_s=100)
        r = Resource(id="W1", kind="equipment", capacity=1.0)
        m.add_resource(r)
        self.assertIs(m.resources["W1"], r)

    def test_duplicate_resource_raises(self):
        m = RTNModel(horizon_s=100)
        m.add_resource(Resource(id="W1", kind="equipment"))
        with self.assertRaises(ValueError):
            m.add_resource(Resource(id="W1", kind="equipment"))

    def test_add_task_validates_coefficient_resources(self):
        m = RTNModel(horizon_s=100)
        with self.assertRaises(ValueError):
            m.add_task(Task(
                id="t1", task_type="dose", duration_s=0,
                coefficients=(ProductionCoefficient("missing_state", 1.0),),
            ))

    def test_duplicate_task_raises(self):
        m = RTNModel(horizon_s=100)
        m.add_resource(Resource(id="W1", kind="equipment"))
        m.add_task(Task(id="t1", task_type="dose", duration_s=0))
        with self.assertRaises(ValueError):
            m.add_task(Task(id="t1", task_type="mix", duration_s=10))


# --- RTNModel: query helpers ---------------------------------------------

class RTNModelQueryTests(unittest.TestCase):
    def _build_simple(self):
        m = RTNModel(horizon_s=1000)
        m.add_resource(Resource(id="W1", kind="equipment", capacity=1.0))
        m.add_resource(Resource(id="W2", kind="equipment", capacity=1.0))
        m.add_resource(Resource(id="W1.in", kind="port", capacity=1.0))
        m.add_resource(Resource(
            id="state.A", kind="state", capacity=10.0,
            metadata={"is_ingredient": True, "ingredient_name": "A"},
        ))
        m.add_resource(Resource(
            id="state.B", kind="state", capacity=5.0,
            metadata={"is_ingredient": True, "ingredient_name": "B"},
        ))
        m.add_resource(Resource(id="state.product", kind="state"))
        m.add_task(Task(
            id="dose_A", task_type="dose", duration_s=0,
            coefficients=(ProductionCoefficient("state.A", +5.0, time_offset="end"),),
        ))
        m.add_task(Task(
            id="dose_B", task_type="dose", duration_s=0,
            coefficients=(ProductionCoefficient("state.B", +2.0, time_offset="end"),),
        ))
        m.add_task(Task(
            id="mix", task_type="mix", duration_s=30,
            coefficients=(
                ProductionCoefficient("state.A", -5.0, time_offset="start"),
                ProductionCoefficient("state.B", -2.0, time_offset="start"),
                ProductionCoefficient("state.product", +1.0, time_offset="end"),
            ),
        ))
        m.precedence = [("dose_A", "mix"), ("dose_B", "mix")]
        return m

    def test_kind_filtering(self):
        m = self._build_simple()
        self.assertEqual({r.id for r in m.equipment_resources()}, {"W1", "W2"})
        self.assertEqual({r.id for r in m.port_resources()}, {"W1.in"})
        self.assertEqual(
            {r.id for r in m.state_resources()},
            {"state.A", "state.B", "state.product"},
        )

    def test_producers_and_consumers(self):
        m = self._build_simple()
        prods_a = {t.id for t in m.producers_of("state.A")}
        cons_a = {t.id for t in m.consumers_of("state.A")}
        self.assertEqual(prods_a, {"dose_A"})
        self.assertEqual(cons_a, {"mix"})

    def test_topological_tasks_respects_precedence(self):
        m = self._build_simple()
        order = [t.id for t in m.topological_tasks()]
        self.assertLess(order.index("dose_A"), order.index("mix"))
        self.assertLess(order.index("dose_B"), order.index("mix"))

    def test_topological_tasks_detects_cycle(self):
        m = self._build_simple()
        m.precedence.append(("mix", "dose_A"))  # introduce cycle
        with self.assertRaises(ValueError):
            m.topological_tasks()

    def test_derive_recipe_inputs(self):
        m = self._build_simple()
        inputs = m.derive_recipe_inputs()
        # derive_recipe_inputs reads each ingredient state's ``capacity`` (which the
        # adapter sets to the total dose volume), not the per-task coefficient.
        self.assertEqual(inputs, {"A": 10.0, "B": 5.0})

    def test_derive_recipe_inputs_skips_non_ingredient_states(self):
        m = self._build_simple()
        inputs = m.derive_recipe_inputs()
        self.assertNotIn("product", inputs)


# --- RTNModel: validate ---------------------------------------------------

class RTNModelValidateTests(unittest.TestCase):
    def test_valid_model(self):
        m = RTNModel(horizon_s=100)
        m.add_resource(Resource(id="W", kind="equipment"))
        m.add_task(Task(id="t", task_type="dose", duration_s=0,
                        eligible_resources=frozenset({"W"})))
        self.assertEqual(m.validate(), [])

    def test_unknown_precedence_task_reported(self):
        m = RTNModel(horizon_s=100)
        m.add_resource(Resource(id="W", kind="equipment"))
        m.add_task(Task(id="t", task_type="dose", duration_s=0))
        m.precedence = [("t", "ghost")]
        errors = m.validate()
        self.assertTrue(any("ghost" in e for e in errors))

    def test_unknown_eligible_resource_reported(self):
        m = RTNModel(horizon_s=100)
        m.add_task(Task(id="t", task_type="dose", duration_s=0,
                        eligible_resources=frozenset({"missing_w"})))
        errors = m.validate()
        self.assertTrue(any("missing_w" in e for e in errors))

    def test_choice_group_with_no_member_task_reported(self):
        m = RTNModel(horizon_s=100)
        m.add_resource(Resource(id="W", kind="equipment"))
        m.choice_groups = {"XOR_x": ["fast"]}  # no task tagged with this group/branch
        errors = m.validate()
        self.assertTrue(any("XOR_x" in e and "fast" in e for e in errors))


# --- RTNModel: serialization roundtrip ------------------------------------

class RTNModelSerializationTests(unittest.TestCase):
    def test_to_from_dict_roundtrip(self):
        m = RTNModel(horizon_s=200)
        m.add_resource(Resource(id="W1", kind="equipment", capacity=1.0))
        m.add_resource(Resource(
            id="state.A", kind="state", capacity=5.0,
            metadata={"is_ingredient": True, "ingredient_name": "A"},
        ))
        m.add_task(Task(
            id="t1", task_type="dose", duration_s=10,
            duration_mode="min",
            coefficients=(ProductionCoefficient("state.A", 5.0, time_offset="end"),),
            eligible_resources=frozenset({"W1"}),
            branch_group_id="XOR_001", branch_id="fast",
            metadata={"recipe_node_name": "Dose A"},
        ))
        m.precedence = [("t1", "t1")]  # silly but exercises the (src, dst) shape
        m.choice_groups = {"XOR_001": ["fast"]}
        m.metadata = {"recipe_id": "Test", "extra": [1, 2, 3]}

        roundtripped = RTNModel.from_dict(m.to_dict())
        self.assertEqual(roundtripped.to_dict(), m.to_dict())
        self.assertEqual(roundtripped.tasks["t1"].eligible_resources, frozenset({"W1"}))
        self.assertEqual(roundtripped.tasks["t1"].coefficients[0].coefficient, 5.0)


# --- TaskNodeView ---------------------------------------------------------

class TaskNodeViewTests(unittest.TestCase):
    def _task(self, **kwargs):
        defaults = dict(
            id="t1", task_type="dose", duration_s=0,
            metadata={"recipe_node_name": "Dose A", "semantic_uri": "uri:Dosing",
                      "params": {"ingredient": "A", "amount_L": 1.0}},
        )
        defaults.update(kwargs)
        return Task(**defaults)

    def test_field_mapping(self):
        v = TaskNodeView(self._task(branch_group_id="XOR_1", branch_id="fast"))
        self.assertEqual(v.id, "t1")
        self.assertEqual(v.node_type, "dose")
        self.assertEqual(v.name, "Dose A")
        self.assertEqual(v.params, {"ingredient": "A", "amount_L": 1.0})
        self.assertEqual(v.semantic_uri, "uri:Dosing")
        self.assertEqual(v.branch_group_id, "XOR_1")
        self.assertEqual(v.branch_id, "fast")

    def test_is_task_is_true(self):
        v = TaskNodeView(self._task())
        self.assertTrue(v.is_task)
        self.assertFalse(v.is_control)

    def test_missing_metadata_falls_back_to_id(self):
        v = TaskNodeView(Task(id="t99", task_type="mix", duration_s=10))
        self.assertEqual(v.name, "t99")
        self.assertEqual(v.params, {})
        self.assertEqual(v.semantic_uri, "")

    def test_underlying_task_is_accessible(self):
        t = self._task()
        v = TaskNodeView(t)
        self.assertIs(v.task, t)


if __name__ == "__main__":
    unittest.main()
