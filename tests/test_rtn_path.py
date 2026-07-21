"""Regression tests: the RTN-first solver entry produces results identical to
the legacy ``solve_recipe_ir_with_cp_sat`` path on representative scenarios.

Phase 1 of the CP-SAT planner refactor introduces ``solve_rtn_with_cp_sat``
which accepts an ``RTNModel`` directly. Internally it delegates to the
existing solver via ``rtn_to_recipe_ir``. These tests pin the parity contract
so future Phase-2 internal rewrites cannot regress observable behavior.
"""
from __future__ import annotations

import unittest
from dataclasses import asdict

try:
    from ..recipe_ir import build_recipe_ir
    from ..recipe_to_rtn import recipe_ir_to_rtn
    from ..cp_sat_planner import (
        PlannerConfig,
        solve_recipe_ir_with_cp_sat,
        solve_rtn_with_cp_sat,
    )
except ImportError:  # pragma: no cover
    from recipe_ir import build_recipe_ir
    from recipe_to_rtn import recipe_ir_to_rtn
    from cp_sat_planner import (
        PlannerConfig,
        solve_recipe_ir_with_cp_sat,
        solve_rtn_with_cp_sat,
    )


def _profit_scenario():
    spec = {
        "id": "UnitTest_RTN_Parity_Profit",
        "volume": 3.0,
        "procedure": [
            {
                "parallel": [
                    {"dose": {"ingredient": "A", "amount_L": 1.0}},
                    {"dose": {"ingredient": "B", "amount_L": 2.0}},
                ],
                "join": "wait_all",
            },
            {"mix": {"rpm": 200, "duration_s": 30}},
            {
                "choice": [
                    {"branch_id": "fast", "steps": [{"usage": {"duration_s": 100}}]},
                    {"branch_id": "slow", "steps": [{"usage": {"duration_s": 300}}]},
                ],
                "select": "optimizer",
            },
        ],
    }
    module_ops = {
        "HC10": [("Draining", 0.2, 3), ("Filling", 0.2, 0), ("Connect", "", 1), ("None", "", 0), ("Stirring", "200", 2)],
        "HC20": [("Draining", 0.2, 3), ("Filling", 0.2, 0), ("Connect", "", 1), ("None", "", 0), ("Stirring", "200", 1)],
        "HC30": [("Draining", 0.2, 3), ("Filling", 0.2, 0), ("Connect", "", 1), ("None", "", 0), ("Stirring", "200", 1)],
    }
    module_interfaces = {
        "HC10": [("Input", "HC10_In1"), ("Output", "HC10_Out1")],
        "HC20": [("Input", "HC20_In1"), ("Output", "HC20_Out1")],
        "HC30": [("Input", "HC30_In1"), ("Input", "HC30_In2"), ("Output", "HC30_Out1")],
    }
    module_resources = {"HC10": ["A", 10], "HC20": ["B", 10]}
    module_max = {"HC10": [10], "HC20": [10], "HC30": [10]}
    cfg = PlannerConfig(
        base_profit=100,
        lambda_per_second=-0.01,
        connect_duration_s=3.0,
        solver_time_limit_s=5,
        num_workers=1,
    )
    return spec, module_ops, module_interfaces, module_max, module_resources, cfg


def _separation_scenario():
    spec = {
        "id": "UnitTest_RTN_Parity_Separation",
        "volume": 2.0,
        "procedure": [
            {"dose": {"ingredient": "A", "amount_L": 1.0}},
            {"dose": {"ingredient": "B", "amount_L": 1.0}},
            {"mix": {"rpm": 200, "duration_s": 20}},
            {"usage": {"duration_s": 60}},
            {"settling": {"duration_s": 30}},
            {"separation": {"order": ["A", "B"]}},
        ],
    }
    module_ops = {
        "W1": [("Draining", 0.2, 3), ("Filling", 0.2, 0), ("Connect", "", 1), ("None", "", 0), ("Stirring", "200", 2), ("Settling", "", 0)],
        "W2": [("Draining", 0.2, 3), ("Filling", 0.2, 0), ("Connect", "", 1), ("None", "", 0)],
    }
    module_interfaces = {
        "W1": [("Input", "W1_In1"), ("Output", "W1_Out1")],
        "W2": [("Input", "W2_In1"), ("Output", "W2_Out1")],
    }
    module_resources = {"W1": ["A", 10]}
    module_max = {"W1": [10], "W2": [10]}
    cfg = PlannerConfig(
        base_profit=80,
        lambda_per_second=-0.01,
        connect_duration_s=2.0,
        solver_time_limit_s=5,
        num_workers=1,
    )
    return spec, module_ops, module_interfaces, module_max, module_resources, cfg


class RTNPathParityTests(unittest.TestCase):
    """Each scenario is run twice (legacy and RTN-first) and compared."""

    def _run_both(self, spec, module_ops, module_interfaces, module_max, module_resources, cfg):
        ir = build_recipe_ir(spec)
        legacy = solve_recipe_ir_with_cp_sat(
            ir,
            module_ops,
            module_interfaces=module_interfaces,
            module_maximum_volume=module_max,
            module_resources=module_resources,
            config=cfg,
        )
        model = recipe_ir_to_rtn(
            ir,
            module_ops=module_ops,
            module_interfaces=module_interfaces,
            module_maximum_volume=module_max,
            module_resources=module_resources,
        )
        rtn_path = solve_rtn_with_cp_sat(model, config=cfg)
        return legacy, rtn_path

    def _assert_results_equal(self, a, b):
        self.assertEqual(a.status, b.status)
        self.assertAlmostEqual(a.objective_profit, b.objective_profit, places=4)
        self.assertAlmostEqual(a.makespan_s, b.makespan_s, places=4)
        self.assertAlmostEqual(a.total_cost, b.total_cost, places=4)
        self.assertEqual(a.selected_branches, b.selected_branches)

        def _material_balance(result):
            balance: dict = {}
            for op in result.operations:
                for mat, qty in (op.material or {}).items():
                    if op.source_module:
                        balance[(op.source_module, mat)] = balance.get((op.source_module, mat), 0.0) - qty
                    if op.target_module:
                        balance[(op.target_module, mat)] = balance.get((op.target_module, mat), 0.0) + qty
            return {k: v for k, v in balance.items() if abs(v) > 1e-9}

        bal_a = _material_balance(a)
        bal_b = _material_balance(b)
        self.assertEqual(bal_a, bal_b, f"Material balance mismatch:\n  {bal_a}\n  !=\n  {bal_b}")

    def test_profit_scenario_parity(self):
        legacy, rtn_path = self._run_both(*_profit_scenario())
        self.assertIn(legacy.status, {"OPTIMAL", "FEASIBLE"})
        self._assert_results_equal(legacy, rtn_path)

    def test_separation_scenario_parity(self):
        # Even if the scenario turns out infeasible due to plant config,
        # both paths should return the same status — that's what we pin here.
        legacy, rtn_path = self._run_both(*_separation_scenario())
        self.assertEqual(legacy.status, rtn_path.status)
        if legacy.status in {"OPTIMAL", "FEASIBLE"}:
            self._assert_results_equal(legacy, rtn_path)

    def test_solve_rtn_with_cp_sat_requires_plant_metadata(self):
        ir = build_recipe_ir(
            {
                "id": "RTN_NoPlant",
                "volume": 50.0,
                "procedure": [{"dose": {"ingredient": "A", "amount_L": 50.0}}],
            }
        )
        # Build RTN without module_ops → metadata['plant']['module_ops'] missing.
        model = recipe_ir_to_rtn(ir)
        with self.assertRaises(ValueError):
            solve_rtn_with_cp_sat(model)


if __name__ == "__main__":
    unittest.main()
