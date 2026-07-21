from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cp_sat_planner import PlannerConfig, solve_recipe_ir_with_cp_sat
from pipeline import plan_general_recipe_runtime_scenarios, plan_general_recipe_xml
from recipe_ir import build_recipe_ir
from schedule_validator import validate_schedule
from sample_data import (
    sample_module_interfaces,
    sample_module_maximum_volume,
    sample_module_ops,
    sample_module_resources,
)
from modplant_recipe import execute_graph, import_general_recipe, import_master_recipe


FIXTURES = Path(__file__).resolve().parent / "fixtures" / "runtime_control"


class RuntimeControlPlanningTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.config = PlannerConfig(solver_time_limit_s=8, num_workers=8)

    def fixture(self, exact_suffix: str) -> Path:
        matches = sorted(FIXTURES.glob(f"*{exact_suffix}"))
        self.assertEqual(len(matches), 1, matches)
        return matches[0]

    def plan(self, path: Path, name: str, **runtime):
        return plan_general_recipe_xml(
            path,
            sample_module_ops(),
            module_interfaces=sample_module_interfaces(),
            module_maximum_volume=sample_module_maximum_volume(),
            module_resources=sample_module_resources(),
            planner_config=self.config,
            out_dir=Path(self.temp.name) / name,
            **runtime,
        )

    def activity_ids(self, result):
        return [
            operation.recipe_node_id
            for operation in result.planner_result.operations
            if operation.operation_type not in {"connect", "transfer", "aux_transfer"}
        ]

    def test_or_selects_two_branches_and_master_executes_both(self):
        path = self.fixture("_OR_42.xml")
        imported = import_general_recipe(path)
        split = next(node for node in imported.nodes if node.gateway_type == "OR_SPLIT")
        self.assertEqual(split.metadata["minBranches"], 2)
        self.assertEqual(split.metadata["maxBranches"], 2)

        result = self.plan(path, "or")
        self.assertEqual(result.planner_result.status, "OPTIMAL")
        self.assertTrue(result.validation_result.valid, result.validation_result.errors)
        selected = result.planner_result.selected_branches[split.metadata["gatewayGroupId"]]
        self.assertEqual(len(selected), 2)
        active_or = {
            operation.branch_id
            for operation in result.planner_result.operations
            if operation.branch_group_id == split.metadata["gatewayGroupId"]
            and operation.operation_type not in {"connect", "transfer", "aux_transfer"}
        }
        self.assertEqual(active_or, set(selected))

        # The planner resolved this OR, so the Master Recipe carries the two
        # branches it chose as parallel work rather than as a choice still to
        # be made; the gateway only survives when a live guard decides it. The
        # branches are named by the steps the plan scheduled for them, because
        # a Master Recipe round trip does not carry node metadata.
        chosen = {
            operation.recipe_node_id
            for operation in result.planner_result.operations
            if operation.branch_group_id == split.metadata["gatewayGroupId"]
            and operation.branch_id in set(selected)
            and operation.operation_type not in {"connect", "transfer", "aux_transfer"}
        }
        self.assertEqual(len(chosen), len(selected))
        master = import_master_recipe(result.master_xml_path)
        self.assertTrue(chosen.issubset({node.id for node in master.nodes}))
        self.assertNotIn(
            "OR_SPLIT", {node.gateway_type for node in master.nodes},
            "a choice the planner already made is not left open in the recipe",
        )
        execution = execute_graph(master)
        self.assertTrue(execution.completed)
        self.assertTrue(chosen.issubset(set(execution.completed_activities)))

    def test_or_selected_material_branches_are_summed_at_join(self):
        recipe = build_recipe_ir({
            "id": "OR_Material_Join",
            "volume": 2.0,
            "procedure": [
                {
                    "inclusive": [
                        {"branch_id": "dose_a", "steps": [{"dose": {"ingredient": "A", "amount_L": 1.0}}]},
                        {"branch_id": "dose_b", "steps": [{"dose": {"ingredient": "B", "amount_L": 1.0}}]},
                    ],
                    "min_branches": 2,
                    "max_branches": 2,
                },
                {"mix": {"rpm": 100, "duration_s": 10}},
                {"settling": {"duration_s": 10}},
                {"separation": {"order": ["A", "B"]}},
            ],
        })
        result = solve_recipe_ir_with_cp_sat(
            recipe,
            sample_module_ops(),
            module_interfaces=sample_module_interfaces(),
            module_maximum_volume=sample_module_maximum_volume(),
            module_resources=sample_module_resources(),
            config=self.config,
        )
        self.assertIn(result.status, {"OPTIMAL", "FEASIBLE"})
        self.assertEqual(result.selected_branches["OR_001"], ["dose_a", "dose_b"])
        validation = validate_schedule(
            recipe,
            result,
            module_maximum_volume=sample_module_maximum_volume(),
            module_resources=sample_module_resources(),
            config=self.config,
        )
        self.assertTrue(validation.valid, validation.errors)

    def test_runtime_loop_iteration_context_changes_finite_plan(self):
        path = self.fixture("_RuntimeLoop_42.xml")
        one = self.plan(path, "loop-one", loop_iterations={"quality_conditioning_loop": 1})
        three = self.plan(path, "loop-three", loop_iterations={"quality_conditioning_loop": 3})
        self.assertTrue(one.validation_result.valid, one.validation_result.errors)
        self.assertTrue(three.validation_result.valid, three.validation_result.errors)
        one_loop = [node for node in self.activity_ids(one) if "quality_conditioning_loop__" in node]
        three_loop = [node for node in self.activity_ids(three) if "quality_conditioning_loop__" in node]
        self.assertEqual(len(one_loop), 1)
        self.assertEqual(len(three_loop), 3)
        master = import_master_recipe(three.master_xml_path)
        self.assertEqual(
            master.metadata["runtimePlanning"]["loopIterations"],
            {"quality_conditioning_loop": 3},
        )

    def test_conditional_jump_context_changes_planned_path(self):
        path = self.fixture("_ConditionalJump_42.xml")
        normal = self.plan(
            path,
            "jump-false",
            planning_context={"quality": {"skipUsage": False}},
        )
        jumped = self.plan(
            path,
            "jump-true",
            planning_context={"quality": {"skipUsage": True}},
        )
        self.assertTrue(normal.validation_result.valid, normal.validation_result.errors)
        self.assertTrue(jumped.validation_result.valid, jumped.validation_result.errors)
        self.assertTrue(any(node.startswith("usage_") for node in self.activity_ids(normal)))
        self.assertFalse(any(node.startswith("usage_") for node in self.activity_ids(jumped)))
        self.assertEqual(
            jumped.recipe_ir.metadata["runtimePlanning"]["jumpDecisions"],
            {"jump_skip_usage": True},
        )

    def test_combined_recipe_builds_a_two_member_jump_plan_family(self):
        path = self.fixture("_OR_Loop_Jump_42.xml")
        scenarios = plan_general_recipe_runtime_scenarios(
            path,
            sample_module_ops(),
            module_interfaces=sample_module_interfaces(),
            module_maximum_volume=sample_module_maximum_volume(),
            module_resources=sample_module_resources(),
            planner_config=self.config,
            out_dir=Path(self.temp.name) / "combined",
            loop_iterations={"quality_conditioning_loop": 2},
            max_scenarios=4,
        )
        self.assertEqual(len(scenarios), 2)
        self.assertEqual(
            {scenario.jump_decisions["combined_jump_skip_optional_work"] for scenario in scenarios},
            {False, True},
        )
        for scenario in scenarios:
            self.assertIn(scenario.result.planner_result.status, {"OPTIMAL", "FEASIBLE"})
            self.assertTrue(scenario.result.validation_result.valid)
            selected_or = [
                value
                for group, value in scenario.result.planner_result.selected_branches.items()
                if group.startswith("OR_")
            ]
            if scenario.jump_decisions["combined_jump_skip_optional_work"]:
                self.assertEqual(selected_or, [])
            else:
                self.assertEqual(len(selected_or), 1)
                self.assertEqual(len(selected_or[0]), 2)


if __name__ == "__main__":
    unittest.main()
