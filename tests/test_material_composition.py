from __future__ import annotations

import unittest

try:
    from cp_sat_planner import (
        MaterialReservation,
        PlannedOperation,
        PlannerConfig,
        PlannerResult,
        solve_recipe_ir_with_cp_sat,
    )
    from material_composition import compositions_proportional
    from recipe_ir import build_recipe_ir
    from schedule_validator import validate_schedule
except ImportError:  # pragma: no cover
    from ..scripts.cp_sat_planner import (
        MaterialReservation,
        PlannedOperation,
        PlannerConfig,
        PlannerResult,
        solve_recipe_ir_with_cp_sat,
    )
    from ..scripts.material_composition import compositions_proportional
    from ..scripts.recipe_ir import build_recipe_ir
    from ..scripts.schedule_validator import validate_schedule


def _capabilities(*, stir: bool = False, usage: bool = False):
    operations = [
        ("Draining", 1.0, 1),
        ("Filling", 1.0, 0),
        ("Connect", "", 2, 0, 0, 1),
        ("Disconnect", "", 2, 0, 0, 1),
    ]
    if stir:
        operations.append(("Stirring", 150, 1))
    if usage:
        operations.append(("None", "", 0))
    return operations


class DynamicCompositionTests(unittest.TestCase):
    @staticmethod
    def _partial_mixture_replay(
        *,
        selective: bool = False,
        selective_transfer_kind: str = "pure_material",
    ):
        ir = build_recipe_ir(
            {
                "id": "PartialMixtureReplay",
                "volume": 3.0,
                "procedure": [
                    {
                        "parallel": [
                            {"dose": {"ingredient": "A", "amount_L": 1.0}},
                            {"dose": {"ingredient": "B", "amount_L": 2.0}},
                        ],
                        "join": "wait_all",
                    }
                ],
            }
        )
        transferred = {"B": 1.0} if selective else {"A": 0.5, "B": 1.0}
        transfer_kind = selective_transfer_kind if selective else "mixture"
        signature = "MATERIAL:B" if selective else "MIX:A=0.5|B=1"
        operations = [
            PlannedOperation(
                step_id=1,
                recipe_node_id="CONNECT_A",
                branch_group_id="",
                branch_id="",
                operation_type="connect",
                operation="Connect A",
                module="A_SOURCE->MIXER",
                source_module="A_SOURCE",
                target_module="MIXER",
                out_port="A_Out1",
                in_port="M_In1",
                connection_path="A_SOURCE.A_Out1 -> MIXER.M_In1",
                start_s=0,
                end_s=1,
                duration_s=1,
                trace={"material_signature": "MATERIAL:A"},
            ),
            PlannedOperation(
                step_id=2,
                recipe_node_id="CONNECT_B",
                branch_group_id="",
                branch_id="",
                operation_type="connect",
                operation="Connect B",
                module="B_SOURCE->MIXER",
                source_module="B_SOURCE",
                target_module="MIXER",
                out_port="B_Out1",
                in_port="M_In2",
                connection_path="B_SOURCE.B_Out1 -> MIXER.M_In2",
                start_s=1,
                end_s=2,
                duration_s=1,
                trace={"material_signature": "MATERIAL:B"},
            ),
            PlannedOperation(
                step_id=3,
                recipe_node_id="dose_001",
                branch_group_id="AND_001",
                branch_id="b1",
                operation_type="dose",
                operation="Dose A",
                module="MIXER",
                source_module="A_SOURCE",
                target_module="MIXER",
                out_port="A_Out1",
                in_port="M_In1",
                connection_path="A_SOURCE.A_Out1 -> MIXER.M_In1",
                start_s=2,
                end_s=3,
                duration_s=1,
                transfer_duration_s=1,
                material={"A": 1.0},
                transfer_kind="pure_material",
            ),
            PlannedOperation(
                step_id=4,
                recipe_node_id="dose_002",
                branch_group_id="AND_001",
                branch_id="b2",
                operation_type="dose",
                operation="Dose B",
                module="MIXER",
                source_module="B_SOURCE",
                target_module="MIXER",
                out_port="B_Out1",
                in_port="M_In2",
                connection_path="B_SOURCE.B_Out1 -> MIXER.M_In2",
                start_s=2,
                end_s=4,
                duration_s=2,
                transfer_duration_s=2,
                material={"B": 2.0},
                transfer_kind="pure_material",
            ),
            PlannedOperation(
                step_id=5,
                recipe_node_id="CONNECT_MIXTURE",
                branch_group_id="",
                branch_id="",
                operation_type="connect",
                operation="Connect mixture route",
                module="MIXER->TARGET",
                source_module="MIXER",
                target_module="TARGET",
                out_port="M_Out1",
                in_port="T_In1",
                connection_path="MIXER.M_Out1 -> TARGET.T_In1",
                start_s=4,
                end_s=5,
                duration_s=1,
                trace={"material_signature": signature},
            ),
            PlannedOperation(
                step_id=6,
                recipe_node_id="AUX_PARTIAL",
                branch_group_id="",
                branch_id="",
                operation_type="aux_transfer",
                operation="Partial material transfer",
                module="TARGET",
                source_module="MIXER",
                target_module="TARGET",
                out_port="M_Out1",
                in_port="T_In1",
                connection_path="MIXER.M_Out1 -> TARGET.T_In1",
                start_s=5,
                end_s=6,
                duration_s=1,
                transfer_duration_s=1,
                material=transferred,
                transfer_kind=transfer_kind,
            ),
        ]
        plan = PlannerResult(
            status="FEASIBLE",
            objective_profit=0,
            base_profit=0,
            total_duration_s=6,
            makespan_s=6,
            total_cost=0,
            selected_branches={},
            operations=operations,
        )
        return validate_schedule(
            ir,
            plan,
            module_maximum_volume={
                "A_SOURCE": [10],
                "B_SOURCE": [10],
                "MIXER": [10],
                "TARGET": [10],
            },
            module_resources={"A_SOURCE": ["A", 10], "B_SOURCE": ["B", 10]},
            config=PlannerConfig(base_profit=0, lambda_per_second=0, require_final_disconnect=False),
        )

    def test_pure_surplus_moves_before_foreign_dose_for_all_solver_variants(self) -> None:
        ir = build_recipe_ir(
            {
                "id": "PureSurplusBeforeContamination",
                "volume": 3.0,
                "procedure": [
                    {
                        "parallel": [
                            {"dose": {"ingredient": "A", "amount_L": 1.0}},
                            {"dose": {"ingredient": "B", "amount_L": 2.0}},
                        ],
                        "join": "wait_all",
                    },
                    {"mix": {"rpm": 150, "duration_s": 1}},
                ],
            }
        )
        module_ops = {
            "A_SOURCE": _capabilities(),
            "B_PROCESS": _capabilities(stir=True),
            "STORAGE": _capabilities(),
        }
        interfaces = {
            "A_SOURCE": [("Output", "A_Out1")],
            "B_PROCESS": [("Input", "B_In1"), ("Output", "B_Out1")],
            "STORAGE": [("Input", "S_In1")],
        }
        maximum = {"A_SOURCE": [10], "B_PROCESS": [10], "STORAGE": [20]}
        resources = {"A_SOURCE": ["A", 10], "B_PROCESS": ["B", 10]}

        for mode in ("lazy", "eager"):
            for workers in (1, 4, 8):
                for seed in (0, 1, 42):
                    with self.subTest(mode=mode, workers=workers, seed=seed):
                        cfg = PlannerConfig(
                            base_profit=100,
                            lambda_per_second=-1,
                            auxiliary_transfer_mode=mode,
                            num_workers=workers,
                            solver_random_seed=seed,
                            solver_time_limit_s=5,
                        )
                        result = solve_recipe_ir_with_cp_sat(
                            ir,
                            module_ops,
                            module_interfaces=interfaces,
                            module_maximum_volume=maximum,
                            module_resources=resources,
                            config=cfg,
                        )
                        replay = validate_schedule(
                            ir,
                            result,
                            module_maximum_volume=maximum,
                            module_resources=resources,
                            config=cfg,
                        )
                        self.assertIn(result.status, {"OPTIMAL", "FEASIBLE"}, result.diagnostics)
                        self.assertTrue(replay.valid, replay.errors)
                        self.assertEqual(replay.composition_errors, [])
                        reservation = next(
                            reservation
                            for reservation in result.local_dose_reservations
                            if reservation.material == "B"
                        )
                        self.assertEqual(reservation.process_module, "B_PROCESS")
                        self.assertAlmostEqual(reservation.amount_l, 2.0)
                        surplus = next(
                            operation
                            for operation in result.operations
                            if operation.operation_type == "aux_transfer"
                            and operation.source_module == "B_PROCESS"
                            and operation.material == {"B": 8.0}
                        )
                        dose_a = next(
                            operation
                            for operation in result.operations
                            if operation.operation_type == "dose" and operation.material == {"A": 1.0}
                        )
                        self.assertEqual(surplus.transfer_kind, "pure_material")
                        self.assertLessEqual(surplus.end_s, dose_a.start_s)

    def test_validator_rejects_pure_b_drain_after_a_contaminates_source(self) -> None:
        ir = build_recipe_ir(
            {
                "id": "RejectSelectiveDrain",
                "volume": 3.0,
                "procedure": [
                    {"dose": {"ingredient": "A", "amount_L": 1.0}},
                    {"dose": {"ingredient": "B", "amount_L": 2.0}},
                    {"mix": {"rpm": 150, "duration_s": 1}},
                ],
            }
        )
        operations = [
            PlannedOperation(
                1, "connect_a", "", "", "connect", "Connect A", "A_SOURCE->B_PROCESS",
                source_module="A_SOURCE", target_module="B_PROCESS", out_port="A_Out1", in_port="B_In1",
                connection_path="A_SOURCE.A_Out1 -> B_PROCESS.B_In1", start_s=0, end_s=1,
                duration_s=1, trace={"material_signature": "MATERIAL:A"},
            ),
            PlannedOperation(
                2, "dose_001", "", "", "dose", "Dose A", "B_PROCESS",
                source_module="A_SOURCE", target_module="B_PROCESS", out_port="A_Out1", in_port="B_In1",
                connection_path="A_SOURCE.A_Out1 -> B_PROCESS.B_In1", start_s=1, end_s=2,
                duration_s=1, transfer_duration_s=1, material={"A": 1.0},
            ),
            PlannedOperation(
                3, "connect_b", "", "", "connect", "Connect B", "B_PROCESS->STORAGE",
                source_module="B_PROCESS", target_module="STORAGE", out_port="B_Out1", in_port="S_In1",
                connection_path="B_PROCESS.B_Out1 -> STORAGE.S_In1", start_s=2, end_s=3,
                duration_s=1, trace={"material_signature": "MATERIAL:B"},
            ),
            PlannedOperation(
                4, "AUX_BAD", "", "", "aux_transfer", "Illegal pure B drain", "STORAGE",
                source_module="B_PROCESS", target_module="STORAGE", out_port="B_Out1", in_port="S_In1",
                connection_path="B_PROCESS.B_Out1 -> STORAGE.S_In1", start_s=3, end_s=11,
                duration_s=8, transfer_duration_s=8, material={"B": 8.0}, transfer_kind="pure_material",
            ),
            PlannedOperation(
                5, "mix_001", "", "", "mix", "Mix", "B_PROCESS",
                start_s=11, end_s=12, duration_s=1,
            ),
        ]
        result = PlannerResult(
            status="FEASIBLE",
            objective_profit=0,
            base_profit=0,
            total_duration_s=12,
            makespan_s=12,
            total_cost=0,
            selected_branches={},
            operations=operations,
            local_dose_reservations=[MaterialReservation("dose_002", "B_PROCESS", "B", 2.0)],
        )
        replay = validate_schedule(
            ir,
            result,
            module_maximum_volume={"A_SOURCE": [10], "B_PROCESS": [10], "STORAGE": [20]},
            module_resources={"A_SOURCE": ["A", 10], "B_PROCESS": ["B", 10]},
            config=PlannerConfig(require_final_disconnect=False),
        )
        self.assertFalse(replay.valid)
        self.assertTrue(
            any("Source composition violation" in error for error in replay.composition_errors),
            replay.composition_errors,
        )

    def test_solver_moves_complete_mixture_between_process_modules(self) -> None:
        ir = build_recipe_ir(
            {
                "id": "MoveMixtureToUsage",
                "volume": 3.0,
                "procedure": [
                    {
                        "parallel": [
                            {"dose": {"ingredient": "A", "amount_L": 1.0}},
                            {"dose": {"ingredient": "B", "amount_L": 2.0}},
                        ],
                        "join": "wait_all",
                    },
                    {"mix": {"rpm": 150, "duration_s": 1}},
                    {"usage": {"duration_s": 1}},
                ],
            }
        )
        module_ops = {
            "A_SOURCE": _capabilities(),
            "B_SOURCE": _capabilities(),
            "MIXER": _capabilities(stir=True),
            "USER": _capabilities(usage=True),
        }
        interfaces = {
            "A_SOURCE": [("Output", "A_Out1")],
            "B_SOURCE": [("Output", "B_Out1")],
            "MIXER": [("Input", "M_In1"), ("Input", "M_In2"), ("Output", "M_Out1")],
            "USER": [("Input", "U_In1")],
        }
        maximum = {module: [10] for module in module_ops}
        resources = {"A_SOURCE": ["A", 10], "B_SOURCE": ["B", 10]}
        for mode in ("lazy", "eager"):
            with self.subTest(mode=mode):
                cfg = PlannerConfig(
                    base_profit=100,
                    lambda_per_second=-1,
                    auxiliary_transfer_mode=mode,
                    solver_time_limit_s=10,
                )
                result = solve_recipe_ir_with_cp_sat(
                    ir,
                    module_ops,
                    module_interfaces=interfaces,
                    module_maximum_volume=maximum,
                    module_resources=resources,
                    config=cfg,
                )
                replay = validate_schedule(
                    ir,
                    result,
                    module_maximum_volume=maximum,
                    module_resources=resources,
                    config=cfg,
                )
                self.assertIn(result.status, {"OPTIMAL", "FEASIBLE"}, result.diagnostics)
                self.assertTrue(replay.valid, replay.errors)
                mixture = next(
                    operation
                    for operation in result.operations
                    if operation.operation_type == "aux_transfer"
                    and operation.transfer_kind == "mixture"
                )
                self.assertEqual(mixture.material, {"A": 1.0, "B": 2.0})
                self.assertEqual(mixture.source_module, "MIXER")
                self.assertEqual(mixture.target_module, "USER")

    def test_partial_mixture_is_proportional_but_selective_component_is_not(self) -> None:
        self.assertTrue(compositions_proportional({"A": 1, "B": 2}, {"A": 0.5, "B": 1}))
        self.assertFalse(compositions_proportional({"A": 1, "B": 2}, {"B": 1}))

        proportional = self._partial_mixture_replay()
        self.assertTrue(proportional.valid, proportional.errors)
        self.assertEqual(proportional.composition_errors, [])
        self.assertTrue(any("AUX_PARTIAL" in line for line in proportional.composition_log))

        selective = self._partial_mixture_replay(selective=True)
        self.assertFalse(selective.valid)
        self.assertTrue(
            any("Source composition violation" in error for error in selective.composition_errors),
            selective.composition_errors,
        )

        selective_as_mixture = self._partial_mixture_replay(
            selective=True,
            selective_transfer_kind="mixture",
        )
        self.assertFalse(selective_as_mixture.valid)
        self.assertTrue(
            any("Mixture transfer ratio violation" in error for error in selective_as_mixture.composition_errors),
            selective_as_mixture.composition_errors,
        )


if __name__ == "__main__":
    unittest.main()
