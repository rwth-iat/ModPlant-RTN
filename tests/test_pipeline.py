from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

try:
    from ..recipe_ir import build_recipe_ir, recipe_spec_from_legacy_order
    from ..isa88_recipe import parse_general_recipe_xml_to_ir, save_general_recipe_xml_from_ir
except ImportError:  # pragma: no cover - direct script execution from CPN/
    from recipe_ir import build_recipe_ir, recipe_spec_from_legacy_order
    from isa88_recipe import parse_general_recipe_xml_to_ir, save_general_recipe_xml_from_ir


def _planner_imports():
    from cp_sat_planner import PlannerConfig, solve_recipe_ir_with_cp_sat
    from schedule_validator import validate_schedule
    return PlannerConfig, solve_recipe_ir_with_cp_sat, validate_schedule


class RecipeIRAndISA88Tests(unittest.TestCase):
    def test_usage_and_time_based_energy_weights_change_equipment_selection(self) -> None:
        PlannerConfig, solve_recipe_ir_with_cp_sat, _ = _planner_imports()
        ir = build_recipe_ir({
            "id": "CostChoice",
            "volume": 1.0,
            "procedure": [{"mix": {"rpm": 100, "duration_s": 10}}],
        })
        module_ops = {
            "LOW_ENERGY": [("Stirring", 100, 5, 0.0, 0.01)],
            "LOW_USAGE": [("Stirring", 100, 0, 0.01, 0.01)],
            "LOW_CO2": [("Stirring", 100, 5, 0.01, 0.0)],
        }
        maximum = {"LOW_ENERGY": [10], "LOW_USAGE": [10], "LOW_CO2": [10]}

        usage_result = solve_recipe_ir_with_cp_sat(
            ir,
            module_ops,
            module_maximum_volume=maximum,
            config=PlannerConfig(
                base_profit=100,
                lambda_per_second=0,
                usage_cost_weight=1,
                energy_cost_weight=0,
                co2_penalty=0,
            ),
        )
        energy_result = solve_recipe_ir_with_cp_sat(
            ir,
            module_ops,
            module_maximum_volume=maximum,
            config=PlannerConfig(
                base_profit=100,
                lambda_per_second=0,
                usage_cost_weight=0,
                energy_cost_weight=1000,
                co2_penalty=0,
                electricity_price_eur_per_kwh=0.3,
            ),
        )
        co2_result = solve_recipe_ir_with_cp_sat(
            ir,
            module_ops,
            module_maximum_volume=maximum,
            config=PlannerConfig(
                base_profit=100,
                lambda_per_second=0,
                usage_cost_weight=0,
                energy_cost_weight=0,
                co2_penalty=1000,
            ),
        )

        self.assertEqual(usage_result.operations[0].module, "LOW_USAGE")
        self.assertEqual(energy_result.operations[0].module, "LOW_ENERGY")
        self.assertEqual(co2_result.operations[0].module, "LOW_CO2")
        self.assertEqual(usage_result.operations[0].energy_consumption_kwh, 0.1)
        self.assertEqual(usage_result.operations[0].energy_cost, 0.03)

    def test_equipment_usage_cost_is_per_invocation_not_per_litre_or_second(self) -> None:
        PlannerConfig, solve_recipe_ir_with_cp_sat, _ = _planner_imports()
        module_ops = {
            "SOURCE": [("Draining", 1.0, 3), ("Connect", "", 1)],
            "TARGET": [("Filling", 1.0, 0), ("Connect", "", 1)],
        }
        interfaces = {"SOURCE": [("Output", "SOURCE_Out")], "TARGET": [("Input", "TARGET_In")]}
        config = PlannerConfig(
            base_profit=100,
            lambda_per_second=0,
            enable_auxiliary_transfers=False,
            require_final_disconnect=True,
        )
        totals = []
        durations = []
        for amount in (1.0, 5.0):
            ir = build_recipe_ir({
                "id": f"Dose{amount:g}",
                "volume": amount,
                "procedure": [{"dose": {"ingredient": "A", "amount_L": amount}}],
            })
            result = solve_recipe_ir_with_cp_sat(
                ir,
                module_ops,
                module_interfaces=interfaces,
                module_maximum_volume={"SOURCE": [10], "TARGET": [10]},
                module_resources={"SOURCE": ["A", 10]},
                config=config,
            )
            totals.append(result.total_usage_cost)
            durations.append(sum(op.duration_s for op in result.operations if op.operation_type == "dose"))

        # 3 usage units for Draining, 2 for Connect (both endpoints), and
        # 2 for the mandatory matching Disconnect. Volume only changes the
        # flow duration, never these per-invocation equipment costs.
        self.assertEqual(totals, [7.0, 7.0])
        self.assertEqual(durations, [1.0, 5.0])

    def test_parallel_and_choice_metadata_roundtrip(self) -> None:
        spec = {
            "id": "UnitTest_Parallel_Choice",
            "volume": 6.0,
            "procedure": [
                {
                    "parallel": [
                        {"dose": {"ingredient": "A", "amount_L": 1.0}},
                        {"dose": {"ingredient": "B", "amount_L": 2.0}},
                        {"dose": {"ingredient": "C", "amount_L": 3.0}},
                    ],
                    "join": "wait_all",
                },
                {"mix": {"rpm": 200, "duration_s": 30}},
                {
                    "choice": [
                        {"branch_id": "fast", "steps": [{"usage": {"duration_s": 1800}}]},
                        {"branch_id": "standard", "steps": [{"usage": {"duration_s": 3600}}]},
                    ],
                    "select": "optimizer",
                },
                {"settling": {"duration_s": 300}},
                {"separation": {"order": ["C", "B", "A"]}},
            ],
        }
        ir = build_recipe_ir(spec)
        with tempfile.NamedTemporaryFile(suffix=".xml") as tmp:
            out = Path(tmp.name)
            save_general_recipe_xml_from_ir(ir, out)
            parsed = parse_general_recipe_xml_to_ir(out)

        self.assertEqual(len(parsed.nodes), len(ir.nodes))
        self.assertEqual(len(parsed.edges), len(ir.edges))
        self.assertIn("AND_001", parsed.choice_groups())
        self.assertTrue(any(n.control_node_type == "AND_JOIN" for n in parsed.nodes.values()))
        self.assertTrue(any(n.control_node_type == "XOR_SPLIT" for n in parsed.nodes.values()))

    def test_legacy_order_conversion(self) -> None:
        order = {
            "volume": 6.0,
            "order": ["A", "B", "C", {"mix": {"rpm": 200, "duration": 30}}],
            "ratio": {"A": [1], "B": [2], "C": [3]},
            "usage_and_settling": [3600, 300],
            "separation_order": ["C", "B", "A"],
        }
        spec = recipe_spec_from_legacy_order(order, parallel_dosing_before_mix=True)
        ir = build_recipe_ir(spec)
        self.assertEqual(ir.inputs, {"A": 1.0, "B": 2.0, "C": 3.0})
        self.assertTrue(any(n.control_node_type == "AND_SPLIT" for n in ir.nodes.values()))


@unittest.skipIf(importlib.util.find_spec("ortools") is None, "OR-Tools is not installed in this Python environment")
class CPSATPipelineTests(unittest.TestCase):
    def test_connection_action_durations_prefer_aas_and_use_separate_fallbacks(self) -> None:
        from cp_sat_planner import PlannerConfig, _connection_action_profile

        pair = ("SOURCE", "Out1", "TARGET", "In1")
        cfg = PlannerConfig(connect_duration_s=7, disconnect_duration_s=5)
        explicit = {
            "SOURCE": [("Connect", "", 2, 0, 0, 3), ("Disconnect", "", 2, 0, 0, 2)],
            "TARGET": [("Connect", "", 2, 0, 0, 3), ("Disconnect", "", 2, 0, 0, 2)],
        }
        connect_duration, connect_cost = _connection_action_profile(pair, "connect", explicit, cfg)
        disconnect_duration, disconnect_cost = _connection_action_profile(pair, "disconnect", explicit, cfg)
        self.assertEqual(connect_duration, 3)
        self.assertEqual(disconnect_duration, 2)
        self.assertEqual(connect_cost.usage_cost, 4)
        self.assertEqual(disconnect_cost.usage_cost, 4)

        no_fixed_duration = {
            "SOURCE": [("Connect", "", 2), ("Disconnect", "", 2)],
            "TARGET": [("Connect", "", 2), ("Disconnect", "", 2)],
        }
        self.assertEqual(_connection_action_profile(pair, "connect", no_fixed_duration, cfg)[0], 7)
        self.assertEqual(_connection_action_profile(pair, "disconnect", no_fixed_duration, cfg)[0], 5)

        missing_disconnect = {
            "SOURCE": [("Connect", "", 2)],
            "TARGET": [("Connect", "", 2)],
        }
        fallback_duration, fallback_cost = _connection_action_profile(
            pair,
            "disconnect",
            missing_disconnect,
            cfg,
        )
        self.assertEqual(fallback_duration, 5)
        self.assertEqual(fallback_cost.usage_cost, 4)

    def test_connection_lifecycle_material_signatures_and_port_exclusivity(self) -> None:
        from cp_sat_planner import PlannedOperation
        from schedule_validator import _check_connection_lifecycle

        def operation(
            step: int,
            kind: str,
            start: float,
            end: float,
            *,
            source: str = "SOURCE",
            target: str = "TARGET",
            out_port: str = "Out1",
            in_port: str = "In1",
            material=None,
            signature: str = "",
        ) -> PlannedOperation:
            return PlannedOperation(
                step_id=step,
                recipe_node_id=f"{kind}_{step}",
                branch_group_id="",
                branch_id="",
                operation_type=kind,
                operation=kind,
                module=f"{source}->{target}",
                source_module=source,
                target_module=target,
                out_port=out_port,
                in_port=in_port,
                connection_path=f"{source}.{out_port} -> {target}.{in_port}",
                start_s=start,
                end_s=end,
                duration_s=end - start,
                material=material or {},
                trace={"material_signature": signature} if signature else {},
            )

        errors: list[str] = []
        log: list[str] = []
        _check_connection_lifecycle(
            [
                operation(1, "connect", 0, 3, signature="MATERIAL:A"),
                operation(2, "dose", 3, 4, material={"A": 1}),
                operation(3, "dose", 4, 9, material={"A": 5}),
                operation(4, "disconnect", 9, 12, signature="MATERIAL:A"),
            ],
            errors,
            log,
            1e-6,
        )
        self.assertEqual(errors, [])
        self.assertEqual(sum("reused" in entry for entry in log), 2)

        open_terminal_errors: list[str] = []
        open_terminal_log: list[str] = []
        terminal_connections = _check_connection_lifecycle(
            [
                operation(1, "connect", 0, 3, signature="MATERIAL:A"),
                operation(2, "dose", 3, 4, material={"A": 1}),
            ],
            open_terminal_errors,
            open_terminal_log,
            1e-6,
            require_final_disconnect=False,
        )
        self.assertEqual(open_terminal_errors, [])
        self.assertEqual(len(terminal_connections), 1)
        self.assertTrue(any("END ACTIVE" in entry for entry in open_terminal_log))

        mixture_errors: list[str] = []
        _check_connection_lifecycle(
            [
                operation(1, "connect", 0, 3, signature="MIX:A=1|B=2"),
                operation(2, "transfer", 3, 4, material={"A": 1, "B": 2}),
                operation(3, "transfer", 4, 5, material={"A": 1, "B": 2}),
                operation(4, "transfer", 5, 6, material={"A": 0.5, "B": 1}),
                operation(5, "disconnect", 6, 9, signature="MIX:A=1|B=2"),
            ],
            mixture_errors,
            [],
            1e-6,
        )
        self.assertTrue(any("material signature mismatch" in error for error in mixture_errors))

        reconfigured_pair_errors: list[str] = []
        _check_connection_lifecycle(
            [
                operation(1, "connect", 0, 3, signature="MATERIAL:A"),
                operation(2, "transfer", 3, 4, material={"A": 1}),
                operation(3, "disconnect", 4, 6, signature="MATERIAL:A"),
                operation(4, "connect", 6, 9, signature="MATERIAL:B"),
                operation(5, "transfer", 9, 10, material={"B": 1}),
            ],
            reconfigured_pair_errors,
            [],
            1e-6,
            require_final_disconnect=False,
        )
        self.assertEqual(reconfigured_pair_errors, [])

        occupied_port_errors: list[str] = []
        _check_connection_lifecycle(
            [
                operation(1, "connect", 0, 3, signature="MATERIAL:A"),
                operation(
                    2,
                    "connect",
                    3,
                    6,
                    target="OTHER",
                    in_port="OtherIn",
                    signature="MATERIAL:A",
                ),
            ],
            occupied_port_errors,
            [],
            1e-6,
        )
        self.assertTrue(any("occupied port" in error for error in occupied_port_errors))

    def test_equivalent_connection_reuse_is_an_objective_tradeoff_not_a_hard_rule(self) -> None:
        PlannerConfig, solve_recipe_ir_with_cp_sat, validate_schedule = _planner_imports()

        ir = build_recipe_ir({
            "id": "ConnectionReuseTradeoff",
            "volume": 200.0,
            "procedure": [
                {
                    "parallel": [
                        {"dose": {"ingredient": "A", "amount_L": 100.0}},
                        {"dose": {"ingredient": "A", "amount_L": 100.0}},
                    ],
                    "join": "wait_all",
                },
                {"mix": {"rpm": 150, "duration_s": 1}},
            ],
        })
        module_ops = {
            "SOURCE": [
                ("Draining", 1.0, 0),
                ("Connect", "", 5, 0, 0, 3),
                ("Disconnect", "", 5, 0, 0, 3),
            ],
            "TARGET": [
                ("Filling", 1.0, 0),
                ("Stirring", "150", 0),
                ("Connect", "", 5, 0, 0, 3),
                ("Disconnect", "", 5, 0, 0, 3),
            ],
        }
        interfaces = {
            "SOURCE": [("Output", "Out1"), ("Output", "Out2")],
            "TARGET": [("Input", "In1"), ("Input", "In2")],
        }
        maximum = {"SOURCE": [250], "TARGET": [250]}
        resources = {"SOURCE": ["A", 200]}

        def solve(time_penalty: float, require_final_disconnect: bool = False):
            cfg = PlannerConfig(
                base_profit=1000,
                lambda_per_second=time_penalty,
                solver_time_limit_s=10,
                enable_auxiliary_transfers=False,
                require_final_disconnect=require_final_disconnect,
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
            self.assertIn(result.status, {"OPTIMAL", "FEASIBLE"})
            self.assertTrue(replay.valid, replay.errors)
            return result

        economical = solve(0.0)
        fast = solve(-1.0)
        strict_teardown = solve(0.0, require_final_disconnect=True)
        economical_connects = [op for op in economical.operations if op.operation_type == "connect"]
        fast_connects = [op for op in fast.operations if op.operation_type == "connect"]

        # One Module is a unary equipment resource.  Extra free ports cannot make
        # two operations on SOURCE/TARGET overlap, so a second equivalent
        # connection offers no time advantage and is naturally rejected by
        # the objective in both economic and time-weighted modes.
        self.assertEqual(len(economical_connects), 1)
        self.assertEqual(len([op for op in economical.operations if op.operation_type == "disconnect"]), 0)
        self.assertEqual(len(economical.diagnostics["terminal_connections"]), 1)
        self.assertEqual(len(fast_connects), 1)
        self.assertEqual(len([op for op in fast.operations if op.operation_type == "disconnect"]), 0)
        self.assertEqual(len(fast.diagnostics["terminal_connections"]), 1)
        self.assertLessEqual(fast.makespan_s, economical.makespan_s)
        self.assertEqual(len([op for op in strict_teardown.operations if op.operation_type == "disconnect"]), 1)
        self.assertEqual(strict_teardown.diagnostics["terminal_connections"], [])

    def test_profit_planner_and_replay(self) -> None:
        PlannerConfig, solve_recipe_ir_with_cp_sat, validate_schedule = _planner_imports()

        spec = {
            "id": "UnitTest_CPSAT",
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
            require_final_disconnect=True,
        )
        ir = build_recipe_ir(spec)
        result = solve_recipe_ir_with_cp_sat(
            ir,
            module_ops,
            module_interfaces=module_interfaces,
            module_maximum_volume=module_max,
            module_resources=module_resources,
            config=cfg,
        )
        replay = validate_schedule(ir, result, module_maximum_volume=module_max, module_resources=module_resources, config=cfg)
        self.assertIn(result.status, {"OPTIMAL", "FEASIBLE"})
        self.assertTrue(replay.valid, replay.errors)
        self.assertEqual(result.selected_branches[next(iter(result.selected_branches))], "fast")
        self.assertTrue(any(row["OutPort"] and row["InPort"] for row in result.to_rows()))
        route_rows = [row for row in result.to_rows() if row["OutPort"] and row["InPort"]]
        connect_rows = [row for row in route_rows if row["Operation Type"] == "connect"]
        disconnect_rows = [row for row in route_rows if row["Operation Type"] == "disconnect"]
        flow_rows = [
            row for row in route_rows
            if row["Operation Type"] not in {"connect", "disconnect"}
        ]
        self.assertTrue(connect_rows)
        self.assertTrue(disconnect_rows)
        self.assertTrue(flow_rows)
        self.assertTrue(all(row["Connect Duration (s)"] == 3.0 and row["Transfer Duration (s)"] == 0.0 for row in connect_rows))
        self.assertTrue(all(row["Connect Duration (s)"] == 0.0 and row["Transfer Duration (s)"] == 0.0 for row in disconnect_rows))
        self.assertTrue(all(row["Connect Duration (s)"] == 0.0 and row["Duration (s)"] == row["Transfer Duration (s)"] for row in flow_rows))
        self.assertAlmostEqual(
            result.objective_profit,
            cfg.base_profit + cfg.lambda_per_second * result.makespan_s - result.total_cost,
        )
        self.assertAlmostEqual(replay.profit_check["makespan_s"], result.makespan_s)
        self.assertEqual(result.diagnostics.get("auxiliary_transfer_mode"), "lazy")
        self.assertFalse(result.diagnostics.get("lazy_fallback_used"))
        eager_count = result.diagnostics.get("eager_auxiliary_candidate_count")
        if eager_count is not None:
            self.assertLess(result.diagnostics.get("lazy_candidate_count", 0), eager_count)
        self.assertTrue(
            any(round_info.get("new_candidate_count", 0) >= 0 for round_info in result.diagnostics.get("lazy_rounds", []))
        )

    def test_single_input_port_serializes_parallel_dosing(self) -> None:
        PlannerConfig, solve_recipe_ir_with_cp_sat, validate_schedule = _planner_imports()

        spec = {
            "id": "UnitTest_Port_Serialization",
            "volume": 2.0,
            "procedure": [
                {
                    "parallel": [
                        {"dose": {"ingredient": "A", "amount_L": 1.0}},
                        {"dose": {"ingredient": "B", "amount_L": 1.0}},
                    ],
                    "join": "wait_all",
                },
                {"mix": {"rpm": 200, "duration_s": 10}},
            ],
        }
        module_ops = {
            "HC10": [("Draining", 1.0, 1), ("Filling", 1.0, 0), ("Connect", "", 1), ("Stirring", "200", 3)],
            "HC20": [("Draining", 1.0, 1), ("Filling", 1.0, 0), ("Connect", "", 1), ("Stirring", "200", 3)],
            "HC30": [("Draining", 1.0, 1), ("Filling", 1.0, 0), ("Connect", "", 1), ("Stirring", "200", 1)],
        }
        module_interfaces = {
            "HC10": [("Input", "HC10_In1"), ("Output", "HC10_Out1")],
            "HC20": [("Input", "HC20_In1"), ("Output", "HC20_Out1")],
            "HC30": [("Input", "HC30_In1"), ("Output", "HC30_Out1")],
        }
        ir = build_recipe_ir(spec)
        result = solve_recipe_ir_with_cp_sat(
            ir,
            module_ops,
            module_interfaces=module_interfaces,
            module_maximum_volume={"HC10": [10], "HC20": [10], "HC30": [10]},
            module_resources={"HC10": ["A", 10], "HC20": ["B", 10]},
            config=PlannerConfig(base_profit=100, lambda_per_second=-0.01, connect_duration_s=3.0, solver_time_limit_s=5),
        )
        replay = validate_schedule(
            ir,
            result,
            module_maximum_volume={"HC10": [10], "HC20": [10], "HC30": [10]},
            module_resources={"HC10": ["A", 10], "HC20": ["B", 10]},
            config=PlannerConfig(base_profit=100, lambda_per_second=-0.01),
        )
        self.assertTrue(replay.valid, replay.errors)
        dose_rows = [row for row in result.to_rows() if row["Operation Type"] == "dose"]
        self.assertEqual({row["InPort"] for row in dose_rows}, {"HC30_In1"})
        first, second = sorted(dose_rows, key=lambda row: row["Start (s)"])
        self.assertLessEqual(first["End (s)"], second["Start (s)"])

    def test_target_module_allows_parallel_filling_from_distinct_sources_and_ports(self) -> None:
        PlannerConfig, solve_recipe_ir_with_cp_sat, validate_schedule = _planner_imports()

        spec = {
            "id": "UnitTest_Module_Connect_Serialization",
            "volume": 20.0,
            "procedure": [
                {
                    "parallel": [
                        {"dose": {"ingredient": "A", "amount_L": 10.0}},
                        {"dose": {"ingredient": "B", "amount_L": 10.0}},
                    ],
                    "join": "wait_all",
                },
                {"mix": {"rpm": 200, "duration_s": 10}},
            ],
        }
        module_ops = {
            "HC10": [("Draining", 1.0, 1), ("Filling", 1.0, 0), ("Connect", "", 1)],
            "HC20": [("Draining", 1.0, 1), ("Filling", 1.0, 0), ("Connect", "", 1)],
            "HC30": [("Draining", 1.0, 1), ("Filling", 1.0, 0), ("Connect", "", 1), ("Stirring", "200", 1)],
        }
        module_interfaces = {
            "HC10": [("Output", "HC10_Out1")],
            "HC20": [("Output", "HC20_Out1")],
            "HC30": [("Input", "HC30_In1"), ("Input", "HC30_In2"), ("Output", "HC30_Out1")],
        }
        module_max = {"HC10": [20], "HC20": [20], "HC30": [30]}
        module_resources = {"HC10": ["A", 20], "HC20": ["B", 20]}
        cfg = PlannerConfig(base_profit=100, lambda_per_second=-1.0, connect_duration_s=3.0, solver_time_limit_s=5)
        ir = build_recipe_ir(spec)
        result = solve_recipe_ir_with_cp_sat(
            ir,
            module_ops,
            module_interfaces=module_interfaces,
            module_maximum_volume=module_max,
            module_resources=module_resources,
            config=cfg,
        )
        replay = validate_schedule(ir, result, module_maximum_volume=module_max, module_resources=module_resources, config=cfg)
        self.assertIn(result.status, {"OPTIMAL", "FEASIBLE"})
        self.assertTrue(replay.valid, replay.errors)
        dose_rows = sorted([row for row in result.to_rows() if row["Operation Type"] == "dose"], key=lambda row: row["Start (s)"])
        connect_rows = sorted([row for row in result.to_rows() if row["Operation Type"] == "connect"], key=lambda row: row["Start (s)"])
        self.assertEqual({row["Target Module"] for row in dose_rows}, {"HC30"})
        self.assertEqual({row["InPort"] for row in dose_rows}, {"HC30_In1", "HC30_In2"})
        self.assertLessEqual(connect_rows[0]["End (s)"], connect_rows[1]["Start (s)"])
        self.assertTrue(
            any(
                connect["Start (s)"] < dose["End (s)"]
                and dose["Start (s)"] < connect["End (s)"]
                and connect["OutPort"] != dose["OutPort"]
                and connect["InPort"] != dose["InPort"]
                for connect in connect_rows
                for dose in dose_rows
            ),
            "a future connection on free ports should be prepared while another Dose is flowing",
        )
        self.assertLess(dose_rows[1]["Start (s)"], dose_rows[0]["End (s)"])
        self.assertAlmostEqual(result.makespan_s, 26.0)

        from cp_sat_planner import PlannedOperation, PlannerResult
        overlapping_plan = PlannerResult(
            status="FEASIBLE",
            objective_profit=0.0,
            base_profit=0.0,
            total_duration_s=8.0,
            makespan_s=4.0,
            total_cost=0.0,
            selected_branches={},
            operations=[
                PlannedOperation(
                    step_id=1,
                    recipe_node_id="dose_001",
                    branch_group_id="AND_001",
                    branch_id="b1",
                    operation_type="dose",
                    operation="overlap A",
                    module="HC30",
                    source_module="HC10",
                    target_module="HC30",
                    out_port="HC10_Out1",
                    in_port="HC30_In1",
                    start_s=0.0,
                    end_s=4.0,
                    duration_s=4.0,
                    connect_duration_s=3.0,
                    transfer_duration_s=1.0,
                    material={"A": 1.0},
                ),
                PlannedOperation(
                    step_id=2,
                    recipe_node_id="dose_002",
                    branch_group_id="AND_001",
                    branch_id="b2",
                    operation_type="dose",
                    operation="same source overlaps a second target",
                    module="HC40",
                    source_module="HC10",
                    target_module="HC40",
                    out_port="HC10_Out2",
                    in_port="HC40_In1",
                    start_s=0.0,
                    end_s=4.0,
                    duration_s=4.0,
                    connect_duration_s=3.0,
                    transfer_duration_s=1.0,
                    material={"A": 1.0},
                ),
            ],
        )
        overlap_replay = validate_schedule(
            ir,
            overlapping_plan,
            module_maximum_volume={**module_max, "HC40": [20]},
            module_resources=module_resources,
            config=cfg,
        )
        self.assertFalse(overlap_replay.valid)
        self.assertTrue(any("Global connect overlap" in error for error in overlap_replay.errors))
        self.assertTrue(any("Resource overlap on HC10" in error for error in overlap_replay.resource_errors))

        transfer_with_one_connect = PlannerResult(
            status="FEASIBLE",
            objective_profit=0.0,
            base_profit=0.0,
            total_duration_s=19.0,
            makespan_s=13.0,
            total_cost=0.0,
            selected_branches={},
            operations=[
                PlannedOperation(
                    step_id=1,
                    recipe_node_id="manual_connect_a",
                    branch_group_id="",
                    branch_id="",
                    operation_type="connect",
                    operation="establish A path",
                    module="HC10->HC30",
                    source_module="HC10",
                    target_module="HC30",
                    out_port="HC10_Out1",
                    in_port="HC30_In1",
                    connection_path="HC10.HC10_Out1 -> HC30.HC30_In1",
                    start_s=0.0,
                    end_s=3.0,
                    duration_s=3.0,
                    connect_duration_s=3.0,
                    material={},
                    trace={"material_signature": "MATERIAL:A"},
                ),
                PlannedOperation(
                    step_id=2,
                    recipe_node_id="dose_001",
                    branch_group_id="AND_001",
                    branch_id="b1",
                    operation_type="dose",
                    operation="flow A",
                    module="HC30",
                    source_module="HC10",
                    target_module="HC30",
                    out_port="HC10_Out1",
                    in_port="HC30_In1",
                    connection_path="HC10.HC10_Out1 -> HC30.HC30_In1",
                    start_s=3.0,
                    end_s=10.0,
                    duration_s=7.0,
                    connect_duration_s=0.0,
                    transfer_duration_s=7.0,
                    material={"A": 1.0},
                ),
                PlannedOperation(
                    step_id=3,
                    recipe_node_id="manual_connect_1",
                    branch_group_id="",
                    branch_id="",
                    operation_type="connect",
                    operation="prepare B path while A is flowing",
                    module="HC20->HC30",
                    source_module="HC20",
                    target_module="HC30",
                    out_port="HC20_Out1",
                    in_port="HC30_In2",
                    connection_path="HC20.HC20_Out1 -> HC30.HC30_In2",
                    start_s=4.0,
                    end_s=7.0,
                    duration_s=3.0,
                    connect_duration_s=3.0,
                    material={},
                ),
                PlannedOperation(
                    step_id=4,
                    recipe_node_id="manual_disconnect_a",
                    branch_group_id="",
                    branch_id="",
                    operation_type="disconnect",
                    operation="release A path",
                    module="HC10->HC30",
                    source_module="HC10",
                    target_module="HC30",
                    out_port="HC10_Out1",
                    in_port="HC30_In1",
                    connection_path="HC10.HC10_Out1 -> HC30.HC30_In1",
                    start_s=10.0,
                    end_s=13.0,
                    duration_s=3.0,
                    material={},
                    trace={"material_signature": "MATERIAL:A"},
                ),
                PlannedOperation(
                    step_id=5,
                    recipe_node_id="manual_disconnect_b",
                    branch_group_id="",
                    branch_id="",
                    operation_type="disconnect",
                    operation="release B path",
                    module="HC20->HC30",
                    source_module="HC20",
                    target_module="HC30",
                    out_port="HC20_Out1",
                    in_port="HC30_In2",
                    connection_path="HC20.HC20_Out1 -> HC30.HC30_In2",
                    start_s=7.0,
                    end_s=10.0,
                    duration_s=3.0,
                    material={},
                ),
            ],
        )
        transfer_one_connect_replay = validate_schedule(
            ir,
            transfer_with_one_connect,
            module_maximum_volume={**module_max, "HC40": [20]},
            module_resources={"HC10": ["A", 20], "HC20": ["B", 20]},
            config=cfg,
        )
        self.assertTrue(transfer_one_connect_replay.valid, transfer_one_connect_replay.errors)

        disconnect_during_its_own_transfer = PlannerResult(
            **{
                **transfer_with_one_connect.__dict__,
                "operations": transfer_with_one_connect.operations[:2]
                + [
                    PlannedOperation(
                        step_id=3,
                        recipe_node_id="invalid_disconnect_a",
                        branch_group_id="",
                        branch_id="",
                        operation_type="disconnect",
                        operation="must not disconnect an actively flowing pair",
                        module="HC10->HC30",
                        source_module="HC10",
                        target_module="HC30",
                        out_port="HC10_Out1",
                        in_port="HC30_In1",
                        connection_path="HC10.HC10_Out1 -> HC30.HC30_In1",
                        start_s=5.0,
                        end_s=7.0,
                        duration_s=2.0,
                        material={},
                        trace={"material_signature": "MATERIAL:A"},
                    )
                ],
            }
        )
        own_transfer_disconnect_replay = validate_schedule(
            ir,
            disconnect_during_its_own_transfer,
            module_maximum_volume={**module_max, "HC40": [20]},
            module_resources={"HC10": ["A", 20], "HC20": ["B", 20]},
            config=PlannerConfig(**{**cfg.__dict__, "require_final_disconnect": False}),
        )
        self.assertFalse(own_transfer_disconnect_replay.valid)
        self.assertTrue(
            any("Port overlap" in error for error in own_transfer_disconnect_replay.resource_errors)
        )

        transfer_with_two_connects = PlannerResult(
            **{
                **transfer_with_one_connect.__dict__,
                "operations": transfer_with_one_connect.operations
                + [
                    PlannedOperation(
                        step_id=3,
                        recipe_node_id="manual_connect_2",
                        branch_group_id="",
                        branch_id="",
                        operation_type="connect",
                        operation="second overlapping connect",
                        module="HC40->HC30",
                        source_module="HC40",
                        target_module="HC30",
                        out_port="HC40_Out1",
                        in_port="HC30_In3",
                        start_s=11.0,
                        end_s=14.0,
                        duration_s=3.0,
                        connect_duration_s=3.0,
                        material={},
                    )
                ],
            }
        )
        transfer_two_connects_replay = validate_schedule(
            ir,
            transfer_with_two_connects,
            module_maximum_volume={**module_max, "HC40": [20]},
            module_resources={"HC10": ["A", 20], "HC20": ["B", 20]},
            config=cfg,
        )
        self.assertFalse(transfer_two_connects_replay.valid)
        self.assertTrue(any("Global connect overlap" in error for error in transfer_two_connects_replay.errors))

    def test_separation_connection_can_be_prepared_during_other_processing(self) -> None:
        PlannerConfig, solve_recipe_ir_with_cp_sat, validate_schedule = _planner_imports()

        spec = {
            "id": "UnitTest_Prepare_Separation_Connect",
            "volume": 5.0,
            "procedure": [
                {"dose": {"ingredient": "A", "amount_L": 5.0}},
                {"mix": {"rpm": 200, "duration_s": 10}},
                {"settling": {"duration_s": 50}},
                {"separation": {"order": ["A"]}},
            ],
        }
        module_ops = {
            "HC10": [("Draining", 1.0, 1), ("Connect", "", 1)],
            "HC30": [("Filling", 1.0, 0), ("Draining", 1.0, 1), ("Stirring", "200", 10), ("Settling", "", 50), ("Connect", "", 1)],
            "HC40": [("Filling", 1.0, 0), ("Connect", "", 1)],
        }
        module_interfaces = {
            "HC10": [("Output", "HC10_Out1")],
            "HC30": [("Input", "HC30_In1"), ("Output", "HC30_Out1")],
            "HC40": [("Input", "HC40_In1")],
        }
        module_max = {"HC10": [10], "HC30": [10], "HC40": [10]}
        module_resources = {"HC10": ["A", 10]}
        cfg = PlannerConfig(base_profit=100, lambda_per_second=-1.0, connect_duration_s=5.0, solver_time_limit_s=5)
        ir = build_recipe_ir(spec)
        result = solve_recipe_ir_with_cp_sat(
            ir,
            module_ops,
            module_interfaces=module_interfaces,
            module_maximum_volume=module_max,
            module_resources=module_resources,
            config=cfg,
        )
        replay = validate_schedule(ir, result, module_maximum_volume=module_max, module_resources=module_resources, config=cfg)

        self.assertIn(result.status, {"OPTIMAL", "FEASIBLE"})
        self.assertTrue(replay.valid, replay.errors)
        rows = result.to_rows()
        settling = next(row for row in rows if row["Operation Type"] == "settling")
        separation = next(row for row in rows if row["Operation Type"] == "separation")
        separation_connect = next(
            row
            for row in rows
            if row["Operation Type"] == "connect" and row["Recipe Node"].endswith("_CONNECT") and "separation" in row["Recipe Node"]
        )
        self.assertGreaterEqual(separation["Start (s)"], settling["End (s)"])
        self.assertLessEqual(separation_connect["End (s)"], separation["Start (s)"])
        self.assertTrue(
            any(
                row["Operation Type"] in {"dose", "mix", "settling"}
                and separation_connect["Start (s)"] < row["End (s)"]
                and row["Start (s)"] < separation_connect["End (s)"]
                for row in rows
            ),
            "the free separation ports should be connectable while another operation is running",
        )

    def test_separations_use_free_source_ports_instead_of_forcing_disconnect(self) -> None:
        PlannerConfig, solve_recipe_ir_with_cp_sat, validate_schedule = _planner_imports()

        ir = build_recipe_ir({
            "id": "ThreePersistentSeparationOutputs",
            "volume": 3.0,
            "procedure": [
                {
                    "parallel": [
                        {"dose": {"ingredient": "A", "amount_L": 1.0}},
                        {"dose": {"ingredient": "B", "amount_L": 1.0}},
                        {"dose": {"ingredient": "C", "amount_L": 1.0}},
                    ],
                    "join": "wait_all",
                },
                {"mix": {"rpm": 100, "duration_s": 1}},
                {"separation": {"order": ["A", "B", "C"]}},
            ],
        })

        def capabilities(extra=()):
            return [
                ("Draining", 1.0, 1),
                ("Filling", 1.0, 0),
                ("Connect", "", 2, 0, 0, 3),
                ("Disconnect", "", 2, 0, 0, 2),
                *extra,
            ]

        module_ops = {
            "A_SRC": capabilities(),
            "B_SRC": capabilities(),
            "C_SRC": capabilities(),
            "PROC": capabilities((("Stirring", "100", 1),)),
        }
        module_interfaces = {
            "A_SRC": [("Input", "A_In1"), ("Output", "A_Out1")],
            "B_SRC": [("Input", "B_In1"), ("Output", "B_Out1")],
            "C_SRC": [("Input", "C_In1"), ("Output", "C_Out1")],
            "PROC": [
                ("Input", "P_In1"),
                ("Input", "P_In2"),
                ("Input", "P_In3"),
                ("Output", "P_Out1"),
                ("Output", "P_Out2"),
                ("Output", "P_Out3"),
            ],
        }
        module_max = {module: [10] for module in module_ops}
        module_resources = {
            "A_SRC": ["A", 5],
            "B_SRC": ["B", 5],
            "C_SRC": ["C", 5],
        }
        cfg = PlannerConfig(
            base_profit=1000,
            lambda_per_second=-0.1,
            solver_time_limit_s=10,
            connect_duration_s=3,
            disconnect_duration_s=2,
            enable_auxiliary_transfers=False,
        )
        result = solve_recipe_ir_with_cp_sat(
            ir,
            module_ops,
            module_interfaces=module_interfaces,
            module_maximum_volume=module_max,
            module_resources=module_resources,
            config=cfg,
        )
        replay = validate_schedule(
            ir,
            result,
            module_maximum_volume=module_max,
            module_resources=module_resources,
            config=cfg,
        )

        self.assertIn(result.status, {"OPTIMAL", "FEASIBLE"})
        self.assertTrue(replay.valid, replay.errors)
        self.assertEqual(
            [operation for operation in result.operations if operation.operation_type == "disconnect"],
            [],
            "a free process output must be preferred over tearing down a still-valid connection",
        )
        separation_routes = [
            operation
            for operation in result.operations
            if operation.operation_type == "separation"
        ]
        self.assertEqual(len(separation_routes), 3)
        self.assertEqual(
            len({operation.out_port for operation in separation_routes}),
            3,
            "all three persistent outgoing connections need distinct physical source ports",
        )
        self.assertEqual(
            result.diagnostics.get("connection_port_search"),
            "all_ports_for_seeded_route",
        )

    def test_usage_cannot_run_on_module_without_mixed_material(self) -> None:
        PlannerConfig, solve_recipe_ir_with_cp_sat, validate_schedule = _planner_imports()

        spec = {
            "id": "UnitTest_Material_Location_For_Usage",
            "volume": 1.0,
            "procedure": [
                {"dose": {"ingredient": "A", "amount_L": 1.0}},
                {"mix": {"rpm": 200, "duration_s": 5}},
                {"usage": {"duration_s": 5}},
            ],
        }
        module_ops = {
            "HC10": [("Draining", 1.0, 1), ("Connect", "", 1)],
            "HC30": [("Filling", 1.0, 0), ("Stirring", "200", 1), ("Connect", "", 1)],
            "HC40": [("None", "", 1)],
        }
        module_interfaces = {
            "HC10": [("Output", "HC10_Out1")],
            "HC30": [("Input", "HC30_In1")],
            "HC40": [("Input", "HC40_In1")],
        }
        ir = build_recipe_ir(spec)
        cfg = PlannerConfig(base_profit=100, lambda_per_second=-1.0, connect_duration_s=1.0, solver_time_limit_s=5)
        result = solve_recipe_ir_with_cp_sat(
            ir,
            module_ops,
            module_interfaces=module_interfaces,
            module_maximum_volume={"HC10": [10], "HC30": [10], "HC40": [10]},
            module_resources={"HC10": ["A", 10]},
            config=cfg,
        )
        self.assertNotIn(result.status, {"OPTIMAL", "FEASIBLE"})

        from cp_sat_planner import PlannedOperation, PlannerResult
        invalid_plan = PlannerResult(
            status="FEASIBLE",
            objective_profit=0.0,
            base_profit=0.0,
            total_duration_s=10.0,
            makespan_s=10.0,
            total_cost=0.0,
            selected_branches={},
            operations=[
                PlannedOperation(
                    step_id=1,
                    recipe_node_id="mix_001",
                    branch_group_id="",
                    branch_id="",
                    operation_type="mix",
                    operation="Mixing",
                    module="HC30",
                    start_s=0.0,
                    end_s=5.0,
                    duration_s=5.0,
                ),
                PlannedOperation(
                    step_id=2,
                    recipe_node_id="usage_001",
                    branch_group_id="",
                    branch_id="",
                    operation_type="usage",
                    operation="Usage",
                    module="HC40",
                    start_s=5.0,
                    end_s=10.0,
                    duration_s=5.0,
                ),
            ],
        )
        replay = validate_schedule(
            ir,
            invalid_plan,
            module_maximum_volume={"HC30": [10], "HC40": [10]},
            module_resources={},
            config=cfg,
        )
        self.assertFalse(replay.valid)
        self.assertTrue(any("Material location violation" in error for error in replay.errors))

    def test_usage_can_run_after_explicit_product_transfer_to_compatible_module(self) -> None:
        PlannerConfig, solve_recipe_ir_with_cp_sat, validate_schedule = _planner_imports()

        spec = {
            "id": "UnitTest_Product_Transfer_To_Usage",
            "volume": 1.0,
            "procedure": [
                {"dose": {"ingredient": "A", "amount_L": 1.0}},
                {"mix": {"rpm": 200, "duration_s": 5}},
                {"usage": {"duration_s": 5}},
            ],
        }
        module_ops = {
            "HC10": [("Draining", 1.0, 1), ("Connect", "", 1)],
            "HC30": [("Filling", 1.0, 0), ("Draining", 1.0, 1), ("Stirring", "200", 1), ("Connect", "", 1)],
            "HC40": [("Filling", 1.0, 0), ("None", "", 1), ("Connect", "", 1)],
        }
        module_interfaces = {
            "HC10": [("Output", "HC10_Out1")],
            "HC30": [("Input", "HC30_In1"), ("Output", "HC30_Out1")],
            "HC40": [("Input", "HC40_In1"), ("Output", "HC40_Out1")],
        }
        module_max = {"HC10": [10], "HC30": [10], "HC40": [10]}
        module_resources = {"HC10": ["A", 10]}
        cfg = PlannerConfig(base_profit=100, lambda_per_second=-1.0, connect_duration_s=1.0, solver_time_limit_s=5)
        ir = build_recipe_ir(spec)
        result = solve_recipe_ir_with_cp_sat(
            ir,
            module_ops,
            module_interfaces=module_interfaces,
            module_maximum_volume=module_max,
            module_resources=module_resources,
            config=cfg,
        )
        replay = validate_schedule(ir, result, module_maximum_volume=module_max, module_resources=module_resources, config=cfg)

        self.assertIn(result.status, {"OPTIMAL", "FEASIBLE"})
        self.assertTrue(replay.valid, replay.errors)
        rows = result.to_rows()
        self.assertEqual(next(row for row in rows if row["Operation Type"] == "mix")["Module"], "HC30")
        self.assertEqual(next(row for row in rows if row["Operation Type"] == "usage")["Module"], "HC40")
        self.assertTrue(any(row["Operation Type"] == "aux_transfer" and row["Source Module"] == "HC30" and row["Target Module"] == "HC40" for row in rows))

    def test_missing_required_input_port_is_infeasible(self) -> None:
        PlannerConfig, solve_recipe_ir_with_cp_sat, _ = _planner_imports()

        spec = {
            "id": "UnitTest_Missing_Input_Port",
            "volume": 1.0,
            "procedure": [
                {"dose": {"ingredient": "A", "amount_L": 1.0}},
                {"mix": {"rpm": 200, "duration_s": 10}},
            ],
        }
        module_ops = {
            "HC10": [("Draining", 1.0, 1), ("Connect", "", 1)],
            "HC30": [("Filling", 1.0, 0), ("Connect", "", 1), ("Stirring", "200", 1)],
        }
        module_interfaces = {
            "HC10": [("Output", "HC10_Out1")],
            "HC30": [("Output", "HC30_Out1")],
        }
        result = solve_recipe_ir_with_cp_sat(
            build_recipe_ir(spec),
            module_ops,
            module_interfaces=module_interfaces,
            module_maximum_volume={"HC10": [10], "HC30": [10]},
            module_resources={"HC10": ["A", 10]},
            config=PlannerConfig(enable_auxiliary_transfers=False, solver_time_limit_s=5),
        )
        self.assertEqual(result.status, "INFEASIBLE")
        self.assertIn("No physical transfer route", result.diagnostics.get("infeasible_reason", ""))

    def test_transfer_requires_connect_capability(self) -> None:
        PlannerConfig, solve_recipe_ir_with_cp_sat, _ = _planner_imports()

        spec = {
            "id": "UnitTest_Missing_Connect",
            "volume": 1.0,
            "procedure": [
                {"dose": {"ingredient": "A", "amount_L": 1.0}},
                {"mix": {"rpm": 200, "duration_s": 10}},
            ],
        }
        module_ops = {
            "HC10": [("Draining", 1.0, 1)],
            "HC30": [("Filling", 1.0, 0), ("Stirring", "200", 1)],
        }
        result = solve_recipe_ir_with_cp_sat(
                build_recipe_ir(spec),
                module_ops,
                module_interfaces={
                    "HC10": [("Output", "HC10_Out1")],
                    "HC30": [("Input", "HC30_In1")],
                },
                module_maximum_volume={"HC10": [10], "HC30": [10]},
                module_resources={"HC10": ["A", 10]},
                config=PlannerConfig(enable_auxiliary_transfers=False, solver_time_limit_s=5),
            )
        self.assertEqual(result.status, "INFEASIBLE")
        self.assertIn("No physical transfer route", result.diagnostics.get("infeasible_reason", ""))

    def test_parallel_material_plan_without_enough_compatible_capacity_is_infeasible(self) -> None:
        PlannerConfig, solve_recipe_ir_with_cp_sat, _ = _planner_imports()

        spec = {
            "id": "UnitTest_Port_Pair_Material_Binding",
            "volume": 2.0,
            "procedure": [
                {
                    "parallel": [
                        {"dose": {"ingredient": "A", "amount_L": 1.0}},
                        {"dose": {"ingredient": "B", "amount_L": 1.0}},
                    ],
                    "join": "wait_all",
                },
                {"mix": {"rpm": 200, "duration_s": 10}},
                {"separation": {"order": ["A", "B"]}},
            ],
        }
        module_ops = {
            "HC10": [("Draining", 1.0, 1), ("Filling", 1.0, 0), ("Connect", "", 1)],
            "HC20": [("Draining", 1.0, 1), ("Connect", "", 1)],
            "HC30": [("Draining", 1.0, 1), ("Filling", 1.0, 0), ("Connect", "", 1), ("Stirring", "200", 1)],
        }
        module_interfaces = {
            "HC10": [("Input", "HC10_In1"), ("Output", "HC10_Out1")],
            "HC20": [("Output", "HC20_Out1")],
            "HC30": [("Input", "HC30_In1"), ("Input", "HC30_In2"), ("Output", "HC30_Out1")],
        }
        result = solve_recipe_ir_with_cp_sat(
            build_recipe_ir(spec),
            module_ops,
            module_interfaces=module_interfaces,
            module_maximum_volume={"HC10": [10], "HC20": [10], "HC30": [10]},
            module_resources={"HC10": ["A", 10], "HC20": ["B", 10]},
            config=PlannerConfig(solver_time_limit_s=5, enable_auxiliary_transfers=False),
        )
        self.assertEqual(result.status, "INFEASIBLE")

    def test_total_source_material_cannot_be_overdrawn(self) -> None:
        PlannerConfig, solve_recipe_ir_with_cp_sat, _ = _planner_imports()

        spec = {
            "id": "UnitTest_Source_Inventory",
            "volume": 12.0,
            "procedure": [
                {
                    "parallel": [
                        {"dose": {"ingredient": "A", "amount_L": 6.0}},
                        {"dose": {"ingredient": "A", "amount_L": 6.0}},
                    ],
                    "join": "wait_all",
                },
                {"mix": {"rpm": 200, "duration_s": 10}},
            ],
        }
        module_ops = {
            "HC10": [("Draining", 1.0, 1), ("Filling", 1.0, 0), ("Connect", "", 1)],
            "HC30": [("Draining", 1.0, 1), ("Filling", 1.0, 0), ("Connect", "", 1), ("Stirring", "200", 1)],
        }
        module_interfaces = {
            "HC10": [("Input", "HC10_In1"), ("Output", "HC10_Out1"), ("Output", "HC10_Out2")],
            "HC30": [("Input", "HC30_In1"), ("Input", "HC30_In2"), ("Output", "HC30_Out1")],
        }
        result = solve_recipe_ir_with_cp_sat(
            build_recipe_ir(spec),
            module_ops,
            module_interfaces=module_interfaces,
            module_maximum_volume={"HC10": [10], "HC30": [20]},
            module_resources={"HC10": ["A", 10]},
            config=PlannerConfig(solver_time_limit_s=5),
        )
        self.assertEqual(result.status, "INFEASIBLE")

    def test_separation_target_cannot_contain_different_initial_material(self) -> None:
        PlannerConfig, solve_recipe_ir_with_cp_sat, _ = _planner_imports()

        spec = {
            "id": "UnitTest_Separation_Target_Material",
            "volume": 1.0,
            "procedure": [
                {"dose": {"ingredient": "A", "amount_L": 1.0}},
                {"mix": {"rpm": 200, "duration_s": 10}},
                {"separation": {"order": ["A"]}},
            ],
        }
        module_ops = {
            "HC10": [("Draining", 1.0, 1), ("Filling", 1.0, 0), ("Connect", "", 1)],
            "HC20": [("Draining", 1.0, 1), ("Filling", 1.0, 0), ("Connect", "", 1)],
            "HC30": [("Draining", 1.0, 1), ("Filling", 1.0, 0), ("Connect", "", 1), ("Stirring", "200", 1)],
        }
        module_interfaces = {
            "HC10": [("Output", "HC10_Out1")],
            "HC20": [("Input", "HC20_In1"), ("Output", "HC20_Out1")],
            "HC30": [("Input", "HC30_In1"), ("Output", "HC30_Out1")],
        }
        result = solve_recipe_ir_with_cp_sat(
            build_recipe_ir(spec),
            module_ops,
            module_interfaces=module_interfaces,
            module_maximum_volume={"HC10": [10], "HC20": [10], "HC30": [10]},
            module_resources={"HC10": ["A", 10], "HC20": ["B", 10]},
            config=PlannerConfig(solver_time_limit_s=5, enable_auxiliary_transfers=False),
        )
        self.assertEqual(result.status, "INFEASIBLE")

    def test_legacy_bfs_config_uses_auxiliary_staging_transfer(self) -> None:
        PlannerConfig, solve_recipe_ir_with_cp_sat, validate_schedule = _planner_imports()

        module_ops = {
            "HC10": [("Draining", 0.1, 3), ("Filling", 0.1, 0), ("Settling", "", 1), ("Stirring", "100", 3), ("Stirring", "200", 3), ("Connect", "", 1), ("Disconnect", "", 0), ("None", "", 0)],
            "HC20": [("Draining", 0.1, 3), ("Filling", 0.1, 0), ("Settling", "", 1), ("Stirring", "150", 3), ("Stirring", "300", 3), ("Connect", "", 1), ("Disconnect", "", 0), ("None", "", 0)],
            "HC30": [("Draining", 0.1, 3), ("Filling", 0.1, 0), ("Settling", "", 1), ("Stirring", "100", 3), ("Stirring", "150", 3), ("Connect", "", 1), ("Disconnect", "", 0), ("None", "", 0)],
            "HC40": [("Draining", 0.1, 3), ("Filling", 0.1, 0), ("Settling", "", 1), ("Connect", "", 1), ("Disconnect", "", 0), ("None", "", 0)],
        }
        module_interfaces = {
            "HC10": [("Input", "HC10_In1"), ("Input", "HC10_In2"), ("Input", "HC10_In3"), ("Output", "HC10_Out1"), ("Output", "HC10_Out2"), ("Output", "HC10_Out3")],
            "HC20": [("Input", "HC20_In1"), ("Input", "HC20_In2"), ("Input", "HC20_In3"), ("Output", "HC20_Out1"), ("Output", "HC20_Out2"), ("Output", "HC20_Out3")],
            "HC30": [("Input", "HC30_In1"), ("Input", "HC30_In2"), ("Input", "HC30_In3"), ("Input", "HC30_In4"), ("Output", "HC30_Out1")],
            "HC40": [("Input", "HC40_In1"), ("Output", "HC40_Out1"), ("Output", "HC40_Out2")],
        }
        module_max = {"HC10": [10], "HC20": [15], "HC30": [10], "HC40": [30]}
        module_resources = {"HC10": ["A", 10], "HC20": ["B", 10], "HC30": ["C", 10]}
        order = {
            "volume": 6.0,
            "order": ["A", "B", "C", {"mix": {"rpm": 150, "duration": 30}}],
            "ratio": {"A": [1], "B": [2], "C": [3]},
            "usage_and_settling": [3600, 300],
            "separation_order": ["C", "B", "A"],
        }
        ir = build_recipe_ir(recipe_spec_from_legacy_order(order, parallel_dosing_before_mix=True))
        cfg = PlannerConfig(base_profit=300, lambda_per_second=-0.5, solver_time_limit_s=30)
        result = solve_recipe_ir_with_cp_sat(
            ir,
            module_ops,
            module_interfaces=module_interfaces,
            module_maximum_volume=module_max,
            module_resources=module_resources,
            config=cfg,
        )
        replay = validate_schedule(ir, result, module_maximum_volume=module_max, module_resources=module_resources, config=cfg)
        self.assertIn(result.status, {"OPTIMAL", "FEASIBLE"})
        self.assertTrue(any(op.operation_type == "aux_transfer" for op in result.operations))
        self.assertTrue(replay.valid, replay.errors)
        self.assertEqual(result.diagnostics.get("auxiliary_transfer_mode"), "lazy")
        eager_count = result.diagnostics.get("eager_auxiliary_candidate_count")
        if eager_count is not None:
            self.assertLess(result.diagnostics.get("lazy_candidate_count", 0), eager_count)


    def test_auxiliary_inventory_rebalancing_with_virtual_self_route_candidates(self) -> None:
        PlannerConfig, solve_recipe_ir_with_cp_sat, validate_schedule = _planner_imports()

        module_ops = {
            "HC10": [("Draining", 0.1, 3), ("Filling", 0.1, 0), ("Settling", "", 1), ("Stirring", "150", 3), ("Connect", "", 1), ("Disconnect", "", 0), ("None", "", 0)],
            "HC20": [("Draining", 0.1, 3), ("Filling", 0.1, 0), ("Settling", "", 1), ("Stirring", "150", 3), ("Connect", "", 1), ("Disconnect", "", 0), ("None", "", 0)],
            "HC30": [("Draining", 0.1, 3), ("Filling", 0.1, 0), ("Settling", "", 1), ("Stirring", "150", 3), ("Connect", "", 1), ("Disconnect", "", 0), ("None", "", 0)],
            "HC40": [("Draining", 0.1, 3), ("Filling", 0.1, 0), ("Settling", "", 1), ("Connect", "", 1), ("Disconnect", "", 0), ("None", "", 0)],
        }
        module_interfaces = {
            "HC10": [("Input", "HC10_In1"), ("Input", "HC10_In3"), ("Output", "HC10_Out1")],
            "HC20": [("Input", "HC20_In1"), ("Input", "HC20_In2"), ("Input", "HC20_In3"), ("Output", "HC20_Out1"), ("Output", "HC20_Out3")],
            "HC30": [("Input", "HC30_In2"), ("Input", "HC30_In4"), ("Output", "HC30_Out1")],
            "HC40": [("Input", "HC40_In1"), ("Output", "HC40_Out1"), ("Output", "HC40_Out2")],
        }
        module_max = {"HC10": [10], "HC20": [15], "HC30": [10], "HC40": [30]}
        module_resources = {"HC10": ["A", 10], "HC20": ["B", 10], "HC30": ["C", 10]}
        order = {
            "volume": 6.0,
            "order": ["A", "B", "C", {"mix": {"rpm": 150, "duration": 30}}],
            "ratio": {"A": [1], "B": [2], "C": [3]},
            "usage_and_settling": [3600, 300],
            "separation_order": ["C", "B", "A"],
        }
        ir = build_recipe_ir(recipe_spec_from_legacy_order(order, parallel_dosing_before_mix=True))
        cfg = PlannerConfig(base_profit=300, lambda_per_second=-0.5, solver_time_limit_s=30)
        result = solve_recipe_ir_with_cp_sat(
            ir,
            module_ops,
            module_interfaces=module_interfaces,
            module_maximum_volume=module_max,
            module_resources=module_resources,
            config=cfg,
        )
        self.assertIn(result.status, {"OPTIMAL", "FEASIBLE"})

        for aux_op in [op for op in result.operations if op.operation_type == "aux_transfer"]:
            aux_material = next(iter(aux_op.material))
            aux_amount = aux_op.material[aux_material]
            initial_at_src = next(
                (qty for wb, raw in module_resources.items()
                 if wb == aux_op.source_module and len(raw) >= 2 and str(raw[0]).upper() == aux_material.upper()
                for qty in [float(raw[1])]),
                0.0,
            )
            self.assertGreater(aux_amount, 0)
            self.assertLessEqual(
                aux_amount,
                initial_at_src + 1e-6,
                msg=f"aux {aux_material} must not overdraw its initial source inventory",
            )

        for op in result.operations:
            if op.operation_type != "dose":
                continue
            self.assertNotEqual(
                op.source_module, op.target_module,
                msg="virtual self-route doses must not surface as physical dose ops",
            )

        replay = validate_schedule(
            ir, result,
            module_maximum_volume=module_max,
            module_resources=module_resources,
            config=cfg,
        )
        self.assertTrue(replay.valid, replay.errors)

    def test_local_recipe_material_is_retained_for_lazy_and_eager_planning(self) -> None:
        PlannerConfig, solve_recipe_ir_with_cp_sat, validate_schedule = _planner_imports()

        spec = {
            "id": "UnitTest_Retain_Local_Recipe_Material",
            "volume": 6.0,
            "procedure": [
                {
                    "parallel": [
                        {"dose": {"ingredient": "A", "amount_L": 1.0}},
                        {"dose": {"ingredient": "B", "amount_L": 2.0}},
                        {"dose": {"ingredient": "C", "amount_L": 3.0}},
                    ],
                    "join": "wait_all",
                },
                {"mix": {"rpm": 150, "duration_s": 10}},
            ],
        }
        module_ops = {
            wb: [
                ("Draining", 1.0, 3),
                ("Filling", 1.0, 0),
                ("Connect", "", 2, 0, 0, 3),
                ("Disconnect", "", 2, 0, 0, 2),
            ]
            for wb in ("HC10", "HC20", "HC30", "HC40")
        }
        for wb in ("HC10", "HC20", "HC30"):
            module_ops[wb].append(("Stirring", 150, 3))
        module_interfaces = {
            wb: [
                ("Input", f"{wb}_In1"),
                ("Output", f"{wb}_Out1"),
            ]
            for wb in ("HC10", "HC20", "HC30", "HC40")
        }
        module_max = {"HC10": [10], "HC20": [10], "HC30": [10], "HC40": [30]}
        module_resources = {"HC10": ["A", 10], "HC20": ["B", 10], "HC30": ["C", 10]}
        required = {"A": 1.0, "B": 2.0, "C": 3.0}
        initial_material = {"HC10": "A", "HC20": "B", "HC30": "C"}
        ir = build_recipe_ir(spec)

        for mode in ("lazy", "eager"):
            with self.subTest(auxiliary_transfer_mode=mode):
                cfg = PlannerConfig(
                    base_profit=1000,
                    lambda_per_second=-1,
                    auxiliary_transfer_mode=mode,
                    solver_time_limit_s=30,
                )
                result = solve_recipe_ir_with_cp_sat(
                    ir,
                    module_ops,
                    module_interfaces=module_interfaces,
                    module_maximum_volume=module_max,
                    module_resources=module_resources,
                    config=cfg,
                )
                replay = validate_schedule(
                    ir,
                    result,
                    module_maximum_volume=module_max,
                    module_resources=module_resources,
                    config=cfg,
                )
                self.assertIn(result.status, {"OPTIMAL", "FEASIBLE"})
                self.assertTrue(replay.valid, replay.errors)

                mix_module = next(op.module for op in result.operations if op.operation_type == "mix")
                native_material = initial_material[mix_module]
                outgoing_native = sum(
                    op.material.get(native_material, 0.0)
                    for op in result.operations
                    if op.operation_type == "aux_transfer"
                    and op.source_module == mix_module
                    and op.material.get(native_material, 0.0) > 0
                )
                self.assertAlmostEqual(outgoing_native, 10.0 - required[native_material])
                self.assertFalse(
                    any(
                        op.operation_type == "dose" and native_material in op.material
                        for op in result.operations
                    ),
                    "material already retained in the process Module must not be physically dosed back",
                )
                selected_routes = result.diagnostics["selected_transfer_routes"]
                self.assertTrue(
                    any(
                        route["is_self"]
                        and route["source_module"] == mix_module
                        and route["material"].upper() == native_material
                        for routes in selected_routes.values()
                        for route in routes
                    )
                )


if __name__ == "__main__":
    unittest.main()
