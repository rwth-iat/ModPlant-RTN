"""The Master Recipe has to be the plan, and has to be readable as one.

The reference is a solved plan for the parallel-choice demo together with the
precedence a process engineer reads off its Gantt chart by hand. Both are
spelled out here, because the whole point of the derivation is that a person
can check it.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from connection_steps import inject_connection_steps  # noqa: E402
from plan_precedence import derive_precedence, successors_of  # noqa: E402
from modplant_recipe import RecipeEdge, RecipeGraph, RecipeNode  # noqa: E402

# recipe node, type, source, out port, target, in port, start, end
SOLVED_PLAN = [
    ("AUX_006_CONNECT", "connect", "HC30", "HC30_Out1", "HC40", "HC40_In1", 0.0, 3.0),
    ("dose_001_CONNECT", "connect", "HC10", "HC10_Out1", "HC30", "HC30_In1", 3.0, 6.0),
    ("AUX_006", "aux_transfer", "HC30", "HC30_Out1", "HC40", "HC40_In1", 3.0, 73.0),
    ("dose_002_CONNECT", "connect", "HC20", "HC20_Out1", "HC30", "HC30_In2", 6.0, 9.0),
    ("separation_002_CONNECT", "connect", "HC30", "HC30_Out3", "HC20", "HC20_In1", 9.0, 12.0),
    ("separation_003_CONNECT", "connect", "HC30", "HC30_Out2", "HC10", "HC10_In1", 12.0, 15.0),
    ("dose_001", "dose", "HC10", "HC10_Out1", "HC30", "HC30_In1", 73.0, 83.0),
    ("dose_002", "dose", "HC20", "HC20_Out1", "HC30", "HC30_In2", 73.0, 93.0),
    ("mix_001", "mix", "HC30", "", "HC30", "", 93.0, 123.0),
    ("usage_001", "usage", "HC30", "", "HC30", "", 123.0, 1923.0),
    ("settling_001", "settling", "HC30", "", "HC30", "", 1923.0, 2223.0),
    ("separation_001", "separation", "HC30", "HC30_Out1", "HC40", "HC40_In1", 2223.0, 2253.0),
    ("separation_002", "separation", "HC30", "HC30_Out3", "HC20", "HC20_In1", 2253.0, 2273.0),
    ("separation_003", "separation", "HC30", "HC30_Out2", "HC10", "HC10_In1", 2273.0, 2283.0),
]

# What the recipe declares by itself: the mix follows both doses. dose_001
# finishes ten seconds early and waits, so the schedule alone cannot show it.
RECIPE_ORDER = {"mix_001": {"dose_001", "dose_002"}}

# Read off the Gantt chart by hand.
EXPECTED_EDGES = {
    ("AUX_006_CONNECT", "AUX_006"),
    ("AUX_006_CONNECT", "dose_001_CONNECT"),
    ("AUX_006", "dose_001"),
    ("AUX_006", "dose_002"),
    ("dose_001_CONNECT", "dose_001"),
    ("dose_001_CONNECT", "dose_002_CONNECT"),
    ("dose_002_CONNECT", "dose_002"),
    ("dose_002_CONNECT", "separation_002_CONNECT"),
    ("separation_002_CONNECT", "separation_003_CONNECT"),
    ("separation_002_CONNECT", "separation_002"),
    ("separation_003_CONNECT", "separation_003"),
    ("dose_001", "mix_001"),
    ("dose_002", "mix_001"),
    ("mix_001", "usage_001"),
    ("usage_001", "settling_001"),
    ("settling_001", "separation_001"),
    ("separation_001", "separation_002"),
    ("separation_002", "separation_003"),
}


def _plan():
    operations = [
        SimpleNamespace(
            recipe_node_id=node, operation_type=kind,
            source_module=source, out_port=out_port,
            target_module=target, in_port=in_port,
            start_s=start, end_s=end, duration_s=end - start,
            module=target, operation=kind, step_id=node,
        )
        for node, kind, source, out_port, target, in_port, start, end in SOLVED_PLAN
    ]
    return SimpleNamespace(operations=operations)


def _edges(precedence: dict[str, set[str]]) -> set[tuple[str, str]]:
    return {(parent, node) for node, parents in precedence.items() for parent in parents}


class PrecedenceFromThePlanTests(unittest.TestCase):
    def test_derivation_matches_the_hand_read_gantt_chart(self):
        precedence = derive_precedence(_plan(), RECIPE_ORDER)
        self.assertEqual(_edges(precedence), EXPECTED_EDGES)

    def test_the_only_element_with_no_predecessor_is_the_first_connection(self):
        precedence = derive_precedence(_plan(), RECIPE_ORDER)
        roots = {node for node, parents in precedence.items() if not parents}
        self.assertEqual(roots, {"AUX_006_CONNECT"})

    def test_a_separation_waits_for_its_own_connection_as_well_as_the_previous_one(self):
        precedence = derive_precedence(_plan(), RECIPE_ORDER)
        self.assertEqual(precedence["separation_002"],
                         {"separation_001", "separation_002_CONNECT"})
        self.assertEqual(precedence["separation_003"],
                         {"separation_002", "separation_003_CONNECT"})

    def test_the_connection_made_at_the_top_of_the_batch_is_not_repeated(self):
        # separation_001 reuses HC30_Out1 -> HC40_In1, opened before the aux
        # transfer. Settling already depends on it, so it must not be listed.
        precedence = derive_precedence(_plan(), RECIPE_ORDER)
        self.assertEqual(precedence["separation_001"], {"settling_001"})

    def test_an_element_that_finished_early_still_holds_up_its_successor(self):
        without = derive_precedence(_plan())
        self.assertNotIn("dose_001", without["mix_001"])
        with_recipe = derive_precedence(_plan(), RECIPE_ORDER)
        self.assertIn("dose_001", with_recipe["mix_001"])

    def test_successors_mirror_predecessors(self):
        precedence = derive_precedence(_plan(), RECIPE_ORDER)
        forward = successors_of(precedence)
        self.assertEqual(
            {(parent, node) for parent, nodes in forward.items() for node in nodes},
            EXPECTED_EDGES,
        )


def _choice_graph(guarded: bool) -> RecipeGraph:
    """Start -> XOR(usage_001 | usage_002) -> End, planner- or runtime-decided."""
    nodes = [
        RecipeNode("__start__", "Start", "Start"),
        RecipeNode("xor_split", "Gateway", "XOR_SPLIT", gateway_type="XOR_SPLIT",
                   metadata={"gatewayGroupId": "XOR_008"}),
        RecipeNode("usage_001", "Activity", "Usage", activity_type="usage",
                   metadata={"branchGroupId": "XOR_008", "branchId": "fast"}),
        RecipeNode("usage_002", "Activity", "Usage", activity_type="usage",
                   metadata={"branchGroupId": "XOR_008", "branchId": "standard"}),
        RecipeNode("xor_join", "Gateway", "XOR_JOIN", gateway_type="XOR_JOIN",
                   metadata={"gatewayGroupId": "XOR_008"}),
        RecipeNode("__end__", "End", "End"),
    ]
    chosen = ({"op": "var", "name": "quality_ok"} if guarded
              else {"op": "literal", "value": True})
    other = ({"op": "not", "value": {"op": "var", "name": "quality_ok"}} if guarded
             else {"op": "literal", "value": False})
    edges = [
        RecipeEdge("e0", "__start__", "xor_split"),
        RecipeEdge("e1", "xor_split", "usage_001", condition=chosen, branch_id="fast"),
        RecipeEdge("e2", "xor_split", "usage_002", condition=other, branch_id="standard"),
        RecipeEdge("e3", "usage_001", "xor_join"),
        RecipeEdge("e4", "usage_002", "xor_join"),
        RecipeEdge("e5", "xor_join", "__end__"),
    ]
    return RecipeGraph(id="Choice", recipe_level="Master", nodes=nodes, edges=edges)


def _usage_only_plan():
    return SimpleNamespace(operations=[SimpleNamespace(
        recipe_node_id="usage_001", operation_type="usage",
        source_module="HC30", out_port="", target_module="HC30", in_port="",
        start_s=0.0, end_s=1800.0, duration_s=1800.0,
        module="HC30", operation="usage", step_id="usage_001")])


class OnlyWhatRunsIsInTheRecipeTests(unittest.TestCase):
    def test_a_choice_the_planner_resolved_keeps_only_the_branch_it_chose(self):
        graph = _choice_graph(guarded=False)
        inject_connection_steps(graph, _usage_only_plan())
        remaining = {node.id for node in graph.nodes}
        self.assertNotIn("usage_002", remaining)
        self.assertNotIn("xor_split", remaining)
        self.assertNotIn("xor_join", remaining)
        self.assertEqual(remaining, {"__start__", "usage_001", "__end__"})

    def test_a_choice_a_live_guard_decides_keeps_every_branch(self):
        graph = _choice_graph(guarded=True)
        inject_connection_steps(graph, _usage_only_plan())
        remaining = {node.id for node in graph.nodes}
        for node_id in ("usage_001", "usage_002", "xor_split", "xor_join"):
            self.assertIn(node_id, remaining, f"{node_id} is still decided at runtime")

    def test_an_element_the_plan_gave_no_operation_is_not_a_step(self):
        # Dosing C: the material is already in the vessel, so nothing runs.
        graph = _linear_graph()
        graph.nodes.append(RecipeNode("dose_003", "Activity", "Dosing C", activity_type="dose"))
        graph.edges.append(RecipeEdge("extra_in", "__start__", "dose_003"))
        graph.edges.append(RecipeEdge("extra_out", "dose_003", "mix_001"))
        inject_connection_steps(graph, _plan())
        self.assertNotIn("dose_003", {node.id for node in graph.nodes})


def _linear_graph() -> RecipeGraph:
    """Start -> dose_001 -> dose_002 -> mix_001 -> ... -> End, no gateways."""
    order = ["dose_001", "dose_002", "mix_001", "usage_001", "settling_001",
             "separation_001", "separation_002", "separation_003"]
    nodes = [RecipeNode("__start__", "Start", "Start")]
    nodes += [RecipeNode(node_id, "Activity", node_id, activity_type=node_id.split("_")[0])
              for node_id in order]
    nodes.append(RecipeNode("__end__", "End", "End"))
    chain = ["__start__", *order, "__end__"]
    edges = [RecipeEdge(f"e{index}", source, target)
             for index, (source, target) in enumerate(zip(chain, chain[1:]))]
    return RecipeGraph(id="Demo", recipe_level="Master", nodes=nodes, edges=edges)


class MasterRecipeShapeTests(unittest.TestCase):
    def setUp(self):
        self.graph = _linear_graph()
        inject_connection_steps(self.graph, _plan())
        self.node_map = self.graph.node_map
        self.incoming = {node.id: [edge.source for edge in self.graph.incoming(node.id)]
                         for node in self.graph.nodes}
        self.outgoing = {node.id: [edge.target for edge in self.graph.outgoing(node.id)]
                         for node in self.graph.nodes}

    def test_every_planned_operation_reaches_the_chart(self):
        for node, *_ in SOLVED_PLAN:
            expected = f"conn_{node}" if node.endswith("_CONNECT") else node
            self.assertIn(expected, self.node_map, f"{node} is missing from the recipe")

    def test_a_linear_stretch_carries_no_gateway(self):
        # mix -> usage -> settling has one predecessor and one successor each.
        for node_id in ("usage_001", "settling_001"):
            self.assertEqual(len(self.incoming[node_id]), 1)
            self.assertEqual(len(self.outgoing[node_id]), 1)
            neighbour = self.node_map[self.incoming[node_id][0]]
            self.assertEqual(neighbour.gateway_type, "",
                             f"{node_id} does not need a gateway to be entered")

    def test_converging_elements_are_entered_through_an_and_join(self):
        for node_id in ("dose_001", "dose_002", "separation_002", "separation_003"):
            entered_by = self.incoming[node_id]
            self.assertEqual(len(entered_by), 1)
            self.assertEqual(self.node_map[entered_by[0]].gateway_type, "AND_JOIN")
            self.assertEqual(len(self.incoming[entered_by[0]]), 2)

    def test_diverging_elements_are_left_through_an_and_split(self):
        for node_id in ("conn_AUX_006_CONNECT", "conn_dose_002_CONNECT"):
            left_by = self.outgoing[node_id]
            self.assertEqual(len(left_by), 1)
            self.assertEqual(self.node_map[left_by[0]].gateway_type, "AND_SPLIT")
            self.assertEqual(len(self.outgoing[left_by[0]]), 2)

    def test_the_recipe_still_starts_and_ends_where_it_did(self):
        self.assertEqual(self.incoming["__start__"], [])
        self.assertEqual(self.outgoing["__start__"], ["conn_AUX_006_CONNECT"])
        self.assertEqual(self.outgoing["__end__"], [])
        self.assertEqual(self.incoming["__end__"], ["separation_003"])

    def test_ordering_lives_in_the_control_flow_alone(self):
        # Duplicating it as Synchronization links let an activity start before
        # its join had fired.
        self.assertEqual([edge.id for edge in self.graph.edges
                          if edge.flow_type != "Control"], [])

    def test_a_connection_step_is_performed_by_a_person(self):
        step = self.node_map["conn_dose_001_CONNECT"]
        self.assertEqual(step.metadata["executionMode"], "manual")
        self.assertEqual(step.metadata["manualAction"], "connect")
        self.assertEqual(step.metadata["outPort"], "HC10_Out1")
        self.assertEqual(step.metadata["inPort"], "HC30_In1")
        self.assertEqual(step.metadata["plannedStartS"], 3.0)


def _choice_recipe(guarded: bool) -> RecipeGraph:
    """Mix, then a two-branch choice of usage, then settling.

    ``guarded`` decides who resolves the choice: a real condition means the
    running process still does, a literal means the planner already has.
    """
    chosen = ({"op": "ref", "name": "quality.fastTrack"} if guarded
              else {"op": "literal", "value": True})
    other = ({"op": "ref", "name": "quality.standardTrack"} if guarded
             else {"op": "literal", "value": False})
    nodes = [
        RecipeNode("__start__", "Start", "Start"),
        RecipeNode("mix_001", "Activity", "Mixing", activity_type="mix",
                   parameters={"rpm": 150, "duration_s": 30}),
        RecipeNode("xor_split_001", "Gateway", "XOR_SPLIT", gateway_type="XOR_SPLIT",
                   metadata={"gatewayGroupId": "XOR_008"}),
        RecipeNode("usage_001", "Activity", "Usage", activity_type="usage",
                   parameters={"duration_s": 1800},
                   metadata={"branchGroupId": "XOR_008", "branchId": "fast"}),
        RecipeNode("usage_002", "Activity", "Usage", activity_type="usage",
                   parameters={"duration_s": 3600},
                   metadata={"branchGroupId": "XOR_008", "branchId": "standard"}),
        RecipeNode("xor_join_001", "Gateway", "XOR_JOIN", gateway_type="XOR_JOIN",
                   metadata={"gatewayGroupId": "XOR_008"}),
        RecipeNode("settling_001", "Activity", "Settling", activity_type="settling",
                   parameters={"duration_s": 300}),
        RecipeNode("__end__", "End", "End"),
    ]
    edges = [
        RecipeEdge("e0", "__start__", "mix_001"),
        RecipeEdge("e1", "mix_001", "xor_split_001"),
        RecipeEdge("e2", "xor_split_001", "usage_001", condition=chosen, branch_id="fast"),
        RecipeEdge("e3", "xor_split_001", "usage_002", condition=other, branch_id="standard"),
        RecipeEdge("e4", "usage_001", "xor_join_001"),
        RecipeEdge("e5", "usage_002", "xor_join_001"),
        RecipeEdge("e6", "xor_join_001", "settling_001"),
        RecipeEdge("e7", "settling_001", "__end__"),
    ]
    return RecipeGraph(id="Choice", recipe_level="General", nodes=nodes, edges=edges)


def _choice_plan():
    def operation(node, kind, start, end):
        return SimpleNamespace(
            recipe_node_id=node, operation_type=kind, module="HC30",
            source_module="HC30", target_module="HC30", out_port="", in_port="",
            start_s=start, end_s=end, duration_s=end - start,
            operation=kind, step_id=node, branch_group_id="", branch_id="")

    return SimpleNamespace(
        operations=[operation("mix_001", "mix", 0.0, 30.0),
                    operation("usage_001", "usage", 30.0, 1830.0),
                    operation("settling_001", "settling", 1830.0, 2130.0)],
        selected_branches={"XOR_008": "fast"},
    )


from recipe_graph_adapter import recipe_graph_to_ir  # noqa: E402


def _exported(guarded: bool):
    """The Master Recipe as it lands on disk, read back."""
    import tempfile
    from pathlib import Path

    from isa88_recipe import save_master_recipe_xml_from_ir
    from modplant_recipe import import_master_recipe

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "MasterRecipe_Choice.xml"
        save_master_recipe_xml_from_ir(
            recipe_graph_to_ir(_choice_recipe(guarded)), path,
            selected_branches={"XOR_008": "fast"}, planner_result=_choice_plan(),
        )
        return import_master_recipe(path)


class TheExportedRecipeCarriesOnlyLiveChoicesTests(unittest.TestCase):
    """Checked on the file itself, not on the graph on the way to it: the export
    binds equipment and prunes after the control flow is built, so it is the
    only place that says what an operator actually receives."""

    def test_a_choice_the_planner_made_leaves_just_the_branch_it_chose(self):
        graph = _exported(guarded=False)
        self.assertEqual(sorted(node.id for node in graph.nodes if "usage" in node.id),
                         ["usage_001"])
        self.assertEqual([node.id for node in graph.nodes
                          if "XOR" in (node.gateway_type or "")], [])

    def test_that_branch_is_spliced_where_the_choice_used_to_be(self):
        graph = _exported(guarded=False)
        self.assertEqual([edge.source for edge in graph.incoming("usage_001")], ["mix_001"])
        self.assertEqual([edge.target for edge in graph.outgoing("usage_001")], ["settling_001"])

    def test_which_choice_it_came_from_is_on_record_before_it_is_written(self):
        """Known limitation: a Master Recipe round trip here carries no node
        metadata at all - not even actualEquipmentId - so where a collapsed
        branch came from can only be checked on the graph being exported."""
        from connection_steps import inject_connection_steps
        from recipe_graph_adapter import recipe_ir_to_graph

        graph = recipe_ir_to_graph(recipe_graph_to_ir(_choice_recipe(guarded=False)),
                                   recipe_level="Master")
        inject_connection_steps(graph, _choice_plan())
        usage = graph.node_map["usage_001"]
        self.assertEqual(usage.metadata.get("branchGroupId"), "XOR_008")
        self.assertEqual(usage.metadata.get("branchId"), "fast")

    def test_a_choice_the_process_makes_keeps_every_branch_and_its_guard(self):
        graph = _exported(guarded=True)
        self.assertEqual(sorted(node.id for node in graph.nodes if "usage" in node.id),
                         ["usage_001", "usage_002"])
        self.assertEqual(sorted(node.gateway_type for node in graph.nodes
                                if "XOR" in (node.gateway_type or "")),
                         ["XOR_JOIN", "XOR_SPLIT"])
        guards = [edge.condition for edge in graph.edges if edge.condition]
        self.assertEqual(len(guards), 2, "both branches keep the condition that picks them")
        for guard in guards:
            self.assertNotEqual(guard.get("op"), "literal")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
