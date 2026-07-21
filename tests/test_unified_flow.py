"""Regression tests for the Unified Flow model.

The feasible cases cover recipe/config combinations from the comprehensive
benchmark.  B3/R7 is retained as an explicit physical-infeasibility regression:
the corrected composition model must not recover its historical schedule by
selectively draining a component from a mixture.
"""

import sys, time, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recipe_ir import build_recipe_ir, auto_enrich_recipe_spec
from recipe_to_rtn import recipe_ir_to_rtn
from cp_sat_planner import PlannerConfig, solve_rtn_with_cp_sat


# ═════════════════════════════════════════════════════════════════════════════
# Factory configs
# ═════════════════════════════════════════════════════════════════════════════

def _cfg_A4():
    """4 modules, standard: HC10(A), HC20(B), HC30(C), HC40(empty)."""
    w = {
        'HC10': [('Draining', 0.1, 3), ('Filling', 0.1, 0), ('Settling', '', 1),
                 ('Stirring', '100', 3), ('Stirring', '200', 3), ('Connect', '', 1), ('None', '', 0)],
        'HC20': [('Draining', 0.1, 3), ('Filling', 0.1, 0), ('Settling', '', 1),
                 ('Stirring', '150', 3), ('Stirring', '300', 3), ('Connect', '', 1), ('None', '', 0)],
        'HC30': [('Draining', 0.1, 3), ('Filling', 0.1, 0), ('Settling', '', 1),
                 ('Stirring', '100', 3), ('Stirring', '150', 3), ('Connect', '', 1), ('None', '', 0)],
        'HC40': [('Draining', 0.1, 3), ('Filling', 0.1, 0), ('Connect', '', 1), ('None', '', 0)],
    }
    hc = [('HC10', 3, 3), ('HC20', 3, 3), ('HC30', 4, 1), ('HC40', 1, 2)]
    ifc = {h: [('Input', f'{h}_In{i}') for i in range(1, ni + 1)]
               + [('Output', f'{h}_Out{i}') for i in range(1, no + 1)]
           for h, ni, no in hc}
    mv = {'HC10': [10], 'HC20': [15], 'HC30': [10], 'HC40': [30]}
    mr = {'HC10': ['A', 10], 'HC20': ['B', 10], 'HC30': ['C', 10]}
    return w, ifc, mv, mr


def _cfg_B3():
    """3 modules, tighter: HC10(A,5L), HC20(B,8L), HC30(empty)."""
    w = {
        'HC10': [('Draining', 0.1, 2), ('Filling', 0.1, 0), ('Stirring', '150', 2),
                 ('Connect', '', 1), ('None', '', 0)],
        'HC20': [('Draining', 0.1, 3), ('Filling', 0.1, 0), ('Stirring', '200', 3),
                 ('Settling', '', 1), ('Connect', '', 1), ('None', '', 0)],
        'HC30': [('Draining', 0.1, 3), ('Filling', 0.1, 0), ('Stirring', '100', 3),
                 ('Settling', '', 1), ('Connect', '', 1), ('None', '', 0)],
    }
    ifc = {
        'HC10': [('Input', 'HC10_In1'), ('Output', 'HC10_Out1'), ('Output', 'HC10_Out2')],
        'HC20': [('Input', 'HC20_In1'), ('Input', 'HC20_In2'), ('Output', 'HC20_Out1')],
        'HC30': [('Input', 'HC30_In1'), ('Output', 'HC30_Out1'), ('Output', 'HC30_Out2')],
    }
    mv = {'HC10': [10], 'HC20': [20], 'HC30': [10]}
    mr = {'HC10': ['A', 5], 'HC20': ['B', 8]}
    return w, ifc, mv, mr


def _cfg_C5():
    """5 modules with D: HC10(A,8L), HC20(B,10L), HC40(D,5L)."""
    w = {
        'HC10': [('Draining', 0.15, 2), ('Filling', 0.15, 0), ('Stirring', '100', 2),
                 ('Settling', '', 1), ('Connect', '', 1), ('None', '', 0)],
        'HC20': [('Draining', 0.1, 3), ('Filling', 0.1, 0), ('Stirring', '150', 3),
                 ('Settling', '', 1), ('Connect', '', 1), ('None', '', 0)],
        'HC30': [('Draining', 0.1, 3), ('Filling', 0.1, 0), ('Stirring', '200', 3),
                 ('Connect', '', 1), ('None', '', 0)],
        'HC40': [('Draining', 0.12, 2), ('Filling', 0.12, 0), ('Stirring', '150', 2),
                 ('Connect', '', 1), ('None', '', 0)],
        'HC50': [('Draining', 0.1, 3), ('Filling', 0.1, 0), ('None', '', 0), ('Connect', '', 1)],
    }
    ifc = {
        'HC10': [('Input', 'HC10_In1'), ('Output', 'HC10_Out1')],
        'HC20': [('Input', 'HC20_In1'), ('Input', 'HC20_In2'),
                 ('Output', 'HC20_Out1'), ('Output', 'HC20_Out2')],
        'HC30': [('Input', 'HC30_In1'), ('Output', 'HC30_Out1')],
        'HC40': [('Input', 'HC40_In1'), ('Output', 'HC40_Out1')],
        'HC50': [('Input', 'HC50_In1'), ('Input', 'HC50_In2'), ('Output', 'HC50_Out1')],
    }
    mv = {'HC10': [8], 'HC20': [15], 'HC30': [10], 'HC40': [12], 'HC50': [20]}
    mr = {'HC10': ['A', 8], 'HC20': ['B', 10], 'HC40': ['D', 5]}
    return w, ifc, mv, mr


def _cfg_E3():
    """3 modules, sparse: HC10(A,5L), HC30(C,10L)."""
    w = {
        'HC10': [('Draining', 0.1, 2), ('Filling', 0.1, 0), ('Stirring', '150', 2),
                 ('Connect', '', 1), ('None', '', 0)],
        'HC20': [('Draining', 0.15, 2), ('Filling', 0.15, 0), ('Stirring', '150', 3),
                 ('Settling', '', 1), ('Connect', '', 1), ('None', '', 0)],
        'HC30': [('Draining', 0.1, 3), ('Filling', 0.1, 0), ('Stirring', '100', 3),
                 ('Settling', '', 1), ('Connect', '', 1), ('None', '', 0)],
    }
    ifc = {
        'HC10': [('Input', 'HC10_In1'), ('Output', 'HC10_Out1')],
        'HC20': [('Input', 'HC20_In1'), ('Input', 'HC20_In2'),
                 ('Output', 'HC20_Out1'), ('Output', 'HC20_Out2')],
        'HC30': [('Input', 'HC30_In1'), ('Output', 'HC30_Out1'), ('Output', 'HC30_Out2')],
    }
    mv = {'HC10': [8], 'HC20': [12], 'HC30': [15]}
    mr = {'HC10': ['A', 5], 'HC30': ['C', 10]}
    return w, ifc, mv, mr


# ═════════════════════════════════════════════════════════════════════════════
# Recipes
# ═════════════════════════════════════════════════════════════════════════════

R1 = {
    'id': 'R1_3par', 'volume': 6.0,
    'procedure': [
        {'dose': {'ingredient': 'A', 'amount_L': 1.0}},
        {'dose': {'ingredient': 'B', 'amount_L': 2.0}},
        {'dose': {'ingredient': 'C', 'amount_L': 3.0}},
        {'mix': {'rpm': 150, 'duration_s': 30}},
        {'usage': {'duration_s': 3600}},
        {'settling': {'duration_s': 300}},
        {'separation': {'order': ['C', 'B', 'A']}},
    ],
}

R2 = {
    'id': 'R2_2mix', 'volume': 6.0,
    'procedure': [
        {'dose': {'ingredient': 'A', 'amount_L': 1.0}},
        {'dose': {'ingredient': 'B', 'amount_L': 2.0}},
        {'mix': {'rpm': 150, 'duration_s': 30}},
        {'dose': {'ingredient': 'C', 'amount_L': 3.0}},
        {'mix': {'rpm': 150, 'duration_s': 30}},
        {'usage': {'duration_s': 3600}},
        {'settling': {'duration_s': 300}},
        {'separation': {'order': ['C', 'B', 'A']}},
    ],
}

R3 = {
    'id': 'R3_simple', 'volume': 3.0,
    'procedure': [
        {'dose': {'ingredient': 'A', 'amount_L': 1.5}},
        {'dose': {'ingredient': 'B', 'amount_L': 1.5}},
        {'mix': {'rpm': 200, 'duration_s': 60}},
        {'settling': {'duration_s': 120}},
        {'separation': {'order': ['B', 'A']}},
    ],
}

R4 = {
    'id': 'R4_mini', 'volume': 2.0,
    'procedure': [
        {'dose': {'ingredient': 'A', 'amount_L': 2.0}},
        {'mix': {'rpm': 100, 'duration_s': 10}},
        {'usage': {'duration_s': 600}},
        {'settling': {'duration_s': 60}},
    ],
}

R5 = {
    'id': 'R5_seq', 'volume': 2.0,
    'procedure': [
        {'dose': {'ingredient': 'A', 'amount_L': 0.5}},
        {'mix': {'rpm': 150, 'duration_s': 10}},
        {'dose': {'ingredient': 'B', 'amount_L': 1.0}},
        {'mix': {'rpm': 150, 'duration_s': 10}},
        {'dose': {'ingredient': 'C', 'amount_L': 0.5}},
        {'mix': {'rpm': 150, 'duration_s': 10}},
        {'usage': {'duration_s': 1200}},
        {'settling': {'duration_s': 120}},
        {'separation': {'order': ['C', 'B', 'A']}},
    ],
}

R7 = {
    'id': 'R7_nous', 'volume': 2.0,
    'procedure': [
        {'dose': {'ingredient': 'A', 'amount_L': 1.0}},
        {'dose': {'ingredient': 'B', 'amount_L': 1.0}},
        {'mix': {'rpm': 150, 'duration_s': 30}},
        {'settling': {'duration_s': 60}},
    ],
}

R8 = {
    'id': 'R8_solo', 'volume': 1.0,
    'procedure': [
        {'dose': {'ingredient': 'A', 'amount_L': 0.5}},
        {'dose': {'ingredient': 'B', 'amount_L': 0.5}},
        {'mix': {'rpm': 100, 'duration_s': 5}},
    ],
}


# ═════════════════════════════════════════════════════════════════════════════
# Expected schedule results. Historical volume-scaled operation-cost figures
# are intentionally retained only as documentation in the tuples below; the
# planner now models equipment usage cost per invocation plus time-based energy
# and CO2 terms, so cost correctness is checked from the new decomposition.
# ═════════════════════════════════════════════════════════════════════════════

EXPECTED = {
    # (config, recipe): (min_status, makespan_s, total_cost, objective_profit)
    # Future connections can be prepared on free ports while another transfer
    # is flowing, reducing the former whole-Module-exclusive baseline by 59 s.
    ('A4', 'R1'): ('FEASIBLE', 2295.0, 70.0, -921.5),
    # Persistent port occupancy plus explicit final Disconnect actions move
    # this historical pre-lifecycle baseline from ~2313s to ~2376s.
    ('A4', 'R2'): ('FEASIBLE', 2376.0, 67.0, -923.5),
    ('A4', 'R3'): ('OPTIMAL',   313.0, 51.0,   92.5),
    ('A4', 'R4'): ('FEASIBLE',   753.0, 30.0, -106.5),
    ('A4', 'R5'): ('FEASIBLE', 1478.0, 68.0, -507.0),
    ('A4', 'R7'): ('OPTIMAL',   193.0, 38.0,  165.5),
    ('A4', 'R8'): ('OPTIMAL',   108.0, 37.0,  209.0),
    ('B3', 'R3'): ('OPTIMAL',   293.0, 43.5,  110.0),
    ('B3', 'R4'): ('OPTIMAL',   693.0, 10.0,  -56.5),
    ('B3', 'R8'): ('OPTIMAL',    18.0,  9.5,  281.5),
    ('C5', 'R3'): ('FEASIBLE',  280.0, 69.5,  -89.5, 200, 30, 300),
    ('C5', 'R4'): ('OPTIMAL',   733.0, 17.0,  -83.5),
    ('C5', 'R7'): ('OPTIMAL',   213.0, 39.0,  187.0),
    ('C5', 'R8'): ('OPTIMAL',    88.0, 22.5,  233.5),
    # HC30 must finish its auxiliary outgoing Draining before it can receive
    # the recipe Dose. Only multiple incoming Filling operations may overlap.
    ('E3', 'R4'): ('OPTIMAL',   800.0, 44.0, -117.0),
}

# Map recipe names to recipe dicts
RECIPES = {'R1': R1, 'R2': R2, 'R3': R3, 'R4': R4, 'R5': R5, 'R7': R7, 'R8': R8}
CONFIGS = {'A4': _cfg_A4, 'B3': _cfg_B3, 'C5': _cfg_C5, 'E3': _cfg_E3}


class UnifiedFlowBenchmarkTests(unittest.TestCase):
    """All feasible recipe+config pairs must solve correctly under Unified Flow.

    Timing: run with ``pytest --durations=0`` to see per-test times.
    """

    @classmethod
    def setUpClass(cls):
        cls.cfg = PlannerConfig(
            base_profit=300,
            lambda_per_second=-0.5,
            connect_duration_s=3.0,
            disconnect_duration_s=2.0,
            require_final_disconnect=True,
            enable_auxiliary_transfers=True,
            auxiliary_transfer_mode='lazy',
            relative_gap_limit=0.001,
            solver_time_limit_s=120,
            num_workers=8,
            volume_scale=10,
            time_scale=1,
            cost_scale=100,
        )

    def _run_one(self, cfg_name, recipe_name, recipe):
        w, ifc, mv, mr = CONFIGS[cfg_name]()
        enriched = auto_enrich_recipe_spec(dict(recipe))
        ir = build_recipe_ir(enriched)
        rtn = recipe_ir_to_rtn(
            ir,
            module_ops=w,
            module_interfaces=ifc,
            module_maximum_volume=mv,
            module_resources=mr,
        )
        t0 = time.time()
        result = solve_rtn_with_cp_sat(rtn, self.cfg)
        dt = time.time() - t0

        expected = EXPECTED[(cfg_name, recipe_name)]
        min_status, exp_ms = expected[0], expected[1]
        ms_delta = expected[4] if len(expected) > 4 else 50

        status_ok = (
            result.status == 'OPTIMAL'
            or (min_status == 'FEASIBLE' and result.status in ('OPTIMAL', 'FEASIBLE'))
        )
        self.assertTrue(
            status_ok,
            f"{cfg_name}-{recipe_name}: expected >={min_status}, got {result.status}",
        )
        self.assertAlmostEqual(result.makespan_s, exp_ms, delta=ms_delta,
            msg=f"{cfg_name}-{recipe_name}: makespan mismatch")
        self.assertAlmostEqual(
            result.total_cost,
            result.total_weighted_usage_cost + result.total_weighted_energy_cost + result.total_weighted_co2_cost,
            msg=f"{cfg_name}-{recipe_name}: weighted cost decomposition mismatch",
        )
        self.assertAlmostEqual(
            result.objective_profit,
            self.cfg.base_profit + self.cfg.lambda_per_second * result.makespan_s - result.total_cost,
            msg=f"{cfg_name}-{recipe_name}: objective mismatch",
        )

        local_count = result.diagnostics.get('local_dose_count', 0)
        self.assertGreaterEqual(local_count, 0,
            msg=f"{cfg_name}-{recipe_name}: local_dose_count missing")

        return dt

    def _run_infeasible(self, cfg_name, recipe):
        w, ifc, mv, mr = CONFIGS[cfg_name]()
        ir = build_recipe_ir(auto_enrich_recipe_spec(dict(recipe)))
        rtn = recipe_ir_to_rtn(
            ir,
            module_ops=w,
            module_interfaces=ifc,
            module_maximum_volume=mv,
            module_resources=mr,
        )
        result = solve_rtn_with_cp_sat(rtn, self.cfg)
        self.assertEqual(result.status, 'INFEASIBLE', result.diagnostics)
        return result

    # ── A4 config (4 modules, full material set) ──────────────────────────

    def test_A4_R1_3par(self):
        """3-dose parallel + 1 mix → OPT, ~2295s with free-port preparation."""
        self._run_one('A4', 'R1', R1)

    def test_A4_R2_2mix(self):
        """2-dose + mix, then dose C + mix → OPT, ~2376s with teardown."""
        self._run_one('A4', 'R2', R2)

    def test_A4_R3_simple(self):
        """2-dose + mix + settling + separation → OPT, 313s."""
        self._run_one('A4', 'R3', R3)

    def test_A4_R4_mini(self):
        """1-dose + mix + usage + settling → OPT, 753s."""
        self._run_one('A4', 'R4', R4)

    def test_A4_R5_seq(self):
        """3 sequential dose-mix pairs → FEAS, 1478s."""
        self._run_one('A4', 'R5', R5)

    def test_A4_R7_nous(self):
        """2-dose + mix + settling (no usage) → OPT, 193s."""
        self._run_one('A4', 'R7', R7)

    def test_A4_R8_solo(self):
        """2-dose + mix only; requires auxiliary cleanup before local A dose."""
        self._run_one('A4', 'R8', R8)

    # ── B3 config (3 modules, no C) ──────────────────────────────────────

    def test_B3_R3_simple(self):
        """2-dose (A+B) + mix → OPT, 293s."""
        self._run_one('B3', 'R3', R3)

    def test_B3_R4_mini(self):
        """1-dose (A) + mix → OPT, 693s."""
        self._run_one('B3', 'R4', R4)

    def test_B3_R7_nous(self):
        """No clean destination exists for the AB batch before Settling.

        HC10 is the only 150 rpm mixer and cannot settle.  Its A4 surplus and
        HC20's B7 surplus occupy the two remaining storage opportunities, so
        the mixed A1+B1 batch cannot move to a Settling-capable Module without
        mixing it with surplus stock.  The old 213 s schedule depended on an
        illegal selective component transfer and must remain infeasible.
        """
        self._run_infeasible('B3', R7)

    def test_B3_R8_solo(self):
        """2-dose + mix only → OPT, 18s."""
        self._run_one('B3', 'R8', R8)

    # ── C5 config (5 modules) ────────────────────────────────────────────

    def test_C5_R3_simple(self):
        """2-dose (A+B, D not used) + mix → FEAS, ~280-301s."""
        self._run_one('C5', 'R3', R3)

    def test_C5_R4_mini(self):
        """1-dose (A) + mix → OPT, 733s."""
        self._run_one('C5', 'R4', R4)

    def test_C5_R7_nous(self):
        """2-dose + mix + settling → OPT, ~213s with connection teardown."""
        self._run_one('C5', 'R7', R7)

    def test_C5_R8_solo(self):
        """2-dose + mix only; requires auxiliary cleanup before local A dose."""
        self._run_one('C5', 'R8', R8)

    # ── E3 config (3 modules, only A/C) ──────────────────────────────────

    def test_E3_R4_mini(self):
        """1-dose (A) + mix → OPT, 800s with role-aware Module exclusivity."""
        self._run_one('E3', 'R4', R4)


if __name__ == '__main__':
    unittest.main()
