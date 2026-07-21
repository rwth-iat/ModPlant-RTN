from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import unittest


RTN_ROOT = Path(__file__).resolve().parents[1]

from modplant_rtn.models import default_plant_model, plant_model_from_rtn_inputs
from modplant_rtn.service import RTNSettings
from notebook_helpers import plan_dataframe
from sample_data import (
    sample_equipment_bindings,
    sample_module_interfaces,
    sample_module_maximum_volume,
    sample_module_ops,
    sample_module_resources,
)


class NotebookSharedKernelTests(unittest.TestCase):
    def test_notebook_demo_capabilities_are_the_app_demo_capabilities(self):
        actual = (
            sample_module_ops(),
            sample_module_interfaces(),
            sample_module_maximum_volume(),
            sample_module_resources(),
        )
        self.assertEqual(actual, default_plant_model().to_rtn_inputs())
        for capabilities in actual[0].values():
            for record in capabilities:
                self.assertEqual(len(record), 6)
                self.assertGreaterEqual(record[3], 0)
                self.assertGreaterEqual(record[4], 0)
            self.assertGreater(next(record[3] for record in capabilities if record[0] == "Draining"), 0)
            self.assertGreater(next(record[4] for record in capabilities if record[0] == "Draining"), 0)
            self.assertEqual(next(record[5] for record in capabilities if record[0] == "Connect"), 3)
            self.assertEqual(next(record[5] for record in capabilities if record[0] == "Disconnect"), 2)

    def test_editable_rtn_inputs_round_trip_through_the_app_plant_model(self):
        expected = default_plant_model().to_rtn_inputs()
        plant = plant_model_from_rtn_inputs(
            *expected,
            equipment_bindings=sample_equipment_bindings(),
        )
        self.assertEqual(plant.to_rtn_inputs(), expected)
        self.assertEqual(
            plant.equipment_bindings(),
            default_plant_model().equipment_bindings(),
        )

    def test_app_settings_own_the_only_planner_config_mapping(self):
        settings = RTNSettings(
            profit_per_litre=321,
            time_penalty_per_second=0.75,
            energy_cost_weight=2.5,
            co2_penalty=3.5,
            disconnect_duration_s=4,
            require_final_disconnect=True,
        )
        config = settings.to_planner_config(6)
        self.assertEqual(config.base_profit, 1926)
        self.assertEqual(config.lambda_per_second, -0.75)
        self.assertEqual(config.energy_cost_weight, 2.5)
        self.assertEqual(config.co2_penalty, 3.5)
        self.assertEqual(config.disconnect_duration_s, 4)
        self.assertTrue(config.require_final_disconnect)

    def test_plan_dataframe_is_dynamic_and_formats_costs_for_display(self):
        session = SimpleNamespace(plan_rows=[{
            "Step": 1,
            "Total Cost": 2.3456,
            "Weighted Energy Cost": 0.004,
            "Future App Field": "automatically visible",
        }])
        frame = plan_dataframe(session)
        self.assertEqual(frame.loc[0, "Total Cost"], 2.35)
        self.assertEqual(frame.loc[0, "Weighted Energy Cost"], 0.0)
        self.assertEqual(frame.loc[0, "Future App Field"], "automatically visible")

    def test_both_notebooks_keep_editable_inputs_but_delegate_to_app_modules(self):
        manual = json.loads((RTN_ROOT / "ModPlant-RTN.ipynb").read_text(encoding="utf-8"))
        widget = json.loads((RTN_ROOT / "ModPlant-RTN-UI.ipynb").read_text(encoding="utf-8"))
        manual_source = "\n".join("".join(cell.get("source", [])) for cell in manual["cells"])
        widget_source = "\n".join("".join(cell.get("source", [])) for cell in widget["cells"])
        widget_helper = (RTN_ROOT / "scripts" / "rtn_notebook_ui.py").read_text(encoding="utf-8")
        notebook_helper = (RTN_ROOT / "scripts" / "notebook_helpers.py").read_text(encoding="utf-8")

        for source in (manual_source, widget_source):
            self.assertIn("module_ops", source)
            self.assertIn("_hc_data = [", source)
            self.assertLess(source.index("_hc_data = ["), source.index("module_interfaces = {"))
            self.assertIn("energy_rate", source)
            self.assertIn("co2_rate", source)
            self.assertIn("equipment_bindings", source)
        self.assertIn("optimize_recipe_ir", manual_source)
        self.assertIn("optimize_recipe_ir", widget_helper)
        self.assertNotIn("PLAN_COLUMNS", widget_helper)
        self.assertIn("display_process_plan_graph(", manual_source)
        self.assertIn("display_process_plan_graph(", widget_helper)
        self.assertIn("visualization.gantt_svg(planner_result, title=title)", notebook_helper)
        self.assertNotIn("include_tooltips=False", notebook_helper)


if __name__ == "__main__":
    unittest.main()
