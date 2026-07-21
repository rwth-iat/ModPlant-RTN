"""ipywidgets interface for the seeded RTN planning workflow."""
from __future__ import annotations

import json
import secrets
import sys
import threading
import time
from html import escape
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from IPython.display import HTML, display

try:
    import ipywidgets as widgets
except ImportError:  # pragma: no cover - depends on the selected Jupyter kernel
    widgets = None

from notebook_helpers import (
    auto_enrich_recipe_spec,
    build_recipe_ir,
    display_connection_graph,
    display_df,
    display_process_plan_graph,
    display_recipe_graph,
    plan_dataframe,
    validate_schedule,
)
from random_recipe import generate_random_recipe_spec
from recipe_graph_adapter import recipe_graph_to_display_ir, recipe_graph_to_ir
from runtime_recipe_variants import (
    combined_runtime_variant,
    conditional_jump_variant,
    or_variant,
    runtime_loop_variant,
)
from isa88_recipe import save_general_recipe_xml_from_ir
from modplant_recipe import export_general_recipe
from sample_data import (
    sample_module_interfaces,
    sample_module_maximum_volume,
    sample_module_ops,
    sample_module_resources,
)
_RTN_ROOT = Path(__file__).resolve().parents[1]
if str(_RTN_ROOT) not in sys.path:
    sys.path.insert(0, str(_RTN_ROOT))
from modplant_rtn.models import plant_model_from_rtn_inputs
from modplant_rtn.service import RTNSettings, export_master_recipe, optimize_recipe_ir


class RTNNotebookUI:
    """Reusable notebook UI for seed -> recipe -> RTN -> CP-SAT -> validation."""

    def __init__(
        self,
        project_root: str | Path,
        *,
        module_ops=None,
        module_interfaces=None,
        module_maximum_volume=None,
        module_resources=None,
        equipment_bindings=None,
    ):
        if widgets is None:
            raise ModuleNotFoundError(
                "ipywidgets is required for the RTN notebook UI. "
                "Install the RTN notebook dependencies with "
                "`%pip install -r scripts/requirements-rtn.txt` in this kernel."
            )

        self.project_root = Path(project_root).resolve()
        # Keep references to the caller's editable dictionaries. Re-running a
        # solve therefore uses in-place notebook changes without rebuilding UI.
        self.module_ops = sample_module_ops() if module_ops is None else module_ops
        self.module_interfaces = sample_module_interfaces() if module_interfaces is None else module_interfaces
        self.module_maximum_volume = sample_module_maximum_volume() if module_maximum_volume is None else module_maximum_volume
        self.module_resources = sample_module_resources() if module_resources is None else module_resources
        self.equipment_bindings = {} if equipment_bindings is None else equipment_bindings

        self.recipe_spec: Optional[Dict[str, Any]] = None
        self.enriched_spec: Optional[Dict[str, Any]] = None
        self.general_graph = None
        self.enriched_graph = None
        self.general_ir = None
        self.enriched_display_ir = None
        self.recipe_ir = None
        self.rtn_model = None
        self.optimization_session = None
        self.planner_config = None
        self.planner_result = None
        self.plan_df = None
        self.validation = None
        self.general_xml_path = ""
        self.master_xml_path = ""
        self.solve_elapsed_s = 0.0
        self._solve_timer_started_at: Optional[float] = None
        self._solve_timer_stop_event: Optional[threading.Event] = None
        self._solve_timer_thread: Optional[threading.Thread] = None

        self._build_controls()
        self._wire_events()

    def _build_controls(self) -> None:
        layout_wide = widgets.Layout(width="360px")
        self.seed = widgets.IntText(
            value=42,
            description="Random Seed",
            style={"description_width": "110px"},
            layout=layout_wide,
        )
        self.random_seed_button = widgets.Button(
            description="New Random Seed",
            icon="refresh",
            layout=widgets.Layout(width="180px"),
        )
        self.topology = widgets.Dropdown(
            options=[
                ("Nonlinear enrichment (AND + XOR)", "nonlinear"),
                ("Parallel enrichment (AND only)", "parallel"),
                ("No enrichment (remain linear)", "linear"),
                ("OR multi-select branches", "or"),
                ("Runtime loop (finite planning projection)", "runtime-loop"),
                ("Conditional jump (continue-path projection)", "conditional-jump"),
                ("Combined OR + loop + jump", "combined"),
            ],
            value="nonlinear",
            description="Topology",
            style={"description_width": "110px"},
            layout=widgets.Layout(width="560px"),
        )
        self.ingredient_count = widgets.IntSlider(
            value=3,
            min=2,
            max=3,
            step=1,
            description="Ingredients",
            style={"description_width": "110px"},
            layout=layout_wide,
        )
        self.time_limit = widgets.FloatSlider(
            value=180,
            min=5,
            max=600,
            step=5,
            description="Time Limit (s)",
            style={"description_width": "110px"},
            layout=widgets.Layout(width="500px"),
        )
        self.num_workers = widgets.IntSlider(
            value=8,
            min=1,
            max=12,
            step=1,
            description="Workers",
            style={"description_width": "110px"},
            layout=widgets.Layout(width="500px"),
        )
        self.time_penalty_per_second = widgets.FloatText(
            value=0.5,
            description="Time penalty / s",
            style={"description_width": "110px"},
            layout=layout_wide,
        )
        # Backward-compatible attribute for notebooks that customized it.
        self.lambda_per_second = self.time_penalty_per_second
        self.profit_per_litre = widgets.FloatText(value=300, description="Revenue / L", style={"description_width": "110px"}, layout=layout_wide)
        self.usage_cost_weight = widgets.FloatText(value=1, description="Usage weight", style={"description_width": "110px"}, layout=layout_wide)
        self.energy_cost_weight = widgets.FloatText(value=1, description="Energy weight", style={"description_width": "110px"}, layout=layout_wide)
        self.co2_penalty = widgets.FloatText(value=1, description="CO₂ penalty", style={"description_width": "110px"}, layout=layout_wide)
        self.electricity_price = widgets.FloatText(value=0.3, description="Electricity €/kWh", style={"description_width": "110px"}, layout=layout_wide)
        self.connect_duration = widgets.FloatText(value=3, description="Connect fallback", style={"description_width": "110px"}, layout=layout_wide)
        self.disconnect_duration = widgets.FloatText(value=2, description="Disconnect fallback", style={"description_width": "110px"}, layout=layout_wide)
        self.relative_gap = widgets.FloatText(value=0.02, description="Relative gap", style={"description_width": "110px"}, layout=layout_wide)
        self.require_final_disconnect = widgets.Checkbox(
            value=False,
            description="Disconnect all connections at plan end",
            indent=False,
            layout=widgets.Layout(width="420px"),
        )

        self.generate_button = widgets.Button(
            description="Generate From Seed",
            icon="random",
            button_style="info",
            layout=widgets.Layout(width="210px"),
        )
        self.run_button = widgets.Button(
            description="Generate & Solve",
            icon="play",
            button_style="success",
            layout=widgets.Layout(width="210px"),
        )
        self.validate_button = widgets.Button(
            description="Validate Latest Schedule",
            icon="check",
            button_style="warning",
            disabled=True,
            layout=widgets.Layout(width="230px"),
        )
        self.status = widgets.HTML(
            value="<span style='color:#6b7280'>Ready.</span>"
        )
        self.solve_timer = widgets.HTML(
            value="<span style='color:#6b7280'>Solve elapsed: not started</span>"
        )
        self.auto_export_master = widgets.Checkbox(
            value=False,
            description="Export automatically after solve",
            indent=False,
        )
        self.export_button = widgets.Button(
            description="Export Master Recipe",
            icon="download",
            button_style="primary",
            disabled=True,
            layout=widgets.Layout(width="220px"),
        )
        self.export_status = widgets.HTML(
            value="<span style='color:#6b7280'>Solve a recipe before exporting.</span>"
        )
        self.validation_status = widgets.HTML(
            value="<span style='color:#6b7280'>Solve a recipe before validating.</span>"
        )

        self.recipe_summary_output = widgets.Output()
        self.control_flow_output = widgets.Output()
        self.solve_output = widgets.Output()
        self.validation_output = widgets.Output()

        self.settings_section = widgets.VBox(
            [
                widgets.HTML("<h3>Settings</h3>"),
                widgets.HBox([self.topology, self.ingredient_count]),
                widgets.HBox([self.time_limit, self.num_workers]),
                widgets.HBox([self.time_penalty_per_second, self.profit_per_litre]),
                widgets.HBox([self.usage_cost_weight, self.energy_cost_weight]),
                widgets.HBox([self.co2_penalty, self.electricity_price]),
                widgets.HBox([self.connect_duration, self.disconnect_duration]),
                widgets.HBox([self.relative_gap, self.require_final_disconnect]),
                widgets.HTML(
                    "<small>Connect/Disconnect durations are fallbacks only; "
                    "fixed durations in <code>module_ops</code> take precedence.</small>"
                ),
            ]
        )
        self.context_section = widgets.VBox(
            [
                widgets.HTML("<h3>Seed and Random Recipe</h3>"),
                widgets.HBox([self.seed, self.random_seed_button, self.generate_button]),
                self.recipe_summary_output,
                self.control_flow_output,
            ]
        )
        self.pipeline_section = widgets.VBox(
            [
                widgets.HTML("<h3>Direct CP-SAT Solve</h3>"),
                widgets.HBox([self.run_button, self.status]),
                self.solve_timer,
                self.solve_output,
            ]
        )
        self.validation_section = widgets.VBox(
            [
                widgets.HTML("<h3>Independent Schedule Validation</h3>"),
                widgets.HBox([self.validate_button, self.validation_status]),
                self.validation_output,
            ]
        )
        self.export_section = widgets.VBox(
            [
                widgets.HTML("<h3>Master Recipe Export</h3>"),
                self.auto_export_master,
                widgets.HBox([self.export_button, self.export_status]),
            ]
        )
        self.app = widgets.VBox(
            [
                self.settings_section,
                self.context_section,
                self.pipeline_section,
                self.validation_section,
                self.export_section,
            ]
        )

    def _wire_events(self) -> None:
        self.random_seed_button.on_click(self._on_new_seed)
        self.generate_button.on_click(self._on_generate)
        self.run_button.on_click(self._on_run)
        self.validate_button.on_click(self._on_validate)
        self.export_button.on_click(self._on_export)

    def build(self) -> "RTNNotebookUI":
        return self

    def _on_new_seed(self, _button) -> None:
        self.seed.value = secrets.randbelow(2_000_000_000)

    def _on_generate(self, _button) -> None:
        self.generate()

    def _on_run(self, _button) -> None:
        self.generate_and_solve()

    def _on_validate(self, _button) -> None:
        try:
            self.run_validation()
        except Exception as exc:
            self.validation_status.value = (
                "<span style='color:#b91c1c'>"
                f"Validation failed: {escape(type(exc).__name__)}</span>"
            )

    def _on_export(self, _button) -> None:
        try:
            self.export_master_recipe()
        except Exception as exc:
            self.export_status.value = (
                "<span style='color:#b91c1c'>"
                f"Export failed: {escape(type(exc).__name__)}</span>"
            )

    def _set_busy(self, busy: bool, message: str) -> None:
        self.generate_button.disabled = busy
        self.run_button.disabled = busy
        self.random_seed_button.disabled = busy
        self.validate_button.disabled = busy or self.planner_result is None
        self.export_button.disabled = (
            busy
            or self.planner_result is None
            or (self.validation is not None and not self.validation.valid)
        )
        color = "#b45309" if busy else "#166534"
        self.status.value = f"<span style='color:{color}'>{message}</span>"

    def _start_solve_timer(self) -> None:
        """Start a live wall-clock timer for the CP-SAT solve only."""
        if self._solve_timer_stop_event is not None:
            self._stop_solve_timer()

        self.solve_elapsed_s = 0.0
        self._solve_timer_started_at = time.monotonic()
        stop_event = threading.Event()
        self._solve_timer_stop_event = stop_event
        self.solve_timer.value = (
            "<span style='color:#b45309'><b>Solving...</b> "
            "elapsed: 0.0 s</span>"
        )

        def update_timer() -> None:
            while not stop_event.wait(0.1):
                started_at = self._solve_timer_started_at
                if started_at is None:
                    return
                elapsed = time.monotonic() - started_at
                self.solve_timer.value = (
                    "<span style='color:#b45309'><b>Solving...</b> "
                    f"elapsed: {elapsed:.1f} s</span>"
                )

        self._solve_timer_thread = threading.Thread(
            target=update_timer,
            name="rtn-solve-timer",
            daemon=True,
        )
        self._solve_timer_thread.start()

    def _stop_solve_timer(self, *, completed: bool = True) -> float:
        """Stop the live solve timer and leave its final duration visible."""
        started_at = self._solve_timer_started_at
        if started_at is None:
            return self.solve_elapsed_s

        elapsed = time.monotonic() - started_at
        stop_event = self._solve_timer_stop_event
        if stop_event is not None:
            stop_event.set()
        timer_thread = self._solve_timer_thread
        if timer_thread is not None and timer_thread is not threading.current_thread():
            timer_thread.join(timeout=0.5)

        self.solve_elapsed_s = elapsed
        self._solve_timer_started_at = None
        self._solve_timer_stop_event = None
        self._solve_timer_thread = None
        if completed:
            self.solve_timer.value = (
                "<span style='color:#166534'><b>Solve completed</b> "
                f"in {elapsed:.1f} s</span>"
            )
        else:
            self.solve_timer.value = (
                "<span style='color:#b91c1c'><b>Solve stopped</b> "
                f"after {elapsed:.1f} s</span>"
            )
        return elapsed

    def _generated_dir(self) -> Path:
        generated_dir = self.project_root / "generated"
        generated_dir.mkdir(parents=True, exist_ok=True)
        return generated_dir

    def _build_recipe_variants(self) -> Any:
        """Build General, enriched-display, and finite planning projections."""
        topology = self.topology.value
        seed = int(self.seed.value)
        ingredient_count = int(self.ingredient_count.value)
        self.general_graph = None
        self.enriched_graph = None

        if topology in {"nonlinear", "parallel", "linear"}:
            self.recipe_spec = generate_random_recipe_spec(
                seed,
                topology="linear",
                ingredient_count=ingredient_count,
            )
            self.recipe_spec["id"] = f"RTN_Random_{topology}_{seed}"
            self.recipe_spec["metadata"] = {
                **self.recipe_spec.get("metadata", {}),
                "topology": topology,
                "generalTopology": "linear",
                "enrichmentTopology": topology,
            }
            self.general_ir = build_recipe_ir(self.recipe_spec)
            self.enriched_spec = auto_enrich_recipe_spec(
                self.recipe_spec,
                parallel_dosing=topology in {"parallel", "nonlinear"},
                usage_alternatives=topology == "nonlinear",
            )
            self.recipe_ir = build_recipe_ir(self.enriched_spec)
            self.enriched_display_ir = self.recipe_ir
            return self.recipe_spec

        factories = {
            "or": or_variant,
            "runtime-loop": runtime_loop_variant,
            "conditional-jump": conditional_jump_variant,
            "combined": combined_runtime_variant,
        }
        factory = factories[topology]
        kwargs = {"ingredient_count": ingredient_count}
        self.general_graph = factory(seed, enriched=False, **kwargs)
        self.enriched_graph = factory(seed, enriched=True, **kwargs)
        self.general_ir = recipe_graph_to_display_ir(self.general_graph)
        self.enriched_display_ir = recipe_graph_to_display_ir(self.enriched_graph)

        jump_decisions = {}
        if topology == "conditional-jump":
            jump_decisions["jump_skip_usage"] = False
        elif topology == "combined":
            jump_decisions["combined_jump_skip_optional_work"] = False
        self.recipe_ir = recipe_graph_to_ir(
            self.enriched_graph,
            jump_decisions=jump_decisions,
        )
        self.recipe_spec = {
            "id": self.general_graph.id,
            "volume": float(self.general_graph.metadata.get("volume", 0.0)),
            "metadata": dict(self.general_graph.metadata),
        }
        self.enriched_spec = None
        return self.general_graph.to_dict()

    def generate(self) -> None:
        """Generate the General and Enriched Recipe graphs from current controls."""
        self._set_busy(True, "Generating recipe and control-flow graphs...")
        self.planner_result = None
        self.validation = None
        self.plan_df = None
        self.optimization_session = None
        self.master_xml_path = ""
        if self._solve_timer_stop_event is not None:
            self._stop_solve_timer(completed=False)
        self.solve_elapsed_s = 0.0
        self.solve_timer.value = (
            "<span style='color:#6b7280'>Solve elapsed: not started</span>"
        )
        self.export_button.disabled = True
        self.export_status.value = (
            "<span style='color:#6b7280'>Solve a recipe before exporting.</span>"
        )
        self.validation_status.value = (
            "<span style='color:#6b7280'>Solve a recipe before validating.</span>"
        )
        self.validation_output.clear_output(wait=True)
        try:
            details_payload = self._build_recipe_variants()
            plant = plant_model_from_rtn_inputs(
                self.module_ops,
                self.module_interfaces,
                self.module_maximum_volume,
                self.module_resources,
                equipment_bindings=self.equipment_bindings,
            )
            generated_dir = self._generated_dir()
            general_xml = generated_dir / f"GeneralRecipe_{self.general_ir.id}.xml"
            if self.general_graph is None:
                self.general_xml_path = save_general_recipe_xml_from_ir(
                    self.general_ir,
                    general_xml,
                )
            else:
                export_general_recipe(
                    self.general_graph,
                    general_xml,
                    mode="annotated",
                )
                self.general_xml_path = str(general_xml)

            with self.recipe_summary_output:
                self.recipe_summary_output.clear_output(wait=True)
                summary = pd.DataFrame(
                    [
                        {
                            "Seed": self.seed.value,
                            "Recipe": self.recipe_spec["id"],
                            "Topology": self.topology.value,
                            "Volume (L)": self.recipe_spec["volume"],
                            "General Nodes": len(self.general_ir.nodes),
                            "Enriched Nodes": len(self.enriched_display_ir.nodes),
                            "Planning Nodes": len(self.recipe_ir.nodes),
                            "Module": len(plant.assets),
                            "Capabilities": sum(len(asset.operations) for asset in plant.assets),
                            "Choice Groups": self.recipe_ir.choice_groups(),
                            "General Recipe XML": self.general_xml_path,
                        }
                    ]
                )
                display_df("Random Recipe Summary", summary)
                display(
                    HTML(
                        "<details><summary><b>Generated General Recipe JSON</b></summary>"
                        f"<pre>{escape(json.dumps(details_payload, indent=2))}</pre></details>"
                    )
                )

            # A single output preserves the required General -> Enriched order.
            with self.control_flow_output:
                self.control_flow_output.clear_output(wait=True)
                display_recipe_graph(
                    self.general_ir,
                    "General Recipe Control Flow",
                )
                display_recipe_graph(
                    self.enriched_display_ir,
                    "Enriched Recipe Control Flow",
                )

            self.solve_output.clear_output(wait=True)
            self._set_busy(False, "Recipe generated. Ready to solve.")
        except Exception as exc:
            self._set_busy(False, f"Generation failed: {type(exc).__name__}")
            with self.recipe_summary_output:
                self.recipe_summary_output.clear_output(wait=True)
                raise

    def export_master_recipe(self) -> str:
        """Export the latest solved plan as a BatchML Master Recipe."""
        if self.optimization_session is None or self.planner_result is None:
            raise RuntimeError("No solved plan is available for export")
        if self.validation is None or not self.validation.valid:
            raise RuntimeError("Only a successfully validated plan can be exported")
        destination = self._generated_dir() / f"MasterRecipe_{self.recipe_ir.id}.xml"
        self.master_xml_path = export_master_recipe(self.optimization_session, destination)
        self.export_status.value = (
            "<span style='color:#166534'><b>Exported:</b> "
            f"{escape(self.master_xml_path)}</span>"
        )
        return self.master_xml_path

    def run_validation(self):
        """Validate the latest solved schedule and render the result in Section 6."""
        if self.planner_result is None or self.rtn_model is None:
            raise RuntimeError("No solved schedule is available for validation")
        if self.planner_config is None:
            raise RuntimeError("No planner configuration is available for validation")

        self.validation_status.value = (
            "<span style='color:#b45309'>Validating latest schedule...</span>"
        )
        self.validation = validate_schedule(
            self.rtn_model,
            self.planner_result,
            module_maximum_volume=self.module_maximum_volume,
            module_resources=self.module_resources,
            config=self.planner_config,
        )
        validation = self.validation
        with self.validation_output:
            self.validation_output.clear_output(wait=True)
            print("Independent Schedule Validation")
            print("Valid                   :", validation.valid)
            print("Errors                  :", len(validation.errors))
            for error in validation.errors:
                print("  -", error)
            print("Warnings                :", len(validation.warnings))
            for warning in validation.warnings:
                print("  -", warning)
            print("\nProfit recomputation:")
            for key, value in validation.profit_check.items():
                print(f"  {key:20s}: {value}")
            profit_match = (
                abs(
                    validation.profit_check["profit"]
                    - self.planner_result.objective_profit
                )
                < 1e-3
            )
            print("\nPlanner objective_profit:", self.planner_result.objective_profit)
            print("Validator profit         :", validation.profit_check["profit"])
            print("Match                    :", profit_match)

        if validation.valid:
            self.validation_status.value = (
                "<span style='color:#166534'><b>Validation passed.</b></span>"
            )
            self.export_status.value = (
                "<span style='color:#6b7280'>Plan ready. Use Export Master Recipe below.</span>"
            )
            self.export_button.disabled = False
        else:
            self.validation_status.value = (
                "<span style='color:#b91c1c'><b>Validation failed.</b></span>"
            )
            self.export_status.value = (
                "<span style='color:#b91c1c'>Validation failed; export is disabled.</span>"
            )
            self.export_button.disabled = True
        return validation

    def generate_and_solve(self) -> None:
        """Run the complete requested notebook pipeline in display order."""
        self._set_busy(True, "Generating recipe...")
        timer_running = False
        try:
            self.generate()
            self._set_busy(True, "Solving RTN model with CP-SAT...")
            settings = RTNSettings(
                solver_time_limit_s=float(self.time_limit.value),
                num_workers=int(self.num_workers.value),
                time_penalty_per_second=abs(float(self.time_penalty_per_second.value)),
                profit_per_litre=float(self.profit_per_litre.value),
                usage_cost_weight=float(self.usage_cost_weight.value),
                energy_cost_weight=float(self.energy_cost_weight.value),
                co2_penalty=float(self.co2_penalty.value),
                electricity_price_eur_per_kwh=float(self.electricity_price.value),
                connect_duration_s=float(self.connect_duration.value),
                disconnect_duration_s=float(self.disconnect_duration.value),
                require_final_disconnect=bool(self.require_final_disconnect.value),
                relative_gap_limit=float(self.relative_gap.value),
                enable_auxiliary_transfers=True,
                allow_process_transfers=True,
                auxiliary_transfer_mode="lazy",
            )
            plant = plant_model_from_rtn_inputs(
                self.module_ops,
                self.module_interfaces,
                self.module_maximum_volume,
                self.module_resources,
                equipment_bindings=self.equipment_bindings,
            )
            self.planner_config = settings.to_planner_config(self.recipe_ir.volume)

            def update_progress(stage, progress, detail):
                self.status.value = (
                    f"<span style='color:#2563eb'><b>{escape(stage.upper())}</b> "
                    f"{progress * 100:.0f}% · {escape(detail)}</span>"
                )

            self._start_solve_timer()
            timer_running = True
            self.optimization_session = optimize_recipe_ir(
                self.recipe_ir,
                plant,
                recipe_path=self.general_xml_path,
                output_dir=self._generated_dir(),
                settings=settings,
                progress_callback=update_progress,
            )
            self.rtn_model = self.optimization_session.pipeline.rtn_model
            self.planner_result = self.optimization_session.pipeline.planner_result
            self._stop_solve_timer()
            timer_running = False

            with self.solve_output:
                self.solve_output.clear_output(wait=True)
                result = self.planner_result
                print("Status            :", result.status)
                print("Selected branches :", result.selected_branches)
                print("Makespan (s)      :", result.makespan_s)
                print("Usage cost        :", round(result.total_weighted_usage_cost, 2))
                print("Energy cost       :", round(result.total_weighted_energy_cost, 2))
                print("CO₂ cost          :", round(result.total_weighted_co2_cost, 2))
                print("Total cost        :", round(result.total_cost, 2))
                print("Revenue objective :", round(result.objective_profit, 2))
                print()

                display_process_plan_graph(
                    self.recipe_ir,
                    result,
                    "CP-SAT Selected Process Plan (Gantt)",
                )

                display_connection_graph(
                    plant,
                    result,
                    "Solved Module Connections",
                )

                plan_df = plan_dataframe(self.optimization_session)
                self.plan_df = plan_df
                display_df("CP-SAT Selected Process Plan", plan_df)

            self.run_validation()

            if self.validation.valid:
                self._set_busy(False, f"Completed: {self.planner_result.status}; validation passed.")
                if self.auto_export_master.value:
                    self.export_master_recipe()
            else:
                self._set_busy(False, f"Completed: {self.planner_result.status}; validation failed.")
        except Exception as exc:
            if timer_running:
                self._stop_solve_timer(completed=False)
            self._set_busy(False, f"Pipeline failed: {type(exc).__name__}")
            with self.solve_output:
                self.solve_output.clear_output(wait=True)
                raise
