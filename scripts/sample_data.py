"""Shared sample recipe + Module plant configuration for the demo scripts.

The four-Module capability data comes from the desktop App's canonical demo
plant.  Notebook callers receive ordinary dictionaries/lists and may edit them
directly before creating the shared App ``PlantModel``.
"""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

_RTN_ROOT = Path(__file__).resolve().parents[1]
if str(_RTN_ROOT) not in sys.path:
    sys.path.insert(0, str(_RTN_ROOT))

from modplant_rtn.models import default_plant_model


def sample_module_ops() -> Dict[str, List[Tuple]]:
    """Return six-field App capabilities, including energy/CO2 and duration."""
    operations, _, _, _ = default_plant_model().to_rtn_inputs()
    return operations


def sample_module_interfaces() -> Dict[str, List[Tuple[str, str]]]:
    """Per-Module input/output port layout."""
    _, interfaces, _, _ = default_plant_model().to_rtn_inputs()
    return interfaces


def sample_hc_data() -> List[Tuple[str, int, int]]:
    """Editable compact port counts used by both notebooks."""
    return [
        (
            asset.asset_id,
            sum(port.port_type == "Input" for port in asset.interfaces),
            sum(port.port_type == "Output" for port in asset.interfaces),
        )
        for asset in default_plant_model().assets
    ]


def sample_module_maximum_volume() -> Dict[str, List[float]]:
    _, _, maximum_volumes, _ = default_plant_model().to_rtn_inputs()
    return maximum_volumes


def sample_module_resources() -> Dict[str, List[Any]]:
    """Initial inventory: each Module holds [material, quantity_L]."""
    _, _, _, resources = default_plant_model().to_rtn_inputs()
    return resources


def sample_equipment_bindings() -> Dict[str, Dict[str, Any]]:
    """OPC UA endpoints, namespaces and operation methods from the App demo."""
    return default_plant_model().equipment_bindings()


def sample_recipe_spec() -> Dict[str, Any]:
    """A recipe with parallel dosing, mix, XOR-chosen usage, settling, separation."""
    return {
        "id": "Module_Parallel_Choice_Example",
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
            {"mix": {"rpm": 150, "duration_s": 30}},
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


def sample_legacy_order() -> Dict[str, Any]:
    """Old-style linear order dict; demonstrates legacy compatibility."""
    return {
        "id": "Legacy_Order_With_Parallel_Dosing",
        "volume": 6.0,
        "order": ["A", "B", "C", {"mix": {"rpm": 150, "duration": 30}}],
        "ratio": {"A": [1], "B": [2], "C": [3]},
        "usage_and_settling": [3600, 300],
        "separation_order": ["C", "B", "A"],
    }


def sample_planner_config_kwargs() -> Dict[str, Any]:
    """Desktop-App-equivalent PlannerConfig defaults for the sample recipe."""
    spec = sample_recipe_spec()
    return {
        "base_profit": 300 * spec["volume"],
        "lambda_per_second": -0.5,
        "usage_cost_weight": 1.0,
        "energy_cost_weight": 1.0,
        "co2_penalty": 1.0,
        "electricity_price_eur_per_kwh": 0.3,
        "connect_duration_s": 3.0,
        "disconnect_duration_s": 2.0,
        "require_final_disconnect": False,
        "enable_auxiliary_transfers": True,
        "allow_process_transfers": True,
        "auxiliary_transfer_mode": "lazy",
        "lazy_auxiliary_eager_audit": False,
        "relative_gap_limit": 0.02,
        "solver_time_limit_s": 180,
        "num_workers": 8,
        "time_scale": 1,
        "cost_scale": 100_000,
    }
