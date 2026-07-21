"""Lazy auxiliary-transfer statistics and reproducibility check.

Regenerates the two evidence blocks stored in ``evaluation/paper_experiments.json``:

``stats``
    How often the lazy auxiliary-transfer loop certifies a schedule on its own and
    how often it falls back to complete enumeration, plus the candidate counts and
    solve times of the lazy and the eager mode on the same instance.

``repro``
    Whether identical inputs reproduce identical results. Every instance is solved
    several times and the runs are compared at three levels: the full operation list
    including generated node identifiers, the physical schedule (unit assignment,
    ports, times, amounts) and the routing (the same without concrete ports).

``audit``
    What the lazy loop costs in solution quality. Every instance is solved several
    times in lazy mode and once with complete enumeration, and the objective values
    are compared. The lazy loop proves optimality with respect to the candidates it
    generated, so a lazy result can be below the optimum of the fully enumerated
    model while still being reported as OPTIMAL.

Plant configurations and recipes are executed straight out of
``tests/ModPlant-Benchmarks.ipynb`` so that this script cannot drift away from the
benchmark it measures.

Usage::

    python scripts/lazy_statistics.py stats [--repeats 3]
    python scripts/lazy_statistics.py repro [--repeats 3]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = REPO_ROOT / "tests" / "ModPlant-Benchmarks.ipynb"

sys.path.insert(0, str(REPO_ROOT / "scripts"))

import notebook_helpers as rtn_nb  # noqa: E402

build_recipe_ir = rtn_nb.build_recipe_ir
auto_enrich_recipe_spec = rtn_nb.auto_enrich_recipe_spec
recipe_ir_to_rtn = rtn_nb.recipe_ir_to_rtn
PlannerConfig = rtn_nb.PlannerConfig
solve_rtn_with_cp_sat = rtn_nb.solve_rtn_with_cp_sat
validate_schedule = rtn_nb.validate_schedule

# The 14 benchmark combinations of the notebook: thirteen feasible ones plus the
# B3 x R7 infeasibility regression.
CASE_KEYS: List[Tuple[str, str]] = [
    ("A4", "R1"), ("A4", "R2"), ("A4", "R3"), ("A4", "R4"), ("A4", "R5"), ("A4", "R7"),
    ("B3", "R3"), ("B3", "R4"), ("B3", "R7"), ("B3", "R8"),
    ("C5", "R3"), ("C5", "R4"), ("C5", "R7"), ("E3", "R4"),
]
# Subset used for the reproducibility check: the largest models, the fallback case and
# two small ones.
REPRO_KEYS: List[Tuple[str, str]] = [
    ("A4", "R1"), ("A4", "R2"), ("A4", "R3"), ("A4", "R5"),
    ("B3", "R3"), ("C5", "R3"), ("C5", "R7"), ("E3", "R4"),
]

FULL_FIELDS = ("step_id", "recipe_node_id", "branch_group_id", "branch_id", "operation_type",
               "operation", "module", "source_module", "target_module", "out_port", "in_port",
               "start_s", "end_s", "duration_s", "total_cost", "transfer_kind")
PHYSICAL_FIELDS = ("operation_type", "module", "source_module", "target_module", "out_port",
                   "in_port", "start_s", "end_s", "duration_s", "total_cost", "transfer_kind")
ROUTING_FIELDS = ("operation_type", "module", "source_module", "target_module", "start_s",
                  "end_s", "duration_s", "total_cost", "transfer_kind")


def load_notebook_definitions() -> Dict[str, Any]:
    """Execute the definition cells of the benchmark notebook and return their namespace."""
    cells = json.loads(NOTEBOOK.read_text())["cells"]
    namespace: Dict[str, Any] = {}
    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if "run_benchmark(" in source or source.lstrip().startswith("%") or "import " in source:
            continue
        exec(compile(source, "<benchmark-notebook>", "exec"), namespace)  # noqa: S102
    return namespace


NS = load_notebook_definitions()
CONFIGS = {name: (NS[f"module_ops_{name}"], NS[f"ifaces_{name}"],
                  NS[f"maxv_{name}"], NS[f"res_{name}"]) for name in ("A4", "B3", "C5", "E3")}
RECIPES = {name: NS[name] for name in ("R1", "R2", "R3", "R4", "R5", "R7", "R8")}


def solve(config_name: str, recipe_name: str, mode: str = "lazy"):
    module_ops, interfaces, max_volume, resources = CONFIGS[config_name]
    recipe = RECIPES[recipe_name]
    recipe_ir = build_recipe_ir(auto_enrich_recipe_spec(recipe))
    rtn_model = recipe_ir_to_rtn(recipe_ir, module_ops=module_ops, module_interfaces=interfaces,
                                 module_maximum_volume=max_volume, module_resources=resources)
    config = PlannerConfig(base_profit=50 * recipe.get("volume", 1.0), lambda_per_second=-0.5,
                           enable_auxiliary_transfers=True, auxiliary_transfer_mode=mode,
                           relative_gap_limit=0.001, solver_time_limit_s=120, num_workers=8)
    started = time.time()
    result = solve_rtn_with_cp_sat(rtn_model, config)
    elapsed = time.time() - started
    validation = validate_schedule(rtn_model, result, module_maximum_volume=max_volume,
                                   module_resources=resources, config=config)
    return result, validation, elapsed, rtn_model


def signature(result, fields) -> str:
    rows = []
    for operation in result.operations:
        row = {field: getattr(operation, field) for field in fields}
        row["material"] = {k: round(v, 6) for k, v in sorted(operation.material.items())}
        rows.append(row)
    rows.sort(key=lambda row: json.dumps(row, sort_keys=True, default=str))
    return hashlib.sha256(json.dumps(rows, sort_keys=True, default=str).encode()).hexdigest()[:12]


def run_stats(repeats: int) -> Dict[str, Any]:
    per_case: Dict[str, Any] = {}
    for config_name, recipe_name in CASE_KEYS:
        key = f"{config_name} x {RECIPES[recipe_name]['id']}"
        iterations, candidates, times, fallbacks, reasons, statuses = [], [], [], [], set(), set()
        for _ in range(repeats):
            result, validation, elapsed, _ = solve(config_name, recipe_name)
            diagnostics = result.diagnostics or {}
            iterations.append(diagnostics.get("lazy_iterations"))
            candidates.append(diagnostics.get("lazy_candidate_count"))
            fallbacks.append(bool(diagnostics.get("lazy_fallback_used")))
            times.append(round(elapsed, 2))
            statuses.add(result.status)
            if diagnostics.get("lazy_fallback_reason"):
                reasons.add(diagnostics["lazy_fallback_reason"])
            assert validation.valid or result.status not in {"OPTIMAL", "FEASIBLE"}
        per_case[key] = {
            "status": sorted(statuses),
            "fallback_used": sorted({*fallbacks}),
            "fallback_reason": sorted(reasons),
            "lazy_iterations": {"min": min(iterations), "max": max(iterations)},
            "lazy_candidate_count": {"min": min(candidates), "max": max(candidates),
                                     "median": statistics.median(candidates)},
            "solve_time_s": {"min": min(times), "max": max(times)},
        }
        print(f"{key:22s} fallback={sorted({*fallbacks})} "
              f"candidates={min(candidates)}-{max(candidates)} "
              f"time={min(times)}-{max(times)}s")

    reference: Dict[str, Any] = {}
    for config_name, recipe_name in (("A4", "R1"), ("C5", "R3")):
        key = f"{config_name} x {RECIPES[recipe_name]['id']}"
        eager_candidates, eager_times = [], []
        for _ in range(repeats):
            result, _, elapsed, _ = solve(config_name, recipe_name, mode="eager")
            eager_candidates.append((result.diagnostics or {}).get("auxiliary_candidate_count"))
            eager_times.append(round(elapsed, 2))
        reference[key] = {
            "eager_candidate_count": sorted({*eager_candidates}),
            "eager_solve_time_s": {"min": min(eager_times), "max": max(eager_times)},
            "lazy_candidate_count": per_case[key]["lazy_candidate_count"],
            "lazy_solve_time_s": per_case[key]["solve_time_s"],
        }
        print(f"{key:22s} eager candidates={sorted({*eager_candidates})} "
              f"time={min(eager_times)}-{max(eager_times)}s")

    feasible = {k: v for k, v in per_case.items() if v["status"] == ["OPTIMAL"]}
    fell_back = [k for k, v in feasible.items() if True in v["fallback_used"]]
    return {
        "repeats_per_instance": repeats,
        "feasible_instances": len(feasible),
        "instances_falling_back_to_complete_enumeration": fell_back,
        "per_case": per_case,
        "lazy_vs_complete_enumeration": reference,
    }


def run_repro(repeats: int) -> Dict[str, Any]:
    per_case: Dict[str, Any] = {}
    for config_name, recipe_name in REPRO_KEYS:
        key = f"{config_name} x {RECIPES[recipe_name]['id']}"
        runs = []
        for _ in range(repeats):
            result, _, _, _ = solve(config_name, recipe_name)
            runs.append({
                "full": signature(result, FULL_FIELDS),
                "physical": signature(result, PHYSICAL_FIELDS),
                "routing": signature(result, ROUTING_FIELDS),
                "status": result.status,
                "objective_profit": round(result.objective_profit, 6),
                "makespan_s": round(result.makespan_s, 6),
                "total_cost": round(result.total_cost, 6),
                "branches": json.dumps(dict(sorted(result.selected_branches.items()))),
                "n_operations": len(result.operations),
            })
        levels = {level: len({run[level] for run in runs}) == 1
                  for level in ("full", "physical", "routing")}
        variants = sorted({(r["status"], r["objective_profit"], r["makespan_s"], r["total_cost"],
                            r["branches"], r["n_operations"]) for r in runs})
        outcome = len(variants) == 1
        per_case[key] = {
            "identical": levels,
            "identical_outcome": outcome,
            "outcome_variants": [
                {"status": v[0], "objective_profit": v[1], "makespan_s": v[2],
                 "total_cost": v[3], "branches": v[4], "n_operations": v[5]} for v in variants
            ],
        }
        print(f"{key:22s} full={levels['full']!s:5s} physical={levels['physical']!s:5s} "
              f"routing={levels['routing']!s:5s} outcome={outcome}")
    return {
        "repeats_per_instance": repeats,
        "instances": len(per_case),
        "identical_outcome": sum(v["identical_outcome"] for v in per_case.values()),
        "identical_full_schedule": sum(v["identical"]["full"] for v in per_case.values()),
        "identical_ignoring_generated_ids": sum(v["identical"]["physical"] for v in per_case.values()),
        "identical_ignoring_ports": sum(v["identical"]["routing"] for v in per_case.values()),
        "per_case": per_case,
    }


def run_audit(repeats: int) -> Dict[str, Any]:
    per_case: Dict[str, Any] = {}
    for config_name, recipe_name in CASE_KEYS:
        key = f"{config_name} x {RECIPES[recipe_name]['id']}"
        lazy_profits, statuses = [], set()
        for _ in range(repeats):
            result, _, _, _ = solve(config_name, recipe_name)
            statuses.add(result.status)
            lazy_profits.append(round(result.objective_profit, 6))
        eager, _, eager_time, _ = solve(config_name, recipe_name, mode="eager")
        eager_profit = round(eager.objective_profit, 6)
        feasible = eager.status in {"OPTIMAL", "FEASIBLE"} and statuses <= {"OPTIMAL", "FEASIBLE"}
        deviation = (round((eager_profit - max(lazy_profits)) / max(1e-9, abs(eager_profit)) * 100, 3)
                     if feasible else None)
        per_case[key] = {
            "lazy_status": sorted(statuses),
            "lazy_objective_profit": sorted({*lazy_profits}),
            "eager_status": eager.status,
            "eager_objective_profit": eager_profit,
            "eager_solve_time_s": round(eager_time, 2),
            "best_lazy_below_enumeration_percent": deviation,
        }
        print(f"{key:22s} lazy={sorted({*lazy_profits})} eager={eager_profit} "
              f"deviation={deviation}%")
    below = [k for k, v in per_case.items()
             if v["best_lazy_below_enumeration_percent"] not in (None, 0.0)
             and v["best_lazy_below_enumeration_percent"] > 0.1]
    return {
        "repeats_per_instance": repeats,
        "instances_where_enumeration_beats_lazy": below,
        "per_case": per_case,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("stats", "repro", "audit"))
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    runner = {"stats": run_stats, "repro": run_repro, "audit": run_audit}[args.mode]
    print(json.dumps(runner(args.repeats), indent=1))


if __name__ == "__main__":
    main()
