# ModPlant-RTN

A resource-task network framework for modeling and scheduling complex recipes in modular plants.

## Paper Implementation

This repository contains the reference implementation and the evaluation data for:

> B. Chen, T. Miny, and T. Kleinert, "A Resource-Task Network Framework for Modeling and Scheduling Complex Recipes in Modular Plants," in *Proc. IECON 2026 — 52nd Annual Conference of the IEEE Industrial Electronics Society*, Doha, Qatar, 2026.

The framework normalizes a branched batch recipe into a finite task graph, maps it together with a machine-readable plant description onto a resource-task network (RTN), and decides branch selection, unit assignment, material routing, and timing in a single constraint-programming solve. Every schedule is re-checked by an independent validator and can be exported as an equipment-assigned master recipe in BatchML.

## What This Repository Contains

- **Recipe normalization and enrichment** — parallel regions, exclusive and inclusive alternatives, and bounded loops are unrolled into a finite acyclic task graph. Missing runtime decisions fail closed instead of being silently assumed.
- **RTN mapping** — units and ports become unary resources, materials and recipe milestones become state resources, and every recipe operation becomes a task with signed coefficients.
- **CP-SAT planner** — one model decides branch selection, unit assignment, unified-flow routing, staged inventory balance, purity, and connection actions, maximizing a profit objective that prices makespan, operation cost, energy, and emissions.
- **Lazy auxiliary transfers** — transfer candidates are generated only from detected inventory violations, with complete enumeration as fallback.
- **Independent schedule validator** — five event-simulation checks (branch consistency, precedence, resource overlap, material balance, objective re-computation) that reuse no planner code.
- **BatchML import and export** — ISA-88 general recipes in, equipment-assigned master recipes with OPC UA method bindings out.
- **AAS plant descriptions** — plant capabilities, ports, capacities, and inventories are read from Asset Administration Shell files.

## Repository Layout

```
ModPlant-RTN.ipynb           end-to-end pipeline: recipe -> RTN -> schedule -> validation -> export
ModPlant-RTN-UI.ipynb        interactive notebook front end for the same pipeline
modplant_rtn/                computational kernel shared with the desktop application
  models.py                  plant model, capabilities, ports, inventories
  aas.py                     AAS/AASX import and export
  service.py                 planning service used by the notebooks
  visualization.py           SFC, Gantt, and connection-graph rendering
scripts/                     planner core
  rtn.py                     RTN data model (Pantelides formalism)
  recipe_ir.py               recipe intermediate representation and auto-enrichment
  recipe_to_rtn.py           recipe graph -> RTN mapping
  cp_sat_planner.py          CP-SAT model: variables, constraint families, objective
  schedule_validator.py      solver-independent schedule validation
  pipeline.py                orchestration of the full run
  isa88_recipe.py            BatchML general and master recipe I/O
  notebook_helpers.py        shared notebook kernel
shared/modplant_recipe/      BatchML recipe graph package with normalization and validation
tests/                       91 unit tests and the benchmark notebook
  ModPlant-Benchmarks.ipynb  thirteen recipe-plant combinations reported in the paper
evaluation/                  raw results behind the numbers reported in the paper
```

## Requirements

- Python 3.10 or newer (tested on 3.13)
- Google OR-Tools CP-SAT 9.15 or newer
- See `requirements.txt` for the full list

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run The Notebook

```bash
jupyter notebook ModPlant-RTN.ipynb
```

The notebook loads the four-module demo plant (HC10–HC40), builds the `Module_Parallel_Choice_Example` recipe, solves it, validates the result, and writes the general and master recipe XML files to `generated/`. It also renders the selected schedule as a Gantt chart and the resulting inter-module connection configuration as a connection graph. `ModPlant-RTN-UI.ipynb` offers the same pipeline behind an interactive form.

To reproduce the benchmark table of the paper:

```bash
jupyter notebook tests/ModPlant-Benchmarks.ipynb
```

## Expected Runtime Behavior

On an Apple M1 Max (10 cores, 64 GB RAM) with OR-Tools CP-SAT 9.15, a 0.1 % relative gap tolerance, a 120 s wall-time limit, eight workers, and a fixed seed:

| Case | Result |
| --- | --- |
| Reference case (parallel dosing + fast/standard choice) | proven optimal in 2.6 s, makespan 2283 s |
| Thirteen benchmark combinations | all proven optimal, median 0.5 s, maximum 18.3 s |
| Capacity-infeasible order | rejected in 0.01 s with a diagnostic |

Solving is deterministic: identical inputs reproduce identical schedules.

## Tests

```bash
python -m pytest tests/ -q
```

All 91 tests cover the recipe intermediate representation, the RTN mapping, the CP-SAT model, the validator, runtime control structures (loops, jumps, inclusive choices), and the BatchML round trip.

## Scope and Limitations

- The planner operates offline on one batch order for one plant configuration at a time. Multi-batch campaign scheduling is future work.
- Cost, duration, energy, and emission coefficients in the demo data are assumed example values. They can be replaced by measured plant data without changing the model.
- The exported master recipe demonstrates equipment assignment at the recipe level. Closed-loop execution on an orchestration environment is outside the scope of this repository.
- The evaluation is a proof of concept on laboratory-scale configurations of three to five units.

## License & Acknowledgments

Released under the MIT License, see `LICENSE`.

Developed at the [Chair of Information and Automation Systems (IAT)](https://www.iat.rwth-aachen.de), RWTH Aachen University. The Honeycomb (HC) modular laboratory plant family used in the evaluation is described [here](https://www.iat.rwth-aachen.de/cms/iat/forschung/anlagen/~zrnbp/iat-modulab/?lidx=1).
