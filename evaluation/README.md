# Evaluation Data

Raw results behind the numbers reported in the IECON 2026 paper. Everything here was produced by the code in this repository and can be regenerated with `tests/ModPlant-Benchmarks.ipynb` and `ModPlant-RTN.ipynb`.

## Environment

All runs used an Apple M1 Max (10 cores, 64 GB RAM), Python 3.13.9, and OR-Tools CP-SAT 9.15.6755, with a 0.1 % relative gap tolerance, a 120 s wall-time limit per solve, eight workers, a fixed seed, and lazy auxiliary transfer generation. The exact configuration is stored under `meta` in `paper_experiments.json`.

## Files

| File | Content | Used for |
| --- | --- | --- |
| `paper_experiments.json` | Environment metadata, one row per benchmark combination (plant, recipe, units, tasks, status, makespan, cost, profit, solve time, validator verdict), and the capacity-infeasible case with its solver diagnostic | Benchmark statistics and the infeasible case in Section V |
| `app_demo_operations.json` | The 14 scheduled operations of the reference case on the demo plant, with unit, source, target, ports, start and end times, and material amounts | The Gantt chart of the reference case (Fig. 3) |
| `app_demo_meta.json` | Planner configuration, status, makespan, total cost, net profit, energy and emissions for the same run | The objective figures quoted in Section V-B |
| `a4_r1_operations.json` | The 16 scheduled operations of the A4 x R1 benchmark combination | Cross-check of the benchmark row for A4 x R1 |

## Lazy Auxiliary Transfers, Reproducibility, and Optimality

`paper_experiments.json` carries three additional blocks that characterise the lazy
auxiliary-transfer loop. All three are regenerated with `scripts/lazy_statistics.py`,
which executes the plant and recipe definitions straight out of
`tests/ModPlant-Benchmarks.ipynb` so that it cannot drift away from the benchmark:

```bash
python scripts/lazy_statistics.py stats --repeats 5   # -> lazy_transfer_statistics
python scripts/lazy_statistics.py repro --repeats 5   # -> reproducibility
python scripts/lazy_statistics.py audit --repeats 3   # -> optimality_audit
```

### `lazy_transfer_statistics` — how often the fallback is needed

Over five repetitions of all 14 combinations, the fallback to complete enumeration was
reached on one feasible instance, `C5 x R3_simple`, which is also the instance with the
longest solve time (18.3 s). `B3 x R7_nous` falls back as well before reporting the
expected `INFEASIBLE`. `C5 x R7_nous` fell back in one earlier run out of roughly ten,
so the set of falling-back instances is itself not fixed. The remaining instances are
certified from targeted candidates alone in every repetition.

The block also stores the direct comparison against complete enumeration:

| Instance | Lazy candidates (min–max) | Lazy solve | Enumerated candidates | Enumerated solve |
| --- | --- | --- | --- | --- |
| `A4 x R1_3par` | 78–117 | 2.1–3.4 s | 690 | 26.3–27.8 s |
| `C5 x R3_simple` | 13 (then fallback) | 17.5–18.8 s | 589 | 17.3–19.1 s |

### `reproducibility` — what a re-run actually reproduces

Eight instances were solved five times each and the runs compared at three levels: the
full operation list including generated node identifiers, the physical schedule (unit,
ports, times, amounts) and the routing (the same without concrete ports).

| Level | Identical across repeats |
| --- | --- |
| Outcome (status, profit, makespan, cost, branch selection, operation count) | 7 of 8 |
| Routing (ports ignored) | 5 of 8 |
| Physical schedule (generated identifiers ignored) | 2 of 8 |
| Full operation list | 2 of 8 |

The relaxed solve of the lazy loop stops at its first solution
(`lazy_auxiliary_relaxed_stop_after_first_solution`) and runs on eight workers, so which
solution arrives first depends on thread timing. Different first solutions lead to
different candidate sets and hence to different, usually equally good, final schedules.

### `optimality_audit` — what the lazy loop costs in solution quality

The lazy loop generates candidates from detected inventory violations, so it proves
optimality with respect to the candidates it generated, not with respect to the fully
enumerated model. Solving every instance in lazy mode and once with complete enumeration:

- The reference case of `ModPlant-RTN.ipynb` reaches the enumerated optimum (profit
  600.4, makespan 2283 s, cost 58.1) in five out of five runs at the 0.1 % gap tolerance.
- 11 of the 13 feasible benchmark rows stored in `benchmark` equal the enumerated
  optimum. The rows `A4 x R2_2mix` (-935.5 against -906.5) and `A4 x R7_nous` (-20.5
  against -10.5) come from runs that stayed below it while reporting `OPTIMAL`.
- How often this happens depends on machine load, because the candidate set follows from
  the first solution of the relaxed solve. Over twelve repetitions on an otherwise idle
  machine, `A4 x R2_2mix` stayed below the enumerated optimum three times and
  `A4 x R4_mini` once. Running the same instance directly after another solve raised the
  latter to six out of twelve. All other instances reached the optimum in every run.

Setting `PlannerConfig.lazy_auxiliary_eager_audit = True` re-solves with complete
enumeration after certification and keeps the better objective, at the cost of the
enumerated solve time.

> `app_demo_meta.json` had been produced with the desktop App defaults, whose 2 % gap
> tolerance lets the solver stop above the optimum, and recorded such a run (profit
> 588.9, makespan 2298 s, cost 62.1). It has been regenerated at the 0.1 % gap tolerance
> that Section V states, which yields 600.4 / 2283 s / 58.1 and matches
> `app_demo_operations.json`. That operation list was unaffected and is unchanged.
> At the 0.1 % gap the reference case is stable: ten consecutive runs produced the same
> objective, makespan, cost and port assignment, with solve times between 2.18 s and
> 2.45 s.

## Notes

- The benchmark file contains 14 rows. Thirteen are feasible and are the combinations reported in the paper. The remaining row, B3 x R7, is retained as an explicit infeasibility regression: the composition model must not recover a schedule by selectively draining a single component out of a mixture.
- Cost, duration, energy, and emission coefficients are assumed example values for this study. They can be replaced by measured plant data without changing the model.
- Re-running with the same seed, worker count, and solver version reproduces the objective value, the branch selection, and the validator verdict of these rows, but not necessarily the identical operation list: see the `reproducibility` block above and `optimality_audit` for the two instances where individual lazy runs stay below the enumerated optimum.
- The solve times in `benchmark` come from a single pass of `tests/ModPlant-Benchmarks.ipynb`. Repeated runs vary by a few tenths of a second, and the lazy candidate counts vary as well, so `lazy_transfer_statistics` reports ranges rather than single values.
