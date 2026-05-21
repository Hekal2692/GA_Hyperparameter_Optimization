# GA-Based Task Scheduling Optimization

## Overview

This repo contains a scheduler GA plus an optional outer tuning GA:

- The scheduler GA public entry point is [`SchedulerGa.py`](SchedulerGa.py). It builds a task schedule and returns its makespan.
- [`GAImplementation.py`](GAImplementation.py) still contains the legacy scheduler implementation details used by `SchedulerGa.py`.
- The outer GA in [`OptimizerGa.py`](OptimizerGa.py) tunes the inner GA hyperparameters by minimizing the mean makespan across repeated inner runs.
  The search space can include both the main inner-GA controls (`pop_size`, `cxpb`, `mutpb`, `ngen`) and the inner mutation sub-probabilities for task order, processor allocation, message priority, and message path genes.

Nested-GA runtime settings live in [`ga_config.json`](ga_config.json). A smaller nested smoke-test preset is available in [`ga_config.small_run.json`](ga_config.small_run.json).

Static baseline settings live separately in [`ga_config.static_baseline.json`](ga_config.static_baseline.json). A tiny static smoke-test preset is available in [`ga_config.static_small_run.json`](ga_config.static_small_run.json).

## Run

Run the static, non-adaptive scheduler baseline:

```bash
python SchedulerGa.py --config ga_config.static_baseline.json
```

Run multiple standalone instances in parallel:

```bash
python SchedulerGa.py example_50T.json example_70T.json --config ga_config.static_baseline.json --parallel-workers 2
```

Override the number of standalone benchmark runs explicitly:

```bash
python SchedulerGa.py example_50T.json example_70T.json --config ga_config.static_baseline.json --parallel-workers 2 --benchmark-repeats 3
```

Run standalone benchmarks and compare them generically against existing nested-GA validation runs:

```bash
python SchedulerGa.py example_50T.json example_70T.json --config ga_config.static_baseline.json --parallel-workers 2 --compare-nested-runs logs/example_50T_20260422_172054 logs/example_70T_20260423_120726
```

Run the comparisons later from already-saved standalone results:

```bash
python SchedulerGa.py --compare-standalone-runs logs/example_50T_standalone_scheduler_ga_20260424_081821 logs/example_70T_standalone_scheduler_ga_20260424_081821 --compare-nested-runs logs/example_50T_20260422_172054 logs/example_70T_20260423_120726
```

Use the separate comparison pipeline:

```bash
python CompareGaRuns.py --standalone-runs logs/example_50T_standalone_scheduler_ga_20260424_081821 logs/example_70T_standalone_scheduler_ga_20260424_081821 --nested-runs logs/example_50T_20260422_172054 logs/example_70T_20260423_120726
```

Compare both nested-GA behaviors:

```bash
python CompareGaRuns.py --standalone-runs logs/example_50T_standalone_scheduler_ga_20260424_081821 logs/example_70T_standalone_scheduler_ga_20260424_081821 --nested-runs logs/example_50T_20260422_172054 logs/example_70T_20260423_120726 --nested-source both
```

Run the nested outer GA tuner:

```bash
python OptimizerGa.py
```

Run a specific example instance:

```bash
python OptimizerGa.py example_40T.json
```

Run with a different config preset:

```bash
python OptimizerGa.py --config ga_config.small_run.json
```

Run a fast static smoke test:

```bash
python SchedulerGa.py --config ga_config.static_small_run.json
```

See all available CLI overrides:

```bash
python OptimizerGa.py --help
python SchedulerGa.py --help
```

## What Gets Logged

Each outer-GA run creates a timestamped folder under `logs/` with:

- a text run log
- a plot of the outer GA objective history
- additional analysis plots for repeat stability, hyperparameter trajectories, runtime tradeoffs, population objective spread, and unseen-seed validation
- a `*_history.json` file with generation-level metrics
- a `*_evaluations.jsonl` file with one record per outer-individual evaluation
- a `*_best_result.json` file with the currently selected best configuration and its best inner schedule
- a `*_validation_results.json` file for the top outer candidates rerun on unseen inner-GA seeds

The outer plot and history optimize the mean makespan objective, not just the best single inner-GA run.

Each static scheduler benchmark creates a `*_standalone_scheduler_ga_*` folder under `logs/` with:

- a text benchmark log
- a `*_runs.jsonl` file with one record per standalone benchmark run
- a main `*.png` convergence plot showing generation-best makespan for each benchmark run plus the mean generation-best makespan across runs
- a `*_repeat_summary.png` plot showing the final makespan of each benchmark run
- a `*_results.json` file with fixed hyperparameters, benchmark run makespans, runtimes, per-run generation histories, aggregated generation statistics, best genome, and best schedule

If `--compare-nested-runs` is used, each standalone run folder also gets:

- a generic tuned-vs-standalone comparison JSON
- a comparison plot showing makespan and runtime distributions
- Welch t-test results for tuned-vs-standalone makespan

The separate [`CompareGaRuns.py`](CompareGaRuns.py) pipeline writes all comparison artifacts into a dedicated `ga_comparison_batch_*` folder instead of mixing them into the standalone run folders.

`--nested-source validation` compares against unseen-seed validation behavior, `--nested-source training` compares against the search-time selected best result, and `--nested-source both` writes both sets of comparisons.

The static benchmark does not adapt scheduler hyperparameters. It uses `static_scheduler_ga` for fixed `pop_size`, `cxpb`, `mutpb`, and `ngen`, and it uses the fixed `scheduler.mutation` values from the config.

The standalone benchmark repeats are only there to estimate stochastic performance and stability. They are not a hyperparameter validation phase. Validation only applies to the tuned nested GA, where unseen seeds are used to test whether the selected hyperparameters generalize.

Internally, the scheduler now remaps sparse processor ids from the JSON platform to dense indices for reconstruction bookkeeping. The processor-allocation genome still uses only the real non-router processor ids from the input model.
