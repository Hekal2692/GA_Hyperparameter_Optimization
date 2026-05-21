# GA-Based Task Scheduling Optimization

## Overview

This repo contains a scheduler GA plus an optional outer tuning GA:

- The scheduler GA public entry point is [`SchedulerGa.py`](SchedulerGa.py). It builds a task schedule and returns its makespan.
- [`GAImplementation.py`](GAImplementation.py) still contains the legacy scheduler implementation details used by `SchedulerGa.py`.
- The outer GA in [`OptimizerGa.py`](OptimizerGa.py) tunes the inner GA hyperparameters.
  In this branch, the default experiment is a fair exact-budget search over only `pop_size` and `ngen`, with the objective set to repeated-run mean makespan only.
  The other inner-GA probabilities are fixed per run: `cxpb`, `mutpb`, and the 4 scheduler mutation sub-probabilities.
  The budget constraint is still controlled through `outer_ga.computation_budget`, for example `pop_size * ngen = 2520`.
- [`BudgetPairSweep.py`](BudgetPairSweep.py) provides the cleaner exhaustive alternative for this branch: instead of using an outer GA, it evaluates every feasible fair `(pop_size, ngen)` pair directly for each chosen budget and each fixed probability combination.

Nested-GA runtime settings live in [`ga_config.json`](ga_config.json). A smaller nested smoke-test preset is available in [`ga_config.small_run.json`](ga_config.small_run.json).

Fixed-parameter preset configs are available as:

- [`ga_config.fixed_params_very_low.json`](ga_config.fixed_params_very_low.json): `cxpb=0.5`, `mutpb=0.1`, shared scheduler mutation probability `0.05`
- [`ga_config.fixed_params_low.json`](ga_config.fixed_params_low.json): `cxpb=0.6`, `mutpb=0.15`, shared scheduler mutation probability `0.10`
- [`ga_config.fixed_params_medium.json`](ga_config.fixed_params_medium.json): `cxpb=0.7`, `mutpb=0.2`, shared scheduler mutation probability `0.15`
- [`ga_config.fixed_params_high.json`](ga_config.fixed_params_high.json): `cxpb=0.8`, `mutpb=0.25`, shared scheduler mutation probability `0.20`
- [`ga_config.fixed_params_very_high.json`](ga_config.fixed_params_very_high.json): `cxpb=0.9`, `mutpb=0.3`, shared scheduler mutation probability `0.25`

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

Run a fair budget sweep across multiple exact `pop_size * ngen` costs:

```bash
python BudgetSweep.py example_50T.json example_70T.json --budgets 1800 2100 2400 2520 3360 --outer-seeds 7 17 27
```

Run the fixed-parameter budget experiment with the medium preset:

```bash
python BudgetSweep.py example_50T.json example_70T.json --config ga_config.fixed_params_medium.json --budgets 1800 2100 2400 2520 3360
```

Run the exhaustive no-outer-GA budget-pair sweep across all feasible fair pairs:

```bash
python BudgetPairSweep.py example_50T.json example_70T.json --config ga_config.json --budgets 1800 2100 2400 2520 3360 --cxpb-values 0.5 0.6 0.7 0.8 0.9 --mutpb-values 0.1 0.15 0.2 0.25 0.3 --shared-mutation-values 0.05 0.1 0.15 0.2 0.25 --parallel-workers 2
```

Run a tiny exhaustive smoke test:

```bash
python BudgetPairSweep.py example_30T.json --config ga_config.small_run.json --budgets 1800 --cxpb-values 0.5 --mutpb-values 0.1 --shared-mutation-values 0.05 --benchmark-repeats 1
```

Plot per-instance summaries from the intermediate `*_best_fixed_params_per_pair.csv` output:

```bash
python PlotBestFixedParamsPerPair.py logs/budget_pair_sweep_.../budget_pair_sweep_..._best_fixed_params_per_pair.csv
```

Run the first-step fair generalization check: generate three unseen DAG variants per instance, preserving task runtimes, platform, message count, and message payloads while changing the dependency edges; then evaluate the five budget-wise best configurations learned on the original `example_50T` and `example_70T` instances:

```bash
python RunGeneralizationEvaluation.py --best-by-budget-csv logs/budget_pair_sweep_.../budget_pair_sweep_..._best_fixed_params_per_pair_best_by_budget.csv --dag-config dag_variant_generator/dag_variant_config_fair_topology.json --num-variants 3 --benchmark-repeats 3
```

Generate and validate the unseen DAG variants without running the scheduler benchmarks:

```bash
python RunGeneralizationEvaluation.py --best-by-budget-csv logs/budget_pair_sweep_.../budget_pair_sweep_..._best_fixed_params_per_pair_best_by_budget.csv --dag-config dag_variant_generator/dag_variant_config_fair_topology.json --num-variants 3 --generate-only
```

Run the Trial 2 continuation: compare the tuned Trial 1 results against continuous random same-budget baselines on the same fair DAG variants:

```bash
python RunGeneralizationRandomBaseline.py --tuned-run-results logs/generalization_eval_fair_topology/generalization_run_results.csv --variant-manifest logs/generalization_eval_fair_topology/generalization_variants.csv --random-configs-per-budget 5 --benchmark-repeats 3 --parallel-workers 4
```

Run the same budget sweep with two parallel workers:

```bash
python BudgetSweep.py example_50T.json example_70T.json --budgets 1800 2100 2400 2520 3360 --outer-seeds 7 17 27 --parallel-workers 2
```

Run a small budget-sweep smoke test:

```bash
python BudgetSweep.py example_30T.json --config ga_config.small_run.json --budgets 1800 2520 --outer-seeds 7
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
- a plot of the outer GA objective history, with the main fitness panel showing best-so-far objective-score progress
- additional analysis plots for repeat stability, objective components, hyperparameter trajectories, runtime tradeoffs, population objective spread, hyperparameter diversity, generation novelty, population origin, best-individual survival, and unseen-seed validation
- a `*_history.json` file with generation-level metrics written on the final archive-wide objective-normalization basis
- a `*_evaluations.jsonl` file with one record per outer-individual evaluation
- a `*_best_result.json` file with the currently selected best configuration and its best inner schedule
- a `*_validation_results.json` file for the top outer candidates rerun on unseen inner-GA seeds

The outer plot and history optimize the configured outer objective. In this branch, the default objective is mean makespan only, while runtime is still logged for analysis.

Each [`BudgetSweep.py`](BudgetSweep.py) batch creates a `budget_sweep_*` folder under `logs/` with:

- a `runs/` subfolder containing the normal nested-GA run folders for each `(instance, budget, outer seed)` combination
- a batch `*_summary.json` file collecting run-level and aggregated budget-level metrics
- plots for selected-candidate validation makespan, selected-candidate validation runtime, best validated makespan, and cross-instance normalized selected validation makespan versus budget

Each [`BudgetPairSweep.py`](BudgetPairSweep.py) batch creates a `budget_pair_sweep_*` folder under `logs/` with:

- a `runs/` subfolder containing one standalone scheduler benchmark folder per `(instance, budget, feasible pair, fixed-parameter combination)`
- a batch `*_summary.json` file collecting all pair-level results
- a `*_pair_results.csv` file with one row per directly evaluated `(pop_size, ngen)` pair
- a `*_best_fixed_params_per_pair.csv` file with the best fixed hyperparameters per `(instance, budget, feasible pair)`
- a `*_best_pair_summary.csv` file with the best feasible pair per `(instance, budget, fixed-parameter combination)`
- plots for best feasible-pair makespan and runtime versus budget

Each [`RunGeneralizationEvaluation.py`](RunGeneralizationEvaluation.py) batch creates a `generalization_eval_*` folder under `logs/` with:

- a `variants/` subfolder containing generated unseen DAG variants for each original instance
- a `validation_logs/` subfolder containing validator output for each generated DAG
- a `runs/` subfolder containing one standalone scheduler benchmark folder per `(original instance, generated variant, budget-wise best configuration)`
- a `generalization_variants.csv` manifest listing the generated DAG files
- a `generalization_validation.csv` file recording validation status for each generated DAG
- a `generalization_run_results.csv` file with one row per evaluated variant/configuration
- a `generalization_summary.csv` file comparing each budget-wise best configuration against its unseen-DAG makespan and runtime behavior
- a `generalization_summary.json` file with the same records plus run metadata

Each [`RunGeneralizationRandomBaseline.py`](RunGeneralizationRandomBaseline.py) batch creates a `generalization_eval_fair_topology_random_baseline_*` folder under `logs/` with:

- a `random_configurations.csv` file listing the sampled continuous random same-budget settings
- a `random_baseline_run_results.csv` file with one row per `(original instance, generated variant, budget, random configuration)`
- a `random_baseline_by_config_summary.csv` file aggregating each random configuration across DAG variants
- a `tuned_vs_random_by_variant_comparison.csv` file comparing tuned versus random separately for each `(original instance, generated variant, budget)`
- a `tuned_vs_random_comparison.csv` file comparing the tuned Trial 1 result against the random baseline distribution per `(instance, budget)`
- a `random_baseline_summary.json` file with the same records plus run metadata

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
