"""Compare tuned generalization results against same-budget random settings."""

import argparse
import copy
import csv
import json
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev

from OptimizerGa import build_feasible_budget_pairs
from RunGeneralizationEvaluation import (
    build_benchmark_config,
    format_float,
    format_percent,
    load_csv_rows,
    write_csv,
    write_json,
)
from SchedulerGa import run_static_scheduler_benchmark
from ga_config import DEFAULT_CONFIG_PATH, load_config


def grouped_rows(rows, key_fields):
    grouped = {}
    for row in rows:
        key = tuple(row[field] for field in key_fields)
        grouped.setdefault(key, []).append(row)
    return grouped


def feasible_pairs_for_budget(base_config, budget):
    config = copy.deepcopy(base_config)
    config.setdefault("outer_ga", {}).setdefault("computation_budget", {})
    config["outer_ga"]["computation_budget"]["mode"] = "exact_product"
    config["outer_ga"]["computation_budget"]["budget"] = int(budget)
    return build_feasible_budget_pairs(config)


def collect_tuned_summary(tuned_rows):
    tuned_summary = {}
    for key, rows in grouped_rows(tuned_rows, ["original_instance_name", "budget"]).items():
        instance_name, budget = key
        variant_means = [float(row["variant_mean_makespan"]) for row in rows]
        runtimes = [float(row["variant_avg_runtime_seconds"]) for row in rows]
        training_mean = float(rows[0]["training_mean_makespan"])
        tuned_mean = mean(variant_means)
        tuned_summary[key] = {
            "original_instance_name": instance_name,
            "budget": int(budget),
            "variant_count": len(rows),
            "training_mean_makespan": training_mean,
            "tuned_pop_size": int(rows[0]["pop_size"]),
            "tuned_ngen": int(rows[0]["ngen"]),
            "tuned_cxpb": float(rows[0]["cxpb"]),
            "tuned_mutpb": float(rows[0]["mutpb"]),
            "tuned_shared_mutation": float(rows[0]["shared_mutation"]),
            "tuned_mean_makespan": tuned_mean,
            "tuned_std_across_dags": pstdev(variant_means),
            "tuned_avg_runtime_seconds": mean(runtimes),
        }
    return tuned_summary


def sample_random_configurations(
    *,
    tuned_summary,
    base_config,
    random_configs_per_budget,
    cxpb_range,
    mutpb_range,
    shared_mutation_range,
    seed,
):
    rng = random.Random(seed)
    rows = []
    for instance_name, budget in sorted(
        tuned_summary,
        key=lambda key: (key[0], int(key[1])),
    ):
        feasible_pairs = feasible_pairs_for_budget(base_config, budget)
        for index in range(1, random_configs_per_budget + 1):
            pop_size, ngen = rng.choice(feasible_pairs)
            rows.append(
                {
                    "original_instance_name": instance_name,
                    "budget": int(budget),
                    "random_config_id": f"random_{index:03d}",
                    "pop_size": int(pop_size),
                    "ngen": int(ngen),
                    "cxpb": format_float(rng.uniform(*cxpb_range), 6),
                    "mutpb": format_float(rng.uniform(*mutpb_range), 6),
                    "shared_mutation": format_float(
                        rng.uniform(*shared_mutation_range), 6
                    ),
                    "feasible_pair_count": len(feasible_pairs),
                    "random_config_seed": int(seed),
                }
            )
    return rows


def build_random_tasks(
    *,
    random_config_rows,
    variant_rows,
    tuned_summary,
    output_dir,
    config_path,
    benchmark_repeats,
    benchmark_seed_base,
):
    variants_by_instance = grouped_rows(variant_rows, ["original_instance_name"])
    tasks = []
    task_index = 0
    for config_row in random_config_rows:
        instance_name = config_row["original_instance_name"]
        budget = int(config_row["budget"])
        tuned_key = (instance_name, str(budget))
        if tuned_key not in tuned_summary:
            tuned_key = (instance_name, budget)
        training_mean = tuned_summary[tuned_key]["training_mean_makespan"]
        for variant_row in variants_by_instance[(instance_name,)]:
            task_index += 1
            variant_path = Path(variant_row["variant_path"])
            task_log_dir = (
                output_dir
                / "runs"
                / instance_name
                / config_row["random_config_id"]
                / variant_path.stem
                / f"budget_{budget}"
            )
            tasks.append(
                {
                    "task_index": task_index,
                    "config_path": str(Path(config_path).resolve()),
                    "task_log_dir": str(task_log_dir.resolve()),
                    "original_instance_name": instance_name,
                    "variant_name": variant_path.stem,
                    "variant_path": str(variant_path.resolve()),
                    "budget": budget,
                    "random_config_id": config_row["random_config_id"],
                    "pop_size": int(config_row["pop_size"]),
                    "ngen": int(config_row["ngen"]),
                    "cxpb": float(config_row["cxpb"]),
                    "mutpb": float(config_row["mutpb"]),
                    "shared_mutation": float(config_row["shared_mutation"]),
                    "training_mean_makespan": float(training_mean),
                    "benchmark_repeats": int(benchmark_repeats),
                    "random_seed": int(benchmark_seed_base + task_index),
                }
            )
    return tasks


def build_random_run_record(task, result_payload):
    training_mean = float(task["training_mean_makespan"])
    variant_mean = float(result_payload["mean_makespan"])
    variant_std = float(result_payload["std_makespan"])
    gap = variant_mean - training_mean
    return {
        "baseline_type": "random_continuous_same_budget",
        "original_instance_name": task["original_instance_name"],
        "variant_name": task["variant_name"],
        "variant_path": task["variant_path"],
        "budget": int(task["budget"]),
        "random_config_id": task["random_config_id"],
        "pop_size": int(task["pop_size"]),
        "ngen": int(task["ngen"]),
        "cxpb": format_float(task["cxpb"], 6),
        "mutpb": format_float(task["mutpb"], 6),
        "shared_mutation": format_float(task["shared_mutation"], 6),
        "training_mean_makespan": format_float(training_mean, 3),
        "variant_mean_makespan": format_float(variant_mean, 3),
        "generalization_gap": format_float(gap, 3),
        "generalization_gap_percent_of_training": format_percent(gap, training_mean, 3),
        "variant_std_makespan": format_float(variant_std, 3),
        "variant_std_makespan_percent_of_training": format_percent(
            variant_std,
            training_mean,
            3,
        ),
        "variant_best_makespan": format_float(result_payload["best_makespan"], 3),
        "variant_avg_runtime_seconds": format_float(
            result_payload["avg_runtime_seconds"],
            3,
        ),
        "variant_std_runtime_seconds": format_float(
            result_payload["std_runtime_seconds"],
            3,
        ),
        "variant_total_runtime_seconds": format_float(
            result_payload["total_runtime_seconds"],
            3,
        ),
        "benchmark_repeats": int(result_payload["benchmark_repeats"]),
        "random_seed": result_payload["random_seed"],
        "run_dir": result_payload["run_dir"],
        "results_path": result_payload["results_path"],
    }


def run_random_task(task):
    base_config = load_config(Path(task["config_path"]))
    config = build_benchmark_config(
        base_config,
        task["task_log_dir"],
        task["shared_mutation"],
    )
    result_payload = run_static_scheduler_benchmark(
        input_json_path=Path(task["variant_path"]),
        config=config,
        pop_size=int(task["pop_size"]),
        cxpb=float(task["cxpb"]),
        mutpb=float(task["mutpb"]),
        ngen=int(task["ngen"]),
        benchmark_repeats=int(task["benchmark_repeats"]),
        random_seed=int(task["random_seed"]),
        show_plot=False,
        console_summary=False,
    )
    return build_random_run_record(task, result_payload)


def summarize_random_configs(random_run_rows):
    summary_rows = []
    group_fields = ["original_instance_name", "budget", "random_config_id"]
    for key, rows in sorted(grouped_rows(random_run_rows, group_fields).items()):
        instance_name, budget, random_config_id = key
        variant_means = [float(row["variant_mean_makespan"]) for row in rows]
        runtimes = [float(row["variant_avg_runtime_seconds"]) for row in rows]
        training_mean = float(rows[0]["training_mean_makespan"])
        mean_makespan = mean(variant_means)
        gap = mean_makespan - training_mean
        summary_rows.append(
            {
                "original_instance_name": instance_name,
                "budget": int(budget),
                "random_config_id": random_config_id,
                "variant_count": len(rows),
                "pop_size": int(rows[0]["pop_size"]),
                "ngen": int(rows[0]["ngen"]),
                "cxpb": rows[0]["cxpb"],
                "mutpb": rows[0]["mutpb"],
                "shared_mutation": rows[0]["shared_mutation"],
                "training_mean_makespan": format_float(training_mean, 3),
                "random_mean_makespan_across_dags": format_float(mean_makespan, 3),
                "random_std_makespan_across_dags": format_float(
                    pstdev(variant_means),
                    3,
                ),
                "random_generalization_gap": format_float(gap, 3),
                "random_generalization_gap_percent_of_training": format_percent(
                    gap,
                    training_mean,
                    3,
                ),
                "random_avg_runtime_seconds": format_float(mean(runtimes), 3),
                "random_worst_runtime_seconds": format_float(max(runtimes), 3),
            }
        )
    return summary_rows


def compare_tuned_vs_random(tuned_summary, random_config_summary_rows):
    comparison_rows = []
    random_by_budget = grouped_rows(
        random_config_summary_rows,
        ["original_instance_name", "budget"],
    )
    for key, tuned in sorted(tuned_summary.items(), key=lambda item: (item[0][0], int(item[0][1]))):
        instance_name, budget = key
        random_rows = random_by_budget.get((instance_name, str(budget)))
        if random_rows is None:
            random_rows = random_by_budget.get((instance_name, int(budget)), [])
        random_means = [
            float(row["random_mean_makespan_across_dags"]) for row in random_rows
        ]
        random_runtimes = [float(row["random_avg_runtime_seconds"]) for row in random_rows]
        tuned_mean = float(tuned["tuned_mean_makespan"])
        random_mean = mean(random_means)
        tuned_improvement = random_mean - tuned_mean
        tuned_runtime = float(tuned["tuned_avg_runtime_seconds"])
        random_runtime = mean(random_runtimes)
        comparison_rows.append(
            {
                "original_instance_name": instance_name,
                "budget": int(budget),
                "random_config_count": len(random_rows),
                "variant_count": int(tuned["variant_count"]),
                "training_mean_makespan": format_float(
                    tuned["training_mean_makespan"],
                    3,
                ),
                "tuned_pop_size": int(tuned["tuned_pop_size"]),
                "tuned_ngen": int(tuned["tuned_ngen"]),
                "tuned_cxpb": format_float(tuned["tuned_cxpb"], 2),
                "tuned_mutpb": format_float(tuned["tuned_mutpb"], 2),
                "tuned_shared_mutation": format_float(
                    tuned["tuned_shared_mutation"],
                    2,
                ),
                "tuned_mean_makespan": format_float(tuned_mean, 3),
                "random_mean_of_config_means": format_float(random_mean, 3),
                "random_best_config_mean_makespan": format_float(min(random_means), 3),
                "random_worst_config_mean_makespan": format_float(max(random_means), 3),
                "random_std_of_config_means": format_float(pstdev(random_means), 3),
                "tuned_improvement_vs_random_mean": format_float(tuned_improvement, 3),
                "tuned_improvement_vs_random_mean_percent": format_percent(
                    tuned_improvement,
                    random_mean,
                    3,
                ),
                "random_configs_beaten_by_tuned": sum(
                    1 for value in random_means if tuned_mean < value
                ),
                "tuned_rank_among_random_plus_tuned": 1
                + sum(1 for value in random_means if value < tuned_mean),
                "tuned_avg_runtime_seconds": format_float(tuned_runtime, 3),
                "random_avg_runtime_seconds": format_float(random_runtime, 3),
                "runtime_delta_tuned_minus_random": format_float(
                    tuned_runtime - random_runtime,
                    3,
                ),
            }
        )
    return comparison_rows


def compare_tuned_vs_random_by_variant(tuned_rows, random_run_rows):
    comparison_rows = []
    tuned_by_variant = {
        key: rows[0]
        for key, rows in grouped_rows(
            tuned_rows,
            ["original_instance_name", "variant_name", "budget"],
        ).items()
    }
    random_by_variant = grouped_rows(
        random_run_rows,
        ["original_instance_name", "variant_name", "budget"],
    )

    for key, tuned_row in sorted(
        tuned_by_variant.items(),
        key=lambda item: (item[0][0], item[0][1], int(item[0][2])),
    ):
        instance_name, variant_name, budget = key
        random_rows = random_by_variant.get(key, [])
        random_means = [float(row["variant_mean_makespan"]) for row in random_rows]
        random_runtimes = [
            float(row["variant_avg_runtime_seconds"]) for row in random_rows
        ]
        tuned_mean = float(tuned_row["variant_mean_makespan"])
        tuned_runtime = float(tuned_row["variant_avg_runtime_seconds"])
        random_mean = mean(random_means)
        tuned_improvement = random_mean - tuned_mean
        comparison_rows.append(
            {
                "original_instance_name": instance_name,
                "variant_name": variant_name,
                "variant_path": tuned_row["variant_path"],
                "budget": int(budget),
                "random_config_count": len(random_rows),
                "training_mean_makespan": tuned_row["training_mean_makespan"],
                "tuned_pop_size": int(tuned_row["pop_size"]),
                "tuned_ngen": int(tuned_row["ngen"]),
                "tuned_cxpb": tuned_row["cxpb"],
                "tuned_mutpb": tuned_row["mutpb"],
                "tuned_shared_mutation": tuned_row["shared_mutation"],
                "tuned_variant_mean_makespan": format_float(tuned_mean, 3),
                "tuned_variant_std_makespan": tuned_row["variant_std_makespan"],
                "random_mean_of_config_means_on_variant": format_float(random_mean, 3),
                "random_best_config_mean_makespan_on_variant": format_float(
                    min(random_means),
                    3,
                ),
                "random_worst_config_mean_makespan_on_variant": format_float(
                    max(random_means),
                    3,
                ),
                "random_std_of_config_means_on_variant": format_float(
                    pstdev(random_means),
                    3,
                ),
                "tuned_improvement_vs_random_mean_on_variant": format_float(
                    tuned_improvement,
                    3,
                ),
                "tuned_improvement_vs_random_mean_on_variant_percent": format_percent(
                    tuned_improvement,
                    random_mean,
                    3,
                ),
                "random_configs_beaten_by_tuned_on_variant": sum(
                    1 for value in random_means if tuned_mean < value
                ),
                "tuned_rank_among_random_plus_tuned_on_variant": 1
                + sum(1 for value in random_means if value < tuned_mean),
                "tuned_variant_avg_runtime_seconds": format_float(tuned_runtime, 3),
                "random_avg_runtime_seconds_on_variant": format_float(
                    mean(random_runtimes),
                    3,
                ),
                "runtime_delta_tuned_minus_random_on_variant": format_float(
                    tuned_runtime - mean(random_runtimes),
                    3,
                ),
            }
        )
    return comparison_rows


def keep_columns(rows, fieldnames):
    return [{field: row[field] for field in fieldnames} for row in rows]


AGGREGATE_COMPARISON_PRESENTATION_FIELDS = [
    "original_instance_name",
    "budget",
    "random_config_count",
    "variant_count",
    "training_mean_makespan",
    "tuned_mean_makespan",
    "random_mean_of_config_means",
    "random_best_config_mean_makespan",
    "random_worst_config_mean_makespan",
    "random_std_of_config_means",
    "tuned_improvement_vs_random_mean",
    "tuned_improvement_vs_random_mean_percent",
    "random_configs_beaten_by_tuned",
    "tuned_rank_among_random_plus_tuned",
    "tuned_avg_runtime_seconds",
    "random_avg_runtime_seconds",
    "runtime_delta_tuned_minus_random",
]


VARIANT_COMPARISON_PRESENTATION_FIELDS = [
    "original_instance_name",
    "variant_name",
    "budget",
    "random_config_count",
    "training_mean_makespan",
    "tuned_variant_mean_makespan",
    "tuned_variant_std_makespan",
    "random_mean_of_config_means_on_variant",
    "random_best_config_mean_makespan_on_variant",
    "random_worst_config_mean_makespan_on_variant",
    "random_std_of_config_means_on_variant",
    "tuned_improvement_vs_random_mean_on_variant",
    "tuned_improvement_vs_random_mean_on_variant_percent",
    "random_configs_beaten_by_tuned_on_variant",
    "tuned_rank_among_random_plus_tuned_on_variant",
    "tuned_variant_avg_runtime_seconds",
    "random_avg_runtime_seconds_on_variant",
    "runtime_delta_tuned_minus_random_on_variant",
]


def parse_pair_range(values, label):
    if len(values) != 2:
        raise ValueError(f"{label} requires exactly two values: min max.")
    lower, upper = float(values[0]), float(values[1])
    if lower > upper:
        raise ValueError(f"{label} lower bound cannot exceed upper bound.")
    return lower, upper


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Run continuous random same-budget baselines on the already generated "
            "fair DAG variants, then compare them against tuned generalization results."
        )
    )
    parser.add_argument(
        "--tuned-run-results",
        type=Path,
        default=Path("logs/generalization_eval_fair_topology/generalization_run_results.csv"),
        help="Trial 1 tuned run-results CSV.",
    )
    parser.add_argument(
        "--variant-manifest",
        type=Path,
        default=Path("logs/generalization_eval_fair_topology/generalization_variants.csv"),
        help="CSV listing the fair DAG variants to reuse.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Scheduler config path. Default: {DEFAULT_CONFIG_PATH.name}.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: logs/generalization_eval_fair_topology_random_baseline_<timestamp>.",
    )
    parser.add_argument(
        "--random-configs-per-budget",
        type=int,
        default=5,
        help="Number of random continuous settings per instance and budget.",
    )
    parser.add_argument(
        "--cxpb-range",
        nargs=2,
        default=[0.50, 0.90],
        metavar=("MIN", "MAX"),
        help="Uniform sampling range for cxpb.",
    )
    parser.add_argument(
        "--mutpb-range",
        nargs=2,
        default=[0.10, 0.30],
        metavar=("MIN", "MAX"),
        help="Uniform sampling range for mutpb.",
    )
    parser.add_argument(
        "--shared-mutation-range",
        nargs=2,
        default=[0.05, 0.25],
        metavar=("MIN", "MAX"),
        help="Uniform sampling range for all scheduler mutation sub-probabilities.",
    )
    parser.add_argument(
        "--random-config-seed",
        type=int,
        default=230000,
        help="Seed used to sample random baseline configurations.",
    )
    parser.add_argument(
        "--benchmark-repeats",
        type=int,
        default=3,
        help="Repeated scheduler runs per random configuration and DAG variant.",
    )
    parser.add_argument(
        "--benchmark-seed-base",
        type=int,
        default=330000,
        help="Base seed for random-baseline scheduler benchmark repeats.",
    )
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=1,
        help="Number of random-baseline cases to evaluate concurrently.",
    )
    parser.add_argument(
        "--generate-configs-only",
        action="store_true",
        help="Write sampled random configurations, then stop before benchmarks.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else Path("logs") / f"generalization_eval_fair_topology_random_baseline_{timestamp}"
    )
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cxpb_range = parse_pair_range(args.cxpb_range, "--cxpb-range")
    mutpb_range = parse_pair_range(args.mutpb_range, "--mutpb-range")
    shared_mutation_range = parse_pair_range(
        args.shared_mutation_range,
        "--shared-mutation-range",
    )

    tuned_rows = load_csv_rows(args.tuned_run_results)
    variant_rows = load_csv_rows(args.variant_manifest)
    tuned_summary = collect_tuned_summary(tuned_rows)
    base_config = load_config(args.config)

    random_config_rows = sample_random_configurations(
        tuned_summary=tuned_summary,
        base_config=base_config,
        random_configs_per_budget=args.random_configs_per_budget,
        cxpb_range=cxpb_range,
        mutpb_range=mutpb_range,
        shared_mutation_range=shared_mutation_range,
        seed=args.random_config_seed,
    )
    write_csv(output_dir / "random_configurations.csv", random_config_rows)

    if args.generate_configs_only:
        print("Random configurations:", (output_dir / "random_configurations.csv").resolve())
        return

    tasks = build_random_tasks(
        random_config_rows=random_config_rows,
        variant_rows=variant_rows,
        tuned_summary=tuned_summary,
        output_dir=output_dir,
        config_path=args.config,
        benchmark_repeats=args.benchmark_repeats,
        benchmark_seed_base=args.benchmark_seed_base,
    )

    random_run_rows = []
    if args.parallel_workers <= 1:
        for task in tasks:
            row = run_random_task(task)
            random_run_rows.append(row)
            print(
                "Completed random baseline run:",
                {
                    "instance": row["original_instance_name"],
                    "variant": row["variant_name"],
                    "budget": row["budget"],
                    "random_config_id": row["random_config_id"],
                    "mean_makespan": row["variant_mean_makespan"],
                    "avg_runtime": row["variant_avg_runtime_seconds"],
                },
            )
    else:
        with ProcessPoolExecutor(max_workers=args.parallel_workers) as executor:
            future_to_task = {executor.submit(run_random_task, task): task for task in tasks}
            for future in as_completed(future_to_task):
                row = future.result()
                random_run_rows.append(row)
                print(
                    "Completed random baseline run:",
                    {
                        "instance": row["original_instance_name"],
                        "variant": row["variant_name"],
                        "budget": row["budget"],
                        "random_config_id": row["random_config_id"],
                        "mean_makespan": row["variant_mean_makespan"],
                        "avg_runtime": row["variant_avg_runtime_seconds"],
                    },
                )

    random_run_rows.sort(
        key=lambda row: (
            row["original_instance_name"],
            int(row["budget"]),
            row["random_config_id"],
            row["variant_name"],
        )
    )
    random_config_summary_rows = summarize_random_configs(random_run_rows)
    comparison_rows = compare_tuned_vs_random(tuned_summary, random_config_summary_rows)
    variant_comparison_rows = compare_tuned_vs_random_by_variant(
        tuned_rows,
        random_run_rows,
    )

    write_csv(output_dir / "random_baseline_run_results.csv", random_run_rows)
    write_csv(output_dir / "random_baseline_by_config_summary.csv", random_config_summary_rows)
    write_csv(output_dir / "tuned_vs_random_comparison.csv", comparison_rows)
    write_csv(
        output_dir / "tuned_vs_random_comparison_clean.csv",
        keep_columns(comparison_rows, AGGREGATE_COMPARISON_PRESENTATION_FIELDS),
    )
    write_csv(
        output_dir / "tuned_vs_random_by_variant_comparison.csv",
        variant_comparison_rows,
    )
    write_csv(
        output_dir / "tuned_vs_random_by_variant_comparison_clean.csv",
        keep_columns(variant_comparison_rows, VARIANT_COMPARISON_PRESENTATION_FIELDS),
    )
    write_json(
        output_dir / "random_baseline_summary.json",
        {
            "description": (
                "Trial 2: continuous random same-budget baseline on the same fair "
                "DAG variants used for Trial 1."
            ),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "tuned_run_results": str(args.tuned_run_results.resolve()),
            "variant_manifest": str(args.variant_manifest.resolve()),
            "config": str(args.config.resolve()),
            "output_dir": str(output_dir),
            "random_configs_per_budget": args.random_configs_per_budget,
            "cxpb_range": list(cxpb_range),
            "mutpb_range": list(mutpb_range),
            "shared_mutation_range": list(shared_mutation_range),
            "benchmark_repeats": args.benchmark_repeats,
            "parallel_workers": args.parallel_workers,
            "random_configurations": random_config_rows,
            "random_run_rows": random_run_rows,
            "random_config_summary_rows": random_config_summary_rows,
            "comparison_rows": comparison_rows,
            "variant_comparison_rows": variant_comparison_rows,
        },
    )
    print("Random baseline run results:", (output_dir / "random_baseline_run_results.csv").resolve())
    print("Random baseline by-config summary:", (output_dir / "random_baseline_by_config_summary.csv").resolve())
    print("Tuned-vs-random comparison:", (output_dir / "tuned_vs_random_comparison.csv").resolve())
    print(
        "Tuned-vs-random clean comparison:",
        (output_dir / "tuned_vs_random_comparison_clean.csv").resolve(),
    )
    print(
        "Tuned-vs-random by-variant comparison:",
        (output_dir / "tuned_vs_random_by_variant_comparison.csv").resolve(),
    )
    print(
        "Tuned-vs-random clean by-variant comparison:",
        (output_dir / "tuned_vs_random_by_variant_comparison_clean.csv").resolve(),
    )


if __name__ == "__main__":
    main()
