"""Run fair exact-budget nested-GA sweeps and summarize how cost changes behavior."""

import argparse
import io
import json
import re
import subprocess
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import redirect_stdout
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from OptimizerGa import build_feasible_budget_pairs, run_default_outer_ga
from ga_config import DEFAULT_CONFIG_PATH, load_config, resolve_config_path


def write_json(path, payload):
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def standard_deviation(values):
    if not values:
        return 0.0
    return pstdev(values)


def example_sort_key(path):
    match = re.search(r"example_(\d+)T\.json$", path.name)
    return int(match.group(1)) if match else path.name


def find_example_json_paths():
    return sorted(Path(".").glob("example_*T.json"), key=example_sort_key)


def parse_internal_args(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--internal-task-file", type=Path, default=None)
    parser.add_argument("--internal-result-file", type=Path, default=None)
    return parser.parse_known_args(argv)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Run a sweep over exact computation budgets for the nested outer GA and "
            "summarize how budget affects training and validation behavior."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Config file to load. Default: {DEFAULT_CONFIG_PATH.name}.",
    )
    parser.add_argument(
        "input_json_paths",
        nargs="*",
        type=Path,
        help="Example JSON file(s), such as example_50T.json.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run every example_*T.json file in this folder, in numeric order.",
    )
    parser.add_argument(
        "--budgets",
        nargs="+",
        type=int,
        required=True,
        help="Exact fair budgets to test, for example: --budgets 1800 2100 2400 2520 3360.",
    )
    parser.add_argument(
        "--outer-seeds",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Outer-GA random seeds to test for each budget and instance. "
            "If omitted, the config seed is used once."
        ),
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Optional short label appended to the batch folder name.",
    )
    parser.add_argument(
        "--show-plot",
        action="store_true",
        help="Display plots after saving them when an interactive backend is available.",
    )
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=1,
        help=(
            "How many budget-sweep runs to execute in parallel. "
            "Use 1 for sequential execution. Default: 1."
        ),
    )
    return parser.parse_args(argv)


def resolve_input_json_paths(args):
    if args.all:
        input_json_paths = find_example_json_paths()
    elif args.input_json_paths:
        input_json_paths = args.input_json_paths
    else:
        raise ValueError("Provide example JSON files or use --all.")

    if not input_json_paths:
        raise ValueError("No example JSON files were found for the requested sweep.")
    return input_json_paths


def create_budget_sweep_output_paths(config, label=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = ""
    if label:
        safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_")
        if safe_label:
            suffix = f"_{safe_label}"

    batch_name = f"budget_sweep{suffix}_{timestamp}"
    log_root = resolve_config_path(config, config["paths"]["log_dir"])
    run_dir = (log_root / batch_name).resolve()
    runs_dir = run_dir / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    return {
        "batch_name": batch_name,
        "run_dir": run_dir,
        "runs_dir": runs_dir,
        "summary_path": run_dir / f"{batch_name}_summary.json",
        "selected_validation_plot_path": run_dir / f"{batch_name}_selected_validation_makespan.png",
        "selected_validation_runtime_plot_path": run_dir / f"{batch_name}_selected_validation_runtime.png",
        "best_validation_plot_path": run_dir / f"{batch_name}_best_validation_makespan.png",
        "normalized_validation_plot_path": run_dir / f"{batch_name}_normalized_selected_validation_makespan.png",
    }


def build_budget_metadata(base_config, budgets):
    rows = []
    for budget in budgets:
        config = deepcopy(base_config)
        config["outer_ga"]["computation_budget"] = {
            "mode": "exact_product",
            "budget": int(budget),
        }
        feasible_pairs = build_feasible_budget_pairs(config)
        rows.append(
            {
                "budget": int(budget),
                "feasible_pair_count": len(feasible_pairs),
                "feasible_pairs": feasible_pairs,
            }
        )
    return rows


def safe_path_component(value):
    text = "none" if value is None else str(value)
    safe_text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")
    return safe_text or "value"


def build_sweep_tasks(input_json_paths, budget_metadata_rows, outer_seeds, runs_dir):
    tasks = []
    task_index = 0
    for input_json_path in input_json_paths:
        instance_name = Path(input_json_path).stem
        for budget_row in budget_metadata_rows:
            for outer_seed in outer_seeds:
                task_index += 1
                task_dir_name = (
                    f"task_{task_index:03d}__{safe_path_component(instance_name)}"
                    f"__budget_{budget_row['budget']}__seed_{safe_path_component(outer_seed)}"
                )
                task_log_dir = (Path(runs_dir) / task_dir_name).resolve()
                tasks.append(
                    {
                        "task_index": task_index,
                        "instance_name": instance_name,
                        "input_json_path": str(Path(input_json_path).resolve()),
                        "budget": int(budget_row["budget"]),
                        "feasible_pair_count": int(budget_row["feasible_pair_count"]),
                        "feasible_pairs": list(budget_row["feasible_pairs"]),
                        "outer_random_seed": outer_seed,
                        "task_log_dir": str(task_log_dir),
                    }
                )
    return tasks


def resolve_outer_seeds(config, requested_outer_seeds):
    if requested_outer_seeds is not None:
        return requested_outer_seeds
    return [config["outer_ga"].get("random_seed")]


def summarize_validation_for_selected_candidate(result):
    if result.validation_results_path is None or not Path(result.validation_results_path).exists():
        return None, None

    validation_payload = load_json(result.validation_results_path)
    validation_results = validation_payload.get("results", [])
    if not validation_results:
        return None, None

    selected_candidate = None
    for candidate in validation_results:
        if candidate.get("training_evaluation_id") == result.best_evaluation_id:
            selected_candidate = candidate
            break
    if selected_candidate is None:
        selected_candidate = validation_results[0]

    best_validation_candidate = min(
        validation_results,
        key=lambda candidate: (
            candidate["validation_mean_makespan"],
            candidate["validation_std_makespan"],
            candidate["training_mean_makespan"],
            candidate["candidate_label"],
        ),
    )
    return selected_candidate, best_validation_candidate


def build_run_record(
    result,
    *,
    budget,
    budget_metadata,
    outer_seed,
    input_json_path,
):
    selected_validation_candidate, best_validation_candidate = (
        summarize_validation_for_selected_candidate(result)
    )
    return {
        "task_log_dir": None,
        "instance_name": Path(input_json_path).stem,
        "input_json_path": str(Path(input_json_path).resolve()),
        "budget": int(budget),
        "feasible_pair_count": budget_metadata["feasible_pair_count"],
        "feasible_pairs": list(budget_metadata["feasible_pairs"]),
        "outer_random_seed": outer_seed,
        "run_dir": str(result.run_dir) if result.run_dir is not None else None,
        "history_path": str(result.history_path) if result.history_path is not None else None,
        "evaluation_log_path": (
            str(result.evaluation_log_path) if result.evaluation_log_path is not None else None
        ),
        "best_result_path": str(result.best_result_path) if result.best_result_path is not None else None,
        "validation_results_path": (
            str(result.validation_results_path) if result.validation_results_path is not None else None
        ),
        "selected_training_evaluation_id": result.best_evaluation_id,
        "selected_training_hyperparameters": dict(result.best_hyperparameters),
        "selected_training_weighted_objective_score": result.best_weighted_objective_score,
        "selected_training_mean_makespan": result.best_makespan,
        "selected_training_avg_inner_runtime_seconds": result.best_avg_inner_runtime_seconds,
        "selected_validation_candidate_label": (
            selected_validation_candidate["candidate_label"]
            if selected_validation_candidate is not None
            else None
        ),
        "selected_validation_mean_makespan": (
            selected_validation_candidate["validation_mean_makespan"]
            if selected_validation_candidate is not None
            else None
        ),
        "selected_validation_std_makespan": (
            selected_validation_candidate["validation_std_makespan"]
            if selected_validation_candidate is not None
            else None
        ),
        "selected_validation_avg_inner_runtime_seconds": (
            selected_validation_candidate["validation_avg_inner_runtime_seconds"]
            if selected_validation_candidate is not None
            else None
        ),
        "best_validation_candidate_label": (
            best_validation_candidate["candidate_label"]
            if best_validation_candidate is not None
            else None
        ),
        "best_validation_mean_makespan": (
            best_validation_candidate["validation_mean_makespan"]
            if best_validation_candidate is not None
            else None
        ),
        "best_validation_std_makespan": (
            best_validation_candidate["validation_std_makespan"]
            if best_validation_candidate is not None
            else None
        ),
        "best_validation_avg_inner_runtime_seconds": (
            best_validation_candidate["validation_avg_inner_runtime_seconds"]
            if best_validation_candidate is not None
            else None
        ),
    }


def build_failed_run_record(task, error, error_traceback=None):
    return {
        "task_index": task["task_index"],
        "task_log_dir": task["task_log_dir"],
        "instance_name": task["instance_name"],
        "input_json_path": task["input_json_path"],
        "budget": int(task["budget"]),
        "outer_random_seed": task["outer_random_seed"],
        "error": str(error),
        "traceback": error_traceback,
    }


def sort_seed_value(seed):
    return -1 if seed is None else seed


def sort_run_record_key(record):
    return (
        record["instance_name"],
        record["budget"],
        sort_seed_value(record["outer_random_seed"]),
        record.get("task_index", 0),
    )


def sort_failed_run_key(record):
    return (
        record["instance_name"],
        record["budget"],
        sort_seed_value(record["outer_random_seed"]),
        record.get("task_index", 0),
    )


def execute_budget_sweep_task(task, config_path, show_plot=False):
    config = load_config(config_path)
    run_config = deepcopy(config)
    task_log_dir = Path(task["task_log_dir"])
    task_log_dir.mkdir(parents=True, exist_ok=True)
    run_config["paths"]["log_dir"] = str(task_log_dir)
    run_config["outer_ga"]["random_seed"] = task["outer_random_seed"]
    run_config["outer_ga"]["computation_budget"] = {
        "mode": "exact_product",
        "budget": int(task["budget"]),
    }

    try:
        with io.StringIO() as stdout_buffer, redirect_stdout(stdout_buffer):
            result = run_default_outer_ga(
                input_json_path=Path(task["input_json_path"]),
                config=run_config,
                show_plot=show_plot,
            )
    except Exception as error:
        return {
            "status": "failed",
            "failed_run": build_failed_run_record(
                task,
                error,
                error_traceback=traceback.format_exc(),
            ),
        }

    run_record = build_run_record(
        result,
        budget=int(task["budget"]),
        budget_metadata={
            "feasible_pair_count": task["feasible_pair_count"],
            "feasible_pairs": task["feasible_pairs"],
        },
        outer_seed=task["outer_random_seed"],
        input_json_path=task["input_json_path"],
    )
    run_record["task_index"] = task["task_index"]
    run_record["task_log_dir"] = task["task_log_dir"]
    return {
        "status": "completed",
        "run_record": run_record,
    }


def run_internal_task_mode(task_file, result_file):
    payload = load_json(task_file)
    outcome = execute_budget_sweep_task(
        payload["task"],
        payload["config_path"],
        payload.get("show_plot", False),
    )
    write_json(result_file, outcome)


def execute_budget_sweep_task_subprocess(
    task,
    *,
    config_path,
    show_plot,
    script_path,
    worker_dir,
):
    worker_dir = Path(worker_dir)
    worker_dir.mkdir(parents=True, exist_ok=True)
    task_file = worker_dir / f"task_{task['task_index']:03d}_input.json"
    result_file = worker_dir / f"task_{task['task_index']:03d}_result.json"
    write_json(
        task_file,
        {
            "task": task,
            "config_path": str(config_path),
            "show_plot": show_plot,
        },
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--internal-task-file",
            str(task_file),
            "--internal-result-file",
            str(result_file),
        ],
        capture_output=True,
        text=True,
    )

    if completed.returncode != 0:
        failed_run = build_failed_run_record(
            task,
            RuntimeError(
                "Parallel worker subprocess failed with return code "
                f"{completed.returncode}."
            ),
        )
        failed_run["worker_stdout"] = completed.stdout
        failed_run["worker_stderr"] = completed.stderr
        return {
            "status": "failed",
            "failed_run": failed_run,
        }

    if not result_file.exists():
        failed_run = build_failed_run_record(
            task,
            RuntimeError("Parallel worker subprocess did not produce a result file."),
        )
        failed_run["worker_stdout"] = completed.stdout
        failed_run["worker_stderr"] = completed.stderr
        return {
            "status": "failed",
            "failed_run": failed_run,
        }

    outcome = load_json(result_file)
    if outcome.get("status") == "failed":
        outcome["failed_run"]["worker_stdout"] = completed.stdout
        outcome["failed_run"]["worker_stderr"] = completed.stderr

    try:
        task_file.unlink()
    except OSError:
        pass
    try:
        result_file.unlink()
    except OSError:
        pass
    return outcome


def mean_and_std(records, key):
    values = [record[key] for record in records if record.get(key) is not None]
    if not values:
        return None, None
    return mean(values), standard_deviation(values)


def build_budget_instance_summary(run_records):
    grouped = {}
    for record in run_records:
        group_key = (record["instance_name"], record["budget"])
        grouped.setdefault(group_key, []).append(record)

    rows = []
    for (instance_name, budget), records in sorted(grouped.items()):
        selected_training_mean, selected_training_std = mean_and_std(
            records,
            "selected_training_mean_makespan",
        )
        selected_validation_mean, selected_validation_std = mean_and_std(
            records,
            "selected_validation_mean_makespan",
        )
        selected_validation_runtime_mean, selected_validation_runtime_std = mean_and_std(
            records,
            "selected_validation_avg_inner_runtime_seconds",
        )
        best_validation_mean, best_validation_std = mean_and_std(
            records,
            "best_validation_mean_makespan",
        )
        weighted_score_mean, weighted_score_std = mean_and_std(
            records,
            "selected_training_weighted_objective_score",
        )
        rows.append(
            {
                "instance_name": instance_name,
                "budget": budget,
                "run_count": len(records),
                "outer_random_seeds": [record["outer_random_seed"] for record in records],
                "feasible_pair_count": records[0]["feasible_pair_count"],
                "feasible_pairs": records[0]["feasible_pairs"],
                "selected_training_mean_makespan_mean": selected_training_mean,
                "selected_training_mean_makespan_std": selected_training_std,
                "selected_training_weighted_objective_score_mean": weighted_score_mean,
                "selected_training_weighted_objective_score_std": weighted_score_std,
                "selected_validation_mean_makespan_mean": selected_validation_mean,
                "selected_validation_mean_makespan_std": selected_validation_std,
                "selected_validation_avg_inner_runtime_seconds_mean": (
                    selected_validation_runtime_mean
                ),
                "selected_validation_avg_inner_runtime_seconds_std": (
                    selected_validation_runtime_std
                ),
                "best_validation_mean_makespan_mean": best_validation_mean,
                "best_validation_mean_makespan_std": best_validation_std,
            }
        )
    return rows


def build_normalized_budget_summary(budget_instance_rows):
    per_instance_minima = {}
    for row in budget_instance_rows:
        metric = row.get("selected_validation_mean_makespan_mean")
        if metric is None:
            continue
        previous = per_instance_minima.get(row["instance_name"])
        if previous is None or metric < previous:
            per_instance_minima[row["instance_name"]] = metric

    grouped = {}
    for row in budget_instance_rows:
        instance_minimum = per_instance_minima.get(row["instance_name"])
        metric = row.get("selected_validation_mean_makespan_mean")
        if instance_minimum is None or metric is None or instance_minimum <= 0:
            continue
        grouped.setdefault(row["budget"], []).append(metric / instance_minimum)

    rows = []
    for budget, normalized_values in sorted(grouped.items()):
        rows.append(
            {
                "budget": budget,
                "instance_count": len(normalized_values),
                "normalized_selected_validation_mean_makespan_mean": mean(normalized_values),
                "normalized_selected_validation_mean_makespan_std": standard_deviation(
                    normalized_values
                ),
            }
        )
    return rows


def plot_budget_metric_by_instance(
    budget_instance_rows,
    *,
    metric_key,
    metric_std_key,
    save_path,
    title,
    ylabel,
    show_plot=False,
):
    grouped = {}
    for row in budget_instance_rows:
        metric_value = row.get(metric_key)
        if metric_value is None:
            continue
        grouped.setdefault(row["instance_name"], []).append(row)

    if not grouped:
        return None

    save_path = Path(save_path)
    fig, axis = plt.subplots(figsize=(10.5, 5.8))
    for instance_name, rows in sorted(grouped.items()):
        sorted_rows = sorted(rows, key=lambda row: row["budget"])
        budgets = [row["budget"] for row in sorted_rows]
        metric_values = [row[metric_key] for row in sorted_rows]
        metric_stds = [
            0.0 if row.get(metric_std_key) is None else row[metric_std_key]
            for row in sorted_rows
        ]
        axis.errorbar(
            budgets,
            metric_values,
            yerr=metric_stds,
            marker="o",
            linewidth=1.8,
            capsize=4,
            label=instance_name,
        )

    axis.set_xlabel("Exact fair computation budget")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    if show_plot and matplotlib.get_backend().lower() != "agg":
        plt.show()
    plt.close(fig)
    return save_path.resolve()


def plot_normalized_budget_summary(
    normalized_budget_rows,
    *,
    save_path,
    show_plot=False,
):
    if not normalized_budget_rows:
        return None

    save_path = Path(save_path)
    budgets = [row["budget"] for row in normalized_budget_rows]
    normalized_means = [
        row["normalized_selected_validation_mean_makespan_mean"]
        for row in normalized_budget_rows
    ]
    normalized_stds = [
        row["normalized_selected_validation_mean_makespan_std"]
        for row in normalized_budget_rows
    ]

    fig, axis = plt.subplots(figsize=(10.5, 5.2))
    axis.errorbar(
        budgets,
        normalized_means,
        yerr=normalized_stds,
        marker="o",
        linewidth=1.8,
        capsize=4,
        color="tab:purple",
    )
    axis.axhline(1.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
    axis.set_xlabel("Exact fair computation budget")
    axis.set_ylabel("Normalized selected validation mean makespan")
    axis.set_title("Cross-instance budget effect on selected validation makespan")
    axis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    if show_plot and matplotlib.get_backend().lower() != "agg":
        plt.show()
    plt.close(fig)
    return save_path.resolve()


def build_batch_summary(
    *,
    config,
    input_json_paths,
    budget_metadata_rows,
    outer_seeds,
    parallel_workers,
    run_records,
    failed_runs,
    output_paths,
    show_plot=False,
):
    budget_instance_rows = build_budget_instance_summary(run_records)
    normalized_budget_rows = build_normalized_budget_summary(budget_instance_rows)

    plot_paths = {
        "selected_validation_makespan": plot_budget_metric_by_instance(
            budget_instance_rows,
            metric_key="selected_validation_mean_makespan_mean",
            metric_std_key="selected_validation_mean_makespan_std",
            save_path=output_paths["selected_validation_plot_path"],
            title="Selected candidate validation makespan vs exact fair budget",
            ylabel="Validation mean makespan",
            show_plot=show_plot,
        ),
        "selected_validation_runtime": plot_budget_metric_by_instance(
            budget_instance_rows,
            metric_key="selected_validation_avg_inner_runtime_seconds_mean",
            metric_std_key="selected_validation_avg_inner_runtime_seconds_std",
            save_path=output_paths["selected_validation_runtime_plot_path"],
            title="Selected candidate validation runtime vs exact fair budget",
            ylabel="Validation average inner runtime (seconds)",
            show_plot=show_plot,
        ),
        "best_validation_makespan": plot_budget_metric_by_instance(
            budget_instance_rows,
            metric_key="best_validation_mean_makespan_mean",
            metric_std_key="best_validation_mean_makespan_std",
            save_path=output_paths["best_validation_plot_path"],
            title="Best validated candidate makespan vs exact fair budget",
            ylabel="Validation mean makespan",
            show_plot=show_plot,
        ),
        "normalized_selected_validation_makespan": plot_normalized_budget_summary(
            normalized_budget_rows,
            save_path=output_paths["normalized_validation_plot_path"],
            show_plot=show_plot,
        ),
    }

    return {
        "description": (
            "Budget sweep summary for exact fair nested-GA computation budgets. "
            "Each budget enforces pop_size * ngen = budget for the inner scheduler GA."
        ),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config_path": config["_meta"]["config_path"],
        "batch_name": output_paths["batch_name"],
        "batch_run_dir": str(output_paths["run_dir"]),
        "input_instances": [str(Path(path).resolve()) for path in input_json_paths],
        "outer_random_seeds": outer_seeds,
        "parallel_workers": parallel_workers,
        "budget_metadata": budget_metadata_rows,
        "planned_run_count": len(input_json_paths) * len(outer_seeds) * len(budget_metadata_rows),
        "completed_run_count": len(run_records),
        "failed_run_count": len(failed_runs),
        "failed_runs": failed_runs,
        "runs": run_records,
        "budget_instance_summary": budget_instance_rows,
        "normalized_budget_summary": normalized_budget_rows,
        "plot_paths": {
            name: (str(path) if path is not None else None)
            for name, path in plot_paths.items()
        },
    }


if __name__ == "__main__":
    internal_args, remaining_args = parse_internal_args()
    if (
        internal_args.internal_task_file is not None
        or internal_args.internal_result_file is not None
    ):
        if (
            internal_args.internal_task_file is None
            or internal_args.internal_result_file is None
        ):
            raise ValueError(
                "--internal-task-file and --internal-result-file must be used together."
            )
        run_internal_task_mode(
            internal_args.internal_task_file,
            internal_args.internal_result_file,
        )
        raise SystemExit(0)

    args = parse_args(remaining_args)
    if args.parallel_workers < 1:
        raise ValueError("--parallel-workers must be at least 1.")

    config = load_config(args.config)
    input_json_paths = resolve_input_json_paths(args)
    outer_seeds = resolve_outer_seeds(config, args.outer_seeds)
    budget_metadata_rows = build_budget_metadata(config, args.budgets)
    output_paths = create_budget_sweep_output_paths(config, label=args.label)
    tasks = build_sweep_tasks(
        input_json_paths,
        budget_metadata_rows,
        outer_seeds,
        output_paths["runs_dir"],
    )

    run_records = []
    failed_runs = []

    print("Budget sweep batch folder:", output_paths["run_dir"])
    print("Budgets to test:", [row["budget"] for row in budget_metadata_rows])
    print("Parallel workers:", args.parallel_workers)
    for row in budget_metadata_rows:
        print(
            "Budget metadata:",
            {
                "budget": row["budget"],
                "feasible_pair_count": row["feasible_pair_count"],
                "feasible_pairs": row["feasible_pairs"],
            },
        )
    print("Planned run count:", len(tasks))

    if args.parallel_workers == 1:
        for task in tasks:
            print(
                "Starting budget-sweep run:",
                {
                    "instance": task["instance_name"],
                    "budget": task["budget"],
                    "outer_random_seed": task["outer_random_seed"],
                    "task_index": task["task_index"],
                },
            )
            outcome = execute_budget_sweep_task(
                task,
                config["_meta"]["config_path"],
                args.show_plot,
            )
            if outcome["status"] == "completed":
                run_record = outcome["run_record"]
                run_records.append(run_record)
                print(
                    "Completed budget-sweep run:",
                    {
                        "instance": run_record["instance_name"],
                        "budget": run_record["budget"],
                        "outer_random_seed": run_record["outer_random_seed"],
                        "task_index": run_record["task_index"],
                        "selected_training_mean_makespan": round(
                            run_record["selected_training_mean_makespan"],
                            3,
                        ),
                        "selected_validation_mean_makespan": (
                            None
                            if run_record["selected_validation_mean_makespan"] is None
                            else round(run_record["selected_validation_mean_makespan"], 3)
                        ),
                    },
                )
            else:
                failed_run = outcome["failed_run"]
                failed_runs.append(failed_run)
                print(
                    "Budget-sweep run failed:",
                    {
                        "instance": failed_run["instance_name"],
                        "budget": failed_run["budget"],
                        "outer_random_seed": failed_run["outer_random_seed"],
                        "task_index": failed_run["task_index"],
                        "error": failed_run["error"],
                    },
                )
    else:
        max_workers = min(args.parallel_workers, len(tasks))
        worker_dir = output_paths["run_dir"] / "_worker_state"
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {
                executor.submit(
                    execute_budget_sweep_task_subprocess,
                    task,
                    config_path=config["_meta"]["config_path"],
                    show_plot=args.show_plot,
                    script_path=Path(__file__).resolve(),
                    worker_dir=worker_dir,
                ): task
                for task in tasks
            }
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    outcome = future.result()
                except Exception as error:
                    failed_run = build_failed_run_record(
                        task,
                        error,
                        error_traceback=traceback.format_exc(),
                    )
                    failed_runs.append(failed_run)
                    print(
                        "Budget-sweep worker crashed:",
                        {
                            "instance": failed_run["instance_name"],
                            "budget": failed_run["budget"],
                            "outer_random_seed": failed_run["outer_random_seed"],
                            "task_index": failed_run["task_index"],
                            "error": failed_run["error"],
                        },
                    )
                    continue

                if outcome["status"] == "completed":
                    run_record = outcome["run_record"]
                    run_records.append(run_record)
                    print(
                        "Completed budget-sweep run:",
                        {
                            "instance": run_record["instance_name"],
                            "budget": run_record["budget"],
                            "outer_random_seed": run_record["outer_random_seed"],
                            "task_index": run_record["task_index"],
                            "selected_training_mean_makespan": round(
                                run_record["selected_training_mean_makespan"],
                                3,
                            ),
                            "selected_validation_mean_makespan": (
                                None
                                if run_record["selected_validation_mean_makespan"] is None
                                else round(run_record["selected_validation_mean_makespan"], 3)
                            ),
                        },
                    )
                else:
                    failed_run = outcome["failed_run"]
                    failed_runs.append(failed_run)
                    print(
                        "Budget-sweep run failed:",
                        {
                            "instance": failed_run["instance_name"],
                            "budget": failed_run["budget"],
                            "outer_random_seed": failed_run["outer_random_seed"],
                            "task_index": failed_run["task_index"],
                            "error": failed_run["error"],
                        },
                    )
        try:
            worker_dir.rmdir()
        except OSError:
            pass

    run_records.sort(key=sort_run_record_key)
    failed_runs.sort(key=sort_failed_run_key)

    summary_payload = build_batch_summary(
        config=config,
        input_json_paths=input_json_paths,
        budget_metadata_rows=budget_metadata_rows,
        outer_seeds=outer_seeds,
        parallel_workers=args.parallel_workers,
        run_records=run_records,
        failed_runs=failed_runs,
        output_paths=output_paths,
        show_plot=args.show_plot,
    )
    write_json(output_paths["summary_path"], summary_payload)

    print("Budget sweep summary saved to:", output_paths["summary_path"].resolve())
    for plot_name, plot_path in summary_payload["plot_paths"].items():
        if plot_path is not None:
            print(f"{plot_name} plot saved to:", plot_path)
