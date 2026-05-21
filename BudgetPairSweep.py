"""Run exhaustive fair-budget scheduler-GA sweeps without an outer GA layer."""

import argparse
import csv
import json
import re
import subprocess
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime
from itertools import product
from pathlib import Path
from statistics import mean, pstdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from OptimizerGa import build_feasible_budget_pairs
from SchedulerGa import run_static_scheduler_benchmark
from ga_config import DEFAULT_CONFIG_PATH, load_config, resolve_config_path


def write_json(path, payload):
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path, rows):
    path = Path(path)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


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
            "Exhaustively evaluate all feasible (pop_size, ngen) pairs under exact "
            "fair budgets, with fixed inner-GA probabilities and no outer GA."
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
        "--cxpb-values",
        nargs="+",
        type=float,
        default=None,
        help="Fixed cxpb values to sweep across runs.",
    )
    parser.add_argument(
        "--mutpb-values",
        nargs="+",
        type=float,
        default=None,
        help="Fixed mutpb values to sweep across runs.",
    )
    parser.add_argument(
        "--shared-mutation-values",
        nargs="+",
        type=float,
        default=None,
        help=(
            "Fixed scheduler mutation values that are applied equally to all 4 scheduler "
            "mutation probabilities in each run."
        ),
    )
    parser.add_argument(
        "--benchmark-repeats",
        type=int,
        default=None,
        help=(
            "How many repeated scheduler-GA runs to execute per fixed configuration. "
            "If omitted, static_scheduler_ga.benchmark_repeats from the config is used."
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
        help="How many pair-evaluation jobs to execute in parallel. Default: 1.",
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


def validate_probability_value(name, value):
    probability = float(value)
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"{name} values must stay within [0, 1], got {value}.")
    return probability


def resolve_fixed_parameter_grid(
    config,
    *,
    cxpb_values=None,
    mutpb_values=None,
    shared_mutation_values=None,
):
    default_cxpb = float(config["static_scheduler_ga"]["cxpb"])
    default_mutpb = float(config["static_scheduler_ga"]["mutpb"])
    default_shared_mutation = float(
        config["scheduler"]["mutation"]["task_order_probability"]
    )

    resolved_cxpb_values = (
        [default_cxpb]
        if not cxpb_values
        else [validate_probability_value("cxpb", value) for value in cxpb_values]
    )
    resolved_mutpb_values = (
        [default_mutpb]
        if not mutpb_values
        else [validate_probability_value("mutpb", value) for value in mutpb_values]
    )
    resolved_shared_mutation_values = (
        [default_shared_mutation]
        if not shared_mutation_values
        else [
            validate_probability_value("shared mutation", value)
            for value in shared_mutation_values
        ]
    )

    grid = []
    seen = set()
    for cxpb, mutpb, shared_mutation_probability in product(
        resolved_cxpb_values,
        resolved_mutpb_values,
        resolved_shared_mutation_values,
    ):
        signature = (
            round(cxpb, 10),
            round(mutpb, 10),
            round(shared_mutation_probability, 10),
        )
        if signature in seen:
            continue
        seen.add(signature)
        grid.append(
            {
                "fixed_cxpb": cxpb,
                "fixed_mutpb": mutpb,
                "shared_scheduler_mutation_probability": shared_mutation_probability,
                "fixed_parameter_combo_label": (
                    f"cxpb={cxpb:.2f}, mutpb={mutpb:.2f}, shared_mut={shared_mutation_probability:.2f}"
                ),
            }
        )
    return grid


def create_budget_pair_sweep_output_paths(config, label=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = ""
    if label:
        safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_")
        if safe_label:
            suffix = f"_{safe_label}"

    batch_name = f"budget_pair_sweep{suffix}_{timestamp}"
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
        "pair_results_csv_path": run_dir / f"{batch_name}_pair_results.csv",
        "best_fixed_params_per_pair_csv_path": (
            run_dir / f"{batch_name}_best_fixed_params_per_pair.csv"
        ),
        "best_pair_summary_csv_path": run_dir / f"{batch_name}_best_pair_summary.csv",
        "best_pair_makespan_plot_path": run_dir / f"{batch_name}_best_pair_makespan.png",
        "best_pair_runtime_plot_path": run_dir / f"{batch_name}_best_pair_runtime.png",
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


def build_pair_sweep_tasks(
    input_json_paths,
    budget_metadata_rows,
    fixed_parameter_grid,
    runs_dir,
):
    tasks = []
    task_index = 0
    for input_json_path in input_json_paths:
        instance_name = Path(input_json_path).stem
        for budget_row in budget_metadata_rows:
            for pop_size, ngen in budget_row["feasible_pairs"]:
                for parameter_combo in fixed_parameter_grid:
                    task_index += 1
                    cxpb_component = safe_path_component(
                        f"{parameter_combo['fixed_cxpb']:.2f}"
                    )
                    mutpb_component = safe_path_component(
                        f"{parameter_combo['fixed_mutpb']:.2f}"
                    )
                    shared_mutation_component = safe_path_component(
                        f"{parameter_combo['shared_scheduler_mutation_probability']:.2f}"
                    )
                    task_dir_name = (
                        f"task_{task_index:03d}__{safe_path_component(instance_name)}"
                        f"__budget_{budget_row['budget']}"
                        f"__pop_{pop_size}"
                        f"__ngen_{ngen}"
                        f"__cxpb_{cxpb_component}"
                        f"__mutpb_{mutpb_component}"
                        f"__sharedmut_{shared_mutation_component}"
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
                            "pop_size": int(pop_size),
                            "ngen": int(ngen),
                            "fixed_cxpb": parameter_combo["fixed_cxpb"],
                            "fixed_mutpb": parameter_combo["fixed_mutpb"],
                            "shared_scheduler_mutation_probability": (
                                parameter_combo["shared_scheduler_mutation_probability"]
                            ),
                            "fixed_parameter_combo_label": parameter_combo[
                                "fixed_parameter_combo_label"
                            ],
                            "task_log_dir": str(task_log_dir),
                        }
                    )
    return tasks


def build_pair_run_record(result_payload, task):
    fixed_hyperparameters = result_payload["fixed_hyperparameters"]
    return {
        "task_index": task["task_index"],
        "task_log_dir": task["task_log_dir"],
        "instance_name": task["instance_name"],
        "input_json_path": task["input_json_path"],
        "budget": task["budget"],
        "feasible_pair_count": task["feasible_pair_count"],
        "feasible_pairs": task["feasible_pairs"],
        "pop_size": task["pop_size"],
        "ngen": task["ngen"],
        "fixed_cxpb": task["fixed_cxpb"],
        "fixed_mutpb": task["fixed_mutpb"],
        "shared_scheduler_mutation_probability": task["shared_scheduler_mutation_probability"],
        "fixed_parameter_combo_label": task["fixed_parameter_combo_label"],
        "run_dir": result_payload["run_dir"],
        "results_path": result_payload["results_path"],
        "repeat_log_path": result_payload["repeat_log_path"],
        "plot_path": result_payload["plot_path"],
        "benchmark_repeats": result_payload["benchmark_repeats"],
        "random_seed": result_payload["random_seed"],
        "mean_makespan": result_payload["mean_makespan"],
        "std_makespan": result_payload["std_makespan"],
        "best_makespan": result_payload["best_makespan"],
        "avg_runtime_seconds": result_payload["avg_runtime_seconds"],
        "std_runtime_seconds": result_payload["std_runtime_seconds"],
        "total_runtime_seconds": result_payload["total_runtime_seconds"],
        "selection_tournament_size": fixed_hyperparameters["selection_tournament_size"],
    }


def build_failed_run_record(task, error, error_traceback=None):
    return {
        "task_index": task["task_index"],
        "task_log_dir": task["task_log_dir"],
        "instance_name": task["instance_name"],
        "input_json_path": task["input_json_path"],
        "budget": task["budget"],
        "pop_size": task["pop_size"],
        "ngen": task["ngen"],
        "fixed_cxpb": task["fixed_cxpb"],
        "fixed_mutpb": task["fixed_mutpb"],
        "shared_scheduler_mutation_probability": task["shared_scheduler_mutation_probability"],
        "fixed_parameter_combo_label": task["fixed_parameter_combo_label"],
        "error": str(error),
        "traceback": error_traceback,
    }


def sort_pair_run_key(record):
    return (
        record["instance_name"],
        record["budget"],
        record["fixed_cxpb"],
        record["fixed_mutpb"],
        record["shared_scheduler_mutation_probability"],
        record["pop_size"],
        record["ngen"],
        record["task_index"],
    )


def execute_pair_sweep_task(
    task,
    config_path,
    *,
    benchmark_repeats=None,
    show_plot=False,
):
    config = load_config(config_path)
    run_config = deepcopy(config)
    task_log_dir = Path(task["task_log_dir"])
    task_log_dir.mkdir(parents=True, exist_ok=True)
    run_config["paths"]["log_dir"] = str(task_log_dir)
    run_config["static_scheduler_ga"]["pop_size"] = int(task["pop_size"])
    run_config["static_scheduler_ga"]["ngen"] = int(task["ngen"])
    run_config["static_scheduler_ga"]["cxpb"] = float(task["fixed_cxpb"])
    run_config["static_scheduler_ga"]["mutpb"] = float(task["fixed_mutpb"])
    if benchmark_repeats is not None:
        run_config["static_scheduler_ga"]["benchmark_repeats"] = int(benchmark_repeats)
    for mutation_name in (
        "task_order_probability",
        "processor_allocation_probability",
        "message_priority_shuffle_probability",
        "message_path_index_probability",
    ):
        run_config["scheduler"]["mutation"][mutation_name] = float(
            task["shared_scheduler_mutation_probability"]
        )

    try:
        result_payload = run_static_scheduler_benchmark(
            input_json_path=Path(task["input_json_path"]),
            config=run_config,
            show_plot=show_plot,
            console_summary=False,
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

    return {
        "status": "completed",
        "run_record": build_pair_run_record(result_payload, task),
    }


def run_internal_task_mode(task_file, result_file):
    payload = load_json(task_file)
    outcome = execute_pair_sweep_task(
        payload["task"],
        payload["config_path"],
        benchmark_repeats=payload.get("benchmark_repeats"),
        show_plot=payload.get("show_plot", False),
    )
    write_json(result_file, outcome)


def execute_pair_sweep_task_subprocess(
    task,
    *,
    config_path,
    benchmark_repeats,
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
            "benchmark_repeats": benchmark_repeats,
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
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
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


def build_best_pair_summary(run_records):
    grouped = {}
    for record in run_records:
        group_key = (
            record["instance_name"],
            record["budget"],
            record["fixed_cxpb"],
            record["fixed_mutpb"],
            record["shared_scheduler_mutation_probability"],
        )
        grouped.setdefault(group_key, []).append(record)

    rows = []
    for (
        instance_name,
        budget,
        fixed_cxpb,
        fixed_mutpb,
        shared_scheduler_mutation_probability,
    ), records in sorted(grouped.items()):
        best_record = min(
            records,
            key=lambda record: (
                record["mean_makespan"],
                record["std_makespan"],
                record["avg_runtime_seconds"],
                record["pop_size"],
                record["ngen"],
            ),
        )
        rows.append(
            {
                "instance_name": instance_name,
                "budget": budget,
                "fixed_cxpb": fixed_cxpb,
                "fixed_mutpb": fixed_mutpb,
                "shared_scheduler_mutation_probability": shared_scheduler_mutation_probability,
                "fixed_parameter_combo_label": best_record["fixed_parameter_combo_label"],
                "feasible_pair_count": best_record["feasible_pair_count"],
                "best_pop_size": best_record["pop_size"],
                "best_ngen": best_record["ngen"],
                "best_mean_makespan": best_record["mean_makespan"],
                "best_std_makespan": best_record["std_makespan"],
                "best_avg_runtime_seconds": best_record["avg_runtime_seconds"],
                "best_total_runtime_seconds": best_record["total_runtime_seconds"],
            }
        )
    return rows


def build_best_fixed_params_per_pair_summary(run_records):
    grouped = {}
    for record in run_records:
        group_key = (
            record["instance_name"],
            record["budget"],
            record["pop_size"],
            record["ngen"],
        )
        grouped.setdefault(group_key, []).append(record)

    rows = []
    for (instance_name, budget, pop_size, ngen), records in sorted(grouped.items()):
        best_record = min(
            records,
            key=lambda record: (
                record["mean_makespan"],
                record["std_makespan"],
                record["avg_runtime_seconds"],
                record["fixed_cxpb"],
                record["fixed_mutpb"],
                record["shared_scheduler_mutation_probability"],
            ),
        )
        rows.append(
            {
                "instance_name": instance_name,
                "budget": budget,
                "pop_size": pop_size,
                "ngen": ngen,
                "feasible_pair_count": best_record["feasible_pair_count"],
                "best_fixed_cxpb": best_record["fixed_cxpb"],
                "best_fixed_mutpb": best_record["fixed_mutpb"],
                "best_shared_scheduler_mutation_probability": (
                    best_record["shared_scheduler_mutation_probability"]
                ),
                "best_fixed_parameter_combo_label": (
                    best_record["fixed_parameter_combo_label"]
                ),
                "best_mean_makespan": best_record["mean_makespan"],
                "best_std_makespan": best_record["std_makespan"],
                "best_avg_runtime_seconds": best_record["avg_runtime_seconds"],
                "best_total_runtime_seconds": best_record["total_runtime_seconds"],
            }
        )
    return rows


def plot_best_pair_metric(
    best_pair_rows,
    *,
    metric_key,
    save_path,
    title,
    ylabel,
    show_plot=False,
):
    if not best_pair_rows:
        return None

    grouped = {}
    for row in best_pair_rows:
        grouped.setdefault(
            (row["instance_name"], row["fixed_parameter_combo_label"]),
            [],
        ).append(row)

    save_path = Path(save_path)
    fig, axis = plt.subplots(figsize=(10.8, 5.8))
    for (instance_name, combo_label), rows in sorted(grouped.items()):
        sorted_rows = sorted(rows, key=lambda row: row["budget"])
        axis.plot(
            [row["budget"] for row in sorted_rows],
            [row[metric_key] for row in sorted_rows],
            marker="o",
            linewidth=1.8,
            label=f"{instance_name} | {combo_label}",
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


def build_batch_summary(
    *,
    config,
    input_json_paths,
    budget_metadata_rows,
    fixed_parameter_grid,
    benchmark_repeats,
    parallel_workers,
    run_records,
    failed_runs,
    output_paths,
    show_plot=False,
):
    best_fixed_params_per_pair_rows = build_best_fixed_params_per_pair_summary(run_records)
    best_pair_rows = build_best_pair_summary(run_records)
    plot_paths = {
        "best_pair_makespan": plot_best_pair_metric(
            best_pair_rows,
            metric_key="best_mean_makespan",
            save_path=output_paths["best_pair_makespan_plot_path"],
            title="Best feasible pair mean makespan vs exact fair budget",
            ylabel="Best feasible pair mean makespan",
            show_plot=show_plot,
        ),
        "best_pair_runtime": plot_best_pair_metric(
            best_pair_rows,
            metric_key="best_avg_runtime_seconds",
            save_path=output_paths["best_pair_runtime_plot_path"],
            title="Best feasible pair runtime vs exact fair budget",
            ylabel="Best feasible pair average runtime (seconds)",
            show_plot=show_plot,
        ),
    }

    write_csv(output_paths["pair_results_csv_path"], run_records)
    write_csv(
        output_paths["best_fixed_params_per_pair_csv_path"],
        best_fixed_params_per_pair_rows,
    )
    write_csv(output_paths["best_pair_summary_csv_path"], best_pair_rows)

    return {
        "description": (
            "Exhaustive fair-budget scheduler-GA sweep. For each exact budget, every feasible "
            "(pop_size, ngen) pair is evaluated directly under fixed inner-GA probabilities."
        ),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config_path": config["_meta"]["config_path"],
        "batch_name": output_paths["batch_name"],
        "batch_run_dir": str(output_paths["run_dir"]),
        "input_instances": [str(Path(path).resolve()) for path in input_json_paths],
        "budget_metadata": budget_metadata_rows,
        "fixed_parameter_grid": fixed_parameter_grid,
        "benchmark_repeats": benchmark_repeats,
        "parallel_workers": parallel_workers,
        "planned_run_count": len(run_records) + len(failed_runs),
        "completed_run_count": len(run_records),
        "failed_run_count": len(failed_runs),
        "failed_runs": failed_runs,
        "pair_results_csv_path": str(output_paths["pair_results_csv_path"]),
        "best_fixed_params_per_pair_csv_path": str(
            output_paths["best_fixed_params_per_pair_csv_path"]
        ),
        "best_pair_summary_csv_path": str(output_paths["best_pair_summary_csv_path"]),
        "runs": run_records,
        "best_fixed_params_per_pair_summary": best_fixed_params_per_pair_rows,
        "best_pair_summary": best_pair_rows,
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
    fixed_parameter_grid = resolve_fixed_parameter_grid(
        config,
        cxpb_values=args.cxpb_values,
        mutpb_values=args.mutpb_values,
        shared_mutation_values=args.shared_mutation_values,
    )
    budget_metadata_rows = build_budget_metadata(config, args.budgets)
    output_paths = create_budget_pair_sweep_output_paths(config, label=args.label)
    tasks = build_pair_sweep_tasks(
        input_json_paths,
        budget_metadata_rows,
        fixed_parameter_grid,
        output_paths["runs_dir"],
    )

    run_records = []
    failed_runs = []

    print("Budget-pair sweep batch folder:", output_paths["run_dir"])
    print("Budgets to test:", [row["budget"] for row in budget_metadata_rows])
    print(
        "Fixed parameter combinations:",
        [row["fixed_parameter_combo_label"] for row in fixed_parameter_grid],
    )
    print("Parallel workers:", args.parallel_workers)
    print("Planned run count:", len(tasks))

    if args.parallel_workers == 1:
        for task in tasks:
            print(
                "Starting exhaustive budget-pair run:",
                {
                    "instance": task["instance_name"],
                    "budget": task["budget"],
                    "pop_size": task["pop_size"],
                    "ngen": task["ngen"],
                    "combo": task["fixed_parameter_combo_label"],
                    "task_index": task["task_index"],
                },
            )
            outcome = execute_pair_sweep_task(
                task,
                config["_meta"]["config_path"],
                benchmark_repeats=args.benchmark_repeats,
                show_plot=args.show_plot,
            )
            if outcome["status"] == "completed":
                run_record = outcome["run_record"]
                run_records.append(run_record)
                print(
                    "Completed exhaustive budget-pair run:",
                    {
                        "instance": run_record["instance_name"],
                        "budget": run_record["budget"],
                        "pop_size": run_record["pop_size"],
                        "ngen": run_record["ngen"],
                        "combo": run_record["fixed_parameter_combo_label"],
                        "mean_makespan": round(run_record["mean_makespan"], 3),
                        "avg_runtime_seconds": round(run_record["avg_runtime_seconds"], 6),
                    },
                )
            else:
                failed_run = outcome["failed_run"]
                failed_runs.append(failed_run)
                print(
                    "Exhaustive budget-pair run failed:",
                    {
                        "instance": failed_run["instance_name"],
                        "budget": failed_run["budget"],
                        "pop_size": failed_run["pop_size"],
                        "ngen": failed_run["ngen"],
                        "combo": failed_run["fixed_parameter_combo_label"],
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
                    execute_pair_sweep_task_subprocess,
                    task,
                    config_path=config["_meta"]["config_path"],
                    benchmark_repeats=args.benchmark_repeats,
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
                        "Exhaustive budget-pair worker crashed:",
                        {
                            "instance": failed_run["instance_name"],
                            "budget": failed_run["budget"],
                            "pop_size": failed_run["pop_size"],
                            "ngen": failed_run["ngen"],
                            "combo": failed_run["fixed_parameter_combo_label"],
                            "task_index": failed_run["task_index"],
                            "error": failed_run["error"],
                        },
                    )
                    continue

                if outcome["status"] == "completed":
                    run_record = outcome["run_record"]
                    run_records.append(run_record)
                    print(
                        "Completed exhaustive budget-pair run:",
                        {
                            "instance": run_record["instance_name"],
                            "budget": run_record["budget"],
                            "pop_size": run_record["pop_size"],
                            "ngen": run_record["ngen"],
                            "combo": run_record["fixed_parameter_combo_label"],
                            "mean_makespan": round(run_record["mean_makespan"], 3),
                            "avg_runtime_seconds": round(run_record["avg_runtime_seconds"], 6),
                        },
                    )
                else:
                    failed_run = outcome["failed_run"]
                    failed_runs.append(failed_run)
                    print(
                        "Exhaustive budget-pair run failed:",
                        {
                            "instance": failed_run["instance_name"],
                            "budget": failed_run["budget"],
                            "pop_size": failed_run["pop_size"],
                            "ngen": failed_run["ngen"],
                            "combo": failed_run["fixed_parameter_combo_label"],
                            "task_index": failed_run["task_index"],
                            "error": failed_run["error"],
                        },
                    )
        try:
            worker_dir.rmdir()
        except OSError:
            pass

    run_records.sort(key=sort_pair_run_key)
    failed_runs.sort(key=sort_pair_run_key)

    summary_payload = build_batch_summary(
        config=config,
        input_json_paths=input_json_paths,
        budget_metadata_rows=budget_metadata_rows,
        fixed_parameter_grid=fixed_parameter_grid,
        benchmark_repeats=args.benchmark_repeats,
        parallel_workers=args.parallel_workers,
        run_records=run_records,
        failed_runs=failed_runs,
        output_paths=output_paths,
        show_plot=args.show_plot,
    )
    write_json(output_paths["summary_path"], summary_payload)

    print("Budget-pair sweep summary saved to:", output_paths["summary_path"].resolve())
    for plot_name, plot_path in summary_payload["plot_paths"].items():
        if plot_path is not None:
            print(f"{plot_name} plot saved to:", plot_path)
