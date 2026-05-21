"""Evaluate budget-wise best scheduler settings on unseen DAG variants."""

import argparse
import copy
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev

from SchedulerGa import run_static_scheduler_benchmark
from ga_config import DEFAULT_CONFIG_PATH, load_config


MUTATION_KEYS = [
    "task_order_probability",
    "processor_allocation_probability",
    "message_priority_shuffle_probability",
    "message_path_index_probability",
]


def write_json(path, payload):
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
        handle.write("\n")


def write_csv(path, rows):
    path = Path(path)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_csv_rows(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def instance_name_from_path(path):
    return Path(path).stem


def task_count_from_input(input_path):
    with Path(input_path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return len(payload["application"]["jobs"])


def group_best_settings_by_instance(best_rows):
    grouped = {}
    for row in best_rows:
        grouped.setdefault(row["instance_name"], []).append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["budget"]))
    return grouped


def run_command(command, *, cwd, log_path=None):
    completed = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if log_path is not None:
        Path(log_path).write_text(
            "COMMAND:\n"
            + " ".join(str(part) for part in command)
            + "\n\nSTDOUT:\n"
            + completed.stdout
            + "\n\nSTDERR:\n"
            + completed.stderr,
            encoding="utf-8",
        )
    if completed.returncode != 0:
        raise RuntimeError(
            "Command failed with exit code "
            f"{completed.returncode}: {' '.join(str(part) for part in command)}"
        )
    return completed


def generate_variants(
    *,
    repo_root,
    input_path,
    dag_config_path,
    output_dir,
    num_variants,
    seed,
    mode,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "dag_variant_generator/generate_dag_variants.py",
        "--input",
        str(input_path),
        "--config",
        str(dag_config_path),
        "--output",
        str(output_dir),
        "--num-variants",
        str(num_variants),
        "--mode",
        mode,
        "--seed",
        str(seed),
    ]
    run_command(command, cwd=repo_root, log_path=output_dir / "generation_command.log")
    task_count = task_count_from_input(input_path)
    return sorted(output_dir.glob(f"{task_count}T_variant_*.json"))


def discover_variants(input_path, output_dir):
    task_count = task_count_from_input(input_path)
    return sorted(output_dir.glob(f"{task_count}T_variant_*.json"))


def validate_variants(*, repo_root, variant_paths, validation_log_dir):
    validation_log_dir.mkdir(parents=True, exist_ok=True)
    validation_records = []
    for variant_path in variant_paths:
        log_path = validation_log_dir / f"{variant_path.stem}_validation.txt"
        command = [
            sys.executable,
            "gen_dag_validate/check_dag.py",
            "--input",
            str(variant_path),
        ]
        run_command(command, cwd=repo_root, log_path=log_path)
        validation_records.append(
            {
                "original_instance_name": variant_path.parent.name,
                "variant_name": variant_path.stem,
                "variant_path": str(variant_path.resolve()),
                "validation_log_path": str(log_path.resolve()),
                "validation_status": "passed",
            }
        )
    return validation_records


def build_benchmark_config(base_config, log_dir, shared_mutation):
    config = copy.deepcopy(base_config)
    config["paths"]["log_dir"] = str(Path(log_dir).resolve())
    scheduler_mutation = config["scheduler"]["mutation"]
    for key in MUTATION_KEYS:
        scheduler_mutation[key] = float(shared_mutation)
    return config


def format_float(value, places=6):
    return f"{float(value):.{places}f}"


def format_percent(numerator, denominator, places=3):
    denominator = float(denominator)
    if denominator == 0:
        return ""
    return format_float(100.0 * float(numerator) / denominator, places)


def build_run_record(*, original_instance, variant_path, best_row, result_payload):
    training_mean_makespan = float(best_row["best_mean_makespan"])
    variant_mean_makespan = float(result_payload["mean_makespan"])
    variant_std_makespan = float(result_payload["std_makespan"])
    generalization_gap = variant_mean_makespan - training_mean_makespan
    return {
        "original_instance_name": original_instance,
        "variant_name": variant_path.stem,
        "variant_path": str(variant_path.resolve()),
        "budget": int(best_row["budget"]),
        "pop_size": int(best_row["best_pop_size"]),
        "ngen": int(best_row["best_ngen"]),
        "cxpb": format_float(best_row["best_fixed_cxpb"], 2),
        "mutpb": format_float(best_row["best_fixed_mutpb"], 2),
        "shared_mutation": format_float(best_row["best_shared_mutation"], 2),
        "training_mean_makespan": format_float(training_mean_makespan, 3),
        "training_avg_runtime_seconds": format_float(
            best_row["best_avg_runtime_seconds"], 3
        ),
        "variant_mean_makespan": format_float(variant_mean_makespan, 3),
        "generalization_gap": format_float(generalization_gap, 3),
        "generalization_gap_percent_of_training": format_percent(
            generalization_gap,
            training_mean_makespan,
            3,
        ),
        "variant_std_makespan": format_float(variant_std_makespan, 3),
        "variant_std_makespan_percent_of_training": format_percent(
            variant_std_makespan,
            training_mean_makespan,
            3,
        ),
        "variant_best_makespan": format_float(result_payload["best_makespan"], 3),
        "variant_avg_runtime_seconds": format_float(
            result_payload["avg_runtime_seconds"], 3
        ),
        "variant_std_runtime_seconds": format_float(
            result_payload["std_runtime_seconds"], 3
        ),
        "variant_total_runtime_seconds": format_float(
            result_payload["total_runtime_seconds"], 3
        ),
        "benchmark_repeats": int(result_payload["benchmark_repeats"]),
        "random_seed": result_payload["random_seed"],
        "run_dir": result_payload["run_dir"],
        "results_path": result_payload["results_path"],
    }


def summarize_generalization(run_records):
    grouped = {}
    for record in run_records:
        key = (record["original_instance_name"], int(record["budget"]))
        grouped.setdefault(key, []).append(record)

    rows = []
    for (instance_name, budget), records in sorted(grouped.items()):
        variant_means = [float(record["variant_mean_makespan"]) for record in records]
        variant_runtimes = [
            float(record["variant_avg_runtime_seconds"]) for record in records
        ]
        training_mean = float(records[0]["training_mean_makespan"])
        variant_mean_average = mean(variant_means)
        variant_mean_std = pstdev(variant_means)
        generalization_gap = variant_mean_average - training_mean
        rows.append(
            {
                "original_instance_name": instance_name,
                "budget": budget,
                "variant_count": len(records),
                "pop_size": records[0]["pop_size"],
                "ngen": records[0]["ngen"],
                "cxpb": records[0]["cxpb"],
                "mutpb": records[0]["mutpb"],
                "shared_mutation": records[0]["shared_mutation"],
                "training_mean_makespan": format_float(training_mean, 3),
                "variant_mean_makespan_mean": format_float(variant_mean_average, 3),
                "generalization_gap_mean": format_float(generalization_gap, 3),
                "generalization_gap_mean_percent_of_training": format_percent(
                    generalization_gap,
                    training_mean,
                    3,
                ),
                "variant_mean_makespan_std_across_dags": format_float(variant_mean_std, 3),
                "variant_mean_makespan_std_across_dags_percent_of_training": format_percent(
                    variant_mean_std,
                    training_mean,
                    3,
                ),
                "variant_mean_makespan_worst": format_float(max(variant_means), 3),
                "variant_avg_runtime_seconds_mean": format_float(
                    mean(variant_runtimes), 3
                ),
                "variant_avg_runtime_seconds_worst": format_float(
                    max(variant_runtimes), 3
                ),
            }
        )
    return rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Generate unseen DAG variants and evaluate the budget-wise best "
            "scheduler settings learned on the original instances."
        )
    )
    parser.add_argument(
        "--best-by-budget-csv",
        required=True,
        type=Path,
        help="CSV containing one best configuration per instance and budget.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Scheduler config path. Default: {DEFAULT_CONFIG_PATH.name}.",
    )
    parser.add_argument(
        "--dag-config",
        type=Path,
        default=Path("dag_variant_generator/dag_variant_config.json"),
        help="DAG generator config path.",
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        default=[Path("example_50T.json"), Path("example_70T.json")],
        help="Original benchmark JSON files to generate variants from.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: logs/generalization_eval_<timestamp>.",
    )
    parser.add_argument(
        "--num-variants",
        type=int,
        default=3,
        help="Number of DAG variants to generate per input instance.",
    )
    parser.add_argument(
        "--generation-mode",
        choices=["application_only", "platform_only", "both"],
        default="application_only",
        help="DAG variant generator mode.",
    )
    parser.add_argument(
        "--variant-seed-base",
        type=int,
        default=90000,
        help="Base seed for deterministic variant generation.",
    )
    parser.add_argument(
        "--benchmark-repeats",
        type=int,
        default=3,
        help="Repeated scheduler runs per variant/configuration.",
    )
    parser.add_argument(
        "--benchmark-seed-base",
        type=int,
        default=120000,
        help="Base seed for deterministic scheduler benchmark repeats.",
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Reuse existing variants in the output variant folders.",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip validator execution after variant generation.",
    )
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Generate and validate variants, then stop before benchmarks.",
    )
    parser.add_argument(
        "--show-plot",
        action="store_true",
        help="Display benchmark plots after saving when an interactive backend is available.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    repo_root = Path.cwd()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else Path("logs") / f"generalization_eval_{timestamp}"
    )
    output_dir = output_dir.resolve()
    variants_root = output_dir / "variants"
    validation_root = output_dir / "validation_logs"
    runs_root = output_dir / "runs"
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_root.mkdir(parents=True, exist_ok=True)

    best_rows = load_csv_rows(args.best_by_budget_csv)
    best_by_instance = group_best_settings_by_instance(best_rows)
    base_config = load_config(args.config)

    all_variant_records = []
    all_validation_records = []
    variant_paths_by_instance = {}

    for input_index, input_path in enumerate(args.inputs):
        input_path = input_path.resolve()
        instance_name = instance_name_from_path(input_path)
        if instance_name not in best_by_instance:
            raise ValueError(
                f"No best-by-budget rows found for input instance {instance_name}."
            )

        instance_variant_dir = variants_root / instance_name
        seed = args.variant_seed_base + input_index * 1000 + task_count_from_input(input_path)
        if args.skip_generation:
            variant_paths = discover_variants(input_path, instance_variant_dir)
        else:
            variant_paths = generate_variants(
                repo_root=repo_root,
                input_path=input_path,
                dag_config_path=args.dag_config,
                output_dir=instance_variant_dir,
                num_variants=args.num_variants,
                seed=seed,
                mode=args.generation_mode,
            )
        if len(variant_paths) != args.num_variants:
            raise RuntimeError(
                f"Expected {args.num_variants} variants for {instance_name}, "
                f"found {len(variant_paths)}."
            )
        variant_paths_by_instance[instance_name] = variant_paths
        for variant_path in variant_paths:
            all_variant_records.append(
                {
                    "original_instance_name": instance_name,
                    "variant_name": variant_path.stem,
                    "variant_path": str(variant_path.resolve()),
                }
            )

        if not args.skip_validation:
            all_validation_records.extend(
                validate_variants(
                    repo_root=repo_root,
                    variant_paths=variant_paths,
                    validation_log_dir=validation_root / instance_name,
                )
            )

    write_csv(output_dir / "generalization_variants.csv", all_variant_records)
    write_csv(output_dir / "generalization_validation.csv", all_validation_records)

    if args.generate_only:
        print("Generated variants and validation records in:", output_dir)
        return

    run_records = []
    task_index = 0
    for instance_name, variant_paths in sorted(variant_paths_by_instance.items()):
        settings_rows = best_by_instance[instance_name]
        for variant_path in variant_paths:
            for best_row in settings_rows:
                task_index += 1
                budget = int(best_row["budget"])
                task_log_dir = runs_root / instance_name / variant_path.stem / f"budget_{budget}"
                config = build_benchmark_config(
                    base_config,
                    task_log_dir,
                    best_row["best_shared_mutation"],
                )
                result_payload = run_static_scheduler_benchmark(
                    input_json_path=variant_path,
                    config=config,
                    pop_size=int(best_row["best_pop_size"]),
                    cxpb=float(best_row["best_fixed_cxpb"]),
                    mutpb=float(best_row["best_fixed_mutpb"]),
                    ngen=int(best_row["best_ngen"]),
                    benchmark_repeats=args.benchmark_repeats,
                    random_seed=args.benchmark_seed_base + task_index,
                    show_plot=args.show_plot,
                    console_summary=False,
                )
                run_record = build_run_record(
                    original_instance=instance_name,
                    variant_path=variant_path,
                    best_row=best_row,
                    result_payload=result_payload,
                )
                run_records.append(run_record)
                print(
                    "Completed generalization run:",
                    {
                        "instance": instance_name,
                        "variant": variant_path.stem,
                        "budget": budget,
                        "mean_makespan": run_record["variant_mean_makespan"],
                        "avg_runtime": run_record["variant_avg_runtime_seconds"],
                    },
                )

    summary_rows = summarize_generalization(run_records)
    write_csv(output_dir / "generalization_run_results.csv", run_records)
    write_csv(output_dir / "generalization_summary.csv", summary_rows)
    write_json(
        output_dir / "generalization_summary.json",
        {
            "description": (
                "Generalization evaluation of budget-wise best scheduler settings "
                "on unseen generated DAG variants."
            ),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "best_by_budget_csv": str(args.best_by_budget_csv.resolve()),
            "scheduler_config": str(args.config.resolve()),
            "dag_config": str(args.dag_config.resolve()),
            "output_dir": str(output_dir),
            "num_variants_per_instance": args.num_variants,
            "benchmark_repeats": args.benchmark_repeats,
            "variant_records": all_variant_records,
            "validation_records": all_validation_records,
            "run_records": run_records,
            "summary_rows": summary_rows,
        },
    )
    print("Generalization run results:", (output_dir / "generalization_run_results.csv").resolve())
    print("Generalization summary:", (output_dir / "generalization_summary.csv").resolve())
    print("Generalization summary JSON:", (output_dir / "generalization_summary.json").resolve())


if __name__ == "__main__":
    main()
