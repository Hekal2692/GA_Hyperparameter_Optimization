import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


HYPERPARAMETER_NAMES = [
    "pop_size",
    "cxpb",
    "mutpb",
    "ngen",
    "task_order_probability",
    "processor_allocation_probability",
    "message_priority_shuffle_probability",
    "message_path_index_probability",
]

SECONDARY_HYPERPARAMETER_NAMES = [
    "cxpb",
    "mutpb",
    "task_order_probability",
    "processor_allocation_probability",
    "message_priority_shuffle_probability",
    "message_path_index_probability",
]


def maybe_show_plot(show_plot):
    if show_plot and matplotlib.get_backend().lower() != "agg":
        plt.show()


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, payload):
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


def discover_evaluation_files(input_paths):
    files = []
    for input_path in input_paths:
        path = Path(input_path)
        if path.is_file() and path.name.endswith("_evaluations.jsonl"):
            files.append(path)
            continue

        if not path.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")

        if path.is_dir():
            files.extend(sorted(path.rglob("*_evaluations.jsonl")))
            continue

        raise ValueError(f"Unsupported path: {path}")

    unique_files = []
    seen = set()
    for path in files:
        resolved = path.resolve()
        if resolved in seen:
            continue
        unique_files.append(path)
        seen.add(resolved)
    return unique_files


def infer_instance_name(path):
    for candidate in [path.name, path.parent.name, path.parent.parent.name]:
        match = re.search(r"(example_\d+T)", candidate)
        if match:
            return match.group(1)
    return "unknown_instance"


def infer_task_metadata(path):
    for parent in path.parents:
        match = re.match(
            r"task_(\d+)__(example_\d+T)__budget_(\d+)__seed_(\d+)",
            parent.name,
        )
        if match:
            return {
                "task_index": int(match.group(1)),
                "instance_name": match.group(2),
                "budget_from_path": int(match.group(3)),
                "outer_seed": int(match.group(4)),
            }
    return {
        "task_index": None,
        "instance_name": infer_instance_name(path),
        "budget_from_path": None,
        "outer_seed": None,
    }


def hyperparameter_signature(hyperparameters):
    return tuple((name, hyperparameters.get(name)) for name in HYPERPARAMETER_NAMES)


def mean_or_none(values):
    return None if not values else mean(values)


def pearson_correlation(x_values, y_values):
    if len(x_values) != len(y_values) or len(x_values) < 2:
        return None

    x_mean = mean(x_values)
    y_mean = mean(y_values)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
    x_denominator = math.sqrt(sum((x - x_mean) ** 2 for x in x_values))
    y_denominator = math.sqrt(sum((y - y_mean) ** 2 for y in y_values))
    if x_denominator == 0 or y_denominator == 0:
        return None
    return numerator / (x_denominator * y_denominator)


def load_records(evaluation_files, instance_filters=None, budget_filters=None):
    records = []
    for evaluation_path in evaluation_files:
        run_metadata = infer_task_metadata(evaluation_path)
        with evaluation_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                instance_name = run_metadata["instance_name"]
                budget = record.get(
                    "target_computation_budget",
                    run_metadata["budget_from_path"],
                )

                if instance_filters and instance_name not in instance_filters:
                    continue
                if budget_filters and budget not in budget_filters:
                    continue

                flattened = {
                    "evaluation_path": str(evaluation_path.resolve()),
                    "instance_name": instance_name,
                    "budget": budget,
                    "outer_seed": run_metadata["outer_seed"],
                    "task_index": run_metadata["task_index"],
                    "evaluation_id": record["evaluation_id"],
                    "generation": record.get("generation"),
                    "population_index": record.get("population_index"),
                    "stage": record.get("stage"),
                    "actual_computation_budget": record.get("actual_computation_budget"),
                    "mean_makespan": record.get("mean_makespan"),
                    "avg_inner_runtime_seconds": record.get("avg_inner_runtime_seconds"),
                    "weighted_objective_score": record.get("weighted_objective_score"),
                }
                hyperparameters = record["hyperparameters"]
                for name in HYPERPARAMETER_NAMES:
                    flattened[name] = hyperparameters.get(name)
                records.append(flattened)
    return records


def build_unique_combo_rows(records):
    grouped = {}
    for record in records:
        signature = (
            record["instance_name"],
            record["budget"],
            hyperparameter_signature({name: record[name] for name in HYPERPARAMETER_NAMES}),
        )
        bucket = grouped.setdefault(
            signature,
            {
                "instance_name": record["instance_name"],
                "budget": record["budget"],
                "evaluation_count": 0,
                "outer_seeds": set(),
                "mean_makespans": [],
                "avg_inner_runtime_seconds_values": [],
                "weighted_objective_scores": [],
                "hyperparameters": {
                    name: record[name]
                    for name in HYPERPARAMETER_NAMES
                },
            },
        )
        bucket["evaluation_count"] += 1
        if record["outer_seed"] is not None:
            bucket["outer_seeds"].add(record["outer_seed"])
        if record["mean_makespan"] is not None:
            bucket["mean_makespans"].append(record["mean_makespan"])
        if record["avg_inner_runtime_seconds"] is not None:
            bucket["avg_inner_runtime_seconds_values"].append(
                record["avg_inner_runtime_seconds"]
            )
        if record["weighted_objective_score"] is not None:
            bucket["weighted_objective_scores"].append(record["weighted_objective_score"])

    rows = []
    for bucket in grouped.values():
        row = {
            "instance_name": bucket["instance_name"],
            "budget": bucket["budget"],
            "evaluation_count": bucket["evaluation_count"],
            "outer_seed_count": len(bucket["outer_seeds"]),
            "mean_makespan_mean": mean_or_none(bucket["mean_makespans"]),
            "mean_makespan_best": min(bucket["mean_makespans"]) if bucket["mean_makespans"] else None,
            "avg_inner_runtime_seconds_mean": mean_or_none(
                bucket["avg_inner_runtime_seconds_values"]
            ),
            "avg_inner_runtime_seconds_best": min(bucket["avg_inner_runtime_seconds_values"])
            if bucket["avg_inner_runtime_seconds_values"]
            else None,
            "weighted_objective_score_mean": mean_or_none(bucket["weighted_objective_scores"]),
            "weighted_objective_score_best": min(bucket["weighted_objective_scores"])
            if bucket["weighted_objective_scores"]
            else None,
        }
        row.update(bucket["hyperparameters"])
        rows.append(row)

    rows.sort(
        key=lambda row: (
            row["instance_name"],
            row["budget"],
            row["pop_size"],
            row["ngen"],
            row["weighted_objective_score_best"]
            if row["weighted_objective_score_best"] is not None
            else float("inf"),
        )
    )
    return rows


def build_pair_summary(records, unique_combo_rows):
    raw_groups = defaultdict(list)
    unique_groups = defaultdict(list)

    for record in records:
        key = (record["instance_name"], record["budget"], record["pop_size"], record["ngen"])
        raw_groups[key].append(record)

    for row in unique_combo_rows:
        key = (row["instance_name"], row["budget"], row["pop_size"], row["ngen"])
        unique_groups[key].append(row)

    pair_rows = []
    for key in sorted(raw_groups):
        instance_name, budget, pop_size, ngen = key
        raw_rows = raw_groups[key]
        unique_rows = unique_groups[key]

        secondary_ranges = {}
        for name in SECONDARY_HYPERPARAMETER_NAMES:
            values = sorted({row[name] for row in unique_rows if row[name] is not None})
            if not values:
                secondary_ranges[name] = None
            else:
                secondary_ranges[name] = {
                    "min": values[0],
                    "max": values[-1],
                    "unique_value_count": len(values),
                }

        pair_rows.append(
            {
                "instance_name": instance_name,
                "budget": budget,
                "pop_size": pop_size,
                "ngen": ngen,
                "raw_evaluation_count": len(raw_rows),
                "unique_combo_count": len(unique_rows),
                "mean_makespan_mean_raw": mean_or_none(
                    [row["mean_makespan"] for row in raw_rows if row["mean_makespan"] is not None]
                ),
                "mean_makespan_best_raw": min(
                    [row["mean_makespan"] for row in raw_rows if row["mean_makespan"] is not None],
                    default=None,
                ),
                "runtime_mean_raw": mean_or_none(
                    [
                        row["avg_inner_runtime_seconds"]
                        for row in raw_rows
                        if row["avg_inner_runtime_seconds"] is not None
                    ]
                ),
                "runtime_best_raw": min(
                    [
                        row["avg_inner_runtime_seconds"]
                        for row in raw_rows
                        if row["avg_inner_runtime_seconds"] is not None
                    ],
                    default=None,
                ),
                "weighted_objective_score_best_raw": min(
                    [
                        row["weighted_objective_score"]
                        for row in raw_rows
                        if row["weighted_objective_score"] is not None
                    ],
                    default=None,
                ),
                "secondary_hyperparameter_ranges": secondary_ranges,
            }
        )
    return pair_rows


def build_generation_summary(records):
    grouped = defaultdict(list)
    for record in records:
        key = (
            record["instance_name"],
            record["budget"],
            record["outer_seed"],
            record["generation"],
        )
        grouped[key].append(record)

    rows = []
    for key in sorted(grouped):
        instance_name, budget, outer_seed, generation = key
        raw_rows = grouped[key]
        pair_counter = Counter((row["pop_size"], row["ngen"]) for row in raw_rows)
        unique_combo_signatures = {
            hyperparameter_signature({name: row[name] for name in HYPERPARAMETER_NAMES})
            for row in raw_rows
        }
        top_pair = None
        top_pair_count = 0
        if pair_counter:
            top_pair, top_pair_count = max(
                pair_counter.items(),
                key=lambda item: (item[1], -item[0][0], -item[0][1]),
            )

        makespans = [row["mean_makespan"] for row in raw_rows if row["mean_makespan"] is not None]
        runtimes = [
            row["avg_inner_runtime_seconds"]
            for row in raw_rows
            if row["avg_inner_runtime_seconds"] is not None
        ]
        weighted_scores = [
            row["weighted_objective_score"]
            for row in raw_rows
            if row["weighted_objective_score"] is not None
        ]

        rows.append(
            {
                "instance_name": instance_name,
                "budget": budget,
                "outer_seed": outer_seed,
                "generation": generation,
                "evaluation_count": len(raw_rows),
                "unique_pair_count": len(pair_counter),
                "unique_combo_count": len(unique_combo_signatures),
                "generation_mean_makespan_mean": mean_or_none(makespans),
                "generation_best_mean_makespan": min(makespans) if makespans else None,
                "generation_mean_runtime_seconds": mean_or_none(runtimes),
                "generation_best_runtime_seconds": min(runtimes) if runtimes else None,
                "generation_best_weighted_objective_score": (
                    min(weighted_scores) if weighted_scores else None
                ),
                "top_pair_pop_size": top_pair[0] if top_pair else None,
                "top_pair_ngen": top_pair[1] if top_pair else None,
                "top_pair_evaluation_count": top_pair_count,
                "pair_presence": sorted(
                    [{"pop_size": pair[0], "ngen": pair[1], "count": count}
                     for pair, count in pair_counter.items()],
                    key=lambda item: (item["pop_size"], item["ngen"]),
                ),
            }
        )
    return rows


def build_generation_pair_summary(records):
    grouped = defaultdict(list)
    for record in records:
        key = (
            record["instance_name"],
            record["budget"],
            record["outer_seed"],
            record["generation"],
            record["pop_size"],
            record["ngen"],
        )
        grouped[key].append(record)

    rows = []
    for key in sorted(grouped):
        instance_name, budget, outer_seed, generation, pop_size, ngen = key
        raw_rows = grouped[key]
        unique_combo_signatures = {
            hyperparameter_signature({name: row[name] for name in HYPERPARAMETER_NAMES})
            for row in raw_rows
        }
        makespans = [row["mean_makespan"] for row in raw_rows if row["mean_makespan"] is not None]
        runtimes = [
            row["avg_inner_runtime_seconds"]
            for row in raw_rows
            if row["avg_inner_runtime_seconds"] is not None
        ]
        weighted_scores = [
            row["weighted_objective_score"]
            for row in raw_rows
            if row["weighted_objective_score"] is not None
        ]
        rows.append(
            {
                "instance_name": instance_name,
                "budget": budget,
                "outer_seed": outer_seed,
                "generation": generation,
                "pop_size": pop_size,
                "ngen": ngen,
                "evaluation_count": len(raw_rows),
                "unique_combo_count": len(unique_combo_signatures),
                "mean_makespan_mean": mean_or_none(makespans),
                "mean_makespan_best": min(makespans) if makespans else None,
                "runtime_mean": mean_or_none(runtimes),
                "runtime_best": min(runtimes) if runtimes else None,
                "weighted_objective_score_best": min(weighted_scores) if weighted_scores else None,
            }
        )
    return rows


def build_correlation_summary(unique_combo_rows):
    metrics = ["mean_makespan_mean", "avg_inner_runtime_seconds_mean"]
    summary = {"overall": {}, "by_budget": {}}

    def correlation_block(rows):
        block = {}
        for metric_name in metrics:
            metric_values = [row[metric_name] for row in rows if row[metric_name] is not None]
            if len(metric_values) < 2:
                block[metric_name] = {}
                continue

            metric_rows = [row for row in rows if row[metric_name] is not None]
            block[metric_name] = {}
            for hyperparameter_name in HYPERPARAMETER_NAMES:
                x_values = [row[hyperparameter_name] for row in metric_rows]
                y_values = [row[metric_name] for row in metric_rows]
                block[metric_name][hyperparameter_name] = pearson_correlation(
                    x_values,
                    y_values,
                )
        return block

    summary["overall"] = correlation_block(unique_combo_rows)

    budgets = sorted({row["budget"] for row in unique_combo_rows})
    for budget in budgets:
        budget_rows = [row for row in unique_combo_rows if row["budget"] == budget]
        summary["by_budget"][str(budget)] = correlation_block(budget_rows)

    return summary


def write_csv(path, rows, fieldnames):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def flatten_pair_rows_for_csv(pair_rows):
    flattened_rows = []
    for row in pair_rows:
        flattened = {
            key: row[key]
            for key in [
                "instance_name",
                "budget",
                "pop_size",
                "ngen",
                "raw_evaluation_count",
                "unique_combo_count",
                "mean_makespan_mean_raw",
                "mean_makespan_best_raw",
                "runtime_mean_raw",
                "runtime_best_raw",
                "weighted_objective_score_best_raw",
            ]
        }
        for name in SECONDARY_HYPERPARAMETER_NAMES:
            range_payload = row["secondary_hyperparameter_ranges"].get(name)
            if range_payload is None:
                flattened[f"{name}_min"] = None
                flattened[f"{name}_max"] = None
                flattened[f"{name}_unique_value_count"] = 0
            else:
                flattened[f"{name}_min"] = range_payload["min"]
                flattened[f"{name}_max"] = range_payload["max"]
                flattened[f"{name}_unique_value_count"] = range_payload["unique_value_count"]
        flattened_rows.append(flattened)
    return flattened_rows


def plot_pair_metric(pair_rows, metric_key, title, ylabel, save_path, show_plot=False):
    if not pair_rows:
        return None

    grouped = defaultdict(list)
    for row in pair_rows:
        grouped[(row["instance_name"], row["budget"])].append(row)

    figure_height = max(4.5, 3.2 * len(grouped))
    fig, axes = plt.subplots(len(grouped), 1, figsize=(13.5, figure_height), squeeze=False)

    for axis, ((instance_name, budget), rows) in zip(axes.flatten(), sorted(grouped.items())):
        rows = sorted(rows, key=lambda row: (row["pop_size"], row["ngen"]))
        labels = [f"{row['pop_size']}x{row['ngen']}" for row in rows]
        values = [row[metric_key] for row in rows]
        axis.plot(labels, values, marker="o", linewidth=1.8, color="tab:blue")
        axis.set_title(f"{instance_name} | budget={budget}")
        axis.set_ylabel(ylabel)
        axis.grid(True, axis="y", alpha=0.3)
        axis.tick_params(axis="x", rotation=35)

    axes[-1][0].set_xlabel("Fair exact (pop_size x ngen) pair")
    fig.suptitle(title, fontsize=15)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    maybe_show_plot(show_plot)
    plt.close(fig)
    return Path(save_path).resolve()


def plot_mutation_scatter(unique_combo_rows, metric_key, title, save_path, show_plot=False):
    if not unique_combo_rows:
        return None

    mutation_names = [
        "task_order_probability",
        "processor_allocation_probability",
        "message_priority_shuffle_probability",
        "message_path_index_probability",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.2))
    budget_values = sorted({row["budget"] for row in unique_combo_rows})
    color_map = plt.get_cmap("tab10")
    budget_to_color = {
        budget: color_map(index % 10)
        for index, budget in enumerate(budget_values)
    }

    for axis, mutation_name in zip(axes.flatten(), mutation_names):
        for budget in budget_values:
            rows = [row for row in unique_combo_rows if row["budget"] == budget]
            axis.scatter(
                [row[mutation_name] for row in rows],
                [row[metric_key] for row in rows],
                alpha=0.65,
                s=28,
                color=budget_to_color[budget],
                label=f"budget={budget}",
            )
        axis.set_title(mutation_name)
        axis.set_xlabel("Probability")
        axis.set_ylabel(metric_key)
        axis.grid(True, alpha=0.3)

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(4, len(handles)))
    fig.suptitle(title, fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(save_path, dpi=200)
    maybe_show_plot(show_plot)
    plt.close(fig)
    return Path(save_path).resolve()


def plot_generation_pair_diversity(generation_rows, save_path, show_plot=False):
    if not generation_rows:
        return None

    grouped = defaultdict(list)
    for row in generation_rows:
        key = (row["instance_name"], row["budget"], row["outer_seed"])
        grouped[key].append(row)

    figure_height = max(4.5, 3.0 * len(grouped))
    fig, axes = plt.subplots(len(grouped), 1, figsize=(13.5, figure_height), squeeze=False)

    for axis, key in zip(axes.flatten(), sorted(grouped)):
        instance_name, budget, outer_seed = key
        rows = sorted(grouped[key], key=lambda row: row["generation"])
        axis.plot(
            [row["generation"] for row in rows],
            [row["unique_pair_count"] for row in rows],
            marker="o",
            linewidth=1.8,
            color="tab:purple",
        )
        axis.set_title(f"{instance_name} | budget={budget} | outer_seed={outer_seed}")
        axis.set_ylabel("Unique fair pairs")
        axis.grid(True, alpha=0.3)

    axes[-1][0].set_xlabel("Generation")
    fig.suptitle("Generation-level fair-pair diversity", fontsize=15)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    maybe_show_plot(show_plot)
    plt.close(fig)
    return Path(save_path).resolve()


def create_output_dir(base_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(base_dir) / f"hyperparameter_relation_analysis_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze outer-GA evaluation logs to study how computation-budget pairs and "
            "other hyperparameters relate to makespan and runtime."
        )
    )
    parser.add_argument(
        "input_paths",
        nargs="+",
        help=(
            "One or more budget-sweep directories, outer-GA run directories, or "
            "*_evaluations.jsonl files."
        ),
    )
    parser.add_argument(
        "--instance",
        action="append",
        default=[],
        help="Filter to one or more instance names such as example_70T.",
    )
    parser.add_argument(
        "--budget",
        type=int,
        action="append",
        default=[],
        help="Filter to one or more exact computation budgets.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("logs"),
        help="Base directory where the analysis output folder will be created.",
    )
    parser.add_argument(
        "--show-plot",
        action="store_true",
        help="Display plots after saving them when an interactive backend is available.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    evaluation_files = discover_evaluation_files(args.input_paths)
    records = load_records(
        evaluation_files,
        instance_filters=set(args.instance) if args.instance else None,
        budget_filters=set(args.budget) if args.budget else None,
    )
    if not records:
        raise SystemExit("No evaluation records matched the requested filters.")

    unique_combo_rows = build_unique_combo_rows(records)
    pair_rows = build_pair_summary(records, unique_combo_rows)
    generation_rows = build_generation_summary(records)
    generation_pair_rows = build_generation_pair_summary(records)
    correlation_summary = build_correlation_summary(unique_combo_rows)

    output_dir = create_output_dir(args.output_dir)

    pair_plot_path = plot_pair_metric(
        pair_rows,
        metric_key="mean_makespan_mean_raw",
        title="Mean makespan by fair computation-budget pair",
        ylabel="Mean makespan",
        save_path=output_dir / "pair_makespan.png",
        show_plot=args.show_plot,
    )
    runtime_plot_path = plot_pair_metric(
        pair_rows,
        metric_key="runtime_mean_raw",
        title="Mean runtime by fair computation-budget pair",
        ylabel="Seconds",
        save_path=output_dir / "pair_runtime.png",
        show_plot=args.show_plot,
    )
    mutation_makespan_plot_path = plot_mutation_scatter(
        unique_combo_rows,
        metric_key="mean_makespan_mean",
        title="Mutation probabilities vs mean makespan",
        save_path=output_dir / "mutation_vs_makespan.png",
        show_plot=args.show_plot,
    )
    mutation_runtime_plot_path = plot_mutation_scatter(
        unique_combo_rows,
        metric_key="avg_inner_runtime_seconds_mean",
        title="Mutation probabilities vs mean runtime",
        save_path=output_dir / "mutation_vs_runtime.png",
        show_plot=args.show_plot,
    )
    generation_pair_diversity_plot_path = plot_generation_pair_diversity(
        generation_rows,
        save_path=output_dir / "generation_pair_diversity.png",
        show_plot=args.show_plot,
    )

    flattened_pair_rows = flatten_pair_rows_for_csv(pair_rows)
    pair_csv_fields = list(flattened_pair_rows[0].keys()) if flattened_pair_rows else []
    unique_combo_csv_fields = list(unique_combo_rows[0].keys()) if unique_combo_rows else []
    generation_csv_fields = [
        "instance_name",
        "budget",
        "outer_seed",
        "generation",
        "evaluation_count",
        "unique_pair_count",
        "unique_combo_count",
        "generation_mean_makespan_mean",
        "generation_best_mean_makespan",
        "generation_mean_runtime_seconds",
        "generation_best_runtime_seconds",
        "generation_best_weighted_objective_score",
        "top_pair_pop_size",
        "top_pair_ngen",
        "top_pair_evaluation_count",
    ]
    generation_pair_csv_fields = list(generation_pair_rows[0].keys()) if generation_pair_rows else []

    if flattened_pair_rows:
        write_csv(output_dir / "pair_summary.csv", flattened_pair_rows, pair_csv_fields)
    if unique_combo_rows:
        write_csv(
            output_dir / "unique_combinations.csv",
            unique_combo_rows,
            unique_combo_csv_fields,
        )
    if generation_rows:
        write_csv(
            output_dir / "generation_summary.csv",
            [{key: row[key] for key in generation_csv_fields} for row in generation_rows],
            generation_csv_fields,
        )
    if generation_pair_rows:
        write_csv(
            output_dir / "generation_pair_summary.csv",
            generation_pair_rows,
            generation_pair_csv_fields,
        )

    summary_payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_paths": [str(Path(path).resolve()) for path in args.input_paths],
        "filters": {
            "instances": args.instance,
            "budgets": args.budget,
        },
        "evaluation_file_count": len(evaluation_files),
        "raw_record_count": len(records),
        "unique_hyperparameter_combination_count": len(unique_combo_rows),
        "pair_summary": pair_rows,
        "generation_summary": generation_rows,
        "correlation_summary": correlation_summary,
        "plot_paths": {
            "pair_makespan": str(pair_plot_path) if pair_plot_path else None,
            "pair_runtime": str(runtime_plot_path) if runtime_plot_path else None,
            "mutation_vs_makespan": (
                str(mutation_makespan_plot_path)
                if mutation_makespan_plot_path
                else None
            ),
            "mutation_vs_runtime": (
                str(mutation_runtime_plot_path)
                if mutation_runtime_plot_path
                else None
            ),
            "generation_pair_diversity": (
                str(generation_pair_diversity_plot_path)
                if generation_pair_diversity_plot_path
                else None
            ),
        },
    }
    write_json(output_dir / "analysis_summary.json", summary_payload)

    print("Analysis output folder:", output_dir.resolve())
    print("Evaluation files scanned:", len(evaluation_files))
    print("Raw evaluation records used:", len(records))
    print("Unique hyperparameter combinations:", len(unique_combo_rows))
    print("Pair summary CSV:", (output_dir / "pair_summary.csv").resolve())
    print("Unique combinations CSV:", (output_dir / "unique_combinations.csv").resolve())
    print("Generation summary CSV:", (output_dir / "generation_summary.csv").resolve())
    print(
        "Generation pair summary CSV:",
        (output_dir / "generation_pair_summary.csv").resolve(),
    )
    print("Summary JSON:", (output_dir / "analysis_summary.json").resolve())
    for plot_name, plot_path in summary_payload["plot_paths"].items():
        if plot_path is not None:
            print(f"{plot_name} plot:", plot_path)


if __name__ == "__main__":
    main()
