"""Public scheduler-GA API plus a static baseline benchmark runner."""

import argparse
import math
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path
from statistics import mean, median, pstdev, stdev, variance
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.stats import t as student_t
from scipy.stats import ttest_ind

from GAImplementation import (
    NEW_GA_V2 as _run_legacy_scheduler_ga,
    get_scheduler_settings,
    load_problem,
)
from ga_config import DEFAULT_CONFIG_PATH, load_config, resolve_config_path


def run_scheduler_ga(
    processor_ids,
    processing_times,
    message_list,
    all_path_indexes_with_costs,
    pop_size,
    cxpb,
    mutpb,
    ngen,
    random_seed=None,
    scheduler_config=None,
    return_generation_history=False,
):
    """Run the scheduler GA once with one complete, explicit configuration."""
    return _run_legacy_scheduler_ga(
        processor_ids,
        processing_times,
        message_list,
        all_path_indexes_with_costs,
        pop_size=pop_size,
        cxpb=cxpb,
        mutpb=mutpb,
        ngen=ngen,
        random_seed=random_seed,
        scheduler_config=scheduler_config,
        return_generation_history=return_generation_history,
    )


# Keep the old function name available for existing imports while new code can use
# the clearer run_scheduler_ga name.
NEW_GA_V2 = run_scheduler_ga


def get_static_scheduler_ga_settings(config):
    return config["static_scheduler_ga"]


def choose_runtime_setting(configured_value, override_value):
    return configured_value if override_value is None else override_value


def resolve_configured_benchmark_repeats(static_settings):
    """Support the clearer benchmark_repeats key while accepting legacy repeats."""
    if "benchmark_repeats" in static_settings:
        return static_settings["benchmark_repeats"]
    if "repeats" in static_settings:
        return static_settings["repeats"]
    raise KeyError("static_scheduler_ga must define benchmark_repeats or repeats.")


def resolve_static_scheduler_runtime_settings(
    config,
    *,
    pop_size=None,
    cxpb=None,
    mutpb=None,
    ngen=None,
    benchmark_repeats=None,
    repeats=None,
    random_seed=None,
    show_plot=None,
):
    """Resolve fixed scheduler settings from config plus optional CLI overrides."""
    static_settings = get_static_scheduler_ga_settings(config)
    configured_benchmark_repeats = resolve_configured_benchmark_repeats(static_settings)
    requested_benchmark_repeats = (
        benchmark_repeats if benchmark_repeats is not None else repeats
    )
    resolved_benchmark_repeats = choose_runtime_setting(
        configured_benchmark_repeats,
        requested_benchmark_repeats,
    )
    return {
        "pop_size": choose_runtime_setting(static_settings["pop_size"], pop_size),
        "cxpb": choose_runtime_setting(static_settings["cxpb"], cxpb),
        "mutpb": choose_runtime_setting(static_settings["mutpb"], mutpb),
        "ngen": choose_runtime_setting(static_settings["ngen"], ngen),
        "benchmark_repeats": resolved_benchmark_repeats,
        # Keep the legacy alias available so older code paths still work.
        "repeats": resolved_benchmark_repeats,
        "random_seed": choose_runtime_setting(static_settings["random_seed"], random_seed),
        "show_plot": choose_runtime_setting(static_settings["show_plot"], show_plot),
    }


def safe_run_name(name):
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")
    return safe_name or "unknown_am"


def extract_instance_name_from_text(text):
    match = re.search(r"(example_\d+T)", str(text))
    return match.group(1) if match else None


def extract_instance_name_from_path(path):
    path = Path(path)
    candidates = [path.name, path.stem, path.parent.name]
    for candidate in candidates:
        instance_name = extract_instance_name_from_text(candidate)
        if instance_name is not None:
            return instance_name
    return path.stem


def create_static_benchmark_output_paths(am_name, config, log_dir=None):
    """Create a timestamped output folder for one static scheduler benchmark."""
    safe_am_name = safe_run_name(am_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    configured_log_dir = log_dir if log_dir is not None else config["paths"]["log_dir"]
    run_name = f"{safe_am_name}_standalone_scheduler_ga_{timestamp}"
    run_dir = resolve_config_path(config, configured_log_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    return {
        "run_dir": run_dir,
        "log_path": run_dir / f"{run_name}.log",
        "repeat_log_path": run_dir / f"{run_name}_runs.jsonl",
        "results_path": run_dir / f"{run_name}_results.json",
        "plot_path": run_dir / f"{run_name}.png",
        "repeat_plot_path": run_dir / f"{run_name}_repeat_summary.png",
    }


def create_comparison_batch_output_paths(config, log_dir=None):
    """Create a dedicated batch folder for standalone-vs-nested comparisons."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    configured_log_dir = log_dir if log_dir is not None else config["paths"]["log_dir"]
    run_name = f"ga_comparison_batch_{timestamp}"
    run_dir = resolve_config_path(config, configured_log_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return {
        "run_dir": run_dir,
        "summary_path": run_dir / f"{run_name}_summary.json",
    }


def append_jsonl(path, record):
    with Path(path).open("a", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=True)
        handle.write("\n")


def write_json(path, payload):
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def maybe_show_plot(show_plot):
    if show_plot and matplotlib.get_backend().lower() != "agg":
        plt.show()


def standard_deviation(values):
    if not values:
        return 0.0
    return pstdev(values)


def sample_standard_deviation(values):
    if len(values) < 2:
        return 0.0
    return stdev(values)


def sample_variance(values):
    if len(values) < 2:
        return 0.0
    return variance(values)


def compute_summary_statistics(values):
    """Return descriptive statistics that are useful for GA benchmarking."""
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "range": None,
            "population_std": 0.0,
            "sample_std": 0.0,
            "sample_variance": 0.0,
            "coefficient_of_variation": None,
            "ci95_low": None,
            "ci95_high": None,
        }

    value_count = len(values)
    value_mean = mean(values)
    value_median = median(values)
    value_min = min(values)
    value_max = max(values)
    population_std = standard_deviation(values)
    sample_std = sample_standard_deviation(values)
    coefficient_of_variation = None if value_mean == 0 else sample_std / value_mean

    if value_count > 1:
        sem = sample_std / math.sqrt(value_count)
        t_critical = student_t.ppf(0.975, df=value_count - 1)
        ci95_low = value_mean - (t_critical * sem)
        ci95_high = value_mean + (t_critical * sem)
    else:
        ci95_low = value_mean
        ci95_high = value_mean

    return {
        "count": value_count,
        "mean": value_mean,
        "median": value_median,
        "min": value_min,
        "max": value_max,
        "range": value_max - value_min,
        "population_std": population_std,
        "sample_std": sample_std,
        "sample_variance": sample_variance(values),
        "coefficient_of_variation": coefficient_of_variation,
        "ci95_low": ci95_low,
        "ci95_high": ci95_high,
    }


def serialize_schedule(schedule):
    serialized = {}
    for task_id, (processor, start_time, end_time, path_info) in sorted(schedule.items()):
        serialized[str(task_id)] = {
            "processor": processor,
            "start_time": start_time,
            "end_time": end_time,
            "predecessor_paths": [
                {
                    "sender": sender,
                    "path_id": path_id,
                    "message_id": message_id,
                }
                for sender, path_id, message_id in path_info
            ],
        }
    return serialized


def serialize_genome(genome):
    if genome is None:
        return None

    task_order, processor_allocation, message_priority_ordering, message_path_index = genome
    return {
        "task_order": list(task_order),
        "processor_allocation": list(processor_allocation),
        "message_priority_ordering": list(message_priority_ordering),
        "message_path_index": list(message_path_index),
    }


def summarize_schedule(schedule):
    if not schedule:
        return {
            "task_count": 0,
            "makespan": None,
            "processor_summary": {},
        }

    processor_summary = {}
    for _, (processor, start_time, end_time, _) in schedule.items():
        processor_key = str(processor)
        stats = processor_summary.setdefault(
            processor_key,
            {"task_count": 0, "busy_time": 0.0, "last_end_time": 0.0},
        )
        stats["task_count"] += 1
        stats["busy_time"] += end_time - start_time
        stats["last_end_time"] = max(stats["last_end_time"], end_time)

    return {
        "task_count": len(schedule),
        "makespan": max(end_time for _, (_, _, end_time, _) in schedule.items()),
        "processor_summary": processor_summary,
    }


def build_repeat_seed(base_seed, repeat_index):
    if base_seed is None:
        return None
    return (base_seed * 1000) + repeat_index


def get_benchmark_repeat_records(results_payload):
    """Load benchmark records from new or legacy standalone result files."""
    benchmark_records = results_payload.get("benchmark_repeat_records")
    if benchmark_records is not None:
        return benchmark_records
    return results_payload.get("repeat_records", [])


def build_generation_summary_statistics(benchmark_repeat_records):
    """Aggregate generation-wise makespans across standalone benchmark runs."""
    if not benchmark_repeat_records:
        return []

    generation_histories = [
        record.get("generation_history", [])
        for record in benchmark_repeat_records
        if record.get("generation_history")
    ]
    if not generation_histories:
        return []

    shared_length = min(len(history) for history in generation_histories)
    generation_statistics = []
    for index in range(shared_length):
        generation = generation_histories[0][index]["generation"]
        best_values = [
            history[index]["generation_best_makespan"]
            for history in generation_histories
        ]
        avg_values = [
            history[index]["generation_avg_makespan"]
            for history in generation_histories
        ]
        worst_values = [
            history[index]["generation_worst_makespan"]
            for history in generation_histories
        ]
        generation_statistics.append(
            {
                "generation": generation,
                "mean_generation_best_makespan": mean(best_values),
                "std_generation_best_makespan": standard_deviation(best_values),
                "min_generation_best_makespan": min(best_values),
                "max_generation_best_makespan": max(best_values),
                "mean_generation_avg_makespan": mean(avg_values),
                "mean_generation_worst_makespan": mean(worst_values),
            }
        )
    return generation_statistics


def plot_static_scheduler_repeat_summary(results_payload, save_path, show_plot=False):
    benchmark_repeat_records = get_benchmark_repeat_records(results_payload)
    if not benchmark_repeat_records:
        return None

    save_path = Path(save_path)
    repeat_indexes = [record["repeat_index"] for record in benchmark_repeat_records]
    makespans = [record["makespan"] for record in benchmark_repeat_records]
    mean_makespan = results_payload["makespan_statistics"]["mean"]
    std_makespan = results_payload["makespan_statistics"]["sample_std"]

    fig, axis = plt.subplots(figsize=(10, 4.8))
    axis.plot(
        repeat_indexes,
        makespans,
        marker="o",
        linewidth=1.8,
        color="tab:blue",
        label="Static scheduler repeat",
    )
    axis.axhline(
        mean_makespan,
        color="tab:red",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean makespan = {mean_makespan:.2f}",
    )
    axis.fill_between(
        repeat_indexes,
        [mean_makespan - std_makespan] * len(repeat_indexes),
        [mean_makespan + std_makespan] * len(repeat_indexes),
        color="tab:red",
        alpha=0.15,
        label=f"Mean ± std = {std_makespan:.2f}",
    )
    axis.set_xlabel("Benchmark run index")
    axis.set_ylabel("Makespan")
    axis.set_title("Standalone scheduler GA benchmark")
    axis.set_xticks(repeat_indexes)
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best")
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    maybe_show_plot(show_plot)
    plt.close(fig)
    return save_path.resolve()


def plot_static_scheduler_generation_convergence(results_payload, save_path, show_plot=False):
    """Plot generation-best makespan per benchmark run plus the mean curve."""
    benchmark_repeat_records = get_benchmark_repeat_records(results_payload)
    generation_statistics = results_payload.get("generation_statistics", [])
    if not benchmark_repeat_records or not generation_statistics:
        return None

    save_path = Path(save_path)
    generations = [row["generation"] for row in generation_statistics]
    mean_generation_best = [
        row["mean_generation_best_makespan"]
        for row in generation_statistics
    ]
    std_generation_best = [
        row["std_generation_best_makespan"]
        for row in generation_statistics
    ]

    fig, axis = plt.subplots(figsize=(10.5, 5.2))
    run_label_used = False
    for repeat_record in benchmark_repeat_records:
        generation_history = repeat_record.get("generation_history", [])
        if not generation_history:
            continue
        run_generations = [row["generation"] for row in generation_history]
        run_best = [row["generation_best_makespan"] for row in generation_history]
        axis.plot(
            run_generations,
            run_best,
            linewidth=1.1,
            alpha=0.35,
            color="tab:gray",
            label="Per-run generation best" if not run_label_used else None,
        )
        run_label_used = True

    axis.plot(
        generations,
        mean_generation_best,
        linewidth=2.4,
        color="tab:blue",
        label="Mean generation best across runs",
    )
    axis.fill_between(
        generations,
        [mean_value - std_value for mean_value, std_value in zip(mean_generation_best, std_generation_best)],
        [mean_value + std_value for mean_value, std_value in zip(mean_generation_best, std_generation_best)],
        color="tab:blue",
        alpha=0.18,
        label="Mean generation best ± std",
    )
    axis.set_xlabel("Generation")
    axis.set_ylabel("Makespan")
    axis.set_title("Standalone GA makespan over generations")
    axis.grid(True, alpha=0.3)
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis.legend(loc="best")

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    maybe_show_plot(show_plot)
    plt.close(fig)
    return save_path.resolve()


def resolve_nested_validation_results_path(nested_path):
    """Resolve a generic nested-GA path into its validation_results.json file."""
    nested_path = Path(nested_path)
    if nested_path.is_dir():
        candidate_paths = sorted(nested_path.glob("*_validation_results.json"))
        if len(candidate_paths) != 1:
            raise FileNotFoundError(
                "Expected exactly one *_validation_results.json inside "
                f"{nested_path}, found {len(candidate_paths)}."
            )
        return candidate_paths[0]

    if nested_path.name.endswith("_validation_results.json"):
        return nested_path

    raise ValueError(
        "Nested comparison paths must point to a run directory or directly to a "
        "*_validation_results.json file."
    )


def resolve_nested_best_result_path(nested_path):
    """Resolve a generic nested-GA path into its best_result.json file."""
    nested_path = Path(nested_path)
    if nested_path.is_dir():
        candidate_paths = sorted(nested_path.glob("*_best_result.json"))
        if len(candidate_paths) != 1:
            raise FileNotFoundError(
                "Expected exactly one *_best_result.json inside "
                f"{nested_path}, found {len(candidate_paths)}."
            )
        return candidate_paths[0]

    if nested_path.name.endswith("_best_result.json"):
        return nested_path

    if nested_path.name.endswith("_validation_results.json"):
        nested_path = nested_path.parent
        return resolve_nested_best_result_path(nested_path)

    raise ValueError(
        "Nested training comparison paths must point to a run directory or directly "
        "to a *_best_result.json file."
    )


def resolve_standalone_results_path(standalone_path):
    """Resolve a standalone benchmark path into its *_results.json file."""
    standalone_path = Path(standalone_path)
    if standalone_path.is_dir():
        candidate_paths = sorted(standalone_path.glob("*_results.json"))
        if len(candidate_paths) != 1:
            raise FileNotFoundError(
                "Expected exactly one *_results.json inside "
                f"{standalone_path}, found {len(candidate_paths)}."
            )
        return candidate_paths[0]

    if standalone_path.name.endswith("_results.json"):
        return standalone_path

    raise ValueError(
        "Standalone comparison paths must point to a standalone run directory or "
        "directly to a *_results.json file."
    )


def load_standalone_results_payload(standalone_path):
    """Load an existing standalone result file so comparisons can be run later."""
    results_path = resolve_standalone_results_path(standalone_path)
    payload = load_json(results_path)
    benchmark_repeat_records = get_benchmark_repeat_records(payload)
    makespans = [record["makespan"] for record in benchmark_repeat_records]
    runtimes = [record["runtime_seconds"] for record in benchmark_repeat_records]

    payload["instance_name"] = payload.get("instance_name") or extract_instance_name_from_path(
        payload.get("input_json_path", results_path)
    )
    payload["run_name"] = payload.get("run_name") or results_path.parent.name
    payload["run_dir"] = payload.get("run_dir") or str(results_path.parent.resolve())
    payload["results_path"] = payload.get("results_path") or str(results_path.resolve())
    payload["benchmark_repeat_records"] = benchmark_repeat_records
    payload.setdefault("repeat_records", benchmark_repeat_records)
    payload["benchmark_repeats"] = payload.get(
        "benchmark_repeats",
        payload.get("repeats", len(benchmark_repeat_records)),
    )
    payload.setdefault("repeats", payload["benchmark_repeats"])
    if "makespan_statistics" not in payload:
        payload["makespan_statistics"] = compute_summary_statistics(makespans)
    if "runtime_statistics" not in payload:
        payload["runtime_statistics"] = compute_summary_statistics(runtimes)
    payload["mean_makespan"] = payload.get("mean_makespan", payload["makespan_statistics"]["mean"])
    payload["std_makespan"] = payload.get("std_makespan", payload["makespan_statistics"]["population_std"])
    payload["avg_runtime_seconds"] = payload.get(
        "avg_runtime_seconds",
        payload["runtime_statistics"]["mean"],
    )
    return payload


def load_nested_validation_candidate(nested_path, candidate_rank=1):
    """Load one tuned candidate from a generic nested-GA validation artifact."""
    validation_results_path = resolve_nested_validation_results_path(nested_path)
    validation_payload = load_json(validation_results_path)
    candidate_record = next(
        (
            result
            for result in validation_payload["results"]
            if result["candidate_rank"] == candidate_rank
        ),
        None,
    )
    if candidate_record is None:
        candidate_record = min(
            validation_payload["results"],
            key=lambda result: result["validation_mean_makespan"],
        )

    return {
        "source_mode": "validation",
        "source_label": (
            f"validation_candidate_rank_{candidate_record['candidate_rank']}"
        ),
        "source_path": str(validation_results_path.resolve()),
        "validation_results_path": str(validation_results_path.resolve()),
        "nested_run_name": validation_results_path.parent.name,
        "instance_name": extract_instance_name_from_path(validation_results_path),
        "candidate_rank": candidate_record["candidate_rank"],
        "candidate_record": candidate_record,
    }


def load_nested_training_candidate(nested_path):
    """Load the nested-GA search-time selected best candidate and its training repeats."""
    best_result_path = resolve_nested_best_result_path(nested_path)
    best_record = load_json(best_result_path)
    return {
        "source_mode": "training",
        "source_label": "training_selected_best",
        "source_path": str(best_result_path.resolve()),
        "best_result_path": str(best_result_path.resolve()),
        "nested_run_name": best_result_path.parent.name,
        "instance_name": extract_instance_name_from_path(best_result_path),
        "candidate_rank": None,
        "candidate_record": best_record,
    }


def load_nested_candidate(nested_path, nested_source="validation", candidate_rank=1):
    """Load one nested-GA candidate from the chosen behavior source."""
    if nested_source == "validation":
        return load_nested_validation_candidate(
            nested_path,
            candidate_rank=candidate_rank,
        )
    if nested_source == "training":
        return load_nested_training_candidate(nested_path)

    raise ValueError(
        f"Unsupported nested comparison source: {nested_source}. "
        "Expected 'validation' or 'training'."
    )


def compute_cohens_d(sample_a, sample_b):
    if len(sample_a) < 2 or len(sample_b) < 2:
        return None

    pooled_denominator = (
        ((len(sample_a) - 1) * sample_variance(sample_a))
        + ((len(sample_b) - 1) * sample_variance(sample_b))
    )
    pooled_denominator /= (len(sample_a) + len(sample_b) - 2)
    if pooled_denominator <= 0:
        return 0.0
    return (mean(sample_a) - mean(sample_b)) / math.sqrt(pooled_denominator)


def compute_welch_ttest(sample_a, sample_b):
    if len(sample_a) < 2 or len(sample_b) < 2:
        return {
            "available": False,
            "reason": "Welch t-test needs at least 2 samples in each group.",
        }

    two_sided = ttest_ind(sample_a, sample_b, equal_var=False, nan_policy="omit")
    try:
        one_sided = ttest_ind(
            sample_a,
            sample_b,
            equal_var=False,
            nan_policy="omit",
            alternative="less",
        )
        one_sided_pvalue = float(one_sided.pvalue)
    except TypeError:
        if two_sided.statistic < 0:
            one_sided_pvalue = float(two_sided.pvalue) / 2.0
        else:
            one_sided_pvalue = 1.0 - (float(two_sided.pvalue) / 2.0)

    return {
        "available": True,
        "t_statistic": float(two_sided.statistic),
        "two_sided_pvalue": float(two_sided.pvalue),
        "one_sided_pvalue_tuned_better": one_sided_pvalue,
    }


def build_static_vs_nested_comparison(
    static_results_payload,
    nested_path,
    nested_source="validation",
    candidate_rank=1,
):
    """Create a generic tuned-vs-standalone comparison payload for one instance."""
    nested_payload = load_nested_candidate(
        nested_path,
        nested_source=nested_source,
        candidate_rank=candidate_rank,
    )
    nested_candidate = nested_payload["candidate_record"]

    static_benchmark_records = get_benchmark_repeat_records(static_results_payload)
    static_makespans = [record["makespan"] for record in static_benchmark_records]
    static_runtimes = [record["runtime_seconds"] for record in static_benchmark_records]
    if nested_source == "validation":
        nested_repeat_records = nested_candidate["validation_runs"]
    else:
        nested_repeat_records = nested_candidate["repeats"]

    tuned_makespans = [record["makespan"] for record in nested_repeat_records]
    tuned_runtimes = [record["runtime_seconds"] for record in nested_repeat_records]

    static_makespan_statistics = compute_summary_statistics(static_makespans)
    tuned_makespan_statistics = compute_summary_statistics(tuned_makespans)
    static_runtime_statistics = compute_summary_statistics(static_runtimes)
    tuned_runtime_statistics = compute_summary_statistics(tuned_runtimes)

    mean_makespan_improvement = (
        static_makespan_statistics["mean"] - tuned_makespan_statistics["mean"]
    )
    mean_runtime_change = (
        tuned_runtime_statistics["mean"] - static_runtime_statistics["mean"]
    )

    if static_makespan_statistics["mean"] in (None, 0):
        relative_makespan_improvement = None
    else:
        relative_makespan_improvement = (
            100.0 * mean_makespan_improvement / static_makespan_statistics["mean"]
        )

    if static_runtime_statistics["mean"] in (None, 0):
        relative_runtime_change = None
    else:
        relative_runtime_change = (
            100.0 * mean_runtime_change / static_runtime_statistics["mean"]
        )

    return {
        "comparison_type": "tuned_nested_ga_vs_standalone_scheduler_benchmark",
        "nested_source_mode": nested_payload["source_mode"],
        "nested_source_label": nested_payload["source_label"],
        "metric_direction": "lower_makespan_is_better",
        "instance_name": static_results_payload["instance_name"],
        "static_run_name": static_results_payload["run_name"],
        "static_results_path": static_results_payload["results_path"],
        "standalone_benchmark_results_path": static_results_payload["results_path"],
        "nested_source_path": nested_payload["source_path"],
        "nested_validation_results_path": nested_payload.get("validation_results_path"),
        "nested_best_result_path": nested_payload.get("best_result_path"),
        "nested_run_name": nested_payload["nested_run_name"],
        "nested_candidate_rank": nested_payload["candidate_rank"],
        "static_hyperparameters": static_results_payload["fixed_hyperparameters"],
        "nested_hyperparameters": nested_candidate["hyperparameters"],
        "standalone_benchmark_repeats": static_results_payload.get(
            "benchmark_repeats",
            len(static_benchmark_records),
        ),
        "nested_repeat_count": len(nested_repeat_records),
        "nested_repeat_record_source": (
            "validation_runs" if nested_source == "validation" else "training_repeats"
        ),
        "standalone_benchmark_makespan_statistics": static_makespan_statistics,
        "nested_candidate_makespan_statistics": tuned_makespan_statistics,
        "standalone_benchmark_runtime_statistics": static_runtime_statistics,
        "nested_candidate_runtime_statistics": tuned_runtime_statistics,
        "standalone_makespan_statistics": static_makespan_statistics,
        "tuned_makespan_statistics": tuned_makespan_statistics,
        "standalone_runtime_statistics": static_runtime_statistics,
        "tuned_runtime_statistics": tuned_runtime_statistics,
        "makespan_repeat_values": {
            "standalone_benchmark": static_makespans,
            "nested_candidate": tuned_makespans,
        },
        "runtime_repeat_values": {
            "standalone_benchmark": static_runtimes,
            "nested_candidate": tuned_runtimes,
        },
        "mean_makespan_improvement_of_tuned": mean_makespan_improvement,
        "relative_makespan_improvement_percent_of_tuned": relative_makespan_improvement,
        "mean_runtime_change_of_tuned_seconds": mean_runtime_change,
        "relative_runtime_change_percent_of_tuned": relative_runtime_change,
        "stability_change_sample_std": (
            static_makespan_statistics["sample_std"]
            - tuned_makespan_statistics["sample_std"]
        ),
        "welch_ttest_makespan": compute_welch_ttest(
            tuned_makespans,
            static_makespans,
        ),
        "cohens_d_tuned_minus_standalone": compute_cohens_d(
            tuned_makespans,
            static_makespans,
        ),
    }


def plot_static_vs_nested_comparison(comparison_payload, save_path, show_plot=False):
    """Plot makespan and runtime tradeoffs for a generic tuned-vs-static comparison."""
    save_path = Path(save_path)
    nested_label = f"Tuned nested ({comparison_payload['nested_source_mode']})"
    makespan_groups = [
        comparison_payload["makespan_repeat_values"]["standalone_benchmark"],
        comparison_payload["makespan_repeat_values"]["nested_candidate"],
    ]
    runtime_groups = [
        comparison_payload["runtime_repeat_values"]["standalone_benchmark"],
        comparison_payload["runtime_repeat_values"]["nested_candidate"],
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    axes[0].boxplot(makespan_groups, labels=["Standalone benchmark", nested_label])
    axes[0].set_ylabel("Makespan")
    axes[0].set_title("Makespan distribution")
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].boxplot(runtime_groups, labels=["Standalone benchmark", nested_label])
    axes[1].set_ylabel("Runtime (seconds)")
    axes[1].set_title("Runtime distribution")
    axes[1].grid(True, axis="y", alpha=0.3)

    ttest_payload = comparison_payload["welch_ttest_makespan"]
    if ttest_payload["available"]:
        ttest_summary = (
            f"Welch t-test (tuned nested < standalone benchmark)\n"
            f"p={ttest_payload['one_sided_pvalue_tuned_better']:.4g}\n"
            f"mean improvement={comparison_payload['mean_makespan_improvement_of_tuned']:.2f}"
        )
    else:
        ttest_summary = ttest_payload["reason"]

    fig.suptitle(
        (
            f"{comparison_payload['instance_name']}: tuned nested GA "
            f"({comparison_payload['nested_source_mode']}) vs standalone benchmark"
        ),
        fontsize=14,
    )
    fig.text(
        0.5,
        0.01,
        ttest_summary,
        ha="center",
        va="bottom",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9},
    )

    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(save_path, dpi=200)
    maybe_show_plot(show_plot)
    plt.close(fig)
    return save_path.resolve()


def create_comparison_artifacts(
    static_results_payload,
    nested_path,
    nested_source="validation",
    candidate_rank=1,
    show_plot=False,
    output_dir=None,
):
    """Write one generic tuned-vs-static comparison JSON and PNG into the run folder."""
    comparison_payload = build_static_vs_nested_comparison(
        static_results_payload,
        nested_path,
        nested_source=nested_source,
        candidate_rank=candidate_rank,
    )
    nested_run_safe = safe_run_name(comparison_payload["nested_run_name"])
    run_dir = Path(output_dir) if output_dir is not None else Path(static_results_payload["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    comparison_base_name = (
        f"{static_results_payload['instance_name']}_standalone_vs_"
        f"{comparison_payload['nested_source_mode']}_{nested_run_safe}"
    )
    comparison_json_path = run_dir / f"{comparison_base_name}_comparison.json"
    comparison_plot_path = run_dir / f"{comparison_base_name}_comparison.png"
    write_json(comparison_json_path, comparison_payload)
    resolved_plot_path = plot_static_vs_nested_comparison(
        comparison_payload,
        save_path=comparison_plot_path,
        show_plot=show_plot,
    )
    comparison_payload["comparison_json_path"] = str(comparison_json_path.resolve())
    comparison_payload["comparison_plot_path"] = str(resolved_plot_path)
    write_json(comparison_json_path, comparison_payload)
    return comparison_payload


def run_static_scheduler_benchmark(
    input_json_path=None,
    config=None,
    pop_size=None,
    cxpb=None,
    mutpb=None,
    ngen=None,
    benchmark_repeats=None,
    repeats=None,
    random_seed=None,
    show_plot=None,
    console_summary=True,
):
    """Run the scheduler GA repeatedly with fixed, non-adaptive hyperparameters."""
    config = config or load_config()
    runtime_settings = resolve_static_scheduler_runtime_settings(
        config,
        pop_size=pop_size,
        cxpb=cxpb,
        mutpb=mutpb,
        ngen=ngen,
        benchmark_repeats=benchmark_repeats,
        repeats=repeats,
        random_seed=random_seed,
        show_plot=show_plot,
    )
    scheduler_config = json.loads(json.dumps(get_scheduler_settings(config)))
    problem = load_problem(input_json_path=input_json_path, config=config)

    am_name = problem["INPUT_JSON_PATH"].stem
    output_paths = create_static_benchmark_output_paths(am_name, config=config)
    run_dir = output_paths["run_dir"]
    log_path = output_paths["log_path"]
    repeat_log_path = output_paths["repeat_log_path"]
    results_path = output_paths["results_path"]
    plot_path = output_paths["plot_path"]
    repeat_plot_path = output_paths["repeat_plot_path"]
    repeat_log_path.write_text("", encoding="utf-8")

    repeat_records = []
    with log_path.open("w", encoding="utf-8", buffering=1) as log_file, redirect_stdout(log_file):
        print("Standalone scheduler GA benchmark log")
        print("AM being tested:", am_name)
        print("Input JSON:", problem["INPUT_JSON_PATH"].resolve())
        print("Config file:", config["_meta"]["config_path"])
        print("Timestamp:", datetime.now().isoformat(timespec="seconds"))
        print("Run folder:", run_dir.resolve())
        print("Benchmark run log file:", repeat_log_path.resolve())
        print("Results file:", results_path.resolve())
        print("Generation plot file:", plot_path.resolve())
        print("Repeat summary plot file:", repeat_plot_path.resolve())
        print("Fixed scheduler GA hyperparameters:", {
            "pop_size": runtime_settings["pop_size"],
            "cxpb": runtime_settings["cxpb"],
            "mutpb": runtime_settings["mutpb"],
            "ngen": runtime_settings["ngen"],
        })
        print("Fixed scheduler config:", scheduler_config)
        print("Benchmark repeats:", runtime_settings["benchmark_repeats"])
        print("Base random seed:", runtime_settings["random_seed"])

        for repeat_index in range(1, runtime_settings["benchmark_repeats"] + 1):
            inner_seed = build_repeat_seed(runtime_settings["random_seed"], repeat_index)
            print("\nRunning standalone benchmark run:", {
                "repeat_index": repeat_index,
                "inner_seed": inner_seed,
            })

            run_start = perf_counter()
            makespan, schedule, genome, generation_history = run_scheduler_ga(
                problem["processor_ids"],
                problem["processing_times"],
                problem["message_list"],
                problem["merged_paths_dict"],
                pop_size=runtime_settings["pop_size"],
                cxpb=runtime_settings["cxpb"],
                mutpb=runtime_settings["mutpb"],
                ngen=runtime_settings["ngen"],
                random_seed=inner_seed,
                scheduler_config=scheduler_config,
                return_generation_history=True,
            )
            elapsed_seconds = perf_counter() - run_start
            repeat_record = {
                "repeat_index": repeat_index,
                "inner_seed": inner_seed,
                "makespan": makespan,
                "runtime_seconds": elapsed_seconds,
                "generation_history": generation_history,
                "schedule_summary": summarize_schedule(schedule),
                "best_genome": serialize_genome(genome),
                "best_schedule": serialize_schedule(schedule),
            }
            repeat_records.append(repeat_record)
            append_jsonl(repeat_log_path, repeat_record)
            print("Completed standalone benchmark run:", {
                "repeat_index": repeat_index,
                "makespan": makespan,
                "runtime_seconds": round(elapsed_seconds, 6),
            })

        makespans = [record["makespan"] for record in repeat_records]
        runtimes = [record["runtime_seconds"] for record in repeat_records]
        makespan_statistics = compute_summary_statistics(makespans)
        runtime_statistics = compute_summary_statistics(runtimes)
        best_repeat_record = min(repeat_records, key=lambda record: record["makespan"])
        generation_statistics = build_generation_summary_statistics(repeat_records)
        results_payload = {
            "benchmark_type": "standalone_scheduler_ga_benchmark",
            "description": (
                "Static baseline: the scheduler GA is run with fixed hyperparameters "
                "from static_scheduler_ga and scheduler.mutation. No outer GA adapts "
                "these values. The repeated runs estimate stochastic benchmark "
                "performance; they are not a hyperparameter validation phase."
            ),
            "instance_name": am_name,
            "run_name": run_dir.name,
            "run_dir": str(run_dir.resolve()),
            "log_path": str(log_path.resolve()),
            "repeat_log_path": str(repeat_log_path.resolve()),
            "results_path": str(results_path.resolve()),
            "plot_path": str(plot_path.resolve()),
            "input_json_path": str(problem["INPUT_JSON_PATH"].resolve()),
            "config_path": config["_meta"]["config_path"],
            "fixed_hyperparameters": {
                "pop_size": runtime_settings["pop_size"],
                "cxpb": runtime_settings["cxpb"],
                "mutpb": runtime_settings["mutpb"],
                "ngen": runtime_settings["ngen"],
                "selection_tournament_size": scheduler_config["selection"]["tournament_size"],
                **scheduler_config["mutation"],
            },
            "benchmark_repeats": runtime_settings["benchmark_repeats"],
            "repeats": runtime_settings["benchmark_repeats"],
            "random_seed": runtime_settings["random_seed"],
            "seed_strategy": (
                "inner_seed = base_random_seed * 1000 + repeat_index"
                if runtime_settings["random_seed"] is not None
                else "unseeded benchmark runs"
            ),
            "makespan_statistics": makespan_statistics,
            "runtime_statistics": runtime_statistics,
            "mean_makespan": makespan_statistics["mean"],
            "std_makespan": makespan_statistics["population_std"],
            "best_makespan": best_repeat_record["makespan"],
            "best_repeat_index": best_repeat_record["repeat_index"],
            "avg_runtime_seconds": runtime_statistics["mean"],
            "std_runtime_seconds": runtime_statistics["population_std"],
            "total_runtime_seconds": sum(runtimes),
            "generation_statistics": generation_statistics,
            "benchmark_repeat_records": repeat_records,
            "repeat_records": repeat_records,
        }
        write_json(results_path, results_payload)
        resolved_plot_path = plot_static_scheduler_generation_convergence(
            results_payload,
            save_path=plot_path,
            show_plot=runtime_settings["show_plot"],
        )
        resolved_repeat_plot_path = plot_static_scheduler_repeat_summary(
            results_payload,
            save_path=repeat_plot_path,
            show_plot=runtime_settings["show_plot"],
        )
        results_payload["plot_path"] = str(resolved_plot_path)
        results_payload["generation_plot_path"] = str(resolved_plot_path)
        results_payload["repeat_plot_path"] = str(resolved_repeat_plot_path)
        write_json(results_path, results_payload)
        print("\nStandalone benchmark mean makespan:", results_payload["mean_makespan"])
        print("Standalone benchmark std makespan:", results_payload["std_makespan"])
        print("Standalone benchmark best makespan:", results_payload["best_makespan"])
        print("Standalone benchmark generation plot:", resolved_plot_path)
        print("Standalone benchmark repeat summary plot:", resolved_repeat_plot_path)

    if console_summary:
        print("Standalone benchmark run folder:", run_dir.resolve())
        print("Standalone benchmark log saved to:", log_path.resolve())
        print("Standalone benchmark run log saved to:", repeat_log_path.resolve())
        print("Standalone benchmark results saved to:", results_path.resolve())
        print("Standalone benchmark generation plot saved to:", plot_path.resolve())
        print("Standalone benchmark repeat summary plot saved to:", repeat_plot_path.resolve())
        print("Standalone benchmark mean makespan:", results_payload["mean_makespan"])
        print("Standalone benchmark std makespan:", results_payload["std_makespan"])
        print("Standalone benchmark best makespan:", results_payload["best_makespan"])
    return results_payload


def run_static_scheduler_benchmark_worker(task):
    """Worker entry point for parallel standalone benchmarks on Windows."""
    config = load_config(task["config_path"])
    return run_static_scheduler_benchmark(
        input_json_path=task["input_json_path"],
        config=config,
        pop_size=task["pop_size"],
        cxpb=task["cxpb"],
        mutpb=task["mutpb"],
        ngen=task["ngen"],
        benchmark_repeats=task.get("benchmark_repeats", task.get("repeats")),
        random_seed=task["random_seed"],
        show_plot=task["show_plot"],
        console_summary=False,
    )


def resolve_parallel_workers(parallel_workers, input_count):
    if input_count <= 1:
        return 1
    if parallel_workers is not None:
        return max(1, min(parallel_workers, input_count))
    return max(1, min(input_count, os.cpu_count() or input_count))


def match_nested_runs_by_instance(nested_paths):
    mapping = {}
    for nested_path in nested_paths:
        instance_name = extract_instance_name_from_path(nested_path)
        mapping[instance_name] = nested_path
    return mapping


def parse_args():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Config file to load. Default: {DEFAULT_CONFIG_PATH.name}.",
    )
    pre_args, remaining = pre_parser.parse_known_args()
    config = load_config(pre_args.config)
    static_settings = get_static_scheduler_ga_settings(config)
    configured_benchmark_repeats = resolve_configured_benchmark_repeats(static_settings)

    parser = argparse.ArgumentParser(
        description="Run a static, non-adaptive scheduler GA benchmark."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=pre_args.config,
        help=f"Config file to load. Default: {DEFAULT_CONFIG_PATH.name}.",
    )
    parser.add_argument(
        "input_json_paths",
        nargs="*",
        type=Path,
        help="One or more example JSON files, such as example_50T.json example_70T.json.",
    )
    parser.add_argument(
        "--pop-size",
        type=int,
        default=None,
        help=f"Fixed scheduler population size. Config default: {static_settings['pop_size']}.",
    )
    parser.add_argument(
        "--cxpb",
        type=float,
        default=None,
        help=f"Fixed scheduler crossover probability. Config default: {static_settings['cxpb']}.",
    )
    parser.add_argument(
        "--mutpb",
        type=float,
        default=None,
        help=f"Fixed scheduler mutation probability. Config default: {static_settings['mutpb']}.",
    )
    parser.add_argument(
        "--ngen",
        type=int,
        default=None,
        help=f"Fixed scheduler generation count. Config default: {static_settings['ngen']}.",
    )
    parser.add_argument(
        "--benchmark-repeats",
        "--repeats",
        dest="benchmark_repeats",
        type=int,
        default=None,
        help=(
            "Number of independent benchmark runs of the fixed standalone GA. "
            "This estimates stochastic performance and stability; it is not a "
            f"hyperparameter validation phase. Config default: {configured_benchmark_repeats}."
        ),
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Base seed used to derive repeat seeds. Omit to use the config default.",
    )
    parser.add_argument(
        "--show-plot",
        action="store_true",
        help="Display plots after saving them when an interactive backend is available.",
    )
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=None,
        help=(
            "Number of worker processes used when multiple standalone instances are "
            "requested. By default, multiple inputs run in parallel up to the CPU count."
        ),
    )
    parser.add_argument(
        "--compare-nested-runs",
        nargs="*",
        type=Path,
        default=[],
        help=(
            "Optional nested-GA run directories. These are matched generically by "
            "instance name and compared against the standalone benchmark runs."
        ),
    )
    parser.add_argument(
        "--compare-standalone-runs",
        nargs="*",
        type=Path,
        default=[],
        help=(
            "Existing standalone run directories or *_results.json files. Use this "
            "when you want to run the tuned-vs-standalone comparisons later without "
            "rerunning the standalone GA."
        ),
    )
    parser.add_argument(
        "--nested-candidate-rank",
        type=int,
        default=1,
        help=(
            "Which validated nested-GA candidate rank to compare against. "
            "Default: 1 (the tuned candidate the outer GA selected)."
        ),
    )
    args = parser.parse_args(remaining)
    if not args.show_plot:
        args.show_plot = static_settings["show_plot"]
    return args, config


if __name__ == "__main__":
    args, config = parse_args()
    run_new_standalone_benchmarks = bool(args.input_json_paths) or not args.compare_standalone_runs
    if run_new_standalone_benchmarks:
        input_json_paths = args.input_json_paths or [None]
        worker_count = resolve_parallel_workers(args.parallel_workers, len(input_json_paths))
        task_descriptors = [
            {
                "config_path": str(args.config.resolve()),
                "input_json_path": None if input_json_path is None else str(input_json_path),
                "pop_size": args.pop_size,
                "cxpb": args.cxpb,
                "mutpb": args.mutpb,
                "ngen": args.ngen,
                "benchmark_repeats": args.benchmark_repeats,
                "random_seed": args.random_seed,
                "show_plot": args.show_plot,
            }
            for input_json_path in input_json_paths
        ]

        if worker_count == 1:
            static_results_payloads = [
                run_static_scheduler_benchmark_worker(task)
                for task in task_descriptors
            ]
        else:
            static_results_payloads = []
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                future_to_task = {
                    executor.submit(run_static_scheduler_benchmark_worker, task): task
                    for task in task_descriptors
                }
                for future in as_completed(future_to_task):
                    static_results_payloads.append(future.result())
    else:
        static_results_payloads = []

    if args.compare_standalone_runs:
        static_results_payloads.extend(
            load_standalone_results_payload(standalone_path)
            for standalone_path in args.compare_standalone_runs
        )

    static_results_payloads.sort(key=lambda payload: payload["instance_name"])
    comparison_payloads = []
    if args.compare_nested_runs:
        nested_runs_by_instance = match_nested_runs_by_instance(args.compare_nested_runs)
        for static_results_payload in static_results_payloads:
            nested_path = nested_runs_by_instance.get(static_results_payload["instance_name"])
            if nested_path is None:
                continue
            comparison_payload = create_comparison_artifacts(
                static_results_payload,
                nested_path=nested_path,
                candidate_rank=args.nested_candidate_rank,
                show_plot=args.show_plot,
            )
            comparison_payloads.append(comparison_payload)

    for static_results_payload in static_results_payloads:
        print(
            "Standalone benchmark:",
            {
                "instance_name": static_results_payload["instance_name"],
                "run_dir": static_results_payload["run_dir"],
                "results_path": static_results_payload["results_path"],
                "mean_makespan": round(static_results_payload["mean_makespan"], 3),
                "std_makespan": round(static_results_payload["std_makespan"], 3),
            },
        )
    for comparison_payload in comparison_payloads:
        print(
            "Comparison against nested GA:",
            {
                "instance_name": comparison_payload["instance_name"],
                "comparison_json_path": comparison_payload["comparison_json_path"],
                "mean_makespan_improvement_of_tuned": round(
                    comparison_payload["mean_makespan_improvement_of_tuned"],
                    3,
                ),
                "one_sided_pvalue_tuned_better": comparison_payload["welch_ttest_makespan"].get(
                    "one_sided_pvalue_tuned_better"
                ),
            },
        )
