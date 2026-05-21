import argparse
import ast
import json
import math
import random
import re
from contextlib import redirect_stdout
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from deap import base, creator, tools

from SchedulerGa import NEW_GA_V2, load_problem
from ga_config import DEFAULT_CONFIG_PATH, load_config, resolve_config_path


OUTER_HYPERPARAMETER_SPECS = (
    ("pop_size", int),
    ("cxpb", float),
    ("mutpb", float),
    ("ngen", int),
    ("task_order_probability", float),
    ("processor_allocation_probability", float),
    ("message_priority_shuffle_probability", float),
    ("message_path_index_probability", float),
)


@dataclass
class OuterGAResult:
    best_hyperparameters: dict
    best_makespan: float
    best_weighted_objective_score: float | None
    best_avg_inner_runtime_seconds: float | None
    history: list[dict]
    evaluation_records: list[dict] | None = None
    best_evaluation_id: int | None = None
    population_evaluation_snapshots: dict[int, list[int]] | None = None
    run_dir: Path | None = None
    log_path: Path | None = None
    plot_path: Path | None = None
    history_path: Path | None = None
    evaluation_log_path: Path | None = None
    best_result_path: Path | None = None
    analysis_plot_paths: dict | None = None
    validation_results_path: Path | None = None


def get_outer_ga_settings(config):
    return config["outer_ga"]


def get_analysis_settings(config):
    return config["analysis"]


def get_search_space(config):
    return config["hyperparameter_search_space"]


def build_inner_scheduler_config(config, hyperparameters):
    """Overlay evolved inner-mutation settings onto the shared scheduler config."""
    scheduler_config = json.loads(json.dumps(config["scheduler"]))
    scheduler_config["mutation"]["task_order_probability"] = hyperparameters[
        "task_order_probability"
    ]
    scheduler_config["mutation"]["processor_allocation_probability"] = hyperparameters[
        "processor_allocation_probability"
    ]
    scheduler_config["mutation"]["message_priority_shuffle_probability"] = hyperparameters[
        "message_priority_shuffle_probability"
    ]
    scheduler_config["mutation"]["message_path_index_probability"] = hyperparameters[
        "message_path_index_probability"
    ]
    return scheduler_config


def choose_runtime_setting(configured_value, override_value):
    """Use the explicit override when present, otherwise keep the config value."""
    return configured_value if override_value is None else override_value


def resolve_random_immigrant_rate(outer_settings, random_immigrant_rate=None):
    """Resolve the immigrant setting to a population fraction."""
    if random_immigrant_rate is not None:
        rate = float(random_immigrant_rate)
    else:
        rate = float(outer_settings["random_immigrant_rate"])

    return min(max(rate, 0.0), 1.0)


def resolve_random_immigrant_count(outer_pop_size, elite_count, random_immigrant_rate):
    """Convert an immigrant rate into a concrete number of immigrants."""
    if random_immigrant_rate <= 0:
        return 0

    immigrant_count = math.ceil(outer_pop_size * random_immigrant_rate)
    return max(0, min(immigrant_count, outer_pop_size - elite_count))


def resolve_outer_objective_weights(outer_settings):
    """Resolve and normalize the weighted outer-GA objective components."""
    configured_weights = outer_settings.get("objective_weights", {})
    makespan_weight = float(configured_weights.get("mean_makespan", 0.8))
    runtime_weight = float(
        configured_weights.get(
            "mean_runtime_seconds",
            configured_weights.get("mean_runtime", 0.2),
        )
    )

    if makespan_weight < 0 or runtime_weight < 0:
        raise ValueError("Outer objective weights must be non-negative.")

    total_weight = makespan_weight + runtime_weight
    if total_weight <= 0:
        raise ValueError("Outer objective weights must sum to a positive value.")

    return {
        "mean_makespan": makespan_weight / total_weight,
        "mean_runtime_seconds": runtime_weight / total_weight,
    }


def resolve_outer_runtime_settings(
    config,
    *,
    show_plot=None,
    outer_pop_size=None,
    outer_ngen=None,
    inner_repeats=None,
    random_seed=None,
    tournament_size=None,
    elite_count=None,
    random_immigrant_rate=None,
    outer_cxpb=None,
    outer_mutpb=None,
    log_population_details=None,
):
    """Resolve all outer-GA runtime settings from config plus optional overrides."""
    outer_settings = get_outer_ga_settings(config)

    resolved = {
        "show_plot": choose_runtime_setting(outer_settings["show_plot"], show_plot),
        "outer_pop_size": choose_runtime_setting(outer_settings["population_size"], outer_pop_size),
        "outer_ngen": choose_runtime_setting(outer_settings["generations"], outer_ngen),
        "inner_repeats": choose_runtime_setting(outer_settings["inner_repeats"], inner_repeats),
        "random_seed": choose_runtime_setting(outer_settings["random_seed"], random_seed),
        "tournament_size": choose_runtime_setting(outer_settings["tournament_size"], tournament_size),
        "elite_count": choose_runtime_setting(outer_settings["elite_count"], elite_count),
        "outer_cxpb": choose_runtime_setting(outer_settings["crossover_probability"], outer_cxpb),
        "outer_mutpb": choose_runtime_setting(outer_settings["mutation_probability"], outer_mutpb),
        "log_population_details": choose_runtime_setting(
            outer_settings["log_population_details"],
            log_population_details,
        ),
        "objective_weights": resolve_outer_objective_weights(outer_settings),
    }
    resolved["random_immigrant_rate"] = resolve_random_immigrant_rate(
        outer_settings,
        random_immigrant_rate=random_immigrant_rate,
    )
    return resolved


def normalize_hyperparameter_types(individual, config):
    """Keep hyperparameters inside the configured search space."""
    search_space = get_search_space(config)
    normalized_values = []

    for index, (name, value_type) in enumerate(OUTER_HYPERPARAMETER_SPECS):
        bounds = search_space[name]
        if value_type is int:
            value = int(round(individual[index]))
        else:
            value = float(individual[index])

        value = min(max(value, bounds["min"]), bounds["max"])
        normalized_values.append(value)

    return normalized_values


def individual_to_dict(individual, config):
    """Convert an outer individual to a readable hyperparameter dictionary."""
    normalized_values = normalize_hyperparameter_types(individual, config)
    hyperparameters = {}
    for (name, value_type), value in zip(OUTER_HYPERPARAMETER_SPECS, normalized_values):
        hyperparameters[name] = value if value_type is int else round(value, 3)
    return hyperparameters


def hyperparameter_signature(individual, config):
    hyperparameters = individual_to_dict(individual, config)
    return tuple(hyperparameters[name] for name, _ in OUTER_HYPERPARAMETER_SPECS)


def create_run_output_paths(am_name, config, log_dir=None):
    """Create a timestamped run folder for the AM currently being tested."""
    safe_am_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", am_name).strip("_")
    if not safe_am_name:
        safe_am_name = "unknown_am"
    instance_match = re.match(r"example_(\d+T)$", safe_am_name)
    instance_label = instance_match.group(1) if instance_match else safe_am_name

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    configured_log_dir = log_dir if log_dir is not None else config["paths"]["log_dir"]
    run_name = f"multiobjectiverun_{instance_label}_{timestamp}"
    run_dir = resolve_config_path(config, configured_log_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return {
        "run_dir": run_dir,
        "log_path": run_dir / f"{run_name}.log",
        "plot_path": run_dir / f"{run_name}.png",
        "history_path": run_dir / f"{run_name}_history.json",
        "evaluation_log_path": run_dir / f"{run_name}_evaluations.jsonl",
        "best_result_path": run_dir / f"{run_name}_best_result.json",
        "repeat_stability_plot_path": run_dir / f"{run_name}_repeat_stability.png",
        "objective_components_plot_path": (
            run_dir / f"{run_name}_objective_components.png"
        ),
        "hyperparameter_trajectory_plot_path": run_dir / f"{run_name}_hyperparameter_trajectories.png",
        "runtime_tradeoff_plot_path": run_dir / f"{run_name}_runtime_tradeoff.png",
        "population_std_plot_path": run_dir / f"{run_name}_population_std.png",
        "hyperparameter_diversity_plot_path": run_dir / f"{run_name}_hyperparameter_diversity.png",
        "generation_novelty_plot_path": run_dir / f"{run_name}_generation_novelty.png",
        "population_origin_plot_path": run_dir / f"{run_name}_population_origin.png",
        "best_individual_survival_plot_path": (
            run_dir / f"{run_name}_best_individual_survival.png"
        ),
        "validation_boxplot_path": run_dir / f"{run_name}_validation_boxplot.png",
        "validation_results_path": run_dir / f"{run_name}_validation_results.json",
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


def load_jsonl(path):
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped_line = line.strip()
            if not stripped_line:
                continue
            records.append(json.loads(stripped_line))
    return records


def maybe_show_plot(show_plot):
    """Only call plt.show when Matplotlib is using an interactive backend."""
    if show_plot and matplotlib.get_backend().lower() != "agg":
        plt.show()


def standard_deviation(values):
    if not values:
        return 0.0
    return pstdev(values)


def hyperparameter_std_from_dicts(hyperparameter_dicts):
    """Measure how spread out each outer-GA hyperparameter is in one population."""
    return {
        name: standard_deviation(
            [float(hyperparameters[name]) for hyperparameters in hyperparameter_dicts]
        )
        for name, _ in OUTER_HYPERPARAMETER_SPECS
    }


def hyperparameter_std_from_population(population, config):
    """Compute per-hyperparameter population diversity for a live outer-GA population."""
    hyperparameter_dicts = []
    for individual in population:
        normalized_values = normalize_hyperparameter_types(individual, config)
        hyperparameter_dicts.append(
            {
                name: value
                for (name, _), value in zip(
                    OUTER_HYPERPARAMETER_SPECS,
                    normalized_values,
                )
            }
        )
    return hyperparameter_std_from_dicts(hyperparameter_dicts)


def hyperparameter_dict_signature(hyperparameters):
    return tuple(hyperparameters[name] for name, _ in OUTER_HYPERPARAMETER_SPECS)


def repeat_makespans(record):
    return [repeat["makespan"] for repeat in record.get("repeats", [])]


def evaluation_records_by_id(evaluation_records):
    return {
        record["evaluation_id"]: record
        for record in evaluation_records
    }


def snapshot_population_evaluation_ids(population):
    evaluation_ids = []
    for index, individual in enumerate(population, start=1):
        evaluation_id = getattr(individual, "evaluation_id", None)
        if evaluation_id is None:
            raise ValueError(
                f"Population snapshot is missing evaluation_id for outer individual {index}."
            )
        evaluation_ids.append(evaluation_id)
    return evaluation_ids


def count_unique_hyperparameters(population, config):
    return len({hyperparameter_signature(individual, config) for individual in population})


def individual_mean_makespan(individual):
    if hasattr(individual, "mean_makespan"):
        return individual.mean_makespan
    if individual.fitness.valid:
        return individual.fitness.values[0]
    return None


def individual_avg_inner_runtime_seconds(individual):
    return getattr(individual, "avg_inner_runtime_seconds", None)


def individual_weighted_objective_score(individual):
    if hasattr(individual, "weighted_objective_score"):
        return individual.weighted_objective_score
    if individual.fitness.valid:
        return individual.fitness.values[0]
    return None


def outer_candidate_sort_key_from_individual(individual):
    runtime = individual_avg_inner_runtime_seconds(individual)
    return (
        individual_weighted_objective_score(individual),
        individual_mean_makespan(individual),
        float("inf") if runtime is None else runtime,
        getattr(individual, "evaluation_id", float("inf")),
    )


def outer_candidate_sort_key_from_record(record):
    runtime = record.get("avg_inner_runtime_seconds")
    return (
        record.get("weighted_objective_score", float("inf")),
        record["mean_makespan"],
        float("inf") if runtime is None else runtime,
        record["evaluation_id"],
    )


def normalize_objective_value(value, minimum, maximum):
    if minimum is None or maximum is None or math.isclose(minimum, maximum):
        return 0.0
    return (value - minimum) / (maximum - minimum)


def refresh_evaluation_archive_scores(evaluation_archive, objective_weights):
    """Update every archived outer evaluation with the latest weighted score."""
    if not evaluation_archive:
        return {
            "mean_makespan": {"min": None, "max": None},
            "mean_runtime_seconds": {"min": None, "max": None},
        }

    makespans = [record["mean_makespan"] for record in evaluation_archive.values()]
    runtimes = [record["avg_inner_runtime_seconds"] for record in evaluation_archive.values()]
    bounds = {
        "mean_makespan": {"min": min(makespans), "max": max(makespans)},
        "mean_runtime_seconds": {"min": min(runtimes), "max": max(runtimes)},
    }

    for record in evaluation_archive.values():
        normalized_mean_makespan = normalize_objective_value(
            record["mean_makespan"],
            bounds["mean_makespan"]["min"],
            bounds["mean_makespan"]["max"],
        )
        normalized_mean_runtime = normalize_objective_value(
            record["avg_inner_runtime_seconds"],
            bounds["mean_runtime_seconds"]["min"],
            bounds["mean_runtime_seconds"]["max"],
        )
        weighted_score = (
            objective_weights["mean_makespan"] * normalized_mean_makespan
            + objective_weights["mean_runtime_seconds"] * normalized_mean_runtime
        )
        record["normalized_mean_makespan"] = normalized_mean_makespan
        record["normalized_avg_inner_runtime_seconds"] = normalized_mean_runtime
        record["weighted_objective_score"] = weighted_score
        record["objective_weights"] = dict(objective_weights)

    return bounds


def apply_evaluation_record_to_individual(individual, evaluation_record):
    individual.mean_makespan = evaluation_record["mean_makespan"]
    individual.avg_inner_runtime_seconds = evaluation_record["avg_inner_runtime_seconds"]
    individual.total_inner_runtime_seconds = evaluation_record["total_inner_runtime_seconds"]
    individual.weighted_objective_score = evaluation_record.get("weighted_objective_score")
    individual.normalized_mean_makespan = evaluation_record.get("normalized_mean_makespan")
    individual.normalized_avg_inner_runtime_seconds = evaluation_record.get(
        "normalized_avg_inner_runtime_seconds"
    )
    if individual.weighted_objective_score is not None:
        individual.fitness.values = (individual.weighted_objective_score,)


def refresh_population_weighted_scores(population, evaluation_archive, objective_weights):
    objective_bounds = refresh_evaluation_archive_scores(evaluation_archive, objective_weights)
    for individual in population:
        evaluation_id = getattr(individual, "evaluation_id", None)
        if evaluation_id is None:
            continue
        apply_evaluation_record_to_individual(individual, evaluation_archive[evaluation_id])
    return objective_bounds


def build_consistent_fitness_history(
    history,
    evaluation_records,
    population_evaluation_snapshots,
):
    """Rescore every generation on one final archive-wide normalization basis."""
    if not history or not evaluation_records or not population_evaluation_snapshots:
        return history

    objective_weights = history[-1].get(
        "objective_weights",
        {"mean_makespan": 1.0, "mean_runtime_seconds": 0.0},
    )
    evaluation_archive = {
        record["evaluation_id"]: dict(record)
        for record in evaluation_records
    }
    final_bounds = refresh_evaluation_archive_scores(evaluation_archive, objective_weights)

    consistent_history = []
    seen_evaluation_ids = set()
    for base_row in history:
        generation = base_row["generation"]
        population_ids = population_evaluation_snapshots.get(generation)
        if not population_ids:
            raise ValueError(
                f"No population snapshot is available for outer generation {generation}."
            )

        missing_ids = [
            evaluation_id
            for evaluation_id in population_ids
            if evaluation_id not in evaluation_archive
        ]
        if missing_ids:
            raise ValueError(
                "Population snapshot references evaluation ids that are missing from the "
                f"evaluation archive for generation {generation}: {missing_ids}"
            )

        population_records = [
            evaluation_archive[evaluation_id]
            for evaluation_id in population_ids
        ]
        generation_best_record = min(
            population_records,
            key=outer_candidate_sort_key_from_record,
        )

        seen_evaluation_ids.update(population_ids)
        best_so_far_record = min(
            (
                evaluation_archive[evaluation_id]
                for evaluation_id in seen_evaluation_ids
            ),
            key=outer_candidate_sort_key_from_record,
        )

        weighted_scores = [
            record["weighted_objective_score"]
            for record in population_records
        ]
        makespans = [record["mean_makespan"] for record in population_records]
        runtimes = [
            record["avg_inner_runtime_seconds"]
            for record in population_records
        ]

        row = dict(base_row)
        row.update(
            {
                "objective_weights": dict(objective_weights),
                "objective_normalization_bounds": final_bounds,
                "population_size": len(population_records),
                "generation_best_weighted_objective_score": generation_best_record[
                    "weighted_objective_score"
                ],
                "generation_avg_weighted_objective_score": mean(weighted_scores),
                "generation_worst_weighted_objective_score": max(weighted_scores),
                "best_so_far_weighted_objective_score": best_so_far_record[
                    "weighted_objective_score"
                ],
                "generation_best_fitness_score": generation_best_record[
                    "weighted_objective_score"
                ],
                "generation_avg_fitness_score": mean(weighted_scores),
                "generation_worst_fitness_score": max(weighted_scores),
                "best_so_far_fitness_score": best_so_far_record[
                    "weighted_objective_score"
                ],
                "generation_best_mean_makespan": generation_best_record[
                    "mean_makespan"
                ],
                "generation_avg_mean_makespan": mean(makespans),
                "generation_worst_mean_makespan": max(makespans),
                "best_so_far_mean_makespan": best_so_far_record["mean_makespan"],
                "generation_best_makespan": generation_best_record["mean_makespan"],
                "generation_avg_makespan": mean(makespans),
                "generation_worst_makespan": max(makespans),
                "best_so_far_makespan": best_so_far_record["mean_makespan"],
                "generation_best_hyperparameters": generation_best_record[
                    "hyperparameters"
                ],
                "best_so_far_hyperparameters": best_so_far_record["hyperparameters"],
                "generation_best_avg_inner_runtime_seconds": generation_best_record[
                    "avg_inner_runtime_seconds"
                ],
                "generation_avg_avg_inner_runtime_seconds": mean(runtimes),
                "generation_worst_avg_inner_runtime_seconds": max(runtimes),
                "generation_best_mean_runtime_seconds": generation_best_record[
                    "avg_inner_runtime_seconds"
                ],
                "generation_avg_mean_runtime_seconds": mean(runtimes),
                "generation_worst_mean_runtime_seconds": max(runtimes),
                "generation_best_repeat_std_makespan": standard_deviation(
                    repeat_makespans(generation_best_record)
                ),
                "best_so_far_avg_inner_runtime_seconds": best_so_far_record[
                    "avg_inner_runtime_seconds"
                ],
                "best_so_far_mean_runtime_seconds": best_so_far_record[
                    "avg_inner_runtime_seconds"
                ],
                "best_so_far_repeat_std_makespan": standard_deviation(
                    repeat_makespans(best_so_far_record)
                ),
                "generation_best_evaluation_id": generation_best_record[
                    "evaluation_id"
                ],
                "best_so_far_evaluation_id": best_so_far_record["evaluation_id"],
                "population_objective_std": standard_deviation(weighted_scores),
                "population_mean_makespan_std": standard_deviation(makespans),
                "population_avg_inner_runtime_seconds_std": standard_deviation(
                    runtimes
                ),
                "fitness_score_basis": "final_archive_normalization",
            }
        )
        consistent_history.append(row)

    return consistent_history


def select_best_evaluation_record(evaluation_archive):
    if not evaluation_archive:
        return None
    return min(evaluation_archive.values(), key=outer_candidate_sort_key_from_record)


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


def print_population_consistency(label, population, expected_size, config, log_details=False):
    """Print outer population size, diversity, and optionally every individual."""
    print(f"\n{label} population summary")
    print(f"Expected outer population size: {expected_size}")
    print(f"Actual outer population size: {len(population)}")
    print(f"Unique hyperparameter configurations: {count_unique_hyperparameters(population, config)}")

    if not log_details:
        return

    for index, individual in enumerate(population, start=1):
        if individual.fitness.valid:
            status = (
                "reuse existing fitness"
                f", mean_makespan={individual_mean_makespan(individual)}"
                f", avg_inner_runtime_seconds={getattr(individual, 'avg_inner_runtime_seconds', 'unknown')}"
                f", evaluation_id={getattr(individual, 'evaluation_id', 'unknown')}"
            )
        else:
            status = "evaluate with NEW_GA_V2"

        print(
            f"Outer individual {index}/{len(population)}: "
            f"{individual_to_dict(individual, config)} -> {status}"
        )


def invalidate_evaluation(individual):
    """Mark an outer individual as needing a fresh inner GA evaluation."""
    if individual.fitness.valid:
        del individual.fitness.values

    for attribute_name in (
        "mean_makespan",
        "avg_inner_runtime_seconds",
        "total_inner_runtime_seconds",
        "inner_run_runtime_seconds",
        "inner_run_makespans",
        "evaluation_id",
        "best_inner_makespan",
        "best_inner_repeat_index",
        "weighted_objective_score",
        "normalized_mean_makespan",
        "normalized_avg_inner_runtime_seconds",
    ):
        if hasattr(individual, attribute_name):
            delattr(individual, attribute_name)


def create_hyperparameter_individual(config):
    """Generate a random individual representing scheduler GA hyperparameters."""
    search_space = get_search_space(config)
    individual = []
    for name, value_type in OUTER_HYPERPARAMETER_SPECS:
        bounds = search_space[name]
        if value_type is int:
            individual.append(random.randint(bounds["min"], bounds["max"]))
        else:
            individual.append(random.uniform(bounds["min"], bounds["max"]))
    return individual


def mutate_hyperparameters(individual, config):
    """Mutate one or more hyperparameters in place."""
    search_space = get_search_space(config)
    for index, (name, value_type) in enumerate(OUTER_HYPERPARAMETER_SPECS):
        if random.random() >= 0.5:
            continue

        bounds = search_space[name]
        if value_type is int:
            individual[index] = random.randint(bounds["min"], bounds["max"])
        else:
            individual[index] = random.uniform(bounds["min"], bounds["max"])

    individual[:] = normalize_hyperparameter_types(individual, config)
    return (individual,)


def crossover_hyperparameters(ind1, ind2, config):
    """Swap hyperparameter genes between two individuals."""
    for index in range(len(ind1)):
        if random.random() < 0.5:
            ind1[index], ind2[index] = ind2[index], ind1[index]

    ind1[:] = normalize_hyperparameter_types(ind1, config)
    ind2[:] = normalize_hyperparameter_types(ind2, config)
    return ind1, ind2


def is_better_outer_candidate(candidate, incumbent):
    return outer_candidate_sort_key_from_individual(candidate) < outer_candidate_sort_key_from_individual(
        incumbent
    )


def build_generation_history_row(
    generation,
    population,
    best_so_far_record,
    immigrants_introduced,
    immigrant_rate,
    config,
    evaluation_archive,
    objective_weights,
    objective_bounds,
):
    generation_best = min(population, key=outer_candidate_sort_key_from_individual)
    weighted_scores = [individual_weighted_objective_score(individual) for individual in population]
    makespans = [individual_mean_makespan(individual) for individual in population]
    runtimes = [
        individual_avg_inner_runtime_seconds(individual)
        for individual in population
    ]
    generation_best_weighted_objective_score = individual_weighted_objective_score(generation_best)
    generation_avg_weighted_objective_score = mean(weighted_scores)
    generation_worst_weighted_objective_score = max(weighted_scores)
    best_so_far_weighted_objective_score = best_so_far_record["weighted_objective_score"]
    generation_best_mean_makespan = individual_mean_makespan(generation_best)
    generation_avg_mean_makespan = mean(makespans)
    generation_worst_mean_makespan = max(makespans)
    best_so_far_mean_makespan = best_so_far_record["mean_makespan"]
    generation_best_avg_inner_runtime_seconds = individual_avg_inner_runtime_seconds(
        generation_best
    )
    generation_avg_avg_inner_runtime_seconds = mean(runtimes)
    generation_worst_avg_inner_runtime_seconds = max(runtimes)
    best_so_far_avg_inner_runtime_seconds = best_so_far_record["avg_inner_runtime_seconds"]
    generation_best_record = evaluation_archive[getattr(generation_best, "evaluation_id")]

    return {
        "objective_name": "weighted_mean_makespan_and_runtime",
        "objective_weights": dict(objective_weights),
        "objective_normalization_bounds": objective_bounds,
        "generation": generation,
        "population_size": len(population),
        "generation_best_weighted_objective_score": generation_best_weighted_objective_score,
        "generation_avg_weighted_objective_score": generation_avg_weighted_objective_score,
        "generation_worst_weighted_objective_score": generation_worst_weighted_objective_score,
        "best_so_far_weighted_objective_score": best_so_far_weighted_objective_score,
        "generation_best_fitness_score": generation_best_weighted_objective_score,
        "generation_avg_fitness_score": generation_avg_weighted_objective_score,
        "generation_worst_fitness_score": generation_worst_weighted_objective_score,
        "best_so_far_fitness_score": best_so_far_weighted_objective_score,
        "generation_best_mean_makespan": generation_best_mean_makespan,
        "generation_avg_mean_makespan": generation_avg_mean_makespan,
        "generation_worst_mean_makespan": generation_worst_mean_makespan,
        "best_so_far_mean_makespan": best_so_far_mean_makespan,
        # Backward-compatible aliases for earlier history files and callers.
        "generation_best_makespan": generation_best_mean_makespan,
        "generation_avg_makespan": generation_avg_mean_makespan,
        "generation_worst_makespan": generation_worst_mean_makespan,
        "best_so_far_makespan": best_so_far_mean_makespan,
        "generation_best_hyperparameters": individual_to_dict(generation_best, config),
        "best_so_far_hyperparameters": best_so_far_record["hyperparameters"],
        "generation_best_avg_inner_runtime_seconds": generation_best_avg_inner_runtime_seconds,
        "generation_avg_avg_inner_runtime_seconds": generation_avg_avg_inner_runtime_seconds,
        "generation_worst_avg_inner_runtime_seconds": generation_worst_avg_inner_runtime_seconds,
        "generation_best_mean_runtime_seconds": generation_best_avg_inner_runtime_seconds,
        "generation_avg_mean_runtime_seconds": generation_avg_avg_inner_runtime_seconds,
        "generation_worst_mean_runtime_seconds": generation_worst_avg_inner_runtime_seconds,
        "generation_best_repeat_std_makespan": standard_deviation(
            repeat_makespans(generation_best_record)
        ),
        "best_so_far_avg_inner_runtime_seconds": best_so_far_avg_inner_runtime_seconds,
        "best_so_far_mean_runtime_seconds": best_so_far_avg_inner_runtime_seconds,
        "best_so_far_repeat_std_makespan": standard_deviation(
            repeat_makespans(best_so_far_record)
        ),
        "generation_best_evaluation_id": getattr(generation_best, "evaluation_id", None),
        "best_so_far_evaluation_id": best_so_far_record["evaluation_id"],
        "population_objective_std": standard_deviation(weighted_scores),
        "population_mean_makespan_std": standard_deviation(makespans),
        "population_avg_inner_runtime_seconds_std": standard_deviation(runtimes),
        "unique_hyperparameter_count": count_unique_hyperparameters(population, config),
        "hyperparameter_std": hyperparameter_std_from_population(population, config),
        "immigrants_introduced": immigrants_introduced,
        "random_immigrant_rate": immigrant_rate,
    }


def format_hyperparameters_for_plot(hyperparameters):
    return (
        f"pop={hyperparameters['pop_size']}, "
        f"cxpb={hyperparameters['cxpb']}, "
        f"mutpb={hyperparameters['mutpb']}, "
        f"ngen={hyperparameters['ngen']}\n"
        f"task_mut={hyperparameters['task_order_probability']}, "
        f"proc_mut={hyperparameters['processor_allocation_probability']}, "
        f"prio_mut={hyperparameters['message_priority_shuffle_probability']}, "
        f"path_mut={hyperparameters['message_path_index_probability']}"
    )


def format_hyperparameters_for_summary(hyperparameters):
    return (
        f"pop={hyperparameters['pop_size']}, cx={hyperparameters['cxpb']}, "
        f"mut={hyperparameters['mutpb']}, ngen={hyperparameters['ngen']}\n"
        f"task={hyperparameters['task_order_probability']}, "
        f"proc={hyperparameters['processor_allocation_probability']}, "
        f"prio={hyperparameters['message_priority_shuffle_probability']}, "
        f"path={hyperparameters['message_path_index_probability']}"
    )


def select_top_distinct_candidates(evaluation_records, top_k):
    sorted_records = sorted(
        evaluation_records,
        key=lambda record: (
            record.get("weighted_objective_score", float("inf")),
            record["mean_makespan"],
            record["avg_inner_runtime_seconds"],
            record["evaluation_id"],
        ),
    )

    selected_records = []
    seen_signatures = set()
    for record in sorted_records:
        signature = hyperparameter_dict_signature(record["hyperparameters"])
        if signature in seen_signatures:
            continue

        selected_records.append(record)
        seen_signatures.add(signature)
        if len(selected_records) >= top_k:
            break

    return selected_records


def build_validation_seeds(random_seed, validation_repeats, validation_seed_offset):
    if validation_repeats <= 0:
        return []

    if random_seed is None:
        base_seed = validation_seed_offset
    else:
        base_seed = (random_seed * 1000) + validation_seed_offset

    return [base_seed + index for index in range(1, validation_repeats + 1)]


def validate_top_candidates_on_unseen_seeds(
    candidate_records,
    problem,
    config,
    validation_seeds,
):
    if not candidate_records or not validation_seeds:
        return []

    validation_results = []
    for rank, candidate_record in enumerate(candidate_records, start=1):
        hyperparameters = candidate_record["hyperparameters"]
        scheduler_config = build_inner_scheduler_config(config, hyperparameters)
        validation_runs = []
        validation_makespans = []
        validation_runtimes = []

        print(
            "Validating top outer candidate on unseen seeds:",
            {
                "candidate_rank": rank,
                "training_evaluation_id": candidate_record["evaluation_id"],
                "hyperparameters": hyperparameters,
                "validation_seeds": validation_seeds,
            },
        )

        for validation_index, inner_seed in enumerate(validation_seeds, start=1):
            run_start = perf_counter()
            makespan, schedule, genome = NEW_GA_V2(
                problem["processor_ids"],
                problem["processing_times"],
                problem["message_list"],
                problem["merged_paths_dict"],
                pop_size=hyperparameters["pop_size"],
                cxpb=hyperparameters["cxpb"],
                mutpb=hyperparameters["mutpb"],
                ngen=hyperparameters["ngen"],
                random_seed=inner_seed,
                scheduler_config=scheduler_config,
            )
            elapsed_seconds = perf_counter() - run_start

            validation_makespans.append(makespan)
            validation_runtimes.append(elapsed_seconds)
            validation_runs.append(
                {
                    "validation_index": validation_index,
                    "inner_seed": inner_seed,
                    "makespan": makespan,
                    "runtime_seconds": elapsed_seconds,
                    "schedule_summary": summarize_schedule(schedule),
                    "best_genome": serialize_genome(genome),
                    "best_schedule": serialize_schedule(schedule),
                }
            )

        result = {
            "candidate_rank": rank,
            "candidate_label": f"C{rank}",
            "training_evaluation_id": candidate_record["evaluation_id"],
            "hyperparameters": hyperparameters,
            "training_weighted_objective_score": candidate_record.get("weighted_objective_score"),
            "training_mean_makespan": candidate_record["mean_makespan"],
            "training_std_makespan": standard_deviation(
                repeat_makespans(candidate_record)
            ),
            "training_avg_inner_runtime_seconds": candidate_record["avg_inner_runtime_seconds"],
            "validation_repeats": len(validation_seeds),
            "validation_seeds": validation_seeds,
            "validation_mean_makespan": mean(validation_makespans),
            "validation_std_makespan": standard_deviation(validation_makespans),
            "validation_avg_inner_runtime_seconds": mean(validation_runtimes),
            "validation_runs": validation_runs,
        }
        validation_results.append(result)

        print(
            "Completed unseen-seed validation for candidate:",
            {
                "candidate_rank": rank,
                "training_mean_makespan": round(result["training_mean_makespan"], 3),
                "validation_mean_makespan": round(result["validation_mean_makespan"], 3),
                "validation_std_makespan": round(result["validation_std_makespan"], 3),
                "validation_avg_inner_runtime_seconds": round(
                    result["validation_avg_inner_runtime_seconds"],
                    6,
                ),
            },
        )

    return validation_results


def outer_ga(
    processor_ids,
    processing_times,
    message_list,
    merged_paths_dict,
    config=None,
    outer_pop_size=None,
    outer_ngen=None,
    inner_repeats=None,
    random_seed=None,
    tournament_size=None,
    elite_count=None,
    random_immigrant_rate=None,
    outer_cxpb=None,
    outer_mutpb=None,
    evaluation_log_path=None,
    history_path=None,
    best_result_path=None,
    log_population_details=None,
):
    config = config or load_config()
    runtime_settings = resolve_outer_runtime_settings(
        config,
        outer_pop_size=outer_pop_size,
        outer_ngen=outer_ngen,
        inner_repeats=inner_repeats,
        random_seed=random_seed,
        tournament_size=tournament_size,
        elite_count=elite_count,
        random_immigrant_rate=random_immigrant_rate,
        outer_cxpb=outer_cxpb,
        outer_mutpb=outer_mutpb,
        log_population_details=log_population_details,
    )
    outer_pop_size = runtime_settings["outer_pop_size"]
    outer_ngen = runtime_settings["outer_ngen"]
    inner_repeats = runtime_settings["inner_repeats"]
    random_seed = runtime_settings["random_seed"]
    tournament_size = runtime_settings["tournament_size"]
    elite_count = runtime_settings["elite_count"]
    random_immigrant_rate = runtime_settings["random_immigrant_rate"]
    outer_cxpb = runtime_settings["outer_cxpb"]
    outer_mutpb = runtime_settings["outer_mutpb"]
    log_population_details = runtime_settings["log_population_details"]
    objective_weights = runtime_settings["objective_weights"]

    if random_seed is not None:
        random.seed(random_seed)

    if "FitnessMinOuter" not in creator.__dict__:
        creator.create("FitnessMinOuter", base.Fitness, weights=(-1.0,))
    if "IndividualOuter" not in creator.__dict__:
        creator.create("IndividualOuter", list, fitness=creator.FitnessMinOuter)

    elite_count = max(1, min(elite_count, outer_pop_size))
    tournament_size = max(2, min(tournament_size, outer_pop_size))
    random_immigrant_count = resolve_random_immigrant_count(
        outer_pop_size,
        elite_count,
        random_immigrant_rate,
    )

    if evaluation_log_path is not None:
        Path(evaluation_log_path).write_text("", encoding="utf-8")

    toolbox = base.Toolbox()
    toolbox.register(
        "individual",
        tools.initIterate,
        creator.IndividualOuter,
        lambda: create_hyperparameter_individual(config),
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("select", tools.selTournament, tournsize=tournament_size)
    toolbox.register("mate", crossover_hyperparameters, config=config)
    toolbox.register("mutate", mutate_hyperparameters, config=config)

    evaluation_counter = 0
    evaluation_archive = {}

    def evaluate_outer(individual, generation, population_index, stage):
        nonlocal evaluation_counter

        individual[:] = normalize_hyperparameter_types(individual, config)
        hyperparameters = individual_to_dict(individual, config)
        pop_size = hyperparameters["pop_size"]
        cxpb = hyperparameters["cxpb"]
        mutpb = hyperparameters["mutpb"]
        ngen = hyperparameters["ngen"]
        scheduler_config = build_inner_scheduler_config(config, hyperparameters)
        makespans = []
        runtimes = []
        repeat_records = []

        print(
            "Evaluating outer individual:",
            {
                "generation": generation,
                "population_index": population_index,
                "stage": stage,
                "hyperparameters": hyperparameters,
            },
        )

        for repeat_index in range(1, inner_repeats + 1):
            inner_seed = None
            if random_seed is not None:
                inner_seed = (random_seed * 1000) + repeat_index

            run_start = perf_counter()
            makespan, schedule, genome = NEW_GA_V2(
                processor_ids,
                processing_times,
                message_list,
                merged_paths_dict,
                pop_size=pop_size,
                cxpb=cxpb,
                mutpb=mutpb,
                ngen=ngen,
                random_seed=inner_seed,
                scheduler_config=scheduler_config,
            )
            elapsed_seconds = perf_counter() - run_start

            makespans.append(makespan)
            runtimes.append(elapsed_seconds)
            repeat_records.append(
                {
                    "repeat_index": repeat_index,
                    "inner_seed": inner_seed,
                    "makespan": makespan,
                    "runtime_seconds": elapsed_seconds,
                    "schedule_summary": summarize_schedule(schedule),
                    "best_genome": serialize_genome(genome),
                    "best_schedule": serialize_schedule(schedule),
                }
            )

        avg_makespan = mean(makespans)
        avg_runtime = mean(runtimes)
        best_repeat_record = min(repeat_records, key=lambda record: record["makespan"])

        evaluation_counter += 1
        evaluation_id = evaluation_counter
        evaluation_record = {
            "evaluation_id": evaluation_id,
            "generation": generation,
            "population_index": population_index,
            "stage": stage,
            "hyperparameters": hyperparameters,
            "mean_makespan": avg_makespan,
            "std_makespan": standard_deviation(makespans),
            "best_repeat_makespan": best_repeat_record["makespan"],
            "avg_inner_runtime_seconds": avg_runtime,
            "std_inner_runtime_seconds": standard_deviation(runtimes),
            "total_inner_runtime_seconds": sum(runtimes),
            "inner_repeats": inner_repeats,
            "repeats": repeat_records,
        }
        evaluation_archive[evaluation_id] = evaluation_record

        individual.mean_makespan = avg_makespan
        individual.inner_run_makespans = makespans
        individual.inner_run_runtime_seconds = runtimes
        individual.avg_inner_runtime_seconds = avg_runtime
        individual.total_inner_runtime_seconds = sum(runtimes)
        individual.best_inner_makespan = best_repeat_record["makespan"]
        individual.best_inner_repeat_index = best_repeat_record["repeat_index"]
        individual.evaluation_id = evaluation_id

        print(
            "Completed outer individual:",
            {
                "evaluation_id": evaluation_id,
                "generation": generation,
                "population_index": population_index,
                "mean_makespan": round(avg_makespan, 3),
                "best_repeat_makespan": round(best_repeat_record["makespan"], 3),
                "avg_inner_runtime_seconds": round(avg_runtime, 6),
                "best_repeat_index": best_repeat_record["repeat_index"],
            },
        )

        return evaluation_record

    population = toolbox.population(n=outer_pop_size)
    population_evaluation_snapshots = {}

    print_population_consistency(
        "Initial outer GA",
        population,
        outer_pop_size,
        config,
        log_details=log_population_details,
    )
    initial_evaluation_records = []
    for index, individual in enumerate(population, start=1):
        initial_evaluation_records.append(
            evaluate_outer(
                individual,
                generation=0,
                population_index=index,
                stage="initial_population",
            )
        )
    objective_bounds = refresh_population_weighted_scores(
        population,
        evaluation_archive,
        objective_weights,
    )
    population_evaluation_snapshots[0] = snapshot_population_evaluation_ids(population)
    if evaluation_log_path is not None:
        for evaluation_record in initial_evaluation_records:
            append_jsonl(evaluation_log_path, evaluation_record)

    best_so_far_record = select_best_evaluation_record(evaluation_archive)
    history = [
        build_generation_history_row(
            generation=0,
            population=population,
            best_so_far_record=best_so_far_record,
            immigrants_introduced=0,
            immigrant_rate=random_immigrant_rate,
            config=config,
            evaluation_archive=evaluation_archive,
            objective_weights=objective_weights,
            objective_bounds=objective_bounds,
        )
    ]

    if history_path is not None:
        write_json(history_path, history)
    if best_result_path is not None:
        write_json(best_result_path, best_so_far_record)

    print("Initial outer GA best hyperparameters:", best_so_far_record["hyperparameters"])
    print("Initial outer GA best mean makespan:", best_so_far_record["mean_makespan"])

    for generation in range(1, outer_ngen + 1):
        print(f"\nOuter GA generation {generation}")

        elites = [toolbox.clone(individual) for individual in tools.selBest(population, elite_count)]
        offspring_count = outer_pop_size - elite_count - random_immigrant_count
        offspring = toolbox.select(population, offspring_count)
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < outer_cxpb:
                toolbox.mate(child1, child2)
                invalidate_evaluation(child1)
                invalidate_evaluation(child2)

        for mutant in offspring:
            if random.random() < outer_mutpb:
                toolbox.mutate(mutant)
                invalidate_evaluation(mutant)

        immigrants = [toolbox.individual() for _ in range(random_immigrant_count)]
        offspring.extend(immigrants)
        offspring.extend(elites)

        print_population_consistency(
            f"Outer GA generation {generation}",
            offspring,
            outer_pop_size,
            config,
            log_details=log_population_details,
        )

        invalid_individuals = [
            (index, individual)
            for index, individual in enumerate(offspring, start=1)
            if not individual.fitness.valid
        ]
        generation_evaluation_records = []
        for index, individual in invalid_individuals:
            generation_evaluation_records.append(
                evaluate_outer(
                    individual,
                    generation=generation,
                    population_index=index,
                    stage="offspring",
                )
            )

        objective_bounds = refresh_population_weighted_scores(
            offspring,
            evaluation_archive,
            objective_weights,
        )
        if evaluation_log_path is not None:
            for evaluation_record in generation_evaluation_records:
                append_jsonl(evaluation_log_path, evaluation_record)

        population[:] = offspring
        population_evaluation_snapshots[generation] = snapshot_population_evaluation_ids(
            population
        )

        best_so_far_record = select_best_evaluation_record(evaluation_archive)
        if best_result_path is not None:
            write_json(best_result_path, best_so_far_record)

        row = build_generation_history_row(
            generation=generation,
            population=population,
            best_so_far_record=best_so_far_record,
            immigrants_introduced=random_immigrant_count,
            immigrant_rate=random_immigrant_rate,
            config=config,
            evaluation_archive=evaluation_archive,
            objective_weights=objective_weights,
            objective_bounds=objective_bounds,
        )
        history.append(row)
        if history_path is not None:
            write_json(history_path, history)

        print("Generation best hyperparameters:", row["generation_best_hyperparameters"])
        print("Generation best mean makespan:", row["generation_best_mean_makespan"])
        print("Generation average mean makespan:", row["generation_avg_mean_makespan"])
        print("Best hyperparameters so far:", row["best_so_far_hyperparameters"])
        print("Best mean makespan so far:", row["best_so_far_mean_makespan"])
        print("Unique hyperparameter configurations:", row["unique_hyperparameter_count"])
        print("Hyperparameter std across population:", row["hyperparameter_std"])
        print("Tournament size:", tournament_size)
        print("Random immigrant rate:", row["random_immigrant_rate"])
        print("Random immigrants introduced:", row["immigrants_introduced"])

    result = OuterGAResult(
        best_hyperparameters=best_so_far_record["hyperparameters"],
        best_makespan=best_so_far_record["mean_makespan"],
        best_weighted_objective_score=best_so_far_record.get("weighted_objective_score"),
        best_avg_inner_runtime_seconds=best_so_far_record["avg_inner_runtime_seconds"],
        history=history,
        evaluation_records=sorted(
            evaluation_archive.values(),
            key=lambda record: record["evaluation_id"],
        ),
        best_evaluation_id=best_so_far_record["evaluation_id"],
        population_evaluation_snapshots=population_evaluation_snapshots,
    )
    return result


def plot_outer_ga_history(history, save_path="outer_ga_hyperparameter_search.png", show_plot=False):
    save_path = Path(save_path)
    generations = [row["generation"] for row in history]
    fitness_generation_best = [
        row.get(
            "generation_best_weighted_objective_score",
            row.get(
                "generation_best_fitness_score",
                row.get("generation_best_mean_makespan", row["generation_best_makespan"]),
            ),
        )
        for row in history
    ]
    fitness_generation_avg = [
        row.get(
            "generation_avg_weighted_objective_score",
            row.get(
                "generation_avg_fitness_score",
                row.get("generation_avg_mean_makespan", row["generation_avg_makespan"]),
            ),
        )
        for row in history
    ]
    fitness_generation_worst = [
        row.get(
            "generation_worst_weighted_objective_score",
            row.get(
                "generation_worst_fitness_score",
                row.get("generation_worst_mean_makespan", row["generation_worst_makespan"]),
            ),
        )
        for row in history
    ]
    makespan_generation_best = [
        row.get("generation_best_mean_makespan", row["generation_best_makespan"])
        for row in history
    ]
    makespan_generation_avg = [
        row.get("generation_avg_mean_makespan", row["generation_avg_makespan"])
        for row in history
    ]
    makespan_generation_worst = [
        row.get("generation_worst_mean_makespan", row["generation_worst_makespan"])
        for row in history
    ]
    unique_configs = [row["unique_hyperparameter_count"] for row in history]
    runtime_generation_best = [row.get("generation_best_avg_inner_runtime_seconds") for row in history]
    runtime_generation_avg = [row.get("generation_avg_avg_inner_runtime_seconds") for row in history]
    runtime_generation_worst = [row.get("generation_worst_avg_inner_runtime_seconds") for row in history]
    has_runtime_values = any(runtime is not None for runtime in runtime_generation_best)
    objective_weights = history[-1].get(
        "objective_weights",
        {"mean_makespan": 1.0, "mean_runtime_seconds": 0.0},
    )
    fitness_score_basis = history[-1].get("fitness_score_basis", "generation_snapshot")
    uses_final_archive_normalization = fitness_score_basis == "final_archive_normalization"
    best_history_row = min(
        history,
        key=lambda row: (
            row.get(
                "best_so_far_weighted_objective_score",
                row.get(
                    "best_so_far_fitness_score",
                    row.get("best_so_far_mean_makespan", row["best_so_far_makespan"]),
                ),
            ),
            row.get("best_so_far_mean_makespan", row["best_so_far_makespan"]),
            row.get("best_so_far_avg_inner_runtime_seconds", float("inf")),
            row["generation"],
        ),
    )

    improvement_rows = []
    current_best = None
    for row in history:
        objective_value = row.get(
            "best_so_far_weighted_objective_score",
            row.get(
                "best_so_far_fitness_score",
                row.get("best_so_far_mean_makespan", row["best_so_far_makespan"]),
            ),
        )
        if current_best is None or objective_value < current_best:
            improvement_rows.append(row)
            current_best = objective_value

    fig = plt.figure(figsize=(15.5, 12.5), constrained_layout=True)
    grid = fig.add_gridspec(
        4,
        2,
        width_ratios=[3.8, 1.45],
        height_ratios=[1.05, 1.0, 1.0, 0.82],
    )
    fitness_axis = fig.add_subplot(grid[0, 0])
    makespan_axis = fig.add_subplot(grid[1, 0], sharex=fitness_axis)
    runtime_axis = fig.add_subplot(grid[2, 0], sharex=fitness_axis)
    diversity_axis = fig.add_subplot(grid[3, 0], sharex=fitness_axis)
    summary_axis = fig.add_subplot(grid[:, 1])
    summary_axis.axis("off")
    fig.suptitle("Outer GA weighted multi-objective progress", fontsize=18)

    fitness_best_line = fitness_axis.plot(
        generations,
        fitness_generation_best,
        marker="s",
        linewidth=1.5,
        color="tab:green",
        label="Generation best",
    )[0]
    fitness_avg_line = fitness_axis.plot(
        generations,
        fitness_generation_avg,
        marker="^",
        linewidth=1.5,
        color="tab:gray",
        label="Generation average",
    )[0]
    fitness_worst_line = fitness_axis.plot(
        generations,
        fitness_generation_worst,
        marker="v",
        linewidth=1.3,
        color="tab:blue",
        label="Generation worst",
    )[0]
    new_best_points = fitness_axis.scatter(
        [row["generation"] for row in improvement_rows],
        [
            row.get(
                "best_so_far_weighted_objective_score",
                row.get(
                    "best_so_far_fitness_score",
                    row.get("best_so_far_mean_makespan", row["best_so_far_makespan"]),
                ),
            )
            for row in improvement_rows
        ],
        marker="*",
        s=140,
        color="tab:red",
        label="New best",
        zorder=5,
    )
    for row in improvement_rows:
        fitness_axis.annotate(
            f"g{row['generation']}",
            xy=(
                row["generation"],
                row.get(
                    "best_so_far_weighted_objective_score",
                    row.get(
                        "best_so_far_fitness_score",
                        row.get("best_so_far_mean_makespan", row["best_so_far_makespan"]),
                    ),
                ),
            ),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color="tab:red",
            fontweight="bold",
        )

    fitness_axis.set_ylabel("Weighted fitness score")
    if uses_final_archive_normalization:
        fitness_axis.set_title("Fitness score curves (final-normalized across the full run)")
    else:
        fitness_axis.set_title("Fitness score curves")
    fitness_axis.grid(True, alpha=0.3)
    fitness_axis.legend(
        handles=[
            fitness_best_line,
            fitness_avg_line,
            fitness_worst_line,
            new_best_points,
        ],
        loc="best",
    )

    makespan_axis.set_ylabel("Mean makespan")
    makespan_axis.set_title("Makespan objective")
    makespan_axis.plot(
        generations,
        makespan_generation_best,
        marker="s",
        linewidth=1.5,
        color="tab:green",
        label="Generation best",
    )
    makespan_axis.plot(
        generations,
        makespan_generation_avg,
        marker="^",
        linewidth=1.5,
        color="tab:gray",
        label="Generation average",
    )
    makespan_axis.plot(
        generations,
        makespan_generation_worst,
        marker="v",
        linewidth=1.3,
        color="tab:blue",
        label="Generation worst",
    )
    makespan_axis.grid(True, alpha=0.3)
    makespan_axis.legend(loc="best")

    unique_config_line = diversity_axis.plot(
        generations,
        unique_configs,
        marker="o",
        linewidth=2,
        color="tab:purple",
        label="Unique configs",
    )[0]
    if has_runtime_values:
        runtime_best_line = runtime_axis.plot(
            generations,
            runtime_generation_best,
            marker="s",
            linewidth=1.5,
            color="tab:orange",
            label="Generation best",
        )[0]
        runtime_avg_line = runtime_axis.plot(
            generations,
            runtime_generation_avg,
            marker="^",
            linewidth=1.5,
            color="tab:gray",
            label="Generation average",
        )[0]
        runtime_worst_line = runtime_axis.plot(
            generations,
            runtime_generation_worst,
            marker="v",
            linewidth=1.3,
            color="tab:blue",
            label="Generation worst",
        )[0]
        runtime_axis.set_ylabel("Seconds")
        runtime_axis.set_title("Runtime objective")
        runtime_axis.grid(True, alpha=0.3)
        runtime_axis.legend(
            handles=[
                runtime_best_line,
                runtime_avg_line,
                runtime_worst_line,
            ],
            loc="best",
        )
    else:
        runtime_axis.text(
            0.5,
            0.5,
            "No runtime data",
            ha="center",
            va="center",
            transform=runtime_axis.transAxes,
            fontsize=11,
        )
        runtime_axis.set_title("Runtime objective")
        runtime_axis.grid(True, alpha=0.3)
        runtime_axis.set_ylabel("Seconds")

    diversity_axis.set_ylabel("Unique configs")
    diversity_axis.set_title("Population diversity")
    diversity_axis.grid(True, alpha=0.3)
    diversity_axis.legend(handles=[unique_config_line], loc="best")

    diversity_axis.set_xlabel("Outer GA generation")
    diversity_axis.set_xticks(generations)
    plt.setp(fitness_axis.get_xticklabels(), visible=False)
    plt.setp(makespan_axis.get_xticklabels(), visible=False)
    plt.setp(runtime_axis.get_xticklabels(), visible=False)

    summary_lines = [
        "Run summary",
        "",
        "Objective: weighted score",
        (
            "Weights: "
            f"makespan={objective_weights['mean_makespan']:.2f}, "
            f"runtime={objective_weights['mean_runtime_seconds']:.2f}"
        ),
        (
            "Fitness basis: final archive-wide normalization"
            if uses_final_archive_normalization
            else "Fitness basis: generation snapshot normalization"
        ),
        (
            "Final selected best\n"
            "score="
            f"{best_history_row.get('best_so_far_weighted_objective_score', best_history_row.get('best_so_far_fitness_score', best_history_row.get('best_so_far_mean_makespan', best_history_row['best_so_far_makespan']))):.4f}\n"
            f"mean={best_history_row.get('best_so_far_mean_makespan', best_history_row['best_so_far_makespan']):.1f}\n"
            f"runtime={best_history_row.get('best_so_far_avg_inner_runtime_seconds', 0.0):.4f}s"
        ),
        format_hyperparameters_for_summary(best_history_row["best_so_far_hyperparameters"]),
        "",
        "Milestones",
    ]
    for row in improvement_rows:
        summary_lines.extend(
            [
                (
                    f"g{row['generation']} -> "
                    f"score={row.get('best_so_far_weighted_objective_score', row.get('best_so_far_fitness_score', row.get('best_so_far_mean_makespan', row['best_so_far_makespan']))):.4f}, "
                    f"mean={row.get('best_so_far_mean_makespan', row['best_so_far_makespan']):.1f}, "
                    f"runtime={row.get('best_so_far_avg_inner_runtime_seconds', 0.0):.4f}s"
                ),
                format_hyperparameters_for_summary(row["best_so_far_hyperparameters"]),
                "",
            ]
        )

    summary_axis.text(
        0.02,
        0.98,
        "\n".join(summary_lines).rstrip(),
        ha="left",
        va="top",
        fontsize=9,
        family="monospace",
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "alpha": 0.9},
    )

    fig.savefig(save_path, dpi=200)
    maybe_show_plot(show_plot)
    plt.close(fig)
    return save_path.resolve()


def plot_outer_objective_components(history, save_path, show_plot=False):
    save_path = Path(save_path)
    generations = [row["generation"] for row in history]
    makespan_generation_best = [
        row.get("generation_best_mean_makespan", row["generation_best_makespan"])
        for row in history
    ]
    makespan_generation_avg = [
        row.get("generation_avg_mean_makespan", row["generation_avg_makespan"])
        for row in history
    ]
    makespan_generation_worst = [
        row.get("generation_worst_mean_makespan", row["generation_worst_makespan"])
        for row in history
    ]
    runtime_generation_best = [row.get("generation_best_avg_inner_runtime_seconds") for row in history]
    runtime_generation_avg = [row.get("generation_avg_avg_inner_runtime_seconds") for row in history]
    runtime_generation_worst = [row.get("generation_worst_avg_inner_runtime_seconds") for row in history]
    has_runtime_values = any(runtime is not None for runtime in runtime_generation_best)

    fig, axes = plt.subplots(2, 1, figsize=(11.5, 8.4), sharex=True)

    axes[0].plot(
        generations,
        makespan_generation_best,
        marker="s",
        linewidth=1.5,
        color="tab:green",
        label="Generation best",
    )
    axes[0].plot(
        generations,
        makespan_generation_avg,
        marker="^",
        linewidth=1.5,
        color="tab:gray",
        label="Generation average",
    )
    axes[0].plot(
        generations,
        makespan_generation_worst,
        marker="v",
        linewidth=1.3,
        color="tab:blue",
        label="Generation worst",
    )
    axes[0].set_ylabel("Mean makespan")
    axes[0].set_title("Makespan objective evolution")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")

    if has_runtime_values:
        axes[1].plot(
            generations,
            runtime_generation_best,
            marker="s",
            linewidth=1.5,
            color="tab:orange",
            label="Generation best",
        )
        axes[1].plot(
            generations,
            runtime_generation_avg,
            marker="^",
            linewidth=1.5,
            color="tab:gray",
            label="Generation average",
        )
        axes[1].plot(
            generations,
            runtime_generation_worst,
            marker="v",
            linewidth=1.3,
            color="tab:blue",
            label="Generation worst",
        )
        axes[1].legend(loc="best")
    else:
        axes[1].text(
            0.5,
            0.5,
            "No runtime data",
            ha="center",
            va="center",
            transform=axes[1].transAxes,
            fontsize=11,
        )

    axes[1].set_xlabel("Outer GA generation")
    axes[1].set_ylabel("Seconds")
    axes[1].set_title("Runtime objective evolution")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(generations)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    maybe_show_plot(show_plot)
    plt.close(fig)
    return save_path.resolve()


def plot_generation_best_repeat_stability(
    history,
    evaluation_records,
    final_best_evaluation_id,
    save_path,
    show_plot=False,
):
    save_path = Path(save_path)
    evaluations_by_id = evaluation_records_by_id(evaluation_records)
    generations = [row["generation"] for row in history]
    generation_best_records = [
        evaluations_by_id[row["generation_best_evaluation_id"]]
        for row in history
    ]
    generation_best_means = [record["mean_makespan"] for record in generation_best_records]
    generation_best_stds = [
        standard_deviation(repeat_makespans(record))
        for record in generation_best_records
    ]
    final_best_record = evaluations_by_id[final_best_evaluation_id]
    final_best_mean = final_best_record["mean_makespan"]
    final_best_std = standard_deviation(repeat_makespans(final_best_record))

    fig, axis = plt.subplots(figsize=(11, 5.5))
    axis.errorbar(
        generations,
        generation_best_means,
        yerr=generation_best_stds,
        fmt="o-",
        capsize=4,
        linewidth=1.5,
        color="tab:blue",
        label="Generation-best mean ± std",
    )
    axis.axhline(
        final_best_mean,
        color="tab:red",
        linestyle="--",
        linewidth=1.5,
        label="Final selected best mean",
    )
    axis.fill_between(
        generations,
        [final_best_mean - final_best_std] * len(generations),
        [final_best_mean + final_best_std] * len(generations),
        color="tab:red",
        alpha=0.15,
        label="Final selected best ± std",
    )
    axis.set_xlabel("Outer GA generation")
    axis.set_ylabel("Mean makespan")
    axis.set_title("Generation-best repeat stability vs final selected best")
    axis.grid(True, alpha=0.3)
    axis.set_xticks(generations)
    axis.legend(loc="best")

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    maybe_show_plot(show_plot)
    plt.close(fig)
    return save_path.resolve()


def plot_best_hyperparameter_trajectories(history, save_path, show_plot=False):
    save_path = Path(save_path)
    generations = [row["generation"] for row in history]
    parameter_names = [name for name, _ in OUTER_HYPERPARAMETER_SPECS]
    fig, axes = plt.subplots(4, 2, figsize=(13, 11), sharex=True)

    for axis, parameter_name in zip(axes.flat, parameter_names):
        values = [row["best_so_far_hyperparameters"][parameter_name] for row in history]
        axis.step(generations, values, where="post", linewidth=1.8, color="tab:blue")
        axis.scatter(generations, values, s=20, color="tab:blue")
        axis.set_title(parameter_name.replace("_", " "))
        axis.grid(True, alpha=0.3)

    for axis in axes[-1]:
        axis.set_xlabel("Outer GA generation")

    fig.suptitle("Best-so-far hyperparameter trajectories", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    maybe_show_plot(show_plot)
    plt.close(fig)
    return save_path.resolve()


def parse_population_hyperparameters_from_log(log_path):
    """Recover full outer populations from a log created with log_population_details."""
    log_path = Path(log_path)
    population_snapshots = {}
    current_generation = None

    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped_line = line.strip()
            if stripped_line == "Initial outer GA population summary":
                current_generation = 0
                population_snapshots.setdefault(current_generation, [])
                continue

            generation_match = re.match(
                r"Outer GA generation (\d+) population summary$",
                stripped_line,
            )
            if generation_match:
                current_generation = int(generation_match.group(1))
                population_snapshots.setdefault(current_generation, [])
                continue

            if current_generation is None or not stripped_line.startswith("Outer individual "):
                continue

            dict_start = stripped_line.find("{")
            dict_end = stripped_line.find("} ->", dict_start)
            if dict_start == -1 or dict_end == -1:
                continue

            try:
                hyperparameters = ast.literal_eval(stripped_line[dict_start:dict_end + 1])
            except (SyntaxError, ValueError):
                continue

            if all(name in hyperparameters for name, _ in OUTER_HYPERPARAMETER_SPECS):
                population_snapshots[current_generation].append(hyperparameters)

    return population_snapshots


def parse_population_evaluation_ids_from_log(log_path):
    """Recover per-generation outer populations as evaluation ids from a detailed log."""
    log_path = Path(log_path)
    indexed_snapshots = {}
    current_generation = None

    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped_line = line.strip()
            if stripped_line == "Initial outer GA population summary":
                current_generation = 0
                indexed_snapshots.setdefault(current_generation, {})
                continue

            generation_match = re.match(
                r"Outer GA generation (\d+) population summary$",
                stripped_line,
            )
            if generation_match:
                current_generation = int(generation_match.group(1))
                indexed_snapshots.setdefault(current_generation, {})
                continue

            if current_generation is not None and stripped_line.startswith("Outer individual "):
                index_match = re.match(r"Outer individual (\d+)/\d+:", stripped_line)
                if not index_match:
                    continue

                population_index = int(index_match.group(1))
                indexed_snapshots[current_generation].setdefault(population_index, None)
                reuse_match = re.search(r"evaluation_id=(\d+)", stripped_line)
                if reuse_match:
                    indexed_snapshots[current_generation][population_index] = int(
                        reuse_match.group(1)
                    )
                continue

            if not stripped_line.startswith("Completed outer individual:"):
                continue

            record_text = stripped_line.split("Completed outer individual:", maxsplit=1)[1].strip()
            try:
                evaluation_record = ast.literal_eval(record_text)
            except (SyntaxError, ValueError):
                continue

            generation = evaluation_record.get("generation")
            population_index = evaluation_record.get("population_index")
            evaluation_id = evaluation_record.get("evaluation_id")
            if generation is None or population_index is None or evaluation_id is None:
                continue

            indexed_snapshots.setdefault(generation, {})
            indexed_snapshots[generation][population_index] = evaluation_id

    population_snapshots = {}
    for generation, indexed_ids in sorted(indexed_snapshots.items()):
        if not indexed_ids:
            continue

        missing_indices = [
            population_index
            for population_index in range(1, max(indexed_ids) + 1)
            if indexed_ids.get(population_index) is None
        ]
        if missing_indices:
            raise ValueError(
                "The outer-GA log is missing population evaluation ids for generation "
                f"{generation}: {missing_indices}"
            )

        population_snapshots[generation] = [
            indexed_ids[population_index]
            for population_index in range(1, max(indexed_ids) + 1)
        ]

    if not population_snapshots:
        raise ValueError(
            "No population snapshots were found. Re-run with log_population_details=true "
            "or use a log that contains outer population details."
        )

    return population_snapshots


def parse_outer_ga_settings_from_log(log_path):
    log_path = Path(log_path)
    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped_line = line.strip()
            if not stripped_line.startswith("Outer GA settings:"):
                continue

            settings_text = stripped_line.split("Outer GA settings:", maxsplit=1)[1].strip()
            try:
                return ast.literal_eval(settings_text)
            except (SyntaxError, ValueError) as exc:
                raise ValueError(
                    f"Could not parse outer-GA settings from {log_path}."
                ) from exc

    raise ValueError(f"No 'Outer GA settings:' line was found in {log_path}.")


def unique_configuration_count_for_evaluation_ids(evaluation_ids, evaluations_by_id):
    return len(
        {
            hyperparameter_dict_signature(evaluations_by_id[evaluation_id]["hyperparameters"])
            for evaluation_id in evaluation_ids
        }
    )


def generation_novelty_rows_from_population_snapshots(
    population_evaluation_snapshots,
    evaluation_records,
):
    evaluations_by_id = evaluation_records_by_id(evaluation_records)
    rows = []
    previous_signature_set = None

    for generation, evaluation_ids in sorted(population_evaluation_snapshots.items()):
        current_signature_set = {
            hyperparameter_dict_signature(
                evaluations_by_id[evaluation_id]["hyperparameters"]
            )
            for evaluation_id in evaluation_ids
        }

        if previous_signature_set is None:
            new_signature_set = set(current_signature_set)
            shared_signature_set = set()
        else:
            new_signature_set = current_signature_set - previous_signature_set
            shared_signature_set = current_signature_set & previous_signature_set

        rows.append(
            {
                "generation": generation,
                "unique_configuration_count": len(current_signature_set),
                "new_unique_configuration_count": len(new_signature_set),
                "shared_unique_configuration_count": len(shared_signature_set),
            }
        )
        previous_signature_set = current_signature_set

    return rows


def population_origin_rows_from_population_snapshots(
    population_evaluation_snapshots,
    evaluation_records,
    outer_ga_settings,
):
    if not population_evaluation_snapshots or not evaluation_records or not outer_ga_settings:
        return []

    evaluations_by_id = evaluation_records_by_id(evaluation_records)
    expected_population_size = int(outer_ga_settings["outer_pop_size"])
    elite_count = int(outer_ga_settings.get("elite_count", 0))
    random_immigrant_count = outer_ga_settings.get("random_immigrant_count")
    if random_immigrant_count is None:
        random_immigrant_count = resolve_random_immigrant_count(
            expected_population_size,
            elite_count,
            float(outer_ga_settings.get("random_immigrant_rate", 0.0)),
        )
    else:
        random_immigrant_count = int(random_immigrant_count)

    rows = []
    previous_evaluation_id_set = set()

    for generation, evaluation_ids in sorted(population_evaluation_snapshots.items()):
        if len(evaluation_ids) != expected_population_size:
            raise ValueError(
                "Population-origin reconstruction expected "
                f"{expected_population_size} individuals, but generation {generation} "
                f"has {len(evaluation_ids)}."
            )

        unique_configuration_count = unique_configuration_count_for_evaluation_ids(
            evaluation_ids,
            evaluations_by_id,
        )

        if generation == 0:
            rows.append(
                {
                    "generation": generation,
                    "population_size": len(evaluation_ids),
                    "initial_population_count": len(evaluation_ids),
                    "changed_offspring_count": 0,
                    "unchanged_selected_clone_count": 0,
                    "random_immigrant_count": 0,
                    "elite_carryover_count": 0,
                    "reused_previous_generation_count": 0,
                    "unique_configuration_count": unique_configuration_count,
                }
            )
            previous_evaluation_id_set = set(evaluation_ids)
            continue

        offspring_count = len(evaluation_ids) - elite_count - random_immigrant_count
        if offspring_count < 0:
            raise ValueError(
                "Population-origin reconstruction computed a negative offspring count "
                f"for generation {generation}."
            )

        offspring_ids = evaluation_ids[:offspring_count]
        immigrant_ids = evaluation_ids[
            offspring_count:offspring_count + random_immigrant_count
        ]
        elite_ids = evaluation_ids[offspring_count + random_immigrant_count:]

        unchanged_selected_clone_count = sum(
            1
            for evaluation_id in offspring_ids
            if evaluation_id in previous_evaluation_id_set
        )
        changed_offspring_count = len(offspring_ids) - unchanged_selected_clone_count

        rows.append(
            {
                "generation": generation,
                "population_size": len(evaluation_ids),
                "initial_population_count": 0,
                "changed_offspring_count": changed_offspring_count,
                "unchanged_selected_clone_count": unchanged_selected_clone_count,
                "random_immigrant_count": len(immigrant_ids),
                "elite_carryover_count": len(elite_ids),
                "reused_previous_generation_count": (
                    unchanged_selected_clone_count + len(elite_ids)
                ),
                "unique_configuration_count": unique_configuration_count,
            }
        )
        previous_evaluation_id_set = set(evaluation_ids)

    return rows


def generation_best_survival_rows_from_history(
    history,
    population_evaluation_snapshots,
):
    if not history or not population_evaluation_snapshots:
        return []

    population_sets = {
        generation: set(evaluation_ids)
        for generation, evaluation_ids in population_evaluation_snapshots.items()
    }
    if not population_sets:
        return []

    max_generation = max(population_sets)
    rows = []

    for row in history:
        generation = row["generation"]
        best_evaluation_id = row.get("generation_best_evaluation_id")
        if best_evaluation_id is None or generation not in population_sets:
            continue

        survival_generations = 0
        last_generation_present = generation - 1
        for future_generation in range(generation, max_generation + 1):
            if best_evaluation_id not in population_sets.get(future_generation, set()):
                break
            survival_generations += 1
            last_generation_present = future_generation

        rows.append(
            {
                "generation": generation,
                "generation_best_evaluation_id": best_evaluation_id,
                "survival_generations_including_origin": survival_generations,
                "future_generations_survived": max(0, survival_generations - 1),
                "last_generation_present": last_generation_present,
            }
        )

    return rows


def hyperparameter_diversity_rows_from_snapshots(population_snapshots):
    rows = []
    for generation, hyperparameter_dicts in sorted(population_snapshots.items()):
        if not hyperparameter_dicts:
            continue

        rows.append(
            {
                "generation": generation,
                "population_size": len(hyperparameter_dicts),
                "hyperparameter_std": hyperparameter_std_from_dicts(hyperparameter_dicts),
            }
        )
    return rows


def hyperparameter_diversity_rows_from_log(log_path):
    return hyperparameter_diversity_rows_from_snapshots(
        parse_population_hyperparameters_from_log(log_path)
    )


def plot_hyperparameter_diversity(history, save_path, show_plot=False):
    save_path = Path(save_path)
    rows = [row for row in history if row.get("hyperparameter_std")]
    if not rows:
        return None

    generations = [row["generation"] for row in rows]
    parameter_names = [name for name, _ in OUTER_HYPERPARAMETER_SPECS]
    fig, axes = plt.subplots(4, 2, figsize=(13, 11), sharex=True)

    for axis, parameter_name in zip(axes.flat, parameter_names):
        values = [
            row["hyperparameter_std"][parameter_name]
            for row in rows
        ]
        axis.plot(
            generations,
            values,
            marker="o",
            linewidth=1.8,
            color="tab:blue",
        )
        axis.set_title(f"{parameter_name.replace('_', ' ')} std")
        axis.set_ylabel("Std")
        axis.grid(True, alpha=0.3)

    for axis in axes[-1]:
        axis.set_xlabel("Outer GA generation")
        axis.set_xticks(generations)

    fig.suptitle("Hyperparameter diversity across the outer population", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    maybe_show_plot(show_plot)
    plt.close(fig)
    return save_path.resolve()


def create_hyperparameter_diversity_artifacts_from_log(log_path, show_plot=False):
    log_path = Path(log_path)
    rows = hyperparameter_diversity_rows_from_log(log_path)
    if not rows:
        raise ValueError(
            "No population snapshots were found. Re-run with log_population_details=true "
            "or use a log that contains 'Outer individual ...' population lines."
        )

    data_path = log_path.with_name(f"{log_path.stem}_hyperparameter_diversity.json")
    plot_path = log_path.with_name(f"{log_path.stem}_hyperparameter_diversity.png")
    write_json(
        data_path,
        {
            "description": (
                "Population-level hyperparameter diversity recovered from the outer-GA log. "
                "Each value is the population standard deviation of one hyperparameter in "
                "one outer generation."
            ),
            "source_log_path": str(log_path.resolve()),
            "rows": rows,
        },
    )
    resolved_plot_path = plot_hyperparameter_diversity(
        rows,
        save_path=plot_path,
        show_plot=show_plot,
    )
    return resolved_plot_path, data_path.resolve(), rows


def plot_generation_novelty(novelty_rows, save_path, show_plot=False):
    save_path = Path(save_path)
    if not novelty_rows:
        return None

    generations = [row["generation"] for row in novelty_rows]
    new_counts = [row["new_unique_configuration_count"] for row in novelty_rows]
    shared_counts = [row["shared_unique_configuration_count"] for row in novelty_rows]
    total_unique_counts = [row["unique_configuration_count"] for row in novelty_rows]

    fig, axis = plt.subplots(figsize=(10.5, 5.2))
    axis.bar(
        generations,
        shared_counts,
        width=0.7,
        color="tab:blue",
        alpha=0.7,
        label="Shared unique configs with previous generation",
    )
    axis.bar(
        generations,
        new_counts,
        width=0.7,
        bottom=shared_counts,
        color="tab:orange",
        alpha=0.8,
        label="New unique configs vs previous generation",
    )
    axis.plot(
        generations,
        total_unique_counts,
        marker="o",
        linewidth=1.8,
        color="black",
        label="Total unique configs in generation",
    )
    axis.set_xlabel("Outer GA generation")
    axis.set_ylabel("Count of unique hyperparameter configurations")
    axis.set_title("Generation novelty relative to the previous generation")
    axis.grid(True, axis="y", alpha=0.3)
    axis.set_xticks(generations)
    axis.legend(loc="best")

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    maybe_show_plot(show_plot)
    plt.close(fig)
    return save_path.resolve()


def plot_population_origin(population_origin_rows, save_path, show_plot=False):
    save_path = Path(save_path)
    if not population_origin_rows:
        return None

    generations = [row["generation"] for row in population_origin_rows]
    initial_counts = [row["initial_population_count"] for row in population_origin_rows]
    changed_offspring_counts = [
        row["changed_offspring_count"] for row in population_origin_rows
    ]
    unchanged_clone_counts = [
        row["unchanged_selected_clone_count"] for row in population_origin_rows
    ]
    immigrant_counts = [row["random_immigrant_count"] for row in population_origin_rows]
    elite_counts = [row["elite_carryover_count"] for row in population_origin_rows]
    unique_configuration_counts = [
        row["unique_configuration_count"] for row in population_origin_rows
    ]
    population_sizes = [row["population_size"] for row in population_origin_rows]

    fig, axis = plt.subplots(figsize=(11, 5.6))
    bottom = [0] * len(generations)

    stacked_series = [
        ("Initial random population", initial_counts, "#B0BEC5"),
        ("Changed offspring", changed_offspring_counts, "tab:orange"),
        ("Unchanged selected clones", unchanged_clone_counts, "tab:blue"),
        ("Random immigrants", immigrant_counts, "tab:green"),
        ("Elite carryover", elite_counts, "tab:red"),
    ]

    for label, values, color in stacked_series:
        axis.bar(
            generations,
            values,
            width=0.72,
            bottom=bottom,
            label=label,
            color=color,
            alpha=0.82,
        )
        bottom = [previous + value for previous, value in zip(bottom, values)]

    axis.plot(
        generations,
        unique_configuration_counts,
        marker="o",
        linewidth=2.0,
        color="black",
        label="Unique configs in generation",
    )
    axis.set_xlabel("Outer GA generation")
    axis.set_ylabel("Individuals / unique configurations")
    axis.set_title("Population origin structure across generations")
    axis.set_xticks(generations)
    axis.set_ylim(0, max(max(population_sizes), max(unique_configuration_counts)) * 1.08)
    axis.grid(True, axis="y", alpha=0.3)
    axis.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))

    fig.tight_layout(rect=(0, 0, 0.78, 1))
    fig.savefig(save_path, dpi=200)
    maybe_show_plot(show_plot)
    plt.close(fig)
    return save_path.resolve()


def plot_generation_best_survival(survival_rows, save_path, show_plot=False):
    save_path = Path(save_path)
    if not survival_rows:
        return None

    generations = [row["generation"] for row in survival_rows]
    future_generations_survived = [
        row["future_generations_survived"] for row in survival_rows
    ]

    fig, axis = plt.subplots(figsize=(10.5, 5.0))
    axis.plot(
        generations,
        future_generations_survived,
        marker="o",
        linewidth=2.0,
        color="tab:purple",
    )
    axis.fill_between(
        generations,
        future_generations_survived,
        color="tab:purple",
        alpha=0.14,
    )
    axis.axhline(
        1,
        color="gray",
        linestyle="--",
        linewidth=1.0,
        alpha=0.6,
        label="1 future generation",
    )
    axis.set_xlabel("Generation where the individual was generation-best")
    axis.set_ylabel("Consecutive future generations survived")
    axis.set_title("Survival of each generation-best individual")
    axis.set_xticks(generations)
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best")

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    maybe_show_plot(show_plot)
    plt.close(fig)
    return save_path.resolve()


def build_padded_axis_limits(values, padding_fraction=0.05):
    if not values:
        return None

    minimum = min(values)
    maximum = max(values)
    if math.isclose(minimum, maximum):
        padding = max(abs(minimum) * padding_fraction, 1.0)
    else:
        padding = (maximum - minimum) * padding_fraction
    return (
        max(0.0, minimum - padding),
        maximum + padding,
    )


def build_runtime_tradeoff_axis_limits(evaluation_record_groups):
    all_records = [
        record
        for evaluation_records in evaluation_record_groups
        for record in evaluation_records
    ]
    if not all_records:
        return None

    return {
        "x": build_padded_axis_limits(
            [record["avg_inner_runtime_seconds"] for record in all_records]
        ),
        "y": build_padded_axis_limits(
            [record["mean_makespan"] for record in all_records]
        ),
    }


def plot_runtime_tradeoff(
    evaluation_records,
    final_best_evaluation_id,
    save_path,
    show_plot=False,
    axis_limits=None,
):
    save_path = Path(save_path)
    runtimes = [record["avg_inner_runtime_seconds"] for record in evaluation_records]
    mean_makespans = [record["mean_makespan"] for record in evaluation_records]
    generations = [record["generation"] for record in evaluation_records]
    final_best_record = evaluation_records_by_id(evaluation_records)[final_best_evaluation_id]

    fig, axis = plt.subplots(figsize=(8.5, 6))
    scatter = axis.scatter(
        runtimes,
        mean_makespans,
        c=generations,
        cmap="viridis",
        s=60,
        alpha=0.8,
        edgecolors="black",
        linewidths=0.3,
    )
    axis.scatter(
        [final_best_record["avg_inner_runtime_seconds"]],
        [final_best_record["mean_makespan"]],
        marker="*",
        s=240,
        color="tab:red",
        edgecolors="black",
        linewidths=0.6,
        label="Final selected best",
        zorder=5,
    )
    axis.set_xlabel("Average inner runtime (seconds)")
    axis.set_ylabel("Mean makespan")
    axis.set_title("Outer evaluations: mean makespan vs average inner runtime")
    axis.grid(True, alpha=0.3)
    if axis_limits is not None:
        if axis_limits.get("x") is not None:
            axis.set_xlim(axis_limits["x"])
        if axis_limits.get("y") is not None:
            axis.set_ylim(axis_limits["y"])
    axis.legend(loc="best")
    colorbar = fig.colorbar(scatter, ax=axis)
    colorbar.set_label("Outer generation")

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    maybe_show_plot(show_plot)
    plt.close(fig)
    return save_path.resolve()


def plot_population_objective_std(history, save_path, show_plot=False):
    save_path = Path(save_path)
    generations = [row["generation"] for row in history]
    population_std = [row["population_objective_std"] for row in history]
    uses_weighted_objective = any(
        "generation_best_weighted_objective_score" in row
        or "generation_best_fitness_score" in row
        for row in history
    )

    fig, axis = plt.subplots(figsize=(10, 4.8))
    axis.plot(
        generations,
        population_std,
        marker="o",
        linewidth=1.8,
        color="tab:purple",
        label="Population objective std",
    )
    axis.set_xlabel("Outer GA generation")
    if uses_weighted_objective:
        axis.set_ylabel("Std of weighted fitness score")
        axis.set_title("Population weighted-objective spread per generation")
    else:
        axis.set_ylabel("Std of population mean makespan")
        axis.set_title("Population objective spread per generation")
    axis.grid(True, alpha=0.3)
    axis.set_xticks(generations)
    axis.legend(loc="best")

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    maybe_show_plot(show_plot)
    plt.close(fig)
    return save_path.resolve()


def plot_validation_boxplot(validation_results, save_path, show_plot=False):
    if not validation_results:
        return None

    save_path = Path(save_path)
    fig, axis = plt.subplots(figsize=(13, 6.5))
    makespan_groups = [
        [run["makespan"] for run in result["validation_runs"]]
        for result in validation_results
    ]
    labels = [
        (
            f"{result['candidate_label']}\n"
            f"train={result['training_mean_makespan']:.1f}\n"
            f"val={result['validation_mean_makespan']:.1f}"
        )
        for result in validation_results
    ]
    axis.boxplot(makespan_groups, labels=labels, patch_artist=True)
    axis.set_xlabel("Top outer candidates validated on unseen inner-GA seeds")
    axis.set_ylabel("Validation makespan")
    axis.set_title("Validation boxplot on unseen seeds")
    axis.grid(True, axis="y", alpha=0.3)

    summary_text = "\n\n".join(
        (
            f"{result['candidate_label']}: "
            f"train_std={result['training_std_makespan']:.2f}, "
            f"val_std={result['validation_std_makespan']:.2f}\n"
            f"{format_hyperparameters_for_plot(result['hyperparameters'])}"
        )
        for result in validation_results
    )
    fig.text(
        0.99,
        0.5,
        summary_text,
        ha="right",
        va="center",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9},
    )
    fig.subplots_adjust(right=0.72)

    fig.tight_layout(rect=(0, 0, 0.72, 1))
    fig.savefig(save_path, dpi=200)
    maybe_show_plot(show_plot)
    plt.close(fig)
    return save_path.resolve()


def resolve_existing_outer_ga_output_paths(run_dir):
    run_dir = Path(run_dir)
    log_paths = sorted(run_dir.glob("*.log"))
    history_paths = sorted(run_dir.glob("*_history.json"))
    evaluation_log_paths = sorted(run_dir.glob("*_evaluations.jsonl"))
    best_result_paths = sorted(run_dir.glob("*_best_result.json"))

    if len(log_paths) != 1 or len(history_paths) != 1 or len(evaluation_log_paths) != 1:
        raise ValueError(
            "Expected exactly one outer-GA log, history JSON, and evaluations JSONL in "
            f"{run_dir}."
        )

    if len(best_result_paths) != 1:
        raise ValueError(
            f"Expected exactly one best-result JSON file in {run_dir}."
        )

    run_name = log_paths[0].stem
    return {
        "run_dir": run_dir,
        "run_name": run_name,
        "log_path": log_paths[0],
        "history_path": history_paths[0],
        "evaluation_log_path": evaluation_log_paths[0],
        "best_result_path": best_result_paths[0],
        "plot_output_path": run_dir / f"{run_name}.png",
        "repeat_stability_plot_path": run_dir / f"{run_name}_repeat_stability.png",
        "objective_components_plot_path": run_dir / f"{run_name}_objective_components.png",
        "hyperparameter_trajectory_plot_path": run_dir / f"{run_name}_hyperparameter_trajectories.png",
        "runtime_tradeoff_plot_path": run_dir / f"{run_name}_runtime_tradeoff.png",
        "population_std_plot_path": run_dir / f"{run_name}_population_std.png",
        "hyperparameter_diversity_plot_path": run_dir / f"{run_name}_hyperparameter_diversity.png",
        "generation_novelty_plot_path": run_dir / f"{run_name}_generation_novelty.png",
        "population_origin_plot_path": run_dir / f"{run_name}_population_origin.png",
        "best_individual_survival_plot_path": (
            run_dir / f"{run_name}_best_individual_survival.png"
        ),
        "validation_boxplot_path": run_dir / f"{run_name}_validation_boxplot.png",
        "validation_results_path": run_dir / f"{run_name}_validation_results.json",
    }


def infer_tradeoff_group_key_from_run_dir(run_dir):
    run_dir = Path(run_dir)
    match = re.search(r"_(\d+T)_", run_dir.name)
    if match:
        return match.group(1)
    return run_dir.name


def load_existing_outer_ga_run_artifacts(run_dir):
    output_paths = resolve_existing_outer_ga_output_paths(run_dir)
    return {
        "output_paths": output_paths,
        "history": load_json(output_paths["history_path"]),
        "evaluation_records": load_jsonl(output_paths["evaluation_log_path"]),
        "best_result": load_json(output_paths["best_result_path"]),
        "outer_ga_settings": parse_outer_ga_settings_from_log(output_paths["log_path"]),
        "population_evaluation_snapshots": parse_population_evaluation_ids_from_log(
            output_paths["log_path"]
        ),
        "tradeoff_group_key": infer_tradeoff_group_key_from_run_dir(run_dir),
    }


def generate_outer_ga_analysis_plots(
    history,
    evaluation_records,
    best_evaluation_id,
    output_paths,
    show_plot=False,
    population_evaluation_snapshots=None,
    outer_ga_settings=None,
    runtime_tradeoff_axis_limits=None,
):
    plot_history = history
    novelty_rows = None
    population_origin_rows = None
    generation_best_survival_rows = None
    if evaluation_records and population_evaluation_snapshots:
        plot_history = build_consistent_fitness_history(
            history,
            evaluation_records,
            population_evaluation_snapshots,
        )
        novelty_rows = generation_novelty_rows_from_population_snapshots(
            population_evaluation_snapshots,
            evaluation_records,
        )
    if evaluation_records and population_evaluation_snapshots and outer_ga_settings:
        population_origin_rows = population_origin_rows_from_population_snapshots(
            population_evaluation_snapshots,
            evaluation_records,
            outer_ga_settings,
        )
    if population_evaluation_snapshots:
        generation_best_survival_rows = generation_best_survival_rows_from_history(
            history,
            population_evaluation_snapshots,
        )

    plot_path = plot_outer_ga_history(
        plot_history,
        save_path=output_paths["plot_output_path"],
        show_plot=show_plot,
    )
    repeat_stability_path = plot_generation_best_repeat_stability(
        plot_history,
        evaluation_records or [],
        best_evaluation_id,
        save_path=output_paths["repeat_stability_plot_path"],
        show_plot=False,
    )
    objective_components_path = plot_outer_objective_components(
        plot_history,
        save_path=output_paths["objective_components_plot_path"],
        show_plot=False,
    )
    hyperparameter_trajectory_path = plot_best_hyperparameter_trajectories(
        plot_history,
        save_path=output_paths["hyperparameter_trajectory_plot_path"],
        show_plot=False,
    )
    runtime_tradeoff_path = plot_runtime_tradeoff(
        evaluation_records or [],
        best_evaluation_id,
        save_path=output_paths["runtime_tradeoff_plot_path"],
        show_plot=False,
        axis_limits=runtime_tradeoff_axis_limits,
    )
    population_std_path = plot_population_objective_std(
        plot_history,
        save_path=output_paths["population_std_plot_path"],
        show_plot=False,
    )
    hyperparameter_diversity_path = plot_hyperparameter_diversity(
        plot_history,
        save_path=output_paths["hyperparameter_diversity_plot_path"],
        show_plot=False,
    )
    generation_novelty_path = plot_generation_novelty(
        novelty_rows,
        save_path=output_paths["generation_novelty_plot_path"],
        show_plot=False,
    )
    population_origin_path = plot_population_origin(
        population_origin_rows,
        save_path=output_paths["population_origin_plot_path"],
        show_plot=False,
    )
    best_individual_survival_path = plot_generation_best_survival(
        generation_best_survival_rows,
        save_path=output_paths["best_individual_survival_plot_path"],
        show_plot=False,
    )

    plot_paths = {
        "main_history": plot_path,
        "repeat_stability": repeat_stability_path,
        "objective_components": objective_components_path,
        "hyperparameter_trajectories": hyperparameter_trajectory_path,
        "runtime_tradeoff": runtime_tradeoff_path,
        "population_objective_std": population_std_path,
        "hyperparameter_diversity": hyperparameter_diversity_path,
        "generation_novelty": generation_novelty_path,
        "population_origin": population_origin_path,
        "best_individual_survival": best_individual_survival_path,
    }

    validation_results_path = output_paths.get("validation_results_path")
    validation_boxplot_path = output_paths.get("validation_boxplot_path")
    if validation_results_path is not None and Path(validation_results_path).exists():
        validation_payload = load_json(validation_results_path)
        validation_plot_path = plot_validation_boxplot(
            validation_payload.get("results", []),
            save_path=validation_boxplot_path,
            show_plot=False,
        )
        if validation_plot_path is not None:
            plot_paths["validation_boxplot"] = validation_plot_path

    return plot_history, plot_paths


def replot_existing_outer_ga_run(run_dir, show_plot=False):
    return replot_existing_outer_ga_runs([run_dir], show_plot=show_plot)[0]


def replot_existing_outer_ga_runs(run_dirs, show_plot=False):
    loaded_runs = [
        load_existing_outer_ga_run_artifacts(run_dir)
        for run_dir in run_dirs
    ]
    axis_limits_by_group = {}
    grouped_evaluation_records = {}
    for run_data in loaded_runs:
        grouped_evaluation_records.setdefault(
            run_data["tradeoff_group_key"],
            [],
        ).append(run_data["evaluation_records"])

    for group_key, evaluation_record_groups in grouped_evaluation_records.items():
        axis_limits_by_group[group_key] = build_runtime_tradeoff_axis_limits(
            evaluation_record_groups
        )

    results = []
    for run_data in loaded_runs:
        output_paths = run_data["output_paths"]
        tradeoff_axis_limits = axis_limits_by_group[run_data["tradeoff_group_key"]]
        plot_history, plot_paths = generate_outer_ga_analysis_plots(
            run_data["history"],
            run_data["evaluation_records"],
            run_data["best_result"]["evaluation_id"],
            output_paths=output_paths,
            show_plot=show_plot,
            population_evaluation_snapshots=run_data["population_evaluation_snapshots"],
            outer_ga_settings=run_data["outer_ga_settings"],
            runtime_tradeoff_axis_limits=tradeoff_axis_limits,
        )
        results.append(
            {
                "run_dir": output_paths["run_dir"].resolve(),
                "plot_history": plot_history,
                "plot_paths": plot_paths,
                "runtime_tradeoff_axis_limits": tradeoff_axis_limits,
                "tradeoff_group_key": run_data["tradeoff_group_key"],
            }
        )
    return results


def run_default_outer_ga(
    input_json_path=None,
    config=None,
    show_plot=None,
    outer_pop_size=None,
    outer_ngen=None,
    inner_repeats=None,
    random_seed=None,
    tournament_size=None,
    elite_count=None,
    random_immigrant_rate=None,
    outer_cxpb=None,
    outer_mutpb=None,
    log_population_details=None,
):
    config = config or load_config()
    analysis_settings = get_analysis_settings(config)
    runtime_settings = resolve_outer_runtime_settings(
        config,
        show_plot=show_plot,
        outer_pop_size=outer_pop_size,
        outer_ngen=outer_ngen,
        inner_repeats=inner_repeats,
        random_seed=random_seed,
        tournament_size=tournament_size,
        elite_count=elite_count,
        random_immigrant_rate=random_immigrant_rate,
        outer_cxpb=outer_cxpb,
        outer_mutpb=outer_mutpb,
        log_population_details=log_population_details,
    )
    show_plot = runtime_settings["show_plot"]
    outer_pop_size = runtime_settings["outer_pop_size"]
    outer_ngen = runtime_settings["outer_ngen"]
    inner_repeats = runtime_settings["inner_repeats"]
    random_seed = runtime_settings["random_seed"]
    tournament_size = runtime_settings["tournament_size"]
    elite_count = runtime_settings["elite_count"]
    random_immigrant_rate = runtime_settings["random_immigrant_rate"]
    outer_cxpb = runtime_settings["outer_cxpb"]
    outer_mutpb = runtime_settings["outer_mutpb"]
    log_population_details = runtime_settings["log_population_details"]
    objective_weights = runtime_settings["objective_weights"]
    top_k_validation_candidates = analysis_settings["top_k_validation_candidates"]
    validation_repeats = analysis_settings["validation_repeats"]
    validation_seed_offset = analysis_settings["validation_seed_offset"]
    random_immigrant_count = resolve_random_immigrant_count(
        outer_pop_size,
        elite_count,
        random_immigrant_rate,
    )

    problem = load_problem(input_json_path=input_json_path, config=config)

    am_name = problem["INPUT_JSON_PATH"].stem
    output_paths = create_run_output_paths(am_name, config=config)
    run_dir = output_paths["run_dir"]
    log_path = output_paths["log_path"]
    plot_output_path = output_paths["plot_path"]
    history_path = output_paths["history_path"]
    evaluation_log_path = output_paths["evaluation_log_path"]
    best_result_path = output_paths["best_result_path"]
    repeat_stability_plot_path = output_paths["repeat_stability_plot_path"]
    objective_components_plot_path = output_paths["objective_components_plot_path"]
    hyperparameter_trajectory_plot_path = output_paths["hyperparameter_trajectory_plot_path"]
    runtime_tradeoff_plot_path = output_paths["runtime_tradeoff_plot_path"]
    population_std_plot_path = output_paths["population_std_plot_path"]
    hyperparameter_diversity_plot_path = output_paths["hyperparameter_diversity_plot_path"]
    generation_novelty_plot_path = output_paths["generation_novelty_plot_path"]
    population_origin_plot_path = output_paths["population_origin_plot_path"]
    best_individual_survival_plot_path = output_paths["best_individual_survival_plot_path"]
    validation_boxplot_path = output_paths["validation_boxplot_path"]
    validation_results_path = output_paths["validation_results_path"]

    with log_path.open("w", encoding="utf-8", buffering=1) as log_file, redirect_stdout(log_file):
        print("Outer GA hyperparameter tuning log")
        print("AM being tested:", am_name)
        print("Input JSON:", problem["INPUT_JSON_PATH"].resolve())
        print("Config file:", config["_meta"]["config_path"])
        print("Timestamp:", datetime.now().isoformat(timespec="seconds"))
        print("Run folder:", run_dir.resolve())
        print("Log file:", log_path.resolve())
        print("History file:", history_path.resolve())
        print("Evaluation log:", evaluation_log_path.resolve())
        print("Best result file:", best_result_path.resolve())
        print("Repeat stability plot:", repeat_stability_plot_path.resolve())
        print("Objective component plot:", objective_components_plot_path.resolve())
        print("Hyperparameter trajectory plot:", hyperparameter_trajectory_plot_path.resolve())
        print("Runtime tradeoff plot:", runtime_tradeoff_plot_path.resolve())
        print("Population std plot:", population_std_plot_path.resolve())
        print("Hyperparameter diversity plot:", hyperparameter_diversity_plot_path.resolve())
        print("Generation novelty plot:", generation_novelty_plot_path.resolve())
        print("Validation boxplot:", validation_boxplot_path.resolve())
        print("Validation results file:", validation_results_path.resolve())
        print(
            "Outer GA settings:",
            {
                "outer_pop_size": outer_pop_size,
                "outer_ngen": outer_ngen,
                "inner_repeats": inner_repeats,
                "random_seed": random_seed,
                "tournament_size": tournament_size,
                "elite_count": elite_count,
                "random_immigrant_rate": random_immigrant_rate,
                "random_immigrant_count": random_immigrant_count,
                "outer_crossover_probability": outer_cxpb,
                "outer_mutation_probability": outer_mutpb,
                "objective_weights": objective_weights,
            },
        )
        print(
            "Analysis settings:",
            {
                "top_k_validation_candidates": top_k_validation_candidates,
                "validation_repeats": validation_repeats,
                "validation_seed_offset": validation_seed_offset,
            },
        )

        result = outer_ga(
            problem["processor_ids"],
            problem["processing_times"],
            problem["message_list"],
            problem["merged_paths_dict"],
            config=config,
            outer_pop_size=outer_pop_size,
            outer_ngen=outer_ngen,
            inner_repeats=inner_repeats,
            random_seed=random_seed,
            tournament_size=tournament_size,
            elite_count=elite_count,
            random_immigrant_rate=random_immigrant_rate,
            outer_cxpb=outer_cxpb,
            outer_mutpb=outer_mutpb,
            evaluation_log_path=evaluation_log_path,
            history_path=history_path,
            best_result_path=best_result_path,
            log_population_details=log_population_details,
        )
        plot_history, analysis_plot_paths = generate_outer_ga_analysis_plots(
            result.history,
            result.evaluation_records or [],
            result.best_evaluation_id,
            output_paths={
                "plot_output_path": plot_output_path,
                "repeat_stability_plot_path": repeat_stability_plot_path,
                "objective_components_plot_path": objective_components_plot_path,
                "hyperparameter_trajectory_plot_path": hyperparameter_trajectory_plot_path,
                "runtime_tradeoff_plot_path": runtime_tradeoff_plot_path,
                "population_std_plot_path": population_std_plot_path,
                "hyperparameter_diversity_plot_path": hyperparameter_diversity_plot_path,
                "generation_novelty_plot_path": generation_novelty_plot_path,
                "population_origin_plot_path": population_origin_plot_path,
                "best_individual_survival_plot_path": best_individual_survival_plot_path,
            },
            show_plot=show_plot,
            population_evaluation_snapshots=result.population_evaluation_snapshots,
            outer_ga_settings={
                "outer_pop_size": outer_pop_size,
                "elite_count": elite_count,
                "random_immigrant_rate": random_immigrant_rate,
                "random_immigrant_count": random_immigrant_count,
            },
        )
        plot_path = analysis_plot_paths["main_history"]

        top_candidate_records = select_top_distinct_candidates(
            result.evaluation_records or [],
            top_k_validation_candidates,
        )
        validation_seeds = build_validation_seeds(
            random_seed,
            validation_repeats,
            validation_seed_offset,
        )
        validation_results = validate_top_candidates_on_unseen_seeds(
            top_candidate_records,
            problem,
            config,
            validation_seeds,
        )
        validation_payload = {
            "description": (
                "Validation results for the top outer candidates on unseen inner-GA seeds. "
                "These seeds were not used during the outer search and are meant to test "
                "whether the selected hyperparameters remain strong and stable on new randomness."
            ),
            "top_k_validation_candidates": top_k_validation_candidates,
            "validation_repeats": validation_repeats,
            "validation_seed_offset": validation_seed_offset,
            "validation_seeds": validation_seeds,
            "results": validation_results,
        }
        write_json(validation_results_path, validation_payload)
        validation_boxplot_resolved_path = plot_validation_boxplot(
            validation_results,
            save_path=validation_boxplot_path,
            show_plot=False,
        )

        result.run_dir = run_dir.resolve()
        result.log_path = log_path.resolve()
        result.plot_path = plot_path
        result.history_path = history_path.resolve()
        result.evaluation_log_path = evaluation_log_path.resolve()
        result.best_result_path = best_result_path.resolve()
        result.history = plot_history
        result.analysis_plot_paths = dict(analysis_plot_paths)
        result.analysis_plot_paths["validation_boxplot"] = validation_boxplot_resolved_path
        result.validation_results_path = validation_results_path.resolve()

    print("Outer GA run folder:", run_dir.resolve())
    print("Outer GA log saved to:", log_path.resolve())
    print("Outer GA history saved to:", history_path.resolve())
    print("Outer GA evaluation log saved to:", evaluation_log_path.resolve())
    print("Outer GA best result saved to:", best_result_path.resolve())
    print("Outer GA plot saved to:", result.plot_path)
    if result.analysis_plot_paths:
        print("Repeat stability plot saved to:", result.analysis_plot_paths["repeat_stability"])
        print("Objective component plot saved to:", result.analysis_plot_paths["objective_components"])
        print(
            "Hyperparameter trajectory plot saved to:",
            result.analysis_plot_paths["hyperparameter_trajectories"],
        )
        print("Runtime tradeoff plot saved to:", result.analysis_plot_paths["runtime_tradeoff"])
        print(
            "Population objective std plot saved to:",
            result.analysis_plot_paths["population_objective_std"],
        )
        print(
            "Hyperparameter diversity plot saved to:",
            result.analysis_plot_paths["hyperparameter_diversity"],
        )
        print(
            "Generation novelty plot saved to:",
            result.analysis_plot_paths["generation_novelty"],
        )
        print(
            "Population origin plot saved to:",
            result.analysis_plot_paths["population_origin"],
        )
        print(
            "Best-individual survival plot saved to:",
            result.analysis_plot_paths["best_individual_survival"],
        )
        print("Validation boxplot saved to:", result.analysis_plot_paths["validation_boxplot"])
    if result.validation_results_path is not None:
        print("Validation results saved to:", result.validation_results_path)
    print("Final best hyperparameters:", result.best_hyperparameters)
    print("Final best mean makespan:", result.best_makespan)
    print("Final best avg inner runtime seconds:", result.best_avg_inner_runtime_seconds)
    print("Final best evaluation id:", result.best_evaluation_id)
    return result


def example_sort_key(path):
    match = re.search(r"example_(\d+)T\.json$", path.name)
    return int(match.group(1)) if match else path.name


def find_example_json_paths():
    return sorted(Path(".").glob("example_*T.json"), key=example_sort_key)


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
    outer_settings = get_outer_ga_settings(config)

    parser = argparse.ArgumentParser(description="Run outer GA tuning for one or more example JSON files.")
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
        help="Example JSON file(s), such as example_40T.json.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run every example_*T.json file in this folder, in numeric order.",
    )
    parser.add_argument(
        "--show-plot",
        action="store_true",
        help="Display plots after saving them.",
    )
    parser.add_argument(
        "--outer-pop-size",
        type=int,
        default=outer_settings["population_size"],
        help=f"Outer GA population size. Default from config: {outer_settings['population_size']}.",
    )
    parser.add_argument(
        "--outer-ngen",
        type=int,
        default=outer_settings["generations"],
        help=f"Number of outer GA generations. Default from config: {outer_settings['generations']}.",
    )
    parser.add_argument(
        "--inner-repeats",
        type=int,
        default=outer_settings["inner_repeats"],
        help="How many times to run the inner GA per outer individual and average the makespan.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=outer_settings["random_seed"],
        help="Optional random seed for reproducible outer search and reproducible inner-GA repeat seeds.",
    )
    parser.add_argument(
        "--tournament-size",
        type=int,
        default=outer_settings["tournament_size"],
        help=(
            "Outer GA tournament size for selection pressure. "
            f"Default from config: {outer_settings['tournament_size']}."
        ),
    )
    parser.add_argument(
        "--elite-count",
        type=int,
        default=outer_settings["elite_count"],
        help=(
            "Number of elite outer individuals kept every generation. "
            f"Default from config: {outer_settings['elite_count']}."
        ),
    )
    parser.add_argument(
        "--random-immigrant-rate",
        type=float,
        default=outer_settings["random_immigrant_rate"],
        help=(
            "Fraction of the outer population replaced by fresh random immigrants "
            f"each generation. Default from config: {outer_settings['random_immigrant_rate']:.3f}."
        ),
    )
    parser.add_argument(
        "--outer-cxpb",
        type=float,
        default=outer_settings["crossover_probability"],
        help=(
            "Outer GA crossover probability. "
            f"Default from config: {outer_settings['crossover_probability']}."
        ),
    )
    parser.add_argument(
        "--outer-mutpb",
        type=float,
        default=outer_settings["mutation_probability"],
        help=(
            "Outer GA mutation probability. "
            f"Default from config: {outer_settings['mutation_probability']}."
        ),
    )
    parser.add_argument(
        "--log-population-details",
        action="store_true",
        help="Print every outer individual and whether its fitness is reused or recomputed.",
    )
    parser.add_argument(
        "--plot-hyperparameter-diversity-from-log",
        type=Path,
        default=None,
        help=(
            "Parse an existing outer-GA log with population details and create a "
            "hyperparameter-diversity JSON plus PNG without starting a new GA run."
        ),
    )
    parser.add_argument(
        "--replot-run-dir",
        nargs="+",
        type=Path,
        default=None,
        help=(
            "Regenerate the history-based plots for one or more existing outer-GA run "
            "folders without launching new experiments."
        ),
    )
    args = parser.parse_args(remaining)

    # Preserve the config default when the flag is not supplied on the command line.
    if not args.show_plot:
        args.show_plot = outer_settings["show_plot"]
    if not args.log_population_details:
        args.log_population_details = outer_settings["log_population_details"]

    return args, config


if __name__ == "__main__":
    args, config = parse_args()
    if args.plot_hyperparameter_diversity_from_log is not None:
        plot_path, data_path, rows = create_hyperparameter_diversity_artifacts_from_log(
            args.plot_hyperparameter_diversity_from_log,
            show_plot=args.show_plot,
        )
        print("Hyperparameter diversity rows parsed:", len(rows))
        print("Hyperparameter diversity data saved to:", data_path)
        print("Hyperparameter diversity plot saved to:", plot_path)
        raise SystemExit(0)

    if args.replot_run_dir is not None:
        for result in replot_existing_outer_ga_runs(
            args.replot_run_dir,
            show_plot=args.show_plot,
        ):
            print("Replotted run folder:", result["run_dir"])
            print(
                "Runtime tradeoff axis group:",
                result["tradeoff_group_key"],
                "limits:",
                result["runtime_tradeoff_axis_limits"],
            )
            for plot_name, plot_path in result["plot_paths"].items():
                print(f"{plot_name} plot saved to:", plot_path)
        raise SystemExit(0)

    if args.all:
        input_json_paths = find_example_json_paths()
    elif args.input_json_paths:
        input_json_paths = args.input_json_paths
    else:
        input_json_paths = [None]

    for input_json_path in input_json_paths:
        run_default_outer_ga(
            input_json_path=input_json_path,
            config=config,
            show_plot=args.show_plot,
            outer_pop_size=args.outer_pop_size,
            outer_ngen=args.outer_ngen,
            inner_repeats=args.inner_repeats,
            random_seed=args.random_seed,
            tournament_size=args.tournament_size,
            elite_count=args.elite_count,
            random_immigrant_rate=args.random_immigrant_rate,
            outer_cxpb=args.outer_cxpb,
            outer_mutpb=args.outer_mutpb,
            log_population_details=args.log_population_details,
        )
