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
    best_avg_inner_runtime_seconds: float | None
    history: list[dict]
    evaluation_records: list[dict] | None = None
    best_evaluation_id: int | None = None
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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    configured_log_dir = log_dir if log_dir is not None else config["paths"]["log_dir"]
    run_dir = resolve_config_path(config, configured_log_dir) / f"{safe_am_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return {
        "run_dir": run_dir,
        "log_path": run_dir / f"{safe_am_name}_{timestamp}.log",
        "plot_path": run_dir / f"{safe_am_name}_{timestamp}.png",
        "history_path": run_dir / f"{safe_am_name}_{timestamp}_history.json",
        "evaluation_log_path": run_dir / f"{safe_am_name}_{timestamp}_evaluations.jsonl",
        "best_result_path": run_dir / f"{safe_am_name}_{timestamp}_best_result.json",
        "repeat_stability_plot_path": run_dir / f"{safe_am_name}_{timestamp}_repeat_stability.png",
        "hyperparameter_trajectory_plot_path": run_dir / f"{safe_am_name}_{timestamp}_hyperparameter_trajectories.png",
        "runtime_tradeoff_plot_path": run_dir / f"{safe_am_name}_{timestamp}_runtime_tradeoff.png",
        "population_std_plot_path": run_dir / f"{safe_am_name}_{timestamp}_population_std.png",
        "hyperparameter_diversity_plot_path": run_dir / f"{safe_am_name}_{timestamp}_hyperparameter_diversity.png",
        "validation_boxplot_path": run_dir / f"{safe_am_name}_{timestamp}_validation_boxplot.png",
        "validation_results_path": run_dir / f"{safe_am_name}_{timestamp}_validation_results.json",
    }


def append_jsonl(path, record):
    with Path(path).open("a", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=True)
        handle.write("\n")


def write_json(path, payload):
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


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


def count_unique_hyperparameters(population, config):
    return len({hyperparameter_signature(individual, config) for individual in population})


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
                f", mean_makespan={individual.fitness.values[0]}"
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
        "avg_inner_runtime_seconds",
        "total_inner_runtime_seconds",
        "inner_run_runtime_seconds",
        "inner_run_makespans",
        "evaluation_id",
        "best_inner_makespan",
        "best_inner_repeat_index",
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
    candidate_makespan = candidate.fitness.values[0]
    incumbent_makespan = incumbent.fitness.values[0]
    if candidate_makespan < incumbent_makespan:
        return True
    if candidate_makespan > incumbent_makespan:
        return False

    candidate_runtime = getattr(candidate, "avg_inner_runtime_seconds", float("inf"))
    incumbent_runtime = getattr(incumbent, "avg_inner_runtime_seconds", float("inf"))
    return candidate_runtime < incumbent_runtime


def build_generation_history_row(
    generation,
    population,
    best_so_far,
    immigrants_introduced,
    immigrant_rate,
    config,
    evaluation_archive,
):
    generation_best = tools.selBest(population, 1)[0]
    makespans = [individual.fitness.values[0] for individual in population]
    generation_best_mean_makespan = generation_best.fitness.values[0]
    generation_avg_mean_makespan = mean(makespans)
    generation_worst_mean_makespan = max(makespans)
    best_so_far_mean_makespan = best_so_far.fitness.values[0]
    generation_best_record = evaluation_archive[getattr(generation_best, "evaluation_id")]
    best_so_far_record = evaluation_archive[getattr(best_so_far, "evaluation_id")]

    return {
        "objective_name": "mean_makespan",
        "generation": generation,
        "population_size": len(population),
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
        "best_so_far_hyperparameters": individual_to_dict(best_so_far, config),
        "generation_best_avg_inner_runtime_seconds": getattr(
            generation_best,
            "avg_inner_runtime_seconds",
            None,
        ),
        "generation_best_repeat_std_makespan": standard_deviation(
            repeat_makespans(generation_best_record)
        ),
        "best_so_far_avg_inner_runtime_seconds": getattr(
            best_so_far,
            "avg_inner_runtime_seconds",
            None,
        ),
        "best_so_far_repeat_std_makespan": standard_deviation(
            repeat_makespans(best_so_far_record)
        ),
        "generation_best_evaluation_id": getattr(generation_best, "evaluation_id", None),
        "best_so_far_evaluation_id": getattr(best_so_far, "evaluation_id", None),
        "population_objective_std": standard_deviation(makespans),
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

        if evaluation_log_path is not None:
            append_jsonl(evaluation_log_path, evaluation_record)

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

        return (avg_makespan,)

    population = toolbox.population(n=outer_pop_size)

    print_population_consistency(
        "Initial outer GA",
        population,
        outer_pop_size,
        config,
        log_details=log_population_details,
    )
    for index, individual in enumerate(population, start=1):
        individual.fitness.values = evaluate_outer(
            individual,
            generation=0,
            population_index=index,
            stage="initial_population",
        )

    best_so_far = toolbox.clone(tools.selBest(population, 1)[0])
    best_record = evaluation_archive[getattr(best_so_far, "evaluation_id")]
    history = [
        build_generation_history_row(
            generation=0,
            population=population,
            best_so_far=best_so_far,
            immigrants_introduced=0,
            immigrant_rate=random_immigrant_rate,
            config=config,
            evaluation_archive=evaluation_archive,
        )
    ]

    if history_path is not None:
        write_json(history_path, history)
    if best_result_path is not None:
        write_json(best_result_path, best_record)

    print("Initial outer GA best hyperparameters:", individual_to_dict(best_so_far, config))
    print("Initial outer GA best mean makespan:", best_so_far.fitness.values[0])

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
        for index, individual in invalid_individuals:
            individual.fitness.values = evaluate_outer(
                individual,
                generation=generation,
                population_index=index,
                stage="offspring",
            )

        population[:] = offspring

        generation_best = tools.selBest(population, 1)[0]
        if is_better_outer_candidate(generation_best, best_so_far):
            best_so_far = toolbox.clone(generation_best)
            best_record = evaluation_archive[getattr(best_so_far, "evaluation_id")]
            if best_result_path is not None:
                write_json(best_result_path, best_record)

        row = build_generation_history_row(
            generation=generation,
            population=population,
            best_so_far=best_so_far,
            immigrants_introduced=random_immigrant_count,
            immigrant_rate=random_immigrant_rate,
            config=config,
            evaluation_archive=evaluation_archive,
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
        best_hyperparameters=individual_to_dict(best_so_far, config),
        best_makespan=best_so_far.fitness.values[0],
        best_avg_inner_runtime_seconds=getattr(best_so_far, "avg_inner_runtime_seconds", None),
        history=history,
        evaluation_records=sorted(
            evaluation_archive.values(),
            key=lambda record: record["evaluation_id"],
        ),
        best_evaluation_id=getattr(best_so_far, "evaluation_id", None),
    )
    return result


def plot_outer_ga_history(history, save_path="outer_ga_hyperparameter_search.png", show_plot=False):
    save_path = Path(save_path)
    generations = [row["generation"] for row in history]
    generation_best = [
        row.get("generation_best_mean_makespan", row["generation_best_makespan"])
        for row in history
    ]
    generation_avg = [
        row.get("generation_avg_mean_makespan", row["generation_avg_makespan"])
        for row in history
    ]
    unique_configs = [row["unique_hyperparameter_count"] for row in history]
    runtimes = [row.get("generation_best_avg_inner_runtime_seconds") for row in history]
    has_runtime_values = any(runtime is not None for runtime in runtimes)
    best_history_row = min(
        history,
        key=lambda row: (row.get("best_so_far_mean_makespan", row["best_so_far_makespan"]), row["generation"]),
    )

    improvement_rows = []
    current_best = None
    for row in history:
        objective_value = row.get("best_so_far_mean_makespan", row["best_so_far_makespan"])
        if current_best is None or objective_value < current_best:
            improvement_rows.append(row)
            current_best = objective_value

    fig = plt.figure(figsize=(15, 10.2), constrained_layout=True)
    grid = fig.add_gridspec(
        3,
        2,
        width_ratios=[3.8, 1.45],
        height_ratios=[1.2, 0.85, 0.85],
    )
    makespan_axis = fig.add_subplot(grid[0, 0])
    diversity_axis = fig.add_subplot(grid[1, 0], sharex=makespan_axis)
    runtime_axis = fig.add_subplot(grid[2, 0], sharex=makespan_axis)
    summary_axis = fig.add_subplot(grid[:, 1])
    summary_axis.axis("off")
    fig.suptitle("Outer GA mean-makespan objective progress", fontsize=18)

    generation_best_line = makespan_axis.plot(
        generations,
        generation_best,
        marker="s",
        linewidth=1.5,
        color="tab:green",
        label="Generation best",
    )[0]
    generation_avg_line = makespan_axis.plot(
        generations,
        generation_avg,
        marker="^",
        linewidth=1.5,
        color="tab:gray",
        label="Generation average",
    )[0]
    new_best_points = makespan_axis.scatter(
        [row["generation"] for row in improvement_rows],
        [
            row.get("best_so_far_mean_makespan", row["best_so_far_makespan"])
            for row in improvement_rows
        ],
        marker="*",
        s=140,
        color="tab:red",
        label="New best",
        zorder=5,
    )
    for row in improvement_rows:
        makespan_axis.annotate(
            f"g{row['generation']}",
            xy=(row["generation"], row.get("best_so_far_mean_makespan", row["best_so_far_makespan"])),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color="tab:red",
            fontweight="bold",
        )

    makespan_axis.set_ylabel("Mean makespan")
    makespan_axis.set_title("Objective curves")
    makespan_axis.grid(True, alpha=0.3)
    makespan_axis.legend(
        handles=[generation_best_line, generation_avg_line, new_best_points],
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02, 1.0, 0.2),
        mode="expand",
        ncol=3,
        frameon=False,
        borderaxespad=0.0,
    )

    unique_config_line = diversity_axis.plot(
        generations,
        unique_configs,
        marker="o",
        linewidth=2,
        color="tab:purple",
        label="Unique configs",
    )[0]
    diversity_axis.set_ylabel("Unique configs")
    diversity_axis.set_title("Population diversity")
    diversity_axis.grid(True, alpha=0.3)
    if has_runtime_values:
        runtime_line = runtime_axis.plot(
            generations,
            runtimes,
            marker="s",
            linewidth=1.5,
            color="tab:orange",
            label="Best inner runtime",
        )[0]
        diversity_axis.legend(
            handles=[unique_config_line],
            loc="lower left",
            bbox_to_anchor=(0.0, 1.02, 1.0, 0.2),
            mode="expand",
            ncol=1,
            frameon=False,
            borderaxespad=0.0,
        )
        runtime_axis.set_ylabel("Seconds")
        runtime_axis.set_title("Generation-best inner runtime")
        runtime_axis.grid(True, alpha=0.3)
        runtime_axis.legend(
            handles=[runtime_line],
            loc="lower left",
            bbox_to_anchor=(0.0, 1.02, 1.0, 0.2),
            mode="expand",
            ncol=1,
            frameon=False,
            borderaxespad=0.0,
        )
    else:
        diversity_axis.legend(
            loc="lower left",
            bbox_to_anchor=(0.0, 1.02, 1.0, 0.2),
            mode="expand",
            ncol=1,
            frameon=False,
            borderaxespad=0.0,
        )
        runtime_axis.text(
            0.5,
            0.5,
            "No runtime data",
            ha="center",
            va="center",
            transform=runtime_axis.transAxes,
            fontsize=11,
        )
        runtime_axis.set_title("Generation-best inner runtime")
        runtime_axis.grid(True, alpha=0.3)
        runtime_axis.set_ylabel("Seconds")

    runtime_axis.set_xlabel("Outer GA generation")
    runtime_axis.set_xticks(generations)
    plt.setp(makespan_axis.get_xticklabels(), visible=False)
    plt.setp(diversity_axis.get_xticklabels(), visible=False)

    summary_lines = [
        "Run summary",
        "",
        "Objective: mean makespan",
        (
            "Final selected best\n"
            f"mean={best_history_row.get('best_so_far_mean_makespan', best_history_row['best_so_far_makespan']):.1f}"
        ),
        format_hyperparameters_for_summary(best_history_row["best_so_far_hyperparameters"]),
        "",
        "Milestones",
    ]
    for row in improvement_rows:
        summary_lines.extend(
            [
                f"g{row['generation']} -> mean={row.get('best_so_far_mean_makespan', row['best_so_far_makespan']):.1f}",
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


def plot_runtime_tradeoff(
    evaluation_records,
    final_best_evaluation_id,
    save_path,
    show_plot=False,
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
    hyperparameter_trajectory_plot_path = output_paths["hyperparameter_trajectory_plot_path"]
    runtime_tradeoff_plot_path = output_paths["runtime_tradeoff_plot_path"]
    population_std_plot_path = output_paths["population_std_plot_path"]
    hyperparameter_diversity_plot_path = output_paths["hyperparameter_diversity_plot_path"]
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
        print("Hyperparameter trajectory plot:", hyperparameter_trajectory_plot_path.resolve())
        print("Runtime tradeoff plot:", runtime_tradeoff_plot_path.resolve())
        print("Population std plot:", population_std_plot_path.resolve())
        print("Hyperparameter diversity plot:", hyperparameter_diversity_plot_path.resolve())
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
        plot_path = plot_outer_ga_history(
            result.history,
            save_path=plot_output_path,
            show_plot=show_plot,
        )
        repeat_stability_path = plot_generation_best_repeat_stability(
            result.history,
            result.evaluation_records or [],
            result.best_evaluation_id,
            save_path=repeat_stability_plot_path,
            show_plot=False,
        )
        hyperparameter_trajectory_path = plot_best_hyperparameter_trajectories(
            result.history,
            save_path=hyperparameter_trajectory_plot_path,
            show_plot=False,
        )
        runtime_tradeoff_path = plot_runtime_tradeoff(
            result.evaluation_records or [],
            result.best_evaluation_id,
            save_path=runtime_tradeoff_plot_path,
            show_plot=False,
        )
        population_std_path = plot_population_objective_std(
            result.history,
            save_path=population_std_plot_path,
            show_plot=False,
        )
        hyperparameter_diversity_path = plot_hyperparameter_diversity(
            result.history,
            save_path=hyperparameter_diversity_plot_path,
            show_plot=False,
        )

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
        result.analysis_plot_paths = {
            "main_history": plot_path,
            "repeat_stability": repeat_stability_path,
            "hyperparameter_trajectories": hyperparameter_trajectory_path,
            "runtime_tradeoff": runtime_tradeoff_path,
            "population_objective_std": population_std_path,
            "hyperparameter_diversity": hyperparameter_diversity_path,
            "validation_boxplot": validation_boxplot_resolved_path,
        }
        result.validation_results_path = validation_results_path.resolve()

    print("Outer GA run folder:", run_dir.resolve())
    print("Outer GA log saved to:", log_path.resolve())
    print("Outer GA history saved to:", history_path.resolve())
    print("Outer GA evaluation log saved to:", evaluation_log_path.resolve())
    print("Outer GA best result saved to:", best_result_path.resolve())
    print("Outer GA plot saved to:", result.plot_path)
    if result.analysis_plot_paths:
        print("Repeat stability plot saved to:", result.analysis_plot_paths["repeat_stability"])
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
