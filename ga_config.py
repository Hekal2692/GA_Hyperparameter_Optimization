"""Shared configuration helpers for the scheduler GA and the outer tuning GA."""

import json
from copy import deepcopy
from pathlib import Path


# Default config file that lives next to this helper module.
DEFAULT_CONFIG_PATH = Path(__file__).with_name("ga_config.json")


# Built-in fallback values.
# These are used first, then overridden by values from a JSON config file.
DEFAULT_CONFIG = {
    # File-system locations used by the project.
    "paths": {
        "default_input_json": "example_40T.json",
        "log_dir": "logs",
    },
    # Problem-building parameters used while creating the communication paths.
    "problem": {
        "k_shortest_paths": 4,
        "self_loop_cost": 1,
    },
    # Inner scheduler-GA behavior that is not part of the outer search space.
    "scheduler": {
        "reconstruction": {
            "fallback_path_id": "290",
        },
        "selection": {
            "tournament_size": 3,
        },
        "message_path_choices": [0, 1, 2, 3],
        "mutation": {
            "task_order_probability": 0.05,
            "processor_allocation_probability": 0.05,
            "message_priority_shuffle_probability": 0.05,
            "message_path_index_probability": 0.05,
        },
    },
    # Outer GA defaults for tuning the scheduler hyperparameters.
    "outer_ga": {
        "population_size": 100,
        "generations": 100,
        "crossover_probability": 0.4,
        "mutation_probability": 0.6,
        "tournament_size": 3,
        "elite_count": 1,
        "random_immigrant_rate": 0.1,
        "inner_repeats": 1,
        "random_seed": None,
        "show_plot": False,
        "log_population_details": False,
        "objective_weights": {
            "mean_makespan": 0.5,
            "mean_runtime_seconds": 0.5,
        },
        "evolved_hyperparameters": [
            "pop_size",
            "cxpb",
            "mutpb",
            "ngen",
            "task_order_probability",
            "processor_allocation_probability",
            "message_priority_shuffle_probability",
            "message_path_index_probability",
        ],
        "computation_budget": {
            "mode": "exact_product",
            "budget": 2520,
        },
    },
    # Fixed scheduler-GA defaults used by SchedulerGa.py as a non-adaptive baseline.
    "static_scheduler_ga": {
        "pop_size": 40,
        "cxpb": 0.45,
        "mutpb": 0.5,
        "ngen": 75,
        "benchmark_repeats": 7,
        "random_seed": None,
        "show_plot": False,
    },
    # Analysis settings for derived plots and validation on unseen inner-GA seeds.
    "analysis": {
        "top_k_validation_candidates": 3,
        "validation_repeats": 7,
        "validation_seed_offset": 100000,
    },
    # Bounds for the hyperparameters that the outer GA is allowed to evolve.
    "hyperparameter_search_space": {
        "pop_size": {"min": 20, "max": 80},
        "cxpb": {"min": 0.2, "max": 0.6},
        "mutpb": {"min": 0.2, "max": 0.8},
        "ngen": {"min": 20, "max": 100},
        "task_order_probability": {"min": 0.0, "max": 0.3},
        "processor_allocation_probability": {"min": 0.0, "max": 0.3},
        "message_priority_shuffle_probability": {"min": 0.0, "max": 0.3},
        "message_path_index_probability": {"min": 0.0, "max": 0.3},
    },
}


def _deep_merge(base, overrides):
    """Recursively merge user config values into the default config tree."""
    for key, value in overrides.items():
        # If both sides are dictionaries, merge them key by key.
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_merge(base[key], value)
        else:
            # Otherwise the override fully replaces the default value.
            base[key] = value
    return base


def load_config(config_path=None):
    """Load a config file and overlay it on top of the built-in defaults."""
    # Start from a deep copy so callers can safely modify the returned config.
    config = deepcopy(DEFAULT_CONFIG)
    resolved_config_path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    resolved_config_path = resolved_config_path.resolve()

    if resolved_config_path.exists():
        with resolved_config_path.open("r", encoding="utf-8") as handle:
            overrides = json.load(handle)
        if not isinstance(overrides, dict):
            raise ValueError(f"Config file must contain a JSON object: {resolved_config_path}")
        config = _deep_merge(config, overrides)
    elif config_path is not None:
        # Missing explicit config paths should fail loudly.
        raise FileNotFoundError(f"Config file not found: {resolved_config_path}")

    # Store resolved path metadata so the rest of the project can resolve
    # relative paths against the config file location.
    config["_meta"] = {
        "config_path": str(resolved_config_path),
        "config_dir": str(resolved_config_path.parent),
    }
    return config


def resolve_config_path(config, path_value):
    """Resolve relative config paths from the directory that owns the config file."""
    path = Path(path_value)
    if path.is_absolute():
        return path

    # Relative paths are interpreted from the config file's directory,
    # not from the current working directory.
    base_dir = Path(config["_meta"]["config_dir"]) if config and "_meta" in config else Path.cwd()
    return (base_dir / path).resolve()
