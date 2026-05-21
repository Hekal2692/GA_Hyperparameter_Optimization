#!/usr/bin/env python3
"""Generate validated DAG/application/platform JSON variants.

This tool is intentionally self-contained and uses only the Python standard
library so it can live inside an experiments repository without changing the
rest of the project.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from collections import defaultdict, deque
from pathlib import Path
from typing import Any


DEFAULT_CONFIG: dict[str, Any] = {
    "generation_mode": "application_only",
    "num_variants": 5,
    "random_seed": 42,
    "output_dir": "generated_variants",
    "application": {
        "preserve_task_count": True,
        "preserve_task_ids": True,
        "preserve_message_count": False,
        "target_message_count": None,
        "layout_strategy": "layered_random",
        "num_layers": 7,
        "min_tasks_per_layer": 1,
        "in_degree": {"min": 0, "max": 3, "target_avg": 1.5},
        "out_degree": {"min": 0, "max": 3, "target_avg": 1.5},
        "force_at_least_one_source": True,
        "force_at_least_one_sink": True,
        "avoid_isolated_tasks": True,
        "message_size_range": [15, 25],
        "period_choices": [10, 30, 50],
        "timetriggered_default": True,
        "regenerate_wcet_fullspeed": True,
        "wcet_fullspeed_range": [3, 100],
        "regenerate_processing_times": True,
        "processing_times_range": [2, 10],
        "regenerate_mcet": False,
        "mcet_default": 0,
        "regenerate_deadline": True,
        "deadline_policy": "random_range",
        "deadline_range": [100, 600],
        "can_run_on_policy": "resample_processors_only",
        "min_can_run_on": 2,
        "max_can_run_on": 5,
    },
    "platform": {
        "modify_platform": False,
        "preserve_node_count": True,
        "preserve_processor_count": True,
        "preserve_router_count": True,
        "processor_count": None,
        "router_count": None,
        "topology_strategy": "layered",
        "ensure_connected": True,
        "extra_link_probability": 0.15,
        "bidirectional_links": False,
        "router_id_policy": "reuse_or_generate",
        "processor_id_policy": "reuse_or_generate",
    },
    "validation": {
        "require_same_task_count": True,
        "require_dag": True,
        "require_no_self_loops": True,
        "require_unique_message_edges": True,
        "require_different_edge_set_from_original": True,
        "min_edge_difference_ratio": 0.30,
        "require_platform_connected": True,
        "require_can_run_on_processors_only": True,
        "max_generation_attempts": 1000,
    },
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return copy.deepcopy(DEFAULT_CONFIG)
    return deep_merge(DEFAULT_CONFIG, load_json(path))


def require_input_shape(data: dict[str, Any]) -> None:
    try:
        jobs = data["application"]["jobs"]
        messages = data["application"]["messages"]
        nodes = data["platform"]["nodes"]
        links = data["platform"]["links"]
    except KeyError as exc:
        raise ValueError(f"Input JSON is missing required key: {exc}") from exc
    for name, value in {
        "application.jobs": jobs,
        "application.messages": messages,
        "platform.nodes": nodes,
        "platform.links": links,
    }.items():
        if not isinstance(value, list):
            raise ValueError(f"{name} must be a list")


def get_task_ids(data: dict) -> list[int]:
    """Return task IDs from data["application"]["jobs"]."""
    return [job["id"] for job in data["application"]["jobs"]]


def get_edges(data: dict) -> list[tuple[int, int]]:
    """Return sender-receiver edges from data["application"]["messages"]."""
    return [
        (message["sender"], message["receiver"])
        for message in data["application"]["messages"]
    ]


def get_platform_node_ids(platform: dict) -> list[int]:
    """Return all platform node IDs."""
    return [node["id"] for node in platform.get("nodes", [])]


def get_processor_ids(platform: dict) -> list[int]:
    """
    Return all platform node IDs where is_router is False.
    These are the only valid IDs for job can_run_on.
    """
    return [
        node["id"]
        for node in platform.get("nodes", [])
        if bool(node.get("is_router", False)) is False
    ]


def get_router_ids(platform: dict) -> list[int]:
    """
    Return all platform node IDs where is_router is True.
    These IDs must never appear in job can_run_on.
    """
    return [
        node["id"]
        for node in platform.get("nodes", [])
        if bool(node.get("is_router", False)) is True
    ]


def unique_preserving_order(values: list[int]) -> list[int]:
    seen: set[int] = set()
    result: list[int] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def validate_can_run_on_jobs(jobs: list[dict], platform: dict) -> list[str]:
    """
    Validate that every job's can_run_on list contains only valid processor IDs.
    Return a list of validation error strings.
    """
    errors: list[str] = []
    node_ids = set(get_platform_node_ids(platform))
    processor_ids = set(get_processor_ids(platform))
    router_ids = set(get_router_ids(platform))

    for job in jobs:
        job_id = job.get("id", "<missing>")
        can_run_on = job.get("can_run_on")
        if "can_run_on" not in job:
            errors.append(f"job {job_id}: can_run_on is missing")
            continue
        if not isinstance(can_run_on, list):
            errors.append(f"job {job_id}: can_run_on must be a list")
            continue
        if not can_run_on:
            errors.append(f"job {job_id}: can_run_on must not be empty")
            continue
        if len(can_run_on) != len(set(can_run_on)):
            errors.append(f"job {job_id}: can_run_on contains duplicate IDs")
        for node_id in can_run_on:
            if node_id not in node_ids:
                errors.append(f"job {job_id}: can_run_on ID {node_id} is not a platform node")
            elif node_id in router_ids:
                errors.append(f"job {job_id}: can_run_on ID {node_id} is a router")
            elif node_id not in processor_ids:
                errors.append(f"job {job_id}: can_run_on ID {node_id} is not a processor")
    return errors


def sample_processor_subset(
    processor_ids: list[int], app_config: dict[str, Any], rng: random.Random
) -> list[int]:
    if not processor_ids:
        raise ValueError("Platform has no processor nodes; cannot populate can_run_on")
    min_count = max(1, int(app_config.get("min_can_run_on", 1)))
    max_count = max(min_count, int(app_config.get("max_can_run_on", min_count)))
    max_count = min(max_count, len(processor_ids))
    min_count = min(min_count, max_count)
    count = rng.randint(min_count, max_count)
    return sorted(rng.sample(processor_ids, count))


def repair_can_run_on_jobs(
    jobs: list[dict],
    platform: dict,
    app_config: dict[str, Any],
    rng: random.Random,
    warnings: list[str],
) -> None:
    processor_ids = sorted(get_processor_ids(platform))
    router_ids = set(get_router_ids(platform))
    node_ids = set(get_platform_node_ids(platform))
    if not processor_ids:
        raise ValueError("Platform must contain at least one processor node")

    for job in jobs:
        original = job.get("can_run_on")
        valid_existing: list[int] = []
        if isinstance(original, list):
            valid_existing = [
                node_id
                for node_id in unique_preserving_order(original)
                if node_id in node_ids and node_id not in router_ids and node_id in processor_ids
            ]
            invalid = [node_id for node_id in original if node_id not in valid_existing]
            if invalid:
                warnings.append(
                    f"job {job.get('id')}: removed invalid can_run_on IDs {unique_preserving_order(invalid)}"
                )
        else:
            warnings.append(f"job {job.get('id')}: replaced missing/non-list can_run_on")

        if not valid_existing:
            valid_existing = sample_processor_subset(processor_ids, app_config, rng)
            warnings.append(f"job {job.get('id')}: resampled can_run_on")
        job["can_run_on"] = sorted(valid_existing)


def topological_sort(task_ids: list[int], edges: list[tuple[int, int]]) -> list[int]:
    """
    Return topological ordering.
    Raise ValueError if graph has a cycle.
    """
    task_set = set(task_ids)
    in_degree = {task_id: 0 for task_id in task_ids}
    adjacency: dict[int, list[int]] = {task_id: [] for task_id in task_ids}
    for sender, receiver in edges:
        if sender not in task_set or receiver not in task_set:
            continue
        adjacency[sender].append(receiver)
        in_degree[receiver] += 1

    ready = deque([task_id for task_id in task_ids if in_degree[task_id] == 0])
    order: list[int] = []
    while ready:
        task_id = ready.popleft()
        order.append(task_id)
        for receiver in adjacency[task_id]:
            in_degree[receiver] -= 1
            if in_degree[receiver] == 0:
                ready.append(receiver)
    if len(order) != len(task_ids):
        raise ValueError("Graph contains a cycle")
    return order


def is_dag(task_ids: list[int], edges: list[tuple[int, int]]) -> bool:
    """Return True if the graph is a DAG."""
    try:
        topological_sort(task_ids, edges)
        return True
    except ValueError:
        return False


def has_cycle(task_ids: list[int], edges: list[tuple[int, int]]) -> bool:
    """Return True if graph has a cycle."""
    return not is_dag(task_ids, edges)


def compare_edge_sets(
    original_edges: list[tuple[int, int]], new_edges: list[tuple[int, int]]
) -> dict:
    """
    Compare original and generated edge sets.

    Return:
    {
      "original_edge_count": int,
      "new_edge_count": int,
      "common_edge_count": int,
      "different_edge_count": int,
      "edge_difference_ratio": float
    }
    """
    original_set = set(original_edges)
    new_set = set(new_edges)
    common = original_set & new_set
    different = new_set - original_set
    denominator = max(len(original_set), len(new_set), 1)
    return {
        "original_edge_count": len(original_set),
        "new_edge_count": len(new_set),
        "common_edge_count": len(common),
        "different_edge_count": len(different),
        "edge_difference_ratio": (len(original_set ^ new_set) / denominator),
    }


def normalize_edge(edge: tuple[int, int], bidirectional: bool) -> tuple[int, int]:
    if bidirectional:
        return edge
    start, end = edge
    return (start, end) if start <= end else (end, start)


def add_platform_edge(
    edges: set[tuple[int, int]], start: int, end: int, bidirectional: bool
) -> None:
    if start == end:
        return
    if bidirectional:
        edges.add((start, end))
        edges.add((end, start))
    else:
        edges.add(normalize_edge((start, end), bidirectional=False))


def is_platform_connected(platform: dict) -> bool:
    """
    Treat platform links as undirected and check whether all nodes are connected.
    """
    node_ids = get_platform_node_ids(platform)
    if not node_ids:
        return False
    adjacency: dict[int, set[int]] = {node_id: set() for node_id in node_ids}
    node_set = set(node_ids)
    for link in platform.get("links", []):
        start = link.get("start")
        end = link.get("end")
        if start in node_set and end in node_set and start != end:
            adjacency[start].add(end)
            adjacency[end].add(start)
    seen = {node_ids[0]}
    queue = deque([node_ids[0]])
    while queue:
        current = queue.popleft()
        for neighbor in adjacency[current]:
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
    return len(seen) == len(node_ids)


def validate_platform(platform: dict, config: dict) -> list[str]:
    """Validate platform nodes and links."""
    errors: list[str] = []
    nodes = platform.get("nodes", [])
    links = platform.get("links", [])
    if not isinstance(nodes, list) or not nodes:
        return ["platform.nodes must be a non-empty list"]
    if not isinstance(links, list):
        errors.append("platform.links must be a list")
        links = []

    node_ids = get_platform_node_ids(platform)
    if len(node_ids) != len(set(node_ids)):
        errors.append("platform node IDs must be unique")
    node_set = set(node_ids)
    if not get_processor_ids(platform):
        errors.append("platform must contain at least one processor")

    seen_links: set[tuple[int, int]] = set()
    bidirectional = bool(config.get("platform", {}).get("bidirectional_links", False))
    for index, link in enumerate(links):
        start = link.get("start")
        end = link.get("end")
        if start not in node_set:
            errors.append(f"platform link {index}: start {start} is not a node")
        if end not in node_set:
            errors.append(f"platform link {index}: end {end} is not a node")
        if start == end:
            errors.append(f"platform link {index}: self-loop {start}->{end}")
        normalized = normalize_edge((start, end), bidirectional)
        if normalized in seen_links:
            errors.append(f"platform link {index}: duplicate link {start}->{end}")
        seen_links.add(normalized)

    validation_config = config.get("validation", {})
    if validation_config.get("require_platform_connected", True) and not is_platform_connected(platform):
        errors.append("platform is not connected")
    return errors


def task_layers(
    task_ids: list[int], app_config: dict[str, Any], strategy: str, rng: random.Random
) -> list[list[int]]:
    count = len(task_ids)
    if count == 0:
        return []
    shuffled = task_ids[:]
    rng.shuffle(shuffled)
    num_layers = int(app_config.get("num_layers", 3))
    min_per_layer = max(1, int(app_config.get("min_tasks_per_layer", 1)))
    num_layers = max(2 if count > 1 else 1, min(num_layers, count))
    while num_layers * min_per_layer > count and num_layers > 1:
        num_layers -= 1

    if strategy == "wide_front_narrow_end":
        weights = [num_layers - index for index in range(num_layers)]
    elif strategy == "narrow_front_wide_middle":
        center = (num_layers - 1) / 2
        weights = [int(num_layers - abs(index - center)) for index in range(num_layers)]
    elif strategy == "fork_join":
        weights = [1] + [max(2, num_layers) for _ in range(max(0, num_layers - 2))] + [1]
    else:
        weights = [1 for _ in range(num_layers)]

    layers: list[list[int]] = [[] for _ in range(num_layers)]
    remaining = shuffled[:]
    for layer in layers:
        for _ in range(min_per_layer):
            if remaining:
                layer.append(remaining.pop())

    weighted_indices = [
        index for index, weight in enumerate(weights[:num_layers]) for _ in range(max(1, weight))
    ]
    while remaining:
        layers[rng.choice(weighted_indices)].append(remaining.pop())

    return [layer for layer in layers if layer]


def candidate_layer_edges(layers: list[list[int]]) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for lower_index, lower_layer in enumerate(layers):
        for higher_layer in layers[lower_index + 1 :]:
            for sender in lower_layer:
                for receiver in higher_layer:
                    candidates.append((sender, receiver))
    return candidates


def target_edge_count(
    original_messages: list[dict], task_count: int, app_config: dict[str, Any]
) -> int:
    if bool(app_config.get("preserve_message_count", False)):
        return len(original_messages)
    explicit = app_config.get("target_message_count")
    if explicit is not None:
        return max(0, int(explicit))
    in_avg = float(app_config.get("in_degree", {}).get("target_avg", 1.5))
    out_avg = float(app_config.get("out_degree", {}).get("target_avg", in_avg))
    return max(0, int(round(task_count * ((in_avg + out_avg) / 2.0))))


def select_edges(
    candidates: list[tuple[int, int]],
    target_count: int,
    app_config: dict[str, Any],
    rng: random.Random,
) -> set[tuple[int, int]]:
    in_max = int(app_config.get("in_degree", {}).get("max", 999999))
    out_max = int(app_config.get("out_degree", {}).get("max", 999999))
    in_count: dict[int, int] = defaultdict(int)
    out_count: dict[int, int] = defaultdict(int)
    selected: set[tuple[int, int]] = set()
    shuffled = candidates[:]
    rng.shuffle(shuffled)
    for sender, receiver in shuffled:
        if len(selected) >= target_count:
            break
        if out_count[sender] >= out_max or in_count[receiver] >= in_max:
            continue
        selected.add((sender, receiver))
        out_count[sender] += 1
        in_count[receiver] += 1
    return selected


def repair_isolated_tasks(
    edges: set[tuple[int, int]],
    layers: list[list[int]],
    app_config: dict[str, Any],
    rng: random.Random,
) -> None:
    if not app_config.get("avoid_isolated_tasks", True) or len(layers) <= 1:
        return
    in_max = int(app_config.get("in_degree", {}).get("max", 999999))
    out_max = int(app_config.get("out_degree", {}).get("max", 999999))

    def degrees() -> tuple[dict[int, int], dict[int, int]]:
        incoming: dict[int, int] = defaultdict(int)
        outgoing: dict[int, int] = defaultdict(int)
        for sender, receiver in edges:
            outgoing[sender] += 1
            incoming[receiver] += 1
        return incoming, outgoing

    layer_index_by_task = {
        task_id: layer_index
        for layer_index, layer in enumerate(layers)
        for task_id in layer
    }
    all_tasks = list(layer_index_by_task)
    rng.shuffle(all_tasks)
    for task_id in all_tasks:
        incoming, outgoing = degrees()
        if incoming[task_id] or outgoing[task_id]:
            continue
        layer_index = layer_index_by_task[task_id]
        possible_in = [
            sender
            for earlier in layers[:layer_index]
            for sender in earlier
            if outgoing[sender] < out_max and incoming[task_id] < in_max
        ]
        possible_out = [
            receiver
            for later in layers[layer_index + 1 :]
            for receiver in later
            if outgoing[task_id] < out_max and incoming[receiver] < in_max
        ]
        if possible_in and (not possible_out or rng.random() < 0.5):
            edges.add((rng.choice(possible_in), task_id))
        elif possible_out:
            edges.add((task_id, rng.choice(possible_out)))


def generate_chain_plus_branches(
    task_ids: list[int],
    original_messages: list[dict],
    app_config: dict[str, Any],
    rng: random.Random,
) -> set[tuple[int, int]]:
    order = task_ids[:]
    rng.shuffle(order)
    target_count = target_edge_count(original_messages, len(task_ids), app_config)
    edges = {(order[index], order[index + 1]) for index in range(len(order) - 1)}
    candidates = [
        (order[i], order[j])
        for i in range(len(order))
        for j in range(i + 2, len(order))
    ]
    extra = select_edges(candidates, max(target_count - len(edges), 0), app_config, rng)
    return edges | extra


def generate_random_dag_edges(
    task_ids: list[int],
    original_messages: list[dict],
    app_config: dict[str, Any],
    rng: random.Random,
) -> set[tuple[int, int]]:
    order = task_ids[:]
    rng.shuffle(order)
    candidates = [
        (order[i], order[j])
        for i in range(len(order))
        for j in range(i + 1, len(order))
    ]
    target_count = target_edge_count(original_messages, len(task_ids), app_config)
    return select_edges(candidates, target_count, app_config, rng)


def generate_layered_edges(
    task_ids: list[int],
    original_messages: list[dict],
    app_config: dict[str, Any],
    strategy: str,
    rng: random.Random,
) -> set[tuple[int, int]]:
    layers = task_layers(task_ids, app_config, strategy, rng)
    candidates = candidate_layer_edges(layers)
    target_count = target_edge_count(original_messages, len(task_ids), app_config)
    edges = select_edges(candidates, target_count, app_config, rng)
    repair_isolated_tasks(edges, layers, app_config, rng)
    return edges


def generate_edges_for_strategy(
    task_ids: list[int],
    original_messages: list[dict],
    app_config: dict[str, Any],
    rng: random.Random,
) -> set[tuple[int, int]]:
    strategy = app_config.get("layout_strategy", "layered_random")
    if strategy == "chain_plus_branches":
        return generate_chain_plus_branches(task_ids, original_messages, app_config, rng)
    if strategy == "random_dag":
        return generate_random_dag_edges(task_ids, original_messages, app_config, rng)
    if strategy in {
        "layered_random",
        "wide_front_narrow_end",
        "narrow_front_wide_middle",
        "fork_join",
    }:
        return generate_layered_edges(task_ids, original_messages, app_config, strategy, rng)
    raise ValueError(f"Unsupported application layout_strategy: {strategy}")


def int_in_range(bounds: list[int], rng: random.Random) -> int:
    low, high = int(bounds[0]), int(bounds[1])
    if low > high:
        low, high = high, low
    return rng.randint(low, high)


def build_messages(
    edges: set[tuple[int, int]],
    app_config: dict[str, Any],
    rng: random.Random,
    original_messages: list[dict] | None = None,
) -> list[dict[str, Any]]:
    if app_config.get("preserve_message_payloads", False) and original_messages:
        payloads = sorted(original_messages, key=lambda message: message.get("id", 0))
        messages = []
        for index, (sender, receiver) in enumerate(sorted(edges)):
            template = copy.deepcopy(payloads[index % len(payloads)])
            template["id"] = index
            template["sender"] = sender
            template["receiver"] = receiver
            messages.append(template)
        return messages

    size_range = app_config.get("message_size_range", [15, 25])
    period_choices = app_config.get("period_choices", [10])
    if not period_choices:
        period_choices = [10]
    timetriggered = bool(app_config.get("timetriggered_default", True))
    return [
        {
            "id": index,
            "sender": sender,
            "receiver": receiver,
            "size": int_in_range(size_range, rng),
            "timetriggered": timetriggered,
            "period": rng.choice(period_choices),
        }
        for index, (sender, receiver) in enumerate(sorted(edges))
    ]


def regenerate_jobs(
    original_jobs: list[dict],
    platform: dict,
    app_config: dict[str, Any],
    rng: random.Random,
    warnings: list[str],
) -> list[dict]:
    processor_ids = sorted(get_processor_ids(platform))
    if not processor_ids:
        raise ValueError("Cannot regenerate jobs because platform has no processors")
    jobs: list[dict] = []
    for index, original_job in enumerate(original_jobs):
        job = copy.deepcopy(original_job)
        if not app_config.get("preserve_task_ids", True):
            job["id"] = index
        if app_config.get("regenerate_wcet_fullspeed", True):
            job["wcet_fullspeed"] = int_in_range(app_config.get("wcet_fullspeed_range", [1, 1]), rng)
        if app_config.get("regenerate_processing_times", True):
            job["processing_times"] = int_in_range(app_config.get("processing_times_range", [1, 1]), rng)
        if app_config.get("regenerate_mcet", False):
            job["mcet"] = app_config.get("mcet_default", 0)
        if app_config.get("regenerate_deadline", True):
            policy = app_config.get("deadline_policy", "random_range")
            if policy == "random_range":
                job["deadline"] = int_in_range(app_config.get("deadline_range", [1, 1]), rng)
        if app_config.get("can_run_on_policy", "resample_processors_only") == "resample_processors_only":
            job["can_run_on"] = sample_processor_subset(processor_ids, app_config, rng)
        jobs.append(job)
    repair_can_run_on_jobs(jobs, platform, app_config, rng, warnings)
    return jobs


def generate_application(
    original_data: dict,
    base_data: dict,
    config: dict,
    rng: random.Random,
    warnings: list[str],
) -> None:
    app_config = config.get("application", {})
    original_jobs = original_data["application"]["jobs"]
    original_messages = original_data["application"]["messages"]
    jobs = regenerate_jobs(original_jobs, base_data["platform"], app_config, rng, warnings)
    task_ids = [job["id"] for job in jobs]
    edges = generate_edges_for_strategy(task_ids, original_messages, app_config, rng)
    if len(edges) < target_edge_count(original_messages, len(task_ids), app_config):
        warnings.append("Generated fewer edges than requested because degree/layer constraints limited candidates")
    base_data["application"]["jobs"] = jobs
    base_data["application"]["messages"] = build_messages(
        edges,
        app_config,
        rng,
        original_messages=original_messages,
    )


def platform_counts(original_platform: dict, platform_config: dict[str, Any]) -> tuple[int, int]:
    original_processors = get_processor_ids(original_platform)
    original_routers = get_router_ids(original_platform)
    if platform_config.get("preserve_processor_count", True):
        processor_count = len(original_processors)
    else:
        processor_count = platform_config.get("processor_count") or len(original_processors)
    if platform_config.get("preserve_router_count", True):
        router_count = len(original_routers)
    else:
        router_count = platform_config.get("router_count") or len(original_routers)
    processor_count = max(1, int(processor_count))
    router_count = max(0, int(router_count))
    if platform_config.get("preserve_node_count", True):
        total = len(original_platform.get("nodes", []))
        if total > 0:
            router_count = max(0, total - processor_count)
    return processor_count, router_count


def generate_node_ids(
    original_platform: dict, processor_count: int, router_count: int
) -> tuple[list[int], list[int]]:
    original_processors = sorted(get_processor_ids(original_platform))
    original_routers = sorted(get_router_ids(original_platform))
    used: set[int] = set()

    processor_ids = original_processors[:processor_count]
    used.update(processor_ids)
    next_id = 0
    while len(processor_ids) < processor_count:
        while next_id in used:
            next_id += 1
        processor_ids.append(next_id)
        used.add(next_id)

    router_ids = [node_id for node_id in original_routers if node_id not in used][:router_count]
    used.update(router_ids)
    while len(router_ids) < router_count:
        while next_id in used:
            next_id += 1
        router_ids.append(next_id)
        used.add(next_id)
    return sorted(processor_ids), sorted(router_ids)


def maybe_add_extra_edges(
    edges: set[tuple[int, int]],
    node_ids: list[int],
    probability: float,
    bidirectional: bool,
    rng: random.Random,
) -> None:
    for i, start in enumerate(node_ids):
        for end in node_ids[i + 1 :]:
            if rng.random() < probability:
                add_platform_edge(edges, start, end, bidirectional)


def generate_platform(original_platform: dict, config: dict, rng: random.Random) -> dict:
    """Generate a new platform according to config."""
    platform_config = config.get("platform", {})
    processor_count, router_count = platform_counts(original_platform, platform_config)
    processor_ids, router_ids = generate_node_ids(original_platform, processor_count, router_count)
    nodes = [{"id": node_id, "is_router": False} for node_id in processor_ids] + [
        {"id": node_id, "is_router": True} for node_id in router_ids
    ]
    nodes.sort(key=lambda node: node["id"])

    strategy = platform_config.get("topology_strategy", "layered")
    probability = float(platform_config.get("extra_link_probability", 0.15))
    bidirectional = bool(platform_config.get("bidirectional_links", False))
    edges: set[tuple[int, int]] = set()
    all_ids = sorted(processor_ids + router_ids)

    if not router_ids:
        shuffled = processor_ids[:]
        rng.shuffle(shuffled)
        for start, end in zip(shuffled, shuffled[1:]):
            add_platform_edge(edges, start, end, bidirectional)
    elif strategy == "star_router":
        centers = router_ids[: max(1, min(len(router_ids), int(math.sqrt(len(router_ids))) or 1))]
        for router_id in router_ids:
            if router_id not in centers:
                add_platform_edge(edges, rng.choice(centers), router_id, bidirectional)
        for processor_id in processor_ids:
            add_platform_edge(edges, processor_id, rng.choice(centers), bidirectional)
    elif strategy == "mesh_router":
        for i, start in enumerate(router_ids):
            for end in router_ids[i + 1 :]:
                add_platform_edge(edges, start, end, bidirectional)
        for processor_id in processor_ids:
            add_platform_edge(edges, processor_id, rng.choice(router_ids), bidirectional)
    elif strategy == "tree":
        ordered_routers = router_ids[:]
        rng.shuffle(ordered_routers)
        for index, router_id in enumerate(ordered_routers[1:], start=1):
            parent = ordered_routers[(index - 1) // 2]
            add_platform_edge(edges, parent, router_id, bidirectional)
        for index, processor_id in enumerate(processor_ids):
            add_platform_edge(edges, processor_id, ordered_routers[index % len(ordered_routers)], bidirectional)
    elif strategy == "random_connected":
        shuffled = all_ids[:]
        rng.shuffle(shuffled)
        for index in range(1, len(shuffled)):
            add_platform_edge(edges, shuffled[index], rng.choice(shuffled[:index]), bidirectional)
    elif strategy == "layered":
        ordered_routers = router_ids[:]
        rng.shuffle(ordered_routers)
        for start, end in zip(ordered_routers, ordered_routers[1:]):
            add_platform_edge(edges, start, end, bidirectional)
        for index, processor_id in enumerate(processor_ids):
            add_platform_edge(edges, processor_id, ordered_routers[index % len(ordered_routers)], bidirectional)
    else:
        raise ValueError(f"Unsupported platform topology_strategy: {strategy}")

    maybe_add_extra_edges(edges, all_ids, probability, bidirectional, rng)
    links = [{"start": start, "end": end} for start, end in sorted(edges)]
    return {"nodes": nodes, "links": links}


def validate_application(original_data: dict, new_data: dict, config: dict) -> list[str]:
    """
    Validate generated application.
    Return list of errors.
    Empty list means valid.
    """
    errors: list[str] = []
    validation_config = config.get("validation", {})
    original_jobs = original_data["application"]["jobs"]
    jobs = new_data["application"]["jobs"]
    messages = new_data["application"]["messages"]
    task_ids = get_task_ids(new_data)
    task_set = set(task_ids)

    if validation_config.get("require_same_task_count", True) and len(jobs) != len(original_jobs):
        errors.append(f"task count changed from {len(original_jobs)} to {len(jobs)}")
    if len(task_ids) != len(task_set):
        errors.append("task IDs must be unique")

    seen_edges: set[tuple[int, int]] = set()
    for index, message in enumerate(messages):
        sender = message.get("sender")
        receiver = message.get("receiver")
        if sender not in task_set:
            errors.append(f"message {index}: sender {sender} is not a task ID")
        if receiver not in task_set:
            errors.append(f"message {index}: receiver {receiver} is not a task ID")
        if validation_config.get("require_no_self_loops", True) and sender == receiver:
            errors.append(f"message {index}: self-loop {sender}->{receiver}")
        edge = (sender, receiver)
        if validation_config.get("require_unique_message_edges", True) and edge in seen_edges:
            errors.append(f"message {index}: duplicate edge {sender}->{receiver}")
        seen_edges.add(edge)

    edges = get_edges(new_data)
    if validation_config.get("require_dag", True) and not is_dag(task_ids, edges):
        errors.append("application graph is not a DAG")

    if validation_config.get("require_different_edge_set_from_original", True):
        comparison = compare_edge_sets(get_edges(original_data), edges)
        minimum = float(validation_config.get("min_edge_difference_ratio", 0.0))
        if comparison["edge_difference_ratio"] < minimum:
            errors.append(
                "generated edge set is too similar to original "
                f"({comparison['edge_difference_ratio']:.3f} < {minimum:.3f})"
            )

    if validation_config.get("require_can_run_on_processors_only", True):
        errors.extend(validate_can_run_on_jobs(jobs, new_data["platform"]))
    return errors


def generate_variant(
    original_data: dict,
    config: dict,
    variant_seed: int,
) -> tuple[dict, list[str]]:
    rng = random.Random(variant_seed)
    mode = config.get("generation_mode", "application_only")
    warnings: list[str] = []
    new_data = copy.deepcopy(original_data)

    if mode in {"platform_only", "both"}:
        new_data["platform"] = generate_platform(original_data["platform"], config, rng)
        repair_can_run_on_jobs(
            new_data["application"]["jobs"],
            new_data["platform"],
            config.get("application", {}),
            rng,
            warnings,
        )
    elif mode != "application_only":
        raise ValueError(f"Unsupported generation_mode: {mode}")

    if mode in {"application_only", "both"}:
        generate_application(original_data, new_data, config, rng, warnings)
    else:
        repair_can_run_on_jobs(
            new_data["application"]["jobs"],
            new_data["platform"],
            config.get("application", {}),
            rng,
            warnings,
        )
    return new_data, warnings


def build_variant_summary(
    original_data: dict,
    new_data: dict,
    config: dict,
    variant_index: int,
    output_file: str,
    variant_seed: int,
    warnings: list[str],
) -> dict[str, Any]:
    edge_comparison = compare_edge_sets(get_edges(original_data), get_edges(new_data))
    app_errors = validate_application(original_data, new_data, config)
    platform_errors = validate_platform(new_data["platform"], config)
    can_run_errors = validate_can_run_on_jobs(new_data["application"]["jobs"], new_data["platform"])
    all_warnings = warnings[:]
    for label, errors in {
        "application": app_errors,
        "platform": platform_errors,
        "can_run_on": can_run_errors,
    }.items():
        all_warnings.extend(f"{label} validation: {error}" for error in errors)

    return {
        "variant_index": variant_index,
        "output_file": output_file,
        "generation_mode": config.get("generation_mode"),
        "random_seed": variant_seed,
        "original_job_count": len(original_data["application"]["jobs"]),
        "generated_job_count": len(new_data["application"]["jobs"]),
        "original_message_count": len(original_data["application"]["messages"]),
        "generated_message_count": len(new_data["application"]["messages"]),
        "original_edge_count": edge_comparison["original_edge_count"],
        "generated_edge_count": edge_comparison["new_edge_count"],
        "common_edge_count": edge_comparison["common_edge_count"],
        "edge_difference_ratio": edge_comparison["edge_difference_ratio"],
        "dag_validation_passed": is_dag(get_task_ids(new_data), get_edges(new_data)),
        "platform_validation_passed": not platform_errors,
        "platform_connected": is_platform_connected(new_data["platform"]),
        "can_run_on_validation_passed": not can_run_errors,
        "processor_ids": sorted(get_processor_ids(new_data["platform"])),
        "router_ids": sorted(get_router_ids(new_data["platform"])),
        "warnings": all_warnings,
    }


def validate_full_variant(original_data: dict, new_data: dict, config: dict) -> list[str]:
    errors = validate_application(original_data, new_data, config)
    errors.extend(validate_platform(new_data["platform"], config))
    errors.extend(validate_can_run_on_jobs(new_data["application"]["jobs"], new_data["platform"]))
    return errors


def generate_validated_variant(
    original_data: dict,
    config: dict,
    variant_index: int,
    base_seed: int,
    verbose: bool,
) -> tuple[dict, list[str], int]:
    max_attempts = int(config.get("validation", {}).get("max_generation_attempts", 1000))
    last_errors: list[str] = []
    for attempt in range(max_attempts):
        variant_seed = base_seed + variant_index * max_attempts + attempt
        candidate, warnings = generate_variant(original_data, config, variant_seed)
        errors = validate_full_variant(original_data, candidate, config)
        if not errors:
            if attempt > 0:
                warnings.append(f"variant accepted after {attempt + 1} attempts")
            return candidate, warnings, variant_seed
        last_errors = errors
        if verbose and attempt < 5:
            print(f"attempt {attempt + 1} failed: {'; '.join(errors)}")
    raise RuntimeError(
        f"Could not generate valid variant {variant_index} after {max_attempts} attempts. "
        f"Last validation errors: {'; '.join(last_errors)}"
    )


def print_dry_run_summary(summary: dict[str, Any]) -> None:
    print("Dry-run summary")
    print(f"  original task count: {summary['original_job_count']}")
    print(f"  generated task count: {summary['generated_job_count']}")
    print(f"  original message count: {summary['original_message_count']}")
    print(f"  generated message count: {summary['generated_message_count']}")
    print(f"  DAG validation passed: {summary['dag_validation_passed']}")
    print(f"  edge difference ratio: {summary['edge_difference_ratio']:.3f}")
    print(f"  processor IDs: {summary['processor_ids']}")
    print(f"  router IDs: {summary['router_ids']}")
    print(f"  can_run_on validation passed: {summary['can_run_on_validation_passed']}")
    print(f"  warnings: {summary['warnings']}")


def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    config = copy.deepcopy(config)
    if args.num_variants is not None:
        config["num_variants"] = args.num_variants
    if args.seed is not None:
        config["random_seed"] = args.seed
    if args.mode is not None:
        config["generation_mode"] = args.mode
    if args.output is not None:
        config["output_dir"] = str(args.output)
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate validated DAG benchmark JSON variants.")
    parser.add_argument("--input", required=True, type=Path, help="Input benchmark JSON file")
    parser.add_argument("--config", type=Path, help="Configuration JSON file")
    parser.add_argument("--output", type=Path, help="Output directory")
    parser.add_argument("--num-variants", type=int, help="Override config num_variants")
    parser.add_argument("--seed", type=int, help="Override config random_seed")
    parser.add_argument(
        "--mode",
        choices=["application_only", "platform_only", "both"],
        help="Override config generation_mode",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate one variant without writing files")
    parser.add_argument("--verbose", action="store_true", help="Print detailed logs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    original_data = load_json(args.input)
    require_input_shape(original_data)
    config = apply_cli_overrides(load_config(args.config), args)
    base_seed = int(config.get("random_seed", 42))

    if args.verbose:
        print(f"input: {args.input}")
        print(f"mode: {config.get('generation_mode')}")
        print(f"base seed: {base_seed}")

    if args.dry_run:
        variant, warnings, variant_seed = generate_validated_variant(
            original_data, config, 0, base_seed, args.verbose
        )
        summary = build_variant_summary(
            original_data, variant, config, 0, "<dry-run>", variant_seed, warnings
        )
        print_dry_run_summary(summary)
        return

    output_dir = Path(config.get("output_dir", "generated_variants"))
    output_dir.mkdir(parents=True, exist_ok=True)
    num_variants = int(config.get("num_variants", 1))

    # Use the actual number of tasks in the input file as a filename prefix.
    # Example: 70 tasks -> 70T_variant_000.json
    task_count = len(original_data["application"]["jobs"])
    file_prefix = f"{task_count}T"

    summaries: list[dict[str, Any]] = []

    for variant_index in range(num_variants):
        variant, warnings, variant_seed = generate_validated_variant(
            original_data, config, variant_index, base_seed, args.verbose
        )
        output_name = f"{file_prefix}_variant_{variant_index:03d}.json"

        output_path = output_dir / output_name
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(variant, handle, indent=2)
            handle.write("\n")
        summary = build_variant_summary(
            original_data, variant, config, variant_index, output_name, variant_seed, warnings
        )
        summaries.append(summary)
        if args.verbose:
            print(
                f"wrote {output_path} "
                f"(messages={summary['generated_message_count']}, "
                f"edge_difference_ratio={summary['edge_difference_ratio']:.3f})"
            )

    top_level_summary = {
    "input_file": str(args.input),
    "task_count": task_count,
    "file_prefix": file_prefix,
    "num_variants_requested": num_variants,
    "num_variants_generated": len(summaries),
    "config_file": str(args.config) if args.config else None,
    "variants": summaries,
    }

    summary_name = f"{file_prefix}_generation_summary.json"
    summary_path = output_dir / summary_name
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(top_level_summary, handle, indent=2)
        handle.write("\n")
    print(f"Generated {len(summaries)} variant(s) in {output_dir}")
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
