#!/usr/bin/env python3
"""
Validate a DAG benchmark JSON file.

This script checks:

1. application.jobs
2. application.messages
3. whether application.messages forms a valid DAG
4. platform.nodes
5. platform.links
6. whether the platform is connected
7. whether every job can_run_on list contains only processor node IDs

Processor rule:
    A processor is a platform node where is_router == false.
    A router is a platform node where is_router == true.
    Router IDs must never appear in can_run_on.

Usage:
  python3 dag_variant_generator/check_application_dag.py \
    --input generated_variants/variant_000.json

Optional:
  python3 dag_variant_generator/check_application_dag.py \
    --input generated_variants/variant_000.json \
    --show-topological-order \
    --verbose
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any


# ============================================================
# JSON LOADING
# ============================================================

def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON file and ensure the top-level object is a dictionary."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Input JSON must contain a JSON object.")

    return data


# ============================================================
# INPUT EXTRACTION
# ============================================================

def get_jobs(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Return application.jobs."""
    try:
        jobs = data["application"]["jobs"]
    except KeyError as exc:
        raise ValueError("Missing required field: application.jobs") from exc

    if not isinstance(jobs, list):
        raise ValueError("application.jobs must be a list.")

    return jobs


def get_messages(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Return application.messages."""
    try:
        messages = data["application"]["messages"]
    except KeyError as exc:
        raise ValueError("Missing required field: application.messages") from exc

    if not isinstance(messages, list):
        raise ValueError("application.messages must be a list.")

    return messages


def get_platform(data: dict[str, Any]) -> dict[str, Any]:
    """Return platform object."""
    try:
        platform = data["platform"]
    except KeyError as exc:
        raise ValueError("Missing required field: platform") from exc

    if not isinstance(platform, dict):
        raise ValueError("platform must be a JSON object.")

    return platform


def extract_task_ids(jobs: list[dict[str, Any]]) -> list[int]:
    """Extract task IDs from application.jobs."""
    task_ids: list[int] = []

    for index, job in enumerate(jobs):
        if not isinstance(job, dict):
            raise ValueError(f"Job at index {index} must be a JSON object.")

        if "id" not in job:
            raise ValueError(f"Job at index {index} is missing field: id")

        task_ids.append(job["id"])

    return task_ids


def extract_edges(messages: list[dict[str, Any]]) -> list[tuple[int, int]]:
    """Extract sender -> receiver edges from application.messages."""
    edges: list[tuple[int, int]] = []

    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"Message at index {index} must be a JSON object.")

        if "sender" not in message:
            raise ValueError(f"Message at index {index} is missing field: sender")

        if "receiver" not in message:
            raise ValueError(f"Message at index {index} is missing field: receiver")

        sender = message["sender"]
        receiver = message["receiver"]
        edges.append((sender, receiver))

    return edges


# ============================================================
# APPLICATION DAG VALIDATION
# ============================================================

def validate_application_edges(
    task_ids: list[int],
    edges: list[tuple[int, int]],
) -> list[str]:
    """
    Validate message edges.

    Checks:
    - duplicate task IDs
    - sender exists
    - receiver exists
    - no self-loops
    - no duplicate message edges
    """
    errors: list[str] = []
    task_set = set(task_ids)

    if len(task_ids) != len(task_set):
        errors.append("Duplicate task IDs found in application.jobs.")

    seen_edges: set[tuple[int, int]] = set()

    for edge_index, (sender, receiver) in enumerate(edges):
        if sender not in task_set:
            errors.append(
                f"Message edge {edge_index}: sender task {sender} does not exist."
            )

        if receiver not in task_set:
            errors.append(
                f"Message edge {edge_index}: receiver task {receiver} does not exist."
            )

        if sender == receiver:
            errors.append(
                f"Message edge {edge_index}: self-loop found: {sender} -> {receiver}"
            )

        if (sender, receiver) in seen_edges:
            errors.append(
                f"Message edge {edge_index}: duplicate edge found: {sender} -> {receiver}"
            )

        seen_edges.add((sender, receiver))

    return errors


def topological_sort(
    task_ids: list[int],
    edges: list[tuple[int, int]],
) -> tuple[bool, list[int]]:
    """
    Check DAG property using Kahn's algorithm.

    Returns:
        (True, topological_order) if graph is a DAG.
        (False, partial_order) if graph has a cycle.
    """
    task_set = set(task_ids)

    adjacency: dict[int, list[int]] = {task_id: [] for task_id in task_ids}
    in_degree: dict[int, int] = {task_id: 0 for task_id in task_ids}

    for sender, receiver in edges:
        # Invalid sender/receiver IDs are reported separately.
        if sender not in task_set or receiver not in task_set:
            continue

        adjacency[sender].append(receiver)
        in_degree[receiver] += 1

    ready = deque([task_id for task_id in task_ids if in_degree[task_id] == 0])
    topo_order: list[int] = []

    while ready:
        current = ready.popleft()
        topo_order.append(current)

        for neighbor in adjacency[current]:
            in_degree[neighbor] -= 1

            if in_degree[neighbor] == 0:
                ready.append(neighbor)

    is_valid_dag = len(topo_order) == len(task_ids)
    return is_valid_dag, topo_order


def find_cycle_path(
    task_ids: list[int],
    edges: list[tuple[int, int]],
) -> list[int]:
    """
    Try to find one cycle path using DFS.

    Returns:
        List of task IDs forming a cycle if found.
        Empty list if no cycle is found.
    """
    task_set = set(task_ids)

    adjacency: dict[int, list[int]] = {task_id: [] for task_id in task_ids}

    for sender, receiver in edges:
        if sender in task_set and receiver in task_set:
            adjacency[sender].append(receiver)

    # 0 = unvisited
    # 1 = visiting
    # 2 = done
    state: dict[int, int] = {task_id: 0 for task_id in task_ids}
    parent: dict[int, int | None] = {task_id: None for task_id in task_ids}

    def dfs(node: int) -> list[int]:
        state[node] = 1

        for neighbor in adjacency[node]:
            if state[neighbor] == 0:
                parent[neighbor] = node
                cycle = dfs(neighbor)

                if cycle:
                    return cycle

            elif state[neighbor] == 1:
                # Found back edge node -> neighbor.
                cycle = [neighbor]
                current = node

                while current != neighbor and current is not None:
                    cycle.append(current)
                    current = parent[current]

                cycle.append(neighbor)
                cycle.reverse()
                return cycle

        state[node] = 2
        return []

    for task_id in task_ids:
        if state[task_id] == 0:
            cycle = dfs(task_id)

            if cycle:
                return cycle

    return []


def compute_sources_and_sinks(
    task_ids: list[int],
    edges: list[tuple[int, int]],
) -> tuple[list[int], list[int]]:
    """Compute source and sink tasks."""
    task_set = set(task_ids)

    incoming_count: dict[int, int] = {task_id: 0 for task_id in task_ids}
    outgoing_count: dict[int, int] = {task_id: 0 for task_id in task_ids}

    for sender, receiver in edges:
        if sender in task_set and receiver in task_set:
            outgoing_count[sender] += 1
            incoming_count[receiver] += 1

    sources = [task_id for task_id in task_ids if incoming_count[task_id] == 0]
    sinks = [task_id for task_id in task_ids if outgoing_count[task_id] == 0]

    return sources, sinks


# ============================================================
# PLATFORM VALIDATION
# ============================================================

def get_platform_nodes(platform: dict[str, Any]) -> list[dict[str, Any]]:
    """Return platform.nodes."""
    nodes = platform.get("nodes")

    if not isinstance(nodes, list):
        raise ValueError("platform.nodes must be a list.")

    return nodes


def get_platform_links(platform: dict[str, Any]) -> list[dict[str, Any]]:
    """Return platform.links."""
    links = platform.get("links")

    if not isinstance(links, list):
        raise ValueError("platform.links must be a list.")

    return links


def get_platform_node_ids(platform: dict[str, Any]) -> list[int]:
    """Return all platform node IDs."""
    nodes = get_platform_nodes(platform)
    return [node["id"] for node in nodes if isinstance(node, dict) and "id" in node]


def get_processor_ids(platform: dict[str, Any]) -> list[int]:
    """
    Return processor node IDs.

    A processor is a platform node where:
        is_router == false
    """
    processor_ids: list[int] = []

    for node in get_platform_nodes(platform):
        if isinstance(node, dict) and node.get("is_router") is False:
            processor_ids.append(node["id"])

    return processor_ids


def get_router_ids(platform: dict[str, Any]) -> list[int]:
    """
    Return router node IDs.

    A router is a platform node where:
        is_router == true
    """
    router_ids: list[int] = []

    for node in get_platform_nodes(platform):
        if isinstance(node, dict) and node.get("is_router") is True:
            router_ids.append(node["id"])

    return router_ids


def normalize_undirected_link(start: int, end: int) -> tuple[int, int]:
    """Normalize a platform link as an undirected edge."""
    return (start, end) if start <= end else (end, start)


def validate_platform(
    platform: dict[str, Any],
    allow_reverse_links: bool = False,
    require_connected: bool = True,
) -> list[str]:
    """
    Validate platform nodes and links.

    Checks:
    - platform.nodes exists and is non-empty
    - platform.links exists
    - node IDs are unique
    - every node has id and is_router
    - is_router is boolean
    - at least one processor exists
    - every link start/end exists
    - no self-loop links
    - no duplicate links
    - platform is connected
    """
    errors: list[str] = []

    try:
        nodes = get_platform_nodes(platform)
        links = get_platform_links(platform)
    except ValueError as exc:
        return [str(exc)]

    if not nodes:
        errors.append("platform.nodes must not be empty.")

    node_ids: list[int] = []

    for index, node in enumerate(nodes):
        if not isinstance(node, dict):
            errors.append(f"platform.nodes[{index}] must be a JSON object.")
            continue

        if "id" not in node:
            errors.append(f"platform.nodes[{index}] is missing field: id.")
            continue

        if "is_router" not in node:
            errors.append(f"platform.nodes[{index}] is missing field: is_router.")

        elif not isinstance(node["is_router"], bool):
            errors.append(
                f"platform.nodes[{index}].is_router must be true or false."
            )

        node_ids.append(node["id"])

    if len(node_ids) != len(set(node_ids)):
        errors.append("Duplicate platform node IDs found.")

    node_set = set(node_ids)
    processor_ids = get_processor_ids(platform)
    router_ids = get_router_ids(platform)

    if not processor_ids:
        errors.append("Platform must contain at least one processor node.")

    seen_links: set[tuple[int, int]] = set()

    for index, link in enumerate(links):
        if not isinstance(link, dict):
            errors.append(f"platform.links[{index}] must be a JSON object.")
            continue

        if "start" not in link:
            errors.append(f"platform.links[{index}] is missing field: start.")
            continue

        if "end" not in link:
            errors.append(f"platform.links[{index}] is missing field: end.")
            continue

        start = link["start"]
        end = link["end"]

        if start not in node_set:
            errors.append(
                f"platform.links[{index}]: start node {start} does not exist."
            )

        if end not in node_set:
            errors.append(
                f"platform.links[{index}]: end node {end} does not exist."
            )

        if start == end:
            errors.append(
                f"platform.links[{index}]: self-loop link found: {start} -> {end}"
            )

        if allow_reverse_links:
            normalized = (start, end)
        else:
            normalized = normalize_undirected_link(start, end)

        if normalized in seen_links:
            errors.append(
                f"platform.links[{index}]: duplicate link found: {start} -> {end}"
            )

        seen_links.add(normalized)

    if require_connected and not is_platform_connected(platform):
        errors.append("Platform is not connected.")

    # This is not an error, but it is useful context.
    # We keep it out of errors because router-less platforms may be valid.
    _ = router_ids

    return errors


def is_platform_connected(platform: dict[str, Any]) -> bool:
    """
    Check whether the platform is connected.

    Platform links are treated as undirected for connectivity.
    """
    try:
        node_ids = get_platform_node_ids(platform)
        links = get_platform_links(platform)
    except ValueError:
        return False

    if not node_ids:
        return False

    node_set = set(node_ids)
    adjacency: dict[int, set[int]] = {node_id: set() for node_id in node_ids}

    for link in links:
        if not isinstance(link, dict):
            continue

        start = link.get("start")
        end = link.get("end")

        if start in node_set and end in node_set and start != end:
            adjacency[start].add(end)
            adjacency[end].add(start)

    start_node = node_ids[0]
    visited = {start_node}
    queue = deque([start_node])

    while queue:
        current = queue.popleft()

        for neighbor in adjacency[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return len(visited) == len(node_ids)


# ============================================================
# CAN_RUN_ON VALIDATION
# ============================================================

def validate_can_run_on(
    jobs: list[dict[str, Any]],
    platform: dict[str, Any],
) -> list[str]:
    """
    Validate job can_run_on lists.

    Rules:
    - can_run_on must exist
    - can_run_on must be a non-empty list
    - every ID in can_run_on must exist in platform.nodes
    - every ID in can_run_on must be a processor
    - router IDs must never appear
    - duplicate IDs are not allowed
    """
    errors: list[str] = []

    node_ids = set(get_platform_node_ids(platform))
    processor_ids = set(get_processor_ids(platform))
    router_ids = set(get_router_ids(platform))

    for job_index, job in enumerate(jobs):
        if not isinstance(job, dict):
            errors.append(f"application.jobs[{job_index}] must be a JSON object.")
            continue

        job_id = job.get("id", f"<index {job_index}>")

        if "can_run_on" not in job:
            errors.append(f"job {job_id}: missing can_run_on.")
            continue

        can_run_on = job["can_run_on"]

        if not isinstance(can_run_on, list):
            errors.append(f"job {job_id}: can_run_on must be a list.")
            continue

        if not can_run_on:
            errors.append(f"job {job_id}: can_run_on must not be empty.")
            continue

        if len(can_run_on) != len(set(can_run_on)):
            errors.append(f"job {job_id}: can_run_on contains duplicate IDs.")

        for node_id in can_run_on:
            if node_id not in node_ids:
                errors.append(
                    f"job {job_id}: can_run_on ID {node_id} is not a platform node."
                )

            elif node_id in router_ids:
                errors.append(
                    f"job {job_id}: can_run_on ID {node_id} is a router. "
                    "Only processors are allowed."
                )

            elif node_id not in processor_ids:
                errors.append(
                    f"job {job_id}: can_run_on ID {node_id} is not a processor."
                )

    return errors


# ============================================================
# REPORTING
# ============================================================

def print_section(title: str) -> None:
    print("-" * 70)
    print(title)
    print("-" * 70)


def print_errors(errors: list[str]) -> None:
    for error in errors:
        print(f"  - {error}")


def print_report(
    input_path: Path,
    task_ids: list[int],
    edges: list[tuple[int, int]],
    platform: dict[str, Any],
    application_errors: list[str],
    platform_errors: list[str],
    can_run_on_errors: list[str],
    is_valid_dag: bool,
    topo_order: list[int],
    show_topological_order: bool,
    verbose: bool,
) -> None:
    sources, sinks = compute_sources_and_sinks(task_ids, edges)

    processor_ids = get_processor_ids(platform)
    router_ids = get_router_ids(platform)
    platform_connected = is_platform_connected(platform)

    print("=" * 70)
    print("Benchmark JSON Validation Report")
    print("=" * 70)
    print(f"Input file              : {input_path}")
    print(f"Number of tasks         : {len(task_ids)}")
    print(f"Number of messages      : {len(edges)}")
    print(f"Number of source tasks  : {len(sources)}")
    print(f"Number of sink tasks    : {len(sinks)}")
    print(f"Number of platform nodes: {len(get_platform_node_ids(platform))}")
    print(f"Number of processors    : {len(processor_ids)}")
    print(f"Number of routers       : {len(router_ids)}")
    print(f"Platform connected      : {platform_connected}")

    if verbose:
        print(f"Processor IDs           : {sorted(processor_ids)}")
        print(f"Router IDs              : {sorted(router_ids)}")
        print(f"Source task IDs         : {sorted(sources)}")
        print(f"Sink task IDs           : {sorted(sinks)}")

    print_section("Application DAG Check")

    if application_errors:
        print("Application edge validation: FAILED")
        print_errors(application_errors)
    else:
        print("Application edge validation: PASSED")

    if is_valid_dag and not application_errors:
        print("DAG check                  : PASSED")
        print("Result                     : The application graph is a valid DAG.")
    elif is_valid_dag and application_errors:
        print("DAG check                  : PASSED, but edge validation has errors.")
        print("Result                     : Fix application edge errors before using this file.")
    else:
        print("DAG check                  : FAILED")
        print("Result                     : The application graph is NOT a DAG.")

        cycle = find_cycle_path(task_ids, edges)

        if cycle:
            print("One detected cycle:")
            print("  " + " -> ".join(str(x) for x in cycle))
        else:
            print("Cycle exists, but a cycle path could not be reconstructed.")

    if show_topological_order and is_valid_dag:
        print("Topological order:")
        print(topo_order)

    print_section("Platform Check")

    if platform_errors:
        print("Platform validation        : FAILED")
        print_errors(platform_errors)
    else:
        print("Platform validation        : PASSED")
        print("Result                     : Platform nodes/links are valid and connected.")

    print_section("can_run_on Check")

    if can_run_on_errors:
        print("can_run_on validation      : FAILED")
        print_errors(can_run_on_errors)
    else:
        print("can_run_on validation      : PASSED")
        print("Result                     : All jobs can run only on processor nodes.")

    print_section("Final Result")

    all_errors = application_errors + platform_errors + can_run_on_errors

    if not is_valid_dag:
        all_errors.append("Application graph is not a DAG.")

    if all_errors:
        print("Overall validation         : FAILED")
        print("The JSON file is not fully valid for scheduling experiments.")
    else:
        print("Overall validation         : PASSED")
        print("The JSON file is valid for scheduling experiments.")

    print("=" * 70)


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate application DAG, platform, and can_run_on fields in a benchmark JSON file."
    )

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to the generated variant JSON file.",
    )

    parser.add_argument(
        "--show-topological-order",
        action="store_true",
        help="Print the topological order if the application graph is a valid DAG.",
    )

    parser.add_argument(
        "--allow-reverse-platform-links",
        action="store_true",
        help=(
            "Allow both A->B and B->A as separate platform links. "
            "By default, platform links are treated as undirected for duplicate detection."
        ),
    )

    parser.add_argument(
        "--skip-platform-connectivity",
        action="store_true",
        help="Skip the platform connectivity check.",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print additional details such as processor/router/source/sink IDs.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data = load_json(args.input)

    jobs = get_jobs(data)
    messages = get_messages(data)
    platform = get_platform(data)

    task_ids = extract_task_ids(jobs)
    edges = extract_edges(messages)

    application_errors = validate_application_edges(task_ids, edges)
    is_valid_dag, topo_order = topological_sort(task_ids, edges)

    platform_errors = validate_platform(
        platform=platform,
        allow_reverse_links=args.allow_reverse_platform_links,
        require_connected=not args.skip_platform_connectivity,
    )

    can_run_on_errors = validate_can_run_on(jobs, platform)

    print_report(
        input_path=args.input,
        task_ids=task_ids,
        edges=edges,
        platform=platform,
        application_errors=application_errors,
        platform_errors=platform_errors,
        can_run_on_errors=can_run_on_errors,
        is_valid_dag=is_valid_dag,
        topo_order=topo_order,
        show_topological_order=args.show_topological_order,
        verbose=args.verbose,
    )

    if application_errors or platform_errors or can_run_on_errors or not is_valid_dag:
        raise SystemExit(1)


if __name__ == "__main__":
    main()