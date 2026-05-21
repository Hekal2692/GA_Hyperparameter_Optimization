"""Separate comparison pipeline for standalone benchmark runs vs nested-GA runs."""

import argparse
from pathlib import Path

from SchedulerGa import (
    create_comparison_artifacts,
    create_comparison_batch_output_paths,
    load_standalone_results_payload,
    match_nested_runs_by_instance,
    write_json,
)
from ga_config import DEFAULT_CONFIG_PATH, load_config


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

    parser = argparse.ArgumentParser(
        description=(
            "Compare previously saved standalone benchmark runs against previously "
            "saved nested-GA training and/or validation results."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=pre_args.config,
        help=f"Config file to load. Default: {DEFAULT_CONFIG_PATH.name}.",
    )
    parser.add_argument(
        "--standalone-runs",
        nargs="+",
        type=Path,
        required=True,
        help=(
            "Standalone benchmark run directories or *_results.json files produced "
            "by SchedulerGa.py."
        ),
    )
    parser.add_argument(
        "--nested-runs",
        nargs="+",
        type=Path,
        required=True,
        help=(
            "Nested-GA run directories, *_validation_results.json files, or "
            "*_best_result.json files produced by OptimizerGa.py."
        ),
    )
    parser.add_argument(
        "--nested-candidate-rank",
        type=int,
        default=1,
        help=(
            "Which validated nested-GA candidate rank to compare against. "
            "Default: 1."
        ),
    )
    parser.add_argument(
        "--nested-source",
        choices=("validation", "training", "both"),
        default="validation",
        help=(
            "Which nested-GA behavior to compare against. "
            "`validation` uses unseen-seed validation results, "
            "`training` uses the search-time selected best result, "
            "and `both` writes both comparisons."
        ),
    )
    parser.add_argument(
        "--show-plot",
        action="store_true",
        help="Display plots after saving them when an interactive backend is available.",
    )
    return parser.parse_args(remaining), config


def build_comparison_batch_summary(
    standalone_results_payloads,
    comparison_payloads,
    unmatched_standalone_runs,
    unmatched_nested_runs,
    failed_comparisons,
):
    return {
        "description": (
            "Separate comparison pipeline summary for standalone benchmark runs "
            "vs nested-GA training and/or validation runs."
        ),
        "matched_comparison_count": len(comparison_payloads),
        "matched_instance_count": len(
            {payload["instance_name"] for payload in comparison_payloads}
        ),
        "standalone_benchmark_runs": [
            {
                "instance_name": payload["instance_name"],
                "run_name": payload["run_name"],
                "results_path": payload["results_path"],
                "benchmark_repeats": payload.get("benchmark_repeats", payload.get("repeats")),
                "mean_makespan": payload["mean_makespan"],
                "std_makespan": payload["std_makespan"],
                "avg_runtime_seconds": payload["avg_runtime_seconds"],
            }
            for payload in standalone_results_payloads
        ],
        "standalone_runs": [
            {
                "instance_name": payload["instance_name"],
                "run_name": payload["run_name"],
                "results_path": payload["results_path"],
                "mean_makespan": payload["mean_makespan"],
                "std_makespan": payload["std_makespan"],
                "avg_runtime_seconds": payload["avg_runtime_seconds"],
            }
            for payload in standalone_results_payloads
        ],
        "comparisons": [
            {
                "instance_name": payload["instance_name"],
                "nested_source_mode": payload["nested_source_mode"],
                "comparison_json_path": payload["comparison_json_path"],
                "comparison_plot_path": payload["comparison_plot_path"],
                "standalone_benchmark_results_path": payload["standalone_benchmark_results_path"],
                "standalone_results_path": payload["static_results_path"],
                "nested_source_path": payload["nested_source_path"],
                "nested_validation_results_path": payload["nested_validation_results_path"],
                "nested_best_result_path": payload["nested_best_result_path"],
                "nested_candidate_rank": payload["nested_candidate_rank"],
                "mean_makespan_improvement_of_tuned": payload["mean_makespan_improvement_of_tuned"],
                "relative_makespan_improvement_percent_of_tuned": payload[
                    "relative_makespan_improvement_percent_of_tuned"
                ],
                "one_sided_pvalue_tuned_better": payload["welch_ttest_makespan"].get(
                    "one_sided_pvalue_tuned_better"
                ),
            }
            for payload in comparison_payloads
        ],
        "unmatched_standalone_runs": unmatched_standalone_runs,
        "unmatched_nested_runs": unmatched_nested_runs,
        "failed_comparisons": failed_comparisons,
    }


if __name__ == "__main__":
    args, config = parse_args()
    standalone_results_payloads = [
        load_standalone_results_payload(standalone_run)
        for standalone_run in args.standalone_runs
    ]
    standalone_results_payloads.sort(key=lambda payload: payload["instance_name"])
    nested_runs_by_instance = match_nested_runs_by_instance(args.nested_runs)
    nested_sources = (
        ["validation", "training"]
        if args.nested_source == "both"
        else [args.nested_source]
    )

    comparison_batch_paths = create_comparison_batch_output_paths(config)
    comparison_output_dir = comparison_batch_paths["run_dir"]

    comparison_payloads = []
    matched_nested_instances = set()
    unmatched_standalone_runs = []
    failed_comparisons = []

    for standalone_results_payload in standalone_results_payloads:
        nested_path = nested_runs_by_instance.get(standalone_results_payload["instance_name"])
        if nested_path is None:
            unmatched_standalone_runs.append(
                {
                    "instance_name": standalone_results_payload["instance_name"],
                    "standalone_results_path": standalone_results_payload["results_path"],
                }
            )
            continue

        for nested_source in nested_sources:
            try:
                comparison_payload = create_comparison_artifacts(
                    standalone_results_payload,
                    nested_path=nested_path,
                    nested_source=nested_source,
                    candidate_rank=args.nested_candidate_rank,
                    show_plot=args.show_plot,
                    output_dir=comparison_output_dir,
                )
            except Exception as error:
                failed_comparisons.append(
                    {
                        "instance_name": standalone_results_payload["instance_name"],
                        "nested_source_mode": nested_source,
                        "standalone_results_path": standalone_results_payload["results_path"],
                        "nested_path": str(Path(nested_path).resolve()),
                        "error": str(error),
                    }
                )
                continue

            comparison_payloads.append(comparison_payload)
        matched_nested_instances.add(standalone_results_payload["instance_name"])

    unmatched_nested_runs = [
        {
            "instance_name": instance_name,
            "nested_path": str(Path(nested_path).resolve()),
        }
        for instance_name, nested_path in nested_runs_by_instance.items()
        if instance_name not in matched_nested_instances
    ]

    summary_payload = build_comparison_batch_summary(
        standalone_results_payloads,
        comparison_payloads,
        unmatched_standalone_runs,
        unmatched_nested_runs,
        failed_comparisons,
    )
    write_json(comparison_batch_paths["summary_path"], summary_payload)

    print("Comparison batch summary saved to:", comparison_batch_paths["summary_path"].resolve())
    for comparison_payload in comparison_payloads:
        print(
            "Comparison:",
            {
                "instance_name": comparison_payload["instance_name"],
                "nested_source_mode": comparison_payload["nested_source_mode"],
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
    if unmatched_standalone_runs:
        print("Unmatched standalone runs:", unmatched_standalone_runs)
    if unmatched_nested_runs:
        print("Unmatched nested runs:", unmatched_nested_runs)
    if failed_comparisons:
        print("Failed comparisons:", failed_comparisons)
