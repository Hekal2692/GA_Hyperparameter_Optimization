"""Plot summaries from best-fixed-params-per-pair CSV output."""

import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def maybe_show_plot(show_plot):
    if show_plot and matplotlib.get_backend().lower() != "agg":
        plt.show()


def load_rows(csv_path):
    with Path(csv_path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    path = Path(path)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def pair_label(row):
    return f"{int(row['pop_size'])}x{int(row['ngen'])}"


def row_selection_key(row):
    return (
        float(row["best_mean_makespan"]),
        float(row["best_avg_runtime_seconds"]),
        int(row["pop_size"]),
        int(row["ngen"]),
    )


def rows_by_instance_and_budget(rows):
    grouped = {}
    for row in rows:
        instance_name = row["instance_name"]
        budget = int(row["budget"])
        grouped.setdefault(instance_name, {}).setdefault(budget, []).append(row)
    return grouped


def format_float(value, places=2):
    return f"{float(value):.{places}f}"


def build_best_budget_rows(grouped_rows):
    rows = []
    for instance_name, budget_rows in sorted(grouped_rows.items()):
        for budget, records in sorted(budget_rows.items()):
            best_row = min(records, key=row_selection_key)
            rows.append(
                {
                    "instance_name": instance_name,
                    "budget": budget,
                    "best_pair": pair_label(best_row),
                    "best_pop_size": int(best_row["pop_size"]),
                    "best_ngen": int(best_row["ngen"]),
                    "best_mean_makespan": format_float(
                        best_row["best_mean_makespan"], 3
                    ),
                    "best_avg_runtime_seconds": format_float(
                        best_row["best_avg_runtime_seconds"], 3
                    ),
                    "best_total_runtime_seconds": format_float(
                        best_row["best_total_runtime_seconds"], 3
                    ),
                    "best_fixed_cxpb": format_float(best_row["best_fixed_cxpb"], 2),
                    "best_fixed_mutpb": format_float(best_row["best_fixed_mutpb"], 2),
                    "best_shared_mutation": format_float(
                        best_row["best_shared_scheduler_mutation_probability"], 2
                    ),
                }
            )
    return rows


def plot_instance_budget_panels(instance_name, budget_rows, output_path, show_plot=False):
    budgets = sorted(budget_rows)
    num_panels = len(budgets)
    num_cols = min(3, max(1, num_panels))
    num_rows = math.ceil(num_panels / num_cols)
    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(6.4 * num_cols, 4.8 * num_rows),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    for axis, budget in zip(axes_flat, budgets):
        rows = sorted(
            budget_rows[budget],
            key=lambda row: (
                float(row["best_mean_makespan"]),
                int(row["pop_size"]),
                int(row["ngen"]),
            ),
        )
        labels = [pair_label(row) for row in rows]
        makespans = [float(row["best_mean_makespan"]) for row in rows]
        colors = ["#c9a227"] + ["#4c78a8"] * (len(rows) - 1)
        best_value = makespans[0]

        bars = axis.bar(
            labels,
            makespans,
            color=colors,
            edgecolor="#1f1f1f",
            linewidth=0.6,
        )
        axis.set_title(f"{instance_name} | budget={budget} | {len(rows)} feasible pairs")
        axis.set_ylabel("Best mean makespan")
        axis.grid(axis="y", alpha=0.3)
        axis.tick_params(axis="x", rotation=45)
        axis.set_axisbelow(True)
        axis.set_ylim(top=max(makespans) * 1.06)
        for bar, value in zip(bars, makespans):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=7,
                color="#2f2f2f",
            )

    for axis in axes_flat[num_panels:]:
        axis.axis("off")

    fig.suptitle(
        f"{instance_name}: best mean makespan by feasible (pop_size, ngen) pair",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=200)
    maybe_show_plot(show_plot)
    plt.close(fig)


def plot_instance_hyperparameter_tables(
    instance_name,
    budget_rows,
    output_path,
    show_plot=False,
):
    budgets = sorted(budget_rows)
    num_panels = len(budgets)
    num_cols = 2
    num_rows = math.ceil(num_panels / num_cols)
    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(14.5, 4.1 * num_rows),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    for axis, budget in zip(axes_flat, budgets):
        rows = sorted(
            budget_rows[budget],
            key=lambda row: (int(row["pop_size"]), int(row["ngen"])),
        )
        best_row = min(rows, key=row_selection_key)
        table_rows = [
            [
                pair_label(row),
                format_float(row["best_fixed_cxpb"], 2),
                format_float(row["best_fixed_mutpb"], 2),
                format_float(row["best_shared_scheduler_mutation_probability"], 2),
                format_float(row["best_mean_makespan"], 1),
                format_float(row["best_avg_runtime_seconds"], 2),
            ]
            for row in rows
        ]
        row_colors = [
            ["#fff3bf"] * 6 if row is best_row else ["#f8fbff"] * 6
            for row in rows
        ]
        axis.axis("off")
        table = axis.table(
            cellText=table_rows,
            colLabels=["Pair", "cxpb", "mutpb", "shared", "mean", "runtime"],
            cellColours=row_colors,
            cellLoc="center",
            colLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.18)
        for (row_index, _column_index), cell in table.get_celld().items():
            if row_index == 0:
                cell.set_facecolor("#263238")
                cell.set_text_props(color="white", weight="bold")
            cell.set_edgecolor("#d6dde3")
        axis.set_title(f"budget={budget}", fontsize=11, pad=8)

    for axis in axes_flat[num_panels:]:
        axis.axis("off")

    fig.suptitle(
        f"{instance_name}: selected fixed hyperparameters per feasible pair",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=200)
    maybe_show_plot(show_plot)
    plt.close(fig)


def plot_instance_best_budget_table(
    instance_name,
    best_budget_rows,
    output_path,
    show_plot=False,
):
    rows = [row for row in best_budget_rows if row["instance_name"] == instance_name]
    table_rows = [
        [
            row["budget"],
            row["best_pair"],
            row["best_mean_makespan"],
            row["best_avg_runtime_seconds"],
            row["best_total_runtime_seconds"],
            row["best_fixed_cxpb"],
            row["best_fixed_mutpb"],
            row["best_shared_mutation"],
        ]
        for row in rows
    ]

    fig, axis = plt.subplots(figsize=(13.5, 3.8))
    axis.axis("off")
    table = axis.table(
        cellText=table_rows,
        colLabels=[
            "Budget",
            "Best pair",
            "Mean makespan",
            "Avg runtime (s)",
            "Total runtime (s)",
            "cxpb",
            "mutpb",
            "shared",
        ],
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)
    for (row_index, _column_index), cell in table.get_celld().items():
        if row_index == 0:
            cell.set_facecolor("#263238")
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#f8fbff")
        cell.set_edgecolor("#d6dde3")
    axis.set_title(
        f"{instance_name}: best feasible pair per budget with runtime",
        fontsize=13,
        pad=10,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    maybe_show_plot(show_plot)
    plt.close(fig)


def build_instance_hyperparameter_rows(instance_name, budget_rows):
    rows = []
    for budget, records in sorted(budget_rows.items()):
        sorted_records = sorted(records, key=row_selection_key)
        best_record = sorted_records[0]
        for rank, row in enumerate(sorted_records, start=1):
            rows.append(
                {
                    "instance_name": instance_name,
                    "budget": budget,
                    "rank_within_budget": rank,
                    "is_best_for_budget": "yes" if row is best_record else "no",
                    "pair": pair_label(row),
                    "pop_size": int(row["pop_size"]),
                    "ngen": int(row["ngen"]),
                    "best_fixed_cxpb": format_float(row["best_fixed_cxpb"], 2),
                    "best_fixed_mutpb": format_float(row["best_fixed_mutpb"], 2),
                    "best_shared_mutation": format_float(
                        row["best_shared_scheduler_mutation_probability"], 2
                    ),
                    "best_mean_makespan": format_float(
                        row["best_mean_makespan"], 3
                    ),
                    "best_avg_runtime_seconds": format_float(
                        row["best_avg_runtime_seconds"], 3
                    ),
                    "best_total_runtime_seconds": format_float(
                        row["best_total_runtime_seconds"], 3
                    ),
                    "best_fixed_parameter_combo_label": row[
                        "best_fixed_parameter_combo_label"
                    ],
                }
            )
    return rows


def print_runtime_report(best_budget_rows):
    print("Best per budget runtime report:")
    for row in best_budget_rows:
        print(
            "{instance} budget={budget} pair={pair} "
            "mean={mean} avg_runtime={avg}s total_runtime={total}s "
            "params=cxpb:{cxpb},mutpb:{mutpb},shared:{shared}".format(
                instance=row["instance_name"],
                budget=row["budget"],
                pair=row["best_pair"],
                mean=row["best_mean_makespan"],
                avg=row["best_avg_runtime_seconds"],
                total=row["best_total_runtime_seconds"],
                cxpb=row["best_fixed_cxpb"],
                mutpb=row["best_fixed_mutpb"],
                shared=row["best_shared_mutation"],
            )
        )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Plot one multi-panel figure per instance from a "
            "*_best_fixed_params_per_pair.csv file."
        )
    )
    parser.add_argument(
        "input_csv",
        type=Path,
        help="Path to a *_best_fixed_params_per_pair.csv file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write the output plots. Default: same folder as the CSV.",
    )
    parser.add_argument(
        "--show-plot",
        action="store_true",
        help="Display plots after saving them when an interactive backend is available.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Write CSV tables only and skip PNG plot generation.",
    )
    parser.add_argument(
        "--skip-csv-tables",
        action="store_true",
        help="Generate plots without rewriting derived CSV table outputs.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    rows = load_rows(args.input_csv)
    if not rows:
        raise ValueError(f"No rows found in CSV: {args.input_csv}")

    output_dir = args.output_dir if args.output_dir is not None else args.input_csv.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    grouped = rows_by_instance_and_budget(rows)
    stem = args.input_csv.stem
    best_budget_rows = build_best_budget_rows(grouped)
    if not args.skip_csv_tables:
        best_budget_csv_path = output_dir / f"{stem}_best_by_budget.csv"
        write_csv(best_budget_csv_path, best_budget_rows)
        print(best_budget_csv_path.resolve())
    print_runtime_report(best_budget_rows)

    for instance_name, budget_rows in sorted(grouped.items()):
        if not args.skip_csv_tables:
            instance_hyperparameter_csv_path = (
                output_dir / f"{stem}_{instance_name}_hyperparameter_table.csv"
            )
            write_csv(
                instance_hyperparameter_csv_path,
                build_instance_hyperparameter_rows(instance_name, budget_rows),
            )
            print(instance_hyperparameter_csv_path.resolve())

            instance_best_budget_csv_path = (
                output_dir / f"{stem}_{instance_name}_best_budget_table.csv"
            )
            write_csv(
                instance_best_budget_csv_path,
                [
                    row
                    for row in best_budget_rows
                    if row["instance_name"] == instance_name
                ],
            )
            print(instance_best_budget_csv_path.resolve())

        if args.skip_plots:
            continue

        output_path = output_dir / f"{stem}_{instance_name}_makespan_panels.png"
        plot_instance_budget_panels(
            instance_name,
            budget_rows,
            output_path,
            show_plot=args.show_plot,
        )
        print(output_path.resolve())

        hyperparameter_output_path = (
            output_dir / f"{stem}_{instance_name}_hyperparameter_tables.png"
        )
        plot_instance_hyperparameter_tables(
            instance_name,
            budget_rows,
            hyperparameter_output_path,
            show_plot=args.show_plot,
        )
        print(hyperparameter_output_path.resolve())

        best_budget_output_path = (
            output_dir / f"{stem}_{instance_name}_best_budget_table.png"
        )
        plot_instance_best_budget_table(
            instance_name,
            best_budget_rows,
            best_budget_output_path,
            show_plot=args.show_plot,
        )
        print(best_budget_output_path.resolve())


if __name__ == "__main__":
    main()
