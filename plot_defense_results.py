import argparse
import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import ttest_rel


DEFENSE_CONFIGS = [
    {"key": "KNN", "display": "$\\mathit{k}$NN", "aliases": ["KNN", "kNN"]},
    {"key": "OneClassSVM", "display": "One-Class SVM", "aliases": ["OneClassSVM"]},
    {"key": "DeepSVDD", "display": "Deep SVDD", "aliases": ["DeepSVDD", "DeepSVDD"]},
    {"key": "LSTMAE", "display": "LSTM-AE", "aliases": ["LSTMAE", "LSTM-AE"]},
    {"key": "MADGAN", "display": "MAD-GAN", "aliases": ["MADGAN", "MAD-GAN"]},
]

MEASURE_ORDER = ["Accuracy", "Precision", "Recall", "F1"]
MEASURE_KEY = {
    "Accuracy": "accuracy",
    "Precision": "precision",
    "Recall": "recall",
    "F1": "f1",
}

COHORT_ORDER = [
    ("less_vulnerable", "Less Vulnerable (OE)"),
    ("samples_training", "Random Samples (OE)"),
    ("more_vulnerable", "More Vulnerable (OE)"),
    ("all_patients_OE", "All Patients (OE)"),
    ("all_patients_benign", "All Patients (Benign)"),
]


def apply_cohort_axis_labels(ax):
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels([label for _, label in COHORT_ORDER])
    # Keep the first cohort at the top.
    ax.invert_yaxis()


def compute_stats(values):
    values = np.array(values, dtype=float)
    mean = np.mean(values)
    if len(values) > 1:
        std = np.std(values, ddof=1)
        ci95 = 1.96 * std / np.sqrt(len(values))
    else:
        std = 0.0
        ci95 = 0.0
    return mean, std, ci95


def find_results_csv(output_directory, aliases):
    for alias in aliases:
        candidate = output_directory / alias / f"{alias}_combined_results.csv"
        if candidate.exists():
            return candidate, alias
    return None, None


def parse_results_csv(csv_path):
    metrics = {
        "less_vulnerable": {"accuracy": [], "precision": [], "recall": [], "f1": []},
        "samples_training": {"accuracy": [], "precision": [], "recall": [], "f1": []},
        "more_vulnerable": {"accuracy": [], "precision": [], "recall": [], "f1": []},
        "all_patients_benign": {"accuracy": [], "precision": [], "recall": [], "f1": []},
        "all_patients_OE": {"accuracy": [], "precision": [], "recall": [], "f1": []},
    }

    with open(csv_path, "r", newline="") as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)
        next(reader, None)

        for row in reader:
            # A full 5-cohort row needs 21 columns (Patient + 5 cohorts x 4 metrics);
            # this reads through row[20]. Stop on short/incomplete rows (e.g. trailing
            # blanks, or stale 4-cohort/17-column files from an older pipeline version)
            # instead of crashing with an IndexError.
            if len(row) < 21 or row[1] == "":
                break

            metrics["less_vulnerable"]["accuracy"].append(float(row[1]))
            metrics["less_vulnerable"]["precision"].append(float(row[2]))
            metrics["less_vulnerable"]["recall"].append(float(row[3]))
            metrics["less_vulnerable"]["f1"].append(float(row[4]))

            metrics["samples_training"]["accuracy"].append(float(row[5]))
            metrics["samples_training"]["precision"].append(float(row[6]))
            metrics["samples_training"]["recall"].append(float(row[7]))
            metrics["samples_training"]["f1"].append(float(row[8]))

            metrics["more_vulnerable"]["accuracy"].append(float(row[9]))
            metrics["more_vulnerable"]["precision"].append(float(row[10]))
            metrics["more_vulnerable"]["recall"].append(float(row[11]))
            metrics["more_vulnerable"]["f1"].append(float(row[12]))

            metrics["all_patients_OE"]["accuracy"].append(float(row[13]))
            metrics["all_patients_OE"]["precision"].append(float(row[14]))
            metrics["all_patients_OE"]["recall"].append(float(row[15]))
            metrics["all_patients_OE"]["f1"].append(float(row[16]))

            metrics["all_patients_benign"]["accuracy"].append(float(row[17]))
            metrics["all_patients_benign"]["precision"].append(float(row[18]))
            metrics["all_patients_benign"]["recall"].append(float(row[19]))
            metrics["all_patients_benign"]["f1"].append(float(row[20]))


    return metrics


def plot_box_with_stats(ax, series, show_legend=False):
    bp = ax.boxplot(series, vert=False)
    positions = range(1, len(series) + 1)

    for pos, data in zip(positions, series):
        if not data:
            continue
        mean, _, ci95 = compute_stats(data)
        ax.plot(mean, pos, "ro")
        ax.errorbar(mean, pos, xerr=ci95, fmt="none", capsize=4, ecolor="blue")

    if show_legend and bp["medians"]:
        median_color = bp["medians"][0].get_color()
        legend_elements = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="red", markersize=6, label="Mean"),
            Line2D([0], [0], color="blue", lw=1, label="95% CI"),
            Line2D([0, 0], [0, 1], color=median_color, lw=2, label="Median"),
        ]
        ax.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1.02, 1.3), fontsize=10)


def print_ttests(defense_name, metrics, output_file):
    for measure in ["precision", "recall"]:
        less = metrics["less_vulnerable"][measure]
        all_patients = metrics["all_patients_benign"][measure]

        if len(less) != len(all_patients) or len(less) < 2:
            continue

        t_stat, p_value = ttest_rel(less, all_patients)
        label = measure.capitalize()
        line1 = f"{defense_name} t-test:"
        line2 = f"{label} T-statistic: {t_stat:.3f}, P-value: {p_value:.11f}"
        if p_value < 0.05:
            line3 = "The improvement is statistically significant (p < 0.05)."
        else:
            line3 = "The improvement is not statistically significant (p >= 0.05)."

        print(line1)
        print(line2)
        print(line3)
        output_file.write(line1 + "\n")
        output_file.write(line2 + "\n")
        output_file.write(line3 + "\n")
        output_file.write("\n")


def plot_individual_defense(dataset, out_dir, defense_key, display_name, metrics, font_size=16):
    defense_dir = out_dir / defense_key
    os.makedirs(defense_dir, exist_ok=True)

    for measure in MEASURE_ORDER:
        measure_key = MEASURE_KEY[measure]
        series = [metrics[cohort][measure_key] for cohort, _ in COHORT_ORDER]

        fig = plt.figure(figsize=(7, 2))
        ax = fig.add_subplot(111)
        ax.set_title(f"{display_name} {measure}", fontsize=font_size + 4)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        plt.xticks(fontsize=font_size)
        plt.yticks(fontsize=font_size)

        plot_box_with_stats(ax, series, show_legend=(dataset == "PhysioNetCinC"))
        apply_cohort_axis_labels(ax)
        plt.tight_layout()
        plt.savefig(defense_dir / f"{dataset}_{defense_key}_{measure}.pdf")
        plt.close(fig)


def plot_combined_by_measure(dataset, out_dir, available_defenses, font_size=16):
    combined_dir = out_dir / "Combined"
    os.makedirs(combined_dir, exist_ok=True)

    for measure in MEASURE_ORDER:
        measure_key = MEASURE_KEY[measure]
        rows = len(available_defenses)
        fig, axes = plt.subplots(rows, 1, figsize=(7, 2 * rows))

        if rows == 1:
            axes = [axes]

        for idx, defense in enumerate(available_defenses):
            ax = axes[idx]
            ax.set_title(defense["display"], fontsize=font_size + 4)
            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()
            ax.tick_params(axis="x", labelsize=font_size)
            ax.tick_params(axis="y", labelsize=font_size)

            series = [defense["metrics"][cohort][measure_key] for cohort, _ in COHORT_ORDER]
            plot_box_with_stats(ax, series, show_legend=(dataset == "PhysioNetCinC" and idx == 0))
            apply_cohort_axis_labels(ax)

        plt.tight_layout()
        plt.savefig(combined_dir / f"{dataset}_{measure}.pdf")
        plt.close(fig)


def plot_combined_by_defense(dataset, out_dir, available_defenses, font_size=16):
    for defense in available_defenses:
        fig, axes = plt.subplots(4, 1, figsize=(13, 15))

        for idx, measure in enumerate(MEASURE_ORDER):
            measure_key = MEASURE_KEY[measure]
            ax = axes[idx]
            ax.set_title(f"{defense['display']} {measure}", fontsize=font_size + 4)
            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()
            ax.tick_params(axis="x", labelsize=font_size)
            ax.tick_params(axis="y", labelsize=font_size)

            series = [defense["metrics"][cohort][measure_key] for cohort, _ in COHORT_ORDER]
            plot_box_with_stats(ax, series, show_legend=False)
            apply_cohort_axis_labels(ax)

        plt.tight_layout()
        plt.savefig(out_dir / defense["key"] / f"{dataset}_{defense['key']}_Combined.pdf")
        plt.close(fig)


def plot_defense_results(dataset, output_directory):
    out_dir = output_directory / "plots"
    os.makedirs(out_dir, exist_ok=True)
    t_test_path = out_dir / "t_test_output.txt"

    available_defenses = []
    for defense_cfg in DEFENSE_CONFIGS:
        csv_path, alias = find_results_csv(output_directory, defense_cfg["aliases"])
        if csv_path is None:
            continue

        metrics = parse_results_csv(csv_path)
        if not metrics["all_patients_benign"]["accuracy"]:
            continue

        available_defenses.append(
            {
                "key": defense_cfg["key"],
                "display": defense_cfg["display"],
                "alias": alias,
                "metrics": metrics,
            }
        )

    if not available_defenses:
        print(f"No defense result files found under {output_directory}")
        return

    print(f"Using defenses: {', '.join(d['key'] for d in available_defenses)}")

    with open(t_test_path, "w") as t_test_file:
        for defense in available_defenses:
            print_ttests(defense["key"], defense["metrics"], t_test_file)
            plot_individual_defense(dataset, out_dir, defense["key"], defense["display"], defense["metrics"])

    plot_combined_by_measure(dataset, out_dir, available_defenses)
    plot_combined_by_defense(dataset, out_dir, available_defenses)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot defense results for one dataset.",
        epilog="Example: python plot_defense_results.py OhioT1DM OhioT1DM/output/defense_output",
    )
    parser.add_argument(
        "dataset",
        choices=["OhioT1DM", "MIMIC", "PhysioNetCinC"],
        help="Dataset name.",
    )
    parser.add_argument(
        "out_dir",
        nargs="?",
        default=None,
        help="Defense output directory. Defaults to {dataset}/output/defense_output.",
    )

    args = parser.parse_args()
    script_dir = Path(__file__).resolve().parent

    if args.out_dir is None:
        output_directory = script_dir / args.dataset / "output" / "defense_output"
    else:
        output_directory = Path(args.out_dir)
        if not output_directory.is_absolute():
            output_directory = script_dir / output_directory

    plot_defense_results(args.dataset, output_directory)
