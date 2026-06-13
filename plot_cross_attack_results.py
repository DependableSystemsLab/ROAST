"""Generate publication-quality figures for the ISSRE_26 submission.

This script consumes the namespaced defense-evaluation outputs produced by the
attack-type / cross-attack-generalization pipeline:

    <Dataset>/output/<NAMESPACE>/defense_output/<DEFENSE>/<DEFENSE>_combined_results.csv

where <NAMESPACE> is either a single attack type (same-attack runs, e.g. ``FGSM``)
or ``<RISK>_to_<EVAL>`` for cross-attack runs (e.g. ``FGSM_to_PGD``). It produces
three families of figures, for each of the four metrics (Recall, Precision, F1,
Accuracy):

  1. attack_comparison/ -- grouped bar charts comparing ROAST (Less Vulnerable, OE)
     against the All Patients (Benign) baseline, per defense, across attack types.
     Extends the paper's RQ2 story to the new PGD / C&W attacks.
  2. cross_attack/       -- heatmaps per (dataset, defense): rows = risk-profiling
     attack, cols = evaluation attack. The diagonal is same-attack performance,
     off-diagonal cells show cross-attack generalization.
  3. box_plots/          -- per-attack, per-cohort box plots in the same style as the
     existing paper figures (one subplot per defense, all five cohorts).

It also writes a per-dataset summary CSV with ROAST-vs-baseline gains and paired
t-tests, to support the quantitative claims in the Evaluation chapter.

The script reuses the CSV parsing and box-plot helpers from ``plot_defense_results``
so the numbers stay identical to the existing pipeline.

Example:
    python plot_cross_attack_results.py
    python plot_cross_attack_results.py --datasets OhioT1DM --metrics Recall
    python plot_cross_attack_results.py --out_dir "../../Papers/In_Progress/ISSRE_26/Figures/generated"
"""

import argparse
import csv
import os
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: must precede the first pyplot import

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_rel

# Reuse the exact parsing / styling primitives from the existing plotting script
# so this publication script reports identical numbers.
from plot_defense_results import (
    DEFENSE_CONFIGS,
    MEASURE_ORDER,
    MEASURE_KEY,
    COHORT_ORDER,
    parse_results_csv,
    find_results_csv,
    compute_stats,
    plot_box_with_stats,
    apply_cohort_axis_labels,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ATTACK_ORDER = ["URET", "FGSM", "PGD", "CW"]
ATTACK_DISPLAY = {"URET": "URET", "FGSM": "FGSM", "PGD": "PGD", "CW": "C&W"}

# The paper's headline contrast (RQ2): ROAST = train on less-vulnerable cluster with
# outlier exposure; baseline = indiscriminate training on all patients (benign only).
ROAST_COHORT = "less_vulnerable"
ROAST_LABEL = "ROAST (Less Vulnerable, OE)"
BASELINE_COHORT = "all_patients_benign"
BASELINE_LABEL = "All Patients (Benign)"

ROAST_COLOR = "#2c7fb8"
BASELINE_COLOR = "#bdbdbd"

DATASETS = ["OhioT1DM", "MIMIC", "PhysioNetCinC"]

CROSS_RE = re.compile(r"^([A-Za-z]+)_to_([A-Za-z]+)$")


def set_publication_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        # Embed TrueType (not Type-3) fonts -- required by most IEEE/ACM submission systems.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
def classify_namespace(name):
    """Return (risk_attack, eval_attack, is_cross) or None if not an attack namespace."""
    match = CROSS_RE.match(name)
    if match:
        risk, ev = match.group(1), match.group(2)
        if risk in ATTACK_ORDER and ev in ATTACK_ORDER:
            return risk, ev, True
        return None
    if name in ATTACK_ORDER:
        return name, name, False
    return None


def discover_namespaces(dataset_root):
    """Map namespace name -> {risk, eval, is_cross, defenses: {key: metrics}}."""
    discovered = {}
    output_dir = dataset_root / "output"
    if not output_dir.is_dir():
        return discovered

    for ns_dir in sorted(output_dir.iterdir()):
        if not ns_dir.is_dir():
            continue
        classified = classify_namespace(ns_dir.name)
        if classified is None:
            continue

        defense_output = ns_dir / "defense_output"
        if not defense_output.is_dir():
            continue

        defenses = {}
        for cfg in DEFENSE_CONFIGS:
            csv_path, _alias = find_results_csv(defense_output, cfg["aliases"])
            if csv_path is None:
                continue
            metrics = parse_results_csv(csv_path)
            if not metrics["all_patients_benign"]["accuracy"]:
                continue
            defenses[cfg["key"]] = metrics

        if defenses:
            risk, ev, is_cross = classified
            discovered[ns_dir.name] = {
                "risk": risk,
                "eval": ev,
                "is_cross": is_cross,
                "defenses": defenses,
            }
    return discovered


def cohort_stat(metrics, cohort, measure_key):
    """Return (mean, std, ci95) for a cohort/measure, or None if no data."""
    values = metrics[cohort][measure_key]
    if not values:
        return None
    return compute_stats(values)


# ---------------------------------------------------------------------------
# Figure 1: attack-type comparison (ROAST vs baseline across attacks)
# ---------------------------------------------------------------------------
def plot_attack_comparison(dataset, namespaces, metric, out_dir):
    measure_key = MEASURE_KEY[metric]
    same = {ns["risk"]: ns for ns in namespaces.values() if not ns["is_cross"]}
    attacks = [a for a in ATTACK_ORDER if a in same]
    if not attacks:
        return None

    defense_keys = [
        cfg["key"]
        for cfg in DEFENSE_CONFIGS
        if any(cfg["key"] in same[a]["defenses"] for a in attacks)
    ]
    if not defense_keys:
        return None

    display_of = {cfg["key"]: cfg["display"] for cfg in DEFENSE_CONFIGS}

    rows = len(defense_keys)
    fig, axes = plt.subplots(rows, 1, figsize=(7, 2.6 * rows), squeeze=False)
    axes = axes[:, 0]

    x = np.arange(len(attacks))
    width = 0.38

    for idx, dkey in enumerate(defense_keys):
        ax = axes[idx]
        roast_means, roast_ci, base_means, base_ci = [], [], [], []
        for a in attacks:
            metrics = same[a]["defenses"].get(dkey)
            rs = cohort_stat(metrics, ROAST_COHORT, measure_key) if metrics else None
            bs = cohort_stat(metrics, BASELINE_COHORT, measure_key) if metrics else None
            roast_means.append(rs[0] if rs else np.nan)
            roast_ci.append(rs[2] if rs else 0.0)
            base_means.append(bs[0] if bs else np.nan)
            base_ci.append(bs[2] if bs else 0.0)

        ax.bar(x - width / 2, roast_means, width, yerr=roast_ci, capsize=3,
               label=ROAST_LABEL, color=ROAST_COLOR)
        ax.bar(x + width / 2, base_means, width, yerr=base_ci, capsize=3,
               label=BASELINE_LABEL, color=BASELINE_COLOR)

        ax.set_xticks(x)
        ax.set_xticklabels([ATTACK_DISPLAY[a] for a in attacks])
        ax.set_ylabel(metric)
        ax.set_ylim(0, 1.05)
        ax.set_title(display_of[dkey])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if idx == 0:
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.42),
                      ncol=2, frameon=False)

    fig.tight_layout()
    path = out_dir / f"{dataset}_{metric}_attack_comparison.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure 2: cross-attack generalization heatmap
# ---------------------------------------------------------------------------
def plot_cross_attack(dataset, namespaces, metric, out_dir):
    measure_key = MEASURE_KEY[metric]
    paths = []

    for cfg in DEFENSE_CONFIGS:
        dkey = cfg["key"]
        cells = {}
        present = set()
        for ns in namespaces.values():
            metrics = ns["defenses"].get(dkey)
            if metrics is None:
                continue
            stat = cohort_stat(metrics, ROAST_COHORT, measure_key)
            if stat is None:
                continue
            cells[(ns["risk"], ns["eval"])] = stat[0]
            present.add(ns["risk"])
            present.add(ns["eval"])

        if not cells:
            continue

        attacks = [a for a in ATTACK_ORDER if a in present]
        has_cross = any(r != e for (r, e) in cells)
        # A heatmap is only meaningful with >=2 attack axes or an explicit cross run.
        if len(attacks) < 2 and not has_cross:
            continue

        matrix = np.full((len(attacks), len(attacks)), np.nan)
        for i, r in enumerate(attacks):
            for j, e in enumerate(attacks):
                if (r, e) in cells:
                    matrix[i, j] = cells[(r, e)]

        size = 1.15 * len(attacks) + 1.8
        fig, ax = plt.subplots(figsize=(size, size - 0.4))
        im = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0, aspect="equal")

        ax.set_xticks(range(len(attacks)))
        ax.set_xticklabels([ATTACK_DISPLAY[a] for a in attacks])
        ax.set_yticks(range(len(attacks)))
        ax.set_yticklabels([ATTACK_DISPLAY[a] for a in attacks])
        ax.set_xlabel("Evaluation attack")
        ax.set_ylabel("Risk-profiling attack")
        ax.set_title(cfg["display"])

        for i in range(len(attacks)):
            for j in range(len(attacks)):
                val = matrix[i, j]
                if np.isnan(val):
                    ax.text(j, i, "--", ha="center", va="center",
                            color="0.4", fontsize=10)
                else:
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            color="white" if val < 0.6 else "black", fontsize=10)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(metric)

        fig.tight_layout()
        path = out_dir / f"{dataset}_{dkey}_{metric}_crossattack.pdf"
        fig.savefig(path)
        plt.close(fig)
        paths.append(path)

    return paths


# ---------------------------------------------------------------------------
# Figure 3: per-attack, per-cohort box plots (existing paper style)
# ---------------------------------------------------------------------------
def plot_attack_box(dataset, namespaces, metric, out_dir):
    measure_key = MEASURE_KEY[metric]
    same = {ns["risk"]: ns for ns in namespaces.values() if not ns["is_cross"]}
    paths = []

    for a in [x for x in ATTACK_ORDER if x in same]:
        ns = same[a]
        defenses = [cfg for cfg in DEFENSE_CONFIGS if cfg["key"] in ns["defenses"]]
        if not defenses:
            continue

        rows = len(defenses)
        fig, axes = plt.subplots(rows, 1, figsize=(7, 2.2 * rows), squeeze=False)
        axes = axes[:, 0]

        for idx, cfg in enumerate(defenses):
            ax = axes[idx]
            metrics = ns["defenses"][cfg["key"]]
            series = [metrics[cohort][measure_key] for cohort, _ in COHORT_ORDER]
            ax.set_title(f"{cfg['display']} ({ATTACK_DISPLAY[a]})")
            plot_box_with_stats(ax, series, show_legend=(idx == 0))
            apply_cohort_axis_labels(ax)
            ax.set_xlabel(metric)
            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()

        fig.tight_layout()
        path = out_dir / f"{dataset}_{a}_{metric}.pdf"
        fig.savefig(path)
        plt.close(fig)
        paths.append(path)

    return paths


# ---------------------------------------------------------------------------
# Numeric summary (for the paper's quantitative claims)
# ---------------------------------------------------------------------------
def write_summary(dataset, namespaces, out_dir):
    rows = []
    for name, ns in sorted(namespaces.items()):
        for cfg in DEFENSE_CONFIGS:
            metrics = ns["defenses"].get(cfg["key"])
            if metrics is None:
                continue
            for metric in MEASURE_ORDER:
                measure_key = MEASURE_KEY[metric]
                roast_vals = metrics[ROAST_COHORT][measure_key]
                base_vals = metrics[BASELINE_COHORT][measure_key]
                if not roast_vals or not base_vals:
                    continue

                roast_mean = float(np.mean(roast_vals))
                base_mean = float(np.mean(base_vals))
                abs_gain = roast_mean - base_mean
                rel_gain = (abs_gain / base_mean * 100.0) if base_mean else float("nan")

                tstat = pvalue = ""
                significant = ""
                if len(roast_vals) == len(base_vals) and len(roast_vals) >= 2:
                    t, p = ttest_rel(roast_vals, base_vals)
                    tstat = f"{t:.3f}"
                    pvalue = f"{p:.3e}"
                    significant = "yes" if p < 0.05 else "no"

                rows.append({
                    "dataset": dataset,
                    "namespace": name,
                    "risk_attack": ns["risk"],
                    "eval_attack": ns["eval"],
                    "is_cross_attack": "yes" if ns["is_cross"] else "no",
                    "defense": cfg["key"],
                    "metric": metric,
                    "roast_mean": f"{roast_mean:.4f}",
                    "baseline_mean": f"{base_mean:.4f}",
                    "abs_gain": f"{abs_gain:.4f}",
                    "rel_gain_pct": f"{rel_gain:.2f}",
                    "tstat": tstat,
                    "pvalue": pvalue,
                    "significant": significant,
                })

    if not rows:
        return None

    path = out_dir / f"{dataset}_summary.csv"
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def describe_namespaces(namespaces):
    same = sorted(n for n, ns in namespaces.items() if not ns["is_cross"])
    cross = sorted(n for n, ns in namespaces.items() if ns["is_cross"])
    parts = []
    if same:
        parts.append("same-attack: " + ", ".join(same))
    if cross:
        parts.append("cross-attack: " + ", ".join(cross))
    return "; ".join(parts) if parts else "none"


def main():
    parser = argparse.ArgumentParser(
        description="Generate publication figures for ISSRE_26 from namespaced "
                    "attack-type / cross-attack defense outputs.",
    )
    parser.add_argument(
        "--root",
        default=None,
        help="ROAST repository root (default: this script's directory).",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Output directory for figures (default: <root>/cross_attack_figures). "
             "Point at the paper's Figures dir to write there directly.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASETS,
        default=DATASETS,
        help="Datasets to process (default: all).",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=MEASURE_ORDER,
        default=MEASURE_ORDER,
        help="Metrics to plot (default: all).",
    )
    parser.add_argument(
        "--figures",
        nargs="+",
        choices=["attack_comparison", "cross_attack", "box_plots"],
        default=["attack_comparison", "cross_attack", "box_plots"],
        help="Figure families to generate (default: all).",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve() if args.root else Path(__file__).resolve().parent
    out_root = Path(args.out_dir).resolve() if args.out_dir else root / "cross_attack_figures"

    set_publication_style()

    family_dirs = {
        "attack_comparison": out_root / "attack_comparison",
        "cross_attack": out_root / "cross_attack",
        "box_plots": out_root / "box_plots",
    }
    summary_dir = out_root / "summaries"
    for fam in args.figures:
        os.makedirs(family_dirs[fam], exist_ok=True)
    os.makedirs(summary_dir, exist_ok=True)

    generated = []
    any_data = False

    for dataset in args.datasets:
        namespaces = discover_namespaces(root / dataset)
        if not namespaces:
            print(f"[{dataset}] no attack-type outputs found under "
                  f"{root / dataset / 'output'} -- skipping.")
            continue

        any_data = True
        print(f"[{dataset}] {describe_namespaces(namespaces)}")

        summary_path = write_summary(dataset, namespaces, summary_dir)
        if summary_path:
            generated.append(summary_path)

        for metric in args.metrics:
            if "attack_comparison" in args.figures:
                path = plot_attack_comparison(
                    dataset, namespaces, metric, family_dirs["attack_comparison"])
                if path:
                    generated.append(path)
            if "cross_attack" in args.figures:
                generated.extend(
                    plot_cross_attack(dataset, namespaces, metric,
                                      family_dirs["cross_attack"]))
            if "box_plots" in args.figures:
                generated.extend(
                    plot_attack_box(dataset, namespaces, metric,
                                    family_dirs["box_plots"]))

    if not any_data:
        print("\nNo namespaced outputs were found for any dataset. Run the pipeline "
              "with the attack-type / cross-attack stages first (e.g. "
              "./run_pipeline.sh --ohiot1dm_attack_type=PGD ...).")
        return

    print(f"\nGenerated {len(generated)} file(s) under {out_root}:")
    for path in generated:
        print(f"  {path.relative_to(out_root)}")


if __name__ == "__main__":
    main()
