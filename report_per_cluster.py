"""Per-cluster (less- vs more-vulnerable test patients) performance reporting.

Standard defense results average each metric over the whole test population. This
script additionally stratifies the *test* side by vulnerability cluster, so for every
training strategy (Less Vulnerable OE, Random, More Vulnerable OE, All OE, All Benign)
we report the metric separately on less-vulnerable and more-vulnerable test patients.

Two data shapes are consumed, unified here:

  * OhioT1DM -- post-hoc, no re-run. Each ``defense_output/<AD>/<AD>_combined_results.csv``
    already has one row per patient (ordered 2018_0..2020_5 = global index 0..11). We
    load ``cluster_output/{Less,More}VulnerablePatientIDs.pkl`` and split the per-patient
    values by cluster index.
  * MIMIC / PhysioNet -- the per-cluster test sets were evaluated separately into
    ``defense_output_less/`` and ``defense_output_more/`` (see generate_defense_dataset.py
    + run_pipeline.sh). We read those directly; the per-fold mean is the cluster value.

Outputs (default <root>/per_cluster_figures/):
  summaries/<dataset>_per_cluster.csv  -- one row per dataset/namespace/detector/metric/
                                          train_strategy/test_cluster (+ overall).
  plots/<dataset>_<detector>_<metric>_per_cluster.pdf -- grouped bars, x = training
                                          strategy, bars = {less-vuln, more-vuln} test.

Example:
    python report_per_cluster.py
    python report_per_cluster.py --datasets OhioT1DM --metrics Recall
"""

import argparse
import csv
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import joblib
import matplotlib.pyplot as plt
import numpy as np

from plot_defense_results import (
    DEFENSE_CONFIGS,
    MEASURE_ORDER,
    MEASURE_KEY,
    COHORT_ORDER,
    parse_results_csv,
    find_results_csv,
)
from plot_cross_attack_results import set_publication_style


DATASETS = ["OhioT1DM", "MIMIC", "PhysioNetCinC"]

# Test-side clusters and their display labels.
TEST_CLUSTERS = [("less", "Less-Vulnerable Test"), ("more", "More-Vulnerable Test")]
CLUSTER_COLOR = {"less": "#2c7fb8", "more": "#d95f0e"}

# OhioT1DM stores per-patient rows; the cluster pkls are indices into that row order.
OHIOT1DM = "OhioT1DM"


def risk_attack_of(namespace):
    """The risk-profiling attack a namespace was clustered under (handles cross-attack)."""
    return namespace.split("_to_")[0]


def discover_namespaces(dataset_root):
    """Namespaces under <dataset>/output that carry defense results."""
    out = dataset_root / "output"
    found = []
    if not out.is_dir():
        return found
    for ns in sorted(out.iterdir()):
        if not ns.is_dir():
            continue
        if (ns / "defense_output").is_dir() or (ns / "defense_output_less").is_dir():
            found.append(ns.name)
    return found


def _mean(values):
    return float(np.mean(values)) if len(values) else float("nan")


def per_cluster_ohiot1dm(dataset_root, namespace):
    """{detector: {cohort: {metric: {'less','more','overall'}}}} from per-patient CSVs."""
    defense_output = dataset_root / "output" / namespace / "defense_output"
    cluster_dir = dataset_root / "output" / risk_attack_of(namespace) / "cluster_output"
    try:
        less_ids = np.asarray(joblib.load(cluster_dir / "LessVulnerablePatientIDs.pkl"), dtype=int)
        more_ids = np.asarray(joblib.load(cluster_dir / "MoreVulnerablePatientIDs.pkl"), dtype=int)
    except FileNotFoundError:
        print(f"[OhioT1DM/{namespace}] cluster pkls not found under {cluster_dir} -- skipping.")
        return {}

    result = {}
    for cfg in DEFENSE_CONFIGS:
        csv_path, _ = find_results_csv(defense_output, cfg["aliases"])
        if csv_path is None:
            continue
        metrics = parse_results_csv(csv_path)
        if not metrics["all_patients_benign"]["accuracy"]:
            continue
        det = {}
        for cohort, _label in COHORT_ORDER:
            det[cohort] = {}
            for metric in MEASURE_ORDER:
                vals = np.asarray(metrics[cohort][MEASURE_KEY[metric]], dtype=float)
                n = len(vals)
                li = less_ids[less_ids < n]
                mi = more_ids[more_ids < n]
                det[cohort][metric] = {
                    "less": _mean(vals[li]),
                    "more": _mean(vals[mi]),
                    "overall": _mean(vals),
                }
        result[cfg["key"]] = det
    return result


def per_cluster_cv(dataset_root, namespace):
    """{detector: {cohort: {metric: {'less','more','overall'}}}} from per-cluster eval dirs."""
    ns_dir = dataset_root / "output" / namespace
    dirs = {
        "less": ns_dir / "defense_output_less",
        "more": ns_dir / "defense_output_more",
        "overall": ns_dir / "defense_output",
    }
    if not dirs["less"].is_dir() or not dirs["more"].is_dir():
        print(f"[{dataset_root.name}/{namespace}] per-cluster eval dirs missing "
              f"(run --per_cluster_report eval first) -- skipping.")
        return {}

    result = {}
    for cfg in DEFENSE_CONFIGS:
        parsed = {}
        for key, d in dirs.items():
            csv_path, _ = find_results_csv(d, cfg["aliases"])
            parsed[key] = parse_results_csv(csv_path) if csv_path else None
        if parsed["less"] is None or parsed["more"] is None:
            continue
        det = {}
        for cohort, _label in COHORT_ORDER:
            det[cohort] = {}
            for metric in MEASURE_ORDER:
                mk = MEASURE_KEY[metric]
                overall = parsed["overall"]
                det[cohort][metric] = {
                    "less": _mean(parsed["less"][cohort][mk]),
                    "more": _mean(parsed["more"][cohort][mk]),
                    "overall": _mean(overall[cohort][mk]) if overall else float("nan"),
                }
        result[cfg["key"]] = det
    return result


def write_summary(dataset, namespace, per_cluster, metrics, out_dir):
    rows = []
    label_of = dict(COHORT_ORDER)
    for det_key, cohorts in per_cluster.items():
        for cohort, _label in COHORT_ORDER:
            for metric in metrics:
                cell = cohorts[cohort][metric]
                for tag, _disp in TEST_CLUSTERS:
                    rows.append({
                        "dataset": dataset,
                        "namespace": namespace,
                        "detector": det_key,
                        "metric": metric,
                        "train_strategy": label_of[cohort],
                        "test_cluster": tag,
                        "value": f"{cell[tag]:.4f}",
                        "overall": f"{cell['overall']:.4f}",
                    })
    if not rows:
        return None
    os.makedirs(out_dir, exist_ok=True)
    path = out_dir / f"{dataset}_{namespace}_per_cluster.csv"
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def plot_per_cluster(dataset, namespace, per_cluster, metrics, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    display_of = {c["key"]: c["display"] for c in DEFENSE_CONFIGS}
    cohorts = [c for c, _ in COHORT_ORDER]
    labels = [lbl for _, lbl in COHORT_ORDER]
    paths = []

    for det_key, cohort_data in per_cluster.items():
        for metric in metrics:
            x = np.arange(len(cohorts))
            width = 0.38
            fig, ax = plt.subplots(figsize=(8, 3.2))
            for i, (tag, disp) in enumerate(TEST_CLUSTERS):
                vals = [cohort_data[c][metric][tag] for c in cohorts]
                ax.bar(x + (i - 0.5) * width, vals, width, label=disp,
                       color=CLUSTER_COLOR[tag])
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=20, ha="right")
            ax.set_ylabel(metric)
            ax.set_title(f"{display_of[det_key]} -- {dataset} ({namespace})", loc="left")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.28), ncol=2, frameon=False)
            fig.tight_layout()
            path = out_dir / f"{dataset}_{namespace}_{det_key}_{metric}_per_cluster.pdf"
            fig.savefig(path)
            plt.close(fig)
            paths.append(path)
    return paths


def main():
    parser = argparse.ArgumentParser(
        description="Per-cluster (less/more-vulnerable test) defense reporting.")
    parser.add_argument("--root", default=None,
                        help="ROAST repo root (default: this script's directory).")
    parser.add_argument("--out_dir", default=None,
                        help="Output dir (default: <root>/per_cluster_figures).")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--metrics", nargs="+", choices=MEASURE_ORDER, default=MEASURE_ORDER)
    args = parser.parse_args()

    root = Path(args.root).resolve() if args.root else Path(__file__).resolve().parent
    out_root = Path(args.out_dir).resolve() if args.out_dir else root / "per_cluster_figures"
    set_publication_style()

    generated = []
    any_data = False
    for dataset in args.datasets:
        dataset_root = root / dataset
        namespaces = discover_namespaces(dataset_root)
        if not namespaces:
            print(f"[{dataset}] no defense outputs found -- skipping.")
            continue
        for namespace in namespaces:
            if dataset == OHIOT1DM:
                per_cluster = per_cluster_ohiot1dm(dataset_root, namespace)
            else:
                per_cluster = per_cluster_cv(dataset_root, namespace)
            if not per_cluster:
                continue
            any_data = True
            print(f"[{dataset}/{namespace}] detectors: {', '.join(per_cluster)}")
            summary = write_summary(dataset, namespace, per_cluster, args.metrics, out_root / "summaries")
            if summary:
                generated.append(summary)
            generated.extend(plot_per_cluster(dataset, namespace, per_cluster, args.metrics, out_root / "plots"))

    if not any_data:
        print("\nNo per-cluster data found. For OhioT1DM, run the defense eval first; for "
              "MIMIC/PhysioNet, run the per-cluster eval (run_pipeline.sh --per_cluster_report=true).")
        return

    print(f"\nGenerated {len(generated)} file(s) under {out_root}:")
    for path in generated:
        print(f"  {path.relative_to(out_root)}")


if __name__ == "__main__":
    main()
