#!/usr/bin/env python3
"""Paper-faithful two-cluster DTW clustering for MIMIC risk profiles."""

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from roast_core import cluster_risk_profiles


def hierarchical_cluster(risk_profiles_directory, output_directory):
    profiles = joblib.load(Path(risk_profiles_directory) / "risk_profiles.pkl")
    records = [(f"p{index}", np.asarray(profile).reshape(-1)) for index, profile in enumerate(profiles)]
    result = cluster_risk_profiles(records)
    index = {label: position for position, label in enumerate(result.labels)}
    less = np.asarray([index[label] for label in result.less_vulnerable], dtype=int)
    more = np.asarray([index[label] for label in result.more_vulnerable], dtype=int)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(np.arange(len(records)), output_directory / "AllPatientIDs.pkl")
    joblib.dump(less, output_directory / "LessVulnerablePatientIDs.pkl")
    joblib.dump(more, output_directory / "MoreVulnerablePatientIDs.pkl")
    np.save(output_directory / "distance_matrix.npy", result.distance_matrix)
    plt.figure(figsize=(10, 6))
    dendrogram(result.linkage_matrix, labels=result.labels)
    plt.xlabel("Patient Index")
    plt.ylabel("DTW Distance")
    plt.tight_layout()
    plt.savefig(output_directory / "Dendrogram.pdf")
    plt.close()
    (output_directory / "cluster_summary.json").write_text(json.dumps({
        "profiling_contract": "roast-v2-standardized-absolute-condensed-dtw-maxclust2-lower-median",
        "less_vulnerable": list(result.less_vulnerable),
        "more_vulnerable": list(result.more_vulnerable),
        "cluster_medians": {str(k): v for k, v in result.cluster_medians.items()},
    }, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("risk_profiles_dir", nargs="?", default="output")
    parser.add_argument("out_dir", nargs="?", default="output/cluster_output")
    args = parser.parse_args()
    base = Path(__file__).resolve().parent
    hierarchical_cluster(base / args.risk_profiles_dir, base / args.out_dir)
