import joblib
import numpy as np
import os
import shutil
import argparse
from pathlib import Path
from sklearn.model_selection import KFold, train_test_split
import random


def write_per_cluster_test_dirs(out_dir, data, cluster_to_patients):
    """Emit sibling <out_dir>_<tag> dirs for per-cluster TEST evaluation.

    Each sibling dir contains a copy of every shared training file from out_dir
    plus cluster-restricted ``*_test_all_{cv}.npy`` files (a 5-fold split over only
    that cluster's patients). Because the train files are identical to the main run,
    pointing the unchanged evaluate_*.py at ``--data_dir=<out_dir>_<tag>`` yields the
    same trained detectors evaluated on a single cluster's test patients.
    """
    out_dir = Path(out_dir)
    nf = data.shape[2]
    train_files = sorted(out_dir.glob("*_train_*.npy"))

    for tag, patient_ids in cluster_to_patients.items():
        cdir = Path(str(out_dir) + f"_{tag}")
        os.makedirs(cdir, exist_ok=True)

        # Reuse the exact training files so only the test set differs per cluster.
        for f in train_files:
            shutil.copy(f, cdir / f.name)

        patient_ids = np.asarray(list(patient_ids))
        n_splits = min(5, len(patient_ids))
        cv = 0
        if n_splits >= 2:
            for _, test_idx in KFold(n_splits=n_splits).split(patient_ids):
                np.save(cdir / f"mimic_test_all_{cv}.npy",
                        data[patient_ids[test_idx]].reshape(-1, nf))
                cv += 1
        # Pad any remaining folds (cluster smaller than 5 patients) with the full
        # cluster so the evaluate loop's `for cv in range(5)` always finds a file.
        while cv < 5:
            np.save(cdir / f"mimic_test_all_{cv}.npy", data[patient_ids].reshape(-1, nf))
            cv += 1

def create_mixed_data(benign_data, adversarial_data, features):
    """Create mixed benign/adversarial data with random selection (vectorized)."""
    # Extract features and reshape for comparison
    benign_selected = benign_data[:, :, features]  # Shape: (n_samples, 72, n_features)
    adversarial_selected = adversarial_data[:, :, features]

    # Check where data differs (across samples and timesteps)
    differs = (benign_selected != adversarial_selected).any(axis=2)  # Shape: (n_samples, 72)

    # Random selection for differing entries
    rand_selection = np.random.randint(0, 2, size=differs.shape) == 0

    # Create mixed data using where
    mixed = np.where(
        differs[:, :, np.newaxis],
        np.where(rand_selection[:, :, np.newaxis], benign_selected, adversarial_selected),
        benign_selected
    )

    # Add label column
    labels = (differs & ~rand_selection)[:, :, np.newaxis].astype(int)
    return np.concatenate([mixed, labels], axis=2)

def generate_defense_dataset(cluster_dir, out_dir, data_dir=None):
    os.makedirs(out_dir, exist_ok=True)
    if data_dir is None:
        data_dir = Path(__file__).resolve().parent / "output"

    features = [143, 83, 89, 118, 183, 114, 177, 47, 75, 100]
    adversarial_data = joblib.load(data_dir / 'adversarial_data.pkl').reshape(-1,72, 432)
    benign_data = joblib.load(data_dir / 'benign_data.pkl').reshape(-1,72, 432)

    # Create mixed data with random selection (already shape (n_samples, 72, n_features+1))
    data = create_mixed_data(benign_data, adversarial_data, features)

    # Create benign-only data for 'all' (vectorized)
    benign_selected = benign_data[:, :, features]
    benign_labels = np.zeros((benign_selected.shape[0], benign_selected.shape[1], 1), dtype=int)
    data_benign = np.concatenate([benign_selected, benign_labels], axis=2)

    LessVulnerablePatientIDs = joblib.load(cluster_dir / 'LessVulnerablePatientIDs.pkl')
    MoreVulnerablePatientIDs = joblib.load(cluster_dir / 'MoreVulnerablePatientIDs.pkl')
    AllPatientIDs = joblib.load(cluster_dir / 'AllPatientIDs.pkl')

    np.save(out_dir / 'mimic_train_less_0.npy', data[LessVulnerablePatientIDs].reshape(-1,data.shape[2]))
    np.save(out_dir / 'mimic_train_more_0.npy', data[MoreVulnerablePatientIDs].reshape(-1,data.shape[2]))
    np.save(out_dir / 'mimic_train_all_0.npy', data[AllPatientIDs].reshape(-1,data.shape[2]))
    np.save(out_dir / 'mimic_train_all_benign_0.npy', data_benign.reshape(-1,data_benign.shape[2]))

    for run in range(5):

        split = train_test_split(AllPatientIDs, train_size=len(LessVulnerablePatientIDs))
        train_indices = split[0]
        # test_indices = split[1][:math.floor(0.2 * len(AllPatientIDs))]

        np.save(out_dir / f'mimic_train_samples_{run}.npy', data[train_indices].reshape(-1, data.shape[2]))


    cv=0
    kf = KFold(n_splits=5)
    for train_indices, test_indices in kf.split(AllPatientIDs):
        np.save(out_dir / f'mimic_test_all_{cv}.npy', data[test_indices].reshape(-1, data.shape[2]))
        cv+=1

    # Per-cluster test sets for less/more-vulnerable test-side breakdown.
    write_per_cluster_test_dirs(out_dir, data, {
        "less": LessVulnerablePatientIDs,
        "more": MoreVulnerablePatientIDs,
    })


if __name__ == '__main__':
	parser = argparse.ArgumentParser(
		description="Run MIMIC script: python generate_defense_dataset.py <cluster_output> <output_directory>",
		epilog="Example: python generate_defense_dataset.py output/cluster_output output/defense_dataset"
	)
	parser.add_argument("cluster_dir", nargs="?", default="output/cluster_output", help="Directory containing cluster output")
	parser.add_argument("out_dir", nargs="?", default="output/defense_dataset", help="Output directory")
	parser.add_argument("--data_dir", default="output", help="Directory containing benign/adversarial model outputs")

	args = parser.parse_args()

	SCRIPT_DIR = Path(__file__).resolve().parent

	cluster_directory = SCRIPT_DIR / args.cluster_dir
	output_directory = SCRIPT_DIR / args.out_dir
	data_directory = SCRIPT_DIR / args.data_dir

	generate_defense_dataset(cluster_directory, output_directory, data_directory)