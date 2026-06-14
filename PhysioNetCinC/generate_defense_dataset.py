import joblib
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import shutil
import argparse
from sklearn.model_selection import KFold, train_test_split


def write_per_cluster_test_dirs(out_dir, all_mixed_data, patient_id_to_indices, cluster_to_patients):
    """Emit sibling <out_dir>_<tag> dirs for per-cluster TEST evaluation.

    Each sibling dir contains a copy of every shared training file from out_dir plus
    cluster-restricted ``sepsis_test_all_{cv}.npy`` files (a 5-fold split over only that
    cluster's patients). Pointing the unchanged evaluate_*.py at
    ``--data_dir=<out_dir>_<tag>`` yields the same trained detectors evaluated on a
    single cluster's test patients.
    """
    out_dir = Path(out_dir)
    train_files = sorted(out_dir.glob("*_train_*.npy"))

    def rows_for(pids):
        rows = []
        for pid in pids:
            rows.extend(patient_id_to_indices.get(pid, []))
        return np.asarray(rows, dtype=int)

    for tag, patient_ids in cluster_to_patients.items():
        cdir = Path(str(out_dir) + f"_{tag}")
        os.makedirs(cdir, exist_ok=True)

        for f in train_files:
            shutil.copy(f, cdir / f.name)

        patient_ids = np.asarray(list(patient_ids))
        n_splits = min(5, len(patient_ids))
        cv = 0
        if n_splits >= 2:
            for _, test_idx in KFold(n_splits=n_splits).split(patient_ids):
                np.save(cdir / f"sepsis_test_all_{cv}.npy",
                        all_mixed_data[rows_for(patient_ids[test_idx])])
                cv += 1
        # Pad remaining folds (cluster smaller than 5 patients) with the full cluster
        # so the evaluate loop's `for cv in range(5)` always finds a file.
        while cv < 5:
            np.save(cdir / f"sepsis_test_all_{cv}.npy", all_mixed_data[rows_for(patient_ids)])
            cv += 1

# Create mixed benign/adversarial data for less, more, samples
def create_mixed_data(data_benign, data_adversarial):
    # Feature columns (all except PatientID and Adversarial)
    feature_columns = [col for col in data_benign.columns if col != 'PatientID']

    benign_array = data_benign[feature_columns].to_numpy()
    adversarial_array = data_adversarial[feature_columns].to_numpy()

    # Find rows that differ
    differs = (benign_array != adversarial_array).any(axis=1)

    # Random selection for differing rows
    rand_selection = np.random.randint(0, 2, size=len(differs)) == 0

    # Create mixed data
    mixed_array = np.where(
        differs[:, np.newaxis],
        np.where(rand_selection[:, np.newaxis], benign_array, adversarial_array),
        benign_array
    )

    # Add label column (0 for benign, 1 for adversarial selected rows)
    labels = (differs & ~rand_selection).astype(int)
    return np.hstack([mixed_array, labels[:, np.newaxis]])

def get_benign_and_adversarial_data(data_dir):
    training_sets = ["training_setA", "training_setB"]

    # get feature names
    data_psv_dir = Path(__file__).resolve().parent / "input"
    psv_candidates = sorted(data_psv_dir.glob("*/*.psv"))
    if not psv_candidates:
        raise FileNotFoundError(
            f"Could not find *.psv in {data_psv_dir}"
        )
    psv_path = psv_candidates[0]

    with open(psv_path) as file_obj:
        header = file_obj.readline().strip()
        features = np.array(header.split('|')[:-1])

    benign_data_df = pd.DataFrame()
    adversarial_data_df = pd.DataFrame()

    for training_set in training_sets:
        benign_data_path = data_dir / training_set / "Data" / "Benign"
        adversarial_data_path = data_dir / training_set / "Data" / "Adversarial"

        for data_file in tqdm(Path(benign_data_path).glob("*.pkl")):
            benign_data = joblib.load(benign_data_path / data_file.name)
            adversarial_data = joblib.load(adversarial_data_path / data_file.name)

            benign_patient_df = pd.DataFrame(benign_data)
            benign_patient_df.columns = features
            benign_patient_df.insert(loc=0, column='PatientID', value=data_file.name[:-4])
            benign_data_df = pd.concat([benign_data_df, benign_patient_df], axis=0, ignore_index=True)

            adversarial_patient_df = pd.DataFrame(adversarial_data)
            adversarial_patient_df.columns = features
            adversarial_patient_df.insert(loc=0, column='PatientID', value=data_file.name[:-4])
            adversarial_data_df = pd.concat([adversarial_data_df, adversarial_patient_df], axis=0, ignore_index=True)


    # pre-processing
    selected_features = ['PatientID', 'HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'Age', 'Gender']
    benign_data_df = benign_data_df.ffill()
    benign_data_df = benign_data_df.fillna(0)

    benign_data = benign_data_df.loc[:, selected_features]
    adversarial_data = adversarial_data_df.loc[:, selected_features]

    # Compare row by row
    row_diff = (benign_data != adversarial_data).any(axis=1)
    adversarial_data.insert(len(adversarial_data.columns), "Adversarial", row_diff)

    joblib.dump(benign_data, data_dir / "benign_data.pkl")
    joblib.dump(adversarial_data, data_dir / "adversarial_data.pkl")


def normalize_patient_id(value):
    """Normalize patient identifiers to a comparable string format."""
    return str(value).strip()


def map_cluster_ids_to_patient_ids(cluster_ids, valid_patient_ids, data_dir):
    """
    Map cluster IDs to real patient IDs.

    Cluster outputs may contain integer indices instead of true patient identifiers.
    In that case, use output/risk_profiles.pkl as the index->PatientID lookup.
    """
    normalized_valid_ids = {normalize_patient_id(pid) for pid in valid_patient_ids}
    normalized_cluster_ids = [normalize_patient_id(pid) for pid in cluster_ids]

    # Fast path: cluster IDs already match patient IDs.
    direct_matches = sum(pid in normalized_valid_ids for pid in normalized_cluster_ids)
    if direct_matches > 0:
        return normalized_cluster_ids

    risk_profiles_path = data_dir / 'risk_profiles.pkl'
    if not risk_profiles_path.exists():
        return normalized_cluster_ids

    risk_profiles = joblib.load(risk_profiles_path)
    index_to_patient_id = [normalize_patient_id(record[0]) for record in risk_profiles]

    mapped_ids = []
    for pid in cluster_ids:
        try:
            idx = int(pid)
        except (TypeError, ValueError):
            mapped_ids.append(normalize_patient_id(pid))
            continue

        if 0 <= idx < len(index_to_patient_id):
            mapped_ids.append(index_to_patient_id[idx])
        else:
            mapped_ids.append(normalize_patient_id(pid))

    return mapped_ids


def generate_defense_dataset(cluster_dir, out_dir, data_dir=None):
    os.makedirs(out_dir, exist_ok=True)
    if data_dir is None:
        data_dir = Path(__file__).resolve().parent / "output"

    file_path_benign = Path(data_dir / 'benign_data.pkl')
    file_path_adversarial = Path(data_dir / 'adversarial_data.pkl')

    if not file_path_benign.exists() or not file_path_adversarial.exists():
        get_benign_and_adversarial_data(data_dir)

    AllPatientsDataBenign = joblib.load(file_path_benign)
    AllPatientsDataAdversarial = joblib.load(file_path_adversarial)

    # Ensure IDs are comparable regardless of original dtype.
    AllPatientsDataBenign['PatientID'] = AllPatientsDataBenign['PatientID'].map(normalize_patient_id)
    AllPatientsDataAdversarial['PatientID'] = AllPatientsDataAdversarial['PatientID'].map(normalize_patient_id)

    AllPatientIDs = joblib.load(cluster_dir / 'AllPatientIDs.pkl')
    MoreVulnerablePatientIDs = joblib.load(cluster_dir / 'MoreVulnerablePatientIDs.pkl')
    LessVulnerablePatientIDs = joblib.load(cluster_dir / 'LessVulnerablePatientIDs.pkl')

    valid_patient_ids = set(AllPatientsDataBenign['PatientID'].values)
    AllPatientIDs = map_cluster_ids_to_patient_ids(AllPatientIDs, valid_patient_ids, data_dir)
    MoreVulnerablePatientIDs = map_cluster_ids_to_patient_ids(MoreVulnerablePatientIDs, valid_patient_ids, data_dir)
    LessVulnerablePatientIDs = map_cluster_ids_to_patient_ids(LessVulnerablePatientIDs, valid_patient_ids, data_dir)

    # Create mixed data with random selection
    all_mixed_data = create_mixed_data(AllPatientsDataBenign, AllPatientsDataAdversarial)

    # Create a mapping of PatientID to row indices
    patient_id_to_indices = {}
    for idx, patient_id in enumerate(AllPatientsDataBenign['PatientID'].values):
        if patient_id not in patient_id_to_indices:
            patient_id_to_indices[patient_id] = []
        patient_id_to_indices[patient_id].append(idx)

    # Get indices for less and more vulnerable patients
    less_indices = []
    more_indices = []
    for patient_id in LessVulnerablePatientIDs:
        less_indices.extend(patient_id_to_indices.get(patient_id, []))
    for patient_id in MoreVulnerablePatientIDs:
        more_indices.extend(patient_id_to_indices.get(patient_id, []))

    if len(less_indices) == 0 or len(more_indices) == 0:
        raise ValueError(
            "Could not map vulnerable patient IDs to data rows. "
            "Check whether cluster IDs and PatientID formats are aligned."
        )

    # Save less and more
    np.save(out_dir / 'sepsis_train_less_0.npy', all_mixed_data[less_indices])
    np.save(out_dir / 'sepsis_train_more_0.npy', all_mixed_data[more_indices])

    # all uses mixed data (benign and adversarial)
    all_indices = list(range(len(AllPatientsDataBenign)))
    np.save(out_dir / 'sepsis_train_all_0.npy', all_mixed_data[all_indices])
    
    # also save benign-only version for reference
    benign_all = AllPatientsDataBenign.drop(columns=['PatientID']).to_numpy().astype(float)
    benign_labels = np.zeros((benign_all.shape[0], 1), dtype=float)
    benign_all_with_label = np.hstack([benign_all, benign_labels])
    np.save(out_dir / 'sepsis_train_all_benign_0.npy', benign_all_with_label)

    for run in range(5):
        split = train_test_split(AllPatientIDs, train_size=len(LessVulnerablePatientIDs))
        train_indices_patients = split[0]

        # Get row indices for these patients
        sample_indices = []
        for patient_id in train_indices_patients:
            sample_indices.extend(patient_id_to_indices.get(patient_id, []))

        np.save(out_dir / f'sepsis_train_samples_{run}.npy', all_mixed_data[sample_indices])


    cv=0
    kf = KFold(n_splits=5)
    for train_indices, test_indices in kf.split(AllPatientIDs):
        test_patient_ids = [AllPatientIDs[i] for i in test_indices]
        test_mask = AllPatientsDataBenign['PatientID'].isin(test_patient_ids)
        test_indices_rows = np.where(test_mask.values)[0]
        np.save(out_dir / f'sepsis_test_all_{cv}.npy', all_mixed_data[test_indices_rows])
        cv+=1

    # Per-cluster test sets for less/more-vulnerable test-side breakdown.
    write_per_cluster_test_dirs(out_dir, all_mixed_data, patient_id_to_indices, {
        "less": LessVulnerablePatientIDs,
        "more": MoreVulnerablePatientIDs,
    })


if __name__ == '__main__':
	parser = argparse.ArgumentParser(
		description="Run PhysioNetCinC script: python generate_defense_dataset.py <cluster_output> <output_directory>",
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