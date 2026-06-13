import os
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import numpy as np
from pathlib import Path
import csv
import pandas as pd
import shutil
import argparse
from pyod.models.deep_svdd import DeepSVDD


def make_detector(n_features):
    # Deep SVDD (Ruff et al., 2018) via PyOD (>=2.0, PyTorch backend). contamination=0.5
    # matches the fixed setting used by the other ADs (no AD-specific tuning).
    return DeepSVDD(n_features=n_features, contamination=0.5, random_state=42)


def combine_results(output_directory):
    # === File paths (edit these) ===
    file_less = output_directory/"less"/"Results.csv"
    file_samples = output_directory/"samples"/"Results.csv"
    file_more = output_directory/"more"/"Results.csv"
    file_all = output_directory/"all"/"Results.csv"
    file_all_benign = output_directory/"all_benign"/"Results.csv"

    output_file = output_directory/"DeepSVDD_combined_results.csv"


    # === Helper: rename columns to avoid collisions ===
    def rename_columns(df, prefix):
        return df.rename(columns={
            "Accuracy": f"{prefix}_Accuracy",
            "Precision": f"{prefix}_Precision",
            "Recall": f"{prefix}_Recall",
            "F1": f"{prefix}_F1"
        })


    # === Load CSVs ===
    df_less = pd.read_csv(file_less)
    df_samples = pd.read_csv(file_samples)
    df_more = pd.read_csv(file_more)
    df_all = pd.read_csv(file_all)
    df_all_benign = pd.read_csv(file_all_benign)

    # === Rename columns ===
    df_less = rename_columns(df_less, "Less")
    df_samples = rename_columns(df_samples, "Samples")
    df_more = rename_columns(df_more, "More")
    df_all = rename_columns(df_all, "All")
    df_all_benign = rename_columns(df_all_benign, "All_Benign")

    # === Merge on CV ===
    merged = df_less.merge(df_samples, on="CV") \
                    .merge(df_more, on="CV") \
                    .merge(df_all, on="CV") \
                    .merge(df_all_benign, on="CV")

    # === Ensure correct column order ===
    ordered_cols = [
        "CV",
        "Less_Accuracy", "Less_Precision", "Less_Recall", "Less_F1",
        "Samples_Accuracy", "Samples_Precision", "Samples_Recall", "Samples_F1",
        "More_Accuracy", "More_Precision", "More_Recall", "More_F1",
        "All_Accuracy", "All_Precision", "All_Recall", "All_F1",
        "All_Benign_Accuracy", "All_Benign_Precision", "All_Benign_Recall", "All_Benign_F1",
    ]

    merged = merged[ordered_cols]


    # === Create custom headers ===
    header1 = [
        "",
        "Less Vulnerable (OE)","","","",
        "Samples Training (OE)","","","",
        "More Vulnerable (OE)","","","",
        "All Patients (OE)","","","",
        "All Patients (Benign)","","",""
    ]

    header2 = [
        "CV",
        "Accuracy","Precision","Recall","F1",
        "Accuracy","Precision","Recall","F1",
        "Accuracy","Precision","Recall","F1",
        "Accuracy","Precision","Recall","F1",
        "Accuracy","Precision","Recall","F1"
    ]


    # === Write output ===
    with open(output_file, "w") as f:
        f.write(",".join(header1) + "\n")
        f.write(",".join(header2) + "\n")
        merged.to_csv(f, index=False, header=False)



def evaluate_deepsvdd(output_directory, data_dir=None):
    os.makedirs(output_directory, exist_ok=True)
    if data_dir is None:
        data_dir = Path(__file__).resolve().parents[1] / "output" / "defense_dataset"

    ######################################################################################################################################
    # less (OE)
    os.makedirs(output_directory / "less", exist_ok=True)
    results = open(output_directory / "less" / "Results.csv", 'w')
    results.write('CV,Accuracy,Precision,Recall,F1\n')

    train = np.load(data_dir / "sepsis_train_less_0.npy")
    train_x = train[:, :-1]
    train_y = train[:, -1]

    clf = make_detector(train_x.shape[1])
    clf.fit(train_x)

    for cv in range(5):
        print('Less\tCV: ' + str(cv))
        test = np.load(data_dir / f"sepsis_test_all_{cv}.npy")
        test_x = test[:, :-1]
        test_y = test[:, -1]

        lst = clf.predict(test_x)

        results.write(str(cv) + ',' + str(accuracy_score(test_y, lst) * 100) + ',' + str(precision_score(test_y, lst)) + ',' + str(
            recall_score(test_y, lst)) + ',' + str(f1_score(test_y, lst)) + '\n')
    results.close()
    ######################################################################################################################################
    # more (OE)
    os.makedirs(output_directory/"more", exist_ok=True)
    results = open(output_directory/"more"/"Results.csv", 'w')
    results.write('CV,Accuracy,Precision,Recall,F1\n')

    train = np.load(data_dir/"sepsis_train_more_0.npy")
    train_x = train[:, :-1]
    train_y = train[:, -1]

    clf = make_detector(train_x.shape[1])
    clf.fit(train_x)

    for cv in range(5):
        print('More\tCV: ' + str(cv))
        test = np.load(data_dir / f"sepsis_test_all_{cv}.npy")
        test_x = test[:, :-1]
        test_y = test[:, -1]

        lst = clf.predict(test_x)

        results.write(str(cv) + ',' + str(accuracy_score(test_y, lst) * 100) + ',' + str(precision_score(test_y, lst)) + ',' + str(
            recall_score(test_y, lst)) + ',' + str(f1_score(test_y, lst)) + '\n')
    results.close()
    ######################################################################################################################################
    #All patients (Benign)
    os.makedirs(output_directory/"all_benign", exist_ok=True)
    results = open(output_directory/"all_benign"/"Results.csv", 'w')
    results.write('CV,Accuracy,Precision,Recall,F1\n')

    train = np.load(data_dir/"sepsis_train_all_benign_0.npy")
    train_x = train[:, :-1]
    train_y = train[:, -1]

    clf = make_detector(train_x.shape[1])
    clf.fit(train_x)

    for cv in range(5):
        print('All\tCV: ' + str(cv))
        test = np.load(data_dir / f"sepsis_test_all_{cv}.npy")
        test_x = test[:, :-1]
        test_y = test[:, -1]

        lst = clf.predict(test_x)

        results.write(str(cv) + ',' + str(accuracy_score(test_y, lst) * 100) + ',' + str(precision_score(test_y, lst)) + ',' + str(
            recall_score(test_y, lst)) + ',' + str(f1_score(test_y, lst)) + '\n')
    results.close()
    ######################################################################################################################################
    #All patients (OE)
    os.makedirs(output_directory/"all", exist_ok=True)
    results = open(output_directory/"all"/"Results.csv", 'w')
    results.write('CV,Accuracy,Precision,Recall,F1\n')

    train = np.load(data_dir/"sepsis_train_all_0.npy")
    train_x = train[:, :-1]
    train_y = train[:, -1]

    clf = make_detector(train_x.shape[1])
    clf.fit(train_x)

    for cv in range(5):
        print('All\tCV: ' + str(cv))
        test = np.load(data_dir / f"sepsis_test_all_{cv}.npy")
        test_x = test[:, :-1]
        test_y = test[:, -1]

        lst = clf.predict(test_x)

        results.write(str(cv) + ',' + str(accuracy_score(test_y, lst) * 100) + ',' + str(precision_score(test_y, lst)) + ',' + str(
            recall_score(test_y, lst)) + ',' + str(f1_score(test_y, lst)) + '\n')
    results.close()
    ######################################################################################################################################
    # Samples (OE)
    os.makedirs(output_directory/"samples", exist_ok=True)
    results = open(output_directory/"samples"/"Results.csv", 'w')
    results.write('CV,Accuracy,Precision,Recall,F1\n')

    for run in range(5):
        train = np.load(data_dir/f"sepsis_train_samples_{run}.npy")
        train_x = train[:, :-1]
        train_y = train[:, -1]

        clf = make_detector(train_x.shape[1])
        clf.fit(train_x)
        Accuracy = []
        Precision = []
        Recall = []
        F1 = []
        for cv in range(5):
            print('Samples '+str(run)+'\tCV: ' + str(cv))
            test = np.load(data_dir / f"sepsis_test_all_{cv}.npy")
            test_x = test[:, :-1]
            test_y = test[:, -1]

            lst = clf.predict(test_x)

            Accuracy.append(accuracy_score(test_y, lst) * 100)
            Precision.append(precision_score(test_y, lst) * 100)
            Recall.append(recall_score(test_y, lst) * 100)
            F1.append(f1_score(test_y, lst) * 100)

        results.write(f'{run},' + str(np.average(np.array(Accuracy))) + ',' + str(np.average(np.array(Precision))) + ',' + str(np.average(np.array(Recall))) + ',' + str(np.average(np.array(F1))) + '\n')
    results.close()
    ######################################################################################################################################

    combine_results(output_directory)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run PhysioNetCinC script: python evaluate_deepsvdd.py <output_directory>",
        epilog="Example: python evaluate_deepsvdd.py output"
    )
    parser.add_argument("out_dir", nargs="?", default="output/defense_output/DeepSVDD", help="Output directory")
    parser.add_argument("--data_dir", default="output/defense_dataset", help="Directory containing generated defense dataset .npy files")

    args = parser.parse_args()

    dataset_root = Path(__file__).resolve().parents[1]

    output_directory = dataset_root / args.out_dir
    data_directory = dataset_root / args.data_dir

    evaluate_deepsvdd(output_directory, data_directory)
