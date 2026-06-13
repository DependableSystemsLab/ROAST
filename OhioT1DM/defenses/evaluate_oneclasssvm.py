import os
from sklearn.svm import OneClassSVM
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import numpy as np
from pathlib import Path
# from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import RobustScaler
import csv
import pandas as pd
import shutil
import argparse


def combine_results(output_directory):
    # === File paths (edit these) ===
    file_less = output_directory/"less"/"Results.csv"
    file_samples = output_directory/"samples"/"Results.csv"
    file_more = output_directory/"more"/"Results.csv"
    file_all = output_directory/"all"/"Results.csv"
    file_all_benign = output_directory/"all_benign"/"Results.csv"

    output_file = output_directory/"OneClassSVM_combined_results.csv"


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

    # === Merge on Patient ===
    merged = df_less.merge(df_samples, on="Patient") \
                    .merge(df_more, on="Patient") \
                    .merge(df_all, on="Patient") \
                    .merge(df_all_benign, on="Patient")

    # === Ensure correct column order ===
    ordered_cols = [
        "Patient",
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
        "Patient",
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



def evaluate_oneclasssvm(output_directory, data_dir=None):
    os.makedirs(output_directory, exist_ok=True)
    if data_dir is None:
        data_dir = Path(__file__).resolve().parents[1] / "output" / "defense_dataset"

    clf = OneClassSVM(kernel='rbf',  gamma='scale', coef0=0, tol=0.001, nu=0.5)
    # scaler = StandardScaler()
    # scaler = RobustScaler()
    # clf = OneClassSVM(kernel='linear',  gamma='auto', coef0=0, tol=1e-5)
    ######################################################################################################################################
    # less (OE)
    os.makedirs(output_directory / "less", exist_ok=True)
    results = open(output_directory / "less" / "Results.csv", 'w')
    results.write('Patient,Accuracy,Precision,Recall,F1\n')

    train = np.load(data_dir / "ohiot1dm_train_less_0.npy")
    train_x = train[:, :-1]
    train_y = train[:, -1]

    clf.fit(train_x)
    # clf.fit(scaler.fit_transform(train_x))

    for year in [2018, 2020]:
        for patient in range(6):
            print("Less\tPatient " + str(year) + "_" + str(patient))
            test = np.load(data_dir / f"ohiot1dm_test_{year}_{patient}.npy")
            test_x = test[:, :-1]
            test_y = test[:, -1]

            results.write(str(year) + '_' + str(patient) + ',')

            lst = clf.predict(test_x)
            # lst = clf.predict(scaler.transform(test_x))
            lst[lst == 1] = 0
            lst[lst == -1] = 1

            results.write(str(accuracy_score(test_y, lst) * 100) + ',' + str(precision_score(test_y, lst)) + ',' + str(
                recall_score(test_y, lst)) + ',' + str(f1_score(test_y, lst)) + '\n')
    results.close()
    ######################################################################################################################################
    # more (OE)
    os.makedirs(output_directory/"more", exist_ok=True)
    results = open(output_directory/"more"/"Results.csv", 'w')
    results.write('Patient,Accuracy,Precision,Recall,F1\n')

    train = np.load(data_dir/"ohiot1dm_train_more_0.npy")
    train_x = train[:, :-1]
    train_y = train[:, -1]

    clf.fit(train_x)
    # clf.fit(scaler.fit_transform(train_x))

    for year in [2018, 2020]:
        for patient in range(6):
            print("More\tPatient " + str(year) + "_" + str(patient))
            test = np.load(data_dir / f"ohiot1dm_test_{year}_{patient}.npy")
            test_x = test[:, :-1]
            test_y = test[:, -1]

            results.write(str(year) + '_' + str(patient) + ',')

            lst = clf.predict(test_x)
            # lst = clf.predict(scaler.transform(test_x))
            lst[lst == 1] = 0
            lst[lst == -1] = 1

            results.write(str(accuracy_score(test_y, lst) * 100) + ',' + str(precision_score(test_y, lst)) + ',' + str(
                recall_score(test_y, lst)) + ',' + str(f1_score(test_y, lst)) + '\n')
    results.close()
    ######################################################################################################################################
    #All patients (Benign)
    os.makedirs(output_directory/"all_benign", exist_ok=True)
    results = open(output_directory/"all_benign"/"Results.csv", 'w')
    results.write('Patient,Accuracy,Precision,Recall,F1\n')

    train = np.load(data_dir/"ohiot1dm_train_all_benign_0.npy")
    train_x = train[:, :-1]
    train_y = train[:, -1]

    clf.fit(train_x)
    # clf.fit(scaler.fit_transform(train_x))

    for year in [2018, 2020]:
        for patient in range(6):
            print("All\tPatient "+str(year)+"_"+str(patient))
            test = np.load(data_dir/f"ohiot1dm_test_{year}_{patient}.npy")
            test_x = test[:, :-1]
            test_y = test[:, -1]

            results.write(str(year)+'_'+str(patient)+',')

            lst = clf.predict(test_x)
            # lst = clf.predict(scaler.transform(test_x))
            lst[lst == 1] = 0
            lst[lst == -1] = 1

            results.write(str(accuracy_score(test_y, lst)*100) + ',' + str(precision_score(test_y, lst)) + ',' + str(recall_score(test_y, lst)) + ',' + str(f1_score(test_y, lst)) + '\n')
    results.close()
    ######################################################################################################################################
    #All patients (OE)
    os.makedirs(output_directory/"all", exist_ok=True)
    results = open(output_directory/"all"/"Results.csv", 'w')
    results.write('Patient,Accuracy,Precision,Recall,F1\n')

    train = np.load(data_dir/"ohiot1dm_train_all_0.npy")
    train_x = train[:, :-1]
    train_y = train[:, -1]

    clf.fit(train_x)
    # clf.fit(scaler.fit_transform(train_x))

    for year in [2018, 2020]:
        for patient in range(6):
            print("All\tPatient "+str(year)+"_"+str(patient))
            test = np.load(data_dir/f"ohiot1dm_test_{year}_{patient}.npy")
            test_x = test[:, :-1]
            test_y = test[:, -1]

            results.write(str(year)+'_'+str(patient)+',')

            lst = clf.predict(test_x)
            # lst = clf.predict(scaler.transform(test_x))
            lst[lst == 1] = 0
            lst[lst == -1] = 1

            results.write(str(accuracy_score(test_y, lst)*100) + ',' + str(precision_score(test_y, lst)) + ',' + str(recall_score(test_y, lst)) + ',' + str(f1_score(test_y, lst)) + '\n')
    results.close()
    ######################################################################################################################################
    # Samples (OE)
    os.makedirs(output_directory/"samples", exist_ok=True)
    results = open(output_directory/"samples"/"Results.csv", 'w')
    results.write('Patient,Accuracy,Precision,Recall,F1\n')

    for year in [2018, 2020]:
        for patient in range(6):
            Accuracy = []
            Precision = []
            Recall = []
            F1 = []
            test = np.load(data_dir/f"ohiot1dm_test_{year}_{patient}.npy")
            test_x = test[:, :-1]
            test_y = test[:, -1]

            results.write(str(year)+'_'+str(patient)+',')
            for sample in range(10):
                print("Samples "+str(sample)+"\tPatient "+str(year)+"_"+str(patient))
                train = np.load(data_dir/f"ohiot1dm_train_samples_{sample}.npy")

                train_x = train[:, :-1]
                train_y = train[:, -1]

                clf.fit(train_x)
                # clf.fit(scaler.fit_transform(train_x))

                lst = clf.predict(test_x)
                # lst = clf.predict(scaler.transform(test_x))
                lst[lst == 1] = 0
                lst[lst == -1] = 1

                Accuracy.insert(len(Accuracy), accuracy_score(test_y, lst)*100)
                Precision.insert(len(Precision), precision_score(test_y, lst))
                Recall.insert(len(Recall), recall_score(test_y, lst))
                F1.insert(len(F1), f1_score(test_y, lst))

                # results.write(str(Accuracy[-1]) + ',' + str(Precision[-1]) + ',' + str(Recall[-1]) + ',' + str(F1[-1]) + '\n')
            results.write(str(year)+'_'+str(patient)+','+str(np.mean(Accuracy)) + ',' + str(np.mean(Precision)) + ',' + str(np.mean(Recall)) + ',' + str(np.mean(F1)) + '\n')

    results.close()
    ######################################################################################################################################

    combine_results(output_directory)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(
		description="Run OhioT1DM script: python evaluate_oneclasssvm.py <output_directory>",
		epilog="Example: python evaluate_oneclasssvm.py output"
	)
	parser.add_argument("out_dir", nargs="?", default="output/defense_output/OneClassSVM", help="Output directory")
	parser.add_argument("--data_dir", default="output/defense_dataset", help="Directory containing generated defense dataset .npy files")

	args = parser.parse_args()

	dataset_root = Path(__file__).resolve().parents[1]

	output_directory = dataset_root / args.out_dir
	data_directory = dataset_root / args.data_dir

	evaluate_oneclasssvm(output_directory, data_directory)