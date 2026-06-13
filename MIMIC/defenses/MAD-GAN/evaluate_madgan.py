import os
import numpy as np
from pathlib import Path
import argparse
import pandas as pd


def combine_results(output_directory):
    # === File paths (edit these) ===
    file_less = output_directory/"less"/"Results.csv"
    file_samples = output_directory/"samples"/"Results.csv"
    file_more = output_directory/"more"/"Results.csv"
    file_all = output_directory/"all"/"Results.csv"
    file_all_benign = output_directory/"all_benign"/"Results.csv"

    output_file = output_directory/"MADGAN_combined_results.csv"


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


def evaluate_madgan(output_directory, data_dir=None):

    os.makedirs(output_directory, exist_ok=True)

    if data_dir is not None:
        os.environ["ROAST_DEFENSE_DATASET_DIR"] = str(data_dir)

    # with is like your try .. finally block in this case
    with open('./experiments/settings/mimic.txt', 'r') as train_file:
        # read a list of lines into data
        train_data = train_file.readlines()

    # now change the 2nd line, note that you have to add a newline
    train_data[2] = "\"year\": \"more\",\n"
    train_data[3] = "\"patient\": \"0\",\n"

    # and write everything back
    with open('./experiments/settings/mimic.txt', 'w') as train_file:
        train_file.writelines(train_data)

    print('Training:')
    os.makedirs(output_directory/"more", exist_ok=True)
    os.system('python RGAN.py --settings_file mimic > '+str(output_directory)+"/more/train.txt")

    # with is like your try .. finally block in this case
    with open('./experiments/settings/mimic_test.txt', 'r') as test_file:
        # read a list of lines into data
        test_data = test_file.readlines()

    print('Testing: ')
    for cv in range(5):
        test_data[3] = "\"patient\": \"" + str(cv) + "\",\n"

        # and write everything back
        with open('./experiments/settings/mimic_test.txt', 'w') as test_file:
            test_file.writelines(test_data)

        print('CV: ' + str(cv))
        os.system('python AD.py --settings_file mimic_test > '+str(output_directory)+"/more/test_more_" + str(cv) + '.txt')

    out = open(str(output_directory)+"/more/Results.csv", "w")
    out.write('CV,Accuracy,Precision,Recall,F1\n')

    Accuracy = []
    Precision = []
    Recall = []
    F1 = []
    for cv in range(5):
        with open(str(output_directory)+"/more/test_more_" + str(cv) + '.txt', 'r') as file:
            # read a list of lines into data
            data = file.readlines()
        Accuracy.append(float(data[-3].split(' ')[4][:-1]))
        Precision.append(float(data[-3].split(' ')[6][:-1]))
        Recall.append(float(data[-3].split(' ')[8][:-1]))
        F1.append(float(data[-3].split(' ')[10][:-1]))
        out.write(str(cv) + ',' + str(data[-3].split(' ')[4][:-1]) + ',' + str(data[-3].split(' ')[6][:-1]) + ',' + str(data[-3].split(' ')[8][:-1]) + ',' + str(data[-3].split(' ')[10]) + '\n')
    # out.write('Average,' + str(np.average(np.array(Accuracy))) + ',' + str(np.average(np.array(Precision))) + ',' + str(np.average(np.array(Recall))) + ',' + str(np.average(np.array(F1))) + '\n')
    out.close()

    print('-----------------------------------------------------------------------------------------------------------------------------')

    # with is like your try .. finally block in this case
    with open('./experiments/settings/mimic.txt', 'r') as train_file:
        # read a list of lines into data
        train_data = train_file.readlines()

    # now change the 2nd line, note that you have to add a newline
    train_data[2] = "\"year\": \"less\",\n"
    train_data[3] = "\"patient\": \"0\",\n"

    # and write everything back
    with open('./experiments/settings/mimic.txt', 'w') as train_file:
        train_file.writelines(train_data)

    print('Training:')
    os.makedirs(output_directory/"less", exist_ok=True)
    os.system('python RGAN.py --settings_file mimic > '+str(output_directory)+"/less/train.txt")

    # with is like your try .. finally block in this case
    with open('./experiments/settings/mimic_test.txt', 'r') as test_file:
        # read a list of lines into data
        test_data = test_file.readlines()

    print('Testing: ')
    for cv in range(5):
        test_data[3] = "\"patient\": \"" + str(cv) + "\",\n"

        # and write everything back
        with open('./experiments/settings/mimic_test.txt', 'w') as test_file:
            test_file.writelines(test_data)

        print('CV: ' + str(cv))
        os.system('python AD.py --settings_file mimic_test > '+str(output_directory)+"/less/test_less_" + str(cv) + '.txt')


    out = open(str(output_directory)+"/less/Results.csv", "w")
    out.write('CV,Accuracy,Precision,Recall,F1\n')

    Accuracy = []
    Precision = []
    Recall = []
    F1 = []
    for cv in range(5):
        with open(str(output_directory)+"/less/test_less_" + str(cv) + '.txt', 'r') as file:
            # read a list of lines into data
            data = file.readlines()
        Accuracy.append(float(data[-3].split(' ')[4][:-1]))
        Precision.append(float(data[-3].split(' ')[6][:-1]))
        Recall.append(float(data[-3].split(' ')[8][:-1]))
        F1.append(float(data[-3].split(' ')[10][:-1]))
        out.write(str(cv) + ',' + str(data[-3].split(' ')[4][:-1]) + ',' + str(data[-3].split(' ')[6][:-1]) + ',' + str(data[-3].split(' ')[8][:-1]) + ',' + str(data[-3].split(' ')[10]) + '\n')
    # out.write('Average,' + str(np.average(np.array(Accuracy))) + ',' + str(np.average(np.array(Precision))) + ',' + str(np.average(np.array(Recall))) + ',' + str(np.average(np.array(F1))) + '\n')
    out.close()

    print('-----------------------------------------------------------------------------------------------------------------------------')

    # with is like your try .. finally block in this case
    with open('./experiments/settings/mimic.txt', 'r') as train_file:
        # read a list of lines into data
        train_data = train_file.readlines()

    # now change the 2nd line, note that you have to add a newline
    train_data[2] = "\"year\": \"all_benign\",\n"
    train_data[3] = "\"patient\": \"0\",\n"

    # and write everything back
    with open('./experiments/settings/mimic.txt', 'w') as train_file:
        train_file.writelines(train_data)

    print('Training:')
    os.makedirs(output_directory/"all_benign", exist_ok=True)
    os.system('python RGAN.py --settings_file mimic > '+str(output_directory)+"/all_benign/train.txt")

    # with is like your try .. finally block in this case
    with open('./experiments/settings/mimic_test.txt', 'r') as test_file:
        # read a list of lines into data
        test_data = test_file.readlines()

    print('Testing: ')
    for cv in range(5):
        test_data[3] = "\"patient\": \"" + str(cv) + "\",\n"

        # and write everything back
        with open('./experiments/settings/mimic_test.txt', 'w') as test_file:
            test_file.writelines(test_data)

        print('CV: ' + str(cv))
        os.system('python AD.py --settings_file mimic_test > '+str(output_directory)+"/all_benign/test_all_" + str(cv) + '.txt')

    out = open(str(output_directory)+"/all_benign/Results.csv", "w")
    out.write('CV,Accuracy,Precision,Recall,F1\n')

    Accuracy = []
    Precision = []
    Recall = []
    F1 = []
    for cv in range(5):
        with open(str(output_directory)+"/all_benign/test_all_" + str(cv) + '.txt', 'r') as file:
            # read a list of lines into data
            data = file.readlines()
        Accuracy.append(float(data[-3].split(' ')[4][:-1]))
        Precision.append(float(data[-3].split(' ')[6][:-1]))
        Recall.append(float(data[-3].split(' ')[8][:-1]))
        F1.append(float(data[-3].split(' ')[10][:-1]))
        out.write(str(cv) + ',' + str(data[-3].split(' ')[4][:-1]) + ',' + str(data[-3].split(' ')[6][:-1]) + ',' + str(data[-3].split(' ')[8][:-1]) + ',' + str(data[-3].split(' ')[10]) + '\n')
    # out.write('Average,' + str(np.average(np.array(Accuracy))) + ',' + str(np.average(np.array(Precision))) + ',' + str(np.average(np.array(Recall))) + ',' + str(np.average(np.array(F1))) + '\n')
    out.close()

    print('-----------------------------------------------------------------------------------------------------------------------------')
    
    # with is like your try .. finally block in this case
    with open('./experiments/settings/mimic.txt', 'r') as train_file:
        # read a list of lines into data
        train_data = train_file.readlines()

    # now change the 2nd line, note that you have to add a newline
    train_data[2] = "\"year\": \"all\",\n"
    train_data[3] = "\"patient\": \"0\",\n"

    # and write everything back
    with open('./experiments/settings/mimic.txt', 'w') as train_file:
        train_file.writelines(train_data)

    print('Training:')
    os.makedirs(output_directory/"all", exist_ok=True)
    os.system('python RGAN.py --settings_file mimic > '+str(output_directory)+"/all/train.txt")

    # with is like your try .. finally block in this case
    with open('./experiments/settings/mimic_test.txt', 'r') as test_file:
        # read a list of lines into data
        test_data = test_file.readlines()

    print('Testing: ')
    for cv in range(5):
        test_data[3] = "\"patient\": \"" + str(cv) + "\",\n"

        # and write everything back
        with open('./experiments/settings/mimic_test.txt', 'w') as test_file:
            test_file.writelines(test_data)

        print('CV: ' + str(cv))
        os.system('python AD.py --settings_file mimic_test > '+str(output_directory)+"/all/test_all_" + str(cv) + '.txt')

    out = open(str(output_directory)+"/all/Results.csv", "w")
    out.write('CV,Accuracy,Precision,Recall,F1\n')

    Accuracy = []
    Precision = []
    Recall = []
    F1 = []
    for cv in range(5):
        with open(str(output_directory)+"/all/test_all_" + str(cv) + '.txt', 'r') as file:
            # read a list of lines into data
            data = file.readlines()
        Accuracy.append(float(data[-3].split(' ')[4][:-1]))
        Precision.append(float(data[-3].split(' ')[6][:-1]))
        Recall.append(float(data[-3].split(' ')[8][:-1]))
        F1.append(float(data[-3].split(' ')[10][:-1]))
        out.write(str(cv) + ',' + str(data[-3].split(' ')[4][:-1]) + ',' + str(data[-3].split(' ')[6][:-1]) + ',' + str(data[-3].split(' ')[8][:-1]) + ',' + str(data[-3].split(' ')[10]) + '\n')
    # out.write('Average,' + str(np.average(np.array(Accuracy))) + ',' + str(np.average(np.array(Precision))) + ',' + str(np.average(np.array(Recall))) + ',' + str(np.average(np.array(F1))) + '\n')
    out.close()

    print('-----------------------------------------------------------------------------------------------------------------------------')

    # with is like your try .. finally block in this case
    with open('./experiments/settings/mimic.txt', 'r') as train_file:
        # read a list of lines into data
        train_data = train_file.readlines()

    # now change the 2nd line, note that you have to add a newline
    train_data[2] = "\"year\": \"samples\",\n"

    for run in range(5):
        # now change the 2nd line, note that you have to add a newline
        train_data[3] = "\"patient\": \"" + str(run) + "\",\n"

        # and write everything back
        with open('./experiments/settings/mimic.txt', 'w') as train_file:
            train_file.writelines(train_data)

        print('Training:')
        os.makedirs(output_directory/"samples", exist_ok=True)
        os.system('python RGAN.py --settings_file mimic > '+str(output_directory)+"/samples/train_"+ str(run) + '.txt')

        # with is like your try .. finally block in this case
        with open('./experiments/settings/mimic_test.txt', 'r') as test_file:
            # read a list of lines into data
            test_data = test_file.readlines()

        print('Testing: ')
        for cv in range(5):
            test_data[3] = "\"patient\": \"" + str(cv) + "\",\n"

            # and write everything back
            with open('./experiments/settings/mimic_test.txt', 'w') as test_file:
                test_file.writelines(test_data)

            print('CV: ' + str(cv))
            os.system('python AD.py --settings_file mimic_test > '+str(output_directory)+"/samples/test_run_" + str(run) + '_' + str(cv) + '.txt')

    
    out = open(str(output_directory)+"/samples/Results.csv", "w")
    out.write('CV,Accuracy,Precision,Recall,F1\n')

    for run in range(5):
        Accuracy = []
        Precision = []
        Recall = []
        F1 = []
        for cv in range(5):
            with open(str(output_directory)+"/samples/test_run_" + str(run) + '_' + str(cv) + '.txt', 'r') as file:
                # read a list of lines into data
                data = file.readlines()
            Accuracy.append(float(data[-3].split(' ')[4][:-1]))
            Precision.append(float(data[-3].split(' ')[6][:-1]))
            Recall.append(float(data[-3].split(' ')[8][:-1]))
            F1.append(float(data[-3].split(' ')[10][:-1]))
            # out.write(str(run) + ',' + str(cv) + ',' + str(data[-3].split(' ')[4][:-1]) + ',' + str(data[-3].split(' ')[6][:-1]) + ',' + str(data[-3].split(' ')[8][:-1]) + ',' + str(data[-3].split(' ')[10]) + '\n')
        out.write(f'{run},' + str(np.average(np.array(Accuracy))) + ',' + str(np.average(np.array(Precision))) + ',' + str(np.average(np.array(Recall))) + ',' + str(np.average(np.array(F1))) + '\n')
    out.close()

    combine_results(output_directory)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(
		description="Run MIMIC script: python evaluate_madgan.py <output_directory>",
		epilog="Example: python evaluate_madgan.py output"
	)
	parser.add_argument("out_dir", nargs="?", default="output/defense_output/MADGAN", help="Output directory")
	parser.add_argument("--data_dir", default="output/defense_dataset", help="Directory containing generated defense dataset .npy files")

	args = parser.parse_args()

	dataset_root = Path(__file__).resolve().parents[2]

	output_directory = dataset_root / args.out_dir
	data_directory = dataset_root / args.data_dir

	evaluate_madgan(output_directory, data_directory)

    