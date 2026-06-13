import os
import argparse
import joblib
import numpy as np
import pandas as pd
import random
from pathlib import Path


def create_mixed_data(benign_data, adversarial_data, index):
    """Create mixed benign/adversarial data with random selection (optimized)."""
    n = len(adversarial_data)

    # Extract relevant fields once
    benign_vals = np.array([benign_data[i][11] for i in range(n)])
    adversarial_vals = np.array([adversarial_data[i][11] for i in range(n)])

    # Check where data differs
    differs = benign_vals[:, 0] != adversarial_vals[:, 0]

    # Generate random selection for all rows at once
    random_select = np.random.randint(0, 2, size=n) == 0

    # Select which data to use (benign when different and random=0, or when same)
    use_benign = ~differs | random_select

    # Build instances array
    instances = np.zeros((n, 6))
    instances[:, 0] = index
    instances[:, 1] = np.where(use_benign, benign_vals[:, 0], adversarial_vals[:, 0])
    instances[:, 2] = np.where(use_benign, benign_vals[:, 1], adversarial_vals[:, 1])
    instances[:, 3] = np.where(use_benign, benign_vals[:, 5], adversarial_vals[:, 5])
    instances[:, 4] = np.where(use_benign, benign_vals[:, 2], adversarial_vals[:, 2])
    instances[:, 5] = (~use_benign & differs).astype(int)

    return instances.tolist()


def generate_defense_dataset(cluster_dir, out_dir, data_dir=None):
    os.makedirs(out_dir, exist_ok=True)
    if data_dir is None:
        data_dir = Path(__file__).resolve().parent / "output"

    index = 0

    AllPatientIDs = joblib.load(cluster_dir/"AllPatientIDs.pkl")
    LessVulnerablePatientIDs = joblib.load(cluster_dir/"LessVulnerablePatientIDs.pkl")
    MoreVulnerablePatientIDs = joblib.load(cluster_dir/"MoreVulnerablePatientIDs.pkl")

    ########################################################################################################################
    # train_year
    # test_year

    for year in ['2018', '2020']:
        instances = []
        if year == '2018':
            patients = ['559', '563', '570', '575', '588', '591'] #, 'allsubs']
        else:
            patients = ['540', '544', '552', '567', '584', '596']  # , 'allsubs']
        for patient in patients:
            benign_data = joblib.load(data_dir/year/patient /"benign_data.pkl")
            adversarial_data = joblib.load(data_dir/year/patient /"adversarial_data.pkl")

            instances.extend(create_mixed_data(benign_data, adversarial_data, index))

            index += 1

        data = np.array(instances)

        train_parts = []
        test_parts = []

        # get unique values from first column
        for idx in np.unique(data[:, 0]):
            group = data[data[:, 0] == idx]

            split = int(len(group) * 0.75)

            train_parts.append(group[:split])
            test_parts.append(group[split:])

        train = np.vstack(train_parts)
        test = np.vstack(test_parts)


        # ohiot1dm_test[:math.floor(0.75 * len(ohiot1dm_test)), :]
        np.save(out_dir / f"ohiot1dm_train_{year}.npy", train)
        np.save(out_dir/ f"ohiot1dm_test_{year}.npy", test)

    ########################################################################################################################
    # train_year_patient
    # test_year_patient
    train_2020 = np.load(out_dir / "ohiot1dm_train_2020.npy")
    test_2020 = np.load(out_dir / "ohiot1dm_test_2020.npy")
    train_2018 = np.load(out_dir / "ohiot1dm_train_2018.npy")
    test_2018 = np.load(out_dir / "ohiot1dm_test_2018.npy")

    patients = np.arange(12)
    for i in patients:
        if i < 6:
            np.save(out_dir / f"ohiot1dm_train_2018_{i}.npy", train_2018[train_2018[:, 0] == i, 1:])
            np.save(out_dir / f"ohiot1dm_test_2018_{i}.npy", test_2018[test_2018[:, 0] == i, 1:])
        else:
            np.save(out_dir / f"ohiot1dm_train_2020_{i-6}.npy", train_2020[train_2020[:, 0] == i, 1:])
            np.save(out_dir / f"ohiot1dm_test_2020_{i-6}.npy", test_2020[test_2020[:, 0] == i, 1:])

    ########################################################################################################################
    # less (mixed benign and adversarial)
    # more (mixed benign and adversarial)
    # all (mixed benign and adversarial)
    less = pd.DataFrame()
    more = pd.DataFrame()
    all_mixed = pd.DataFrame()
    all_benign = pd.DataFrame()

    LVindex = 0
    MVindex = 0
    i = 0

    for year in ['2018', '2020']:
        for patient in [0, 1, 2, 3, 4, 5]:
            p = np.load(out_dir/f"ohiot1dm_train_{year}_{patient}.npy")

            if i in LessVulnerablePatientIDs:
                if not LVindex:
                    less = pd.DataFrame(p) #'glucose', 'dose', 'finger', 'carbs', 'adversarial'
                else:
                    less = pd.concat([less, pd.DataFrame(p)], ignore_index=True)
                LVindex += 1

            if i in MoreVulnerablePatientIDs:
                if not MVindex:
                    more = pd.DataFrame(p)
                else:
                    more = pd.concat([more, pd.DataFrame(p)], ignore_index=True)
                MVindex += 1

            # Accumulate all mixed data
            if not i:
                all_mixed = pd.DataFrame(p)
            else:
                all_mixed = pd.concat([all_mixed, pd.DataFrame(p)], ignore_index=True)

            i+=1

    # Create all from benign data only
    for year in ['2018', '2020']:
        instances_benign = []
        if year == '2018':
            patients = ['559', '563', '570', '575', '588', '591']
        else:
            patients = ['540', '544', '552', '567', '584', '596']
        for patient in patients:
            benign_data = joblib.load(data_dir/year/patient /"benign_data.pkl")
            for i in range(len(benign_data)):
                instance = [benign_data[i][11][0], benign_data[i][11][1], benign_data[i][11][5], benign_data[i][11][2], 0]
                instances_benign.append(instance)

    all_benign = pd.DataFrame(instances_benign)

    less = np.array(less.fillna(0))
    more = np.array(more.fillna(0))
    all_mixed = np.array(all_mixed.fillna(0))
    all_benign = np.array(all_benign.fillna(0))


    np.save(out_dir / f"ohiot1dm_train_less_0.npy", less)
    np.save(out_dir / f"ohiot1dm_train_more_0.npy", more)
    np.save(out_dir / f"ohiot1dm_train_all_0.npy", all_mixed)
    np.save(out_dir / f"ohiot1dm_train_all_benign_0.npy", all_benign)

    ########################################################################################################################
    # samples
    f = open(out_dir/"Samples.txt", "w")

    patients = []
    for year in ["2018", "2020"]:
        for i in range(6):
            patients.append(year+"_"+str(i))

    for i in range(10):
        selection = []

        while len(selection) != 10:
            x = np.random.randint(12)
            if not any(num == x for num in selection): # and x != 1 and x != 2 and x != 11:
                selection.append(x)
                f.write(patients[x]+'\t')

        f.write('\n')

        for p in range(len(selection)):
            patient_data = pd.DataFrame(np.load(out_dir/f"ohiot1dm_train_{patients[selection[p]]}.npy"))
            if p == 0:
                train_data = patient_data
            else:
                train_data = pd.concat([train_data, patient_data])

        train_data = np.array(train_data)
        np.save(out_dir/f"ohiot1dm_train_samples_{i}.npy", train_data)

    f.close()
    ########################################################################################################################


if __name__ == '__main__':
	parser = argparse.ArgumentParser(
		description="Run OhioT1DM script: python generate_defense_dataset.py <cluster_output> <output_directory>",
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