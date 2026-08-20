import argparse

from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
import pandas as pd
import warnings
import math
import numpy as np
import joblib
from scipy import stats
from dtaidistance import dtw, clustering
import itertools
from tqdm import tqdm
import os
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import squareform

warnings.filterwarnings("ignore")

def risk_profile(output_directory):
    years = ['2018', '2020']
    patients_2018 = ['559', '563', '570', '575', '588', '591']
    patients_2020 = ['540', '544', '552', '567', '584', '596']
    window = 12
    no_of_features = 8

    timeseries_original = []
    severity_coefficients = []
    i = 0
    for year in years:
        if year == '2018':
            patients = patients_2018
        else:
            patients = patients_2020
        for patient in patients:
            try:
                benign_data_path = output_directory/year/patient/"benign_data.pkl"
                adversarial_output_path = output_directory/year/patient/"predicted_output.pkl"
                benign_output_path = output_directory/year/patient/"actual_output.pkl"
                benign_data = joblib.load(benign_data_path)
                adversarial_output = joblib.load(adversarial_output_path)
                benign_output = joblib.load(benign_output_path)
            except FileNotFoundError:
                print(f"{year} results not available. Skipping...")
                continue
            timeseries_original.append(pow(adversarial_output - benign_output, 2))

            if i == 0:
                data = benign_data
                output = adversarial_output
                actual_output = benign_output
            else:
                data = np.concatenate((data, benign_data), axis=0)
                output = np.concatenate((output, adversarial_output), axis=0)
                actual_output = np.concatenate((actual_output, benign_output), axis=0)
            ##################################################################################
            coefficient = np.empty([benign_output.shape[0], benign_output.shape[1]])
            for i in range(len(benign_output)):
                postprandial = any([benign_data[i][0][7], benign_data[i][1][7], benign_data[i][2][7], benign_data[i][3][7],
                                    benign_data[i][4][7], benign_data[i][5][7], benign_data[i][6][7], benign_data[i][7][7],
                                    benign_data[i][8][7], benign_data[i][9][7], benign_data[i][10][7],
                                    benign_data[i][11][7]])  # check if postprandial (True) or fasting (False)
                for j in range(len(benign_output[i])):
                    if not postprandial:  # fasting
                        if benign_output[i][j] < 70 and adversarial_output[i][j] > 125:  # actual (hypo), predicted (hyper)
                            coefficient[i][j] = 64
                        elif 70 < benign_output[i][j] < 125 < adversarial_output[i][j]:  # actual (normal), predicted (hyper)
                            coefficient[i][j] = 32
                        elif benign_output[i][j] < 70 < adversarial_output[i][j] < 125:  # actual (hypo), predicted (normal)
                            coefficient[i][j] = 16
                        elif benign_output[i][j] > 125 and adversarial_output[i][j] < 70:  # actual (hyper), predicted (hypo)
                            coefficient[i][j] = 8
                        elif benign_output[i][j] > 125 > adversarial_output[i][j] > 70:  # actual (hyper), predicted (normal)
                            coefficient[i][j] = 4
                        elif 125 > benign_output[i][j] > 70 > adversarial_output[i][j]:  # actual (normal), predicted (hypo)
                            coefficient[i][j] = 2
                        else:
                            coefficient[i][j] = 0
                    else:  # postprandial
                        if benign_output[i][j] < 70 and adversarial_output[i][j] > 180:  # actual (hypo), predicted (hyper)
                            coefficient[i][j] = 64
                        elif 70 < benign_output[i][j] < 180 < adversarial_output[i][j]:  # actual (normal), predicted (hyper)
                            coefficient[i][j] = 32
                        elif benign_output[i][j] < 70 < adversarial_output[i][j] < 180:  # actual (hypo), predicted (normal)
                            coefficient[i][j] = 16
                        elif benign_output[i][j] > 180 and adversarial_output[i][j] < 70:  # actual (hyper), predicted (hypo)
                            coefficient[i][j] = 8
                        elif benign_output[i][j] > 180 > adversarial_output[i][j] > 70:  # actual (hyper), predicted (normal)
                            coefficient[i][j] = 4
                        elif 180 > benign_output[i][j] > 70 > adversarial_output[i][j]:  # actual (normal), predicted (hypo)
                            coefficient[i][j] = 2
                        else:
                            coefficient[i][j] = 0
            severity_coefficients.append(coefficient)
            ##################################################################################
            i+=1

    # CGM is the only manipulated channel, so the normalized absolute influence
    # weight is exactly one.  Keep the instantaneous impact non-negative and avoid
    # the former categorical severity multiplier, which did not match the paper's
    # declared regression-weight rule.
    timeseries = [np.asarray(ts, dtype=float) for ts in timeseries_original]

    joblib.dump(timeseries, output_directory/"risk_profiles.pkl")


if __name__ == '__main__':
	parser = argparse.ArgumentParser(
		description="Run OhioT1DM script: python risk_profile.py <output_directory>",
		epilog="Example: python risk_profile.py output"
	)
	parser.add_argument("out_dir", nargs="?", default="output", help="Output directory")

	args = parser.parse_args()

	SCRIPT_DIR = Path(__file__).resolve().parent

	output_directory = SCRIPT_DIR / args.out_dir

	risk_profile(output_directory)
