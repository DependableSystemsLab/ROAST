import csv
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import joblib
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")


def risk_profile(output_directory):
    adversarial_data = joblib.load(output_directory/"adversarial_data.pkl")
    benign_data = joblib.load(output_directory/"benign_data.pkl")
    adversarial_output = joblib.load(output_directory/"adversarial_output.pkl")

    # get feature numbers for CHART columns from the first two rows of dynamic.csv
    data_csv_dir = Path(__file__).resolve().parent / "data" / "csv"
    dynamic_csv_candidates = sorted(data_csv_dir.glob("*/dynamic.csv"))
    if not dynamic_csv_candidates:
        raise FileNotFoundError(
            f"Could not find dynamic.csv in {data_csv_dir}"
        )
    dynamic_csv_path = dynamic_csv_candidates[0]

    with open(dynamic_csv_path) as file_obj:
        reader_obj = csv.reader(file_obj)
        row_1 = next(reader_obj, [])
        row_2 = next(reader_obj, [])

    features = np.array([
        int(col_id)
        for section, col_id in zip(row_1, row_2)
        if section.strip().upper() == "CHART"
    ])

    # pre-processing
    adversarial_data = adversarial_data.reshape(-1, 72 * 432)
    adversarial_data = preprocessing.normalize(adversarial_data)
    benign_data = benign_data.reshape(-1, 72 * 432)
    benign_data = preprocessing.normalize(benign_data)
    adversarial_output = adversarial_output >= 0.5

    # logistic regression for feature importance
    # define dataset
    # X = adversarial_data
    X = benign_data
    y = adversarial_output
    # define the model
    model = LogisticRegression(max_iter=2000, random_state=42)
    # fit the model
    model.fit(StandardScaler().fit_transform(X), y)
    # get importance
    importance = np.abs(model.coef_[0])
    importance = importance / importance.sum() if importance.sum() else np.full_like(importance, 1 / len(importance))

    # timeseries = importance*adversarial_data
    timeseries = importance*pow(adversarial_data-benign_data, 2)
    timeseries = timeseries.reshape(-1, 72, 432)

    risk_profiles = []
    features = [143, 83, 89, 118, 183, 114, 177, 47, 75, 100]

    for i in range(len(timeseries)):
        df = pd.DataFrame(timeseries[i])
        risk_profiles.append(df[features].to_numpy().reshape(72*len(features)))

    joblib.dump(risk_profiles, output_directory / 'risk_profiles.pkl')


if __name__ == '__main__':
	parser = argparse.ArgumentParser(
		description="Run MIMIC script: python risk_profile.py <output_directory>",
		epilog="Example: python risk_profile.py output"
	)
	parser.add_argument("out_dir", nargs="?", default="output", help="Output directory")

	args = parser.parse_args()

	SCRIPT_DIR = Path(__file__).resolve().parent

	output_directory = SCRIPT_DIR / args.out_dir

	risk_profile(output_directory)
