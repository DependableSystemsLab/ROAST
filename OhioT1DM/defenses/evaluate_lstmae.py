import os
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import numpy as np
from pathlib import Path
import csv
import pandas as pd
import shutil
import argparse
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# LSTM Autoencoder time-series anomaly detector (Malhotra et al., 2016).
#
# A sliding window of WINDOW consecutive feature rows is encoded by an LSTM into a
# latent vector and reconstructed by a decoder LSTM. The model is trained to
# reconstruct the (mostly benign) training windows; the per-window reconstruction
# error is the anomaly score. One window ends at each row (left-padded at the start),
# so the detector produces one prediction per row -- aligned with the labels exactly
# like the point detectors (kNN, One-Class SVM, Deep SVDD), enabling a fair
# comparison. The decision threshold is the 50th percentile of training errors, i.e.
# contamination = 0.5, matching the fixed setting used by the other ADs.
# ---------------------------------------------------------------------------
WINDOW = 10
HIDDEN = 32
EPOCHS = 20
BATCH = 256
LR = 1e-3
CONTAMINATION = 0.5
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features, hidden=HIDDEN):
        super().__init__()
        self.encoder = nn.LSTM(n_features, hidden, batch_first=True)
        self.decoder = nn.LSTM(hidden, hidden, batch_first=True)
        self.output = nn.Linear(hidden, n_features)

    def forward(self, x):
        # x: (batch, window, n_features)
        _, (h, _) = self.encoder(x)
        window = x.size(1)
        z = h[-1].unsqueeze(1).repeat(1, window, 1)
        dec, _ = self.decoder(z)
        return self.output(dec)


def make_windows(features, window):
    """Return (n_rows, window, n_features); window i ends at row i, left-padded."""
    n, f = features.shape
    pad = np.repeat(features[:1], window - 1, axis=0)
    padded = np.concatenate([pad, features], axis=0)
    idx = np.arange(window)[None, :] + np.arange(n)[:, None]
    return padded[idx]


def _scores(model, x):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, x.size(0), BATCH):
            batch = x[i:i + BATCH]
            recon = model(batch)
            err = nn.functional.mse_loss(recon, batch, reduction="none").mean(dim=(1, 2))
            out.append(err.cpu().numpy())
    return np.concatenate(out) if out else np.array([])


def fit_lstmae(train):
    """Fit the LSTM-AE on a training array (last column is the ignored label)."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    feats = train[:, :-1].astype(np.float32)
    mean = feats.mean(axis=0)
    std = feats.std(axis=0)
    std[std == 0] = 1.0
    feats_n = (feats - mean) / std

    windows = make_windows(feats_n, WINDOW)
    x = torch.tensor(windows, dtype=torch.float32, device=DEVICE)

    model = LSTMAutoencoder(feats.shape[1]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(EPOCHS):
        perm = torch.randperm(x.size(0), device=DEVICE)
        for i in range(0, x.size(0), BATCH):
            batch = x[perm[i:i + BATCH]]
            optimizer.zero_grad()
            loss = loss_fn(model(batch), batch)
            loss.backward()
            optimizer.step()

    train_scores = _scores(model, x)
    threshold = float(np.percentile(train_scores, 100 * (1 - CONTAMINATION)))
    return {"model": model, "mean": mean, "std": std, "threshold": threshold}


def predict_lstmae(detector, test):
    """Return per-row 0/1 predictions for a test array (last column is the label)."""
    feats = test[:, :-1].astype(np.float32)
    feats_n = (feats - detector["mean"]) / detector["std"]
    windows = make_windows(feats_n, WINDOW)
    x = torch.tensor(windows, dtype=torch.float32, device=DEVICE)
    scores = _scores(detector["model"], x)
    return (scores > detector["threshold"]).astype(int)


def combine_results(output_directory):
    # === File paths (edit these) ===
    file_less = output_directory/"less"/"Results.csv"
    file_samples = output_directory/"samples"/"Results.csv"
    file_more = output_directory/"more"/"Results.csv"
    file_all = output_directory/"all"/"Results.csv"
    file_all_benign = output_directory/"all_benign"/"Results.csv"

    output_file = output_directory/"LSTMAE_combined_results.csv"


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



def evaluate_lstmae(output_directory, data_dir=None):
    os.makedirs(output_directory, exist_ok=True)
    if data_dir is None:
        data_dir = Path(__file__).resolve().parents[1] / "output" / "defense_dataset"

    ######################################################################################################################################
    # less (OE)
    os.makedirs(output_directory / "less", exist_ok=True)
    results = open(output_directory / "less" / "Results.csv", 'w')
    results.write('Patient,Accuracy,Precision,Recall,F1\n')

    train = np.load(data_dir / "ohiot1dm_train_less_0.npy")
    detector = fit_lstmae(train)

    for year in [2018, 2020]:
        for patient in range(6):
            print("Less\tPatient " + str(year) + "_" + str(patient))
            test = np.load(data_dir / f"ohiot1dm_test_{year}_{patient}.npy")
            test_y = test[:, -1]

            results.write(str(year) + '_' + str(patient) + ',')

            lst = predict_lstmae(detector, test)

            results.write(str(accuracy_score(test_y, lst) * 100) + ',' + str(precision_score(test_y, lst)) + ',' + str(
                recall_score(test_y, lst)) + ',' + str(f1_score(test_y, lst)) + '\n')
    results.close()
    ######################################################################################################################################
    # more (OE)
    os.makedirs(output_directory/"more", exist_ok=True)
    results = open(output_directory/"more"/"Results.csv", 'w')
    results.write('Patient,Accuracy,Precision,Recall,F1\n')

    train = np.load(data_dir/"ohiot1dm_train_more_0.npy")
    detector = fit_lstmae(train)

    for year in [2018, 2020]:
        for patient in range(6):
            print("More\tPatient " + str(year) + "_" + str(patient))
            test = np.load(data_dir / f"ohiot1dm_test_{year}_{patient}.npy")
            test_y = test[:, -1]

            results.write(str(year) + '_' + str(patient) + ',')

            lst = predict_lstmae(detector, test)

            results.write(str(accuracy_score(test_y, lst) * 100) + ',' + str(precision_score(test_y, lst)) + ',' + str(
                recall_score(test_y, lst)) + ',' + str(f1_score(test_y, lst)) + '\n')
    results.close()
    ######################################################################################################################################
    #All patients (Benign)
    os.makedirs(output_directory/"all_benign", exist_ok=True)
    results = open(output_directory/"all_benign"/"Results.csv", 'w')
    results.write('Patient,Accuracy,Precision,Recall,F1\n')

    train = np.load(data_dir/"ohiot1dm_train_all_benign_0.npy")
    detector = fit_lstmae(train)

    for year in [2018, 2020]:
        for patient in range(6):
            print("All\tPatient "+str(year)+"_"+str(patient))
            test = np.load(data_dir/f"ohiot1dm_test_{year}_{patient}.npy")
            test_y = test[:, -1]

            results.write(str(year)+'_'+str(patient)+',')

            lst = predict_lstmae(detector, test)

            results.write(str(accuracy_score(test_y, lst)*100) + ',' + str(precision_score(test_y, lst)) + ',' + str(recall_score(test_y, lst)) + ',' + str(f1_score(test_y, lst)) + '\n')
    results.close()
    ######################################################################################################################################
    #All patients (OE)
    os.makedirs(output_directory/"all", exist_ok=True)
    results = open(output_directory/"all"/"Results.csv", 'w')
    results.write('Patient,Accuracy,Precision,Recall,F1\n')

    train = np.load(data_dir/"ohiot1dm_train_all_0.npy")
    detector = fit_lstmae(train)

    for year in [2018, 2020]:
        for patient in range(6):
            print("All\tPatient "+str(year)+"_"+str(patient))
            test = np.load(data_dir/f"ohiot1dm_test_{year}_{patient}.npy")
            test_y = test[:, -1]

            results.write(str(year)+'_'+str(patient)+',')

            lst = predict_lstmae(detector, test)

            results.write(str(accuracy_score(test_y, lst)*100) + ',' + str(precision_score(test_y, lst)) + ',' + str(recall_score(test_y, lst)) + ',' + str(f1_score(test_y, lst)) + '\n')
    results.close()
    ######################################################################################################################################
    # Samples (OE)
    os.makedirs(output_directory/"samples", exist_ok=True)
    results = open(output_directory/"samples"/"Results.csv", 'w')
    results.write('Patient,Accuracy,Precision,Recall,F1\n')

    # Fit one detector per random-sample training set once (independent of the test
    # patient), then reuse across patients -- identical results to refitting inside the
    # loop, but avoids retraining a neural net 12x per sample.
    sample_detectors = []
    for sample in range(10):
        train = np.load(data_dir/f"ohiot1dm_train_samples_{sample}.npy")
        sample_detectors.append(fit_lstmae(train))

    for year in [2018, 2020]:
        for patient in range(6):
            Accuracy = []
            Precision = []
            Recall = []
            F1 = []
            test = np.load(data_dir/f"ohiot1dm_test_{year}_{patient}.npy")
            test_y = test[:, -1]

            results.write(str(year)+'_'+str(patient)+',')
            for sample in range(10):
                print("Samples "+str(sample)+"\tPatient "+str(year)+"_"+str(patient))
                lst = predict_lstmae(sample_detectors[sample], test)

                Accuracy.insert(len(Accuracy), accuracy_score(test_y, lst)*100)
                Precision.insert(len(Precision), precision_score(test_y, lst))
                Recall.insert(len(Recall), recall_score(test_y, lst))
                F1.insert(len(F1), f1_score(test_y, lst))

            results.write(str(year)+'_'+str(patient)+','+str(np.mean(Accuracy)) + ',' + str(np.mean(Precision)) + ',' + str(np.mean(Recall)) + ',' + str(np.mean(F1)) + '\n')

    results.close()
    ######################################################################################################################################

    combine_results(output_directory)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(
		description="Run OhioT1DM script: python evaluate_lstmae.py <output_directory>",
		epilog="Example: python evaluate_lstmae.py output"
	)
	parser.add_argument("out_dir", nargs="?", default="output/defense_output/LSTMAE", help="Output directory")
	parser.add_argument("--data_dir", default="output/defense_dataset", help="Directory containing generated defense dataset .npy files")

	args = parser.parse_args()

	dataset_root = Path(__file__).resolve().parents[1]

	output_directory = dataset_root / args.out_dir
	data_directory = dataset_root / args.data_dir

	evaluate_lstmae(output_directory, data_directory)
