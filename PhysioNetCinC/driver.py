#!/usr/bin/env python

import numpy as np, os, sys
from get_sepsis_score import load_sepsis_model, get_sepsis_score
# N's start
import joblib
import argparse
from pathlib import Path
# N's end

def load_challenge_data(file):
    with open(file, 'r') as f:
        header = f.readline().strip()
        column_names = header.split('|')
        data = np.loadtxt(f, delimiter='|')

    # Ignore SepsisLabel column if present.
    if column_names[-1] == 'SepsisLabel':
        column_names = column_names[:-1]
        data = data[:, :-1]

    return data

def save_challenge_predictions(file, scores, labels):
    with open(file, 'w') as f:
        f.write('PredictedProbability|PredictedLabel\n')
        for (s, l) in zip(scores, labels):
            f.write('%g|%d\n' % (s, l))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run PhysioNetCinC script: python driver.py <input_directory> <output_directory>",
        epilog="Example: python driver.py input output"
    )
    parser.add_argument("data_dir", nargs="?", default="input", help="Directory containing dataset")
    parser.add_argument("out_dir", nargs="?", default="output", help="Output directory")
    parser.add_argument("--attack_type", choices=["URET", "FGSM", "PGD", "CW"], default=None, help="Override the adversarial attack type (defaults to pipeline_config.yml)")

    args = parser.parse_args()

    SCRIPT_DIR = Path(__file__).resolve().parent

    input_directory = Path(SCRIPT_DIR/args.data_dir)
    output_directory = Path(SCRIPT_DIR/args.out_dir)

    os.makedirs(output_directory, exist_ok=True)
    os.makedirs(output_directory / "Predictions" / "Benign", exist_ok=True)
    os.makedirs(output_directory / "Predictions" / "Adversarial", exist_ok=True)
    os.makedirs(output_directory / "Data" / "Benign", exist_ok=True)
    os.makedirs(output_directory / "Data" / "Adversarial", exist_ok=True)
    # N's end

    # Find files.
    files = []
    for f in os.listdir(str(input_directory)):
        if os.path.isfile(input_directory / f) and not f.lower().startswith('.') and f.lower().endswith('psv'):
            files.append(f)

    # Load model.
    model = load_sepsis_model()

    # Iterate over files.
    for f in files:
        # Load data.
        input_file = input_directory / f
        data = load_challenge_data(str(input_file))
        # Make predictions.
        num_rows = len(data)
        scores = np.zeros(num_rows)
        labels = np.zeros(num_rows)
        # N's start
        scores_adversarial = np.zeros(num_rows)
        labels_adversarial = np.zeros(num_rows)
        adversarial_data = np.random.rand(data.shape[0], data.shape[1])
        # N's end
        for t in range(num_rows):
            current_data = data[:t+1]
            current_score, current_label = get_sepsis_score(current_data, model, attack_type_override=args.attack_type)
            scores[t] = current_score
            labels[t] = current_label
            # N's start
            current_score_adversarial, current_label_adversarial = get_sepsis_score(current_data, model, adversary=True, adversarial_data=adversarial_data, attack_type_override=args.attack_type)
            scores_adversarial[t] = current_score_adversarial
            labels_adversarial[t] = current_label_adversarial
            # N's end

        # Save results.
        # N's start
        output_file = output_directory / "Predictions" / "Benign" / f
        # N's end
        save_challenge_predictions(str(output_file), scores, labels)

        # N's start
        output_file = output_directory / "Predictions" / "Adversarial" / f
        save_challenge_predictions(str(output_file), scores_adversarial, labels_adversarial)
        joblib.dump(data, output_directory / "Data" / "Benign" / f'{f[:-4]}.pkl')
        joblib.dump(adversarial_data, output_directory / "Data" / "Adversarial" / f'{f[:-4]}.pkl')
        # N's end