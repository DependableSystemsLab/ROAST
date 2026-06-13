#!/usr/bin/env python

import pandas as pd
import numpy as np
import glob
from torch.utils.data import Dataset
import copy
import os
import shutil
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

# N's start
from pathlib import Path
import joblib
import sys
import yaml

from URET.uret.utils.config import process_config_file
cf = str(Path(__file__).resolve().parent/"URET"/"brute.yml")

def feature_extractor(x):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    return torch.tensor(x, dtype=torch.float).to(device)

def mse(output, target):
	loss=torch.mean((output - target) ** 2)
	return loss
# N's end

class Model(nn.Module):
    # N's start
    def __init__(self,input_size,hidden_size,hidden_depth, device=None):
    # N's end
        super(Model, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_depth = hidden_depth
        # N's start
        self.device = device
        self.lstm = nn.LSTM(input_size, self.hidden_size, self.hidden_depth, bidirectional=False, dropout=0, device = self.device)
        self.fc1 = nn.Linear(1*self.hidden_size, 2, device = self.device)
        # N's end
    def forward(self, x):
        h0 = torch.zeros(self.hidden_depth * 1, 1, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.hidden_depth * 1, 1, self.hidden_size).to(x.device)

        x = x.view(-1, 1, self.input_size)
        x, (hn, cn) = self.lstm(x, (h0, c0))
        x = self.fc1(x)
        x = x.view(-1, 2)
        x = x[-1, :].view(1, 2)
        return x, F.softmax(x, dim=1)


# N's start
def get_sepsis_score(data, model, adversary=False, adversarial_data=None, attack_type_override=None):
# N's end
    data = pd.DataFrame(data)
    data = data.fillna(method='ffill')
    data = data.fillna(0).values
    # N's start
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    backcast_length = data.shape[0]
    nv = data.shape[1]

    norm = np.array([2.800e+02, 1.000e+02, 5.000e+01, 3.000e+02, 3.000e+02, 3.000e+02, 1.000e+02,
                     1.000e+02, 1.000e+02, 5.500e+01, 4.000e+03, 7.930e+00, 1.000e+02, 1.000e+02,
                     9.961e+03, 2.680e+02, 3.833e+03, 2.790e+01, 1.450e+02, 4.660e+01, 3.750e+01,
                     9.880e+02, 3.100e+01, 9.800e+00, 1.880e+01, 2.750e+01, 4.960e+01, 4.400e+02,
                     7.170e+01, 3.200e+01, 2.500e+02, 4.400e+02, 1.760e+03, 2.322e+03, 1.000e+02,
                     1.000e+00, 1.000e+00, 1.000e+00, 2.399e+01, 3.360e+02])

    config_path = str(Path(__file__).resolve().parent.parent / "pipeline_config.yml")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            attack_cfg = yaml.safe_load(f)
    else:
        attack_cfg = {"physionetcinc_attack_type": "URET"}
        
    attack_type = attack_type_override or attack_cfg.get("physionetcinc_attack_type", "URET")

    if adversary:
        if attack_type == "FGSM":
            # --- FGSM Attack ---
            data_normalized = data / norm
            eps = 0.02

            data_tensor = torch.tensor(data_normalized, dtype=torch.float32, device=device, requires_grad=True)
            model.train()
            logits, _ = model(data_tensor)
            
            target = torch.tensor([0], dtype=torch.long).to(device)
            loss = F.cross_entropy(logits, target)

            model.zero_grad()
            loss.backward()
            model.eval()

            with torch.no_grad():
                data_adv = data_tensor - eps * data_tensor.grad.sign()
                data_adv = torch.clamp(data_adv, min=0.0)

            data_normalized = data_adv.cpu().numpy().reshape(backcast_length, nv)
            adversarial_data[backcast_length - 1] = (data_normalized * norm)[backcast_length - 1]
            data = data_normalized
        elif attack_type == "PGD":
            # --- PGD Attack ---
            data_normalized = data / norm
            eps = 0.02
            alpha = 0.004
            num_steps = 10

            data_orig = torch.tensor(data_normalized, dtype=torch.float32, device=device)
            data_adv = data_orig.clone()

            model.train()
            for step in range(num_steps):
                data_adv = data_adv.clone().detach().requires_grad_(True)
                logits, _ = model(data_adv)

                target = torch.tensor([0], dtype=torch.long).to(device)
                loss = F.cross_entropy(logits, target)

                model.zero_grad()
                loss.backward()

                with torch.no_grad():
                    data_adv = data_adv - alpha * data_adv.grad.sign()
                    data_adv = torch.max(torch.min(data_adv, data_orig + eps), data_orig - eps)
                    data_adv = torch.clamp(data_adv, min=0.0)
            model.eval()

            data_normalized = data_adv.cpu().numpy().reshape(backcast_length, nv)
            adversarial_data[backcast_length - 1] = (data_normalized * norm)[backcast_length - 1]
            data = data_normalized
        elif attack_type == "CW":
            # --- C&W Attack ---
            data_normalized = data / norm
            c = 1.0
            lr = 0.01
            num_steps = 50

            data_orig = torch.tensor(data_normalized, dtype=torch.float32, device=device)
            delta = torch.zeros_like(data_orig, requires_grad=True)
            optimizer = optim.Adam([delta], lr=lr)

            model.train()
            for step in range(num_steps):
                data_adv = torch.clamp(data_orig + delta, min=0.0)
                logits, _ = model(data_adv)

                target = torch.tensor([0], dtype=torch.long).to(device)
                loss_attack = F.cross_entropy(logits, target)
                loss_perturb = torch.mean(delta ** 2)

                # attack succeeds when cross-entropy toward class 0 is small, so it's added (not subtracted)
                loss = loss_perturb + c * loss_attack

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            model.eval()

            data_adv = torch.clamp(data_orig + delta, min=0.0)
            data_normalized = data_adv.detach().cpu().numpy().reshape(backcast_length, nv)
            adversarial_data[backcast_length - 1] = (data_normalized * norm)[backcast_length - 1]
            data = data_normalized
        else:
            # --- URET Attack ---
            explorer = process_config_file(cf, model, feature_extractor=feature_extractor, input_processor_list=[])
            explorer.scoring_function = mse
            explore_params = [data, backcast_length, nv]
            allPatients_adversarial = np.array(explorer.explore(explore_params))
            
            if allPatients_adversarial[0] is None:
                allPatients_adversarial = data
                
            allPatients_adversarial = allPatients_adversarial.reshape(backcast_length, nv)
            adversarial_data[backcast_length - 1] = allPatients_adversarial[backcast_length - 1]
            data = allPatients_adversarial / norm
    else:
        # Benign mode
        data = data / norm

    # N's start
    data = torch.Tensor(data).float().to(device)
    # N's end
    threshold = 0.10
    _, probs = model(data)
    probs = probs[:, 1]
    argmax = (probs > threshold)
    # N's start
    return probs.cpu().data.numpy()[0], argmax.cpu().data.numpy()[0]
    # N's end

def load_sepsis_model():
    modelpth = str(Path(__file__).resolve().parent/'model_1561740354_cv_0_16_0.09277561709115473')
    # N's start
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = Model(40,100,2, device)
    # N's end
    model.load_state_dict(torch.load(modelpth))
    model.eval()
    return model
