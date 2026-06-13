import ipywidgets as widgets
import sys
from pathlib import Path
import os
import importlib
import argparse

module_path='preprocessing/day_intervals_preproc'
if module_path not in sys.path:
    sys.path.append(module_path)

module_path='utils'
if module_path not in sys.path:
    sys.path.append(module_path)
    
module_path='preprocessing/hosp_module_preproc'
if module_path not in sys.path:
    sys.path.append(module_path)
    
module_path='model'
if module_path not in sys.path:
    sys.path.append(module_path)
#print(sys.path)
root_dir = os.path.dirname(os.path.abspath('UserInterface.ipynb'))
import day_intervals_cohort
from day_intervals_cohort import *

import day_intervals_cohort_v2
from day_intervals_cohort_v2 import *

import data_generation_icu

import data_generation
import evaluation

import feature_selection_hosp
from feature_selection_hosp import *

# import train
# from train import *


import ml_models
from ml_models import *

import dl_train
from dl_train import *

import tokenization
from tokenization import *


import behrt_train
from behrt_train import *

import feature_selection_icu
from feature_selection_icu import *
import fairness
import callibrate_output

importlib.reload(day_intervals_cohort)
import day_intervals_cohort
from day_intervals_cohort import *

importlib.reload(day_intervals_cohort_v2)
import day_intervals_cohort_v2
from day_intervals_cohort_v2 import *

importlib.reload(data_generation_icu)
import data_generation_icu
importlib.reload(data_generation)
import data_generation

importlib.reload(feature_selection_hosp)
import feature_selection_hosp
from feature_selection_hosp import *

importlib.reload(feature_selection_icu)
import feature_selection_icu
from feature_selection_icu import *

importlib.reload(tokenization)
import tokenization
from tokenization import *

importlib.reload(ml_models)
import ml_models
from ml_models import *

importlib.reload(dl_train)
import dl_train
from dl_train import *

importlib.reload(behrt_train)
import behrt_train
from behrt_train import *

importlib.reload(fairness)
import fairness

importlib.reload(callibrate_output)
import callibrate_output

importlib.reload(evaluation)
import evaluation



# N's start
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run MIMIC script: python run.py --out_dir <output_directory> --train_test <1/0>",
        epilog="Example: python run.py --out_dir output --train_test 1"
    )
    parser.add_argument("--out_dir", nargs="?", default="output", help="Output directory")
    parser.add_argument("--train_test", nargs="?", default="1", help="Train/test flag (1 for train, 0 for test)")
    parser.add_argument("--attack_type", choices=["URET", "FGSM", "PGD", "CW"], default=None, help="Override the adversarial attack type (defaults to pipeline_config.yml)")

    args = parser.parse_args()

    if args.train_test == '0':
        train = False
    elif args.train_test == '1':
        train = True
    else:
        raise ValueError("Train flag must be 0 or 1")

    SCRIPT_DIR = Path(__file__).resolve().parent

    output_dir = str(SCRIPT_DIR / args.out_dir)
    #def __init__(self,data_icu,diag_flag,proc_flag,out_flag,chart_flag,med_flag,lab_flag,model_type,k_fold,oversampling,model_name,train):
    #model=dl_train.DL_models(data_icu,diag_flag,proc_flag,out_flag,chart_flag,med_flag,False,radio_input6.value,cv,oversampling=radio_input8.value=='True',model_name='attn_icu_read',train=True)
    model=dl_train.DL_models(True,True,True,True,True,True,False,'Time-series LSTM',int(5),True,model_name='attn_icu_read',train=train, output_dir=output_dir, attack_type_override=args.attack_type)
