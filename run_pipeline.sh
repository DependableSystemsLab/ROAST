#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/pipeline_config.yml"

# Check for yq and install if missing
if ! command -v yq &>/dev/null; then
    echo "'yq' not found. Installing..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS: use Homebrew
        if ! command -v brew &>/dev/null; then
            echo "Error: Homebrew not found. Install it from https://brew.sh then re-run."
            exit 1
        fi
        brew install yq
    else
        # Linux: download prebuilt binary
        YQ_BIN="/usr/local/bin/yq"
        sudo wget -q https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64 -O "$YQ_BIN"
        sudo chmod +x "$YQ_BIN"
    fi
    if ! command -v yq &>/dev/null; then
        echo "Error: Failed to install yq."
        exit 1
    fi
    echo "'yq' installed successfully."
fi

# ---------------------------
# Load flags from YAML config
# ---------------------------
RUN_OHIOT1DM_PRE=$(yq e '.ohiot1dm_preprocess' "$CONFIG_FILE")
RUN_OHIOT1DM_MOD=$(yq e '.ohiot1dm_model' "$CONFIG_FILE")
RUN_OHIOT1DM_DATASET=$(yq e '.ohiot1dm_dataset' "$CONFIG_FILE")

RUN_MIMIC_PRE=$(yq e '.mimic_preprocess' "$CONFIG_FILE")
RUN_MIMIC_MOD_TRAIN=$(yq e '.mimic_model_train' "$CONFIG_FILE")
RUN_MIMIC_MOD_TEST=$(yq e '.mimic_model_test' "$CONFIG_FILE")

RUN_PHYSIONET_MOD=$(yq e '.physionetcinc_model' "$CONFIG_FILE")
RUN_PHYSIONET_DATASET=$(yq e '.physionetcinc_dataset' "$CONFIG_FILE")

# Risk Profile and Cluster flags
RUN_OHIOT1DM_RISK=$(yq e '.ohiot1dm_risk_profile' "$CONFIG_FILE")
RUN_OHIOT1DM_CLUS=$(yq e '.ohiot1dm_cluster' "$CONFIG_FILE")
RUN_OHIOT1DM_CLUS_METHOD=$(yq e '.ohiot1dm_cluster_method' "$CONFIG_FILE")

RUN_MIMIC_RISK=$(yq e '.mimic_risk_profile' "$CONFIG_FILE")
RUN_MIMIC_CLUS=$(yq e '.mimic_cluster' "$CONFIG_FILE")
RUN_MIMIC_CLUS_METHOD=$(yq e '.mimic_cluster_method' "$CONFIG_FILE")

RUN_PHYS_RISK=$(yq e '.physionetcinc_risk_profile' "$CONFIG_FILE")
RUN_PHYS_CLUS=$(yq e '.physionetcinc_cluster' "$CONFIG_FILE")
RUN_PHYS_CLUS_METHOD=$(yq e '.physionetcinc_cluster_method' "$CONFIG_FILE")

# Generate Defense Dataset flags
RUN_OHIOT1DM_GEN_DEF=$(yq e '.ohiot1dm_generate_defense_datasets' "$CONFIG_FILE")
RUN_MIMIC_GEN_DEF=$(yq e '.mimic_generate_defense_datasets' "$CONFIG_FILE")
RUN_PHYS_GEN_DEF=$(yq e '.physionetcinc_generate_defense_datasets' "$CONFIG_FILE")

# Evaluate Defense flags
RUN_OHIOT1DM_EVAL_DEF=$(yq e '.ohiot1dm_evaluate_defense' "$CONFIG_FILE")
RUN_OHIOT1DM_DEF_TYPE=$(yq e '.ohiot1dm_defense_type' "$CONFIG_FILE")
RUN_OHIOT1DM_PLOT_DEF=$(yq e '.ohiot1dm_plot_defense_results' "$CONFIG_FILE")
RUN_MIMIC_EVAL_DEF=$(yq e '.mimic_evaluate_defense' "$CONFIG_FILE")
RUN_MIMIC_DEF_TYPE=$(yq e '.mimic_defense_type' "$CONFIG_FILE")
RUN_MIMIC_PLOT_DEF=$(yq e '.mimic_plot_defense_results' "$CONFIG_FILE")
RUN_PHYS_EVAL_DEF=$(yq e '.physionetcinc_evaluate_defense' "$CONFIG_FILE")
RUN_PHYS_DEF_TYPE=$(yq e '.physionetcinc_defense_type' "$CONFIG_FILE")
RUN_PHYS_PLOT_DEF=$(yq e '.physionetcinc_plot_defense_results' "$CONFIG_FILE")

# Cross-attack comparison figures flag (cross-dataset)
RUN_CROSS_ATTACK_PLOTS=$(yq e '.cross_attack_plots' "$CONFIG_FILE")

# Adversarial Attack Type flags
RUN_OHIOT1DM_ATTACK_TYPE=$(yq e '.ohiot1dm_attack_type' "$CONFIG_FILE")
RUN_OHIOT1DM_EVAL_ATTACK_TYPE=$(yq e '.ohiot1dm_eval_attack_type' "$CONFIG_FILE")
RUN_MIMIC_ATTACK_TYPE=$(yq e '.mimic_attack_type' "$CONFIG_FILE")
RUN_MIMIC_EVAL_ATTACK_TYPE=$(yq e '.mimic_eval_attack_type' "$CONFIG_FILE")
RUN_PHYS_ATTACK_TYPE=$(yq e '.physionetcinc_attack_type' "$CONFIG_FILE")
RUN_PHYS_EVAL_ATTACK_TYPE=$(yq e '.physionetcinc_eval_attack_type' "$CONFIG_FILE")

RUN_GLOBAL_RISK="false"
RUN_GLOBAL_CLUS="false"

# ---------------------------
# Parse command-line overrides
# ---------------------------
for arg in "$@"; do
    case $arg in
        --ohiot1dm_preprocess=*) RUN_OHIOT1DM_PRE="${arg#*=}" ;;
        --ohiot1dm_model=*)      RUN_OHIOT1DM_MOD="${arg#*=}" ;;
        --ohiot1dm_dataset=*)    RUN_OHIOT1DM_DATASET="${arg#*=}" ;;
        --mimic_preprocess=*)    RUN_MIMIC_PRE="${arg#*=}" ;;
        --mimic_model_train=*)   RUN_MIMIC_MOD_TRAIN="${arg#*=}" ;;
        --mimic_model_test=*)    RUN_MIMIC_MOD_TEST="${arg#*=}" ;;
        --physionetcinc_model=*)      RUN_PHYSIONET_MOD="${arg#*=}" ;;
        --physionetcinc_dataset=*)    RUN_PHYSIONET_DATASET="${arg#*=}" ;;
        --ohiot1dm_risk_profile=*)    RUN_OHIOT1DM_RISK="${arg#*=}" ;;
        --ohiot1dm_cluster=*)         RUN_OHIOT1DM_CLUS="${arg#*=}" ;;
        --ohiot1dm_cluster_method=*)  RUN_OHIOT1DM_CLUS_METHOD="${arg#*=}" ;;
        --mimic_risk_profile=*)       RUN_MIMIC_RISK="${arg#*=}" ;;
        --mimic_cluster=*)            RUN_MIMIC_CLUS="${arg#*=}" ;;
        --mimic_cluster_method=*)     RUN_MIMIC_CLUS_METHOD="${arg#*=}" ;;
        --physionetcinc_risk_profile=*) RUN_PHYS_RISK="${arg#*=}" ;;
        --physionetcinc_cluster=*)      RUN_PHYS_CLUS="${arg#*=}" ;;
        --physionetcinc_cluster_method=*) RUN_PHYS_CLUS_METHOD="${arg#*=}" ;;
        --ohiot1dm_generate_defense_datasets=*) RUN_OHIOT1DM_GEN_DEF="${arg#*=}" ;;
        --mimic_generate_defense_datasets=*) RUN_MIMIC_GEN_DEF="${arg#*=}" ;;
        --physionetcinc_generate_defense_datasets=*) RUN_PHYS_GEN_DEF="${arg#*=}" ;;
        --ohiot1dm_evaluate_defense=*) RUN_OHIOT1DM_EVAL_DEF="${arg#*=}" ;;
        --ohiot1dm_defense_type=*) RUN_OHIOT1DM_DEF_TYPE="${arg#*=}" ;;
        --ohiot1dm_plot_defense_results=*) RUN_OHIOT1DM_PLOT_DEF="${arg#*=}" ;;
        --mimic_evaluate_defense=*) RUN_MIMIC_EVAL_DEF="${arg#*=}" ;;
        --mimic_defense_type=*) RUN_MIMIC_DEF_TYPE="${arg#*=}" ;;
        --mimic_plot_defense_results=*) RUN_MIMIC_PLOT_DEF="${arg#*=}" ;;
        --physionetcinc_evaluate_defense=*) RUN_PHYS_EVAL_DEF="${arg#*=}" ;;
        --physionetcinc_defense_type=*) RUN_PHYS_DEF_TYPE="${arg#*=}" ;;
        --physionetcinc_plot_defense_results=*) RUN_PHYS_PLOT_DEF="${arg#*=}" ;;
        --ohiot1dm_attack_type=*) RUN_OHIOT1DM_ATTACK_TYPE="${arg#*=}" ;;
        --ohiot1dm_eval_attack_type=*) RUN_OHIOT1DM_EVAL_ATTACK_TYPE="${arg#*=}" ;;
        --mimic_attack_type=*) RUN_MIMIC_ATTACK_TYPE="${arg#*=}" ;;
        --mimic_eval_attack_type=*) RUN_MIMIC_EVAL_ATTACK_TYPE="${arg#*=}" ;;
        --physionetcinc_attack_type=*) RUN_PHYS_ATTACK_TYPE="${arg#*=}" ;;
        --physionetcinc_eval_attack_type=*) RUN_PHYS_EVAL_ATTACK_TYPE="${arg#*=}" ;;
        --cross_attack_plots=*) RUN_CROSS_ATTACK_PLOTS="${arg#*=}" ;;
        --risk_profile=*)             RUN_GLOBAL_RISK="${arg#*=}" ;;
        --cluster=*)                  RUN_GLOBAL_CLUS="${arg#*=}" ;;
        -h|--help)
            echo "Usage: $0 [--ohiot1dm_preprocess=true|false] [--ohiot1dm_model=true|false]"
            echo "       [--ohiot1dm_dataset=2018|2020|all]"
            echo "       [--mimic_preprocess=true|false] [--mimic_model_train=true|false] [--mimic_model_test=true|false]"
            echo "       [--physionetcinc_model=true|false]"
            echo "       [--physionetcinc_dataset=A|B|all]"
            echo "       [--risk_profile=true|false] [--cluster=true|false]"
            echo "       [--ohiot1dm_risk_profile=true|false] [--ohiot1dm_cluster=true|false] [--ohiot1dm_cluster_method=hierarchical|kmeans]"
            echo "       [--mimic_risk_profile=true|false] [--mimic_cluster=true|false] [--mimic_cluster_method=hierarchical|kmeans]"
            echo "       [--physionetcinc_risk_profile=true|false] [--physionetcinc_cluster=true|false] [--physionetcinc_cluster_method=hierarchical|kmeans]"
            echo "       [--ohiot1dm_attack_type=URET|FGSM|PGD|CW] [--ohiot1dm_eval_attack_type=same|URET|FGSM|PGD|CW]"
            echo "       [--mimic_attack_type=URET|FGSM|PGD|CW] [--mimic_eval_attack_type=same|URET|FGSM|PGD|CW]"
            echo "       [--physionetcinc_attack_type=URET|FGSM|PGD|CW] [--physionetcinc_eval_attack_type=same|URET|FGSM|PGD|CW]"
            echo "       [--ohiot1dm_generate_defense_datasets=true|false]"
            echo "       [--mimic_generate_defense_datasets=true|false]"
            echo "       [--physionetcinc_generate_defense_datasets=true|false]"
            echo "       [--ohiot1dm_evaluate_defense=true|false] [--ohiot1dm_defense_type=knn|oneclasssvm|deepsvdd|lstmae|madgan|all]"
            echo "       [--mimic_evaluate_defense=true|false] [--mimic_defense_type=knn|oneclasssvm|deepsvdd|lstmae|madgan|all]"
            echo "       [--physionetcinc_evaluate_defense=true|false] [--physionetcinc_defense_type=knn|oneclasssvm|deepsvdd|lstmae|madgan|all]"
            echo "       [--ohiot1dm_plot_defense_results=true|false]"
            echo "       [--mimic_plot_defense_results=true|false]"
            echo "       [--physionetcinc_plot_defense_results=true|false]"
            echo "       [--cross_attack_plots=true|false]"
            exit 0
            ;;
        *) echo "Unknown option: $arg"; exit 1 ;;
    esac
done

# Convert all flags to lowercase
RUN_OHIOT1DM_PRE=$(echo "$RUN_OHIOT1DM_PRE" | tr '[:upper:]' '[:lower:]')
RUN_OHIOT1DM_MOD=$(echo "$RUN_OHIOT1DM_MOD" | tr '[:upper:]' '[:lower:]')

RUN_MIMIC_PRE=$(echo "$RUN_MIMIC_PRE" | tr '[:upper:]' '[:lower:]')
RUN_MIMIC_MOD_TRAIN=$(echo "$RUN_MIMIC_MOD_TRAIN" | tr '[:upper:]' '[:lower:]')
RUN_MIMIC_MOD_TEST=$(echo "$RUN_MIMIC_MOD_TEST" | tr '[:upper:]' '[:lower:]')

RUN_PHYSIONET_MOD=$(echo "$RUN_PHYSIONET_MOD" | tr '[:upper:]' '[:lower:]')

RUN_OHIOT1DM_RISK=$(echo "$RUN_OHIOT1DM_RISK" | tr '[:upper:]' '[:lower:]')
RUN_OHIOT1DM_CLUS=$(echo "$RUN_OHIOT1DM_CLUS" | tr '[:upper:]' '[:lower:]')
RUN_MIMIC_RISK=$(echo "$RUN_MIMIC_RISK" | tr '[:upper:]' '[:lower:]')
RUN_MIMIC_CLUS=$(echo "$RUN_MIMIC_CLUS" | tr '[:upper:]' '[:lower:]')
RUN_PHYS_RISK=$(echo "$RUN_PHYS_RISK" | tr '[:upper:]' '[:lower:]')
RUN_PHYS_CLUS=$(echo "$RUN_PHYS_CLUS" | tr '[:upper:]' '[:lower:]')
RUN_OHIOT1DM_CLUS_METHOD=$(echo "$RUN_OHIOT1DM_CLUS_METHOD" | tr '[:upper:]' '[:lower:]')
RUN_MIMIC_CLUS_METHOD=$(echo "$RUN_MIMIC_CLUS_METHOD" | tr '[:upper:]' '[:lower:]')
RUN_PHYS_CLUS_METHOD=$(echo "$RUN_PHYS_CLUS_METHOD" | tr '[:upper:]' '[:lower:]')

RUN_OHIOT1DM_GEN_DEF=$(echo "$RUN_OHIOT1DM_GEN_DEF" | tr '[:upper:]' '[:lower:]')
RUN_MIMIC_GEN_DEF=$(echo "$RUN_MIMIC_GEN_DEF" | tr '[:upper:]' '[:lower:]')
RUN_PHYS_GEN_DEF=$(echo "$RUN_PHYS_GEN_DEF" | tr '[:upper:]' '[:lower:]')
RUN_OHIOT1DM_EVAL_DEF=$(echo "$RUN_OHIOT1DM_EVAL_DEF" | tr '[:upper:]' '[:lower:]')
RUN_OHIOT1DM_DEF_TYPE=$(echo "$RUN_OHIOT1DM_DEF_TYPE" | tr '[:upper:]' '[:lower:]')
RUN_OHIOT1DM_PLOT_DEF=$(echo "$RUN_OHIOT1DM_PLOT_DEF" | tr '[:upper:]' '[:lower:]')
RUN_MIMIC_EVAL_DEF=$(echo "$RUN_MIMIC_EVAL_DEF" | tr '[:upper:]' '[:lower:]')
RUN_MIMIC_DEF_TYPE=$(echo "$RUN_MIMIC_DEF_TYPE" | tr '[:upper:]' '[:lower:]')
RUN_MIMIC_PLOT_DEF=$(echo "$RUN_MIMIC_PLOT_DEF" | tr '[:upper:]' '[:lower:]')
RUN_PHYS_EVAL_DEF=$(echo "$RUN_PHYS_EVAL_DEF" | tr '[:upper:]' '[:lower:]')
RUN_PHYS_DEF_TYPE=$(echo "$RUN_PHYS_DEF_TYPE" | tr '[:upper:]' '[:lower:]')
RUN_PHYS_PLOT_DEF=$(echo "$RUN_PHYS_PLOT_DEF" | tr '[:upper:]' '[:lower:]')

RUN_CROSS_ATTACK_PLOTS=$(echo "$RUN_CROSS_ATTACK_PLOTS" | tr '[:upper:]' '[:lower:]')

RUN_GLOBAL_RISK=$(echo "$RUN_GLOBAL_RISK" | tr '[:upper:]' '[:lower:]')
RUN_GLOBAL_CLUS=$(echo "$RUN_GLOBAL_CLUS" | tr '[:upper:]' '[:lower:]')

# Apply global overrides
if [ "$RUN_GLOBAL_RISK" = "true" ]; then
    RUN_OHIOT1DM_RISK="true"
    RUN_MIMIC_RISK="true"
    RUN_PHYS_RISK="true"
fi

if [ "$RUN_GLOBAL_CLUS" = "true" ]; then
    RUN_OHIOT1DM_CLUS="true"
    RUN_MIMIC_CLUS="true"
    RUN_PHYS_CLUS="true"
fi

# ---------------------------
# Compute attack-type namespaces
# ---------------------------
# NAMESPACE: output/<NAMESPACE>/{defense_dataset,defense_output} for the
# generate/evaluate/plot stages. DATA_ATTACK: which output/<ATTACK_TYPE> directory
# holds the benign/adversarial model outputs to source for defense generation.
# ATTACK_TYPES: the set of attack types the model stage must produce output for.
OHIOT1DM_NAMESPACE="$RUN_OHIOT1DM_ATTACK_TYPE"
OHIOT1DM_DATA_ATTACK="$RUN_OHIOT1DM_ATTACK_TYPE"
OHIOT1DM_ATTACK_TYPES=("$RUN_OHIOT1DM_ATTACK_TYPE")
if [ "$RUN_OHIOT1DM_EVAL_ATTACK_TYPE" != "same" ] && [ "$RUN_OHIOT1DM_EVAL_ATTACK_TYPE" != "$RUN_OHIOT1DM_ATTACK_TYPE" ]; then
    OHIOT1DM_NAMESPACE="${RUN_OHIOT1DM_ATTACK_TYPE}_to_${RUN_OHIOT1DM_EVAL_ATTACK_TYPE}"
    OHIOT1DM_DATA_ATTACK="$RUN_OHIOT1DM_EVAL_ATTACK_TYPE"
    OHIOT1DM_ATTACK_TYPES+=("$RUN_OHIOT1DM_EVAL_ATTACK_TYPE")
fi

MIMIC_NAMESPACE="$RUN_MIMIC_ATTACK_TYPE"
MIMIC_DATA_ATTACK="$RUN_MIMIC_ATTACK_TYPE"
MIMIC_ATTACK_TYPES=("$RUN_MIMIC_ATTACK_TYPE")
if [ "$RUN_MIMIC_EVAL_ATTACK_TYPE" != "same" ] && [ "$RUN_MIMIC_EVAL_ATTACK_TYPE" != "$RUN_MIMIC_ATTACK_TYPE" ]; then
    MIMIC_NAMESPACE="${RUN_MIMIC_ATTACK_TYPE}_to_${RUN_MIMIC_EVAL_ATTACK_TYPE}"
    MIMIC_DATA_ATTACK="$RUN_MIMIC_EVAL_ATTACK_TYPE"
    MIMIC_ATTACK_TYPES+=("$RUN_MIMIC_EVAL_ATTACK_TYPE")
fi

PHYS_NAMESPACE="$RUN_PHYS_ATTACK_TYPE"
PHYS_DATA_ATTACK="$RUN_PHYS_ATTACK_TYPE"
PHYS_ATTACK_TYPES=("$RUN_PHYS_ATTACK_TYPE")
if [ "$RUN_PHYS_EVAL_ATTACK_TYPE" != "same" ] && [ "$RUN_PHYS_EVAL_ATTACK_TYPE" != "$RUN_PHYS_ATTACK_TYPE" ]; then
    PHYS_NAMESPACE="${RUN_PHYS_ATTACK_TYPE}_to_${RUN_PHYS_EVAL_ATTACK_TYPE}"
    PHYS_DATA_ATTACK="$RUN_PHYS_EVAL_ATTACK_TYPE"
    PHYS_ATTACK_TYPES+=("$RUN_PHYS_EVAL_ATTACK_TYPE")
fi

# ---------------------------
# Helper to run scripts inside environments
# ---------------------------
run_in_env_path() {
    local env_path=$1
    local work_dir=$2
    local cmd=$3

    echo ">>> Activating environment $env_path and running script in $work_dir ..."
    source "$SCRIPT_DIR/$env_path/bin/activate"
    cd "$SCRIPT_DIR/$work_dir"
    eval "$cmd"
    cd "$SCRIPT_DIR"
    deactivate || true
}

run_in_env() {
    local env_dir=$1
    local target_dir=$2
    local cmd=$3
    run_in_env_path "$target_dir/$env_dir" "$target_dir" "$cmd"
}

run_defense_eval_scripts() {
    local env_dir=$1
    local target_dir=$2
    local dataset_key=$3
    local defense_type=$4
    local namespace=$5

    local data_dir_arg="output/${namespace}/defense_dataset"
    local defense_out_base="output/${namespace}/defense_output"

    local defense_types=()
    if [ "$defense_type" = "all" ]; then
        defense_types=("knn" "oneclasssvm" "deepsvdd" "lstmae" "madgan")
    elif [ "$defense_type" = "knn" ] || [ "$defense_type" = "oneclasssvm" ] || [ "$defense_type" = "deepsvdd" ] || [ "$defense_type" = "lstmae" ] || [ "$defense_type" = "madgan" ]; then
        defense_types=("$defense_type")
    else
        echo "Error: Invalid defense type '$defense_type' for $dataset_key."
        echo "Valid values are: knn, oneclasssvm, deepsvdd, lstmae, madgan, all"
        exit 1
    fi

    for dtype in "${defense_types[@]}"; do
        if [ "$dtype" = "madgan" ]; then
            local madgan_script="defenses/MAD-GAN/evaluate_madgan.py"
            local madgan_work_dir="$target_dir/defenses/MAD-GAN"
            local madgan_env_path="$madgan_work_dir/venv_madgan"

            if [ ! -f "$SCRIPT_DIR/$target_dir/$madgan_script" ]; then
                echo "Error: Defense evaluation script not found: $target_dir/$madgan_script"
                exit 1
            fi
            if [ ! -d "$SCRIPT_DIR/$madgan_env_path" ]; then
                echo "Error: MAD-GAN environment not found: $madgan_env_path"
                exit 1
            fi
            run_in_env_path "$madgan_env_path" "$madgan_work_dir" "python evaluate_madgan.py ${defense_out_base}/MADGAN --data_dir=${data_dir_arg}"
        elif [ "$dtype" = "knn" ]; then
            local knn_script="defenses/evaluate_knn.py"
            if [ ! -f "$SCRIPT_DIR/$target_dir/$knn_script" ]; then
                echo "Error: Defense evaluation script not found: $target_dir/$knn_script"
                exit 1
            fi
            run_in_env "$env_dir" "$target_dir" "python $knn_script ${defense_out_base}/KNN --data_dir=${data_dir_arg}"
        elif [ "$dtype" = "oneclasssvm" ]; then
            local ocsvm_script="defenses/evaluate_oneclasssvm.py"
            if [ ! -f "$SCRIPT_DIR/$target_dir/$ocsvm_script" ]; then
                echo "Error: Defense evaluation script not found: $target_dir/$ocsvm_script"
                exit 1
            fi
            run_in_env "$env_dir" "$target_dir" "python $ocsvm_script ${defense_out_base}/OneClassSVM --data_dir=${data_dir_arg}"
        elif [ "$dtype" = "deepsvdd" ]; then
            local deepsvdd_script="defenses/evaluate_deepsvdd.py"
            if [ ! -f "$SCRIPT_DIR/$target_dir/$deepsvdd_script" ]; then
                echo "Error: Defense evaluation script not found: $target_dir/$deepsvdd_script"
                exit 1
            fi
            run_in_env "$env_dir" "$target_dir" "python $deepsvdd_script ${defense_out_base}/DeepSVDD --data_dir=${data_dir_arg}"
        elif [ "$dtype" = "lstmae" ]; then
            local lstmae_script="defenses/evaluate_lstmae.py"
            if [ ! -f "$SCRIPT_DIR/$target_dir/$lstmae_script" ]; then
                echo "Error: Defense evaluation script not found: $target_dir/$lstmae_script"
                exit 1
            fi
            run_in_env "$env_dir" "$target_dir" "python $lstmae_script ${defense_out_base}/LSTMAE --data_dir=${data_dir_arg}"
        fi
    done
}

# ---------------------------
# Run pipeline stages
# ---------------------------
if [ "$RUN_OHIOT1DM_PRE" = "true" ]; then
    echo "Preprocessing OhioT1DM dataset..."
    run_in_env "venv_ohiot1dm" "OhioT1DM" "python convert_data.py data/raw data/processed"
fi

if [ "$RUN_OHIOT1DM_MOD" = "true" ]; then
    for ATYPE in "${OHIOT1DM_ATTACK_TYPES[@]}"; do
        if [ "$RUN_OHIOT1DM_DATASET" = "all" ]; then
            echo "Running OhioT1DM model for all datasets (2018 and 2020) with attack type ${ATYPE}..."
            run_in_env "venv_ohiot1dm" "OhioT1DM" "python drtf.py data/processed/2018data output/${ATYPE}/2018 --attack_type=${ATYPE}"
            run_in_env "venv_ohiot1dm" "OhioT1DM" "python drtf.py data/processed/2020data output/${ATYPE}/2020 --attack_type=${ATYPE}"
        else
            echo "Running OhioT1DM model for dataset ${RUN_OHIOT1DM_DATASET} with attack type ${ATYPE}..."
            run_in_env "venv_ohiot1dm" "OhioT1DM" "python drtf.py data/processed/${RUN_OHIOT1DM_DATASET}data output/${ATYPE}/${RUN_OHIOT1DM_DATASET} --attack_type=${ATYPE}"
        fi
    done
fi

if [ "$RUN_MIMIC_PRE" = "true" ]; then
    echo "Preprocessing MIMIC dataset..."
    run_in_env "venv_mimic" "MIMIC" "jupyter nbconvert --execute --inplace mainPipeline.ipynb"
fi

if [ "$RUN_MIMIC_MOD_TRAIN" = "true" ]; then
    echo "Running MIMIC model training..."
    run_in_env "venv_mimic" "MIMIC" "python run.py --train_test 1"
fi

if [ "$RUN_MIMIC_MOD_TEST" = "true" ]; then
    for ATYPE in "${MIMIC_ATTACK_TYPES[@]}"; do
        echo "Running MIMIC model testing with attack type ${ATYPE}..."
        run_in_env "venv_mimic" "MIMIC" "python run.py --train_test 0 --out_dir output/${ATYPE} --attack_type=${ATYPE}"
    done
fi

if [ "$RUN_PHYSIONET_MOD" = "true" ]; then
    for ATYPE in "${PHYS_ATTACK_TYPES[@]}"; do
        if [ "$RUN_PHYSIONET_DATASET" = "all" ]; then
            echo "Running PhysioNetCinC model for all datasets (A and B) with attack type ${ATYPE}..."
            run_in_env "venv_physionetcinc" "PhysioNetCinC" "python driver.py input/training_setA output/${ATYPE}/training_setA --attack_type=${ATYPE}"
            run_in_env "venv_physionetcinc" "PhysioNetCinC" "python driver.py input/training_setB output/${ATYPE}/training_setB --attack_type=${ATYPE}"
        else
            echo "Running PhysioNetCinC model for dataset ${RUN_PHYSIONET_DATASET} with attack type ${ATYPE}..."
            run_in_env "venv_physionetcinc" "PhysioNetCinC" "python driver.py input/training_set${RUN_PHYSIONET_DATASET} output/${ATYPE}/training_set${RUN_PHYSIONET_DATASET} --attack_type=${ATYPE}"
        fi
    done
fi

# ---------------------------
# Risk Profiling and Clustering
# ---------------------------
if [ "$RUN_OHIOT1DM_RISK" = "true" ]; then
    echo "Running Risk Profile for OhioT1DM (attack type ${RUN_OHIOT1DM_ATTACK_TYPE})..."
    run_in_env "venv_ohiot1dm" "OhioT1DM" "python risk_profile.py output/${RUN_OHIOT1DM_ATTACK_TYPE}"
fi

if [ "$RUN_OHIOT1DM_CLUS" = "true" ]; then
    echo "Running Clustering for OhioT1DM (attack type ${RUN_OHIOT1DM_ATTACK_TYPE})..."
    if [ "$RUN_OHIOT1DM_CLUS_METHOD" = "kmeans" ]; then
        run_in_env "venv_ohiot1dm" "OhioT1DM" "python kmeans_cluster.py output/${RUN_OHIOT1DM_ATTACK_TYPE} output/${RUN_OHIOT1DM_ATTACK_TYPE}/cluster_output"
    else
        run_in_env "venv_ohiot1dm" "OhioT1DM" "python hierarchical_cluster.py output/${RUN_OHIOT1DM_ATTACK_TYPE} output/${RUN_OHIOT1DM_ATTACK_TYPE}/cluster_output"
    fi
fi

if [ "$RUN_MIMIC_RISK" = "true" ]; then
    echo "Running Risk Profile for MIMIC (attack type ${RUN_MIMIC_ATTACK_TYPE})..."
    run_in_env "venv_mimic" "MIMIC" "python risk_profile.py output/${RUN_MIMIC_ATTACK_TYPE}"
fi

if [ "$RUN_MIMIC_CLUS" = "true" ]; then
    echo "Running Clustering for MIMIC (attack type ${RUN_MIMIC_ATTACK_TYPE})..."
    if [ "$RUN_MIMIC_CLUS_METHOD" = "kmeans" ]; then
        run_in_env "venv_mimic" "MIMIC" "python kmeans_cluster.py output/${RUN_MIMIC_ATTACK_TYPE} output/${RUN_MIMIC_ATTACK_TYPE}/cluster_output"
    else
        run_in_env "venv_mimic" "MIMIC" "python hierarchical_cluster.py output/${RUN_MIMIC_ATTACK_TYPE} output/${RUN_MIMIC_ATTACK_TYPE}/cluster_output"
    fi
fi

if [ "$RUN_PHYS_RISK" = "true" ]; then
    echo "Running Risk Profile for PhysioNetCinC (attack type ${RUN_PHYS_ATTACK_TYPE})..."
    run_in_env "venv_physionetcinc" "PhysioNetCinC" "python risk_profile.py output/${RUN_PHYS_ATTACK_TYPE}"
fi

if [ "$RUN_PHYS_CLUS" = "true" ]; then
    echo "Running Clustering for PhysioNetCinC (attack type ${RUN_PHYS_ATTACK_TYPE})..."
    if [ "$RUN_PHYS_CLUS_METHOD" = "kmeans" ]; then
        run_in_env "venv_physionetcinc" "PhysioNetCinC" "python kmeans_cluster.py output/${RUN_PHYS_ATTACK_TYPE} output/${RUN_PHYS_ATTACK_TYPE}/cluster_output"
    else
        run_in_env "venv_physionetcinc" "PhysioNetCinC" "python hierarchical_cluster.py output/${RUN_PHYS_ATTACK_TYPE} output/${RUN_PHYS_ATTACK_TYPE}/cluster_output"
    fi
fi

# ---------------------------
# Generate Defense Datasets
# ---------------------------
if [ "$RUN_OHIOT1DM_GEN_DEF" = "true" ]; then
    echo "Generating Defense Dataset for OhioT1DM (namespace ${OHIOT1DM_NAMESPACE})..."
    run_in_env "venv_ohiot1dm" "OhioT1DM" "python generate_defense_dataset.py output/${RUN_OHIOT1DM_ATTACK_TYPE}/cluster_output output/${OHIOT1DM_NAMESPACE}/defense_dataset --data_dir=output/${OHIOT1DM_DATA_ATTACK}"
fi

if [ "$RUN_MIMIC_GEN_DEF" = "true" ]; then
    echo "Generating Defense Dataset for MIMIC (namespace ${MIMIC_NAMESPACE})..."
    run_in_env "venv_mimic" "MIMIC" "python generate_defense_dataset.py output/${RUN_MIMIC_ATTACK_TYPE}/cluster_output output/${MIMIC_NAMESPACE}/defense_dataset --data_dir=output/${MIMIC_DATA_ATTACK}"
fi

if [ "$RUN_PHYS_GEN_DEF" = "true" ]; then
    echo "Generating Defense Dataset for PhysioNetCinC (namespace ${PHYS_NAMESPACE})..."
    run_in_env "venv_physionetcinc" "PhysioNetCinC" "python generate_defense_dataset.py output/${RUN_PHYS_ATTACK_TYPE}/cluster_output output/${PHYS_NAMESPACE}/defense_dataset --data_dir=output/${PHYS_DATA_ATTACK}"
fi

# ---------------------------
# Evaluate Defenses
# ---------------------------
if [ "$RUN_OHIOT1DM_EVAL_DEF" = "true" ]; then
    echo "Evaluating defenses for OhioT1DM (${RUN_OHIOT1DM_DEF_TYPE}, namespace ${OHIOT1DM_NAMESPACE})..."
    run_defense_eval_scripts "venv_ohiot1dm" "OhioT1DM" "ohiot1dm" "$RUN_OHIOT1DM_DEF_TYPE" "$OHIOT1DM_NAMESPACE"
fi

if [ "$RUN_MIMIC_EVAL_DEF" = "true" ]; then
    echo "Evaluating defenses for MIMIC (${RUN_MIMIC_DEF_TYPE}, namespace ${MIMIC_NAMESPACE})..."
    run_defense_eval_scripts "venv_mimic" "MIMIC" "mimic" "$RUN_MIMIC_DEF_TYPE" "$MIMIC_NAMESPACE"
fi

if [ "$RUN_PHYS_EVAL_DEF" = "true" ]; then
    echo "Evaluating defenses for PhysioNetCinC (${RUN_PHYS_DEF_TYPE}, namespace ${PHYS_NAMESPACE})..."
    run_defense_eval_scripts "venv_physionetcinc" "PhysioNetCinC" "physionetcinc" "$RUN_PHYS_DEF_TYPE" "$PHYS_NAMESPACE"
fi

# ---------------------------
# Plot Defense Results
# ---------------------------
if [ "$RUN_OHIOT1DM_PLOT_DEF" = "true" ]; then
    echo "Plotting defense results for OhioT1DM (namespace ${OHIOT1DM_NAMESPACE})..."
    run_in_env_path "OhioT1DM/venv_ohiot1dm" "." "python plot_defense_results.py OhioT1DM OhioT1DM/output/${OHIOT1DM_NAMESPACE}/defense_output"
fi

if [ "$RUN_MIMIC_PLOT_DEF" = "true" ]; then
    echo "Plotting defense results for MIMIC (namespace ${MIMIC_NAMESPACE})..."
    run_in_env_path "MIMIC/venv_mimic" "." "python plot_defense_results.py MIMIC MIMIC/output/${MIMIC_NAMESPACE}/defense_output"
fi

if [ "$RUN_PHYS_PLOT_DEF" = "true" ]; then
    echo "Plotting defense results for PhysioNetCinC (namespace ${PHYS_NAMESPACE})..."
    run_in_env_path "PhysioNetCinC/venv_physionetcinc" "." "python plot_defense_results.py PhysioNetCinC PhysioNetCinC/output/${PHYS_NAMESPACE}/defense_output"
fi

# ---------------------------
# Cross-attack comparison figures (cross-dataset)
# ---------------------------
# Auto-discovers all namespaced attack-type / cross-attack outputs across the three
# datasets and emits comparison figures + summary CSVs under cross_attack_figures/. Runs
# once from the repo root inside OhioT1DM's venv (which provides matplotlib and scipy).
if [ "$RUN_CROSS_ATTACK_PLOTS" = "true" ]; then
    echo "Generating cross-attack comparison figures (attack-type comparison, cross-attack heatmaps, box plots)..."
    run_in_env_path "OhioT1DM/venv_ohiot1dm" "." "python plot_cross_attack_results.py"
fi

echo "Pipeline completed successfully."
