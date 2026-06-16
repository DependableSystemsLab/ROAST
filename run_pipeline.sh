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

# Per-cluster (less/more-vulnerable test) reporting flag (cross-dataset)
RUN_PER_CLUSTER_REPORT=$(yq e '.per_cluster_report' "$CONFIG_FILE")

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
        --per_cluster_report=*) RUN_PER_CLUSTER_REPORT="${arg#*=}" ;;
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
            echo "       [--ohiot1dm_attack_type=URET|FGSM|PGD|CW|all] [--ohiot1dm_eval_attack_type=same|URET|FGSM|PGD|CW|all]"
            echo "       [--mimic_attack_type=URET|FGSM|PGD|CW|all] [--mimic_eval_attack_type=same|URET|FGSM|PGD|CW|all]"
            echo "       [--physionetcinc_attack_type=URET|FGSM|PGD|CW|all] [--physionetcinc_eval_attack_type=same|URET|FGSM|PGD|CW|all]"
            echo "       (attack_type=all + eval_attack_type=all sweeps every off-diagonal cross-attack pair)"
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
            echo "       [--per_cluster_report=true|false]"
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
RUN_PER_CLUSTER_REPORT=$(echo "$RUN_PER_CLUSTER_REPORT" | tr '[:upper:]' '[:lower:]')

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
# Compute attack-type namespaces (supports cross-attack sweeps)
# ---------------------------
# Each dataset expands (attack_type x eval_attack_type) into a set of
# (risk, namespace, data_attack) triples that the generate/evaluate/plot stages loop
# over, plus the de-duplicated risk set (for risk-profiling/clustering) and model
# attack-type set. Config values:
#   - attack_type:      URET | FGSM | PGD | CW | all   ("all" = every attack as risk)
#   - eval_attack_type: same | URET | FGSM | PGD | CW | all
#       "same" -> diagonal (namespace=<risk>); "all" -> every off-diagonal eval.
# So attack_type=all + eval_attack_type=all generates the full off-diagonal sweep.
# namespace = <risk> (same-attack) or <risk>_to_<eval> (cross); data_attack = the
# attack whose output/<ATTACK_TYPE> supplies the benign/adversarial defense data.
ATTACK_ALL=(URET FGSM PGD CW)

risk_set() {   # $1=attack_type config -> risk attack(s), one per line
    if [ "$1" = "all" ]; then printf '%s\n' "${ATTACK_ALL[@]}"; else echo "$1"; fi
}

eval_set() {   # $1=risk $2=eval_attack_type config -> eval attack(s), one per line
    local risk=$1 evalcfg=$2 a
    if [ "$evalcfg" = "all" ]; then
        for a in "${ATTACK_ALL[@]}"; do [ "$a" != "$risk" ] && echo "$a"; done
    elif [ "$evalcfg" = "same" ]; then
        echo "$risk"
    else
        echo "$evalcfg"
    fi
}

expand_cross() {   # $1=attack cfg $2=eval cfg -> lines "<risk> <namespace> <data_attack>"
    local r e
    for r in $(risk_set "$1"); do
        for e in $(eval_set "$r" "$2"); do
            if [ "$e" = "$r" ]; then echo "$r $r $r"; else echo "$r ${r}_to_${e} $e"; fi
        done
    done
}

populate_cross() {   # $1=PREFIX $2=attack cfg $3=eval cfg
    # Fills <PREFIX>_{NS_RISKS,NAMESPACES,DATA_ATTACKS} (parallel) + de-duped
    # <PREFIX>_{RISKS,ATTACK_TYPES}.
    local p=$1 r ns da
    eval "${p}_NS_RISKS=(); ${p}_NAMESPACES=(); ${p}_DATA_ATTACKS=(); ${p}_RISKS=(); ${p}_ATTACK_TYPES=()"
    while read -r r ns da; do
        eval "${p}_NS_RISKS+=(\"$r\"); ${p}_NAMESPACES+=(\"$ns\"); ${p}_DATA_ATTACKS+=(\"$da\")"
        eval "case \" \${${p}_RISKS[*]} \" in *\" $r \"*) ;; *) ${p}_RISKS+=(\"$r\");; esac"
        eval "case \" \${${p}_ATTACK_TYPES[*]} \" in *\" $r \"*) ;; *) ${p}_ATTACK_TYPES+=(\"$r\");; esac"
        eval "case \" \${${p}_ATTACK_TYPES[*]} \" in *\" $da \"*) ;; *) ${p}_ATTACK_TYPES+=(\"$da\");; esac"
    done < <(expand_cross "$2" "$3")
    return 0
}

populate_cross OHIOT1DM "$RUN_OHIOT1DM_ATTACK_TYPE" "$RUN_OHIOT1DM_EVAL_ATTACK_TYPE"
populate_cross MIMIC "$RUN_MIMIC_ATTACK_TYPE" "$RUN_MIMIC_EVAL_ATTACK_TYPE"
populate_cross PHYS "$RUN_PHYS_ATTACK_TYPE" "$RUN_PHYS_EVAL_ATTACK_TYPE"

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
    local variant=$6   # optional: "" (full), "less", or "more" for per-cluster test sets

    local suffix=""
    if [ -n "$variant" ]; then
        suffix="_${variant}"
    fi
    local data_dir_arg="output/${namespace}/defense_dataset${suffix}"
    local defense_out_base="output/${namespace}/defense_output${suffix}"

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
    for RISK in "${OHIOT1DM_RISKS[@]}"; do
        echo "Running Risk Profile for OhioT1DM (attack type ${RISK})..."
        run_in_env "venv_ohiot1dm" "OhioT1DM" "python risk_profile.py output/${RISK}"
    done
fi

if [ "$RUN_OHIOT1DM_CLUS" = "true" ]; then
    for RISK in "${OHIOT1DM_RISKS[@]}"; do
        echo "Running Clustering for OhioT1DM (attack type ${RISK})..."
        if [ "$RUN_OHIOT1DM_CLUS_METHOD" = "kmeans" ]; then
            run_in_env "venv_ohiot1dm" "OhioT1DM" "python kmeans_cluster.py output/${RISK} output/${RISK}/cluster_output"
        else
            run_in_env "venv_ohiot1dm" "OhioT1DM" "python hierarchical_cluster.py output/${RISK} output/${RISK}/cluster_output"
        fi
    done
fi

if [ "$RUN_MIMIC_RISK" = "true" ]; then
    for RISK in "${MIMIC_RISKS[@]}"; do
        echo "Running Risk Profile for MIMIC (attack type ${RISK})..."
        run_in_env "venv_mimic" "MIMIC" "python risk_profile.py output/${RISK}"
    done
fi

if [ "$RUN_MIMIC_CLUS" = "true" ]; then
    for RISK in "${MIMIC_RISKS[@]}"; do
        echo "Running Clustering for MIMIC (attack type ${RISK})..."
        if [ "$RUN_MIMIC_CLUS_METHOD" = "kmeans" ]; then
            run_in_env "venv_mimic" "MIMIC" "python kmeans_cluster.py output/${RISK} output/${RISK}/cluster_output"
        else
            run_in_env "venv_mimic" "MIMIC" "python hierarchical_cluster.py output/${RISK} output/${RISK}/cluster_output"
        fi
    done
fi

if [ "$RUN_PHYS_RISK" = "true" ]; then
    for RISK in "${PHYS_RISKS[@]}"; do
        echo "Running Risk Profile for PhysioNetCinC (attack type ${RISK})..."
        run_in_env "venv_physionetcinc" "PhysioNetCinC" "python risk_profile.py output/${RISK}"
    done
fi

if [ "$RUN_PHYS_CLUS" = "true" ]; then
    for RISK in "${PHYS_RISKS[@]}"; do
        echo "Running Clustering for PhysioNetCinC (attack type ${RISK})..."
        if [ "$RUN_PHYS_CLUS_METHOD" = "kmeans" ]; then
            run_in_env "venv_physionetcinc" "PhysioNetCinC" "python kmeans_cluster.py output/${RISK} output/${RISK}/cluster_output"
        else
            run_in_env "venv_physionetcinc" "PhysioNetCinC" "python hierarchical_cluster.py output/${RISK} output/${RISK}/cluster_output"
        fi
    done
fi

# ---------------------------
# Generate Defense Datasets
# ---------------------------
if [ "$RUN_OHIOT1DM_GEN_DEF" = "true" ]; then
    for i in "${!OHIOT1DM_NAMESPACES[@]}"; do
        NS="${OHIOT1DM_NAMESPACES[$i]}"; DA="${OHIOT1DM_DATA_ATTACKS[$i]}"; RISK="${OHIOT1DM_NS_RISKS[$i]}"
        echo "Generating Defense Dataset for OhioT1DM (namespace ${NS})..."
        run_in_env "venv_ohiot1dm" "OhioT1DM" "python generate_defense_dataset.py output/${RISK}/cluster_output output/${NS}/defense_dataset --data_dir=output/${DA}"
    done
fi

if [ "$RUN_MIMIC_GEN_DEF" = "true" ]; then
    for i in "${!MIMIC_NAMESPACES[@]}"; do
        NS="${MIMIC_NAMESPACES[$i]}"; DA="${MIMIC_DATA_ATTACKS[$i]}"; RISK="${MIMIC_NS_RISKS[$i]}"
        echo "Generating Defense Dataset for MIMIC (namespace ${NS})..."
        run_in_env "venv_mimic" "MIMIC" "python generate_defense_dataset.py output/${RISK}/cluster_output output/${NS}/defense_dataset --data_dir=output/${DA}"
    done
fi

if [ "$RUN_PHYS_GEN_DEF" = "true" ]; then
    for i in "${!PHYS_NAMESPACES[@]}"; do
        NS="${PHYS_NAMESPACES[$i]}"; DA="${PHYS_DATA_ATTACKS[$i]}"; RISK="${PHYS_NS_RISKS[$i]}"
        echo "Generating Defense Dataset for PhysioNetCinC (namespace ${NS})..."
        run_in_env "venv_physionetcinc" "PhysioNetCinC" "python generate_defense_dataset.py output/${RISK}/cluster_output output/${NS}/defense_dataset --data_dir=output/${DA}"
    done
fi

# ---------------------------
# Evaluate Defenses
# ---------------------------
if [ "$RUN_OHIOT1DM_EVAL_DEF" = "true" ]; then
    for NS in "${OHIOT1DM_NAMESPACES[@]}"; do
        echo "Evaluating defenses for OhioT1DM (${RUN_OHIOT1DM_DEF_TYPE}, namespace ${NS})..."
        run_defense_eval_scripts "venv_ohiot1dm" "OhioT1DM" "ohiot1dm" "$RUN_OHIOT1DM_DEF_TYPE" "$NS"
    done
fi

if [ "$RUN_MIMIC_EVAL_DEF" = "true" ]; then
    for NS in "${MIMIC_NAMESPACES[@]}"; do
        echo "Evaluating defenses for MIMIC (${RUN_MIMIC_DEF_TYPE}, namespace ${NS})..."
        run_defense_eval_scripts "venv_mimic" "MIMIC" "mimic" "$RUN_MIMIC_DEF_TYPE" "$NS"
    done
fi

if [ "$RUN_PHYS_EVAL_DEF" = "true" ]; then
    for NS in "${PHYS_NAMESPACES[@]}"; do
        echo "Evaluating defenses for PhysioNetCinC (${RUN_PHYS_DEF_TYPE}, namespace ${NS})..."
        run_defense_eval_scripts "venv_physionetcinc" "PhysioNetCinC" "physionetcinc" "$RUN_PHYS_DEF_TYPE" "$NS"
    done
fi

# ---------------------------
# Plot Defense Results
# ---------------------------
if [ "$RUN_OHIOT1DM_PLOT_DEF" = "true" ]; then
    for NS in "${OHIOT1DM_NAMESPACES[@]}"; do
        echo "Plotting defense results for OhioT1DM (namespace ${NS})..."
        run_in_env_path "OhioT1DM/venv_ohiot1dm" "." "python plot_defense_results.py OhioT1DM OhioT1DM/output/${NS}/defense_output"
    done
fi

if [ "$RUN_MIMIC_PLOT_DEF" = "true" ]; then
    for NS in "${MIMIC_NAMESPACES[@]}"; do
        echo "Plotting defense results for MIMIC (namespace ${NS})..."
        run_in_env_path "MIMIC/venv_mimic" "." "python plot_defense_results.py MIMIC MIMIC/output/${NS}/defense_output"
    done
fi

if [ "$RUN_PHYS_PLOT_DEF" = "true" ]; then
    for NS in "${PHYS_NAMESPACES[@]}"; do
        echo "Plotting defense results for PhysioNetCinC (namespace ${NS})..."
        run_in_env_path "PhysioNetCinC/venv_physionetcinc" "." "python plot_defense_results.py PhysioNetCinC PhysioNetCinC/output/${NS}/defense_output"
    done
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

# ---------------------------
# Per-cluster (less/more-vulnerable test) reporting (cross-dataset)
# ---------------------------
# OhioT1DM is post-hoc (per-patient combined CSVs already exist) -- no extra eval.
# MIMIC/PhysioNet need the per-cluster test sets evaluated: re-run the existing
# detectors against the defense_dataset_{less,more} dirs emitted by
# generate_defense_dataset.py (requires those generate + the normal eval to have run).
# Honor each dataset's defense_type so MAD-GAN (expensive) is included only if selected.
if [ "$RUN_PER_CLUSTER_REPORT" = "true" ]; then
    # Only re-run per-cluster eval for a dataset whose per-cluster test sets exist
    # (created by that dataset's generate_defense_datasets step). Datasets without them
    # are skipped, so single-dataset / partial configs work. OhioT1DM is post-hoc.
    for NS in "${MIMIC_NAMESPACES[@]}"; do
        for VARIANT in less more; do
            if [ -d "$SCRIPT_DIR/MIMIC/output/${NS}/defense_dataset_${VARIANT}" ]; then
                echo "Per-cluster eval (${VARIANT}) for MIMIC (namespace ${NS})..."
                run_defense_eval_scripts "venv_mimic" "MIMIC" "mimic" "$RUN_MIMIC_DEF_TYPE" "$NS" "$VARIANT"
            else
                echo "Skipping MIMIC per-cluster eval (${VARIANT}): output/${NS}/defense_dataset_${VARIANT} not found."
            fi
        done
    done
    for NS in "${PHYS_NAMESPACES[@]}"; do
        for VARIANT in less more; do
            if [ -d "$SCRIPT_DIR/PhysioNetCinC/output/${NS}/defense_dataset_${VARIANT}" ]; then
                echo "Per-cluster eval (${VARIANT}) for PhysioNetCinC (namespace ${NS})..."
                run_defense_eval_scripts "venv_physionetcinc" "PhysioNetCinC" "physionetcinc" "$RUN_PHYS_DEF_TYPE" "$NS" "$VARIANT"
            else
                echo "Skipping PhysioNetCinC per-cluster eval (${VARIANT}): output/${NS}/defense_dataset_${VARIANT} not found."
            fi
        done
    done

    echo "Generating per-cluster breakdown CSVs + grouped-bar plots..."
    run_in_env_path "OhioT1DM/venv_ohiot1dm" "." "python report_per_cluster.py"
fi

echo "Pipeline completed successfully."
