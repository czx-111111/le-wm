#!/usr/bin/env bash
# Single installation script for dino-wm-admm-planning
# Sets up conda env, optional Mujoco, dataset path, and PushT frame extraction.

set -e

CONDA_ENV_NAME="grasp"
BASHRC_MARKER="# dino-wm-admm-planning install"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MUJOCO_URL="https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz"
OSF_DATASET_URL="https://osf.io/bmw48/?view_only=a56a296ce3b24cceaf408383a175ce28"

# Option flags (default: conda on)
DO_CONDA=true
DO_MUJOCO=false
DO_DATASET_DIR=false
DATASET_DIR_PATH=""
DO_EXTRACT_PUSHT=false
DO_DOWNLOAD_DATASETS=false
# User path = base dir; datasets go in base/datasets, pre-trained models in base/outputs
SAVED_DATASET_DIR=""
SAVED_BASE_DIR=""

# Standard layout under DATASET_DIR: point_maze/, pusht/, wall_single/
# PushT raw videos: pusht/train/obses, pusht/val/obses; extracted frames go under pusht/train/obses, pusht/val/obses
OSF_PROJECT_ID="bmw48"

# User input collected up front
CONDA_ENV_CHOICE=""

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Install dino-wm-admm-planning: conda environment and optional Mujoco,
dataset path, and PushT frame extraction.

Options:
  --help                 Show this help and exit.
  --conda                Create/update conda env from environment.yaml (default: on).
  --no-conda             Skip conda environment setup.
  --mujoco               Install Mujoco210 to ~/.mujoco and add to LD_LIBRARY_PATH.
  --dataset-dir [PATH]   Base directory: datasets go to <base>/datasets, pre-trained models to <base>/outputs. DATASET_DIR=<base>/datasets is persisted. If PATH omitted, prompt.
  --download-datasets     If dataset dir is set, download missing datasets from OSF (requires osfclient).
  --extract-pusht-frames Run PushT frame extraction (uses DATASET_DIR; expects data in pusht/train/obses, pusht/val/obses).
  --all                  Run conda + mujoco + dataset-dir (prompt) + download-datasets (with y/n confirmation) + extract-pusht-frames.

Examples:
  $0 --conda
  $0 --conda --mujoco --dataset-dir /path/to/base
  $0 --all --dataset-dir /path/to/base
  $0 --no-conda --extract-pusht-frames
EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --help)
            usage
            exit 0
            ;;
        --conda)
            DO_CONDA=true
            shift
            ;;
        --no-conda)
            DO_CONDA=false
            shift
            ;;
        --mujoco)
            DO_MUJOCO=true
            shift
            ;;
        --dataset-dir)
            DO_DATASET_DIR=true
            if [[ -n "${2:-}" && "$2" != --* ]]; then
                DATASET_DIR_PATH="$2"
                shift 2
            else
                shift
            fi
            ;;
        --extract-pusht-frames)
            DO_EXTRACT_PUSHT=true
            shift
            ;;
        --download-datasets)
            DO_DOWNLOAD_DATASETS=true
            shift
            ;;
        --all)
            DO_CONDA=true
            DO_MUJOCO=true
            DO_DATASET_DIR=true
            DO_DOWNLOAD_DATASETS=true
            DO_EXTRACT_PUSHT=true
            shift
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

# Ensure we are in repo root
cd "$REPO_ROOT"
if [[ ! -f environment.yaml ]]; then
    echo "Error: environment.yaml not found. Run this script from the repository root." >&2
    exit 1
fi

echo "=== dino-wm-admm-planning installer ==="
echo "Repository root: $REPO_ROOT"
echo "Steps to run: conda=$DO_CONDA mujoco=$DO_MUJOCO dataset_dir=$DO_DATASET_DIR download_datasets=$DO_DOWNLOAD_DATASETS extract_pusht=$DO_EXTRACT_PUSHT"
echo ""

# --- Gather user input (all interaction happens here) ---
if $DO_CONDA; then
    if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
        read -p "Conda env '${CONDA_ENV_NAME}' exists. Skip / Update / Recreate? [s/u/r] (default: s): " CONDA_ENV_CHOICE
        CONDA_ENV_CHOICE="${CONDA_ENV_CHOICE:-s}"
    else
        CONDA_ENV_CHOICE="c"
    fi
fi

if $DO_DATASET_DIR; then
    if [[ -z "$DATASET_DIR_PATH" ]]; then
        echo "Enter base directory (datasets will be in <base>/datasets, pre-trained models in <base>/outputs)."
        read -p "Base directory path (or leave empty to skip): " DATASET_DIR_PATH
        DATASET_DIR_PATH="$(echo "$DATASET_DIR_PATH" | sed 's/^ *//;s/ *$//')"
    fi
    if [[ -n "$DATASET_DIR_PATH" ]]; then
        DATASET_DIR_PATH="$(echo "$DATASET_DIR_PATH" | sed "s|^~|$HOME|")"
        if [[ ! -d "$DATASET_DIR_PATH" ]]; then
            echo "Warning: Directory does not exist yet: $DATASET_DIR_PATH"
            read -p "Still use this path and add DATASET_DIR=<base>/datasets to ~/.bashrc? [y/N]: " _confirm_dataset
            if [[ "${_confirm_dataset:-n}" != [yY]* ]]; then
                DATASET_DIR_PATH=""
            fi
        fi
    fi
fi

# --- Helpers for .bashrc ---
# Each step uses a distinct marker so re-runs don't duplicate blocks.
bashrc_has_marker() {
    local marker="${1:-$BASHRC_MARKER}"
    grep -q "$marker" ~/.bashrc 2>/dev/null || true
}

bashrc_append_block() {
    local block="$1"
    local marker="${2:-$BASHRC_MARKER}"
    if bashrc_has_marker "$marker"; then
        echo "Note: already present in ~/.bashrc (marker: $marker); skipping."
        return
    fi
    echo "" >> ~/.bashrc
    echo "$marker" >> ~/.bashrc
    echo "$block" >> ~/.bashrc
    echo "Appended to ~/.bashrc."
}

# --- Conda ---
run_conda() {
    echo ">>> Setting up conda environment '$CONDA_ENV_NAME'..."
    if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
        echo "Environment '$CONDA_ENV_NAME' already exists."
        case "$CONDA_ENV_CHOICE" in
            u|U)
                conda env update -f environment.yaml -n "$CONDA_ENV_NAME"
                echo "Updated environment '$CONDA_ENV_NAME'."
                ;;
            r|R)
                conda env remove -n "$CONDA_ENV_NAME" -y
                conda env create -f environment.yaml
                echo "Recreated environment '$CONDA_ENV_NAME'."
                ;;
            c|C)
                echo "Env exists; 'c' (create) choice is ignored. Skipping conda env."
                ;;
            *)
                echo "Skipping conda env."
                ;;
        esac
    else
        if [[ "$CONDA_ENV_CHOICE" == "c" || -z "$CONDA_ENV_CHOICE" ]]; then
            conda env create -f environment.yaml
            echo "Created environment '$CONDA_ENV_NAME'."
        else
            echo "Conda env '$CONDA_ENV_NAME' does not exist and choice='$CONDA_ENV_CHOICE'; creating it."
            conda env create -f environment.yaml
            echo "Created environment '$CONDA_ENV_NAME'."
        fi
    fi
    if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
        echo "Ensuring CMake is installed in env (required for building egl-probe)..."
        conda install -n "$CONDA_ENV_NAME" cmake -y
    fi
}

# --- Mujoco ---
run_mujoco() {
    echo ">>> Installing Mujoco210..."
    local mujoco_dir="$HOME/.mujoco"
    local mujoco_bin="$mujoco_dir/mujoco210"
    if [[ -d "$mujoco_bin" ]]; then
        echo "Mujoco210 already present at $mujoco_bin. Skipping download."
    else
        mkdir -p "$mujoco_dir"
        wget -q --show-progress "$MUJOCO_URL" -P "$mujoco_dir" || {
            echo "Error: wget failed. Try downloading manually from $MUJOCO_URL" >&2
            return 1
        }
        (cd "$mujoco_dir" && tar -xzf mujoco210-linux-x86_64.tar.gz)
        echo "Extracted Mujoco210."
    fi
    local block="# Mujoco Path
export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin

# NVIDIA Library Path (if using NVIDIA GPUs)
export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/lib/nvidia"
    bashrc_append_block "$block" "${BASHRC_MARKER} mujoco"
    echo "Mujoco step done."
}

# --- Dataset dir (user path = base; DATASET_DIR = base/datasets) ---
run_dataset_dir() {
    echo ">>> Setting DATASET_DIR..."
    local base="$DATASET_DIR_PATH"
    if [[ -z "$base" ]]; then
        echo "No base directory path set; skipping DATASET_DIR."
        return 0
    fi
    base="$(echo "$base" | sed "s|^~|$HOME|")"
    if [[ ! -d "$base" ]]; then
        echo "Warning: Base directory does not exist yet: $base"
    fi
    local dataset_root="$base/datasets"
    mkdir -p "$dataset_root"
    local block="export DATASET_DIR=\"$dataset_root\"
export CKPT_BASE_PATH=\"$base\""
    bashrc_append_block "$block" "${BASHRC_MARKER} dataset_dir"
    SAVED_DATASET_DIR="$dataset_root"
    SAVED_BASE_DIR="$base"
    echo ""
    echo "DATASET_DIR=$dataset_root"
    echo "CKPT_BASE_PATH=$base (used by plan.py to load from $base/outputs)"
    echo "Datasets will be in: $dataset_root (use --download-datasets to fetch from OSF)."
    echo "Pre-trained models will be in: $base/outputs (when using --download-datasets)"
    echo "Expected layout under DATASET_DIR: point_maze/, pusht/, wall_single/"
    echo "DATASET_DIR step done."
}

# --- Download datasets from OSF ---
run_download_datasets() {
    echo ">>> Downloading datasets from OSF (project $OSF_PROJECT_ID)..."
    local base="${SAVED_BASE_DIR:-$DATASET_DIR_PATH}"
    base="$(echo "$base" | sed "s|^~|$HOME|")"
    if [[ -z "$base" ]]; then
        echo "No base directory set; skipping download. Use --dataset-dir PATH first." >&2
        return 0
    fi
    if [[ ! -d "$base" ]]; then
        mkdir -p "$base"
    fi
    mkdir -p "$base/datasets" "$base/outputs"
    if ! conda run -n "$CONDA_ENV_NAME" osf --help &>/dev/null; then
        echo "Installing osfclient in conda env for OSF download..."
        conda run -n "$CONDA_ENV_NAME" pip install osfclient
    fi
    local need_download=false
    for sub in point_maze pusht wall_single; do
        if [[ ! -d "$base/datasets/$sub" ]]; then
            need_download=true
            break
        fi
    done
    if [[ "$need_download" != true ]]; then
        echo "All dataset folders (point_maze, pusht, wall_single) already present in $base/datasets; skipping download."
        return 0
    fi
    read -p "Download missing datasets from OSF (project $OSF_PROJECT_ID)? [y/N]: " _confirm_download
    if [[ "${_confirm_download:-n}" != [yY]* ]]; then
        echo "Skipping dataset download."
        return 0
    fi
    local tmp_dir="$base/.osf_clone_$$"
    mkdir -p "$tmp_dir"
    # Run osf with conda activated in this shell so the process keeps the TTY; osfclient then shows a tqdm progress bar (file count + files/s)
    local conda_sh
    conda_sh="$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh"
    [[ -f "$conda_sh" ]] || conda_sh="${CONDA_PREFIX:-}/etc/profile.d/conda.sh"
    if [[ -f "$conda_sh" ]]; then
        set +e
        ( source "$conda_sh" && conda activate "$CONDA_ENV_NAME" && PYTHONUNBUFFERED=1 osf -p "$OSF_PROJECT_ID" clone "$tmp_dir" )
        local clone_ret=$?
        set -e
        if [[ $clone_ret -ne 0 ]]; then
            echo "Error: osf clone failed. You may need to run 'osf login' (in the conda env) for private projects." >&2
            rm -rf "$tmp_dir"
            return 1
        fi
    else
        if ! PYTHONUNBUFFERED=1 conda run -n "$CONDA_ENV_NAME" osf -p "$OSF_PROJECT_ID" clone "$tmp_dir"; then
            echo "Error: osf clone failed. You may need to run 'osf login' (in the conda env) for private projects." >&2
            rm -rf "$tmp_dir"
            return 1
        fi
    fi
    # Move clone top-level (e.g. osfstorage) into base
    local src="$tmp_dir"
    if [[ -d "$tmp_dir/$OSF_PROJECT_ID" ]]; then
        src="$tmp_dir/$OSF_PROJECT_ID"
    fi
    for item in "$src"/*; do
        [[ -e "$item" ]] || continue
        name="$(basename "$item")"
        if [[ -d "$base/$name" ]]; then
            echo "  Skipping $name (already exists)."
        else
            mv "$item" "$base/"
            echo "  Placed $name"
        fi
    done
    rm -rf "$tmp_dir"

    # Unpack osfstorage: move datasets/* into base/datasets; move checkpoints/* into base/outputs (plan.py loads from ckpt_base_path/outputs)
    if [[ -d "$base/osfstorage" ]]; then
        if [[ -d "$base/osfstorage/datasets" ]]; then
            for item in "$base/osfstorage/datasets"/*; do
                [[ -e "$item" ]] || continue
                name="$(basename "$item")"
                if [[ "$name" == deformable* ]]; then
                    echo "  Skipping $name (deformable env not used)."
                    continue
                fi
                if [[ -e "$base/datasets/$name" ]]; then
                    echo "  Skipping $name (already exists in base/datasets)."
                else
                    mv "$item" "$base/datasets/"
                    echo "  Placed $name in datasets/"
                fi
            done
        fi
        if [[ -d "$base/osfstorage/checkpoints" ]]; then
            for item in "$base/osfstorage/checkpoints"/*; do
                [[ -e "$item" ]] || continue
                name="$(basename "$item")"
                if [[ -e "$base/outputs/$name" ]]; then
                    echo "  Skipping outputs/$name (already exists)."
                else
                    mv "$item" "$base/outputs/"
                    echo "  Placed $name in outputs/"
                fi
            done
        fi
        rm -rf "$base/osfstorage"
    fi

    # Extract zip archives in base/datasets and base/outputs
    for dir in "$base/datasets" "$base/outputs"; do
        [[ -d "$dir" ]] || continue
        while IFS= read -r -d '' z; do
            echo "  Extracting $z ..."
            (cd "$(dirname "$z")" && unzip -o -q "$z" && rm -f "$(basename "$z")") || true
        done < <(find "$dir" -maxdepth 2 -name '*.zip' -print0 2>/dev/null)
    done

    # Remove any deformable data that came out of zips
    for d in "$base/datasets"/deformable*; do
        if [[ -e "$d" ]]; then
            echo "  Removing $d (deformable env not used)."
            rm -rf "$d"
        fi
    done

    # Standardize naming: pusht_noise -> pusht
    if [[ -d "$base/datasets/pusht_noise" && ! -d "$base/datasets/pusht" ]]; then
        mv "$base/datasets/pusht_noise" "$base/datasets/pusht"
        echo "  Renamed pusht_noise -> pusht"
    fi
    echo "Download step done."
}

# --- Extract PushT frames ---
run_extract_pusht() {
    echo ">>> PushT frame extraction..."
    echo "This will convert MP4 videos to PNG frames for faster data loading"
    echo ""
    local base_dir="${SAVED_DATASET_DIR:-$DATASET_DIR_PATH}"
    base_dir="$(echo "$base_dir" | sed "s|^~|$HOME|")"
    if [[ -z "$base_dir" ]]; then
        echo "Error: DATASET_DIR is not set. Set it with --dataset-dir PATH so PushT data is at <PATH>/pusht/train/obses and <PATH>/pusht/val/obses." >&2
        return 1
    fi
    local input_train="$base_dir/pusht/train/obses"
    local input_val="$base_dir/pusht/val/obses"
    local output="$base_dir/pusht"

    if [[ ! -d "$input_train" ]]; then
        echo "Error: PushT train data not found at: $input_train" >&2
        echo "Expected raw observation videos in: $base_dir/pusht/train/obses/ and $base_dir/pusht/val/obses/" >&2
        echo "Download datasets with --download-datasets or place the data manually." >&2
        return 1
    fi
    if [[ ! -d "$input_val" ]]; then
        echo "Error: PushT val data not found at: $input_val" >&2
        echo "Expected raw observation videos in: $base_dir/pusht/train/obses/ and $base_dir/pusht/val/obses/" >&2
        echo "Download datasets with --download-datasets or place the data manually." >&2
        return 1
    fi

    echo "Using PushT paths: train=$input_train val=$input_val output=$output"
    mkdir -p "$output"
    conda run -n "$CONDA_ENV_NAME" python extract_pusht_frames.py \
        --input-train "$input_train" \
        --input-val "$input_val" \
        --output "$output" \
        --split both
    echo "PushT frame extraction done."
}

# --- Run selected steps ---
if $DO_CONDA; then
    run_conda || exit 1
fi
if $DO_MUJOCO; then
    run_mujoco || exit 1
fi
if $DO_DATASET_DIR; then
    run_dataset_dir || exit 1
fi
if $DO_DOWNLOAD_DATASETS; then
    run_download_datasets || exit 1
fi
if $DO_EXTRACT_PUSHT; then
    run_extract_pusht || exit 1
fi

echo ""
echo "=== Done ==="
echo "Run: source ~/.bashrc   (or open a new shell)"
echo "Then: conda activate $CONDA_ENV_NAME"
