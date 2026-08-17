#!/bin/bash
# Setup an adapter repo for additional environment support.
#
# Usage:
#   bash setup_adapter.sh <adapter_name> [target_dir]
#
# Available adapters:
#   jepa-wms-grasp   -  JEPA World Model environments (Metaworld, DROID, RoboCasa)
#   le-wm-grasp      -  LeWorldModel environments (PushT, TwoRoom, Reacher, Cube)
#
# Examples:
#   bash setup_adapter.sh jepa-wms-grasp
#   bash setup_adapter.sh le-wm-grasp ./my_adapters/le-wm-grasp
#   bash setup_adapter.sh all

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ADAPTER_DIR="${SCRIPT_DIR}/adapters"

ADAPTERS=(
    "jepa-wms-grasp|git@github.com:michael-psenka/jepa-wms-grasp.git"
    "le-wm-grasp|git@github.com:michael-psenka/le-wm-grasp.git"
)

usage() {
    echo "Usage: bash setup_adapter.sh <adapter_name> [target_dir]"
    echo ""
    echo "Available adapters:"
    for entry in "${ADAPTERS[@]}"; do
        name="${entry%%|*}"
        echo "  $name"
    done
    echo "  all  (clone all adapters)"
    echo ""
    echo "Default target directory: <repo_root>/adapters/<adapter_name>"
    exit 1
}

get_repo_url() {
    local name="$1"
    for entry in "${ADAPTERS[@]}"; do
        entry_name="${entry%%|*}"
        entry_url="${entry#*|}"
        if [ "$entry_name" = "$name" ]; then
            echo "$entry_url"
            return 0
        fi
    done
    return 1
}

clone_adapter() {
    local name="$1"
    local target="$2"

    local url
    url="$(get_repo_url "$name")" || { echo "Error: unknown adapter '$name'"; usage; }

    if [ -d "$target" ]; then
        echo "Adapter '$name' already exists at $target — skipping clone."
        echo "  To update, run: cd $target && git pull"
        return 0
    fi

    echo "Cloning $name into $target ..."
    mkdir -p "$(dirname "$target")"
    git clone "$url" "$target"
    echo "Done: $target"
}

ensure_gitignore() {
    local gitignore="${SCRIPT_DIR}/.gitignore"
    if ! grep -qxF 'adapters/' "$gitignore" 2>/dev/null; then
        echo "" >> "$gitignore"
        echo "adapters/" >> "$gitignore"
        echo "Added 'adapters/' to .gitignore"
    fi
}

# --- main ---

if [ $# -lt 1 ]; then
    usage
fi

ADAPTER_NAME="$1"
TARGET_DIR="${2:-}"

ensure_gitignore

if [ "$ADAPTER_NAME" = "all" ]; then
    for entry in "${ADAPTERS[@]}"; do
        name="${entry%%|*}"
        clone_adapter "$name" "${DEFAULT_ADAPTER_DIR}/${name}"
    done
else
    if [ -z "$TARGET_DIR" ]; then
        TARGET_DIR="${DEFAULT_ADAPTER_DIR}/${ADAPTER_NAME}"
    fi
    clone_adapter "$ADAPTER_NAME" "$TARGET_DIR"
fi
