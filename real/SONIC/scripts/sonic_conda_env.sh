#!/usr/bin/env bash
# Shared environment for Psi0's official-SONIC wrappers.  All Python processes
# use the existing conda `sonic` environment; no per-repo venv is created.

set -euo pipefail

SONIC_ENV_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PSI_ROOT="${PSI_ROOT:-$(cd "$SONIC_ENV_SCRIPT_DIR/../../.." && pwd)}"
export SONIC_DIR="${SONIC_DIR:-$PSI_ROOT/third_party/GR00T-WholeBodyControl}"
if [[ -z "${SONIC_CONDA_PREFIX:-}" ]]; then
    if [[ "${CONDA_DEFAULT_ENV:-}" == "sonic" && -n "${CONDA_PREFIX:-}" ]]; then
        SONIC_CONDA_PREFIX="$CONDA_PREFIX"
    else
        SONIC_CONDA_PREFIX="$HOME/miniconda3/envs/sonic"
    fi
fi
export SONIC_CONDA_PREFIX
export SONIC_PYTHON="$SONIC_CONDA_PREFIX/bin/python"

if [ ! -x "$SONIC_PYTHON" ]; then
    echo "Missing conda sonic Python: $SONIC_PYTHON" >&2
    return 1 2>/dev/null || exit 1
fi

# Shell snapshots on this workstation contain IsaacLab paths which shadow
# typing_extensions, OpenCV, SciPy, and Torch from the sonic environment.
unset PYTHONPATH
unset LD_LIBRARY_PATH
export PATH="$SONIC_CONDA_PREFIX/bin:$PATH"
export PYTHONPATH="$PSI_ROOT:$SONIC_DIR"
export LD_LIBRARY_PATH="$SONIC_CONDA_PREFIX/lib:$SONIC_DIR/external_dependencies/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64/lib"

export SONIC_CHECKPOINT="${SONIC_CHECKPOINT:-policy/sonic_v1_1/model}"
export SONIC_OBS_CONFIG="${SONIC_OBS_CONFIG:-policy/sonic_v1_1/observation_config.yaml}"
export DEX1_VIRTUAL_STATS="${DEX1_VIRTUAL_STATS:-$PSI_ROOT/real/SONIC/assets/dex1_virtual_mapping_stats.json}"
export G1_NETWORK_INTERFACE="${G1_NETWORK_INTERFACE:-enp4s0}"
