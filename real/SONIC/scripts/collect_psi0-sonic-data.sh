#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/sonic_conda_env.sh"

exec "$SONIC_PYTHON" "$PSI_ROOT/real/SONIC/launch_data_collection_dex1.py" "$@"
