#!/bin/bash
set -euo pipefail

PSI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$PSI_ROOT/real/SONIC/scripts/sonic_conda_env.sh"

exec "$SONIC_PYTHON" -u \
    "$PSI_ROOT/real/SONIC/run_psi0_rtc_sonic_dex1.py" \
    "$@"
