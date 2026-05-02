#!/usr/bin/env bash
set -euo pipefail
export LD_LIBRARY_PATH=/run/current-system/sw/share/nix-ld/lib:${LD_LIBRARY_PATH:-}
.venv/bin/python src/main.py "$@"