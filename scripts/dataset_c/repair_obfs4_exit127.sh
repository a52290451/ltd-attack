#!/usr/bin/env bash
# DEPRECATED P16C diagnostic recovery path, retained in Git history.
# Obsolete: env -u workaround failed because executable was absent.
# Working procedure: scripts/dataset_c/restore_three_tor_from_working.sh
# New instance setup: scripts/dataset_c/three_instance_qa.py
set -euo pipefail
echo "STOP: Este script está OBSOLETO. No reejecutar reparaciones fallidas de Tor." >&2
echo "Tor ya fue restaurado; comprobar: python3 scripts/dataset_c/three_instance_qa.py status" >&2
exit 2
