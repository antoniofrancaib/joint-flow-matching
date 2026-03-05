#!/bin/bash
# Regenerate all data: molecules.json and trajectory JSONs.
set -e
cd "$(dirname "$0")/.."
python3 scripts/gen_molecules.py
python3 scripts/gen_trajectories.py
