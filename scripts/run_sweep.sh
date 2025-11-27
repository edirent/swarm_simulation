#!/usr/bin/env bash
set -euo pipefail

# Simple sweep over configs. Renders disabled for speed/headless runs.
for cfg in configs/*.yaml; do
  echo "Running config: ${cfg}"
  MPLBACKEND=Agg python scripts/run_sim.py --config "${cfg}" --no-render
done
