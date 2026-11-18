#!/usr/bin/env bash
# Run all of Experiment 1 (dataset, three estimators, plots) for every h
# in H_LIST.  Logs each step to a .log file and times it.

set -euo pipefail

cd "$(dirname "$0")"
mkdir -p data results

for stage in dataset bucketing nadaraya_watson polynomial plot; do
    echo "=========================================="
    echo "  Running ${stage}.py"
    echo "=========================================="
    { time python "${stage}.py"; } 2>&1 | tee "${stage}.log"
done

echo
echo "All done.  Plots are in results/experiment1_h_1_*.pdf"
ls -1 results/experiment1_*.pdf