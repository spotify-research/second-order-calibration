#!/usr/bin/env bash
set -euo pipefail

# Defaults aimed at an paper-faithful regime that still leaves usable epistemic signal for
# Experiment 2 and active-learning diagnostics.
#
# Defaults used if not already set in the environment:
#   EXP1_ENSEMBLE_MEMBER_DATA_MODE=same
#   EXP1_ENSEMBLE_SIZE=20
#   EXP1_ENSEMBLE_TRAIN_N=24
#   EXP1_ENSEMBLE_EPOCHS=60
#   EXP1_S_H_SCALE=1.0
#   EXP2_CAL_SNAPSHOTS=2
#   EXP2_TARGET_MODE=paper
#   EXP2_FIXED_DEGREE=12
#   EXP2_PROJECT_FEASIBLE_MOMENTS=1
#   EXP2_SAVE_RANK_DIAGNOSTIC=1
#
# Main outputs:
#   results/experiment2_h_*.pdf          scatter + binned monotonicity + CE2
#   results/experiment2_rankdiag_h_*.pdf top-k active-learning diagnostics
#   results/experiment2_h_*.npz          full metrics and saved arrays

: "${EXP1_ENSEMBLE_MEMBER_DATA_MODE:=same}"
: "${EXP1_ENSEMBLE_SIZE:=20}"
: "${EXP1_ENSEMBLE_TRAIN_N:=24}"
: "${EXP1_ENSEMBLE_EPOCHS:=60}"
: "${EXP1_S_H_SCALE:=1.0}"
: "${EXP2_CAL_SNAPSHOTS:=2}"
: "${EXP2_FIXED_DEGREE:=12}"
: "${EXP2_TARGET_MODE:=paper}"
: "${EXP2_PROJECT_FEASIBLE_MOMENTS:=1}"
: "${EXP2_SAVE_RANK_DIAGNOSTIC:=1}"
: "${EXP2_SAVE_RAW_DIAGNOSTIC:=0}"

export EXP1_ENSEMBLE_MEMBER_DATA_MODE
export EXP1_ENSEMBLE_SIZE
export EXP1_ENSEMBLE_TRAIN_N
export EXP1_ENSEMBLE_EPOCHS
export EXP1_S_H_SCALE
export EXP2_CAL_SNAPSHOTS
export EXP2_FIXED_DEGREE
export EXP2_TARGET_MODE
export EXP2_PROJECT_FEASIBLE_MOMENTS
export EXP2_SAVE_RANK_DIAGNOSTIC
export EXP2_SAVE_RAW_DIAGNOSTIC

python dataset.py
python experiment2.py
