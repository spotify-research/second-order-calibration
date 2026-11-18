# Experiment 4: crowd disagreement audit (Weather Sentiment)

Code for Experiment 4 of the paper, "the same effect on real labels" (Figure 5, Appendix B.4).

## Purpose

This is a stress test of second-order Platt scaling: a disagreement audit on repeated labels
from Weather Sentiment.

The experiment asks the following question. With a limited audit budget, which items should
be sent for additional review if the goal is to find items where two fresh workers would
disagree?

For an item, let `theta` be the probability that a fresh worker agrees with a provisional
binary orientation. The audit value is

```
g(theta) = P(two workers disagree | theta) = 2 theta (1 - theta)
```

Given a score `S = (m, s)`, second-order calibration estimates

```
eta1(S) = E[theta   | S]
eta2(S) = E[theta^2 | S]
```

and uses the calibrated score

```
E[g(theta) | S] = 2 eta1(S) - 2 eta2(S)
```

## Data

The default dataset is Weather Sentiment AMT, converted into one-vs-rest binary tasks for
each of the five labels. On the first run the script downloads
`WeatherSentiment_amt.csv` to `data/`, next to the script. Set
`EXP_CROWD_DISAGREE_FORCE_REDOWNLOAD=1` to download it again.

The labels are real repeated AMT labels, which makes this a controlled real-label stress test.
The score pair `S = (m, s)`, however, is deliberately constructed to create a setting where
items share the same first moment but differ in the second moment.

## Important default design

By default the script reselects the best Weather class and cohort/score configuration on a
held-out selection split (`EXP_CROWD_DISAGREE_RESELECT=1`).

The default score-mode grid is `xor,noisy_xor,banana,linear`. In XOR mode, the
one-dimensional marginals of `m` and `s` carry little information on their own. Their joint
pattern, however, separates aleatoric items from hidden-subtype items. This tests whether a
2D second-moment calibrator can recover an audit value that is genuinely second-order.

## Main methods

| Method | Score |
| --- | --- |
| raw plug-in from `m` | `2 m (1 - m)` |
| raw high-s coordinate | `s` |
| raw low-s coordinate | `1 - s` |
| 1D first-order Platt | `2 eta1(m) (1 - eta1(m))` |
| 1D second-order Platt | `2 eta1(m) - 2 eta2(m)` |
| 2D first-order Platt | `2 eta1(m,s) (1 - eta1(m,s))` |
| 2D second-order Platt | `2 eta1(m,s) - 2 eta2(m,s)` |
| oracle | `2 theta (1 - theta)` |

## Main output

The script writes two figures:

- **`*_curves.pdf`** shows the expected number of worker disagreements found per 100
  audited items, as a function of the audit fraction.
- **`*_contrasts.pdf`** shows the paired lift of 2D second-order Platt over the key
  baselines: 2D first-order Platt, 1D first-order Platt, raw plug-in from `m`, and the
  raw high-s and low-s coordinates.

## Running

The default settings are the ones used for the paper figures:

```bash
bash run_exp_crowd_weather_disagreement.sh
```

To regenerate both figures from the saved `.npz` without re-running the experiment:

```bash
python replot.py
```

### Useful knobs

Fast debug run:

```bash
EXP_CROWD_DISAGREE_SELECT_REPS=10 \
EXP_CROWD_DISAGREE_EVAL_REPS=50 \
EXP_CROWD_DISAGREE_N_JOBS=1 \
bash run_exp_crowd_weather_disagreement.sh
```

Confirm a fixed configuration without reselection:

```bash
EXP_CROWD_DISAGREE_RESELECT=0 \
EXP_CROWD_DISAGREE_DATASET='Weather Sentiment class 1 vs rest' \
EXP_CROWD_DISAGREE_SCORE_MODE=xor \
bash run_exp_crowd_weather_disagreement.sh
```

Strict paper-style two-snapshot calibration:

```bash
EXP_CROWD_DISAGREE_TARGET_MODE=two_snapshot \
bash run_exp_crowd_weather_disagreement.sh
```

## Files written

All outputs go to `results/`. The filenames below use the default prefix
`exp_crowd_weather_disagreement` (set with `EXP_CROWD_DISAGREE_OUT_PREFIX`).

| File | Contents |
| --- | --- |
| `exp_crowd_weather_disagreement_curves.pdf` (and `.png`) | Disagreements found vs. audit fraction |
| `exp_crowd_weather_disagreement_contrasts.pdf` (and `.png`) | Paired lift of 2D second-order over baselines |
| `exp_crowd_weather_disagreement.npz` | Curves, per-repeat results, selected config, and summary (read by `replot.py`) |
| `exp_crowd_weather_disagreement_selected_config.json` | Selected configuration, selection result, and summary |
| `exp_crowd_weather_disagreement_screen.csv` | Screening table of all candidate configs (written only when `RESELECT=1`, the default) |

## Environment variables

All variables have the prefix `EXP_CROWD_DISAGREE_`.

| Variable | Default | Meaning |
| --- | --- | --- |
| `RESELECT` | `1` | `1` = reselect the config on the selection split; `0` = use the fixed config below |
| `SELECT_REPS` | `60` | Repeats per candidate during selection |
| `EVAL_REPS` | `1000` | Evaluation repeats for the selected config |
| `TARGET_MODE` | `pairs` | `pairs` (all-pairs estimator) or `two_snapshot` |
| `N_JOBS` | `min(8, CPUs - 1)` | Parallel workers; set to `1` for sequential runs |
| `SEED` | `123` | Base random seed |
| `SEL_FRAC` | `0.25` | Fraction of items used for selection |
| `CAL_FRAC` | `0.35` | Fraction of items used for calibration |
| `DEGREE_1D` | `4` | Polynomial degree for the 1D calibrators |
| `DEGREE_2D` | `3` | Polynomial degree for the 2D calibrators |
| `RIDGE` | `1e-3` | Ridge penalty |
| `SCORE_MODES` | `xor,noisy_xor,banana,linear` | Score modes screened during selection |
| `REVIEW_FRACS` | `0,0.005,0.01,0.02,0.05,0.10,0.20,0.30` | Audit fractions on the x-axis |
| `SUMMARY_FRAC` | `0.05` | Audit fraction used for the printed summary |
| `AUC_MAX_FRAC` | `0.30` | Upper audit fraction for the area-under-curve summary |
| `DATASET` | `Weather Sentiment class 1 vs rest` | Fixed dataset (only when `RESELECT=0`) |
| `SCORE_MODE` | `xor` | Fixed score mode (only when `RESELECT=0`) |
| `EXTREME_THR` | `0.18` | Fixed extreme-item threshold (only when `RESELECT=0`) |
| `ALE_WIDTH` | `0.22` | Fixed aleatoric width (only when `RESELECT=0`) |
| `M_NOISE` | `0.05` | Fixed `m` noise (only when `RESELECT=0`) |
| `S_NOISE` | `0.22` | Fixed `s` noise (only when `RESELECT=0`) |
| `S_GAP` | `0.50` | Fixed `s` gap (only when `RESELECT=0`) |
| `OUT_PREFIX` | `exp_crowd_weather_disagreement` | Output filename prefix |
| `FORCE_REDOWNLOAD` | `0` | `1` = download the dataset again |

The script also reads selection-grid variables (`EXTREME_THRESHOLDS`, `ALEATORIC_WIDTHS`,
`M_NOISE_GRID`, `S_NOISE_GRID`, `S_GAP_GRID`), minimum-size filters (`MIN_ITEMS`,
`MIN_VOTES_PER_ITEM`, `MIN_ELIGIBLE_CAL`, `MIN_ELIGIBLE_EVAL`, `MIN_GROUP`,
`BALANCE_GROUPS`), and parallelism controls (`PARALLEL_BATCH_SIZE`, `PARALLEL_BACKEND`,
`SAVE_PARTIAL`, `MAX_CONFIGS`). See the top of `experiment_crowd_weather_disagreement.py`
for their defaults.

## Real-data accounting

- **Selection labels** are used to choose the Weather class and the cohort/score construction.
- **Calibration labels** fit `eta1` and `eta2`.
- **Evaluation items** are disjoint from the selection and calibration items.

In `pairs` mode, the script uses all repeated labels on calibration items through the
all-pairs estimator `a(a-1) / (n(n-1))`. Here `a` is the number of labels that agree with
the provisional orientation and `n` is the number of labels. In `two_snapshot` mode, it uses
exactly two worker-label snapshots per calibration item per repeat.
