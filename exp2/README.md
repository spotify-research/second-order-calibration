# Experiment 2: the recalibrated σ² tracks the truth

Code for Experiment 2 of *The Minimax Rate of Perturbed Second-Order Calibration*
(NeurIPS 2026, [arXiv:2605.07808](https://arxiv.org/abs/2605.07808)). It produces
Figure 3 and the plots in Appendix B.2.

## Purpose

A binary predictor reports a score pair `S = (m, s)`: a probability `m ∈ [0, 1]`
and a variance `s ∈ [0, 0.25]`. Write `p(x)` for the true event probability, and
let `Y⁽¹⁾, Y⁽²⁾` be two independent Bernoulli(`p(X)`) snapshots. The conditional moments are

- `η1(S) = E[Y⁽¹⁾ | S] = E[p | S]`
- `η2(S) = E[Y⁽¹⁾ Y⁽²⁾ | S] = E[p² | S]`

The second-order calibration error is

```
CE2(S) = E[ |η1(S) − m| + |η2(S) − (m² + s)| ]
```

and the ground-truth epistemic variance is `σ²_epi(S) = η2(S) − η1(S)²`.

The experiment takes a deep ensemble whose reported variance `s` carries almost
no information about `σ²_epi`, and fits a second-order, Platt-style polynomial
recalibrator `(m, s) ↦ (η̂1, η̂2)`. From that fit it forms the corrected variance

```
σ̂² = clip(η̂2 − η̂1², 0, 0.25)
```

It then compares the reported `σ²` ("before") and `σ̂²` ("after") with the
ground-truth `σ²_epi` in four ways:

1. pointwise scatter plots with Pearson and Spearman correlations;
2. equal-count binned curves;
3. ground-truth CE2 before and after recalibration, over 20 calibration seeds;
4. active-learning style top-k acquisition diagnostics.

The score pair is a deterministic function of `x`. To make CE2 well defined,
the scores are smoothed with the paper's additive sech perturbation. The
bandwidth takes the values `h ∈ {1/4, 1/16, 1/64}`.

## Files

| File | Contents |
|---|---|
| `common.py` | Paths, the h grid (`EXP_H_LIST`), filename tags (`h_1_4`, …), split loaders, target mode |
| `dataset.py` | Synthetic data, deep ensemble, sech perturbation, ground-truth surfaces, per-seed splits |
| `polynomial.py` | Tensor-Chebyshev ridge regression (primal or dual solver) |
| `experiment2.py` | Recalibration, diagnostics, plots and saved results |
| `replot.py` | Redraws the plots from `results/*.npz` without recomputing anything |
| `run_experiment2.sh` | Sets the defaults, then runs `dataset.py` and `experiment2.py` |

## Data and ensemble (`dataset.py`)

- **Data.** `X ∈ R⁴` comes from a mixture of 10 Gaussian components. The centres are drawn
  from `N(0, 1.5² I)` and each component is `N(centre, I)`. The true probability is
  `p(X) = sigmoid(X·w)` with `w ~ N(0, I)/√4` and bias 0, and `Y ~ Bernoulli(p(X))`.
  The data, the ensemble and its training set all come from `np.random.default_rng(0)`.
- **Ensemble.** It has 20 MLPs, each `Linear(4,64)–ReLU–Linear(64,64)–ReLU–Linear(64,1)`.
  Each member trains with `BCEWithLogitsLoss` and Adam (lr 1e-3, batch 256) for 60 epochs.
  In the default `same` mode, all members share one small training set of
  24 points. They differ only in their torch seeds (`1000 + k`). The `bootstrap`
  and `independent` modes are also available.
- **Score pair.** `m` is the mean of the member probabilities, clipped to `[1e-6, 1 − 1e-6]`.
  `s` is their population variance, clipped to `[0, m(1 − m)]`. The small training
  set is what leaves a weak but nonzero epistemic signal in `s`.

## Perturbation and two-snapshot targets

- **Sech kernel.** On an interval `[a, b]`, `k_h(t | c) ∝ sech((t − c)/h)`, truncated
  and renormalised. Its CDF uses `G(t) = 2h·arctan(tanh((t − c)/(2h)))`, which has a
  closed-form inverse (`_G`, `_G_inv`, `sech_perturb`).
- **Perturbed scores.** Each coordinate is perturbed independently by inverse-CDF sampling:
  `M` on `[0, 1]` with bandwidth `h`, and `Σ` on `[0, 0.25]` with `h_s = EXP1_S_H_SCALE · h`.
  The default `h_s` equals `h`. The recalibrator sees only `(M, Σ)`.
- **Ground truth.** Each of 2 replicates draws 2¹⁸ scrambled-Sobol samples of `X`.
  The values `(1, p, p²)` are deposited bilinearly on a 1025 × 257 grid over
  `[0, 1] × [0, 0.25]`. Both axes are then smoothed with the sech kernel, which
  gives the surfaces `η1`, `η2` and `mass`. The replicates are averaged, and their
  half-span is recorded as a precision check. The "before" CE2 for each `h` is
  computed from these surfaces. An unperturbed surface (`gt_raw_*`) is also stored
  for the raw diagnostic.
- **Splits.** For each of 20 seeds there are `train`, `hyper` and `valid` splits,
  each with 50,000 fresh rows. Each row stores `(m, s)` and `EXP2_CAL_SNAPSHOTS = 2`
  Bernoulli snapshots. The per-seed RNG is `default_rng(10000 + seed)`. For each `h`,
  the raw scores are perturbed once with `default_rng(100000 + 10000·seed + round(1/h))`.
  All bandwidths share the same snapshots. Experiment 2 calibrates on `train`,
  evaluates on `valid`, and does not use `hyper`.
- **Targets (`EXP2_TARGET_MODE`).** In `paper` mode (the default), `r1 = Y⁽¹⁾` and
  `r2 = Y⁽¹⁾Y⁽²⁾`. In `pooled` mode, `r1` is the mean of the K snapshots and `r2` is
  the mean product over all unordered pairs. With K = 2, `pooled` mode gives
  `r1 = (Y⁽¹⁾ + Y⁽²⁾)/2`.

## Recalibration and diagnostics (`experiment2.py`)

- **Features.** The scores are mapped to Chebyshev coordinates `x = 2M − 1` and
  `y = 2(Σ/0.25) − 1`. The features form a tensor-product Chebyshev basis of
  fixed degree `l = 12`, giving `(l + 1)² = 169` features. No degree or ridge
  search takes place.
- **Ridge fit.** The ridge parameter is `λ = (α + 1e-12)·trace(G)/(l + 1)²`, with
  `α = EXP2_ERM_ALPHA = 0` by default. The solver is primal (Cholesky on the Gram
  matrix) when `n ≥ 169` and dual otherwise. With the default `cal_n = 20000`,
  it is always primal. The model fits `(r1, r2)` jointly.
- **Moment projection.** `η̂1` is clipped to `[0, 1]`. With
  `EXP2_PROJECT_FEASIBLE_MOMENTS=1`, `η̂2` is clipped to `[η̂1², η̂1]`; otherwise it
  is clipped to `[0, 1]`. Then `σ̂² = clip(η̂2 − η̂1², 0, 0.25)`.
- **After-CE2.** For each calibration seed, the fitted map is evaluated at every
  node of the ground-truth grid, and the ground-truth mass and moments are
  re-deposited onto a grid indexed by `(η̂1, σ̂²)`. The CE2 of that transported
  surface is the "after" value.
- **Evaluation.** The evaluation set is the first 20,000 rows of the `valid` split
  for seed `EXP2_SCATTER_SEED = 0`, scored with the seed-0 model. Bilinear
  interpolation of the ground-truth surfaces gives `σ²_epi` at each `(M, Σ)`.
- **Diagnostics.**
  - Pearson, Spearman and Kendall correlations.
  - Equal-count binned curves (20 bins).
  - Top-k overlap with the oracle, oracle ratio and lift, at budgets of
    0.5%, 1%, 2%, 5% and 10%. The oracle ranks points by ground truth.
  - Threshold-exceedance fractions at 1e-3, 5e-3 and 1e-2. These are saved but
    not plotted.

## How to run

Install the dependencies listed in the repository's `requirements.txt`. A LaTeX
installation is also required, because matplotlib renders text with `text.usetex`
in the Latin Modern Roman serif font. Then run from inside `exp2/`:

```bash
cd exp2
bash run_experiment2.sh
```

The script runs `python dataset.py` and then `python experiment2.py`. The first
step regenerates `./data/dataset.npz`. That file is not shipped with the repository.

To run the steps by hand, use the commands below. The defaults in `dataset.py` and
`experiment2.py` match the defaults the script exports.

```bash
python dataset.py       # writes ./data/dataset.npz
python experiment2.py   # writes ./results/experiment2_*.{pdf,npz}
```

To redraw every plot from the saved results without recomputing, run:

```bash
python replot.py        # reads results/experiment2_h_*.npz
```

To run a subset of bandwidths or change a setting, set environment variables first:

```bash
EXP_H_LIST="1/16,1/64" bash run_experiment2.sh
EXP2_SAVE_RAW_DIAGNOSTIC=1 bash run_experiment2.sh
```

`EXP_H_LIST` is read by both `dataset.py` and `experiment2.py`. The dataset has to
be regenerated with the same list, because the per-`h` surfaces and splits are
stored in `dataset.npz`.

## Outputs

All outputs are written to `./results/`. The tag is `h_1_4`, `h_1_16` or `h_1_64`.

| File | Produced | Contents |
|---|---|---|
| `experiment2_{tag}_before.pdf` | always | Reported `σ²` against ground-truth `σ²_epi` (scatter) |
| `experiment2_{tag}_after.pdf` | always | Corrected `σ̂²` against ground-truth `σ²_epi` (scatter) |
| `experiment2_{tag}_binned.pdf` | always | Equal-count binned curves, before and after |
| `experiment2_{tag}_ce2.pdf` | always | Ground-truth CE2: one "before" bar, and the "after" mean over 20 seeds with 10–90% error bar and per-seed points |
| `experiment2_rankdiag_{tag}_oracle.pdf` | when `EXP2_SAVE_RANK_DIAGNOSTIC=1` (default) | Top-k oracle ratio against acquisition budget |
| `experiment2_rankdiag_{tag}_overlap.pdf` | when `EXP2_SAVE_RANK_DIAGNOSTIC=1` (default) | Top-k overlap with the oracle |
| `experiment2_rawdiag_{tag}.pdf` | only when `EXP2_SAVE_RAW_DIAGNOSTIC=1` (default 0), or when `replot.py` is run | Unperturbed ensemble `s` against the unperturbed ground-truth surface. Every `h` gives the same plot apart from the title. |
| `experiment2_{tag}.npz` | always | All metrics, evaluation arrays, binned curves, top-k arrays, configuration metadata and the raw-score block |

`replot.py` always draws the rank diagnostics. It also draws the raw diagnostic
whenever the npz contains the raw-score arrays, and the arrays are saved by default.

Main npz keys:

- CE2: `ce2_before_gt` (scalar) and `ce2_after_gt` (one value per seed).
- Evaluation arrays: `before_epistemic_eval`, `pointwise_gt_epistemic_eval`,
  `corrected_epistemic_eval`, `transported_gt_epistemic_eval` and
  `corrected_eta{1,2}_eval`.
- Correlations: `before_vs_pointwise_gt_corr`, `after_vs_pointwise_gt_corr`,
  `{before,after}_spearman` and `{before,after}_kendall`.
- Top-k: `topk_fracs` and `{before,after}_topk_{overlap,oracle_ratio,lift}`.
- Binned curves: `{before,after}_bins_{x,y}`.
- Raw-score block: `raw_score_*`.

## Configuration

| Variable | Default | Read by | Meaning |
|---|---|---|---|
| `EXP_H_LIST` | `1/4,1/16,1/64` | `common.py` | Perturbation bandwidths (fractions or decimals) |
| `EXP1_DIM` | 4 | `dataset.py` | Input dimension |
| `EXP1_N_MIXTURE` | 10 | `dataset.py` | Number of mixture components |
| `EXP1_N_MAX` | 50000 | `dataset.py` | Rows per (seed, split) |
| `EXP1_N_SEEDS` | 20 | `dataset.py` | Number of seeds |
| `EXP1_HIDDEN` | 64 | `dataset.py` | Hidden width of each MLP |
| `EXP1_ENSEMBLE_SIZE` | 20 | `dataset.py`, script | Number of ensemble members |
| `EXP1_ENSEMBLE_TRAIN_N` | 24 | `dataset.py`, script | Training-set size per member |
| `EXP1_ENSEMBLE_EPOCHS` | 60 | `dataset.py`, script | Training epochs per member |
| `EXP1_ENSEMBLE_BATCH` / `EXP1_ENSEMBLE_LR` | 256 / 1e-3 | `dataset.py` | Adam batch size and learning rate |
| `EXP1_ENSEMBLE_MEMBER_DATA_MODE` | `same` | `dataset.py`, script | `same`, `bootstrap` or `independent` |
| `EXP1_S_H_SCALE` | 1.0 | `dataset.py`, script | Sets `h_s = scale · h` |
| `EXP1_GT_QMC_N` / `EXP1_GT_QMC_REPEATS` | 262144 / 2 | `dataset.py` | Sobol points per replicate / number of replicates |
| `EXP1_GT_GRID_M` / `EXP1_GT_GRID_S` | 1025 / 257 | `dataset.py` | Size of the ground-truth grid |
| `EXP1_SAVE_RAW_SPLITS` | 1 | `dataset.py` | Store the unperturbed splits (needed for the raw diagnostic) |
| `EXP1_INFER_BATCH` / `EXP1_TORCH_NUM_THREADS` | 16384 / 1 | `dataset.py` | Inference batch size / torch CPU threads |
| `EXP2_CAL_SNAPSHOTS` | 2 | `dataset.py`, script | Bernoulli snapshots per row (at least 2) |
| `EXP2_TARGET_MODE` | `paper` | `common.py`, script | `paper` or `pooled` targets |
| `EXP2_FIXED_DEGREE` | 12 | `experiment2.py`, script | Chebyshev degree per axis |
| `EXP2_ERM_ALPHA` | 0.0 | `experiment2.py` | Ridge α, added to the 1e-12 floor |
| `EXP2_CAL_SPLIT` / `EXP2_EVAL_SPLIT` | `train` / `valid` | `experiment2.py` | Calibration split / evaluation split |
| `EXP2_CAL_N` / `EXP2_EVAL_N` | 20000 / 20000 | `experiment2.py` | Rows used, capped at `EXP1_N_MAX` |
| `EXP2_N_REPEATS` | 20 | `experiment2.py` | Calibration seeds, capped at `EXP1_N_SEEDS` |
| `EXP2_SCATTER_SEED` | 0 | `experiment2.py` | Seed used for the evaluation and scatter plots |
| `EXP2_PROJECT_FEASIBLE_MOMENTS` | 1 | `experiment2.py`, script | Project `η̂2` onto `[η̂1², η̂1]` |
| `EXP2_RANK_BINS` | 20 | `experiment2.py` | Number of equal-count bins |
| `EXP2_TOPK_FRACS` | `0.005,0.01,0.02,0.05,0.1` | `experiment2.py` | Top-k budgets |
| `EXP2_SAVE_RANK_DIAGNOSTIC` | 1 | `experiment2.py`, script | Write the rankdiag PDFs |
| `EXP2_SAVE_RAW_DIAGNOSTIC` | 0 | `experiment2.py`, script | Write the rawdiag PDFs |
| `EXP2_PRED_CHUNK` | 65536 | `experiment2.py` | Prediction batch size |

"script" means that `run_experiment2.sh` exports the variable. The values it
exports are the same as the Python defaults. Values already set in the
environment take precedence.

## Runtime

Runtime has not been benchmarked. Most of the time goes to `dataset.py`, which
trains the ensemble, builds the ground-truth surfaces and scores
20 seeds × 3 splits × 50,000 rows. It uses CUDA when available. `experiment2.py`
fits 20 small ridge problems with 169 features each for every `h`. `replot.py`
takes seconds.
