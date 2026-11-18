# Experiment 1: the predicted rate

Code for Experiment 1 of *The Minimax Rate of Perturbed Second-Order
Calibration* (NeurIPS 2026, [arXiv:2605.07808](https://arxiv.org/abs/2605.07808)).
It produces Figure 2. Appendix B.1 of the paper gives the full details.

## Purpose

The experiment measures how fast three estimators of the perturbed
second-order calibration error `CE2` converge as the sample size `n` grows. It
uses a synthetic problem where the true `CE2` can be computed to high
precision. For a perturbed score `S~ = (m~, s~)`:

```
CE2 = E[ |eta1(S~) - m~| + |eta2(S~) - (m~^2 + s~)| ]
eta1(q) = E[p(X)   | S~ = q]
eta2(q) = E[p(X)^2 | S~ = q]
```

Each estimator fits `eta1_hat` and `eta2_hat` on `n` samples and evaluates the
plug-in estimate on `n` held-out points `(m_i, s_i)`:

```
CE2_hat = (1/n) sum_i |eta1_hat(S_i) - m_i| + (1/n) sum_i |eta2_hat(S_i) - (m_i^2 + s_i)|
```

This is `common.empirical_ce2`. The experiment reports `|CE2_hat - CE2|`
against `n` on log-log axes, with `n` in
`N_GRID = [500, 1000, 2000, 5000, 10000, 20000, 50000]`. It is repeated for
three perturbation bandwidths, `h` in `H_LIST = [1/4, 1/16, 1/64]`, and
produces one plot for each bandwidth. The polynomial estimator should converge
faster than the bucketing and Nadaraya-Watson baselines.

## Data-generating process

All of this is in `dataset.py`.

- **Covariates.** `X` lies in `R^4`. It is drawn from a mixture of 10 Gaussians
  with equal weights: `X = c_k + Z`, where `Z ~ N(0, I)`. The centres are drawn
  once as `c_k ~ N(0, 1.5^2 I)`.
- **Labels.** `p(X) = sigmoid(w . X)` with no bias, where `w ~ N(0, I) / sqrt(4)`.
  Each point gets two independent labels, `Y1, Y2 ~ Bernoulli(p(X))`. Their
  product `Y1*Y2` is an unbiased estimate of `p(X)^2`.
- **Predictor.** The predictor is an ensemble of 5 MLPs. Each MLP has the
  architecture 4 -> 64 -> 64 -> 1 with ReLU activations, and is trained with
  BCE loss and Adam (learning rate 1e-3, batch size 256, 30 epochs). All
  members train on the same 5,000 fresh samples; member `k` uses the torch seed
  `1000 + k`. The raw score is `m` = the mean of the members' sigmoid outputs,
  clipped to `[1e-6, 1 - 1e-6]`, and `s` = their population variance
  (`ddof = 0`), clipped to `[0, m(1 - m)]`. Scores lie in the rectangle
  `[0, 1] x [0, 0.25]`.
- **Perturbation.** Each coordinate is perturbed independently with a sech
  kernel truncated to its interval (`[0, 1]` for `m`, `[0, 0.25]` for `s`):
  `k_h(t | s) ∝ sech((t - s) / h)`. Samples are drawn with the closed-form
  inverse CDF of `G(t) = 2h * arctan(tanh((t - s) / (2h)))`.
- **Splits.** Each of the 20 seeds gets three splits, `train`, `hyper` and
  `valid`, with 50,000 points each. They come from the RNG
  `default_rng(10000 + seed)`. The raw scores and labels are the same for every
  `h`. Only the perturbation is redrawn, with the RNG
  `default_rng(100000 + 10000*seed + 1/h)`. A run at sample size `n` uses the
  first `n` points of each split, so the smaller samples are prefixes of the
  larger ones.
- **Ground truth.** The true `CE2` is computed separately for each `h`. The
  distribution of the raw scores is approximated with `2^18` scrambled-Sobol
  points (2 independent replicates, seeds 42 and 10042). These points are
  deposited bilinearly on a 1025 x 257 grid over `(m, s)`. The truncated sech
  kernel is then applied exactly as dense matrices: `D = Km W0 Ks^T`, and in
  the same way `eta1 = N1 / D` and `eta2 = N2 / D`. The loss is integrated
  against `D` with trapezoidal weights. The half-span between the two
  replicates is saved as a check on precision.

## Estimators

All three estimators regress `Y1` on `(m, s)` to get `eta1`, and `Y1*Y2` on
`(m, s)` to get `eta2`. Hyperparameters are tuned only on the squared-error fit
loss on the responses (`common.fit_loss`). The true `CE2` is never used for
tuning.

**Bucketing** (`bucketing.py`). The rectangle is split into a regular
`K x K` grid. Each bucket predicts the mean of its training responses. Empty
buckets fall back to the global mean. The grid size is
`K = clip(round(c * (n / h^2)^(1/4)), 2, 200)`. The constant `c` starts from
the grid `{0.25, 0.5, 1, 2, 4, 8}` and is tuned on the `hyper` split. If the
best `c` is at an end of the grid and `K` is not clipped there, the grid is
extended by halving or doubling `c` until the best value is inside it.

**Nadaraya-Watson** (`nadaraya_watson.py`). The `s` axis is multiplied by 4
so that both axes have unit length. The estimator uses an isotropic Gaussian
kernel truncated at `5b`, and finds neighbours with a scikit-learn `BallTree`
in chunks of 2048 query points. The bandwidth is
`b = c * h^(1/2) * n^(-1/4)`, where `c` is chosen on the `hyper` split from
`{0.25, 0.5, 1, 2, 4, 8}`. This grid is not extended. If a point has no
neighbours, or its kernel weights sum to less than 1e-12, it gets the global
mean.

**Polynomial** (`polynomial.py`). This is ridge regression on a tensor
product of Chebyshev polynomials of degree at most `l` in each of
`x = 2m - 1` and `y = 8s - 1`, which gives `(l+1)^2` features. Both responses
are fitted together. The ridge penalty is
`lambda = (alpha + 1e-12) * max(scale, 1)`, where `scale = trace(G) / (l+1)^2`
(or `trace(K) / (l+1)^2` in dual form). `alpha` is chosen from
`{1e-6, 1e-5, ..., 1e-1, 1}`. The estimator works in primal form when
`n >= (l+1)^2` and in dual (kernel) form otherwise, with Cholesky solves in
both. Predictions are clipped to `[0, 1]`.

The candidate degrees for each `(n, h)` are computed as follows:

```
theta(h)  = h*pi/4 + sqrt(1 + (h*pi/4)^2)
l_paper   = ceil(2 * log(n) / log(theta(h)))
L_cap(h)  = min(ceil(-log(1e-4) / log(theta(h))), {1/4: 13, 1/16: 44, 1/64: 88}[h])
l_max     = max(4, min(l_paper, L_cap(h)))
schedule  = l_max, l_max // 2, l_max // 4, ... down to 4 (4 is always included)
```

The hard cap is the limit that applies at every point of the experiment's
grid. `test_hard_cap_always_binds_on_experiment_grid` checks this. Two
variants are reported:

- **Poly (held-out val)**: `(l, alpha)` is chosen on the `hyper` split. The
  model is fitted on `train` and evaluated on `valid`. The results are stored
  as `ce2_val` and `err_val`.
- **Poly (CV)**: uses only the `valid` split. `(l, alpha)` is chosen by 5-fold
  cross-validation (`CV_K = 5`, permutation seed `70000 + 13*seed + n`).
  `CE2_hat` is then computed from closed-form leave-one-out ridge predictions
  `(pred_i - h_ii * Y_i) / (1 - h_ii)` on that same split. The results are
  stored as `ce2_erm` and `err_erm`.

## Requirements

You need Python 3.10+, NumPy, SciPy, PyTorch (CUDA is optional and is used if
available), scikit-learn, joblib and matplotlib. `plot.py` sets
`text.usetex = True`, so it also needs a working LaTeX installation with the
Latin Modern fonts. See `../requirements.txt`.

## How to run

To run every stage, use the following command from inside `exp1/`:

```bash
bash run_all.sh
```

This runs `dataset.py`, `bucketing.py`, `nadaraya_watson.py`, `polynomial.py`
and `plot.py` in that order. It prints how long each stage takes and lists the
final PDFs.

To run the stages one at a time (all paths are relative, so run them from
`exp1/`):

```bash
python dataset.py           # writes ./data/dataset.npz
python bucketing.py         # reads data, writes results/bucketing_h_1_*.npz
python nadaraya_watson.py   # reads data, writes results/nadaraya_watson_h_1_*.npz
python polynomial.py        # reads data, writes results/polynomial_h_1_*.npz
python plot.py              # reads the three results files per h, writes results/experiment1_h_1_*.pdf
```

The dataset is not included in this repository. `python dataset.py` generates
it deterministically into `./data/dataset.npz`. Each estimator runs all three
values of `h`.

## Outputs

The suffix `h_1_{4,16,64}` stands for `h = 1/4, 1/16, 1/64`.

| File | Contents |
|---|---|
| `data/dataset.npz` | Arrays `s{seed}_h_1_{k}_{train,hyper,valid}_{m,s,y1,y2}`; the true value `ce2_gt_h_1_{k}`, with `ce2_gt_half_diff_*` and `ce2_gt_replicates_*`; metadata (`n_seeds`, `n_max`, `h_list`, `gt_qmc_n`, `gt_qmc_repeats`, `gt_grid_m`, `gt_grid_s`) |
| `results/bucketing_h_1_{k}.npz` | `results` (one dict per (seed, n): `seed, n, h, ce2_hat, err, chosen_c, K`), `ce2_gt`, `h` |
| `results/nadaraya_watson_h_1_{k}.npz` | as above, with `bandwidth` in place of `K` |
| `results/polynomial_h_1_{k}.npz` | dicts with `seed, n, h, ce2_val, err_val, ce2_erm, err_erm, chosen_deg_val, chosen_alpha_val, chosen_deg_erm, chosen_alpha_erm, deg_schedule`; `ce2_gt`, `h` |
| `results/experiment1_h_1_{k}.pdf` | The panels of Figure 2 |

The `results` arrays are object arrays, so load them with
`np.load(..., allow_pickle=True)`.

Each plot shows four curves: Bucketing, Nadaraya-Watson, Poly (held-out val)
and Poly (CV). Each curve is the mean error over the 20 seeds, with a shaded
90% Student-t confidence interval for the mean (`t.ppf(0.95, df=19)`). Each
curve has an inline label with its empirical rate `n^a`, where `a` is the
least-squares slope of `log10(mean error)` against `log10(n)`. `plot.py` also
prints these slopes. `plot.per_seed_slope`, which averages the slopes of the
individual seeds, is available but not used in the figure.

## Configuration

Grids that are fixed in the code:

| Name | Value | Location |
|---|---|---|
| `N_GRID` | `[500, 1000, 2000, 5000, 10000, 20000, 50000]` | `common.py` |
| `H_LIST` | `[1/4, 1/16, 1/64]` | `common.py` |
| `CV_K` | `5` | `common.py` |
| `C_GRID` | `[0.25, 0.5, 1, 2, 4, 8]` | `bucketing.py`, `nadaraya_watson.py` |
| `K_MIN`, `K_MAX` | `2`, `200` | `bucketing.py` |
| `LAMBDA_GRID` | `[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]` | `polynomial.py` |
| `DEG_MIN`, `NOISE_FLOOR`, `L_HARD_CAPS` | `4`, `1e-4`, `{4: 13, 16: 44, 64: 88}` (keyed by `1/h`) | `polynomial.py` |

Environment variables. The defaults reproduce the paper. Smaller values are
useful for smoke tests.

| Variable | Default | Meaning |
|---|---|---|
| `EXP1_DIM` | 4 | Dimension of `X` |
| `EXP1_N_MIXTURE` | 10 | Number of mixture components |
| `EXP1_N_MAX` | 50000 | Points per split |
| `EXP1_N_SEEDS` | 20 | Number of seeds |
| `EXP1_ENSEMBLE_SIZE` | 5 | Number of MLPs in the ensemble |
| `EXP1_HIDDEN` | 64 | Width of the MLP hidden layers |
| `EXP1_ENSEMBLE_TRAIN_N` | 5000 | Size of the ensemble's training set |
| `EXP1_ENSEMBLE_EPOCHS` | 30 | Training epochs |
| `EXP1_ENSEMBLE_BATCH` | 256 | Training batch size |
| `EXP1_INFER_BATCH` | 16384 | Inference batch size |
| `EXP1_GT_QMC_N` | 262144 (`2^18`) | Sobol points per ground-truth replicate |
| `EXP1_GT_QMC_REPEATS` | 2 | Number of ground-truth replicates |
| `EXP1_GT_GRID_M`, `EXP1_GT_GRID_S` | 1025, 257 | Ground-truth grid nodes in `m` and `s` |
| `POLY_N_JOBS` | number of CPU cores | Polynomial workers, one seed per worker (capped at `n_seeds`) |
| `POLY_ROW_CHUNK` | 4096 | Row chunk size when the feature matrix is not materialised |
| `POLY_MAX_PHI_MB` | 384 | Largest feature matrix to materialise, in MB |

The `EXP1_*` variables are read only by `dataset.py`. If `EXP1_N_MAX` is below
50000, the downstream scripts still use the full `N_GRID`, so `N_GRID` must be
reduced to match. `polynomial.py` sets the BLAS and OpenMP thread variables to
1 by default and parallelises over seeds with joblib. Bucketing and
Nadaraya-Watson parallelise over (seed, n) jobs on all cores.

## Tests

To run the tests, use the following command from inside `exp1/`:

```bash
python -m pytest test_units.py -v
```

The tests cover the shared helpers, the bucket and bandwidth rules, the
fallbacks for empty buckets and empty neighbourhoods, the Chebyshev features,
the consistency between primal and dual forms, ridge scaling, the ground-truth
grid (mass conservation and normalisation), the degree caps, and dataset
reproducibility. They use small synthetic inputs and do not need
`data/dataset.npz`.

## Runtime

These times are approximate, from a workstation with one GPU and about 10 CPU
workers:

| Stage | Time |
|---|---|
| `dataset.py` | under a minute with a GPU (slower on CPU) |
| `bucketing.py` | seconds |
| `nadaraya_watson.py` | about 15 minutes |
| `polynomial.py` | about 1.5 hours (most of the total) |
| `plot.py` | seconds |
