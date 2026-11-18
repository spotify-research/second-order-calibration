# Experiment 3: specialist referral (synthetic)

Code for Experiment 3 of the paper, "magnitudes, not just rankings" (Figure 4, Appendix B.3).
The experiment is designed so that first-order calibration is not enough in a *typical*
repeat, not just on average.

## Protocol

1. Simulate a screening model score pair `(m, s)`:
   - `m` = posterior mean response probability
   - `s` = posterior epistemic variance
2. Restrict to a "borderline" cohort with `m` in `[0.35, 0.65]`. Only these cases are
   eligible for specialist referral.
3. If not referred, the clinic gives a generic treatment with 50% success.
4. If referred, one specialist molecular panel vote `Y'` is obtained and the targeted
   treatment follows that vote.
5. The patient's eventual response `Y` is modeled as an independent Bernoulli draw with
   the same latent `theta`.

Therefore the true gain from referral is

```
g(theta) = theta^2 + (1 - theta)^2 - 0.5 - c
         = 2 theta^2 - 2 theta + 0.5 - c
```

where `c` is the referral cost. Given the score `S`, the expected gain is
`E[g(theta) | S] = 2 eta2(S) - 2 eta1(S) + 0.5 - c`, with `eta1(S) = E[theta | S]` and
`eta2(S) = E[theta^2 | S]`. It depends on the *second conditional moment* of `theta`.

## Why the experiment is strong

Among eligible borderline cases there are two clinically distinct groups with very similar
posterior mean `m`:

- **Common borderline aleatoric cases:** `theta` near 0.5, low `s`.
- **Hidden-subtype epistemic cases:** `theta` near 0.1 or 0.9 (the simulator draws it
  around 0.12 or 0.88), high `s`.

Because `E[theta | m, s]` is still about 0.5 in the hidden-subtype group, a first-order
calibrator on the full 2D score cannot recover the specialist value. A second-order
calibrator can, because `E[theta^2 | m, s]` is much larger there.

## Methods

- raw plug-in from `m`
- raw plug-in from `(m, s)` (uses `eta1 = m`, `eta2 = s + m^2`)
- raw variance `s` (computed and saved, but left out of the plots)
- 1D first-order Platt
- 2D first-order Platt
- 1D second-order Platt
- 2D second-order Platt
- oracle gain

## Main metric

The decision policy is "refer iff predicted gain > `tau`", evaluated over a sweep of referral
thresholds `tau`. `tau = 0` is the natural cost-aware threshold, because the predicted gain
already absorbs the referral cost `c`. The figure reports realized net utility per 100
borderline cases as a function of `tau`.

This is *not* an ordering-only metric. Two methods with the same ranking can have different
threshold-policy curves if their score magnitudes differ. This is what surfaces the benefit
of second-order Platt over raw plug-in `(m, s)`. Both rank borderline cases similarly, but
raw plug-in `(m, s)` underestimates the gain magnitude on the epistemic group. It therefore
loses utility as soon as the cost-aware threshold becomes mildly conservative.

Per-repeat scalar:

```
PGI = mean realized gain per 100 cases over tau in [0, PGI_TAU_MAX]   (default PGI_TAU_MAX = 0.20)
    = (1 / PGI_TAU_MAX) * trapezoidal integral of realized_gain(tau) over that range
```

The script also reports gain at `tau = 0` as a sanity check.

## Running

Default run (the settings used for the paper figures):

```bash
bash run_exp_med.sh
```

Large repeat run:

```bash
EXP_MED_N_REPEATS=1000 bash run_exp_med.sh
```

## Files written

All outputs go to `results/`, which is created next to the script. The filenames below use the default prefix `exp_med_strong` (set with `EXP_MED_OUT_PREFIX`).

| File | Contents |
| --- | --- |
| `results/exp_med_strong.npz` | Curves, PGI values, and settings |
| `results/exp_med_strong_score_space.pdf` | Score-space diagnostic |
| `results/exp_med_strong_decision_curve.pdf` | Realized gain vs. `tau` |
| `results/exp_med_strong_decision_curve_legend.pdf` | Legend for the decision curve |
| `results/exp_med_strong_pgi_distribution.pdf` | Per-repeat PGI distribution |

## Environment variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `EXP_MED_N_REPEATS` | `200` | Number of repeats |
| `EXP_MED_CAL_N` | `3000` | Calibration sample size (before the borderline filter) |
| `EXP_MED_EVAL_N` | `8000` | Evaluation sample size (before the borderline filter) |
| `EXP_MED_DIAG_N` | `3000` | Points kept for the score-space diagnostic plot |
| `EXP_MED_DEGREE_1D` | `6` | Chebyshev degree for the 1D calibrators |
| `EXP_MED_DEGREE_2D` | `4` | Chebyshev degree for the 2D calibrators |
| `EXP_MED_RIDGE` | `1e-4` | Ridge penalty for the calibrator fits |
| `EXP_MED_REFERRAL_COST` | `0.06` | Referral cost `c` |
| `EXP_MED_TARGET_MODE` | `two_snapshot` | `two_snapshot` or `theta` |
| `EXP_MED_BORDERLINE_LOW` | `0.35` | Lower end of the borderline `m` range |
| `EXP_MED_BORDERLINE_HIGH` | `0.65` | Upper end of the borderline `m` range |
| `EXP_MED_TAU_MIN` | `-0.05` | Left end of the threshold sweep |
| `EXP_MED_TAU_MAX` | `0.30` | Right end of the threshold sweep |
| `EXP_MED_N_TAUS` | `71` | Grid points in `[TAU_MIN, TAU_MAX]` |
| `EXP_MED_PGI_TAU_MAX` | `0.20` | Upper bound for PGI integration |
| `EXP_MED_OUT_PREFIX` | `exp_med_strong` | Output filename prefix |
