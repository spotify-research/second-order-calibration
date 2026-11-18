#!/usr/bin/env python3
"""
exp_med_strong: synthetic medical-style referral experiment with strong median
separation between first-order and second-order calibration.

Story
-----
A screening model outputs a compressed score pair
    S = (m_hat, s_hat),
where
    m_hat = posterior mean of a latent patient-specific response probability,
    s_hat = posterior epistemic variance of that probability.

Only *borderline* cases are eligible for an expensive specialist molecular panel.
If the panel is not ordered, the clinic gives a generic treatment with 50% chance
of success.
If the panel is ordered, it returns one noisy binary recommendation Y' and the
clinic follows it with a targeted treatment. The patient's eventual treatment
response Y is modeled as an independent Bernoulli draw with the same latent
probability theta. Therefore the expected treatment success after referral is

    P(Y' = Y | theta) = theta^2 + (1 - theta)^2.

So the *true* gain from referral is

    g(theta) = theta^2 + (1-theta)^2 - 0.5 - c
             = 2 theta^2 - 2 theta + 0.5 - c,

where c is the referral cost.

This is the key point: the conditional expected gain given the score S depends
on the *second conditional moment* of theta,

    E[g(theta) | S] = 2 eta2(S) - 2 eta1(S) + 0.5 - c,

where
    eta1(S) = E[theta | S],
    eta2(S) = E[theta^2 | S].

A first-order calibrator that learns only eta1 cannot in general recover this.
In the synthetic DGP below, there are many borderline cases with the *same mean*
correctness/risk but different epistemic spread. That makes the second moment
operationally necessary rather than just diagnostically interesting.

Calibration protocol
--------------------
By default, calibration uses the paper-style two-snapshot targets on a separate
calibration split:
    Y1 ~ Bernoulli(theta),
    Y2 ~ Bernoulli(theta),
independent given theta.
Then
    E[Y1 | S] = eta1(S),
    E[Y1 Y2 | S] = eta2(S).

Important honesty note
----------------------
To keep the experiment focused on calibration and decision-making, the screening
score pair (m_hat, s_hat) is simulated *directly* rather than produced by a
trained network. This is intentional: it removes optimizer/training noise and
makes the experiment about whether second-order calibration helps when the score
pair truly contains useful second-order information.
"""

from __future__ import annotations

import os
for _name in (
    'OMP_NUM_THREADS',
    'OPENBLAS_NUM_THREADS',
    'MKL_NUM_THREADS',
    'VECLIB_MAXIMUM_THREADS',
    'NUMEXPR_NUM_THREADS',
):
    os.environ.setdefault(_name, '1')

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.chebyshev import chebvander

plt.rcParams.update({
    'text.usetex': True,
    'font.size': 12,
    'font.family': 'lmodern',
    'legend.fontsize': 17,
})


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_REPEATS = int(os.environ.get('EXP_MED_N_REPEATS', '200'))
N_CAL = int(os.environ.get('EXP_MED_CAL_N', '3000'))
N_EVAL = int(os.environ.get('EXP_MED_EVAL_N', '8000'))
N_DIAG = int(os.environ.get('EXP_MED_DIAG_N', '3000'))
DEGREE_1D = int(os.environ.get('EXP_MED_DEGREE_1D', '6'))
DEGREE_2D = int(os.environ.get('EXP_MED_DEGREE_2D', '4'))
RIDGE = float(os.environ.get('EXP_MED_RIDGE', '1e-4'))
REFERRAL_COST = float(os.environ.get('EXP_MED_REFERRAL_COST', '0.06'))
TARGET_MODE = os.environ.get('EXP_MED_TARGET_MODE', 'two_snapshot').strip().lower()
if TARGET_MODE not in {'two_snapshot', 'theta'}:
    raise ValueError('EXP_MED_TARGET_MODE must be two_snapshot or theta')
OUT_PREFIX = os.environ.get('EXP_MED_OUT_PREFIX', 'exp_med_strong')

# Decision policy: refer iff predicted gain > tau. tau = 0 is the natural
# cost-aware threshold since the gain definition already absorbs the cost.
TAU_MIN = float(os.environ.get('EXP_MED_TAU_MIN', '-0.05'))
TAU_MAX = float(os.environ.get('EXP_MED_TAU_MAX', '0.30'))
N_TAUS = int(os.environ.get('EXP_MED_N_TAUS', '71'))
TAU_GRID = np.linspace(TAU_MIN, TAU_MAX, N_TAUS, dtype=np.float64)
TAU_ZERO_IDX = int(np.argmin(np.abs(TAU_GRID - 0.0)))
# Per-repeat scalar metric: average realized gain over tau in [0, PGI_TAU_MAX].
# This integrates the policy over a band of cost regimes; ties of magnitude-
# miscalibrated methods at tau = 0 are broken once tau exceeds their (biased)
# predicted gain on positive-gain cases.
PGI_TAU_MAX = float(os.environ.get('EXP_MED_PGI_TAU_MAX', '0.20'))
PGI_MASK = (TAU_GRID >= -1e-12) & (TAU_GRID <= PGI_TAU_MAX + 1e-12)

# Borderline triage cohort: only cases in this screening range are eligible for
# specialist referral.
BORDERLINE_LOW = float(os.environ.get('EXP_MED_BORDERLINE_LOW', '0.35'))
BORDERLINE_HIGH = float(os.environ.get('EXP_MED_BORDERLINE_HIGH', '0.65'))

GROUP_NAMES = np.array([
    'clear negative',
    'clear positive',
    'common borderline aleatoric',
    'borderline hidden-subtype epistemic',
], dtype=object)
GROUP_WEIGHTS = np.array([0.25, 0.25, 0.35, 0.15], dtype=np.float64)


# -----------------------------------------------------------------------------
# Synthetic score-space DGP
# -----------------------------------------------------------------------------
# The crucial construction is that groups 2 and 3 have very similar m_hat but
# very different theta distributions. In group 2, theta is genuinely close to
# 0.5 (aleatoric). In group 3, theta is near 0.1 or 0.9 with equal probability
# (epistemic hidden-subtype mixture), so E[theta | m,s] ~= 0.5 but E[theta^2 |
# m,s] is much larger. That is why first-order calibration fails even on the 2D
# score pair, while second-order calibration can succeed.

def sample_score_population(n: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    group = rng.choice(len(GROUP_WEIGHTS), size=n, p=GROUP_WEIGHTS)
    m = np.zeros(n, dtype=np.float64)
    s = np.zeros(n, dtype=np.float64)
    theta = np.zeros(n, dtype=np.float64)
    subtype = np.full(n, -1, dtype=np.int64)

    for k in range(len(GROUP_WEIGHTS)):
        idx = np.flatnonzero(group == k)
        z = idx.size
        if z == 0:
            continue

        if k == 0:
            m[idx] = np.clip(rng.normal(0.08, 0.025, size=z), 0.01, 0.20)
            s[idx] = np.clip(rng.normal(0.0030, 0.0008, size=z), 0.0003, 0.010)
            theta[idx] = np.clip(rng.normal(0.08, 0.020, size=z), 0.01, 0.20)
        elif k == 1:
            m[idx] = np.clip(rng.normal(0.92, 0.025, size=z), 0.80, 0.99)
            s[idx] = np.clip(rng.normal(0.0030, 0.0008, size=z), 0.0003, 0.010)
            theta[idx] = np.clip(rng.normal(0.92, 0.020, size=z), 0.80, 0.99)
        elif k == 2:
            m[idx] = np.clip(rng.normal(0.50, 0.035, size=z), BORDERLINE_LOW, BORDERLINE_HIGH)
            s[idx] = np.clip(rng.normal(0.010, 0.002, size=z), 0.002, 0.030)
            theta[idx] = np.clip(rng.normal(0.50, 0.040, size=z), BORDERLINE_LOW, BORDERLINE_HIGH)
        elif k == 3:
            m[idx] = np.clip(rng.normal(0.50, 0.035, size=z), BORDERLINE_LOW, BORDERLINE_HIGH)
            s[idx] = np.clip(rng.normal(0.060, 0.010, size=z), 0.030, 0.120)
            subtype[idx] = rng.integers(0, 2, size=z)
            theta[idx] = np.where(
                subtype[idx] == 0,
                np.clip(rng.normal(0.12, 0.030, size=z), 0.02, 0.25),
                np.clip(rng.normal(0.88, 0.030, size=z), 0.75, 0.98),
            )

    return m, s, theta, group.astype(np.int64), subtype


# -----------------------------------------------------------------------------
# Polynomial calibration utilities
# -----------------------------------------------------------------------------
def cheb_design_1d(m: np.ndarray, degree: int) -> np.ndarray:
    x = 2.0 * np.asarray(m, dtype=np.float64) - 1.0
    return np.ascontiguousarray(chebvander(x, degree), dtype=np.float64)


def cheb_design_2d(m: np.ndarray, s: np.ndarray, degree: int) -> np.ndarray:
    x = 2.0 * np.asarray(m, dtype=np.float64) - 1.0
    y = 2.0 * (np.asarray(s, dtype=np.float64) / 0.25) - 1.0
    vx = chebvander(x, degree)
    vy = chebvander(y, degree)
    return np.ascontiguousarray((vx[:, :, None] * vy[:, None, :]).reshape(len(m), -1), dtype=np.float64)


def fit_poly_erm(design: np.ndarray, targets: np.ndarray, ridge: float) -> np.ndarray:
    gram = design.T @ design
    rhs = design.T @ targets
    gram[np.diag_indices_from(gram)] += ridge
    return np.linalg.solve(gram, rhs)


def project_moments(eta1: np.ndarray, eta2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    eta1 = np.clip(np.asarray(eta1, dtype=np.float64), 0.0, 1.0)
    eta2 = np.asarray(eta2, dtype=np.float64)
    eta2 = np.clip(eta2, eta1 * eta1, eta1)
    return eta1, eta2


# -----------------------------------------------------------------------------
# Decision problem and evaluation
# -----------------------------------------------------------------------------
def true_referral_gain(theta: np.ndarray, referral_cost: float) -> np.ndarray:
    theta = np.asarray(theta, dtype=np.float64)
    return 2.0 * theta * theta - 2.0 * theta + 0.5 - referral_cost


def plugin_gain_from_mean(mu: np.ndarray, referral_cost: float) -> np.ndarray:
    mu = np.clip(np.asarray(mu, dtype=np.float64), 0.0, 1.0)
    return 2.0 * mu * mu - 2.0 * mu + 0.5 - referral_cost


def plugin_gain_from_raw_moments(m: np.ndarray, s: np.ndarray, referral_cost: float) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    eta1 = m
    eta2 = s + m * m
    eta1, eta2 = project_moments(eta1, eta2)
    return 2.0 * eta2 - 2.0 * eta1 + 0.5 - referral_cost


def exact_gain_from_moments(eta1: np.ndarray, eta2: np.ndarray, referral_cost: float) -> np.ndarray:
    eta1, eta2 = project_moments(eta1, eta2)
    return 2.0 * eta2 - 2.0 * eta1 + 0.5 - referral_cost


def threshold_policy_curve(
    score: np.ndarray, true_gain: np.ndarray, taus: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Realized utility per 100 borderline cases under the policy
    "refer iff score > tau", evaluated on a grid of taus.

    A magnitude-calibrated score (predicted = E[g | S]) makes the natural
    threshold tau = 0 optimal. Methods whose magnitude is biased downward
    refer too few cases at tau >= 0 and lose realized utility once tau
    exceeds their biased predicted gain on truly positive-gain cases.
    """
    score = np.asarray(score, dtype=np.float64)
    true_gain = np.asarray(true_gain, dtype=np.float64)
    taus = np.asarray(taus, dtype=np.float64)
    n = score.shape[0]
    gain_per100 = np.zeros_like(taus, dtype=np.float64)
    frac_referred = np.zeros_like(taus, dtype=np.float64)
    if n == 0:
        return gain_per100, frac_referred
    order = np.argsort(score, kind='mergesort')
    sorted_score = score[order]
    sorted_gain = true_gain[order]
    cum_from_top = np.cumsum(sorted_gain[::-1])[::-1]
    idx = np.searchsorted(sorted_score, taus, side='right')
    n_above = n - idx
    above = idx < n
    gain_per100[above] = 100.0 * cum_from_top[idx[above]] / n
    frac_referred[:] = n_above / n
    return gain_per100, frac_referred


def sem95(arr: np.ndarray) -> np.ndarray:
    if arr.shape[0] <= 1:
        return np.zeros(arr.shape[1:], dtype=np.float64)
    return 1.96 * arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])


def duplicate_groups(curves: Dict[str, np.ndarray], rel_tol: float = 0.05) -> Dict[str, list[str]]:
    """Group curves that are visually indistinguishable.

    Two curves are considered tied when their max absolute difference is at most
    ``rel_tol`` times the overall y-range across all curves. ``rel_tol=0.05``
    catches curves that overlap on screen while leaving genuinely distinct
    curves separated.
    """
    names = list(curves.keys())
    if not names:
        return {}
    all_vals = np.concatenate([np.ravel(curves[n]) for n in names])
    y_range = max(float(np.ptp(all_vals)), 1e-12)
    atol = rel_tol * y_range
    used: set[str] = set()
    out: Dict[str, list[str]] = {}
    for i, name in enumerate(names):
        if name in used:
            continue
        group = [name]
        used.add(name)
        for other in names[i + 1:]:
            if other in used:
                continue
            if np.max(np.abs(np.asarray(curves[name]) - np.asarray(curves[other]))) <= atol:
                group.append(other)
                used.add(other)
        out[name] = group
    return out


# -----------------------------------------------------------------------------
# Main experiment
# -----------------------------------------------------------------------------
def main() -> None:
    method_order = [
        'raw plug-in from m',
        'raw plug-in from (m, s)',
        'raw variance s',
        '1D first-order Platt',
        '2D first-order Platt',
        '1D second-order Platt',
        '2D second-order Platt',
        'oracle gain',
    ]

    gain_curves = {k: [] for k in method_order}
    frac_referred_curves = {k: [] for k in method_order}
    pgi_values = {k: [] for k in method_order}
    gain_at_zero_values = {k: [] for k in method_order}
    representative = None

    for rep in range(N_REPEATS):
        rng = np.random.default_rng(rep)

        m_cal, s_cal, theta_cal, group_cal, subtype_cal = sample_score_population(N_CAL, rng)
        m_eval, s_eval, theta_eval, group_eval, subtype_eval = sample_score_population(N_EVAL, rng)

        cal_mask = (m_cal >= BORDERLINE_LOW) & (m_cal <= BORDERLINE_HIGH)
        eval_mask = (m_eval >= BORDERLINE_LOW) & (m_eval <= BORDERLINE_HIGH)

        m_cal = m_cal[cal_mask]
        s_cal = s_cal[cal_mask]
        theta_cal = theta_cal[cal_mask]
        group_cal = group_cal[cal_mask]
        subtype_cal = subtype_cal[cal_mask]

        m_eval = m_eval[eval_mask]
        s_eval = s_eval[eval_mask]
        theta_eval = theta_eval[eval_mask]
        group_eval = group_eval[eval_mask]
        subtype_eval = subtype_eval[eval_mask]

        if TARGET_MODE == 'two_snapshot':
            y1 = (rng.random(len(theta_cal)) < theta_cal).astype(np.float64)
            y2 = (rng.random(len(theta_cal)) < theta_cal).astype(np.float64)
            first_order_targets = y1[:, None]
            second_order_targets = np.column_stack([y1, y1 * y2])
        else:
            first_order_targets = theta_cal[:, None]
            second_order_targets = np.column_stack([theta_cal, theta_cal * theta_cal])

        d1_cal = cheb_design_1d(m_cal, DEGREE_1D)
        d2_cal = cheb_design_2d(m_cal, s_cal, DEGREE_2D)
        d1_eval = cheb_design_1d(m_eval, DEGREE_1D)
        d2_eval = cheb_design_2d(m_eval, s_eval, DEGREE_2D)

        beta_1d_first = fit_poly_erm(d1_cal, first_order_targets, RIDGE)
        beta_2d_first = fit_poly_erm(d2_cal, first_order_targets, RIDGE)
        beta_1d_second = fit_poly_erm(d1_cal, second_order_targets, RIDGE)
        beta_2d_second = fit_poly_erm(d2_cal, second_order_targets, RIDGE)

        mu_1d = np.clip((d1_eval @ beta_1d_first).ravel(), 0.0, 1.0)
        mu_2d = np.clip((d2_eval @ beta_2d_first).ravel(), 0.0, 1.0)
        eta1_1d, eta2_1d = (d1_eval @ beta_1d_second).T
        eta1_2d, eta2_2d = (d2_eval @ beta_2d_second).T
        eta1_1d, eta2_1d = project_moments(eta1_1d, eta2_1d)
        eta1_2d, eta2_2d = project_moments(eta1_2d, eta2_2d)

        true_gain = true_referral_gain(theta_eval, REFERRAL_COST)
        scores = {
            'raw plug-in from m': plugin_gain_from_mean(m_eval, REFERRAL_COST),
            'raw plug-in from (m, s)': plugin_gain_from_raw_moments(m_eval, s_eval, REFERRAL_COST),
            'raw variance s': s_eval,
            '1D first-order Platt': plugin_gain_from_mean(mu_1d, REFERRAL_COST),
            '2D first-order Platt': plugin_gain_from_mean(mu_2d, REFERRAL_COST),
            '1D second-order Platt': exact_gain_from_moments(eta1_1d, eta2_1d, REFERRAL_COST),
            '2D second-order Platt': exact_gain_from_moments(eta1_2d, eta2_2d, REFERRAL_COST),
            'oracle gain': true_gain,
        }

        for name in method_order:
            gain_curve, frac_curve = threshold_policy_curve(scores[name], true_gain, TAU_GRID)
            gain_curves[name].append(gain_curve)
            frac_referred_curves[name].append(frac_curve)
            integral = float(np.trapezoid(gain_curve[PGI_MASK], TAU_GRID[PGI_MASK]))
            pgi_values[name].append(integral / PGI_TAU_MAX)
            gain_at_zero_values[name].append(float(gain_curve[TAU_ZERO_IDX]))

        if representative is None:
            representative = {
                'm_eval': m_eval[:N_DIAG].copy(),
                's_eval': s_eval[:N_DIAG].copy(),
                'theta_eval': theta_eval[:N_DIAG].copy(),
                'group_eval': group_eval[:N_DIAG].copy(),
                'subtype_eval': subtype_eval[:N_DIAG].copy(),
            }

    gain_mean = {k: np.mean(np.vstack(v), axis=0) for k, v in gain_curves.items()}
    gain_sem = {k: sem95(np.vstack(v)) for k, v in gain_curves.items()}
    frac_mean = {k: np.mean(np.vstack(v), axis=0) for k, v in frac_referred_curves.items()}
    dup = duplicate_groups(gain_mean)

    # Save arrays.
    out_npz = RESULTS_DIR / f'{OUT_PREFIX}.npz'
    np.savez_compressed(
        out_npz,
        tau_grid=TAU_GRID,
        target_mode=np.array(TARGET_MODE),
        n_repeats=np.array(N_REPEATS),
        n_cal=np.array(N_CAL),
        n_eval=np.array(N_EVAL),
        degree_1d=np.array(DEGREE_1D),
        degree_2d=np.array(DEGREE_2D),
        ridge=np.array(RIDGE),
        referral_cost=np.array(REFERRAL_COST),
        borderline_low=np.array(BORDERLINE_LOW),
        borderline_high=np.array(BORDERLINE_HIGH),
        pgi_tau_max=np.array(PGI_TAU_MAX),
        duplicate_groups=np.array([f'{k}: ' + ', '.join(v) for k, v in dup.items()], dtype=object),
        method_order=np.array(method_order, dtype=object),
        group_names=GROUP_NAMES,
        m_diag=representative['m_eval'],
        s_diag=representative['s_eval'],
        theta_diag=representative['theta_eval'],
        group_diag=representative['group_eval'],
        subtype_diag=representative['subtype_eval'],
        **{f'gain_curve_mean__{k}': gain_mean[k] for k in method_order},
        **{f'gain_curve_sem95__{k}': gain_sem[k] for k in method_order},
        **{f'frac_referred_mean__{k}': frac_mean[k] for k in method_order},
        **{f'pgi__{k}': np.asarray(pgi_values[k], dtype=np.float64) for k in method_order},
        **{f'gain_at_zero__{k}': np.asarray(gain_at_zero_values[k], dtype=np.float64) for k in method_order},
    )

    # Plot. Each figure is saved to its own PDF; no side-by-side panels.

    # Figure 1: score space.
    fig1, ax = plt.subplots(figsize=(6.0, 4.5), constrained_layout=True)
    sc = ax.scatter(
        representative['m_eval'],
        representative['s_eval'],
        c=true_referral_gain(representative['theta_eval'], REFERRAL_COST),
        s=12,
        alpha=0.75,
        linewidths=0.0,
    )
    ax.set_title('Eligible borderline cohort in score space')
    ax.set_xlabel(r'screening posterior mean $m$')
    ax.set_ylabel(r'screening epistemic variance $s$')
    cb = fig1.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label('true specialist value')
    # Sketch the group means.
    for g in [2, 3]:
        idx = representative['group_eval'] == g
        if np.any(idx):
            ax.scatter(
                representative['m_eval'][idx].mean(),
                representative['s_eval'][idx].mean(),
                s=90,
                marker='x',
                linewidths=2.0,
                color='k',
            )
            ax.text(
                representative['m_eval'][idx].mean() + 0.005,
                representative['s_eval'][idx].mean() + 0.002,
                GROUP_NAMES[g],
                fontsize=8,
            )
    out_pdf_score = RESULTS_DIR / f'{OUT_PREFIX}_score_space.pdf'
    fig1.savefig(out_pdf_score)
    plt.close(fig1)

    # Figure 2: decision curve.
    # Distinct marker per method, with marker positions staggered within tie
    # groups so that overlapping curves remain individually visible.
    method_markers = {
        'raw plug-in from m': 'o',
        'raw plug-in from (m, s)': 's',
        'raw variance s': '*',
        '1D first-order Platt': '^',
        '2D first-order Platt': 'v',
        '1D second-order Platt': 'D',
        '2D second-order Platt': 'P',
        'oracle gain': 'X',
    }
    # Wong colorblind-safe palette; grey for raw baselines, blue family for
    # first-order, green/vermillion for second-order, black for oracle.
    method_colors = {
        'raw plug-in from m':       '#56B4E9',
        'raw plug-in from (m, s)':  '#E69F00',
        'raw variance s':           '#F0E442',
        '1D first-order Platt':     '#CC79A7',
        '2D first-order Platt':     '#0072B2',
        '1D second-order Platt':    '#D55E00',
        '2D second-order Platt':    '#009E73',
        'oracle gain':              '#000000',
    }
    tie_group_of = {name: [name] for name in method_order}
    for _rep, _grp in dup.items():
        for _name in _grp:
            tie_group_of[_name] = _grp
    n_points = len(TAU_GRID)
    marker_stride = max(2, n_points // 7)

    plot_methods = [n for n in method_order if n != 'raw variance s']
    fig2, ax = plt.subplots(figsize=(3.5, 2.6), constrained_layout=True)
    for name in plot_methods:
        y = gain_mean[name]
        e = gain_sem[name]
        group = tie_group_of[name]
        gsize = len(group)
        j = group.index(name)
        # Stagger marker offset within tie group so markers from tied curves
        # interleave instead of stacking.
        marker_start = j % marker_stride
        if gsize > 1:
            others = [g for g in group if g != name]
            label = rf"{name} ($\approx$ {', '.join(others)})"
        else:
            label = name
        line, = ax.plot(
            TAU_GRID,
            y,
            color=method_colors[name],
            linewidth=1.5,
            marker=method_markers[name],
            markevery=(marker_start, marker_stride),
            markersize=5.0,
            markeredgecolor='white',
            markeredgewidth=0.5,
            label=label,
        )
        ax.fill_between(TAU_GRID, y - e, y + e, color=method_colors[name], alpha=0.14)
    ax.axhline(0.0, color='k', linewidth=1.0, alpha=0.5)
    ax.axvline(0.0, color='k', linewidth=1.0, alpha=0.35, linestyle='--')
    ax.set_xlabel(r'Referral threshold $\tau$')
    ax.set_ylabel('Realized gain per 100 cases')
    handles, labels = ax.get_legend_handles_labels()
    out_pdf_curve = RESULTS_DIR / f'{OUT_PREFIX}_decision_curve.pdf'
    fig2.savefig(out_pdf_curve)
    plt.close(fig2)

    # Figure 2b: legend for the decision curve, saved as a separate PDF so it
    # can be placed alongside the plot in the paper without crowding the axes.
    fig2b, ax_legend = plt.subplots(figsize=(5.5, 1.9), constrained_layout=True)
    ax_legend.axis('off')
    ax_legend.legend(
        handles,
        labels,
        loc='center',
        ncol=1,
        frameon=False,
        handlelength=2.5,
        borderaxespad=0.0,
    )
    out_pdf_legend = RESULTS_DIR / f'{OUT_PREFIX}_decision_curve_legend.pdf'
    fig2b.savefig(out_pdf_legend, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig2b)

    # Figure 3: per-repeat policy-gain distribution.
    fig3, ax = plt.subplots(figsize=(6.0, 4.5), constrained_layout=True)
    box_methods = [
        '1D first-order Platt',
        '2D first-order Platt',
        '1D second-order Platt',
        'raw plug-in from (m, s)',
        '2D second-order Platt',
        'oracle gain',
    ]
    box_data = [np.asarray(pgi_values[k], dtype=np.float64) for k in box_methods]
    bp = ax.boxplot(box_data, patch_artist=True, showfliers=False)
    for patch in bp['boxes']:
        patch.set_alpha(0.35)
    ax.set_xticklabels(['1D-1st', '2D-1st', '1D-2nd', 'raw (m,s)', '2D-2nd', 'Oracle'], rotation=20)
    ax.set_title('Per-repeat policy-gain distribution')
    ax.set_ylabel(rf'Mean realized gain per 100 cases over $\tau \in [0, {PGI_TAU_MAX:.2f}]$')
    out_pdf_box = RESULTS_DIR / f'{OUT_PREFIX}_pgi_distribution.pdf'
    fig3.savefig(out_pdf_box)
    plt.close(fig3)

    # Console summary.
    print(f'Saved {out_npz}')
    print(f'Saved {out_pdf_score}')
    print(f'Saved {out_pdf_curve}')
    print(f'Saved {out_pdf_legend}')
    print(f'Saved {out_pdf_box}')
    print('=== exp_med_strong summary ===')
    print('Synthetic borderline-referral medical story: only borderline patients are eligible;')
    print('specialist referral value is exact second-order in theta and should therefore')
    print('benefit from eta2-calibration when high-s epistemic cases share the same mean as')
    print('low-s aleatoric cases.')
    print(f'Calibration label accounting: {2 * N_CAL * N_REPEATS} synthetic specialist labels in total in two-snapshot mode.')

    for name in method_order:
        arr_pgi = np.asarray(pgi_values[name], dtype=np.float64)
        arr_at0 = np.asarray(gain_at_zero_values[name], dtype=np.float64)
        print(
            f'  {name}: PGI mean={arr_pgi.mean():.5f}, median={np.median(arr_pgi):.5f}; '
            f'gain@tau=0 mean={arr_at0.mean():.5f}, median={np.median(arr_at0):.5f}'
        )

    diff_2d2_vs_raw_ms = np.asarray(pgi_values['2D second-order Platt']) - np.asarray(pgi_values['raw plug-in from (m, s)'])
    diff_2d2_vs_2d1 = np.asarray(pgi_values['2D second-order Platt']) - np.asarray(pgi_values['2D first-order Platt'])
    print(
        f'  Median PGI lift of 2D second-order over raw plug-in (m, s) = {np.median(diff_2d2_vs_raw_ms):.5f}; '
        f'win fraction = {np.mean(diff_2d2_vs_raw_ms > 0.0):.3f}'
    )
    print(
        f'  Median PGI lift of 2D second-order over 2D first-order = {np.median(diff_2d2_vs_2d1):.5f}; '
        f'win fraction = {np.mean(diff_2d2_vs_2d1 > 0.0):.3f}'
    )


if __name__ == '__main__':
    main()
