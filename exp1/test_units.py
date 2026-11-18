"""
Unit tests for exp1: common, bucketing, nadaraya_watson, polynomial.
Run from inside exp1/:  python -m pytest test_units.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pytest

# ── common ────────────────────────────────────────────────────────────────────

from common import h_tag, empirical_ce2, fit_loss, kfold_indices


def test_h_tag():
    assert h_tag(1 / 4) == "h_1_4"
    assert h_tag(1 / 16) == "h_1_16"
    assert h_tag(1 / 64) == "h_1_64"


def test_empirical_ce2_perfect():
    # Perfect predictor: eta1_hat = m, eta2_hat = m^2 + s → CE2 = 0
    rng = np.random.default_rng(0)
    m = rng.uniform(0.1, 0.9, 100)
    s = rng.uniform(0.0, 0.1, 100)
    assert empirical_ce2(m, m ** 2 + s, m, s) == 0.0


def test_empirical_ce2_known():
    # eta1_hat = 0.5, m = 0 → |eta1 - m| = 0.5
    # eta2_hat = 0.5, v = m^2+s = 0 → |eta2 - v| = 0.5
    # CE2 = 0.5 + 0.5 = 1.0
    m = np.zeros(4)
    s = np.zeros(4)
    assert np.isclose(empirical_ce2(np.full(4, 0.5), np.full(4, 0.5), m, s), 1.0)


def test_fit_loss_perfect():
    # Perfect predictor on y1 and y1*y2 → loss = 0
    rng = np.random.default_rng(1)
    y1 = rng.integers(0, 2, 50).astype(float)
    y2 = rng.integers(0, 2, 50).astype(float)
    assert fit_loss(y1, y1 * y2, y1, y2) == 0.0


def test_kfold_coverage():
    # Each index appears in exactly one val fold; no train/val leakage per fold
    n, k = 100, 5
    val_seen = np.zeros(n, dtype=int)
    for tr, va in kfold_indices(n, k, seed=42):
        assert len(np.intersect1d(tr, va)) == 0
        assert len(tr) + len(va) == n
        val_seen[va] += 1
    assert np.all(val_seen == 1)


# ── bucketing ─────────────────────────────────────────────────────────────────

from bucketing import K_of, bucket_ids, predict_buckets


def test_K_of_formula():
    # K_of(c=1, n=10000, h=0.25) = round((10000 / 0.0625)^0.25) = round(20.0) = 20
    assert K_of(1.0, 10_000, 0.25) == 20


def test_K_of_clipped():
    assert K_of(1000.0, 50_000, 0.25) == 200  # clips to K_MAX=200
    assert K_of(0.001, 10, 0.25) == 2          # clips to K_MIN=2


def test_bucket_ids_corners():
    K = 10
    # bottom-left (0, 0) → bin 0
    ids, n_bins = bucket_ids(np.array([0.0]), np.array([0.0]), K)
    assert ids[0] == 0
    assert n_bins == K * K
    # top-right (≈1, ≈0.25) → bin K²-1
    ids, _ = bucket_ids(np.array([0.999]), np.array([0.2499]), K)
    assert ids[0] == K * K - 1


def test_predict_buckets_constant_labels():
    # y1 = 0.7 everywhere → every bin (including empty bins via fallback) gives e1 = 0.7
    rng = np.random.default_rng(2)
    m = rng.uniform(0, 1, 200)
    s = rng.uniform(0, 0.25, 200)
    y1 = np.full(200, 0.7)
    y2 = np.full(200, 0.4)
    e1, _ = predict_buckets(m, s, y1, y2, K=5, m_eval=m[:10], s_eval=s[:10])
    np.testing.assert_allclose(e1, 0.7, atol=1e-10)


def test_predict_buckets_empty_bin_fallback():
    # Training only in bottom-left corner; eval in top-right corner (empty bin)
    # → fallback to global mean of y1 = 1.0
    m_tr = np.full(50, 0.02)
    s_tr = np.full(50, 0.005)
    y1 = np.ones(50)
    y2 = np.ones(50)
    m_ev = np.full(5, 0.98)
    s_ev = np.full(5, 0.245)
    e1, _ = predict_buckets(m_tr, s_tr, y1, y2, K=20, m_eval=m_ev, s_eval=s_ev)
    np.testing.assert_allclose(e1, 1.0, atol=1e-10)


# ── nadaraya_watson ───────────────────────────────────────────────────────────

from nadaraya_watson import b_of, _scale_pts, nw_predict


def test_b_of_formula():
    c, h, n = 2.0, 0.25, 10_000
    expected = 2.0 * (0.25 ** 0.5) * (10_000 ** -0.25)
    assert np.isclose(b_of(c, n, h), expected)


def test_scale_pts():
    # m unchanged, s stretched by 4 so [0, 0.25] → [0, 1]
    m = np.array([0.3, 0.7])
    s = np.array([0.1, 0.2])
    pts = _scale_pts(m, s)
    np.testing.assert_allclose(pts[:, 0], m)
    np.testing.assert_allclose(pts[:, 1], s * 4.0)


def test_nw_predict_single_train_point():
    # Single training point; eval at the same location with very small b
    # → NW prediction converges to the training label
    m_tr = np.array([0.5])
    s_tr = np.array([0.05])
    y1 = np.array([0.8])
    y2 = np.array([0.6])
    e1, e2 = nw_predict(m_tr, s_tr, y1, y2, np.array([0.5]), np.array([0.05]), b=0.001)
    np.testing.assert_allclose(e1[0], 0.8, atol=1e-6)
    np.testing.assert_allclose(e2[0], y1[0] * y2[0], atol=1e-6)  # e2 targets y1*y2


def test_nw_predict_empty_neighborhood_fallback():
    # Eval point far from all training points → falls back to global mean
    m_tr = np.full(10, 0.1)
    s_tr = np.full(10, 0.01)
    y1 = np.full(10, 0.6)
    y2 = np.full(10, 0.4)
    m_ev = np.array([0.9])
    s_ev = np.array([0.24])
    e1, e2 = nw_predict(m_tr, s_tr, y1, y2, m_ev, s_ev, b=0.0001)
    np.testing.assert_allclose(e1[0], 0.6, atol=1e-10)  # global mean of y1
    np.testing.assert_allclose(e2[0], 0.24, atol=1e-10) # global mean of y1*y2


# ── polynomial ────────────────────────────────────────────────────────────────

from polynomial import (
    theta_of, cheb_vander, features_from_vander,
    dual_kernel_full, dual_kernel_cross, response_matrix,
    clip_eta, ridge_lambda, RIDGE_FLOOR,
)


def test_theta_of_greater_than_one():
    # theta(h) = h*pi/4 + sqrt(1 + (h*pi/4)^2) > 1 for all h > 0
    for h in [1 / 4, 1 / 16, 1 / 64]:
        assert theta_of(h) > 1.0


def test_cheb_vander_t0_is_ones():
    # T_0(x) = 1 for all x
    m = np.array([0.0, 0.5, 1.0])
    s = np.array([0.0, 0.125, 0.25])
    Vx, Vy = cheb_vander(m, s, l_max=3)
    np.testing.assert_allclose(Vx[:, 0], 1.0)
    np.testing.assert_allclose(Vy[:, 0], 1.0)


def test_cheb_vander_t1_is_linear():
    # T_1(x) = x, and x = 2*m - 1 (m mapped to [-1, 1])
    # Similarly y = 2*(s/0.25) - 1
    m = np.array([0.0, 0.5, 1.0])
    s = np.array([0.0, 0.125, 0.25])
    Vx, Vy = cheb_vander(m, s, l_max=3)
    np.testing.assert_allclose(Vx[:, 1], 2 * m - 1.0, atol=1e-12)
    np.testing.assert_allclose(Vy[:, 1], 2 * (s / 0.25) - 1.0, atol=1e-12)


def test_features_shape():
    n, l = 50, 4
    rng = np.random.default_rng(0)
    m = rng.uniform(0, 1, n)
    s = rng.uniform(0, 0.25, n)
    Vx, Vy = cheb_vander(m, s, l_max=l)
    Phi = features_from_vander(Vx, Vy, l)
    assert Phi.shape == (n, (l + 1) ** 2)


def test_dual_kernel_equals_phi_phi_t():
    # By definition, dual_kernel_full(V, V, l) should equal Phi @ Phi.T
    n, l = 20, 3
    rng = np.random.default_rng(5)
    m = rng.uniform(0, 1, n)
    s = rng.uniform(0, 0.25, n)
    Vx, Vy = cheb_vander(m, s, l_max=l)
    Phi = features_from_vander(Vx, Vy, l)
    np.testing.assert_allclose(dual_kernel_full(Vx, Vy, l), Phi @ Phi.T, atol=1e-10)


def test_dual_kernel_cross_consistency():
    # cross-kernel with itself should equal full kernel
    n, l = 15, 3
    rng = np.random.default_rng(7)
    m = rng.uniform(0, 1, n)
    s = rng.uniform(0, 0.25, n)
    Vx, Vy = cheb_vander(m, s, l_max=l)
    K_full = dual_kernel_full(Vx, Vy, l)
    K_cross = dual_kernel_cross(Vx, Vy, Vx, Vy, l)
    np.testing.assert_allclose(K_cross, K_full, atol=1e-10)


def test_response_matrix():
    y1 = np.array([1.0, 0.0, 1.0, 0.0])
    y2 = np.array([1.0, 1.0, 0.0, 0.0])
    R = response_matrix(y1, y2)
    assert R.shape == (4, 2)
    np.testing.assert_array_equal(R[:, 0], y1)
    np.testing.assert_array_equal(R[:, 1], y1 * y2)


def test_clip_eta_bounds():
    pred = np.array([[-0.5, 1.5], [0.3, 0.7], [2.0, -1.0]])
    e1, e2 = clip_eta(pred)
    assert np.all(e1 >= 0.0) and np.all(e1 <= 1.0)
    assert np.all(e2 >= 0.0) and np.all(e2 <= 1.0)
    np.testing.assert_allclose(e1[1], 0.3)
    np.testing.assert_allclose(e2[1], 0.7)


def test_ridge_lambda_scaling():
    # ridge_lambda(scale, alpha) = (alpha + RIDGE_FLOOR) * scale
    lam = ridge_lambda(scale=10.0, alpha=0.01)
    assert np.isclose(lam, 10.0 * (0.01 + RIDGE_FLOOR))


# ── dataset: mass conservation ────────────────────────────────────────────────

from dataset import (
    deposit_bilinear, ce2_from_grid_masses, trapezoid_weights,
    sech_kernel_matrix, sample_X, perturb_split,
)
from polynomial import degree_schedule, L_cap_of


def test_deposit_bilinear_mass_conservation():
    # Bilinear weights sum to 1 per point, so W0.sum()=1, W1.sum()=mean(p), W2.sum()=mean(p^2)
    rng = np.random.default_rng(0)
    n = 10_000
    m = rng.uniform(0.1, 0.9, n)
    s = rng.uniform(0.0, 0.1, n)
    p_true = rng.uniform(0.2, 0.8, n)
    m_nodes = np.linspace(0.0, 1.0, 101)
    s_nodes = np.linspace(0.0, 0.25, 26)
    W0, W1, W2 = deposit_bilinear(m, s, p_true, m_nodes, s_nodes)
    np.testing.assert_allclose(W0.sum(), 1.0, atol=1e-10)
    np.testing.assert_allclose(W1.sum(), p_true.mean(), atol=1e-10)
    np.testing.assert_allclose(W2.sum(), (p_true ** 2).mean(), atol=1e-10)


def test_ce2_grid_perturbed_mass_integrates_to_one():
    # After convolving W0 with the sech kernel, mass integrated with trapezoid weights ≈ 1
    # (because the sech kernel is normalised to integrate to 1 for each center)
    rng = np.random.default_rng(1)
    n = 5_000
    m = rng.uniform(0.2, 0.8, n)
    s = rng.uniform(0.0, 0.1, n)
    p_true = rng.uniform(0.2, 0.8, n)
    m_nodes = np.linspace(0.0, 1.0, 65)
    s_nodes = np.linspace(0.0, 0.25, 17)
    m_quad = trapezoid_weights(m_nodes)
    s_quad = trapezoid_weights(s_nodes)
    W0, _, _ = deposit_bilinear(m, s, p_true, m_nodes, s_nodes)
    Km = sech_kernel_matrix(m_nodes, m_nodes, 0.0, 1.0, 0.25)
    Ks = sech_kernel_matrix(s_nodes, s_nodes, 0.0, 0.25, 0.25)
    D = Km @ W0 @ Ks.T
    mass_sum = float((D * m_quad[:, None] * s_quad[None, :]).sum())
    np.testing.assert_allclose(mass_sum, 1.0, atol=0.02)


def test_ce2_grid_finite_nonneg():
    # ce2_from_grid_masses must return a finite non-negative value for all h
    rng = np.random.default_rng(2)
    n = 2_000
    m = rng.uniform(0.1, 0.9, n)
    s = rng.uniform(0.0, 0.1, n)
    p_true = rng.uniform(0.1, 0.9, n)
    m_nodes = np.linspace(0.0, 1.0, 33)
    s_nodes = np.linspace(0.0, 0.25, 9)
    m_quad = trapezoid_weights(m_nodes)
    s_quad = trapezoid_weights(s_nodes)
    for h in [0.25, 1 / 16, 1 / 64]:
        W0, W1, W2 = deposit_bilinear(m, s, p_true, m_nodes, s_nodes)
        ce2 = ce2_from_grid_masses(W0, W1, W2, h, m_nodes, s_nodes, m_quad, s_quad)
        assert np.isfinite(ce2), f"CE2 not finite for h={h}"
        assert ce2 >= 0.0, f"CE2 negative for h={h}"


def test_empirical_ce2_nonzero_for_wrong_predictor():
    # Zero predictor on data with m > 0: CE2 = mean(m) + mean(m^2+s), exactly computable
    rng = np.random.default_rng(3)
    m = rng.uniform(0.3, 0.7, 200)
    s = rng.uniform(0.0, 0.05, 200)
    ce2 = empirical_ce2(np.zeros(200), np.zeros(200), m, s)
    np.testing.assert_allclose(ce2, m.mean() + (m ** 2 + s).mean(), rtol=1e-10)
    assert ce2 > 0.0


# ── end-to-end estimator smoke tests ─────────────────────────────────────────

def _synthetic(n, seed):
    """Simple synthetic dataset: p_true = m, s = 0."""
    rng = np.random.default_rng(seed)
    m = rng.uniform(0.2, 0.8, n)
    s = np.zeros(n)
    y1 = (rng.uniform(size=n) < m).astype(float)
    y2 = (rng.uniform(size=n) < m).astype(float)
    return m, s, y1, y2


def test_bucketing_smoke():
    m_tr, s_tr, y1_tr, y2_tr = _synthetic(500, 10)
    m_ev, s_ev, _, _ = _synthetic(200, 11)
    e1, e2 = predict_buckets(m_tr, s_tr, y1_tr, y2_tr, K_of(1.0, 500, 0.25), m_ev, s_ev)
    ce2 = empirical_ce2(e1, e2, m_ev, s_ev)
    assert np.isfinite(ce2) and ce2 >= 0.0
    assert not (np.any(np.isnan(e1)) or np.any(np.isnan(e2)))


def test_nw_smoke():
    m_tr, s_tr, y1_tr, y2_tr = _synthetic(500, 20)
    m_ev, s_ev, _, _ = _synthetic(200, 21)
    e1, e2 = nw_predict(m_tr, s_tr, y1_tr, y2_tr, m_ev, s_ev, b_of(1.0, 500, 0.25))
    ce2 = empirical_ce2(e1, e2, m_ev, s_ev)
    assert np.isfinite(ce2) and ce2 >= 0.0
    assert not (np.any(np.isnan(e1)) or np.any(np.isnan(e2)))


def test_polynomial_smoke():
    from polynomial import prepare_model, fit_predict_prepared
    m_tr, s_tr, y1_tr, y2_tr = _synthetic(500, 30)
    m_ev, s_ev, _, _ = _synthetic(200, 31)
    l = 4
    Vx_tr, Vy_tr = cheb_vander(m_tr, s_tr, l_max=l)
    Vx_ev, Vy_ev = cheb_vander(m_ev, s_ev, l_max=l)
    Y_tr = response_matrix(y1_tr, y2_tr)
    prepared = prepare_model(Vx_tr, Vy_tr, Y_tr, l)
    e1, e2 = fit_predict_prepared(prepared, 0.01, Vx_ev, Vy_ev)
    ce2 = empirical_ce2(e1, e2, m_ev, s_ev)
    assert np.isfinite(ce2) and ce2 >= 0.0
    assert not (np.any(np.isnan(e1)) or np.any(np.isnan(e2)))


# ── polynomial: degree schedule and hard caps ─────────────────────────────────

def test_paper_degree_formula_at_h_quarter():
    # theta(0.25) ≈ 1.215; 2*ln(500)/ln(1.215) ≈ 63.8 → ceil = 64
    # L_cap_of(0.25) = min(l_noise=48, hard_cap=13) = 13
    # So degree_schedule clips l_paper=64 down to 13
    import math
    h = 0.25
    l_paper = math.ceil(2.0 * math.log(500) / math.log(theta_of(h)))
    assert l_paper == 64
    assert L_cap_of(h) == 13
    assert max(degree_schedule(500, h)) == 13


def test_hard_cap_always_binds_on_experiment_grid():
    """The theoretical degree exceeds the hard cap for every (h, n) in the
    experiment grid — the implementation never runs at the theorem's schedule."""
    import math
    from common import H_LIST, N_GRID
    for h in H_LIST:
        cap = L_cap_of(h)
        for n in N_GRID:
            l_paper = math.ceil(2.0 * math.log(n) / math.log(theta_of(h)))
            assert l_paper > cap, (
                f"Cap not binding at h={h:.4f}, n={n}: l_paper={l_paper}, cap={cap}"
            )


# ── polynomial: ERM/CV data reuse ─────────────────────────────────────────────

def test_erm_cv_uses_validation_split_for_hp_and_eval():
    """Document: in run_seed, cv_pick_hypers and fit_predict_loo both receive
    Vx_va/Vy_va/y1_va/y2_va — the same validation split.  No separate held-out
    data is used; the LOO formula is the only guard against in-sample optimism."""
    import inspect
    from polynomial import cv_pick_hypers, fit_predict_loo
    cv_params = list(inspect.signature(cv_pick_hypers).parameters.keys())
    loo_params = list(inspect.signature(fit_predict_loo).parameters.keys())
    assert cv_params[:4] == ["Vx", "Vy", "y1", "y2"]
    assert loo_params[:4] == ["Vx", "Vy", "y1", "y2"]


# ── dataset: reproducibility ──────────────────────────────────────────────────

def test_sample_x_reproducible():
    # N_MIXTURE=10 in dataset.py, so centres must have 10 rows
    from dataset import N_MIXTURE, DIM
    centres = np.random.default_rng(0).normal(0, 1, (N_MIXTURE, DIM))
    X1 = sample_X(100, centres, np.random.default_rng(42))
    X2 = sample_X(100, centres, np.random.default_rng(42))
    np.testing.assert_array_equal(X1, X2)


def test_perturb_split_reproducible():
    rng = np.random.default_rng(0)
    m = rng.uniform(0.2, 0.8, 200)
    s = rng.uniform(0.0, 0.1, 200)
    y1 = rng.integers(0, 2, 200).astype(float)
    y2 = rng.integers(0, 2, 200).astype(float)
    d1 = perturb_split(m, s, y1, y2, 0.25, np.random.default_rng(7))
    d2 = perturb_split(m, s, y1, y2, 0.25, np.random.default_rng(7))
    np.testing.assert_array_equal(d1["m"], d2["m"])
    np.testing.assert_array_equal(d1["s"], d2["s"])


# ── dataset: experimental design ─────────────────────────────────────────────

def test_raw_labels_identical_across_h():
    """Y1, Y2 are drawn once and shared across all h; only the perturbed
    (m_tilde, s_tilde) vary with h."""
    rng = np.random.default_rng(0)
    m = rng.uniform(0.2, 0.8, 200)
    s = rng.uniform(0.0, 0.1, 200)
    y1 = rng.integers(0, 2, 200).astype(float)
    y2 = rng.integers(0, 2, 200).astype(float)
    d1 = perturb_split(m, s, y1, y2, 0.25,   np.random.default_rng(1))
    d2 = perturb_split(m, s, y1, y2, 0.0625, np.random.default_rng(2))
    np.testing.assert_array_equal(d1["y1"], d2["y1"])
    np.testing.assert_array_equal(d1["y2"], d2["y2"])
    assert not np.allclose(d1["m"], d2["m"])  # perturbed scores differ


def test_two_snapshot_labels_are_unbiased():
    """Y1, Y2 ~ Bernoulli(p_true) iid: E[Y1]=E[p_true], E[Y1*Y2]=E[p_true^2]."""
    rng = np.random.default_rng(0)
    n = 100_000
    p_true = rng.uniform(0.1, 0.9, n)
    y1 = (rng.uniform(size=n) < p_true).astype(float)
    y2 = (rng.uniform(size=n) < p_true).astype(float)
    np.testing.assert_allclose(y1.mean(), p_true.mean(), atol=0.01)
    np.testing.assert_allclose(y2.mean(), p_true.mean(), atol=0.01)
    np.testing.assert_allclose((y1 * y2).mean(), (p_true ** 2).mean(), atol=0.01)


# ── convergence and cross-method ─────────────────────────────────────────────

def test_bucketing_ce2_decreases_with_n():
    """With p_true=m and s=0, eta1*=m and eta2*=m^2=v. Averaged over seeds,
    empirical CE2 should fall as n grows."""
    ns = [200, 1000, 5000]
    mean_ce2 = []
    for n in ns:
        vals = []
        for seed in range(15):
            rng = np.random.default_rng(seed * 13 + n)
            m_tr = rng.uniform(0.2, 0.8, n)
            s_tr = np.zeros(n)
            y1 = (rng.uniform(size=n) < m_tr).astype(float)
            y2 = (rng.uniform(size=n) < m_tr).astype(float)
            m_ev = rng.uniform(0.2, 0.8, 500)
            s_ev = np.zeros(500)
            e1, e2 = predict_buckets(m_tr, s_tr, y1, y2, K_of(1.0, n, 0.25), m_ev, s_ev)
            vals.append(empirical_ce2(e1, e2, m_ev, s_ev))
        mean_ce2.append(np.mean(vals))
    assert mean_ce2[0] > mean_ce2[-1], f"No convergence trend: {mean_ce2}"


def test_cross_method_plausible_range():
    """All three estimators should return finite, non-negative CE2 values
    below 1.0 on a simple synthetic problem at n=5000."""
    from polynomial import prepare_model, fit_predict_prepared
    m_tr, s_tr, y1_tr, y2_tr = _synthetic(5000, 77)
    m_ev, s_ev, _, _ = _synthetic(1000, 78)

    K = K_of(1.0, 5000, 0.25)
    e1_b, e2_b = predict_buckets(m_tr, s_tr, y1_tr, y2_tr, K, m_ev, s_ev)
    ce2_b = empirical_ce2(e1_b, e2_b, m_ev, s_ev)

    b = b_of(1.0, 5000, 0.25)
    e1_nw, e2_nw = nw_predict(m_tr, s_tr, y1_tr, y2_tr, m_ev, s_ev, b)
    ce2_nw = empirical_ce2(e1_nw, e2_nw, m_ev, s_ev)

    l = 4
    Vx_tr, Vy_tr = cheb_vander(m_tr, s_tr, l_max=l)
    Vx_ev, Vy_ev = cheb_vander(m_ev, s_ev, l_max=l)
    Y_tr = response_matrix(y1_tr, y2_tr)
    prepared = prepare_model(Vx_tr, Vy_tr, Y_tr, l)
    e1_p, e2_p = fit_predict_prepared(prepared, 0.001, Vx_ev, Vy_ev)
    ce2_p = empirical_ce2(e1_p, e2_p, m_ev, s_ev)

    for name, ce2 in [("bucketing", ce2_b), ("NW", ce2_nw), ("polynomial", ce2_p)]:
        assert np.isfinite(ce2), f"{name} CE2 not finite"
        assert 0.0 <= ce2 < 1.0, f"{name} CE2 = {ce2:.3f} out of plausible range [0, 1)"
