"""
Polynomial estimator for CE2 (optimized fixed version).

Main changes relative to polynomial.py:
  * use all available cores by default (still override with POLY_N_JOBS);
  * reuse dual-mode kernel matrices exactly via separability of the tensor basis;
  * precompute eval/train cross-kernels once per degree in dual mode;
  * avoid retaining prepared models for every degree during hyper search;
  * avoid giant validation feature matrices in primal CV when they exceed memory cap;
  * compute primal leverages from a single triangular solve.
"""

import os

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_name, "1")

import math
import numpy as np
from joblib import Parallel, delayed
from numpy.polynomial.chebyshev import chebvander
from scipy.linalg import cho_factor, cho_solve, solve_triangular

from common import (load_dataset, get_split, empirical_ce2, fit_loss,
                    N_GRID, RESULTS_DIR, H_LIST, h_tag, kfold_indices, CV_K)

DEG_MIN = 4
NOISE_FLOOR = 1e-4
L_HARD_CAPS = {4: 13, 16: 44, 64: 88}
DTYPE = np.float64
RIDGE_FLOOR = 1e-12
ROW_CHUNK = int(os.environ.get("POLY_ROW_CHUNK", "4096"))
MAX_PHI_BYTES = int(float(os.environ.get("POLY_MAX_PHI_MB", "384")) * 1024 ** 2)
N_JOBS = int(os.environ.get("POLY_N_JOBS", str(max(1, os.cpu_count() or 1))))
LAMBDA_GRID = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]


def theta_of(h):
    return h * math.pi / 4.0 + math.sqrt(1.0 + h * h * math.pi * math.pi / 16.0)


def hard_cap_of(h):
    return L_HARD_CAPS[int(round(1.0 / h))]


def L_cap_of(h):
    th = theta_of(h)
    l_noise = int(math.ceil(-math.log(NOISE_FLOOR) / math.log(th)))
    return min(l_noise, hard_cap_of(h))


def degree_schedule(n, h):
    l_paper = int(math.ceil(2.0 * math.log(n) / math.log(theta_of(h))))
    l_max = max(DEG_MIN, min(l_paper, L_cap_of(h)))
    degs = [l_max]
    while degs[-1] // 2 >= DEG_MIN:
        degs.append(degs[-1] // 2)
        if degs[-1] == DEG_MIN:
            break
    if DEG_MIN not in degs:
        degs.append(DEG_MIN)
    out = []
    seen = set()
    for d in degs:
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def cheb_vander(m, s, l_max):
    x = 2.0 * m.astype(np.float64) - 1.0
    y = 2.0 * (s.astype(np.float64) / 0.25) - 1.0
    return (np.ascontiguousarray(chebvander(x, l_max), dtype=DTYPE),
            np.ascontiguousarray(chebvander(y, l_max), dtype=DTYPE))


def features_from_vander(Vx, Vy, l):
    Vx_l = Vx[:, :l + 1]
    Vy_l = Vy[:, :l + 1]
    Phi = (Vx_l[:, :, None] * Vy_l[:, None, :]).reshape(len(Vx), -1)
    return np.ascontiguousarray(Phi, dtype=DTYPE)


def dual_kernel_full(Vx, Vy, l):
    Vx_l = Vx[:, :l + 1]
    Vy_l = Vy[:, :l + 1]
    return (Vx_l @ Vx_l.T) * (Vy_l @ Vy_l.T)


def dual_kernel_cross(Vx_a, Vy_a, Vx_b, Vy_b, l):
    Vx_a_l = Vx_a[:, :l + 1]
    Vy_a_l = Vy_a[:, :l + 1]
    Vx_b_l = Vx_b[:, :l + 1]
    Vy_b_l = Vy_b[:, :l + 1]
    return (Vx_a_l @ Vx_b_l.T) * (Vy_a_l @ Vy_b_l.T)


def response_matrix(y1, y2):
    return np.column_stack([y1, y1 * y2]).astype(DTYPE, copy=False)


def phi_bytes(n, d):
    return n * d * np.dtype(DTYPE).itemsize


def can_materialize_phi(n, d):
    return phi_bytes(n, d) <= MAX_PHI_BYTES


def ridge_lambda(scale, alpha):
    scale = max(float(scale), 1.0)
    return alpha * scale + RIDGE_FLOOR * scale


def solve_primal_from_gram(G, rhs, alpha, scale=None):
    d = G.shape[0]
    if scale is None:
        scale = float(np.trace(G)) / d
    lam = ridge_lambda(scale, alpha)
    A = G.copy()
    A[np.diag_indices_from(A)] += lam
    cho, low = cho_factor(A, lower=True, check_finite=False)
    beta = cho_solve((cho, low), rhs, check_finite=False)
    return lam, cho, low, beta


def solve_dual_from_kernel(K, Y, alpha, scale=None):
    n = K.shape[0]
    if scale is None:
        scale = float(np.trace(K)) / max(n, 1)
    lam = ridge_lambda(scale, alpha)
    A = K.copy()
    A[np.diag_indices_from(A)] += lam
    cho, low = cho_factor(A, lower=True, check_finite=False)
    coef = cho_solve((cho, low), Y, check_finite=False)
    return lam, cho, low, coef


def build_gram_rhs(Vx, Vy, Y, l):
    n = Vx.shape[0]
    d = (l + 1) ** 2
    if can_materialize_phi(n, d):
        Phi = features_from_vander(Vx, Vy, l)
        return Phi.T @ Phi, Phi.T @ Y, Phi
    G = np.zeros((d, d), dtype=DTYPE)
    rhs = np.zeros((d, Y.shape[1]), dtype=DTYPE)
    for start in range(0, n, ROW_CHUNK):
        end = min(start + ROW_CHUNK, n)
        Pb = features_from_vander(Vx[start:end], Vy[start:end], l)
        G += Pb.T @ Pb
        rhs += Pb.T @ Y[start:end]
    return G, rhs, None


def build_subset_gram_rhs(Vx, Vy, Y, idx, l):
    m = len(idx)
    d = (l + 1) ** 2
    if can_materialize_phi(m, d):
        Phi = features_from_vander(Vx[idx], Vy[idx], l)
        Y_sub = Y[idx]
        return Phi.T @ Phi, Phi.T @ Y_sub, Phi, Y_sub
    G = np.zeros((d, d), dtype=DTYPE)
    rhs = np.zeros((d, Y.shape[1]), dtype=DTYPE)
    for start in range(0, m, ROW_CHUNK):
        sl = idx[start:start + ROW_CHUNK]
        Pb = features_from_vander(Vx[sl], Vy[sl], l)
        Yb = Y[sl]
        G += Pb.T @ Pb
        rhs += Pb.T @ Yb
    return G, rhs, None, None


def predict_primal(Vx_ev, Vy_ev, l, beta, Phi_ev=None):
    if Phi_ev is not None:
        return Phi_ev @ beta
    n = Vx_ev.shape[0]
    pred = np.empty((n, beta.shape[1]), dtype=DTYPE)
    for start in range(0, n, ROW_CHUNK):
        end = min(start + ROW_CHUNK, n)
        Pb = features_from_vander(Vx_ev[start:end], Vy_ev[start:end], l)
        pred[start:end] = Pb @ beta
    return pred


def clip_eta(pred):
    e1 = np.clip(pred[:, 0].astype(np.float64, copy=False), 0.0, 1.0)
    e2 = np.clip(pred[:, 1].astype(np.float64, copy=False), 0.0, 1.0)
    return e1, e2


def ensure_positive_denom(denom):
    min_denom = float(np.min(denom))
    if min_denom <= 0.0:
        if min_denom < -1e-10:
            raise RuntimeError(f"ridge LOO produced non-positive denominator: {min_denom}")
        denom = np.maximum(denom, 1e-12)
    return denom


def prepare_model(Vx_tr, Vy_tr, Y_tr, l):
    n = Vx_tr.shape[0]
    d = (l + 1) ** 2
    if n < d:
        K = dual_kernel_full(Vx_tr, Vy_tr, l)
        scale = float(np.trace(K)) / d
        return {
            "mode": "dual",
            "l": l,
            "Vx_tr": Vx_tr,
            "Vy_tr": Vy_tr,
            "K": K,
            "Y": Y_tr,
            "scale": scale,
        }
    G, rhs, Phi_tr = build_gram_rhs(Vx_tr, Vy_tr, Y_tr, l)
    scale = float(np.trace(G)) / d
    return {
        "mode": "primal",
        "l": l,
        "Phi_tr": Phi_tr,
        "G": G,
        "rhs": rhs,
        "Y": Y_tr,
        "scale": scale,
        "Vx_tr": Vx_tr,
        "Vy_tr": Vy_tr,
    }


def fit_predict_prepared(prepared, alpha, Vx_ev, Vy_ev, Phi_ev=None, K_ev_tr=None):
    if prepared["mode"] == "dual":
        if K_ev_tr is None:
            K_ev_tr = dual_kernel_cross(Vx_ev, Vy_ev, prepared["Vx_tr"], prepared["Vy_tr"], prepared["l"])
        _, _, _, coef = solve_dual_from_kernel(prepared["K"], prepared["Y"], alpha, prepared["scale"])
        pred = K_ev_tr @ coef
        return clip_eta(pred)

    _, _, _, beta = solve_primal_from_gram(prepared["G"], prepared["rhs"], alpha, prepared["scale"])
    pred = predict_primal(Vx_ev, Vy_ev, prepared["l"], beta, Phi_ev=Phi_ev)
    return clip_eta(pred)


def fit_predict_loo(Vx, Vy, y1, y2, l, alpha):
    Y = response_matrix(y1, y2)
    n = len(Y)
    d = (l + 1) ** 2

    if n < d:
        K = dual_kernel_full(Vx, Vy, l)
        scale = float(np.trace(K)) / d
        lam, cho, low, coef = solve_dual_from_kernel(K, Y, alpha, scale)
        pred = Y - lam * coef
        inv_reg = cho_solve((cho, low), np.eye(n, dtype=DTYPE), check_finite=False)
        h = 1.0 - lam * np.diag(inv_reg)
        denom = ensure_positive_denom(1.0 - h)
        pred_loo = (pred - h[:, None] * Y) / denom[:, None]
        return clip_eta(pred_loo)

    G, rhs, Phi = build_gram_rhs(Vx, Vy, Y, l)
    scale = float(np.trace(G)) / d
    _, cho, low, beta = solve_primal_from_gram(G, rhs, alpha, scale)

    pred = predict_primal(Vx, Vy, l, beta, Phi_ev=Phi)
    leverages = np.empty(n, dtype=np.float64)
    if Phi is not None:
        for start in range(0, n, ROW_CHUNK):
            end = min(start + ROW_CHUNK, n)
            Pb = Phi[start:end]
            Z = solve_triangular(cho, Pb.T, lower=low, trans=0, check_finite=False)
            leverages[start:end] = np.sum(Z * Z, axis=0)
    else:
        for start in range(0, n, ROW_CHUNK):
            end = min(start + ROW_CHUNK, n)
            Pb = features_from_vander(Vx[start:end], Vy[start:end], l)
            Z = solve_triangular(cho, Pb.T, lower=low, trans=0, check_finite=False)
            leverages[start:end] = np.sum(Z * Z, axis=0)

    denom = ensure_positive_denom(1.0 - leverages)
    pred_loo = (pred.astype(np.float64, copy=False) - leverages[:, None] * Y.astype(np.float64, copy=False)) / denom[:, None]
    return clip_eta(pred_loo)


def losses_from_linear_predictions(pred_all, y1, y2):
    n_alphas = pred_all.shape[1] // 2
    out = np.empty(n_alphas, dtype=np.float64)
    for k in range(n_alphas):
        pred = pred_all[:, 2 * k: 2 * k + 2]
        e1, e2 = clip_eta(pred)
        out[k] = fit_loss(e1, e2, y1, y2)
    return out


def losses_from_betas_on_subset(Vx, Vy, y1, y2, idx, l, betas, Phi_sub=None):
    B = np.concatenate(betas, axis=1)
    n_alphas = len(betas)
    if Phi_sub is not None:
        pred_all = Phi_sub @ B
        return losses_from_linear_predictions(pred_all, y1[idx], y2[idx])

    totals = np.zeros(n_alphas, dtype=np.float64)
    count = 0
    for start in range(0, len(idx), ROW_CHUNK):
        sl = idx[start:start + ROW_CHUNK]
        Pb = features_from_vander(Vx[sl], Vy[sl], l)
        pred_all = Pb @ B
        p = y1[sl] * y2[sl]
        for k in range(n_alphas):
            pred = pred_all[:, 2 * k: 2 * k + 2]
            e1, e2 = clip_eta(pred)
            totals[k] += float(np.sum((e1 - y1[sl]) ** 2) + np.sum((e2 - p) ** 2))
        count += len(sl)
    return totals / max(count, 1)


def select_on_hyper(Vx_tr, Vy_tr, y1_tr, y2_tr, Vx_hp, Vy_hp, y1_hp, y2_hp, degs):
    Y_tr = response_matrix(y1_tr, y2_tr)
    best_loss = np.inf
    best_d = degs[0]
    best_a = LAMBDA_GRID[0]
    best_prepared = None

    for l in degs:
        prepared = prepare_model(Vx_tr, Vy_tr, Y_tr, l)
        alpha_losses = np.empty(len(LAMBDA_GRID), dtype=np.float64)

        if prepared["mode"] == "dual":
            K_hp = dual_kernel_cross(Vx_hp, Vy_hp, Vx_tr, Vy_tr, l)
            coefs = [solve_dual_from_kernel(prepared["K"], prepared["Y"], alpha, prepared["scale"])[3]
                     for alpha in LAMBDA_GRID]
            pred_all = K_hp @ np.concatenate(coefs, axis=1)
            alpha_losses[:] = losses_from_linear_predictions(pred_all, y1_hp, y2_hp)
        else:
            betas = [solve_primal_from_gram(prepared["G"], prepared["rhs"], alpha, prepared["scale"])[3]
                     for alpha in LAMBDA_GRID]
            d = (l + 1) ** 2
            if can_materialize_phi(len(Vx_hp), d):
                Phi_hp = features_from_vander(Vx_hp, Vy_hp, l)
                pred_all = Phi_hp @ np.concatenate(betas, axis=1)
                alpha_losses[:] = losses_from_linear_predictions(pred_all, y1_hp, y2_hp)
            else:
                alpha_losses[:] = losses_from_betas_on_subset(Vx_hp, Vy_hp, y1_hp, y2_hp, np.arange(len(Vx_hp)), l, betas, Phi_sub=None)

        k_star = int(np.argmin(alpha_losses))
        if alpha_losses[k_star] < best_loss:
            best_loss = float(alpha_losses[k_star])
            best_d = l
            best_a = LAMBDA_GRID[k_star]
            best_prepared = prepared
    return best_d, best_a, best_prepared


def cv_pick_hypers(Vx, Vy, y1, y2, degs, seed):
    Y = response_matrix(y1, y2)
    n = len(Y)
    folds = list(kfold_indices(n, CV_K, seed))
    losses = np.zeros((len(degs), len(LAMBDA_GRID)), dtype=np.float64)

    for j, l in enumerate(degs):
        d = (l + 1) ** 2

        if n < d:
            K_full = dual_kernel_full(Vx, Vy, l)
            scale_full = float(np.trace(K_full)) / d
            for tr_idx, va_idx in folds:
                K = K_full[np.ix_(tr_idx, tr_idx)]
                K_va = K_full[np.ix_(va_idx, tr_idx)]
                Y_tr = Y[tr_idx]
                coefs = [
                    solve_dual_from_kernel(K, Y_tr, alpha, scale_full)[3]
                    for alpha in LAMBDA_GRID
                ]
                pred_all = K_va @ np.concatenate(coefs, axis=1)
                losses[j, :] += losses_from_linear_predictions(
                    pred_all, y1[va_idx], y2[va_idx]
                )
            continue

        G_full, rhs_full, Phi = build_gram_rhs(Vx, Vy, Y, l)
        scale_full = float(np.trace(G_full)) / d
        for tr_idx, va_idx in folds:
            if Phi is not None:
                Phi_va = Phi[va_idx]
                Y_va = Y[va_idx]
                G_va = Phi_va.T @ Phi_va
                rhs_va = Phi_va.T @ Y_va
            else:
                G_va, rhs_va, Phi_va, _ = build_subset_gram_rhs(Vx, Vy, Y, va_idx, l)

            G_tr = G_full - G_va
            rhs_tr = rhs_full - rhs_va
            betas = [
                solve_primal_from_gram(G_tr, rhs_tr, alpha, scale_full)[3]
                for alpha in LAMBDA_GRID
            ]
            losses[j, :] += losses_from_betas_on_subset(
                Vx, Vy, y1, y2, va_idx, l, betas, Phi_sub=Phi_va
            )

    j_star, k_star = np.unravel_index(int(np.argmin(losses)), losses.shape)
    return degs[j_star], LAMBDA_GRID[k_star]


def run_seed(seed, h, ce2_gt):
    D = load_dataset()
    m_tr, s_tr, y1_tr, y2_tr = get_split(D, seed, "train", h)
    m_hp, s_hp, y1_hp, y2_hp = get_split(D, seed, "hyper", h)
    m_va, s_va, y1_va, y2_va = get_split(D, seed, "valid", h)

    l_max = L_cap_of(h)
    Vx_tr, Vy_tr = cheb_vander(m_tr, s_tr, l_max)
    Vx_hp, Vy_hp = cheb_vander(m_hp, s_hp, l_max)
    Vx_va, Vy_va = cheb_vander(m_va, s_va, l_max)

    out = []
    for n in N_GRID:
        degs = degree_schedule(n, h)

        best_d, best_a, best_prepared = select_on_hyper(
            Vx_tr[:n], Vy_tr[:n], y1_tr[:n], y2_tr[:n],
            Vx_hp[:n], Vy_hp[:n], y1_hp[:n], y2_hp[:n],
            degs,
        )
        Phi_va_best = None
        K_va_best = None
        if best_prepared["mode"] == "dual":
            K_va_best = dual_kernel_cross(Vx_va[:n], Vy_va[:n], Vx_tr[:n], Vy_tr[:n], best_d)
        elif can_materialize_phi(n, (best_d + 1) ** 2):
            Phi_va_best = features_from_vander(Vx_va[:n], Vy_va[:n], best_d)
        e1_va, e2_va = fit_predict_prepared(best_prepared, best_a, Vx_va[:n], Vy_va[:n], Phi_ev=Phi_va_best, K_ev_tr=K_va_best)
        ce2_val = empirical_ce2(e1_va, e2_va, m_va[:n], s_va[:n])

        best_d_erm, best_a_erm = cv_pick_hypers(
            Vx_va[:n], Vy_va[:n], y1_va[:n], y2_va[:n],
            degs,
            seed=70_000 + 13 * seed + n,
        )
        e1_erm, e2_erm = fit_predict_loo(
            Vx_va[:n], Vy_va[:n], y1_va[:n], y2_va[:n],
            best_d_erm, best_a_erm)
        ce2_erm = empirical_ce2(e1_erm, e2_erm, m_va[:n], s_va[:n])

        out.append(dict(
            seed=seed,
            n=n,
            h=h,
            ce2_val=float(ce2_val),
            err_val=float(abs(ce2_val - ce2_gt)),
            ce2_erm=float(ce2_erm),
            err_erm=float(abs(ce2_erm - ce2_gt)),
            chosen_deg_val=int(best_d),
            chosen_alpha_val=float(best_a),
            chosen_deg_erm=int(best_d_erm),
            chosen_alpha_erm=float(best_a_erm),
            deg_schedule=degs,
        ))
    return out


def main():
    D = load_dataset()
    n_seeds = int(D["n_seeds"])
    for h in H_LIST:
        ce2_gt = float(D[f"ce2_gt_{h_tag(h)}"])
        print(f"--- polynomial, h={h:.5f} ({h_tag(h)}), L_cap={L_cap_of(h)}, GT CE2={ce2_gt:.6f} ---")
        per_seed = Parallel(n_jobs=min(N_JOBS, n_seeds), backend="loky", verbose=5)(
            delayed(run_seed)(seed, h, ce2_gt) for seed in range(n_seeds)
        )
        results = [r for seed_res in per_seed for r in seed_res]
        out = os.path.join(RESULTS_DIR, f"polynomial_{h_tag(h)}.npz")
        np.savez(out, results=np.array(results, dtype=object), ce2_gt=ce2_gt, h=h)
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
