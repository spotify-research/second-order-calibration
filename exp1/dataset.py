"""
Dataset generation for Experiment 1.

Same X's, same ensemble, same raw scores, same labels - perturbed THREE
 times with h in H_LIST.

Ground truth CE2 is now computed from a large QMC approximation of the RAW
score distribution together with exact application of the truncated-sech
perturbation kernel on a fine Cartesian grid, which avoids the self-fit
bias of a histogram-based estimate.

Storage layout in dataset.npz:
  s{seed}_h_1_{1/h}_{split}_{m|s|y1|y2}    perturbed scores + 2-snapshots
  ce2_gt_h_1_{1/h}                          ground-truth CE2 (perturbed)
  ce2_gt_half_diff_h_1_{1/h}                half-span across independent
                                            scrambled-Sobol GT replicates
  ce2_gt_replicates_h_1_{1/h}               per-replicate CE2 estimates
  n_seeds, n_max, h_list                    metadata
"""

import os
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import qmc, norm

from common import H_LIST, h_tag


# -------------------------------------------------------------------------
# Small env helpers (useful for smoke tests without changing defaults).
# -------------------------------------------------------------------------

def _env_int(name, default):
    val = os.environ.get(name)
    return int(val) if val is not None else int(default)


def _env_float(name, default):
    val = os.environ.get(name)
    return float(val) if val is not None else float(default)


# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------

DIM = _env_int("EXP1_DIM", 4)
N_MIXTURE = _env_int("EXP1_N_MIXTURE", 10)
N_MAX = _env_int("EXP1_N_MAX", 50_000)
N_SEEDS = _env_int("EXP1_N_SEEDS", 20)

ENSEMBLE_SIZE = _env_int("EXP1_ENSEMBLE_SIZE", 5)
HIDDEN = _env_int("EXP1_HIDDEN", 64)
ENSEMBLE_TRAIN_N = _env_int("EXP1_ENSEMBLE_TRAIN_N", 5_000)
ENSEMBLE_EPOCHS = _env_int("EXP1_ENSEMBLE_EPOCHS", 30)
ENSEMBLE_BATCH = _env_int("EXP1_ENSEMBLE_BATCH", 256)
INFER_BATCH = _env_int("EXP1_INFER_BATCH", 16_384)

GT_QMC_N = _env_int("EXP1_GT_QMC_N", 1 << 18)
GT_QMC_REPEATS = _env_int("EXP1_GT_QMC_REPEATS", 2)
GT_GRID_M = _env_int("EXP1_GT_GRID_M", 1025)
GT_GRID_S = _env_int("EXP1_GT_GRID_S", 257)

DATA_DIR = "./data"
os.makedirs(DATA_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------------------------------------------------------
# Ground-truth DGP
# -------------------------------------------------------------------------

def make_dgp(rng):
    centres = rng.normal(0.0, 1.5, size=(N_MIXTURE, DIM))
    w = rng.normal(0.0, 1.0, size=DIM) / np.sqrt(DIM)
    return centres, w, 0.0


def sample_X(n, centres, rng):
    comp = rng.integers(0, N_MIXTURE, size=n)
    noise = rng.normal(0.0, 1.0, size=(n, DIM))
    return centres[comp] + noise


def sample_X_qmc(n, centres, seed):
    """Sobol approximation of the mixture distribution of X."""
    sobol = qmc.Sobol(d=DIM + 1, scramble=True, seed=seed)
    if n > 0 and (n & (n - 1)) == 0:
        u = sobol.random_base2(int(np.log2(n)))
    else:
        u = sobol.random(n)
    comp = np.minimum((u[:, 0] * N_MIXTURE).astype(np.int64), N_MIXTURE - 1)
    z = norm.ppf(np.clip(u[:, 1:], 1e-12, 1.0 - 1e-12))
    return centres[comp] + z


def f_star(x, w, bias):
    z = x @ w + bias
    return 1.0 / (1.0 + np.exp(-z))


# -------------------------------------------------------------------------
# Ensemble
# -------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, dim=DIM, hidden=HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)



def train_one_member(seed, X, Y):
    torch.manual_seed(seed)
    model = MLP().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    Xt = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    Yt = torch.tensor(Y, dtype=torch.float32, device=DEVICE)
    bs, n = ENSEMBLE_BATCH, len(X)
    for _ in range(ENSEMBLE_EPOCHS):
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            loss = loss_fn(model(Xt[idx]), Yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model



def train_ensemble(centres, w, bias, rng):
    X = sample_X(ENSEMBLE_TRAIN_N, centres, rng)
    p = f_star(X, w, bias)
    Y = (rng.uniform(size=ENSEMBLE_TRAIN_N) < p).astype(np.float32)
    return [train_one_member(1000 + k, X, Y) for k in range(ENSEMBLE_SIZE)]


@torch.no_grad()
def ensemble_scores(members, X, batch_size=INFER_BATCH):
    n = len(X)
    m_out = np.empty(n, dtype=np.float64)
    s2_out = np.empty(n, dtype=np.float64)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        Xt = torch.tensor(X[start:end], dtype=torch.float32, device=DEVICE)
        probs = torch.stack([torch.sigmoid(m(Xt)) for m in members], dim=0)
        m = probs.mean(0).cpu().numpy().astype(np.float64, copy=False)
        s2 = probs.var(0, unbiased=False).cpu().numpy().astype(np.float64, copy=False)
        m = np.clip(m, 1e-6, 1.0 - 1e-6)
        s2 = np.clip(s2, 0.0, m * (1.0 - m))
        m_out[start:end] = m
        s2_out[start:end] = s2
    return m_out, s2_out


# -------------------------------------------------------------------------
# Sech perturbation on an interval (closed-form inverse CDF).
# k_h(t|s) ∝ sech((t-s)/h),  G(t) = 2h * arctan(tanh((t-s)/(2h)))
# -------------------------------------------------------------------------

def _G(t, s, h):
    return 2.0 * h * np.arctan(np.tanh((t - s) / (2.0 * h)))



def _G_inv(g, s, h):
    return s + 2.0 * h * np.arctanh(np.tan(g / (2.0 * h)))



def sech_perturb(scores, low, high, h, rng):
    Ga, Gb = _G(low, scores, h), _G(high, scores, h)
    u = rng.uniform(size=scores.shape)
    g = Ga + u * (Gb - Ga)
    return _G_inv(g, scores, h)



def sech_kernel_matrix(query_points, center_points, low, high, h):
    """Dense matrix K[q, c] = k_h(query_q | center_c)."""
    Z = _G(high, center_points, h) - _G(low, center_points, h)
    return (1.0 / np.cosh((query_points[:, None] - center_points[None, :]) / h)
            / Z[None, :])


# -------------------------------------------------------------------------
# Ground-truth CE2 of the perturbed predictor.
#
# Let R=(m_raw,s_raw) and Q=(m_tilde,s_tilde).  Conditional on X,
# Q has density k_m(q_m | m_raw(X)) k_s(q_s | s_raw(X)).  Therefore,
#
#   eta1(q) = E[p_true(X)     k(q | R(X))] / E[k(q | R(X))]
#   eta2(q) = E[p_true(X)^2   k(q | R(X))] / E[k(q | R(X))]
#
# where expectations are only over X.  We estimate the raw score law of R by a
# large scrambled-Sobol sample and then apply the perturbation kernel exactly on
# a fine tensor grid.  This avoids both the old histogram bias and the extra
# Monte Carlo variance from explicitly resampling perturbations inside the GT.
# -------------------------------------------------------------------------


def trapezoid_weights(nodes):
    w = np.empty_like(nodes, dtype=np.float64)
    gaps = np.diff(nodes)
    w[0] = 0.5 * gaps[0]
    w[-1] = 0.5 * gaps[-1]
    w[1:-1] = 0.5 * (gaps[:-1] + gaps[1:])
    return w



def _bilinear_indices(values, low, high, n_nodes):
    step = (high - low) / (n_nodes - 1)
    pos = np.clip((values - low) / step, 0.0, n_nodes - 1.0)
    idx0 = np.floor(pos).astype(np.int64)
    idx0 = np.minimum(idx0, n_nodes - 2)
    frac = pos - idx0
    return idx0, frac



def deposit_bilinear(m, s, p_true, m_nodes, s_nodes):
    """Deposit raw-score mass and p, p^2 moments onto a tensor grid."""
    M = len(m_nodes)
    S = len(s_nodes)
    i0, tx = _bilinear_indices(m, 0.0, 1.0, M)
    j0, ty = _bilinear_indices(s, 0.0, 0.25, S)

    w00 = (1.0 - tx) * (1.0 - ty)
    w10 = tx * (1.0 - ty)
    w01 = (1.0 - tx) * ty
    w11 = tx * ty

    flat00 = i0 * S + j0
    flat10 = (i0 + 1) * S + j0
    flat01 = i0 * S + (j0 + 1)
    flat11 = (i0 + 1) * S + (j0 + 1)

    base = np.full(len(m), 1.0 / len(m), dtype=np.float64)
    p1 = p_true.astype(np.float64, copy=False) / len(m)
    p2 = (p_true.astype(np.float64, copy=False) ** 2) / len(m)

    W0 = np.zeros(M * S, dtype=np.float64)
    W1 = np.zeros(M * S, dtype=np.float64)
    W2 = np.zeros(M * S, dtype=np.float64)

    for idx, ww in ((flat00, w00), (flat10, w10), (flat01, w01), (flat11, w11)):
        np.add.at(W0, idx, base * ww)
        np.add.at(W1, idx, p1 * ww)
        np.add.at(W2, idx, p2 * ww)

    return W0.reshape(M, S), W1.reshape(M, S), W2.reshape(M, S)



def ce2_from_grid_masses(W0, W1, W2, h, m_nodes, s_nodes, m_quad, s_quad):
    Km = sech_kernel_matrix(m_nodes, m_nodes, 0.0, 1.0, h)
    Ks = sech_kernel_matrix(s_nodes, s_nodes, 0.0, 0.25, h)

    D = Km @ W0 @ Ks.T
    N1 = Km @ W1 @ Ks.T
    N2 = Km @ W2 @ Ks.T

    D = np.maximum(D, np.finfo(np.float64).tiny)
    eta1 = N1 / D
    eta2 = N2 / D

    m_grid = m_nodes[:, None]
    s_grid = s_nodes[None, :]
    v_grid = m_grid ** 2 + s_grid
    loss = np.abs(eta1 - m_grid) + np.abs(eta2 - v_grid)

    mass = D * m_quad[:, None] * s_quad[None, :]
    mass_sum = float(mass.sum())
    if not np.isfinite(mass_sum) or mass_sum <= 0.0:
        raise RuntimeError("invalid perturbed-score mass in GT computation")
    return float(np.sum(loss * mass) / mass_sum)



def compute_ground_truth_ce2_all_h(members, centres, w, bias):
    m_nodes = np.linspace(0.0, 1.0, GT_GRID_M, dtype=np.float64)
    s_nodes = np.linspace(0.0, 0.25, GT_GRID_S, dtype=np.float64)
    m_quad = trapezoid_weights(m_nodes)
    s_quad = trapezoid_weights(s_nodes)

    replicate_masses = []
    for rep in range(GT_QMC_REPEATS):
        print(f"    GT replicate {rep + 1}/{GT_QMC_REPEATS} with {GT_QMC_N} Sobol points")
        X = sample_X_qmc(GT_QMC_N, centres, seed=42 + 10_000 * rep)
        p_true = f_star(X, w, bias)
        m_raw, s_raw = ensemble_scores(members, X)
        replicate_masses.append(deposit_bilinear(m_raw, s_raw, p_true, m_nodes, s_nodes))

    W0 = sum(r[0] for r in replicate_masses) / len(replicate_masses)
    W1 = sum(r[1] for r in replicate_masses) / len(replicate_masses)
    W2 = sum(r[2] for r in replicate_masses) / len(replicate_masses)

    out = {}
    for h in H_LIST:
        reps = [
            ce2_from_grid_masses(r0, r1, r2, h, m_nodes, s_nodes, m_quad, s_quad)
            for (r0, r1, r2) in replicate_masses
        ]
        ce2 = ce2_from_grid_masses(W0, W1, W2, h, m_nodes, s_nodes, m_quad, s_quad)
        half_diff = 0.5 * (max(reps) - min(reps)) if len(reps) > 1 else 0.0
        out[h] = (ce2, half_diff, np.array(reps, dtype=np.float64))
    return out


# -------------------------------------------------------------------------
# Per-(seed, split, h) generation: same X's and raw scores per (seed, split),
# then perturb three times.
# -------------------------------------------------------------------------

def make_split(members, centres, w, bias, n, rng):
    """Sample X's and labels (h-independent), and raw ensemble scores."""
    X = sample_X(n, centres, rng)
    p_true = f_star(X, w, bias)
    m_raw, s2_raw = ensemble_scores(members, X)
    Y1 = (rng.uniform(size=n) < p_true).astype(np.float32)
    Y2 = (rng.uniform(size=n) < p_true).astype(np.float32)
    return m_raw, s2_raw, Y1, Y2



def perturb_split(m_raw, s2_raw, Y1, Y2, h, rng):
    m_t = sech_perturb(m_raw, 0.0, 1.0, h, rng)
    s_t = sech_perturb(s2_raw, 0.0, 0.25, h, rng)
    return dict(m=m_t.astype(np.float32),
                s=s_t.astype(np.float32),
                y1=Y1,
                y2=Y2)


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main():
    print("=== Building DGP and ensemble ===")
    rng_dgp = np.random.default_rng(0)
    centres, w, bias = make_dgp(rng_dgp)
    members = train_ensemble(centres, w, bias, rng_dgp)

    save = dict(
        n_seeds=N_SEEDS,
        n_max=N_MAX,
        h_list=np.array(H_LIST, dtype=np.float64),
        gt_qmc_n=GT_QMC_N,
        gt_qmc_repeats=GT_QMC_REPEATS,
        gt_grid_m=GT_GRID_M,
        gt_grid_s=GT_GRID_S,
    )

    print("=== Computing ground-truth CE2 per h ===")
    gt_all = compute_ground_truth_ce2_all_h(members, centres, w, bias)
    for h in H_LIST:
        ce2_gt, gt_half_diff, gt_reps = gt_all[h]
        save[f"ce2_gt_{h_tag(h)}"] = ce2_gt
        save[f"ce2_gt_half_diff_{h_tag(h)}"] = gt_half_diff
        save[f"ce2_gt_replicates_{h_tag(h)}"] = gt_reps
        print(
            f"  h = {h:.5f} ({h_tag(h)}): CE2 = {ce2_gt:.6f}, "
            f"replicate half-span = {gt_half_diff:.6f}, reps = {gt_reps}"
        )

    print("=== Generating per-seed splits and perturbations ===")
    for seed in range(N_SEEDS):
        print(f"  seed {seed} ...")
        rng = np.random.default_rng(10_000 + seed)
        raw_splits = {}
        for split_name in ("train", "hyper", "valid"):
            raw_splits[split_name] = make_split(
                members, centres, w, bias, N_MAX, rng)
        for h in H_LIST:
            rng_h = np.random.default_rng(
                100_000 + 10_000 * seed + int(round(1.0 / h)))
            for split_name, raw in raw_splits.items():
                d = perturb_split(*raw, h, rng_h)
                pref = f"s{seed}_{h_tag(h)}_{split_name}_"
                for k, v in d.items():
                    save[pref + k] = v

    out_path = os.path.join(DATA_DIR, "dataset.npz")
    np.savez_compressed(out_path, **save)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
