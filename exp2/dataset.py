"""
Dataset generation for Experiment 1 / 2.

Same X's, same ensemble, same raw scores, same labels - perturbed THREE
 times with h in H_LIST.

Uses the Experiment 1 data-generating scheme with the Experiment 2 defaults:
  * same-data ensemble members by default (no bagging)
  * exactly two Bernoulli snapshots by default for calibration targets
  * perturb both coordinates with the same bandwidth by default (h_s = h)

Useful defaults in this file:
  * EXP1_ENSEMBLE_MEMBER_DATA_MODE=same
  * EXP1_ENSEMBLE_SIZE=20
  * EXP1_ENSEMBLE_TRAIN_N=24
  * EXP1_ENSEMBLE_EPOCHS=60
  * EXP1_S_H_SCALE=1.0
  * EXP2_CAL_SNAPSHOTS=2

Two practical controls are also available that are useful for diagnosing whether
there is any epistemic signal left to calibrate:
  * EXP1_ENSEMBLE_MEMBER_DATA_MODE in {same, bootstrap, independent}
      - same:        all members train on the same dataset (original behaviour)
      - bootstrap:   each member trains on a bootstrap resample of one shared pool
      - independent: each member trains on its own iid dataset
  * EXP1_S_H_SCALE >= 0
      - the perturbation bandwidth applied to the second coordinate is
        h_s = EXP1_S_H_SCALE * h.  The original paper setting is 1.0.
        Setting it smaller (or even 0.0) prevents the variance coordinate from
        being dominated by perturbation noise when raw ensemble disagreement is tiny.

Storage layout in dataset.npz additionally includes:
  s{seed}_raw_{split}_{m|s|y1|y2}          unperturbed raw scores + 2-snapshots
  s{seed}_{h}_{split}_{r1|r2}              low-variance aggregated targets for Exp. 2
  ce2_gt_raw                               ground-truth CE2 on the raw score grid
  gt_raw_mass, gt_raw_eta1, gt_raw_eta2    raw-score GT surface (diagnostic)
  s_h_scale                                metadata for the second-coordinate bandwidth
  ensemble_member_data_mode                metadata for ensemble training mode
  exp2_cal_snapshots                        number of Bernoulli snapshots saved for optional pooled targets

The saved arrays still include both the original two-snapshot responses (y1, y2)
and optional pooled moment targets (r1, r2).  Experiment 2 can choose between
them via EXP2_TARGET_MODE in common.py / experiment2.py.
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



def _env_bool(name, default):
    val = os.environ.get(name)
    if val is None:
        return bool(default)
    return str(val).strip().lower() not in {"0", "false", "no", "off"}


# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------

DIM = _env_int("EXP1_DIM", 4)
N_MIXTURE = _env_int("EXP1_N_MIXTURE", 10)
N_MAX = _env_int("EXP1_N_MAX", 50_000)
N_SEEDS = _env_int("EXP1_N_SEEDS", 20)

ENSEMBLE_SIZE = _env_int("EXP1_ENSEMBLE_SIZE", 20)
HIDDEN = _env_int("EXP1_HIDDEN", 64)
ENSEMBLE_TRAIN_N = _env_int("EXP1_ENSEMBLE_TRAIN_N", 24)
ENSEMBLE_EPOCHS = _env_int("EXP1_ENSEMBLE_EPOCHS", 60)
ENSEMBLE_BATCH = _env_int("EXP1_ENSEMBLE_BATCH", 256)
INFER_BATCH = _env_int("EXP1_INFER_BATCH", 16_384)
ENSEMBLE_MEMBER_DATA_MODE = os.environ.get(
    "EXP1_ENSEMBLE_MEMBER_DATA_MODE", "same"
).strip().lower()
if ENSEMBLE_MEMBER_DATA_MODE not in {"same", "bootstrap", "independent"}:
    raise ValueError(
        "EXP1_ENSEMBLE_MEMBER_DATA_MODE must be one of: same, bootstrap, independent"
    )

GT_QMC_N = _env_int("EXP1_GT_QMC_N", 1 << 18)
GT_QMC_REPEATS = _env_int("EXP1_GT_QMC_REPEATS", 2)
GT_GRID_M = _env_int("EXP1_GT_GRID_M", 1025)
GT_GRID_S = _env_int("EXP1_GT_GRID_S", 257)

S_H_SCALE = _env_float("EXP1_S_H_SCALE", 1.0)
SAVE_RAW_SPLITS = _env_bool("EXP1_SAVE_RAW_SPLITS", True)
EXP2_CAL_SNAPSHOTS = _env_int("EXP2_CAL_SNAPSHOTS", 2)
if EXP2_CAL_SNAPSHOTS < 2:
    raise ValueError("EXP2_CAL_SNAPSHOTS must be at least 2")

DATA_DIR = "./data"
os.makedirs(DATA_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(int(os.environ.get("EXP1_TORCH_NUM_THREADS", "1")))
ENSEMBLE_LR = _env_float("EXP1_ENSEMBLE_LR", 1e-3)


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
    opt = torch.optim.Adam(model.parameters(), lr=ENSEMBLE_LR)
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



def _member_training_sample(centres, w, bias, pool, member_rng):
    if ENSEMBLE_MEMBER_DATA_MODE == "same":
        return pool
    if ENSEMBLE_MEMBER_DATA_MODE == "bootstrap":
        X_base, Y_base = pool
        idx = member_rng.integers(0, len(X_base), size=len(X_base))
        return X_base[idx], Y_base[idx]
    X = sample_X(ENSEMBLE_TRAIN_N, centres, member_rng)
    p = f_star(X, w, bias)
    Y = (member_rng.uniform(size=ENSEMBLE_TRAIN_N) < p).astype(np.float32)
    return X, Y



def train_ensemble(centres, w, bias, rng):
    X = sample_X(ENSEMBLE_TRAIN_N, centres, rng)
    p = f_star(X, w, bias)
    Y = (rng.uniform(size=ENSEMBLE_TRAIN_N) < p).astype(np.float32)
    pool = (X, Y)

    base_seed = int(rng.integers(1_000_000_000))
    members = []
    for k in range(ENSEMBLE_SIZE):
        member_rng = np.random.default_rng(base_seed + 10_000 * k + 17)
        Xk, Yk = _member_training_sample(centres, w, bias, pool, member_rng)
        members.append(train_one_member(1000 + k, Xk, Yk))
    return members


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
    if h <= 0.0:
        return np.clip(np.asarray(scores, dtype=np.float64), low, high).copy()
    Ga, Gb = _G(low, scores, h), _G(high, scores, h)
    u = rng.uniform(size=scores.shape)
    g = Ga + u * (Gb - Ga)
    return _G_inv(g, scores, h)



def sech_kernel_matrix(query_points, center_points, low, high, h):
    """Dense matrix K[q, c] = k_h(query_q | center_c)."""
    if h <= 0.0:
        if (len(query_points) == len(center_points)
                and np.allclose(query_points, center_points)):
            return np.eye(len(query_points), dtype=np.float64)
        raise ValueError("h <= 0 is only supported for identity-grid convolutions")
    Z = _G(high, center_points, h) - _G(low, center_points, h)
    return (1.0 / np.cosh((query_points[:, None] - center_points[None, :]) / h)
            / Z[None, :])


# -------------------------------------------------------------------------
# Ground-truth CE2 of the perturbed predictor.
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



def deposit_weighted_bilinear(m, s, w0, w1, w2, m_nodes, s_nodes):
    """Deposit weighted mass and two weighted moments onto a tensor grid."""
    M = len(m_nodes)
    S = len(s_nodes)
    i0, tx = _bilinear_indices(m, float(m_nodes[0]), float(m_nodes[-1]), M)
    j0, ty = _bilinear_indices(s, float(s_nodes[0]), float(s_nodes[-1]), S)

    w00 = (1.0 - tx) * (1.0 - ty)
    w10 = tx * (1.0 - ty)
    w01 = (1.0 - tx) * ty
    w11 = tx * ty

    flat00 = i0 * S + j0
    flat10 = (i0 + 1) * S + j0
    flat01 = i0 * S + (j0 + 1)
    flat11 = (i0 + 1) * S + (j0 + 1)

    W0 = np.zeros(M * S, dtype=np.float64)
    W1 = np.zeros(M * S, dtype=np.float64)
    W2 = np.zeros(M * S, dtype=np.float64)

    for idx, ww in ((flat00, w00), (flat10, w10), (flat01, w01), (flat11, w11)):
        np.add.at(W0, idx, w0 * ww)
        np.add.at(W1, idx, w1 * ww)
        np.add.at(W2, idx, w2 * ww)

    return W0.reshape(M, S), W1.reshape(M, S), W2.reshape(M, S)



def deposit_bilinear(m, s, p_true, m_nodes, s_nodes):
    """Deposit raw-score mass and p, p^2 moments onto a tensor grid."""
    base = np.full(len(m), 1.0 / len(m), dtype=np.float64)
    p1 = p_true.astype(np.float64, copy=False) / len(m)
    p2 = (p_true.astype(np.float64, copy=False) ** 2) / len(m)
    return deposit_weighted_bilinear(m, s, base, p1, p2, m_nodes, s_nodes)



def ce2_from_surface(mass, eta1, eta2, m_nodes, s_nodes):
    m_grid = m_nodes[:, None]
    s_grid = s_nodes[None, :]
    v_grid = m_grid ** 2 + s_grid
    loss = np.abs(eta1 - m_grid) + np.abs(eta2 - v_grid)
    mass_sum = float(mass.sum())
    if not np.isfinite(mass_sum) or mass_sum <= 0.0:
        raise RuntimeError("invalid score-space mass in GT computation")
    return float(np.sum(loss * mass) / mass_sum)



def raw_surface_from_grid_masses(W0, W1, W2, m_nodes, s_nodes):
    denom = np.maximum(W0, np.finfo(np.float64).tiny)
    eta1 = W1 / denom
    eta2 = W2 / denom
    mass = np.maximum(W0, 0.0)
    mass_sum = float(mass.sum())
    if not np.isfinite(mass_sum) or mass_sum <= 0.0:
        raise RuntimeError("invalid raw-score mass in GT computation")
    mass = mass / mass_sum
    ce2 = ce2_from_surface(mass, eta1, eta2, m_nodes, s_nodes)
    return dict(mass=mass, eta1=eta1, eta2=eta2, ce2=ce2)



def perturbed_surface_from_grid_masses(W0, W1, W2, h_m, m_nodes, s_nodes, m_quad, s_quad,
                                       h_s=None):
    if h_s is None:
        h_s = h_m

    if h_m > 0.0:
        Km = sech_kernel_matrix(m_nodes, m_nodes, 0.0, 1.0, h_m)
        D = Km @ W0
        N1 = Km @ W1
        N2 = Km @ W2
    else:
        D = W0.copy()
        N1 = W1.copy()
        N2 = W2.copy()

    if h_s > 0.0:
        Ks = sech_kernel_matrix(s_nodes, s_nodes, 0.0, 0.25, h_s)
        D = D @ Ks.T
        N1 = N1 @ Ks.T
        N2 = N2 @ Ks.T

    D = np.maximum(D, np.finfo(np.float64).tiny)
    eta1 = N1 / D
    eta2 = N2 / D

    mass = D * m_quad[:, None] * s_quad[None, :]
    mass_sum = float(mass.sum())
    if not np.isfinite(mass_sum) or mass_sum <= 0.0:
        raise RuntimeError("invalid perturbed-score mass in GT computation")
    mass = mass / mass_sum
    ce2 = ce2_from_surface(mass, eta1, eta2, m_nodes, s_nodes)
    return dict(mass=mass, eta1=eta1, eta2=eta2, ce2=ce2)



def ce2_from_grid_masses(W0, W1, W2, h, m_nodes, s_nodes, m_quad, s_quad, h_s=None):
    return perturbed_surface_from_grid_masses(
        W0, W1, W2, h, m_nodes, s_nodes, m_quad, s_quad, h_s=h_s
    )["ce2"]



def interpolate_grid_bilinear(grid, m, s, m_nodes, s_nodes):
    """Bilinearly interpolate a score-space grid at points (m, s)."""
    M = len(m_nodes)
    S = len(s_nodes)
    i0, tx = _bilinear_indices(m, float(m_nodes[0]), float(m_nodes[-1]), M)
    j0, ty = _bilinear_indices(s, float(s_nodes[0]), float(s_nodes[-1]), S)
    g00 = grid[i0, j0]
    g10 = grid[i0 + 1, j0]
    g01 = grid[i0, j0 + 1]
    g11 = grid[i0 + 1, j0 + 1]
    return ((1.0 - tx) * (1.0 - ty) * g00
            + tx * (1.0 - ty) * g10
            + (1.0 - tx) * ty * g01
            + tx * ty * g11)



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

    raw_surface = raw_surface_from_grid_masses(W0, W1, W2, m_nodes, s_nodes)

    out = {}
    for h in H_LIST:
        h_s = S_H_SCALE * h
        rep_surfaces = [
            perturbed_surface_from_grid_masses(
                r0, r1, r2, h, m_nodes, s_nodes, m_quad, s_quad, h_s=h_s
            )
            for (r0, r1, r2) in replicate_masses
        ]
        surface = perturbed_surface_from_grid_masses(
            W0, W1, W2, h, m_nodes, s_nodes, m_quad, s_quad, h_s=h_s
        )
        reps = [r["ce2"] for r in rep_surfaces]
        half_diff = 0.5 * (max(reps) - min(reps)) if len(reps) > 1 else 0.0
        out[h] = dict(
            ce2=surface["ce2"],
            half_diff=half_diff,
            reps=np.array(reps, dtype=np.float64),
            mass=surface["mass"],
            eta1=surface["eta1"],
            eta2=surface["eta2"],
            h_s=h_s,
        )
    return m_nodes, s_nodes, raw_surface, out


# -------------------------------------------------------------------------
# Per-(seed, split, h) generation: same X's and raw scores per (seed, split),
# then perturb three times.
# -------------------------------------------------------------------------

def make_split(members, centres, w, bias, n, rng):
    """Sample X's and labels (h-independent), and raw ensemble scores.

    The original paper-faithful targets are the first snapshot Y1 and the
    product Y1 * Y2 of two independent snapshots.  For optional variance
    reduction experiments, we also save pooled moment targets
      r1 = average of K Bernoulli snapshots, unbiased for p
      r2 = average of all unordered pair products, unbiased for p^2
    where K = EXP2_CAL_SNAPSHOTS.  If K = 2, then r2 = Y1 * Y2 and
    r1 = (Y1 + Y2) / 2.
    """
    X = sample_X(n, centres, rng)
    p_true = f_star(X, w, bias)
    m_raw, s2_raw = ensemble_scores(members, X)

    K = EXP2_CAL_SNAPSHOTS
    Y = (rng.uniform(size=(n, K)) < p_true[:, None]).astype(np.float32)
    count = Y.sum(axis=1, dtype=np.float32)
    Y1 = Y[:, 0].astype(np.float32, copy=False)
    Y2 = Y[:, 1].astype(np.float32, copy=False)
    r1 = (count / float(K)).astype(np.float32, copy=False)
    r2 = (count * np.maximum(count - 1.0, 0.0) / float(K * (K - 1))).astype(np.float32, copy=False)
    return m_raw, s2_raw, Y1, Y2, r1, r2



def perturb_split(m_raw, s2_raw, Y1, Y2, r1, r2, h, rng):
    h_s = S_H_SCALE * h
    m_t = sech_perturb(m_raw, 0.0, 1.0, h, rng)
    s_t = sech_perturb(s2_raw, 0.0, 0.25, h_s, rng)
    return dict(m=m_t.astype(np.float32),
                s=s_t.astype(np.float32),
                y1=Y1,
                y2=Y2,
                r1=r1,
                r2=r2)


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main():
    print("=== Building DGP and ensemble ===")
    print(
        f"ensemble_train_n={ENSEMBLE_TRAIN_N}, epochs={ENSEMBLE_EPOCHS}, "
        f"member_data_mode={ENSEMBLE_MEMBER_DATA_MODE}, ensemble_size={ENSEMBLE_SIZE}, "
        f"s_h_scale={S_H_SCALE}, exp2_cal_snapshots={EXP2_CAL_SNAPSHOTS}"
    )
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
        s_h_scale=np.array(S_H_SCALE, dtype=np.float64),
        ensemble_member_data_mode=np.array(ENSEMBLE_MEMBER_DATA_MODE),
        ensemble_size=np.array(ENSEMBLE_SIZE, dtype=np.int64),
        ensemble_hidden=np.array(HIDDEN, dtype=np.int64),
        ensemble_batch=np.array(ENSEMBLE_BATCH, dtype=np.int64),
        ensemble_lr=np.array(ENSEMBLE_LR, dtype=np.float64),
        ensemble_train_n=np.array(ENSEMBLE_TRAIN_N, dtype=np.int64),
        ensemble_epochs=np.array(ENSEMBLE_EPOCHS, dtype=np.int64),
        exp2_cal_snapshots=np.array(EXP2_CAL_SNAPSHOTS, dtype=np.int64),
    )

    print("=== Computing ground-truth CE2 per h ===")
    m_nodes, s_nodes, raw_surface, gt_all = compute_ground_truth_ce2_all_h(
        members, centres, w, bias
    )
    save["gt_m_nodes"] = m_nodes
    save["gt_s_nodes"] = s_nodes
    save["ce2_gt_raw"] = raw_surface["ce2"]
    save["gt_raw_mass"] = raw_surface["mass"]
    save["gt_raw_eta1"] = raw_surface["eta1"]
    save["gt_raw_eta2"] = raw_surface["eta2"]
    print(f"  raw-score CE2: {raw_surface['ce2']:.6f}")

    for h in H_LIST:
        gt = gt_all[h]
        ce2_gt = gt["ce2"]
        gt_half_diff = gt["half_diff"]
        gt_reps = gt["reps"]
        tag = h_tag(h)
        save[f"ce2_gt_{tag}"] = ce2_gt
        save[f"ce2_gt_half_diff_{tag}"] = gt_half_diff
        save[f"ce2_gt_replicates_{tag}"] = gt_reps
        save[f"gt_mass_{tag}"] = gt["mass"]
        save[f"gt_eta1_{tag}"] = gt["eta1"]
        save[f"gt_eta2_{tag}"] = gt["eta2"]
        print(
            f"  h = {h:.5f} ({tag}): CE2 = {ce2_gt:.6f}, h_s = {gt['h_s']:.6f}, "
            f"replicate half-span = {gt_half_diff:.6f}, reps = {gt_reps}"
        )

    print("=== Generating per-seed splits and perturbations ===")
    for seed in range(N_SEEDS):
        print(f"  seed {seed} ...")
        rng = np.random.default_rng(10_000 + seed)
        raw_splits = {}
        for split_name in ("train", "hyper", "valid"):
            raw_splits[split_name] = make_split(
                members, centres, w, bias, N_MAX, rng
            )
            if SAVE_RAW_SPLITS:
                m_raw, s_raw, y1_raw, y2_raw, r1_raw, r2_raw = raw_splits[split_name]
                pref_raw = f"s{seed}_raw_{split_name}_"
                save[pref_raw + "m"] = m_raw.astype(np.float32)
                save[pref_raw + "s"] = s_raw.astype(np.float32)
                save[pref_raw + "y1"] = y1_raw
                save[pref_raw + "y2"] = y2_raw
                save[pref_raw + "r1"] = r1_raw
                save[pref_raw + "r2"] = r2_raw
        for h in H_LIST:
            rng_h = np.random.default_rng(
                100_000 + 10_000 * seed + int(round(1.0 / h))
            )
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
