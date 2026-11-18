"""Shared helpers for all estimators."""

import os
import numpy as np

DATA_PATH = "./data/dataset.npz"
RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Sample sizes from the paper's experiment 1.
N_GRID = [500, 1000, 2000, 5000, 10_000, 20_000, 50_000]

# Sech perturbation bandwidths to study.
H_LIST = [1.0 / 4.0, 1.0 / 16.0, 1.0 / 64.0]

# K for K-fold CV (used in the ERM variant of the polynomial method).
CV_K = 5


def h_tag(h):
    """A safe filename-style tag like 'h_1_16'."""
    return f"h_1_{int(round(1.0 / h))}"


def load_dataset():
    return np.load(DATA_PATH)


def get_split(D, seed, split_name, h, n=None):
    """Return (m, s, y1, y2) for the given (seed, split, h), optionally
    truncated to the first n samples."""
    pref = f"s{seed}_{h_tag(h)}_{split_name}_"
    out = (D[pref + "m"], D[pref + "s"], D[pref + "y1"], D[pref + "y2"])
    if n is not None:
        out = tuple(a[:n] for a in out)
    return out


def empirical_ce2(eta1_hat_at, eta2_hat_at, m_t, s_t):
    """CE2_hat = (1/n) sum |eta1_hat(S_i) - m_i| + |eta2_hat(S_i) - v_i|."""
    v_t = m_t ** 2 + s_t
    return (np.mean(np.abs(eta1_hat_at - m_t))
            + np.mean(np.abs(eta2_hat_at - v_t)))


def fit_loss(eta1_hat_at, eta2_hat_at, y1, y2):
    """Squared-error loss on the response variables (used for hyper tuning).
    eta1 is regressed on Y1; eta2 is regressed on the product Y1*Y2.
    """
    p = y1 * y2
    return (np.mean((eta1_hat_at - y1) ** 2)
            + np.mean((eta2_hat_at - p) ** 2))


def kfold_indices(n, k, seed):
    """Yield (train_idx, val_idx) for K-fold splits of range(n)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    fold_size = n // k
    for fold in range(k):
        start = fold * fold_size
        end = (fold + 1) * fold_size if fold < k - 1 else n
        val_idx = perm[start:end]
        train_idx = np.concatenate([perm[:start], perm[end:]])
        yield train_idx, val_idx
