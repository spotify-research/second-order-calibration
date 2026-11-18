"""Shared helpers for all estimators."""

import os
import numpy as np

DATA_PATH = "./data/dataset.npz"
RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Sample sizes from the paper's experiment 1.
N_GRID = [500, 1000, 2000, 5000, 10_000, 20_000, 50_000]

# Sech perturbation bandwidths to study.
# Optional override: EXP_H_LIST="1/4,1/16" or EXP_H_LIST="0.25,0.0625".
def _parse_h_list():
    raw = os.environ.get("EXP_H_LIST")
    if raw is None or not raw.strip():
        return [1.0 / 4.0, 1.0 / 16.0, 1.0 / 64.0]
    out = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "/" in tok:
            a, b = tok.split("/", 1)
            out.append(float(a) / float(b))
        else:
            out.append(float(tok))
    if not out:
        raise ValueError("EXP_H_LIST did not contain any valid bandwidths")
    return out


H_LIST = _parse_h_list()

# K for K-fold CV (used in the ERM variant of the polynomial method).
CV_K = 5


TARGET_MODE = os.environ.get("EXP2_TARGET_MODE", "paper").strip().lower()
if TARGET_MODE not in {"paper", "pooled"}:
    raise ValueError("EXP2_TARGET_MODE must be one of: paper, pooled")



def h_tag(h):
    """A safe filename-style tag like 'h_1_16'."""
    return f"h_1_{int(round(1.0 / h))}"


def load_dataset():
    return np.load(DATA_PATH, allow_pickle=True)


def get_split(D, seed, split_name, h, n=None):
    """Return (m, s, y1, y2) for the given (seed, split, h), optionally
    truncated to the first n samples."""
    pref = f"s{seed}_{h_tag(h)}_{split_name}_"
    out = (D[pref + "m"], D[pref + "s"], D[pref + "y1"], D[pref + "y2"])
    if n is not None:
        out = tuple(a[:n] for a in out)
    return out


def get_split_targets(D, seed, split_name, h, n=None, target_mode=None):
    """Return (m, s, t1, t2) for Experiment 2 calibration.

    target_mode = "paper"  -> use the original two-snapshot targets
        t1 = Y^(1),  t2 = Y^(1) Y^(2)

    target_mode = "pooled" -> use the optional pooled moment targets saved by
    dataset.py when available:
        r1 = average of K Bernoulli snapshots, unbiased for p
        r2 = average of all unordered pair products, unbiased for p^2

    The default is controlled by EXP2_TARGET_MODE and is set to "paper" in the
    updated Experiment 2 package.
    """
    mode = TARGET_MODE if target_mode is None else str(target_mode).strip().lower()
    if mode not in {"paper", "pooled"}:
        raise ValueError("target_mode must be one of: paper, pooled")

    pref = f"s{seed}_{h_tag(h)}_{split_name}_"
    if mode == "pooled" and pref + "r1" in D and pref + "r2" in D:
        out = (D[pref + "m"], D[pref + "s"], D[pref + "r1"], D[pref + "r2"])
    else:
        y1 = D[pref + "y1"]
        y2 = D[pref + "y2"]
        out = (D[pref + "m"], D[pref + "s"], y1, y1 * y2)
    if n is not None:
        out = tuple(a[:n] for a in out)
    return out


def get_raw_split(D, seed, split_name, n=None):
    """Return (m_raw, s_raw, y1, y2) for the given (seed, split) if present."""
    pref = f"s{seed}_raw_{split_name}_"
    required = (pref + "m", pref + "s", pref + "y1", pref + "y2")
    for key in required:
        if key not in D:
            raise KeyError(
                "dataset.npz is missing raw-score splits. "
                f"Please regenerate it with the updated dataset.py (missing key: {key})."
            )
    out = tuple(D[key] for key in required)
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
