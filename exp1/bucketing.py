"""
Bucketing estimator for CE2.

2D regular grid on [0,1] x [0, 0.25].  The number of buckets per axis
is set by the standard 2D-histogram MSE-balance for an L-Lipschitz
target (L = O(1/h) under sech smoothing):

    K = clip( c * (n / h^2)^{1/4},  K_min,  K_max ).

The constant c is tuned on the hypers split via the squared-error fit
loss (NOT ground truth).  Loops over H_LIST.
"""

import os
import numpy as np
from joblib import Parallel, delayed

from common import (load_dataset, get_split, empirical_ce2, fit_loss,
                    N_GRID, RESULTS_DIR, H_LIST, h_tag)

C_GRID = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
K_MIN, K_MAX = 2, 200


def K_of(c, n, h):
    K = int(round(c * (n / (h ** 2)) ** (1.0 / 4.0)))
    return int(np.clip(K, K_MIN, K_MAX))


def bucket_ids(m, s, K):
    bm = np.minimum((m * K).astype(np.int64), K - 1)
    bs = np.minimum((s / 0.25 * K).astype(np.int64), K - 1)
    return bm * K + bs, K * K


def predict_buckets(m_train, s_train, y1, y2, K, m_eval, s_eval):
    ids_tr, n_bins = bucket_ids(m_train, s_train, K)
    cnt = np.bincount(ids_tr, minlength=n_bins).astype(np.float64)
    s1 = np.bincount(ids_tr, weights=y1, minlength=n_bins)
    p = y1 * y2
    s2 = np.bincount(ids_tr, weights=p, minlength=n_bins)
    safe = np.maximum(cnt, 1.0)
    e1_bin = s1 / safe
    e2_bin = s2 / safe
    e1_global = y1.mean(); e2_global = p.mean()
    ids_ev, _ = bucket_ids(m_eval, s_eval, K)
    e1 = e1_bin[ids_ev]; e2 = e2_bin[ids_ev]
    empty = (cnt[ids_ev] == 0)
    e1[empty] = e1_global; e2[empty] = e2_global
    return e1, e2


def tune_c_adaptive(m_tr, s_tr, y1_tr, y2_tr, m_hp, s_hp, y1_hp, y2_hp, n, h):
    """Tune c on the hyper split, expanding the grid if the best value
    lands on a search boundary.
    """
    tried = {}
    grid = list(C_GRID)

    def loss_of(c):
        if c not in tried:
            K = K_of(c, n, h)
            e1_hp, e2_hp = predict_buckets(m_tr, s_tr, y1_tr, y2_tr, K, m_hp, s_hp)
            tried[c] = fit_loss(e1_hp, e2_hp, y1_hp, y2_hp)
        return tried[c]

    for c in grid:
        loss_of(c)

    while True:
        best_c = min(grid, key=loss_of)
        best_idx = grid.index(best_c)

        if best_idx == 0 and K_of(best_c, n, h) > K_MIN:
            new_c = best_c / 2.0
            grid.insert(0, new_c)
            loss_of(new_c)
            continue

        if best_idx == len(grid) - 1 and K_of(best_c, n, h) < K_MAX:
            new_c = best_c * 2.0
            grid.append(new_c)
            loss_of(new_c)
            continue

        return best_c


def run_one(seed, n, h, ce2_gt):
    D = load_dataset()
    m_tr, s_tr, y1_tr, y2_tr = get_split(D, seed, "train", h, n)
    m_hp, s_hp, y1_hp, y2_hp = get_split(D, seed, "hyper", h, n)
    m_va, s_va, _, _ = get_split(D, seed, "valid", h, n)
    best_c = tune_c_adaptive(
        m_tr, s_tr, y1_tr, y2_tr,
        m_hp, s_hp, y1_hp, y2_hp,
        n, h,
    )
    K = K_of(best_c, n, h)
    e1_va, e2_va = predict_buckets(
        m_tr, s_tr, y1_tr, y2_tr, K, m_va, s_va)
    ce2_hat = empirical_ce2(e1_va, e2_va, m_va, s_va)
    return dict(seed=seed, n=n, h=h,
                ce2_hat=float(ce2_hat),
                err=float(abs(ce2_hat - ce2_gt)),
                chosen_c=float(best_c), K=K)


def main():
    D = load_dataset()
    n_seeds = int(D["n_seeds"])
    for h in H_LIST:
        ce2_gt = float(D[f"ce2_gt_{h_tag(h)}"])
        jobs = [(seed, n, h) for n in N_GRID for seed in range(n_seeds)]
        print(f"--- bucketing, h={h:.5f} ({h_tag(h)}), GT CE2={ce2_gt:.4f} ---")
        results = Parallel(n_jobs=-1, backend="loky", verbose=5)(
            delayed(run_one)(seed, n, h, ce2_gt) for seed, n, h in jobs)
        out = os.path.join(RESULTS_DIR, f"bucketing_{h_tag(h)}.npz")
        np.savez(out, results=np.array(results, dtype=object), ce2_gt=ce2_gt, h=h)
        print(f"Saved {out}")


if __name__ == "__main__":
    main()