"""
Nadaraya-Watson estimator for CE2.

We use the NW estimator with bandwidth of the form

    b = c * h^(1/2) * n^(-1/4).

This is optimal in the distribution-free setting 
(ie. distributon of (m,s) completely arbitrary).

We tune the multiplier c on the hypers split via the squared-error fit
loss.  We use a Gaussian kernel with truncation at 5 sigma and a
BallTree to keep neighbour queries fast on 50k x 50k.
"""

import os
import math
import numpy as np
from joblib import Parallel, delayed
from sklearn.neighbors import BallTree

from common import (load_dataset, get_split, empirical_ce2, fit_loss,
                    N_GRID, RESULTS_DIR, H_LIST, h_tag)

C_GRID = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
RADIUS_FACTOR = 5.0


def b_of(c, n, h):
    return c * (h ** (1.0 / 2.0)) * (n ** (-1.0 / 4.0))


def _scale_pts(m, s):
    """Stretch the s-axis so the kernel bandwidth is meaningful in both
    dimensions (both axes have unit support)."""
    return np.stack([m, s * 4.0], axis=1)


def nw_predict(m_tr, s_tr, y1_tr, y2_tr, m_ev, s_ev, b):
    P_tr = _scale_pts(m_tr, s_tr)
    P_ev = _scale_pts(m_ev, s_ev)
    p_tr = y1_tr * y2_tr
    tree = BallTree(P_tr)
    r = RADIUS_FACTOR * b
    inv_2bb = 1.0 / (2.0 * b * b)
    n_ev = len(P_ev)
    e1 = np.empty(n_ev); e2 = np.empty(n_ev)
    e1_global = y1_tr.mean(); e2_global = p_tr.mean()
    CHUNK = 2048
    for start in range(0, n_ev, CHUNK):
        end = min(start + CHUNK, n_ev)
        idxs, dists = tree.query_radius(
            P_ev[start:end], r=r, return_distance=True)
        for j, (ii, dd) in enumerate(zip(idxs, dists)):
            row = start + j
            if len(ii) == 0:
                e1[row] = e1_global; e2[row] = e2_global; continue
            w = np.exp(-(dd ** 2) * inv_2bb)
            wsum = w.sum()
            if wsum < 1e-12:
                e1[row] = e1_global; e2[row] = e2_global
            else:
                e1[row] = (w * y1_tr[ii]).sum() / wsum
                e2[row] = (w * p_tr[ii]).sum() / wsum
    return e1, e2


def run_one(seed, n, h, ce2_gt):
    D = load_dataset()
    m_tr, s_tr, y1_tr, y2_tr = get_split(D, seed, "train", h, n)
    m_hp, s_hp, y1_hp, y2_hp = get_split(D, seed, "hyper", h, n)
    m_va, s_va, _, _ = get_split(D, seed, "valid", h, n)
    best_c, best_loss = C_GRID[0], np.inf
    for c in C_GRID:
        b = b_of(c, n, h)
        e1_hp, e2_hp = nw_predict(m_tr, s_tr, y1_tr, y2_tr,
                                  m_hp, s_hp, b)
        loss = fit_loss(e1_hp, e2_hp, y1_hp, y2_hp)
        if loss < best_loss:
            best_loss, best_c = loss, c
    b = b_of(best_c, n, h)
    e1_va, e2_va = nw_predict(m_tr, s_tr, y1_tr, y2_tr, m_va, s_va, b)
    ce2_hat = empirical_ce2(e1_va, e2_va, m_va, s_va)
    return dict(seed=seed, n=n, h=h,
                ce2_hat=float(ce2_hat),
                err=float(abs(ce2_hat - ce2_gt)),
                chosen_c=float(best_c), bandwidth=float(b))


def main():
    D = load_dataset()
    n_seeds = int(D["n_seeds"])
    for h in H_LIST:
        ce2_gt = float(D[f"ce2_gt_{h_tag(h)}"])
        jobs = [(seed, n, h) for n in N_GRID for seed in range(n_seeds)]
        print(f"--- NW, h={h:.5f} ({h_tag(h)}), GT CE2={ce2_gt:.4f} ---")
        results = Parallel(n_jobs=-1, backend="loky", verbose=5)(
            delayed(run_one)(seed, n, h, ce2_gt) for seed, n, h in jobs)
        out = os.path.join(RESULTS_DIR, f"nadaraya_watson_{h_tag(h)}.npz")
        np.savez(out, results=np.array(results, dtype=object), ce2_gt=ce2_gt, h=h)
        print(f"Saved {out}")


if __name__ == "__main__":
    main()