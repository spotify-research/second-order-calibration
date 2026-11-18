"""
Regenerate experiment 2 plots from previously-saved results/*.npz files.

This does NOT re-run the experiment.  It loads the precomputed evaluation
arrays and diagnostic statistics and dispatches to the plotting functions in
``experiment2.py``.
"""

import os
import glob
import numpy as np

from common import RESULTS_DIR, h_tag, H_LIST
from experiment2 import (
    plot_main_figure,
    plot_rank_diagnostic,
    plot_raw_diagnostic,
)


def _diag_from_npz(npz):
    return {
        "before_vs_pointwise_gt_corr": float(npz["before_vs_pointwise_gt_corr"]),
        "after_vs_pointwise_gt_corr": float(npz["after_vs_pointwise_gt_corr"]),
        "before_rank": {
            "spearman": float(npz["before_spearman"]),
            "fracs": np.asarray(npz["topk_fracs"], dtype=np.float64),
            "overlap": np.asarray(npz["before_topk_overlap"], dtype=np.float64),
            "oracle_ratio": np.asarray(npz["before_topk_oracle_ratio"], dtype=np.float64),
        },
        "after_rank": {
            "spearman": float(npz["after_spearman"]),
            "fracs": np.asarray(npz["topk_fracs"], dtype=np.float64),
            "overlap": np.asarray(npz["after_topk_overlap"], dtype=np.float64),
            "oracle_ratio": np.asarray(npz["after_topk_oracle_ratio"], dtype=np.float64),
        },
        "before_bins": {
            "x": np.asarray(npz["before_bins_x"], dtype=np.float64),
            "y": np.asarray(npz["before_bins_y"], dtype=np.float64),
        },
        "after_bins": {
            "x": np.asarray(npz["after_bins_x"], dtype=np.float64),
            "y": np.asarray(npz["after_bins_y"], dtype=np.float64),
        },
    }


def _replot_one(npz_path):
    print(f"loading {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    h = float(data["h"])
    cal_n = int(data["cal_n"])
    eval_n = int(data["eval_n"])
    ce2_before = float(data["ce2_before_gt"])
    ce2_after = np.asarray(data["ce2_after_gt"], dtype=np.float64)
    before_epi = np.asarray(data["before_epistemic_eval"], dtype=np.float64)
    pointwise_gt_epi = np.asarray(data["pointwise_gt_epistemic_eval"], dtype=np.float64)
    corr_epi = np.asarray(data["corrected_epistemic_eval"], dtype=np.float64)
    diag = _diag_from_npz(data)

    plot_main_figure(
        h, ce2_before, ce2_after, before_epi, pointwise_gt_epi,
        corr_epi, diag, cal_n, eval_n,
    )
    plot_rank_diagnostic(h, diag)

    if "raw_score_before_epistemic_eval" in data.files and "raw_score_gt_epistemic_eval" in data.files:
        raw_before = np.asarray(data["raw_score_before_epistemic_eval"], dtype=np.float64)
        raw_gt = np.asarray(data["raw_score_gt_epistemic_eval"], dtype=np.float64)
        plot_raw_diagnostic(h, raw_before, raw_gt, eval_n)


def main():
    targets = []
    for h in H_LIST:
        path = os.path.join(RESULTS_DIR, f"experiment2_{h_tag(h)}.npz")
        if os.path.exists(path):
            targets.append(path)
    if not targets:
        targets = sorted(glob.glob(os.path.join(RESULTS_DIR, "experiment2_h_*.npz")))
    if not targets:
        raise SystemExit(f"no experiment2 npz files found under {RESULTS_DIR}")
    for p in targets:
        _replot_one(p)


if __name__ == "__main__":
    main()
