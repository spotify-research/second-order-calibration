"""
Regenerate the exp_crowd_weather_disagreement figures from the saved .npz
without re-running the experiment.
"""
from __future__ import annotations

import json
import numpy as np

from experiment_crowd_weather_disagreement import (
    CandidateConfig,
    METHODS,
    OUT_PREFIX,
    RESULTS_DIR,
    SearchResult,
    plot_results,
    safe_key,
)


def main() -> None:
    npz_path = RESULTS_DIR / f"{OUT_PREFIX}.npz"
    data = np.load(npz_path, allow_pickle=True)
    cfg_dict = json.loads(str(data["selected_config_json"]))
    cfg = CandidateConfig(**cfg_dict)
    sel = json.loads(str(data["selection_result_json"]))
    summary = json.loads(str(data["summary_json"]))

    curve_mean = {}
    curve_sem = {}
    curve_stack = {}
    for m in METHODS:
        key = safe_key(m)
        curve_mean[m] = np.asarray(data[f"curve_mean_{key}"], dtype=float)
        curve_sem[m] = np.asarray(data[f"curve_sem_{key}"], dtype=float)
        rk = f"curve_repeats_{key}"
        if rk in data.files:
            curve_stack[m] = np.asarray(data[rk], dtype=float)
    if not curve_stack:
        curve_stack = None

    best = SearchResult(
        config=cfg,
        lift_vs_all=float(sel["lift_vs_all"]),
        lift_vs_first=float(sel["lift_vs_first"]),
        auc_2d_second=float(sel["auc_2d_second"]),
        auc_best_competitor=float(sel["auc_best_competitor"]),
        aucs={k: float(v) for k, v in sel["aucs"].items()},
        valid_reps=int(sel["valid_reps"]),
        n_items=int(sel["n_items"]),
        mean_votes_per_item=float(sel["mean_votes_per_item"]),
        cohort_size=float(sel["cohort_size"]),
        hidden_frac=float(sel["hidden_frac"]),
    )

    plot_results(curve_mean, curve_sem, summary, cfg, best, curve_stack)


if __name__ == "__main__":
    main()
