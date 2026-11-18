"""
Experiment 2: recalibration quality (synthetic), with active-learning diagnostics.

Defaults:

1) the calibration target is the two-snapshot pair (Y^(1), Y^(1) Y^(2));
2) the score perturbation uses h on both coordinates (h_s = h);
3) the ensemble defaults in dataset.py are tuned to leave more usable epistemic
   signal without resorting to pooled multi-snapshot targets.

The main scatter still uses the same pointwise GT epistemic target before and
after calibration.  The post-calibration transported GT surface is used for
the true post-calibration CE2 computation.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau

from common import load_dataset, get_split_targets, get_raw_split, RESULTS_DIR, H_LIST, h_tag, TARGET_MODE
from dataset import deposit_weighted_bilinear, interpolate_grid_bilinear, ce2_from_surface
from polynomial import (
    cheb_vander,
    prepare_model,
    solve_primal_from_gram,
    solve_dual_from_kernel,
    dual_kernel_cross,
)

plt.rcParams.update({
    "text.usetex": True,
    "font.size": 34,
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman"],
    "axes.labelsize": 35,
    "axes.titlesize": 35,
    "xtick.labelsize": 33,
    "ytick.labelsize": 33,
    "legend.fontsize": 29,
    "legend.frameon": False,
})

CAL_SPLIT = os.environ.get("EXP2_CAL_SPLIT", "train")
EVAL_SPLIT = os.environ.get("EXP2_EVAL_SPLIT", "valid")
CAL_N_REQUESTED = int(os.environ.get("EXP2_CAL_N", "20000"))
EVAL_N_REQUESTED = int(os.environ.get("EXP2_EVAL_N", "20000"))
N_REPEATS_REQUESTED = int(os.environ.get("EXP2_N_REPEATS", "20"))
SCATTER_SEED = int(os.environ.get("EXP2_SCATTER_SEED", "0"))
PRED_CHUNK = int(os.environ.get("EXP2_PRED_CHUNK", "65536"))
ERM_ALPHA = float(os.environ.get("EXP2_ERM_ALPHA", "0.0"))
FIXED_DEGREE = int(os.environ.get("EXP2_FIXED_DEGREE", "12"))
SAVE_RAW_DIAGNOSTIC = os.environ.get("EXP2_SAVE_RAW_DIAGNOSTIC", "0").strip().lower() in {"1", "true", "yes", "on"}
SAVE_RANK_DIAGNOSTIC = os.environ.get("EXP2_SAVE_RANK_DIAGNOSTIC", "1").strip().lower() in {"1", "true", "yes", "on"}
PROJECT_FEASIBLE_MOMENTS = os.environ.get("EXP2_PROJECT_FEASIBLE_MOMENTS", "1").strip().lower() in {"1", "true", "yes", "on"}
RANK_N_BINS = int(os.environ.get("EXP2_RANK_BINS", "20"))
TOPK_FRACS = os.environ.get("EXP2_TOPK_FRACS", "0.005,0.01,0.02,0.05,0.1")
TOPK_FRACS = np.array([float(tok.strip()) for tok in TOPK_FRACS.split(",") if tok.strip()], dtype=np.float64)
TOPK_FRACS = np.array(sorted(TOPK_FRACS.tolist()), dtype=np.float64)
TARGET_MODE = TARGET_MODE
if FIXED_DEGREE < 1:
    raise ValueError("EXP2_FIXED_DEGREE must be a positive integer")
if RANK_N_BINS < 4:
    raise ValueError("EXP2_RANK_BINS must be at least 4")
if TOPK_FRACS.size == 0 or np.any(TOPK_FRACS <= 0.0) or np.any(TOPK_FRACS >= 1.0):
    raise ValueError("EXP2_TOPK_FRACS must contain fractions strictly between 0 and 1")


def clip_prob(x):
    return np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)



def clip_epistemic(var):
    return np.clip(np.asarray(var, dtype=np.float64), 0.0, 0.25)



def project_moment_pair(e1, e2):
    e1 = clip_prob(e1)
    e2 = np.asarray(e2, dtype=np.float64)
    if PROJECT_FEASIBLE_MOMENTS:
        lower = e1 ** 2
        upper = e1
        e2 = np.clip(e2, lower, upper)
    else:
        e2 = clip_prob(e2)
    return e1, e2



def load_gt_surface(D, h):
    tag = h_tag(h)
    required = [
        "gt_m_nodes",
        "gt_s_nodes",
        f"gt_mass_{tag}",
        f"gt_eta1_{tag}",
        f"gt_eta2_{tag}",
        f"ce2_gt_{tag}",
    ]
    for key in required:
        if key not in D:
            raise KeyError(
                "dataset.npz is missing ground-truth calibration surfaces. "
                "Please regenerate it with the updated dataset.py before running "
                f"experiment2.py (missing key: {key})."
            )
    return {
        "m_nodes": D["gt_m_nodes"].astype(np.float64, copy=False),
        "s_nodes": D["gt_s_nodes"].astype(np.float64, copy=False),
        "mass": D[f"gt_mass_{tag}"].astype(np.float64, copy=False),
        "eta1": D[f"gt_eta1_{tag}"].astype(np.float64, copy=False),
        "eta2": D[f"gt_eta2_{tag}"].astype(np.float64, copy=False),
    }



def load_raw_gt_surface(D):
    required = [
        "gt_m_nodes",
        "gt_s_nodes",
        "gt_raw_mass",
        "gt_raw_eta1",
        "gt_raw_eta2",
        "ce2_gt_raw",
    ]
    for key in required:
        if key not in D:
            raise KeyError(
                "dataset.npz is missing raw-score GT surfaces. "
                "Please regenerate it with the updated dataset.py before running "
                f"experiment2.py (missing key: {key})."
            )
    return {
        "m_nodes": D["gt_m_nodes"].astype(np.float64, copy=False),
        "s_nodes": D["gt_s_nodes"].astype(np.float64, copy=False),
        "mass": D["gt_raw_mass"].astype(np.float64, copy=False),
        "eta1": D["gt_raw_eta1"].astype(np.float64, copy=False),
        "eta2": D["gt_raw_eta2"].astype(np.float64, copy=False),
    }



def interpolate_gt_epistemic(gt, m, s):
    eta1 = interpolate_grid_bilinear(gt["eta1"], m, s, gt["m_nodes"], gt["s_nodes"])
    eta2 = interpolate_grid_bilinear(gt["eta2"], m, s, gt["m_nodes"], gt["s_nodes"])
    return clip_epistemic(eta2 - eta1 ** 2), eta1, eta2



def choose_erm_degree(n, h):
    del n, h
    return FIXED_DEGREE



def fit_erm_polynomial_model(m, s, r1, r2, h):
    n = len(m)
    degree = choose_erm_degree(n, h)
    Vx, Vy = cheb_vander(m, s, degree)
    Y = np.column_stack([r1, r2]).astype(np.float64, copy=False)
    prepared = prepare_model(Vx, Vy, Y, degree)

    model = {
        "degree": degree,
        "alpha": float(ERM_ALPHA),
        "mode": prepared["mode"],
    }

    if prepared["mode"] == "primal":
        _, _, _, beta = solve_primal_from_gram(
            prepared["G"], prepared["rhs"], ERM_ALPHA, prepared["scale"]
        )
        beta = np.asarray(beta, dtype=np.float64)
        q = degree + 1
        model["beta"] = beta
        model["B1"] = beta[:, 0].reshape(q, q)
        model["B2"] = beta[:, 1].reshape(q, q)
    else:
        _, _, _, coef = solve_dual_from_kernel(
            prepared["K"], prepared["Y"], ERM_ALPHA, prepared["scale"]
        )
        model["coef"] = np.asarray(coef, dtype=np.float64)
        model["Vx_tr"] = prepared["Vx_tr"]
        model["Vy_tr"] = prepared["Vy_tr"]
    return model



def pairwise_predict_from_primal(Vx, Vy, B1, B2):
    p1 = np.sum((Vx @ B1) * Vy, axis=1)
    p2 = np.sum((Vx @ B2) * Vy, axis=1)
    return project_moment_pair(p1, p2)



def predict_eta(model, m, s, chunk_size=PRED_CHUNK):
    n = len(m)
    out1 = np.empty(n, dtype=np.float64)
    out2 = np.empty(n, dtype=np.float64)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        Vx, Vy = cheb_vander(m[start:end], s[start:end], model["degree"])
        if model["mode"] == "primal":
            e1, e2 = pairwise_predict_from_primal(Vx, Vy, model["B1"], model["B2"])
        else:
            K = dual_kernel_cross(Vx, Vy, model["Vx_tr"], model["Vy_tr"], model["degree"])
            pred = K @ model["coef"]
            e1, e2 = project_moment_pair(pred[:, 0], pred[:, 1])
        out1[start:end] = e1
        out2[start:end] = e2
    return out1, out2



def predict_eta_on_tensor_grid(model, m_nodes, s_nodes):
    if model["mode"] != "primal":
        m_grid, s_grid = np.meshgrid(m_nodes, s_nodes, indexing="ij")
        return predict_eta(model, m_grid.reshape(-1), s_grid.reshape(-1), chunk_size=PRED_CHUNK)

    Vx_nodes, _ = cheb_vander(m_nodes, np.zeros_like(m_nodes), model["degree"])
    _, Vy_nodes = cheb_vander(np.zeros_like(s_nodes), s_nodes, model["degree"])
    e1, e2 = project_moment_pair(Vx_nodes @ model["B1"] @ Vy_nodes.T,
                                 Vx_nodes @ model["B2"] @ Vy_nodes.T)
    return e1, e2



def corrected_epistemic_from_scores(e1_hat, e2_hat):
    return clip_epistemic(e2_hat - e1_hat ** 2)



def make_gt_transport_cache(gt):
    mass_flat = gt["mass"].reshape(-1).astype(np.float64, copy=False)
    eta1_flat = gt["eta1"].reshape(-1).astype(np.float64, copy=False)
    eta2_flat = gt["eta2"].reshape(-1).astype(np.float64, copy=False)
    return {
        "mass_flat": mass_flat,
        "w1_flat": mass_flat * eta1_flat,
        "w2_flat": mass_flat * eta2_flat,
        "m_nodes": gt["m_nodes"],
        "s_nodes": gt["s_nodes"],
    }



def transport_ground_truth_surface(model, gt_cache):
    m_nodes = gt_cache["m_nodes"]
    s_nodes = gt_cache["s_nodes"]
    e1_grid, e2_grid = predict_eta_on_tensor_grid(model, m_nodes, s_nodes)
    s_corr = corrected_epistemic_from_scores(e1_grid, e2_grid)

    W0, W1, W2 = deposit_weighted_bilinear(
        e1_grid.reshape(-1),
        s_corr.reshape(-1),
        gt_cache["mass_flat"],
        gt_cache["w1_flat"],
        gt_cache["w2_flat"],
        m_nodes,
        s_nodes,
    )
    mass = np.maximum(W0, 0.0)
    mass_sum = float(mass.sum())
    if not np.isfinite(mass_sum) or mass_sum <= 0.0:
        raise RuntimeError("invalid transported GT mass after calibration")
    mass = mass / mass_sum
    denom = np.maximum(W0, np.finfo(np.float64).tiny)
    eta1_corr = W1 / denom
    eta2_corr = W2 / denom
    ce2 = ce2_from_surface(mass, eta1_corr, eta2_corr, m_nodes, s_nodes)
    return {
        "m_nodes": m_nodes,
        "s_nodes": s_nodes,
        "mass": mass,
        "eta1": eta1_corr,
        "eta2": eta2_corr,
        "ce2": ce2,
    }



def summarise_array(x):
    q = np.quantile(x, [0.5, 0.9, 0.99])
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "q50": float(q[0]),
        "q90": float(q[1]),
        "q99": float(q[2]),
        "max": float(np.max(x)),
    }



def safe_corr(a, b):
    sa = float(np.std(a))
    sb = float(np.std(b))
    if sa <= 1e-15 or sb <= 1e-15:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])



def safe_spearman(a, b):
    sa = float(np.std(a))
    sb = float(np.std(b))
    if sa <= 1e-15 or sb <= 1e-15:
        return float("nan")
    val = spearmanr(a, b).statistic
    return float(val)


def safe_kendall(a, b):
    sa = float(np.std(a))
    sb = float(np.std(b))
    if sa <= 1e-15 or sb <= 1e-15:
        return float("nan")
    val = kendalltau(a, b).statistic
    return float(val)



def signal_scale_diagnostic(raw_s, h_s):
    q = np.quantile(raw_s, [0.5, 0.9, 0.99])
    if float(h_s) > 0.0:
        mean_over_hs = float(np.mean(raw_s) / float(h_s))
        q90_over_hs = float(q[1] / float(h_s))
    else:
        mean_over_hs = float("nan")
        q90_over_hs = float("nan")
    return {
        "raw_s_mean": float(np.mean(raw_s)),
        "raw_s_q50": float(q[0]),
        "raw_s_q90": float(q[1]),
        "raw_s_q99": float(q[2]),
        "h_s": float(h_s),
        "mean_over_hs": mean_over_hs,
        "q90_over_hs": q90_over_hs,
    }



def equal_count_bin_curve(score, gt, n_bins):
    order = np.argsort(score, kind="mergesort")
    bins = np.array_split(order, n_bins)
    xs, ys, counts = [], [], []
    for idx in bins:
        if len(idx) == 0:
            continue
        xs.append(float(np.mean(score[idx])))
        ys.append(float(np.mean(gt[idx])))
        counts.append(int(len(idx)))
    return {
        "x": np.array(xs, dtype=np.float64),
        "y": np.array(ys, dtype=np.float64),
        "count": np.array(counts, dtype=np.int64),
    }



def active_learning_metrics(score, gt, fracs):
    n = len(score)
    order_score = np.argsort(score)[::-1]
    order_gt = np.argsort(gt)[::-1]
    gt_mean = float(np.mean(gt))
    overlap = np.empty(len(fracs), dtype=np.float64)
    oracle_ratio = np.empty(len(fracs), dtype=np.float64)
    lift = np.empty(len(fracs), dtype=np.float64)
    selected_mean = np.empty(len(fracs), dtype=np.float64)
    oracle_mean = np.empty(len(fracs), dtype=np.float64)
    for i, frac in enumerate(fracs):
        k = max(1, int(np.ceil(float(frac) * n)))
        sel = order_score[:k]
        oracle = order_gt[:k]
        overlap[i] = np.intersect1d(sel, oracle, assume_unique=False).size / float(k)
        selected_mean[i] = float(np.mean(gt[sel]))
        oracle_mean[i] = float(np.mean(gt[oracle]))
        oracle_ratio[i] = selected_mean[i] / oracle_mean[i] if oracle_mean[i] > 0.0 else np.nan
        lift[i] = selected_mean[i] / gt_mean if gt_mean > 0.0 else np.nan
    return {
        "spearman": safe_spearman(score, gt),
        "kendall": safe_kendall(score, gt),
        "fracs": fracs.astype(np.float64, copy=True),
        "overlap": overlap,
        "oracle_ratio": oracle_ratio,
        "lift": lift,
        "selected_mean": selected_mean,
        "oracle_mean": oracle_mean,
        "global_mean": gt_mean,
    }



def make_main_diagnostics(before_epi, pointwise_gt_epi, corr_epi, transported_gt_epi):
    thresholds = np.array([1e-3, 5e-3, 1e-2], dtype=np.float64)
    frac_before = np.array([np.mean(before_epi > t) for t in thresholds], dtype=np.float64)
    frac_gt = np.array([np.mean(pointwise_gt_epi > t) for t in thresholds], dtype=np.float64)
    frac_after = np.array([np.mean(corr_epi > t) for t in thresholds], dtype=np.float64)
    frac_after_transport = np.array([np.mean(transported_gt_epi > t) for t in thresholds], dtype=np.float64)
    before_rank = active_learning_metrics(before_epi, pointwise_gt_epi, TOPK_FRACS)
    after_rank = active_learning_metrics(corr_epi, pointwise_gt_epi, TOPK_FRACS)
    return {
        "thresholds": thresholds,
        "before_summary": summarise_array(before_epi),
        "pointwise_gt_summary": summarise_array(pointwise_gt_epi),
        "after_summary": summarise_array(corr_epi),
        "transported_gt_summary": summarise_array(transported_gt_epi),
        "before_vs_pointwise_gt_corr": safe_corr(before_epi, pointwise_gt_epi),
        "after_vs_pointwise_gt_corr": safe_corr(corr_epi, pointwise_gt_epi),
        "after_vs_transported_gt_corr": safe_corr(corr_epi, transported_gt_epi),
        "frac_before_above": frac_before,
        "frac_pointwise_gt_above": frac_gt,
        "frac_after_above": frac_after,
        "frac_transported_gt_above": frac_after_transport,
        "before_rank": before_rank,
        "after_rank": after_rank,
        "before_bins": equal_count_bin_curve(before_epi, pointwise_gt_epi, RANK_N_BINS),
        "after_bins": equal_count_bin_curve(corr_epi, pointwise_gt_epi, RANK_N_BINS),
    }



def print_diagnostics(h, ce2_before, ce2_after, diag, per_seed, raw_score_diag=None,
                      signal_diag=None, dataset_meta=None):
    tag = h_tag(h)
    after_q = np.quantile(ce2_after, [0.1, 0.5, 0.9])
    print(f"\n--- Experiment 2, {tag} ---")
    print(
        f"GT CE2 before calibration = {ce2_before:.6f}; "
        f"after calibration mean/median = {np.mean(ce2_after):.6f}/{after_q[1]:.6f}; "
        f"10/90% quantiles = [{after_q[0]:.6f}, {after_q[2]:.6f}]"
    )
    print(
        f"Fixed ERM polynomial degree across seeds: "
        f"{[int(r['degree']) for r in per_seed]}"
    )
    print(f"Configured fixed ERM degree = {FIXED_DEGREE}")
    print(f"Project predicted moments to feasible set = {PROJECT_FEASIBLE_MOMENTS}")
    print(
        f"Numerical ridge alpha across seeds: "
        f"{[float(r['alpha']) for r in per_seed]}"
    )
    if dataset_meta is not None:
        print(
            "Dataset defaults used here: "
            f"ensemble_size = {dataset_meta['ensemble_size']}, "
            f"ensemble_train_n = {dataset_meta['ensemble_train_n']}, "
            f"member_data_mode = {dataset_meta['ensemble_member_data_mode']}, "
            f"ensemble_epochs = {dataset_meta['ensemble_epochs']}, "
            f"ensemble_lr = {dataset_meta['ensemble_lr']:.4g}, "
            f"hidden = {dataset_meta['ensemble_hidden']}, "
            f"batch = {dataset_meta['ensemble_batch']}, "
            f"s_h_scale = {dataset_meta['s_h_scale']:.4f}, "
            f"exp2_cal_snapshots = {dataset_meta['exp2_cal_snapshots']}, "
            f"target_mode = {dataset_meta['target_mode']}"
        )
    print(
        "Main scatter correlations vs the same pointwise GT epistemic target: "
        f"before = {diag['before_vs_pointwise_gt_corr']:.4f}, "
        f"after = {diag['after_vs_pointwise_gt_corr']:.4f}"
    )
    print(
        "Rank correlations vs the same pointwise GT target: "
        f"spearman before/after = {diag['before_rank']['spearman']:.4f}/{diag['after_rank']['spearman']:.4f}; "
        f"kendall before/after = {diag['before_rank']['kendall']:.4f}/{diag['after_rank']['kendall']:.4f}"
    )
    print(
        "Secondary diagnostic (after vs transported post-cal GT surface): "
        f"{diag['after_vs_transported_gt_corr']:.4f}"
    )
    print(
        "Before  epistemic mean/q90/q99/max = "
        f"{diag['before_summary']['mean']:.5f} / {diag['before_summary']['q90']:.5f} / "
        f"{diag['before_summary']['q99']:.5f} / {diag['before_summary']['max']:.5f}"
    )
    print(
        "Pointwise-GT epistemic mean/q90/q99/max = "
        f"{diag['pointwise_gt_summary']['mean']:.5f} / {diag['pointwise_gt_summary']['q90']:.5f} / "
        f"{diag['pointwise_gt_summary']['q99']:.5f} / {diag['pointwise_gt_summary']['max']:.5f}"
    )
    print(
        "After   epistemic mean/q90/q99/max = "
        f"{diag['after_summary']['mean']:.5f} / {diag['after_summary']['q90']:.5f} / "
        f"{diag['after_summary']['q99']:.5f} / {diag['after_summary']['max']:.5f}"
    )
    for t, fb, fgt, fa in zip(
        diag["thresholds"],
        diag["frac_before_above"],
        diag["frac_pointwise_gt_above"],
        diag["frac_after_above"],
    ):
        print(
            f"Frac(epistemic > {t:.3g}): "
            f"before = {fb:.3f}, pointwise-GT = {fgt:.3f}, after = {fa:.3f}"
        )
    print("Active-learning style top-k diagnostics (same pointwise GT target):")
    for frac, br, ar, bo, ao in zip(
        diag["before_rank"]["fracs"],
        diag["before_rank"]["oracle_ratio"],
        diag["after_rank"]["oracle_ratio"],
        diag["before_rank"]["overlap"],
        diag["after_rank"]["overlap"],
    ):
        print(
            f"  top {100.0 * frac:.1f}%: oracle-ratio before/after = {br:.3f}/{ar:.3f}; "
            f"overlap before/after = {bo:.3f}/{ao:.3f}"
        )

    if raw_score_diag is not None:
        print(
            "Separate raw-score diagnostic: corr(raw ensemble variance, raw GT epistemic) = "
            f"{raw_score_diag['corr']:.4f}; spearman = {raw_score_diag['spearman']:.4f}; "
            f"mean raw s = {raw_score_diag['raw_summary']['mean']:.6f}, "
            f"mean raw GT epistemic = {raw_score_diag['gt_summary']['mean']:.6f}"
        )

    if signal_diag is not None:
        if signal_diag["h_s"] > 0.0:
            print(
                f"Second-coordinate perturbation scale: h_s = {signal_diag['h_s']:.6f}; "
                f"raw-s mean/h_s = {signal_diag['mean_over_hs']:.4f}, "
                f"raw-s q90/h_s = {signal_diag['q90_over_hs']:.4f}"
            )
        else:
            print(
                "Second-coordinate perturbation scale: h_s = 0.000000 "
                "(no perturbation on the variance coordinate)"
            )



def plot_main_figure(h, ce2_before, ce2_after, before_epi, pointwise_gt_epi,
                     corr_epi, diag, cal_n, eval_n):
    tag = h_tag(h)
    h_label = f"$h = 1/{int(round(1.0 / h))}$"

    upper_main = float(np.quantile(
        np.concatenate([before_epi, corr_epi, pointwise_gt_epi]), 0.995
    ))
    upper_main = min(max(upper_main * 1.05, 1e-4), 0.25)
    xs = np.array([0.0, upper_main], dtype=np.float64)

    fig, ax = plt.subplots(1, 1, figsize=(10, 7.5))
    ax.scatter(before_epi, pointwise_gt_epi, s=6, alpha=0.30, c="#d62728", edgecolors="none")
    ax.plot(xs, xs, "k--", linewidth=1.0, alpha=0.8)
    ax.set_xlim(0.0, upper_main)
    ax.set_ylim(0.0, upper_main)
    ax.set_yticks([0, 0.05, 0.1, 0.15])
    ax.set_xlabel(r"Reported epistemic variance $\sigma^2$")
    ax.set_ylabel("Ground-truth\nepistemic variance")
    ax.set_title(
        f"Before ({h_label})\npearson = {diag['before_vs_pointwise_gt_corr']:.3f}, "
        f"spearman = {diag['before_rank']['spearman']:.3f}"
    )
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, f"experiment2_{tag}_before.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"saved {out}")

    fig, ax = plt.subplots(1, 1, figsize=(10, 7.5))
    ax.scatter(corr_epi, pointwise_gt_epi, s=6, alpha=0.30, c="#2ca02c", edgecolors="none")
    ax.plot(xs, xs, "k--", linewidth=1.0, alpha=0.8)
    ax.set_xlim(0.0, upper_main)
    ax.set_ylim(0.0, upper_main)
    ax.set_yticks([0, 0.05, 0.1, 0.15])
    ax.set_xlabel(r"Corrected epistemic variance $(\sigma^2)'$")
    ax.set_ylabel("Ground-truth\nepistemic variance")
    ax.set_title(
        f"After ({h_label})\npearson = {diag['after_vs_pointwise_gt_corr']:.3f}, "
        f"spearman = {diag['after_rank']['spearman']:.3f}"
    )
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, f"experiment2_{tag}_after.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"saved {out}")

    fig, ax = plt.subplots(1, 1, figsize=(5.0, 4.4))
    bb = diag["before_bins"]
    ab = diag["after_bins"]
    ax.plot(bb["x"], bb["y"], marker="o", markersize=3.0, linewidth=1.2,
            color="#d62728", label="before")
    ax.plot(ab["x"], ab["y"], marker="o", markersize=3.0, linewidth=1.2,
            color="#2ca02c", label="after")
    ax.plot(xs, xs, "k--", linewidth=1.0, alpha=0.8)
    ax.set_xlim(0.0, upper_main)
    ax.set_ylim(0.0, upper_main)
    ax.set_yticks([0, 0.05, 0.1, 0.15])
    ax.set_xlabel("Mean reported epistemic variance (equal-count bins)")
    ax.set_ylabel("Mean GT epistemic variance")
    ax.set_title(f"Binned monotonicity / calibration ({h_label})")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, f"experiment2_{tag}_binned.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"saved {out}")

    fig, ax = plt.subplots(1, 1, figsize=(5.0, 4.4))
    after_mean = float(np.mean(ce2_after))
    q10, q90 = np.quantile(ce2_after, [0.1, 0.9])
    ax.bar([0, 1], [ce2_before, after_mean], width=0.58,
           color=["#d62728", "#2ca02c"], alpha=0.35)
    ax.errorbar([1], [after_mean],
                yerr=[[after_mean - q10], [q90 - after_mean]],
                fmt="none", ecolor="k", capsize=4, linewidth=1.0)
    rng = np.random.default_rng(0)
    jitter = rng.normal(0.0, 0.03, size=len(ce2_after))
    ax.scatter(np.full(len(ce2_after), 1.0) + jitter, ce2_after,
               s=18, alpha=0.85, c="#2ca02c")
    ax.set_xticks([0, 1], ["before", "after"])
    ax.set_ylabel(r"Ground-truth CE$_2$")
    ax.set_title(f"Recalibration effect across seeds ({h_label})")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, f"experiment2_{tag}_ce2.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"saved {out}")



def plot_rank_diagnostic(h, diag):
    tag = h_tag(h)
    h_label = f"$h = 1/{int(round(1.0 / h))}$"
    fracs_pct = 100.0 * diag["before_rank"]["fracs"]

    fig, ax = plt.subplots(1, 1, figsize=(5.2, 4.2))
    ax.plot(fracs_pct, diag["before_rank"]["oracle_ratio"], marker="o", linewidth=1.4,
            color="#d62728", label="before")
    ax.plot(fracs_pct, diag["after_rank"]["oracle_ratio"], marker="o", linewidth=1.4,
            color="#2ca02c", label="after")
    ax.axhline(1.0, linestyle="--", linewidth=1.0, color="k", alpha=0.8)
    ax.set_xlabel("Acquisition budget (\\%)")
    ax.set_ylabel("Mean GT selected / oracle")
    ax.set_title(f"Top-k oracle ratio ({h_label})")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, f"experiment2_rankdiag_{tag}_oracle.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"saved {out}")

    fig, ax = plt.subplots(1, 1, figsize=(5.2, 4.2))
    ax.plot(fracs_pct, diag["before_rank"]["overlap"], marker="o", linewidth=1.4,
            color="#d62728", label="before")
    ax.plot(fracs_pct, diag["after_rank"]["overlap"], marker="o", linewidth=1.4,
            color="#2ca02c", label="after")
    ax.set_xlabel("Acquisition budget (\\%)")
    ax.set_ylabel("Top-k overlap with oracle")
    ax.set_title(f"Selection overlap ({h_label})")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, f"experiment2_rankdiag_{tag}_overlap.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"saved {out}")



def plot_raw_diagnostic(h, raw_before, raw_gt, eval_n):
    tag = h_tag(h)
    upper = float(np.quantile(np.concatenate([raw_before, raw_gt]), 0.995))
    upper = min(max(upper * 1.05, 1e-6), 0.25)
    xs = np.array([0.0, upper], dtype=np.float64)
    fig, ax = plt.subplots(1, 1, figsize=(5.3, 4.3))
    ax.scatter(raw_before, raw_gt, s=6, alpha=0.30, c="#1f77b4", edgecolors="none")
    ax.plot(xs, xs, "k--", linewidth=1.0, alpha=0.8)
    ax.set_xlim(0.0, upper)
    ax.set_ylim(0.0, upper)
    ax.set_xlabel(r"Raw ensemble epistemic variance $\sigma^2$")
    ax.set_ylabel("Raw-score GT epistemic variance")
    ax.set_title(f"Raw diagnostic at $h = 1/{int(round(1.0 / h))}$ (eval n = {eval_n:,})")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, f"experiment2_rawdiag_{tag}.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"saved {out}")



def run_one_h(D, h):
    tag = h_tag(h)
    n_max = int(D["n_max"])
    n_seeds = int(D["n_seeds"])
    cal_n = min(CAL_N_REQUESTED, n_max)
    eval_n = min(EVAL_N_REQUESTED, n_max)
    n_repeats = min(N_REPEATS_REQUESTED, n_seeds)
    s_h_scale = float(D["s_h_scale"]) if "s_h_scale" in D else 1.0
    h_s = s_h_scale * h

    gt = load_gt_surface(D, h)
    gt_cache = make_gt_transport_cache(gt)
    ce2_before = float(D[f"ce2_gt_{tag}"])

    per_seed = []
    scatter_model = None
    for seed in range(n_repeats):
        print(
            f"fitting {tag}: calibration seed {seed + 1}/{n_repeats} "
            f"with fixed degree {FIXED_DEGREE} and target mode {TARGET_MODE}"
        )
        m_cal, s_cal, r1_cal, r2_cal = get_split_targets(D, seed, CAL_SPLIT, h, n=cal_n, target_mode=TARGET_MODE)
        model = fit_erm_polynomial_model(m_cal, s_cal, r1_cal, r2_cal, h)
        gt_after_surface = transport_ground_truth_surface(model, gt_cache)
        per_seed.append({
            "seed": seed,
            "degree": model["degree"],
            "alpha": model["alpha"],
            "ce2_after_gt": gt_after_surface["ce2"],
        })
        if seed == SCATTER_SEED:
            scatter_model = model

    if scatter_model is None:
        m_cal, s_cal, r1_cal, r2_cal = get_split_targets(D, SCATTER_SEED, CAL_SPLIT, h, n=cal_n, target_mode=TARGET_MODE)
        scatter_model = fit_erm_polynomial_model(m_cal, s_cal, r1_cal, r2_cal, h)

    m_eval, s_eval, _, _ = get_split_targets(D, SCATTER_SEED, EVAL_SPLIT, h, n=eval_n, target_mode=TARGET_MODE)
    before_epi_eval = s_eval.astype(np.float64, copy=False)
    pointwise_gt_epi_eval, _, _ = interpolate_gt_epistemic(gt, m_eval, s_eval)
    e1_hat_eval, e2_hat_eval = predict_eta(scatter_model, m_eval, s_eval)
    corr_epi_eval = corrected_epistemic_from_scores(e1_hat_eval, e2_hat_eval)

    gt_after_surface = transport_ground_truth_surface(scatter_model, gt_cache)
    transported_gt_epi_eval, _, _ = interpolate_gt_epistemic(gt_after_surface, e1_hat_eval, corr_epi_eval)

    raw_score_diag = None
    raw_before_epi = None
    raw_gt_epi_eval = None
    signal_diag = None
    try:
        raw_gt = load_raw_gt_surface(D)
        m_raw_eval, s_raw_eval, _, _ = get_raw_split(D, SCATTER_SEED, EVAL_SPLIT, n=eval_n)
        raw_gt_epi_eval, _, _ = interpolate_gt_epistemic(raw_gt, m_raw_eval, s_raw_eval)
        raw_before_epi = s_raw_eval.astype(np.float64, copy=False)
        raw_score_diag = {
            "corr": safe_corr(raw_before_epi, raw_gt_epi_eval),
            "spearman": safe_spearman(raw_before_epi, raw_gt_epi_eval),
            "raw_summary": summarise_array(raw_before_epi),
            "gt_summary": summarise_array(raw_gt_epi_eval),
        }
        signal_diag = signal_scale_diagnostic(raw_before_epi, h_s)
    except KeyError:
        pass

    ce2_after_all = np.array([r["ce2_after_gt"] for r in per_seed], dtype=np.float64)
    diag = make_main_diagnostics(
        before_epi_eval,
        pointwise_gt_epi_eval,
        corr_epi_eval,
        transported_gt_epi_eval,
    )
    dataset_meta = {
        "ensemble_size": int(D["ensemble_size"]) if "ensemble_size" in D else -1,
        "ensemble_train_n": int(D["ensemble_train_n"]) if "ensemble_train_n" in D else -1,
        "ensemble_member_data_mode": str(D["ensemble_member_data_mode"]) if "ensemble_member_data_mode" in D else "unknown",
        "ensemble_epochs": int(D["ensemble_epochs"]) if "ensemble_epochs" in D else -1,
        "ensemble_lr": float(D["ensemble_lr"]) if "ensemble_lr" in D else float("nan"),
        "ensemble_hidden": int(D["ensemble_hidden"]) if "ensemble_hidden" in D else -1,
        "ensemble_batch": int(D["ensemble_batch"]) if "ensemble_batch" in D else -1,
        "s_h_scale": float(D["s_h_scale"]) if "s_h_scale" in D else float("nan"),
        "exp2_cal_snapshots": int(D["exp2_cal_snapshots"]) if "exp2_cal_snapshots" in D else 2,
        "target_mode": TARGET_MODE,
    }
    print_diagnostics(
        h,
        ce2_before,
        ce2_after_all,
        diag,
        per_seed,
        raw_score_diag=raw_score_diag,
        signal_diag=signal_diag,
        dataset_meta=dataset_meta,
    )
    plot_main_figure(
        h,
        ce2_before,
        ce2_after_all,
        before_epi_eval,
        pointwise_gt_epi_eval,
        corr_epi_eval,
        diag,
        cal_n,
        eval_n,
    )
    if SAVE_RANK_DIAGNOSTIC:
        plot_rank_diagnostic(h, diag)
    if SAVE_RAW_DIAGNOSTIC and raw_before_epi is not None and raw_gt_epi_eval is not None:
        plot_raw_diagnostic(h, raw_before_epi, raw_gt_epi_eval, eval_n)

    out = os.path.join(RESULTS_DIR, f"experiment2_{tag}.npz")
    save_dict = dict(
        h=h,
        cal_split=CAL_SPLIT,
        eval_split=EVAL_SPLIT,
        cal_n=cal_n,
        eval_n=eval_n,
        scatter_seed=SCATTER_SEED,
        ce2_before_gt=ce2_before,
        ce2_after_gt=ce2_after_all,
        degrees=np.array([r["degree"] for r in per_seed], dtype=np.int64),
        fixed_degree=np.array(FIXED_DEGREE, dtype=np.int64),
        alphas=np.array([r["alpha"] for r in per_seed], dtype=np.float64),
        before_epistemic_eval=before_epi_eval,
        pointwise_gt_epistemic_eval=pointwise_gt_epi_eval,
        corrected_epistemic_eval=corr_epi_eval,
        transported_gt_epistemic_eval=transported_gt_epi_eval,
        corrected_eta1_eval=e1_hat_eval,
        corrected_eta2_eval=e2_hat_eval,
        thresholds=diag["thresholds"],
        frac_before_above=diag["frac_before_above"],
        frac_pointwise_gt_above=diag["frac_pointwise_gt_above"],
        frac_after_above=diag["frac_after_above"],
        frac_transported_gt_above=diag["frac_transported_gt_above"],
        before_vs_pointwise_gt_corr=diag["before_vs_pointwise_gt_corr"],
        after_vs_pointwise_gt_corr=diag["after_vs_pointwise_gt_corr"],
        after_vs_transported_gt_corr=diag["after_vs_transported_gt_corr"],
        before_spearman=diag["before_rank"]["spearman"],
        after_spearman=diag["after_rank"]["spearman"],
        before_kendall=diag["before_rank"]["kendall"],
        after_kendall=diag["after_rank"]["kendall"],
        topk_fracs=diag["before_rank"]["fracs"],
        before_topk_overlap=diag["before_rank"]["overlap"],
        after_topk_overlap=diag["after_rank"]["overlap"],
        before_topk_oracle_ratio=diag["before_rank"]["oracle_ratio"],
        after_topk_oracle_ratio=diag["after_rank"]["oracle_ratio"],
        before_topk_lift=diag["before_rank"]["lift"],
        after_topk_lift=diag["after_rank"]["lift"],
        before_bins_x=diag["before_bins"]["x"],
        before_bins_y=diag["before_bins"]["y"],
        after_bins_x=diag["after_bins"]["x"],
        after_bins_y=diag["after_bins"]["y"],
        before_summary=np.array([diag["before_summary"]], dtype=object),
        pointwise_gt_summary=np.array([diag["pointwise_gt_summary"]], dtype=object),
        after_summary=np.array([diag["after_summary"]], dtype=object),
        transported_gt_summary=np.array([diag["transported_gt_summary"]], dtype=object),
        s_h_scale=np.array(s_h_scale, dtype=np.float64),
        h_s=np.array(h_s, dtype=np.float64),
        erm_alpha=np.array(ERM_ALPHA, dtype=np.float64),
        ensemble_size=np.array(int(D["ensemble_size"]) if "ensemble_size" in D else -1, dtype=np.int64),
        ensemble_train_n=np.array(int(D["ensemble_train_n"]) if "ensemble_train_n" in D else -1, dtype=np.int64),
        ensemble_epochs=np.array(int(D["ensemble_epochs"]) if "ensemble_epochs" in D else -1, dtype=np.int64),
        ensemble_hidden=np.array(int(D["ensemble_hidden"]) if "ensemble_hidden" in D else -1, dtype=np.int64),
        ensemble_batch=np.array(int(D["ensemble_batch"]) if "ensemble_batch" in D else -1, dtype=np.int64),
        ensemble_lr=np.array(float(D["ensemble_lr"]) if "ensemble_lr" in D else np.nan, dtype=np.float64),
        ensemble_member_data_mode=np.array(str(D["ensemble_member_data_mode"]) if "ensemble_member_data_mode" in D else "unknown"),
        exp2_cal_snapshots=np.array(int(D["exp2_cal_snapshots"]) if "exp2_cal_snapshots" in D else 2, dtype=np.int64),
        exp2_target_mode=np.array(TARGET_MODE),
        project_feasible_moments=np.array(bool(PROJECT_FEASIBLE_MOMENTS)),
        rank_n_bins=np.array(RANK_N_BINS, dtype=np.int64),
    )
    if raw_score_diag is not None:
        save_dict.update(
            raw_score_raw_vs_gt_corr=np.array(raw_score_diag["corr"], dtype=np.float64),
            raw_score_raw_vs_gt_spearman=np.array(raw_score_diag["spearman"], dtype=np.float64),
            raw_score_raw_summary=np.array([raw_score_diag["raw_summary"]], dtype=object),
            raw_score_gt_summary=np.array([raw_score_diag["gt_summary"]], dtype=object),
            raw_score_before_epistemic_eval=raw_before_epi,
            raw_score_gt_epistemic_eval=raw_gt_epi_eval,
        )
    if signal_diag is not None:
        save_dict.update(
            raw_signal_scale=np.array([signal_diag], dtype=object),
        )
    np.savez(out, **save_dict)
    print(f"saved {out}")



def main():
    D = load_dataset()
    for h in H_LIST:
        run_one_h(D, h)


if __name__ == "__main__":
    main()
