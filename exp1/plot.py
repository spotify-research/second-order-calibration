"""
Plot the log-log convergence curves for Experiment 1.
One PDF per perturbation bandwidth h.

For each n, we plot the mean of the per-seed errors and a 90% confidence
interval for that mean using a Student-t critical value.
Empirical rates are computed by least-squares fitting log10(mean error)
against log10(n) over the swept grid.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from common import N_GRID, RESULTS_DIR, H_LIST, h_tag

plt.rcParams.update({
    "text.usetex": True,
    "font.size": 34,
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman"],
    "axes.labelsize": 35,
    "axes.titlesize": 35,
    "xtick.labelsize": 33,
    "ytick.labelsize": 33,
})

def load(name, h):
    p = os.path.join(RESULTS_DIR, f"{name}_{h_tag(h)}.npz")
    arr = np.load(p, allow_pickle=True)["results"]
    return list(arr)


def aggregate(results, key):
    """Return (ns, mean, lo, hi) where [lo, hi] is a 90% confidence
    interval for the mean across seeds (t-distribution)."""
    from scipy.stats import t as tdist
    by_n = {}
    for r in results:
        by_n.setdefault(r["n"], []).append(r[key])
    ns = sorted(by_n.keys())
    mean, lo, hi = [], [], []
    for n in ns:
        vals = np.array(by_n[n], dtype=float)
        k = len(vals)
        mu = vals.mean()
        se = vals.std(ddof=1) / np.sqrt(k) if k > 1 else 0.0
        t_crit = tdist.ppf(0.95, df=max(k - 1, 1))
        mean.append(mu)
        lo.append(mu - t_crit * se)
        hi.append(mu + t_crit * se)
    return np.array(ns), np.array(mean), np.array(lo), np.array(hi)


def empirical_slope(ns, vals):
    x = np.log10(ns); y = np.log10(np.maximum(vals, 1e-12))
    a, _ = np.polyfit(x, y, 1)
    return a


def per_seed_slope(results, key):
    """Fit log-log slope per seed, then average — estimates E[slope_i]."""
    by_seed = {}
    for r in results:
        by_seed.setdefault(r["seed"], []).append((r["n"], r[key]))
    slopes = []
    for pts in by_seed.values():
        pts.sort()
        ns = np.array([p[0] for p in pts])
        vs = np.array([p[1] for p in pts])
        a, _ = np.polyfit(np.log10(ns), np.log10(np.maximum(vs, 1e-12)), 1)
        slopes.append(a)
    return float(np.mean(slopes))


LABEL_FS = 29
BBOX = dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.88)
ARROW = dict(arrowstyle="-", color="0.4", lw=0.8, shrinkA=0, shrinkB=4)

# Per-series inline-label placement.
# Each entry: (x_index along the 7-point N_GRID,
#              (dx, dy) offset in points from the anchor,
#              text alignment (ha, va)).
# Labels are staggered across x so they never overlap, and nudged into
# the gap between neighbouring curves.  An arrow connects each label to
# its anchor point on the line.
LABEL_PLACEMENT = {
    "Poly (CV)":      (2, ( 0,-50), ("center", "top")),
    "Bucketing":       (5, ( 0, 30), ("center", "bottom")),
    "Nadaraya-Watson": (4, ( 0, 30), ("center", "bottom")),
    "Poly (held-out val)":      (4, ( 0,-50), ("center", "top")),
}


def plot_one(h):
    bucketing = load("bucketing", h)
    nw = load("nadaraya_watson", h)
    poly = load("polynomial", h)
    ce2_gt = float(np.load(os.path.join(
        RESULTS_DIR, f"polynomial_{h_tag(h)}.npz"))["ce2_gt"])

    fig, ax = plt.subplots(figsize=(10, 7.5))
    # Wong colorblind-safe palette
    series = []
    n, m, lo, hi = aggregate(bucketing, "err")
    series.append(("Bucketing", n, m, lo, hi, "#E69F00", "s"))
    n, m, lo, hi = aggregate(nw, "err")
    series.append(("Nadaraya-Watson", n, m, lo, hi, "#009E73", "^"))
    n, m, lo, hi = aggregate(poly, "err_val")
    series.append(("Poly (held-out val)", n, m, lo, hi, "#0072B2", "o"))
    n, m, lo, hi = aggregate(poly, "err_erm")
    series.append(("Poly (CV)", n, m, lo, hi, "#CC79A7", "D"))

    print(f"\n--- h = {h:.5f} ({h_tag(h)})  GT CE2 = {ce2_gt:.4f} ---")
    print("Empirical slopes (log-log regression on the mean):")
    for label, ns, mu, lo_, hi_, color, marker in series:
        slope = empirical_slope(ns, mu)
        print(f"  {label:30s}  slope = {slope:+.3f}")
        ax.plot(ns, mu, color=color, marker=marker, linewidth=1.5,
                markersize=8, zorder=3)
        ax.fill_between(ns, np.maximum(lo_, 1e-12), hi_,
                        color=color, alpha=0.15)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$n$ (samples)")
    ax.set_ylabel(r"$|\widehat{\mathrm{CE}}_2 - \mathrm{CE}_2^{\mathrm{pert}}|$")
    ax.set_title(
        f"Experiment 1: convergence at $h = 1/{int(round(1.0/h))}$\n"
        f"(GT CE$_2$ = {ce2_gt:.3f}, mean $\\pm$ 90\\% CI)")
    ax.grid(True, which="both", alpha=0.3)

    # Inline labels — placed at staggered x-positions inside the axes
    for label, ns, mu, lo_, hi_, color, marker in series:
        slope = empirical_slope(ns, mu)
        idx, (dx, dy), (ha, va) = LABEL_PLACEMENT[label]
        ax.annotate(
            rf"\textbf{{{label}}} ($n^{{{slope:.2f}}}$)",
            xy=(ns[idx], mu[idx]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=LABEL_FS,
            color=color,
            ha=ha, va=va,
            bbox=BBOX,
            arrowprops=ARROW,
            zorder=5,
        )

    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, f"experiment1_{h_tag(h)}.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")


def main():
    for h in H_LIST:
        plot_one(h)


if __name__ == "__main__":
    main()