from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def rolling_mean_estimate(x: np.ndarray, n: int, t: int) -> float:
    w = x[max(0, t - n) : t]
    return float(np.mean(w)) if len(w) else 0.0


def make_series(
    T: int, zeta: float, sigma: float, seed: int, drift: bool
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = np.arange(T, dtype=float)
    mu = zeta * t if drift else np.zeros(T, dtype=float)
    x = mu + rng.normal(0.0, sigma, size=T)
    return mu, x


def mae_curve(mu: np.ndarray, x: np.ndarray, ns: np.ndarray, warmup: int) -> np.ndarray:
    maes = []
    for n in ns:
        errs = []
        for t in range(warmup, len(mu)):
            est = rolling_mean_estimate(x, int(n), t)
            errs.append(abs(est - mu[t]))
        maes.append(float(np.mean(errs)))
    return np.array(maes, dtype=float)


def main() -> None:
    sigma = 1.0
    T = 12000
    warmup = 1000
    ns = np.array([5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 400, 600])
    zetas = np.array([0.0008, 0.0012, 0.0018, 0.0027, 0.0040, 0.0060])

    # Stationary vs drifting representatives
    mu_stat, x_stat = make_series(T, 0.0, sigma, seed=7, drift=False)
    mu_drift, x_drift = make_series(T, 0.002, sigma, seed=11, drift=True)
    maes_stat = mae_curve(mu_stat, x_stat, ns, warmup)
    maes_drift = mae_curve(mu_drift, x_drift, ns, warmup)

    n_stat = int(ns[np.argmin(maes_stat)])
    n_drift = int(ns[np.argmin(maes_drift)])

    # Scaling sweep
    best_ns = []
    best_maes = []
    curves = []
    for i, zeta in enumerate(zetas):
        mu, x = make_series(T, zeta, sigma, seed=100 + i, drift=True)
        maes = mae_curve(mu, x, ns, warmup)
        curves.append(maes)
        idx = int(np.argmin(maes))
        best_ns.append(float(ns[idx]))
        best_maes.append(float(maes[idx]))

    best_ns = np.array(best_ns)
    best_maes = np.array(best_maes)
    slope_n, intercept_n = np.polyfit(np.log(zetas), np.log(best_ns), 1)
    slope_e, intercept_e = np.polyfit(np.log(zetas), np.log(best_maes), 1)

    # Visual style
    bg = "#0b0f14"
    fg = "#d7e1ea"
    gridc = "#233040"
    accent1 = "#4cc9f0"
    accent2 = "#f9844a"
    accent3 = "#90be6d"
    accent4 = "#c77dff"

    fig = plt.figure(figsize=(14, 10), facecolor=bg)
    gs = fig.add_gridspec(2, 2, hspace=0.26, wspace=0.18)

    def stylize(ax):
        ax.set_facecolor("#111922")
        for sp in ax.spines.values():
            sp.set_color(gridc)
        ax.tick_params(colors=fg, labelsize=9)
        ax.xaxis.label.set_color(fg)
        ax.yaxis.label.set_color(fg)
        ax.title.set_color(fg)
        ax.grid(True, color=gridc, alpha=0.45, lw=0.7)

    # Panel A: stationary regime vs one drifting regime
    ax = fig.add_subplot(gs[0, 0])
    stylize(ax)
    ax.loglog(ns, maes_stat, "o-", color=accent1, lw=2.5, ms=6, label="stationary")
    ax.loglog(ns, maes_drift, "o-", color=accent2, lw=2.5, ms=6, label="drift")
    ax.axvline(n_stat, color=accent1, ls="--", lw=1.2, alpha=0.8)
    ax.axvline(n_drift, color=accent2, ls="--", lw=1.2, alpha=0.8)
    ax.legend(
        fontsize=8.5,
        facecolor="#1a1f27",
        labelcolor="white",
        edgecolor=gridc,
        loc="upper right",
    )
    ax.set_title("Stationary vs drifting mean", pad=10)
    ax.set_xlabel("window n")
    ax.set_ylabel("rolling MAE")
    ax.text(
        0.03,
        0.94,
        "A",
        transform=ax.transAxes,
        fontsize=16,
        fontweight="bold",
        color=fg,
    )

    # Panel B: a few drifting curves
    ax = fig.add_subplot(gs[0, 1])
    stylize(ax)
    drift_idxs = [0, 2, 4]
    drift_palette = [accent2, accent3, accent4]
    for k, idx in enumerate(drift_idxs):
        ax.loglog(
            ns,
            curves[idx],
            "o-",
            color=drift_palette[k],
            lw=2.0,
            ms=5,
            label=f"ζ={zetas[idx]:.4f}",
        )
    ax.legend(
        fontsize=8,
        facecolor="#1a1f27",
        labelcolor="white",
        edgecolor=gridc,
        loc="upper right",
    )
    ax.set_title("Drift creates a finite horizon", pad=10)
    ax.set_xlabel("window n")
    ax.set_ylabel("rolling MAE")
    ax.text(
        0.03,
        0.94,
        "B",
        transform=ax.transAxes,
        fontsize=16,
        fontweight="bold",
        color=fg,
    )

    # Panel C: scaling law for n*
    ax = fig.add_subplot(gs[1, 0])
    stylize(ax)
    ax.loglog(zetas, best_ns, "o-", color=accent3, lw=2.5, ms=6, label="empirical n*")
    ax.loglog(
        zetas,
        np.exp(intercept_n) * zetas**slope_n,
        "--",
        color=accent3,
        lw=1.3,
        label=f"fit slope={slope_n:.3f}",
    )
    theory_n = (zetas / zetas[0]) ** (-2 / 3) * best_ns[0]
    ax.loglog(zetas, theory_n, ":", color=fg, lw=1.6, label="theory -2/3")
    ax.set_xlabel("drift rate ζ")
    ax.set_ylabel("optimal horizon n*")
    ax.set_title("Optimal horizon scales as a power law", pad=10)
    ax.text(
        0.03,
        0.94,
        "C",
        transform=ax.transAxes,
        fontsize=16,
        fontweight="bold",
        color=fg,
    )
    ax.legend(
        fontsize=8,
        facecolor="#1a1f27",
        labelcolor="white",
        edgecolor=gridc,
        loc="lower left",
    )

    # Panel D: scaling law for Emin
    ax = fig.add_subplot(gs[1, 1])
    stylize(ax)
    ax.loglog(
        zetas, best_maes, "s-", color=accent4, lw=2.5, ms=6, label="empirical Emin"
    )
    ax.loglog(
        zetas,
        np.exp(intercept_e) * zetas**slope_e,
        "--",
        color=accent4,
        lw=1.3,
        label=f"fit slope={slope_e:.3f}",
    )
    theory_e = (zetas / zetas[0]) ** (1 / 3) * best_maes[0]
    ax.loglog(zetas, theory_e, ":", color=fg, lw=1.6, label="theory +1/3")
    ax.set_xlabel("drift rate ζ")
    ax.set_ylabel("minimum error")
    ax.set_title("Minimum error scales as a power law", pad=10)
    ax.text(
        0.03,
        0.94,
        "D",
        transform=ax.transAxes,
        fontsize=16,
        fontweight="bold",
        color=fg,
    )
    ax.legend(
        fontsize=8,
        facecolor="#1a1f27",
        labelcolor="white",
        edgecolor=gridc,
        loc="lower left",
    )

    fig.suptitle(
        "LLN in stationary time, finite horizon under drift",
        color=fg,
        fontsize=15,
        fontweight="bold",
    )

    out_png = "figures/lln_temporal_validity_story.png"
    out_pdf = "figures/lln_temporal_validity_story.pdf"
    plt.savefig(out_png, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.savefig(out_pdf, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f"saved {out_png}")
    print(f"saved {out_pdf}")
    print(f"stationary best n: {n_stat}, drifting best n: {n_drift}")
    print(f"drift scaling slope n*: {slope_n:.3f} (theory -2/3)")
    print(f"drift scaling slope Emin: {slope_e:.3f} (theory +1/3)")


if __name__ == "__main__":
    main()
