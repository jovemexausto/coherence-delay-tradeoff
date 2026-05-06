from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "experiments" / "artifacts"
OUTPUT = ROOT / "figures" / "fig_sensitivity_summary.pdf"


def _style_axis(axis) -> None:
    axis.grid(alpha=0.2, linewidth=0.5)
    axis.tick_params(labelsize=8)


def main() -> None:
    sinkhorn = pd.read_csv(ARTIFACTS / "gaussian" / "gaussian_sinkhorn_runtime.csv")
    kuairand_lambda = pd.read_csv(
        ARTIFACTS / "kuairand" / "kuairand_followup_lambda.csv"
    )
    kuairand_e0 = pd.read_csv(ARTIFACTS / "kuairand" / "kuairand_followup_e0.csv")
    kuairand_proxy = pd.read_csv(ARTIFACTS / "kuairand" / "kuairand_followup_proxy.csv")
    kuairand_threshold = pd.read_csv(
        ARTIFACTS / "kuairand" / "kuairand_followup_threshold.csv"
    )
    particle = pd.read_csv(ARTIFACTS / "particle" / "particle_masking_grid_summary.csv")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.6), constrained_layout=True)
    flat_axes = axes.ravel()

    # Panel A: epsilon trade-off
    ax = flat_axes[0]
    eps_rows = sinkhorn[
        (sinkhorn["window_size"] == 100) & (sinkhorn["dimension"].isin([8, 256]))
    ]
    ax2 = ax.twinx()
    colors = {8: "tab:blue", 256: "tab:orange"}
    for dimension in (8, 256):
        rows = eps_rows[eps_rows["dimension"] == dimension].sort_values("epsilon")
        ax.plot(
            rows["epsilon"],
            rows["mean_runtime_ms"],
            marker="o",
            linewidth=1.6,
            color=colors[dimension],
            label=f"runtime d={dimension}",
        )
        ax2.plot(
            rows["epsilon"],
            rows["mean_abs_bias"],
            marker="s",
            linewidth=1.4,
            linestyle="--",
            color=colors[dimension],
            label=f"bias d={dimension}",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax2.set_yscale("log")
    ax.set_xlabel(r"Regularization $\varepsilon$")
    ax.set_ylabel("Runtime (ms)")
    ax2.set_ylabel("Abs. bias")
    ax.set_title("A. Sinkhorn epsilon trade-off")
    _style_axis(ax)
    ax2.tick_params(labelsize=8)
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(
        lines, [line.get_label() for line in lines], loc="upper right", fontsize=7
    )

    # Panel B: lambda sensitivity
    ax = flat_axes[1]
    rows = kuairand_lambda[kuairand_lambda["phase"] == "bubble_detection"].sort_values(
        "lambda"
    )
    ax2 = ax.twinx()
    ax.plot(rows["lambda"], rows["rate"], marker="o", linewidth=1.8, color="tab:red")
    ax.fill_between(
        rows["lambda"], rows["ci_low"], rows["ci_high"], color="tab:red", alpha=0.15
    )
    ax2.plot(
        rows["lambda"],
        rows["healthy_fp_per_user"],
        marker="s",
        linewidth=1.4,
        linestyle="--",
        color="0.35",
    )
    ax.axvspan(2.0, 3.0, color="tab:green", alpha=0.08)
    ax.axvline(3.0, color="tab:red", linestyle=":", linewidth=1.0)
    ax.set_xlabel(r"Effort penalty $\lambda$")
    ax.set_ylabel("Bubble detection rate")
    ax2.set_ylabel("Healthy FP / user")
    ax.set_ylim(0.4, 0.82)
    ax.set_title("B. KuaiRand lambda sensitivity")
    _style_axis(ax)
    ax2.tick_params(labelsize=8)

    # Panel C: E0 sensitivity
    ax = flat_axes[2]
    rows = kuairand_e0[kuairand_e0["phase"] == "bubble_detection"].sort_values(
        "e0_scale"
    )
    ax2 = ax.twinx()
    ax.plot(
        rows["e0_scale"], rows["rate"], marker="o", linewidth=1.8, color="tab:purple"
    )
    ax.fill_between(
        rows["e0_scale"],
        rows["ci_low"],
        rows["ci_high"],
        color="tab:purple",
        alpha=0.15,
    )
    ax2.plot(
        rows["e0_scale"],
        rows["healthy_fp_per_user"],
        marker="s",
        linewidth=1.4,
        linestyle="--",
        color="0.35",
    )
    ax.axvline(1.0, color="tab:purple", linestyle=":", linewidth=1.0)
    ax.set_xlabel(r"Reference effort scale $E_0$ multiplier")
    ax.set_ylabel("Bubble detection rate")
    ax2.set_ylabel("Healthy FP / user")
    ax.set_ylim(0.45, 0.9)
    ax.set_title("C. KuaiRand E0 sensitivity")
    _style_axis(ax)
    ax2.tick_params(labelsize=8)

    # Panel D: proxy sensitivity
    ax = flat_axes[3]
    rows = kuairand_proxy[kuairand_proxy["phase"] == "bubble_detection"].copy()
    rows["proxy"] = rows["proxy"].str.upper()
    rows = rows.sort_values("rate", ascending=False)
    x = range(len(rows))
    rate_err_low = rows["rate"] - rows["ci_low"]
    rate_err_high = rows["ci_high"] - rows["rate"]
    ax.bar(x, rows["rate"], color=["tab:blue", "tab:orange", "tab:green"], alpha=0.8)
    ax.errorbar(
        x,
        rows["rate"],
        yerr=[rate_err_low, rate_err_high],
        fmt="none",
        ecolor="black",
        capsize=3,
        linewidth=1.0,
    )
    ax2 = ax.twinx()
    ax2.plot(
        x,
        rows["healthy_fp_per_user"],
        marker="s",
        linewidth=1.3,
        linestyle="--",
        color="0.35",
    )
    ax.set_xticks(list(x), rows["proxy"])
    ax.set_ylabel("Bubble detection rate")
    ax2.set_ylabel("Healthy FP / user")
    ax.set_ylim(0.45, 0.8)
    ax.set_title("D. Logged effort proxy swap")
    _style_axis(ax)
    ax2.tick_params(labelsize=8)

    # Panel E: threshold sensitivity
    ax = flat_axes[4]
    rows = kuairand_threshold[
        kuairand_threshold["phase"] == "bubble_detection"
    ].sort_values("threshold_quantile")
    detector_colors = {"CI": "0.45", "CI^E": "tab:red", "CI^E-EWMA": "tab:blue"}
    for detector in ("CI", "CI^E", "CI^E-EWMA"):
        det_rows = rows[rows["detector"] == detector]
        ax.plot(
            det_rows["threshold_quantile"],
            det_rows["rate"],
            marker="o",
            linewidth=1.6,
            color=detector_colors[detector],
            label=detector,
        )
    ax.axvline(0.2, color="tab:red", linestyle=":", linewidth=1.0)
    ax.set_xlabel("Healthy-window threshold quantile")
    ax.set_ylabel("Bubble detection rate")
    ax.set_ylim(0.2, 0.85)
    ax.set_title("E. Threshold sensitivity")
    _style_axis(ax)
    ax.legend(loc="upper left", fontsize=7)

    # Panel F: particle masking gap
    ax = flat_axes[5]
    rows = particle[particle["regime"] == "coercive"].copy()
    focus_influences = [0.1, 0.3, 0.5, 0.9]
    palette = {0.1: "tab:blue", 0.3: "tab:red", 0.5: "tab:green", 0.9: "tab:purple"}
    for influence in focus_influences:
        det_rows = rows[rows["influence"] == influence].sort_values("lambda")
        ax.plot(
            det_rows["lambda"],
            det_rows["mean_masking_gap"],
            marker="o",
            linewidth=1.6,
            color=palette[influence],
            label=rf"$\alpha={influence:.1f}$",
        )
    ax.axvline(3.0, color="tab:red", linestyle=":", linewidth=1.0)
    ax.set_xlabel(r"Effort penalty $\lambda$")
    ax.set_ylabel("Mean masking gap")
    ax.set_ylim(0.0, 0.6)
    ax.set_title("F. Particle masking-gap sensitivity")
    _style_axis(ax)
    ax.legend(loc="upper left", fontsize=7, ncol=2)

    fig.suptitle(
        "Consolidated sensitivity analysis from existing artifacts", fontsize=14
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT)
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=180)
    plt.close(fig)
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
