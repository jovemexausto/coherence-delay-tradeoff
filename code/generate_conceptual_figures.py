from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def conceptual_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
        }
    )


def generate_two_clocks(output_path: Path, csv_path: Path) -> None:
    # Normalize the theorem exactly by n* and E_min.
    # If x = n / n*, then
    # variance / E_min = (2/3) x^{-1/2}
    # staleness / E_min = (1/3) x
    # total / E_min = (2/3) x^{-1/2} + (1/3) x
    x = np.linspace(0.35, 3.0, 500)
    variance = (2.0 / 3.0) * x ** (-0.5)
    staleness = (1.0 / 3.0) * x
    total = variance + staleness
    x_star = 1.0
    x_cross = 2.0 ** (2.0 / 3.0)
    y_star = 1.0

    rows = [
        {
            "normalized_horizon": round(float(xi), 6),
            "variance_cost_over_Emin": round(float(vi), 6),
            "staleness_cost_over_Emin": round(float(si), 6),
            "total_error_over_Emin": round(float(ti), 6),
        }
        for xi, vi, si, ti in zip(x, variance, staleness, total, strict=True)
    ]
    write_csv(csv_path, rows)

    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    ax.axvspan(x_star, 2.2, color="#f6ddd2", alpha=0.75, linewidth=0, zorder=0)
    ax.plot(x, variance, color="#315caf", linewidth=2.3)
    ax.plot(x, staleness, color="#b23a3a", linewidth=2.3)
    ax.plot(x, total, color="black", linewidth=2.6)
    ax.axvline(x_star, color="black", linestyle="--", linewidth=1.0)
    ax.axvline(x_cross, color="#666666", linestyle="--", linewidth=1.0)
    ax.scatter([x_star], [y_star], color="black", s=22, zorder=3)

    ax.set_xlim(0.35, 3.0)
    ax.set_ylim(0.0, 1.55)
    ax.set_xlabel(r"Normalized horizon $n / n^*$")
    ax.set_ylabel(r"Normalized tracking error / $E_{\min}$")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.18, linewidth=0.6)

    handles = [
        plt.Line2D([0], [0], color="#315caf", linewidth=2.8),
        plt.Line2D([0], [0], color="#b23a3a", linewidth=2.8),
        plt.Line2D([0], [0], color="black", linewidth=3.0),
    ]
    labels = [
        r"Variance $C_K n^{-1/2}$",
        r"Staleness $\frac{1}{2}\zeta n$",
        r"Total $\mathcal{E}(n)$",
    ]
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=3,
        frameon=False,
        handlelength=2.1,
        handletextpad=0.6,
        columnspacing=1.8,
        fontsize=9,
    )

    ax.annotate(
        r"optimal $n^*$",
        xy=(x_star, y_star),
        xytext=(0.82, 1.17),
        fontsize=8,
        ha="right",
        arrowprops={"arrowstyle": "-", "linewidth": 0.9, "color": "black"},
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.95,
        },
    )
    ax.annotate(
        "costs cross",
        xy=(x_cross, float((2.0 / 3.0) * x_cross ** (-0.5))),
        xytext=(1.82, 0.9),
        fontsize=8,
        ha="left",
        arrowprops={"arrowstyle": "-", "linewidth": 0.9, "color": "#666666"},
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.95,
        },
        color="#444444",
    )
    ax.text(
        1.58,
        1.42,
        "detector-silent staleness",
        ha="center",
        fontsize=8,
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.9,
        },
    )

    fig.subplots_adjust(left=0.12, right=0.98, top=0.82, bottom=0.18)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def generate_lower_bound(output_path: Path, csv_path: Path) -> None:
    # Exact witness shape up to scale: mu^{\pm,h}(j) = \pm (1 - j / h)_+
    m = 6.0
    h = 3.0
    age = np.linspace(0.0, m, 500)
    mu_plus = np.maximum(1.0 - age / h, 0.0)
    mu_minus = -mu_plus

    rows = [
        {
            "sample_age": round(float(a), 6),
            "mu_plus": round(float(p), 6),
            "mu_minus": round(float(n), 6),
        }
        for a, p, n in zip(age, mu_plus, mu_minus, strict=True)
    ]
    write_csv(csv_path, rows)

    fig, ax = plt.subplots(figsize=(6.6, 3.5))
    ax.axvspan(0.0, h, color="#eef4ff", linewidth=0, zorder=0)
    ax.axvspan(h, m, color="#f3f3f3", linewidth=0, zorder=0)
    ax.fill_between(age, mu_minus, mu_plus, where=age <= h, color="#dbe7ff", zorder=1)
    ax.plot(
        age, mu_plus, color="#315caf", linewidth=2.8, solid_capstyle="round", zorder=3
    )
    ax.plot(
        age, mu_minus, color="#b23a3a", linewidth=2.8, solid_capstyle="round", zorder=3
    )
    ax.axhline(0.0, color="#8a8a8a", linewidth=0.9, zorder=2)
    ax.axvline(h, color="#666666", linestyle="--", linewidth=1.0, zorder=2)

    ax.set_xlim(-0.1, m + 0.1)
    ax.set_ylim(-1.25, 1.40)
    ax.set_xlabel("Sample age inside retained memory")
    ax.set_ylabel("Mean shift")
    ax.set_xticks([0, h, m], ["present", "h", "m"])
    ax.set_yticks([])
    ax.tick_params(labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)

    label_box = {
        "boxstyle": "round,pad=0.14",
        "facecolor": "white",
        "edgecolor": "none",
        "alpha": 0.96,
    }
    ax.text(
        0.34,
        0.88,
        r"$\mu_t^{+,h}$",
        color="#315caf",
        fontsize=11,
        weight="bold",
        bbox=label_box,
        zorder=5,
    )
    ax.text(
        0.34,
        -1.08,
        r"$\mu_t^{-,h}$",
        color="#b23a3a",
        fontsize=11,
        weight="bold",
        bbox=label_box,
        zorder=5,
    )
    ax.text(
        1.5, 1.20, "short differing segment", ha="center", fontsize=9, color="#2f4f8f"
    )
    ax.text(4.5, 0.1, "shared history", ha="center", fontsize=9, color="#666666")

    ax.annotate(
        "endpoint gap",
        xy=(0.32, 0.0),
        xytext=(1.02, 0.30),
        arrowprops={"arrowstyle": "-|>", "linewidth": 0.9, "color": "black"},
        fontsize=9,
        bbox={
            "boxstyle": "round,pad=0.16",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.95,
        },
    )
    ax.annotate(
        "",
        xy=(0.32, -1.0),
        xytext=(0.32, 1.0),
        arrowprops={"arrowstyle": "<->", "linewidth": 1.0, "color": "black"},
    )

    bracket_y = -1.12
    ax.annotate(
        "",
        xy=(0.0, bracket_y),
        xytext=(h, bracket_y),
        arrowprops={"arrowstyle": "<->", "linewidth": 0.9, "color": "#444444"},
    )
    ax.text(h / 2.0, bracket_y - 0.14, "width h", ha="center", va="top", fontsize=9)
    ax.annotate(
        "",
        xy=(h, bracket_y),
        xytext=(m, bracket_y),
        arrowprops={"arrowstyle": "<->", "linewidth": 0.9, "color": "#777777"},
    )
    ax.text(
        (h + m) / 2.0,
        bracket_y - 0.14,
        "identical over m-h",
        ha="center",
        va="top",
        fontsize=9,
        color="#666666",
    )

    fig.subplots_adjust(left=0.11, right=0.99, top=0.92, bottom=0.22)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate conceptual Paper 1 figures.")
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("artifacts/figures/conceptual"),
    )
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=Path("artifacts/csv/conceptual"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    conceptual_style()
    generate_two_clocks(
        args.figures_dir / "fig_two_clocks_of_drift.pdf",
        args.csv_dir / "two_clocks_of_drift.csv",
    )
    generate_lower_bound(
        args.figures_dir / "fig_lower_bound_witness.pdf",
        args.csv_dir / "lower_bound_witness.csv",
    )


if __name__ == "__main__":
    main()
