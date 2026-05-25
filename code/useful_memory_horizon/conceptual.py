from __future__ import annotations

import argparse
import csv
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from .useful_memory_region import (
    continuous_optimal_horizon,
    normalized_envelope_ratio,
    useful_memory_interval,
)


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


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
    a = 0.5
    cases = [
        {"label": r"$H=0.50$", "H": 0.5, "zeta": 0.7, "color": "#315caf"},
        {"label": r"$H=0.75$", "H": 0.75, "zeta": 0.6, "color": "#2f8f5b"},
        {"label": r"$H=1.00$", "H": 1.0, "zeta": 0.45, "color": "#b23a3a"},
    ]
    x = np.linspace(0.25, 4.0, 600)
    x_norm = np.linspace(0.35, 3.0, 600)

    rows: list[dict[str, Any]] = []
    for case in cases:
        H = float(case["H"])
        zeta = float(case["zeta"])
        n_star = continuous_optimal_horizon(1.0, a, 1.0, zeta, H)
        raw_variance = x ** (-a)
        raw_staleness = zeta * x**H
        raw_total = raw_variance + raw_staleness
        normalized = normalized_envelope_ratio(x_norm, a, H)
        lower, upper = useful_memory_interval(a, H, 0.12)
        for xi, vi, si, ti in zip(
            x, raw_variance, raw_staleness, raw_total, strict=True
        ):
            rows.append(
                {
                    "panel": "raw",
                    "H": H,
                    "zeta": zeta,
                    "n": round(float(xi), 6),
                    "n_over_n_star": round(float(xi / n_star), 6),
                    "variance_cost": round(float(vi), 6),
                    "staleness_cost": round(float(si), 6),
                    "total_cost": round(float(ti), 6),
                    "n_star": round(float(n_star), 6),
                    "useful_lower_over_n_star": round(float(lower), 6),
                    "useful_upper_over_n_star": round(float(upper), 6),
                }
            )
        for xn, yn in zip(x_norm, normalized, strict=True):
            rows.append(
                {
                    "panel": "normalized",
                    "H": H,
                    "zeta": zeta,
                    "n": round(float(xn * n_star), 6),
                    "n_over_n_star": round(float(xn), 6),
                    "variance_cost": round(float(xn ** (-a)), 6),
                    "staleness_cost": round(float(zeta * (xn * n_star) ** H), 6),
                    "total_cost": round(float(yn), 6),
                    "n_star": round(float(n_star), 6),
                    "useful_lower_over_n_star": round(float(lower), 6),
                    "useful_upper_over_n_star": round(float(upper), 6),
                }
            )

    write_csv(csv_path, rows)

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(9.5, 3.15))

    for case in cases:
        H = float(case["H"])
        zeta = float(case["zeta"])
        color = str(case["color"])
        n_star = continuous_optimal_horizon(1.0, a, 1.0, zeta, H)
        raw_variance = x ** (-a)
        raw_staleness = zeta * x**H
        raw_total = raw_variance + raw_staleness
        ax_left.plot(x, raw_total, color=color, linewidth=2.2, label=case["label"])
        ax_left.axvline(n_star, color=color, linestyle="--", linewidth=0.9, alpha=0.7)
        ax_right.plot(
            x_norm,
            normalized_envelope_ratio(x_norm, a, H),
            color=color,
            linewidth=2.3,
            label=case["label"],
        )

    ax_left.axhline(1.0, color="#666666", linestyle=":", linewidth=0.9)
    ax_left.set_xlabel(r"Memory length $n$")
    ax_left.set_ylabel(r"Envelope $\Phi(n)$")
    ax_left.set_title("Raw validity profiles")
    ax_left.spines["top"].set_visible(False)
    ax_left.spines["right"].set_visible(False)
    ax_left.grid(axis="y", alpha=0.18, linewidth=0.6)

    ax_right.axvspan(1 - 0.12, 1 + 0.12, color="#e7f2df", alpha=0.8, linewidth=0)
    ax_right.axvline(1.0, color="black", linestyle="--", linewidth=1.0)
    ax_right.set_xlabel(r"Normalized horizon $n / n^*$")
    ax_right.set_ylabel(r"Normalized profile $\Psi(n/n^*)$")
    ax_right.set_title("Normalized validity profile")
    ax_right.spines["top"].set_visible(False)
    ax_right.spines["right"].set_visible(False)
    ax_right.grid(axis="y", alpha=0.18, linewidth=0.6)

    fig.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=False,
        handlelength=2.0,
        handletextpad=0.5,
        columnspacing=1.4,
        fontsize=8,
    )

    fig.subplots_adjust(left=0.08, right=0.985, top=0.78, bottom=0.18, wspace=0.28)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def generate_lower_bound(output_path: Path, csv_path: Path) -> None:
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
    ax.fill_between(
        age,
        mu_minus,
        mu_plus,
        where=(age <= h).tolist(),
        color="#dbe7ff",
        zorder=1,
    )
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

    label_box: dict[str, object] = {
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


def _extract_float(setting: str, marker: str) -> float:
    match = re.search(rf"{re.escape(marker)}([0-9.]+)", setting)
    if match is None:
        raise ValueError(f"could not parse {marker!r} from {setting!r}")
    return float(match.group(1))


def generate_carrier_layers(
    output_path: Path, csv_path: Path, table_path: Path
) -> None:
    from .carrier_roughness_research import (
        CarrierRoughnessResearchConfig,
        run_carrier_roughness_research,
    )
    from .glue_theorem_useful import UsefulCarrierConfig, run_useful_carrier_research

    useful_result = run_useful_carrier_research(
        UsefulCarrierConfig(
            ambient_intrinsic_pairs=((8, 1),),
            spans=(0.10, 0.25, 0.50),
            sample_sizes=(32, 64, 128, 256, 512),
            replications=8,
        )
    )
    practical_result = run_carrier_roughness_research(
        CarrierRoughnessResearchConfig(
            raw_dims=(),
            ambient_intrinsic_pairs=(),
            triangular_dims=(),
            H_values=(),
            fixed_spans=(),
            span_growth_fractions=(),
            sinkhorn_epsilons=(0.50, 0.20, 0.10, 0.05),
            sinkhorn_ambient_intrinsic_pairs=((8, 1),),
            sinkhorn_sample_sizes=(24, 48, 96, 160),
            sinkhorn_seed_count=10,
        )
    )

    useful_rows = [
        row
        for row in useful_result.summary_rows
        if row["experiment"] == "useful-fixed-span"
    ]
    practical_rows = [
        row
        for row in practical_result.summary_rows
        if row["experiment"] == "sinkhorn-fixed-span"
    ]

    rows: list[dict[str, str | float]] = []
    useful_tri_x: list[float] = []
    useful_tri_y: list[float] = []
    useful_iid_x: list[float] = []
    useful_iid_y: list[float] = []
    for row in useful_rows:
        span = _extract_float(str(row["setting"]), "span=")
        carrier_a = float(row["carrier_a"])
        rows.append(
            {
                "layer": "extended",
                "setting": str(row["setting"]),
                "parameter": round(span, 2),
                "carrier_a": round(carrier_a, 4),
                "comment": str(row["comment"]),
            }
        )
        if "triangular" in str(row["setting"]):
            useful_tri_x.append(span)
            useful_tri_y.append(carrier_a)
        else:
            useful_iid_x.append(span)
            useful_iid_y.append(carrier_a)

    practical_tri_x: list[float] = []
    practical_tri_y: list[float] = []
    practical_iid_x: list[float] = []
    practical_iid_y: list[float] = []
    for row in practical_rows:
        epsilon = _extract_float(str(row["setting"]), "eps=")
        carrier_a = float(row["carrier_a"])
        rows.append(
            {
                "layer": "operational",
                "setting": str(row["setting"]),
                "parameter": round(epsilon, 2),
                "carrier_a": round(carrier_a, 4),
                "comment": str(row["comment"]),
            }
        )
        if "triangular" in str(row["setting"]):
            practical_tri_x.append(epsilon)
            practical_tri_y.append(carrier_a)
        else:
            practical_iid_x.append(epsilon)
            practical_iid_y.append(carrier_a)

    write_csv(csv_path, rows)

    fig, (ax_useful, ax_practical) = plt.subplots(1, 2, figsize=(9.0, 3.6))

    ax_useful.plot(
        useful_tri_x, useful_tri_y, marker="o", color="#315caf", linewidth=2.1
    )
    ax_useful.plot(
        useful_iid_x,
        useful_iid_y,
        marker="o",
        color="#b23a3a",
        linewidth=2.1,
    )
    ax_useful.set_title("Extended regime")
    ax_useful.set_xlabel("span")
    ax_useful.set_ylabel(r"estimated carrier $a$")
    ax_useful.set_ylim(0.38, 0.58)
    ax_useful.spines["top"].set_visible(False)
    ax_useful.spines["right"].set_visible(False)
    ax_useful.grid(axis="y", alpha=0.18, linewidth=0.6)
    ax_useful.legend(["triangular", "i.i.d. mixture"], frameon=False, loc="best")

    ax_practical.plot(
        practical_tri_x,
        practical_tri_y,
        marker="o",
        color="#315caf",
        linewidth=2.1,
    )
    ax_practical.plot(
        practical_iid_x,
        practical_iid_y,
        marker="o",
        color="#b23a3a",
        linewidth=2.1,
    )
    ax_practical.set_title("Operational regime")
    ax_practical.set_xlabel(r"Sinkhorn $\epsilon$")
    ax_practical.set_ylabel(r"estimated carrier $a$")
    ax_practical.set_ylim(0.40, 0.56)
    ax_practical.spines["top"].set_visible(False)
    ax_practical.spines["right"].set_visible(False)
    ax_practical.grid(axis="y", alpha=0.18, linewidth=0.6)
    ax_practical.legend(["triangular", "i.i.d. mixture"], frameon=False, loc="best")

    fig.suptitle("Carrier regimes beyond the minimum kernel", y=1.02, fontsize=10)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    table_lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\small",
        r"\caption{Summary rows for the extended and operational carrier regimes. The extended rows compare triangular and i.i.d. mixture carriers across span, while the operational rows compare fixed-$\epsilon$ Sinkhorn across $\epsilon$ on embedded low-intrinsic-dimensional support.}",
        r"\label{tab:carrier_layers}",
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"Regime & Setting & Parameter & Carrier $a$ " + r"\\",
        r"\midrule",
    ]
    for row in rows:
        table_lines.append(
            f"{row['layer']} & {row['setting']} & {row['parameter']:.2f} & {row['carrier_a']:.4f} "
            + r"\\"
        )
    table_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    write_text(table_path, "\n".join(table_lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate conceptual paper figures.")
    parser.add_argument(
        "--figures-dir", type=Path, default=Path("artifacts/figures/conceptual")
    )
    parser.add_argument(
        "--csv-dir", type=Path, default=Path("artifacts/csv/conceptual")
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
    generate_carrier_layers(
        args.figures_dir / "fig_carrier_layers.pdf",
        args.csv_dir / "carrier_layers.csv",
        Path("artifacts/tables/conceptual") / "carrier_layers.tex",
    )
