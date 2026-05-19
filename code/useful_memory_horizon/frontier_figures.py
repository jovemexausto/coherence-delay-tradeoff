from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .common import export_rows_csv
from .operational_complexity_inheritance import certify_operational_theorem_candidate
from .operational_dual_inheritance_kernel import (
    critical_dual_smoothness_for_parametric_region,
)
from .operational_region_thresholds import maximal_stable_epsilon_band
from .operational_regime_frontier import map_operational_regime
from .regular_family_frontier import (
    regular_family_horizon_exponent,
    regular_family_metric_carrier_exponent,
    regular_family_rate_exponent,
)


PAIR_ORDER = ((8, 1), (8, 2), (12, 1), (12, 2))
EPSILONS = (0.8, 0.5, 0.3, 0.2)


def frontier_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
        }
    )


def _pair_label(pair: tuple[int, int]) -> str:
    return f"d={pair[0]}, k={pair[1]}"


def generate_two_benchmarks_one_theory(output_path: Path, csv_path: Path) -> None:
    rows = [
        {
            "benchmark": "canonical",
            "title": "Canonical Benchmark",
            "geometry": "W2 / Gaussian location",
            "carrier": "a = 1/2",
            "status": "closed theorem",
            "role": "clean distributional benchmark",
        },
        {
            "benchmark": "operational",
            "title": "Operational Benchmark",
            "geometry": "fixed-epsilon Sinkhorn",
            "carrier": "a = a_epsilon",
            "status": "operational frontier",
            "role": "usable mid/high-dimensional regime",
        },
    ]
    export_rows_csv(rows, csv_path)

    fig, ax = plt.subplots(figsize=(11.2, 4.8))
    ax.axis("off")

    central_title = "Carrier-Roughness Horizon Law"
    central_body = (
        r"$\mathbb{E}\,d(\widehat P_t^{(n)},P_t) \leq C_K n^{-a} + C_S\zeta n^H$"
        "\n"
        r"$n^*(a,H) \propto (C_K/\zeta)^{1/(a+H)}$"
    )

    box_style_left = dict(
        boxstyle="round,pad=0.35",
        facecolor="#eef4ff",
        edgecolor="#315caf",
        linewidth=1.2,
    )
    box_style_right = dict(
        boxstyle="round,pad=0.35",
        facecolor="#fff1e8",
        edgecolor="#cc6f2c",
        linewidth=1.2,
    )
    center_style = dict(
        boxstyle="round,pad=0.4", facecolor="white", edgecolor="black", linewidth=1.3
    )

    ax.text(
        0.5,
        0.83,
        "Two Benchmarks, One Theory",
        ha="center",
        va="center",
        fontsize=14,
        weight="bold",
    )
    ax.text(
        0.5,
        0.61,
        central_title,
        ha="center",
        va="center",
        fontsize=11,
        weight="bold",
        bbox=center_style,
    )
    ax.text(0.5, 0.46, central_body, ha="center", va="center", fontsize=11)
    ax.text(
        0.18,
        0.56,
        "Canonical Benchmark\n\nGeometry: W2 / Gaussian location\nStatus: closed theorem\nCarrier: a = 1/2\nRole: conceptual closure",
        ha="center",
        va="center",
        fontsize=10.2,
        bbox=box_style_left,
    )
    ax.text(
        0.82,
        0.56,
        "Operational Benchmark\n\nGeometry: fixed-epsilon Sinkhorn\nStatus: operational frontier\nCarrier: a = a_epsilon\nRole: mid/high-dimensional regime",
        ha="center",
        va="center",
        fontsize=10.2,
        bbox=box_style_right,
    )
    ax.annotate(
        "",
        xy=(0.39, 0.56),
        xytext=(0.31, 0.56),
        arrowprops=dict(arrowstyle="-|>", linewidth=1.2, color="#315caf"),
    )
    ax.annotate(
        "",
        xy=(0.61, 0.56),
        xytext=(0.69, 0.56),
        arrowprops=dict(arrowstyle="-|>", linewidth=1.2, color="#cc6f2c"),
    )
    ax.text(
        0.5,
        0.17,
        "The benchmark theorem closes the clean distributional case; the operational benchmark carries the theory into usable high-dimensional practice.",
        ha="center",
        fontsize=9.5,
        color="#333333",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def generate_operational_frontier_map(output_path: Path, csv_path: Path) -> None:
    rows = map_operational_regime(PAIR_ORDER, epsilons=EPSILONS, seed_count=24)
    csv_rows = [
        {
            "ambient_dim": row.ambient_dim,
            "intrinsic_dim": row.intrinsic_dim,
            "epsilon": row.epsilon,
            "iid_a": row.iid_a,
            "triangular_a": row.triangular_a,
            "gap": row.gap,
            "useful": row.useful,
            "min_a": min(row.iid_a, row.triangular_a),
        }
        for row in rows
    ]
    export_rows_csv(csv_rows, csv_path)

    ambients = sorted({pair[0] for pair in PAIR_ORDER})
    intrinsics = sorted({pair[1] for pair in PAIR_ORDER})
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.8), constrained_layout=True)
    for ax, epsilon in zip(axes.flat, EPSILONS, strict=True):
        matrix = np.full((len(intrinsics), len(ambients)), np.nan)
        useful = np.zeros_like(matrix, dtype=bool)
        for row in rows:
            if abs(row.epsilon - epsilon) > 1e-12:
                continue
            i = intrinsics.index(row.intrinsic_dim)
            j = ambients.index(row.ambient_dim)
            matrix[i, j] = min(row.iid_a, row.triangular_a)
            useful[i, j] = row.useful
        im = ax.imshow(matrix, vmin=0.35, vmax=0.60, cmap="viridis", origin="lower")
        for i in range(len(intrinsics)):
            for j in range(len(ambients)):
                if not np.isnan(matrix[i, j]):
                    color = "white" if matrix[i, j] < 0.48 else "black"
                    ax.text(
                        j,
                        i,
                        f"{matrix[i, j]:.2f}",
                        ha="center",
                        va="center",
                        color=color,
                        fontsize=9,
                    )
                    if not useful[i, j]:
                        ax.plot(
                            [j - 0.35, j + 0.35],
                            [i - 0.35, i + 0.35],
                            color="crimson",
                            linewidth=2,
                        )
                        ax.plot(
                            [j - 0.35, j + 0.35],
                            [i + 0.35, i - 0.35],
                            color="crimson",
                            linewidth=2,
                        )
        ax.set_xticks(range(len(ambients)), [f"d={d}" for d in ambients])
        ax.set_yticks(range(len(intrinsics)), [f"k={k}" for k in intrinsics])
        ax.set_title(rf"$\varepsilon={epsilon:.2f}$")
        ax.set_xlabel("Ambient dimension")
        ax.set_ylabel("Intrinsic dimension")
    cbar = fig.colorbar(im, ax=axes, shrink=0.82)
    cbar.set_label(r"min$(a_{iid}, a_{tri})$")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def generate_operational_theorem_mechanism(output_path: Path, csv_path: Path) -> None:
    rows = map_operational_regime(PAIR_ORDER, epsilons=EPSILONS, seed_count=24)
    threshold_rows = []
    for pair in PAIR_ORDER:
        threshold_rows.append(
            {
                "ambient_dim": pair[0],
                "intrinsic_dim": pair[1],
                "epsilon_max": maximal_stable_epsilon_band(rows, pair[0], pair[1]),
            }
        )
    export_rows_csv(threshold_rows, csv_path)

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), constrained_layout=True)

    ks = np.array([1, 2, 4, 6], dtype=float)
    axes[0].plot(ks, 0.5 * ks, marker="o", color="#315caf", linewidth=2.2)
    axes[0].axhline(2.0, linestyle="--", color="#cc6f2c", linewidth=1.2)
    axes[0].set_title(r"Dual threshold $\alpha > k/2$")
    axes[0].set_xlabel("effective intrinsic complexity $k$")
    axes[0].set_ylabel(r"critical smoothness $\alpha_{crit}$")
    axes[0].grid(alpha=0.18)

    eps = np.array(EPSILONS)
    ratio_k1 = np.ones_like(eps)
    ratio_k2 = np.ones_like(eps)
    axes[1].plot(eps, ratio_k1, marker="o", label="k=1", color="#315caf", linewidth=2.2)
    axes[1].plot(eps, ratio_k2, marker="s", label="k=2", color="#cc6f2c", linewidth=2.2)
    axes[1].axhline(1.0, color="black", linewidth=1.0, linestyle="--")
    axes[1].set_xscale("log")
    axes[1].invert_xaxis()
    axes[1].set_title("Exact support-complexity inheritance")
    axes[1].set_xlabel(r"regularization $\varepsilon$")
    axes[1].set_ylabel("triangular / iid covering ratio")
    axes[1].legend(frameon=False)

    labels = [_pair_label(pair) for pair in PAIR_ORDER]
    values = [row["epsilon_max"] for row in threshold_rows]
    colors = ["#315caf" if value >= 0.5 else "#cc6f2c" for value in values]
    axes[2].bar(labels, values, color=colors)
    axes[2].set_ylim(0.0, 0.55)
    axes[2].set_title("Maximal stable epsilon band")
    axes[2].set_ylabel(r"largest stable $\varepsilon_{max}$")
    axes[2].tick_params(axis="x", rotation=20)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def generate_regular_family_inheritance(output_path: Path, csv_path: Path) -> None:
    H = 1.0
    alphas = np.linspace(0.5, 2.0, 300)
    rows = [
        {
            "alpha": round(float(alpha), 6),
            "carrier_a": round(float(regular_family_metric_carrier_exponent(alpha)), 6),
            "rate_exponent": round(float(regular_family_rate_exponent(alpha, H)), 6),
            "horizon_exponent": round(
                float(regular_family_horizon_exponent(alpha, H)), 6
            ),
        }
        for alpha in alphas
    ]
    export_rows_csv(rows, csv_path)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), constrained_layout=True)
    axes[0].plot(alphas, 0.5 * alphas, color="#315caf", linewidth=2.3)
    axes[0].axvline(1.0, color="#666666", linestyle="--", linewidth=1.0)
    axes[0].scatter([1.0], [0.5], color="black", zorder=3)
    axes[0].set_title(r"Carrier exponent $a = \alpha/2$")
    axes[0].set_xlabel(r"metric exponent $\alpha$")
    axes[0].set_ylabel(r"carrier exponent $a$")
    axes[0].grid(alpha=0.18)
    axes[0].annotate(
        "Gaussian location / scale",
        xy=(1.0, 0.5),
        xytext=(1.15, 0.62),
        fontsize=8,
        arrowprops=dict(arrowstyle="-", linewidth=0.8),
    )

    for H_value, color in ((0.5, "#cc6f2c"), (1.0, "#2f7d4a")):
        axes[1].plot(
            alphas,
            [regular_family_horizon_exponent(alpha, H_value) for alpha in alphas],
            color=color,
            linewidth=2.3,
            label=rf"$H={H_value}$",
        )
    axes[1].axvline(1.0, color="#666666", linestyle="--", linewidth=1.0)
    axes[1].set_title(r"Horizon exponent $1/[\alpha(1/2+H)]$")
    axes[1].set_xlabel(r"metric exponent $\alpha$")
    axes[1].set_ylabel(r"power on $(C/\zeta)$")
    axes[1].grid(alpha=0.18)
    axes[1].legend(frameon=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def generate_distributional_taxonomy(output_path: Path, csv_path: Path) -> None:
    rows = [
        {
            "class": "regular dominated",
            "examples": "Gaussian location, Gaussian scale",
            "local_W2": "regular metric scaling",
            "local_testing": "quadratic KL / Hellinger",
            "theorem_status": "theorem target",
        },
        {
            "class": "support-changing nonregular",
            "examples": "uniform scale",
            "local_W2": "can remain linear",
            "local_testing": "nonquadratic / one-sided KL",
            "theorem_status": "separate theory needed",
        },
        {
            "class": "singular / atomic",
            "examples": "two-point scale",
            "local_W2": "can remain linear",
            "local_testing": "singular",
            "theorem_status": "exclusion / counterexample class",
        },
    ]
    export_rows_csv(rows, csv_path)

    fig, ax = plt.subplots(figsize=(12.0, 4.8))
    ax.axis("off")
    ax.text(
        0.5,
        0.92,
        "Distributional Extension Taxonomy",
        ha="center",
        va="center",
        fontsize=14,
        weight="bold",
    )
    card_specs = [
        (0.17, "#eef4ff", "#315caf", rows[0]),
        (0.50, "#fff4ea", "#cc6f2c", rows[1]),
        (0.83, "#f8ecec", "#b23a3a", rows[2]),
    ]
    for x, face, edge, row in card_specs:
        body = (
            f"Examples: {row['examples']}\n"
            f"Local W2: {row['local_W2']}\n"
            f"Testing: {row['local_testing']}\n"
            f"Role: {row['theorem_status']}"
        )
        ax.text(
            x,
            0.48,
            row["class"].title() + "\n\n" + body,
            ha="center",
            va="center",
            fontsize=9.5,
            bbox=dict(
                boxstyle="round,pad=0.45", facecolor=face, edgecolor=edge, linewidth=1.3
            ),
        )
    ax.text(
        0.5,
        0.14,
        "Transport-Hölder control alone is too weak for a universal extension theorem. The correct theorem classes are regular dominated families and operational geometries with analogous local structure.",
        ha="center",
        fontsize=9.2,
        color="#333333",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def generate_prescriptive_memory_law(output_path: Path, csv_path: Path) -> None:
    zeta_values = np.logspace(-3, -1, 200)
    rows = []
    for a, label in (
        (0.5, "canonical / parametric"),
        (5.0 / 12.0, "operational degraded"),
        (1.0 / 3.0, "rough noncanonical"),
    ):
        for zeta in zeta_values:
            n_star = (1.0 / zeta) ** (1.0 / (a + 1.0))
            risk_star = zeta ** (a / (a + 1.0))
            rows.append(
                {
                    "label": label,
                    "a": a,
                    "zeta": float(zeta),
                    "n_star": float(n_star),
                    "risk_star": float(risk_star),
                }
            )
    export_rows_csv(rows, csv_path)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)
    for a, label, color in (
        (0.5, "canonical / parametric", "#315caf"),
        (5.0 / 12.0, "operational degraded", "#cc6f2c"),
        (1.0 / 3.0, "rough noncanonical", "#2f7d4a"),
    ):
        n_star = (1.0 / zeta_values) ** (1.0 / (a + 1.0))
        risk_star = zeta_values ** (a / (a + 1.0))
        axes[0].plot(zeta_values, n_star, label=label, color=color, linewidth=2.3)
        axes[1].plot(zeta_values, risk_star, label=label, color=color, linewidth=2.3)
    for ax, ylabel in zip(
        axes, (r"optimal memory $n^*$", r"optimized error scale"), strict=True
    ):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"roughness level $\zeta$")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.18, which="both")
    axes[0].set_title("Prescriptive horizon law")
    axes[1].set_title("Prescriptive error law")
    axes[1].legend(frameon=False, loc="upper right")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate frontier figures.")
    parser.add_argument(
        "--figures-dir", type=Path, default=Path("artifacts/figures/frontier")
    )
    parser.add_argument("--csv-dir", type=Path, default=Path("artifacts/csv/frontier"))
    return parser.parse_args()


def main() -> None:
    frontier_style()
    args = parse_args()
    generate_two_benchmarks_one_theory(
        args.figures_dir / "fig_two_benchmarks_one_theory.pdf",
        args.csv_dir / "two_benchmarks_one_theory.csv",
    )
    generate_operational_frontier_map(
        args.figures_dir / "fig_operational_frontier_map.pdf",
        args.csv_dir / "operational_frontier_map.csv",
    )
    generate_operational_theorem_mechanism(
        args.figures_dir / "fig_operational_theorem_mechanism.pdf",
        args.csv_dir / "operational_theorem_mechanism.csv",
    )
    generate_regular_family_inheritance(
        args.figures_dir / "fig_regular_family_inheritance.pdf",
        args.csv_dir / "regular_family_inheritance.csv",
    )
    generate_distributional_taxonomy(
        args.figures_dir / "fig_distributional_taxonomy.pdf",
        args.csv_dir / "distributional_taxonomy.csv",
    )
    generate_prescriptive_memory_law(
        args.figures_dir / "fig_prescriptive_memory_law.pdf",
        args.csv_dir / "prescriptive_memory_law.csv",
    )


if __name__ == "__main__":
    main()
