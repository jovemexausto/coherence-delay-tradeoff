from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import cast

import numpy as np
from scipy.optimize import linear_sum_assignment  # pyright: ignore[reportUnknownVariableType]

from .sinkhorn import debiased_sinkhorn_divergence

ROW_END = r"\\"


def write_csv(path: Path, rows: list[dict[str, str | float]]) -> None:
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


def exact_empirical_w2(x: np.ndarray, y: np.ndarray) -> float:
    x_norm = np.sum(x * x, axis=1, keepdims=True)
    y_norm = np.sum(y * y, axis=1, keepdims=True).T
    cost = np.maximum(x_norm + y_norm - 2.0 * x @ y.T, 0.0)
    row_ind, col_ind = cast(tuple[np.ndarray, np.ndarray], linear_sum_assignment(cost))
    matched_costs = cast(np.ndarray, cost[row_ind, col_ind])
    return float(np.sqrt(np.mean(matched_costs)))


def estimate_raw_w2_slopes() -> list[dict[str, str | float]]:
    dims = (1, 2, 4, 5)
    sample_sizes = np.asarray([10, 20, 40, 80], dtype=int)
    seeds = range(10)
    rows: list[dict[str, str | float]] = []

    for dim in dims:
        means: list[float] = []
        for n in sample_sizes:
            values: list[float] = []
            for seed in seeds:
                rng = np.random.default_rng(1000 * dim + 10 * n + seed)
                x = rng.uniform(-1.0, 1.0, size=(n, dim))
                y = rng.uniform(-1.0, 1.0, size=(n, dim))
                values.append(exact_empirical_w2(x, y))
            means.append(float(np.mean(values)))
        slope = float(np.polyfit(np.log(sample_sizes), np.log(means), 1)[0])
        rows.append(
            {
                "setting": f"$W_2$, $d={dim}$",
                "estimated_slope": f"{slope:.2f}",
                "comment": "close to $-1/2$" if dim == 1 else "slower decay",
            }
        )
    return rows


def estimate_sinkhorn_rows() -> list[dict[str, str | float]]:
    sample_sizes = np.asarray([20, 40, 80, 120], dtype=int)
    epsilons = (0.50, 0.10, 0.05)
    seeds = range(8)
    rows: list[dict[str, str | float]] = []

    for epsilon in epsilons:
        means: list[float] = []
        for n in sample_sizes:
            values: list[float] = []
            for seed in seeds:
                rng = np.random.default_rng(5000 + int(100 * epsilon) + 10 * n + seed)
                x = rng.uniform(-1.0, 1.0, size=(n, 1))
                y = rng.uniform(-1.0, 1.0, size=(n, 1))
                result = debiased_sinkhorn_divergence(x, y, epsilon)
                values.append(abs(float(result.cost)) ** 0.5)
            means.append(float(np.mean(values)))
        slope = float(np.polyfit(np.log(sample_sizes), np.log(means), 1)[0])
        rows.append(
            {
                "setting": f"Sinkhorn $\\varepsilon={epsilon:.2f}$",
                "estimated_slope": f"{slope:.2f}",
                "comment": "proxy constant changes",
            }
        )
    return rows


def _path_means(n: int, dim: int, zeta: float, H: float) -> np.ndarray:
    lags = np.arange(n, dtype=float)
    means = np.zeros((n, dim), dtype=float)
    means[:, 0] = -zeta * lags**H
    return means


def _triangular_window_sample(
    n: int, dim: int, zeta: float, H: float, rng: np.random.Generator
) -> np.ndarray:
    means = _path_means(n, dim, zeta, H)
    return means + rng.normal(size=(n, dim))


def _mixture_sample(
    n: int, dim: int, zeta: float, H: float, rng: np.random.Generator
) -> np.ndarray:
    means = _path_means(n, dim, zeta, H)
    component_index = rng.integers(0, n, size=n)
    return means[component_index] + rng.normal(size=(n, dim))


def estimate_triangular_array_rows() -> tuple[
    list[dict[str, str | float]], list[dict[str, str | float]]
]:
    dim = 1
    H_values = (0.5, 1.0)
    span = 0.25
    sample_sizes = np.asarray([32, 64, 128, 256], dtype=int)
    seeds = range(24)
    summary_rows: list[dict[str, str | float]] = []
    curve_rows: list[dict[str, str | float]] = []

    for H in H_values:
        triangular_means: list[float] = []
        iid_means: list[float] = []
        for n in sample_sizes:
            zeta = span / (n**H)
            triangular_values: list[float] = []
            iid_values: list[float] = []
            for seed in seeds:
                triangular_rng = np.random.default_rng(
                    20_000 + 100 * int(10 * H) + 10 * n + seed
                )
                mixture_rng = np.random.default_rng(
                    30_000 + 100 * int(10 * H) + 10 * n + seed
                )
                comparison_rng = np.random.default_rng(
                    40_000 + 100 * int(10 * H) + 10 * n + seed
                )

                triangular_sample = _triangular_window_sample(
                    n, dim, zeta, H, triangular_rng
                )
                mixture_sample = _mixture_sample(n, dim, zeta, H, mixture_rng)
                iid_sample_left = _mixture_sample(n, dim, zeta, H, comparison_rng)
                iid_sample_right = _mixture_sample(n, dim, zeta, H, comparison_rng)

                triangular_values.append(
                    exact_empirical_w2(triangular_sample, mixture_sample)
                )
                iid_values.append(exact_empirical_w2(iid_sample_left, iid_sample_right))

            triangular_mean = float(np.mean(triangular_values))
            iid_mean = float(np.mean(iid_values))
            triangular_means.append(triangular_mean)
            iid_means.append(iid_mean)
            curve_rows.append(
                {
                    "setting": "triangular",
                    "dimension": dim,
                    "H": H,
                    "window_span": span,
                    "zeta": round(float(zeta), 8),
                    "sample_size": n,
                    "mean_w2": round(triangular_mean, 6),
                }
            )
            curve_rows.append(
                {
                    "setting": "iid-mixture",
                    "dimension": dim,
                    "H": H,
                    "window_span": span,
                    "zeta": round(float(zeta), 8),
                    "sample_size": n,
                    "mean_w2": round(iid_mean, 6),
                }
            )

        triangular_slope = float(
            np.polyfit(np.log(sample_sizes), np.log(triangular_means), 1)[0]
        )
        iid_slope = float(np.polyfit(np.log(sample_sizes), np.log(iid_means), 1)[0])
        summary_rows.append(
            {
                "setting": rf"$W_2$, i.i.d. mixture, $d=1$, $H={H:.1f}$",
                "estimated_slope": f"{iid_slope:.2f}",
                "comment": rf"fixed span $\zeta n^H={span}$",
            }
        )
        summary_rows.append(
            {
                "setting": rf"$W_2$, triangular, $d=1$, $H={H:.1f}$",
                "estimated_slope": f"{triangular_slope:.2f}",
                "comment": rf"fixed span $\zeta n^H={span}$",
            }
        )

    return summary_rows, curve_rows


def build_table(rows: list[dict[str, str | float]]) -> str:
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\small",
        r"\caption{Empirical scaling check for the finite-sample term. The bounded-support i.i.d. $W_2$ experiment is close to the expected $n^{-1/2}$ rate in low dimension, the corresponding triangular-array check compares one sample from each drifted component against an i.i.d. sample from the window mixture, and the Sinkhorn proxy is reported at fixed $\varepsilon$. These rows are numerical support for the carrier-regime discussion rather than theorem statements.}",
        r"\label{tab:finite_sample_geometry}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Setting & Estimated slope & Comment " + ROW_END,
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['setting']} & {row['estimated_slope']} & {row['comment']} " + ROW_END
        )
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}", ""])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate finite-sample geometry support."
    )
    parser.add_argument("--csv-dir", type=Path, default=Path("artifacts/csv/gaussian"))
    parser.add_argument(
        "--tables-dir", type=Path, default=Path("artifacts/tables/gaussian")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    triangular_rows, curve_rows = estimate_triangular_array_rows()
    rows = estimate_raw_w2_slopes() + triangular_rows + estimate_sinkhorn_rows()
    write_csv(args.csv_dir / "finite_sample_geometry.csv", rows)
    write_csv(args.csv_dir / "finite_sample_geometry_curves.csv", curve_rows)
    write_text(args.tables_dir / "finite_sample_geometry.tex", build_table(rows))
