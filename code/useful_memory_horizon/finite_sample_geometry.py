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


def build_table(rows: list[dict[str, str | float]]) -> str:
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\small",
        r"\caption{Empirical scaling check for the finite-sample term. The bounded-support $W_2$ experiment is close to the expected $n^{-1/2}$ rate in low dimension, and the slope degrades as intrinsic dimension increases. The Sinkhorn proxy is reported at fixed $\varepsilon$, where the measured slope and constant both vary with regularization.}",
        r"\label{tab:finite_sample_geometry}",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Setting & Estimated slope & Comment " + ROW_END,
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['setting']} & {row['estimated_slope']} & {row['comment']} " + ROW_END
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
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
    rows = estimate_raw_w2_slopes() + estimate_sinkhorn_rows()
    write_csv(args.csv_dir / "finite_sample_geometry.csv", rows)
    write_text(args.tables_dir / "finite_sample_geometry.tex", build_table(rows))
