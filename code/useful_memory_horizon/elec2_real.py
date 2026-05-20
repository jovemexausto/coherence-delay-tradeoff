from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from river import datasets
from scipy.stats import wasserstein_distance

from .common import export_rows_csv


def _default_window_sizes() -> tuple[int, ...]:
    values = np.unique(np.round(np.geomspace(8, 1024, 18)).astype(int)).tolist()
    return tuple(int(value) for value in values)


@dataclass(frozen=True, slots=True)
class Elec2DiagnosticConfig:
    max_samples: int = 15_000
    anchor_size: int = 48
    step: int = 24
    useful_delta: float = 0.05
    variable: str = "nswprice"
    window_sizes: tuple[int, ...] = field(default_factory=_default_window_sizes)


@dataclass(frozen=True, slots=True)
class Elec2DiagnosticResult:
    config: Elec2DiagnosticConfig
    values: np.ndarray
    window_sizes: np.ndarray
    mean_w1: np.ndarray
    best_window: int
    best_error: float
    useful_windows: np.ndarray


def _load_values(config: Elec2DiagnosticConfig) -> np.ndarray:
    dataset = datasets.Elec2()
    path = Path(dataset.path)
    values: list[float] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader):
            values.append(float(row[config.variable]))
            if index >= config.max_samples:
                break
    return np.asarray(values, dtype=float)


def run_elec2_diagnostic(
    config: Elec2DiagnosticConfig | None = None,
) -> Elec2DiagnosticResult:
    cfg = config or Elec2DiagnosticConfig()
    values = _load_values(cfg)
    window_sizes = np.asarray(cfg.window_sizes, dtype=int)
    max_window = int(window_sizes.max())
    times = range(max_window, len(values) - cfg.anchor_size, cfg.step)
    mean_w1 = np.zeros(window_sizes.size, dtype=float)

    for index, window in enumerate(window_sizes):
        errors: list[float] = []
        for time in times:
            past = values[time - window : time]
            present = values[time : time + cfg.anchor_size]
            errors.append(float(wasserstein_distance(past, present)))
        mean_w1[index] = float(np.mean(errors))

    best_index = int(np.argmin(mean_w1))
    best_window = int(window_sizes[best_index])
    best_error = float(mean_w1[best_index])
    useful_mask = mean_w1 <= (1.0 + cfg.useful_delta) * best_error
    useful_windows = window_sizes[useful_mask]
    return Elec2DiagnosticResult(
        config=cfg,
        values=values,
        window_sizes=window_sizes,
        mean_w1=mean_w1,
        best_window=best_window,
        best_error=best_error,
        useful_windows=useful_windows,
    )


def build_elec2_rows(result: Elec2DiagnosticResult) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for window, error in zip(result.window_sizes, result.mean_w1, strict=True):
        rows.append(
            {
                "window": int(window),
                "mean_w1": round(float(error), 8),
                "best_window": int(result.best_window),
                "best_error": round(float(result.best_error), 8),
            }
        )
    return rows


def save_elec2_figure(result: Elec2DiagnosticResult, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(7.2, 4.6))
    axis.plot(
        result.window_sizes, result.mean_w1, linewidth=1.8, marker="o", markersize=3
    )
    axis.set_xscale("log")
    axis.set_xlabel("Memory window n")
    axis.set_ylabel(r"Mean anchor-block $W_1$")
    axis.set_title("ELEC2 real-stream diagnostic")
    axis.grid(alpha=0.2, linewidth=0.5)
    axis.scatter(
        [result.best_window], [result.best_error], color="black", s=28, zorder=4
    )

    if result.useful_windows.size > 0:
        axis.axvspan(
            float(result.useful_windows.min()),
            float(result.useful_windows.max()),
            color="#e7f2df",
            alpha=0.9,
            linewidth=0,
            zorder=0,
        )
        axis.text(
            float(np.sqrt(result.useful_windows.min() * result.useful_windows.max())),
            float(
                result.mean_w1.max()
                - 0.08 * (result.mean_w1.max() - result.mean_w1.min())
            ),
            "useful-memory band",
            ha="center",
            fontsize=8,
            bbox={
                "boxstyle": "round,pad=0.16",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.95,
            },
        )

    axis.annotate(
        rf"best $n={result.best_window}$",
        xy=(result.best_window, result.best_error),
        xytext=(1.12 * result.best_window, 1.04 * result.best_error),
        fontsize=8,
        arrowprops={"arrowstyle": "-", "linewidth": 0.9, "color": "black"},
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.95,
        },
    )

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate ELEC2 real-stream diagnostics."
    )
    parser.add_argument(
        "--figures-dir", type=Path, default=Path("artifacts/figures/elec2")
    )
    parser.add_argument("--csv-dir", type=Path, default=Path("artifacts/csv/elec2"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_elec2_diagnostic(Elec2DiagnosticConfig())
    save_elec2_figure(result, args.figures_dir / "fig_elec2_ucurve.pdf")
    export_rows_csv(build_elec2_rows(result), args.csv_dir / "elec2_ucurve.csv")


if __name__ == "__main__":
    main()
