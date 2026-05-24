from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from river import datasets
from scipy.stats import wasserstein_distance

from .common import build_manifest_row, export_rows_csv, file_sha256, stable_run_id


def _default_window_sizes() -> tuple[int, ...]:
    values = np.unique(np.round(np.geomspace(8, 1024, 18)).astype(int)).tolist()
    return tuple(int(value) for value in values)


@dataclass(frozen=True, slots=True)
class Elec2DiagnosticConfig:
    dataset_name: str = "Elec2"
    max_samples: int = 15_000
    start_index: int = 0
    anchor_size: int = 48
    step: int = 24
    useful_delta: float = 0.05
    variable: str = "nswprice"
    window_sizes: tuple[int, ...] = field(default_factory=_default_window_sizes)


@dataclass(frozen=True, slots=True)
class Elec2DatasetMetadata:
    dataset_name: str
    dataset_path: str
    dataset_sha256: str
    requested_max_samples: int
    loaded_sample_count: int
    start_index: int
    variable: str


@dataclass(frozen=True, slots=True)
class Elec2DiagnosticResult:
    config: Elec2DiagnosticConfig
    metadata: Elec2DatasetMetadata
    values: np.ndarray
    window_sizes: np.ndarray
    mean_w1: np.ndarray
    best_window: int
    best_error: float
    useful_windows: np.ndarray


def _parse_csv_ints(text: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in text.split(",") if part.strip())


def _parse_csv_strings(text: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in text.split(",") if part.strip())


def _load_values(
    config: Elec2DiagnosticConfig,
) -> tuple[np.ndarray, Elec2DatasetMetadata]:
    dataset = datasets.Elec2()
    path = Path(dataset.path)
    values: list[float] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader):
            if index < config.start_index:
                continue
            values.append(float(row[config.variable]))
            if len(values) >= config.max_samples:
                break
    metadata = Elec2DatasetMetadata(
        dataset_name=config.dataset_name,
        dataset_path=str(path),
        dataset_sha256=file_sha256(path),
        requested_max_samples=config.max_samples,
        loaded_sample_count=len(values),
        start_index=config.start_index,
        variable=config.variable,
    )
    return np.asarray(values, dtype=float), metadata


def run_elec2_diagnostic(
    config: Elec2DiagnosticConfig | None = None,
) -> Elec2DiagnosticResult:
    cfg = config or Elec2DiagnosticConfig()
    values, metadata = _load_values(cfg)
    window_sizes = np.asarray(cfg.window_sizes, dtype=int)
    max_window = int(window_sizes.max())
    if values.size <= max_window + cfg.anchor_size:
        raise ValueError("not enough observations for the requested ELEC2 diagnostic")
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
        metadata=metadata,
        values=values,
        window_sizes=window_sizes,
        mean_w1=mean_w1,
        best_window=best_window,
        best_error=best_error,
        useful_windows=useful_windows,
    )


def build_elec2_rows(result: Elec2DiagnosticResult) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int | str | bool]] = []
    normalizer = max(float(result.best_error), 1.0e-12)
    normalized = result.mean_w1 / normalizer
    for window, error in zip(result.window_sizes, result.mean_w1, strict=True):
        normalized_error = normalized[np.where(result.window_sizes == window)[0][0]]
        rows.append(
            {
                "run_id": stable_run_id(asdict(result.config)),
                "dataset_name": result.metadata.dataset_name,
                "variable": result.metadata.variable,
                "start_index": result.metadata.start_index,
                "loaded_sample_count": result.metadata.loaded_sample_count,
                "anchor_size": result.config.anchor_size,
                "step": result.config.step,
                "window": int(window),
                "mean_w1": round(float(error), 8),
                "normalized_mean_w1": round(float(normalized_error), 8),
                "is_best": int(window) == result.best_window,
                "in_useful_band": int(window in set(result.useful_windows.tolist())),
                "best_window": int(result.best_window),
                "best_error": round(float(result.best_error), 8),
                "degenerate_best_error": int(float(result.best_error) <= 1.0e-12),
            }
        )
    return rows


def build_elec2_summary_rows(
    results: list[Elec2DiagnosticResult],
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for result in results:
        run_id = stable_run_id(asdict(result.config))
        useful_min = (
            int(result.useful_windows.min()) if result.useful_windows.size else ""
        )
        useful_max = (
            int(result.useful_windows.max()) if result.useful_windows.size else ""
        )
        edge_ratio = float(
            max(result.mean_w1[0], result.mean_w1[-1]) / max(result.best_error, 1.0e-12)
        )
        rows.append(
            {
                "run_id": run_id,
                "dataset_name": result.metadata.dataset_name,
                "dataset_path": result.metadata.dataset_path,
                "dataset_sha256": result.metadata.dataset_sha256,
                "variable": result.metadata.variable,
                "start_index": result.metadata.start_index,
                "requested_max_samples": result.metadata.requested_max_samples,
                "loaded_sample_count": result.metadata.loaded_sample_count,
                "anchor_size": result.config.anchor_size,
                "step": result.config.step,
                "useful_delta": result.config.useful_delta,
                "best_window": result.best_window,
                "best_error": round(float(result.best_error), 8),
                "useful_min_window": useful_min,
                "useful_max_window": useful_max,
                "useful_window_count": int(result.useful_windows.size),
                "interior_optimum": int(
                    result.best_window > int(result.window_sizes[0])
                    and result.best_window < int(result.window_sizes[-1])
                ),
                "degenerate_best_error": int(float(result.best_error) <= 1.0e-12),
                "edge_to_best_ratio": round(edge_ratio, 6),
            }
        )
    return rows


def run_elec2_robustness_sweep(
    configs: list[Elec2DiagnosticConfig],
) -> list[Elec2DiagnosticResult]:
    return [run_elec2_diagnostic(config) for config in configs]


def build_elec2_robustness_configs(
    *,
    variables: tuple[str, ...],
    anchor_sizes: tuple[int, ...],
    start_indices: tuple[int, ...],
    max_samples: int,
    step: int,
    useful_delta: float,
    window_sizes: tuple[int, ...] | None = None,
) -> list[Elec2DiagnosticConfig]:
    configs: list[Elec2DiagnosticConfig] = []
    for variable in variables:
        for anchor_size in anchor_sizes:
            for start_index in start_indices:
                configs.append(
                    Elec2DiagnosticConfig(
                        variable=variable,
                        anchor_size=anchor_size,
                        start_index=start_index,
                        max_samples=max_samples,
                        step=step,
                        useful_delta=useful_delta,
                        window_sizes=_default_window_sizes()
                        if window_sizes is None
                        else window_sizes,
                    )
                )
    return configs


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
    parser.add_argument("--variable", type=str, default="nswprice")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=15_000)
    parser.add_argument("--anchor-size", type=int, default=48)
    parser.add_argument("--step", type=int, default=24)
    parser.add_argument(
        "--variables",
        type=str,
        default="",
        help="Comma-separated variables for robustness sweep.",
    )
    parser.add_argument(
        "--anchor-sizes",
        type=str,
        default="",
        help="Comma-separated anchor sizes for robustness sweep.",
    )
    parser.add_argument(
        "--start-indices",
        type=str,
        default="",
        help="Comma-separated start indices for robustness sweep.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    variables = (
        (args.variable,) if not args.variables else _parse_csv_strings(args.variables)
    )
    anchor_sizes = (
        (args.anchor_size,)
        if not args.anchor_sizes
        else _parse_csv_ints(args.anchor_sizes)
    )
    start_indices = (
        (args.start_index,)
        if not args.start_indices
        else _parse_csv_ints(args.start_indices)
    )
    configs = build_elec2_robustness_configs(
        variables=variables,
        anchor_sizes=anchor_sizes,
        start_indices=start_indices,
        max_samples=args.max_samples,
        step=args.step,
        useful_delta=0.05,
    )
    results = run_elec2_robustness_sweep(configs)
    result = results[0]
    save_elec2_figure(result, args.figures_dir / "fig_elec2_ucurve.pdf")
    all_curve_rows: list[dict[str, float | int | str | bool]] = []
    for sweep_result in results:
        all_curve_rows.extend(build_elec2_rows(sweep_result))
    export_rows_csv(all_curve_rows, args.csv_dir / "elec2_ucurve.csv")
    export_rows_csv(build_elec2_summary_rows(results), args.csv_dir / "summary.csv")
    manifest = build_manifest_row(
        "elec2_real",
        {
            "variables": variables,
            "anchor_sizes": anchor_sizes,
            "start_indices": start_indices,
            "max_samples": args.max_samples,
            "step": args.step,
        },
        run_id=stable_run_id(
            {
                "variables": variables,
                "anchor_sizes": anchor_sizes,
                "start_indices": start_indices,
                "max_samples": args.max_samples,
                "step": args.step,
            }
        ),
        notes="Real-stream diagnostic with dataset fingerprint metadata and robustness sweep support.",
    )
    export_rows_csv([manifest], args.csv_dir / "manifest.csv")


if __name__ == "__main__":
    main()
