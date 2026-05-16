from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .common import export_rows_csv
from .finite_sample_geometry import exact_empirical_w2


@dataclass(slots=True)
class UsefulCarrierConfig:
    ambient_intrinsic_pairs: tuple[tuple[int, int], ...] = ((8, 1), (8, 2))
    spans: tuple[float, ...] = (0.1, 0.25, 0.5)
    sample_sizes: tuple[int, ...] = (32, 64, 128, 256, 512)
    replications: int = 8


@dataclass(slots=True)
class UsefulCarrierResult:
    summary_rows: list[dict[str, str | float]]
    curve_rows: list[dict[str, str | float]]


def embedded_fixed_span_means(
    n: int, ambient_dim: int, intrinsic_dim: int, span: float
) -> np.ndarray:
    raw = np.arange(n, dtype=float)
    raw -= raw.min()
    scale = float(raw.max()) if raw.max() > 0 else 1.0
    means = np.zeros((n, ambient_dim), dtype=float)
    means[:, 0] = span * raw / scale
    return means


def embedded_uniform_window_sample(
    n: int, ambient_dim: int, intrinsic_dim: int, span: float, rng: np.random.Generator
) -> np.ndarray:
    means = embedded_fixed_span_means(n, ambient_dim, intrinsic_dim, span)
    sample = means.copy()
    sample[:, :intrinsic_dim] += rng.uniform(-1.0, 1.0, size=(n, intrinsic_dim))
    return sample


def estimate_log_slope(sample_sizes: np.ndarray, values: list[float]) -> float:
    return float(np.polyfit(np.log(sample_sizes), np.log(values), 1)[0])


def run_useful_carrier_research(
    config: UsefulCarrierConfig | None = None,
) -> UsefulCarrierResult:
    if config is None:
        config = UsefulCarrierConfig()

    summary_rows: list[dict[str, str | float]] = []
    curve_rows: list[dict[str, str | float]] = []
    sample_sizes = np.asarray(config.sample_sizes, dtype=int)

    for ambient_dim, intrinsic_dim in config.ambient_intrinsic_pairs:
        for span in config.spans:
            triangular_means: list[float] = []
            iid_means: list[float] = []

            for n in sample_sizes:
                means = embedded_fixed_span_means(n, ambient_dim, intrinsic_dim, span)
                triangular_values: list[float] = []
                iid_values: list[float] = []

                for seed in range(config.replications):
                    triangular_rng = np.random.default_rng(
                        10_000 + 100 * ambient_dim + 10 * n + seed
                    )
                    iid_rng = np.random.default_rng(
                        20_000 + 100 * ambient_dim + 10 * n + seed
                    )
                    mixture_rng = np.random.default_rng(
                        30_000 + 100 * ambient_dim + 10 * n + seed
                    )

                    triangular_sample = embedded_uniform_window_sample(
                        n, ambient_dim, intrinsic_dim, span, triangular_rng
                    )
                    iid_index = iid_rng.integers(0, n, size=n)
                    iid_sample = means[iid_index].copy()
                    iid_sample[:, :intrinsic_dim] += iid_rng.uniform(
                        -1.0, 1.0, size=(n, intrinsic_dim)
                    )
                    mixture_index = mixture_rng.integers(0, n, size=n)
                    mixture_sample = means[mixture_index].copy()
                    mixture_sample[:, :intrinsic_dim] += mixture_rng.uniform(
                        -1.0, 1.0, size=(n, intrinsic_dim)
                    )

                    triangular_values.append(
                        exact_empirical_w2(triangular_sample, mixture_sample)
                    )
                    iid_values.append(exact_empirical_w2(iid_sample, mixture_sample))

                triangular_means.append(float(np.mean(triangular_values)))
                iid_means.append(float(np.mean(iid_values)))

            triangular_slope = estimate_log_slope(sample_sizes, triangular_means)
            iid_slope = estimate_log_slope(sample_sizes, iid_means)

            summary_rows.append(
                {
                    "experiment": "useful-fixed-span",
                    "setting": f"triangular ambient d={ambient_dim}, intrinsic k={intrinsic_dim}, span={span:.2f}",
                    "estimated_slope": round(triangular_slope, 4),
                    "carrier_a": round(-triangular_slope, 4),
                    "comment": "fixed-span inheritance check",
                }
            )
            summary_rows.append(
                {
                    "experiment": "useful-fixed-span",
                    "setting": f"iid mixture ambient d={ambient_dim}, intrinsic k={intrinsic_dim}, span={span:.2f}",
                    "estimated_slope": round(iid_slope, 4),
                    "carrier_a": round(-iid_slope, 4),
                    "comment": "mixture benchmark",
                }
            )

            for n, tri_mean, iid_mean in zip(
                sample_sizes, triangular_means, iid_means, strict=True
            ):
                curve_rows.append(
                    {
                        "experiment": "useful-fixed-span",
                        "setting": f"triangular ambient d={ambient_dim}, intrinsic k={intrinsic_dim}, span={span:.2f}",
                        "ambient_dim": ambient_dim,
                        "intrinsic_dim": intrinsic_dim,
                        "span": span,
                        "sample_size": int(n),
                        "mean_w2": round(tri_mean, 6),
                    }
                )
                curve_rows.append(
                    {
                        "experiment": "useful-fixed-span",
                        "setting": f"iid mixture ambient d={ambient_dim}, intrinsic k={intrinsic_dim}, span={span:.2f}",
                        "ambient_dim": ambient_dim,
                        "intrinsic_dim": intrinsic_dim,
                        "span": span,
                        "sample_size": int(n),
                        "mean_w2": round(iid_mean, 6),
                    }
                )

    return UsefulCarrierResult(summary_rows=summary_rows, curve_rows=curve_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run useful-layer carrier sweeps.")
    parser.add_argument(
        "--csv-dir", type=Path, default=Path("artifacts/csv/glue_theorem_useful")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_useful_carrier_research()
    export_rows_csv(
        result.summary_rows, args.csv_dir / "glue_theorem_useful_summary.csv"
    )
    export_rows_csv(result.curve_rows, args.csv_dir / "glue_theorem_useful_curves.csv")


if __name__ == "__main__":
    main()
