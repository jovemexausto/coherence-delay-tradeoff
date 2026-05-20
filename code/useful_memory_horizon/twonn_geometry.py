from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from .common import export_rows_csv
from .twonn_dimension import twonn_dimension_estimate


@dataclass(frozen=True, slots=True)
class TwonnGeometryConfig:
    ambient_dim: int = 8
    intrinsic_dim: int = 1
    holder_exponents: tuple[float, ...] = (0.5, 0.75)
    roughness_scale: float = 0.5
    time_steps: int = 300
    sample_size_per_time: int = 256
    point_noise_scale: float = 0.15
    lag_values: tuple[int, ...] = (4, 8, 16, 32)
    history: int = 220
    path_seed_count: int = 4


@dataclass(frozen=True, slots=True)
class TwonnGeometryRow:
    holder_exponent: float
    median_k_hat: float
    naive_holder_mae: float
    aggregated_holder_mae: float
    median_naive_holder: float
    median_aggregated_holder: float


@dataclass(frozen=True, slots=True)
class TwonnGeometryResult:
    config: TwonnGeometryConfig
    rows: tuple[TwonnGeometryRow, ...]


@lru_cache(maxsize=None)
def _sqrt_fbm_covariance(time_steps: int, holder_exponent: float) -> np.ndarray:
    t = np.arange(1, time_steps + 1, dtype=float)
    covariance = np.empty((time_steps, time_steps), dtype=float)
    for i, s in enumerate(t):
        for j, u in enumerate(t):
            covariance[i, j] = 0.5 * (
                s ** (2.0 * holder_exponent)
                + u ** (2.0 * holder_exponent)
                - abs(u - s) ** (2.0 * holder_exponent)
            )
    covariance += 1e-6 * np.eye(time_steps)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    return eigenvectors @ np.diag(np.sqrt(eigenvalues))


def _embedded_cloud(
    center: float,
    ambient_dim: int,
    intrinsic_dim: int,
    sample_size: int,
    point_noise_scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    intrinsic = rng.normal(scale=point_noise_scale, size=(sample_size, intrinsic_dim))
    intrinsic[:, 0] += center
    cloud = np.zeros((sample_size, ambient_dim), dtype=float)
    cloud[:, :intrinsic_dim] = intrinsic
    return cloud


def _embedded_stream(
    config: TwonnGeometryConfig,
    holder_exponent: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    path = (
        config.roughness_scale
        * (
            _sqrt_fbm_covariance(config.time_steps, holder_exponent)
            @ rng.standard_normal(config.time_steps)
        )
        / (config.time_steps**holder_exponent)
    )
    clouds = np.empty(
        (config.time_steps, config.sample_size_per_time, config.ambient_dim),
        dtype=float,
    )
    barycenters = np.empty((config.time_steps, config.ambient_dim), dtype=float)
    for t, center in enumerate(path):
        cloud = _embedded_cloud(
            center=float(center),
            ambient_dim=config.ambient_dim,
            intrinsic_dim=config.intrinsic_dim,
            sample_size=config.sample_size_per_time,
            point_noise_scale=config.point_noise_scale,
            rng=rng,
        )
        clouds[t] = cloud
        barycenters[t] = cloud.mean(axis=0)
    return clouds, barycenters


def _holder_from_logfit(lags: np.ndarray, values: np.ndarray) -> float:
    slope, _ = np.polyfit(np.log(lags), np.log(np.maximum(values, 1e-12)), 1)
    return float(slope)


def naive_holder_estimate(
    barycenters: np.ndarray,
    time_index: int,
    lag_values: tuple[int, ...],
) -> float:
    recent = barycenters[time_index]
    discrepancies = np.asarray(
        [np.linalg.norm(recent - barycenters[time_index - lag]) for lag in lag_values],
        dtype=float,
    )
    return _holder_from_logfit(np.asarray(lag_values, dtype=float), discrepancies)


def aggregated_holder_estimate(
    barycenters: np.ndarray,
    time_index: int,
    lag_values: tuple[int, ...],
    history: int,
) -> float:
    lag_array = np.asarray(lag_values, dtype=float)
    summaries = np.empty(lag_array.size, dtype=float)
    for index, lag in enumerate(lag_values):
        start = max(lag, time_index - history + 1)
        discrepancies = np.asarray(
            [
                np.linalg.norm(barycenters[s] - barycenters[s - lag])
                for s in range(start, time_index + 1)
            ],
            dtype=float,
        )
        summaries[index] = float(np.mean(discrepancies))
    return _holder_from_logfit(lag_array, summaries)


def run_twonn_geometry_experiment(
    config: TwonnGeometryConfig | None = None,
) -> TwonnGeometryResult:
    cfg = config or TwonnGeometryConfig()
    rows: list[TwonnGeometryRow] = []
    for holder_exponent in cfg.holder_exponents:
        k_estimates: list[float] = []
        naive_errors: list[float] = []
        aggregated_errors: list[float] = []
        naive_holders: list[float] = []
        aggregated_holders: list[float] = []
        for seed in range(cfg.path_seed_count):
            clouds, barycenters = _embedded_stream(cfg, holder_exponent, seed)
            time_index = cfg.time_steps - 1
            k_estimates.append(twonn_dimension_estimate(clouds[time_index]))
            naive_holder = naive_holder_estimate(
                barycenters, time_index, cfg.lag_values
            )
            aggregated_holder = aggregated_holder_estimate(
                barycenters, time_index, cfg.lag_values, cfg.history
            )
            naive_holders.append(naive_holder)
            aggregated_holders.append(aggregated_holder)
            naive_errors.append(abs(naive_holder - holder_exponent))
            aggregated_errors.append(abs(aggregated_holder - holder_exponent))
        rows.append(
            TwonnGeometryRow(
                holder_exponent=holder_exponent,
                median_k_hat=float(np.median(k_estimates)),
                naive_holder_mae=float(np.mean(naive_errors)),
                aggregated_holder_mae=float(np.mean(aggregated_errors)),
                median_naive_holder=float(np.median(naive_holders)),
                median_aggregated_holder=float(np.median(aggregated_holders)),
            )
        )
    return TwonnGeometryResult(config=cfg, rows=tuple(rows))


def build_twonn_geometry_rows(
    result: TwonnGeometryResult,
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for row in result.rows:
        rows.append(
            {
                "holder_exponent": row.holder_exponent,
                "median_k_hat": round(row.median_k_hat, 6),
                "naive_holder_mae": round(row.naive_holder_mae, 6),
                "aggregated_holder_mae": round(row.aggregated_holder_mae, 6),
                "median_naive_holder": round(row.median_naive_holder, 6),
                "median_aggregated_holder": round(row.median_aggregated_holder, 6),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the TwoNN-to-online-geometry diagnostic."
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=Path("artifacts/csv/online/twonn_geometry.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_twonn_geometry_experiment()
    export_rows_csv(build_twonn_geometry_rows(result), args.csv_path)
    for row in result.rows:
        print(
            f"H={row.holder_exponent:.2f}  "
            f"k_hat~{row.median_k_hat:.2f}  "
            f"naive_mae={row.naive_holder_mae:.3f}  "
            f"agg_mae={row.aggregated_holder_mae:.3f}"
        )


if __name__ == "__main__":
    main()
