from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .online_horizon_adaptation import OnlineAdaptationConfig, phase_profile


@dataclass(frozen=True, slots=True)
class WeightSensitivityRow:
    scheme: str
    best_window: int
    mean_absolute_error: float
    w1_to_uniform: float


@dataclass(frozen=True, slots=True)
class WeightSensitivityResult:
    config: OnlineAdaptationConfig
    rows: tuple[WeightSensitivityRow, ...]


def _normalize(weights: np.ndarray) -> np.ndarray:
    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError("weights must sum to a positive value")
    return weights / total


def lag_weights(window: int, scheme: str) -> np.ndarray:
    if window <= 0:
        raise ValueError("window must be positive")
    if scheme == "uniform":
        return np.ones(window, dtype=float)
    if scheme == "triangular":
        return np.arange(1.0, window + 1.0, dtype=float)
    if scheme == "geometric":
        return np.geomspace(1.0, 0.2, window, dtype=float)[::-1]
    raise ValueError(f"unknown scheme: {scheme}")


def lag_weight_w1(weights: np.ndarray) -> float:
    p = _normalize(np.asarray(weights, dtype=float))
    q = np.full(p.size, 1.0 / p.size, dtype=float)
    return float(np.sum(np.abs(np.cumsum(p - q)[:-1])))


def weighted_moving_average(
    values: np.ndarray, end_index: int, weights: np.ndarray
) -> float:
    weights = _normalize(np.asarray(weights, dtype=float))
    usable = min(weights.size, end_index + 1)
    block = values[end_index - usable + 1 : end_index + 1]
    return float(np.dot(block, _normalize(weights[-usable:])))


def run_weight_sensitivity_experiment(
    config: OnlineAdaptationConfig | None = None,
) -> WeightSensitivityResult:
    cfg = config or OnlineAdaptationConfig()
    latent_mean, _, _ = phase_profile(cfg)
    rng = np.random.default_rng(cfg.seed)
    observations = latent_mean + rng.normal(
        scale=cfg.observation_scale, size=latent_mean.size
    )
    valid = np.arange(cfg.warmup, latent_mean.size)

    rows: list[WeightSensitivityRow] = []
    for scheme in ("uniform", "triangular", "geometric"):
        best_window = min(cfg.candidate_windows)
        best_error = float("inf")
        for window in cfg.candidate_windows:
            estimates = np.full(latent_mean.size, np.nan, dtype=float)
            weights = lag_weights(window, scheme)
            for t in valid:
                estimates[t] = weighted_moving_average(observations, t, weights)
            error = float(np.mean(np.abs(latent_mean[valid] - estimates[valid])))
            if error < best_error:
                best_error = error
                best_window = window
        rows.append(
            WeightSensitivityRow(
                scheme=scheme,
                best_window=best_window,
                mean_absolute_error=best_error,
                w1_to_uniform=lag_weight_w1(lag_weights(best_window, scheme)),
            )
        )
    return WeightSensitivityResult(config=cfg, rows=tuple(rows))


def save_weight_sensitivity_summary(output_path: Path) -> WeightSensitivityResult:
    result = run_weight_sensitivity_experiment()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["scheme", "best_window", "mean_absolute_error", "w1_to_uniform"]
        )
        for row in result.rows:
            writer.writerow(
                [
                    row.scheme,
                    row.best_window,
                    row.mean_absolute_error,
                    row.w1_to_uniform,
                ]
            )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a weight-sensitivity summary."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/csv/weight_sensitivity/summary.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    save_weight_sensitivity_summary(args.output)


if __name__ == "__main__":
    main()
