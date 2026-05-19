from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .carrier_roughness_research import _embedded_uniform_window_sample
from .common import export_rows_csv
from .operational_region_thresholds import maximal_stable_epsilon_band
from .operational_regime_frontier import map_operational_regime
from .twonn_dimension import twonn_dimension_estimate


@dataclass(frozen=True, slots=True)
class TwoNNOperationalDiagnosticConfig:
    pairs: tuple[tuple[int, int], ...] = (
        (8, 1),
        (8, 2),
        (8, 3),
        (12, 1),
        (12, 2),
        (12, 3),
        (16, 1),
        (16, 2),
        (16, 3),
    )
    epsilons: tuple[float, ...] = (0.8, 0.5, 0.3, 0.2, 0.1, 0.05)
    frontier_sample_sizes: tuple[int, ...] = (24, 48, 96, 160)
    frontier_seed_count: int = 12
    twonn_sample_size: int = 256
    twonn_seed_count: int = 4
    span: float = 0.25
    cut_epsilon: float = 0.2


@dataclass(frozen=True, slots=True)
class TwoNNOperationalDiagnosticResult:
    pair_rows: list[dict[str, str | float]]
    comparison_rows: list[dict[str, str | float]]


def _sample_pair(
    n: int,
    ambient_dim: int,
    intrinsic_dim: int,
    span: float,
    seed: int,
) -> np.ndarray:
    return _embedded_uniform_window_sample(
        n,
        ambient_dim,
        intrinsic_dim,
        span,
        np.random.default_rng(
            50_000 + 1000 * ambient_dim + 100 * intrinsic_dim + 10 * n + seed
        ),
    )


def _loo_linear_predictions(features: np.ndarray, target: np.ndarray) -> np.ndarray:
    if features.ndim != 1 or target.ndim != 1:
        raise ValueError("features and target must be one-dimensional")
    if features.size != target.size:
        raise ValueError("features and target must have the same length")
    if features.size < 3:
        raise ValueError("at least three points are required for leave-one-out")

    predictions: list[float] = []
    for index in range(features.size):
        mask = np.ones(features.size, dtype=bool)
        mask[index] = False
        x = features[mask]
        y = target[mask]
        design = np.column_stack([x, np.ones_like(x)])
        slope, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
        predictions.append(float(slope * features[index] + intercept))
    return np.asarray(predictions, dtype=float)


def _pairwise_mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def _cut_accuracy(
    actual_targets: np.ndarray,
    predicted_targets: np.ndarray,
    cut_epsilon: float,
) -> float:
    actual_positive = actual_targets >= cut_epsilon - 1e-12
    predicted_positive = predicted_targets >= cut_epsilon - 1e-12
    return float(np.mean(actual_positive == predicted_positive))


def compare_twonn_to_ambient_on_operational_frontier(
    config: TwoNNOperationalDiagnosticConfig | None = None,
) -> TwoNNOperationalDiagnosticResult:
    if config is None:
        config = TwoNNOperationalDiagnosticConfig()

    frontier_rows = map_operational_regime(
        ambient_intrinsic_pairs=config.pairs,
        epsilons=config.epsilons,
        sample_sizes=config.frontier_sample_sizes,
        seed_count=config.frontier_seed_count,
    )
    pair_rows: list[dict[str, str | float]] = []
    ambient_features: list[float] = []
    twonn_features: list[float] = []
    epsilon_targets: list[float] = []

    for ambient_dim, intrinsic_dim in config.pairs:
        estimates: list[float] = []
        for seed in range(config.twonn_seed_count):
            sample = _sample_pair(
                config.twonn_sample_size,
                ambient_dim,
                intrinsic_dim,
                config.span,
                seed,
            )
            estimates.append(twonn_dimension_estimate(sample))

        epsilon_max = maximal_stable_epsilon_band(
            frontier_rows, ambient_dim=ambient_dim, intrinsic_dim=intrinsic_dim
        )
        epsilon_max_value = 0.0 if epsilon_max is None else float(epsilon_max)
        ambient_features.append(float(ambient_dim))
        twonn_features.append(float(np.mean(estimates)))
        epsilon_targets.append(epsilon_max_value)
        pair_rows.append(
            {
                "ambient_dim": ambient_dim,
                "intrinsic_dim": intrinsic_dim,
                "twonn_k_hat": round(float(np.mean(estimates)), 6),
                "twonn_k_hat_std": round(float(np.std(estimates)), 6),
                "epsilon_max": round(epsilon_max_value, 6),
                "epsilon_max_defined": epsilon_max is not None,
            }
        )

    ambient_features_array = np.asarray(ambient_features, dtype=float)
    twonn_features_array = np.asarray(twonn_features, dtype=float)
    epsilon_targets_array = np.asarray(epsilon_targets, dtype=float)

    ambient_predictions = _loo_linear_predictions(
        ambient_features_array, epsilon_targets_array
    )
    twonn_predictions = _loo_linear_predictions(
        twonn_features_array, epsilon_targets_array
    )

    comparison_rows = [
        {
            "feature": "ambient_dim",
            "loo_mae_epsilon_max": round(
                _pairwise_mae(epsilon_targets_array, ambient_predictions), 6
            ),
            "stability_cut_accuracy": round(
                _cut_accuracy(
                    epsilon_targets_array, ambient_predictions, config.cut_epsilon
                ),
                6,
            ),
        },
        {
            "feature": "twonn_k_hat",
            "loo_mae_epsilon_max": round(
                _pairwise_mae(epsilon_targets_array, twonn_predictions), 6
            ),
            "stability_cut_accuracy": round(
                _cut_accuracy(
                    epsilon_targets_array, twonn_predictions, config.cut_epsilon
                ),
                6,
            ),
        },
    ]
    return TwoNNOperationalDiagnosticResult(
        pair_rows=pair_rows,
        comparison_rows=comparison_rows,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare TwoNN intrinsic dimension to ambient dimension."
    )
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=Path("artifacts/csv/twonn_operational_diagnostic"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = compare_twonn_to_ambient_on_operational_frontier()
    export_rows_csv(result.pair_rows, args.csv_dir / "pair_rows.csv")
    export_rows_csv(result.comparison_rows, args.csv_dir / "comparison_rows.csv")


if __name__ == "__main__":
    main()
