from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .carrier_roughness_research import (
    _embedded_uniform_mixture_sample,
    _embedded_uniform_window_sample,
)
from .common import export_rows_csv
from .sinkhorn_jacobian_probe import (
    _centered_logv_jacobian,
    _probe_matrix_stats,
    _sinkhorn_scalings,
)


def _twonn(points: np.ndarray) -> float:
    x = np.asarray(points, float)
    d2 = np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=2)
    np.fill_diagonal(d2, np.inf)
    part = np.partition(d2, 2, axis=1)
    r1 = np.sqrt(part[:, 1])
    r2 = np.sqrt(part[:, 2])
    u = r2 / np.maximum(r1, 1e-15)
    u = u[np.isfinite(u) & (u > 1)]
    return float(len(u) / np.sum(np.log(u)))


@dataclass(slots=True)
class TwoNNModerateBandCheckConfig:
    pairs: tuple[tuple[int, int], ...] = ((8, 1), (8, 2), (12, 1), (12, 2))
    epsilons: tuple[float, ...] = (0.2, 0.3, 0.4, 0.5, 0.65, 0.8)
    sample_sizes: tuple[int, ...] = (32, 64, 128, 256, 384, 512)
    seed_count: int = 16
    span: float = 0.25


@dataclass(slots=True)
class TwoNNModerateBandCheckResult:
    estimate_rows: list[dict[str, str | float]]
    band_rows: list[dict[str, str | float]]


def run_twonn_moderate_band_check(
    config: TwoNNModerateBandCheckConfig | None = None,
) -> TwoNNModerateBandCheckResult:
    if config is None:
        config = TwoNNModerateBandCheckConfig()

    estimate_rows: list[dict[str, str | float]] = []
    bucketed: dict[tuple[int, int], dict[str, list[tuple[np.ndarray, np.ndarray]]]] = (
        defaultdict(lambda: {"triangular": [], "iid": []})
    )

    for ambient_dim, intrinsic_dim in config.pairs:
        for n in config.sample_sizes:
            ests: list[float] = []
            hits: list[int] = []
            for seed in range(config.seed_count):
                x = _embedded_uniform_window_sample(
                    n,
                    ambient_dim,
                    intrinsic_dim,
                    config.span,
                    np.random.default_rng(
                        10_000
                        + 1000 * ambient_dim
                        + 100 * intrinsic_dim
                        + 10 * n
                        + seed
                    ),
                )
                y = _embedded_uniform_mixture_sample(
                    n,
                    ambient_dim,
                    intrinsic_dim,
                    config.span,
                    np.random.default_rng(
                        20_000
                        + 1000 * ambient_dim
                        + 100 * intrinsic_dim
                        + 10 * n
                        + seed
                    ),
                )
                d_hat = _twonn(x) / 2.0
                k_hat = max(1, int(np.rint(d_hat)))
                ests.append(d_hat)
                hits.append(int(k_hat == intrinsic_dim))
                if k_hat in (1, 2):
                    bucketed[(ambient_dim, k_hat)]["triangular"].append((x, y))
                    bucketed[(ambient_dim, k_hat)]["iid"].append(
                        (
                            _embedded_uniform_mixture_sample(
                                n,
                                ambient_dim,
                                intrinsic_dim,
                                config.span,
                                np.random.default_rng(
                                    30_000
                                    + 1000 * ambient_dim
                                    + 100 * intrinsic_dim
                                    + 10 * n
                                    + seed
                                ),
                            ),
                            _embedded_uniform_mixture_sample(
                                n,
                                ambient_dim,
                                intrinsic_dim,
                                config.span,
                                np.random.default_rng(
                                    40_000
                                    + 1000 * ambient_dim
                                    + 100 * intrinsic_dim
                                    + 10 * n
                                    + seed
                                ),
                            ),
                        )
                    )
            estimate_rows.append(
                {
                    "ambient_dim": ambient_dim,
                    "intrinsic_dim": intrinsic_dim,
                    "sample_size": n,
                    "mean_twonn_half": round(float(np.mean(ests)), 8),
                    "median_twonn_half": round(float(np.median(ests)), 8),
                    "sd_twonn_half": round(float(np.std(ests)), 8),
                    "rounded_accuracy": round(float(np.mean(hits)), 8),
                }
            )

    band_rows: list[dict[str, str | float]] = []
    for (ambient_dim, k_hat), groups in bucketed.items():
        for coupling in ("triangular", "iid"):
            values_by_epsilon: dict[float, list[float]] = defaultdict(list)
            for left, right in groups[coupling]:
                for epsilon in config.epsilons:
                    a, b, kernel, u, v = _sinkhorn_scalings(
                        left, right, epsilon, max_iters=800, tol=1e-12
                    )
                    jac = _centered_logv_jacobian(a, b, kernel, u, v)
                    values_by_epsilon[epsilon].append(
                        _probe_matrix_stats(jac)["inverse_norm"]
                    )

            if not values_by_epsilon:
                continue
            band_rows.append(
                {
                    "ambient_dim": ambient_dim,
                    "estimated_k": k_hat,
                    "coupling": coupling,
                    "max_inverse_norm_over_band": round(
                        max(
                            float(np.mean(vals)) for vals in values_by_epsilon.values()
                        ),
                        8,
                    ),
                    "mean_inverse_norm_over_band": round(
                        float(
                            np.mean(
                                [
                                    float(np.mean(vals))
                                    for vals in values_by_epsilon.values()
                                ]
                            )
                        ),
                        8,
                    ),
                    "epsilon_count": len(values_by_epsilon),
                }
            )

    return TwoNNModerateBandCheckResult(
        estimate_rows=estimate_rows, band_rows=band_rows
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check moderate-band Sinkhorn stability with TwoNN-estimated k."
    )
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=Path("artifacts/csv/twonn_moderate_band_check"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_twonn_moderate_band_check()
    export_rows_csv(result.estimate_rows, args.csv_dir / "estimates.csv")
    export_rows_csv(result.band_rows, args.csv_dir / "bands.csv")


if __name__ == "__main__":
    main()
