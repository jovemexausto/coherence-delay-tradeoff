from __future__ import annotations

import argparse
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


@dataclass(slots=True)
class ModerateBandProbeConfig:
    pairs: tuple[tuple[int, int], ...] = ((8, 2), (12, 2))
    epsilons: tuple[float, ...] = (0.2, 0.3, 0.4, 0.5, 0.65, 0.8)
    sample_sizes: tuple[int, ...] = (32, 64, 128, 256, 384, 512)
    seed_count: int = 16
    span: float = 0.25


def _sample_pair(
    n: int,
    ambient_dim: int,
    intrinsic_dim: int,
    span: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    x = _embedded_uniform_window_sample(
        n,
        ambient_dim,
        intrinsic_dim,
        span,
        np.random.default_rng(
            10_000 + 1000 * ambient_dim + 100 * intrinsic_dim + 10 * n + seed
        ),
    )
    y = _embedded_uniform_mixture_sample(
        n,
        ambient_dim,
        intrinsic_dim,
        span,
        np.random.default_rng(
            20_000 + 1000 * ambient_dim + 100 * intrinsic_dim + 10 * n + seed
        ),
    )
    return x, y


def run_moderate_band_probe(
    config: ModerateBandProbeConfig | None = None,
) -> tuple[list[dict[str, str | float]], list[dict[str, str | float]]]:
    if config is None:
        config = ModerateBandProbeConfig()

    detailed_rows: list[dict[str, str | float]] = []
    summary_rows: list[dict[str, str | float]] = []

    for ambient_dim, intrinsic_dim in config.pairs:
        pair_rows: list[dict[str, str | float]] = []
        for n in config.sample_sizes:
            for epsilon in config.epsilons:
                stats_by_coupling: dict[str, list[dict[str, float]]] = {
                    "xy": [],
                    "xx": [],
                    "yy": [],
                }
                for seed in range(config.seed_count):
                    x, y = _sample_pair(
                        n, ambient_dim, intrinsic_dim, config.span, seed
                    )
                    for label, left, right in (
                        ("xy", x, y),
                        ("xx", x, x),
                        ("yy", y, y),
                    ):
                        a, b, kernel, u, v = _sinkhorn_scalings(
                            left, right, epsilon, max_iters=800, tol=1e-12
                        )
                        jacobian = _centered_logv_jacobian(a, b, kernel, u, v)
                        stats_by_coupling[label].append(_probe_matrix_stats(jacobian))

                for coupling, stats in stats_by_coupling.items():
                    row = {
                        "ambient_dim": ambient_dim,
                        "intrinsic_dim": intrinsic_dim,
                        "sample_size": n,
                        "epsilon": epsilon,
                        "coupling": coupling,
                        "mean_spectral_radius": round(
                            float(np.mean([s["spectral_radius"] for s in stats])), 8
                        ),
                        "max_spectral_radius": round(
                            float(np.max([s["spectral_radius"] for s in stats])), 8
                        ),
                        "mean_inverse_norm": round(
                            float(np.mean([s["inverse_norm"] for s in stats])), 8
                        ),
                        "max_inverse_norm": round(
                            float(np.max([s["inverse_norm"] for s in stats])), 8
                        ),
                        "min_spectral_gap": round(
                            float(np.min([s["spectral_gap"] for s in stats])), 8
                        ),
                    }
                    detailed_rows.append(row)
                    pair_rows.append(row)

        for coupling in ("xy", "xx", "yy"):
            rows = [row for row in pair_rows if row["coupling"] == coupling]
            summary_rows.append(
                {
                    "ambient_dim": ambient_dim,
                    "intrinsic_dim": intrinsic_dim,
                    "coupling": coupling,
                    "worst_max_spectral_radius": round(
                        max(float(row["max_spectral_radius"]) for row in rows), 8
                    ),
                    "worst_max_inverse_norm": round(
                        max(float(row["max_inverse_norm"]) for row in rows), 8
                    ),
                    "worst_min_spectral_gap": round(
                        min(float(row["min_spectral_gap"]) for row in rows), 8
                    ),
                    "largest_n_mean_inverse_norm": round(
                        float(
                            np.mean(
                                [
                                    float(row["mean_inverse_norm"])
                                    for row in rows
                                    if int(row["sample_size"])
                                    == max(config.sample_sizes)
                                ]
                            )
                        ),
                        8,
                    ),
                }
            )

    return detailed_rows, summary_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe moderate-band Sinkhorn Jacobians."
    )
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=Path("artifacts/csv/sinkhorn_moderate_band_probe"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detailed_rows, summary_rows = run_moderate_band_probe()
    export_rows_csv(detailed_rows, args.csv_dir / "detailed.csv")
    export_rows_csv(summary_rows, args.csv_dir / "summary.csv")


if __name__ == "__main__":
    main()
