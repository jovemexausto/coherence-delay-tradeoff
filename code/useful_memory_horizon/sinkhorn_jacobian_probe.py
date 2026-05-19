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


def _pairwise_squared_distances(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x_norm = np.sum(x * x, axis=1, keepdims=True)
    y_norm = np.sum(y * y, axis=1, keepdims=True).T
    return np.maximum(x_norm + y_norm - 2.0 * x @ y.T, 0.0)


def _sinkhorn_scalings(
    x: np.ndarray,
    y: np.ndarray,
    epsilon: float,
    *,
    max_iters: int = 400,
    tol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n, m = x.shape[0], y.shape[0]
    a = np.full(n, 1.0 / n)
    b = np.full(m, 1.0 / m)
    cost = _pairwise_squared_distances(x, y)
    kernel = np.exp(-(cost - float(np.min(cost))) / epsilon)
    u = np.ones(n)
    v = np.ones(m)
    for _ in range(max_iters):
        u_prev = u.copy()
        u = a / np.maximum(kernel @ v, 1e-300)
        v = b / np.maximum(kernel.T @ u, 1e-300)
        if np.max(np.abs(u - u_prev) / np.maximum(np.abs(u_prev), 1e-12)) < tol:
            break
    return a, b, kernel, u, v


def _center_projector(n: int) -> np.ndarray:
    return np.eye(n) - np.ones((n, n)) / n


def _centered_logv_jacobian(
    a: np.ndarray,
    b: np.ndarray,
    kernel: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
) -> np.ndarray:
    p = np.maximum(kernel @ v, 1e-300)
    q = np.maximum(kernel.T @ u, 1e-300)
    middle = kernel.T @ (np.diag(a / (p**2)) @ (kernel @ np.diag(v)))
    j_raw = np.diag(1.0 / q) @ middle
    p_center = _center_projector(v.shape[0])
    return p_center @ j_raw @ p_center


def _probe_matrix_stats(jacobian: np.ndarray) -> dict[str, float]:
    eigvals = np.linalg.eigvals(jacobian)
    spectral_radius = float(np.max(np.abs(eigvals)))
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    operator_norm = float(np.max(singular_values))
    centered_dim = jacobian.shape[0]
    ident = _center_projector(centered_dim)
    inverse_norm = float(np.linalg.norm(np.linalg.pinv(ident - jacobian), 2))
    gap = float(1.0 - spectral_radius)
    return {
        "spectral_radius": spectral_radius,
        "operator_norm": operator_norm,
        "inverse_norm": inverse_norm,
        "spectral_gap": gap,
    }


@dataclass(slots=True)
class SinkhornJacobianProbeConfig:
    pairs: tuple[tuple[int, int], ...] = ((8, 1), (8, 2), (12, 1), (12, 2))
    epsilons: tuple[float, ...] = (0.05, 0.1, 0.2, 0.4, 0.8)
    sample_sizes: tuple[int, ...] = (32, 64, 128)
    seed_count: int = 12
    span: float = 0.25


def run_sinkhorn_jacobian_probe(
    config: SinkhornJacobianProbeConfig | None = None,
) -> list[dict[str, str | float]]:
    if config is None:
        config = SinkhornJacobianProbeConfig()
    rows: list[dict[str, str | float]] = []
    for ambient_dim, intrinsic_dim in config.pairs:
        for n in config.sample_sizes:
            for epsilon in config.epsilons:
                stats_xy: list[dict[str, float]] = []
                stats_xx: list[dict[str, float]] = []
                stats_yy: list[dict[str, float]] = []
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
                    a, b, kernel, u, v = _sinkhorn_scalings(x, y, epsilon)
                    stats_xy.append(
                        _probe_matrix_stats(_centered_logv_jacobian(a, b, kernel, u, v))
                    )
                    a_xx, b_xx, kernel_xx, u_xx, v_xx = _sinkhorn_scalings(
                        x, x, epsilon
                    )
                    stats_xx.append(
                        _probe_matrix_stats(
                            _centered_logv_jacobian(a_xx, b_xx, kernel_xx, u_xx, v_xx)
                        )
                    )
                    a_yy, b_yy, kernel_yy, u_yy, v_yy = _sinkhorn_scalings(
                        y, y, epsilon
                    )
                    stats_yy.append(
                        _probe_matrix_stats(
                            _centered_logv_jacobian(a_yy, b_yy, kernel_yy, u_yy, v_yy)
                        )
                    )
                for label, stats in (
                    ("xy", stats_xy),
                    ("xx", stats_xx),
                    ("yy", stats_yy),
                ):
                    rows.append(
                        {
                            "ambient_dim": ambient_dim,
                            "intrinsic_dim": intrinsic_dim,
                            "sample_size": n,
                            "epsilon": epsilon,
                            "coupling": label,
                            "mean_spectral_radius": round(
                                float(np.mean([s["spectral_radius"] for s in stats])), 8
                            ),
                            "max_spectral_radius": round(
                                float(np.max([s["spectral_radius"] for s in stats])), 8
                            ),
                            "mean_operator_norm": round(
                                float(np.mean([s["operator_norm"] for s in stats])), 8
                            ),
                            "max_operator_norm": round(
                                float(np.max([s["operator_norm"] for s in stats])), 8
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
                    )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe Jacobian regularity of the Sinkhorn fixed-point map."
    )
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=Path("artifacts/csv/sinkhorn_jacobian_probe"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = run_sinkhorn_jacobian_probe()
    export_rows_csv(rows, args.csv_dir / "summary.csv")


if __name__ == "__main__":
    main()
