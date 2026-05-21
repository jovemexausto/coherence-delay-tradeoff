from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _logsumexp(z: np.ndarray) -> float:
    z = np.asarray(z, dtype=float)
    m = float(np.max(z))
    return m + float(np.log(np.sum(np.exp(z - m))))


def _softmax(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    m = float(np.max(z))
    e = np.exp(z - m)
    return e / float(np.sum(e))


def smooth_functional(z: np.ndarray) -> float:
    return _logsumexp(z)


def smooth_functional_gradient(z: np.ndarray) -> np.ndarray:
    return _softmax(z)


def drift_mean(step: int, total_steps: int, H: float, zeta: float) -> np.ndarray:
    if step < 0:
        raise ValueError("step must be nonnegative")
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    frac = step / max(total_steps - 1, 1)
    scale = zeta * (frac**H)
    return scale * np.array([1.0, 0.5, -0.25], dtype=float)


def sample_path(
    total_steps: int,
    H: float,
    zeta: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    means = np.vstack(
        [drift_mean(step, total_steps, H, zeta) for step in range(total_steps)]
    )
    samples = means + rng.normal(size=means.shape)
    return means, samples


def window_slice(anchor: int, window: int) -> slice:
    if window <= 0:
        raise ValueError("window must be positive")
    if anchor < window - 1:
        raise ValueError("anchor too small for window")
    return slice(anchor - window + 1, anchor + 1)


def window_average(array: np.ndarray, anchor: int, window: int) -> np.ndarray:
    slc = window_slice(anchor, window)
    return np.mean(array[slc], axis=0)


def linearization_residual(
    estimate: np.ndarray,
    target: np.ndarray,
) -> float:
    estimate = np.asarray(estimate, dtype=float)
    target = np.asarray(target, dtype=float)
    linear_term = float(smooth_functional_gradient(target) @ (estimate - target))
    return float(
        abs(smooth_functional(estimate) - smooth_functional(target) - linear_term)
    )


def quadratic_remainder_bound(
    estimate: np.ndarray,
    target: np.ndarray,
    multiplier: float = 1.0,
) -> float:
    estimate = np.asarray(estimate, dtype=float)
    target = np.asarray(target, dtype=float)
    if multiplier <= 0.0:
        raise ValueError("multiplier must be positive")
    delta = estimate - target
    return float(multiplier * float(np.dot(delta, delta)))


@dataclass(frozen=True, slots=True)
class MultivariateTransferProxyRow:
    window: int
    mean_total_error: float
    mean_finite_sample_error: float
    mean_drift_error: float
    mean_linearization_residual: float


@dataclass(frozen=True, slots=True)
class MultivariateTransferProxyResult:
    rows: tuple[MultivariateTransferProxyRow, ...]


def run_multivariate_transfer_proxy_experiment(
    total_steps: int = 192,
    H: float = 0.75,
    zeta: float = 1.6,
    windows: tuple[int, ...] = (8, 16, 32, 64),
    seed_count: int = 16,
) -> MultivariateTransferProxyResult:
    if total_steps <= max(windows):
        raise ValueError("total_steps must exceed the largest window")
    anchor = total_steps - 1
    rows: list[MultivariateTransferProxyRow] = []
    for window in windows:
        total_errors = []
        finite_errors = []
        drift_errors = []
        residuals = []
        for seed in range(seed_count):
            means, samples = sample_path(total_steps, H, zeta, seed)
            target = means[anchor]
            window_target = window_average(means, anchor, window)
            sample_target = window_average(samples, anchor, window)
            total_errors.append(
                abs(smooth_functional(sample_target) - smooth_functional(target))
            )
            finite_errors.append(
                abs(smooth_functional(sample_target) - smooth_functional(window_target))
            )
            drift_errors.append(
                abs(smooth_functional(window_target) - smooth_functional(target))
            )
            residuals.append(linearization_residual(sample_target, window_target))
        rows.append(
            MultivariateTransferProxyRow(
                window=window,
                mean_total_error=float(np.mean(total_errors)),
                mean_finite_sample_error=float(np.mean(finite_errors)),
                mean_drift_error=float(np.mean(drift_errors)),
                mean_linearization_residual=float(np.mean(residuals)),
            )
        )
    return MultivariateTransferProxyResult(rows=tuple(rows))


def save_multivariate_transfer_proxy_summary(
    output_path: Path,
) -> MultivariateTransferProxyResult:
    result = run_multivariate_transfer_proxy_experiment()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "window",
                "mean_total_error",
                "mean_finite_sample_error",
                "mean_drift_error",
                "mean_linearization_residual",
            ]
        )
        for row in result.rows:
            writer.writerow(
                [
                    row.window,
                    row.mean_total_error,
                    row.mean_finite_sample_error,
                    row.mean_drift_error,
                    row.mean_linearization_residual,
                ]
            )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate multivariate transfer proxy summary."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/csv/multivariate_transfer_proxy/summary.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    save_multivariate_transfer_proxy_summary(args.output)


if __name__ == "__main__":
    main()
