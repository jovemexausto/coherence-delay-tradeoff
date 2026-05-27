from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from temporalbridge.core.alarms import calibrate_alarms, detect_alarms


@dataclass(frozen=True)
class ConformalBenchmarkConfig:
    calibration_windows: tuple[int, ...] = (8, 12, 16, 24, 32, 48, 64, 96)
    repetitions: int = 16
    steps: int = 240
    switch_step: int = 120
    alpha: float = 0.1
    beta0: float = 1.0
    drift_rate: float = 0.012
    noise_sigma: float = 0.35
    alarm_window: int = 8
    alarm_persistence: int = 3
    coverage_slack: float = 0.02
    useful_tol: float = 0.05
    score_penalty: float = 18.0


@dataclass(frozen=True)
class ConformalWindowRow:
    seed: int
    calibration_window: int
    mean_coverage: float
    target_coverage: float
    coverage_gap: float
    mean_width: float
    score: float
    safe: bool
    useful: bool
    coverage_crossing_step: int
    alarm_step: int
    coverage_before_alarm_gap: int


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 0:
        raise ValueError("window must be positive")
    series = np.asarray(values, dtype=float)
    if series.size < window:
        return np.array([], dtype=float)
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(series, kernel, mode="valid")


def _split_conformal_radius(abs_residuals: np.ndarray, alpha: float) -> float:
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    values = np.sort(np.asarray(abs_residuals, dtype=float))
    if values.size == 0:
        raise ValueError("abs_residuals must be non-empty")
    rank = int(np.ceil((values.size + 1) * (1.0 - alpha)))
    rank = min(max(rank, 1), values.size)
    return float(values[rank - 1])


def _simulate_stream(
    seed: int, config: ConformalBenchmarkConfig
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=config.steps)
    drift = np.maximum(0.0, np.arange(config.steps, dtype=float) - config.switch_step)
    beta = config.beta0 + config.drift_rate * drift
    noise = rng.normal(scale=config.noise_sigma, size=config.steps)
    y = beta * x + noise
    y_hat = config.beta0 * x
    return y_hat, y - y_hat


def _alarm_step(
    calibration_abs: np.ndarray,
    post_abs: np.ndarray,
    *,
    alarm_window: int,
    alarm_persistence: int,
) -> int:
    calib_series = _rolling_mean(calibration_abs, alarm_window)
    post_series = _rolling_mean(post_abs, alarm_window)
    if calib_series.size == 0 or post_series.size == 0:
        return int(post_abs.size)

    calibrated = calibrate_alarms(
        {},
        {"method": "wild", "diagnostic_bootstrap": {"abs_residual_mean": calib_series}},
        {"abs_residual_mean": calib_series},
    )
    detected = detect_alarms(
        {"abs_residual_mean": post_series},
        calibrated["thresholds"],
        persistence_windows=alarm_persistence,
    )
    hits = detected["alarms"]["abs_residual_mean"]
    if not hits:
        return int(post_abs.size)
    return int(hits[0] + alarm_window)


def _coverage_crossing_step(
    post_abs: np.ndarray,
    *,
    radius: float,
    target_coverage: float,
    slack: float,
    window: int,
) -> int:
    coverage_series = _rolling_mean((post_abs <= radius).astype(float), window)
    if coverage_series.size == 0:
        return int(post_abs.size)
    below = np.flatnonzero(coverage_series < (target_coverage - slack))
    if below.size == 0:
        return int(post_abs.size)
    return int(below[0] + window)


def run_conformal_benchmark(
    *,
    config: ConformalBenchmarkConfig = ConformalBenchmarkConfig(),
    rng_seed: int = 0,
) -> dict[str, object]:
    target_coverage = 1.0 - config.alpha
    rows: list[ConformalWindowRow] = []

    for rep in range(config.repetitions):
        _y_hat, residuals = _simulate_stream(rng_seed + rep, config)
        if config.switch_step <= 0 or config.switch_step >= residuals.size:
            raise ValueError("switch_step must lie strictly within the stream")

        post_abs = np.abs(residuals[config.switch_step :])
        for window in config.calibration_windows:
            if window >= config.switch_step:
                continue
            calib_abs = np.abs(
                residuals[config.switch_step - window : config.switch_step]
            )
            radius = _split_conformal_radius(calib_abs, config.alpha)
            mean_coverage = float(np.mean(post_abs <= radius))
            mean_width = float(2.0 * radius)
            coverage_gap = float(max(target_coverage - mean_coverage, 0.0))
            score = float(mean_width + config.score_penalty * coverage_gap)
            coverage_crossing = _coverage_crossing_step(
                post_abs,
                radius=radius,
                target_coverage=target_coverage,
                slack=config.coverage_slack,
                window=config.alarm_window,
            )
            alarm_step = _alarm_step(
                calib_abs,
                post_abs,
                alarm_window=config.alarm_window,
                alarm_persistence=config.alarm_persistence,
            )
            rows.append(
                ConformalWindowRow(
                    seed=rng_seed + rep,
                    calibration_window=window,
                    mean_coverage=mean_coverage,
                    target_coverage=target_coverage,
                    coverage_gap=coverage_gap,
                    mean_width=mean_width,
                    score=score,
                    safe=bool(mean_coverage >= target_coverage - config.coverage_slack),
                    useful=False,
                    coverage_crossing_step=coverage_crossing,
                    alarm_step=alarm_step,
                    coverage_before_alarm_gap=int(alarm_step - coverage_crossing),
                )
            )

    df = pd.DataFrame([asdict(row) for row in rows])
    window_summary = (
        df.groupby("calibration_window", as_index=False)
        .agg(
            mean_coverage=("mean_coverage", "mean"),
            target_coverage=("target_coverage", "mean"),
            coverage_gap=("coverage_gap", "mean"),
            mean_width=("mean_width", "mean"),
            score=("score", "mean"),
            safe_rate=("safe", "mean"),
            mean_coverage_crossing_step=("coverage_crossing_step", "mean"),
            mean_alarm_step=("alarm_step", "mean"),
            mean_coverage_before_alarm_gap=("coverage_before_alarm_gap", "mean"),
        )
        .sort_values("calibration_window")
        .reset_index(drop=True)
    )

    best_idx = int(window_summary["score"].idxmin())
    best_row = window_summary.iloc[best_idx]
    best_window = int(best_row["calibration_window"])
    min_score = float(best_row["score"])
    max_coverage = float(window_summary["mean_coverage"].max())
    useful_windows = window_summary[
        window_summary["score"] <= min_score * (1.0 + config.useful_tol)
    ]
    safe_windows = window_summary[
        window_summary["mean_coverage"] >= max_coverage - config.coverage_slack
    ]
    useful_set = set(int(value) for value in useful_windows["calibration_window"])
    safe_set = set(int(value) for value in safe_windows["calibration_window"])
    overlap = sorted(useful_set & safe_set)

    ordered_windows = list(int(value) for value in window_summary["calibration_window"])
    best_pos = ordered_windows.index(best_window)
    if 0 < best_pos < len(ordered_windows) - 1:
        left_slope = float(
            window_summary.iloc[best_pos]["score"]
            - window_summary.iloc[best_pos - 1]["score"]
        )
        right_slope = float(
            window_summary.iloc[best_pos + 1]["score"]
            - window_summary.iloc[best_pos]["score"]
        )
    else:
        left_slope = float("nan")
        right_slope = float("nan")

    best_rows = df[df["calibration_window"] == best_window]
    summary = {
        "target_coverage": target_coverage,
        "best_window": best_window,
        "best_score": min_score,
        "best_window_is_safe": bool(best_window in safe_set),
        "best_window_is_useful": bool(best_window in useful_set),
        "coverage_plateau": max_coverage,
        "safe_coverage_floor": max_coverage - config.coverage_slack,
        "safe_useful_overlap_fraction": float(
            len(overlap) / max(len(safe_set | useful_set), 1)
        ),
        "mean_coverage_before_alarm_gap": float(
            best_rows["coverage_before_alarm_gap"].mean()
        ),
        "positive_gap_rate": float(np.mean(best_rows["coverage_before_alarm_gap"] > 0)),
        "mean_alarm_step": float(best_rows["alarm_step"].mean()),
        "mean_coverage_crossing_step": float(
            best_rows["coverage_crossing_step"].mean()
        ),
        "best_score_left_slope": left_slope,
        "best_score_right_slope": right_slope,
        "u_curve": bool(
            np.isfinite(left_slope) and left_slope < 0.0 and right_slope > 0.0
        ),
        "safe_windows": sorted(safe_set),
        "useful_windows": sorted(useful_set),
        "overlap_windows": overlap,
    }
    return {
        "rows": [asdict(row) for row in rows],
        "window_summary": window_summary.to_dict(orient="records"),
        "summary": summary,
    }
