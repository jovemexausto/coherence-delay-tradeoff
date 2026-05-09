from __future__ import annotations

from dataclasses import dataclass, replace
from collections.abc import Callable

import numpy as np

from ..bikes.model import load_bikes_values
from ..core.horizon_baselines import compute_umr_regulator
from ..core.rff_ensemble import (
    prequential_rff_predictions,
    scale_for_miscalibration,
)
from ..elec2.model import load_elec2_values


DATASET_LOADERS: dict[str, Callable[[], np.ndarray]] = {
    "elec2": load_elec2_values,
    "bikes": load_bikes_values,
}


@dataclass(slots=True)
class RajputConfig:
    dataset: str = "elec2"
    context_length: int = 100
    buffer_sizes: tuple[int, ...] = (1, 5, 20, 40, 200)
    num_features: int = 64
    ridge: float = 1.0
    noise_floor: float = 1e-6
    calibration_samples: int = 2000
    seed: int = 42
    umr_block_size: int = 100
    umr_ema_alpha: float = 0.05
    umr_scale: float = 1.25
    umr_baseline_window: int = 100
    umr_min_window: int = 1
    umr_max_window: int = 200


@dataclass(slots=True)
class BufferCalibration:
    buffer_size: int
    alpha: float
    calibration_mae: float
    calibration_miscalibration: float


@dataclass(slots=True)
class RajputResult:
    config: RajputConfig
    values: np.ndarray
    targets: np.ndarray
    feature_matrix: np.ndarray
    buffer_sizes: np.ndarray
    caps: np.ndarray
    calibration_slice: slice
    test_slice: slice
    calibrations: list[BufferCalibration]
    model_means: np.ndarray
    model_stds: np.ndarray
    scaled_model_stds: np.ndarray
    single_index: int
    single_mean: np.ndarray
    single_std: np.ndarray
    naive_mean: np.ndarray
    naive_std: np.ndarray
    uq_mean: np.ndarray
    uq_std: np.ndarray
    uq_umr_mean: np.ndarray
    uq_umr_std: np.ndarray


def _build_supervised_series(
    values: np.ndarray, context_length: int
) -> tuple[np.ndarray, np.ndarray]:
    if context_length <= 0:
        raise ValueError("context_length must be positive")
    windows = np.lib.stride_tricks.sliding_window_view(values, context_length + 1)
    features = np.asarray(windows[:, :-1], dtype=float)
    targets = np.asarray(windows[:, -1], dtype=float)
    return features, targets


def _load_dataset(name: str) -> np.ndarray:
    loader = DATASET_LOADERS.get(name.lower())
    if loader is None:
        raise ValueError(f"Unknown dataset: {name}")
    return loader()


def _ensemble_mean_and_std(means: np.ndarray, stds: np.ndarray) -> tuple[float, float]:
    if means.size == 0:
        return float("nan"), float("nan")
    mean = float(np.mean(means))
    variance = float(np.mean(stds * stds) + np.var(means))
    return mean, float(np.sqrt(max(variance, 1e-12)))


def _weighted_mean_and_std(means: np.ndarray, stds: np.ndarray) -> tuple[float, float]:
    valid = np.isfinite(means) & np.isfinite(stds)
    if not np.any(valid):
        return float("nan"), float("nan")
    means = means[valid]
    stds = np.maximum(stds[valid], 1e-9)
    weights = 1.0 / (stds * stds)
    total = float(np.sum(weights))
    if total <= 0.0:
        return float(np.mean(means)), float(np.std(means))
    mean = float(np.sum(weights * means) / total)
    std = float(np.sqrt(1.0 / total))
    return mean, std


def _calibrate_models(
    features: np.ndarray,
    targets: np.ndarray,
    buffer_sizes: np.ndarray,
    config: RajputConfig,
    calibration_slice: slice,
) -> tuple[np.ndarray, np.ndarray, list[BufferCalibration]]:
    model_means = np.full((buffer_sizes.size, targets.size), np.nan, dtype=float)
    model_stds = np.full_like(model_means, np.nan)
    calibrations: list[BufferCalibration] = []
    for index, buffer_size in enumerate(buffer_sizes):
        preds, stds = prequential_rff_predictions(
            features,
            targets,
            buffer_size=int(buffer_size),
            num_features=config.num_features,
            ridge=config.ridge,
            noise_floor=config.noise_floor,
            seed=config.seed + 100 * index,
        )
        model_means[index] = preds
        model_stds[index] = stds
        cal_y = targets[calibration_slice]
        cal_mean = preds[calibration_slice]
        cal_std = stds[calibration_slice]
        alpha, miscal = scale_for_miscalibration(cal_y, cal_mean, cal_std)
        valid = np.isfinite(cal_y) & np.isfinite(cal_mean)
        cal_mae = (
            float(np.mean(np.abs(cal_y[valid] - cal_mean[valid])))
            if np.any(valid)
            else float("nan")
        )
        calibrations.append(
            BufferCalibration(
                buffer_size=int(buffer_size),
                alpha=alpha,
                calibration_mae=cal_mae,
                calibration_miscalibration=miscal,
            )
        )
    scaled_model_stds = (
        model_stds * np.asarray([c.alpha for c in calibrations], dtype=float)[:, None]
    )
    return model_means, model_stds, scaled_model_stds, calibrations


def _compute_cap_series(
    values: np.ndarray, config: RajputConfig, sample_count: int
) -> np.ndarray:
    regulator = compute_umr_regulator(
        values,
        block_size=config.umr_block_size,
        ema_alpha=config.umr_ema_alpha,
        baseline_window=config.umr_baseline_window,
        prefix_length=min(
            len(values), max(config.calibration_samples, config.context_length + 1)
        ),
        scale=config.umr_scale,
        min_window=config.umr_min_window,
        max_window=config.umr_max_window,
    )
    caps = np.asarray(
        regulator.window_sizes[
            config.context_length : config.context_length + sample_count
        ],
        dtype=float,
    )
    return np.clip(caps, float(config.umr_min_window), float(config.umr_max_window))


def run_rajput_benchmark(config: RajputConfig | None = None) -> RajputResult:
    config = config or RajputConfig()
    values = _load_dataset(config.dataset)
    features, targets = _build_supervised_series(values, config.context_length)

    buffer_sizes = np.asarray(
        sorted({int(size) for size in config.buffer_sizes}), dtype=int
    )
    warmup = int(buffer_sizes.max())
    calibration_end = min(max(config.calibration_samples, warmup + 1), targets.size)
    calibration_slice = slice(warmup, calibration_end)
    test_slice = slice(calibration_end, targets.size)

    model_means, model_stds, scaled_model_stds, calibrations = _calibrate_models(
        features, targets, buffer_sizes, config, calibration_slice
    )

    calibration_maes = np.asarray(
        [c.calibration_mae for c in calibrations], dtype=float
    )
    single_index = int(np.nanargmin(calibration_maes))

    caps = _compute_cap_series(values, config, targets.size)

    single_mean = np.full(targets.size, np.nan, dtype=float)
    single_std = np.full(targets.size, np.nan, dtype=float)
    naive_mean = np.full(targets.size, np.nan, dtype=float)
    naive_std = np.full(targets.size, np.nan, dtype=float)
    uq_mean = np.full(targets.size, np.nan, dtype=float)
    uq_std = np.full(targets.size, np.nan, dtype=float)
    uq_umr_mean = np.full(targets.size, np.nan, dtype=float)
    uq_umr_std = np.full(targets.size, np.nan, dtype=float)

    for index in range(warmup, targets.size):
        available = np.isfinite(model_means[:, index]) & np.isfinite(
            scaled_model_stds[:, index]
        )
        if not np.any(available):
            continue

        selected_means = model_means[:, index][available]
        selected_stds = scaled_model_stds[:, index][available]
        single_mean[index] = float(model_means[single_index, index])
        single_std[index] = float(scaled_model_stds[single_index, index])

        naive_mean[index], naive_std[index] = _ensemble_mean_and_std(
            selected_means, selected_stds
        )
        uq_mean[index], uq_std[index] = _weighted_mean_and_std(
            selected_means, selected_stds
        )

        allowed = available & (buffer_sizes <= caps[index])
        if not np.any(allowed):
            allowed = available
        umr_means = model_means[:, index][allowed]
        umr_stds = scaled_model_stds[:, index][allowed]
        uq_umr_mean[index], uq_umr_std[index] = _weighted_mean_and_std(
            umr_means, umr_stds
        )

    return RajputResult(
        config=config,
        values=values,
        targets=targets,
        feature_matrix=features,
        buffer_sizes=buffer_sizes,
        caps=caps,
        calibration_slice=calibration_slice,
        test_slice=test_slice,
        calibrations=calibrations,
        model_means=model_means,
        model_stds=model_stds,
        scaled_model_stds=scaled_model_stds,
        single_index=single_index,
        single_mean=single_mean,
        single_std=single_std,
        naive_mean=naive_mean,
        naive_std=naive_std,
        uq_mean=uq_mean,
        uq_std=uq_std,
        uq_umr_mean=uq_umr_mean,
        uq_umr_std=uq_umr_std,
    )


def run_rajput_benchmarks(
    config: RajputConfig | None = None,
) -> dict[str, RajputResult]:
    config = config or RajputConfig()
    results: dict[str, RajputResult] = {}
    for dataset in DATASET_LOADERS:
        results[dataset] = run_rajput_benchmark(replace(config, dataset=dataset))
    return results
