from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from river import datasets, drift


@dataclass(slots=True)
class Elec2Config:
    demand_key: str = "nswdemand"
    page_hinkley_delta: float = 0.01
    page_hinkley_threshold: float = 84.0
    page_hinkley_alpha: float = 0.9999
    warning_threshold: float = 0.295
    max_gap: int = 5000
    fixed_window: int = 100
    dynamic_alpha: float = 0.03
    dynamic_window_delta: int = 24
    dynamic_min_window: int = 30
    dynamic_max_window: int = 300
    dynamic_scale: float = 1.25
    dynamic_baseline_window: int = 100
    adwin_delta: float = 0.03


@dataclass(slots=True)
class Elec2DetectionResult:
    sigma: np.ndarray
    estimate: np.ndarray
    window_sizes: np.ndarray
    warnings: list[int]
    matched_warnings: list[int]
    matched_events: list[int]
    lead_times: list[int]


@dataclass(slots=True)
class Elec2ExperimentResult:
    config: Elec2Config
    values: np.ndarray
    events: list[int]
    fixed_100: Elec2DetectionResult
    fixed_50: Elec2DetectionResult
    fixed_300: Elec2DetectionResult
    dynamic: Elec2DetectionResult
    adwin: Elec2DetectionResult
    residual_signal: np.ndarray
    dynamic_drift_estimate: np.ndarray


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same")


def load_elec2_values(config: Elec2Config | None = None) -> np.ndarray:
    config = config or Elec2Config()
    values = np.asarray(
        [features[config.demand_key] for features, _ in datasets.Elec2()],
        dtype=float,
    )
    return (values - values.mean()) / values.std()


def detect_page_hinkley_events(
    values: np.ndarray,
    config: Elec2Config,
) -> list[int]:
    detector = drift.PageHinkley(
        min_instances=30,
        delta=config.page_hinkley_delta,
        threshold=config.page_hinkley_threshold,
        alpha=config.page_hinkley_alpha,
        mode="both",
    )
    events: list[int] = []
    for index, value in enumerate(values):
        detector.update(value)
        if detector.drift_detected:
            events.append(index)
    return events


def _compute_sigma_fixed(
    values: np.ndarray, window: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sigma = np.full(values.size, np.nan)
    estimate = np.full(values.size, np.nan)
    windows = np.full(values.size, float(window))
    for index in range(window - 1, values.size):
        mean_value = float(np.mean(values[index - window + 1 : index + 1]))
        estimate[index] = mean_value
        sigma[index] = 1.0 / (1.0 + 0.5 * (values[index] - mean_value) ** 2)
    return sigma, estimate, windows


def _compute_sigma_dynamic(
    values: np.ndarray,
    config: Elec2Config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sigma = np.full(values.size, np.nan)
    estimate = np.full(values.size, np.nan)
    windows = np.full(values.size, float(config.dynamic_baseline_window))
    zeta_hat = np.zeros(values.size)

    d = config.dynamic_window_delta
    ck = float(
        np.mean(np.abs(values[:2000] - np.mean(values[:2000])))
        * np.sqrt(config.dynamic_baseline_window)
        * config.dynamic_scale
    )
    ema = 0.0

    for index in range(values.size):
        if index >= 2 * d:
            recent_mean = float(np.mean(values[index - d : index]))
            previous_mean = float(np.mean(values[index - 2 * d : index - d]))
            local_drift = abs(recent_mean - previous_mean) / d
            ema = (
                config.dynamic_alpha * local_drift + (1.0 - config.dynamic_alpha) * ema
            )
            zeta_hat[index] = ema
            windows[index] = np.clip(
                (ck / max(ema, 1e-9)) ** (2.0 / 3.0),
                config.dynamic_min_window,
                config.dynamic_max_window,
            )
        window = int(round(windows[index]))
        if index >= window - 1:
            mean_value = float(np.mean(values[index - window + 1 : index + 1]))
            estimate[index] = mean_value
            sigma[index] = 1.0 / (1.0 + 0.5 * (values[index] - mean_value) ** 2)

    return sigma, estimate, windows, zeta_hat


def _extract_warnings(sigma: np.ndarray, threshold: float) -> list[int]:
    warnings: list[int] = []
    below = False
    for index, value in enumerate(sigma):
        if np.isnan(value):
            continue
        if value < threshold and not below:
            warnings.append(index)
            below = True
        elif value >= threshold:
            below = False
    return warnings


def _match_warnings_to_events(
    warnings: list[int],
    events: list[int],
    max_gap: int,
) -> tuple[list[int], list[int], list[int]]:
    matched_warnings: list[int] = []
    matched_events: list[int] = []
    lead_times: list[int] = []
    event_index = 0

    for warning in warnings:
        while event_index < len(events) and events[event_index] < warning:
            event_index += 1
        if event_index < len(events):
            lead_time = events[event_index] - warning
            if lead_time <= max_gap:
                matched_warnings.append(warning)
                matched_events.append(events[event_index])
                lead_times.append(lead_time)
                event_index += 1

    return matched_warnings, matched_events, lead_times


def _build_detection_result(
    sigma: np.ndarray,
    estimate: np.ndarray,
    windows: np.ndarray,
    events: list[int],
    config: Elec2Config,
) -> Elec2DetectionResult:
    warnings = _extract_warnings(sigma, config.warning_threshold)
    matched_warnings, matched_events, lead_times = _match_warnings_to_events(
        warnings,
        events,
        config.max_gap,
    )
    return Elec2DetectionResult(
        sigma=sigma,
        estimate=estimate,
        window_sizes=windows,
        warnings=warnings,
        matched_warnings=matched_warnings,
        matched_events=matched_events,
        lead_times=lead_times,
    )


def _build_detection_result_from_warnings(
    signal: np.ndarray,
    warnings: list[int],
    events: list[int],
    config: Elec2Config,
) -> Elec2DetectionResult:
    matched_warnings, matched_events, lead_times = _match_warnings_to_events(
        warnings,
        events,
        config.max_gap,
    )
    return Elec2DetectionResult(
        sigma=signal,
        estimate=np.full(signal.size, np.nan),
        window_sizes=np.full(signal.size, np.nan),
        warnings=warnings,
        matched_warnings=matched_warnings,
        matched_events=matched_events,
        lead_times=lead_times,
    )


def _detect_adwin_warnings(signal: np.ndarray, delta: float) -> list[int]:
    detector = drift.ADWIN(delta=delta)
    warnings: list[int] = []
    for index, value in enumerate(signal):
        detector.update(float(value))
        if detector.drift_detected:
            warnings.append(index)
    return warnings


def run_elec2_experiments(config: Elec2Config | None = None) -> Elec2ExperimentResult:
    config = config or Elec2Config()
    values = load_elec2_values(config)
    events = detect_page_hinkley_events(values, config)
    adaptive_gap = (
        max(1, int(np.median(np.diff(events)) * 0.5))
        if len(events) > 1
        else config.max_gap
    )
    config.max_gap = min(config.max_gap, adaptive_gap)

    sigma_100, estimate_100, windows_100 = _compute_sigma_fixed(
        values, config.fixed_window
    )
    sigma_50, estimate_50, windows_50 = _compute_sigma_fixed(values, 50)
    sigma_300, estimate_300, windows_300 = _compute_sigma_fixed(values, 300)
    sigma_dynamic, estimate_dynamic, windows_dynamic, zeta_hat = _compute_sigma_dynamic(
        values,
        config,
    )
    residual_signal = np.abs(estimate_100 - values)
    residual_signal = np.nan_to_num(residual_signal, nan=0.0)
    adwin_warnings = _detect_adwin_warnings(residual_signal, config.adwin_delta)

    return Elec2ExperimentResult(
        config=config,
        values=values,
        events=events,
        fixed_100=_build_detection_result(
            sigma_100, estimate_100, windows_100, events, config
        ),
        fixed_50=_build_detection_result(
            sigma_50, estimate_50, windows_50, events, config
        ),
        fixed_300=_build_detection_result(
            sigma_300, estimate_300, windows_300, events, config
        ),
        dynamic=_build_detection_result(
            sigma_dynamic, estimate_dynamic, windows_dynamic, events, config
        ),
        adwin=_build_detection_result_from_warnings(
            residual_signal,
            adwin_warnings,
            events,
            config,
        ),
        residual_signal=residual_signal,
        dynamic_drift_estimate=zeta_hat,
    )


def summarize_detection(result: Elec2DetectionResult) -> dict[str, float]:
    lead_times = np.asarray(result.lead_times, dtype=float)
    return {
        "warnings": float(len(result.warnings)),
        "leads": float(len(result.lead_times)),
        "precision": float(len(result.lead_times) / len(result.warnings))
        if result.warnings
        else 0.0,
        "median_lead": float(np.median(lead_times)) if lead_times.size else 0.0,
        "mean_lead": float(np.mean(lead_times)) if lead_times.size else 0.0,
        "min_lead": float(np.min(lead_times)) if lead_times.size else 0.0,
        "max_lead": float(np.max(lead_times)) if lead_times.size else 0.0,
    }


def build_elec2_rows(result: Elec2ExperimentResult) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for name, detection in (
        ("fixed_50", result.fixed_50),
        ("fixed_100", result.fixed_100),
        ("fixed_300", result.fixed_300),
        ("dynamic", result.dynamic),
        ("adwin", result.adwin),
    ):
        summary = summarize_detection(detection)
        rows.append(
            {
                "strategy": name,
                "warnings": int(summary["warnings"]),
                "leads": int(summary["leads"]),
                "precision": round(summary["precision"], 3),
                "median_lead": round(summary["median_lead"], 1),
                "mean_lead": round(summary["mean_lead"], 1),
                "min_lead": round(summary["min_lead"], 1),
                "max_lead": round(summary["max_lead"], 1),
            }
        )
    return rows


def save_elec2_figure(result: Elec2ExperimentResult, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    detection = result.fixed_100
    time = np.arange(result.values.size)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True)
    axes[0].plot(
        time, _rolling_mean(detection.sigma, 100), color="tab:blue", linewidth=1.2
    )
    axes[0].axhline(
        result.config.warning_threshold,
        color="tab:red",
        linestyle="--",
        linewidth=1.0,
        label="TCI threshold",
    )
    for warning in detection.warnings:
        axes[0].axvline(warning, color="tab:red", alpha=0.08, linewidth=0.8)
    for warning in result.adwin.warnings:
        axes[0].axvline(warning, color="tab:purple", alpha=0.05, linewidth=0.8)
    for event in result.events:
        axes[0].axvline(event, color="0.5", alpha=0.08, linewidth=0.8)
    axes[0].set_ylabel(r"$\hat\sigma_P$")
    axes[0].set_title("ELEC2 early-warning diagnostic (fixed n=100 vs ADWIN)")
    axes[0].plot([], [], color="tab:red", linewidth=1.0, label="TCI warnings")
    axes[0].plot([], [], color="tab:purple", linewidth=1.0, label="ADWIN warnings")
    axes[0].legend(loc="lower left", ncol=3)

    axes[1].plot(
        time,
        _rolling_mean(result.residual_signal, 50),
        color="tab:orange",
        linewidth=1.2,
        label="residual input",
    )
    axes[1].set_ylabel("Residual")
    axes[1].set_xlabel("Time step")
    axes[1].legend(loc="upper left")

    for axis in axes:
        axis.grid(alpha=0.2, linewidth=0.5)

    fig.tight_layout()
    fig.savefig(output_path)
    fig.savefig(output_path.with_suffix(".png"), dpi=180)
    plt.close(fig)


def save_dynamic_nstar_figure(result: Elec2ExperimentResult, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    time = np.arange(result.values.size)
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=False)

    axes[0].plot(
        time,
        _rolling_mean(result.fixed_50.sigma, 100),
        label="fixed n=50",
        linewidth=1.1,
    )
    axes[0].plot(
        time,
        _rolling_mean(result.fixed_300.sigma, 100),
        label="fixed n=300",
        linewidth=1.1,
    )
    axes[0].plot(
        time,
        _rolling_mean(result.dynamic.sigma, 100),
        label="dynamic n*_t",
        linewidth=1.2,
    )
    for event in result.events:
        axes[0].axvline(event, color="0.7", alpha=0.05, linewidth=0.8)
    axes[0].set_ylabel(r"$\hat\sigma_P$")
    axes[0].set_title("Dynamic window adaptation on ELEC2")
    axes[0].legend(loc="lower left", ncol=3)

    axes[1].plot(time, result.dynamic.window_sizes, color="tab:green", linewidth=1.1)
    axes[1].set_ylabel(r"$n^*_t$")

    lead_data = [
        result.fixed_50.lead_times,
        result.fixed_300.lead_times,
        result.dynamic.lead_times,
        result.adwin.lead_times,
    ]
    axes[2].boxplot(
        lead_data,
        labels=["fixed 50", "fixed 300", "dynamic", "ADWIN"],
        showfliers=False,
    )
    axes[2].set_ylabel("Lead time")
    axes[2].set_xlabel("Strategy")

    for axis in axes:
        axis.grid(alpha=0.2, linewidth=0.5)

    fig.tight_layout()
    fig.savefig(output_path)
    fig.savefig(output_path.with_suffix(".png"), dpi=180)
    plt.close(fig)
