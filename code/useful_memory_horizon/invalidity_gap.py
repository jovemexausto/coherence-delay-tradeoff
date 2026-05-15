from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from river import drift as river_drift

from .common import export_rows_csv, rolling_mean

LIPSCHITZ_STALENESS_COEFFICIENT = 3.0 ** (-0.5)


@dataclass(slots=True)
class InvalidityGapConfig:
    seeds: tuple[int, ...] = tuple(range(12))
    steps: int = 3600
    warmup: int = 400
    phase_lengths: tuple[int, int, int] = (1000, 1400, 1200)
    low_drift: float = 0.00008
    high_drift: float = 0.0025
    observation_scale: float = 1.0
    process_scale: float = 0.0
    operating_window: int = 220
    detector_delta: float = 0.002
    detector_deltas: tuple[float, ...] = (0.0005, 0.001, 0.002, 0.004)
    detector_name: Literal["adwin", "page_hinkley"] = "adwin"
    page_hinkley_delta: float = 0.005
    page_hinkley_threshold: float = 50.0
    page_hinkley_alpha: float = 0.9999
    page_hinkley_min_instances: int = 30
    n_min: int = 20
    n_max: int = 320
    Ck: float = 1.0
    rolling_window: int = 80
    persistence: int = 40


@dataclass(slots=True)
class InvalidityGapTrace:
    seed: int
    time: np.ndarray
    drift_path: np.ndarray
    latent_mean: np.ndarray
    observations: np.ndarray
    operating_estimate: np.ndarray
    oracle_estimate: np.ndarray
    operating_error: np.ndarray
    oracle_error: np.ndarray
    oracle_horizon: np.ndarray
    residual_stream: np.ndarray
    detector_events: np.ndarray
    t_valid: int | None
    t_detect: int | None
    invalidity_gap: int | None


@dataclass(slots=True)
class InvalidityGapSummary:
    detector_name: str
    detector_delta: float
    mean_t_valid: float
    mean_t_detect: float
    mean_gap: float
    std_gap: float
    positive_gap_rate: float
    detection_rate: float


@dataclass(slots=True)
class InvalidityGapResult:
    config: InvalidityGapConfig
    representative: InvalidityGapTrace
    traces: list[InvalidityGapTrace]
    summaries: list[InvalidityGapSummary]


def _build_detector(
    config: InvalidityGapConfig, detector_delta: float
) -> river_drift.ADWIN | river_drift.PageHinkley:
    if config.detector_name == "adwin":
        return river_drift.ADWIN(delta=detector_delta)
    return river_drift.PageHinkley(
        min_instances=config.page_hinkley_min_instances,
        delta=config.page_hinkley_delta,
        threshold=config.page_hinkley_threshold,
        alpha=config.page_hinkley_alpha,
        mode="both",
    )


def _simulate_stream(
    seed: int, config: InvalidityGapConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    p1, p2, _ = config.phase_lengths
    drift_path = np.empty(config.steps, dtype=float)
    drift_path[:p1] = config.low_drift
    drift_path[p1 : p1 + p2] = np.linspace(
        config.low_drift, config.high_drift, p2, endpoint=True, dtype=float
    )
    drift_path[p1 + p2 :] = config.high_drift
    latent_mean = np.zeros(config.steps, dtype=float)
    for step in range(1, config.steps):
        latent_mean[step] = (
            latent_mean[step - 1]
            + drift_path[step]
            + rng.normal(scale=config.process_scale)
        )
    observations = latent_mean + rng.normal(
        scale=config.observation_scale, size=config.steps
    )
    return drift_path, latent_mean, observations


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    estimate = np.zeros(values.size, dtype=float)
    for index in range(values.size):
        start = max(0, index - window + 1)
        estimate[index] = float(np.mean(values[start : index + 1]))
    return estimate


def _oracle_horizon(config: InvalidityGapConfig, drift_path: np.ndarray) -> np.ndarray:
    horizon = (
        config.Ck
        / (2.0 * LIPSCHITZ_STALENESS_COEFFICIENT * np.maximum(drift_path, 1e-12))
    ) ** (2.0 / 3.0)
    return np.clip(horizon, config.n_min, config.n_max)


def _variable_window_estimate(values: np.ndarray, windows: np.ndarray) -> np.ndarray:
    estimate = np.zeros(values.size, dtype=float)
    for index in range(values.size):
        window = max(1, min(int(round(float(windows[index]))), index + 1))
        start = max(0, index - window + 1)
        estimate[index] = float(np.mean(values[start : index + 1]))
    return estimate


def _first_persistent_true(
    mask: np.ndarray, persistence: int, warmup: int
) -> int | None:
    run = 0
    for index in range(max(warmup, 0), mask.size):
        run = run + 1 if mask[index] else 0
        if run >= persistence:
            return index - persistence + 1
    return None


def _first_detector_time(events: np.ndarray, start: int | None) -> int | None:
    if start is None:
        return None
    hits = np.flatnonzero(events[start:])
    return None if hits.size == 0 else int(start + hits[0])


def _run_single_trace(
    seed: int, config: InvalidityGapConfig, detector_delta: float
) -> InvalidityGapTrace:
    drift_path, latent_mean, observations = _simulate_stream(seed, config)
    oracle_horizon = _oracle_horizon(config, drift_path)
    operating_estimate = _moving_average(observations, config.operating_window)
    oracle_estimate = _variable_window_estimate(observations, oracle_horizon)
    operating_error = np.abs(latent_mean - operating_estimate)
    oracle_error = np.abs(latent_mean - oracle_estimate)
    residual_stream = np.abs(observations - operating_estimate)
    detector = _build_detector(config, detector_delta)
    detector_events = np.zeros(config.steps, dtype=bool)
    for index, value in enumerate(observations):
        detector.update(float(value))
        detector_events[index] = bool(detector.drift_detected)
    stale_mask = config.operating_window > oracle_horizon
    t_valid = _first_persistent_true(stale_mask, config.persistence, config.warmup)
    t_detect = _first_detector_time(detector_events, t_valid)
    invalidity_gap = None if t_valid is None or t_detect is None else t_detect - t_valid
    return InvalidityGapTrace(
        seed=seed,
        time=np.arange(config.steps),
        drift_path=drift_path,
        latent_mean=latent_mean,
        observations=observations,
        operating_estimate=operating_estimate,
        oracle_estimate=oracle_estimate,
        operating_error=operating_error,
        oracle_error=oracle_error,
        oracle_horizon=oracle_horizon,
        residual_stream=residual_stream,
        detector_events=detector_events,
        t_valid=t_valid,
        t_detect=t_detect,
        invalidity_gap=invalidity_gap,
    )


def _summarize(
    traces: list[InvalidityGapTrace], detector_name: str, detector_delta: float
) -> InvalidityGapSummary:
    t_valid = [trace.t_valid for trace in traces if trace.t_valid is not None]
    t_detect = [trace.t_detect for trace in traces if trace.t_detect is not None]
    gaps = [
        trace.invalidity_gap for trace in traces if trace.invalidity_gap is not None
    ]
    return InvalidityGapSummary(
        detector_name=detector_name,
        detector_delta=detector_delta,
        mean_t_valid=float(np.mean(t_valid)) if t_valid else float("nan"),
        mean_t_detect=float(np.mean(t_detect)) if t_detect else float("nan"),
        mean_gap=float(np.mean(gaps)) if gaps else float("nan"),
        std_gap=float(np.std(gaps)) if gaps else float("nan"),
        positive_gap_rate=float(np.mean([gap > 0 for gap in gaps])) if gaps else 0.0,
        detection_rate=float(np.mean([trace.t_detect is not None for trace in traces])),
    )


def run_invalidity_gap_experiment(
    config: InvalidityGapConfig | None = None,
) -> InvalidityGapResult:
    cfg = config or InvalidityGapConfig()
    traces = [_run_single_trace(seed, cfg, cfg.detector_delta) for seed in cfg.seeds]
    summaries: list[InvalidityGapSummary] = []
    for delta in cfg.detector_deltas:
        summaries.append(
            _summarize(
                [_run_single_trace(seed, cfg, delta) for seed in cfg.seeds],
                cfg.detector_name,
                delta,
            )
        )
    representative = max(
        traces,
        key=lambda trace: -1 if trace.invalidity_gap is None else trace.invalidity_gap,
    )
    return InvalidityGapResult(
        config=cfg, representative=representative, traces=traces, summaries=summaries
    )


def run_detector_comparison() -> list[InvalidityGapSummary]:
    configs = (
        InvalidityGapConfig(
            detector_name="adwin", detector_delta=0.002, detector_deltas=(0.002,)
        ),
        InvalidityGapConfig(detector_name="page_hinkley", detector_deltas=(0.005,)),
    )
    rows: list[InvalidityGapSummary] = []
    for config in configs:
        result = run_invalidity_gap_experiment(config)
        summary = result.summaries[0]
        rows.append(
            InvalidityGapSummary(
                detector_name=config.detector_name,
                detector_delta=summary.detector_delta,
                mean_t_valid=summary.mean_t_valid,
                mean_t_detect=summary.mean_t_detect,
                mean_gap=summary.mean_gap,
                std_gap=summary.std_gap,
                positive_gap_rate=summary.positive_gap_rate,
                detection_rate=summary.detection_rate,
            )
        )
    return rows


def _rolling_error(
    trace: InvalidityGapTrace, window: int
) -> tuple[np.ndarray, np.ndarray]:
    return rolling_mean(trace.operating_error, window), rolling_mean(
        trace.oracle_error, window
    )


def save_invalidity_gap_figure(result: InvalidityGapResult, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trace = result.representative
    rolling_operating, rolling_oracle = _rolling_error(
        trace, result.config.rolling_window
    )
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.2), sharex=True)
    useful_line = axes[0].plot(
        trace.time,
        trace.oracle_horizon,
        color="black",
        linewidth=2.0,
        label=r"useful-memory horizon $n_t^*$",
    )[0]
    operating_line = axes[0].axhline(
        result.config.operating_window,
        color="tab:red",
        linewidth=1.5,
        linestyle="--",
        label="operating horizon",
    )
    drift_axis = axes[0].twinx()
    drift_line = drift_axis.plot(
        trace.time,
        trace.drift_path,
        color="tab:purple",
        linewidth=1.6,
        alpha=0.9,
        label=r"local drift $\zeta_t$",
    )[0]
    drift_axis.set_ylabel(r"$\zeta_t$")
    if trace.t_valid is not None:
        axes[0].axvline(
            trace.t_valid,
            color="tab:red",
            linewidth=1.8,
            linestyle=":",
            zorder=5,
            label=r"$t_{\mathrm{valid}}$",
        )
    if trace.t_detect is not None:
        axes[0].axvline(
            trace.t_detect,
            color="tab:blue",
            linewidth=1.8,
            linestyle="--",
            zorder=5,
            label=r"$t_{\mathrm{detect}}$",
        )
    if (
        trace.t_valid is not None
        and trace.t_detect is not None
        and trace.t_detect >= trace.t_valid
    ):
        axes[0].axvspan(
            trace.t_valid, trace.t_detect, color="gold", alpha=0.18, linewidth=0
        )
    axes[0].set_ylabel("Horizon")
    axes[0].set_title("Detector-silent staleness and the invalidity gap")
    axes[0].legend(
        handles=[useful_line, operating_line, drift_line],
        loc="upper right",
        frameon=False,
    )

    axes[1].plot(
        trace.time,
        rolling_operating,
        color="tab:red",
        linewidth=1.5,
        label="long-horizon error",
    )
    axes[1].plot(
        trace.time,
        rolling_oracle,
        color="black",
        linewidth=1.5,
        label="oracle-horizon error",
    )
    if trace.t_valid is not None:
        axes[1].axvline(
            trace.t_valid,
            color="tab:red",
            linewidth=1.8,
            linestyle=":",
            label=r"$t_{\mathrm{valid}}$",
        )
    if trace.t_detect is not None:
        axes[1].axvline(
            trace.t_detect,
            color="tab:blue",
            linewidth=1.8,
            linestyle="--",
            label=r"$t_{\mathrm{detect}}$",
        )
    if (
        trace.t_valid is not None
        and trace.t_detect is not None
        and trace.t_detect >= trace.t_valid
    ):
        axes[1].axvspan(
            trace.t_valid, trace.t_detect, color="gold", alpha=0.18, linewidth=0
        )
        axes[1].text(
            trace.t_valid + 20,
            float(max(rolling_operating.max(), rolling_oracle.max())) * 0.92,
            rf"$\Delta_{{inv}}={trace.t_detect - trace.t_valid}$",
            fontsize=10,
            bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.9},
        )
    axes[1].set_xlabel("Time step")
    axes[1].set_ylabel("Rolling MAE")
    axes[1].legend(loc="upper left", frameon=False)

    for axis in axes:
        axis.grid(alpha=0.2, linewidth=0.5)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def build_gap_rows(result: InvalidityGapResult) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for summary in result.summaries:
        rows.append(
            {
                "detector": summary.detector_name,
                "detector_delta": round(float(summary.detector_delta), 4),
                "mean_t_valid": round(float(summary.mean_t_valid), 1),
                "mean_t_detect": round(float(summary.mean_t_detect), 1),
                "mean_gap": round(float(summary.mean_gap), 1),
                "std_gap": round(float(summary.std_gap), 1),
                "positive_gap_rate": round(float(summary.positive_gap_rate), 3),
                "detection_rate": round(float(summary.detection_rate), 3),
            }
        )
    return rows


def build_detector_comparison_rows(
    summaries: list[InvalidityGapSummary],
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for summary in summaries:
        rows.append(
            {
                "detector": summary.detector_name,
                "mean_t_valid": round(float(summary.mean_t_valid), 1),
                "mean_t_detect": round(float(summary.mean_t_detect), 1),
                "mean_gap": round(float(summary.mean_gap), 1),
                "positive_gap_rate": round(float(summary.positive_gap_rate), 3),
            }
        )
    return rows


def build_trace_rows(result: InvalidityGapResult) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for trace in result.traces:
        rows.append(
            {
                "seed": trace.seed,
                "t_valid": "" if trace.t_valid is None else trace.t_valid,
                "t_detect": "" if trace.t_detect is None else trace.t_detect,
                "invalidity_gap": ""
                if trace.invalidity_gap is None
                else trace.invalidity_gap,
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate invalidity-gap artifacts.")
    parser.add_argument(
        "--figures-dir", type=Path, default=Path("artifacts/figures/invalidity_gap")
    )
    parser.add_argument(
        "--csv-dir", type=Path, default=Path("artifacts/csv/invalidity_gap")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    args.csv_dir.mkdir(parents=True, exist_ok=True)
    result = run_invalidity_gap_experiment(InvalidityGapConfig())
    save_invalidity_gap_figure(result, args.figures_dir / "fig_invalidity_gap.pdf")
    export_rows_csv(
        build_gap_rows(result), args.csv_dir / "invalidity_gap_ablation.csv"
    )
    export_rows_csv(
        build_detector_comparison_rows(run_detector_comparison()),
        args.csv_dir / "invalidity_gap_detector_comparison.csv",
    )
    export_rows_csv(
        build_trace_rows(result), args.csv_dir / "invalidity_gap_traces.csv"
    )
