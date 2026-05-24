from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from river import drift as river_drift

from .common import build_manifest_row, export_rows_csv, rolling_mean, stable_run_id

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
    detector_input: Literal["observation", "signed_residual", "absolute_residual"] = (
        "observation"
    )
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
    pre_detection_excess_area: float


@dataclass(slots=True)
class InvalidityGapSummary:
    detector_name: str
    detector_input: str
    detector_delta: float
    mean_t_valid: float
    mean_t_detect: float
    mean_gap: float
    median_gap: float
    gap_q10: float
    gap_q90: float
    std_gap: float
    mean_gap_se: float
    positive_gap_rate: float
    detection_rate: float
    undetected_invalid_rate: float
    mean_pre_detection_excess_area: float


@dataclass(slots=True)
class InvalidityGapResult:
    config: InvalidityGapConfig
    representative: InvalidityGapTrace
    traces: list[InvalidityGapTrace]
    summaries: list[InvalidityGapSummary]


def _parse_csv_ints(text: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in text.split(",") if part.strip())


def _parse_csv_floats(text: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in text.split(",") if part.strip())


def _parse_csv_strings(text: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in text.split(",") if part.strip())


def _build_detector(
    config: InvalidityGapConfig, detector_delta: float
) -> river_drift.ADWIN | river_drift.PageHinkley:
    if config.detector_name == "adwin":
        return river_drift.ADWIN(delta=detector_delta)
    return river_drift.PageHinkley(
        min_instances=config.page_hinkley_min_instances,
        delta=detector_delta,
        threshold=config.page_hinkley_threshold,
        alpha=config.page_hinkley_alpha,
        mode="both",
    )


def _detector_stream(
    trace: InvalidityGapTrace, config: InvalidityGapConfig
) -> np.ndarray:
    if config.detector_input == "observation":
        return trace.observations
    if config.detector_input == "signed_residual":
        return trace.observations - trace.operating_estimate
    return trace.residual_stream


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


def _pre_detection_excess_area(
    operating_error: np.ndarray,
    oracle_error: np.ndarray,
    t_valid: int | None,
    t_detect: int | None,
) -> float:
    if t_valid is None or t_detect is None or t_detect <= t_valid:
        return 0.0
    excess = np.maximum(
        operating_error[t_valid:t_detect] - oracle_error[t_valid:t_detect], 0.0
    )
    return float(np.sum(excess))


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
    signed_residual_stream = observations - operating_estimate
    detector = _build_detector(config, detector_delta)
    detector_events = np.zeros(config.steps, dtype=bool)
    detector_trace = InvalidityGapTrace(
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
        t_valid=None,
        t_detect=None,
        invalidity_gap=None,
        pre_detection_excess_area=0.0,
    )
    detector_stream = _detector_stream(detector_trace, config)
    if config.detector_input == "signed_residual":
        detector_stream = signed_residual_stream
    for index, value in enumerate(detector_stream):
        detector.update(float(value))
        detector_events[index] = bool(detector.drift_detected)
    stale_mask = config.operating_window > oracle_horizon
    t_valid = _first_persistent_true(stale_mask, config.persistence, config.warmup)
    t_detect = _first_detector_time(detector_events, t_valid)
    invalidity_gap = None if t_valid is None or t_detect is None else t_detect - t_valid
    pre_detection_excess_area = _pre_detection_excess_area(
        operating_error,
        oracle_error,
        t_valid,
        t_detect,
    )
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
        pre_detection_excess_area=pre_detection_excess_area,
    )


def _summarize(
    traces: list[InvalidityGapTrace],
    detector_name: str,
    detector_input: str,
    detector_delta: float,
) -> InvalidityGapSummary:
    t_valid = [trace.t_valid for trace in traces if trace.t_valid is not None]
    t_detect = [trace.t_detect for trace in traces if trace.t_detect is not None]
    gaps = [
        trace.invalidity_gap for trace in traces if trace.invalidity_gap is not None
    ]
    gap_array = np.asarray(gaps, dtype=float) if gaps else np.asarray([], dtype=float)
    excess_areas = np.asarray(
        [trace.pre_detection_excess_area for trace in traces],
        dtype=float,
    )
    undetected_invalid_rate = float(
        np.mean(
            [trace.t_valid is not None and trace.t_detect is None for trace in traces]
        )
    )
    return InvalidityGapSummary(
        detector_name=detector_name,
        detector_input=detector_input,
        detector_delta=detector_delta,
        mean_t_valid=float(np.mean(t_valid)) if t_valid else float("nan"),
        mean_t_detect=float(np.mean(t_detect)) if t_detect else float("nan"),
        mean_gap=float(np.mean(gaps)) if gaps else float("nan"),
        median_gap=float(np.median(gap_array)) if gaps else float("nan"),
        gap_q10=float(np.quantile(gap_array, 0.1)) if gaps else float("nan"),
        gap_q90=float(np.quantile(gap_array, 0.9)) if gaps else float("nan"),
        std_gap=float(np.std(gaps)) if gaps else float("nan"),
        mean_gap_se=float(np.std(gap_array, ddof=1) / np.sqrt(gap_array.size))
        if gap_array.size >= 2
        else 0.0,
        positive_gap_rate=float(np.mean([gap > 0 for gap in gaps])) if gaps else 0.0,
        detection_rate=float(np.mean([trace.t_detect is not None for trace in traces])),
        undetected_invalid_rate=undetected_invalid_rate,
        mean_pre_detection_excess_area=float(np.mean(excess_areas)),
    )


def run_invalidity_gap_experiment(
    config: InvalidityGapConfig | None = None,
) -> InvalidityGapResult:
    cfg = config or InvalidityGapConfig()
    traces = [_run_single_trace(seed, cfg, cfg.detector_delta) for seed in cfg.seeds]
    summaries: list[InvalidityGapSummary] = []
    for delta in cfg.detector_deltas:
        detector_traces = [_run_single_trace(seed, cfg, delta) for seed in cfg.seeds]
        summaries.append(
            _summarize(
                detector_traces,
                cfg.detector_name,
                cfg.detector_input,
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
        InvalidityGapConfig(
            detector_name="page_hinkley",
            detector_delta=0.005,
            detector_deltas=(0.005,),
        ),
    )
    rows: list[InvalidityGapSummary] = []
    for config in configs:
        result = run_invalidity_gap_experiment(config)
        summary = result.summaries[0]
        rows.append(
            InvalidityGapSummary(
                detector_name=config.detector_name,
                detector_input=summary.detector_input,
                detector_delta=summary.detector_delta,
                mean_t_valid=summary.mean_t_valid,
                mean_t_detect=summary.mean_t_detect,
                mean_gap=summary.mean_gap,
                median_gap=summary.median_gap,
                gap_q10=summary.gap_q10,
                gap_q90=summary.gap_q90,
                std_gap=summary.std_gap,
                mean_gap_se=summary.mean_gap_se,
                positive_gap_rate=summary.positive_gap_rate,
                detection_rate=summary.detection_rate,
                undetected_invalid_rate=summary.undetected_invalid_rate,
                mean_pre_detection_excess_area=summary.mean_pre_detection_excess_area,
            )
        )
    return rows


def build_invalidity_gap_sweep_configs(
    *,
    detector_names: tuple[str, ...],
    detector_inputs: tuple[str, ...],
    operating_windows: tuple[int, ...],
    detector_deltas: tuple[float, ...],
    seeds: tuple[int, ...],
    steps: int,
    phase_lengths: tuple[int, int, int] = (1000, 1400, 1200),
    low_drift: float,
    high_drift: float,
    persistence: int,
) -> list[InvalidityGapConfig]:
    configs: list[InvalidityGapConfig] = []
    for detector_name in detector_names:
        for detector_input in detector_inputs:
            for operating_window in operating_windows:
                base_delta = detector_deltas[0]
                configs.append(
                    InvalidityGapConfig(
                        seeds=seeds,
                        steps=steps,
                        phase_lengths=phase_lengths,
                        low_drift=low_drift,
                        high_drift=high_drift,
                        operating_window=operating_window,
                        detector_name=detector_name,  # type: ignore[arg-type]
                        detector_input=detector_input,  # type: ignore[arg-type]
                        detector_delta=base_delta,
                        detector_deltas=detector_deltas,
                        persistence=persistence,
                    )
                )
    return configs


def run_invalidity_gap_sweep(
    configs: list[InvalidityGapConfig],
) -> list[InvalidityGapResult]:
    return [run_invalidity_gap_experiment(config) for config in configs]


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
    run_id = stable_run_id(asdict(result.config))
    for summary in result.summaries:
        rows.append(
            {
                "run_id": run_id,
                "detector": summary.detector_name,
                "detector_input": summary.detector_input,
                "detector_delta": round(float(summary.detector_delta), 4),
                "mean_t_valid": round(float(summary.mean_t_valid), 1),
                "mean_t_detect": round(float(summary.mean_t_detect), 1),
                "mean_gap": round(float(summary.mean_gap), 1),
                "median_gap": round(float(summary.median_gap), 1),
                "gap_q10": round(float(summary.gap_q10), 1),
                "gap_q90": round(float(summary.gap_q90), 1),
                "std_gap": round(float(summary.std_gap), 1),
                "mean_gap_se": round(float(summary.mean_gap_se), 2),
                "positive_gap_rate": round(float(summary.positive_gap_rate), 3),
                "detection_rate": round(float(summary.detection_rate), 3),
                "undetected_invalid_rate": round(
                    float(summary.undetected_invalid_rate), 3
                ),
                "mean_pre_detection_excess_area": round(
                    float(summary.mean_pre_detection_excess_area), 3
                ),
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
                "detector_input": summary.detector_input,
                "mean_t_valid": round(float(summary.mean_t_valid), 1),
                "mean_t_detect": round(float(summary.mean_t_detect), 1),
                "mean_gap": round(float(summary.mean_gap), 1),
                "median_gap": round(float(summary.median_gap), 1),
                "positive_gap_rate": round(float(summary.positive_gap_rate), 3),
                "mean_pre_detection_excess_area": round(
                    float(summary.mean_pre_detection_excess_area), 3
                ),
            }
        )
    return rows


def build_trace_rows(result: InvalidityGapResult) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    run_id = stable_run_id(asdict(result.config))
    for trace in result.traces:
        rows.append(
            {
                "run_id": run_id,
                "seed": trace.seed,
                "t_valid": "" if trace.t_valid is None else trace.t_valid,
                "t_detect": "" if trace.t_detect is None else trace.t_detect,
                "invalidity_gap": ""
                if trace.invalidity_gap is None
                else trace.invalidity_gap,
                "detected": int(trace.t_detect is not None),
                "pre_detection_excess_area": round(
                    float(trace.pre_detection_excess_area), 6
                ),
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
    parser.add_argument("--detector-name", type=str, default="adwin")
    parser.add_argument("--detector-input", type=str, default="observation")
    parser.add_argument("--operating-window", type=int, default=220)
    parser.add_argument(
        "--detector-deltas", type=str, default="0.0005,0.001,0.002,0.004"
    )
    parser.add_argument("--detector-names", type=str, default="")
    parser.add_argument("--detector-inputs", type=str, default="")
    parser.add_argument("--operating-windows", type=str, default="")
    parser.add_argument("--steps", type=int, default=3600)
    parser.add_argument("--low-drift", type=float, default=0.00008)
    parser.add_argument("--high-drift", type=float, default=0.0025)
    parser.add_argument("--persistence", type=int, default=40)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    args.csv_dir.mkdir(parents=True, exist_ok=True)
    detector_names = (
        (args.detector_name,)
        if not args.detector_names
        else _parse_csv_strings(args.detector_names)
    )
    detector_inputs = (
        (args.detector_input,)
        if not args.detector_inputs
        else _parse_csv_strings(args.detector_inputs)
    )
    operating_windows = (
        (args.operating_window,)
        if not args.operating_windows
        else _parse_csv_ints(args.operating_windows)
    )
    detector_deltas = _parse_csv_floats(args.detector_deltas)
    configs = build_invalidity_gap_sweep_configs(
        detector_names=detector_names,
        detector_inputs=detector_inputs,
        operating_windows=operating_windows,
        detector_deltas=detector_deltas,
        seeds=tuple(range(12)),
        steps=args.steps,
        phase_lengths=(1000, 1400, 1200),
        low_drift=args.low_drift,
        high_drift=args.high_drift,
        persistence=args.persistence,
    )
    results = run_invalidity_gap_sweep(configs)
    result = results[0]
    save_invalidity_gap_figure(result, args.figures_dir / "fig_invalidity_gap.pdf")
    all_gap_rows: list[dict[str, float | str]] = []
    all_trace_rows: list[dict[str, float | int | str]] = []
    for sweep_result in results:
        all_gap_rows.extend(build_gap_rows(sweep_result))
        all_trace_rows.extend(build_trace_rows(sweep_result))
    export_rows_csv(all_gap_rows, args.csv_dir / "invalidity_gap_ablation.csv")
    export_rows_csv(
        build_detector_comparison_rows(run_detector_comparison()),
        args.csv_dir / "invalidity_gap_detector_comparison.csv",
    )
    export_rows_csv(all_trace_rows, args.csv_dir / "invalidity_gap_traces.csv")
    manifest = build_manifest_row(
        "invalidity_gap",
        {
            "detector_names": detector_names,
            "detector_inputs": detector_inputs,
            "operating_windows": operating_windows,
            "detector_deltas": detector_deltas,
            "steps": args.steps,
            "low_drift": args.low_drift,
            "high_drift": args.high_drift,
            "persistence": args.persistence,
        },
        run_id=stable_run_id(
            {
                "detector_names": detector_names,
                "detector_inputs": detector_inputs,
                "operating_windows": operating_windows,
                "detector_deltas": detector_deltas,
                "steps": args.steps,
                "low_drift": args.low_drift,
                "high_drift": args.high_drift,
                "persistence": args.persistence,
            }
        ),
        notes="Detector-input aware invalidity-gap robustness sweep.",
    )
    export_rows_csv([manifest], args.csv_dir / "manifest.csv")
