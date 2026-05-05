from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from river import drift


@dataclass(slots=True)
class KuaiRandConfig:
    data_dir: Path = Path("../data/kuairand/KuaiRand-Pure/data")
    window_size: int = 20
    min_phase_count: int = 20
    max_users: int = 1000
    seed: int = 7
    tci_threshold: float = 0.80
    tcie_threshold: float = 0.80
    auto_calibrate_thresholds: bool = True
    threshold_quantile: float = 0.20
    tcie_lambda: float = 3.0
    adwin_delta: float = 0.03
    page_hinkley_delta: float = 0.005
    page_hinkley_threshold: float = 20.0
    page_hinkley_alpha: float = 0.9999
    kswin_window_size: int = 30
    kswin_stat_size: int = 10
    kswin_alpha: float = 0.001


@dataclass(slots=True)
class KuaiRandUserSignals:
    user_id: int
    signals: pd.DataFrame
    random_end: int
    coercive_end: int
    baseline_watch_mean: float
    baseline_watch_std: float
    baseline_tag_dist: dict[str, float]
    baseline_effort_scale: float


@dataclass(slots=True)
class KuaiRandUserDetectionResult:
    user_id: int
    tci_warnings: list[int]
    tcie_warnings: list[int]
    raw_warnings: dict[str, list[int]]
    masking_detection: dict[str, dict[str, float | int | None]]
    collapse_detection: dict[str, dict[str, float | int | None]]


@dataclass(slots=True)
class KuaiRandBenchmarkResult:
    config: KuaiRandConfig
    user_results: list[KuaiRandUserDetectionResult]
    user_signals: list[KuaiRandUserSignals]
    summary_rows: list[dict[str, str | float | int]]


ACTIVE_BASELINE_DETECTORS = ("ADWIN", "PageHinkley", "KSWIN", "NoDrift")


def _read_logs(base: Path) -> pd.DataFrame:
    cols = [
        "user_id",
        "video_id",
        "time_ms",
        "play_time_ms",
        "duration_ms",
        "is_click",
        "is_like",
        "is_follow",
        "is_comment",
        "is_forward",
        "is_hate",
        "long_view",
        "is_rand",
        "tab",
    ]
    random = pd.read_csv(
        base / "log_random_4_22_to_5_08_pure.csv",
        usecols=cols,
    )
    random["phase"] = "healthy"
    standard = pd.read_csv(
        base / "log_standard_4_22_to_5_08_pure.csv",
        usecols=cols,
    )
    standard["phase"] = "later"
    logs = pd.concat([random, standard], ignore_index=True)
    logs["watch_ratio"] = logs["play_time_ms"] / logs["duration_ms"].clip(lower=1)
    logs["watch_ratio"] = logs["watch_ratio"].replace([np.inf, -np.inf], np.nan)
    logs = logs.dropna(subset=["watch_ratio"])
    return logs


def _load_video_tags(base: Path) -> pd.Series:
    video = pd.read_csv(
        base / "video_features_basic_pure.csv", usecols=["video_id", "tag"]
    )
    return video.set_index("video_id")["tag"].astype(str)


def _kl_divergence(
    p: dict[str, float], q: dict[str, float], alpha: float = 1e-6
) -> float:
    keys = sorted(set(p) | set(q))
    p_arr = np.asarray([p.get(key, 0.0) + alpha for key in keys], dtype=float)
    q_arr = np.asarray([q.get(key, 0.0) + alpha for key in keys], dtype=float)
    p_arr /= p_arr.sum()
    q_arr /= q_arr.sum()
    return float(np.sum(p_arr * np.log(p_arr / q_arr)))


def _distribution(values: list[str]) -> dict[str, float]:
    if not values:
        return {}
    counts = Counter(values)
    total = float(sum(counts.values()))
    return {key: count / total for key, count in counts.items()}


def _entropy(dist: dict[str, float]) -> float:
    probs = np.asarray(list(dist.values()), dtype=float)
    probs = probs[probs > 0]
    if probs.size == 0:
        return 0.0
    return float(-(probs * np.log(probs)).sum())


def _match_warnings_to_events(
    warnings: list[int], events: list[int], max_gap: int
) -> tuple[list[int], list[int], list[int]]:
    matched_warnings: list[int] = []
    matched_events: list[int] = []
    lead_times: list[int] = []
    event_index = 0
    for warning in warnings:
        while event_index < len(events) and events[event_index] < warning:
            event_index += 1
        if event_index < len(events):
            lead = events[event_index] - warning
            if lead <= max_gap:
                matched_warnings.append(warning)
                matched_events.append(events[event_index])
                lead_times.append(lead)
                event_index += 1
    return matched_warnings, matched_events, lead_times


def _threshold_warnings(values: np.ndarray, threshold: float) -> list[int]:
    warnings: list[int] = []
    below = False
    for index, value in enumerate(values):
        if np.isnan(value):
            continue
        if value < threshold and not below:
            warnings.append(index)
            below = True
        elif value >= threshold:
            below = False
    return warnings


def _run_detector(
    signal: np.ndarray, detector_name: str, config: KuaiRandConfig
) -> list[int]:
    if detector_name == "ADWIN":
        detector = drift.ADWIN(delta=config.adwin_delta)
    elif detector_name == "PageHinkley":
        detector = drift.PageHinkley(
            delta=config.page_hinkley_delta,
            threshold=config.page_hinkley_threshold,
            alpha=config.page_hinkley_alpha,
            mode="both",
        )
    elif detector_name == "KSWIN":
        detector = drift.KSWIN(
            window_size=config.kswin_window_size,
            stat_size=config.kswin_stat_size,
            alpha=config.kswin_alpha,
        )
    elif detector_name == "NoDrift":
        detector = drift.NoDrift()
    else:
        raise ValueError(detector_name)

    warnings: list[int] = []
    for index, value in enumerate(signal):
        if np.isnan(value):
            continue
        detector.update(float(value))
        if detector.drift_detected:
            warnings.append(index)
    return warnings


def _window_signals(
    tags: list[str],
    watch_ratios: list[float],
    baseline_tag_dist: dict[str, float],
    baseline_watch_mean: float,
    baseline_watch_std: float,
    window_size: int,
    tcie_lambda: float,
    baseline_effort_scale: float,
    tag_vocab_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(tags)
    tci = np.full(n, np.nan)
    tcie = np.full(n, np.nan)
    sigma_p = np.full(n, np.nan)
    sigma_a = np.full(n, np.nan)
    sigma_phi = np.full(n, np.nan)
    effort = np.full(n, np.nan)

    tag_window: deque[str] = deque(maxlen=window_size)
    watch_window: deque[float] = deque(maxlen=window_size)

    for index, (tag, watch) in enumerate(zip(tags, watch_ratios, strict=True)):
        tag_window.append(tag)
        watch_window.append(float(watch))
        current_tag_dist = _distribution(list(tag_window))
        current_watch_mean = float(np.mean(watch_window))
        current_watch_std = float(np.std(watch_window))

        tag_kl = _kl_divergence(current_tag_dist, baseline_tag_dist)
        effort[index] = tag_kl
        watch_gap = abs(current_watch_mean - baseline_watch_mean)
        sigma_p[index] = 1.0 / (1.0 + watch_gap / max(baseline_watch_std, 1e-6))
        entropy = _entropy(current_tag_dist)
        sigma_a[index] = 1.0 / (1.0 + entropy / np.log(max(tag_vocab_size, 2)))
        sigma_phi[index] = 1.0 / (1.0 + current_watch_std)
        tci[index] = min(sigma_p[index], sigma_a[index], sigma_phi[index])
        tcie[index] = min(
            sigma_p[index]
            * np.exp(-tcie_lambda * tag_kl / max(baseline_effort_scale, 1e-6)),
            sigma_a[index],
            sigma_phi[index],
        )

    return tci, tcie, sigma_p, sigma_a, sigma_phi, effort


def _prepare_user_signals(
    user_id: int,
    user_logs: pd.DataFrame,
    tag_map: pd.Series,
    config: KuaiRandConfig,
) -> KuaiRandUserSignals | None:
    user_logs = user_logs.sort_values("time_ms").copy()
    if len(user_logs) < 2 * config.min_phase_count:
        return None

    healthy = user_logs[user_logs["phase"] == "healthy"]
    later = user_logs[user_logs["phase"] == "later"]
    if len(healthy) < config.min_phase_count or len(later) < 2 * config.min_phase_count:
        return None

    later = later.sort_values("time_ms")
    split = len(later) // 2
    coercive = later.iloc[:split]
    collapse = later.iloc[split:]

    eval_logs = pd.concat([healthy, coercive, collapse], ignore_index=True)
    eval_logs = eval_logs.sort_values("time_ms")
    eval_tags = tag_map.reindex(eval_logs["video_id"]).fillna("unknown").tolist()
    eval_watch = eval_logs["watch_ratio"].astype(float).tolist()

    healthy_tag_dist = _distribution(
        tag_map.reindex(healthy["video_id"]).fillna("unknown").tolist()
    )
    tag_vocab_size = int(tag_map.nunique())
    healthy_watch_mean = float(healthy["watch_ratio"].mean())
    healthy_watch_std = float(healthy["watch_ratio"].std(ddof=0) or 1.0)
    _, _, _, _, _, effort = _window_signals(
        eval_tags,
        eval_watch,
        healthy_tag_dist,
        healthy_watch_mean,
        healthy_watch_std,
        config.window_size,
        config.tcie_lambda,
        1.0,
        tag_vocab_size,
    )
    healthy_effort_scale = float(np.nanmedian(effort[: len(healthy)]))
    if not np.isfinite(healthy_effort_scale) or healthy_effort_scale <= 1e-6:
        healthy_effort_scale = 1.0

    tci, tcie, sigma_p, sigma_a, sigma_phi, effort = _window_signals(
        eval_tags,
        eval_watch,
        healthy_tag_dist,
        healthy_watch_mean,
        healthy_watch_std,
        config.window_size,
        config.tcie_lambda,
        healthy_effort_scale,
        tag_vocab_size,
    )

    signals = pd.DataFrame(
        {
            "tag": eval_tags,
            "watch_ratio": eval_watch,
            "sigma_p": sigma_p,
            "sigma_a": sigma_a,
            "sigma_phi": sigma_phi,
            "tci": tci,
            "tcie": tcie,
        }
    )

    random_end = len(healthy)
    coercive_end = len(healthy) + split
    return KuaiRandUserSignals(
        user_id=user_id,
        signals=signals,
        random_end=random_end,
        coercive_end=coercive_end,
        baseline_watch_mean=healthy_watch_mean,
        baseline_watch_std=healthy_watch_std,
        baseline_tag_dist=healthy_tag_dist,
        baseline_effort_scale=healthy_effort_scale,
    )


def load_kuairand_users(
    config: KuaiRandConfig | None = None,
) -> list[KuaiRandUserSignals]:
    config = config or KuaiRandConfig()
    base = config.data_dir
    logs = _read_logs(base)
    tag_map = _load_video_tags(base)

    counts = logs.groupby(["user_id", "phase"]).size().unstack(fill_value=0)
    eligible = counts[
        (counts["healthy"] >= config.min_phase_count)
        & (counts["later"] >= 2 * config.min_phase_count)
    ]
    eligible_users = (
        eligible.index.to_series()
        .sample(
            n=min(config.max_users, len(eligible)),
            random_state=config.seed,
            replace=False,
        )
        .tolist()
    )

    user_signals: list[KuaiRandUserSignals] = []
    grouped = logs[logs["user_id"].isin(eligible_users)].groupby("user_id")
    for user_id, user_logs in grouped:
        prepared = _prepare_user_signals(int(user_id), user_logs, tag_map, config)
        if prepared is not None:
            user_signals.append(prepared)
    return user_signals


def summarize_user_detection(
    warnings: list[int],
    event: int,
    max_gap: int,
) -> dict[str, float | int | None]:
    matched, _, lead_times = _match_warnings_to_events(warnings, [event], max_gap)
    if not matched:
        return {
            "detections": 0,
            "rate": 0.0,
            "median_delay": None,
        }
    return {
        "detections": 1,
        "rate": 1.0,
        "median_delay": float(np.median(lead_times)),
    }


def run_kuairand_active_benchmark(
    config: KuaiRandConfig | None = None,
) -> KuaiRandBenchmarkResult:
    config = config or KuaiRandConfig()
    users = load_kuairand_users(config)
    if config.auto_calibrate_thresholds and users:
        healthy_tci = np.concatenate(
            [user.signals["tci"].to_numpy()[: user.random_end] for user in users]
        )
        healthy_tcie = np.concatenate(
            [user.signals["tcie"].to_numpy()[: user.random_end] for user in users]
        )
        config = KuaiRandConfig(
            data_dir=config.data_dir,
            window_size=config.window_size,
            min_phase_count=config.min_phase_count,
            max_users=config.max_users,
            seed=config.seed,
            tci_threshold=float(np.nanquantile(healthy_tci, config.threshold_quantile)),
            tcie_threshold=float(
                np.nanquantile(healthy_tcie, config.threshold_quantile)
            ),
            auto_calibrate_thresholds=config.auto_calibrate_thresholds,
            threshold_quantile=config.threshold_quantile,
            tcie_lambda=config.tcie_lambda,
            adwin_delta=config.adwin_delta,
            page_hinkley_delta=config.page_hinkley_delta,
            page_hinkley_threshold=config.page_hinkley_threshold,
            page_hinkley_alpha=config.page_hinkley_alpha,
            kswin_window_size=config.kswin_window_size,
            kswin_stat_size=config.kswin_stat_size,
            kswin_alpha=config.kswin_alpha,
        )
    user_results: list[KuaiRandUserDetectionResult] = []

    for user in users:
        tci_warnings = _threshold_warnings(
            user.signals["tci"].to_numpy(), config.tci_threshold
        )
        tcie_warnings = _threshold_warnings(
            user.signals["tcie"].to_numpy(), config.tcie_threshold
        )
        baseline_signal = 1.0 - user.signals["tcie"].to_numpy()
        raw_warnings = {
            detector_name: _run_detector(baseline_signal, detector_name, config)
            for detector_name in ACTIVE_BASELINE_DETECTORS
        }
        user_results.append(
            KuaiRandUserDetectionResult(
                user_id=user.user_id,
                tci_warnings=tci_warnings,
                tcie_warnings=tcie_warnings,
                raw_warnings=raw_warnings,
                masking_detection={
                    "TCI": summarize_user_detection(
                        tci_warnings, user.random_end, config.window_size * 4
                    ),
                    "TCIE": summarize_user_detection(
                        tcie_warnings, user.random_end, config.window_size * 4
                    ),
                    **{
                        detector_name: summarize_user_detection(
                            warnings, user.random_end, config.window_size * 4
                        )
                        for detector_name, warnings in raw_warnings.items()
                    },
                },
                collapse_detection={
                    "TCI": summarize_user_detection(
                        tci_warnings, user.coercive_end, config.window_size * 4
                    ),
                    "TCIE": summarize_user_detection(
                        tcie_warnings, user.coercive_end, config.window_size * 4
                    ),
                    **{
                        detector_name: summarize_user_detection(
                            warnings, user.coercive_end, config.window_size * 4
                        )
                        for detector_name, warnings in raw_warnings.items()
                    },
                },
            )
        )

    summary_rows = build_kuairand_summary_rows(user_results)
    return KuaiRandBenchmarkResult(
        config=config,
        user_results=user_results,
        user_signals=users,
        summary_rows=summary_rows,
    )


def build_kuairand_summary_rows(
    results: list[KuaiRandUserDetectionResult],
) -> list[dict[str, str | float | int]]:
    rows: list[dict[str, str | float | int]] = []
    phase_labels = {
        "masking_detection": "bubble_detection",
        "collapse_detection": "collapse_detection",
    }
    for phase in ("masking_detection", "collapse_detection"):
        for detector in ("TCI", "TCIE", *ACTIVE_BASELINE_DETECTORS):
            detections = 0
            delays: list[float] = []
            for result in results:
                summary = getattr(result, phase)[detector]
                if summary["detections"]:
                    detections += 1
                    if summary["median_delay"] is not None:
                        delays.append(float(summary["median_delay"]))
            rows.append(
                {
                    "phase": phase_labels[phase],
                    "detector": detector,
                    "n_users": len(results),
                    "detections": detections,
                    "rate": round(detections / len(results), 3) if results else 0.0,
                    "median_delay": round(float(np.median(delays)), 1)
                    if delays
                    else "NA",
                }
            )
    return rows


def save_kuairand_figure(result: KuaiRandBenchmarkResult, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Use median trajectory aligned by phase fractions.
    max_len = max(len(user.signals) for user in result.user_signals)
    grid = np.full((len(result.user_signals), max_len), np.nan)
    grid_tcie = np.full((len(result.user_signals), max_len), np.nan)
    for row, user in enumerate(result.user_signals):
        tci = user.signals["tci"].to_numpy()
        tcie = user.signals["tcie"].to_numpy()
        grid[row, : len(tci)] = tci
        grid_tcie[row, : len(tcie)] = tcie

    med_tci = np.nanmedian(grid, axis=0)
    med_tcie = np.nanmedian(grid_tcie, axis=0)
    phases = [
        np.nanmedian([user.random_end for user in result.user_signals]),
        np.nanmedian([user.coercive_end for user in result.user_signals]),
    ]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(med_tci, label="Score", linewidth=1.5)
    ax.plot(med_tcie, label="Effort-corrected score", linewidth=1.5)
    for boundary in phases:
        ax.axvline(boundary, color="0.4", linestyle="--", linewidth=1.0)
    ax.set_ylabel("Median score")
    ax.set_xlabel("Time step")
    ax.set_title("KuaiRand logged benchmark: median trajectories")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.2, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(output_path)
    fig.savefig(output_path.with_suffix(".png"), dpi=180)
    plt.close(fig)
