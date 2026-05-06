from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from ..core.common import threshold_crossings
from ..core.detectors import run_river_drift_detector
from ..core.types import SummaryRows


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
    effort_proxy: str = "kl"
    effort_scale_multiplier: float = 1.0
    ewma_alpha: float = 0.15
    tcie_ewma_threshold: float = 0.80
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
    tcie_ewma_warnings: list[int]
    raw_warnings: dict[str, list[int]]
    masking_detection: dict[str, dict[str, float | int | None]]
    collapse_detection: dict[str, dict[str, float | int | None]]


@dataclass(slots=True)
class KuaiRandBenchmarkResult:
    config: KuaiRandConfig
    user_results: list[KuaiRandUserDetectionResult]
    user_signals: list[KuaiRandUserSignals]
    summary_rows: SummaryRows


def summarize_user_detection(
    warnings: list[int],
    event: int,
    max_gap: int,
) -> dict[str, float | int | None]:
    matched = [
        warning
        for warning in warnings
        if warning <= event and event - warning <= max_gap
    ]
    if not matched:
        return {"detections": 0, "rate": 0.0, "median_delay": None}
    lead_times = [event - warning for warning in matched]
    return {"detections": 1, "rate": 1.0, "median_delay": float(np.median(lead_times))}


ACTIVE_BASELINE_DETECTORS = ("ADWIN", "PageHinkley", "KSWIN", "NoDrift")


def _read_csv(base: Path, filename: str, usecols: tuple[str, ...]) -> pd.DataFrame:
    return cast(
        pd.DataFrame,
        pd.read_csv(base / filename, usecols=usecols),  # pyright: ignore[reportCallIssue, reportArgumentType]
    )


def _read_logs(base: Path) -> pd.DataFrame:
    cols: tuple[str, ...] = (
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
    )
    random_logs = _read_csv(base, "log_random_4_22_to_5_08_pure.csv", cols)
    random_logs["phase"] = "healthy"
    standard_logs = _read_csv(base, "log_standard_4_22_to_5_08_pure.csv", cols)
    standard_logs["phase"] = "later"
    logs = cast(
        pd.DataFrame, pd.concat([random_logs, standard_logs], ignore_index=True)
    )
    logs["watch_ratio"] = logs["play_time_ms"] / logs["duration_ms"].clip(lower=1)
    logs["watch_ratio"] = logs["watch_ratio"].replace([np.inf, -np.inf], np.nan)
    logs = logs.dropna(subset=["watch_ratio"])
    return logs


def _load_video_tags(base: Path) -> pd.Series:
    video = _read_csv(base, "video_features_basic_pure.csv", ("video_id", "tag"))
    return cast(pd.Series, video.set_index("video_id")["tag"].astype(str))


def _kl_divergence(
    p: dict[str, float], q: dict[str, float], alpha: float = 1e-6
) -> float:
    keys = sorted(set(p) | set(q))
    p_arr = np.asarray([p.get(key, 0.0) + alpha for key in keys], dtype=float)
    q_arr = np.asarray([q.get(key, 0.0) + alpha for key in keys], dtype=float)
    p_arr /= p_arr.sum()
    q_arr /= q_arr.sum()
    return float(np.sum(p_arr * np.log(p_arr / q_arr)))


def _total_variation_distance(p: dict[str, float], q: dict[str, float]) -> float:
    keys = sorted(set(p) | set(q))
    if not keys:
        return 0.0
    return 0.5 * float(sum(abs(p.get(key, 0.0) - q.get(key, 0.0)) for key in keys))


def _gini_coefficient(dist: dict[str, float]) -> float:
    if not dist:
        return 0.0
    probs = np.sort(np.asarray(list(dist.values()), dtype=float))
    total = float(probs.sum())
    if total <= 0.0:
        return 0.0
    count = probs.size
    weighted = float(np.sum((2 * np.arange(1, count + 1) - count - 1) * probs))
    gini = weighted / (count * total)
    max_gini = (count - 1) / count if count > 1 else 1.0
    if max_gini <= 0.0:
        return 0.0
    return float(np.clip(gini / max_gini, 0.0, 1.0))


def _effort_proxy_value(
    current_tag_dist: dict[str, float],
    baseline_tag_dist: dict[str, float],
    proxy_name: str,
) -> float:
    if proxy_name == "kl":
        return _kl_divergence(current_tag_dist, baseline_tag_dist)
    if proxy_name == "tv":
        return _total_variation_distance(current_tag_dist, baseline_tag_dist)
    if proxy_name == "gini":
        return max(
            0.0,
            _gini_coefficient(current_tag_dist) - _gini_coefficient(baseline_tag_dist),
        )
    raise ValueError(f"Unknown effort proxy: {proxy_name}")


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


def _ewma(values: np.ndarray, alpha: float) -> np.ndarray:
    if values.size == 0:
        return np.empty_like(values, dtype=float)
    smoothed = np.empty_like(values, dtype=float)
    ema = float(values[0])
    for index, value in enumerate(values):
        ema = alpha * float(value) + (1.0 - alpha) * ema
        smoothed[index] = ema
    return smoothed


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
    effort_proxy: str,
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

        proxy_effort = _effort_proxy_value(
            current_tag_dist,
            baseline_tag_dist,
            effort_proxy,
        )
        effort[index] = proxy_effort
        watch_gap = abs(current_watch_mean - baseline_watch_mean)
        sigma_p[index] = 1.0 / (1.0 + watch_gap / max(baseline_watch_std, 1e-6))
        entropy = _entropy(current_tag_dist)
        sigma_a[index] = 1.0 / (1.0 + entropy / np.log(max(tag_vocab_size, 2)))
        sigma_phi[index] = 1.0 / (1.0 + current_watch_std)
        tci[index] = min(sigma_p[index], sigma_a[index], sigma_phi[index])
        tcie[index] = min(
            sigma_p[index]
            * np.exp(-tcie_lambda * proxy_effort / max(baseline_effort_scale, 1e-6)),
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
    user_logs = cast(pd.DataFrame, user_logs.sort_values("time_ms").copy())
    if len(user_logs) < 2 * config.min_phase_count:
        return None

    healthy = cast(pd.DataFrame, user_logs[user_logs["phase"] == "healthy"])
    later = cast(pd.DataFrame, user_logs[user_logs["phase"] == "later"])
    if len(healthy) < config.min_phase_count or len(later) < 2 * config.min_phase_count:
        return None

    later = cast(pd.DataFrame, later.sort_values("time_ms"))
    split = len(later) // 2
    coercive = cast(pd.DataFrame, later.iloc[:split])
    collapse = cast(pd.DataFrame, later.iloc[split:])

    eval_logs = cast(
        pd.DataFrame,
        pd.concat([healthy, coercive, collapse], ignore_index=True),
    )
    eval_logs = cast(pd.DataFrame, eval_logs.sort_values("time_ms"))
    eval_tags = cast(
        list[str], tag_map.reindex(eval_logs["video_id"]).fillna("unknown").tolist()
    )
    eval_watch = cast(list[float], eval_logs["watch_ratio"].astype(float).tolist())

    healthy_tag_dist = _distribution(
        cast(list[str], tag_map.reindex(healthy["video_id"]).fillna("unknown").tolist())
    )
    tag_vocab_size = int(tag_map.nunique())
    healthy_watch = cast(pd.Series, healthy["watch_ratio"])
    healthy_watch_mean = float(healthy_watch.mean())
    healthy_watch_std = float(healthy_watch.std(ddof=0) or 1.0)
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
        config.effort_proxy,
    )
    healthy_effort_scale = float(np.nanmedian(effort[: len(healthy)]))
    if not np.isfinite(healthy_effort_scale) or healthy_effort_scale <= 1e-6:
        healthy_effort_scale = 1.0
    healthy_effort_scale *= config.effort_scale_multiplier

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
        config.effort_proxy,
    )
    tcie_ewma = _ewma(tcie, config.ewma_alpha)

    signals = pd.DataFrame(
        {
            "tag": eval_tags,
            "watch_ratio": eval_watch,
            "is_click": eval_logs["is_click"].astype(float).to_numpy(),
            "is_like": eval_logs["is_like"].astype(float).to_numpy(),
            "long_view": eval_logs["long_view"].astype(float).to_numpy(),
            "sigma_p": sigma_p,
            "sigma_a": sigma_a,
            "sigma_phi": sigma_phi,
            "effort": effort,
            "tci": tci,
            "tcie": tcie,
            "tcie_ewma": tcie_ewma,
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

    counts = cast(
        pd.DataFrame,
        logs.groupby(["user_id", "phase"]).size().unstack(fill_value=0),
    )
    healthy_counts = cast(pd.Series, counts["healthy"])
    later_counts = cast(pd.Series, counts["later"])
    eligible_index = cast(
        pd.Index,
        counts.index[
            (healthy_counts >= config.min_phase_count)
            & (later_counts >= 2 * config.min_phase_count)
        ],
    )
    eligible_users = cast(
        list[int],
        eligible_index.to_series()
        .sample(
            n=min(config.max_users, len(eligible_index)),
            random_state=config.seed,
            replace=False,
        )
        .astype(int)
        .tolist(),
    )

    user_signals: list[KuaiRandUserSignals] = []
    for user_id in eligible_users:
        user_logs = cast(pd.DataFrame, logs[logs["user_id"].astype(int) == user_id])
        prepared = _prepare_user_signals(user_id, user_logs, tag_map, config)
        if prepared is not None:
            user_signals.append(prepared)
    return user_signals


def run_kuairand_active_benchmark(
    config: KuaiRandConfig | None = None,
) -> KuaiRandBenchmarkResult:
    config = config or KuaiRandConfig()
    users = load_kuairand_users(config)
    if config.auto_calibrate_thresholds and users:
        healthy_tci = np.concatenate(
            [
                cast(np.ndarray, user.signals["tci"].to_numpy()[: user.random_end])
                for user in users
            ]
        )
        healthy_tcie = np.concatenate(
            [
                cast(np.ndarray, user.signals["tcie"].to_numpy()[: user.random_end])
                for user in users
            ]
        )
        healthy_tcie_ewma = np.concatenate(
            [
                cast(
                    np.ndarray, user.signals["tcie_ewma"].to_numpy()[: user.random_end]
                )
                for user in users
            ]
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
            ewma_alpha=config.ewma_alpha,
            tcie_ewma_threshold=float(
                np.nanquantile(healthy_tcie_ewma, config.threshold_quantile)
            ),
            auto_calibrate_thresholds=config.auto_calibrate_thresholds,
            threshold_quantile=config.threshold_quantile,
            tcie_lambda=config.tcie_lambda,
            effort_proxy=config.effort_proxy,
            effort_scale_multiplier=config.effort_scale_multiplier,
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
        tci_signal = cast(np.ndarray, user.signals["tci"].to_numpy())
        tcie_signal = cast(np.ndarray, user.signals["tcie"].to_numpy())
        tcie_ewma_signal = cast(np.ndarray, user.signals["tcie_ewma"].to_numpy())
        tci_warnings = threshold_crossings(tci_signal, config.tci_threshold)
        tcie_warnings = threshold_crossings(tcie_signal, config.tcie_threshold)
        tcie_ewma_warnings = threshold_crossings(
            tcie_ewma_signal, config.tcie_ewma_threshold
        )
        baseline_signal = cast(np.ndarray, 1.0 - tcie_signal)
        raw_warnings = {
            detector_name: run_river_drift_detector(
                baseline_signal,
                detector_name,
                adwin_delta=config.adwin_delta,
                page_hinkley_delta=config.page_hinkley_delta,
                page_hinkley_threshold=config.page_hinkley_threshold,
                page_hinkley_alpha=config.page_hinkley_alpha,
                kswin_window_size=config.kswin_window_size,
                kswin_stat_size=config.kswin_stat_size,
                kswin_alpha=config.kswin_alpha,
            )
            for detector_name in ACTIVE_BASELINE_DETECTORS
        }
        user_results.append(
            KuaiRandUserDetectionResult(
                user_id=user.user_id,
                tci_warnings=tci_warnings,
                tcie_warnings=tcie_warnings,
                tcie_ewma_warnings=tcie_ewma_warnings,
                raw_warnings=raw_warnings,
                masking_detection={
                    "TCI": summarize_user_detection(
                        tci_warnings, user.random_end, config.window_size * 4
                    ),
                    "TCIE": summarize_user_detection(
                        tcie_warnings, user.random_end, config.window_size * 4
                    ),
                    "TCIE-EWMA": summarize_user_detection(
                        tcie_ewma_warnings, user.random_end, config.window_size * 4
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
                    "TCIE-EWMA": summarize_user_detection(
                        tcie_ewma_warnings, user.coercive_end, config.window_size * 4
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

    return KuaiRandBenchmarkResult(
        config=config,
        user_results=user_results,
        user_signals=users,
        summary_rows=[],
    )
