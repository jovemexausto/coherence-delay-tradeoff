from __future__ import annotations

from collections import Counter, deque
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import numpy as np
from scipy.stats import binomtest, spearmanr

from ..core.common import (
    export_summary_csv,
    format_summary_markdown,
    threshold_crossings,
)
from .model import (
    KuaiRandBenchmarkResult,
    KuaiRandConfig,
    KuaiRandUserSignals,
    run_kuairand_active_benchmark,
)


DISPLAY_TO_INTERNAL = {
    "CI": "TCI",
    "CI^E": "TCIE",
    "CI^E-EWMA": "TCIE-EWMA",
    "ADWIN": "ADWIN",
    "PageHinkley": "PageHinkley",
    "KSWIN": "KSWIN",
    "NoDrift": "NoDrift",
}
INTERNAL_TO_DISPLAY = {value: key for key, value in DISPLAY_TO_INTERNAL.items()}
DETECTORS = tuple(DISPLAY_TO_INTERNAL.keys())
PHASES = (
    ("masking_detection", "bubble_detection"),
    ("collapse_detection", "collapse_detection"),
)


def _clone_config(config: KuaiRandConfig, **updates: Any) -> KuaiRandConfig:
    return KuaiRandConfig(**{**asdict(config), **updates})


def _distribution(values: list[str]) -> dict[str, float]:
    if not values:
        return {}
    counts = Counter(values)
    total = float(sum(counts.values()))
    return {key: count / total for key, count in counts.items()}


def _normalized_entropy(values: list[str]) -> float:
    dist = _distribution(values)
    if not dist:
        return 0.0
    probs = np.asarray(list(dist.values()), dtype=float)
    entropy = float(-(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum())
    return entropy / np.log(max(len(probs), 2))


def _detector_warnings(
    result: KuaiRandBenchmarkResult,
    user_index: int,
    detector: str,
) -> list[int]:
    user_result = result.user_results[user_index]
    internal = DISPLAY_TO_INTERNAL.get(detector, detector)
    if internal == "TCI":
        return user_result.tci_warnings
    if internal == "TCIE":
        return user_result.tcie_warnings
    if internal == "TCIE-EWMA":
        return user_result.tcie_ewma_warnings
    return user_result.raw_warnings[internal]


def _detection_array(
    result: KuaiRandBenchmarkResult,
    phase: str,
    detector: str,
) -> np.ndarray:
    internal = DISPLAY_TO_INTERNAL.get(detector, detector)
    return np.asarray(
        [
            int(getattr(user_result, phase)[internal]["detections"])
            for user_result in result.user_results
        ],
        dtype=float,
    )


def _healthy_false_positives_per_user(
    result: KuaiRandBenchmarkResult,
    detector: str,
) -> np.ndarray:
    counts: list[float] = []
    for index, user in enumerate(result.user_signals):
        warnings = _detector_warnings(result, index, detector)
        counts.append(
            float(sum(1 for warning in warnings if warning < user.random_end))
        )
    return np.asarray(counts, dtype=float)


def _median_delay(
    result: KuaiRandBenchmarkResult,
    phase: str,
    detector: str,
) -> float | str:
    internal = DISPLAY_TO_INTERNAL.get(detector, detector)
    delays = [
        float(getattr(user_result, phase)[internal]["median_delay"])
        for user_result in result.user_results
        if getattr(user_result, phase)[internal]["median_delay"] is not None
    ]
    return round(float(np.median(delays)), 1) if delays else "NA"


def _bootstrap_mean_ci(
    values: np.ndarray,
    *,
    seed: int,
    n_boot: int = 2000,
) -> tuple[float, float, float]:
    if values.size == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(n_boot, values.size))
    samples = values[indices].mean(axis=1)
    return (
        float(values.mean()),
        float(np.quantile(samples, 0.025)),
        float(np.quantile(samples, 0.975)),
    )


def _bootstrap_diff_ci(
    left: np.ndarray,
    right: np.ndarray,
    *,
    seed: int,
    n_boot: int = 2000,
) -> tuple[float, float, float]:
    if left.size == 0 or right.size == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    left_idx = rng.integers(0, left.size, size=(n_boot, left.size))
    right_idx = rng.integers(0, right.size, size=(n_boot, right.size))
    diffs = left[left_idx].mean(axis=1) - right[right_idx].mean(axis=1)
    return (
        float(left.mean() - right.mean()),
        float(np.quantile(diffs, 0.025)),
        float(np.quantile(diffs, 0.975)),
    )


def _paired_binary_difference(
    baseline: np.ndarray,
    comparator: np.ndarray,
) -> tuple[float, float, float, float]:
    diff = comparator - baseline
    mean_diff, ci_low, ci_high = _bootstrap_mean_ci(diff, seed=17)
    improved = int(np.sum((baseline == 0) & (comparator == 1)))
    regressed = int(np.sum((baseline == 1) & (comparator == 0)))
    discordant = improved + regressed
    if discordant == 0:
        p_value = 1.0
    else:
        p_value = float(binomtest(min(improved, regressed), n=discordant, p=0.5).pvalue)
    return mean_diff, ci_low, ci_high, p_value


def build_default_followup_rows(
    result: KuaiRandBenchmarkResult,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for phase_key, phase_label in PHASES:
        for detector in DETECTORS:
            detections = _detection_array(result, phase_key, detector)
            mean_rate, ci_low, ci_high = _bootstrap_mean_ci(detections, seed=13)
            healthy_fp = _healthy_false_positives_per_user(result, detector)
            rows.append(
                {
                    "phase": phase_label,
                    "detector": detector,
                    "rate": round(mean_rate, 3),
                    "ci_low": round(ci_low, 3),
                    "ci_high": round(ci_high, 3),
                    "median_delay": _median_delay(result, phase_key, detector),
                    "healthy_fp_per_user": round(float(np.mean(healthy_fp)), 3),
                }
            )
    return rows


def build_improvement_rows(result: KuaiRandBenchmarkResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for phase_key, phase_label in PHASES:
        tci = _detection_array(result, phase_key, "CI")
        tcie = _detection_array(result, phase_key, "CI^E")
        diff, ci_low, ci_high, p_value = _paired_binary_difference(tci, tcie)
        rows.append(
            {
                "phase": phase_label,
                "comparison": "CI^E - CI",
                "delta_rate": round(diff, 3),
                "ci_low": round(ci_low, 3),
                "ci_high": round(ci_high, 3),
                "p_value": round(p_value, 6),
            }
        )
    return rows


def build_lambda_rows(
    *,
    base_config: KuaiRandConfig,
    lambdas: list[float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lambda_value in lambdas:
        result = run_kuairand_active_benchmark(
            _clone_config(base_config, tcie_lambda=lambda_value)
        )
        healthy_fp = _healthy_false_positives_per_user(result, "CI^E")
        for phase_key, phase_label in PHASES:
            detections = _detection_array(result, phase_key, "CI^E")
            mean_rate, ci_low, ci_high = _bootstrap_mean_ci(
                detections, seed=int(100 * lambda_value) + 1
            )
            rows.append(
                {
                    "lambda": lambda_value,
                    "phase": phase_label,
                    "proxy": base_config.effort_proxy,
                    "e0_scale": base_config.effort_scale_multiplier,
                    "rate": round(mean_rate, 3),
                    "ci_low": round(ci_low, 3),
                    "ci_high": round(ci_high, 3),
                    "healthy_fp_per_user": round(float(np.mean(healthy_fp)), 3),
                }
            )
    return rows


def build_e0_rows(
    *,
    base_config: KuaiRandConfig,
    e0_scales: list[float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scale in e0_scales:
        result = run_kuairand_active_benchmark(
            _clone_config(base_config, effort_scale_multiplier=scale)
        )
        healthy_fp = _healthy_false_positives_per_user(result, "CI^E")
        for phase_key, phase_label in PHASES:
            detections = _detection_array(result, phase_key, "CI^E")
            mean_rate, ci_low, ci_high = _bootstrap_mean_ci(
                detections, seed=int(100 * scale) + 3
            )
            rows.append(
                {
                    "e0_scale": scale,
                    "phase": phase_label,
                    "proxy": base_config.effort_proxy,
                    "lambda": base_config.tcie_lambda,
                    "rate": round(mean_rate, 3),
                    "ci_low": round(ci_low, 3),
                    "ci_high": round(ci_high, 3),
                    "healthy_fp_per_user": round(float(np.mean(healthy_fp)), 3),
                }
            )
    return rows


def build_proxy_rows(
    *,
    base_config: KuaiRandConfig,
    proxies: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for proxy_name in proxies:
        result = run_kuairand_active_benchmark(
            _clone_config(base_config, effort_proxy=proxy_name)
        )
        healthy_fp = _healthy_false_positives_per_user(result, "CI^E")
        for phase_key, phase_label in PHASES:
            detections = _detection_array(result, phase_key, "CI^E")
            mean_rate, ci_low, ci_high = _bootstrap_mean_ci(
                detections, seed=11 + len(proxy_name)
            )
            rows.append(
                {
                    "proxy": proxy_name,
                    "phase": phase_label,
                    "lambda": base_config.tcie_lambda,
                    "e0_scale": base_config.effort_scale_multiplier,
                    "rate": round(mean_rate, 3),
                    "ci_low": round(ci_low, 3),
                    "ci_high": round(ci_high, 3),
                    "healthy_fp_per_user": round(float(np.mean(healthy_fp)), 3),
                }
            )
    return rows


def build_threshold_rows(
    *,
    base_config: KuaiRandConfig,
    quantiles: list[float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for quantile in quantiles:
        result = run_kuairand_active_benchmark(
            _clone_config(base_config, threshold_quantile=quantile)
        )
        for detector in ("CI", "CI^E", "CI^E-EWMA"):
            healthy_fp = _healthy_false_positives_per_user(result, detector)
            for phase_key, phase_label in PHASES:
                detections = _detection_array(result, phase_key, detector)
                mean_rate, ci_low, ci_high = _bootstrap_mean_ci(
                    detections, seed=int(1000 * quantile) + len(detector)
                )
                rows.append(
                    {
                        "threshold_quantile": quantile,
                        "detector": detector,
                        "phase": phase_label,
                        "rate": round(mean_rate, 3),
                        "ci_low": round(ci_low, 3),
                        "ci_high": round(ci_high, 3),
                        "healthy_fp_per_user": round(float(np.mean(healthy_fp)), 3),
                    }
                )
    return rows


def _build_offpolicy_signal(user: KuaiRandUserSignals, window_size: int) -> np.ndarray:
    tags = cast(list[str], user.signals["tag"].tolist())
    watch = cast(list[float], user.signals["watch_ratio"].astype(float).tolist())
    signal = np.full(len(tags), np.nan)
    tag_window: deque[str] = deque(maxlen=window_size)
    watch_window: deque[float] = deque(maxlen=window_size)

    for index, (tag, reward) in enumerate(zip(tags, watch, strict=True)):
        tag_window.append(tag)
        watch_window.append(float(reward))
        current_tags = list(tag_window)
        current_rewards = np.asarray(list(watch_window), dtype=float)
        current_dist = _distribution(current_tags)
        weights = np.asarray(
            [
                user.baseline_tag_dist.get(current_tag, 0.0)
                / max(current_dist.get(current_tag, 0.0), 1e-6)
                for current_tag in current_tags
            ],
            dtype=float,
        )
        weights = np.clip(weights, 0.0, 10.0)
        if float(weights.sum()) <= 0.0:
            snips_estimate = user.baseline_watch_mean
        else:
            snips_estimate = float(np.dot(weights, current_rewards) / weights.sum())
        watch_gap = max(0.0, user.baseline_watch_mean - snips_estimate)
        signal[index] = 1.0 / (1.0 + watch_gap / max(user.baseline_watch_std, 1e-6))
    return signal


def build_offpolicy_rows(
    result: KuaiRandBenchmarkResult,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    healthy_scores = [
        _build_offpolicy_signal(user, result.config.window_size)[: user.random_end]
        for user in result.user_signals
    ]
    threshold = float(
        np.nanquantile(np.concatenate(healthy_scores), result.config.threshold_quantile)
    )

    phase_rows: list[dict[str, Any]] = []
    per_user_detection: dict[str, list[float]] = {
        phase_key: [] for phase_key, _ in PHASES
    }
    per_user_min_score: dict[str, list[float]] = {
        phase_key: [] for phase_key, _ in PHASES
    }
    per_user_tcie_min: dict[str, list[float]] = {
        phase_key: [] for phase_key, _ in PHASES
    }

    for user, user_result in zip(result.user_signals, result.user_results, strict=True):
        signal = _build_offpolicy_signal(user, result.config.window_size)
        warnings = threshold_crossings(signal, threshold)
        healthy_fp = float(sum(1 for warning in warnings if warning < user.random_end))
        for phase_key, phase_label in PHASES:
            event = (
                user.random_end
                if phase_key == "masking_detection"
                else user.coercive_end
            )
            start = event
            end = user.coercive_end if phase_key == "masking_detection" else len(signal)
            matched = [
                warning
                for warning in warnings
                if warning <= event and event - warning <= result.config.window_size * 4
            ]
            detected = 1.0 if matched else 0.0
            per_user_detection[phase_key].append(detected)
            per_user_min_score[phase_key].append(float(np.nanmin(signal[start:end])))
            per_user_tcie_min[phase_key].append(
                float(
                    np.nanmin(
                        cast(np.ndarray, user.signals["tcie"].to_numpy())[start:end]
                    )
                )
            )
            phase_rows.append(
                {
                    "user_id": user.user_id,
                    "phase": phase_label,
                    "detected": int(detected),
                    "healthy_fp": healthy_fp,
                }
            )

    summary_rows: list[dict[str, Any]] = []
    healthy_fp_mean = round(
        float(
            np.mean(
                [
                    row["healthy_fp"]
                    for row in phase_rows
                    if row["phase"] == "bubble_detection"
                ]
            )
        ),
        3,
    )
    for phase_key, phase_label in PHASES:
        detections = np.asarray(per_user_detection[phase_key], dtype=float)
        mean_rate, ci_low, ci_high = _bootstrap_mean_ci(detections, seed=29)
        tcie_detection = _detection_array(result, phase_key, "CI^E")
        agreement = float(np.mean(detections == tcie_detection))
        corr = spearmanr(
            per_user_min_score[phase_key], per_user_tcie_min[phase_key]
        ).statistic
        summary_rows.append(
            {
                "phase": phase_label,
                "detector": "tag_snips_proxy",
                "threshold": round(threshold, 3),
                "rate": round(mean_rate, 3),
                "ci_low": round(ci_low, 3),
                "ci_high": round(ci_high, 3),
                "healthy_fp_per_user": healthy_fp_mean,
                "agreement_with_ci_e": round(agreement, 3),
                "spearman_min_score_vs_ci_e": round(float(corr), 3),
            }
        )
    return summary_rows, phase_rows


def _collapse_outcome_vectors(
    result: KuaiRandBenchmarkResult,
) -> dict[str, np.ndarray]:
    watch: list[float] = []
    likes: list[float] = []
    long_views: list[float] = []
    entropy: list[float] = []
    for user in result.user_signals:
        collapse = user.signals.iloc[user.coercive_end :]
        watch.append(float(collapse["watch_ratio"].mean()))
        likes.append(float(collapse["is_like"].mean()))
        long_views.append(float(collapse["long_view"].mean()))
        entropy.append(_normalized_entropy(cast(list[str], collapse["tag"].tolist())))
    return {
        "collapse_watch_ratio": np.asarray(watch, dtype=float),
        "collapse_like_rate": np.asarray(likes, dtype=float),
        "collapse_long_view_rate": np.asarray(long_views, dtype=float),
        "collapse_tag_entropy": np.asarray(entropy, dtype=float),
    }


def build_downstream_rows(result: KuaiRandBenchmarkResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    outcomes = _collapse_outcome_vectors(result)
    bubble_tci = _detection_array(result, "masking_detection", "CI").astype(bool)
    bubble_tcie = _detection_array(result, "masking_detection", "CI^E").astype(bool)
    comparisons = {
        "CI flagged vs not": (bubble_tci, ~bubble_tci),
        "CI^E flagged vs not": (bubble_tcie, ~bubble_tcie),
        "CI^E-only vs neither": (bubble_tcie & ~bubble_tci, ~bubble_tcie & ~bubble_tci),
    }
    for comparison, (left_mask, right_mask) in comparisons.items():
        for outcome_name, outcome_values in outcomes.items():
            left = outcome_values[left_mask]
            right = outcome_values[right_mask]
            diff, ci_low, ci_high = _bootstrap_diff_ci(left, right, seed=41)
            rows.append(
                {
                    "comparison": comparison,
                    "outcome": outcome_name,
                    "left_mean": round(float(left.mean()), 3) if left.size else "NA",
                    "right_mean": round(float(right.mean()), 3) if right.size else "NA",
                    "delta": round(diff, 3),
                    "ci_low": round(ci_low, 3),
                    "ci_high": round(ci_high, 3),
                    "n_left": int(left.size),
                    "n_right": int(right.size),
                }
            )
    return rows


def write_followup_report(
    output_dir: Path, base_config: KuaiRandConfig | None = None
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    base_config = base_config or KuaiRandConfig(max_users=367)
    default_result = run_kuairand_active_benchmark(base_config)

    default_rows = build_default_followup_rows(default_result)
    improvement_rows = build_improvement_rows(default_result)
    lambda_rows = build_lambda_rows(
        base_config=base_config, lambdas=[0.5, 1.0, 2.0, 3.0, 4.0]
    )
    e0_rows = build_e0_rows(base_config=base_config, e0_scales=[0.5, 1.0, 2.0])
    proxy_rows = build_proxy_rows(base_config=base_config, proxies=["kl", "tv", "gini"])
    threshold_rows = build_threshold_rows(
        base_config=base_config, quantiles=[0.1, 0.2, 0.3]
    )
    offpolicy_summary_rows, offpolicy_user_rows = build_offpolicy_rows(default_result)
    downstream_rows = build_downstream_rows(default_result)

    outputs = {
        "default_csv": output_dir / "kuairand_followup_default.csv",
        "improvement_csv": output_dir / "kuairand_followup_improvement.csv",
        "lambda_csv": output_dir / "kuairand_followup_lambda.csv",
        "e0_csv": output_dir / "kuairand_followup_e0.csv",
        "proxy_csv": output_dir / "kuairand_followup_proxy.csv",
        "threshold_csv": output_dir / "kuairand_followup_threshold.csv",
        "offpolicy_csv": output_dir / "kuairand_followup_offpolicy.csv",
        "offpolicy_user_csv": output_dir / "kuairand_followup_offpolicy_user.csv",
        "downstream_csv": output_dir / "kuairand_followup_downstream.csv",
        "report_md": output_dir / "kuairand_followup_report.md",
    }

    export_summary_csv(default_rows, outputs["default_csv"])
    export_summary_csv(improvement_rows, outputs["improvement_csv"])
    export_summary_csv(lambda_rows, outputs["lambda_csv"])
    export_summary_csv(e0_rows, outputs["e0_csv"])
    export_summary_csv(proxy_rows, outputs["proxy_csv"])
    export_summary_csv(threshold_rows, outputs["threshold_csv"])
    export_summary_csv(offpolicy_summary_rows, outputs["offpolicy_csv"])
    export_summary_csv(offpolicy_user_rows, outputs["offpolicy_user_csv"])
    export_summary_csv(downstream_rows, outputs["downstream_csv"])

    report = "\n".join(
        [
            "# KuaiRand Follow-up Analyses",
            "",
            "## Default Summary with 95% bootstrap intervals",
            "",
            format_summary_markdown(default_rows),
            "## Paired improvement of CI^E over CI",
            "",
            format_summary_markdown(improvement_rows),
            "## Lambda sensitivity for CI^E",
            "",
            format_summary_markdown(lambda_rows),
            "## E0 sensitivity for CI^E",
            "",
            format_summary_markdown(e0_rows),
            "## Effort proxy sensitivity for CI^E",
            "",
            format_summary_markdown(proxy_rows),
            "## Threshold quantile sensitivity",
            "",
            format_summary_markdown(threshold_rows),
            "## Tag-reweighted off-policy control",
            "",
            "This control uses a clipped self-normalized tag-frequency reweighting signal.",
            "It is a logged-data sanity check, not a causal IPS guarantee, because the",
            "repository does not contain item-level propensities or replica policies.",
            "",
            format_summary_markdown(offpolicy_summary_rows),
            "## Downstream consequences after bubble flags",
            "",
            "These rows compare collapse-phase outcomes for users flagged early versus",
            "users not flagged early. They are not causal welfare estimates, but they do",
            "test whether early `CI^E` alarms line up with worse downstream behavior.",
            "",
            format_summary_markdown(downstream_rows),
        ]
    )
    outputs["report_md"].write_text(report, encoding="utf-8")
    return outputs
