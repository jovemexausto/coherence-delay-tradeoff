from __future__ import annotations

import numpy as np

from ..core.types import SummaryRows
from .model import UMRBenchmarkResult, _fixed_window_estimate


def _display_method(method: str) -> str:
    labels = {
        "fixed": "fixed-100",
        "fixed_long": "fixed-500",
        "ewma": "EWMA",
        "adwin": "ADWIN",
        "cube": "ADWIN + UMR",
    }
    return labels.get(method, method)


def _phase_edges(
    result: UMRBenchmarkResult,
) -> tuple[np.ndarray, tuple[float, ...]]:
    if result.config.piecewise_drifts:
        drifts = result.config.piecewise_drifts
        if result.config.piecewise_lengths:
            edges = [0]
            for length in result.config.piecewise_lengths:
                edges.append(min(result.config.steps, edges[-1] + max(0, int(length))))
            edges[-1] = result.config.steps
            return np.asarray(edges, dtype=int), drifts
        edges = np.linspace(0, result.config.steps, len(drifts) + 1, dtype=int)
        return edges, drifts
    return np.asarray([0, result.config.steps], dtype=int), (result.config.drift,)


def build_summary_rows(result: UMRBenchmarkResult) -> SummaryRows:
    rows: list[dict[str, float | str | int]] = []
    for method, summary in result.summaries.items():
        rows.append(
            {
                "method": _display_method(method),
                "tail_mae_mean": round(summary.tail_mae_mean, 4),
                "tail_mae_std": round(summary.tail_mae_std, 4),
                "tail_rmse_mean": round(summary.tail_rmse_mean, 4),
                "tail_rmse_std": round(summary.tail_rmse_std, 4),
                "tail_width_mean": round(summary.tail_width_mean, 2),
                "tail_width_std": round(summary.tail_width_std, 2),
                "event_count_mean": round(summary.event_count_mean, 2),
                "event_count_std": round(summary.event_count_std, 2),
                "cap_count_mean": ""
                if summary.cap_count_mean is None
                else round(summary.cap_count_mean, 2),
                "cap_only_count_mean": ""
                if summary.cap_only_count_mean is None
                else round(summary.cap_only_count_mean, 2),
                "first_cap_time_mean": ""
                if summary.first_cap_time_mean is None
                else round(summary.first_cap_time_mean, 1),
                "first_drift_time_mean": ""
                if summary.first_drift_time_mean is None
                else round(summary.first_drift_time_mean, 1),
                "cap_before_drift_delay_mean": ""
                if summary.cap_before_drift_delay_mean is None
                else round(summary.cap_before_drift_delay_mean, 1),
            }
        )
    return rows


def build_event_rows(result: UMRBenchmarkResult) -> SummaryRows:
    representative = result.representative
    return [
        {
            "event": "drift_detected",
            "method": _display_method("adwin"),
            "count": len(representative.adwin_drift_detected),
            "first_time": representative.adwin_drift_detected[0]
            if representative.adwin_drift_detected
            else "",
        },
        {
            "event": "drift_detected",
            "method": _display_method("cube"),
            "count": len(representative.cube_drift_detected),
            "first_time": representative.cube_drift_detected[0]
            if representative.cube_drift_detected
            else "",
        },
        {
            "event": "cap_triggered",
            "method": _display_method("cube"),
            "count": len(representative.cube_cap_triggered),
            "first_time": representative.cube_cap_triggered[0]
            if representative.cube_cap_triggered
            else "",
        },
        {
            "event": "cap_only",
            "method": _display_method("cube"),
            "count": len(
                set(representative.cube_cap_triggered)
                - set(representative.cube_drift_detected)
            ),
            "first_time": next(
                (
                    t
                    for t in representative.cube_cap_triggered
                    if t not in set(representative.cube_drift_detected)
                ),
                "",
            ),
        },
    ]


def build_phase_rows(result: UMRBenchmarkResult) -> SummaryRows:
    edges, drifts = _phase_edges(result)

    rows: list[dict[str, float | int | str]] = []
    for phase, drift in enumerate(drifts):
        start = int(edges[phase])
        stop = int(edges[phase + 1])
        for method, series in result.series.items():
            rows.append(
                {
                    "phase": phase,
                    "start": start,
                    "stop": stop,
                    "drift": float(drift),
                    "method": _display_method(method),
                    "mean_abs_error": round(
                        float(np.mean(series.absolute_error_mean[start:stop])),
                        4,
                    ),
                    "mean_width": round(
                        float(np.mean(series.memory_horizon_mean[start:stop])),
                        2,
                    ),
                }
            )
    return rows


def build_oracle_phase_rows(result: UMRBenchmarkResult) -> SummaryRows:
    edges, drifts = _phase_edges(result)

    rows: list[dict[str, float | int | str]] = []
    for phase, drift in enumerate(drifts):
        start = int(edges[phase])
        stop = int(edges[phase + 1])
        oracle_scores: list[tuple[int, float]] = []
        for window in result.config.oracle_windows:
            per_seed_errors: list[float] = []
            for trace in result.traces:
                estimate, _ = _fixed_window_estimate(trace.observations, window)
                per_seed_errors.append(
                    float(
                        np.mean(
                            np.abs(trace.latent_mean[start:stop] - estimate[start:stop])
                        )
                    )
                )
            oracle_scores.append((window, float(np.mean(per_seed_errors))))
        oracle_window, oracle_error = min(oracle_scores, key=lambda item: item[1])

        cube = result.series["cube"]
        fixed = result.series["fixed"]
        fixed_long = result.series["fixed_long"]
        ewma = result.series["ewma"]
        adwin = result.series["adwin"]
        rows.append(
            {
                "phase": phase,
                "start": start,
                "stop": stop,
                "drift": float(drift),
                "oracle_window": int(oracle_window),
                "oracle_mae": round(oracle_error, 4),
                "cube_mae": round(
                    float(np.mean(cube.absolute_error_mean[start:stop])), 4
                ),
                "cube_mean_n_star": round(
                    float(np.mean(cube.memory_horizon_mean[start:stop])), 2
                ),
                "fixed_mae": round(
                    float(np.mean(fixed.absolute_error_mean[start:stop])), 4
                ),
                "fixed_long_mae": round(
                    float(np.mean(fixed_long.absolute_error_mean[start:stop])), 4
                ),
                "ewma_mae": round(
                    float(np.mean(ewma.absolute_error_mean[start:stop])), 4
                ),
                "adwin_mae": round(
                    float(np.mean(adwin.absolute_error_mean[start:stop])), 4
                ),
                "adwin_mean_width": round(
                    float(np.mean(adwin.memory_horizon_mean[start:stop])), 2
                ),
            }
        )
    return rows


def build_frontier_rows(
    result: UMRBenchmarkResult,
    windows: tuple[int, ...] | None = None,
) -> SummaryRows:
    windows = windows or tuple(range(10, 501, 10))
    rows: list[dict[str, float | int | str]] = []

    for window in windows:
        per_seed_mae: list[float] = []
        per_seed_width: list[float] = []
        for trace in result.traces:
            estimate, width = _fixed_window_estimate(trace.observations, window)
            tail_start = int(estimate.size * (1.0 - result.config.tail_fraction))
            per_seed_mae.append(
                float(
                    np.mean(
                        np.abs(trace.latent_mean[tail_start:] - estimate[tail_start:])
                    )
                )
            )
            per_seed_width.append(float(np.mean(width[tail_start:])))

        rows.append(
            {
                "method": "fixed_sweep",
                "window": int(window),
                "tail_mae_mean": round(float(np.mean(per_seed_mae)), 4),
                "tail_mae_std": round(float(np.std(per_seed_mae)), 4),
                "tail_width_mean": round(float(np.mean(per_seed_width)), 2),
                "tail_width_std": round(float(np.std(per_seed_width)), 2),
            }
        )

    for method in ("fixed", "fixed_long", "ewma", "adwin", "cube"):
        summary = result.summaries[method]
        rows.append(
            {
                "method": method,
                "window": "",
                "tail_mae_mean": round(summary.tail_mae_mean, 4),
                "tail_mae_std": round(summary.tail_mae_std, 4),
                "tail_width_mean": round(summary.tail_width_mean, 2),
                "tail_width_std": round(summary.tail_width_std, 2),
            }
        )

    return rows


def build_delay_rows(result: UMRBenchmarkResult) -> SummaryRows:
    cube = result.summaries["cube"]
    adwin = result.summaries["adwin"]
    lead_time = (
        None
        if cube.first_cap_time_mean is None or adwin.first_drift_time_mean is None
        else float(adwin.first_drift_time_mean - cube.first_cap_time_mean)
    )
    return [
        {
            "drift": round(float(result.config.drift), 6),
            "cube_first_cap_time_mean": ""
            if cube.first_cap_time_mean is None
            else round(cube.first_cap_time_mean, 2),
            "adwin_first_drift_time_mean": ""
            if adwin.first_drift_time_mean is None
            else round(adwin.first_drift_time_mean, 2),
            "lead_time_mean": "" if lead_time is None else round(lead_time, 2),
            "cube_cap_only_count_mean": ""
            if cube.cap_only_count_mean is None
            else round(cube.cap_only_count_mean, 2),
        }
    ]


def build_horizon_instability_rows(result: UMRBenchmarkResult) -> SummaryRows:
    rep = result.representative
    Ck = 1.0 if result.config.Ck is None else float(result.config.Ck)
    drift_path = np.maximum(rep.drift_path, 1e-6)
    oracle_horizon = np.power(Ck / drift_path, 2.0 / 3.0)
    oracle_delta = np.abs(np.diff(oracle_horizon, prepend=oracle_horizon[0]))
    fixed_100_regret = np.abs(rep.fixed_width - oracle_horizon)
    fixed_200_regret = np.abs(rep.fixed_long_width - oracle_horizon)
    adwin_regret = np.abs(rep.adwin_width - oracle_horizon)
    cube_regret = np.abs(rep.cube_width - oracle_horizon)
    adwin_log_regret = np.abs(
        np.log(np.maximum(rep.adwin_width, 1e-6) / oracle_horizon)
    )
    cube_log_regret = np.abs(np.log(np.maximum(rep.cube_width, 1e-6) / oracle_horizon))
    return [
        {
            "time": int(t),
            "drift": round(float(rep.drift_path[t]), 6),
            "oracle_horizon": round(float(oracle_horizon[t]), 2),
            "oracle_horizon_delta": round(float(oracle_delta[t]), 4),
            "fixed_100": round(float(rep.fixed_width[t]), 2),
            "fixed_200": round(float(rep.fixed_long_width[t]), 2),
            "adwin_width": round(float(rep.adwin_width[t]), 2),
            "cube_n_star": round(float(rep.cube_n_star[t]), 2),
            "cube_width": round(float(rep.cube_width[t]), 2),
            "fixed_100_regret": round(float(fixed_100_regret[t]), 2),
            "fixed_200_regret": round(float(fixed_200_regret[t]), 2),
            "adwin_regret": round(float(adwin_regret[t]), 2),
            "cube_regret": round(float(cube_regret[t]), 2),
            "adwin_log_regret": round(float(adwin_log_regret[t]), 4),
            "cube_log_regret": round(float(cube_log_regret[t]), 4),
            "cube_error": round(float(rep.cube_error[t]), 4),
            "adwin_error": round(float(rep.adwin_error[t]), 4),
            "fixed_error": round(float(rep.fixed_error[t]), 4),
            "fixed_long_error": round(float(rep.fixed_long_error[t]), 4),
        }
        for t in range(rep.drift_path.size)
    ]


def build_horizon_transition_rows(
    result: UMRBenchmarkResult, window_steps: int = 50
) -> SummaryRows:
    if not result.config.piecewise_drifts:
        return []

    edges, drifts = _phase_edges(result)
    Ck = 1.0 if result.config.Ck is None else float(result.config.Ck)
    methods = {
        "fixed_100": lambda trace: trace.fixed_width,
        "fixed_200": lambda trace: trace.fixed_long_width,
        "adwin": lambda trace: trace.adwin_width,
        "cube": lambda trace: trace.cube_width,
    }
    rows: list[dict[str, float | int | str]] = []

    for phase in range(len(drifts) - 1):
        boundary = int(edges[phase + 1])
        stop = int(edges[phase + 2])
        old_oracle = float(np.power(Ck / max(float(drifts[phase]), 1e-6), 2.0 / 3.0))
        new_oracle = float(
            np.power(Ck / max(float(drifts[phase + 1]), 1e-6), 2.0 / 3.0)
        )
        transition_type = "contraction" if new_oracle < old_oracle else "expansion"
        window_stop = min(stop, boundary + window_steps)

        for method, getter in methods.items():
            per_seed_regret: list[float] = []
            per_seed_width: list[float] = []
            for trace in result.traces:
                width = getter(trace)
                segment = width[boundary:window_stop]
                if segment.size == 0:
                    continue
                per_seed_regret.append(float(np.mean(np.abs(segment - new_oracle))))
                per_seed_width.append(float(np.mean(segment)))

            rows.append(
                {
                    "phase": phase,
                    "boundary": boundary,
                    "transition_type": transition_type,
                    "from_drift": round(float(drifts[phase]), 6),
                    "to_drift": round(float(drifts[phase + 1]), 6),
                    "from_oracle": round(old_oracle, 2),
                    "to_oracle": round(new_oracle, 2),
                    "window_steps": window_steps,
                    "method": method,
                    "window_width_mean": round(float(np.mean(per_seed_width)), 2)
                    if per_seed_width
                    else "",
                    "window_regret_mean": round(float(np.mean(per_seed_regret)), 2)
                    if per_seed_regret
                    else "",
                    "window_regret_std": round(float(np.std(per_seed_regret)), 2)
                    if per_seed_regret
                    else "",
                }
            )
    return rows


def build_drift_ema_ablation_rows(
    results: list[UMRBenchmarkResult],
    alphas: list[float],
) -> SummaryRows:
    rows: list[dict[str, float | int | str]] = []
    if len(results) != len(alphas):
        raise ValueError("results and alphas must have the same length")

    for alpha, result in zip(alphas, results, strict=True):
        transition_rows = [
            row
            for row in build_horizon_transition_rows(result)
            if row["method"] == "cube"
        ]
        contraction = [
            float(row["window_regret_mean"])
            for row in transition_rows
            if row["transition_type"] == "contraction"
            and row["window_regret_mean"] != ""
        ]
        expansion = [
            float(row["window_regret_mean"])
            for row in transition_rows
            if row["transition_type"] == "expansion" and row["window_regret_mean"] != ""
        ]
        summary = result.summaries["cube"]
        contraction_mean = float(np.mean(contraction)) if contraction else float("nan")
        expansion_mean = float(np.mean(expansion)) if expansion else float("nan")
        ratio = (
            expansion_mean / contraction_mean if contraction_mean > 0 else float("nan")
        )
        rows.append(
            {
                "drift_ema_alpha": round(float(alpha), 4),
                "contraction_regret_mean": round(contraction_mean, 2),
                "contraction_regret_std": round(float(np.std(contraction)), 2)
                if contraction
                else "",
                "expansion_regret_mean": round(expansion_mean, 2),
                "expansion_regret_std": round(float(np.std(expansion)), 2)
                if expansion
                else "",
                "expansion_to_contraction_ratio": round(ratio, 2),
                "tail_mae_mean": round(summary.tail_mae_mean, 4),
                "tail_mae_std": round(summary.tail_mae_std, 4),
                "cap_count_mean": ""
                if summary.cap_count_mean is None
                else round(summary.cap_count_mean, 2),
                "cap_only_count_mean": ""
                if summary.cap_only_count_mean is None
                else round(summary.cap_only_count_mean, 2),
                "first_cap_time_mean": ""
                if summary.first_cap_time_mean is None
                else round(summary.first_cap_time_mean, 2),
                "first_drift_time_mean": ""
                if summary.first_drift_time_mean is None
                else round(summary.first_drift_time_mean, 2),
                "cap_before_drift_delay_mean": ""
                if summary.cap_before_drift_delay_mean is None
                else round(summary.cap_before_drift_delay_mean, 2),
            }
        )
    return rows


def build_horizon_gap_curve_rows(
    result: UMRBenchmarkResult, bins: int = 24
) -> SummaryRows:
    methods = ("fixed", "fixed_long", "ewma", "adwin", "cube")
    width_attrs = {
        "fixed": "fixed_width",
        "fixed_long": "fixed_long_width",
        "ewma": "ewma_width",
        "adwin": "adwin_width",
        "cube": "cube_width",
    }

    absolute_points: dict[str, list[np.ndarray]] = {method: [] for method in methods}
    absolute_costs: dict[str, list[np.ndarray]] = {method: [] for method in methods}
    relative_points: dict[str, list[np.ndarray]] = {method: [] for method in methods}
    relative_costs: dict[str, list[np.ndarray]] = {method: [] for method in methods}

    for trace in result.traces:
        oracle_estimates = [
            _fixed_window_estimate(trace.observations, window)[0]
            for window in result.config.oracle_windows
        ]
        oracle_stack = np.stack(oracle_estimates)
        oracle_error = np.min(np.abs(oracle_stack - trace.latent_mean), axis=0)
        oracle_horizon = np.power(
            (1.0 if result.config.Ck is None else float(result.config.Ck))
            / np.maximum(trace.drift_path, 1e-6),
            2.0 / 3.0,
        )

        for method in methods:
            error = getattr(trace, f"{method}_error")
            width = getattr(trace, width_attrs[method])
            excess_error = error - oracle_error
            absolute_gap = np.abs(width - oracle_horizon)
            relative_gap = np.abs(np.log(np.maximum(width, 1e-6) / oracle_horizon))
            absolute_points[method].append(np.asarray(absolute_gap, dtype=float))
            absolute_costs[method].append(np.asarray(excess_error, dtype=float))
            relative_points[method].append(np.asarray(relative_gap, dtype=float))
            relative_costs[method].append(np.asarray(excess_error, dtype=float))

    def _bin_rows(
        *,
        gap_kind: str,
        point_map: dict[str, list[np.ndarray]],
        cost_map: dict[str, list[np.ndarray]],
    ) -> list[dict[str, float | int | str]]:
        transformed: dict[str, np.ndarray] = {}
        all_points: list[np.ndarray] = []
        for method in methods:
            if point_map[method]:
                points = np.concatenate(point_map[method])
                costs = np.concatenate(cost_map[method])
            else:
                points = np.asarray([], dtype=float)
                costs = np.asarray([], dtype=float)
            transformed[method] = np.log1p(points)
            all_points.append(transformed[method])

        valid = np.concatenate([points for points in all_points if points.size])
        if valid.size == 0:
            return []

        lo = float(np.min(valid))
        hi = float(np.max(valid))
        if np.isclose(lo, hi):
            hi = lo + 1e-6
        edges = np.linspace(lo, hi, bins + 1)

        rows: list[dict[str, float | int | str]] = []
        for method in methods:
            x = transformed[method]
            if point_map[method]:
                y = np.concatenate(cost_map[method])
            else:
                y = np.asarray([], dtype=float)
            if x.size == 0:
                continue
            bucket = np.clip(np.digitize(x, edges, right=False) - 1, 0, bins - 1)
            for index in range(bins):
                mask = bucket == index
                if not np.any(mask):
                    continue
                x_center = float(np.expm1((edges[index] + edges[index + 1]) / 2.0))
                values = y[mask]
                rows.append(
                    {
                        "gap_kind": gap_kind,
                        "method": method,
                        "gap_center": round(x_center, 6),
                        "mean_excess_error": round(float(np.mean(values)), 4),
                        "median_excess_error": round(float(np.median(values)), 4),
                        "q10_excess_error": round(float(np.quantile(values, 0.1)), 4),
                        "q90_excess_error": round(float(np.quantile(values, 0.9)), 4),
                        "count": int(values.size),
                    }
                )
        return rows

    return _bin_rows(
        gap_kind="absolute",
        point_map=absolute_points,
        cost_map=absolute_costs,
    ) + _bin_rows(
        gap_kind="relative",
        point_map=relative_points,
        cost_map=relative_costs,
    )
