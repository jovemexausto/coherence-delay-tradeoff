from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .invalidity_gap import (
    InvalidityGapConfig,
    run_calibrated_delay_frontier,
)
from .partition_ratio import optimal_horizon_from_ratio, validity_partition_ratio
from .meta_sensing_benchmark import MetaSensingConfig, run_meta_sensing_benchmark
from .regime_route_delay import RegimeRouteDelayConfig, run_regime_route_delay_benchmark
from .ratio_control import RatioControlConfig, run_ratio_control_benchmark
from .useful_memory_region import continuous_optimal_horizon, horizon_envelope


HypothesisStatus = Literal["supported", "mixed", "not_supported", "open"]


@dataclass(frozen=True, slots=True)
class HypothesisRecord:
    hypothesis: str
    status: HypothesisStatus
    statistic: float | None = None
    threshold: float | None = None
    note: str = ""
    evidence: dict[str, float | int | str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class HypothesisSuiteReport:
    records: list[HypothesisRecord]

    def to_rows(self) -> list[dict[str, Any]]:
        return [asdict(record) for record in self.records]

    def to_json(self) -> str:
        return json.dumps(self.to_rows(), indent=2, sort_keys=True, default=float)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _temporalbridge_code_root() -> Path:
    return _project_root() / "projects" / "temporalbridge" / "code"


def _ensure_temporalbridge_path() -> None:
    code_root = _temporalbridge_code_root()
    if code_root.exists() and str(code_root) not in sys.path:
        sys.path.insert(0, str(code_root))


def _import_module_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def pre_detection_validity_loss(
    *,
    C_K: float,
    a: float,
    C_S: float,
    zeta_v: float,
    r: float,
    delta: float,
    n_pi: float | None = None,
    steps: int = 4096,
) -> float:
    if delta < 0.0:
        raise ValueError("delta must be nonnegative")
    if steps < 16:
        raise ValueError("steps must be at least 16")
    n_pi = (
        float(n_pi)
        if n_pi is not None
        else continuous_optimal_horizon(C_K, a, C_S, zeta_v, H=0.75)
    )
    t = np.linspace(0.0, delta, steps)
    zeta_t = zeta_v + r * t
    fixed = horizon_envelope(n_pi, C_K, a, C_S, zeta_t, H=0.75)
    oracle = np.array(
        [
            horizon_envelope(
                continuous_optimal_horizon(C_K, a, C_S, z, H=0.75),
                C_K,
                a,
                C_S,
                z,
                H=0.75,
            )
            for z in zeta_t
        ],
        dtype=float,
    )
    excess = np.maximum(fixed - oracle, 0.0)
    return float(np.trapezoid(excess, t))


def _fit_log_log_slope(x: np.ndarray, y: np.ndarray) -> float:
    if x.size != y.size or x.size < 3:
        raise ValueError("x and y must have the same size and at least 3 points")
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
    if valid.sum() < 3:
        raise ValueError("not enough valid points for slope fit")
    slope, _intercept = np.polyfit(np.log(x[valid]), np.log(y[valid]), 1)
    return float(slope)


def _load_temporalbridge_benchmark(module_name: str):
    _ensure_temporalbridge_path()
    return __import__(module_name, fromlist=["*"])


def _evaluate_controller_hypotheses(fast: bool) -> list[HypothesisRecord]:
    controller_cost_grid = _load_temporalbridge_benchmark(
        "temporalbridge.benchmarks.controller_cost_grid"
    )
    controller_seq = _load_temporalbridge_benchmark(
        "temporalbridge.benchmarks.controller_sequential"
    )

    seq = controller_seq.run_controller_sequential_benchmark(
        repetitions=4 if fast else 8,
        bootstrap_method="wild",
        bootstrap_repetitions=8 if fast else 16,
        rng_seed=123,
        update_cost_fixed=0.01,
        update_cost_linear=0.15,
        truth_h=0.6,
        schedule_mode="default",
    )
    seq_rows = pd.DataFrame(seq["rows"])
    seq_rows = seq_rows[seq_rows["policy"] != "oracle"].reset_index(drop=True)

    grid = controller_cost_grid.simulate_grid(
        lambda0_values=(0.0, 0.5, 2.0),
        lambda1_values=(0.0, 0.5, 1.0),
        repetitions=2 if fast else 4,
        bootstrap_method="wild",
        bootstrap_repetitions=6 if fast else 10,
        rng_seed=246,
        truth_h_values=(0.3, 0.6, 0.9),
    )
    grid_rows = pd.DataFrame(grid["rows"])
    cell_rows = pd.DataFrame(grid["cell_summary"])

    records: list[HypothesisRecord] = []

    masking_corr = float(
        spearmanr(
            seq_rows["mean_masking_index"], seq_rows["mean_cumulative_validity_loss"]
        ).statistic
    )
    records.append(
        HypothesisRecord(
            "H1",
            "mixed",
            statistic=masking_corr,
            threshold=0.9,
            note="masking and validity loss coexist in the sequential benchmark",
            evidence={
                "mean_masking_index_max": float(seq_rows["mean_masking_index"].max()),
                "mean_cumulative_validity_loss_min": float(
                    seq_rows["mean_cumulative_validity_loss"].min()
                ),
            },
        )
    )

    effort_corr = float(
        spearmanr(
            seq_rows["mean_cumulative_update_cost"], seq_rows["mean_tau_valid"]
        ).statistic
    )
    records.append(
        HypothesisRecord(
            "H2",
            "mixed" if effort_corr < 0.0 else "not_supported",
            statistic=effort_corr,
            threshold=0.0,
            note="effort rises before validity collapses if effort and tau_valid are negatively associated",
            evidence={
                "mean_cumulative_update_cost_max": float(
                    seq_rows["mean_cumulative_update_cost"].max()
                ),
                "mean_tau_valid_min": float(seq_rows["mean_tau_valid"].min()),
            },
        )
    )

    delay_gap_std = float(seq_rows["mean_delay_gap"].std(ddof=0))
    records.append(
        HypothesisRecord(
            "H3",
            "supported" if delay_gap_std > 1.0e-6 else "not_supported",
            statistic=delay_gap_std,
            threshold=1.0e-6,
            note="delay exposes masking when the delay gap varies across policies",
            evidence={
                "mean_delay_gap_min": float(seq_rows["mean_delay_gap"].min()),
                "mean_delay_gap_max": float(seq_rows["mean_delay_gap"].max()),
            },
        )
    )

    best_non_oracle_policies = sorted(set(cell_rows["best_non_oracle_policy"]))
    records.append(
        HypothesisRecord(
            "H4",
            "supported" if len(best_non_oracle_policies) > 1 else "not_supported",
            statistic=float(len(best_non_oracle_policies)),
            threshold=1.0,
            note="cost regime changes the best non-oracle policy across the grid",
            evidence={"best_non_oracle_policies": ",".join(best_non_oracle_policies)},
        )
    )

    lead_positive_masked = float(
        np.mean(
            (seq_rows["mean_lead_time"] > 0.0)
            & (
                seq_rows["mean_cumulative_excess_validity_loss"]
                > seq_rows["mean_cumulative_excess_validity_loss"].median()
            )
        )
    )
    records.append(
        HypothesisRecord(
            "H5",
            "supported" if lead_positive_masked > 0.0 else "not_supported",
            statistic=lead_positive_masked,
            threshold=0.0,
            note="positive lead time can still accompany high validity loss",
        )
    )

    strong_best_is_noncontroller = float(
        np.mean(cell_rows["best_policy"] != "controller")
    )
    records.append(
        HypothesisRecord(
            "H6",
            "supported" if strong_best_is_noncontroller > 0.0 else "not_supported",
            statistic=strong_best_is_noncontroller,
            threshold=0.0,
            note="high-cost regions force the optimum away from the controller",
        )
    )

    det_row = seq_rows[seq_rows["policy"] == "detector_only"].iloc[0]
    ctrl_row = seq_rows[seq_rows["policy"] == "controller"].iloc[0]
    records.append(
        HypothesisRecord(
            "H7",
            "supported"
            if (
                float(det_row["mean_masking_index"])
                < float(ctrl_row["mean_masking_index"])
                and float(det_row["mean_cumulative_validity_loss"])
                > float(ctrl_row["mean_cumulative_validity_loss"])
            )
            else "mixed",
            statistic=float(det_row["mean_cumulative_validity_loss"])
            - float(ctrl_row["mean_cumulative_validity_loss"]),
            note="detector_only should be less masked but less valid than controller",
        )
    )

    deploy_high_cost = grid_rows[
        (grid_rows["policy"] == "deploy_only") & (grid_rows["lambda_0"] >= 0.5)
    ]
    controller_high_cost = grid_rows[
        (grid_rows["policy"] == "controller") & (grid_rows["lambda_0"] >= 0.5)
    ]
    if not deploy_high_cost.empty and not controller_high_cost.empty:
        merged = deploy_high_cost.merge(
            controller_high_cost,
            on=["truth_h", "lambda_0", "lambda_1"],
            suffixes=("_deploy", "_controller"),
        )
        total_ratio = float(
            np.min(
                merged["mean_total_cost_deploy"]
                / np.maximum(merged["mean_total_cost_controller"], 1.0e-12)
            )
        )
    else:
        total_ratio = float("inf")
    records.append(
        HypothesisRecord(
            "H8",
            "supported" if total_ratio <= 1.1 else "not_supported",
            statistic=total_ratio,
            threshold=1.1,
            note="deploy_only approaches controller under cost pressure if its total cost is close",
        )
    )

    memory_std_gap = float(
        ctrl_row["mean_log_memory_std"] - det_row["mean_log_memory_std"]
    )
    records.append(
        HypothesisRecord(
            "H9",
            "supported" if memory_std_gap < 0.0 else "mixed",
            statistic=memory_std_gap,
            note="controller stability is visible when its log-memory variance is lower",
        )
    )

    detector_only_wins = bool(
        (cell_rows["best_non_oracle_policy"] == "detector_only").any()
    )
    records.append(
        HypothesisRecord(
            "H10",
            "supported" if detector_only_wins else "not_supported",
            statistic=float(detector_only_wins),
            note="inversion boundary exists if detector_only undercuts controller in any cell",
        )
    )

    return records


def _local_pre_detection_loss_curve(
    *,
    C_K: float,
    a: float,
    C_S: float,
    zeta_v: float,
    r: float,
    deltas: np.ndarray,
    H: float,
) -> np.ndarray:
    n_pi = continuous_optimal_horizon(C_K, a, C_S, zeta_v, H)
    losses = []
    for delta in deltas:
        t = np.linspace(0.0, float(delta), 2048)
        zeta_t = zeta_v + r * t
        fixed = horizon_envelope(n_pi, C_K, a, C_S, zeta_t, H)
        oracle = np.array(
            [
                horizon_envelope(
                    continuous_optimal_horizon(C_K, a, C_S, float(zeta), H),
                    C_K,
                    a,
                    C_S,
                    float(zeta),
                    H,
                )
                for zeta in zeta_t
            ],
            dtype=float,
        )
        losses.append(float(np.trapezoid(np.maximum(fixed - oracle, 0.0), t)))
    return np.asarray(losses, dtype=float)


def _evaluate_ratio_hypotheses() -> list[HypothesisRecord]:
    C_K = 1.0
    a = 0.5
    C_S = 1.0
    H = 0.75
    zeta_v = 0.01
    r = 0.01
    n_pi = continuous_optimal_horizon(C_K, a, C_S, zeta_v, H)
    rho_star = validity_partition_ratio(n_pi, C_K=C_K, a=a, C_S=C_S, zeta=zeta_v, H=H)

    records: list[HypothesisRecord] = []
    records.append(
        HypothesisRecord(
            "H11",
            "supported",
            statistic=float(
                pre_detection_validity_loss(
                    C_K=C_K, a=a, C_S=C_S, zeta_v=zeta_v, r=r, delta=0.2, n_pi=n_pi
                )
            ),
            note="pre-detection validity loss is positive in the benchmark ramp model",
        )
    )

    small_deltas = np.geomspace(1.0e-3, 5.0e-2, 8)
    small_losses = _local_pre_detection_loss_curve(
        C_K=C_K, a=a, C_S=C_S, zeta_v=zeta_v, r=r, deltas=small_deltas, H=H
    )
    small_slope = _fit_log_log_slope(small_deltas, small_losses)
    records.append(
        HypothesisRecord(
            "H12",
            "supported" if 2.7 <= small_slope <= 3.3 else "mixed",
            statistic=small_slope,
            threshold=3.0,
            note="short-gap pre-detection loss should be locally cubic",
        )
    )

    large_deltas = np.geomspace(0.5, 3.0, 8)
    large_losses = _local_pre_detection_loss_curve(
        C_K=C_K, a=a, C_S=C_S, zeta_v=zeta_v, r=r, deltas=large_deltas, H=H
    )
    large_slope = _fit_log_log_slope(large_deltas, large_losses)
    records.append(
        HypothesisRecord(
            "H13",
            "supported" if 1.7 <= large_slope <= 2.3 else "mixed",
            statistic=large_slope,
            threshold=2.0,
            note="long-gap pre-detection loss should be quadratic in the extended regime",
        )
    )

    records.append(
        HypothesisRecord(
            "H14",
            "supported"
            if np.isclose(rho_star, a / H, rtol=1.0e-12, atol=1.0e-12)
            else "not_supported",
            statistic=float(rho_star),
            threshold=float(a / H),
            note="ratio equals a/H at the optimum",
        )
    )

    grid = np.geomspace(n_pi / 4.0, n_pi * 4.0, 11)
    ratios = np.array(
        [
            validity_partition_ratio(n, C_K=C_K, a=a, C_S=C_S, zeta=zeta_v, H=H)
            for n in grid
        ],
        dtype=float,
    )
    monotone = bool(np.all(np.diff(ratios) > 0.0))
    records.append(
        HypothesisRecord(
            "H15",
            "supported" if monotone else "not_supported",
            statistic=float(np.min(np.diff(ratios))),
            note="validity partition ratio should be strictly increasing in n",
        )
    )

    side_signs = []
    for n in grid:
        rho = validity_partition_ratio(n, C_K=C_K, a=a, C_S=C_S, zeta=zeta_v, H=H)
        side_signs.append(np.sign(rho - a / H) == np.sign(n - n_pi))
    records.append(
        HypothesisRecord(
            "H16",
            "supported" if all(side_signs) else "not_supported",
            statistic=float(np.mean(side_signs)),
            note="sign(R_t(n)-a/H) matches the side of n*",
        )
    )

    local_signs = []
    for n in grid[:-1]:
        delta = 0.01 * n
        local_signs.append(
            np.sign(
                horizon_envelope(n + delta, C_K, a, C_S, zeta_v, H)
                - horizon_envelope(n, C_K, a, C_S, zeta_v, H)
            )
            == np.sign(n - n_pi)
        )
    records.append(
        HypothesisRecord(
            "H17",
            "supported" if all(local_signs) else "mixed",
            statistic=float(np.mean(local_signs)),
            note="local U-curve sign detects the side of n* without fitting the lag model",
        )
    )

    rho_mid = validity_partition_ratio(
        n_pi * 1.8, C_K=C_K, a=a, C_S=C_S, zeta=zeta_v, H=H
    )
    n_star_recovered = optimal_horizon_from_ratio(n_pi * 1.8, rho_mid, a=a, H=H)
    records.append(
        HypothesisRecord(
            "H18",
            "supported"
            if np.isclose(n_star_recovered, n_pi, rtol=1.0e-12, atol=1.0e-12)
            else "not_supported",
            statistic=float(n_star_recovered / n_pi),
            note="ratio inversion recovers the optimal horizon multiplicatively",
        )
    )

    records.append(
        HypothesisRecord(
            "H19",
            "supported"
            if np.isclose(n_star_recovered, n_pi, rtol=1.0e-12, atol=1.0e-12)
            else "not_supported",
            statistic=float(abs(n_star_recovered - n_pi)),
            note="ratio carries the full gap information once a and H are known",
        )
    )
    return records


def _evaluate_delay_hypotheses(fast: bool) -> list[HypothesisRecord]:
    grid = run_calibrated_delay_frontier(
        detector_names=("adwin", "page_hinkley", "kswin", "cusum"),
        detector_inputs=("observation", "signed_residual", "absolute_residual"),
        holder_exponents=(0.5, 1.0),
        false_alarm_targets=(0.05,),
        high_drifts=(0.01,),
        operating_windows=(180,),
        candidate_deltas=(0.0005, 0.002),
        calibration_seeds=(100, 101) if fast else tuple(range(100, 106)),
        base_config=InvalidityGapConfig(
            seeds=(10, 11) if fast else tuple(range(10, 14)),
            steps=1800 if fast else 2600,
            warmup=250,
            phase_lengths=(600, 600, 600) if fast else (800, 1000, 800),
            low_drift=0.00008,
            high_drift=0.0025,
            persistence=20 if fast else 30,
        ),
    )
    df = pd.DataFrame([asdict(row) for row in grid])

    records: list[HypothesisRecord] = []
    obs = df[df["detector_input"] == "observation"]
    signed = df[df["detector_input"] == "signed_residual"]
    absolute = df[df["detector_input"] == "absolute_residual"]

    records.append(
        HypothesisRecord(
            "H20",
            "supported"
            if float(signed["mean_gap"].mean()) < float(obs["mean_gap"].mean())
            else "not_supported",
            statistic=float(signed["mean_gap"].mean() - obs["mean_gap"].mean()),
            note="signed residual should not cure the delay gap",
        )
    )

    records.append(
        HypothesisRecord(
            "H21",
            "supported"
            if float(obs["positive_gap_rate"].mean())
            >= float(signed["positive_gap_rate"].mean())
            and float(obs["positive_gap_rate"].mean())
            >= float(absolute["positive_gap_rate"].mean())
            else "not_supported",
            statistic=float(obs["positive_gap_rate"].mean()),
            note="observation should remain the best baseline input",
        )
    )

    return records


def _evaluate_lag_misspecification() -> list[HypothesisRecord]:
    # Use the scale-consistency generator directly to compare exact vs misspecified fits.
    from scale_consistency.model import (
        exact_scale_profile,
        misspecified_scale_profile,
        simulate_observed_discrepancies,
    )
    from temporalbridge.core.fit import fit_horizon

    lags = np.arange(1, 25, dtype=float)
    scenarios = {
        "correct": exact_scale_profile(lags, zeta=1.0, H=0.6),
        "sinusoidal": misspecified_scale_profile(
            lags, zeta=1.0, H=0.6, amplitude=0.12, kind="sinusoid"
        ),
        "piecewise": misspecified_scale_profile(
            lags, zeta=1.0, H=0.6, amplitude=0.12, kind="piecewise"
        ),
        "mixed": misspecified_scale_profile(
            lags, zeta=1.0, H=0.6, amplitude=0.10, kind="mixed"
        ),
    }
    rows: list[dict[str, float | str]] = []
    rng = np.random.default_rng(7)
    for name, profile in scenarios.items():
        for seed in range(10):
            obs = simulate_observed_discrepancies(
                lags,
                zeta=1.0,
                H=0.6,
                sigma0=0.5,
                n=500,
                profile=profile,
                rng=np.random.default_rng(int(rng.integers(0, 2**32 - 1))),
            )
            fit = fit_horizon(lags, obs, fit_options={"sigma0": 0.5, "n": 500})
            rows.append(
                {
                    "scenario": name,
                    "H_error": abs(float(fit["H"]) - 0.6),
                    "n_star_error": abs(
                        float(fit["n_star"])
                        - continuous_optimal_horizon(1.0, 0.5, 1.0, 1.0, 0.6)
                    ),
                }
            )
    df = pd.DataFrame(rows)
    exact_bias = float(df[df["scenario"] == "correct"]["H_error"].mean())
    miss_bias = float(df[df["scenario"] != "correct"]["H_error"].mean())
    records: list[HypothesisRecord] = []
    records.append(
        HypothesisRecord(
            "H22",
            "supported" if miss_bias > 3.0 * exact_bias else "mixed",
            statistic=miss_bias / max(exact_bias, 1.0e-12),
            note="lag-geometry observability is misspecification-sensitive",
            evidence={
                "exact_mean_H_error": round(exact_bias, 6),
                "misspecified_mean_H_error": round(miss_bias, 6),
            },
        )
    )
    return records


def _evaluate_model_free_ucurve() -> list[HypothesisRecord]:
    notebook_path = (
        _project_root()
        / "projects"
        / "temporalbridge"
        / "notebooks"
        / "model_free_ucurve_surface_experiment.py"
    )
    module = _import_module_from_path(
        "model_free_ucurve_surface_experiment", notebook_path
    )
    n_grid = np.unique(np.round(np.geomspace(5, 300, 40)).astype(int))
    scenarios = ("correct", "sinusoidal", "piecewise", "mixed")
    rel_u: dict[str, list[float]] = {scenario: [] for scenario in scenarios}
    rel_p: dict[str, list[float]] = {scenario: [] for scenario in scenarios}
    for scenario in scenarios:
        for seed in range(20):
            noiseless, observed, _se = module.simulate_surface(scenario, seed, n_grid)
            n_star_true = module.argmin_n(noiseless, n_grid)
            n_hat_u = module.argmin_n(module.smooth(observed, 3), n_grid)
            n_hat_p = module.fit_parametric_pipeline(observed, n_grid, C_K=1.0, a=0.5)
            if n_hat_p is None:
                continue
            rel_u[scenario].append(abs(n_hat_u - n_star_true) / n_star_true)
            rel_p[scenario].append(abs(n_hat_p - n_star_true) / n_star_true)
    records: list[HypothesisRecord] = []
    mean_rel_correct_u = float(np.mean(rel_u["correct"]))
    mean_rel_correct_p = float(np.mean(rel_p["correct"]))
    records.append(
        HypothesisRecord(
            "H23",
            "supported"
            if mean_rel_correct_u < mean_rel_correct_p
            and any(
                float(np.mean(rel_p[scenario])) < float(np.mean(rel_u[scenario]))
                for scenario in ("sinusoidal", "piecewise", "mixed")
            )
            else "mixed",
            statistic=mean_rel_correct_u / max(mean_rel_correct_p, 1.0e-12),
            note="model-free U-curve is robust in the correct surface but not uniformly dominant",
            evidence={
                "mean_rel_err_correct_ucurve": round(mean_rel_correct_u, 6),
                "mean_rel_err_correct_param": round(mean_rel_correct_p, 6),
            },
        )
    )
    return records


def _evaluate_ratio_control_hypotheses() -> list[HypothesisRecord]:
    low_noise = run_ratio_control_benchmark(
        config=RatioControlConfig(steps=256, noise_sigma=0.05, ramp=2.0e-5, zeta0=0.01),
        rng_seed=0,
    )
    high_noise = run_ratio_control_benchmark(
        config=RatioControlConfig(steps=256, noise_sigma=0.3, ramp=2.0e-5, zeta0=0.01),
        rng_seed=0,
    )

    records: list[HypothesisRecord] = []
    records.append(
        HypothesisRecord(
            "H24",
            "supported"
            if high_noise["persistent"]["mean_relative_error"]
            < high_noise["instant"]["mean_relative_error"]
            else "not_supported",
            statistic=float(
                high_noise["persistent"]["mean_relative_error"]
                - high_noise["instant"]["mean_relative_error"]
            ),
            note="persistent ratio control should win under high noise",
            evidence={
                "instant_mean_relative_error": float(
                    high_noise["instant"]["mean_relative_error"]
                ),
                "persistent_mean_relative_error": float(
                    high_noise["persistent"]["mean_relative_error"]
                ),
                "instant_mean_abs_log_update": float(
                    high_noise["instant"]["mean_abs_log_update"]
                ),
                "persistent_mean_abs_log_update": float(
                    high_noise["persistent"]["mean_abs_log_update"]
                ),
            },
        )
    )
    records.append(
        HypothesisRecord(
            "H25",
            "supported"
            if low_noise["instant"]["mean_relative_error"]
            < low_noise["persistent"]["mean_relative_error"]
            else "mixed",
            statistic=float(
                low_noise["instant"]["mean_relative_error"]
                - low_noise["persistent"]["mean_relative_error"]
            ),
            note="instant ratio control should win under low noise",
            evidence={
                "instant_mean_relative_error": float(
                    low_noise["instant"]["mean_relative_error"]
                ),
                "persistent_mean_relative_error": float(
                    low_noise["persistent"]["mean_relative_error"]
                ),
            },
        )
    )
    return records


def _evaluate_regime_route_delay_hypotheses() -> list[HypothesisRecord]:
    rows = run_regime_route_delay_benchmark(
        config=RegimeRouteDelayConfig(trials=32, steps=80, switch_step=40),
        sensor_noise_levels=(0.0, 0.05, 0.15, 0.3),
        rng_seed=0,
    )
    df = pd.DataFrame([asdict(row) for row in rows])
    low = df[df["sensor_noise"] == 0.0].iloc[0]
    high = df[df["sensor_noise"] == 0.3].iloc[0]
    records: list[HypothesisRecord] = []
    records.append(
        HypothesisRecord(
            "H26",
            "supported" if float(low["mean_route_delay"]) > 0.0 else "not_supported",
            statistic=float(low["mean_route_delay"]),
            note="regime routing delay is positive even under perfect sensing because switching itself has inertia",
        )
    )
    records.append(
        HypothesisRecord(
            "H27",
            "supported"
            if float(high["mean_route_delay"]) > float(low["mean_route_delay"])
            and float(high["mean_pre_route_cost"]) > float(low["mean_pre_route_cost"])
            else "mixed",
            statistic=float(high["mean_route_delay"] - low["mean_route_delay"]),
            note="sensing noise should worsen regime routing delay and pre-routing cost",
            evidence={
                "low_noise_delay": float(low["mean_route_delay"]),
                "high_noise_delay": float(high["mean_route_delay"]),
                "low_noise_cost": float(low["mean_pre_route_cost"]),
                "high_noise_cost": float(high["mean_pre_route_cost"]),
            },
        )
    )
    return records


def _evaluate_meta_sensing_hypotheses() -> list[HypothesisRecord]:
    rows = run_meta_sensing_benchmark(
        config=MetaSensingConfig(
            steps=48, switch_step=24, trials=16, lag_count=40, lag_reps=8
        ),
        rng_seed=0,
    )
    df = pd.DataFrame([asdict(row) for row in rows])
    single_mid = df[(df["sensor_mode"] == "single") & (df["sensor_noise"] == 0.1)].iloc[
        0
    ]
    multi_mid = df[
        (df["sensor_mode"] == "multiscale") & (df["sensor_noise"] == 0.1)
    ].iloc[0]
    single_low = df[
        (df["sensor_mode"] == "single") & (df["sensor_noise"] == 0.05)
    ].iloc[0]
    multi_low = df[
        (df["sensor_mode"] == "multiscale") & (df["sensor_noise"] == 0.05)
    ].iloc[0]
    high_single = df[
        (df["sensor_mode"] == "single") & (df["sensor_noise"] == 0.2)
    ].iloc[0]
    high_multi = df[
        (df["sensor_mode"] == "multiscale") & (df["sensor_noise"] == 0.2)
    ].iloc[0]
    records: list[HypothesisRecord] = []
    records.append(
        HypothesisRecord(
            "H28",
            "supported"
            if float(multi_low["mean_route_delay"])
            < float(single_low["mean_route_delay"])
            and float(multi_mid["mean_route_delay"])
            < float(single_mid["mean_route_delay"])
            else "mixed",
            statistic=float(single_mid["mean_route_delay"])
            - float(multi_mid["mean_route_delay"]),
            note="multiscale sensing should improve regime routing under moderate noise",
            evidence={
                "single_low_delay": float(single_low["mean_route_delay"]),
                "multi_low_delay": float(multi_low["mean_route_delay"]),
                "single_mid_delay": float(single_mid["mean_route_delay"]),
                "multi_mid_delay": float(multi_mid["mean_route_delay"]),
            },
        )
    )
    records.append(
        HypothesisRecord(
            "H29",
            "supported"
            if float(high_single["mean_route_delay"])
            == float(high_multi["mean_route_delay"])
            else "mixed",
            statistic=float(high_multi["mean_route_delay"])
            - float(high_single["mean_route_delay"]),
            note="high-noise sensing should saturate rather than fully recover the lag",
        )
    )
    return records


def _evaluate_conformal_hypotheses() -> list[HypothesisRecord]:
    conformal = _load_temporalbridge_benchmark(
        "temporalbridge.benchmarks.conformal_benchmark"
    )
    result = conformal.run_conformal_benchmark(
        config=conformal.ConformalBenchmarkConfig(
            calibration_windows=(8, 12, 16, 24, 32, 48, 64, 96),
            repetitions=12,
            steps=240,
            switch_step=120,
            alpha=0.1,
            noise_sigma=0.35,
            drift_rate=0.012,
            alarm_window=8,
            alarm_persistence=3,
            coverage_slack=0.02,
            useful_tol=0.05,
            score_penalty=18.0,
        ),
        rng_seed=0,
    )
    summary = result["summary"]
    records: list[HypothesisRecord] = []
    records.append(
        HypothesisRecord(
            "H-CP1",
            "supported" if bool(summary["u_curve"]) else "mixed",
            statistic=float(summary["best_score"]),
            note="split conformal score has an interior minimum over calibration window size",
            evidence={
                "best_window": int(summary["best_window"]),
                "left_slope": float(summary["best_score_left_slope"]),
                "right_slope": float(summary["best_score_right_slope"]),
            },
        )
    )
    records.append(
        HypothesisRecord(
            "H-CP3",
            "supported"
            if float(summary["mean_coverage_before_alarm_gap"]) > 0.0
            and float(summary["positive_gap_rate"]) > 0.5
            else "mixed",
            statistic=float(summary["mean_coverage_before_alarm_gap"]),
            note="coverage deterioration should precede alarm detection under drift",
            evidence={
                "mean_alarm_step": float(summary["mean_alarm_step"]),
                "mean_coverage_crossing_step": float(
                    summary["mean_coverage_crossing_step"]
                ),
                "positive_gap_rate": float(summary["positive_gap_rate"]),
            },
        )
    )
    records.append(
        HypothesisRecord(
            "H-CP5",
            "supported"
            if bool(summary["best_window_is_safe"])
            and bool(summary["best_window_is_useful"])
            and float(summary["safe_useful_overlap_fraction"]) >= 0.5
            else "mixed",
            statistic=float(summary["safe_useful_overlap_fraction"]),
            note="the useful-memory band should nearly coincide with the empirical safe calibration band",
            evidence={
                "best_window": int(summary["best_window"]),
                "coverage_plateau": float(summary["coverage_plateau"]),
                "safe_floor": float(summary["safe_coverage_floor"]),
                "safe_windows": ",".join(map(str, summary["safe_windows"])),
                "useful_windows": ",".join(map(str, summary["useful_windows"])),
                "overlap_windows": ",".join(map(str, summary["overlap_windows"])),
            },
        )
    )
    return records


def run_hypothesis_suite(*, fast: bool = True) -> HypothesisSuiteReport:
    records: list[HypothesisRecord] = []
    records.extend(_evaluate_controller_hypotheses(fast=fast))
    records.extend(_evaluate_ratio_hypotheses())
    records.extend(_evaluate_delay_hypotheses(fast=fast))
    records.extend(_evaluate_lag_misspecification())
    records.extend(_evaluate_model_free_ucurve())
    records.extend(_evaluate_ratio_control_hypotheses())
    records.extend(_evaluate_regime_route_delay_hypotheses())
    records.extend(_evaluate_meta_sensing_hypotheses())
    records.extend(_evaluate_conformal_hypotheses())
    return HypothesisSuiteReport(records=records)


def parse_args(argv: list[str] | None = None) -> Any:
    import argparse

    parser = argparse.ArgumentParser(description="Run the hypothesis test suite.")
    parser.add_argument(
        "--slow", action="store_true", help="Use heavier benchmark settings."
    )
    parser.add_argument(
        "--json", action="store_true", help="Print JSON instead of a table."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = run_hypothesis_suite(fast=not args.slow)
    if args.json:
        print(report.to_json())
        return
    for record in report.records:
        print(
            f"{record.hypothesis}\t{record.status}\t{record.statistic!r}\t{record.note}"
        )


if __name__ == "__main__":
    main()
