from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kstest, norm

from .common import (
    build_manifest_row,
    export_rows_csv,
    spawn_rng,
    stable_run_id,
)
from .useful_memory_region import (
    continuous_optimal_horizon,
    horizon_envelope,
    useful_memory_interval,
)


@dataclass(frozen=True, slots=True)
class InferableHorizonConfig:
    H_values: tuple[float, ...] = (0.3, 0.6, 0.9)
    lag_counts: tuple[int, ...] = (10, 20, 50)
    sample_sizes: tuple[int, ...] = (500, 1000, 5000)
    zeta_values: tuple[float, ...] = (0.5, 1.0, 2.0)
    delta_values: tuple[float, ...] = (0.05, 0.1, 0.2)
    a: float = 0.5
    C_K: float = 100.0
    C_S: float = 1.0
    sigma0: float = 1.0
    repetitions: int = 4000
    seed: int = 12345
    suite_name: str = "default"


@dataclass(frozen=True, slots=True)
class InferableHorizonDraws:
    alpha_hats: np.ndarray
    H_hats: np.ndarray
    log_horizon_hats: np.ndarray


@dataclass(frozen=True, slots=True)
class InferableHorizonSuite:
    scenario_rows: list[dict[str, float | int | str]]
    joint_clt_rows: list[dict[str, float | int]]
    plugin_clt_rows: list[dict[str, float | int]]
    coverage_rows: list[dict[str, float | int]]
    regret_rows: list[dict[str, float | int]]


def export_inferable_horizon_suite(
    suite: InferableHorizonSuite,
    output_root: Path,
) -> None:
    csv_root = output_root / "csv" / "inferable_horizon"
    export_rows_csv(suite.scenario_rows, csv_root / "scenarios.csv")
    export_rows_csv(suite.joint_clt_rows, csv_root / "joint_clt.csv")
    export_rows_csv(suite.plugin_clt_rows, csv_root / "plugin_clt.csv")
    export_rows_csv(suite.coverage_rows, csv_root / "coverage.csv")
    export_rows_csv(suite.regret_rows, csv_root / "regret.csv")


def transition_coverage_config(repetitions: int, seed: int) -> InferableHorizonConfig:
    return InferableHorizonConfig(
        H_values=(0.3, 0.6, 0.9),
        lag_counts=(3, 5, 8, 10),
        sample_sizes=(10, 20, 50, 100, 200),
        zeta_values=(0.5, 1.0, 2.0, 4.0),
        delta_values=(0.005, 0.01, 0.02, 0.05),
        repetitions=repetitions,
        seed=seed,
        suite_name="transition",
    )


def select_representative_coverage_rows(
    coverage_rows: list[dict[str, float | int]],
    targets: tuple[float, ...] = (0.5, 1.0, 2.0, 3.0),
) -> list[dict[str, float | int]]:
    selected: list[dict[str, float | int]] = []
    used_indices: set[int] = set()
    for target in targets:
        best_index = min(
            (index for index in range(len(coverage_rows)) if index not in used_indices),
            key=lambda index: abs(
                float(coverage_rows[index]["identifiability_score"]) - target
            ),
        )
        used_indices.add(best_index)
        selected.append(coverage_rows[best_index])
    return selected


def write_representative_coverage_table(
    rows: list[dict[str, float | int]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\small",
        r"\caption{Representative regimes across the inferable-horizon transition.}",
        r"\label{tab:inferable_horizon}",
        r"\begin{tabular}{rrrrrrr}",
        r"\toprule",
        r"$L$ & $m$ & $H$ & $\delta$ & $\mathfrak S$ & Empirical & Gaussian \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{int(row['lag_count'])} & {int(row['sample_size'])} & {float(row['H']):.1f} & {float(row['delta']):.3f} & {float(row['identifiability_score']):.3f} & {float(row['empirical_hit_rate']):.3f} & {float(row['gaussian_hit_rate']):.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_coverage_collapse_figure(
    coverage_rows: list[dict[str, float | int]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 3.9))
    max_score = max(float(row["identifiability_score"]) for row in coverage_rows)
    scores = np.linspace(0.0, max_score, 400)
    grouped: dict[float, list[dict[str, float | int]]] = {}
    for row in coverage_rows:
        grouped.setdefault(float(row["delta"]), []).append(row)
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    for index, (delta, rows) in enumerate(sorted(grouped.items())):
        rows = sorted(rows, key=lambda row: float(row["identifiability_score"]))
        x = np.asarray([float(row["identifiability_score"]) for row in rows])
        y = np.asarray([float(row["empirical_hit_rate"]) for row in rows])
        y_se = np.asarray(
            [float(row.get("empirical_hit_rate_se", 0.0)) for row in rows]
        )
        n_bins = min(6, max(3, len(rows) // 2))
        bins = np.array_split(np.arange(len(rows)), n_bins)
        bin_x: list[float] = []
        bin_y: list[float] = []
        bin_lo: list[float] = []
        bin_hi: list[float] = []
        for bin_indices in bins:
            if bin_indices.size == 0:
                continue
            local_x = x[bin_indices]
            local_y = y[bin_indices]
            local_se = y_se[bin_indices]
            mean_x = float(np.mean(local_x))
            mean_y = float(np.mean(local_y))
            mean_se = float(np.sqrt(np.mean(local_se**2)))
            bin_x.append(mean_x)
            bin_y.append(mean_y)
            bin_lo.append(max(0.0, mean_y - 1.96 * mean_se))
            bin_hi.append(min(1.0, mean_y + 1.96 * mean_se))
        color = colors[index % len(colors)] if colors else None
        ax.plot(
            bin_x,
            bin_y,
            marker="o",
            linewidth=1.6,
            markersize=4.5,
            color=color,
            label=rf"$\delta={delta:.3f}$",
        )
        ax.fill_between(
            bin_x,
            bin_lo,
            bin_hi,
            color=color,
            alpha=0.12,
            linewidth=0,
        )
    ax.plot(
        scores,
        2.0 * norm.cdf(scores) - 1.0,
        color="black",
        linewidth=1.3,
        linestyle="--",
        label="Gaussian approximation",
    )
    ax.set_xlabel(r"Operational identifiability score $\mathfrak{S}_{\delta,m,L}$")
    ax.set_ylabel(r"$\Pr(\widehat r^* \in \mathcal{U}_\delta)$")
    ax.set_ylim(0.0, 1.02)
    ax.set_xlim(0.0, max_score)
    ax.grid(alpha=0.2, linewidth=0.5)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_regret_scaling_figure(
    regret_rows: list[dict[str, float | int]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    xs = [float(row["theoretical_relative_regret"]) for row in regret_rows]
    ys = [float(row["empirical_relative_regret"]) for row in regret_rows]
    ax.scatter(xs, ys, s=18, alpha=0.85)
    upper = max(xs + ys)
    ax.plot([0.0, upper], [0.0, upper], color="black", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Theoretical relative regret")
    ax.set_ylabel("Empirical relative regret")
    ax.grid(alpha=0.2, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def lag_grid(lag_count: int) -> np.ndarray:
    if lag_count < 2:
        raise ValueError("lag_count must be at least 2")
    return np.arange(1, lag_count + 1, dtype=float)


def lag_energy(sample_size: int, lags: np.ndarray, H: float) -> float:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    lag_array = np.asarray(lags, dtype=float)
    return float(sample_size) * float(np.sum(lag_array ** (2.0 * H)))


def horizon_log_map(
    alpha: np.ndarray | float,
    H: np.ndarray | float,
    C_K: float,
    a: float,
    C_S: float,
) -> np.ndarray | float:
    if C_K <= 0.0 or C_S <= 0.0:
        raise ValueError("C_K and C_S must be positive")
    alpha_array = np.asarray(alpha, dtype=float)
    H_array = np.asarray(H, dtype=float)
    if a <= 0.0 or np.any(H_array <= 0.0):
        raise ValueError("a and H must be positive")
    return (np.log(a * C_K) - np.log(H_array) - np.log(C_S) - alpha_array) / (
        a + H_array
    )


def horizon_gradient_log_map(
    alpha: float,
    H: float,
    C_K: float,
    a: float,
    C_S: float,
) -> np.ndarray:
    if a <= 0.0 or H <= 0.0:
        raise ValueError("a and H must be positive")
    log_n_star = horizon_log_map(alpha, H, C_K, a, C_S)
    d_alpha = -1.0 / (a + H)
    d_H = -(1.0 / (H * (a + H))) - log_n_star / (a + H)
    return np.asarray([d_alpha, d_H], dtype=float)


def horizon_log_lipschitz_constant(
    alpha_bounds: tuple[float, float],
    H_bounds: tuple[float, float],
    C_K: float,
    a: float,
    C_S: float,
) -> float:
    alpha_min, alpha_max = alpha_bounds
    H_min, H_max = H_bounds
    if alpha_min > alpha_max:
        raise ValueError("alpha_bounds must be ordered")
    if H_min <= 0.0 or H_min > H_max:
        raise ValueError("H_bounds must be ordered and positive")
    corner_values = [
        abs(float(horizon_log_map(alpha, H, C_K, a, C_S)))
        for alpha in (alpha_min, alpha_max)
        for H in (H_min, H_max)
    ]
    max_abs_log_horizon = max(corner_values)
    d_alpha_bound = 1.0 / (a + H_min)
    d_H_bound = (1.0 / (H_min * (a + H_min))) + max_abs_log_horizon / (a + H_min)
    return float(d_alpha_bound + d_H_bound)


def useful_region_parameter_radius(
    alpha_bounds: tuple[float, float],
    H_bounds: tuple[float, float],
    C_K: float,
    a: float,
    C_S: float,
    H: float,
    delta: float,
) -> float:
    lipschitz = horizon_log_lipschitz_constant(alpha_bounds, H_bounds, C_K, a, C_S)
    return float(useful_region_log_radius(a, H, delta) / lipschitz)


def exact_relative_regret_from_log_ratio(
    log_ratio: np.ndarray | float, a: float, H: float
) -> np.ndarray | float:
    log_ratio_array = np.asarray(log_ratio, dtype=float)
    return (H * np.exp(-a * log_ratio_array) + a * np.exp(H * log_ratio_array)) / (
        a + H
    ) - 1.0


def deterministic_relative_regret_bound(
    log_error_radius: float, a: float, H: float
) -> float:
    if log_error_radius < 0.0:
        raise ValueError("log_error_radius must be nonnegative")
    upper = float(exact_relative_regret_from_log_ratio(log_error_radius, a, H))
    lower = float(exact_relative_regret_from_log_ratio(-log_error_radius, a, H))
    return max(upper, lower)


def useful_region_log_radius(a: float, H: float, delta: float) -> float:
    lower_x, upper_x = useful_memory_interval(a, H, delta)
    return float(min(-np.log(lower_x), np.log(upper_x)))


def weighted_log_design_summaries(
    lags: np.ndarray, H: float
) -> tuple[float, float, float, float]:
    lag_array = np.asarray(lags, dtype=float)
    x = np.log(lag_array)
    w = lag_array ** (2.0 * H)
    R0 = float(np.sum(w))
    R1 = float(np.sum(w * x))
    R2 = float(np.sum(w * x * x))
    det = R0 * R2 - R1 * R1
    if det <= 0.0:
        raise ValueError("weighted design must have positive determinant")
    return (R0, R1, R2, det)


def theoretical_joint_covariance_scaled(
    lags: np.ndarray,
    H: float,
    zeta: float,
    sigma0: float,
) -> np.ndarray:
    if zeta <= 0.0 or sigma0 <= 0.0:
        raise ValueError("zeta and sigma0 must be positive")
    _, R1, R2, det = weighted_log_design_summaries(lags, H)
    R0, _, _, _ = weighted_log_design_summaries(lags, H)
    prefactor = (sigma0 / zeta) ** 2 / det
    return prefactor * np.asarray([[R2, -R1], [-R1, R0]], dtype=float)


def plug_in_horizon_tau_squared(
    lags: np.ndarray,
    H: float,
    zeta: float,
    sigma0: float,
    C_K: float,
    a: float,
    C_S: float,
) -> float:
    alpha = float(np.log(zeta))
    gradient = horizon_gradient_log_map(alpha, H, C_K, a, C_S)
    covariance = theoretical_joint_covariance_scaled(lags, H, zeta, sigma0)
    return float(gradient @ covariance @ gradient)


def plug_in_horizon_information_tau_squared(
    lags: np.ndarray,
    H: float,
    zeta: float,
    sigma0: float,
    C_K: float,
    a: float,
    C_S: float,
) -> float:
    lag_array = np.asarray(lags, dtype=float)
    return float(np.sum(lag_array ** (2.0 * H))) * plug_in_horizon_tau_squared(
        lag_array,
        H,
        zeta,
        sigma0,
        C_K,
        a,
        C_S,
    )


def relative_useful_memory_regret(
    n_hat: float,
    C_K: float,
    a: float,
    C_S: float,
    zeta: float,
    H: float,
) -> float:
    n_star = continuous_optimal_horizon(C_K, a, C_S, zeta, H)
    phi_star = horizon_envelope(n_star, C_K, a, C_S, zeta, H)
    phi_hat = horizon_envelope(n_hat, C_K, a, C_S, zeta, H)
    return float((phi_hat - phi_star) / phi_star)


def quadratic_relative_regret(
    log_ratio: np.ndarray | float, a: float, H: float
) -> np.ndarray | float:
    return 0.5 * a * H * np.asarray(log_ratio) ** 2


def operational_identifiability_score(
    sample_size: int,
    lags: np.ndarray,
    H: float,
    zeta: float,
    sigma0: float,
    C_K: float,
    a: float,
    C_S: float,
    delta: float,
) -> float:
    tau = np.sqrt(
        plug_in_horizon_information_tau_squared(
            lags,
            H,
            zeta,
            sigma0,
            C_K,
            a,
            C_S,
        )
    )
    if tau <= 0.0:
        raise ValueError("tau must be positive")
    return float(
        np.sqrt(lag_energy(sample_size, lags, H))
        * useful_region_log_radius(a, H, delta)
        / tau
    )


def _pilot_ols(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    x_centered = x - float(np.mean(x))
    y_centered = y - float(np.mean(y))
    H_hat = float(np.dot(x_centered, y_centered) / np.dot(x_centered, x_centered))
    alpha_hat = float(np.mean(y) - H_hat * np.mean(x))
    return (alpha_hat, H_hat)


def fwls_estimate(
    observed_discrepancies: np.ndarray,
    sample_size: int,
    sigma0: float,
) -> tuple[float, float]:
    observed = np.asarray(observed_discrepancies, dtype=float)
    if np.any(observed <= 0.0):
        raise ValueError("observed discrepancies must be positive")
    if sample_size <= 0 or sigma0 <= 0.0:
        raise ValueError("sample_size and sigma0 must be positive")
    lags = lag_grid(observed.size)
    x = np.log(lags)
    y = np.log(observed)
    alpha_pilot, H_pilot = _pilot_ols(y, x)
    fitted_scale = np.exp(alpha_pilot) * lags**H_pilot
    weights = float(sample_size) * fitted_scale**2 / sigma0**2
    X = np.column_stack([np.ones_like(x), x])
    sqrt_weights = np.sqrt(weights)
    Xw = X * sqrt_weights[:, None]
    yw = y * sqrt_weights
    beta = np.linalg.solve(Xw.T @ Xw, Xw.T @ yw)
    return (float(beta[0]), max(float(beta[1]), 1.0e-6))


def simulate_fwls_joint_estimates(
    H: float,
    zeta: float,
    sigma0: float,
    sample_size: int,
    lag_count: int,
    repetitions: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    draws = simulate_fwls_draws(
        H,
        zeta,
        sigma0,
        sample_size,
        lag_count,
        repetitions,
        rng,
        C_K=1.0,
        a=0.5,
        C_S=1.0,
    )
    return (draws.alpha_hats, draws.H_hats)


def simulate_fwls_draws(
    H: float,
    zeta: float,
    sigma0: float,
    sample_size: int,
    lag_count: int,
    repetitions: int,
    rng: np.random.Generator,
    *,
    C_K: float,
    a: float,
    C_S: float,
) -> InferableHorizonDraws:
    lags = lag_grid(lag_count)
    signal = zeta * lags**H
    alpha_hats = np.empty(repetitions, dtype=float)
    H_hats = np.empty(repetitions, dtype=float)
    noise_scale = sigma0 / np.sqrt(float(sample_size))
    for index in range(repetitions):
        observed = np.maximum(
            signal + rng.normal(scale=noise_scale, size=lag_count),
            1.0e-12,
        )
        alpha_hats[index], H_hats[index] = fwls_estimate(
            observed,
            sample_size,
            sigma0,
        )
    log_horizon_hats = horizon_log_map(alpha_hats, H_hats, C_K, a, C_S)
    return InferableHorizonDraws(
        alpha_hats=alpha_hats,
        H_hats=H_hats,
        log_horizon_hats=np.asarray(log_horizon_hats, dtype=float),
    )


def _scenario_metadata(
    config: InferableHorizonConfig,
    *,
    H: float,
    lag_count: int,
    sample_size: int,
    zeta: float,
) -> dict[str, float | int | str]:
    payload = {
        "suite_name": config.suite_name,
        "H": H,
        "lag_count": lag_count,
        "sample_size": sample_size,
        "zeta": zeta,
        "sigma0": config.sigma0,
        "a": config.a,
        "C_K": config.C_K,
        "C_S": config.C_S,
    }
    scenario_id = stable_run_id(payload)
    scenario_seed = int(scenario_id, 16)
    return {
        "scenario_id": scenario_id,
        "suite_name": config.suite_name,
        "master_seed": config.seed,
        "scenario_seed": scenario_seed,
        "repetitions": config.repetitions,
        "lag_count": lag_count,
        "sample_size": sample_size,
        "H": H,
        "zeta": zeta,
        "sigma0": config.sigma0,
        "a": config.a,
        "C_K": config.C_K,
        "C_S": config.C_S,
    }


def _std_error_from_sample(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    if array.size <= 1:
        return 0.0
    return float(np.std(array, ddof=1) / np.sqrt(array.size))


def _rate_standard_error(success_rate: float, repetitions: int) -> float:
    if repetitions <= 0:
        return 0.0
    clipped = min(max(float(success_rate), 0.0), 1.0)
    return float(np.sqrt(clipped * (1.0 - clipped) / float(repetitions)))


def _joint_clt_row(
    scenario: dict[str, float | int | str],
    draws: InferableHorizonDraws,
) -> dict[str, float | int]:
    H = float(scenario["H"])
    zeta = float(scenario["zeta"])
    sigma0 = float(scenario["sigma0"])
    sample_size = int(scenario["sample_size"])
    lag_count = int(scenario["lag_count"])
    repetitions = int(scenario["repetitions"])
    lags = lag_grid(lag_count)
    alpha_true = float(np.log(zeta))
    covariance_theory = theoretical_joint_covariance_scaled(lags, H, zeta, sigma0)
    alpha_scaled = np.sqrt(sample_size) * (draws.alpha_hats - alpha_true)
    H_scaled = np.sqrt(sample_size) * (draws.H_hats - H)
    covariance_empirical = np.cov(
        alpha_scaled,
        H_scaled,
    )
    cov_products = (alpha_scaled - np.mean(alpha_scaled)) * (
        H_scaled - np.mean(H_scaled)
    )
    return {
        **scenario,
        "var_alpha_theory_scaled": float(covariance_theory[0, 0]),
        "var_alpha_empirical_scaled": float(covariance_empirical[0, 0]),
        "var_alpha_empirical_scaled_se": float(
            abs(covariance_empirical[0, 0]) * np.sqrt(2.0 / max(repetitions - 1, 1))
        ),
        "var_H_theory_scaled": float(covariance_theory[1, 1]),
        "var_H_empirical_scaled": float(covariance_empirical[1, 1]),
        "var_H_empirical_scaled_se": float(
            abs(covariance_empirical[1, 1]) * np.sqrt(2.0 / max(repetitions - 1, 1))
        ),
        "cov_theory_scaled": float(covariance_theory[0, 1]),
        "cov_empirical_scaled": float(covariance_empirical[0, 1]),
        "cov_empirical_scaled_se": _std_error_from_sample(cov_products),
    }


def _plugin_clt_row(
    scenario: dict[str, float | int | str],
    draws: InferableHorizonDraws,
) -> dict[str, float | int]:
    H = float(scenario["H"])
    zeta = float(scenario["zeta"])
    sigma0 = float(scenario["sigma0"])
    sample_size = int(scenario["sample_size"])
    lag_count = int(scenario["lag_count"])
    repetitions = int(scenario["repetitions"])
    a = float(scenario["a"])
    C_K = float(scenario["C_K"])
    C_S = float(scenario["C_S"])
    lags = lag_grid(lag_count)
    log_horizon_true = horizon_log_map(float(np.log(zeta)), H, C_K, a, C_S)
    tau2 = plug_in_horizon_tau_squared(lags, H, zeta, sigma0, C_K, a, C_S)
    std_theory = float(np.sqrt(tau2 / sample_size))
    standardized = (draws.log_horizon_hats - log_horizon_true) / std_theory
    ci_half_width = norm.ppf(0.975) * std_theory
    ci_coverage = float(
        np.mean(np.abs(draws.log_horizon_hats - log_horizon_true) <= ci_half_width)
    )
    std_empirical = float(np.std(draws.log_horizon_hats, ddof=1))
    return {
        **scenario,
        "tau2": tau2,
        "std_log_horizon_theory": std_theory,
        "std_log_horizon_empirical": std_empirical,
        "std_log_horizon_empirical_se": float(
            std_empirical / np.sqrt(2.0 * max(repetitions - 1, 1))
        ),
        "ks_pvalue": float(kstest(standardized, "norm").pvalue),
        "ci95_coverage": ci_coverage,
        "ci95_coverage_se": _rate_standard_error(ci_coverage, repetitions),
    }


def _coverage_rows(
    scenario: dict[str, float | int | str],
    delta_values: tuple[float, ...],
    draws: InferableHorizonDraws,
) -> list[dict[str, float | int]]:
    H = float(scenario["H"])
    zeta = float(scenario["zeta"])
    sigma0 = float(scenario["sigma0"])
    sample_size = int(scenario["sample_size"])
    lag_count = int(scenario["lag_count"])
    repetitions = int(scenario["repetitions"])
    a = float(scenario["a"])
    C_K = float(scenario["C_K"])
    C_S = float(scenario["C_S"])
    lags = lag_grid(lag_count)
    n_star = continuous_optimal_horizon(C_K, a, C_S, zeta, H)
    n_hats = np.exp(draws.log_horizon_hats)
    tau = float(
        np.sqrt(
            plug_in_horizon_information_tau_squared(
                lags,
                H,
                zeta,
                sigma0,
                C_K,
                a,
                C_S,
            )
        )
    )
    rows: list[dict[str, float | int]] = []
    for delta in delta_values:
        lower_x, upper_x = useful_memory_interval(a, H, delta)
        lower = n_star * lower_x
        upper = n_star * upper_x
        radius = useful_region_log_radius(a, H, delta)
        score = np.sqrt(lag_energy(sample_size, lags, H)) * radius / tau
        empirical_hit_rate = float(np.mean((n_hats >= lower) & (n_hats <= upper)))
        rows.append(
            {
                **scenario,
                "delta": delta,
                "radius": radius,
                "identifiability_score": float(score),
                "lower": float(lower),
                "upper": float(upper),
                "empirical_hit_rate": empirical_hit_rate,
                "empirical_hit_rate_se": _rate_standard_error(
                    empirical_hit_rate, repetitions
                ),
                "gaussian_hit_rate": float(2.0 * norm.cdf(score) - 1.0),
            }
        )
    return rows


def _regret_row(
    scenario: dict[str, float | int | str],
    draws: InferableHorizonDraws,
) -> dict[str, float | int]:
    H = float(scenario["H"])
    zeta = float(scenario["zeta"])
    sigma0 = float(scenario["sigma0"])
    sample_size = int(scenario["sample_size"])
    lag_count = int(scenario["lag_count"])
    a = float(scenario["a"])
    C_K = float(scenario["C_K"])
    C_S = float(scenario["C_S"])
    lags = lag_grid(lag_count)
    n_hats = np.exp(draws.log_horizon_hats)
    regrets = np.asarray(
        [
            relative_useful_memory_regret(n_hat, C_K, a, C_S, zeta, H)
            for n_hat in n_hats
        ],
        dtype=float,
    )
    tau2 = plug_in_horizon_tau_squared(lags, H, zeta, sigma0, C_K, a, C_S)
    theory = 0.5 * a * H * tau2 / float(sample_size)
    empirical_regret = float(np.mean(regrets))
    return {
        **scenario,
        "empirical_relative_regret": empirical_regret,
        "empirical_relative_regret_se": _std_error_from_sample(regrets),
        "theoretical_relative_regret": float(theory),
        "empirical_to_theoretical_ratio": float(empirical_regret / theory),
    }


def run_inferable_horizon_suite(
    config: InferableHorizonConfig = InferableHorizonConfig(),
) -> InferableHorizonSuite:
    scenario_rows: list[dict[str, float | int | str]] = []
    joint_rows: list[dict[str, float | int]] = []
    plugin_rows: list[dict[str, float | int]] = []
    coverage_rows: list[dict[str, float | int]] = []
    regret_rows: list[dict[str, float | int]] = []

    for H in config.H_values:
        for lag_count in config.lag_counts:
            for sample_size in config.sample_sizes:
                for zeta in config.zeta_values:
                    scenario = _scenario_metadata(
                        config,
                        H=H,
                        lag_count=lag_count,
                        sample_size=sample_size,
                        zeta=zeta,
                    )
                    scenario_rows.append(scenario)
                    rng = spawn_rng(
                        config.seed, "inferable-horizon", str(scenario["scenario_id"])
                    )
                    draws = simulate_fwls_draws(
                        H,
                        zeta,
                        config.sigma0,
                        sample_size,
                        lag_count,
                        config.repetitions,
                        rng,
                        C_K=config.C_K,
                        a=config.a,
                        C_S=config.C_S,
                    )
                    joint_rows.append(_joint_clt_row(scenario, draws))
                    plugin_rows.append(_plugin_clt_row(scenario, draws))
                    coverage_rows.extend(
                        _coverage_rows(
                            scenario,
                            config.delta_values,
                            draws,
                        )
                    )
                    regret_rows.append(_regret_row(scenario, draws))

    return InferableHorizonSuite(
        scenario_rows=scenario_rows,
        joint_clt_rows=joint_rows,
        plugin_clt_rows=plugin_rows,
        coverage_rows=coverage_rows,
        regret_rows=regret_rows,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate inferable horizon bridge artifacts."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts"),
        help="Artifact root directory.",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=1000,
        help="Monte Carlo repetitions per configuration.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Master seed for deterministic scenario-level Monte Carlo.",
    )
    parser.add_argument(
        "--transition-seed",
        type=int,
        default=2026,
        help="Master seed for the transition-coverage suite.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    suite = run_inferable_horizon_suite(
        InferableHorizonConfig(repetitions=args.repetitions, seed=args.seed)
    )
    export_inferable_horizon_suite(suite, args.output_root)
    transition_suite = run_inferable_horizon_suite(
        transition_coverage_config(
            repetitions=max(args.repetitions, 1000), seed=args.transition_seed
        )
    )
    export_rows_csv(
        transition_suite.coverage_rows,
        args.output_root / "csv" / "inferable_horizon" / "coverage_transition.csv",
    )
    export_rows_csv(
        transition_suite.scenario_rows,
        args.output_root / "csv" / "inferable_horizon" / "transition_scenarios.csv",
    )
    figure_root = args.output_root / "figures" / "inferable_horizon"
    save_coverage_collapse_figure(
        transition_suite.coverage_rows,
        figure_root / "fig_operational_identifiability.pdf",
    )
    save_regret_scaling_figure(
        suite.regret_rows,
        figure_root / "fig_operational_regret.pdf",
    )
    write_representative_coverage_table(
        select_representative_coverage_rows(transition_suite.coverage_rows),
        args.output_root / "tables" / "inferable_horizon" / "tab_inferable_horizon.tex",
    )
    manifest = build_manifest_row(
        "inferable_horizon",
        asdict(InferableHorizonConfig(repetitions=args.repetitions, seed=args.seed)),
        run_id=stable_run_id(
            {
                "suite_name": "default",
                "repetitions": args.repetitions,
                "seed": args.seed,
            }
        ),
        seed=args.seed,
        notes="Scenario-seeded inferable-horizon bridge with Monte Carlo uncertainty fields.",
    )
    export_rows_csv(
        [manifest], args.output_root / "csv" / "inferable_horizon" / "manifest.csv"
    )


if __name__ == "__main__":
    main()
