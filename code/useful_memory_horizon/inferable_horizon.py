from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kstest, norm

from .common import export_rows_csv
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


@dataclass(frozen=True, slots=True)
class InferableHorizonSuite:
    joint_clt_rows: list[dict[str, float | int]]
    plugin_clt_rows: list[dict[str, float | int]]
    coverage_rows: list[dict[str, float | int]]
    regret_rows: list[dict[str, float | int]]


def export_inferable_horizon_suite(
    suite: InferableHorizonSuite,
    output_root: Path,
) -> None:
    csv_root = output_root / "csv" / "inferable_horizon"
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
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    grouped: dict[float, list[dict[str, float | int]]] = {}
    for row in coverage_rows:
        grouped.setdefault(float(row["delta"]), []).append(row)
    for delta, rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda row: float(row["identifiability_score"]))
        ax.scatter(
            [float(row["identifiability_score"]) for row in rows],
            [float(row["empirical_hit_rate"]) for row in rows],
            s=18,
            alpha=0.8,
            label=rf"$\delta={delta:.2f}$",
        )
    scores = np.linspace(
        0.0, max(float(row["identifiability_score"]) for row in coverage_rows), 400
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
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.2, linewidth=0.5)
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
    return (alpha_hats, H_hats)


def _joint_clt_row(
    H: float,
    zeta: float,
    sigma0: float,
    sample_size: int,
    lag_count: int,
    repetitions: int,
    rng: np.random.Generator,
) -> dict[str, float | int]:
    lags = lag_grid(lag_count)
    alpha_hats, H_hats = simulate_fwls_joint_estimates(
        H,
        zeta,
        sigma0,
        sample_size,
        lag_count,
        repetitions,
        rng,
    )
    alpha_true = float(np.log(zeta))
    covariance_theory = theoretical_joint_covariance_scaled(lags, H, zeta, sigma0)
    covariance_empirical = np.cov(
        np.sqrt(sample_size) * (alpha_hats - alpha_true),
        np.sqrt(sample_size) * (H_hats - H),
    )
    return {
        "lag_count": lag_count,
        "sample_size": sample_size,
        "H": H,
        "zeta": zeta,
        "var_alpha_theory_scaled": float(covariance_theory[0, 0]),
        "var_alpha_empirical_scaled": float(covariance_empirical[0, 0]),
        "var_H_theory_scaled": float(covariance_theory[1, 1]),
        "var_H_empirical_scaled": float(covariance_empirical[1, 1]),
        "cov_theory_scaled": float(covariance_theory[0, 1]),
        "cov_empirical_scaled": float(covariance_empirical[0, 1]),
    }


def _plugin_clt_row(
    H: float,
    zeta: float,
    sigma0: float,
    sample_size: int,
    lag_count: int,
    repetitions: int,
    a: float,
    C_K: float,
    C_S: float,
    rng: np.random.Generator,
) -> tuple[dict[str, float | int], np.ndarray, np.ndarray, np.ndarray]:
    lags = lag_grid(lag_count)
    alpha_hats, H_hats = simulate_fwls_joint_estimates(
        H,
        zeta,
        sigma0,
        sample_size,
        lag_count,
        repetitions,
        rng,
    )
    log_horizon_hats = horizon_log_map(alpha_hats, H_hats, C_K, a, C_S)
    log_horizon_true = horizon_log_map(float(np.log(zeta)), H, C_K, a, C_S)
    tau2 = plug_in_horizon_tau_squared(lags, H, zeta, sigma0, C_K, a, C_S)
    std_theory = float(np.sqrt(tau2 / sample_size))
    standardized = (log_horizon_hats - log_horizon_true) / std_theory
    ci_half_width = norm.ppf(0.975) * std_theory
    ci_coverage = float(
        np.mean(np.abs(log_horizon_hats - log_horizon_true) <= ci_half_width)
    )
    row = {
        "lag_count": lag_count,
        "sample_size": sample_size,
        "H": H,
        "zeta": zeta,
        "tau2": tau2,
        "std_log_horizon_theory": std_theory,
        "std_log_horizon_empirical": float(np.std(log_horizon_hats, ddof=1)),
        "ks_pvalue": float(kstest(standardized, "norm").pvalue),
        "ci95_coverage": ci_coverage,
    }
    return (row, alpha_hats, H_hats, log_horizon_hats)


def _coverage_rows(
    H: float,
    zeta: float,
    sigma0: float,
    sample_size: int,
    lag_count: int,
    a: float,
    C_K: float,
    C_S: float,
    delta_values: tuple[float, ...],
    log_horizon_hats: np.ndarray,
) -> list[dict[str, float | int]]:
    lags = lag_grid(lag_count)
    n_star = continuous_optimal_horizon(C_K, a, C_S, zeta, H)
    n_hats = np.exp(log_horizon_hats)
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
        rows.append(
            {
                "lag_count": lag_count,
                "sample_size": sample_size,
                "H": H,
                "zeta": zeta,
                "delta": delta,
                "radius": radius,
                "identifiability_score": float(score),
                "lower": float(lower),
                "upper": float(upper),
                "empirical_hit_rate": float(
                    np.mean((n_hats >= lower) & (n_hats <= upper))
                ),
                "gaussian_hit_rate": float(2.0 * norm.cdf(score) - 1.0),
            }
        )
    return rows


def _regret_row(
    H: float,
    zeta: float,
    sigma0: float,
    sample_size: int,
    lag_count: int,
    a: float,
    C_K: float,
    C_S: float,
    log_horizon_hats: np.ndarray,
) -> dict[str, float | int]:
    lags = lag_grid(lag_count)
    n_hats = np.exp(log_horizon_hats)
    regrets = np.asarray(
        [
            relative_useful_memory_regret(n_hat, C_K, a, C_S, zeta, H)
            for n_hat in n_hats
        ],
        dtype=float,
    )
    tau2 = plug_in_horizon_tau_squared(lags, H, zeta, sigma0, C_K, a, C_S)
    theory = 0.5 * a * H * tau2 / float(sample_size)
    return {
        "lag_count": lag_count,
        "sample_size": sample_size,
        "H": H,
        "zeta": zeta,
        "empirical_relative_regret": float(np.mean(regrets)),
        "theoretical_relative_regret": float(theory),
        "empirical_to_theoretical_ratio": float(np.mean(regrets) / theory),
    }


def run_inferable_horizon_suite(
    config: InferableHorizonConfig = InferableHorizonConfig(),
) -> InferableHorizonSuite:
    rng = np.random.default_rng(config.seed)
    joint_rows: list[dict[str, float | int]] = []
    plugin_rows: list[dict[str, float | int]] = []
    coverage_rows: list[dict[str, float | int]] = []
    regret_rows: list[dict[str, float | int]] = []

    for H in config.H_values:
        for lag_count in config.lag_counts:
            for sample_size in config.sample_sizes:
                for zeta in config.zeta_values:
                    joint_rows.append(
                        _joint_clt_row(
                            H,
                            zeta,
                            config.sigma0,
                            sample_size,
                            lag_count,
                            config.repetitions,
                            rng,
                        )
                    )
                    plugin_row, _, _, log_horizon_hats = _plugin_clt_row(
                        H,
                        zeta,
                        config.sigma0,
                        sample_size,
                        lag_count,
                        config.repetitions,
                        config.a,
                        config.C_K,
                        config.C_S,
                        rng,
                    )
                    plugin_rows.append(plugin_row)
                    coverage_rows.extend(
                        _coverage_rows(
                            H,
                            zeta,
                            config.sigma0,
                            sample_size,
                            lag_count,
                            config.a,
                            config.C_K,
                            config.C_S,
                            config.delta_values,
                            log_horizon_hats,
                        )
                    )
                    regret_rows.append(
                        _regret_row(
                            H,
                            zeta,
                            config.sigma0,
                            sample_size,
                            lag_count,
                            config.a,
                            config.C_K,
                            config.C_S,
                            log_horizon_hats,
                        )
                    )

    return InferableHorizonSuite(
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    suite = run_inferable_horizon_suite(
        InferableHorizonConfig(repetitions=args.repetitions)
    )
    export_inferable_horizon_suite(suite, args.output_root)
    transition_suite = run_inferable_horizon_suite(
        transition_coverage_config(
            repetitions=max(args.repetitions, 1000), seed=args.repetitions + 2026
        )
    )
    export_rows_csv(
        transition_suite.coverage_rows,
        args.output_root / "csv" / "inferable_horizon" / "coverage_transition.csv",
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


if __name__ == "__main__":
    main()
