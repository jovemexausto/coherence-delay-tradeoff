from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

TGT_CONDITIONS = ("full", "fm1", "fm2", "fm3")
TGT_LABELS = {
    "full": "Full TGT",
    "fm1": "FM-1 static Pot",
    "fm2": "FM-2 noisy Act",
    "fm3": "FM-3 absent Conv",
}
TGT_COLORS = {
    "full": "tab:blue",
    "fm1": "tab:purple",
    "fm2": "tab:orange",
    "fm3": "tab:red",
}


@dataclass(slots=True)
class TGTConfig:
    steps: int = 500
    seed: int = 42
    drift: float = 0.02
    process_scale: float = 0.1
    observation_scale: float = 1.0
    initial_mean: float = 0.0
    initial_variance: float = 1.0
    condition: str = "full"
    influence: float = 0.0
    fm1_frozen_mean: float = 0.0
    fm2_action_scale: float = 0.5


@dataclass(slots=True)
class TGTResult:
    config: TGTConfig
    condition: str
    latent_mean: np.ndarray
    free_mean: np.ndarray
    observations: np.ndarray
    estimate_mean: np.ndarray
    estimate_variance: np.ndarray
    action: np.ndarray
    v_p: np.ndarray
    v_a: np.ndarray
    v_phi: np.ndarray
    sigma_p: np.ndarray
    sigma_a: np.ndarray
    sigma_phi: np.ndarray
    tci: np.ndarray
    v_total: np.ndarray


@dataclass(slots=True)
class UCurveResult:
    drift_values: np.ndarray
    window_sizes: np.ndarray
    mean_error_grid: np.ndarray
    std_error_grid: np.ndarray
    empirical_n_star: np.ndarray
    empirical_e_min: np.ndarray
    slope: float
    scaled_constant: float


@dataclass(slots=True)
class SampleComplexityResult:
    window_sizes: np.ndarray
    mean_absolute_error: np.ndarray
    std_absolute_error: np.ndarray
    slope: float


def _steady_state_variance(process_scale: float, observation_scale: float) -> float:
    q = process_scale * process_scale
    r = observation_scale * observation_scale
    return 0.5 * (q + np.sqrt(q * q + 4.0 * q * r))


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same")


def _simulate_environment(
    config: TGTConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(config.seed)
    latent_mean = np.zeros(config.steps)
    free_mean = np.zeros(config.steps)
    observations = np.zeros(config.steps)
    actions = np.zeros(config.steps)

    latent_mean[0] = config.initial_mean
    free_mean[0] = config.initial_mean
    observations[0] = latent_mean[0] + rng.normal(scale=config.observation_scale)

    for step in range(1, config.steps):
        free_mean[step] = (
            latent_mean[step - 1]
            + config.drift
            + rng.normal(scale=config.process_scale)
        )
        latent_mean[step] = free_mean[step] + config.influence * (
            actions[step - 1] - free_mean[step]
        )
        observations[step] = latent_mean[step] + rng.normal(
            scale=config.observation_scale
        )

    return latent_mean, free_mean, observations


def run_tgt_experiment(config: TGTConfig | None = None) -> TGTResult:
    config = config or TGTConfig()
    if config.condition not in TGT_CONDITIONS:
        raise ValueError(f"Unknown TGT condition: {config.condition}")

    rng = np.random.default_rng(config.seed + 17)
    latent_mean, free_mean, observations = _simulate_environment(config)
    steady_variance = _steady_state_variance(
        config.process_scale,
        config.observation_scale,
    )

    estimate_mean = np.zeros(config.steps)
    estimate_variance = np.zeros(config.steps)
    action = np.zeros(config.steps)
    v_p = np.zeros(config.steps)
    v_a = np.zeros(config.steps)
    v_phi = np.zeros(config.steps)
    sigma_p = np.zeros(config.steps)
    sigma_a = np.zeros(config.steps)
    sigma_phi = np.zeros(config.steps)
    tci = np.zeros(config.steps)
    v_total = np.zeros(config.steps)

    current_mean = config.initial_mean
    current_variance = config.initial_variance

    for step in range(config.steps):
        if config.condition == "fm1":
            current_mean = config.fm1_frozen_mean
            current_variance = steady_variance
        elif config.condition == "fm3":
            # FM-3 removes convergence entirely: the filter keeps propagating its
            # prior but never assimilates observations. The variance therefore
            # grows without bound under the random-walk model, which is exactly
            # what makes V_Phi explode and sigma_Phi collapse in this ablation.
            predicted_mean = current_mean + config.drift
            predicted_variance = current_variance + config.process_scale**2
            current_mean = predicted_mean
            current_variance = predicted_variance
        else:
            predicted_mean = current_mean + config.drift
            predicted_variance = current_variance + config.process_scale**2
            kalman_gain = predicted_variance / (
                predicted_variance + config.observation_scale**2
            )
            innovation = observations[step] - predicted_mean
            current_mean = predicted_mean + kalman_gain * innovation
            current_variance = (1.0 - kalman_gain) * predicted_variance

        estimate_mean[step] = current_mean
        estimate_variance[step] = current_variance

        if config.condition == "fm2":
            action_noise = rng.normal(scale=config.fm2_action_scale)
            action[step] = current_mean + action_noise
            v_a[step] = (action_noise / max(config.fm2_action_scale, 1e-9)) ** 2
        else:
            action[step] = current_mean
            v_a[step] = 0.0

        v_p[step] = 0.5 * (estimate_mean[step] - latent_mean[step]) ** 2
        v_phi[step] = abs(estimate_variance[step] - steady_variance)
        sigma_p[step] = 1.0 / (1.0 + v_p[step])
        sigma_a[step] = np.exp(-v_a[step])
        sigma_phi[step] = 1.0 / (1.0 + v_phi[step])
        tci[step] = min(sigma_p[step], sigma_a[step], sigma_phi[step])
        v_total[step] = v_p[step] + v_a[step] + v_phi[step]

    return TGTResult(
        config=config,
        condition=config.condition,
        latent_mean=latent_mean,
        free_mean=free_mean,
        observations=observations,
        estimate_mean=estimate_mean,
        estimate_variance=estimate_variance,
        action=action,
        v_p=v_p,
        v_a=v_a,
        v_phi=v_phi,
        sigma_p=sigma_p,
        sigma_a=sigma_a,
        sigma_phi=sigma_phi,
        tci=tci,
        v_total=v_total,
    )


def run_tgt_ablation(config: TGTConfig | None = None) -> dict[str, TGTResult]:
    base_config = config or TGTConfig()
    results: dict[str, TGTResult] = {}
    for condition in TGT_CONDITIONS:
        condition_config = TGTConfig(
            steps=base_config.steps,
            seed=base_config.seed,
            drift=base_config.drift,
            process_scale=base_config.process_scale,
            observation_scale=base_config.observation_scale,
            initial_mean=base_config.initial_mean,
            initial_variance=base_config.initial_variance,
            condition=condition,
            influence=base_config.influence,
            fm1_frozen_mean=base_config.fm1_frozen_mean,
            fm2_action_scale=base_config.fm2_action_scale,
        )
        results[condition] = run_tgt_experiment(condition_config)
    return results


def summarize_tgt_result(result: TGTResult, tail_window: int = 100) -> dict[str, float]:
    tail = slice(-min(tail_window, result.config.steps), None)
    return {
        "mean_v_p": float(np.mean(result.v_p[tail])),
        "mean_v_a": float(np.mean(result.v_a[tail])),
        "mean_v_phi": float(np.mean(result.v_phi[tail])),
        "mean_sigma_p": float(np.mean(result.sigma_p[tail])),
        "mean_sigma_a": float(np.mean(result.sigma_a[tail])),
        "mean_sigma_phi": float(np.mean(result.sigma_phi[tail])),
        "mean_tci": float(np.mean(result.tci[tail])),
        "mean_v_total": float(np.mean(result.v_total[tail])),
    }


def build_ablation_rows(results: dict[str, TGTResult]) -> list[dict[str, str | float]]:
    rows: list[dict[str, str | float]] = []
    for condition in TGT_CONDITIONS:
        summary = summarize_tgt_result(results[condition])
        rows.append(
            {
                "condition": condition,
                "mean_v_p": round(summary["mean_v_p"], 3),
                "mean_v_a": round(summary["mean_v_a"], 3),
                "mean_v_phi": round(summary["mean_v_phi"], 3),
                "mean_sigma_p": round(summary["mean_sigma_p"], 3),
                "mean_sigma_a": round(summary["mean_sigma_a"], 3),
                "mean_sigma_phi": round(summary["mean_sigma_phi"], 3),
                "mean_tci": round(summary["mean_tci"], 3),
                "mean_v_total": round(summary["mean_v_total"], 3),
            }
        )
    return rows


def export_rows_csv(rows: list[dict[str, str | float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_ablation_figure(results: dict[str, TGTResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    time = np.arange(next(iter(results.values())).config.steps)
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    for condition in TGT_CONDITIONS:
        result = results[condition]
        axes[0].plot(
            time,
            _rolling_mean(result.v_total, 20),
            color=TGT_COLORS[condition],
            linewidth=1.3,
            label=TGT_LABELS[condition],
        )
        axes[1].plot(
            time,
            _rolling_mean(result.tci, 20),
            color=TGT_COLORS[condition],
            linewidth=1.3,
            label=TGT_LABELS[condition],
        )

    axes[0].set_yscale("log")
    axes[0].set_ylabel(r"$V_{total}$")
    axes[0].set_title("Component ablation under passive drift")
    axes[0].legend(loc="upper left", ncol=2)
    axes[1].set_ylabel("Score")
    axes[1].set_xlabel("Time step")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].legend(loc="lower left", ncol=2)

    for axis in axes:
        axis.grid(alpha=0.2, linewidth=0.5)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _window_errors_for_drift(
    drift: float,
    window_sizes: np.ndarray,
    seeds: list[int],
    steps: int = 6000,
    process_scale: float = 0.1,
    observation_scale: float = 1.0,
) -> np.ndarray:
    errors = np.zeros((len(seeds), len(window_sizes)))
    tail_start = steps // 2
    for seed_index, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        mu = np.zeros(steps)
        obs = np.zeros(steps)
        for step in range(1, steps):
            mu[step] = mu[step - 1] + drift + rng.normal(scale=process_scale)
        obs = mu + rng.normal(scale=observation_scale, size=steps)
        for window_index, window in enumerate(window_sizes):
            estimate = np.full(steps, np.nan)
            for step in range(window - 1, steps):
                estimate[step] = np.mean(obs[step - window + 1 : step + 1])
            valid = np.arange(tail_start, steps)
            valid = valid[~np.isnan(estimate[tail_start:])]
            tail_estimate = estimate[valid]
            tail_mu = mu[valid]
            errors[seed_index, window_index] = float(
                np.mean(np.abs(tail_mu - tail_estimate))
            )
    return errors


def run_ucurve_experiment() -> UCurveResult:
    # This experiment intentionally uses a simple sliding-window mean instead of
    # the Kalman filter. The goal is to isolate the lag-vs-variance tradeoff of
    # finite windows directly, without the extra adaptation dynamics of Conv.
    drift_values = np.asarray([0.001, 0.005, 0.01, 0.05], dtype=float)
    window_sizes = np.asarray([5, 10, 20, 50, 75, 100, 150, 200, 300, 500], dtype=int)
    seeds = list(range(20))
    error_grid = np.zeros((drift_values.size, window_sizes.size))
    std_grid = np.zeros((drift_values.size, window_sizes.size))
    empirical_n_star = np.zeros(drift_values.size, dtype=int)
    empirical_e_min = np.zeros(drift_values.size)

    for index, drift in enumerate(drift_values):
        errors = _window_errors_for_drift(
            drift,
            window_sizes,
            seeds,
            process_scale=0.0,
            observation_scale=1.0,
        )
        mean_errors = np.mean(errors, axis=0)
        std_errors = np.std(errors, axis=0)
        error_grid[index] = mean_errors
        std_grid[index] = std_errors
        best = int(np.argmin(mean_errors))
        empirical_n_star[index] = int(window_sizes[best])
        empirical_e_min[index] = float(mean_errors[best])

    slope = float(np.polyfit(np.log(drift_values), np.log(empirical_e_min), 1)[0])
    scaled_constant = float(np.mean(empirical_e_min / np.cbrt(drift_values)))
    return UCurveResult(
        drift_values=drift_values,
        window_sizes=window_sizes,
        mean_error_grid=error_grid,
        std_error_grid=std_grid,
        empirical_n_star=empirical_n_star,
        empirical_e_min=empirical_e_min,
        slope=slope,
        scaled_constant=scaled_constant,
    )


def save_ucurve_figure(result: UCurveResult, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for index, drift in enumerate(result.drift_values):
        axes[0].plot(
            result.window_sizes,
            result.mean_error_grid[index],
            linewidth=1.4,
            marker="o",
            markersize=3,
            label=rf"$\zeta={drift:.3f}$",
        )
        best = int(np.argmin(result.mean_error_grid[index]))
        axes[0].scatter(
            result.window_sizes[best],
            result.mean_error_grid[index, best],
            color="black",
            s=28,
            zorder=4,
        )

    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Window size n")
    axes[0].set_ylabel("Mean absolute tracking error")
    axes[0].set_title("Lag-drift U-curve")
    axes[0].legend(loc="upper right")

    axes[1].plot(result.drift_values, result.empirical_e_min, marker="o", linewidth=1.5)
    fit_constant = np.exp(
        np.polyfit(np.log(result.drift_values), np.log(result.empirical_e_min), 1)[1]
    )
    fit_curve = fit_constant * result.drift_values**result.slope
    axes[1].plot(
        result.drift_values,
        fit_curve,
        linestyle="--",
        color="0.35",
        label=rf"fit slope={result.slope:.3f}",
    )
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"Drift rate $\zeta$")
    axes[1].set_ylabel(r"$\mathcal{E}_{min}$")
    axes[1].set_title("Minimum error vs drift")
    axes[1].legend(loc="upper left")

    for axis in axes:
        axis.grid(alpha=0.2, linewidth=0.5)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def build_ucurve_rows(result: UCurveResult) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for drift, n_star, e_min in zip(
        result.drift_values,
        result.empirical_n_star,
        result.empirical_e_min,
        strict=True,
    ):
        rows.append(
            {
                "drift": round(float(drift), 4),
                "n_star": int(n_star),
                "e_min": round(float(e_min), 4),
                "scaled_constant": round(float(e_min / np.cbrt(drift)), 4),
            }
        )
    return rows


def run_sample_complexity_experiment() -> SampleComplexityResult:
    # The paper still refers to this artifact as fig_sinkhorn for continuity,
    # but the implemented experiment measures the sample complexity of the
    # sigma_P estimator in a slow-drift regime rather than transport itself.
    window_sizes = np.asarray([5, 10, 25, 50, 100], dtype=int)
    seeds = list(range(50))
    steps = 5000
    drift = 0.001
    process_scale = 0.035
    observation_scale = 1.0
    maes = np.zeros((len(seeds), len(window_sizes)))

    for seed_index, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        mu = np.zeros(steps)
        obs = np.zeros(steps)
        for step in range(1, steps):
            mu[step] = mu[step - 1] + drift + rng.normal(scale=process_scale)
        obs = mu + rng.normal(scale=observation_scale, size=steps)
        tail_start = int(steps * 0.6)
        for window_index, window in enumerate(window_sizes):
            sigma_est = np.full(steps, np.nan)
            sigma_true = np.full(steps, np.nan)
            for step in range(window - 1, steps):
                window_mean = np.mean(obs[step - window + 1 : step + 1])
                estimated_v_p = 0.5 * (window_mean - mu[step]) ** 2
                sigma_est[step] = 1.0 / (1.0 + estimated_v_p)
                sigma_true[step] = 1.0
            valid = np.arange(tail_start, steps)
            maes[seed_index, window_index] = float(
                np.nanmean(np.abs(sigma_true[valid] - sigma_est[valid]))
            )

    mean_absolute_error = np.mean(maes, axis=0)
    std_absolute_error = np.std(maes, axis=0)
    slope = float(np.polyfit(np.log(window_sizes), np.log(mean_absolute_error), 1)[0])
    return SampleComplexityResult(
        window_sizes=window_sizes,
        mean_absolute_error=mean_absolute_error,
        std_absolute_error=std_absolute_error,
        slope=slope,
    )


def save_sigma_p_complexity_figure(
    result: SampleComplexityResult,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(6.6, 4.6))
    axis.plot(
        result.window_sizes,
        result.mean_absolute_error,
        marker="o",
        linewidth=1.5,
        color="tab:blue",
        label="measured MAE",
    )
    fit_constant = np.exp(
        np.polyfit(np.log(result.window_sizes), np.log(result.mean_absolute_error), 1)[
            1
        ]
    )
    fit_curve = fit_constant * result.window_sizes**result.slope
    axis.plot(
        result.window_sizes,
        fit_curve,
        linestyle="--",
        color="0.35",
        label=rf"fit slope={result.slope:.3f}",
    )
    axis.fill_between(
        result.window_sizes,
        result.mean_absolute_error - result.std_absolute_error,
        result.mean_absolute_error + result.std_absolute_error,
        color="tab:blue",
        alpha=0.15,
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Window size n")
    axis.set_ylabel(r"MAE $|\sigma_P - \hat\sigma_P|$")
    axis.set_title("Variance-dominated sample complexity")
    axis.grid(alpha=0.2, linewidth=0.5)
    axis.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_sinkhorn_figure(result: SampleComplexityResult, output_path: Path) -> None:
    # Backward-compatible wrapper for the paper artifact name figures/fig_sinkhorn.pdf.
    save_sigma_p_complexity_figure(result, output_path)


def build_sample_complexity_rows(
    result: SampleComplexityResult,
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for window_size, mean_error, std_error in zip(
        result.window_sizes,
        result.mean_absolute_error,
        result.std_absolute_error,
        strict=True,
    ):
        rows.append(
            {
                "window_size": int(window_size),
                "mean_absolute_error": round(float(mean_error), 4),
                "std_absolute_error": round(float(std_error), 4),
            }
        )
    return rows
