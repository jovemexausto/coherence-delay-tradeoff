from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from river import drift as river_drift
from scipy.special import logsumexp
from scipy.stats import t as student_t

ABLATION_CONDITIONS = ("full", "fm1", "fm2", "fm3")
CONDITION_LABELS = {
    "full": "Full TPT",
    "fm1": "FM-1 static Pot",
    "fm2": "FM-2 noisy Act",
    "fm3": "FM-3 absent Conv",
}
CONDITION_OFFSETS = {"full": 1, "fm1": 2, "fm2": 3, "fm3": 4}


@dataclass(slots=True)
class TPTConfig:
    steps: int = 300
    particles: int = 750
    seed: int = 7
    drift: float = 0.04
    influence: float = 0.0
    process_scale: float = 0.35
    observation_scale: float = 0.9
    observation_df: int = 4
    resample_threshold: float = 0.45
    prior_mean: float = 0.0
    prior_scale: float = 1.0
    condition: str = "full"
    actuation_noise_scale: float = 0.9
    fm1_sigma_phi_level: float = 1.0
    fm3_sigma_phi_floor: float = 0.08
    effort_penalty_lambda: float = 3.0
    effort_floor: float = 1e-3


@dataclass(slots=True)
class TPTResult:
    config: TPTConfig
    condition: str
    latent_state: np.ndarray
    uncontrolled_state: np.ndarray
    observations: np.ndarray
    posterior_mean: np.ndarray
    posterior_std: np.ndarray
    actions: np.ndarray
    action_gap: np.ndarray
    effort_signal: np.ndarray
    tracking_error: np.ndarray
    sigma_p: np.ndarray
    sigma_p_eff: np.ndarray
    sigma_a: np.ndarray
    sigma_phi: np.ndarray
    tci: np.ndarray
    tcie: np.ndarray
    ess: np.ndarray
    entropy: np.ndarray
    log_evidence: np.ndarray
    resampled: np.ndarray


@dataclass(slots=True)
class TPTActiveBenchmarkConfig:
    steps: int = 600
    particles: int = 750
    seed: int = 7
    masking_start: int = 200
    collapse_start: int = 400
    healthy_drift: float = 0.01
    masking_drift: float = 0.04
    collapse_drift: float = 0.40
    influence: float = 0.3
    process_scale: float = 0.35
    observation_scale: float = 0.9
    observation_df: int = 4
    prior_mean: float = 0.0
    prior_scale: float = 1.0
    resample_threshold: float = 0.45
    effort_penalty_lambda: float = 3.0
    tci_threshold: float = 0.80
    tcie_threshold: float = 0.80
    adwin_delta: float = 0.20
    page_hinkley_delta: float = 0.005
    page_hinkley_threshold: float = 20.0
    page_hinkley_alpha: float = 0.9999
    kswin_window_size: int = 30
    kswin_stat_size: int = 10
    kswin_alpha: float = 0.001


@dataclass(slots=True)
class TPTActiveBenchmarkResult:
    config: TPTActiveBenchmarkConfig
    latent_state: np.ndarray
    uncontrolled_state: np.ndarray
    observations: np.ndarray
    posterior_mean: np.ndarray
    action_gap: np.ndarray
    effort_signal: np.ndarray
    tci: np.ndarray
    tcie: np.ndarray
    adwin_signal: np.ndarray
    tci_warnings: list[int]
    tcie_warnings: list[int]
    adwin_warnings: list[int]
    baseline_warnings: dict[str, list[int]]
    masking_detection: dict[str, float | int | None]
    collapse_detection: dict[str, float | int | None]


ACTIVE_BASELINE_DETECTORS = ("ADWIN", "PageHinkley", "KSWIN", "NoDrift")


def _state_dynamics(state: np.ndarray, drift: float) -> np.ndarray:
    return 0.72 * state + 0.18 * np.sin(1.4 * state) + drift


def _observation_model(state: np.ndarray) -> np.ndarray:
    return 0.35 * state**2 - 0.4 * np.cos(state)


def _apply_control(
    state: np.ndarray, action: np.ndarray | float, influence: float
) -> np.ndarray:
    return state + influence * (action - state)


def _condition_label(condition: str) -> str:
    return CONDITION_LABELS.get(condition, condition)


def _first_warning_after(warnings: list[int], start: int) -> int | None:
    for warning in warnings:
        if warning >= start:
            return warning
    return None


def _threshold_warnings(signal: np.ndarray, threshold: float) -> list[int]:
    warnings: list[int] = []
    below = False
    for index, value in enumerate(signal):
        if np.isnan(value):
            continue
        if value < threshold and not below:
            warnings.append(index)
            below = True
        elif value >= threshold:
            below = False
    return warnings


def _run_drift_detector(
    signal: np.ndarray, detector_name: str, config: TPTActiveBenchmarkConfig
) -> list[int]:
    if detector_name == "ADWIN":
        detector = river_drift.ADWIN(delta=config.adwin_delta)
    elif detector_name == "PageHinkley":
        detector = river_drift.PageHinkley(
            delta=config.page_hinkley_delta,
            threshold=config.page_hinkley_threshold,
            alpha=config.page_hinkley_alpha,
            mode="both",
        )
    elif detector_name == "KSWIN":
        detector = river_drift.KSWIN(
            window_size=config.kswin_window_size,
            stat_size=config.kswin_stat_size,
            alpha=config.kswin_alpha,
        )
    elif detector_name == "NoDrift":
        detector = river_drift.NoDrift()
    else:
        raise ValueError(f"Unknown detector: {detector_name}")

    warnings: list[int] = []
    for index, value in enumerate(signal):
        detector.update(float(value))
        if detector.drift_detected:
            warnings.append(index)
    return warnings


def _summarize_detector_onset(
    warnings: list[int],
    start: int,
) -> dict[str, float | int | None]:
    first = _first_warning_after(warnings, start)
    return {
        "first_warning": first,
        "delay": None if first is None else first - start,
    }


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same")


def _systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    count = weights.size
    positions = (rng.random() + np.arange(count)) / count
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, positions, side="left")


def _observe_state(state: float, config: TPTConfig, rng: np.random.Generator) -> float:
    return float(
        _observation_model(state)
        + student_t.rvs(
            df=config.observation_df,
            loc=0.0,
            scale=config.observation_scale,
            random_state=rng,
        )
    )


def _propagate_latent_state(
    previous_state: float,
    previous_action: float,
    config: TPTConfig,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    uncontrolled_state = float(
        _state_dynamics(previous_state, config.drift)
        + rng.normal(scale=config.process_scale)
    )
    latent_state = float(
        _apply_control(uncontrolled_state, previous_action, config.influence)
    )
    effort = float(config.influence * abs(previous_action - uncontrolled_state))
    return uncontrolled_state, latent_state, effort


def _propagate_particles(
    particles: np.ndarray,
    previous_action: float,
    config: TPTConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    uncontrolled = _state_dynamics(particles, config.drift) + rng.normal(
        scale=config.process_scale,
        size=particles.size,
    )
    return _apply_control(uncontrolled, previous_action, config.influence)


def run_tpt_experiment(config: TPTConfig | None = None) -> TPTResult:
    config = config or TPTConfig()
    if config.condition not in ABLATION_CONDITIONS:
        raise ValueError(f"Unknown TPT condition: {config.condition}")

    rng = np.random.default_rng(config.seed + CONDITION_OFFSETS[config.condition])

    latent_state = np.zeros(config.steps)
    uncontrolled_state = np.zeros(config.steps)
    observations = np.zeros(config.steps)
    posterior_mean = np.zeros(config.steps)
    posterior_std = np.zeros(config.steps)
    actions = np.zeros(config.steps)
    action_gap = np.zeros(config.steps)
    effort_signal = np.zeros(config.steps)
    tracking_error = np.zeros(config.steps)
    sigma_p = np.zeros(config.steps)
    sigma_p_eff = np.zeros(config.steps)
    sigma_a = np.zeros(config.steps)
    sigma_phi = np.zeros(config.steps)
    tci = np.zeros(config.steps)
    tcie = np.zeros(config.steps)
    ess = np.zeros(config.steps)
    entropy = np.zeros(config.steps)
    log_evidence = np.zeros(config.steps)
    resampled = np.zeros(config.steps, dtype=bool)

    latent_state[0] = config.prior_mean + rng.normal(scale=config.prior_scale)
    uncontrolled_state[0] = latent_state[0]
    observations[0] = _observe_state(latent_state[0], config, rng)

    particles = rng.normal(
        loc=config.prior_mean,
        scale=config.prior_scale,
        size=config.particles,
    )
    uniform_weights = np.full(config.particles, 1.0 / config.particles)
    frozen_particles = particles.copy()
    frozen_mean = float(np.mean(frozen_particles))
    frozen_std = float(np.std(frozen_particles))
    # Use the prior state scale as the reference effort scale. This is more
    # interpretable than mixing multiple magnitudes via max(...), and keeps the
    # Effort penalty tied to the model's nominal state scale.
    effort_reference = max(config.effort_floor, config.prior_scale)

    for step in range(config.steps):
        if step > 0:
            (
                uncontrolled_state[step],
                latent_state[step],
                _,
            ) = _propagate_latent_state(
                latent_state[step - 1], actions[step - 1], config, rng
            )
            observations[step] = _observe_state(latent_state[step], config, rng)

        if config.condition == "fm1":
            candidate_particles = frozen_particles
        elif step == 0:
            candidate_particles = particles
        else:
            candidate_particles = _propagate_particles(
                particles, actions[step - 1], config, rng
            )

        predicted_observation = _observation_model(candidate_particles)
        log_likelihood = student_t.logpdf(
            observations[step],
            df=config.observation_df,
            loc=predicted_observation,
            scale=config.observation_scale,
        )
        log_evidence[step] = logsumexp(log_likelihood) - np.log(config.particles)

        if config.condition == "fm3":
            weights = uniform_weights
            posterior_mean[step] = float(np.mean(candidate_particles))
            posterior_std[step] = float(np.std(candidate_particles))
            ess_ratio = 0.0
            entropy_ratio = 0.0
            sigma_phi_value = config.fm3_sigma_phi_floor
        elif config.condition == "fm1":
            weights = np.exp(log_likelihood - logsumexp(log_likelihood))
            posterior_mean[step] = frozen_mean
            posterior_std[step] = frozen_std
            ess_ratio = 1.0
            entropy_ratio = 1.0
            sigma_phi_value = config.fm1_sigma_phi_level
        else:
            weights = np.exp(log_likelihood - logsumexp(log_likelihood))
            posterior_mean[step] = float(np.sum(weights * candidate_particles))
            centered = candidate_particles - posterior_mean[step]
            posterior_std[step] = float(np.sqrt(np.sum(weights * centered * centered)))
            ess_ratio = float(1.0 / (config.particles * np.sum(weights * weights)))
            entropy_ratio = float(
                -np.sum(weights * np.log(weights + 1e-12)) / np.log(config.particles)
            )
            sigma_phi_value = 0.5 * (ess_ratio + entropy_ratio)

        target_action = posterior_mean[step]
        if config.condition == "fm2":
            actions[step] = target_action + rng.normal(
                scale=config.actuation_noise_scale
            )
        else:
            actions[step] = target_action

        tracking_error[step] = posterior_mean[step] - latent_state[step]
        sigma_p[step] = 1.0 / (1.0 + 0.5 * tracking_error[step] * tracking_error[step])

        actuation_gap = (actions[step] - target_action) / max(
            config.actuation_noise_scale,
            1e-9,
        )
        sigma_a[step] = float(np.exp(-0.5 * actuation_gap * actuation_gap))
        action_gap[step] = abs(actions[step] - uncontrolled_state[step])
        effort_signal[step] = config.influence * action_gap[step]
        sigma_p_eff[step] = sigma_p[step] * np.exp(
            -config.effort_penalty_lambda * effort_signal[step] / effort_reference
        )
        sigma_phi[step] = sigma_phi_value
        tci[step] = min(sigma_p[step], sigma_a[step], sigma_phi[step])
        tcie[step] = min(sigma_p_eff[step], sigma_a[step], sigma_phi[step])
        ess[step] = ess_ratio
        entropy[step] = entropy_ratio

        if config.condition in {"full", "fm2"}:
            particles = candidate_particles
            if ess_ratio < config.resample_threshold:
                indexes = _systematic_resample(weights, rng)
                particles = particles[indexes]
                resampled[step] = True
        elif config.condition == "fm3":
            particles = candidate_particles
        else:
            particles = frozen_particles

    return TPTResult(
        config=config,
        condition=config.condition,
        latent_state=latent_state,
        uncontrolled_state=uncontrolled_state,
        observations=observations,
        posterior_mean=posterior_mean,
        posterior_std=posterior_std,
        actions=actions,
        action_gap=action_gap,
        effort_signal=effort_signal,
        tracking_error=tracking_error,
        sigma_p=sigma_p,
        sigma_p_eff=sigma_p_eff,
        sigma_a=sigma_a,
        sigma_phi=sigma_phi,
        tci=tci,
        tcie=tcie,
        ess=ess,
        entropy=entropy,
        log_evidence=log_evidence,
        resampled=resampled,
    )


def run_tpt_active_benchmark(
    config: TPTActiveBenchmarkConfig | None = None,
) -> TPTActiveBenchmarkResult:
    config = config or TPTActiveBenchmarkConfig()
    rng = np.random.default_rng(config.seed)

    latent_state = np.zeros(config.steps)
    uncontrolled_state = np.zeros(config.steps)
    observations = np.zeros(config.steps)
    posterior_mean = np.zeros(config.steps)
    posterior_std = np.zeros(config.steps)
    actions = np.zeros(config.steps)
    action_gap = np.zeros(config.steps)
    effort_signal = np.zeros(config.steps)
    tci = np.zeros(config.steps)
    tcie = np.zeros(config.steps)
    sigma_p = np.zeros(config.steps)
    sigma_p_eff = np.zeros(config.steps)
    sigma_phi = np.zeros(config.steps)
    sigma_a = np.ones(config.steps)
    baseline_signal = np.zeros(config.steps)

    latent_state[0] = config.prior_mean + rng.normal(scale=config.prior_scale)
    uncontrolled_state[0] = latent_state[0]
    observations[0] = _observe_state(
        latent_state[0],
        TPTConfig(
            observation_df=config.observation_df,
            observation_scale=config.observation_scale,
        ),
        rng,
    )

    particles = rng.normal(
        loc=config.prior_mean,
        scale=config.prior_scale,
        size=config.particles,
    )
    frozen_particles: np.ndarray | None = None
    frozen_mean = config.prior_mean
    frozen_std = config.prior_scale
    effort_reference = max(config.prior_scale, 1e-3)

    current_mean = config.prior_mean
    current_std = config.prior_scale

    for step in range(config.steps):
        if step > 0:
            if step < config.masking_start:
                drift_value = config.healthy_drift
                influence_value = 0.0
            elif step < config.collapse_start:
                drift_value = config.masking_drift
                influence_value = config.influence
            else:
                drift_value = config.collapse_drift
                influence_value = config.influence

            uncontrolled_state[step] = float(
                _state_dynamics(latent_state[step - 1], drift_value)
                + rng.normal(scale=config.process_scale)
            )
            latent_state[step] = float(
                _apply_control(
                    uncontrolled_state[step], actions[step - 1], influence_value
                )
            )
            observations[step] = _observe_state(
                latent_state[step],
                TPTConfig(
                    observation_df=config.observation_df,
                    observation_scale=config.observation_scale,
                ),
                rng,
            )

        if step == config.masking_start:
            frozen_particles = particles.copy()
            frozen_mean = float(current_mean)
            frozen_std = float(current_std)

        if step < config.masking_start:
            candidate_particles = _state_dynamics(
                particles, config.healthy_drift
            ) + rng.normal(
                scale=config.process_scale,
                size=config.particles,
            )
            predicted_observation = _observation_model(candidate_particles)
            log_likelihood = student_t.logpdf(
                observations[step],
                df=config.observation_df,
                loc=predicted_observation,
                scale=config.observation_scale,
            )
            weights = np.exp(log_likelihood - logsumexp(log_likelihood))
            current_mean = float(np.sum(weights * candidate_particles))
            centered = candidate_particles - current_mean
            current_std = float(np.sqrt(np.sum(weights * centered * centered)))
            particles = candidate_particles
            ess_ratio = float(1.0 / (config.particles * np.sum(weights * weights)))
            if ess_ratio < config.resample_threshold:
                indexes = _systematic_resample(weights, rng)
                particles = particles[indexes]
            sigma_phi_value = 0.5 * (
                ess_ratio
                + float(
                    -np.sum(weights * np.log(weights + 1e-12))
                    / np.log(config.particles)
                )
            )
        else:
            assert frozen_particles is not None
            current_mean = frozen_mean
            current_std = frozen_std
            sigma_phi_value = 1.0

        posterior_mean[step] = current_mean
        posterior_std[step] = current_std
        actions[step] = current_mean

        tracking_error = posterior_mean[step] - latent_state[step]
        sigma_p[step] = 1.0 / (1.0 + 0.5 * tracking_error * tracking_error)
        action_gap[step] = abs(actions[step] - uncontrolled_state[step])
        current_influence = 0.0 if step < config.masking_start else config.influence
        effort_signal[step] = current_influence * action_gap[step]
        sigma_p_eff[step] = sigma_p[step] * np.exp(
            -config.effort_penalty_lambda * effort_signal[step] / effort_reference
        )
        sigma_phi[step] = sigma_phi_value
        tci[step] = min(sigma_p[step], sigma_a[step], sigma_phi[step])
        tcie[step] = min(sigma_p_eff[step], sigma_a[step], sigma_phi[step])
        baseline_signal[step] = 1.0 - tcie[step]

    tci_warnings = _threshold_warnings(tci, config.tci_threshold)
    tcie_warnings = _threshold_warnings(tcie, config.tcie_threshold)
    baseline_warnings = {
        detector_name: _run_drift_detector(baseline_signal, detector_name, config)
        for detector_name in ACTIVE_BASELINE_DETECTORS
    }
    adwin_warnings = baseline_warnings["ADWIN"]

    return TPTActiveBenchmarkResult(
        config=config,
        latent_state=latent_state,
        uncontrolled_state=uncontrolled_state,
        observations=observations,
        posterior_mean=posterior_mean,
        action_gap=action_gap,
        effort_signal=effort_signal,
        tci=tci,
        tcie=tcie,
        adwin_signal=baseline_signal,
        tci_warnings=tci_warnings,
        tcie_warnings=tcie_warnings,
        adwin_warnings=adwin_warnings,
        baseline_warnings=baseline_warnings,
        masking_detection={
            "TCI": _summarize_detector_onset(tci_warnings, config.masking_start),
            "TCIE": _summarize_detector_onset(tcie_warnings, config.masking_start),
            **{
                detector_name: _summarize_detector_onset(warnings, config.masking_start)
                for detector_name, warnings in baseline_warnings.items()
            },
        },
        collapse_detection={
            "TCI": _summarize_detector_onset(tci_warnings, config.collapse_start),
            "TCIE": _summarize_detector_onset(tcie_warnings, config.collapse_start),
            **{
                detector_name: _summarize_detector_onset(
                    warnings, config.collapse_start
                )
                for detector_name, warnings in baseline_warnings.items()
            },
        },
    )


def build_active_benchmark_rows(
    results: list[TPTActiveBenchmarkResult],
) -> list[dict[str, str | float | int]]:
    rows: list[dict[str, str | float | int]] = []
    for phase in ("masking_detection", "collapse_detection"):
        for detector in ("TCI", "TCIE", *ACTIVE_BASELINE_DETECTORS):
            delays = []
            detections = 0
            for result in results:
                summary = getattr(result, phase)[detector]
                delay = summary["delay"]
                if delay is not None:
                    detections += 1
                    delays.append(float(delay))
            rows.append(
                {
                    "phase": phase.replace("_", " "),
                    "detector": detector,
                    "detections": detections,
                    "n_runs": len(results),
                    "detection_rate": round(detections / len(results), 3),
                    "mean_delay": round(float(np.mean(delays)), 1) if delays else "NA",
                    "median_delay": round(float(np.median(delays)), 1)
                    if delays
                    else "NA",
                }
            )
    return rows


def build_tcie_calibration_rows(
    results: list[TPTActiveBenchmarkResult],
    lambdas: list[float],
    thresholds: list[float],
) -> list[dict[str, str | float | int]]:
    rows: list[dict[str, str | float | int]] = []
    grouped: dict[float, list[TPTActiveBenchmarkResult]] = {
        lambda_value: [] for lambda_value in lambdas
    }
    for result in results:
        grouped.setdefault(result.config.effort_penalty_lambda, []).append(result)

    for lambda_value in lambdas:
        lambda_results = grouped.get(lambda_value, [])
        if not lambda_results:
            continue
        for threshold in thresholds:
            masking_delays: list[float] = []
            collapse_delays: list[float] = []
            healthy_false_positives = 0
            masking_detections = 0
            collapse_detections = 0

            for result in lambda_results:
                masking_warnings = _threshold_warnings(result.tcie, threshold)
                healthy_false_positives += sum(
                    1
                    for warning in masking_warnings
                    if warning < result.config.masking_start
                )
                masking_summary = _summarize_detector_onset(
                    masking_warnings, result.config.masking_start
                )
                collapse_summary = _summarize_detector_onset(
                    masking_warnings, result.config.collapse_start
                )
                if masking_summary["delay"] is not None:
                    masking_detections += 1
                    masking_delays.append(float(masking_summary["delay"]))
                if collapse_summary["delay"] is not None:
                    collapse_detections += 1
                    collapse_delays.append(float(collapse_summary["delay"]))

            rows.append(
                {
                    "lambda": round(lambda_value, 3),
                    "threshold": round(threshold, 3),
                    "n_runs": len(lambda_results),
                    "masking_rate": round(masking_detections / len(lambda_results), 3),
                    "masking_median_delay": round(float(np.median(masking_delays)), 1)
                    if masking_delays
                    else "NA",
                    "collapse_rate": round(
                        collapse_detections / len(lambda_results), 3
                    ),
                    "collapse_median_delay": round(float(np.median(collapse_delays)), 1)
                    if collapse_delays
                    else "NA",
                    "healthy_false_positives": healthy_false_positives,
                    "mean_healthy_false_positives": round(
                        healthy_false_positives / len(lambda_results), 3
                    ),
                }
            )
    return rows


def save_tcie_calibration_figure(
    rows: list[dict[str, str | float | int]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lambdas = sorted({float(row["lambda"]) for row in rows})
    thresholds = sorted({float(row["threshold"]) for row in rows})
    masking_rate = np.full((len(lambdas), len(thresholds)), np.nan)
    masking_delay = np.full((len(lambdas), len(thresholds)), np.nan)
    healthy_fp = np.full((len(lambdas), len(thresholds)), np.nan)

    lambda_index = {value: index for index, value in enumerate(lambdas)}
    threshold_index = {value: index for index, value in enumerate(thresholds)}
    for row in rows:
        i = lambda_index[float(row["lambda"])]
        j = threshold_index[float(row["threshold"])]
        masking_rate[i, j] = float(row["masking_rate"])
        masking_delay[i, j] = (
            np.nan
            if row["masking_median_delay"] == "NA"
            else float(row["masking_median_delay"])
        )
        healthy_fp[i, j] = float(row["mean_healthy_false_positives"])

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), constrained_layout=True)
    panels = [
        (masking_rate, "Masking rate"),
        (masking_delay, "Masking median delay"),
        (healthy_fp, "Healthy false positives / run"),
    ]

    for axis, (matrix, title) in zip(axes, panels, strict=True):
        image = axis.imshow(matrix, aspect="auto", origin="lower")
        axis.set_title(title)
        axis.set_xticks(
            np.arange(len(thresholds)), [f"{value:.2f}" for value in thresholds]
        )
        axis.set_yticks(np.arange(len(lambdas)), [f"{value:.1f}" for value in lambdas])
        axis.set_xlabel("Threshold")
        axis.set_ylabel(r"$\lambda$")
        for row_index in range(matrix.shape[0]):
            for col_index in range(matrix.shape[1]):
                value = matrix[row_index, col_index]
                if np.isnan(value):
                    continue
                axis.text(
                    col_index,
                    row_index,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if value < np.nanmedian(matrix) else "black",
                )
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    axes[0].set_title("Effort-corrected score calibration on the active benchmark")
    fig.savefig(output_path)
    plt.close(fig)


def save_active_benchmark_figure(
    result: TPTActiveBenchmarkResult,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    time = np.arange(result.config.steps)
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)

    axes[0].plot(
        time, result.latent_state, color="black", linewidth=1.3, label="latent"
    )
    axes[0].plot(
        time,
        result.uncontrolled_state,
        color="0.6",
        linewidth=1.0,
        linestyle="--",
        label="free state",
    )
    axes[0].plot(
        time,
        result.posterior_mean,
        color="tab:blue",
        linewidth=1.2,
        label="posterior mean",
    )
    axes[0].axvline(result.config.masking_start, color="tab:orange", linestyle="--")
    axes[0].axvline(result.config.collapse_start, color="tab:red", linestyle="--")
    axes[0].set_ylabel("State")
    axes[0].set_title("Active benchmark: healthy -> coercive masking -> collapse")
    axes[0].legend(loc="upper left", ncol=3)

    axes[1].plot(
        time,
        result.adwin_signal,
        color="tab:purple",
        linewidth=1.1,
        label="ADWIN input",
    )
    for warning in result.adwin_warnings:
        axes[1].axvline(warning, color="tab:purple", alpha=0.08, linewidth=0.8)
    axes[1].set_ylabel("Observation")
    axes[1].legend(loc="upper left")

    axes[2].plot(
        time, result.action_gap, color="tab:brown", linewidth=1.1, label="action gap"
    )
    axes[2].plot(
        time,
        result.effort_signal,
        color="tab:pink",
        linewidth=1.3,
        label="coercive effort",
    )
    axes[2].set_ylabel("Effort")
    axes[2].legend(loc="upper left", ncol=2)

    axes[3].plot(
        time, result.tci, color="0.45", linewidth=1.0, linestyle="--", label="Score"
    )
    axes[3].plot(
        time,
        result.tcie,
        color="tab:red",
        linewidth=1.4,
        label="Effort-corrected score",
    )
    axes[3].axhline(
        result.config.tci_threshold, color="0.5", linestyle=":", linewidth=0.9
    )
    axes[3].axhline(
        result.config.tcie_threshold, color="tab:red", linestyle=":", linewidth=0.9
    )
    for warning in result.tci_warnings:
        axes[3].axvline(warning, color="0.6", alpha=0.08, linewidth=0.8)
    for warning in result.tcie_warnings:
        axes[3].axvline(warning, color="tab:red", alpha=0.08, linewidth=0.8)
    axes[3].set_ylabel("Score")
    axes[3].set_xlabel("Time step")
    axes[3].set_ylim(0.0, 1.05)
    axes[3].legend(loc="lower left", ncol=2)

    for axis in axes:
        axis.axvline(
            result.config.masking_start,
            color="tab:orange",
            linestyle="--",
            linewidth=0.9,
        )
        axis.axvline(
            result.config.collapse_start, color="tab:red", linestyle="--", linewidth=0.9
        )
        axis.grid(alpha=0.2, linewidth=0.5)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def run_tpt_ablation(config: TPTConfig | None = None) -> dict[str, TPTResult]:
    base_config = config or TPTConfig()
    results: dict[str, TPTResult] = {}
    for condition in ABLATION_CONDITIONS:
        condition_config = TPTConfig(
            steps=base_config.steps,
            particles=base_config.particles,
            seed=base_config.seed,
            drift=base_config.drift,
            influence=base_config.influence,
            process_scale=base_config.process_scale,
            observation_scale=base_config.observation_scale,
            observation_df=base_config.observation_df,
            resample_threshold=base_config.resample_threshold,
            prior_mean=base_config.prior_mean,
            prior_scale=base_config.prior_scale,
            condition=condition,
            actuation_noise_scale=base_config.actuation_noise_scale,
            fm1_sigma_phi_level=base_config.fm1_sigma_phi_level,
            fm3_sigma_phi_floor=base_config.fm3_sigma_phi_floor,
            effort_penalty_lambda=base_config.effort_penalty_lambda,
            effort_floor=base_config.effort_floor,
        )
        results[condition] = run_tpt_experiment(condition_config)
    return results


def run_coercive_masking_experiment(
    config: TPTConfig | None = None,
    active_influence: float = 0.3,
) -> dict[str, TPTResult]:
    base_config = config or TPTConfig(condition="fm1")
    base_kwargs = asdict(base_config)
    passive_config = TPTConfig(**{**base_kwargs, "condition": "fm1", "influence": 0.0})
    coercive_config = TPTConfig(
        **{**base_kwargs, "condition": "fm1", "influence": active_influence}
    )
    return {
        "passive": run_tpt_experiment(passive_config),
        "coercive": run_tpt_experiment(coercive_config),
    }


def summarize_result(result: TPTResult, tail_window: int = 60) -> dict[str, float]:
    tail = slice(-min(tail_window, result.config.steps), None)
    return {
        "mean_abs_error": float(np.mean(np.abs(result.tracking_error[tail]))),
        "mean_action_gap": float(np.mean(result.action_gap[tail])),
        "mean_effort": float(np.mean(result.effort_signal[tail])),
        "mean_sigma_p": float(np.mean(result.sigma_p[tail])),
        "mean_sigma_p_eff": float(np.mean(result.sigma_p_eff[tail])),
        "mean_sigma_a": float(np.mean(result.sigma_a[tail])),
        "mean_sigma_phi": float(np.mean(result.sigma_phi[tail])),
        "mean_tci": float(np.mean(result.tci[tail])),
        "mean_tcie": float(np.mean(result.tcie[tail])),
        "resampling_steps": float(np.sum(result.resampled)),
    }


def export_summary_csv(
    rows: list[dict[str, str | float | int]], output_path: Path
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_summary_markdown(rows: list[dict[str, str | float | int]]) -> str:
    if not rows:
        return ""
    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = [str(row[header]) for header in headers]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def build_masking_summary_rows(
    results: dict[str, TPTResult], tail_window: int = 60
) -> list[dict[str, str | float | int]]:
    rows: list[dict[str, str | float | int]] = []
    for regime in ("passive", "coercive"):
        result = results[regime]
        summary = summarize_result(result, tail_window=tail_window)
        rows.append(
            {
                "regime": regime,
                "condition": result.condition,
                "influence": round(result.config.influence, 3),
                "tail_abs_error": round(summary["mean_abs_error"], 3),
                "tail_action_gap": round(summary["mean_action_gap"], 3),
                "tail_effort": round(summary["mean_effort"], 3),
                "tail_sigma_p": round(summary["mean_sigma_p"], 3),
                "tail_sigma_p_eff": round(summary["mean_sigma_p_eff"], 3),
                "tail_sigma_a": round(summary["mean_sigma_a"], 3),
                "tail_sigma_phi": round(summary["mean_sigma_phi"], 3),
                "tail_tci": round(summary["mean_tci"], 3),
                "tail_tcie": round(summary["mean_tcie"], 3),
            }
        )
    return rows


def build_masking_grid_rows(
    config: TPTConfig,
    influences: list[float],
    lambdas: list[float],
    seeds: list[int],
    tail_window: int = 60,
) -> tuple[list[dict[str, str | float | int]], list[dict[str, str | float | int]]]:
    raw_rows: list[dict[str, str | float | int]] = []
    numeric_fields = [
        "tail_abs_error",
        "tail_action_gap",
        "tail_effort",
        "tail_sigma_p",
        "tail_sigma_p_eff",
        "tail_sigma_a",
        "tail_sigma_phi",
        "tail_tci",
        "tail_tcie",
    ]

    for influence in influences:
        for penalty in lambdas:
            for seed in seeds:
                run_config = TPTConfig(
                    steps=config.steps,
                    particles=config.particles,
                    seed=seed,
                    drift=config.drift,
                    influence=influence,
                    process_scale=config.process_scale,
                    observation_scale=config.observation_scale,
                    observation_df=config.observation_df,
                    resample_threshold=config.resample_threshold,
                    prior_mean=config.prior_mean,
                    prior_scale=config.prior_scale,
                    condition="fm1",
                    actuation_noise_scale=config.actuation_noise_scale,
                    fm1_sigma_phi_level=config.fm1_sigma_phi_level,
                    fm3_sigma_phi_floor=config.fm3_sigma_phi_floor,
                    effort_penalty_lambda=penalty,
                    effort_floor=config.effort_floor,
                )
                results = run_coercive_masking_experiment(
                    run_config,
                    active_influence=influence,
                )
                for regime, result in results.items():
                    summary = summarize_result(result, tail_window=tail_window)
                    raw_rows.append(
                        {
                            "seed": seed,
                            "regime": regime,
                            "condition": result.condition,
                            "influence": round(influence, 3),
                            "lambda": round(penalty, 3),
                            "tail_abs_error": round(summary["mean_abs_error"], 6),
                            "tail_action_gap": round(summary["mean_action_gap"], 6),
                            "tail_effort": round(summary["mean_effort"], 6),
                            "tail_sigma_p": round(summary["mean_sigma_p"], 6),
                            "tail_sigma_p_eff": round(summary["mean_sigma_p_eff"], 6),
                            "tail_sigma_a": round(summary["mean_sigma_a"], 6),
                            "tail_sigma_phi": round(summary["mean_sigma_phi"], 6),
                            "tail_tci": round(summary["mean_tci"], 6),
                            "tail_tcie": round(summary["mean_tcie"], 6),
                        }
                    )

    grouped: dict[
        tuple[str, float, float], dict[str, list[float] | str | float | int]
    ] = {}
    for row in raw_rows:
        key = (str(row["regime"]), float(row["influence"]), float(row["lambda"]))
        if key not in grouped:
            grouped[key] = {
                "regime": row["regime"],
                "influence": row["influence"],
                "lambda": row["lambda"],
                "n_seeds": 0,
                **{field: [] for field in numeric_fields},
            }
        grouped_row = grouped[key]
        grouped_row["n_seeds"] = int(grouped_row["n_seeds"]) + 1
        for field in numeric_fields:
            values = grouped_row[field]
            assert isinstance(values, list)
            values.append(float(row[field]))

    summary_rows: list[dict[str, str | float | int]] = []
    for key in sorted(grouped):
        grouped_row = grouped[key]
        output_row: dict[str, str | float | int] = {
            "regime": str(grouped_row["regime"]),
            "influence": grouped_row["influence"],
            "lambda": grouped_row["lambda"],
            "n_seeds": grouped_row["n_seeds"],
        }
        for field in numeric_fields:
            values = np.asarray(grouped_row[field], dtype=float)
            output_row[f"mean_{field}"] = round(float(np.mean(values)), 3)
            output_row[f"std_{field}"] = round(float(np.std(values)), 3)
        if output_row["regime"] == "coercive":
            output_row["mean_masking_gap"] = round(
                float(output_row["mean_tail_tci"])
                - float(output_row["mean_tail_tcie"]),
                3,
            )
        else:
            output_row["mean_masking_gap"] = 0.0
        summary_rows.append(output_row)

    return raw_rows, summary_rows


def save_masking_grid_figure(
    summary_rows: list[dict[str, str | float | int]],
    influences: list[float],
    lambdas: list[float],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    coercive_rows = [row for row in summary_rows if row["regime"] == "coercive"]
    influence_index = {value: index for index, value in enumerate(influences)}
    lambda_index = {value: index for index, value in enumerate(lambdas)}
    mean_tci = np.full((len(lambdas), len(influences)), np.nan)
    mean_tcie = np.full((len(lambdas), len(influences)), np.nan)
    masking_gap = np.full((len(lambdas), len(influences)), np.nan)

    for row in coercive_rows:
        i = lambda_index[float(row["lambda"])]
        j = influence_index[float(row["influence"])]
        mean_tci[i, j] = float(row["mean_tail_tci"])
        mean_tcie[i, j] = float(row["mean_tail_tcie"])
        masking_gap[i, j] = float(row["mean_masking_gap"])

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), constrained_layout=True)
    panels = [
        (mean_tci, "Coercive mean TCI"),
        (mean_tcie, "Coercive mean TCIE"),
        (masking_gap, "Masking gap: TCI - TCIE"),
    ]

    for axis, (matrix, title) in zip(axes, panels, strict=True):
        image = axis.imshow(matrix, aspect="auto", origin="lower", vmin=0.0, vmax=1.0)
        axis.set_title(title)
        axis.set_xticks(
            np.arange(len(influences)), [f"{value:.1f}" for value in influences]
        )
        axis.set_yticks(np.arange(len(lambdas)), [f"{value:.1f}" for value in lambdas])
        axis.set_xlabel("Influence")
        axis.set_ylabel(r"$\lambda$")
        for row_index in range(matrix.shape[0]):
            for col_index in range(matrix.shape[1]):
                value = matrix[row_index, col_index]
                axis.text(
                    col_index,
                    row_index,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if value < 0.45 else "black",
                )
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    fig.savefig(output_path)
    plt.close(fig)


def save_tpt_figure(result: TPTResult, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    time = np.arange(result.config.steps)
    band = 2.0 * result.posterior_std

    fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True)

    axes[0].plot(
        time, result.latent_state, label="latent state", color="black", linewidth=1.4
    )
    if result.config.influence > 0:
        axes[0].plot(
            time,
            result.uncontrolled_state,
            label="uncontrolled state",
            color="0.55",
            linewidth=1.0,
            linestyle="--",
        )
    axes[0].plot(
        time,
        result.posterior_mean,
        label="posterior mean",
        color="tab:blue",
        linewidth=1.3,
    )
    axes[0].fill_between(
        time,
        result.posterior_mean - band,
        result.posterior_mean + band,
        color="tab:blue",
        alpha=0.16,
        label="posterior +/- 2 std",
    )
    axes[0].scatter(
        time,
        result.observations,
        s=10,
        alpha=0.28,
        color="tab:orange",
        label="observations",
    )
    axes[0].set_ylabel("State")
    axes[0].set_title(
        f"Triadic Particle Tracker: {_condition_label(result.condition)} (influence={result.config.influence:.2f})"
    )
    axes[0].legend(loc="upper left", ncol=2)

    axes[1].plot(time, result.ess, label="ESS / N", color="tab:green", linewidth=1.2)
    axes[1].plot(
        time, result.entropy, label="weight entropy", color="tab:purple", linewidth=1.2
    )
    axes[1].axhline(
        result.config.resample_threshold,
        color="tab:red",
        linestyle="--",
        linewidth=1.0,
        label="resample threshold",
    )
    resampled_steps = time[result.resampled]
    if resampled_steps.size:
        axes[1].vlines(
            resampled_steps, 0.0, 1.0, color="tab:red", alpha=0.08, linewidth=0.8
        )
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_ylabel("Convergence")
    axes[1].legend(loc="lower left", ncol=3)

    axes[2].plot(
        time,
        result.action_gap,
        label="action gap",
        color="tab:brown",
        linewidth=1.1,
    )
    axes[2].plot(
        time,
        result.effort_signal,
        label="coercive effort",
        color="tab:pink",
        linewidth=1.3,
    )
    axes[2].set_ylabel("Effort")
    axes[2].legend(loc="upper left", ncol=2)

    axes[3].plot(
        time, result.sigma_p, label=r"$\sigma_P$", color="tab:blue", linewidth=1.0
    )
    axes[3].plot(
        time,
        result.sigma_p_eff,
        label=r"$\sigma_P^E$",
        color="tab:purple",
        linewidth=1.2,
    )
    axes[3].plot(
        time,
        result.sigma_a,
        label=r"$\sigma_A$",
        color="tab:orange",
        linewidth=1.0,
        alpha=0.8,
    )
    axes[3].plot(
        time,
        result.sigma_phi,
        label=r"$\sigma_\Phi$",
        color="tab:green",
        linewidth=1.0,
        alpha=0.8,
    )
    axes[3].plot(
        time, result.tci, label="Score", color="0.55", linewidth=1.0, linestyle="--"
    )
    axes[3].plot(
        time,
        result.tcie,
        label="Effort-corrected score",
        color="tab:red",
        linewidth=1.4,
    )
    axes[3].set_ylim(0.0, 1.05)
    axes[3].set_ylabel("Score")
    axes[3].set_xlabel("Time step")
    axes[3].legend(loc="lower left", ncol=6)

    for axis in axes:
        axis.grid(alpha=0.2, linewidth=0.5)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_tpt_ablation_figure(results: dict[str, TPTResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    time = np.arange(next(iter(results.values())).config.steps)
    colors = {
        "full": "tab:blue",
        "fm1": "tab:purple",
        "fm2": "tab:orange",
        "fm3": "tab:red",
    }

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    for condition in ABLATION_CONDITIONS:
        result = results[condition]
        axes[0].plot(
            time,
            _rolling_mean(np.abs(result.tracking_error), 20),
            color=colors[condition],
            linewidth=1.4,
            label=_condition_label(condition),
        )
        axes[1].plot(
            time,
            _rolling_mean(result.tcie, 20),
            color=colors[condition],
            linewidth=1.4,
            label=_condition_label(condition),
        )

    axes[0].set_title(
        "Particle-tracker ablation: rolling absolute tracking error and score"
    )
    axes[0].set_ylabel("|tracking error|")
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


def save_coercive_masking_figure(
    results: dict[str, TPTResult], output_path: Path
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    time = np.arange(next(iter(results.values())).config.steps)
    passive = results["passive"]
    coercive = results["coercive"]
    passive_summary = summarize_result(passive)
    coercive_summary = summarize_result(coercive)

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

    axes[0].plot(
        time,
        _rolling_mean(np.abs(passive.tracking_error), 20),
        color="tab:blue",
        linewidth=1.3,
        label="Passive FM-1 |error|",
    )
    axes[0].plot(
        time,
        _rolling_mean(np.abs(coercive.tracking_error), 20),
        color="tab:red",
        linewidth=1.3,
        label="Coercive FM-1 |error|",
    )
    axes[0].set_ylabel("|tracking error|")
    axes[0].set_title(
        "Coercive masking: score stays high while effort-corrected score pays for effort"
    )
    axes[0].legend(loc="upper left")

    axes[1].plot(
        time,
        _rolling_mean(passive.action_gap, 20),
        color="tab:blue",
        linewidth=1.3,
        label="Passive action gap",
    )
    axes[1].plot(
        time,
        _rolling_mean(coercive.action_gap, 20),
        color="tab:red",
        linewidth=1.3,
        label="Coercive action gap",
    )
    axes[1].plot(
        time,
        _rolling_mean(passive.effort_signal, 20),
        color="tab:cyan",
        linewidth=1.1,
        linestyle="--",
        label="Passive coercive effort",
    )
    axes[1].plot(
        time,
        _rolling_mean(coercive.effort_signal, 20),
        color="tab:orange",
        linewidth=1.2,
        linestyle="--",
        label="Coercive effort",
    )
    axes[1].set_ylabel("Effort")
    axes[1].legend(loc="upper left", ncol=2)

    axes[2].plot(
        time,
        _rolling_mean(passive.tci, 20),
        color="tab:blue",
        linewidth=1.2,
        linestyle="--",
        label="Passive score",
    )
    axes[2].plot(
        time,
        _rolling_mean(coercive.tci, 20),
        color="tab:red",
        linewidth=1.2,
        linestyle="--",
        label="Coercive score",
    )
    axes[2].plot(
        time,
        _rolling_mean(passive.tcie, 20),
        color="tab:cyan",
        linewidth=1.2,
        label="Passive effort-corrected score",
    )
    axes[2].plot(
        time,
        _rolling_mean(coercive.tcie, 20),
        color="tab:orange",
        linewidth=1.4,
        label="Coercive effort-corrected score",
    )
    axes[2].set_ylabel("Score")
    axes[2].set_xlabel("Time step")
    axes[2].set_ylim(0.0, 1.05)
    axes[2].legend(loc="lower left", ncol=2)
    axes[2].text(
        0.02,
        0.98,
        (
            "Passive\n"
            f"score={passive_summary['mean_tci']:.3f}\n"
            f"effort-corrected={passive_summary['mean_tcie']:.3f}\n"
            f"effort={passive_summary['mean_effort']:.3f}"
        ),
        transform=axes[2].transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.9},
    )
    axes[2].text(
        0.98,
        0.98,
        (
            "Coercive\n"
            f"score={coercive_summary['mean_tci']:.3f}\n"
            f"effort-corrected={coercive_summary['mean_tcie']:.3f}\n"
            f"effort={coercive_summary['mean_effort']:.3f}"
        ),
        transform=axes[2].transAxes,
        va="top",
        ha="right",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.9},
    )

    for axis in axes:
        axis.grid(alpha=0.2, linewidth=0.5)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
