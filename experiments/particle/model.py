from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np
from scipy.special import logsumexp  # pyright: ignore[reportUnknownVariableType]
from scipy.stats import t as student_t

from ..core.common import OnsetSummary, summarize_onset
from ..core.detectors import create_river_drift_detector

ABLATION_CONDITIONS = ("full", "fm1", "fm2", "fm3")
CONDITION_LABELS = {
    "full": "Full tracker",
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
    masking_detection: dict[str, OnsetSummary]
    collapse_detection: dict[str, OnsetSummary]


ACTIVE_BASELINE_DETECTORS = ("ADWIN", "PageHinkley", "KSWIN", "NoDrift")


def _state_dynamics(state: np.ndarray | float, drift: float) -> np.ndarray | float:
    return 0.72 * state + 0.18 * np.sin(1.4 * state) + drift


def _observation_model(state: np.ndarray | float) -> np.ndarray | float:
    return 0.35 * state**2 - 0.4 * np.cos(state)


def _apply_control(
    state: np.ndarray | float, action: np.ndarray | float, influence: float
) -> np.ndarray | float:
    return state + influence * (action - state)


def _condition_label(condition: str) -> str:
    return CONDITION_LABELS.get(condition, condition)


def _systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    count = weights.size
    positions = (rng.random() + np.arange(count)) / count
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, positions, side="left")


def _observe_state(state: float, config: TPTConfig, rng: np.random.Generator) -> float:
    observation = float(_observation_model(state))
    noise = float(rng.standard_t(config.observation_df) * config.observation_scale)
    return observation + noise


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
    return np.asarray(_apply_control(uncontrolled, previous_action, config.influence))


def run_particle_tracking_experiment(config: TPTConfig | None = None) -> TPTResult:
    config = config or TPTConfig()
    if config.condition not in ABLATION_CONDITIONS:
        raise ValueError(f"Unknown tracker condition: {config.condition}")

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
        log_likelihood = np.asarray(
            student_t.logpdf(
                observations[step],
                df=config.observation_df,
                loc=predicted_observation,
                scale=config.observation_scale,
            ),
            dtype=float,
        )
        log_evidence[step] = float(logsumexp(log_likelihood)) - math.log(  # pyright: ignore[reportUnknownArgumentType]
            config.particles
        )

        if config.condition == "fm3":
            weights = uniform_weights
            posterior_mean[step] = float(np.mean(candidate_particles))
            posterior_std[step] = float(np.std(candidate_particles))
            ess_ratio = 0.0
            entropy_ratio = 0.0
            sigma_phi_value = config.fm3_sigma_phi_floor
        elif config.condition == "fm1":
            weights = np.exp(log_likelihood - float(logsumexp(log_likelihood)))  # pyright: ignore[reportUnknownArgumentType]
            posterior_mean[step] = frozen_mean
            posterior_std[step] = frozen_std
            ess_ratio = 1.0
            entropy_ratio = 1.0
            sigma_phi_value = config.fm1_sigma_phi_level
        else:
            weights = np.exp(log_likelihood - float(logsumexp(log_likelihood)))
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


def run_particle_tracking_ablation(
    config: TPTConfig | None = None,
) -> dict[str, TPTResult]:
    config = config or TPTConfig()
    results: dict[str, TPTResult] = {}
    for condition in ABLATION_CONDITIONS:
        results[condition] = run_particle_tracking_experiment(
            replace(config, condition=condition)
        )
    return results


def run_coercive_masking_experiment(
    config: TPTConfig | None = None,
    active_influence: float | None = None,
) -> dict[str, TPTResult]:
    config = config or TPTConfig()
    passive_config = replace(config, influence=0.0)
    coercive_config = replace(
        config,
        influence=config.influence if active_influence is None else active_influence,
    )
    return {
        "passive": run_particle_tracking_experiment(passive_config),
        "coercive": run_particle_tracking_experiment(coercive_config),
    }


def run_particle_tracking_active_benchmark(
    config: TPTActiveBenchmarkConfig | None = None,
    verbose: bool = False,
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
    tci_warnings: list[int] = []
    tcie_warnings: list[int] = []
    tci_below = False
    tcie_below = False
    baseline_detectors = {
        name: create_river_drift_detector(
            name,
            adwin_delta=config.adwin_delta,
            page_hinkley_delta=config.page_hinkley_delta,
            page_hinkley_threshold=config.page_hinkley_threshold,
            page_hinkley_alpha=config.page_hinkley_alpha,
            kswin_window_size=config.kswin_window_size,
            kswin_stat_size=config.kswin_stat_size,
            kswin_alpha=config.kswin_alpha,
        )
        for name in ACTIVE_BASELINE_DETECTORS
    }
    baseline_warnings: dict[str, list[int]] = {
        name: [] for name in ACTIVE_BASELINE_DETECTORS
    }

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

    if verbose:
        print(
            f"active benchmark: steps={config.steps}, particles={config.particles}",
            flush=True,
        )

    for step in range(config.steps):
        if verbose and step % max(config.steps // 5, 1) == 0:
            print(f"  progress {step}/{config.steps}", flush=True)
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
            log_likelihood = np.asarray(
                student_t.logpdf(
                    observations[step],
                    df=config.observation_df,
                    loc=predicted_observation,
                    scale=config.observation_scale,
                ),
                dtype=float,
            )
            weights = np.exp(log_likelihood - float(logsumexp(log_likelihood)))
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

        if not tci_below and tci[step] < config.tci_threshold:
            tci_warnings.append(step)
            tci_below = True
        elif tci[step] >= config.tci_threshold:
            tci_below = False

        if not tcie_below and tcie[step] < config.tcie_threshold:
            tcie_warnings.append(step)
            tcie_below = True
        elif tcie[step] >= config.tcie_threshold:
            tcie_below = False

        for detector_name, detector in baseline_detectors.items():
            detector.update(float(baseline_signal[step]))
            if detector.drift_detected:
                baseline_warnings[detector_name].append(step)
    adwin_warnings = baseline_warnings["ADWIN"]

    if verbose:
        print("active benchmark: done", flush=True)

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
            "TCI": summarize_onset(tci_warnings, config.masking_start),
            "TCIE": summarize_onset(tcie_warnings, config.masking_start),
            **{
                detector_name: summarize_onset(warnings, config.masking_start)
                for detector_name, warnings in baseline_warnings.items()
            },
        },
        collapse_detection={
            "TCI": summarize_onset(tci_warnings, config.collapse_start),
            "TCIE": summarize_onset(tcie_warnings, config.collapse_start),
            **{
                detector_name: summarize_onset(warnings, config.collapse_start)
                for detector_name, warnings in baseline_warnings.items()
            },
        },
    )
