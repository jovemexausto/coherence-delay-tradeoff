from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from river import drift as river_drift

from ..core.horizon_baselines import compute_useful_memory_regulation


@dataclass(slots=True)
class TemporalInvalidityConfig:
    seeds: tuple[int, ...] = (42, 43, 44, 45, 46, 47)
    steps: int = 6000
    dimensions: int = 4
    phase_lengths: tuple[int, int, int] = (2000, 2000, 2000)
    phase_zeta: tuple[float, float, float] = (0.0005, 0.003, 0.0008)
    feature_scale: float = 1.8
    mean_amplitude: float = 1.0
    label_noise: float = 0.15
    warmup: int = 500
    fixed_long_window: int = 400
    fixed_short_window: int = 100
    adwin_delta: float = 0.002
    umr_block_size: int = 30
    umr_ema_alpha: float = 0.05
    umr_baseline_window: int = 500
    umr_prefix_length: int = 500
    umr_scale: float = 2.0
    umr_min_window: int = 50
    umr_max_window: int = 400
    logistic_l2: float = 1e-3
    logistic_max_iter: int = 8
    logistic_tol: float = 1e-6
    rolling_window: int = 200


@dataclass(slots=True)
class WindowedLogisticRegression:
    n_features: int
    l2: float = 1e-3
    max_iter: int = 8
    tol: float = 1e-6
    coefficients: np.ndarray | None = None

    @staticmethod
    def _sigmoid(score: np.ndarray) -> np.ndarray:
        out = np.empty_like(score, dtype=float)
        positive = score >= 0.0
        out[positive] = 1.0 / (1.0 + np.exp(-score[positive]))
        exp_score = np.exp(score[~positive])
        out[~positive] = exp_score / (1.0 + exp_score)
        return out

    def fit(self, x: np.ndarray, y: np.ndarray) -> "WindowedLogisticRegression":
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.ndim != 2:
            raise ValueError("x must be 2-D")
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must have matching rows")
        if x.shape[0] == 0:
            self.coefficients = np.zeros(self.n_features + 1, dtype=float)
            return self

        x_aug = np.column_stack([np.ones(x.shape[0]), x])
        beta = np.zeros(self.n_features + 1, dtype=float)
        reg = np.eye(self.n_features + 1, dtype=float)
        reg[0, 0] = 0.0

        for _ in range(self.max_iter):
            scores = x_aug @ beta
            probs = self._sigmoid(scores)
            gradient = (x_aug.T @ (probs - y)) / x.shape[0] + self.l2 * (reg @ beta)
            weights = probs * (1.0 - probs)
            hessian = (x_aug.T * weights) @ x_aug / x.shape[0] + self.l2 * reg
            hessian += 1e-8 * np.eye(hessian.shape[0], dtype=float)
            step = np.linalg.solve(hessian, gradient)
            beta -= step
            if float(np.linalg.norm(step)) < self.tol:
                break

        self.coefficients = beta
        return self

    def predict_proba(self, x: np.ndarray) -> float:
        if self.coefficients is None:
            return 0.5
        x = np.asarray(x, dtype=float)
        score = float(self.coefficients[0] + np.dot(self.coefficients[1:], x))
        return float(self._sigmoid(np.asarray([score]))[0])


@dataclass(slots=True)
class TemporalInvaliditySeedTrace:
    seed: int
    features: np.ndarray
    targets: np.ndarray
    true_zeta: np.ndarray
    phase_index: np.ndarray
    drift_signal: np.ndarray
    umr_width_raw: np.ndarray
    umr_width_policy: np.ndarray
    adwin_width_raw: np.ndarray
    adwin_width_policy: np.ndarray
    adwin_event: np.ndarray
    adwin_silent_for_prediction: np.ndarray
    cap_only_mask: np.ndarray
    fixed_400_prob: np.ndarray
    fixed_100_prob: np.ndarray
    adwin_prob: np.ndarray
    umr_prob: np.ndarray
    fixed_400_accuracy: np.ndarray
    fixed_100_accuracy: np.ndarray
    adwin_accuracy: np.ndarray
    umr_accuracy: np.ndarray
    fixed_400_log_loss: np.ndarray
    fixed_100_log_loss: np.ndarray
    adwin_log_loss: np.ndarray
    umr_log_loss: np.ndarray


@dataclass(slots=True)
class MethodSummary:
    global_accuracy_mean: float
    global_accuracy_std: float
    global_log_loss_mean: float
    global_log_loss_std: float
    cap_only_accuracy_mean: float
    cap_only_accuracy_std: float
    cap_only_log_loss_mean: float
    cap_only_log_loss_std: float
    delta_cap_only_vs_fixed_400_pp_mean: float
    delta_cap_only_vs_fixed_400_pp_std: float


@dataclass(slots=True)
class TemporalInvalidityResult:
    config: TemporalInvalidityConfig
    time: np.ndarray
    representative: TemporalInvaliditySeedTrace
    traces: list[TemporalInvaliditySeedTrace]
    summaries: dict[str, MethodSummary]
    cap_only_segments: list[tuple[int, int]]


def _simulate_stream(
    seed: int, config: TemporalInvalidityConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    features = np.zeros((config.steps, config.dimensions), dtype=float)
    targets = np.zeros(config.steps, dtype=int)
    true_zeta = np.zeros(config.steps, dtype=float)
    phase_index = np.zeros(config.steps, dtype=int)
    drift_signal = np.zeros(config.steps, dtype=float)

    angle = 0.0
    for step in range(config.steps):
        phase = 0
        if step >= config.phase_lengths[0] + config.phase_lengths[1]:
            phase = 2
        elif step >= config.phase_lengths[0]:
            phase = 1
        phase_index[step] = phase
        zeta = config.phase_zeta[phase]
        true_zeta[step] = zeta
        angle += zeta
        mean = np.array(
            [
                config.mean_amplitude * np.cos(angle),
                config.mean_amplitude * np.sin(angle),
                0.5 * config.mean_amplitude * np.cos(angle / 2.0),
                0.5 * config.mean_amplitude * np.sin(angle / 2.0),
            ],
            dtype=float,
        )
        w = np.array(
            [
                np.cos(angle),
                np.sin(angle),
                np.cos(angle / 3.0),
                np.sin(angle / 3.0),
            ],
            dtype=float,
        )
        w /= float(np.linalg.norm(w))
        x = mean + rng.normal(scale=config.feature_scale, size=config.dimensions)
        score = float(np.dot(x, w) + rng.normal(scale=config.label_noise))
        features[step] = x
        targets[step] = int(score > 0.0)
        drift_signal[step] = float(x[0])

    return features, targets, true_zeta, phase_index, drift_signal


def _fit_predict_windowed_logistic(
    features: np.ndarray,
    targets: np.ndarray,
    window_series: np.ndarray,
    *,
    warmup: int,
    l2: float,
    max_iter: int,
    tol: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    probabilities = np.full(targets.size, np.nan, dtype=float)
    accuracy = np.full(targets.size, np.nan, dtype=float)
    log_loss = np.full(targets.size, np.nan, dtype=float)

    for step in range(targets.size):
        if step < max(1, warmup):
            continue
        window = int(round(float(window_series[step])))
        window = max(1, min(window, step))
        start = max(0, step - window)
        x_train = features[start:step]
        y_train = targets[start:step].astype(float)
        if x_train.shape[0] < 2 or np.unique(y_train).size < 2:
            probabilities[step] = 0.5
        else:
            model = WindowedLogisticRegression(
                n_features=features.shape[1],
                l2=l2,
                max_iter=max_iter,
                tol=tol,
            )
            model.fit(x_train, y_train)
            probabilities[step] = model.predict_proba(features[step])

        pred = float(probabilities[step] >= 0.5)
        accuracy[step] = float(pred == targets[step])
        p = float(np.clip(probabilities[step], 1e-9, 1.0 - 1e-9))
        log_loss[step] = float(
            -(targets[step] * np.log(p) + (1.0 - targets[step]) * np.log(1.0 - p))
        )

    return probabilities, accuracy, log_loss


def _build_adwin_series(
    reference_accuracy: np.ndarray,
    config: TemporalInvalidityConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    detector = river_drift.ADWIN(delta=config.adwin_delta)
    width_after_update = np.zeros(reference_accuracy.size, dtype=float)
    event = np.zeros(reference_accuracy.size, dtype=bool)
    for step, value in enumerate(reference_accuracy):
        if not np.isfinite(value):
            width_after_update[step] = (
                width_after_update[step - 1]
                if step > 0
                else float(config.fixed_long_window)
            )
            event[step] = False
            continue
        detector.update(float(1.0 - value))
        width_after_update[step] = float(detector.width)
        event[step] = bool(detector.drift_detected)

    policy_width = np.full(reference_accuracy.size, float(config.fixed_long_window))
    policy_width[1:] = width_after_update[:-1]
    policy_width = np.clip(
        policy_width, float(config.umr_min_window), float(config.fixed_long_window)
    )

    silent_for_prediction = np.ones(reference_accuracy.size, dtype=bool)
    silent_for_prediction[1:] = ~np.maximum.accumulate(event[:-1])
    return policy_width, event, silent_for_prediction


def _build_cap_only_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    start: int | None = None
    for index, active in enumerate(mask):
        if active and start is None:
            start = index
        elif not active and start is not None:
            segments.append((start, index))
            start = None
    if start is not None:
        segments.append((start, mask.size))
    return segments


def _summarize_seed(
    trace: TemporalInvaliditySeedTrace,
    *,
    config: TemporalInvalidityConfig,
) -> dict[str, dict[str, float]]:
    valid = np.arange(trace.targets.size) >= config.warmup
    cap = valid & trace.cap_only_mask
    summaries: dict[str, dict[str, float]] = {}
    for name, accuracy, log_loss in (
        ("fixed_400", trace.fixed_400_accuracy, trace.fixed_400_log_loss),
        ("fixed_100", trace.fixed_100_accuracy, trace.fixed_100_log_loss),
        ("adwin", trace.adwin_accuracy, trace.adwin_log_loss),
        ("umr", trace.umr_accuracy, trace.umr_log_loss),
    ):
        global_mask = valid & np.isfinite(accuracy)
        cap_mask = cap & np.isfinite(accuracy)
        summaries[name] = {
            "global_accuracy": float(np.mean(accuracy[global_mask]))
            if np.any(global_mask)
            else float("nan"),
            "global_log_loss": float(np.mean(log_loss[global_mask]))
            if np.any(global_mask)
            else float("nan"),
            "cap_only_accuracy": float(np.mean(accuracy[cap_mask]))
            if np.any(cap_mask)
            else float("nan"),
            "cap_only_log_loss": float(np.mean(log_loss[cap_mask]))
            if np.any(cap_mask)
            else float("nan"),
        }
    summaries["cap_only"] = {
        "steps": float(np.sum(cap)),
        "segments": float(len(_build_cap_only_segments(cap))),
        "mean_umr_width": float(np.mean(trace.umr_width_policy[cap]))
        if np.any(cap)
        else float("nan"),
        "mean_adwin_width": float(np.mean(trace.adwin_width_policy[cap]))
        if np.any(cap)
        else float("nan"),
    }
    return summaries


def _aggregate_summaries(
    traces: list[TemporalInvaliditySeedTrace],
    config: TemporalInvalidityConfig,
) -> dict[str, MethodSummary]:
    per_method: dict[str, dict[str, list[float]]] = {
        "fixed_400": {
            k: []
            for k in (
                "global_accuracy",
                "global_log_loss",
                "cap_only_accuracy",
                "cap_only_log_loss",
            )
        },
        "fixed_100": {
            k: []
            for k in (
                "global_accuracy",
                "global_log_loss",
                "cap_only_accuracy",
                "cap_only_log_loss",
            )
        },
        "adwin": {
            k: []
            for k in (
                "global_accuracy",
                "global_log_loss",
                "cap_only_accuracy",
                "cap_only_log_loss",
            )
        },
        "umr": {
            k: []
            for k in (
                "global_accuracy",
                "global_log_loss",
                "cap_only_accuracy",
                "cap_only_log_loss",
            )
        },
        "delta": {k: [] for k in ("delta_cap_only_vs_fixed_400_pp",)},
    }
    for trace in traces:
        summary = _summarize_seed(trace, config=config)
        for method in ("fixed_400", "fixed_100", "adwin", "umr"):
            for key in (
                "global_accuracy",
                "global_log_loss",
                "cap_only_accuracy",
                "cap_only_log_loss",
            ):
                per_method[method][key].append(summary[method][key])
        if np.isfinite(summary["fixed_400"]["cap_only_accuracy"]) and np.isfinite(
            summary["umr"]["cap_only_accuracy"]
        ):
            per_method["delta"]["delta_cap_only_vs_fixed_400_pp"].append(
                100.0
                * (
                    summary["umr"]["cap_only_accuracy"]
                    - summary["fixed_400"]["cap_only_accuracy"]
                )
            )

    summaries: dict[str, MethodSummary] = {}
    for method in ("fixed_400", "fixed_100", "adwin", "umr"):
        summaries[method] = MethodSummary(
            global_accuracy_mean=float(
                np.nanmean(per_method[method]["global_accuracy"])
            ),
            global_accuracy_std=float(np.nanstd(per_method[method]["global_accuracy"])),
            global_log_loss_mean=float(
                np.nanmean(per_method[method]["global_log_loss"])
            ),
            global_log_loss_std=float(np.nanstd(per_method[method]["global_log_loss"])),
            cap_only_accuracy_mean=float(
                np.nanmean(per_method[method]["cap_only_accuracy"])
            ),
            cap_only_accuracy_std=float(
                np.nanstd(per_method[method]["cap_only_accuracy"])
            ),
            cap_only_log_loss_mean=float(
                np.nanmean(per_method[method]["cap_only_log_loss"])
            ),
            cap_only_log_loss_std=float(
                np.nanstd(per_method[method]["cap_only_log_loss"])
            ),
            delta_cap_only_vs_fixed_400_pp_mean=float(
                np.nanmean(per_method["delta"]["delta_cap_only_vs_fixed_400_pp"])
            )
            if method == "umr"
            else 0.0,
            delta_cap_only_vs_fixed_400_pp_std=float(
                np.nanstd(per_method["delta"]["delta_cap_only_vs_fixed_400_pp"])
            )
            if method == "umr"
            else 0.0,
        )
    return summaries


def run_temporal_invalidity_benchmark(
    config: TemporalInvalidityConfig | None = None,
) -> TemporalInvalidityResult:
    config = config or TemporalInvalidityConfig()
    traces: list[TemporalInvaliditySeedTrace] = []

    for seed in config.seeds:
        features, targets, true_zeta, phase_index, drift_signal = _simulate_stream(
            seed, config
        )
        umr_regulator = compute_useful_memory_regulation(
            drift_signal,
            block_size=config.umr_block_size,
            ema_alpha=config.umr_ema_alpha,
            baseline_window=config.umr_baseline_window,
            prefix_length=min(config.umr_prefix_length, config.steps),
            scale=config.umr_scale,
            min_window=config.umr_min_window,
            max_window=config.umr_max_window,
        )
        umr_width_raw = np.clip(
            umr_regulator.window_sizes.astype(float),
            float(config.umr_min_window),
            float(config.umr_max_window),
        )

        fixed_400_prob, fixed_400_accuracy, fixed_400_log_loss = (
            _fit_predict_windowed_logistic(
                features,
                targets,
                np.full(config.steps, float(config.fixed_long_window), dtype=float),
                warmup=config.warmup,
                l2=config.logistic_l2,
                max_iter=config.logistic_max_iter,
                tol=config.logistic_tol,
            )
        )
        fixed_100_prob, fixed_100_accuracy, fixed_100_log_loss = (
            _fit_predict_windowed_logistic(
                features,
                targets,
                np.full(config.steps, float(config.fixed_short_window), dtype=float),
                warmup=config.warmup,
                l2=config.logistic_l2,
                max_iter=config.logistic_max_iter,
                tol=config.logistic_tol,
            )
        )

        adwin_width_raw, adwin_event, adwin_silent_for_prediction = _build_adwin_series(
            fixed_400_accuracy, config
        )
        adwin_width_policy = np.full(config.steps, float(config.fixed_long_window))
        adwin_width_policy[1:] = adwin_width_raw[:-1]
        adwin_width_policy = np.clip(
            adwin_width_policy,
            float(config.umr_min_window),
            float(config.fixed_long_window),
        )

        adwin_prob, adwin_accuracy, adwin_log_loss = _fit_predict_windowed_logistic(
            features,
            targets,
            adwin_width_policy,
            warmup=config.warmup,
            l2=config.logistic_l2,
            max_iter=config.logistic_max_iter,
            tol=config.logistic_tol,
        )
        umr_prob, umr_accuracy, umr_log_loss = _fit_predict_windowed_logistic(
            features,
            targets,
            umr_width_raw,
            warmup=config.warmup,
            l2=config.logistic_l2,
            max_iter=config.logistic_max_iter,
            tol=config.logistic_tol,
        )

        cap_only_mask = (
            (phase_index == 1)
            & (np.arange(config.steps) >= config.warmup)
            & (umr_width_raw < float(config.fixed_long_window))
            & adwin_silent_for_prediction
        )

        traces.append(
            TemporalInvaliditySeedTrace(
                seed=seed,
                features=features,
                targets=targets,
                true_zeta=true_zeta,
                phase_index=phase_index,
                drift_signal=drift_signal,
                umr_width_raw=umr_width_raw,
                umr_width_policy=umr_width_raw,
                adwin_width_raw=adwin_width_raw,
                adwin_width_policy=adwin_width_policy,
                adwin_event=adwin_event,
                adwin_silent_for_prediction=adwin_silent_for_prediction,
                cap_only_mask=cap_only_mask,
                fixed_400_prob=fixed_400_prob,
                fixed_100_prob=fixed_100_prob,
                adwin_prob=adwin_prob,
                umr_prob=umr_prob,
                fixed_400_accuracy=fixed_400_accuracy,
                fixed_100_accuracy=fixed_100_accuracy,
                adwin_accuracy=adwin_accuracy,
                umr_accuracy=umr_accuracy,
                fixed_400_log_loss=fixed_400_log_loss,
                fixed_100_log_loss=fixed_100_log_loss,
                adwin_log_loss=adwin_log_loss,
                umr_log_loss=umr_log_loss,
            )
        )

    summaries = _aggregate_summaries(traces, config)
    representative = traces[0]
    cap_only_segments = _build_cap_only_segments(representative.cap_only_mask)
    return TemporalInvalidityResult(
        config=config,
        time=np.arange(config.steps),
        representative=representative,
        traces=traces,
        summaries=summaries,
        cap_only_segments=cap_only_segments,
    )
