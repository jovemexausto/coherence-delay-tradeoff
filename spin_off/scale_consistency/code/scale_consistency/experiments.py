from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .estimation import (
    feasible_wls,
    oracle_wls,
    residual_statistic,
    run_scale_consistency_test,
)
from .model import simulate_observed_discrepancies
from .model import misspecified_scale_profile
from .theory_diagnostics import (
    chi_square_null_mean,
    chi_square_null_variance,
    kappa_boundary,
    scaled_rmse_constant,
)


@dataclass(frozen=True)
class NullCalibrationConfig:
    L_values: tuple[int, ...] = (10, 20)
    n_values: tuple[int, ...] = (200, 1000)
    H_values: tuple[float, ...] = (0.3, 0.6)
    sigma0_values: tuple[float, ...] = (1.0,)
    zeta: float = 1.0
    alpha_level: float = 0.05
    repetitions: int = 200
    seed: int = 1234


@dataclass(frozen=True)
class NullCalibrationRow:
    L: int
    n: int
    H: float
    sigma0: float
    empirical_size: float
    q_mean: float
    q_variance: float
    q_mean_theory: float
    q_variance_theory: float


@dataclass(frozen=True)
class FWLSOracleConfig:
    L_values: tuple[int, ...] = (10, 20)
    n_values: tuple[int, ...] = (200, 1000)
    H: float = 0.6
    sigma0: float = 1.0
    zeta: float = 1.0
    repetitions: int = 200
    seed: int = 4321


@dataclass(frozen=True)
class FWLSOracleRow:
    L: int
    n: int
    rmse_h_gap: float
    mean_abs_q_gap: float
    variance_ratio: float


@dataclass(frozen=True)
class BoundaryPowerConfig:
    L: int = 20
    n_values: tuple[int, ...] = (200, 1000)
    c_values: tuple[float, ...] = (0.5, 1.0, 2.0, 3.0)
    H: float = 0.6
    sigma0: float = 1.0
    zeta: float = 1.0
    alpha_level: float = 0.05
    repetitions: int = 200
    seed: int = 2468


@dataclass(frozen=True)
class BoundaryPowerRow:
    n: int
    c: float
    kappa: float
    empirical_power: float


@dataclass(frozen=True)
class RateConstantConfig:
    L: int = 20
    n_values: tuple[int, ...] = (200, 500, 1000)
    H: float = 0.6
    sigma0: float = 1.0
    zeta: float = 1.0
    repetitions: int = 200
    seed: int = 1357


@dataclass(frozen=True)
class RateConstantRow:
    n: int
    rmse_h: float
    scaled_constant: float
    oracle_scaled_constant: float


@dataclass(frozen=True)
class MisspecificationConfig:
    L: int = 20
    n: int = 1000
    H: float = 0.6
    sigma0: float = 1.0
    zeta: float = 1.0
    amplitudes: tuple[float, ...] = (0.0, 0.05, 0.10, 0.20)
    kinds: tuple[str, ...] = ("bump", "sinusoid", "slope_shift")
    alpha_level: float = 0.05
    repetitions: int = 200
    seed: int = 97531


@dataclass(frozen=True)
class MisspecificationRow:
    kind: str
    amplitude: float
    empirical_size: float
    mean_h: float
    mean_q: float


@dataclass(frozen=True)
class NoiseRobustnessConfig:
    L: int = 20
    n: int = 1000
    H: float = 0.6
    sigma0: float = 1.0
    zeta: float = 1.0
    noise_models: tuple[str, ...] = ("gaussian", "bounded", "student")
    student_df: float = 8.0
    alpha_level: float = 0.05
    repetitions: int = 200
    seed: int = 86420


@dataclass(frozen=True)
class NoiseRobustnessRow:
    noise: str
    empirical_size: float
    mean_h: float
    mean_q: float


def _lags(L: int) -> np.ndarray:
    return np.arange(1, L + 1, dtype=float)


def run_null_calibration_experiment(
    config: NullCalibrationConfig = NullCalibrationConfig(),
) -> list[NullCalibrationRow]:
    rng = np.random.default_rng(config.seed)
    rows: list[NullCalibrationRow] = []
    for L in config.L_values:
        lags = _lags(L)
        for n in config.n_values:
            for H in config.H_values:
                for sigma0 in config.sigma0_values:
                    rejections = 0
                    statistics: list[float] = []
                    for _ in range(config.repetitions):
                        obs = simulate_observed_discrepancies(
                            lags,
                            config.zeta,
                            H,
                            sigma0,
                            n,
                            rng=rng,
                        )
                        result = run_scale_consistency_test(
                            obs,
                            lags,
                            sigma0,
                            n,
                            alpha_level=config.alpha_level,
                        )
                        rejections += int(result.reject)
                        statistics.append(result.statistic)
                    q_array = np.asarray(statistics, dtype=float)
                    rows.append(
                        NullCalibrationRow(
                            L=L,
                            n=n,
                            H=H,
                            sigma0=sigma0,
                            empirical_size=float(rejections)
                            / float(config.repetitions),
                            q_mean=float(np.mean(q_array)),
                            q_variance=float(np.var(q_array, ddof=1)),
                            q_mean_theory=chi_square_null_mean(L),
                            q_variance_theory=chi_square_null_variance(L),
                        )
                    )
    return rows


def run_fwls_oracle_experiment(
    config: FWLSOracleConfig = FWLSOracleConfig(),
) -> list[FWLSOracleRow]:
    rng = np.random.default_rng(config.seed)
    rows: list[FWLSOracleRow] = []
    for L in config.L_values:
        lags = _lags(L)
        for n in config.n_values:
            h_gaps: list[float] = []
            q_gaps: list[float] = []
            fwls_values: list[float] = []
            oracle_values: list[float] = []
            for _ in range(config.repetitions):
                obs = simulate_observed_discrepancies(
                    lags,
                    config.zeta,
                    config.H,
                    config.sigma0,
                    n,
                    rng=rng,
                )
                y = np.log(obs)
                fwls = feasible_wls(y, lags, config.sigma0, n)
                oracle = oracle_wls(y, lags, config.zeta, config.H, config.sigma0, n)
                h_gaps.append(fwls.H - oracle.H)
                q_gaps.append(
                    residual_statistic(fwls.residuals, fwls.weights)
                    - residual_statistic(oracle.residuals, oracle.weights)
                )
                fwls_values.append(fwls.H)
                oracle_values.append(oracle.H)
            fwls_array = np.asarray(fwls_values, dtype=float)
            oracle_array = np.asarray(oracle_values, dtype=float)
            rows.append(
                FWLSOracleRow(
                    L=L,
                    n=n,
                    rmse_h_gap=float(np.sqrt(np.mean(np.square(h_gaps)))),
                    mean_abs_q_gap=float(np.mean(np.abs(q_gaps))),
                    variance_ratio=float(
                        np.var(fwls_array, ddof=1) / np.var(oracle_array, ddof=1)
                    ),
                )
            )
    return rows


def run_boundary_power_experiment(
    config: BoundaryPowerConfig = BoundaryPowerConfig(),
) -> list[BoundaryPowerRow]:
    rng = np.random.default_rng(config.seed)
    rows: list[BoundaryPowerRow] = []
    lags = _lags(config.L)
    for n in config.n_values:
        for c in config.c_values:
            kappa = c * kappa_boundary(n, config.L)
            rejections = 0
            for _ in range(config.repetitions):
                obs = simulate_observed_discrepancies(
                    lags,
                    config.zeta,
                    config.H,
                    config.sigma0,
                    n,
                    kappa=kappa,
                    rng=rng,
                )
                result = run_scale_consistency_test(
                    obs,
                    lags,
                    config.sigma0,
                    n,
                    alpha_level=config.alpha_level,
                )
                rejections += int(result.reject)
            rows.append(
                BoundaryPowerRow(
                    n=n,
                    c=c,
                    kappa=kappa,
                    empirical_power=float(rejections) / float(config.repetitions),
                )
            )
    return rows


def run_rate_constant_experiment(
    config: RateConstantConfig = RateConstantConfig(),
) -> list[RateConstantRow]:
    rng = np.random.default_rng(config.seed)
    rows: list[RateConstantRow] = []
    lags = _lags(config.L)
    for n in config.n_values:
        fwls_errors: list[float] = []
        oracle_errors: list[float] = []
        for _ in range(config.repetitions):
            obs = simulate_observed_discrepancies(
                lags,
                config.zeta,
                config.H,
                config.sigma0,
                n,
                rng=rng,
            )
            y = np.log(obs)
            fwls = feasible_wls(y, lags, config.sigma0, n)
            oracle = oracle_wls(y, lags, config.zeta, config.H, config.sigma0, n)
            fwls_errors.append(fwls.H - config.H)
            oracle_errors.append(oracle.H - config.H)
        rmse_h = float(np.sqrt(np.mean(np.square(fwls_errors))))
        rmse_oracle = float(np.sqrt(np.mean(np.square(oracle_errors))))
        rows.append(
            RateConstantRow(
                n=n,
                rmse_h=rmse_h,
                scaled_constant=scaled_rmse_constant(rmse_h, n, config.L, config.H),
                oracle_scaled_constant=scaled_rmse_constant(
                    rmse_oracle, n, config.L, config.H
                ),
            )
        )
    return rows


def run_misspecification_experiment(
    config: MisspecificationConfig = MisspecificationConfig(),
) -> list[MisspecificationRow]:
    rng = np.random.default_rng(config.seed)
    rows: list[MisspecificationRow] = []
    lags = _lags(config.L)
    for kind in config.kinds:
        for amplitude in config.amplitudes:
            rejections = 0
            h_values: list[float] = []
            q_values: list[float] = []
            profile = misspecified_scale_profile(
                lags,
                config.zeta,
                config.H,
                amplitude,
                kind=kind,
            )
            for _ in range(config.repetitions):
                obs = simulate_observed_discrepancies(
                    lags,
                    config.zeta,
                    config.H,
                    config.sigma0,
                    config.n,
                    rng=rng,
                    profile=profile,
                )
                result = run_scale_consistency_test(
                    obs,
                    lags,
                    config.sigma0,
                    config.n,
                    alpha_level=config.alpha_level,
                )
                rejections += int(result.reject)
                h_values.append(result.estimate.H)
                q_values.append(result.statistic)
            rows.append(
                MisspecificationRow(
                    kind=kind,
                    amplitude=amplitude,
                    empirical_size=float(rejections) / float(config.repetitions),
                    mean_h=float(np.mean(h_values)),
                    mean_q=float(np.mean(q_values)),
                )
            )
    return rows


def run_noise_robustness_experiment(
    config: NoiseRobustnessConfig = NoiseRobustnessConfig(),
) -> list[NoiseRobustnessRow]:
    rng = np.random.default_rng(config.seed)
    rows: list[NoiseRobustnessRow] = []
    lags = _lags(config.L)
    for noise in config.noise_models:
        rejections = 0
        h_values: list[float] = []
        q_values: list[float] = []
        for _ in range(config.repetitions):
            obs = simulate_observed_discrepancies(
                lags,
                config.zeta,
                config.H,
                config.sigma0,
                config.n,
                rng=rng,
                noise=noise,
                student_df=config.student_df,
            )
            result = run_scale_consistency_test(
                obs,
                lags,
                config.sigma0,
                config.n,
                alpha_level=config.alpha_level,
            )
            rejections += int(result.reject)
            h_values.append(result.estimate.H)
            q_values.append(result.statistic)
        rows.append(
            NoiseRobustnessRow(
                noise=noise,
                empirical_size=float(rejections) / float(config.repetitions),
                mean_h=float(np.mean(h_values)),
                mean_q=float(np.mean(q_values)),
            )
        )
    return rows
