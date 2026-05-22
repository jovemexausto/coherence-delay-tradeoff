from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import f

from .estimation import (
    feasible_wls,
    oracle_wls,
    residual_statistic,
    run_split_scale_consistency_test,
    run_scale_consistency_test,
)
from .model import simulate_observed_discrepancies
from .model import misspecified_scale_profile
from .theory_diagnostics import (
    chi_square_null_mean,
    chi_square_null_variance,
    information_scale,
    kappa_boundary,
    lag_energy,
    scaled_rmse_constant,
)


@dataclass(frozen=True)
class NullCalibrationConfig:
    L_values: tuple[int, ...] = (10, 20, 30, 50)
    n_values: tuple[int, ...] = (1000,)
    H_values: tuple[float, ...] = (0.6,)
    sigma0_values: tuple[float, ...] = (1.0,)
    zeta: float = 1.0
    alpha_level: float = 0.05
    repetitions: int = 8000
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
    L_values: tuple[int, ...] = (20,)
    n_values: tuple[int, ...] = (200, 500, 1000, 2000, 5000)
    H: float = 0.6
    sigma0: float = 1.0
    zeta: float = 1.0
    repetitions: int = 5000
    seed: int = 4321


@dataclass(frozen=True)
class FWLSOracleRow:
    L: int
    n: int
    rmse_h_fwls: float
    rmse_h_oracle: float
    rmse_ratio: float
    rmse_h_gap: float
    mean_abs_q_gap: float


@dataclass(frozen=True)
class BoundaryPowerConfig:
    L: int = 20
    n_values: tuple[int, ...] = (1000,)
    c_values: tuple[float, ...] = (0.5, 1.0, 2.0, 3.0, 5.0)
    H: float = 0.6
    sigma0: float = 1.0
    zeta: float = 1.0
    alpha_level: float = 0.05
    repetitions: int = 5000
    seed: int = 2468


@dataclass(frozen=True)
class BoundaryPowerRow:
    n: int
    c: float
    kappa: float
    lag_energy: float
    information_scale: float
    boundary_scale: float
    empirical_power: float


@dataclass(frozen=True)
class RateConstantConfig:
    L: int = 20
    n_values: tuple[int, ...] = (200, 500, 1000, 2000, 5000)
    H: float = 0.6
    sigma0: float = 1.0
    zeta: float = 1.0
    repetitions: int = 5000
    seed: int = 1357


@dataclass(frozen=True)
class RateConstantRow:
    n: int
    rmse_h: float
    information_scale: float
    scaled_constant: float
    oracle_scaled_constant: float


@dataclass(frozen=True)
class Sigma0PluginConfig:
    L_values: tuple[int, ...] = (10, 20, 30, 50)
    n_values: tuple[int, ...] = (500, 1000)
    H_values: tuple[float, ...] = (0.6,)
    sigma0_values: tuple[float, ...] = (1.0,)
    zeta: float = 1.0
    alpha_level: float = 0.05
    repetitions: int = 300
    bootstrap_repetitions: int = 400
    seed: int = 24680


@dataclass(frozen=True)
class Sigma0PluginRow:
    L: int
    n: int
    H: float
    sigma0: float
    empirical_size_naive: float
    empirical_size_bootstrap: float
    empirical_size_oracle_split_f: float
    empirical_size_split_f: float
    mean_sigma0_hat: float
    mean_sigma0_hat_ratio: float
    mean_df_naive: float


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
            rmse_fwls = float(np.sqrt(np.mean(np.square(fwls_array - config.H))))
            rmse_oracle = float(np.sqrt(np.mean(np.square(oracle_array - config.H))))
            rows.append(
                FWLSOracleRow(
                    L=L,
                    n=n,
                    rmse_h_fwls=rmse_fwls,
                    rmse_h_oracle=rmse_oracle,
                    rmse_ratio=rmse_fwls / rmse_oracle,
                    rmse_h_gap=float(np.sqrt(np.mean(np.square(h_gaps)))),
                    mean_abs_q_gap=float(np.mean(np.abs(q_gaps))),
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
            energy = lag_energy(lags, config.H)
            info = information_scale(n, lags, config.H)
            kappa = kappa_boundary(n, lags, config.H, c)
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
                    lag_energy=energy,
                    information_scale=info,
                    boundary_scale=(kappa**2) * info,
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
                information_scale=information_scale(n, lags, config.H),
                scaled_constant=scaled_rmse_constant(rmse_h, n, config.L, config.H),
                oracle_scaled_constant=scaled_rmse_constant(
                    rmse_oracle, n, config.L, config.H
                ),
            )
        )
    return rows


def run_sigma0_plugin_experiment(
    config: Sigma0PluginConfig = Sigma0PluginConfig(),
) -> list[Sigma0PluginRow]:
    rng = np.random.default_rng(config.seed)
    rows: list[Sigma0PluginRow] = []
    for L in config.L_values:
        lags = _lags(L)
        for n in config.n_values:
            for H in config.H_values:
                for sigma0 in config.sigma0_values:
                    rejections = 0
                    bootstrap_rejections = 0
                    oracle_split_f_rejections = 0
                    split_f_rejections = 0
                    sigma0_hats: list[float] = []
                    dfs: list[int] = []
                    n_scale = max(n // 2, 1)
                    n_test = max(n - n_scale, 1)
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
                            None,
                            n,
                            alpha_level=config.alpha_level,
                            calibration="chi2",
                        )
                        bootstrap_result = run_scale_consistency_test(
                            obs,
                            lags,
                            None,
                            n,
                            alpha_level=config.alpha_level,
                            calibration="bootstrap",
                            bootstrap_repetitions=config.bootstrap_repetitions,
                            rng=rng,
                        )
                        scale_obs = simulate_observed_discrepancies(
                            lags,
                            config.zeta,
                            H,
                            sigma0,
                            n_scale,
                            rng=rng,
                        )
                        test_obs = simulate_observed_discrepancies(
                            lags,
                            config.zeta,
                            H,
                            sigma0,
                            n_test,
                            rng=rng,
                        )
                        split_result = run_split_scale_consistency_test(
                            scale_obs,
                            test_obs,
                            lags,
                            n_scale,
                            n_test,
                            alpha_level=config.alpha_level,
                        )
                        scale_y = np.log(scale_obs)
                        test_y = np.log(test_obs)
                        true_signal = np.log(config.zeta) + H * np.log(lags)
                        true_scale = config.zeta * lags**H
                        oracle_sigma0_sq_hat = (
                            float(n_scale)
                            * float(np.sum((true_scale * (scale_y - true_signal)) ** 2))
                            / float(len(lags))
                        )
                        oracle_sigma0_hat = float(np.sqrt(oracle_sigma0_sq_hat))
                        oracle_test = oracle_wls(
                            test_y,
                            lags,
                            config.zeta,
                            H,
                            oracle_sigma0_hat,
                            n_test,
                        )
                        oracle_split_statistic = residual_statistic(
                            oracle_test.residuals, oracle_test.weights
                        )
                        oracle_split_critical_value = float(
                            (len(lags) - 2)
                            * f.ppf(
                                1.0 - config.alpha_level,
                                len(lags) - 2,
                                len(lags),
                            )
                        )
                        rejections += int(result.reject)
                        bootstrap_rejections += int(bootstrap_result.reject)
                        oracle_split_f_rejections += int(
                            oracle_split_statistic > oracle_split_critical_value
                        )
                        split_f_rejections += int(split_result.reject)
                        sigma0_hats.append(
                            float(result.estimate.sigma0_hat)
                            if result.estimate.sigma0_hat is not None
                            else float("nan")
                        )
                        dfs.append(result.degrees_of_freedom)
                    sigma0_hat_array = np.asarray(sigma0_hats, dtype=float)
                    rows.append(
                        Sigma0PluginRow(
                            L=L,
                            n=n,
                            H=H,
                            sigma0=sigma0,
                            empirical_size_naive=float(rejections)
                            / float(config.repetitions),
                            empirical_size_bootstrap=float(bootstrap_rejections)
                            / float(config.repetitions),
                            empirical_size_oracle_split_f=float(
                                oracle_split_f_rejections
                            )
                            / float(config.repetitions),
                            empirical_size_split_f=float(split_f_rejections)
                            / float(config.repetitions),
                            mean_sigma0_hat=float(np.nanmean(sigma0_hat_array)),
                            mean_sigma0_hat_ratio=float(np.nanmean(sigma0_hat_array))
                            / float(sigma0),
                            mean_df_naive=float(np.mean(np.asarray(dfs, dtype=float))),
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
