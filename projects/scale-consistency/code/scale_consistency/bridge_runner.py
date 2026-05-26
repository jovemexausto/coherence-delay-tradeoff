from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .bridge_plots import (
    save_bridge_misspecification_figure,
    save_bridge_recovery_figure,
)
from .horizon_bridge import (
    BridgeMisspecificationConfig,
    BridgeMisspecificationRow,
    BridgeRecoveryConfig,
    BridgeRecoveryRow,
    bootstrap_lag_power_law,
    continuous_optimal_horizon,
    run_bridge_misspecification_experiment,
    run_bridge_recovery_experiment,
)
from .model import simulate_observed_discrepancies
from .report import export_rows_csv


@dataclass(frozen=True)
class BridgeBootstrapCoverageRow:
    regime: str
    lag_count: int
    H: float
    method: str
    coverage_H: float
    coverage_n_star: float
    mean_interval_width_H: float
    mean_interval_width_n_star: float


def run_bridge_bootstrap_coverage_experiment(
    output_root: Path,
    *,
    repetitions: int = 30,
    bootstrap_repetitions: int = 120,
) -> dict[str, Any]:
    methods = ("parametric", "wild", "moving_block")
    regimes = (
        {
            "name": "power",
            "noise": "heteroskedastic_power",
            "alpha": 4.0,
            "beta": 1.5,
            "rho": 0.0,
        },
        {
            "name": "ar",
            "noise": "heteroskedastic_ar",
            "alpha": 0.35,
            "beta": 1.0,
            "rho": 0.8,
        },
    )
    rows: list[BridgeBootstrapCoverageRow] = []
    rng = np.random.default_rng(20240526)
    for regime in regimes:
        for lag_count in (80, 160):
            lags = np.arange(1, lag_count + 1, dtype=float)
            for H in (0.4, 0.8):
                true_n_star = continuous_optimal_horizon(1.0, 0.5, 1.0, 1.0, H)
                for method in methods:
                    H_coverages: list[float] = []
                    n_coverages: list[float] = []
                    H_widths: list[float] = []
                    n_widths: list[float] = []
                    for rep in range(repetitions):
                        obs = simulate_observed_discrepancies(
                            lags,
                            zeta=1.0,
                            H=H,
                            sigma0=0.5,
                            n=500,
                            noise=regime["noise"],
                            heteroskedastic_alpha=regime["alpha"],
                            heteroskedastic_beta=regime["beta"],
                            heteroskedastic_rho=regime["rho"],
                            rng=rng,
                        )
                        H_interval, n_interval = bootstrap_lag_power_law(
                            obs,
                            lags,
                            sigma0=0.5,
                            n=500,
                            bootstrap_repetitions=bootstrap_repetitions,
                            interval_level=0.95,
                            C_K=1.0,
                            a=0.5,
                            C_S=1.0,
                            method=method,
                            block_length=8,
                            rng=np.random.default_rng(1000 + rep),
                        )
                        H_coverages.append(
                            float(H_interval.lower <= H <= H_interval.upper)
                        )
                        n_coverages.append(
                            float(n_interval.lower <= true_n_star <= n_interval.upper)
                        )
                        H_widths.append(H_interval.upper - H_interval.lower)
                        n_widths.append(n_interval.upper - n_interval.lower)
                    rows.append(
                        BridgeBootstrapCoverageRow(
                            regime=regime["name"],
                            lag_count=lag_count,
                            H=H,
                            method=method,
                            coverage_H=float(np.mean(H_coverages)),
                            coverage_n_star=float(np.mean(n_coverages)),
                            mean_interval_width_H=float(np.mean(H_widths)),
                            mean_interval_width_n_star=float(np.mean(n_widths)),
                        )
                    )

    csv_root = output_root / "csv" / "horizon_bridge"
    csv_root.mkdir(parents=True, exist_ok=True)
    export_rows_csv(rows, csv_root / "bridge_bootstrap_coverage.csv")
    summary = {
        "rows": [asdict(row) for row in rows],
        "aggregated": [],
    }
    for regime_name in ("power", "ar"):
        for method in methods:
            subset = [
                row
                for row in rows
                if row.regime == regime_name and row.method == method
            ]
            summary["aggregated"].append(
                {
                    "regime": regime_name,
                    "method": method,
                    "coverage_H": float(np.mean([row.coverage_H for row in subset])),
                    "coverage_n_star": float(
                        np.mean([row.coverage_n_star for row in subset])
                    ),
                    "mean_interval_width_H": float(
                        np.mean([row.mean_interval_width_H for row in subset])
                    ),
                    "mean_interval_width_n_star": float(
                        np.mean([row.mean_interval_width_n_star for row in subset])
                    ),
                }
            )
    summary_path = _json_path(output_root, "bridge_bootstrap_coverage.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _figure_path(output_root: Path, name: str) -> Path:
    path = output_root / "figures" / "horizon_bridge" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _json_path(output_root: Path, name: str) -> Path:
    path = output_root / "tables" / "horizon_bridge" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def build_bridge_exploratory_plan() -> tuple[
    list[BridgeRecoveryConfig], list[BridgeMisspecificationConfig]
]:
    recovery_configs = [
        BridgeRecoveryConfig(
            lags=lag_count,
            n_values=(200, 500, 1000),
            H_values=(0.2, 0.4, 0.6, 0.8),
            zeta_values=(0.5, 1.0, 2.0),
            sigma0_values=(0.1, 0.5, 1.0),
            repetitions=200,
            bootstrap_repetitions=500,
            seed=1200 + lag_count,
        )
        for lag_count in (100, 200)
    ]
    shape_kinds = ("bump", "mixed", "piecewise", "sinusoid", "slope_shift")
    misspec_configs = [
        BridgeMisspecificationConfig(
            lags=lag_count,
            n=1000,
            H=H,
            zeta=1.0,
            sigma0=0.5,
            amplitudes=(0.0, 0.1, 0.2),
            kinds=shape_kinds,
            repetitions=200,
            seed=2400 + lag_count + int(100 * H),
        )
        for lag_count in (100, 200)
        for H in (0.4, 0.6, 0.8)
    ] + [
        BridgeMisspecificationConfig(
            lags=lag_count,
            n=1000,
            H=H,
            zeta=1.0,
            sigma0=0.5,
            amplitudes=(0.0, 0.5, 1.0),
            kinds=("heteroskedastic",),
            repetitions=200,
            seed=3400 + lag_count + int(100 * H),
        )
        for lag_count in (100, 200)
        for H in (0.4, 0.6, 0.8)
    ]
    return recovery_configs, misspec_configs


def build_bridge_smoke_plan() -> tuple[
    list[BridgeRecoveryConfig], list[BridgeMisspecificationConfig]
]:
    recovery_configs = [
        BridgeRecoveryConfig(
            lags=100,
            n_values=(200,),
            H_values=(0.6,),
            zeta_values=(1.0,),
            sigma0_values=(0.5,),
            repetitions=10,
            bootstrap_repetitions=20,
            seed=1100,
        )
    ]
    misspec_configs = [
        BridgeMisspecificationConfig(
            lags=100,
            n=200,
            H=0.6,
            zeta=1.0,
            sigma0=0.5,
            amplitudes=(0.0, 0.1),
            kinds=("sinusoid",),
            repetitions=10,
            seed=2100,
        )
    ]
    return recovery_configs, misspec_configs


def build_bridge_final_plan() -> tuple[
    list[BridgeRecoveryConfig], list[BridgeMisspecificationConfig]
]:
    recovery_configs = [
        BridgeRecoveryConfig(
            lags=lag_count,
            n_values=(200, 500, 1000),
            H_values=(0.2, 0.4, 0.6, 0.8),
            zeta_values=(0.5, 1.0, 2.0),
            sigma0_values=(0.1, 0.5, 1.0),
            repetitions=1000,
            bootstrap_repetitions=2000,
            seed=5200 + lag_count,
        )
        for lag_count in (100, 200)
    ]
    shape_kinds = ("bump", "mixed", "piecewise", "sinusoid", "slope_shift")
    misspec_configs = [
        BridgeMisspecificationConfig(
            lags=lag_count,
            n=1000,
            H=H,
            zeta=1.0,
            sigma0=0.5,
            amplitudes=(0.0, 0.1, 0.2, 0.3),
            kinds=shape_kinds,
            repetitions=1000,
            seed=6400 + lag_count + int(100 * H),
        )
        for lag_count in (100, 200)
        for H in (0.4, 0.6, 0.8)
    ] + [
        BridgeMisspecificationConfig(
            lags=lag_count,
            n=1000,
            H=H,
            zeta=1.0,
            sigma0=0.5,
            amplitudes=(0.0, 0.25, 0.5, 0.75),
            kinds=("heteroskedastic",),
            repetitions=1000,
            seed=7400 + lag_count + int(100 * H),
        )
        for lag_count in (100, 200)
        for H in (0.4, 0.6, 0.8)
    ]
    return recovery_configs, misspec_configs


def build_bridge_hetero_plan() -> tuple[
    list[BridgeRecoveryConfig], list[BridgeMisspecificationConfig]
]:
    recovery_configs = [
        BridgeRecoveryConfig(
            lags=lag_count,
            n_values=(500, 1000),
            H_values=(0.4, 0.6, 0.8),
            zeta_values=(1.0,),
            sigma0_values=(0.5,),
            repetitions=300,
            bootstrap_repetitions=500,
            seed=8200 + lag_count,
        )
        for lag_count in (100, 200)
    ]
    misspec_configs = [
        BridgeMisspecificationConfig(
            lags=lag_count,
            n=1000,
            H=0.6,
            zeta=1.0,
            sigma0=0.5,
            amplitudes=(0.0, 0.5, 1.0, 2.0),
            kinds=("heteroskedastic",),
            heteroskedastic_mode=mode,
            heteroskedastic_beta=beta,
            heteroskedastic_jump_lag=jump_lag,
            repetitions=300,
            seed=9200 + lag_count + int(10 * beta),
        )
        for lag_count in (100, 200)
        for mode, beta, jump_lag in (
            ("power", 1.0, None),
            ("power", 2.0, None),
            ("jump", 1.0, 50.0),
        )
    ]
    return recovery_configs, misspec_configs


def run_bridge_suite(
    output_root: Path,
    *,
    recovery_configs: list[BridgeRecoveryConfig],
    misspec_configs: list[BridgeMisspecificationConfig],
    label: str,
) -> dict[str, Any]:
    recovery_rows: list[BridgeRecoveryRow] = []
    misspec_rows: list[BridgeMisspecificationRow] = []
    for config in recovery_configs:
        recovery_rows.extend(run_bridge_recovery_experiment(config))
    for config in misspec_configs:
        misspec_rows.extend(run_bridge_misspecification_experiment(config))

    csv_root = output_root / "csv" / "horizon_bridge"
    csv_root.mkdir(parents=True, exist_ok=True)
    export_rows_csv(recovery_rows, csv_root / f"bridge_recovery_{label}.csv")
    export_rows_csv(
        misspec_rows,
        csv_root / f"bridge_misspecification_{label}.csv",
    )

    recovery_figure = _figure_path(output_root, f"fig_bridge_recovery_{label}.pdf")
    misspec_figure = _figure_path(
        output_root, f"fig_bridge_misspecification_{label}.pdf"
    )
    save_bridge_recovery_figure(recovery_rows, recovery_figure)
    save_bridge_misspecification_figure(misspec_rows, misspec_figure)

    summary = {
        "label": label,
        "recovery_rows": [asdict(row) for row in recovery_rows],
        "misspecification_rows": [asdict(row) for row in misspec_rows],
        "recovery_figure": str(recovery_figure),
        "misspecification_figure": str(misspec_figure),
    }
    summary_path = _json_path(output_root, f"bridge_suite_{label}.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run horizon-bridge experiment suites")
    parser.add_argument(
        "--mode",
        choices=("smoke", "exploratory", "final", "hetero", "bootstrap"),
        default="exploratory",
    )
    parser.add_argument(
        "--output-root",
        default=str(Path(__file__).resolve().parents[2] / "artifacts"),
    )
    args = parser.parse_args()

    if args.mode == "smoke":
        recovery_configs, misspec_configs = build_bridge_smoke_plan()
    elif args.mode == "exploratory":
        recovery_configs, misspec_configs = build_bridge_exploratory_plan()
    elif args.mode == "bootstrap":
        run_bridge_bootstrap_coverage_experiment(Path(args.output_root))
        return
    elif args.mode == "hetero":
        recovery_configs, misspec_configs = build_bridge_hetero_plan()
    else:
        recovery_configs, misspec_configs = build_bridge_final_plan()

    run_bridge_suite(
        Path(args.output_root),
        recovery_configs=recovery_configs,
        misspec_configs=misspec_configs,
        label=args.mode,
    )


if __name__ == "__main__":
    main()
