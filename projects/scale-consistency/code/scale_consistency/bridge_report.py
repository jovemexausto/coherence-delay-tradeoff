from __future__ import annotations

from pathlib import Path
from typing import Any

from .horizon_bridge import (
    BridgeMisspecificationConfig,
    BridgeRecoveryConfig,
    run_bridge_misspecification_experiment,
    run_bridge_recovery_experiment,
)
from .report import export_rows_csv, write_latex_table


def generate_bridge_reports(
    output_root: Path,
    *,
    recovery_config: BridgeRecoveryConfig | None = None,
    misspec_config: BridgeMisspecificationConfig | None = None,
) -> dict[str, list[Any]]:
    csv_root = output_root / "csv" / "horizon_bridge"
    table_root = output_root / "tables" / "horizon_bridge"

    recovery_rows = run_bridge_recovery_experiment(
        BridgeRecoveryConfig() if recovery_config is None else recovery_config
    )
    misspec_rows = run_bridge_misspecification_experiment(
        BridgeMisspecificationConfig() if misspec_config is None else misspec_config
    )

    export_rows_csv(recovery_rows, csv_root / "bridge_recovery.csv")
    export_rows_csv(misspec_rows, csv_root / "bridge_misspecification.csv")

    write_latex_table(
        recovery_rows,
        table_root / "tab_bridge_recovery.tex",
        columns=[
            ("lag_count", "$L$"),
            ("n", "$n$"),
            ("H", "$H$"),
            ("zeta", "$\\zeta$"),
            ("information_scale", "$\\mathcal I_{n,L}(H)$"),
            ("rmse_H", "RMSE($\\widehat H$)"),
            ("coverage_H", "Cov.($\\widehat H$)"),
            ("rmse_zeta", "RMSE($\\widehat\\zeta$)"),
            ("rmse_n_star", "RMSE($\\widehat n_\\star$)"),
            ("coverage_n_star", "Cov.($\\widehat n_\\star$)"),
            ("mean_residual_slope", "Mean residual slope"),
            ("mean_durbin_watson", "Mean DW"),
            ("mean_curvature_p_value", "Mean curvature $p$"),
            ("mean_periodogram_peak_frequency", "Peak freq."),
            ("mean_periodogram_peak_power", "Peak power"),
            ("mean_tail_kl_residual", "Tail KL(res)"),
            ("max_tail_kl_residual", "Max KL(res)"),
            (
                "mean_tail_kl_standardized_residual",
                "Tail KL(std res)",
            ),
            (
                "max_tail_kl_standardized_residual",
                "Max KL(std res)",
            ),
            ("mean_tail_kl_variance", "Tail KL(var)"),
            ("max_tail_kl_variance", "Max KL(var)"),
            ("mean_tail_kl_log_observed", "Tail KL(log D)"),
            ("max_tail_kl_log_observed", "Max KL(log D)"),
        ],
        caption="Synthetic bridge recovery for the lag-geometry to horizon plug-in pipeline.",
        label="tab:bridge-recovery",
        column_spec="rrrrrrrrrrrrrrrrrrrrrrr",
    )
    write_latex_table(
        misspec_rows,
        table_root / "tab_bridge_misspecification.tex",
        columns=[
            ("lag_count", "$L$"),
            ("kind", "Perturbation"),
            ("amplitude", "Amplitude"),
            ("bias_H", "Bias($\\widehat H$)"),
            ("bias_n_star", "Bias($\\widehat n_\\star$)"),
            ("rmse_n_star", "RMSE($\\widehat n_\\star$)"),
            ("mean_residual_slope", "Mean residual slope"),
            ("mean_durbin_watson", "Mean DW"),
            ("mean_curvature_p_value", "Mean curvature $p$"),
            ("mean_periodogram_peak_frequency", "Peak freq."),
            ("mean_periodogram_peak_power", "Peak power"),
            ("mean_tail_kl_residual", "Tail KL(res)"),
            ("max_tail_kl_residual", "Max KL(res)"),
            (
                "mean_tail_kl_standardized_residual",
                "Tail KL(std res)",
            ),
            (
                "max_tail_kl_standardized_residual",
                "Max KL(std res)",
            ),
            ("mean_tail_kl_variance", "Tail KL(var)"),
            ("max_tail_kl_variance", "Max KL(var)"),
            ("mean_tail_kl_log_observed", "Tail KL(log D)"),
            ("max_tail_kl_log_observed", "Max KL(log D)"),
        ],
        caption="Sensitivity of horizon recovery to lag-law misspecification.",
        label="tab:bridge-misspecification",
        column_spec="rlrrrrrrrrrrrrrrrrr",
    )

    return {
        "recovery": recovery_rows,
        "misspecification": misspec_rows,
    }


def main() -> None:
    workspace_root = Path(__file__).resolve().parents[2]
    generate_bridge_reports(workspace_root / "artifacts")


if __name__ == "__main__":
    main()
