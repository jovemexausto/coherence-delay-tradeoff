from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable

from .experiments import (
    BoundaryPowerConfig,
    FWLSOracleConfig,
    NullCalibrationConfig,
    MisspecificationConfig,
    NoiseRobustnessConfig,
    RateConstantConfig,
    run_boundary_power_experiment,
    run_fwls_oracle_experiment,
    run_misspecification_experiment,
    run_noise_robustness_experiment,
    run_null_calibration_experiment,
    run_rate_constant_experiment,
)


def _row_to_dict(row: Any) -> dict[str, Any]:
    if is_dataclass(row):
        return asdict(row)
    if isinstance(row, dict):
        return row
    raise TypeError(f"unsupported row type: {type(row)!r}")


def export_rows_csv(rows: Iterable[Any], output_path: Path) -> None:
    import csv

    row_dicts = [_row_to_dict(row) for row in rows]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not row_dicts:
        output_path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in row_dicts:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(row_dicts)


def _format_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}".rstrip("0").rstrip(".")
    if isinstance(value, int):
        return str(value)
    return rf"\detokenize{{{value}}}"


def write_latex_table(
    rows: Iterable[Any],
    output_path: Path,
    *,
    columns: list[tuple[str, str]],
    caption: str,
    label: str,
    column_spec: str,
) -> None:
    row_dicts = [_row_to_dict(row) for row in rows]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\small",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\resizebox{\\linewidth}{!}{%",
        f"\\begin{{tabular}}{{{column_spec}}}",
        "\\toprule",
        " & ".join(header for _, header in columns) + r" \\",
        "\\midrule",
    ]
    for row in row_dicts:
        lines.append(" & ".join(_format_cell(row[key]) for key, _ in columns) + r" \\")
    lines.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\end{table}"])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def generate_v1_reports(
    output_root: Path,
    *,
    null_config: NullCalibrationConfig | None = None,
    fwls_config: FWLSOracleConfig | None = None,
    boundary_config: BoundaryPowerConfig | None = None,
    rate_config: RateConstantConfig | None = None,
    misspec_config: MisspecificationConfig | None = None,
    noise_config: NoiseRobustnessConfig | None = None,
) -> dict[str, list[Any]]:
    csv_root = output_root / "csv" / "scale_consistency"
    table_root = output_root / "tables" / "scale_consistency"

    null_rows = run_null_calibration_experiment(
        NullCalibrationConfig() if null_config is None else null_config
    )
    fwls_rows = run_fwls_oracle_experiment(
        FWLSOracleConfig() if fwls_config is None else fwls_config
    )
    boundary_rows = run_boundary_power_experiment(
        BoundaryPowerConfig() if boundary_config is None else boundary_config
    )
    rate_rows = run_rate_constant_experiment(
        RateConstantConfig() if rate_config is None else rate_config
    )
    misspec_rows = run_misspecification_experiment(
        MisspecificationConfig() if misspec_config is None else misspec_config
    )
    noise_rows = run_noise_robustness_experiment(
        NoiseRobustnessConfig() if noise_config is None else noise_config
    )

    export_rows_csv(null_rows, csv_root / "null_calibration.csv")
    export_rows_csv(fwls_rows, csv_root / "fwls_oracle.csv")
    export_rows_csv(boundary_rows, csv_root / "boundary_power.csv")
    export_rows_csv(rate_rows, csv_root / "rate_constant.csv")
    export_rows_csv(misspec_rows, csv_root / "misspecification.csv")
    export_rows_csv(noise_rows, csv_root / "noise_robustness.csv")

    write_latex_table(
        null_rows,
        table_root / "tab_null_size.tex",
        columns=[
            ("L", "$L$"),
            ("n", "$n$"),
            ("H", "$H$"),
            ("empirical_size", "Empirical size"),
            ("q_mean", "Mean($Q$)"),
            ("q_mean_theory", "$\\mathbb E[\\chi^2]$"),
        ],
        caption="Null calibration summary for the adequacy statistic.",
        label="tab:null-size-generated",
        column_spec="rrrrrr",
    )
    write_latex_table(
        fwls_rows,
        table_root / "tab_fwls_oracle.tex",
        columns=[
            ("L", "$L$"),
            ("n", "$n$"),
            ("rmse_h_gap", "RMSE gap"),
            ("mean_abs_q_gap", "Mean abs. $Q$ gap"),
            ("variance_ratio", "Variance ratio"),
        ],
        caption="FWLS-oracle agreement diagnostics.",
        label="tab:fwls-oracle-generated",
        column_spec="rrrrr",
    )
    write_latex_table(
        boundary_rows,
        table_root / "tab_boundary_power.tex",
        columns=[
            ("n", "$n$"),
            ("c", "$c$"),
            ("kappa", "$\\kappa$"),
            ("empirical_power", "Empirical power"),
        ],
        caption="Power near the minimax boundary.",
        label="tab:boundary-power-generated",
        column_spec="rrrr",
    )
    write_latex_table(
        rate_rows,
        table_root / "tab_rate_constant.tex",
        columns=[
            ("n", "$n$"),
            ("rmse_h", "RMSE($\\widehat H$)"),
            ("scaled_constant", "$C$"),
            ("oracle_scaled_constant", "Oracle $C$"),
        ],
        caption="Rate-constant stabilization for FWLS and oracle WLS.",
        label="tab:rate-constant-generated",
        column_spec="rrrr",
    )
    write_latex_table(
        misspec_rows,
        table_root / "tab_misspecification.tex",
        columns=[
            ("kind", "Kind"),
            ("amplitude", "Amplitude"),
            ("empirical_size", "Size"),
            ("mean_h", "Mean $H$"),
            ("mean_q", "Mean $Q$"),
        ],
        caption="Sensitivity of the adequacy statistic under scale misspecification.",
        label="tab:misspecification-generated",
        column_spec="lrrrr",
    )
    write_latex_table(
        noise_rows,
        table_root / "tab_noise_robustness.tex",
        columns=[
            ("noise", "Noise"),
            ("empirical_size", "Size"),
            ("mean_h", "Mean $H$"),
            ("mean_q", "Mean $Q$"),
        ],
        caption="Robustness of the adequacy procedure under alternative noise laws.",
        label="tab:noise-robustness-generated",
        column_spec="lrrr",
    )

    return {
        "null": null_rows,
        "fwls_oracle": fwls_rows,
        "boundary_power": boundary_rows,
        "rate_constant": rate_rows,
        "misspecification": misspec_rows,
        "noise_robustness": noise_rows,
    }


def main() -> None:
    workspace_root = Path(__file__).resolve().parents[2]
    generate_v1_reports(workspace_root / "artifacts")


if __name__ == "__main__":
    main()
