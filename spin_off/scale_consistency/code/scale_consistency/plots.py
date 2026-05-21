from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from .experiments import (
    BoundaryPowerConfig,
    FWLSOracleConfig,
    MisspecificationConfig,
    NoiseRobustnessConfig,
    NullCalibrationConfig,
    RateConstantConfig,
)
from .report import generate_v1_reports


def _figure_path(output_root: Path, name: str) -> Path:
    path = output_root / "figures" / "scale_consistency" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_null_calibration_figure(rows: list[object], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    xs = [f"L={row.L}, n={row.n}" for row in rows]
    empirical = [row.q_mean for row in rows]
    theory = [row.q_mean_theory for row in rows]
    ax.plot(xs, empirical, "o-", label="Empirical mean($Q$)")
    ax.plot(xs, theory, "s--", label=r"$\chi^2(L-2)$ mean")
    ax.set_ylabel("Mean value")
    ax.set_title("Finite-sample null calibration")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_fwls_oracle_gap_figure(rows: list[object], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    grouped: dict[int, list[object]] = {}
    for row in rows:
        grouped.setdefault(row.L, []).append(row)
    for L, series in sorted(grouped.items()):
        series = sorted(series, key=lambda row: row.n)
        ax.plot(
            [row.n for row in series],
            [row.rmse_ratio for row in series],
            "o-",
            label=f"L={L}",
        )
    ax.set_xscale("log")
    ax.set_xlabel("n")
    ax.set_ylabel(r"RMSE$(\widehat H)$/RMSE$(\widehat H^*)$")
    ax.set_title("Feasible-oracle agreement")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_boundary_power_figure(rows: list[object], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    grouped: dict[int, list[object]] = {}
    for row in rows:
        grouped.setdefault(row.n, []).append(row)
    for n, series in sorted(grouped.items()):
        series = sorted(series, key=lambda row: row.c)
        ax.plot(
            [row.c for row in series],
            [row.empirical_power for row in series],
            "o-",
            label=f"n={n}",
        )
    ax.set_xlabel(r"$c = \kappa / \kappa^*$")
    ax.set_ylabel("Empirical power")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Power at the information-scale boundary")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_rate_constant_figure(rows: list[object], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    rows = sorted(rows, key=lambda row: row.n)
    ax.plot(
        [row.n for row in rows],
        [row.scaled_constant for row in rows],
        "o-",
        label="FWLS",
    )
    ax.plot(
        [row.n for row in rows],
        [row.oracle_scaled_constant for row in rows],
        "s--",
        label="Oracle",
    )
    ax.set_xscale("log")
    ax.set_xlabel("n")
    ax.set_ylabel("Information-normalized constant")
    ax.set_title("Information-normalized error constant")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_robustness_figure(
    misspec_rows: list[object],
    noise_rows: list[object],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))

    ax = axes[0]
    grouped: dict[str, list[object]] = {}
    for row in misspec_rows:
        grouped.setdefault(row.kind, []).append(row)
    for kind, series in sorted(grouped.items()):
        series = sorted(series, key=lambda row: row.amplitude)
        ax.plot(
            [row.amplitude for row in series],
            [row.empirical_size for row in series],
            "o-",
            label=kind,
        )
    ax.set_xlabel("Misspecification amplitude")
    ax.set_ylabel("Empirical size")
    ax.set_title("Power-law misspecification")
    ax.set_ylim(0.0, 1.05)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    ax.plot(
        [row.noise for row in noise_rows],
        [row.empirical_size for row in noise_rows],
        "o-",
        color="tab:purple",
    )
    ax.set_xlabel("Noise law")
    ax.set_ylabel("Empirical size")
    ax.set_title("Sensitivity to noise law")
    ax.set_ylim(0.0, 1.05)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def generate_v1_figures(
    output_root: Path,
    *,
    null_config: NullCalibrationConfig | None = None,
    fwls_config: FWLSOracleConfig | None = None,
    boundary_config: BoundaryPowerConfig | None = None,
    rate_config: RateConstantConfig | None = None,
    misspec_config: MisspecificationConfig | None = None,
    noise_config: NoiseRobustnessConfig | None = None,
) -> dict[str, Path]:
    rows = generate_v1_reports(
        output_root,
        null_config=null_config,
        fwls_config=fwls_config,
        boundary_config=boundary_config,
        rate_config=rate_config,
        misspec_config=misspec_config,
        noise_config=noise_config,
    )
    null_path = _figure_path(output_root, "fig_null_calibration.pdf")
    fwls_path = _figure_path(output_root, "fig_fwls_oracle_gap.pdf")
    power_path = _figure_path(output_root, "fig_power_boundary.pdf")
    rate_path = _figure_path(output_root, "fig_rate_constant.pdf")
    robustness_path = _figure_path(output_root, "fig_robustness.pdf")
    save_null_calibration_figure(rows["null"], null_path)
    save_fwls_oracle_gap_figure(rows["fwls_oracle"], fwls_path)
    save_boundary_power_figure(rows["boundary_power"], power_path)
    save_rate_constant_figure(rows["rate_constant"], rate_path)
    save_robustness_figure(
        rows["misspecification"], rows["noise_robustness"], robustness_path
    )
    return {
        "null_calibration": null_path,
        "fwls_oracle_gap": fwls_path,
        "power_boundary": power_path,
        "rate_constant": rate_path,
        "robustness": robustness_path,
    }


def main() -> None:
    workspace_root = Path(__file__).resolve().parents[2]
    generate_v1_figures(workspace_root / "artifacts")


if __name__ == "__main__":
    main()
