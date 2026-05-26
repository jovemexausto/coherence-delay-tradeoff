from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from .bridge_report import generate_bridge_reports
from .horizon_bridge import BridgeMisspecificationConfig, BridgeRecoveryConfig


def _figure_path(output_root: Path, name: str) -> Path:
    path = output_root / "figures" / "horizon_bridge" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_bridge_recovery_figure(rows: list[object], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))

    grouped: dict[tuple[int, float, float], list[object]] = {}
    for row in rows:
        grouped.setdefault((row.lag_count, row.H, row.zeta), []).append(row)

    ax = axes[0]
    for (lag_count, H, zeta), series in sorted(grouped.items()):
        series = sorted(series, key=lambda row: row.information_scale)
        ax.plot(
            [row.information_scale for row in series],
            [row.rmse_H for row in series],
            "o-",
            label=f"L={lag_count}, H={H}, zeta={zeta}",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("I_{n,L}(H)")
    ax.set_ylabel("RMSE(H_hat)")
    ax.set_title("Lag-regularity recovery")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    for (lag_count, H, zeta), series in sorted(grouped.items()):
        series = sorted(series, key=lambda row: row.information_scale)
        ax.plot(
            [row.information_scale for row in series],
            [row.rmse_n_star for row in series],
            "o-",
            label=f"L={lag_count}, H={H}, zeta={zeta}",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("I_{n,L}(H)")
    ax.set_ylabel("RMSE(n_star_hat)")
    ax.set_title("Plug-in horizon recovery")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_bridge_misspecification_figure(rows: list[object], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))

    grouped: dict[tuple[int, str], list[object]] = {}
    for row in rows:
        grouped.setdefault((row.lag_count, row.kind), []).append(row)

    ax = axes[0]
    for (lag_count, kind), series in sorted(grouped.items()):
        series = sorted(series, key=lambda row: row.amplitude)
        ax.plot(
            [row.amplitude for row in series],
            [abs(row.bias_H) for row in series],
            "o-",
            label=f"L={lag_count}, {kind}",
        )
    ax.set_xlabel("Amplitude")
    ax.set_ylabel("Abs. bias(H_hat)")
    ax.set_title("Regularity bias under misspecification")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    for (lag_count, kind), series in sorted(grouped.items()):
        series = sorted(series, key=lambda row: row.amplitude)
        ax.plot(
            [row.amplitude for row in series],
            [abs(row.bias_n_star) for row in series],
            "o-",
            label=f"L={lag_count}, {kind}",
        )
    ax.set_xlabel("Amplitude")
    ax.set_ylabel("Abs. bias(n_star_hat)")
    ax.set_title("Horizon bias under misspecification")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def generate_bridge_figures(
    output_root: Path,
    *,
    recovery_config: BridgeRecoveryConfig | None = None,
    misspec_config: BridgeMisspecificationConfig | None = None,
) -> dict[str, Path]:
    rows = generate_bridge_reports(
        output_root,
        recovery_config=recovery_config,
        misspec_config=misspec_config,
    )
    recovery_path = _figure_path(output_root, "fig_bridge_recovery.pdf")
    misspec_path = _figure_path(output_root, "fig_bridge_misspecification.pdf")
    save_bridge_recovery_figure(rows["recovery"], recovery_path)
    save_bridge_misspecification_figure(rows["misspecification"], misspec_path)
    return {
        "recovery": recovery_path,
        "misspecification": misspec_path,
    }


def main() -> None:
    workspace_root = Path(__file__).resolve().parents[2]
    generate_bridge_figures(workspace_root / "artifacts")


if __name__ == "__main__":
    main()
