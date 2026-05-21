from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from .common import export_rows_csv
from .sinkhorn_moderate_band_probe import run_moderate_band_probe


@dataclass(frozen=True, slots=True)
class SelfCouplingCertificateRow:
    ambient_dim: int
    intrinsic_dim: int
    coupling: str
    worst_max_spectral_radius: float
    worst_max_inverse_norm: float
    largest_n_mean_inverse_norm: float
    worst_min_spectral_gap: float
    stable_proxy: bool


def certify_self_coupling_stability(
    max_spectral_radius: float,
    max_largest_n_mean_inverse_norm: float,
) -> list[dict[str, str | float | bool]]:
    _, summary_rows = run_moderate_band_probe()
    rows: list[dict[str, str | float | bool]] = []
    for row in summary_rows:
        coupling = str(row["coupling"])
        stable_proxy = (
            coupling in {"xx", "yy"}
            and float(row["worst_max_spectral_radius"]) <= max_spectral_radius
            and float(row["largest_n_mean_inverse_norm"])
            <= max_largest_n_mean_inverse_norm
            and float(row["worst_min_spectral_gap"]) > 0.0
        )
        rows.append(
            {
                "ambient_dim": int(row["ambient_dim"]),
                "intrinsic_dim": int(row["intrinsic_dim"]),
                "coupling": coupling,
                "worst_max_spectral_radius": float(row["worst_max_spectral_radius"]),
                "worst_max_inverse_norm": float(row["worst_max_inverse_norm"]),
                "largest_n_mean_inverse_norm": float(
                    row["largest_n_mean_inverse_norm"]
                ),
                "worst_min_spectral_gap": float(row["worst_min_spectral_gap"]),
                "stable_proxy": stable_proxy,
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate self-coupling stability proxy certificate."
    )
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=Path("artifacts/csv/sinkhorn_self_coupling_certificate"),
    )
    parser.add_argument("--max-spectral-radius", type=float, default=0.97)
    parser.add_argument("--max-largest-n-mean-inverse-norm", type=float, default=3.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = certify_self_coupling_stability(
        max_spectral_radius=args.max_spectral_radius,
        max_largest_n_mean_inverse_norm=args.max_largest_n_mean_inverse_norm,
    )
    export_rows_csv(rows, args.csv_dir / "summary.csv")


if __name__ == "__main__":
    main()
