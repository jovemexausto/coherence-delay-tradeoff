from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from .common import export_rows_csv
from .smooth_fixed_support_sinkhorn import (
    SmoothSinkhornConfig,
    run_smooth_fixed_support_sinkhorn,
)


@dataclass(slots=True)
class BandwiseFixedSupportResult:
    summary_rows: list[dict[str, str | float]]


def derive_bandwise_fixed_support_sinkhorn(
    epsilons: tuple[float, ...] = (0.8, 0.5, 0.3, 0.2, 0.1),
) -> BandwiseFixedSupportResult:
    result = run_smooth_fixed_support_sinkhorn(
        SmoothSinkhornConfig(
            n_values=(40, 80, 160, 320),
            replications=12,
            span_values=(0.25,),
            H_values=(0.5, 1.0),
            epsilons=epsilons,
        )
    )
    summary_rows = []
    for row in result.summary_rows:
        summary_rows.append(
            {
                "span": row["span"],
                "H": row["H"],
                "epsilon": row["epsilon"],
                "tri_slope": row["tri_slope"],
                "iid_slope": row["iid_slope"],
                "slope_gap": row["slope_gap"],
                "status": row["status"],
            }
        )
    return BandwiseFixedSupportResult(summary_rows=summary_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Derive compact-band fixed-support Sinkhorn diagnostics."
    )
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=Path("artifacts/csv/bandwise_fixed_support_sinkhorn"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = derive_bandwise_fixed_support_sinkhorn()
    export_rows_csv(result.summary_rows, args.csv_dir / "summary.csv")


if __name__ == "__main__":
    main()
