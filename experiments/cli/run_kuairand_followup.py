"""KuaiRand follow-up analysis entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..kuairand.followup import write_followup_report
from ..kuairand.model import KuaiRandConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("./artifacts/kuairand"),
    )
    parser.add_argument(
        "--kuairand-data-dir",
        type=Path,
        default=Path("../data/kuairand/KuaiRand-Pure/data"),
    )
    parser.add_argument("--kuairand-window-size", type=int, default=20)
    parser.add_argument("--kuairand-min-phase-count", type=int, default=20)
    parser.add_argument("--kuairand-max-users", type=int, default=367)
    parser.add_argument("--kuairand-threshold-quantile", type=float, default=0.20)
    parser.add_argument("--kuairand-tcie-lambda", type=float, default=3.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = write_followup_report(
        args.artifacts_dir,
        KuaiRandConfig(
            data_dir=args.kuairand_data_dir,
            window_size=args.kuairand_window_size,
            min_phase_count=args.kuairand_min_phase_count,
            max_users=args.kuairand_max_users,
            threshold_quantile=args.kuairand_threshold_quantile,
            tcie_lambda=args.kuairand_tcie_lambda,
        ),
    )
    for path in outputs.values():
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
