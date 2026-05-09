"""Rajput-style uncertainty-guided ensemble benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..core.common import export_summary_csv
from ..rajput.artifacts import save_rajput_figure
from ..rajput.model import RajputConfig, run_rajput_benchmark
from ..rajput.reports import build_rajput_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("elec2", "bikes", "both"),
        default="both",
        help="Dataset to run the Rajput-style benchmark on",
    )
    parser.add_argument("--figures-dir", type=Path, default=Path("figures/rajput"))
    parser.add_argument(
        "--artifacts-dir", type=Path, default=Path("./artifacts/rajput")
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _run_one(dataset: str, figures_dir: Path, artifacts_dir: Path, seed: int) -> None:
    result = run_rajput_benchmark(RajputConfig(dataset=dataset, seed=seed))
    save_rajput_figure(result, figures_dir / f"fig_{dataset}_rajput.pdf")
    rows = build_rajput_rows(result)
    export_summary_csv(rows, artifacts_dir / f"{dataset}_rajput_summary.csv")
    for row in rows:
        print(row)
    print(f"Saved {figures_dir / f'fig_{dataset}_rajput.pdf'}")
    print(f"Saved {artifacts_dir / f'{dataset}_rajput_summary.csv'}")


def main() -> None:
    args = parse_args()
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    datasets = ("elec2", "bikes") if args.dataset == "both" else (args.dataset,)
    for dataset in datasets:
        _run_one(dataset, args.figures_dir, args.artifacts_dir, args.seed)


if __name__ == "__main__":
    main()
