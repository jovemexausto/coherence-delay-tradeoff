from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import pandas as pd

from .meta_sensing_benchmark import MetaSensingConfig, run_meta_sensing_benchmark


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts"
FIGURE_ROOT = ARTIFACT_ROOT / "figures" / "meta_sensing"
TABLE_ROOT = ARTIFACT_ROOT / "tables" / "meta_sensing"
for root in (FIGURE_ROOT, TABLE_ROOT):
    root.mkdir(parents=True, exist_ok=True)


def build_meta_sensing_figure(
    *,
    config: MetaSensingConfig = MetaSensingConfig(),
    rng_seed: int = 0,
    output_path: Path | None = None,
) -> pd.DataFrame:
    rows = run_meta_sensing_benchmark(config=config, rng_seed=rng_seed)
    df = pd.DataFrame([asdict(row) for row in rows])
    df.to_csv(TABLE_ROOT / "meta_sensing_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    palette = {"single": "#1f77b4", "multiscale": "#d62728"}

    for sensor_mode, sub in df.groupby("sensor_mode", sort=False):
        sub = sub.sort_values("sensor_noise")
        axes[0].plot(
            sub["sensor_noise"],
            sub["mean_route_delay"],
            marker="o",
            linewidth=2.0,
            color=palette.get(sensor_mode, "#333333"),
            label=sensor_mode,
        )
        axes[1].plot(
            sub["sensor_noise"],
            sub["mean_pre_route_cost"],
            marker="o",
            linewidth=2.0,
            color=palette.get(sensor_mode, "#333333"),
            label=sensor_mode,
        )

    axes[0].set_title("Routing Delay vs Sensor Noise")
    axes[0].set_xlabel("sensor noise")
    axes[0].set_ylabel("mean route delay")
    axes[0].legend(frameon=False)

    axes[1].set_title("Pre-routing Cost vs Sensor Noise")
    axes[1].set_xlabel("sensor noise")
    axes[1].set_ylabel("mean pre-route cost")
    axes[1].legend(frameon=False)

    fig.savefig(
        output_path or (FIGURE_ROOT / "fig_meta_sensing_frontier.pdf"),
        bbox_inches="tight",
    )
    fig.savefig(
        FIGURE_ROOT / "fig_meta_sensing_frontier.png", dpi=220, bbox_inches="tight"
    )
    plt.close(fig)
    return df


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build meta-sensing figure.")
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--switch-step", type=int, default=24)
    parser.add_argument("--trials", type=int, default=16)
    parser.add_argument("--lag-count", type=int, default=40)
    parser.add_argument("--lag-reps", type=int, default=8)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = MetaSensingConfig(
        steps=args.steps,
        switch_step=args.switch_step,
        trials=args.trials,
        lag_count=args.lag_count,
        lag_reps=args.lag_reps,
    )
    df = build_meta_sensing_figure(config=config, rng_seed=0)
    if args.json:
        print(df.to_json(orient="records", indent=2))
        return
    print(df)


if __name__ == "__main__":
    main()
