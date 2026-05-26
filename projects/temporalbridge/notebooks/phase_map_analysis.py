# %%
from __future__ import annotations

import os
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from temporalbridge.benchmarks import simulate_grid


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts"
CSV_ROOT = ARTIFACT_ROOT / "csv" / "phase_map_analysis"
FIGURE_ROOT = ARTIFACT_ROOT / "figures" / "phase_map_analysis"
TABLE_ROOT = ARTIFACT_ROOT / "tables" / "phase_map_analysis"
for root in (CSV_ROOT, FIGURE_ROOT, TABLE_ROOT):
    root.mkdir(parents=True, exist_ok=True)


LAMBDA0_VALUES = (0.0, 0.5, 1.0, 2.0, 4.0)
LAMBDA1_VALUES = (0.0, 0.5, 1.0, 2.0, 4.0)
TRUTH_H_VALUES = (0.1, 0.3, 0.5, 0.7, 0.9)
SCHEDULE_MODES = ("default", "strong")


def _parse_float_list(name: str, default: tuple[float, ...]) -> tuple[float, ...]:
    value = os.environ.get(name)
    if not value:
        return default
    return tuple(float(item) for item in value.split(",") if item.strip())


def _parse_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value else default


def _policy_code(policy: str) -> int:
    mapping = {
        "controller": 0,
        "detector_only": 1,
        "deploy_only": 2,
        "fixed_policy": 3,
        "oracle": 4,
    }
    return mapping.get(policy, -1)


def _plot_winner_heatmaps(
    cell_df: pd.DataFrame,
    *,
    schedule_mode: str,
    truth_h_values: tuple[float, ...],
    lambda0_values: tuple[float, ...],
    lambda1_values: tuple[float, ...],
) -> None:
    fig, axes = plt.subplots(
        len(truth_h_values), 1, figsize=(7.0, 2.4 * len(truth_h_values)), sharex=True
    )
    if len(truth_h_values) == 1:
        axes = [axes]

    policy_labels = ["controller", "detector_only", "deploy_only", "fixed_policy"]
    cmap = plt.get_cmap("tab10", len(policy_labels))

    for ax, truth_h in zip(axes, truth_h_values, strict=False):
        subset = cell_df[
            (cell_df["schedule_mode"] == schedule_mode)
            & (cell_df["truth_h"] == truth_h)
        ]
        pivot = subset.pivot(
            index="lambda_0", columns="lambda_1", values="best_non_oracle_policy"
        )
        codes = pivot.apply(lambda col: col.map(_policy_code)).to_numpy(dtype=float)
        im = ax.imshow(
            codes, origin="lower", cmap=cmap, vmin=0, vmax=len(policy_labels) - 1
        )
        ax.set_title(f"{schedule_mode} | H={truth_h}")
        ax.set_xticks(range(len(lambda1_values)))
        ax.set_xticklabels([str(v) for v in lambda1_values])
        ax.set_yticks(range(len(lambda0_values)))
        ax.set_yticklabels([str(v) for v in lambda0_values])
        ax.set_xlabel("lambda_1")
        ax.set_ylabel("lambda_0")

        for i, lam0 in enumerate(lambda0_values):
            for j, lam1 in enumerate(lambda1_values):
                policy = pivot.loc[lam0, lam1]
                ax.text(j, i, policy[:2], ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    cbar.set_ticks(range(len(policy_labels)))
    cbar.set_ticklabels(policy_labels)
    fig.tight_layout()
    fig.savefig(FIGURE_ROOT / f"phase_winner_heatmaps_{schedule_mode}.png", dpi=220)
    plt.close(fig)


def _plot_controller_advantage(
    row_df: pd.DataFrame,
    *,
    schedule_mode: str,
    truth_h_values: tuple[float, ...],
    lambda0_values: tuple[float, ...],
    lambda1_values: tuple[float, ...],
) -> None:
    fig, axes = plt.subplots(
        len(truth_h_values), 1, figsize=(7.0, 2.4 * len(truth_h_values)), sharex=True
    )
    if len(truth_h_values) == 1:
        axes = [axes]

    for ax, truth_h in zip(axes, truth_h_values, strict=False):
        controller_subset = row_df[
            (row_df["schedule_mode"] == schedule_mode)
            & (row_df["truth_h"] == truth_h)
            & (row_df["policy"] == "controller")
        ]
        detector_subset = row_df[
            (row_df["schedule_mode"] == schedule_mode)
            & (row_df["truth_h"] == truth_h)
            & (row_df["policy"] == "detector_only")
        ]
        controller = controller_subset.pivot_table(
            index="lambda_0",
            columns="lambda_1",
            values="mean_total_cost",
            aggfunc="mean",
        )
        detector = detector_subset.pivot_table(
            index="lambda_0",
            columns="lambda_1",
            values="mean_total_cost",
            aggfunc="mean",
        )
        diff = controller.to_numpy(dtype=float) - detector.to_numpy(dtype=float)
        im = ax.imshow(diff, origin="lower", cmap="coolwarm")
        ax.set_title(f"{schedule_mode} | H={truth_h} | controller - detector_only")
        ax.set_xticks(range(len(lambda1_values)))
        ax.set_xticklabels([str(v) for v in lambda1_values])
        ax.set_yticks(range(len(lambda0_values)))
        ax.set_yticklabels([str(v) for v in lambda0_values])
        ax.set_xlabel("lambda_1")
        ax.set_ylabel("lambda_0")

    fig.colorbar(
        im, ax=axes, fraction=0.02, pad=0.02, label="controller - detector_only"
    )
    fig.tight_layout()
    fig.savefig(
        FIGURE_ROOT / f"phase_controller_advantage_{schedule_mode}.png", dpi=220
    )
    plt.close(fig)


def main() -> None:
    lambda0_values = _parse_float_list("PHASEMAP_LAMBDA0", LAMBDA0_VALUES)
    lambda1_values = _parse_float_list("PHASEMAP_LAMBDA1", LAMBDA1_VALUES)
    truth_h_values = _parse_float_list("PHASEMAP_H", TRUTH_H_VALUES)
    repetitions = _parse_int("PHASEMAP_REPETITIONS", 6)
    bootstrap_repetitions = _parse_int("PHASEMAP_BOOTSTRAP_REPETITIONS", 6)
    schedule_modes = tuple(
        mode.strip()
        for mode in os.environ.get(
            "PHASEMAP_SCHEDULE_MODES", ",".join(SCHEDULE_MODES)
        ).split(",")
        if mode.strip()
    )

    rows: list[dict[str, object]] = []
    cell_summary: list[dict[str, object]] = []

    for schedule_mode in schedule_modes:
        result = simulate_grid(
            lambda0_values=lambda0_values,
            lambda1_values=lambda1_values,
            truth_h_values=truth_h_values,
            repetitions=repetitions,
            bootstrap_method="wild",
            bootstrap_repetitions=bootstrap_repetitions,
            rng_seed=123,
            validity_slack_fraction=0.25,
            schedule_mode=schedule_mode,
        )
        for row in result["rows"]:
            row = dict(row)
            row["schedule_mode"] = schedule_mode
            rows.append(row)
        for cell in result["cell_summary"]:
            cell = dict(cell)
            cell["schedule_mode"] = schedule_mode
            cell_summary.append(cell)

    row_df = pd.DataFrame(rows)
    cell_df = pd.DataFrame(cell_summary)

    row_df.to_csv(CSV_ROOT / "phase_map_rows.csv", index=False)
    cell_df.to_csv(CSV_ROOT / "phase_map_cells.csv", index=False)

    summary = {
        "schedule_modes": schedule_modes,
        "lambda0_values": lambda0_values,
        "lambda1_values": lambda1_values,
        "truth_h_values": truth_h_values,
        "repetitions": repetitions,
        "bootstrap_repetitions": bootstrap_repetitions,
        "row_count": int(len(row_df)),
        "cell_count": int(len(cell_df)),
        "best_non_oracle_counts": (
            cell_df.groupby(["schedule_mode", "truth_h", "best_non_oracle_policy"])
            .size()
            .reset_index(name="cell_count")
            .to_dict(orient="records")
        ),
    }
    (TABLE_ROOT / "phase_map_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    for schedule_mode in schedule_modes:
        subset = cell_df[cell_df["schedule_mode"] == schedule_mode]
        _plot_winner_heatmaps(
            subset,
            schedule_mode=schedule_mode,
            truth_h_values=truth_h_values,
            lambda0_values=lambda0_values,
            lambda1_values=lambda1_values,
        )
        _plot_controller_advantage(
            row_df,
            schedule_mode=schedule_mode,
            truth_h_values=truth_h_values,
            lambda0_values=lambda0_values,
            lambda1_values=lambda1_values,
        )

    # Paper-facing compact table
    paper_table = (
        cell_df.groupby(["schedule_mode", "truth_h", "best_non_oracle_policy"])
        .size()
        .reset_index(name="cell_count")
        .sort_values(
            ["schedule_mode", "truth_h", "cell_count"], ascending=[True, True, False]
        )
    )
    paper_table.to_csv(TABLE_ROOT / "tab_phase_map_policy_counts.csv", index=False)

    print("Phase-map artifacts written under:")
    print(CSV_ROOT)
    print(FIGURE_ROOT)
    print(TABLE_ROOT)


if __name__ == "__main__":
    main()
