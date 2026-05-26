# %%
from __future__ import annotations

from pathlib import Path

import json

import matplotlib.pyplot as plt
import pandas as pd

from temporalbridge.benchmarks import (
    run_controller_grid_benchmark,
    run_controller_monte_carlo,
    run_controller_sequential_benchmark,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts"
CSV_ROOT = ARTIFACT_ROOT / "csv" / "controller_analysis"
FIGURE_ROOT = ARTIFACT_ROOT / "figures" / "controller_analysis"
TABLE_ROOT = ARTIFACT_ROOT / "tables" / "controller_analysis"
for root in (CSV_ROOT, FIGURE_ROOT, TABLE_ROOT):
    root.mkdir(parents=True, exist_ok=True)


# %%
# Benchmark execution

grid_result = run_controller_grid_benchmark(rng_seed=123, bootstrap_method="wild")
mc_result = run_controller_monte_carlo(
    repetitions=50,
    bootstrap_method="wild",
    bootstrap_repetitions=50,
    rng_seed=123,
)
sequential_result = run_controller_sequential_benchmark(
    repetitions=100,
    bootstrap_method="wild",
    bootstrap_repetitions=20,
    rng_seed=123,
)


# %%
# DataFrames and artifact export

grid_df = pd.DataFrame(grid_result["rows"])
mc_df = pd.DataFrame(mc_result["rows"])
mc_agg_df = pd.DataFrame(mc_result["aggregated"])
seq_df = pd.DataFrame(sequential_result["rows"])
seq_traj_df = pd.DataFrame(sequential_result["trajectory_rows"])

grid_df.to_csv(CSV_ROOT / "controller_grid.csv", index=False)
mc_df.to_csv(CSV_ROOT / "controller_monte_carlo_rows.csv", index=False)
mc_agg_df.to_csv(CSV_ROOT / "controller_monte_carlo_aggregated.csv", index=False)
seq_df.to_csv(CSV_ROOT / "controller_sequential_rows.csv", index=False)
seq_traj_df.to_csv(CSV_ROOT / "controller_sequential_trajectory_rows.csv", index=False)

summary = {
    "grid": grid_result,
    "monte_carlo": mc_result,
    "sequential": sequential_result,
}
(TABLE_ROOT / "controller_analysis_summary.json").write_text(
    json.dumps(summary, indent=2), encoding="utf-8"
)


# %%
# Table surfaces to inspect directly in notebook mode

print("Sequential aggregated rows")
print(seq_df)

print("\nMonte Carlo aggregated rows")
print(mc_agg_df)


# %%
# Figure 1: sequential regret / excess validity loss by policy

plot_df = seq_df.sort_values("mean_regret")
fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))

axes[0].bar(plot_df["policy"], plot_df["mean_regret"])
axes[0].set_title("Mean Regret by Policy")
axes[0].set_ylabel("Mean Regret")
axes[0].tick_params(axis="x", rotation=30)

axes[1].bar(plot_df["policy"], plot_df["mean_cumulative_excess_validity_loss"])
axes[1].set_title("Mean Excess Validity Loss")
axes[1].set_ylabel("Excess Loss")
axes[1].tick_params(axis="x", rotation=30)

fig.tight_layout()
fig.savefig(FIGURE_ROOT / "fig_controller_regret_and_excess_loss.png", dpi=200)
plt.close(fig)


# %%
# Figure 2: lead time and false alarm summary

fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))

axes[0].bar(seq_df["policy"], seq_df["mean_lead_time"])
axes[0].axhline(0.0, color="black", linewidth=1.0)
axes[0].set_title("Mean Lead Time")
axes[0].set_ylabel("tau_valid - tau_detect")
axes[0].tick_params(axis="x", rotation=30)

axes[1].bar(seq_df["policy"], seq_df["mean_false_alarm_rate"])
axes[1].set_title("Mean False Alarm Rate")
axes[1].set_ylabel("False Alarm Rate")
axes[1].tick_params(axis="x", rotation=30)

fig.tight_layout()
fig.savefig(FIGURE_ROOT / "fig_controller_lead_time_false_alarm.png", dpi=200)
plt.close(fig)


# %%
# Figure 3: update counts versus regret

fig, ax = plt.subplots(figsize=(5.5, 4.5))
for _, row in seq_df.iterrows():
    ax.scatter(row["mean_update_count"], row["mean_regret"], s=60)
    ax.annotate(row["policy"], (row["mean_update_count"], row["mean_regret"]))

ax.set_xlabel("Mean Update Count")
ax.set_ylabel("Mean Regret")
ax.set_title("Update Count vs Regret")
fig.tight_layout()
fig.savefig(FIGURE_ROOT / "fig_controller_update_count_vs_regret.png", dpi=200)
plt.close(fig)


# %%
# Figure 4: trajectory-level regret distribution

fig, ax = plt.subplots(figsize=(8.0, 4.5))
traj_plot_df = seq_traj_df[seq_traj_df["policy"] != "oracle"].copy()
policy_order = [
    "controller",
    "detector_only",
    "deploy_only",
    "fixed_policy",
]
box_data = [
    traj_plot_df.loc[traj_plot_df["policy"] == policy, "regret"].to_numpy()
    for policy in policy_order
]
ax.boxplot(box_data, tick_labels=policy_order)
ax.set_title("Trajectory-Level Regret Distribution")
ax.set_ylabel("Regret")
ax.tick_params(axis="x", rotation=20)
fig.tight_layout()
fig.savefig(FIGURE_ROOT / "fig_controller_trajectory_regret_boxplot.png", dpi=200)
plt.close(fig)


# %%
# Expected paper-facing tables

paper_table = seq_df[
    [
        "policy",
        "mean_action_accuracy",
        "mean_false_alarm_rate",
        "mean_lead_time",
        "mean_cumulative_validity_loss",
        "mean_cumulative_excess_validity_loss",
        "mean_regret",
        "mean_update_count",
    ]
].sort_values("mean_regret")
paper_table.to_csv(TABLE_ROOT / "tab_controller_sequential_summary.csv", index=False)
print("\nPaper-facing sequential summary")
print(paper_table)


# %%
# Ablation placeholders

# Suggested next cells:
# 1. Compare sequential benchmark under bootstrap_method in {wild, moving_block}
# 2. Sweep MemoryDynamicsParams.update_cost_linear and tracking_gain
# 3. Compare controller trajectory regret quantiles across schedules

print("Artifacts written under:")
print(CSV_ROOT)
print(FIGURE_ROOT)
print(TABLE_ROOT)
