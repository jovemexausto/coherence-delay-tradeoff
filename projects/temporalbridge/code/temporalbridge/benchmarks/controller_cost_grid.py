from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

from temporalbridge.benchmarks.controller_sequential import (
    run_controller_sequential_benchmark,
)


@dataclass(frozen=True)
class ControllerCostGridRow:
    truth_h: float
    lambda_0: float
    lambda_1: float
    policy: str
    mean_cumulative_validity_loss: float
    mean_cumulative_update_cost: float
    mean_total_cost: float
    mean_regret: float
    mean_lead_time: float
    mean_log_memory_std: float
    mean_tau_valid: float
    mean_tau_detect: float
    mean_delay_gap: float
    mean_masking_index: float


def simulate_grid(
    *,
    lambda0_values: Iterable[float],
    lambda1_values: Iterable[float],
    repetitions: int = 12,
    bootstrap_method: str = "wild",
    bootstrap_repetitions: int = 12,
    rng_seed: int = 123,
    validity_slack_fraction: float = 0.25,
    schedule_mode: str = "default",
    truth_h_values: Iterable[float] = (0.6,),
) -> dict[str, object]:
    rows: list[ControllerCostGridRow] = []
    cell_summary: list[dict[str, object]] = []
    for truth_h in truth_h_values:
        for lambda_0 in lambda0_values:
            for lambda_1 in lambda1_values:
                result = run_controller_sequential_benchmark(
                    repetitions=repetitions,
                    bootstrap_method=bootstrap_method,
                    bootstrap_repetitions=bootstrap_repetitions,
                    rng_seed=rng_seed,
                    update_cost_fixed=lambda_0,
                    update_cost_linear=lambda_1,
                    validity_slack_fraction=validity_slack_fraction,
                    schedule_mode=schedule_mode,
                    truth_h=truth_h,
                )
                policy_rows = result["rows"]
                best_row = min(
                    policy_rows,
                    key=lambda row: (
                        row["mean_cumulative_validity_loss"]
                        + row["mean_cumulative_update_cost"]
                    ),
                )
                non_oracle_rows = [
                    row for row in policy_rows if row["policy"] != "oracle"
                ]
                best_non_oracle_row = min(
                    non_oracle_rows,
                    key=lambda row: (
                        row["mean_cumulative_validity_loss"]
                        + row["mean_cumulative_update_cost"]
                    ),
                )
                cell_summary.append(
                    {
                        "truth_h": float(truth_h),
                        "lambda_0": float(lambda_0),
                        "lambda_1": float(lambda_1),
                        "best_policy": best_row["policy"],
                        "best_total_cost": float(
                            best_row["mean_cumulative_validity_loss"]
                            + best_row["mean_cumulative_update_cost"]
                        ),
                        "best_non_oracle_policy": best_non_oracle_row["policy"],
                        "best_non_oracle_total_cost": float(
                            best_non_oracle_row["mean_cumulative_validity_loss"]
                            + best_non_oracle_row["mean_cumulative_update_cost"]
                        ),
                    }
                )
                for row in policy_rows:
                    rows.append(
                        ControllerCostGridRow(
                            truth_h=float(truth_h),
                            lambda_0=float(lambda_0),
                            lambda_1=float(lambda_1),
                            policy=str(row["policy"]),
                            mean_cumulative_validity_loss=float(
                                row["mean_cumulative_validity_loss"]
                            ),
                            mean_cumulative_update_cost=float(
                                row["mean_cumulative_update_cost"]
                            ),
                            mean_total_cost=float(
                                row["mean_cumulative_validity_loss"]
                                + row["mean_cumulative_update_cost"]
                            ),
                            mean_regret=float(row["mean_regret"]),
                            mean_lead_time=float(row["mean_lead_time"]),
                            mean_log_memory_std=float(row["mean_log_memory_std"]),
                            mean_tau_valid=float(row["mean_tau_valid"]),
                            mean_tau_detect=float(row["mean_tau_detect"]),
                            mean_delay_gap=float(row["mean_delay_gap"]),
                            mean_masking_index=float(row["mean_masking_index"]),
                        )
                    )
    return {
        "rows": [asdict(row) for row in rows],
        "cell_summary": cell_summary,
    }
