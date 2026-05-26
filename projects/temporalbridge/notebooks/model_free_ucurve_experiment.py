# %%
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts"
CSV_ROOT = ARTIFACT_ROOT / "csv" / "model_free_ucurve_experiment"
FIGURE_ROOT = ARTIFACT_ROOT / "figures" / "model_free_ucurve_experiment"
TABLE_ROOT = ARTIFACT_ROOT / "tables" / "model_free_ucurve_experiment"
for root in (CSV_ROOT, FIGURE_ROOT, TABLE_ROOT):
    root.mkdir(parents=True, exist_ok=True)


def true_n_star(C_K: float, zeta: float, a: float, H: float) -> float:
    return float((C_K / zeta) ** (1.0 / (a + H)))


def tracking_error_w1(window: np.ndarray, anchor: np.ndarray) -> float:
    n = max(len(window), len(anchor))
    if n < 2:
        return float("nan")
    u = np.linspace(0.0, 1.0, n + 1)[1:-1]
    qw = np.quantile(window, u)
    qa = np.quantile(anchor, u)
    return float(np.mean(np.abs(qw - qa)))


def compute_ucurve(arr: np.ndarray, n_grid: np.ndarray, anchor_size: int) -> np.ndarray:
    anchor = arr[-anchor_size:]
    errors = []
    end = len(arr) - anchor_size
    for n in n_grid:
        start = end - int(n)
        if start < 0:
            errors.append(np.nan)
            continue
        errors.append(tracking_error_w1(arr[start:end], anchor))
    return np.asarray(errors, dtype=float)


def minimize_ucurve(errors: np.ndarray, n_grid: np.ndarray) -> int | None:
    valid = ~np.isnan(errors)
    if valid.sum() < 4:
        return None
    idx = np.where(valid)[0]
    return int(n_grid[idx[np.nanargmin(errors[idx])]])


def smooth_errors(errors: np.ndarray, width: int = 3) -> np.ndarray:
    out = errors.copy()
    valid = ~np.isnan(out)
    if valid.sum() < width:
        return out
    vals = out[valid]
    kernel = np.ones(width, dtype=float) / float(width)
    smoothed = np.convolve(vals, kernel, mode="same")
    out[valid] = smoothed
    return out


def fit_parametric_pipeline(
    errors: np.ndarray, n_grid: np.ndarray, C_K: float = 1.0, a: float = 0.5
) -> float | None:
    valid = ~np.isnan(errors) & (errors > 0)
    if valid.sum() < 4:
        return None
    x = np.log(n_grid[valid])
    y = np.log(errors[valid])
    slope, intercept = np.polyfit(x, y, 1)
    H_hat = float(np.clip(slope, 0.05, 1.5))
    zeta_hat = float(np.exp(intercept))
    return true_n_star(C_K, zeta_hat, a, H_hat)


@dataclass
class ExperimentRow:
    scenario: str
    seed: int
    n_star_true: float
    n_hat_ucurve: float | None
    n_hat_param: float | None
    abs_err_ucurve: float | None
    abs_err_param: float | None
    rel_err_ucurve: float | None
    rel_err_param: float | None


def make_stream(
    scenario: str, seed: int, n_total: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    sigma = 1.0
    mu = np.zeros(n_total, dtype=float)
    if scenario == "correct":
        H, zeta, t0 = 0.75, 0.008, 400
        for t in range(n_total):
            mu[t] = zeta * (max(0, t - t0) ** H) if t > t0 else 0.0
    elif scenario == "sinusoidal":
        for t in range(n_total):
            mu[t] = 0.22 * np.sin(2 * np.pi * t / 220.0) + 0.003 * t if t > 400 else 0.0
    elif scenario == "piecewise":
        for t in range(n_total):
            if t <= 400:
                mu[t] = 0.0
            else:
                phase = ((t - 400) // 300) % 2
                local = (t - 400) % 300
                mu[t] = (
                    0.55 * local / 300.0 if phase == 0 else 0.55 - 0.55 * local / 300.0
                )
    elif scenario == "mixed":
        for t in range(n_total):
            base = 0.005 * max(0, t - 350)
            wobble = 0.12 * np.sin(2 * np.pi * t / 260.0)
            mu[t] = base + wobble if t > 350 else 0.0
    else:
        raise ValueError(f"unknown scenario: {scenario}")
    y = mu + rng.normal(0.0, sigma, size=n_total)
    return y, mu


def run_snapshot_experiment():
    n_grid = np.unique(np.round(np.geomspace(5, 300, 40)).astype(int))
    anchor_size = 30
    n_total = 4000
    C_K, a = 1.0, 0.5
    scenarios = ["correct", "sinusoidal", "piecewise", "mixed"]
    rows: list[ExperimentRow] = []

    for scenario in scenarios:
        for seed in range(20):
            y, mu = make_stream(scenario, seed, n_total)
            t_eval = 3500
            obs = y[:t_eval]
            truth = mu[:t_eval]
            errors_obs = compute_ucurve(obs, n_grid, anchor_size)
            errors_true = compute_ucurve(truth, n_grid, anchor_size)
            n_hat_ucurve = minimize_ucurve(smooth_errors(errors_obs, 3), n_grid)
            n_hat_param = fit_parametric_pipeline(errors_obs, n_grid, C_K=C_K, a=a)
            n_star = minimize_ucurve(errors_true, n_grid)
            if n_star is None:
                continue
            rows.append(
                ExperimentRow(
                    scenario=scenario,
                    seed=seed,
                    n_star_true=float(n_star),
                    n_hat_ucurve=float(n_hat_ucurve)
                    if n_hat_ucurve is not None
                    else None,
                    n_hat_param=float(n_hat_param) if n_hat_param is not None else None,
                    abs_err_ucurve=(
                        abs(n_hat_ucurve - n_star) if n_hat_ucurve is not None else None
                    ),
                    abs_err_param=(
                        abs(n_hat_param - n_star) if n_hat_param is not None else None
                    ),
                    rel_err_ucurve=(
                        abs(n_hat_ucurve - n_star) / n_star
                        if n_hat_ucurve is not None
                        else None
                    ),
                    rel_err_param=(
                        abs(n_hat_param - n_star) / n_star
                        if n_hat_param is not None
                        else None
                    ),
                )
            )

    return n_grid, anchor_size, rows


def summarize(rows: list[ExperimentRow]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for scenario in sorted({r.scenario for r in rows}):
        subset = [r for r in rows if r.scenario == scenario]
        summary[scenario] = {
            "count": float(len(subset)),
            "mean_true_n_star": float(np.mean([r.n_star_true for r in subset])),
            "mean_abs_err_ucurve": float(
                np.mean(
                    [r.abs_err_ucurve for r in subset if r.abs_err_ucurve is not None]
                )
            ),
            "mean_abs_err_param": float(
                np.mean(
                    [r.abs_err_param for r in subset if r.abs_err_param is not None]
                )
            ),
            "mean_rel_err_ucurve": float(
                np.mean(
                    [r.rel_err_ucurve for r in subset if r.rel_err_ucurve is not None]
                )
            ),
            "mean_rel_err_param": float(
                np.mean(
                    [r.rel_err_param for r in subset if r.rel_err_param is not None]
                )
            ),
            "ucurve_better_fraction": float(
                np.mean(
                    [
                        (r.abs_err_ucurve or np.inf) < (r.abs_err_param or np.inf)
                        for r in subset
                        if r.abs_err_ucurve is not None and r.abs_err_param is not None
                    ]
                )
            ),
        }
    return summary


def plot_summary(
    rows: list[ExperimentRow], n_grid: np.ndarray, anchor_size: int
) -> None:
    scenarios = ["correct", "sinusoidal", "piecewise", "mixed"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.ravel()
    colors = {
        "correct": "#39d0d8",
        "sinusoidal": "#ff6b6b",
        "piecewise": "#f0b429",
        "mixed": "#56e39f",
    }

    for ax, scenario in zip(axes, scenarios, strict=False):
        subset = [r for r in rows if r.scenario == scenario]
        xs = np.arange(len(subset))
        u = np.array([r.rel_err_ucurve for r in subset], dtype=float)
        p = np.array([r.rel_err_param for r in subset], dtype=float)
        ax.boxplot(
            [u, p],
            tick_labels=["U-curve", "Parametric"],
            patch_artist=True,
            boxprops=dict(facecolor="#161b22", color="#30363d"),
            medianprops=dict(color="#e6edf3"),
        )
        ax.set_title(scenario)
        ax.set_ylabel("Relative error")
        ax.grid(True, alpha=0.25)

    fig.suptitle("Model-Free U-Curve vs Parametric Horizon Pipeline", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURE_ROOT / "ucurve_vs_parametric_boxplots.png", dpi=200)
    plt.close(fig)


def main() -> None:
    n_grid, anchor_size, rows = run_snapshot_experiment()
    summary = summarize(rows)
    row_dicts = [asdict(r) for r in rows]
    (TABLE_ROOT / "ucurve_experiment_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (TABLE_ROOT / "ucurve_experiment_rows.json").write_text(
        json.dumps(row_dicts, indent=2), encoding="utf-8"
    )

    # minimal CSV without optional None entries
    import pandas as pd

    df = pd.DataFrame(row_dicts)
    df.to_csv(CSV_ROOT / "ucurve_experiment_rows.csv", index=False)

    plot_summary(rows, n_grid, anchor_size)

    print("Model-free horizon experiment completed.")
    for scenario, stats in summary.items():
        print(scenario, stats)


if __name__ == "__main__":
    main()
