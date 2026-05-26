# %%
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts"
CSV_ROOT = ARTIFACT_ROOT / "csv" / "model_free_ucurve_surface_experiment"
FIGURE_ROOT = ARTIFACT_ROOT / "figures" / "model_free_ucurve_surface_experiment"
TABLE_ROOT = ARTIFACT_ROOT / "tables" / "model_free_ucurve_surface_experiment"
for root in (CSV_ROOT, FIGURE_ROOT, TABLE_ROOT):
    root.mkdir(parents=True, exist_ok=True)


def true_n_star(C_K: float, zeta: float, a: float, H: float) -> float:
    return float((C_K / zeta) ** (1.0 / (a + H)))


def base_surface(
    n_grid: np.ndarray, C_K: float, a: float, zeta: float, H: float
) -> np.ndarray:
    return C_K * n_grid ** (-a) + zeta * n_grid**H


def misspec_term(n_grid: np.ndarray, scenario: str) -> np.ndarray:
    x = np.log(n_grid)
    if scenario == "correct":
        return np.zeros_like(n_grid, dtype=float)
    if scenario == "sinusoidal":
        return 0.06 * np.sin(2.2 * x)
    if scenario == "piecewise":
        return 0.10 * (n_grid >= np.median(n_grid)).astype(float)
    if scenario == "mixed":
        return 0.05 * np.sin(2.0 * x) + 0.07 * (n_grid >= np.median(n_grid)).astype(
            float
        )
    raise ValueError(f"unknown scenario: {scenario}")


def fit_parametric_pipeline(
    errors: np.ndarray, n_grid: np.ndarray, C_K: float = 1.0, a: float = 0.5
) -> float | None:
    valid = np.isfinite(errors) & (errors > 0)
    if valid.sum() < 4:
        return None
    x = np.log(n_grid[valid])
    y = np.log(errors[valid])
    slope, intercept = np.polyfit(x, y, 1)
    H_hat = float(np.clip(slope, 0.05, 1.5))
    zeta_hat = float(np.exp(intercept))
    return true_n_star(C_K, zeta_hat, a, H_hat)


def smooth(arr: np.ndarray, width: int = 3) -> np.ndarray:
    if width <= 1:
        return arr.copy()
    kernel = np.ones(width, dtype=float) / float(width)
    return np.convolve(arr, kernel, mode="same")


def argmin_n(errors: np.ndarray, n_grid: np.ndarray) -> int:
    return int(n_grid[np.nanargmin(errors)])


@dataclass
class SurfaceRow:
    scenario: str
    seed: int
    n_star_true: float
    n_hat_ucurve: float
    n_hat_param: float
    abs_err_ucurve: float
    abs_err_param: float
    rel_err_ucurve: float
    rel_err_param: float
    coverage_ucurve: float
    coverage_param: float


def simulate_surface(
    scenario: str, seed: int, n_grid: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    C_K, a, zeta, H = 1.0, 0.5, 0.01, 0.75
    if scenario == "piecewise":
        H = 0.60
        zeta = 0.02
    if scenario == "mixed":
        H = 0.70
        zeta = 0.015
    if scenario == "sinusoidal":
        H = 0.75
        zeta = 0.01
    noiseless = base_surface(n_grid, C_K, a, zeta, H) + misspec_term(n_grid, scenario)
    noise_scale = 0.015 if scenario == "correct" else 0.02
    observed = noiseless + rng.normal(0.0, noise_scale, size=n_grid.size)
    se = np.full_like(observed, noise_scale, dtype=float)
    return noiseless, observed, se


def bootstrap_minimizer_ci(
    errors: np.ndarray,
    se: np.ndarray,
    n_grid: np.ndarray,
    reps: int = 500,
    confidence: float = 0.9,
) -> tuple[int, int]:
    valid = np.isfinite(errors)
    if valid.sum() < 4:
        return int(n_grid[0]), int(n_grid[-1])
    mins = []
    for _ in range(reps):
        perturbed = errors + np.random.normal(0.0, se, size=errors.shape)
        mins.append(argmin_n(perturbed, n_grid))
    lo = np.quantile(mins, (1.0 - confidence) / 2.0)
    hi = np.quantile(mins, 1.0 - (1.0 - confidence) / 2.0)
    return int(lo), int(hi)


def run_experiment() -> tuple[
    np.ndarray,
    list[SurfaceRow],
    dict[str, dict[str, float]],
    dict[str, dict[str, np.ndarray]],
]:
    n_grid = np.unique(np.round(np.geomspace(5, 300, 40)).astype(int))
    scenarios = ["correct", "sinusoidal", "piecewise", "mixed"]
    rows: list[SurfaceRow] = []
    summary: dict[str, dict[str, float]] = {}
    example_surfaces: dict[str, dict[str, np.ndarray]] = {}

    for scenario in scenarios:
        rel_u, rel_p, cov_u, cov_p = [], [], [], []
        example_surfaces[scenario] = {}
        for seed in range(100):
            noiseless, observed, se = simulate_surface(scenario, seed, n_grid)
            n_star_true = argmin_n(noiseless, n_grid)
            n_hat_u = argmin_n(smooth(observed, 3), n_grid)
            n_hat_p = fit_parametric_pipeline(observed, n_grid, C_K=1.0, a=0.5)
            if n_hat_p is None:
                continue
            ci_lo, ci_hi = bootstrap_minimizer_ci(
                observed, se, n_grid, reps=200, confidence=0.9
            )
            rows.append(
                SurfaceRow(
                    scenario=scenario,
                    seed=seed,
                    n_star_true=float(n_star_true),
                    n_hat_ucurve=float(n_hat_u),
                    n_hat_param=float(n_hat_p),
                    abs_err_ucurve=float(abs(n_hat_u - n_star_true)),
                    abs_err_param=float(abs(n_hat_p - n_star_true)),
                    rel_err_ucurve=float(abs(n_hat_u - n_star_true) / n_star_true),
                    rel_err_param=float(abs(n_hat_p - n_star_true) / n_star_true),
                    coverage_ucurve=float(ci_lo <= n_star_true <= ci_hi),
                    coverage_param=float(ci_lo <= n_star_true <= ci_hi),
                )
            )
            rel_u.append(abs(n_hat_u - n_star_true) / n_star_true)
            rel_p.append(abs(n_hat_p - n_star_true) / n_star_true)
            cov_u.append(ci_lo <= n_star_true <= ci_hi)
            cov_p.append(ci_lo <= n_star_true <= ci_hi)
            if seed == 0:
                example_surfaces[scenario]["noiseless"] = noiseless
                example_surfaces[scenario]["observed"] = observed
                example_surfaces[scenario]["se"] = se
                example_surfaces[scenario]["n_star_true"] = np.array([n_star_true])
                example_surfaces[scenario]["n_hat_u"] = np.array([n_hat_u])
                example_surfaces[scenario]["n_hat_p"] = np.array([n_hat_p])
                example_surfaces[scenario]["ci_lo"] = np.array([ci_lo])
                example_surfaces[scenario]["ci_hi"] = np.array([ci_hi])

        summary[scenario] = {
            "count": float(len(rel_u)),
            "mean_rel_err_ucurve": float(np.mean(rel_u)),
            "mean_rel_err_param": float(np.mean(rel_p)),
            "ucurve_better_fraction": float(np.mean(np.array(rel_u) < np.array(rel_p))),
            "ci_coverage": float(np.mean(cov_u)),
        }
    return n_grid, rows, summary, example_surfaces


def plot(
    n_grid: np.ndarray,
    rows: list[SurfaceRow],
    examples: dict[str, dict[str, np.ndarray]],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes = axes.ravel()
    scenarios = ["correct", "sinusoidal", "piecewise", "mixed"]
    for ax, scenario in zip(axes, scenarios, strict=False):
        ex = examples[scenario]
        if not ex:
            continue
        ax.plot(
            n_grid, ex["noiseless"], color="#39d0d8", lw=2.0, label="noiseless surface"
        )
        ax.plot(
            n_grid,
            ex["observed"],
            color="#ff6b6b",
            lw=1.2,
            alpha=0.8,
            label="observed surface",
        )
        ax.axvline(
            float(ex["n_star_true"][0]),
            color="#f0b429",
            lw=1.8,
            ls="--",
            label="true argmin",
        )
        ax.axvline(
            float(ex["n_hat_u"][0]),
            color="#56e39f",
            lw=1.8,
            ls=":",
            label="U-curve argmin",
        )
        ax.axvline(
            float(ex["n_hat_p"][0]),
            color="#b48ead",
            lw=1.8,
            ls="-.",
            label="parametric estimate",
        )
        ax.set_title(scenario)
        ax.set_xlabel("window size n")
        ax.set_ylabel("error surface")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURE_ROOT / "surface_examples.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    scenarios = ["correct", "sinusoidal", "piecewise", "mixed"]
    u = [
        np.mean([r.rel_err_ucurve for r in rows if r.scenario == s]) for s in scenarios
    ]
    p = [np.mean([r.rel_err_param for r in rows if r.scenario == s]) for s in scenarios]
    x = np.arange(len(scenarios))
    ax.bar(x - 0.18, u, width=0.36, color="#39d0d8", label="U-curve")
    ax.bar(x + 0.18, p, width=0.36, color="#ff6b6b", label="Parametric")
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.set_ylabel("mean relative error")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_ROOT / "relative_error_comparison.png", dpi=200)
    plt.close(fig)


def main() -> None:
    n_grid, rows, summary, examples = run_experiment()
    df_rows = [asdict(r) for r in rows]
    (TABLE_ROOT / "ucurve_surface_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (TABLE_ROOT / "ucurve_surface_rows.json").write_text(
        json.dumps(df_rows, indent=2), encoding="utf-8"
    )

    import pandas as pd

    pd.DataFrame(df_rows).to_csv(CSV_ROOT / "ucurve_surface_rows.csv", index=False)
    plot(n_grid, rows, examples)

    print("Surface experiment completed.")
    for scenario, stats in summary.items():
        print(scenario, stats)


if __name__ == "__main__":
    main()
