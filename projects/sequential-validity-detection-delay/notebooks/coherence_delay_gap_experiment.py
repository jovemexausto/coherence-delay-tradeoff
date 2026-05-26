# %%
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts"
CSV_ROOT = ARTIFACT_ROOT / "csv" / "coherence_delay_gap_experiment"
FIGURE_ROOT = ARTIFACT_ROOT / "figures" / "coherence_delay_gap_experiment"
TABLE_ROOT = ARTIFACT_ROOT / "tables" / "coherence_delay_gap_experiment"
for root in (CSV_ROOT, FIGURE_ROOT, TABLE_ROOT):
    root.mkdir(parents=True, exist_ok=True)


def simulate_path(
    H: float, zeta: float, t0: int, T: int, sigma: float, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mu = np.zeros(T, dtype=float)
    for t in range(t0, T):
        mu[t] = zeta * max(0.0, float(t - t0)) ** H
    return mu + rng.normal(0.0, sigma, size=T)


def page_hinkley(x: np.ndarray, threshold: float, k: float = 0.0) -> int:
    g = 0.0
    for t, val in enumerate(x):
        g = max(0.0, g + float(val) - k)
        if g > threshold:
            return t
    return len(x)


def calibrate_threshold(
    alpha: float, T: int, reps: int, sigma: float, seed: int
) -> float:
    rng = np.random.default_rng(seed)
    lo, hi = 0.1, 250.0
    for _ in range(22):
        mid = 0.5 * (lo + hi)
        fa = []
        for _ in range(reps):
            x = rng.normal(0.0, sigma, size=T)
            det = page_hinkley(x, mid)
            fa.append(det < T)
        rate = float(np.mean(fa))
        if rate > alpha:
            lo = mid
        else:
            hi = mid
    return float(hi)


@dataclass
class GapRow:
    H: float
    zeta: float
    alpha: float
    threshold: float
    t_valid: int
    mean_tau_detect: float
    mean_delay_gap: float
    median_delay_gap: float
    p_positive_gap: float
    q10_gap: float
    q90_gap: float


def run_experiment() -> list[GapRow]:
    H_values = (0.30, 0.50, 0.75, 1.00)
    zeta_values = (0.002, 0.005, 0.010)
    alpha_values = (0.05, 0.10, 0.20)
    sigma = 1.0
    T = 1500
    t0 = 400
    n_op = 80
    t_valid = t0 + n_op
    rows: list[GapRow] = []

    for alpha in alpha_values:
        thr = calibrate_threshold(alpha=alpha, T=T, reps=200, sigma=sigma, seed=11)
        for H in H_values:
            for zeta in zeta_values:
                gaps = []
                detects = []
                for seed in range(250):
                    x = simulate_path(
                        H=H, zeta=zeta, t0=t0, T=T, sigma=sigma, seed=seed
                    )
                    tau_detect = page_hinkley(x, thr)
                    gap = float(tau_detect - t_valid)
                    gaps.append(gap)
                    detects.append(float(tau_detect))
                gaps_arr = np.asarray(gaps, dtype=float)
                rows.append(
                    GapRow(
                        H=H,
                        zeta=zeta,
                        alpha=alpha,
                        threshold=thr,
                        t_valid=t_valid,
                        mean_tau_detect=float(np.mean(detects)),
                        mean_delay_gap=float(np.mean(gaps_arr)),
                        median_delay_gap=float(np.median(gaps_arr)),
                        p_positive_gap=float(np.mean(gaps_arr > 0.0)),
                        q10_gap=float(np.quantile(gaps_arr, 0.1)),
                        q90_gap=float(np.quantile(gaps_arr, 0.9)),
                    )
                )
    return rows


def plot(rows: list[GapRow]) -> None:
    import pandas as pd

    df = pd.DataFrame([asdict(r) for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)

    for ax, alpha in zip(axes, sorted(df["alpha"].unique()), strict=False):
        sub = df[df["alpha"] == alpha]
        for H in sorted(sub["H"].unique()):
            hs = sub[sub["H"] == H].sort_values("zeta")
            ax.plot(hs["zeta"], hs["mean_delay_gap"], marker="o", label=f"H={H}")
            ax.fill_between(hs["zeta"], hs["q10_gap"], hs["q90_gap"], alpha=0.12)
        ax.axhline(0.0, color="black", lw=1.0, ls="--")
        ax.set_xscale("log")
        ax.set_title(f"alpha={alpha}")
        ax.set_xlabel("zeta")
        ax.set_ylabel("mean delay gap = tau_detect - tau_valid")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(FIGURE_ROOT / "delay_gap_by_H_and_zeta.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for alpha in sorted(df["alpha"].unique()):
        sub = df[df["alpha"] == alpha]
        ax.plot(
            sub.groupby("H")["p_positive_gap"].mean().index,
            sub.groupby("H")["p_positive_gap"].mean().values,
            marker="o",
            label=f"alpha={alpha}",
        )
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("H")
    ax.set_ylabel("P(delay gap > 0)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_ROOT / "positive_gap_probability.png", dpi=200)
    plt.close(fig)


def main() -> None:
    rows = run_experiment()
    import pandas as pd

    df = pd.DataFrame([asdict(r) for r in rows])
    df.to_csv(CSV_ROOT / "coherence_delay_gap_rows.csv", index=False)
    summary = {
        "row_count": int(len(df)),
        "mean_gap_by_alpha": df.groupby("alpha")["mean_delay_gap"].mean().to_dict(),
        "positive_gap_by_alpha": df.groupby("alpha")["p_positive_gap"].mean().to_dict(),
        "max_positive_gap": float(df["mean_delay_gap"].max()),
        "min_positive_gap": float(df["mean_delay_gap"].min()),
    }
    (TABLE_ROOT / "coherence_delay_gap_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    plot(rows)
    print("Coherence-delay gap experiment completed.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
