from __future__ import annotations

from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np

from .model import KuaiRandBenchmarkResult


def save_kuairand_figure(result: KuaiRandBenchmarkResult, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    max_len = max(len(user.signals) for user in result.user_signals)
    grid = np.full((len(result.user_signals), max_len), np.nan)
    grid_tcie = np.full((len(result.user_signals), max_len), np.nan)
    grid_tcie_ewma = np.full((len(result.user_signals), max_len), np.nan)
    for row, user in enumerate(result.user_signals):
        tci = cast(np.ndarray, user.signals["tci"].to_numpy())
        tcie = cast(np.ndarray, user.signals["tcie"].to_numpy())
        tcie_ewma = cast(np.ndarray, user.signals["tcie_ewma"].to_numpy())
        grid[row, : len(tci)] = tci
        grid_tcie[row, : len(tcie)] = tcie
        grid_tcie_ewma[row, : len(tcie_ewma)] = tcie_ewma

    med_tci = np.nanmedian(grid, axis=0)
    med_tcie = np.nanmedian(grid_tcie, axis=0)
    med_tcie_ewma = np.nanmedian(grid_tcie_ewma, axis=0)
    phases = [
        np.nanmedian([user.random_end for user in result.user_signals]),
        np.nanmedian([user.coercive_end for user in result.user_signals]),
    ]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(med_tci, label="CI", linewidth=1.5)
    ax.plot(med_tcie, label="CI^E", linewidth=1.5)
    ax.plot(med_tcie_ewma, label="CI^E-EWMA", linewidth=1.2, linestyle="--")
    for boundary in phases:
        ax.axvline(boundary, color="0.4", linestyle="--", linewidth=1.0)
    ax.set_ylabel("Median CI")
    ax.set_xlabel("Time step")
    ax.set_title("KuaiRand logged benchmark: median CI trajectories")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.2, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(output_path)
    fig.savefig(output_path.with_suffix(".png"), dpi=180)
    plt.close(fig)
