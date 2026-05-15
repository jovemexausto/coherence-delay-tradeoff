from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

ROW_END = r"\\"


def write_csv(path: Path, rows: list[dict[str, str | float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def table_lines(caption: str, label: str, columns: str, header: str) -> list[str]:
    return [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\small",
        caption,
        label,
        f"\\begin{{tabular}}{{{columns}}}",
        r"\toprule",
        header + ROW_END,
        r"\midrule",
    ]


def calibration_cost_rows() -> list[dict[str, str | float]]:
    comments = {
        0.90: "modest underestimation already moves the horizon",
        0.95: "local error is very small near the optimum",
        1.00: "oracle calibration",
        1.05: "local error is very small near the optimum",
        1.10: "moderate overestimation remains mild",
        1.25: "larger bias becomes visible",
        1.50: "large bias moves the operating point",
    }
    rows: list[dict[str, str | float]] = []
    for r in (0.90, 0.95, 1.00, 1.05, 1.10, 1.25, 1.50):
        delta = r ** (-1.0 / 3.0) + 0.5 * r ** (2.0 / 3.0) - 1.5
        rows.append(
            {
                "ratio": f"{r:.2f}",
                "delta_over_emin": f"{delta:.4f}",
                "comment": comments[round(r, 2)],
            }
        )
    return rows


def calibration_cost_table(rows: list[dict[str, str | float]]) -> str:
    lines = table_lines(
        r"\caption{Local calibration-cost sensitivity for the horizon constant. The excess error is normalized by $E_{\min}$, so the table is scale-free.}",
        r"\label{tab:calibration_cost}",
        "lrp{0.42\\linewidth}",
        r"$r = \widehat C_K/C_K$ & Exact $\Delta E/E_{\min}$ & Comment ",
    )
    for row in rows:
        lines.append(
            f"{row['ratio']} & {row['delta_over_emin']} & {row['comment']} " + ROW_END
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def zeta_robustness_rows() -> list[dict[str, str | float]]:
    alphas = (0.01, 0.02, 0.05, 0.10, 0.20)
    sigma_delta = 0.25
    zeta = 1.0
    ck = 1.0
    burn_in = 5000
    steps = 250000
    rng = np.random.default_rng(123)

    rows: list[dict[str, str | float]] = []
    for alpha in alphas:
        noise = rng.normal(loc=0.0, scale=sigma_delta, size=steps)
        estimate = np.empty(steps, dtype=float)
        estimate[0] = zeta
        for index in range(1, steps):
            delta = zeta + noise[index]
            estimate[index] = alpha * delta + (1.0 - alpha) * estimate[index - 1]
        stationary = estimate[burn_in:]
        theory_var = (alpha / (2.0 - alpha)) * (sigma_delta**2)
        horizon_ratio = np.mean((ck / np.maximum(stationary, 1e-9)) ** (2.0 / 3.0))
        rows.append(
            {
                "alpha": f"{alpha:.2f}" if alpha < 0.1 else f"{alpha:.1f}",
                "mean_zeta_hat": f"{np.mean(stationary):.4f}",
                "var_zeta_hat": f"{np.var(stationary):.5f}",
                "theory_var": f"{theory_var:.5f}",
                "mean_horizon_ratio": f"{horizon_ratio:.4f}",
            }
        )
    return rows


def zeta_robustness_table(rows: list[dict[str, str | float]]) -> str:
    lines = table_lines(
        r"\caption{EMA robustness check for a constant drift proxy with additive noise. The empirical mean matches the target drift, the empirical variance matches the closed-form $\alpha/(2-\alpha)$ law, and the implied horizon distortion stays close to one when the proxy is unbiased.}",
        r"\label{tab:zeta_robustness}",
        "lrrrr",
        r"$\alpha$ & $\mathbb E[\widehat\zeta_t]$ & $\mathrm{Var}(\widehat\zeta_t)$ & Theory & $\mathbb E[\widehat n^*]/n^*$ ",
    )
    for row in rows:
        lines.append(
            f"{row['alpha']} & {row['mean_zeta_hat']} & {row['var_zeta_hat']} & {row['theory_var']} & {row['mean_horizon_ratio']} "
            + ROW_END
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate zeta-support artifacts.")
    parser.add_argument(
        "--csv-dir", type=Path, default=Path("artifacts/csv/calibration")
    )
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=Path("artifacts/tables/calibration"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cost_rows = calibration_cost_rows()
    write_csv(args.csv_dir / "calibration_cost.csv", cost_rows)
    write_text(
        args.tables_dir / "calibration_cost.tex",
        calibration_cost_table(cost_rows),
    )

    robustness_rows = zeta_robustness_rows()
    write_csv(args.csv_dir / "zeta_robustness.csv", robustness_rows)
    write_text(
        args.tables_dir / "zeta_robustness.tex",
        zeta_robustness_table(robustness_rows),
    )


if __name__ == "__main__":
    main()
