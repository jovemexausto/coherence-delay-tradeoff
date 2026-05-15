from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.cuberoot_adwin.model import UMRBenchmarkConfig, run_benchmark

ROW_END = r"\\"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def table_lines(caption: str, label: str, columns: str, header: str) -> list[str]:
    return [
        r"\begin{table}[!htbp]",
        r"\centering",
        caption,
        label,
        f"\\begin{{tabular}}{{{columns}}}",
        r"\toprule",
        header + ROW_END,
        r"\midrule",
    ]


def continuous_benchmark_table(rows: list[dict[str, str]]) -> str:
    display_labels = {
        "fixed-100": "Fixed-100",
        "fixed-500": "Fixed-500",
        "EWMA": "EWMA",
        "ADWIN": "ADWIN",
        "ADWIN + UMR": "ADWIN + UMR",
    }
    lines = table_lines(
        r"\caption{Continuous-drift benchmark. The key gap is between statistical evidence and temporal validity: ADWIN keeps a much wider window than the UMR-capped horizon scale, while the long fixed window collapses under lag.}",
        r"\label{tab:continuous_benchmark}",
        "lrrrr",
        r"Method & Tail MAE & Mean horizon & Detector events & Cap events ",
    )
    for row in rows:
        method = display_labels.get(row["method"], row["method"])
        detector = (
            "--"
            if method in {"Fixed-100", "Fixed-500", "EWMA"}
            else row["event_count_mean"]
        )
        caps = (
            "--"
            if method in {"Fixed-100", "Fixed-500", "EWMA"}
            else (row["cap_count_mean"] or "--")
        )
        lines.append(
            f"{method} & {float(row['tail_mae_mean']):.4f} & {float(row['tail_width_mean']):.1f} & {detector} & {caps} "
            + ROW_END
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def bootstrap_table() -> str:
    config = UMRBenchmarkConfig(
        seeds=tuple(range(20)),
        drift=0.001,
        fixed_window=100,
        fixed_long_window=500,
        ewma_alpha=0.05,
        adwin_delta=0.002,
        Ck=1.0,
        drift_window=100,
    )
    result = run_benchmark(config)
    method_map = {
        "Fixed-100": "fixed_error",
        "Fixed-500": "fixed_long_error",
        "EWMA": "ewma_error",
        "ADWIN": "adwin_error",
        "ADWIN + UMR": "cube_error",
    }
    tail_start = int(config.steps * (1.0 - config.tail_fraction))
    per_method: dict[str, np.ndarray] = {}
    for label, attr in method_map.items():
        values = [
            float(np.mean(getattr(trace, attr)[tail_start:])) for trace in result.traces # type: ignore
        ]
        per_method[label] = np.asarray(values, dtype=float)

    cube = per_method["ADWIN + UMR"]
    rng = np.random.default_rng(17)
    bootstrap_samples = 5000

    lines = table_lines(
        r"\caption{Seed-wise bootstrap summary for the main synthetic benchmark. The first column block reports tail-MAE means with 95\% bootstrap intervals across the 20 seeds; the second block reports paired differences versus ADWIN+$\,$UMR.}",
        r"\label{tab:bootstrap_tail_mae}",
        "lrr",
        r"Method & Tail MAE [95\% CI] & $\Delta$ vs ADWIN+$\,$UMR [95\% CI] ",
    )
    for label, values in per_method.items():
        bootstrap_means = np.mean(
            rng.choice(values, size=(bootstrap_samples, values.size), replace=True),
            axis=1,
        )
        mean = float(np.mean(values))
        low, high = np.percentile(bootstrap_means, [2.5, 97.5])
        if label == "ADWIN + UMR":
            delta_text = "--"
        else:
            diff = values - cube
            bootstrap_diff = np.mean(
                rng.choice(diff, size=(bootstrap_samples, diff.size), replace=True),
                axis=1,
            )
            diff_mean = float(np.mean(diff))
            diff_low, diff_high = np.percentile(bootstrap_diff, [2.5, 97.5])
            delta_text = f"{diff_mean:+.4f} [{diff_low:+.4f}, {diff_high:+.4f}]"
        lines.append(
            f"{label} & {mean:.4f} [{low:.4f}, {high:.4f}] & {delta_text} " + ROW_END
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def hysteresis_transition_table(rows: list[dict[str, str]]) -> str:
    labels = {
        "fixed_100": "fixed-100",
        "fixed_200": "fixed-200",
        "adwin": "ADWIN",
        "cube": "ADWIN + UMR",
    }
    grouped: dict[str, dict[str, list[float]]] = {
        label: {"contraction": [], "expansion": []} for label in labels.values()
    }
    for row in rows:
        label = labels[row["method"]]
        grouped[label][row["transition_type"]].append(float(row["window_regret_mean"]))

    lines = table_lines(
        r"\caption{Mean horizon regret over the first 50 steps after each regime change. Contraction averages over the two fast-drift transitions, and expansion averages over the two slow-drift recoveries. Lower is better.}",
        r"\label{tab:hysteresis_transition}",
        "lrr",
        r"Method & Contraction & Expansion ",
    )
    for label in ("fixed-100", "fixed-200", "ADWIN", "ADWIN + UMR"):
        contraction = np.mean(grouped[label]["contraction"])
        expansion = np.mean(grouped[label]["expansion"])
        lines.append(f"{label} & {contraction:.1f} & {expansion:.1f} " + ROW_END)
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def zeta_sweep_table(rows: list[dict[str, str]]) -> str:
    lines = table_lines(
        r"\caption{Drift-EMA sensitivity on the alternating-timescales benchmark. The ratio remains above one across the sweep, so re-expansion is slower than contraction for every tested $\alpha$.}",
        r"\label{tab:zeta_sweep}",
        "lrrrr",
        r"$\alpha$ & Contraction & Expansion & Ratio & Tail MAE ",
    )
    for row in rows:
        lines.append(
            f"{row['drift_ema_alpha']} & {float(row['contraction_regret_mean']):.2f} & {float(row['expansion_regret_mean']):.2f} & {float(row['expansion_to_contraction_ratio']):.2f} & {float(row['tail_mae_mean']):.4f} "
            + ROW_END
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def piecewise_oracle_table(rows: list[dict[str, str]]) -> str:
    lines = table_lines(
        r"\caption{Oracle horizon recovery on piecewise drift. The same calibrated cap tracks the oracle scale across regimes: it is close in the fast-drift phase and remains within the right order of magnitude in the smoother phases.}",
        r"\label{tab:piecewise_oracle}",
        "lrrrr",
        r"Drift & Oracle window & Regulated $n^*$ & Oracle MAE & ADWIN + UMR MAE ",
    )
    for row in rows:
        lines.append(
            f"${float(row['drift']):.4f}$ & {int(float(row['oracle_window']))} & {float(row['cube_mean_n_star']):.1f} & {float(row['oracle_mae']):.4f} & {float(row['cube_mae']):.4f} "
            + ROW_END
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def real_world_umr_table(rows: list[dict[str, str]]) -> str:
    labels = {
        ("ewma", "base"): "EWMA",
        ("ewma", "umr"): "EWMA + UMR",
        ("window_dilemma", "base"): "Window Dilemma",
        ("window_dilemma", "umr"): "Window Dilemma + UMR",
        ("melo", "base"): "MELO-style",
        ("melo", "umr"): "MELO-style + UMR",
        ("adwin", "base"): "ADWIN",
        ("adwin", "umr"): "ADWIN + UMR",
    }
    order = [
        ("ewma", "base"),
        ("ewma", "umr"),
        ("window_dilemma", "base"),
        ("window_dilemma", "umr"),
        ("melo", "base"),
        ("melo", "umr"),
        ("adwin", "base"),
        ("adwin", "umr"),
    ]
    indexed = {(row["baseline"], row["condition"]): row for row in rows}

    lines = table_lines(
        r"\caption{Bikes public-stream summary. Precision is matched-warning precision against Page-Hinkley events; median lead is the median warning lead in steps.}",
        r"\label{tab:real_world_umr}",
        "lrrr",
        r"Method & Precision & Median lead & Leads ",
    )
    for key in order:
        row = indexed[key]
        lines.append(
            f"{labels[key]} & {float(row['precision']):.3f} & {float(row['median_lead']):.1f} & {int(float(row['leads']))} "
            + ROW_END
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def temporal_invalidity_table(rows: list[dict[str, str]]) -> str:
    lines = table_lines(
        r"\caption{Controlled downstream benchmark summary. The reported delta is UMR minus Fixed-400 within the cap-only interval, averaged across six seeds.}",
        r"\label{tab:temporal_invalidity}",
        "lrrrr",
        r"Method & Global acc. & Cap-only acc. & $\Delta$ cap-only & Mean cap-only width ",
    )
    for row in rows:
        lines.append(
            f"{row['method']} & {float(row['global_accuracy']):.4f} & {float(row['cap_only_accuracy']):.4f} & {float(row['delta_cap_only_vs_fixed_400_pp']):+.2f} & {float(row['mean_umr_width_cap_only']):.1f} "
            + ROW_END
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Paper 1 LaTeX tables.")
    parser.add_argument("--csv-root", type=Path, default=Path("artifacts/csv"))
    parser.add_argument("--tables-root", type=Path, default=Path("artifacts/tables"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cuberoot_root = args.csv_root / "cuberoot_adwin"

    write_text(
        args.tables_root / "cuberoot_adwin" / "continuous_benchmark.tex",
        continuous_benchmark_table(
            read_csv_rows(cuberoot_root / "cuberoot_adwin_summary.csv")
        ),
    )
    write_text(
        args.tables_root / "cuberoot_adwin" / "bootstrap_tail_mae.tex",
        bootstrap_table(),
    )
    write_text(
        args.tables_root / "cuberoot_adwin" / "hysteresis_transition.tex",
        hysteresis_transition_table(
            read_csv_rows(
                cuberoot_root / "cuberoot_adwin_horizon_transition_ablation.csv"
            )
        ),
    )
    write_text(
        args.tables_root / "cuberoot_adwin" / "zeta_sweep.tex",
        zeta_sweep_table(
            read_csv_rows(cuberoot_root / "cuberoot_adwin_drift_ema_ablation.csv")
        ),
    )
    write_text(
        args.tables_root / "cuberoot_adwin" / "piecewise_oracle.tex",
        piecewise_oracle_table(
            read_csv_rows(cuberoot_root / "cuberoot_adwin_piecewise_oracle_phases.csv")
        ),
    )
    write_text(
        args.tables_root / "bikes" / "real_world_umr.tex",
        real_world_umr_table(
            read_csv_rows(args.csv_root / "bikes" / "bikes_arena_summary.csv")
        ),
    )
    write_text(
        args.tables_root / "temporal_invalidity" / "temporal_invalidity.tex",
        temporal_invalidity_table(
            read_csv_rows(
                args.csv_root
                / "temporal_invalidity"
                / "temporal_invalidity_summary.csv"
            )
        ),
    )


if __name__ == "__main__":
    main()
