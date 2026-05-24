from __future__ import annotations

import argparse
from pathlib import Path

from .bandwise_sinkhorn_frontier import derive_bandwise_sinkhorn_frontier
from .common import build_manifest_row, export_rows_csv, stable_run_id
from .sinkhorn_self_coupling_certificate import certify_self_coupling_stability
from .sinkhorn_theorem_ready_band import run_sinkhorn_theorem_ready_band_report


def build_sinkhorn_frontier_summary(
    pairs: tuple[tuple[int, int], ...] = ((8, 1), (8, 2), (12, 1), (12, 2)),
) -> list[dict[str, float | int | bool | str]]:
    frontier = derive_bandwise_sinkhorn_frontier(pairs=pairs)
    theorem_rows = run_sinkhorn_theorem_ready_band_report(ambient_intrinsic_pairs=pairs)
    coupling_rows = certify_self_coupling_stability(
        max_spectral_radius=0.97,
        max_largest_n_mean_inverse_norm=3.0,
    )

    band_lookup = {
        (int(row["ambient_dim"]), int(row["intrinsic_dim"])): row
        for row in frontier.band_summary
    }
    theorem_lookup = {
        (int(row["ambient_dim"]), int(row["intrinsic_dim"])): row
        for row in theorem_rows
    }
    coupling_lookup = {
        (int(row["ambient_dim"]), int(row["intrinsic_dim"]), str(row["coupling"])): row
        for row in coupling_rows
    }

    rows: list[dict[str, float | int | bool | str]] = []
    for ambient_dim, intrinsic_dim in pairs:
        band_row = band_lookup[(ambient_dim, intrinsic_dim)]
        theorem_row = theorem_lookup[(ambient_dim, intrinsic_dim)]
        xx_row = coupling_lookup.get((ambient_dim, intrinsic_dim, "xx"))
        yy_row = coupling_lookup.get((ambient_dim, intrinsic_dim, "yy"))
        if xx_row is None or yy_row is None:
            self_coupling_status = "not_probed"
            self_coupling_positive = False
        else:
            self_coupling_positive = bool(
                bool(xx_row["stable_proxy"]) and bool(yy_row["stable_proxy"])
            )
            self_coupling_status = (
                "positive" if self_coupling_positive else "not_positive"
            )
        rows.append(
            {
                "ambient_dim": ambient_dim,
                "intrinsic_dim": intrinsic_dim,
                "empirical_epsilon_max": band_row["epsilon_max"],
                "theorem_ready_epsilon_max": theorem_row["theorem_ready_epsilon_max"],
                "rows_in_band": theorem_row["rows_in_band"],
                "min_iid_a": theorem_row["min_iid_a"],
                "min_triangular_a": theorem_row["min_triangular_a"],
                "max_gap": theorem_row["max_gap"],
                "critical_smoothness_alpha": theorem_row["critical_smoothness_alpha"],
                "exact_complexity_inheritance": theorem_row[
                    "exact_complexity_inheritance"
                ],
                "parametric_region_holds": theorem_row["parametric_region_holds"],
                "self_coupling_status": self_coupling_status,
                "self_coupling_positive": self_coupling_positive,
                "xx_largest_n_mean_inverse_norm": ""
                if xx_row is None
                else xx_row["largest_n_mean_inverse_norm"],
                "yy_largest_n_mean_inverse_norm": ""
                if yy_row is None
                else yy_row["largest_n_mean_inverse_norm"],
                "xx_worst_max_spectral_radius": ""
                if xx_row is None
                else xx_row["worst_max_spectral_radius"],
                "yy_worst_max_spectral_radius": ""
                if yy_row is None
                else yy_row["worst_max_spectral_radius"],
            }
        )
    return rows


def write_sinkhorn_frontier_summary_table(
    rows: list[dict[str, float | int | bool | str]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    body_lines: list[str] = []
    for row in rows:
        pair = f"$({int(row['ambient_dim'])},{int(row['intrinsic_dim'])})$"
        empirical_eps = row["empirical_epsilon_max"]
        theorem_eps = row["theorem_ready_epsilon_max"]
        self_proxy = {
            "positive": "yes",
            "not_positive": "no",
            "not_probed": "--",
        }[str(row["self_coupling_status"])]
        iid_a = float(row["min_iid_a"])
        tri_a = float(row["min_triangular_a"])
        body_lines.append(
            f"{pair} & {empirical_eps} & {theorem_eps} & {iid_a:.3f} & {tri_a:.3f} & {self_proxy} \\\\"
        )
    table = "\n".join(
        [
            r"\begin{table}[!htbp]",
            r"\centering",
            r"\small",
            r"\begin{tabular}{cccccc}",
            r"\toprule",
            r"Pair & Emp. $\varepsilon_{\max}$ & Thm.-ready $\varepsilon_{\max}$ & Min iid $a$ & Min tri. $a$ & Self \\",
            r"\midrule",
            *body_lines,
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Combined moderate-band Sinkhorn summary on the calibrated embedded grid. The table reports the empirical stable band, the theorem-ready band, the minimum recovered carrier exponents inside the theorem-ready band, and whether both self-coupling blocks satisfy the current spectral-radius and inverse-norm proxy thresholds.}",
            r"\label{tab:sinkhorn_frontier_summary}",
            r"\end{table}",
        ]
    )
    output_path.write_text(table + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a combined Sinkhorn frontier summary."
    )
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=Path("artifacts/csv/sinkhorn_frontier_summary"),
    )
    parser.add_argument(
        "--table-path",
        type=Path,
        default=Path("artifacts/tables/conceptual/sinkhorn_frontier_summary.tex"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_sinkhorn_frontier_summary()
    export_rows_csv(rows, args.csv_dir / "summary.csv")
    write_sinkhorn_frontier_summary_table(rows, args.table_path)
    manifest = build_manifest_row(
        "sinkhorn_frontier_summary",
        {"pairs": ((8, 1), (8, 2), (12, 1), (12, 2))},
        run_id=stable_run_id({"pairs": ((8, 1), (8, 2), (12, 1), (12, 2))}),
        notes="Combined Sinkhorn frontier, theorem-band, and self-coupling summary.",
    )
    export_rows_csv([manifest], args.csv_dir / "manifest.csv")


if __name__ == "__main__":
    main()
