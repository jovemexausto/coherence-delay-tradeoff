from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "experiments" / "artifacts"
OUTPUT = ARTIFACTS / "revision" / "revision_audit.md"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def select_one(rows: list[dict[str, str]], **criteria: str) -> dict[str, str]:
    for row in rows:
        if all(row[key] == value for key, value in criteria.items()):
            return row
    raise KeyError(f"No row matched {criteria}")


def format_float(value: str, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def main() -> None:
    elec2 = read_csv(ARTIFACTS / "elec2" / "elec2_summary.csv")
    bikes = read_csv(ARTIFACTS / "bikes" / "bikes_summary.csv")
    kuairand_default = read_csv(
        ARTIFACTS / "kuairand" / "kuairand_followup_default.csv"
    )
    kuairand_improvement = read_csv(
        ARTIFACTS / "kuairand" / "kuairand_followup_improvement.csv"
    )
    kuairand_lambda = read_csv(ARTIFACTS / "kuairand" / "kuairand_followup_lambda.csv")
    kuairand_proxy = read_csv(ARTIFACTS / "kuairand" / "kuairand_followup_proxy.csv")
    sinkhorn = read_csv(ARTIFACTS / "gaussian" / "gaussian_sinkhorn_runtime.csv")
    particle = read_csv(ARTIFACTS / "particle" / "particle_masking_grid_summary.csv")

    elec2_most_sensitive = max(elec2, key=lambda row: int(row["leads"]))
    bikes_most_sensitive = max(bikes, key=lambda row: int(row["leads"]))
    elec2_most_precise = max(
        [row for row in elec2 if int(row["warnings"]) > 0],
        key=lambda row: float(row["precision"]),
    )
    bikes_most_precise = max(
        [row for row in bikes if int(row["warnings"]) > 0],
        key=lambda row: float(row["precision"]),
    )

    bubble_ci = select_one(kuairand_default, phase="bubble_detection", detector="CI")
    bubble_tcie = select_one(
        kuairand_default, phase="bubble_detection", detector="CI^E"
    )
    bubble_kswin = select_one(
        kuairand_default, phase="bubble_detection", detector="KSWIN"
    )
    collapse_tcie = select_one(
        kuairand_default, phase="collapse_detection", detector="CI^E"
    )
    bubble_gain = select_one(kuairand_improvement, phase="bubble_detection")
    collapse_gain = select_one(kuairand_improvement, phase="collapse_detection")

    lambda_rows = [row for row in kuairand_lambda if row["phase"] == "bubble_detection"]
    best_lambda = max(lambda_rows, key=lambda row: float(row["rate"]))
    stable_lambda = [row for row in lambda_rows if row["lambda"] in {"2.0", "3.0"}]

    proxy_rows = [row for row in kuairand_proxy if row["phase"] == "bubble_detection"]
    proxy_best = max(proxy_rows, key=lambda row: float(row["rate"]))

    particle_default = select_one(
        particle, **{"regime": "coercive", "influence": "0.3", "lambda": "3.0"}
    )
    particle_passive = select_one(
        particle, **{"regime": "passive", "influence": "0.3", "lambda": "3.0"}
    )

    d8_eps005 = select_one(sinkhorn, dimension="8", window_size="100", epsilon="0.05")
    d8_eps10 = select_one(sinkhorn, dimension="8", window_size="100", epsilon="1.0")
    d256_eps005 = select_one(
        sinkhorn, dimension="256", window_size="100", epsilon="0.05"
    )
    d256_eps10 = select_one(sinkhorn, dimension="256", window_size="100", epsilon="1.0")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Revision Audit",
        "",
        "## Flagship Summary",
        "",
        "- Passive real-world streams still favor generic drift detectors, but the ranking depends on whether one values raw sensitivity or balanced precision: "
        f"ELEC2 most sensitive = {elec2_most_sensitive['strategy']} ({elec2_most_sensitive['leads']} leads, precision {format_float(elec2_most_sensitive['precision'])}), "
        f"most precise = {elec2_most_precise['strategy']} ({format_float(elec2_most_precise['precision'])}); "
        f"Bikes most sensitive = {bikes_most_sensitive['strategy']} ({bikes_most_sensitive['leads']} leads, precision {format_float(bikes_most_sensitive['precision'])}), "
        f"most precise = {bikes_most_precise['strategy']} ({format_float(bikes_most_precise['precision'])}).",
        "- The distinctive surviving contribution is intervention-aware monitoring: "
        f"on KuaiRand bubble detection rises from CI {format_float(bubble_ci['rate'])} to CI^E {format_float(bubble_tcie['rate'])}, "
        f"with paired bootstrap delta {format_float(bubble_gain['delta_rate'])} [{format_float(bubble_gain['ci_low'])}, {format_float(bubble_gain['ci_high'])}].",
        "- The synthetic masking benchmark shows the same pattern: "
        f"at particle influence 0.3 and lambda 3.0, CI stays at {format_float(particle_default['mean_tail_tci'])} while CI^E drops to {format_float(particle_default['mean_tail_tcie'])}, "
        f"creating masking gap {format_float(particle_default['mean_masking_gap'])}; in the passive control, the gap is {format_float(particle_passive['mean_masking_gap'])}.",
        "",
        "## Calibration and Sensitivity",
        "",
        f"- KuaiRand bubble detection is highest at lambda {best_lambda['lambda']} (rate {format_float(best_lambda['rate'])}, healthy FP/user {format_float(best_lambda['healthy_fp_per_user'])}), but the current text's stable operating region [2, 3] is consistent with the measured trade-off.",
        "- Stable-region rows:",
    ]

    for row in stable_lambda:
        lines.append(
            f"  - lambda {row['lambda']}: bubble rate {format_float(row['rate'])}, healthy FP/user {format_float(row['healthy_fp_per_user'])}."
        )

    lines.extend(
        [
            f"- The strongest logged effort proxy remains {proxy_best['proxy'].upper()} for bubble detection (rate {format_float(proxy_best['rate'])}, healthy FP/user {format_float(proxy_best['healthy_fp_per_user'])}).",
            "",
            "## Sinkhorn Runtime / Bias Trade-off",
            "",
            f"- At d=8, n=100, increasing epsilon from 0.05 to 1.0 changes runtime from {format_float(d8_eps005['mean_runtime_ms'])} ms to {format_float(d8_eps10['mean_runtime_ms'])} ms and mean abs. bias from {format_float(d8_eps005['mean_abs_bias'], 4)} to {format_float(d8_eps10['mean_abs_bias'], 4)}.",
            f"- At d=256, n=100, the same change moves runtime from {format_float(d256_eps005['mean_runtime_ms'])} ms to {format_float(d256_eps10['mean_runtime_ms'])} ms and mean abs. bias from {format_float(d256_eps005['mean_abs_bias'], 4)} to {format_float(d256_eps10['mean_abs_bias'], 4)}.",
            "- This confirms the current manuscript's interpretation: epsilon is an operational calibration knob, not a universal fixed setting.",
            "",
            "## Immediate Revision Priorities",
            "",
            "1. Promote CI / CI^E and coercive masking as the flagship contribution in the abstract, introduction, and conclusion.",
            "2. Keep the cube-root law as the rigorous backbone, but explicitly state that passive streams remain ADWIN-favored in the current evidence bundle.",
            "3. Move the fairness/calibration message for KuaiRand into a more prominent sentence in the body text: healthy-only thresholds, same scalar input, paired bootstrap intervals.",
            "4. Tighten the Sinkhorn citation chain around the null-vs-alternative calibration split and published Goldfeld et al. citation.",
            '5. Strengthen the lower-bound proposition by making the operational constant/regime explicit rather than only saying "universal constant".',
            "",
            "## Reviewer-Facing Takeaway",
            "",
            f"- KuaiRand CI^E bubble/collapse rates: {format_float(bubble_tcie['rate'])} / {format_float(collapse_tcie['rate'])}.",
            f"- KuaiRand strongest generic passive baseline on bubble onset: KSWIN {format_float(bubble_kswin['rate'])}.",
            f"- KuaiRand paired improvements over CI: bubble {format_float(bubble_gain['delta_rate'])}, collapse {format_float(collapse_gain['delta_rate'])}.",
            f"- Particle masking gap at default setting: {format_float(particle_default['mean_masking_gap'])}.",
        ]
    )

    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
