from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def _lookup(rows: list[dict[str, object]], **criteria: object) -> dict[str, object]:
    for row in rows:
        if all(row.get(key) == value for key, value in criteria.items()):
            return row
    raise KeyError(f"No row matched {criteria}")


def save_regime_first_summary_figure(
    elec2_rows: list[dict[str, object]],
    bikes_rows: list[dict[str, object]],
    active_rows: list[dict[str, object]],
    logged_rows: list[dict[str, object]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6), constrained_layout=True)

    # Passive regime: generic alarm wins.
    passive_specs = [
        (
            "ELEC2",
            _lookup(elec2_rows, strategy="fixed_100"),
            _lookup(elec2_rows, strategy="adwin"),
        ),
        (
            "Bikes",
            _lookup(bikes_rows, strategy="fixed_100"),
            _lookup(bikes_rows, strategy="adwin"),
        ),
    ]
    x = range(len(passive_specs))
    width = 0.34
    axis = axes[0]
    ci_leads = [float(spec[1]["leads"]) for spec in passive_specs]
    adwin_leads = [float(spec[2]["leads"]) for spec in passive_specs]
    axis.bar([i - width / 2 for i in x], ci_leads, width, label="CI", color="tab:blue")
    axis.bar(
        [i + width / 2 for i in x],
        adwin_leads,
        width,
        label="ADWIN",
        color="tab:purple",
    )
    axis.set_xticks(list(x), [spec[0] for spec in passive_specs])
    axis.set_ylabel("Matched leads")
    axis.set_title("Passive regime\nADWIN remains the default alarm")
    axis.text(
        0.02, 0.95, "Higher is better", transform=axis.transAxes, fontsize=8, va="top"
    )
    axis.legend(loc="upper left")
    for i, value in enumerate(ci_leads):
        axis.text(i - width / 2, value + 4, f"{value:.0f}", ha="center", fontsize=8)
    for i, value in enumerate(adwin_leads):
        axis.text(i + width / 2, value + 4, f"{value:.0f}", ha="center", fontsize=8)
    axis.grid(axis="y", alpha=0.2)

    # Active regime: effort adjustment wins.
    axis = axes[1]
    active_specs = [
        (
            "Masking onset",
            _lookup(active_rows, phase="masking detection", detector="TCI"),
            _lookup(active_rows, phase="masking detection", detector="TCIE"),
        ),
        (
            "Collapse onset",
            _lookup(active_rows, phase="collapse detection", detector="TCI"),
            _lookup(active_rows, phase="collapse detection", detector="TCIE"),
        ),
    ]
    ci_delay = [float(spec[1]["median_delay"]) for spec in active_specs]
    tcie_delay = [float(spec[2]["median_delay"]) for spec in active_specs]
    axis.bar([i - width / 2 for i in x], ci_delay, width, label="CI", color="tab:blue")
    axis.bar(
        [i + width / 2 for i in x], tcie_delay, width, label="CI^E", color="tab:red"
    )
    axis.set_xticks(list(x), [spec[0] for spec in active_specs])
    axis.set_ylabel("Median delay")
    axis.set_title("Active regime\nCI^E removes coercive masking")
    axis.text(
        0.02, 0.95, "Lower is better", transform=axis.transAxes, fontsize=8, va="top"
    )
    axis.legend(loc="upper left")
    for i, value in enumerate(ci_delay):
        axis.text(i - width / 2, value + 1.5, f"{value:.1f}", ha="center", fontsize=8)
    for i, value in enumerate(tcie_delay):
        axis.text(i + width / 2, value + 1.5, f"{value:.1f}", ha="center", fontsize=8)
    axis.grid(axis="y", alpha=0.2)

    # Logged regime: proxy-aware diagnostic wins.
    axis = axes[2]
    logged_specs = [
        (
            "Bubble onset",
            _lookup(logged_rows, phase="bubble_detection", detector="TCI"),
            _lookup(logged_rows, phase="bubble_detection", detector="TCIE"),
        ),
        (
            "Collapse onset",
            _lookup(logged_rows, phase="collapse_detection", detector="TCI"),
            _lookup(logged_rows, phase="collapse_detection", detector="TCIE"),
        ),
    ]
    ci_rate = [float(spec[1]["rate"]) for spec in logged_specs]
    tcie_rate = [float(spec[2]["rate"]) for spec in logged_specs]
    axis.bar([i - width / 2 for i in x], ci_rate, width, label="CI", color="tab:blue")
    axis.bar(
        [i + width / 2 for i in x], tcie_rate, width, label="CI^E", color="tab:red"
    )
    axis.set_xticks(list(x), [spec[0] for spec in logged_specs])
    axis.set_ylabel("Detection rate")
    axis.set_ylim(0, 1.0)
    axis.set_title("Logged regime\nCI^E wins under proxy effort")
    axis.legend(loc="upper left")
    for i, value in enumerate(ci_rate):
        axis.text(i - width / 2, value + 0.03, f"{value:.3f}", ha="center", fontsize=8)
    for i, value in enumerate(tcie_rate):
        axis.text(i + width / 2, value + 0.03, f"{value:.3f}", ha="center", fontsize=8)
    axis.grid(axis="y", alpha=0.2)

    fig.suptitle("Diagnostic regime map")
    fig.savefig(output_path)
    plt.close(fig)
