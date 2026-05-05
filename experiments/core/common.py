from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 0:
        raise ValueError("window must be positive")
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same")


def threshold_crossings(values: np.ndarray, threshold: float) -> list[int]:
    warnings: list[int] = []
    below = False
    for index, value in enumerate(values):
        if np.isnan(value):
            continue
        if value < threshold and not below:
            warnings.append(index)
            below = True
        elif value >= threshold:
            below = False
    return warnings


def match_warnings_to_events(
    warnings: list[int], events: list[int], max_gap: int
) -> tuple[list[int], list[int], list[int]]:
    matched_warnings: list[int] = []
    matched_events: list[int] = []
    lead_times: list[int] = []
    event_index = 0
    for warning in warnings:
        while event_index < len(events) and events[event_index] < warning:
            event_index += 1
        if event_index < len(events):
            lead_time = events[event_index] - warning
            if lead_time <= max_gap:
                matched_warnings.append(warning)
                matched_events.append(events[event_index])
                lead_times.append(lead_time)
                event_index += 1
    return matched_warnings, matched_events, lead_times


def summarize_onset(warnings: list[int], start: int) -> dict[str, float | int | None]:
    first = next((warning for warning in warnings if warning >= start), None)
    return {"first_warning": first, "delay": None if first is None else first - start}


def export_summary_csv(rows: Sequence[Mapping[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_summary_markdown(rows: Sequence[Mapping[str, object]]) -> str:
    if not rows:
        return ""
    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = [str(row[header]) for header in headers]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"
