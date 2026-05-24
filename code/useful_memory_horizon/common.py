from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 0:
        raise ValueError("window must be positive")
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(values, kernel, mode="same")


def export_rows_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def stable_run_id(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:12]


def spawn_rng(master_seed: int, *keys: int | str) -> np.random.Generator:
    sequence_parts: list[int] = [int(master_seed)]
    for key in keys:
        if isinstance(key, str):
            digest = hashlib.sha256(key.encode("utf-8")).digest()
            sequence_parts.append(
                int.from_bytes(digest[:8], byteorder="big", signed=False)
            )
        else:
            sequence_parts.append(int(key))
    seed_sequence = np.random.SeedSequence(sequence_parts)
    return np.random.default_rng(seed_sequence)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def runtime_metadata() -> dict[str, str | None]:
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "numpy_version": package_version("numpy"),
        "scipy_version": package_version("scipy"),
        "matplotlib_version": package_version("matplotlib"),
        "river_version": package_version("river"),
        "package_version": package_version("useful-memory-horizon"),
    }


def build_manifest_row(
    experiment: str,
    config: dict[str, Any],
    *,
    run_id: str | None = None,
    seed: int | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    manifest = {
        "experiment": experiment,
        "run_id": run_id or stable_run_id({"experiment": experiment, "config": config}),
        "seed": seed,
        "notes": notes,
        "config": config,
        "runtime": runtime_metadata(),
    }
    return manifest
