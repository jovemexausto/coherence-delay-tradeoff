from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .common import export_summary_csv
from .types import SummaryRows


@dataclass(slots=True)
class ExperimentHarness:
    figures_dir: Path
    artifacts_dir: Path

    def ensure(self) -> None:
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def figure_path(self, domain: str, filename: str) -> Path:
        return self.figures_dir / domain / filename

    def artifact_path(self, domain: str, filename: str) -> Path:
        return self.artifacts_dir / domain / filename

    def figure_file(self, filename: str) -> Path:
        return self.figures_dir / filename

    def save_summary_csv(self, rows: SummaryRows, domain: str, filename: str) -> None:
        export_summary_csv(rows, self.artifact_path(domain, filename))
