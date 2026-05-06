from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypeAlias

SummaryValue: TypeAlias = str | float | int
SummaryRow: TypeAlias = Mapping[str, SummaryValue]
SummaryRows: TypeAlias = Sequence[SummaryRow]
