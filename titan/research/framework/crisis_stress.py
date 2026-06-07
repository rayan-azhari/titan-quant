"""Named-crisis-window stress test (V3.7).

MC block bootstrap shuffles bars uniformly and therefore under-samples
named crisis events (1987 Oct, 2000-02, 2008 Sep-Nov, 2010 Flash, 2015
Aug, 2018 Q4, 2020 Mar, 2022 Q1, 2023 Mar SVB, etc). Real risk
management requires explicit per-event stress reporting.

This module:
    1. Defines a registry of named historical crisis windows.
    2. Computes per-window strategy returns + drawdown.
    3. Reports per-window pass/fail vs a max-DD threshold.
    4. Aggregates to a "worst-crisis-MaxDD" metric for the audit.

Usage:

    from titan.research.framework.crisis_stress import run_crisis_stress

    res = run_crisis_stress(
        strategy_returns=stitched_oos_returns,
        max_dd_threshold=0.25,
    )
    print(res.report())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

# Named crisis windows. Each is (label, start_date, end_date, description).
NAMED_CRISES: list[tuple[str, date, date, str]] = [
    (
        "1987_blackmonday",
        date(1987, 10, 1),
        date(1987, 12, 31),
        "Oct 19 -22% one-day; Reagan crash",
    ),
    ("2000_dotcom", date(2000, 3, 1), date(2002, 12, 31), "Dot-com bear, -49% SPY peak-trough"),
    ("2008_gfc", date(2008, 9, 1), date(2009, 3, 31), "Lehman + AIG; -57% peak-trough"),
    ("2010_flashcrash", date(2010, 5, 6), date(2010, 5, 7), "May 6 intraday -9%, recovered"),
    ("2011_eurozone", date(2011, 7, 1), date(2011, 10, 31), "US debt downgrade + EZ debt crisis"),
    ("2015_china", date(2015, 8, 1), date(2015, 10, 31), "Yuan devaluation, VIX spike"),
    ("2018_volmageddon", date(2018, 2, 1), date(2018, 2, 28), "Feb 5 XIV blowup"),
    ("2018_q4", date(2018, 10, 1), date(2018, 12, 31), "Fed-hike-cycle bear, -20% SPY"),
    ("2020_covid", date(2020, 2, 19), date(2020, 4, 30), "Mar -34% in 23 days"),
    (
        "2022_trend_reversal",
        date(2022, 1, 1),
        date(2022, 10, 31),
        "Inflation + Fed; bonds+stocks down",
    ),
    ("2023_svb", date(2023, 3, 8), date(2023, 3, 31), "SVB collapse, regional bank stress"),
    ("2025_tradewar", date(2025, 3, 1), date(2025, 5, 31), "Tariff-driven equity selloff"),
]


@dataclass
class CrisisWindowResult:
    """Per-named-crisis stats for a single strategy."""

    label: str
    start: date
    end: date
    n_bars: int
    total_return: float
    max_drawdown: float
    description: str

    def passes(self, max_dd_threshold: float = 0.25) -> bool:
        """Pass if drawdown is within threshold (negative number)."""
        return self.max_drawdown > -abs(max_dd_threshold)


@dataclass
class CrisisStressResult:
    """Aggregated crisis-stress output across all named windows."""

    windows: list[CrisisWindowResult] = field(default_factory=list)
    max_dd_threshold: float = 0.25

    @property
    def n_tested(self) -> int:
        return len(self.windows)

    @property
    def n_passes(self) -> int:
        return sum(1 for w in self.windows if w.passes(self.max_dd_threshold))

    @property
    def worst_crisis(self) -> CrisisWindowResult | None:
        if not self.windows:
            return None
        return min(self.windows, key=lambda w: w.max_drawdown)

    def report(self) -> str:
        lines = [
            f"Crisis Stress Report (threshold = {self.max_dd_threshold:.1%})",
            f"{'Window':<24} {'Start':>10} {'End':>10} {'TotRet':>10} {'MaxDD':>10} {'Pass':>6}",
            "─" * 80,
        ]
        for w in self.windows:
            mark = "✓" if w.passes(self.max_dd_threshold) else "✗"
            lines.append(
                f"{w.label:<24} {str(w.start):>10} {str(w.end):>10} "
                f"{w.total_return:>+9.2%} {w.max_drawdown:>+9.2%} {mark:>6}"
            )
        lines.append("─" * 80)
        ww = self.worst_crisis
        if ww:
            lines.append(f"Worst: {ww.label}  MaxDD = {ww.max_drawdown:+.2%}")
        lines.append(f"PASSES: {self.n_passes}/{self.n_tested}")
        return "\n".join(lines)


def _max_drawdown(returns: np.ndarray) -> float:
    if len(returns) < 2:
        return 0.0
    eq = np.concatenate([[1.0], np.cumprod(1.0 + returns)])
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    return float(dd.min())


def run_crisis_stress(
    strategy_returns: pd.Series,
    *,
    max_dd_threshold: float = 0.25,
    crises: list[tuple[str, date, date, str]] | None = None,
) -> CrisisStressResult:
    """Compute per-crisis-window total return + MaxDD for a strategy.

    Parameters:
        strategy_returns: daily per-bar returns with DatetimeIndex.
        max_dd_threshold: drawdown level that defines a "fail" for each
            crisis window. Default 25%.
        crises: optional override of the crisis registry.

    Returns:
        CrisisStressResult with per-window stats + summary.
    """
    if crises is None:
        crises = NAMED_CRISES
    s = strategy_returns.dropna()
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index)
    s_index_dates = s.index.date

    result = CrisisStressResult(max_dd_threshold=max_dd_threshold)
    for label, start_d, end_d, desc in crises:
        mask = (s_index_dates >= start_d) & (s_index_dates <= end_d)
        window = s[mask]
        if len(window) < 2:
            continue
        arr = window.to_numpy()
        total_ret = float(np.prod(1.0 + arr) - 1.0)
        mdd = _max_drawdown(arr)
        result.windows.append(
            CrisisWindowResult(
                label=label,
                start=start_d,
                end=end_d,
                n_bars=len(window),
                total_return=total_ret,
                max_drawdown=mdd,
                description=desc,
            )
        )
    return result


__all__ = ["CrisisStressResult", "CrisisWindowResult", "NAMED_CRISES", "run_crisis_stress"]
