"""Correlation dial — top-of-book leverage governor (validated B1, 2026-06-05).

The trailing average pairwise correlation of the survivorship-free liquid equity panel is the
state of the market's risk-budgeting machine: when it spikes, vol-control/risk-parity funds
force-deleverage indiscriminately (crashes cluster there). The dial scales AGGREGATE book
leverage inverse to the correlation z-score — de-grossing as the market gets brittle, re-grossing
in calm.

Validation (directives/Pre-Reg Correlation Dial Risk Governor 2026-06-05.md, build bc0e346 + rank
re-confirmation): forward-21d SPY vol rises monotonically with corr; Spearman(corr_z, fwd-vol)
FULL +0.47 / H1 +0.61 / H2 +0.31 (all p < 1e-14); applied as a leverage scalar it cut SPY MaxDD
55%->36% and joint P_kill 44%->28% across 18/18 cells. It is a SIZER, not a return forecaster.

Design notes:
  * FAIL-SAFE: missing/stale data, disabled config, or any error -> returns 1.0 (no leverage
    change). The governor must never break live sizing.
  * CAUSAL: correlation uses only trailing data; the latest z is vs the expanding history.
  * Off by default — operator enables it (config/risk.toml [allocation.correlation_dial]).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from titan.research.metrics import expanding_zscore

logger = logging.getLogger(__name__)

_PANEL_DIR = Path(__file__).resolve().parents[2] / "data" / "clean_equity"


@dataclass(frozen=True)
class CorrelationDialConfig:
    """Validated canonical cell (B1): k=0.5, clip (0.3, 1.5), 63d window, top-150 liquid names."""

    enabled: bool = False  # operator enables deliberately
    k: float = 0.5  # de-gross strength on the correlation z-score
    lev_min: float = 0.3
    lev_max: float = 1.5
    corr_window: int = 63
    top_k: int = 150
    z_min_periods: int = 50
    step: int = 5  # weekly cadence for the (cached) correlation series
    # P4.1 (2026-06-24): if the panel's latest observation is older than this
    # many days the dial is acting on stale market state -> fail-safe to 1.0.
    # The panel is a weekly-refreshed file; >10d means a refresh was missed.
    max_staleness_days: int = 10


def dial_leverage(corr_z: float, cfg: CorrelationDialConfig) -> float:
    """Pure map: correlation z-score -> clipped leverage. High z (brittle) -> low leverage."""
    if not np.isfinite(corr_z):
        return 1.0
    return float(np.clip(1.0 - cfg.k * corr_z, cfg.lev_min, cfg.lev_max))


def _avg_pairwise_corr(block: np.ndarray) -> float:
    if block.shape[1] < 5:
        return np.nan
    c = np.corrcoef(block, rowvar=False)
    n = c.shape[0]
    return float((np.nansum(c) - n) / (n * (n - 1)))


def compute_corr_series(cfg: CorrelationDialConfig, panel_dir: Path = _PANEL_DIR) -> pd.Series:
    """Trailing avg pairwise correlation of the top-K liquid alive names, weekly, causal."""
    tc = pd.read_parquet(panel_dir / "tr_close.parquet")
    mm = pd.read_parquet(panel_dir / "membership_mask.parquet").astype(bool)
    dv = pd.read_parquet(panel_dir / "dollar_volume.parquet")
    for df in (tc, mm, dv):
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    rets = tc.pct_change()
    idx = tc.index
    out = pd.Series(index=idx, dtype=float)
    for i in range(cfg.corr_window, len(idx), cfg.step):
        top = dv.iloc[i].where(mm.iloc[i]).nlargest(cfg.top_k).index
        block = rets.iloc[i - cfg.corr_window : i][list(top)].dropna(axis=1, how="any")
        if block.shape[1] >= 30:
            out.iloc[i] = _avg_pairwise_corr(block.values)
    return out.dropna()


class CorrelationDial:
    """Lazy, cached leverage governor. ``leverage_scalar(asof)`` -> [lev_min, lev_max], or 1.0."""

    def __init__(self, cfg: CorrelationDialConfig | None = None) -> None:
        self._cfg = cfg or CorrelationDialConfig()
        self._z: pd.Series | None = None

    def _ensure(self) -> None:
        if self._z is not None or not self._cfg.enabled:
            return
        try:
            corr = compute_corr_series(self._cfg)
            self._z = expanding_zscore(corr, min_periods=self._cfg.z_min_periods).dropna()
        except Exception as e:  # noqa: BLE001 — fail-safe: never break sizing
            logger.warning("[CorrelationDial] disabled — could not build correlation series: %s", e)
            self._z = pd.Series(dtype=float)

    def leverage_scalar(self, asof: date | None = None) -> float:
        """Causal leverage as of ``asof`` (latest <= asof). 1.0 if disabled / no data / error."""
        if not self._cfg.enabled:
            return 1.0
        self._ensure()
        if self._z is None or self._z.empty:
            return 1.0
        z = self._z
        ref = pd.Timestamp(asof) if asof is not None else pd.Timestamp.today().normalize()
        if asof is not None:
            z = z[z.index <= pd.Timestamp(asof)]
            if z.empty:
                return 1.0
        # P4.1 freshness gate: refuse to act on a stale panel (a missed weekly
        # refresh). Fail-safe to 1.0 rather than de-gross/re-gross on an old
        # reading -- the panel was 28d stale on 2026-06-24 while the dial sat at
        # its 1.5x ceiling (MLC-3/ALLOC-1).
        age_days = (ref - z.index[-1]).days
        if age_days > self._cfg.max_staleness_days:
            logger.warning(
                "[CorrelationDial] panel stale (%dd > %dd) — leverage forced to 1.0",
                age_days,
                self._cfg.max_staleness_days,
            )
            return 1.0
        return dial_leverage(float(z.iloc[-1]), self._cfg)

    def refresh(self) -> None:
        """Force a recompute on next call (e.g., after a data refresh)."""
        self._z = None


def correlation_dial_from_config(alloc_cfg: dict) -> CorrelationDial:
    """Build the dial from the [allocation.correlation_dial] sub-table (or defaults = disabled)."""
    sub = (alloc_cfg or {}).get("correlation_dial", {}) if alloc_cfg else {}
    cfg = CorrelationDialConfig(
        enabled=bool(sub.get("enabled", False)),
        k=float(sub.get("k", 0.5)),
        lev_min=float(sub.get("lev_min", 0.3)),
        lev_max=float(sub.get("lev_max", 1.5)),
        corr_window=int(sub.get("corr_window", 63)),
        top_k=int(sub.get("top_k", 150)),
        max_staleness_days=int(sub.get("max_staleness_days", 10)),
    )
    return CorrelationDial(cfg)
