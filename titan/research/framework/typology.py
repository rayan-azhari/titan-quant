"""Strategy-class typology + per-class defaults.

Specified in directives/Methodology Audit & Unified Framework 2026-05-14.md
§2.1-§2.7. Every NEW audit MUST classify the strategy under audit into one
of these classes; defaults are then drawn from the typology table.

A strategy class's "primary metric" determines the gate-binding Sharpe
variant; the "secondary metric" is reported for diagnostics. For
sparse-trade strategies (INTRADAY_BREAKOUT, META_LABELING) the primary
is per-trade Sharpe (annualised at trades_per_year); for the rest it's
per-bar or per-day MTM.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

SharpeConvention = Literal["per_bar", "per_trade", "per_day_mtm"]


class StrategyClass(Enum):
    """The 9 typology classes per directive §2.1.

    Add new classes via a dedicated pre-registration directive that
    appends to this enum AND specifies the corresponding row in the
    defaults table below. No silent additions.
    """

    INTRADAY_MICROSTRUCTURE = "intraday_microstructure"  # H1+, sparse trades, mean-rev/range-exp
    INTRADAY_BREAKOUT = "intraday_breakout"  # M5/M15, ORB-style, very sparse
    DAILY_TREND = "daily_trend"  # D, persistent long-only
    DAILY_MEAN_REVERSION = "daily_mean_reversion"  # D, oscillator-based
    DAILY_MEAN_REVERSION_VOL_CARRY = "daily_mean_reversion_vol_carry"  # D, short-vol carry
    CROSS_ASSET_MOMENTUM = "cross_asset_momentum"  # D, always-on long
    PAIRS = "pairs"  # D, market-neutral
    ML_CLASSIFIER = "ml_classifier"  # any TF, predicts label, holds until flip
    META_LABELING = "meta_labeling"  # primary + ML filter
    CARRY = "carry"  # FX, slow-moving


@dataclass(frozen=True)
class CostModel:
    """Default cost model per asset class (directive §2.7).

    All fields in basis points (bps) except `commission_usd_per_side`
    (absolute USD per fill). Per-trade round-trip cost in USD on a
    position of notional N is:

        round_trip_cost = 2 * N * (spread_bps + slip_bps) / 1e4
                       + 2 * commission_usd_per_side
    """

    spread_bps: float
    slip_bps: float
    commission_usd_per_side: float

    @property
    def round_trip_bps_no_commission(self) -> float:
        return 2.0 * (self.spread_bps + self.slip_bps)


# Pre-committed cost models per asset class (directive §2.7).
COST_CME_FUTURES_LIQUID = CostModel(spread_bps=1.0, slip_bps=1.0, commission_usd_per_side=1.0)
COST_US_EQUITY_LARGE_CAP = CostModel(spread_bps=0.5, slip_bps=0.5, commission_usd_per_side=0.5)
COST_US_ETF_LIQUID = CostModel(spread_bps=1.0, slip_bps=0.5, commission_usd_per_side=0.35)
COST_UCITS_ETF = CostModel(spread_bps=8.0, slip_bps=2.0, commission_usd_per_side=2.0)
COST_FX_MAJOR = CostModel(spread_bps=0.5, slip_bps=0.3, commission_usd_per_side=0.0)
COST_IG_DFB_INDEX = CostModel(spread_bps=2.5, slip_bps=1.0, commission_usd_per_side=0.0)


@dataclass(frozen=True)
class WfoConfig:
    """Walk-forward design per strategy class (directive §2.3).

    Attributes:
        is_min_years: minimum in-sample window length.
        oos_years: out-of-sample window length per fold.
        fold_count: explicit fold count. If ``auto_fold_count`` is True
                    this value is treated as the FLOOR -- the framework
                    will use ``max(fold_count, auto-derived)``.
        is_mode: "expanding" or "rolling".
        stride_overlap_allowed: rolling-mode stride halving.
        auto_fold_count: if True, ``build_folds`` derives the fold count
                         from the visible window length to ensure OOS
                         spans (close to) the full available history.
                         Capped at ``auto_fold_count_max``.
        auto_fold_count_max: ceiling on auto-derived fold count.
    """

    is_min_years: float
    oos_years: float
    fold_count: int
    is_mode: Literal["expanding", "rolling"]
    stride_overlap_allowed: bool
    auto_fold_count: bool = True
    auto_fold_count_max: int = 60
    sanctuary_months: int = 12


@dataclass(frozen=True)
class McConfig:
    """Monte Carlo config per strategy class (directive §2.4)."""

    block_size_bars: int
    n_paths: int
    bootstrap_method: Literal["block", "shared_block", "stationary"]
    max_dd_threshold_pct: float  # e.g. 0.25 = 25%
    max_dd_pass_prob: float  # threshold for P(MaxDD > X)


@dataclass(frozen=True)
class SharpeReporting:
    """Which Sharpe to use as primary gate vs secondary diagnostic
    (directive §2.2).
    """

    primary: SharpeConvention
    secondary: SharpeConvention
    primary_periods_per_year: int | None = None  # None means strategy-data-dependent


@dataclass(frozen=True)
class StrategyClassDefaults:
    """Bundle of all framework defaults for a single strategy class."""

    sharpe: SharpeReporting
    wfo: WfoConfig
    mc: McConfig
    # Note: cost model is asset-class-specific, not strategy-class-specific,
    # so it's selected separately at audit time. See COST_* constants above.


# ── The defaults table (directive §2.2 + §2.3 + §2.4) ─────────────────────


DEFAULTS: dict[StrategyClass, StrategyClassDefaults] = {
    StrategyClass.INTRADAY_MICROSTRUCTURE: StrategyClassDefaults(
        sharpe=SharpeReporting(primary="per_bar", secondary="per_trade"),
        wfo=WfoConfig(
            is_min_years=1.0,
            oos_years=1.0,
            fold_count=5,
            is_mode="expanding",
            stride_overlap_allowed=False,
        ),
        mc=McConfig(
            block_size_bars=50,
            n_paths=200,
            bootstrap_method="block",
            max_dd_threshold_pct=0.25,
            max_dd_pass_prob=0.05,
        ),
    ),
    StrategyClass.INTRADAY_BREAKOUT: StrategyClassDefaults(
        sharpe=SharpeReporting(primary="per_trade", secondary="per_bar"),
        wfo=WfoConfig(
            is_min_years=1.0,
            oos_years=1.0,
            fold_count=5,
            is_mode="expanding",
            stride_overlap_allowed=False,
        ),
        mc=McConfig(
            block_size_bars=20,
            n_paths=200,
            bootstrap_method="block",
            max_dd_threshold_pct=0.15,
            max_dd_pass_prob=0.10,
        ),
    ),
    StrategyClass.DAILY_TREND: StrategyClassDefaults(
        sharpe=SharpeReporting(
            primary="per_day_mtm", secondary="per_trade", primary_periods_per_year=252
        ),
        wfo=WfoConfig(
            is_min_years=3.0,
            oos_years=1.0,
            fold_count=5,
            is_mode="expanding",
            stride_overlap_allowed=False,
        ),
        mc=McConfig(
            block_size_bars=21,
            n_paths=200,
            bootstrap_method="block",
            max_dd_threshold_pct=0.35,
            max_dd_pass_prob=0.10,
        ),
    ),
    StrategyClass.DAILY_MEAN_REVERSION: StrategyClassDefaults(
        sharpe=SharpeReporting(
            primary="per_day_mtm", secondary="per_trade", primary_periods_per_year=252
        ),
        wfo=WfoConfig(
            is_min_years=3.0,
            oos_years=1.0,
            fold_count=5,
            is_mode="expanding",
            stride_overlap_allowed=False,
        ),
        mc=McConfig(
            block_size_bars=21,
            n_paths=200,
            bootstrap_method="block",
            max_dd_threshold_pct=0.25,
            max_dd_pass_prob=0.10,
        ),
    ),
    # L25 (2026-05-15): short-vol-carry sub-class. Same Sharpe / WFO as
    # DAILY_MEAN_REVERSION but with a relaxed MC threshold calibrated to
    # the empirical Eraker-Wu / Cheng range for VRP-harvest strategies
    # (MaxDDs of 30-70% are structural, not failure modes).
    StrategyClass.DAILY_MEAN_REVERSION_VOL_CARRY: StrategyClassDefaults(
        sharpe=SharpeReporting(
            primary="per_day_mtm", secondary="per_trade", primary_periods_per_year=252
        ),
        wfo=WfoConfig(
            is_min_years=3.0,
            oos_years=1.0,
            fold_count=5,
            is_mode="expanding",
            stride_overlap_allowed=False,
        ),
        mc=McConfig(
            block_size_bars=21,
            n_paths=200,
            bootstrap_method="block",
            max_dd_threshold_pct=0.50,  # short-vol exposure: 50% MaxDD is the realistic tail
            max_dd_pass_prob=0.10,
        ),
    ),
    StrategyClass.CROSS_ASSET_MOMENTUM: StrategyClassDefaults(
        sharpe=SharpeReporting(
            primary="per_day_mtm", secondary="per_trade", primary_periods_per_year=252
        ),
        wfo=WfoConfig(
            is_min_years=2.0,
            oos_years=0.5,
            fold_count=8,
            is_mode="rolling",
            stride_overlap_allowed=True,
        ),
        # 63-bar (3-month) blocks preserve typical bond-equity correlation
        # regimes; threshold recalibrated from broken 25%/5% per
        # Bond-Equity Audit §4.2-c
        mc=McConfig(
            block_size_bars=63,
            n_paths=200,
            bootstrap_method="shared_block",
            max_dd_threshold_pct=0.35,
            max_dd_pass_prob=0.10,
        ),
    ),
    StrategyClass.PAIRS: StrategyClassDefaults(
        sharpe=SharpeReporting(
            primary="per_day_mtm", secondary="per_trade", primary_periods_per_year=252
        ),
        wfo=WfoConfig(
            is_min_years=3.0,
            oos_years=1.0,
            fold_count=5,
            is_mode="expanding",
            stride_overlap_allowed=False,
        ),
        mc=McConfig(
            block_size_bars=21,
            n_paths=200,
            bootstrap_method="shared_block",
            max_dd_threshold_pct=0.20,
            max_dd_pass_prob=0.05,
        ),
    ),
    StrategyClass.ML_CLASSIFIER: StrategyClassDefaults(
        sharpe=SharpeReporting(primary="per_bar", secondary="per_trade"),
        wfo=WfoConfig(
            is_min_years=2.0,
            oos_years=0.5,
            fold_count=8,
            is_mode="rolling",
            stride_overlap_allowed=True,
        ),
        mc=McConfig(
            block_size_bars=50,
            n_paths=200,
            bootstrap_method="block",
            max_dd_threshold_pct=0.25,
            max_dd_pass_prob=0.10,
        ),
    ),
    StrategyClass.META_LABELING: StrategyClassDefaults(
        sharpe=SharpeReporting(primary="per_trade", secondary="per_bar"),
        wfo=WfoConfig(
            is_min_years=2.0,
            oos_years=0.5,
            fold_count=8,
            is_mode="rolling",
            stride_overlap_allowed=True,
        ),
        mc=McConfig(
            block_size_bars=50,
            n_paths=200,
            bootstrap_method="block",
            max_dd_threshold_pct=0.25,
            max_dd_pass_prob=0.10,
        ),
    ),
    StrategyClass.CARRY: StrategyClassDefaults(
        sharpe=SharpeReporting(
            primary="per_day_mtm", secondary="per_trade", primary_periods_per_year=252
        ),
        wfo=WfoConfig(
            is_min_years=5.0,
            oos_years=1.0,
            fold_count=5,
            is_mode="expanding",
            stride_overlap_allowed=False,
        ),
        mc=McConfig(
            block_size_bars=21,
            n_paths=200,
            bootstrap_method="block",
            max_dd_threshold_pct=0.30,
            max_dd_pass_prob=0.10,
        ),
    ),
}


def defaults_for(cls: StrategyClass) -> StrategyClassDefaults:
    """Lookup the framework defaults for a strategy class. Raises if missing."""
    if cls not in DEFAULTS:
        raise KeyError(
            f"No defaults for {cls.value}. Add a row to titan.research.framework.typology.DEFAULTS "
            f"via a pre-registration directive before using this class."
        )
    return DEFAULTS[cls]
