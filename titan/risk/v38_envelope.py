"""V3.8 envelope dataclasses + enums consumed by the PortfolioRiskManager.

Per `directives/PRM Integration V3.8 2026-05-24.md` §2.1 + §3.1 + §4.1 +
§4.4. This module defines the types that flow between the strategy side
(submitting candidate trades) and the PRM side (evaluating against the
V3.8 envelope). The PRM itself owns the orchestration (in
`portfolio_risk_manager.py`); this module owns the schemas.

The dataclasses are frozen for the same reason `CalmarPromotionResult` /
`HeatCheckResult` / `LeverageCheckResult` are frozen: a decision snapshot
should be immutable after the PRM returns it.

This module is **behaviourally inert** on its own -- no I/O, no global
state, no PRM coupling. The week-1 PR's gate per the integration
directive is "All new types have unit tests; register_strategy unchanged".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


class PreTradeRejectionReason(str, Enum):
    """Typed rejection reasons surfaced via PreTradeDecision.reason.

    Each enum value is what flows through Nautilus's order-rejected callback
    path (per §4.1 of the integration directive) so strategies can dispatch
    on a known string rather than parse a free-form text reason.
    """

    HEAT_CAP_HIT = "heat_cap_hit"
    LEVERAGE_CAP_HIT = "leverage_cap_hit"
    SPAN_BUFFER_LOW = "span_buffer_low"
    PER_TRADE_R_CAP_HIT = "per_trade_r_cap_hit"
    CRISIS_REGIME_HEAT_REDUCED = "crisis_regime_heat_reduced"
    DD_THROTTLE_HALVED = "dd_throttle_halved"
    DD_THROTTLE_FLATTEN = "dd_throttle_flatten"


@dataclass(frozen=True)
class PreTradeDecision:
    """Result of `PortfolioRiskManager.check_pre_trade(...)`.

    Attributes:
    ----------
    accepted:
        True iff the trade may proceed. Strategies should NOT submit if
        False.
    reason:
        The specific control that rejected the trade. `None` when
        `accepted is True`.
    diagnostic:
        Control-specific details for logging / Slack alerts.
        Empty dict when no diagnostic is relevant.
    enforced:
        True iff this decision was enforced (live mode). False iff the
        decision was evaluated in shadow mode (would_have_rejected
        telemetry only). Lets the strategy distinguish "the envelope is
        watching" from "the envelope is acting".
    """

    accepted: bool
    reason: PreTradeRejectionReason | None = None
    diagnostic: dict[str, float | str] = field(default_factory=dict)
    enforced: bool = True


@dataclass(frozen=True)
class OnBarDecisions:
    """Result of `PortfolioRiskManager.on_bar(...)`.

    The PRM evaluates regime + DD throttle + flatten recommendations
    once per bar (per §2.2 of the integration directive); strategies
    consume this via `prm.current_regime()` / `prm.current_dd_throttle()`
    rather than calling on_bar themselves.

    Attributes:
    ----------
    is_crisis_regime:
        Latest crisis-regime evaluation. True iff VIX > 30 OR per-instrument
        realised vol > 90th percentile (rolling 5y).
    regime_reasons:
        Tuple of human-readable trigger descriptions when `is_crisis_regime
        is True`. Empty tuple when normal regime.
    dd_throttle_multiplier:
        Current Kelly-fraction multiplier from the DD throttle. 1.0 normal,
        0.5 throttled.
    dd_throttle_triggered:
        True iff the DD throttle is currently engaged.
    portfolio_dd_from_60d_peak:
        Latest rolling 60-bar DD as a non-positive float.
    flatten_recommended:
        Tuple of strategy_ids whose pre-emptive flatten is recommended per
        §4.6 C4. Empty tuple when portfolio DD < -12% threshold.
    """

    is_crisis_regime: bool
    regime_reasons: tuple[str, ...]
    dd_throttle_multiplier: float
    dd_throttle_triggered: bool
    portfolio_dd_from_60d_peak: float
    flatten_recommended: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class StrategyV38Telemetry:
    """Per-strategy V3.8 telemetry slice. Aggregated by `V38Telemetry`."""

    strategy_id: str
    enabled: bool
    mode: Literal["shadow", "live"]
    would_have_rejected_count: int
    actual_rejected_count: int
    last_rejection_reason: PreTradeRejectionReason | None
    current_heat_contribution: float
    current_leverage_contribution: float


@dataclass(frozen=True)
class V38Telemetry:
    """Snapshot of the V3.8 envelope state for daily_summary + watchdog.

    Returned by `PortfolioRiskManager.get_v38_telemetry()` (per §2.1 of
    the integration directive). Consumed by the `daily_summary` ops
    strategy AM/PM rollups and the `subscription_health` watchdog.

    Attributes:
    ----------
    per_strategy:
        Per-strategy `StrategyV38Telemetry` records.
    portfolio_heat:
        Current sum-of-R / equity. Same units as `HeatCheckResult.portfolio_heat`.
    gross_leverage:
        Current sum-of-abs-notional / equity. Same units as
        `LeverageCheckResult.gross_leverage`.
    is_crisis_regime:
        Current regime-detector state.
    dd_throttle_state:
        Tuple of (multiplier, triggered) for the DD throttle.
    portfolio_dd_from_60d_peak:
        Latest rolling 60-bar DD.
    """

    per_strategy: tuple[StrategyV38Telemetry, ...]
    portfolio_heat: float
    gross_leverage: float
    is_crisis_regime: bool
    dd_throttle_state: tuple[float, bool]
    portfolio_dd_from_60d_peak: float


@dataclass(frozen=True)
class StrategyV38Config:
    """Per-strategy V3.8 envelope config, set via
    `PortfolioRiskManager.set_strategy_v38_config(strategy_id, config)`.

    Default constructor returns the V3.7-compatibility state: envelope
    disabled, mode is irrelevant. Per the integration directive §3, the
    runtime reads each strategy's TOML at startup (or on SIGHUP) and
    calls `set_strategy_v38_config` to register the per-strategy state.
    """

    enabled: bool = False
    mode: Literal["shadow", "live"] = "shadow"


__all__ = [
    "PreTradeRejectionReason",
    "PreTradeDecision",
    "OnBarDecisions",
    "StrategyV38Telemetry",
    "V38Telemetry",
    "StrategyV38Config",
]
