"""Gross leverage + SPAN margin envelope (V3.8 §4.6 controls C1 + C2).

Per `directives/Objective Reframe 2026-05-23.md` §4.6:

    C1 -- Max gross leverage: sum(abs_notional_i) / equity <= 8x in normal
          regime, <= 6x in crisis regime. PRM pre-trade reject; log
          WARNING leverage_cap_hit.

    C2 -- SPAN margin buffer: maintain >= 2x exchange minimum maintenance
          margin per open position. If broker-reported margin drops below
          2x minimum, reduce position to restore buffer at next bar.
          Protects against mid-crisis SPAN expansion (March 2020 +60%).

This module implements C1 + C2 as pure functions over a snapshot of
positions. The PRM integration that calls them on every pre-trade lives
in `titan/risk/portfolio_risk_manager.py::check_pre_trade(...)`. The
crisis-regime detection that determines whether to use the 8x or 6x cap
(C3) lives in `titan/strategies/regime_filter/regime.py` (separate module).

Pre-emptive flatten (C4) at -12% DD is NOT in this module -- it depends on
position-level stop placement and lives alongside the PRM integration.

Why "abs notional", not signed
==============================

A short ES position and a long ES position both consume gross leverage --
they don't net out for the purpose of "what happens if all positions gap
the wrong way simultaneously". The directive specifies
`sum(abs_notional_i)`, i.e. the L1 norm of position notionals. Internally
hedged strategies (e.g. a top-N rotation that's 3 long + 3 short) consume
6x heat against the 8x cap even though the net position is zero -- this
is by design (see §13 caveat 4 in the directive).

Why a buffer ratio, not a margin headroom in dollars
====================================================

SPAN margin grows multiplicatively in a vol spike (March 2020 saw a 60%
increase in a week). A dollar-headroom check would pass at peace and fail
at exactly the moment it was needed. A ratio of "posted / exchange_min"
captures the multiplicative headroom that survives proportional margin
expansion. A 2x ratio absorbs up to 100% SPAN expansion before forced
liquidation.

Worked example: ES position with exchange min margin = $5,000. Buffer 2x
means broker is holding $10,000 in maintenance margin. If SPAN expands
60% (margin -> $8,000), the position is still inside the buffer. If SPAN
expands 110% (margin -> $10,500), the position breaches and C2 says
"reduce position to restore buffer at next bar".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

# V3.8 §4.6 defaults.
DEFAULT_NORMAL_LEVERAGE_CAP: float = 8.0
DEFAULT_CRISIS_LEVERAGE_CAP: float = 6.0
DEFAULT_SPAN_BUFFER_MIN: float = 2.0


@dataclass(frozen=True)
class PositionSnapshot:
    """One open position's contribution to the leverage + SPAN envelope.

    Attributes:
    ----------
    symbol:
        Instrument identifier (e.g. "ES", "MES", "ZB").
    abs_notional:
        Absolute notional value of the position in the account base
        currency. For a 1-contract ES position at $5000 multiplier and
        index 4500 with USD account, abs_notional = $2,250,000. Caller
        is responsible for currency conversion (see
        `titan.risk.fx_units.convert_notional_to_units`).
    margin_posted:
        Broker-reported maintenance margin in the account base currency.
        None means margin data is not available (typical in backtests);
        SPAN check is skipped for that position.
    exchange_min_margin:
        Exchange minimum maintenance margin in the account base
        currency. None means data is not available; SPAN check is
        skipped for that position.
    """

    symbol: str
    abs_notional: float
    margin_posted: float | None = None
    exchange_min_margin: float | None = None


@dataclass(frozen=True)
class LeverageCheckResult:
    """Combined C1 + C2 envelope evaluation against the current portfolio.

    Returned by `evaluate_leverage_envelope` and used by the PRM
    `check_pre_trade` hook to accept / reject a candidate trade.

    Attributes:
    ----------
    gross_leverage:
        `sum(abs_notional_i) / equity` including the candidate (if any).
    leverage_cap:
        Cap actually applied (8x normal or 6x crisis per §4.6 C3).
    leverage_pass:
        True iff `gross_leverage <= leverage_cap`.
    span_buffer_ratios:
        Per-symbol `margin_posted / exchange_min_margin` ratio.
        Symbols with missing margin data are NOT included.
    span_buffer_min_ratio:
        Minimum ratio across all positions with margin data, or
        positive infinity if no position has margin data.
    span_buffer_pass:
        True iff every position with margin data has ratio >= min_ratio.
        Vacuously True when no position has margin data.
    passes:
        True iff BOTH leverage_pass AND span_buffer_pass.
    reasons:
        Human-readable explanation of any failing control. Empty tuple
        on full PASS.
    span_breaching_symbols:
        Tuple of symbol names whose buffer ratio fell below the minimum.
        Useful for the PRM to know which positions to reduce on next bar.
    """

    gross_leverage: float
    leverage_cap: float
    leverage_pass: bool
    span_buffer_ratios: dict[str, float]
    span_buffer_min_ratio: float
    span_buffer_min_required: float
    span_buffer_pass: bool
    passes: bool
    reasons: tuple[str, ...]
    span_breaching_symbols: tuple[str, ...] = field(default_factory=tuple)


def compute_gross_leverage(
    positions: Iterable[PositionSnapshot],
    equity: float,
) -> float:
    """`sum(abs_notional_i) / equity`. Returns 0.0 if equity <= 0."""
    if equity <= 0.0:
        return 0.0
    total = sum(max(0.0, p.abs_notional) for p in positions)
    return total / equity


def compute_span_buffer_ratio(
    margin_posted: float | None,
    exchange_min_margin: float | None,
) -> float | None:
    """Per-position buffer ratio: `posted / exchange_min`.

    Returns None when either input is None or exchange_min is non-positive
    (broker / exchange data unavailable). Returns positive infinity if
    margin_posted > 0 but exchange_min is zero (which is a data quality
    issue — caller should investigate).
    """
    if margin_posted is None or exchange_min_margin is None:
        return None
    if exchange_min_margin <= 0.0:
        return float("inf") if margin_posted > 0 else None
    return margin_posted / exchange_min_margin


def evaluate_leverage_envelope(
    positions: Iterable[PositionSnapshot],
    equity: float,
    candidate: PositionSnapshot | None = None,
    *,
    regime_normal: bool = True,
    leverage_cap_normal: float = DEFAULT_NORMAL_LEVERAGE_CAP,
    leverage_cap_crisis: float = DEFAULT_CRISIS_LEVERAGE_CAP,
    span_buffer_min: float = DEFAULT_SPAN_BUFFER_MIN,
) -> LeverageCheckResult:
    """Evaluate C1 (leverage) and C2 (SPAN buffer) against the portfolio.

    Pass `candidate` to evaluate "what if I added this trade?" -- the
    candidate's abs_notional is INCLUDED in the leverage sum, and its
    SPAN buffer is INCLUDED in the per-symbol map. Pass `candidate=None`
    to evaluate the current portfolio's compliance.

    Parameters
    ----------
    positions:
        Iterable of open `PositionSnapshot`s.
    equity:
        Account equity in the same currency as `abs_notional`. Returned
        gross_leverage is 0.0 if equity <= 0.
    candidate:
        Optional proposed new position. When evaluating a PRM
        pre-trade hook, pass the candidate here.
    regime_normal:
        True for normal regime (8x cap), False for crisis (6x cap) per
        §4.6 C3. Detection lives in `regime_filter/regime.py`.
    leverage_cap_normal, leverage_cap_crisis, span_buffer_min:
        Override defaults if needed.

    Returns:
    -------
    `LeverageCheckResult` with the gross leverage, applied cap, per-symbol
    SPAN ratios, and combined PASS/FAIL with diagnostic reasons.
    """
    pos_list = list(positions)
    if candidate is not None:
        pos_list = pos_list + [candidate]

    leverage_cap = leverage_cap_normal if regime_normal else leverage_cap_crisis
    gross = compute_gross_leverage(pos_list, equity)
    leverage_pass = gross <= leverage_cap

    span_ratios: dict[str, float] = {}
    breaching: list[str] = []
    for p in pos_list:
        ratio = compute_span_buffer_ratio(p.margin_posted, p.exchange_min_margin)
        if ratio is None:
            continue
        span_ratios[p.symbol] = ratio
        if ratio < span_buffer_min:
            breaching.append(p.symbol)

    if span_ratios:
        span_min_ratio = min(span_ratios.values())
        span_pass = span_min_ratio >= span_buffer_min
    else:
        # Vacuously pass when no broker / exchange margin data is available
        # (typical in backtests). Caller may wrap this in a stricter check.
        span_min_ratio = float("inf")
        span_pass = True

    reasons: list[str] = []
    if not leverage_pass:
        reasons.append(
            f"Gross leverage {gross:.2f}x above cap {leverage_cap:.2f}x "
            f"({'crisis' if not regime_normal else 'normal'} regime)"
        )
    if not span_pass:
        sym_list = ", ".join(sorted(breaching))
        reasons.append(
            f"SPAN buffer ratio {span_min_ratio:.2f}x below required "
            f"{span_buffer_min:.2f}x on: {sym_list}"
        )

    return LeverageCheckResult(
        gross_leverage=gross,
        leverage_cap=leverage_cap,
        leverage_pass=leverage_pass,
        span_buffer_ratios=span_ratios,
        span_buffer_min_ratio=span_min_ratio,
        span_buffer_min_required=span_buffer_min,
        span_buffer_pass=span_pass,
        passes=leverage_pass and span_pass,
        reasons=tuple(reasons),
        span_breaching_symbols=tuple(sorted(breaching)),
    )


def would_candidate_breach_leverage(
    current_positions: Iterable[PositionSnapshot],
    equity: float,
    candidate: PositionSnapshot,
    *,
    regime_normal: bool = True,
    leverage_cap_normal: float = DEFAULT_NORMAL_LEVERAGE_CAP,
    leverage_cap_crisis: float = DEFAULT_CRISIS_LEVERAGE_CAP,
) -> bool:
    """Convenience wrapper for the PRM pre-trade hook.

    Returns True iff adding `candidate` would push gross leverage above
    the applicable cap. The PRM uses this to reject the trade before
    submission (per V3.8 rule V3.8-11).

    Equivalent to checking
    `evaluate_leverage_envelope(...).leverage_pass is False`.
    """
    res = evaluate_leverage_envelope(
        current_positions,
        equity,
        candidate=candidate,
        regime_normal=regime_normal,
        leverage_cap_normal=leverage_cap_normal,
        leverage_cap_crisis=leverage_cap_crisis,
    )
    return not res.leverage_pass


__all__ = [
    "DEFAULT_NORMAL_LEVERAGE_CAP",
    "DEFAULT_CRISIS_LEVERAGE_CAP",
    "DEFAULT_SPAN_BUFFER_MIN",
    "PositionSnapshot",
    "LeverageCheckResult",
    "compute_gross_leverage",
    "compute_span_buffer_ratio",
    "evaluate_leverage_envelope",
    "would_candidate_breach_leverage",
]
