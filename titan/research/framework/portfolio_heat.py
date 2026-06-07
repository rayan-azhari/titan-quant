"""Portfolio heat envelope (V3.8 §4.2 + §4.6 C3 crisis-regime reduction).

Per `directives/Objective Reframe 2026-05-23.md` §4.2:

    Definition: portfolio_heat = sum across open positions of
                (entry_price - stop_price) * units / total_equity.
    Cap: <= 8%.
    Enforcement: PRM-side pre-trade check. If a new entry would push heat
                 > 8%, the entry is REJECTED (not partially filled). Log a
                 WARNING heat_cap_hit event.
    Rationale: 4 simultaneous correlated trades at 2% R each = 8% real risk
               on a single market gap. The cap exists to prevent the
               "uncorrelated portfolio that becomes correlated in the tail"
               failure mode (LTCM, 2008 quant quake, 2020 March).

Per §4.6 C3 the cap drops from 8% to 6% in crisis regime (VIX > 30 OR
realised vol > 90th pct). The cap is parameterised here; the regime
detection lives in `titan/strategies/regime_filter/regime.py`.

This module implements the heat envelope as a pure function over a list
of `PositionHeat` records, each capturing one position's risk amount
(R in account base currency). The PRM is responsible for computing R
from entry / stop / units / fx_rate before constructing the record --
the framework primitive doesn't need to know about prices or units, just
the aggregated R per position.

Why "risk amount" not entry / stop / units
==========================================

The directive defines heat as `(entry - stop) * units / equity` per
position. That formula combines instrument-level price, contract
multiplier, units, and FX into a single dollar number ("R"). The PRM
already does that conversion via `titan.risk.fx_units.convert_notional_to_units`
+ contract multiplier lookup, so the framework primitive takes R as a
pre-computed scalar.

For shorts the directive's formula naturally gives `R > 0` because
`stop > entry` for shorts (stop above entry; loss if price rises). The
PRM constructs R as `abs((entry - stop) * units) * fx_rate`, which works
for both directions.

The framework primitive validates `risk_amount >= 0` and silently clamps
malformed negatives to zero (defensive — matches `leverage_envelope`).

Heat is a SUM of bounded losses (the floor)
============================================

If every stop fills exactly at the stop price, portfolio heat is the
maximum nominal loss the portfolio can take across simultaneous stops.
This is the contractual floor; actual realised loss can exceed heat if:

1. Gap risk -- overnight or pre-bar gaps through the stop. V3.8 §V3.8-9
   budgets 1.5R expected per position, so heat * 1.5 is a better
   first-order estimate of realised loss under gap risk.
2. Slippage on simultaneous market-on-stop orders -- order book depth
   absorption.
3. Stop-hunting -- not captured in heat; mitigated via ATR-anchored
   stops per §V3.8-8.

Heat is the right primary control because it's COMPUTABLE pre-trade with
no path-dependent assumptions. Gap-risk reserve (§4.6 C3 -> heat 8% to 6%
in crisis) is the regime-conditional adjustment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

# V3.8 §4.2 + §4.6 C3 defaults.
DEFAULT_NORMAL_HEAT_CAP: float = 0.08
DEFAULT_CRISIS_HEAT_CAP: float = 0.06


@dataclass(frozen=True)
class PositionHeat:
    """One open position's contribution to portfolio heat.

    Attributes:
    ----------
    symbol:
        Instrument identifier (e.g. "ES", "MES", "ZB").
    risk_amount:
        Maximum nominal loss at stop fill, in the account base currency.
        Computed by the PRM as `abs((entry - stop) * units) * fx_rate`
        (works for both long and short positions; always non-negative).
        Negative values are silently clamped to zero (defensive).
    """

    symbol: str
    risk_amount: float


@dataclass(frozen=True)
class HeatCheckResult:
    """Combined heat envelope evaluation against the current portfolio.

    Returned by `evaluate_heat_envelope` and consumed by the PRM
    `check_pre_trade` hook to accept / reject a candidate trade.

    Attributes:
    ----------
    portfolio_heat:
        `sum(risk_amount_i) / equity` including the candidate (if any).
    heat_cap:
        Cap actually applied (8% normal or 6% crisis per §4.6 C3).
    heat_pass:
        True iff `portfolio_heat <= heat_cap`.
    per_position_r:
        Per-symbol risk amount in account base currency. Useful for the
        PRM to know which positions contribute most to heat.
    total_r:
        Sum of per-position risk amounts (numerator of portfolio_heat).
    reasons:
        Human-readable explanation of any failing control. Empty tuple
        on PASS.
    """

    portfolio_heat: float
    heat_cap: float
    heat_pass: bool
    per_position_r: dict[str, float]
    total_r: float
    reasons: tuple[str, ...] = field(default_factory=tuple)

    @property
    def passes(self) -> bool:
        """Alias for heat_pass; matches the `passes` convention of
        leverage_envelope / calmar promotion / dd_throttle results.
        """
        return self.heat_pass


def compute_portfolio_heat(
    positions: Iterable[PositionHeat],
    equity: float,
) -> float:
    """`sum(risk_amount_i) / equity`. Returns 0.0 if equity <= 0.

    Negative `risk_amount` values are silently clamped to zero
    (defensive against malformed inputs; matches leverage_envelope's
    `compute_gross_leverage` convention).
    """
    if equity <= 0.0:
        return 0.0
    total = sum(max(0.0, p.risk_amount) for p in positions)
    return total / equity


def evaluate_heat_envelope(
    positions: Iterable[PositionHeat],
    equity: float,
    candidate: PositionHeat | None = None,
    *,
    regime_normal: bool = True,
    normal_cap: float = DEFAULT_NORMAL_HEAT_CAP,
    crisis_cap: float = DEFAULT_CRISIS_HEAT_CAP,
) -> HeatCheckResult:
    """Evaluate the V3.8 §4.2 portfolio-heat cap with §4.6 C3 regime adjustment.

    Pass `candidate` to evaluate "what if I added this trade?" -- the
    candidate's risk_amount is INCLUDED in the heat sum. Pass
    `candidate=None` to evaluate the current portfolio's compliance.

    Parameters
    ----------
    positions:
        Iterable of open `PositionHeat` records.
    equity:
        Account equity in the same currency as `risk_amount`. Returned
        portfolio_heat is 0.0 if equity <= 0.
    candidate:
        Optional proposed new position. When evaluating a PRM pre-trade
        hook, pass the candidate here.
    regime_normal:
        True for normal regime (8% cap), False for crisis (6% cap) per
        §4.6 C3. Detection lives in `regime_filter/regime.py`.
    normal_cap, crisis_cap:
        Override defaults if needed.

    Returns:
    -------
    `HeatCheckResult` with portfolio_heat, applied cap, per-symbol R map,
    and PASS/FAIL with diagnostic reason on failure.
    """
    pos_list = list(positions)
    if candidate is not None:
        pos_list = pos_list + [candidate]

    heat_cap = normal_cap if regime_normal else crisis_cap
    heat = compute_portfolio_heat(pos_list, equity)
    heat_pass = heat <= heat_cap

    per_position_r = {p.symbol: max(0.0, p.risk_amount) for p in pos_list}
    total_r = sum(per_position_r.values())

    reasons: list[str] = []
    if not heat_pass:
        reasons.append(
            f"Portfolio heat {heat * 100:.2f}% above cap {heat_cap * 100:.2f}% "
            f"({'crisis' if not regime_normal else 'normal'} regime)"
        )

    return HeatCheckResult(
        portfolio_heat=heat,
        heat_cap=heat_cap,
        heat_pass=heat_pass,
        per_position_r=per_position_r,
        total_r=total_r,
        reasons=tuple(reasons),
    )


def would_candidate_breach_heat(
    current_positions: Iterable[PositionHeat],
    equity: float,
    candidate: PositionHeat,
    *,
    regime_normal: bool = True,
    normal_cap: float = DEFAULT_NORMAL_HEAT_CAP,
    crisis_cap: float = DEFAULT_CRISIS_HEAT_CAP,
) -> bool:
    """Convenience wrapper for the PRM pre-trade hook.

    Returns True iff adding `candidate` would push portfolio heat above
    the applicable cap. The PRM uses this to reject the trade before
    submission (per V3.8 rule V3.8-2).
    """
    res = evaluate_heat_envelope(
        current_positions,
        equity,
        candidate=candidate,
        regime_normal=regime_normal,
        normal_cap=normal_cap,
        crisis_cap=crisis_cap,
    )
    return not res.heat_pass


__all__ = [
    "DEFAULT_NORMAL_HEAT_CAP",
    "DEFAULT_CRISIS_HEAT_CAP",
    "PositionHeat",
    "HeatCheckResult",
    "compute_portfolio_heat",
    "evaluate_heat_envelope",
    "would_candidate_breach_heat",
]
