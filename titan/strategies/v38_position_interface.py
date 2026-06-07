"""V3.8 position-visibility Protocol + helper functions.

Per `directives/PRM Integration V3.8 2026-05-24.md` §5. Live strategies
that opt into the V3.8 envelope expose their open positions via two
methods so the PortfolioRiskManager can poll them on every bar without
caching state (avoids the L73 cutover-diff failure mode).

Strategies that do NOT opt in (the V3.7 LIVE roster currently) need no
modification -- the PRM uses `get_heat_from(strategy)` and
`get_snapshots_from(strategy)` helpers which fall back to empty lists
when the strategy doesn't implement the Protocol. This avoids touching
any existing strategy file while keeping the interface explicit for
new V3.8-compliant strategies (Sleeve B GEM-futures-wrapper, etc.).

Design choice: Protocol over abstract base class
================================================

Nautilus strategies subclass `nautilus_trader.trading.strategy.Strategy`
directly -- there is no Titan-specific base class to extend. Adding one
would require modifying every existing strategy file. Instead this
module defines a `typing.Protocol` that strategies implement informally
(duck-typed). PRM-side code uses the helper functions which check via
`hasattr` -- O(1), no runtime isinstance cost, no inheritance machinery.

The Protocol is `@runtime_checkable` so tests can `isinstance(strategy,
V38PositionVisible)` if explicit verification is needed.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from titan.research.framework.leverage_envelope import PositionSnapshot
from titan.research.framework.portfolio_heat import PositionHeat


@runtime_checkable
class V38PositionVisible(Protocol):
    """Optional interface for strategies that opt into V3.8 envelope.

    Strategies that do not implement this Protocol are NOT subject to the
    V3.8 envelope -- the PRM helper functions below return empty lists
    for them, which makes the envelope checks pass trivially (sum = 0
    <= any cap).
    """

    def get_open_position_heat(self) -> list[PositionHeat]:
        """Returns the strategy's open positions as `PositionHeat` records.

        Each `PositionHeat.risk_amount` is the maximum nominal loss at
        stop fill in the account base currency, computed by the strategy
        from entry / stop / units / fx_rate.
        """
        ...

    def get_open_position_snapshots(self) -> list[PositionSnapshot]:
        """Returns the strategy's open positions as `PositionSnapshot` records.

        Each snapshot includes abs_notional + optional margin posted /
        exchange-minimum margin for the SPAN buffer check.
        """
        ...


def build_v38_candidate(
    symbol: str,
    notional: float,
    *,
    stop_distance_pct: float = 0.05,
    fx_rate: float = 1.0,
) -> tuple[PositionHeat, PositionSnapshot]:
    """Build the (PositionHeat, PositionSnapshot) candidate for a pending order
    from its notional + stop distance (audit P0-3 -- the shared builder the
    demo_set strategies pass to ``check_pre_trade`` in shadow mode).

    ``risk_amount = |notional| * stop_distance_pct * fx_rate`` -- the nominal
    loss at the stop. Pass the strategy's REAL stop fraction where it has one
    (turtle / ic_equity: ``stop_dist / price``); a vol-targeted strategy with
    no per-position stop (GEM, like Sleeve B) uses the 5% notional heuristic.
    ``abs_notional`` is the order's base-currency exposure. SPAN/margin is
    omitted (None) -- skipped for these cash/ETF-style legs.
    """
    abs_notional = abs(notional) * fx_rate
    heat = PositionHeat(symbol=symbol, risk_amount=abs_notional * stop_distance_pct)
    snapshot = PositionSnapshot(symbol=symbol, abs_notional=abs_notional)
    return heat, snapshot


def position_records(
    symbol: str,
    qty: float,
    price: float,
    *,
    stop_price: float | None = None,
    atr_stop: float | None = None,
    heuristic_pct: float = 0.05,
) -> tuple[PositionHeat, PositionSnapshot]:
    """Build (PositionHeat, PositionSnapshot) for ONE OPEN position (audit P0-5).

    The shared per-position record builder the demo_set strategies use to
    implement the V38PositionVisible protocol, so the PRM sees their real open
    exposure (not just the candidate). ``abs_notional = |qty| * price``; the
    stop risk is, in priority:
      * ``|price - stop_price| * |qty|`` -- an explicit stop price (turtle);
      * ``atr_stop * |qty|``             -- an ATR-derived stop distance (ic_equity);
      * ``heuristic_pct * notional``     -- vol-targeted, no stop (GEM).
    """
    aq = abs(qty)
    notional = aq * price
    if stop_price is not None and price > 0:
        risk = abs(price - stop_price) * aq
    elif atr_stop is not None:
        risk = abs(atr_stop) * aq
    else:
        risk = notional * heuristic_pct
    return PositionHeat(symbol=symbol, risk_amount=risk), PositionSnapshot(
        symbol=symbol, abs_notional=notional
    )


def get_heat_from(strategy: Any) -> list[PositionHeat]:
    """Returns the strategy's `PositionHeat` list, or empty if the
    strategy doesn't implement the V3.8 position-visibility Protocol.

    Used by the PRM's `on_bar` and `check_pre_trade` to evaluate the
    portfolio-heat envelope without requiring every strategy to be
    V3.8-aware.
    """
    method = getattr(strategy, "get_open_position_heat", None)
    if method is None:
        return []
    result = method()
    if result is None:
        return []
    return list(result)


def get_snapshots_from(strategy: Any) -> list[PositionSnapshot]:
    """Returns the strategy's `PositionSnapshot` list, or empty if the
    strategy doesn't implement the V3.8 position-visibility Protocol.

    Used by the PRM's `on_bar` and `check_pre_trade` to evaluate the
    leverage envelope.
    """
    method = getattr(strategy, "get_open_position_snapshots", None)
    if method is None:
        return []
    result = method()
    if result is None:
        return []
    return list(result)


__all__ = [
    "V38PositionVisible",
    "build_v38_candidate",
    "position_records",
    "get_heat_from",
    "get_snapshots_from",
]
