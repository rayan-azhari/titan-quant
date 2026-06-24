"""Shared position-sizing helpers with a hard leverage ceiling.

Consolidates the vol-target -> portfolio-overlay -> integer-units pipeline that
was copy-pasted across the strategy layer. Every copy applied the
``max_leverage`` cap BEFORE multiplying by the allocator weight and the PRM
scale factor::

    notional = equity * (vol_target / ann_vol)
    notional = min(notional, equity * max_leverage)   # cap applied here ...
    notional *= alloc * scale_factor                  # ... then breached here

so the cap never bounded the *final* notional. With the correlation dial pinned
at its 1.5x ceiling (``alloc`` can exceed 1.0), the deployed notional could
reach ``book * max_leverage * 1.5`` -- 50% above the advertised ceiling. See the
2026-06-23 sizing/margin investigation (finding SIZE-1 / ALLOC-1) and
``directives/Sizing-Margin-Leverage Remediation 2026-06-23.md``.

This module enforces the missing invariant: the leverage cap is applied LAST, as
the outermost clamp, so ``book_equity * max_leverage`` is a TRUE ceiling
regardless of the allocator weight, the correlation dial, or the scale factor.

Pure functions with no NautilusTrader / IO / singleton dependencies, so the
sizing invariant is unit-testable in isolation (tests/test_sizing_helper.py).
"""

from __future__ import annotations


def bounded_target_notional(
    *,
    book_equity: float,
    vol_target_pct: float,
    ann_vol: float,
    max_leverage: float,
    alloc_weight: float,
    scale_factor: float,
) -> float:
    """Vol-targeted gross notional, hard-capped at ``book_equity * max_leverage``.

    Parameters
    ----------
    book_equity:
        Per-strategy capital base in the portfolio base currency (USD). This is
        the ceiling reference -- the returned notional never exceeds
        ``book_equity * max_leverage``.
    vol_target_pct:
        Annualised volatility target (e.g. ``0.10`` for 10%).
    ann_vol:
        Annualised realised volatility of the instrument.
    max_leverage:
        Per-book leverage cap. The returned notional is hard-clamped to
        ``book_equity * max_leverage`` AFTER all overlays.
    alloc_weight:
        Gross overlay multiplier, expected in [0, 1] -- the correlation-dial
        scalar clamped <=1.0 (P1.3 / SIZE-4). The cross-sleeve inverse-vol tilt
        is expressed in ``book_equity`` (auto-equity partitions NLV once), NOT
        re-applied here -- passing the allocator's inverse-vol fraction would
        double-count the partition. Negative values are floored at 0.
    scale_factor:
        PortfolioRiskManager composite throttle in [0, 1+]. Negative values are
        floored at 0.

    Returns:
    -------
    Target gross notional in base currency, ``>= 0`` and
    ``<= book_equity * max_leverage``. Returns ``0.0`` on degenerate inputs
    (non-positive book, vol, or leverage).
    """
    if book_equity <= 0.0 or ann_vol <= 0.0 or max_leverage <= 0.0:
        return 0.0

    ceiling = book_equity * max_leverage

    # 1. Vol-target notional, bounded by the per-book leverage cap.
    notional = book_equity * (vol_target_pct / ann_vol)
    notional = min(notional, ceiling)

    # 2. Portfolio overlays: inverse-vol weight x correlation dial, PRM throttle.
    notional *= max(0.0, alloc_weight) * max(0.0, scale_factor)

    # 3. HARD CEILING applied LAST. The overlays in step 2 can push notional
    #    above the cap (the correlation dial alone is up to 1.5x); clamping here
    #    -- after every multiplier -- makes book_equity * max_leverage a true
    #    upper bound. This is the invariant the pre-fix per-strategy code lacked.
    notional = min(notional, ceiling)

    return max(0.0, notional)


def bounded_target_units(price: float, **kwargs: float) -> int:
    """Integer units from :func:`bounded_target_notional`, floored at 0.

    ``price`` is the instrument price in its quote currency (callers pass a
    USD-quoted price for USD sleeves). Remaining keyword arguments are forwarded
    verbatim to :func:`bounded_target_notional`. Returns ``0`` when ``price`` is
    non-positive.
    """
    if price <= 0.0:
        return 0
    return max(0, int(bounded_target_notional(**kwargs) / price))


def submit_unit_ceiling(book_equity: float, price: float, max_leverage: float) -> int:
    """Max integer units a sleeve may hold or submit: ``floor(book*max_leverage/price)``.

    The absolute per-sleeve ceiling, used as (a) a submit-time clamp on new
    entries (P1.4) and (b) the trim target for the size-drift band on held
    positions (P2.2). It is independent of HOW a unit count was computed, so it
    backstops any sizing path -- a bug that bypasses :func:`bounded_target_units`
    still cannot push a sleeve above ``book*max_leverage`` at the submit price.
    Returns ``0`` on degenerate inputs.
    """
    if book_equity <= 0.0 or price <= 0.0 or max_leverage <= 0.0:
        return 0
    return int((book_equity * max_leverage) / price)


def open_position_mtm(positions, last_price: float) -> float:
    """Unrealized P&L (base ccy) of USD-quoted open positions at ``last_price``.

    Strategies must call ``StrategyEquityTracker.set_mtm`` each bar with this
    value so ``current_equity()`` reflects open P&L, not just realized -- without
    it a multi-day hold that draws down is invisible to the PRM drawdown / kill
    switch and to the sizing book until the position closes (2026-06-23 audit,
    CCY-6).

    Parameters
    ----------
    positions:
        Iterable of ``(signed_qty, avg_px_open)`` pairs. For USD-quoted
        instruments unrealized P&L is already in base ccy (fx 1.0). A short
        position (negative ``signed_qty``) is handled by the sign.
    last_price:
        Latest traded-instrument price in base ccy.

    Returns:
    -------
    Total unrealized P&L in base ccy; ``0.0`` for a flat/empty book.
    """
    total = 0.0
    for signed_qty, avg_px_open in positions:
        total += float(signed_qty) * (float(last_price) - float(avg_px_open))
    return total
