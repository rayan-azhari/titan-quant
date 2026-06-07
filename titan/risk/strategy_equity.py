"""Per-strategy equity tracking and FX unit conversion.

Fixes three concrete defects in the previous live-risk wiring:

1. Strategies used ``account.balance_total(list(balances.keys())[0])`` which
   returned the *whole account* net liquidation in a non-deterministic currency.
   Every strategy therefore fed the PortfolioRiskManager an identical equity
   series, collapsing the inverse-vol allocator and correlation gate.

2. ``ccys[0]`` silently picked whichever currency sat first in the dict. On a
   multi-currency IBKR account that could be USD / JPY / EUR across restarts.

3. FX strategies computed ``notional_usd / price`` where ``price`` was quoted
   in a non-USD currency (e.g., JPY-per-AUD for AUD/JPY), producing garbage
   unit counts.

Design
------
Each strategy instantiates one ``StrategyEquityTracker`` in ``on_start``. The
tracker owns:

* ``initial_equity`` -- the notional seed capital assigned to this strategy
  (expressed in the portfolio base currency, typically USD).
* ``realized_pnl`` -- cumulative realised P&L accumulated on
  ``on_position_closed`` events, converted to base currency.
* Optional mark-to-market helper that values open positions at a last price
  in base currency.

The tracker exposes ``current_equity()`` which is what the strategy passes to
``portfolio_risk_manager.update``. This gives the risk manager a true
per-strategy equity stream.

For FX unit sizing the helper ``convert_notional_to_units`` handles the three
quote conventions we use:

* USD-quoted equities/ETFs: trivial ``units = notional_usd / price``.
* ``AUD/USD`` style pairs (quote ccy == account ccy): same formula.
* ``AUD/JPY`` style pairs (quote ccy != account ccy): convert notional to the
  base ccy of the pair using an explicit FX rate supplied by the strategy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from decimal import Decimal

logger = logging.getLogger(__name__)

# Portfolio base currency. All per-strategy equity is accounted in this.
DEFAULT_BASE_CCY = "USD"


@dataclass
class StrategyEquityTracker:
    """Per-strategy equity ledger.

    Parameters
    ----------
    prm_id:
        Portfolio-risk-manager identifier for this strategy.
    initial_equity:
        Seed capital in base currency.
    base_ccy:
        Portfolio accounting currency (default "USD").
    """

    prm_id: str
    initial_equity: float
    base_ccy: str = DEFAULT_BASE_CCY
    realized_pnl_base: float = 0.0
    # Last observed mark-to-market notional of open positions in base ccy.
    mtm_base: float = 0.0
    # Per-instrument running quantities, used when strategies prefer to
    # compute MTM themselves from bar closes without fetching unrealized P&L.
    _open_positions: dict[str, tuple[float, float]] = field(default_factory=dict)

    def on_position_closed(self, realized_pnl: float, fx_to_base: float = 1.0) -> None:
        """Record a closed position's realised P&L.

        ``realized_pnl`` is in the instrument's quote currency; ``fx_to_base``
        is the multiplicative rate to convert one unit of that currency into
        base currency. For USD-quoted instruments use ``1.0``.
        """
        self.realized_pnl_base += realized_pnl * fx_to_base

    def set_mtm(self, mtm_base: float) -> None:
        """Set the current mark-to-market of open positions in base currency."""
        self.mtm_base = float(mtm_base)

    def current_equity(self) -> float:
        """Total strategy equity = seed + realised + MTM of open positions."""
        return self.initial_equity + self.realized_pnl_base + self.mtm_base


# ── Currency / FX helpers ─────────────────────────────────────────────────────


def get_base_balance(account, base_ccy: str = DEFAULT_BASE_CCY) -> float | None:
    """Fetch account balance in an explicit base currency.

    Returns ``None`` if the account has no balance in that ccy -- callers
    must fall back to their tracker's internal equity instead of picking an
    arbitrary currency via ``list(balances.keys())[0]`` (which is the bug
    this helper exists to prevent).
    """
    if account is None:
        return None
    try:
        balances = account.balances()
    except Exception:
        return None

    # Match by string form of the currency key; NautilusTrader uses Currency
    # objects that compare via their ISO code.
    for ccy, _ in balances.items():
        if str(ccy) == base_ccy:
            try:
                return float(account.balance_total(ccy).as_double())
            except Exception:
                return None
    return None


def convert_notional_to_units(
    notional_base: float,
    price: float,
    *,
    quote_ccy: str = DEFAULT_BASE_CCY,
    base_ccy: str = DEFAULT_BASE_CCY,
    fx_rate_quote_to_base: float | None = None,
) -> int:
    """Convert a base-currency notional into instrument units.

    Parameters
    ----------
    notional_base:
        Desired exposure in base currency (usually USD).
    price:
        Quoted price of the instrument (quote ccy per 1 unit of instrument).
    quote_ccy:
        Currency in which ``price`` is quoted. For USD-quoted stocks this is
        ``"USD"``. For ``AUD/JPY`` this is ``"JPY"``.
    base_ccy:
        Portfolio accounting currency.
    fx_rate_quote_to_base:
        Multiplicative rate: 1 unit of ``quote_ccy`` = this many units of
        ``base_ccy``. Required when ``quote_ccy != base_ccy``. For JPY->USD
        pass ~0.0067. Pass ``None`` to force a ValueError if the caller
        forgot -- we refuse to silently assume 1.0.

    Returns:
    -------
    Integer number of instrument units. Zero if any input is invalid.
    """
    if price <= 0 or notional_base <= 0:
        return 0

    if quote_ccy == base_ccy:
        return int(notional_base / price)

    if fx_rate_quote_to_base is None or fx_rate_quote_to_base <= 0:
        raise ValueError(
            f"convert_notional_to_units: quote_ccy={quote_ccy!r} != "
            f"base_ccy={base_ccy!r} so fx_rate_quote_to_base is required. "
            f"Pass the rate explicitly -- never assume 1.0."
        )

    # notional_base expressed in quote ccy = notional_base / fx_rate_quote_to_base
    notional_quote = notional_base / fx_rate_quote_to_base
    return int(notional_quote / price)


def split_fx_pair(instrument_symbol: str) -> tuple[str, str] | None:
    """Split "AUD/JPY" into ("AUD", "JPY"). Returns None if not a pair."""
    if "/" not in instrument_symbol:
        return None
    left, right = instrument_symbol.split("/", 1)
    # Strip venue suffix like "AUD/JPY.IDEALPRO" -> "AUD", "JPY"
    right = right.split(".", 1)[0]
    return left.upper(), right.upper()


def d(value: float | str) -> Decimal:
    """Shorthand for Decimal construction (used by size rounding helpers)."""
    return Decimal(str(value))


# ── Migration-friendly drop-in replacement ───────────────────────────────────


def report_equity_and_check(
    strategy,
    prm_id: str,
    bar,
    *,
    tracker: "StrategyEquityTracker | None" = None,
    fallback_account_ccy: str = DEFAULT_BASE_CCY,
) -> tuple[float, bool]:
    """Standard per-bar call that every strategy's on_bar uses.

    Replaces the previous ``balance_total(ccys[0])`` anti-pattern everywhere.
    Returns ``(equity, halted)`` where ``halted`` is the portfolio halt flag
    the caller should branch on.

    Resolution order for equity:
      1. ``tracker.current_equity()`` if the strategy has wired one up
         (preferred -- gives true per-strategy equity).
      2. ``get_base_balance(account, fallback_account_ccy)`` -- account NLV
         in a *deterministic* base currency.
      3. ``None`` falls through as 0.0 (strategy should skip the bar).
    """
    from titan.risk.portfolio_risk_manager import portfolio_risk_manager

    equity: float | None = None
    if tracker is not None:
        equity = tracker.current_equity()

    if equity is None or equity <= 0:
        accounts = strategy.cache.accounts() if hasattr(strategy, "cache") else []
        if accounts:
            equity = get_base_balance(accounts[0], fallback_account_ccy)

    equity_val = float(equity) if equity is not None else 0.0

    # Pass raw nanosecond epoch -- the PRM converts it to a UTC Timestamp.
    ts = getattr(bar, "ts_event", None) if bar is not None else None

    if equity_val > 0:
        portfolio_risk_manager.update(prm_id, equity_val, ts=ts)

    return equity_val, portfolio_risk_manager.halt_all
