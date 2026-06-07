"""Return + volatility hygiene primitives (audit P1-22).

Specified in directives/Audit Remediation Plan 2026-05-29.md row P1-22
(finding H24): *"Drop ffill-before-returns; model the CL settle explicitly
(cap/model, not phantom flat day); exclude pure-missing dates from the vol
denominator; keep cash-day fillna(0.0) (that part is correct)."*

The bug class
=============

Two distinct kinds of "zero return" get silently conflated:

  * **Cash day** -- the strategy is deliberately flat, so it genuinely earns
    0% that bar. This 0.0 is REAL and must stay in the vol/Sharpe denominator.
  * **Pure-missing day** -- the instrument did not print (holiday, illiquid
    session, a masked anomaly like the 2020-04-20 CL -$37.63 settle). There is
    no return; it is undefined.

Forward-filling prices BEFORE differencing turns a pure-missing day into a
phantom 0% "flat day". Those phantom zeros inflate the denominator and deflate
realised vol -- which flatters Sharpe and understates drawdown probabilities,
exactly the optimism the audit flagged.

The fix (this module)
======================

  1. ``mask_nonpositive_prices`` -- model non-positive prints (CL settle) as
     MISSING (NaN), never a tradable price or a phantom flat day.
  2. ``returns_from_prices`` -- difference WITHOUT forward-filling
     (``pct_change(fill_method=None)`` / log of the same), so a missing price
     yields a NaN return, not a phantom 0%.
  3. ``realised_vol`` / ``annualise_vol`` -- drop NaN (pure-missing) from the
     denominator, but KEEP genuine 0.0 cash-day returns.
  4. ``mark_pure_missing`` -- the bridge for strategies: set returns to NaN on
     pure-missing dates (where the underlying had no real print) while leaving
     cash-day 0.0 returns intact, so the two are no longer fungible.

Adoption note
=============

Wiring this into the LIVE ewmac_regime audit (research/ewmac/run_b2e_audit.py)
+ re-freezing data/i1v2_c6_frozen.json changes that deployed strategy's verdict
numbers and its hash-locked artefact (P1-23), so it is an operator-gated
re-audit step, NOT applied here. This module is the validated primitive that
re-audit adopts; the signal path (EWMA needs continuous prices) stays separate
from the return/vol path (which must exclude pure-missing dates).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def mask_nonpositive_prices(prices: pd.Series) -> pd.Series:
    """Model non-positive prints as MISSING (NaN), not phantom flat days.

    The canonical case is the 2020-04-20 CL -$37.63 settle: it is neither a
    tradable price nor a 0% day, so masking it to NaN makes the returns into
    and out of it undefined (and thus excluded from the vol denominator) rather
    than forward-filled into a fake flat day. No forward-fill is applied.
    """
    p = prices.astype(float)
    return p.where(p > 0.0)


def returns_from_prices(prices: pd.Series, *, kind: str = "log") -> pd.Series:
    """Per-bar returns WITHOUT forward-filling prices first.

    Uses ``pct_change(fill_method=None)`` (or the log of the same), so a missing
    or masked price produces a NaN return instead of a phantom 0% "flat day".
    This is the "drop ffill-before-returns" half of P1-22.

    Parameters
    ----------
    prices:
        Price series (mask anomalies first via ``mask_nonpositive_prices``).
    kind:
        ``"log"`` (default) -> ``log(p_t / p_{t-1})``; ``"simple"`` ->
        ``p_t / p_{t-1} - 1``.
    """
    p = prices.astype(float)
    if kind == "simple":
        return p.pct_change(fill_method=None)
    if kind == "log":
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.log(p / p.shift(1))
        return r.replace([np.inf, -np.inf], np.nan)
    raise ValueError(f"kind must be 'log' or 'simple', got {kind!r}")


def mark_pure_missing(returns: pd.Series, valid_mask: pd.Series) -> pd.Series:
    """Set returns to NaN on pure-missing dates, leaving cash-day 0.0 intact.

    ``valid_mask`` is True where the underlying instrument actually printed
    (so the bar's 0.0 -- if any -- is a genuine cash-day return) and False on
    pure-missing dates (no print). The False dates become NaN so they drop out
    of the vol/Sharpe denominator; the True dates (including real 0.0s) are
    untouched. This is the cash-vs-missing partition the audit calls for.
    """
    out = returns.astype(float).copy()
    mask = valid_mask.reindex(out.index).fillna(False).astype(bool)
    out[~mask] = np.nan
    return out


def realised_vol(returns: pd.Series | np.ndarray, periods_per_year: int, *, ddof: int = 1) -> float:
    """Annualised realised volatility that EXCLUDES pure-missing dates.

    NaN returns (pure-missing) are dropped from the denominator; genuine 0.0
    cash-day returns are kept. Returns ``nan`` if fewer than two finite
    observations remain.
    """
    r = pd.Series(returns, dtype=float).dropna()
    if len(r) < 2:
        return float("nan")
    return float(r.std(ddof=ddof) * np.sqrt(periods_per_year))


def n_vol_observations(returns: pd.Series | np.ndarray) -> int:
    """Count of finite returns that enter the vol denominator (post-dropna).

    Diagnostic counterpart to :func:`realised_vol` -- a phantom-filled series
    reports MORE observations than a hygienic one, which is the tell-tale of
    deflated vol.
    """
    return int(pd.Series(returns, dtype=float).dropna().shape[0])
