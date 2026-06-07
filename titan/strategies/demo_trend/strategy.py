"""A simple, educational moving-average trend rule.

This is a teaching aid for the titan-quant framework. It has NO expected edge and
must never be traded with real money. Its only jobs are:

1. Provide a causal, shift-disciplined signal the validation framework can grade.
2. Show the canonical signal shape the live integration contract wraps
   (see ``live_contract.py`` and the book, Part IV).

The signal is long when the fast MA is above the slow MA, flat otherwise. It is
*long-only* and *shift-disciplined*: the position decided at the close of bar t
earns the return from t -> t+1. See the book chapter "A backtest you can trust".
"""

from __future__ import annotations

import pandas as pd


def demo_trend_signal(close: pd.Series, fast: int = 20, slow: int = 100) -> pd.Series:
    """Raw target position in {0, 1} decided AT each bar's close (not yet lagged).

    Long (1) when the fast SMA is above the slow SMA, else flat (0). The caller is
    responsible for lagging this before multiplying by a return (see
    :func:`demo_trend_positions`), so that a decision never earns its own bar.
    """
    if fast >= slow:
        raise ValueError(f"fast ({fast}) must be < slow ({slow})")
    fast_ma = close.rolling(fast, min_periods=fast).mean()
    slow_ma = close.rolling(slow, min_periods=slow).mean()
    return (fast_ma > slow_ma).astype(float)


def demo_trend_positions(close: pd.Series, fast: int = 20, slow: int = 100) -> pd.Series:
    """Tradable, LAGGED positions: the signal at close t applied to the t -> t+1 bar.

    This is the series you multiply by forward returns. The ``.shift(1)`` is the
    whole point: without it the backtest would let a decision earn the very bar it
    was computed on (look-ahead). See the failure-mode catalogue in the book.
    """
    return demo_trend_signal(close, fast=fast, slow=slow).shift(1).fillna(0.0)
