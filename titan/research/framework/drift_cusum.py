"""CUSUM drift monitor for live PnL (V3.7).

Detects structural breaks in live strategy performance vs the OOS-stitched
baseline using a CUSUM (cumulative sum) test on standardised excess returns.

CUSUM is the standard SPC tool for detecting parameter drift: if a process
mean drifts beyond a control limit, the cumulative sum of deviations
crosses a threshold and the test triggers.

For trading strategies:
    - Compute z(t) = (live_ret(t) - oos_mean) / oos_std
    - S(t) = max(0, S(t-1) + z(t) - k)   (upper CUSUM, catches mean shifts UP)
    - S_neg(t) = max(0, S_neg(t-1) - z(t) - k)  (lower CUSUM, catches mean shifts DOWN)
    - Alert if S(t) > h or S_neg(t) > h

The relevant alert for live trading is the LOWER CUSUM (S_neg) — detecting
when live Sharpe is materially below research expectations. The UPPER
CUSUM is informational (live is BETTER than expected, which is suspicious
in a different way — possibly a methodology bug).

Defaults (Page 1954 standard):
    k = 0.5 (half a sigma per bar tolerance)
    h = 5.0 (~ 1% false-positive rate at 252 bars)

Reference:
    Page, E.S. 1954. "Continuous Inspection Schemes." Biometrika 41.
    Lopez de Prado 2018, "Advances in Financial Machine Learning",
    ch. 17 on structural breaks.

Usage:

    from titan.research.framework.drift_cusum import run_cusum_drift

    res = run_cusum_drift(
        live_returns=live_pnl_series,
        oos_returns=stitched_oos_returns,
    )
    if res.lower_breach:
        # Live strategy is underperforming research baseline
        flag_for_re_audit()
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class CusumResult:
    """CUSUM drift test output.

    Attributes:
        upper_cusum: latest upper-CUSUM value (catches mean shift UP).
        lower_cusum: latest lower-CUSUM value (catches mean shift DOWN).
        upper_breach: True if upper CUSUM exceeds threshold h.
        lower_breach: True if lower CUSUM exceeds threshold h (the
            operationally important alert for live trading).
        upper_path: full upper-CUSUM series.
        lower_path: full lower-CUSUM series.
        breach_bar: first bar where any CUSUM crossed h, or None.
        oos_mean: per-bar OOS sample mean (baseline).
        oos_std: per-bar OOS sample std (baseline).
        live_mean: per-bar live sample mean (observed).
        live_std: per-bar live sample std (observed).
        n_live: live observations.
        k_slack: tolerance (per bar in sigma units).
        h_threshold: alert threshold (in cumulative sigma units).
    """

    upper_cusum: float
    lower_cusum: float
    upper_breach: bool
    lower_breach: bool
    upper_path: pd.Series
    lower_path: pd.Series
    breach_bar: pd.Timestamp | None
    oos_mean: float
    oos_std: float
    live_mean: float
    live_std: float
    n_live: int
    k_slack: float
    h_threshold: float

    def report(self) -> str:
        breach_str = "─"
        if self.lower_breach:
            breach_str = f"LOWER BREACH at {self.breach_bar}"
        elif self.upper_breach:
            breach_str = f"UPPER BREACH at {self.breach_bar} (informational)"
        return (
            f"CusumDriftResult(n_live={self.n_live}, k={self.k_slack}, h={self.h_threshold})\n"
            f"  OOS baseline: mean={self.oos_mean:+.6f} std={self.oos_std:.6f}\n"
            f"  Live observed: mean={self.live_mean:+.6f} std={self.live_std:.6f}\n"
            f"  Upper CUSUM: {self.upper_cusum:.3f}  Lower CUSUM: {self.lower_cusum:.3f}\n"
            f"  Status: {breach_str}"
        )


def run_cusum_drift(
    live_returns: pd.Series,
    oos_returns: pd.Series,
    *,
    k_slack: float = 0.5,
    h_threshold: float = 5.0,
) -> CusumResult:
    """Run CUSUM drift test on live returns vs OOS baseline.

    Parameters:
        live_returns: per-bar returns observed live.
        oos_returns: per-bar returns from research OOS (the baseline).
        k_slack: tolerance per bar (sigma units). Default 0.5.
        h_threshold: alert threshold (cumulative sigma units). Default 5.0.

    Returns:
        CusumResult.
    """
    live = live_returns.dropna()
    oos = oos_returns.dropna()
    if len(oos) < 30:
        raise ValueError(f"OOS baseline too short: {len(oos)} bars")
    if len(live) == 0:
        # Return empty result.
        return CusumResult(
            upper_cusum=0.0,
            lower_cusum=0.0,
            upper_breach=False,
            lower_breach=False,
            upper_path=pd.Series(dtype=float),
            lower_path=pd.Series(dtype=float),
            breach_bar=None,
            oos_mean=float(oos.mean()),
            oos_std=float(oos.std(ddof=1)),
            live_mean=0.0,
            live_std=0.0,
            n_live=0,
            k_slack=k_slack,
            h_threshold=h_threshold,
        )
    oos_mean = float(oos.mean())
    oos_std = float(oos.std(ddof=1))
    if oos_std < 1e-12:
        raise ValueError("OOS std is ~zero; cannot compute CUSUM")

    z = (live - oos_mean) / oos_std
    upper = np.zeros(len(z))
    lower = np.zeros(len(z))
    breach_bar = None
    for i in range(len(z)):
        prev_u = upper[i - 1] if i > 0 else 0.0
        prev_l = lower[i - 1] if i > 0 else 0.0
        upper[i] = max(0.0, prev_u + z.iloc[i] - k_slack)
        lower[i] = max(0.0, prev_l - z.iloc[i] - k_slack)
        if breach_bar is None and (upper[i] > h_threshold or lower[i] > h_threshold):
            breach_bar = live.index[i]

    upper_path = pd.Series(upper, index=live.index, name="upper_cusum")
    lower_path = pd.Series(lower, index=live.index, name="lower_cusum")

    return CusumResult(
        upper_cusum=float(upper_path.iloc[-1]),
        lower_cusum=float(lower_path.iloc[-1]),
        upper_breach=bool(upper_path.max() > h_threshold),
        lower_breach=bool(lower_path.max() > h_threshold),
        upper_path=upper_path,
        lower_path=lower_path,
        breach_bar=breach_bar,
        oos_mean=oos_mean,
        oos_std=oos_std,
        live_mean=float(live.mean()),
        live_std=float(live.std(ddof=1)),
        n_live=len(live),
        k_slack=k_slack,
        h_threshold=h_threshold,
    )


__all__ = ["CusumResult", "run_cusum_drift"]
