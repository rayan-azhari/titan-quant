"""Cost-realism primitives for audit re-runs (audit P1-12 / P1-13 / P1-14).

Turn a GROSS strategy return series into a net-of-cost series and re-test the
deployment gate at realistic cost. Three cost components, each a pure function
over a return series so any audit/sleeve can apply the relevant ones:

  * ``apply_cost_model``    -- per-period transaction cost = turnover x the
    typology ``CostModel`` spread+slip bps (P1-12);
  * ``apply_roll_cost``     -- a fixed full-notional cost on each futures roll
    date (P1-13; ~5 bps/roll, independent of signal turnover);
  * ``apply_sqrt_impact_slippage`` -- state-dependent slippage in the
    sqrt-market-impact form of ``titan.models.spread.estimate_slippage`` (P1-14).

``realistic_cost_gate`` then re-runs the bootstrap-Sharpe CI on the net series
and reports whether ``CI_lo > 0`` still holds at the realistic cost point -- the
P1-12 requirement that a deployed edge survive realistic cost, not just gross.

Turnover convention: ONE-WAY notional fraction traded per period (annual
turnover 1.0 = the book traded through once one-way per year). A full round trip
is turnover 2.0, costing ``CostModel.round_trip_bps_no_commission`` -- consistent
with the typology definition. Commission (absolute USD/side in ``CostModel``) is
notional-dependent; pass it as an explicit bps via ``commission_bps`` if material.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from titan.research.framework.typology import CostModel
from titan.research.metrics import bootstrap_sharpe_ci, sharpe


def apply_cost_model(
    returns: pd.Series,
    turnover: float | pd.Series,
    cost_model: CostModel,
    *,
    commission_bps: float = 0.0,
) -> pd.Series:
    """Net-of-cost returns: subtract ``turnover * (spread+slip+commission)`` bps
    each period. ``turnover`` is the one-way notional fraction (scalar or a
    per-period Series aligned to ``returns``).
    """
    cost_bps = cost_model.spread_bps + cost_model.slip_bps + commission_bps
    if isinstance(turnover, pd.Series):
        turnover = turnover.reindex(returns.index).fillna(0.0)
    cost = (turnover * cost_bps) / 1e4
    return returns - cost


def apply_roll_cost(
    returns: pd.Series,
    roll_dates: Iterable,
    *,
    bps_per_roll: float = 5.0,
) -> pd.Series:
    """Subtract a fixed full-notional cost on each roll date (P1-13). Independent
    of signal turnover -- a futures position pays the calendar-spread + double
    commission every time the contract rolls (~5 bps/roll).
    """
    out = returns.copy()
    if len(out) == 0:
        return out
    roll_idx = pd.DatetimeIndex(list(roll_dates))
    hit = out.index.intersection(roll_idx)
    out.loc[hit] = out.loc[hit] - (bps_per_roll / 1e4)
    return out


def apply_sqrt_impact_slippage(
    returns: pd.Series,
    turnover: float | pd.Series,
    participation: float | pd.Series,
    *,
    coef: float = 1e-4,
) -> pd.Series:
    """State-dependent slippage in the sqrt-market-impact form used by
    ``titan.models.spread.estimate_slippage`` (slip = coef*sqrt(participation),
    participation = order size / ADV). Cost = turnover * slip each period (P1-14).
    """
    part = participation
    if isinstance(part, pd.Series):
        part = part.reindex(returns.index).fillna(0.0)
    slip = coef * np.sqrt(np.clip(part, 0.0, None))
    if isinstance(turnover, pd.Series):
        turnover = turnover.reindex(returns.index).fillna(0.0)
    return returns - turnover * slip


@dataclass(frozen=True)
class CostSensitivity:
    """Gross vs net-of-cost deployment-gate result."""

    gross_sharpe: float
    net_sharpe: float
    net_ci_lo: float
    net_ci_hi: float
    annual_cost_drag: float  # annualised return lost to cost
    passes: bool  # net 95% CI_lo > 0 (the P1-12 realistic-cost requirement)


def realistic_cost_gate(
    returns: pd.Series,
    turnover: float | pd.Series,
    cost_model: CostModel,
    *,
    periods_per_year: int,
    commission_bps: float = 0.0,
    n_resamples: int = 1000,
    seed: int = 42,
) -> CostSensitivity:
    """Re-run the bootstrap-Sharpe CI on the net-of-cost series; pass requires
    ``CI_lo > 0`` at the realistic cost point (P1-12).
    """
    net = apply_cost_model(returns, turnover, cost_model, commission_bps=commission_bps)
    ci_lo, ci_hi = bootstrap_sharpe_ci(
        net.dropna(), periods_per_year, n_resamples=n_resamples, seed=seed
    )
    drag = float((returns - net).mean() * periods_per_year)
    return CostSensitivity(
        gross_sharpe=float(sharpe(returns.dropna(), periods_per_year)),
        net_sharpe=float(sharpe(net.dropna(), periods_per_year)),
        net_ci_lo=float(ci_lo),
        net_ci_hi=float(ci_hi),
        annual_cost_drag=drag,
        passes=bool(ci_lo > 0),
    )


@dataclass(frozen=True)
class CostReconciliation:
    """Realised-vs-audit cost comparison for one live strategy (P1-16)."""

    strategy: str
    audit_bps: float  # round-trip bps assumed in the promotion audit
    realised_bps: float  # round-trip bps observed live (from IBKR fills)
    ratio: float  # realised / audit
    alarm: bool  # ratio > alarm_ratio -> the audit understated real cost


def reconcile_cost(
    strategy: str,
    audit_bps: float,
    realised_bps: float,
    *,
    alarm_ratio: float = 1.5,
) -> CostReconciliation:
    """Compare a strategy's realised round-trip cost (from live fills) against
    the cost assumed in its promotion audit; alarm when realised exceeds the
    audit by more than ``alarm_ratio`` (default 1.5x) -- the standing monthly
    cost-reconciliation check (P1-16). A strategy promoted on optimistic cost
    that trades dearer live has an inflated, unverified edge.
    """
    ratio = realised_bps / audit_bps if audit_bps > 0 else float("inf")
    return CostReconciliation(
        strategy=strategy,
        audit_bps=float(audit_bps),
        realised_bps=float(realised_bps),
        ratio=float(ratio),
        alarm=bool(ratio > alarm_ratio),
    )


def reconcile_costs(
    records: dict[str, tuple[float, float]],
    *,
    alarm_ratio: float = 1.5,
) -> list[CostReconciliation]:
    """Batch reconcile: ``{strategy: (audit_bps, realised_bps)}`` -> results."""
    return [
        reconcile_cost(name, audit, realised, alarm_ratio=alarm_ratio)
        for name, (audit, realised) in records.items()
    ]


__all__ = [
    "apply_cost_model",
    "apply_roll_cost",
    "apply_sqrt_impact_slippage",
    "CostSensitivity",
    "realistic_cost_gate",
    "CostReconciliation",
    "reconcile_cost",
    "reconcile_costs",
]
