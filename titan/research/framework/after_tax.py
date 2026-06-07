"""After-tax / wrapper-aware accounting for a UK retail operator (V3.8 Diagnostic A).

Pre-reg: the Wave-1 diagnostics directive, Diagnostic A. The program ranks on GROSS
Sharpe, but the operator's objective is after-tax wealth in the best UK wrapper. Runs
one annual P&L stream through four wrappers:

    ISA  -- tax-free; capped (GBP20k/yr); cash instruments only (no futures/shorts).
    SIPP -- tax-free like ISA; relief + age-57 lock are capital-flow effects (not in
            the return path -> sipp == isa here).
    GIA  -- taxable: CGT (annual realised gain + GBP3k exemption + loss carry-forward)
            + 0.5% SDRT on UK SHARE buys (not US/futures) + dividend tax above GBP500.
    SB   -- spread bet: CGT/income/stamp = 0 (HMRC BIM22020/CG56105) BUT losses are
            non-deductible and you pay IG overnight financing on the notional held.

Decisive wedges:
  * SB vs GIA = (CGT + stamp saved) - IG financing. Stamp (0.5% x turnover) dominates
    for high-turnover UK shares; for US/futures (no stamp) financing wins for GIA/ISA.
  * ISA dominates SB for unleveraged ISA-eligible cash -> SB only matters beyond the
    GBP20k/yr allowance, for leverage, or for non-ISA instruments (futures/shorts).
  * SB's non-deductible losses penalise neg-skew vs GIA -- see the carry-forward sim.

All functions take ANNUAL P&L in GBP + a capital base (GBP3k exemption is fixed cash).
UK 2024/25 params are the defaults.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WrapperParams:
    """UK 2024/25 tax + IG-financing parameters (override as policy changes)."""

    cgt_rate: float = 0.24  # higher-rate CGT on non-property gains (2024/25)
    cgt_annual_exemption: float = 3_000.0  # GBP annual exempt amount (2024/25)
    stamp_rate_uk_shares: float = 0.005  # 0.5% SDRT on UK share purchases (0 for US / futures)
    dividend_allowance: float = 500.0
    dividend_tax_rate: float = 0.3375  # higher-rate dividend tax
    sonia: float = 0.045  # ~current SONIA
    ig_financing_spread: float = 0.025  # IG overnight admin spread over SONIA (~2.5-3.0%)
    # DEALING costs per unit turnover (round-trip fraction of notional). Wrapper-specific: direct
    # dealing (ISA/SIPP/GIA) = commission + market half-spread + slippage; an IG spread bet pays
    # IG's EMBEDDED spread, WIDER than direct -> a turnover-scaled SB penalty.
    direct_cost_per_turnover: float = 0.0010  # ~10 bps RT, liquid UK/US shares direct
    sb_cost_per_turnover: float = 0.0020  # ~20 bps RT, IG SB embedded spread on liquid shares
    isa_annual_allowance: float = 20_000.0
    sipp_contribution_relief: float = (
        0.0  # one-off relief on capital IN (not modelled in the return path)
    )

    @property
    def ig_financing_rate(self) -> float:
        return self.sonia + self.ig_financing_spread


def isa_after_tax(
    annual_pnl: np.ndarray,
    annual_turnover_notional: np.ndarray,
    *,
    is_uk_shares: bool = False,
    params: WrapperParams = WrapperParams(),
) -> np.ndarray:
    """ISA: CGT/dividend-tax-free, minus DIRECT dealing AND stamp (SDRT applies to UK share buys
    in an ISA too -- only a spread bet avoids it). annual_pnl is PRE-dealing gross P&L in GBP.
    """
    pnl = np.asarray(annual_pnl, dtype=float)
    turn = np.asarray(annual_turnover_notional, dtype=float)
    stamp = params.stamp_rate_uk_shares * turn if is_uk_shares else 0.0
    return pnl - params.direct_cost_per_turnover * turn - stamp


def sipp_after_tax(
    annual_pnl: np.ndarray,
    annual_turnover_notional: np.ndarray,
    *,
    is_uk_shares: bool = False,
    params: WrapperParams = WrapperParams(),
) -> np.ndarray:
    """SIPP: tax-free like ISA here (incl. stamp); relief/lock not modelled."""
    return isa_after_tax(
        annual_pnl, annual_turnover_notional, is_uk_shares=is_uk_shares, params=params
    )


def gia_after_tax(
    annual_pnl: np.ndarray,
    annual_turnover_notional: np.ndarray,
    *,
    is_uk_shares: bool,
    annual_dividend_income: np.ndarray | None = None,
    params: WrapperParams = WrapperParams(),
) -> np.ndarray:
    """GIA after-tax P&L: DIRECT dealing + stamp + CGT (GBP3k exempt + carry-fwd) + dividend tax.

    annual_pnl: PRE-dealing-cost gross P&L in GBP (one element per year).
    annual_turnover_notional: GBP value of purchases per year (drives dealing + stamp).
    is_uk_shares: True -> 0.5% SDRT on purchases; False (US shares / futures) -> no stamp.
    """
    pnl = np.asarray(annual_pnl, dtype=float)
    turn = np.asarray(annual_turnover_notional, dtype=float)
    div = (
        np.zeros_like(pnl)
        if annual_dividend_income is None
        else np.asarray(annual_dividend_income, dtype=float)
    )
    out = np.empty_like(pnl)
    carryforward = 0.0
    for y in range(len(pnl)):
        dealing = params.direct_cost_per_turnover * turn[y]
        stamp = params.stamp_rate_uk_shares * turn[y] if is_uk_shares else 0.0
        gain = pnl[y] - dealing - stamp
        div_tax = params.dividend_tax_rate * max(div[y] - params.dividend_allowance, 0.0)
        if gain > 0:
            offset = min(carryforward, gain)  # losses offset gains first
            after_cf = gain - offset
            carryforward -= offset
            taxable = max(after_cf - params.cgt_annual_exemption, 0.0)
            cgt = params.cgt_rate * taxable
            out[y] = gain - cgt - div_tax
        else:
            carryforward += -gain  # bank the loss as a future tax asset
            out[y] = gain - div_tax
    return out


def sb_after_tax(
    annual_pnl: np.ndarray,
    annual_turnover_notional: float | np.ndarray,
    avg_notional: float | np.ndarray,
    *,
    params: WrapperParams = WrapperParams(),
) -> np.ndarray:
    """Spread bet: no CGT/income/stamp, minus IG EMBEDDED dealing spread (turnover-scaled, wider
    than direct) AND IG overnight financing on the notional held.

    annual_pnl is PRE-dealing gross P&L. Losses are NOT banked (the GIA asymmetry is the absence of
    a tax-asset, not a return haircut).
    """
    pnl = np.asarray(annual_pnl, dtype=float)

    def _arr(x):
        return np.full_like(pnl, float(x)) if np.isscalar(x) else np.asarray(x, dtype=float)

    turn = _arr(annual_turnover_notional)
    notional = _arr(avg_notional)
    dealing = params.sb_cost_per_turnover * turn
    financing = params.ig_financing_rate * notional
    return pnl - dealing - financing


def best_wrapper(
    annual_pnl: np.ndarray,
    *,
    capital: float,
    annual_turnover_x: float,
    is_uk_shares: bool,
    leverage: float = 1.0,
    isa_eligible: bool = True,
    params: WrapperParams = WrapperParams(),
) -> dict:
    """Rank wrappers by mean after-tax annual return for a strategy.

    annual_pnl: annual P&L in GBP (one element/year). capital: GBP capital base.
    annual_turnover_x: annual purchase turnover as a multiple of capital (drives GIA stamp).
    is_uk_shares / isa_eligible: instrument flags. leverage: SB notional / capital.
    """
    pnl = np.asarray(annual_pnl, dtype=float)
    turn = np.full_like(pnl, annual_turnover_x * capital)
    nets = {
        "ISA": isa_after_tax(pnl, turn, is_uk_shares=is_uk_shares, params=params)
        if isa_eligible
        else None,
        "SIPP": sipp_after_tax(pnl, turn, is_uk_shares=is_uk_shares, params=params)
        if isa_eligible
        else None,
        "GIA": gia_after_tax(pnl, turn, is_uk_shares=is_uk_shares, params=params),
        "SB": sb_after_tax(pnl, turn, leverage * capital, params=params),
    }
    means = {k: float(np.mean(v)) / capital for k, v in nets.items() if v is not None}
    ranked = sorted(means.items(), key=lambda kv: kv[1], reverse=True)
    return {"net_return_by_wrapper": means, "ranked": ranked, "best": ranked[0][0]}


__all__ = [
    "WrapperParams",
    "best_wrapper",
    "gia_after_tax",
    "isa_after_tax",
    "sb_after_tax",
    "sipp_after_tax",
]
