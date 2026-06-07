"""Fractional Kelly sizing per strategy (V3.7 / L65 + L67; audit P1-25 / P1-26).

Computes the Kelly-optimal LEVERAGE for each strategy from WFO-stitched OOS
returns, applies a fractional-Kelly scaling (default 0.25x for
parameter-estimation uncertainty, per MacLean/Thorp/Ziemba 2010), and gates
strategies with statistically-insignificant edge.

LEVERAGE, NOT WEIGHT (audit H5 / P1-25)
---------------------------------------
``kelly_leverage = f* = mu / sigma^2`` is a **leverage multiple** -- gross
exposure per unit capital for this strategy IN ISOLATION. It is NOT a portfolio
capital weight and is routinely > 1 (a low-vol edge implies high optimal
leverage -- GEM's f* was ~9). The ERC allocator and ``assess_joint_ruin`` work
in capital WEIGHTS that sum to <= 1. Convert with ``normalise_kelly_to_weights``
-- never feed a raw Kelly leverage into a weight-summing context.

Confidence interval + gate (audit H6 / P1-26)
---------------------------------------------
f* inherits the sampling uncertainty of the Sharpe estimate. We attach a 95% CI
via the Lo (2002) analytic Sharpe standard error and gate on ``ci_lo > 0`` (the
edge is statistically positive) rather than the old "f* >= 5%" floor (5%
leverage is a meaningless threshold). We also report the EMPIRICAL (geometric)
Kelly -- argmax_f E[log(1 + f r)] on the actual return sample -- alongside the
Gaussian f*, so fat tails / skew that the Gaussian formula misses are visible.

Why fractional, not full Kelly: full Kelly assumes the edge estimate is exact
and has crippling drawdowns; fractional (0.25-0.5x) trades a little long-run
growth for materially lower drawdowns.

Reference: Kelly 1956; MacLean/Thorp/Ziemba 2010; Lo 2002 "The Statistics of
Sharpe Ratios"; Bailey & Lopez de Prado 2014 (DSR).

Usage:

    from titan.research.framework.kelly import compute_kelly_fraction
    kelly = compute_kelly_fraction(returns, periods_per_year=252, fractional=0.25)
    if kelly.passes_gate():            # ci_lo > 0
        weight = normalise_kelly_to_weights({sid: kelly.fractional_leverage})[sid]
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

_Z_95 = 1.959963984540054  # two-sided 95% normal quantile


@dataclass(frozen=True)
class KellyFraction:
    """Per-strategy Kelly analysis. ``kelly_leverage`` is a LEVERAGE multiple
    (see module docstring), not a capital weight.

    Attributes:
        kelly_leverage: Gaussian f* = mu / sigma^2 (annualised-invariant).
        kelly_leverage_ci_lo / _hi: 95% CI on f* from the Lo-2002 Sharpe SE.
        empirical_kelly: geometric Kelly, argmax_f E[log(1 + f r)] on the
            sample (captures fat tails / skew the Gaussian f* ignores).
        fractional_leverage: fractional_factor * kelly_leverage (still a
            leverage, not a weight). Default 0.25x.
        sample_sharpe / deflated_sharpe: annualised Sharpe (raw / DSR-adjusted).
        sharpe_ci_lo / _hi: 95% CI on the annualised sample Sharpe (Lo 2002).
        annual_return / annual_vol: annualised mean / vol.
        fractional_factor: scaling applied (typically 0.25 or 0.5).
        n_obs: number of return observations.
    """

    kelly_leverage: float
    kelly_leverage_ci_lo: float
    kelly_leverage_ci_hi: float
    empirical_kelly: float
    fractional_leverage: float
    sample_sharpe: float
    deflated_sharpe: float | None
    sharpe_ci_lo: float
    sharpe_ci_hi: float
    annual_return: float
    annual_vol: float
    fractional_factor: float
    n_obs: int

    # ── Deprecated aliases (pre-P1-25 names; "weight" was a misnomer) ──────
    @property
    def full_kelly(self) -> float:
        """Deprecated alias for ``kelly_leverage`` (it is a leverage, not a
        weight).
        """
        return self.kelly_leverage

    @property
    def fractional_weight(self) -> float:
        """Deprecated alias for ``fractional_leverage`` (a leverage, not a
        weight -- do not feed into a weight-summing allocator; use
        ``normalise_kelly_to_weights``).
        """
        return self.fractional_leverage

    def passes_gate(
        self, *, require_ci_lo_positive: bool = True, min_kelly_leverage: float = 0.05
    ) -> bool:
        """V3.8-3 gate (P1-26): the Kelly leverage is STATISTICALLY POSITIVE
        (95% CI lower bound > 0), i.e. the edge survives sampling uncertainty.

        Set ``require_ci_lo_positive=False`` to fall back to the legacy
        magnitude floor (``kelly_leverage >= min_kelly_leverage``); the CI gate
        is preferred -- a 5% leverage floor is not an economically meaningful
        threshold.
        """
        if require_ci_lo_positive:
            return self.kelly_leverage_ci_lo > 0.0
        return self.kelly_leverage >= min_kelly_leverage

    def report(self) -> str:
        dsr_str = f"{self.deflated_sharpe:+.4f}" if self.deflated_sharpe is not None else "n/a"
        return (
            f"KellyFraction(n_obs={self.n_obs})\n"
            f"  Sample Sharpe (ann):     {self.sample_sharpe:+.4f} "
            f"[95% CI {self.sharpe_ci_lo:+.3f}, {self.sharpe_ci_hi:+.3f}]\n"
            f"  DSR-deflated Sharpe:     {dsr_str}\n"
            f"  Annual return / vol:     {self.annual_return:+.4%} / {self.annual_vol:.4%}\n"
            f"  Kelly LEVERAGE (f*):     {self.kelly_leverage:+.3f}x "
            f"[95% CI {self.kelly_leverage_ci_lo:+.3f}, {self.kelly_leverage_ci_hi:+.3f}]\n"
            f"  Empirical (geom) Kelly:  {self.empirical_kelly:+.3f}x\n"
            f"  Fractional ({self.fractional_factor:.2f}x):  {self.fractional_leverage:+.3f}x "
            f"(LEVERAGE, not a weight)\n"
            f"  Passes gate (ci_lo>0):   {self.passes_gate()}"
        )


def _empirical_kelly(returns_arr: np.ndarray) -> float:
    """Geometric Kelly: argmax_{f>=0} mean(log(1 + f r)) on the sample.

    Grid search over the feasible range (f < 1/|worst loss| keeps 1+f r > 0).
    Log-concave in f, so a fine grid is robust + deterministic (no RNG / scipy).
    """
    r = returns_arr[np.isfinite(returns_arr)]
    if len(r) < 2:
        return 0.0
    min_r = float(r.min())
    f_ub = (0.999 / abs(min_r)) if min_r < 0 else 50.0
    if f_ub <= 0:
        return 0.0
    best_f, best_g = 0.0, 0.0  # f=0 -> log(1)=0 baseline
    for f in np.linspace(0.0, f_ub, 2001)[1:]:
        v = 1.0 + f * r
        if np.any(v <= 0.0):
            break
        g = float(np.mean(np.log(v)))
        if g > best_g:
            best_g, best_f = g, float(f)
    return best_f


def compute_kelly_fraction(
    returns: pd.Series,
    *,
    periods_per_year: int,
    dsr_deflated_sharpe: float | None = None,
    fractional: float = 0.25,
) -> KellyFraction:
    """Compute the fractional Kelly LEVERAGE from a return series (audit P1-25/26).

    ``periods_per_year`` annualises (252 daily, 6048 FX H1, 1764 US-equity RTH
    H1). ``dsr_deflated_sharpe``, if given, replaces the sample Sharpe in the
    point f* (more conservative); the CI always reflects sample sampling
    uncertainty (Lo 2002). ``fractional`` scales f* (0.25 standard).
    """
    s = returns.dropna()
    n = len(s)
    if n < 30:
        return KellyFraction(
            kelly_leverage=0.0,
            kelly_leverage_ci_lo=0.0,
            kelly_leverage_ci_hi=0.0,
            empirical_kelly=0.0,
            fractional_leverage=0.0,
            sample_sharpe=0.0,
            deflated_sharpe=dsr_deflated_sharpe,
            sharpe_ci_lo=0.0,
            sharpe_ci_hi=0.0,
            annual_return=0.0,
            annual_vol=0.0,
            fractional_factor=fractional,
            n_obs=n,
        )

    mu_per = float(s.mean())
    sigma_per = float(s.std(ddof=1))
    annual_return = mu_per * periods_per_year
    annual_vol = sigma_per * np.sqrt(periods_per_year)
    sr_per = (mu_per / sigma_per) if sigma_per > 1e-12 else 0.0
    sample_sharpe = sr_per * np.sqrt(periods_per_year)

    # Lo (2002) IID Sharpe standard error: SE(SR_per) = sqrt((1 + 0.5 SR_per^2)/n);
    # annualise SR and SE by sqrt(ppy). CI on the annualised SAMPLE Sharpe.
    se_per = np.sqrt((1.0 + 0.5 * sr_per**2) / n)
    se_ann = se_per * np.sqrt(periods_per_year)
    sharpe_ci_lo = sample_sharpe - _Z_95 * se_ann
    sharpe_ci_hi = sample_sharpe + _Z_95 * se_ann

    # Gaussian Kelly leverage f* = edge_sharpe / annual_vol (annualised-invariant
    # of mu/sigma^2). Use the deflated Sharpe for the point estimate if given.
    edge_sharpe = dsr_deflated_sharpe if dsr_deflated_sharpe is not None else sample_sharpe
    kelly_leverage = max(0.0, edge_sharpe / annual_vol) if annual_vol > 1e-12 else 0.0
    # CI on f* from the sample-Sharpe CI (vol at point estimate).
    if annual_vol > 1e-12:
        kelly_leverage_ci_lo = sharpe_ci_lo / annual_vol
        kelly_leverage_ci_hi = sharpe_ci_hi / annual_vol
    else:
        kelly_leverage_ci_lo = kelly_leverage_ci_hi = 0.0

    empirical_kelly = _empirical_kelly(s.to_numpy(dtype=float))
    fractional_leverage = fractional * kelly_leverage

    return KellyFraction(
        kelly_leverage=float(kelly_leverage),
        kelly_leverage_ci_lo=float(kelly_leverage_ci_lo),
        kelly_leverage_ci_hi=float(kelly_leverage_ci_hi),
        empirical_kelly=float(empirical_kelly),
        fractional_leverage=float(fractional_leverage),
        sample_sharpe=float(sample_sharpe),
        deflated_sharpe=dsr_deflated_sharpe,
        sharpe_ci_lo=float(sharpe_ci_lo),
        sharpe_ci_hi=float(sharpe_ci_hi),
        annual_return=float(annual_return),
        annual_vol=float(annual_vol),
        fractional_factor=fractional,
        n_obs=n,
    )


def normalise_kelly_to_weights(
    leverages: dict[str, float], *, gross_cap: float = 1.0
) -> dict[str, float]:
    """Convert per-strategy Kelly LEVERAGE multiples to capital WEIGHTS summing
    to <= ``gross_cap`` (audit P1-25 / H5 -- the explicit leverage->weight
    reconciliation with the ERC / ``assess_joint_ruin`` sum<=1 convention).

    Non-positive leverages -> 0. If the total leverage already fits the budget
    the weights ARE the leverages (no scaling UP -- Kelly never demands more
    than its computed leverage); otherwise weights scale down proportionally to
    fit ``gross_cap``.
    """
    pos = {k: max(0.0, float(v)) for k, v in leverages.items()}
    total = sum(pos.values())
    if total <= 0.0:
        return dict.fromkeys(leverages, 0.0)
    if total <= gross_cap:
        return pos
    scale = gross_cap / total
    return {k: v * scale for k, v in pos.items()}


__all__ = ["KellyFraction", "compute_kelly_fraction", "normalise_kelly_to_weights"]
