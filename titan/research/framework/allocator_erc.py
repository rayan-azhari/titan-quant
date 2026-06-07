"""Equal Risk Contribution (ERC) allocator (V3.7).

Allocates portfolio weights so each strategy contributes equally to the
portfolio's total volatility. This is the standard "risk parity"
allocation method (Maillard, Roncalli & Teiletche 2010).

Why ERC over inverse-vol:
    - Inverse-vol ignores correlations: two highly-correlated strategies
      at equal vol get equal weight, but their combined risk contribution
      is doubled.
    - ERC accounts for correlation structure via the covariance matrix.
    - It's the deterministic "risk parity" answer to "how should I split
      capital across uncorrelated bets?"

Why HRP later (not now):
    - HRP (Lopez de Prado 2016) does tree-clustering on correlations and
      recursive bisection. It's a refinement of ERC that handles clusters
      better — but at 3-10 strategies the gain is marginal. Add HRP when
      portfolio has 10+ strategies.

Reference:
    Maillard, S., T. Roncalli, J. Teiletche (2010). "The Properties of
    Equally Weighted Risk Contribution Portfolios." Journal of Portfolio
    Management 36(4): 60-70.

Usage:

    from titan.research.framework.allocator_erc import compute_erc_weights

    weights = compute_erc_weights(
        strategy_returns={"demo_a": gem_ret, "turtle": turtle_ret},
        target_vol_ann=0.10,  # optional rescale to target portfolio vol
    )
    # weights = {"demo_a": 0.65, "turtle": 0.35} (illustrative)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from titan.research.framework.covariance import (
    ShrunkCovariance,
    inverse_vol_weights,
    shrink_covariance,
)


@dataclass(frozen=True)
class ErcResult:
    """ERC allocation output.

    Attributes:
        weights: mapping name -> normalised weight (sum to 1.0).
        risk_contributions: mapping name -> fraction of portfolio variance
            contributed by this strategy. ERC objective: all equal (1/N).
        portfolio_vol_ann: annualised portfolio volatility at output weights.
        target_vol_rescale: scaling factor applied if target_vol_ann was set.
            Final deployable weights = weights * target_vol_rescale.
        n_obs: number of common observations used.
    """

    weights: dict[str, float]
    risk_contributions: dict[str, float]
    portfolio_vol_ann: float
    target_vol_rescale: float
    n_obs: int

    def deployment_weights(self) -> dict[str, float]:
        """Final per-strategy capital weights after target-vol rescale."""
        return {n: w * self.target_vol_rescale for n, w in self.weights.items()}

    def report(self) -> str:
        lines = ["ERC Allocation:"]
        lines.append(f"  n_obs = {self.n_obs}")
        lines.append(f"  portfolio vol (ann) = {self.portfolio_vol_ann:.4%}")
        lines.append(f"  target-vol rescale  = {self.target_vol_rescale:.4f}")
        lines.append(f"  {'name':<20} {'weight':>10} {'risk_contrib':>14} {'deploy_w':>10}")
        deploy = self.deployment_weights()
        for name in self.weights:
            lines.append(
                f"  {name:<20} {self.weights[name]:>10.4%} "
                f"{self.risk_contributions[name]:>14.4%} {deploy[name]:>10.4%}"
            )
        return "\n".join(lines)


def _cov_matrix(
    strategy_returns: dict[str, pd.Series], *, crisis_rho: float | None = None
) -> ShrunkCovariance:
    """Build a Ledoit-Wolf SHRUNK covariance from aligned returns (audit P1-27).

    Replaces the raw sample covariance, which is noisy / ill-conditioned at the
    small N (~3 strategies) + short window the live stack runs at. ``crisis_rho``
    floors off-diagonal correlations (crisis overlay). The returned
    ShrunkCovariance carries a PSD/condition flag so the caller can fall back to
    inverse-vol when the estimate is degenerate.
    """
    names = list(strategy_returns.keys())
    common = None
    for r in strategy_returns.values():
        r = r.dropna()
        common = r.index if common is None else common.intersection(r.index)
    if common is None or len(common) < 30:
        raise ValueError(f"Common index too short: {len(common) if common is not None else 0}")
    df = pd.DataFrame({n: strategy_returns[n].reindex(common).fillna(0.0) for n in names})
    return shrink_covariance(df, crisis_rho=crisis_rho)


def _risk_contributions(weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Per-strategy fractional risk contributions. Sum to 1."""
    portfolio_var = weights @ cov @ weights
    if portfolio_var <= 1e-18:
        return np.full_like(weights, 1.0 / len(weights))
    marginal = cov @ weights
    rc = weights * marginal / portfolio_var
    return rc


def compute_erc_weights(
    strategy_returns: dict[str, pd.Series],
    *,
    target_vol_ann: float | None = None,
    periods_per_year: int = 252,
    min_weight: float = 0.01,
    max_iter: int = 500,
    crisis_rho: float | None = None,
) -> ErcResult:
    """Solve for equal-risk-contribution weights.

    Parameters:
        strategy_returns: name -> per-bar net returns (OOS preferred).
        target_vol_ann: if provided, computes a `target_vol_rescale` factor
            to scale the ERC weights so that w * rescale produces a
            portfolio with this annualised vol. Useful for sizing the
            stack to a vol budget.
        periods_per_year: annualisation factor for target_vol_rescale.
        min_weight: minimum per-strategy weight (regularisation).
        max_iter: SLSQP iterations.

    Returns:
        ErcResult with weights, risk contributions, vol, rescale factor.
    """
    sc = _cov_matrix(strategy_returns, crisis_rho=crisis_rho)
    cov, names, n_obs = sc.cov, sc.names, sc.n_obs
    n = len(names)

    if not sc.is_well_conditioned:
        # P1-27 fallback: a degenerate / ill-conditioned covariance can't be
        # trusted for ERC (the inverse is unstable) -- allocate inverse-vol
        # (diagonal only), ignoring the unreliable off-diagonal correlations.
        w = inverse_vol_weights(cov)
    else:

        def objective(w: np.ndarray) -> float:
            rc = _risk_contributions(w, cov)
            target = 1.0 / n
            return float(np.sum((rc - target) ** 2))

        # Constraints: weights sum to 1, all >= min_weight.
        cons = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        ]
        bounds = [(min_weight, 1.0 - min_weight * (n - 1)) for _ in range(n)]
        x0 = np.full(n, 1.0 / n)

        sol = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={"maxiter": max_iter, "ftol": 1e-12},
        )
        w = sol.x
    w = w / w.sum()  # ensure exact sum-to-1

    rc = _risk_contributions(w, cov)
    portfolio_var = w @ cov @ w
    portfolio_vol = float(np.sqrt(portfolio_var) * np.sqrt(periods_per_year))

    if target_vol_ann is not None and portfolio_vol > 1e-12:
        target_vol_rescale = target_vol_ann / portfolio_vol
    else:
        target_vol_rescale = 1.0

    return ErcResult(
        weights={names[i]: float(w[i]) for i in range(n)},
        risk_contributions={names[i]: float(rc[i]) for i in range(n)},
        portfolio_vol_ann=portfolio_vol,
        target_vol_rescale=target_vol_rescale,
        n_obs=n_obs,
    )


__all__ = ["ErcResult", "compute_erc_weights"]
