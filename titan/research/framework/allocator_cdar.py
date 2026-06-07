"""CDaR (Conditional Drawdown-at-Risk) drawdown-constrained allocator (audit P3-1).

The most direct encoding of the RoR~=0 / MaxDD<20% mandate: instead of equalising
risk contributions (ERC) or inverse-vol, choose deployment weights that MINIMISE
the portfolio's tail drawdown subject to a return floor:

    min_w  CDaR_{tail}(portfolio)
    s.t.   sum(w) = 1,   0 <= w_n <= max_weight,
           mean portfolio return >= cagr_floor / periods_per_year   (optional)

CDaR_{tail} = the average of the worst ``tail`` fraction of drawdown depths
(Chekhlov, Uryasev & Zabarankin 2005). Minimising it is a LINEAR PROGRAM in the
weights plus per-bar running-peak / drawdown auxiliaries -- solved exactly with
HiGHS. The drawdown is modelled on the (uncompounded) cumulative-return path,
the standard tractable CDaR LP; the REALISED (geometric) CDaR of the solution is
reported via ``metrics.cdar`` for verification.

If the LP is infeasible (e.g. the return floor can't be met) it falls back to an
inverse-vol allocation (``covariance.inverse_vol_weights``) with ``success=False``
so the caller can react. The audit recipe is: warm-start intuition from ERC/Kelly,
then VERIFY the chosen weights with ``assess_joint_ruin``.

Reference: Chekhlov, Uryasev & Zabarankin 2005, "Drawdown Measure in Portfolio
Optimization."
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from titan.research.framework.covariance import inverse_vol_weights
from titan.research.metrics import cdar

DEFAULT_TAIL = 0.05  # CDaR_95: average of the worst 5% of drawdowns
DEFAULT_MAX_WEIGHT = 0.60


@dataclass(frozen=True)
class CdarResult:
    """CDaR allocation output."""

    weights: dict[str, float]
    cdar: float  # realised CDaR_{tail} of the solution (geometric, metrics.cdar)
    lp_cdar: float  # LP objective value (additive-cumret-path CDaR)
    mean_return_ann: float  # realised annualised mean of the solution portfolio
    n_obs: int
    success: bool  # True if the LP solved; False if it fell back to inverse-vol
    status: str


def _aligned_matrix(strategy_returns: dict[str, pd.Series]) -> tuple[np.ndarray, list[str]]:
    names = list(strategy_returns.keys())
    common = None
    for r in strategy_returns.values():
        r = r.dropna()
        common = r.index if common is None else common.intersection(r.index)
    if common is None or len(common) < 30:
        raise ValueError(f"Common index too short: {len(common) if common is not None else 0}")
    df = pd.DataFrame({n: strategy_returns[n].reindex(common).fillna(0.0) for n in names})
    return df.to_numpy(dtype=float), names


def _portfolio_returns(returns_mat: np.ndarray, w: np.ndarray) -> np.ndarray:
    return returns_mat @ w


def compute_cdar_weights(
    strategy_returns: dict[str, pd.Series],
    *,
    tail: float = DEFAULT_TAIL,
    cagr_floor: float | None = None,
    max_weight: float = DEFAULT_MAX_WEIGHT,
    min_weight: float = 0.0,
    periods_per_year: int = 252,
) -> CdarResult:
    """Solve the CDaR-minimising deployment weights (audit P3-1).

    ``tail`` is the drawdown tail fraction (0.05 = CDaR_95). ``cagr_floor`` is an
    optional ANNUAL mean-return floor (linearised to a per-bar mean constraint).
    ``max_weight`` caps single-strategy concentration.
    """
    r_mat, names = _aligned_matrix(strategy_returns)
    t_n, n = r_mat.shape
    # Uncompounded cumulative-return path: cumret_t = C[t] @ w.
    c_cum = np.cumsum(r_mat, axis=0)

    # LP variables x = [ w(n) | u(T) | zeta(1) | y(T) ];  len = n + 2T + 1.
    wu, zi, yo = n, n + t_n, n + t_n + 1
    n_vars = n + 2 * t_n + 1

    cost = np.zeros(n_vars)
    cost[zi] = 1.0
    cost[yo : yo + t_n] = 1.0 / (tail * t_n)

    rows, cols, data, b_ub = [], [], [], []
    rr = 0

    def _add(r: int, c: int, v: float) -> None:
        rows.append(r)
        cols.append(c)
        data.append(v)

    # (1) y_t >= u_t - cumret_t - zeta  ->  u_t - C[t].w - zeta - y_t <= 0
    for t in range(t_n):
        for j in range(n):
            _add(rr, j, -c_cum[t, j])
        _add(rr, wu + t, 1.0)
        _add(rr, zi, -1.0)
        _add(rr, yo + t, -1.0)
        b_ub.append(0.0)
        rr += 1
    # (2) u_t >= cumret_t  ->  C[t].w - u_t <= 0
    for t in range(t_n):
        for j in range(n):
            _add(rr, j, c_cum[t, j])
        _add(rr, wu + t, -1.0)
        b_ub.append(0.0)
        rr += 1
    # (3) running-max monotonicity: u_{t-1} - u_t <= 0
    for t in range(1, t_n):
        _add(rr, wu + t - 1, 1.0)
        _add(rr, wu + t, -1.0)
        b_ub.append(0.0)
        rr += 1
    # (4) optional return floor: -mean(R).w <= -floor_per_bar
    if cagr_floor is not None:
        mean_r = r_mat.mean(axis=0)
        floor_per_bar = cagr_floor / periods_per_year
        for j in range(n):
            _add(rr, j, -mean_r[j])
        b_ub.append(-floor_per_bar)
        rr += 1

    a_ub = coo_matrix((data, (rows, cols)), shape=(rr, n_vars)).tocsr()
    a_eq = coo_matrix(([1.0] * n, ([0] * n, list(range(n)))), shape=(1, n_vars)).tocsr()
    bounds = (
        [(min_weight, max_weight)] * n
        + [(0.0, None)] * t_n  # u >= 0 (peak from a 0 baseline)
        + [(None, None)]  # zeta free
        + [(0.0, None)] * t_n  # y >= 0
    )

    res = linprog(
        cost,
        A_ub=a_ub,
        b_ub=np.array(b_ub),
        A_eq=a_eq,
        b_eq=np.array([1.0]),
        bounds=bounds,
        method="highs",
    )

    if res.success:
        w = np.clip(res.x[:n], 0.0, None)
        w = w / w.sum() if w.sum() > 0 else np.full(n, 1.0 / n)
        lp_cdar = float(res.fun)
        success, status = True, "optimal"
    else:
        # Infeasible (e.g. return floor unreachable) -> inverse-vol fallback.
        cov = np.cov(r_mat, rowvar=False)
        cov = np.atleast_2d(cov)
        w = inverse_vol_weights(cov)
        lp_cdar = float("nan")
        success, status = False, f"lp_failed:{res.status}->inverse_vol"

    port = _portfolio_returns(r_mat, w)
    return CdarResult(
        weights={names[i]: float(w[i]) for i in range(n)},
        cdar=float(cdar(pd.Series(port), alpha=tail)),
        lp_cdar=lp_cdar,
        mean_return_ann=float(port.mean() * periods_per_year),
        n_obs=t_n,
        success=success,
        status=status,
    )


__all__ = ["CdarResult", "compute_cdar_weights", "DEFAULT_TAIL", "DEFAULT_MAX_WEIGHT"]
