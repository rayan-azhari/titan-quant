"""Information Coefficient (IC) analysis -- cross-sectional factor diagnostics.

Borrows the Alphalens methodology (IC decay + quantile/return spread) into the
Titan framework, but routes every annualisation through
``titan.research.metrics`` so the convention is explicit and consistent with the
rest of the framework -- no hidden ``sqrt(252)``. In particular the *annualised
IC information ratio* is, by construction, the Sharpe ratio of the per-period IC
series, so :func:`summarise_ic` calls :func:`titan.research.metrics.sharpe`
directly rather than re-deriving the factor.

Two diagnostics, both standard tools for screening a signal *before* it becomes a
strategy. The IC sleeves (``ic_equity_daily``, ``ic_mtf``) screen a universe this
way -- ``ic_equity_daily`` screened 482 S&P 500 / Russell 100 names before
isolating its final basket.

    IC decay        At each forward horizon ``h``, how strongly does the factor
                    at time ``t`` rank-correlate with the ``h``-bar-ahead return?
                    A signal with edge has consistently-signed IC that decays
                    smoothly as ``h`` grows; the horizon where ``|IC|`` peaks is
                    the natural holding period. Cross-sectional: one IC per date
                    across the universe, then summarised over time.

    Quantile spread Sort the universe into ``N`` quantiles by factor each period
                    and measure the mean forward return of each bucket. A monotone
                    ladder (top bucket >> bottom bucket) is the evidence that the
                    factor *sorts* returns; the top-minus-bottom spread is the
                    long/short edge before costs.

Look-ahead note
---------------
Forward returns are future-derived BY DESIGN here -- IC analysis measures how
well a factor *predicts* the future, so the return must look ahead of the
factor. These functions only ever produce *measurements*, never a live signal,
so the discipline rule (never let a forward return leak into a model input) is
unaffected. The one mild caveat is pooled (single-instrument) quantile binning,
which uses full-sample quantile edges -- flagged inline on :func:`quantile_returns`.

Overlap caveat
--------------
For horizon ``h > 1`` sampled every bar, consecutive forward returns overlap, so
the IC series is autocorrelated. The IC ``t``-stat and annualised IC-IR then
OVERSTATE significance (the same issue Alphalens documents, and the same caveat
:func:`titan.research.metrics.bootstrap_sharpe_ci` carries). Treat them as
optimistic unless you sample the factor every ``h`` bars (non-overlapping).

References:
----------
* Grinold & Kahn (2000), *Active Portfolio Management* -- the IC / IR framework.
* Alphalens (Quantopian) -- cross-sectional IC + ``mean_return_by_quantile``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from titan.research import metrics

_VALID_METHODS = ("spearman", "pearson")


@dataclass(frozen=True)
class IcSummary:
    """Summary statistics of a per-period IC series at one forward horizon."""

    horizon: int
    method: str  # "spearman" | "pearson"
    n_periods: int  # number of dates with a defined IC
    ic_mean: float
    ic_std: float  # sample std (ddof=1)
    ic_ir: float  # un-annualised IC information ratio: ic_mean / ic_std
    ic_ir_annualised: float  # == metrics.sharpe(ic_series, periods_per_year)
    t_stat: float  # ic_ir * sqrt(n_periods)
    pct_positive: float  # fraction of periods with IC > 0


@dataclass(frozen=True)
class QuantileResult:
    """Mean forward return per factor quantile, plus the top-minus-bottom spread."""

    horizon: int
    n_quantiles: int
    mean_returns: tuple[float, ...]  # per quantile, low factor -> high factor
    counts: tuple[int, ...]  # observation count per quantile
    spread: float  # mean return of top quantile - bottom quantile (per horizon)
    spread_tstat: float  # Welch t-stat, top-bucket returns vs bottom-bucket returns
    monotonicity: float  # Spearman corr of quantile index vs mean_returns, in [-1, 1]
    ann_spread: float  # spread annualised: spread * periods_per_year / horizon
    cross_sectional: bool  # True if binned per date across a universe; False if pooled


# ── Internal helpers ────────────────────────────────────────────────────────


def _as_panel(x: pd.DataFrame | pd.Series) -> pd.DataFrame:
    """Coerce a Series (single instrument) to a one-column DataFrame."""
    if isinstance(x, pd.DataFrame):
        return x
    if isinstance(x, pd.Series):
        return x.to_frame(name=x.name if x.name is not None else "asset")
    raise TypeError(f"expected pd.DataFrame or pd.Series, got {type(x).__name__}")


def _check_method(method: str) -> None:
    if method not in _VALID_METHODS:
        raise ValueError(f"method must be one of {_VALID_METHODS}, got {method!r}")


def _pairwise_corr(a: np.ndarray, b: np.ndarray, method: str) -> float:
    """Rank (spearman) or linear (pearson) correlation, NaN on degenerate input.

    Returns NaN -- rather than raising or warning -- when either side is
    constant or has fewer than two points, so callers can simply drop NaNs.
    """
    if len(a) < 2:
        return float("nan")
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    if method == "spearman":
        r, _ = stats.spearmanr(a, b)
    else:  # pearson (validated by caller)
        r, _ = stats.pearsonr(a, b)
    return float(r)


# ── Forward returns ─────────────────────────────────────────────────────────


def forward_returns(
    prices: pd.DataFrame | pd.Series,
    horizon: int,
    *,
    log: bool = True,
) -> pd.DataFrame | pd.Series:
    """Forward return over ``horizon`` bars: ``r_t = f(price_{t+h} / price_t)``.

    Returns the same shape as ``prices`` (Series in -> Series out, frame in ->
    frame out). The last ``horizon`` rows are NaN (no future price). Uses log
    returns by default (matches the IC sleeves' ``log(close.shift(-1)/close)``);
    pass ``log=False`` for simple returns.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    shifted = prices.shift(-horizon)
    if log:
        return np.log(shifted / prices)
    return shifted / prices - 1.0


# ── Cross-sectional IC ───────────────────────────────────────────────────────


def cross_sectional_ic(
    factor: pd.DataFrame,
    fwd_ret: pd.DataFrame,
    *,
    method: str = "spearman",
    min_assets: int = 5,
) -> pd.Series:
    """Per-date IC: rank-correlate the factor cross-section against forward returns.

    At each date the factor values across the universe are correlated with the
    same date's forward returns, yielding one IC per date. Dates with fewer than
    ``min_assets`` non-NaN factor/return pairs are skipped.

    Parameters
    ----------
    factor:
        Wide panel, index = date, columns = symbol. Needs >= 2 columns -- a
        single instrument has no cross-section; use :func:`rolling_ic` instead.
    fwd_ret:
        Forward returns aligned to ``factor`` (e.g. ``forward_returns(prices, h)``).
    method:
        ``"spearman"`` (rank IC, the default and the Alphalens convention) or
        ``"pearson"``.
    min_assets:
        Minimum non-NaN pairs required to compute an IC for a date.

    Returns:
    -------
    pd.Series
        IC indexed by date, sorted ascending. Empty if no date qualifies.
    """
    _check_method(method)
    f = _as_panel(factor)
    r = _as_panel(fwd_ret)
    if f.shape[1] < 2:
        raise ValueError(
            "cross_sectional_ic needs >= 2 symbols (a cross-section); "
            "for a single instrument use rolling_ic"
        )
    cols = f.columns.intersection(r.columns)
    idx = f.index.intersection(r.index)
    f = f.loc[idx, cols]
    r = r.loc[idx, cols]

    out: dict = {}
    for t in f.index:
        pair = pd.concat([f.loc[t], r.loc[t]], axis=1).dropna()
        if len(pair) < min_assets:
            continue
        ic = _pairwise_corr(pair.iloc[:, 0].to_numpy(), pair.iloc[:, 1].to_numpy(), method)
        if not math.isnan(ic):
            out[t] = ic
    return pd.Series(out, dtype=float).sort_index()


def rolling_ic(
    factor: pd.Series,
    fwd_ret: pd.Series,
    window: int,
    *,
    method: str = "spearman",
    min_periods: int | None = None,
) -> pd.Series:
    """Rolling time-series IC for a *single* instrument.

    Correlates the trailing ``window`` of factor values against the same window
    of forward returns -- the single-symbol analogue of :func:`cross_sectional_ic`,
    useful for watching IC stability / decay over time on one name (e.g. the
    per-symbol calibration the IC sleeves do). The value at bar ``t`` uses the
    window ending at ``t``; because ``fwd_ret`` already looks ``h`` bars ahead,
    interpret this as a measurement, not a tradeable series.
    """
    _check_method(method)
    f = pd.Series(factor, dtype=float)
    r = pd.Series(fwd_ret, dtype=float).reindex(f.index)
    mp = min_periods if min_periods is not None else max(2, window // 2)

    vals = np.full(len(f), np.nan)
    fa = f.to_numpy()
    ra = r.to_numpy()
    for i in range(len(f)):
        lo = max(0, i - window + 1)
        a = fa[lo : i + 1]
        b = ra[lo : i + 1]
        mask = ~(np.isnan(a) | np.isnan(b))
        if mask.sum() < mp:
            continue
        vals[i] = _pairwise_corr(a[mask], b[mask], method)
    return pd.Series(vals, index=f.index)


def summarise_ic(
    ic_series: pd.Series | np.ndarray,
    *,
    periods_per_year: int,
    horizon: int,
    method: str = "spearman",
) -> IcSummary:
    """Reduce a per-period IC series to an :class:`IcSummary`.

    The annualised IC information ratio is exactly the Sharpe ratio of the IC
    series, so it is delegated to :func:`titan.research.metrics.sharpe` -- the
    single routing point that keeps annualisation honest. Note ``sharpe``
    returns ``0.0`` for fewer than 20 IC observations (its small-sample guard),
    while ``ic_ir`` (un-annualised) is still reported for short series.
    """
    _check_method(method)
    s = pd.Series(ic_series, dtype=float).dropna()
    n = len(s)
    if n == 0:
        return IcSummary(horizon, method, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    mean = float(s.mean())
    std = float(s.std(ddof=1)) if n > 1 else 0.0
    ic_ir = mean / std if std > 1e-12 else 0.0
    t_stat = ic_ir * math.sqrt(n) if std > 1e-12 else 0.0
    ann = metrics.sharpe(s, periods_per_year)
    pct_pos = float((s > 0).mean())
    return IcSummary(
        horizon=horizon,
        method=method,
        n_periods=n,
        ic_mean=mean,
        ic_std=std,
        ic_ir=ic_ir,
        ic_ir_annualised=ann,
        t_stat=t_stat,
        pct_positive=pct_pos,
    )


def ic_decay(
    factor: pd.DataFrame,
    prices: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 2, 3, 5, 10, 21),
    *,
    method: str = "spearman",
    periods_per_year: int,
    log: bool = True,
    min_assets: int = 5,
) -> list[IcSummary]:
    """IC decay curve: one :class:`IcSummary` per forward horizon.

    For each ``h`` in ``horizons`` this computes forward returns, the
    cross-sectional IC series, and its summary. Scan the resulting ``ic_mean`` /
    ``t_stat`` across horizons to find where predictive power peaks and how fast
    it decays -- the peak ``|IC|`` horizon is the natural holding period.

    ``periods_per_year`` is required (no default) -- it must match the bar
    frequency of ``prices``, per the framework's annualisation discipline.
    """
    _check_method(method)
    summaries: list[IcSummary] = []
    for h in horizons:
        fr = forward_returns(prices, h, log=log)
        ic = cross_sectional_ic(factor, fr, method=method, min_assets=min_assets)
        summaries.append(
            summarise_ic(ic, periods_per_year=periods_per_year, horizon=h, method=method)
        )
    return summaries


# ── Quantile spread ───────────────────────────────────────────────────────────


def quantile_returns(
    factor: pd.DataFrame | pd.Series,
    prices: pd.DataFrame | pd.Series,
    horizon: int,
    *,
    n_quantiles: int = 5,
    periods_per_year: int,
    log: bool = True,
    min_assets: int | None = None,
) -> QuantileResult:
    """Mean forward return per factor quantile, plus the top-minus-bottom spread.

    Two modes, auto-detected from the factor shape:

    * **Cross-sectional** (panel with >= 2 columns): at each date the universe is
      bucketed into ``n_quantiles`` by that date's factor cross-section, then
      ``(quantile, forward_return)`` pairs are pooled across all dates. This is
      the Alphalens ``mean_return_by_quantile`` (by-date) convention and is
      free of look-ahead in the binning.
    * **Pooled** (single instrument): all ``(factor, forward_return)`` pairs are
      bucketed over the *whole* series. Quantile edges then use full-sample
      information -- a mild look-ahead acceptable for a diagnostic but not for a
      live decision. Flagged via ``cross_sectional=False`` on the result.

    A monotone ladder of ``mean_returns`` (``monotonicity`` near +1) with a
    positive, significant ``spread`` is the evidence the factor sorts returns.
    """
    if n_quantiles < 2:
        raise ValueError(f"n_quantiles must be >= 2, got {n_quantiles}")
    fr = forward_returns(prices, horizon, log=log)
    f_panel = _as_panel(factor)
    cross_sectional = f_panel.shape[1] >= 2

    # Collect forward returns per quantile bucket (0 = lowest factor).
    buckets: list[list[float]] = [[] for _ in range(n_quantiles)]

    if cross_sectional:
        r_panel = _as_panel(fr)
        cols = f_panel.columns.intersection(r_panel.columns)
        idx = f_panel.index.intersection(r_panel.index)
        f_panel = f_panel.loc[idx, cols]
        r_panel = r_panel.loc[idx, cols]
        floor = min_assets if min_assets is not None else n_quantiles
        for t in f_panel.index:
            pair = pd.concat([f_panel.loc[t], r_panel.loc[t]], axis=1).dropna()
            if len(pair) < floor:
                continue
            try:
                q = pd.qcut(pair.iloc[:, 0], n_quantiles, labels=False, duplicates="drop")
            except ValueError:
                continue
            # Ties collapsed a bin -> the ladder is incomparable for this date.
            if q.nunique() < n_quantiles:
                continue
            for qi, ret in zip(q.to_numpy(), pair.iloc[:, 1].to_numpy()):
                buckets[int(qi)].append(float(ret))
    else:
        f_ser = f_panel.iloc[:, 0]
        r_ser = fr if isinstance(fr, pd.Series) else fr.iloc[:, 0]
        pair = pd.concat([f_ser, r_ser], axis=1).dropna()
        if len(pair) >= n_quantiles:
            q = pd.qcut(pair.iloc[:, 0], n_quantiles, labels=False, duplicates="drop")
            for qi, ret in zip(q.to_numpy(), pair.iloc[:, 1].to_numpy()):
                buckets[int(qi)].append(float(ret))

    mean_returns = tuple(float(np.mean(b)) if b else float("nan") for b in buckets)
    counts = tuple(len(b) for b in buckets)

    top, bottom = buckets[-1], buckets[0]
    spread = float(np.mean(top) - np.mean(bottom)) if top and bottom else float("nan")
    if len(top) >= 2 and len(bottom) >= 2:
        t_res = stats.ttest_ind(top, bottom, equal_var=False)
        spread_tstat = float(t_res.statistic)
    else:
        spread_tstat = float("nan")

    # Monotonicity: does the mean-return ladder rise with quantile index?
    valid = [(i, m) for i, m in enumerate(mean_returns) if not math.isnan(m)]
    if len(valid) >= 2:
        idxs = [v[0] for v in valid]
        rets = [v[1] for v in valid]
        monotonicity = _pairwise_corr(np.array(idxs, float), np.array(rets, float), "spearman")
    else:
        monotonicity = float("nan")

    ann_spread = spread * periods_per_year / horizon if not math.isnan(spread) else float("nan")

    return QuantileResult(
        horizon=horizon,
        n_quantiles=n_quantiles,
        mean_returns=mean_returns,
        counts=counts,
        spread=spread,
        spread_tstat=spread_tstat,
        monotonicity=monotonicity,
        ann_spread=ann_spread,
        cross_sectional=cross_sectional,
    )
