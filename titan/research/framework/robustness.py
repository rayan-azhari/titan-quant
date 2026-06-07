"""Noise-injection robustness gate (Varma).

Specified in directives/Strategy Backlog 2026-05-14.md §J3. Implements
Varma's "Signal vs Noise" robustness test: a strategy whose realised
Sharpe degrades catastrophically under a small perturbation of its inputs
is fragile -- it has overfit to specific data quirks rather than to a
durable signal.

The test:
    1. Take the strategy's returns on the unperturbed data: ``sr_base``.
    2. For each noise level ``σ`` in cfg.noise_levels:
         a. Add zero-mean Gaussian noise to each input series, scaled
            by ``σ × σ_input`` (i.e. relative to the input's own σ).
         b. Re-run the strategy on the perturbed series for ``cfg.n_trials``
            independent draws.
         c. Compute the mean and 5th-percentile Sharpe across draws.
    3. Report degradation = (sr_base - sr_at_noise_level) / |sr_base|.
       The strategy PASSES the gate iff degradation < cfg.max_degradation
       at the LARGEST noise level. (Equivalently: Sharpe must remain at
       least ``(1 - max_degradation) * sr_base`` under 0.3σ relative noise.)

Why this matters in the V2.0 framework:
    - Detects parameter-spike vs parameter-plateau (V3.2) without
      requiring an explicit grid sweep.
    - Catches "lucky-seed" overfits where a strategy passes its own gates
      on one realisation but breaks under any reasonable perturbation.
    - Becomes the 5th axis of the decision matrix when wired into
      ``decide(...)`` (planned follow-up; for now the gate is run
      standalone and the result appended to the audit log).

Caveats:
    - For path-dependent strategies (anything with stops, regime gates,
      pyramiding), the response to noise can be highly non-linear. A
      strategy that PASSES this test is robust to small input noise;
      a strategy that FAILS is fragile. PASS does not imply alpha.
    - The noise is on the INPUT (typically close prices). Strategies
      that derive features from close are tested end-to-end.
    - If the strategy uses external regime indicators (e.g. VIX), pass
      those in ``extra_series`` and they'll be perturbed with the same
      σ scaling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from titan.research.metrics import sharpe


@dataclass(frozen=True)
class NoiseConfig:
    """Configuration for the noise-injection robustness gate.

    Attributes:
        noise_levels: σ multipliers relative to each input series' own σ.
                      e.g. [0.1, 0.3, 0.5] perturbs by 10%, 30%, 50% of the
                      input's standard deviation.
        n_trials: Independent random draws per noise level.
        max_degradation: PASS gate. Sharpe at the LARGEST noise level must
                         remain at least (1 - max_degradation) of the base
                         Sharpe. Default 0.3 means "Sharpe may drop at most
                         30% under 0.5σ relative noise".
        seed: Base RNG seed; each trial uses seed + trial_id.
        method: "additive" -> noise added to log-prices then re-cumprod'd
                "multiplicative" -> noise applied to bar returns directly
                Default "additive" preserves price-path realism (no negative
                prices) and matches how vol-of-vol papers add perturbations.
    """

    noise_levels: tuple[float, ...] = (0.1, 0.3, 0.5)
    n_trials: int = 10
    max_degradation: float = 0.30
    seed: int = 42
    method: str = "additive"


@dataclass(frozen=True)
class NoiseLevelResult:
    """Per-noise-level summary across the n_trials draws."""

    noise_level: float
    n_trials: int
    sharpes: tuple[float, ...]
    sharpe_mean: float
    sharpe_p5: float
    sharpe_median: float
    sharpe_p95: float
    degradation_mean: float  # (sr_base - sr_mean) / |sr_base|
    degradation_p5: float  # (sr_base - sr_p5) / |sr_base|  -- worst case


@dataclass(frozen=True)
class NoiseRobustnessResult:
    """Aggregate result returned by run_noise_robustness.

    `passes` is True iff at every noise level, degradation_mean < cfg.max_degradation.
    `worst_case_passes` is True iff at every noise level,
    degradation_p5 < cfg.max_degradation (stricter; uses 5th percentile).
    """

    base_sharpe: float
    per_level: tuple[NoiseLevelResult, ...]
    cfg: NoiseConfig
    passes: bool
    worst_case_passes: bool


def _perturb_additive(
    series: pd.Series,
    noise_sigma: float,
    rng: np.random.Generator,
) -> pd.Series:
    """Add Gaussian noise to log-prices, then re-exponentiate.

    Preserves: positivity, rough scale.
    Disturbs: short-horizon return autocorrelation. (That's the point.)
    """
    log_p = np.log(series.astype(float))
    sigma_log = float(log_p.diff().std(ddof=1))
    if not np.isfinite(sigma_log) or sigma_log <= 0:
        return series.copy()
    noise = rng.normal(0.0, noise_sigma * sigma_log, size=len(log_p))
    perturbed_log = log_p.values + noise
    out = pd.Series(np.exp(perturbed_log), index=series.index, name=series.name)
    return out


def _perturb_multiplicative(
    series: pd.Series,
    noise_sigma: float,
    rng: np.random.Generator,
) -> pd.Series:
    """Multiply bar-returns by (1 + ε) where ε ~ N(0, noise_sigma * σ_ret).

    Re-build the price path from the perturbed returns.
    """
    rets = series.astype(float).pct_change().fillna(0.0)
    sigma_r = float(rets.std(ddof=1))
    if not np.isfinite(sigma_r) or sigma_r <= 0:
        return series.copy()
    noise = rng.normal(0.0, noise_sigma * sigma_r, size=len(rets))
    perturbed_rets = rets.values + noise
    out = pd.Series(
        series.iloc[0] * np.cumprod(1.0 + perturbed_rets),
        index=series.index,
        name=series.name,
    )
    return out


def run_noise_robustness(
    closes: pd.DataFrame,
    strategy_returns_fn: Callable[[pd.DataFrame], pd.Series],
    *,
    periods_per_year: int,
    cfg: NoiseConfig | None = None,
) -> NoiseRobustnessResult:
    """Run the Varma noise-injection robustness test.

    Parameters
    ----------
    closes:
        The unperturbed input DataFrame the strategy consumes (typically
        close prices).
    strategy_returns_fn:
        Callable mapping ``closes -> per-bar return Series`` -- the same
        contract as ``gem_returns``, ``run_block_mc.strategy_fn``, etc.
        This is what the audit harness already builds.
    periods_per_year:
        Annualisation factor matching the bar frequency.
    cfg:
        NoiseConfig. Default is the Varma-recommended sweep.

    Returns:
    -------
    NoiseRobustnessResult with per-level diagnostics and a binary `passes` flag.
    """
    if cfg is None:
        cfg = NoiseConfig()
    if cfg.method not in ("additive", "multiplicative"):
        raise ValueError(
            f"NoiseConfig.method must be 'additive' or 'multiplicative', got {cfg.method!r}"
        )

    perturb = _perturb_additive if cfg.method == "additive" else _perturb_multiplicative

    # Base Sharpe (unperturbed).
    base_ret = strategy_returns_fn(closes)
    sr_base = float(sharpe(base_ret, periods_per_year=periods_per_year))

    per_level_results: list[NoiseLevelResult] = []
    abs_base = abs(sr_base) if abs(sr_base) > 1e-9 else 1.0  # guard /0

    for level in cfg.noise_levels:
        sharpes_at_level: list[float] = []
        for trial in range(cfg.n_trials):
            rng = np.random.default_rng(cfg.seed + 1000 * int(level * 100) + trial)
            perturbed = closes.copy()
            for col in closes.columns:
                perturbed[col] = perturb(closes[col], level, rng)
            try:
                ret = strategy_returns_fn(perturbed)
            except Exception:
                # Treat strategy failure under noise as Sharpe = 0 (graceless degradation).
                sharpes_at_level.append(0.0)
                continue
            sh = float(sharpe(ret, periods_per_year=periods_per_year))
            if not np.isfinite(sh):
                sh = 0.0
            sharpes_at_level.append(sh)

        arr = np.asarray(sharpes_at_level)
        mean_sh = float(arr.mean())
        p5 = float(np.quantile(arr, 0.05))
        p50 = float(np.quantile(arr, 0.50))
        p95 = float(np.quantile(arr, 0.95))
        per_level_results.append(
            NoiseLevelResult(
                noise_level=level,
                n_trials=cfg.n_trials,
                sharpes=tuple(round(s, 4) for s in sharpes_at_level),
                sharpe_mean=round(mean_sh, 4),
                sharpe_p5=round(p5, 4),
                sharpe_median=round(p50, 4),
                sharpe_p95=round(p95, 4),
                degradation_mean=round((sr_base - mean_sh) / abs_base, 4),
                degradation_p5=round((sr_base - p5) / abs_base, 4),
            )
        )

    passes = all(r.degradation_mean < cfg.max_degradation for r in per_level_results)
    worst_case_passes = all(r.degradation_p5 < cfg.max_degradation for r in per_level_results)

    return NoiseRobustnessResult(
        base_sharpe=round(sr_base, 4),
        per_level=tuple(per_level_results),
        cfg=cfg,
        passes=passes,
        worst_case_passes=worst_case_passes,
    )
