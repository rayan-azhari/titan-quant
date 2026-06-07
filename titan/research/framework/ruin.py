"""Risk-of-ruin assessment module (V3.6 L65).

Computes formal P(ruin) over a deployment horizon for a strategy at a
specified deployment weight. Block-bootstrap based, preserves empirical
return distribution and autocorrelation.

Designed to fill the gap between MC MaxDD gate (per-strategy tail-risk
proxy) and actual portfolio survival probability at deployed size.

Two main entry points:

- ``assess_strategy_ruin``: single-strategy ruin probability at a
  deployment weight, with respect to a portfolio kill-switch threshold.
- ``assess_joint_ruin``: portfolio-level ruin probability across multiple
  simultaneously-deployed strategies (preserves empirical cross-
  correlations via aligned-index block bootstrap).

Usage::

    from titan.research.framework.ruin import assess_strategy_ruin

    res = assess_strategy_ruin(
        strategy_returns=stitched_oos_returns,   # net of cost
        deployment_weight=0.30,
        portfolio_kill_threshold=0.15,           # 15% portfolio NAV DD
        horizon_bars=252,                        # 1 year of daily bars
        block_size=21,                           # ~1 month
        n_paths=1000,
        seed=42,
    )
    print(res.report())
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

# ── Deployment gates (doctrine-versioned) ──────────────────────────────────

# Trading days in the V3.8 doctrine's 10-year ruin horizon (252 * 10).
HORIZON_10Y_DAILY = 2520


@dataclass(frozen=True)
class RuinGate:
    """A named, doctrine-versioned risk-of-ruin deployment gate.

    Bundles the SIMULATION parameters (horizon, the DD level that defines
    "ruin", the MaxDD-constraint level) with the pass/fail probability
    tolerances. Pairing them is the point: ``RuinAssessment.passes_gate``
    refuses to evaluate a gate against an assessment simulated under different
    params, so the V3.8 thresholds can NEVER be silently applied to a
    V3.7-simulated assessment (External Quant Audit 2026-05-29, P1-2 / C7 / H2
    / H3 -- the prior code shipped V3.7 defaults and no caller overrode them).

    Attributes:
        name: Doctrine label (e.g. "V3.7", "V3.8").
        horizon_bars: Forward MC horizon the assessment must be run at.
        kill_dd: Portfolio NAV drawdown that defines ruin (positive fraction;
            e.g. 0.40 for V3.8). ``p_kill_trip`` is measured against this.
        max_p_kill: Tolerance on P(ruin) over the horizon (e.g. 1e-3).
        maxdd_constraint_level: DD level for the P(MaxDD > X) constraint
            (e.g. 0.25). ``p_maxdd_gt_constraint`` is measured against this.
        max_p_maxdd_constraint: Tolerance on P(MaxDD > level) (e.g. 0.05).
        max_p_dd_50pct: Catastrophic-strategy guard -- P(strategy DD > 50% at
            full size).
    """

    name: str
    horizon_bars: int
    kill_dd: float
    max_p_kill: float
    maxdd_constraint_level: float
    max_p_maxdd_constraint: float
    max_p_dd_50pct: float = 0.05


# V3.7 (legacy L65): 1-year horizon, 15% kill, 1% P(kill).
GATE_V37 = RuinGate(
    name="V3.7",
    horizon_bars=252,
    kill_dd=0.15,
    max_p_kill=0.01,
    maxdd_constraint_level=0.25,
    max_p_maxdd_constraint=0.05,
    max_p_dd_50pct=0.05,
)

# V3.8 (Objective Reframe 2026-05-23 §2.1): 10-year horizon,
# P_kill(DD > 40%) <= 1e-3 AND P(MaxDD > 25%) <= 5%.
# *** RATIFIED binding objective (operator P4-4 decision, 2026-05-31): this gate
# IS the formalisation of "RoR ~= 0 / MaxDD < 20%". See
# directives/Objective Ratification (P4-4) 2026-05-31.md. Parameters are
# locked by tests/test_p4_4_objective_ratification.py. ***
GATE_V38 = RuinGate(
    name="V3.8",
    horizon_bars=HORIZON_10Y_DAILY,
    kill_dd=0.40,
    max_p_kill=1e-3,
    maxdd_constraint_level=0.25,
    max_p_maxdd_constraint=0.05,
    max_p_dd_50pct=0.05,
)

# V3.8-STRICT: an OPTIONAL, conservative tighter preset (P(MaxDD>20%) <= 5% AND
# P_kill(DD>30%) <= 1e-3). NOT the binding gate -- the 2026-05-31 P4-4 ratification
# chose GATE_V38 (25%/40%); see directives/Objective Ratification (P4-4) ...md.
# (Earlier code called this "operator-ratified"; that was a placeholder, never
# actually ratified.) Use it only for conservative sensitivity checks.
GATE_V38_STRICT = RuinGate(
    name="V3.8-STRICT",
    horizon_bars=HORIZON_10Y_DAILY,
    kill_dd=0.30,
    max_p_kill=1e-3,
    maxdd_constraint_level=0.20,
    max_p_maxdd_constraint=0.05,
    max_p_dd_50pct=0.05,
)


@dataclass(frozen=True)
class RuinAssessment:
    """Per-strategy or joint ruin-risk assessment.

    Attributes:
        deployment_weight: Fraction of full size the strategy is deployed at.
        portfolio_kill_threshold: Portfolio NAV drawdown that trips kill
            switch (e.g., 0.15 = 15%).
        horizon_bars: Forward horizon in bars (e.g., 252 = 1 year daily).
        p_kill_trip: Probability that scaled-strategy MaxDD crosses
            portfolio_kill_threshold within horizon.
        p_dd_50pct_strategy: Probability that the strategy itself
            crosses 50% MaxDD at full size (irrespective of deployment
            scaling) -- catastrophic-strategy probability.
        median_maxdd_at_size: Median MaxDD over MC paths at the deployed
            weight (negative number).
        p95_maxdd_at_size: 95th-percentile MaxDD at deployed weight
            (more negative than median).
        median_recovery_bars: Median bars to recover from median MaxDD,
            measured as bars until equity returns to the pre-DD peak.
            ``None`` if not recovered within horizon.
        n_paths: Number of MC paths simulated.
        block_size: Block size used in bootstrap.
        maxdd_constraint_level: DD level (positive fraction) at which
            ``p_maxdd_gt_constraint`` was measured (e.g. 0.25). Recorded so a
            ``RuinGate`` can verify the assessment matches its constraint level.
        p_maxdd_gt_constraint: Probability that MaxDD at the deployed weight
            exceeds ``maxdd_constraint_level`` over the horizon -- the quantity
            the V3.8 ``P(MaxDD > 25%) <= 5%`` constraint gates on.
    """

    deployment_weight: float
    portfolio_kill_threshold: float
    horizon_bars: int
    p_kill_trip: float
    p_dd_50pct_strategy: float
    median_maxdd_at_size: float
    p95_maxdd_at_size: float
    median_recovery_bars: int | None
    n_paths: int
    block_size: int
    maxdd_constraint_level: float = 0.25
    p_maxdd_gt_constraint: float = 0.0

    def passes(
        self,
        *,
        max_p_kill_trip: float = 0.01,
        max_p_dd_50pct: float = 0.05,
        max_p95_dd: float = -0.25,
    ) -> bool:
        """DEPRECATED legacy V3.7 L65 gate (bare defaults).

        Use ``passes_gate(GATE_V37)`` or ``passes_gate(GATE_V38)`` instead --
        the explicit, doctrine-versioned API. This method is retained only so
        the frozen historical audit scripts keep running; it emits a
        ``DeprecationWarning`` because the V3.8 doctrine (P_kill at DD>40%,
        1e-3, 10y) cannot be expressed through its bare defaults (audit P1-2).

        Pass criteria:
            - P(portfolio kill switch trips in horizon) < max_p_kill_trip
            - P(strategy DD > 50% at full size) < max_p_dd_50pct
            - 95th-percentile MaxDD at deployed weight better than
              max_p95_dd (i.e., closer to 0)
        """
        warnings.warn(
            "RuinAssessment.passes() is the legacy V3.7 gate with bare "
            "defaults; use passes_gate(GATE_V37) or passes_gate(GATE_V38) to "
            "make the doctrine version explicit (External Quant Audit "
            "2026-05-29, P1-2).",
            DeprecationWarning,
            stacklevel=2,
        )
        return (
            self.p_kill_trip < max_p_kill_trip
            and self.p_dd_50pct_strategy < max_p_dd_50pct
            and self.p95_maxdd_at_size >= max_p95_dd  # less-negative is better
        )

    def passes_gate(self, gate: RuinGate) -> bool:
        """Doctrine-versioned deployment gate.

        RAISES ``ValueError`` if this assessment was NOT simulated under the
        gate's horizon / kill-DD / MaxDD-constraint level -- so a V3.8 gate
        cannot be silently evaluated against a V3.7-simulated assessment (the
        exact drift the audit found: every caller applied V3.7 numbers). Run
        ``assess_*_ruin(gate=GATE_V38)`` to produce a matching assessment.

        Pass criteria (all must hold):
            - p_kill_trip           < gate.max_p_kill            (P(ruin at kill_dd))
            - p_maxdd_gt_constraint < gate.max_p_maxdd_constraint
            - p_dd_50pct_strategy   < gate.max_p_dd_50pct
        """
        if self.horizon_bars != gate.horizon_bars:
            raise ValueError(
                f"Assessment horizon ({self.horizon_bars}b) != gate "
                f"'{gate.name}' horizon ({gate.horizon_bars}b). Re-run "
                f"assess_*_ruin(gate=GATE_{gate.name.replace('.', '')}) to match."
            )
        if abs(self.portfolio_kill_threshold - gate.kill_dd) > 1e-9:
            raise ValueError(
                f"Assessment kill-DD ({self.portfolio_kill_threshold:.2%}) != "
                f"gate '{gate.name}' kill-DD ({gate.kill_dd:.2%}). Re-run under the gate."
            )
        if abs(self.maxdd_constraint_level - gate.maxdd_constraint_level) > 1e-9:
            raise ValueError(
                f"Assessment MaxDD-constraint level "
                f"({self.maxdd_constraint_level:.2%}) != gate '{gate.name}' "
                f"level ({gate.maxdd_constraint_level:.2%})."
            )
        return (
            self.p_kill_trip < gate.max_p_kill
            and self.p_maxdd_gt_constraint < gate.max_p_maxdd_constraint
            and self.p_dd_50pct_strategy < gate.max_p_dd_50pct
        )

    def report(self) -> str:
        """Human-readable summary string."""
        rec_str = (
            f"{self.median_recovery_bars} bars"
            if self.median_recovery_bars is not None
            else f"NOT RECOVERED within {self.horizon_bars} bars"
        )
        return (
            f"RuinAssessment(weight={self.deployment_weight:.2%}, "
            f"kill_thresh={self.portfolio_kill_threshold:.2%}, "
            f"horizon={self.horizon_bars}b, n_paths={self.n_paths})\n"
            f"  P(kill @ {self.portfolio_kill_threshold:.0%}) = {self.p_kill_trip:.3%}\n"
            f"  P(MaxDD > {self.maxdd_constraint_level:.0%}) = {self.p_maxdd_gt_constraint:.3%}\n"
            f"  P(strategy DD > 50% full size) = {self.p_dd_50pct_strategy:.3%}\n"
            f"  median MaxDD at deployed weight = {self.median_maxdd_at_size:.3%}\n"
            f"  95th-pct MaxDD at deployed weight = {self.p95_maxdd_at_size:.3%}\n"
            f"  median recovery from median DD = {rec_str}\n"
            f"  (apply a gate via passes_gate(GATE_V37|GATE_V38))"
        )


def _block_bootstrap_path(
    returns_arr: np.ndarray, horizon_bars: int, block_size: int, rng: np.random.Generator
) -> np.ndarray:
    """Generate one bootstrap path of length horizon_bars from returns_arr."""
    n_blocks = (horizon_bars + block_size - 1) // block_size
    n_available = len(returns_arr) - block_size + 1
    if n_available <= 0:
        return returns_arr[:horizon_bars]
    starts = rng.integers(0, n_available, size=n_blocks)
    path = np.concatenate([returns_arr[s : s + block_size] for s in starts])
    return path[:horizon_bars]


def _max_drawdown(returns_path: np.ndarray) -> float:
    """Maximum drawdown of the compounded equity curve (non-positive float).

    ``returns_path`` is a series of SIMPLE (arithmetic) per-bar returns.
    Equity compounds geometrically from a starting capital of 1.0 and the
    drawdown is the fractional decline from the running peak,
    ``equity / running_peak - 1`` -- the same convention as
    ``titan.research.metrics.max_drawdown`` and ``crisis_stress``.

    V3.8 audit fix P1-1 (External Quant Audit 2026-05-29): the prior
    implementation used ``np.cumsum`` (additive) and compared a raw
    cumulative-return delta against fractional thresholds (e.g. -0.15),
    which is both internally inconsistent with the rest of the framework and
    not a fraction of equity. Callers MUST pass simple returns; a caller that
    passes log returns will be silently mis-scaled (tracked for the P2 caller
    sweep). The starting 1.0 is prepended so a first-bar loss is a real
    drawdown from 1.0, matching ``metrics.max_drawdown``.
    """
    if len(returns_path) == 0:
        return 0.0
    eq = np.empty(len(returns_path) + 1, dtype=float)
    eq[0] = 1.0
    np.cumprod(1.0 + returns_path, out=eq[1:])
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1.0
    return float(dd.min())


def _recovery_bars(returns_path: np.ndarray, threshold_dd: float) -> int | None:
    """Bars from the first crossing of ``threshold_dd`` to recovery to the
    pre-drawdown peak. ``None`` if not recovered within the path.

    ``threshold_dd`` is a fractional drawdown (negative). Computed on the
    geometrically-compounded equity curve (P1-1), consistent with
    ``_max_drawdown``.
    """
    eq = np.cumprod(1.0 + returns_path)
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1.0
    if dd.min() > threshold_dd:  # threshold_dd is negative
        return 0  # never had a worse DD
    # Find first bar where DD hits threshold or worse.
    dd_hit_idx = int(np.argmax(dd <= threshold_dd))
    # Equity peak at that point.
    peak_at_dd = peak[dd_hit_idx]
    # Find first subsequent bar where equity recovers to that peak.
    forward = eq[dd_hit_idx:]
    recovery_mask = forward >= peak_at_dd
    if not recovery_mask.any():
        return None
    return int(np.argmax(recovery_mask))


def assess_strategy_ruin(
    strategy_returns: pd.Series,
    *,
    deployment_weight: float,
    gate: RuinGate | None = None,
    portfolio_kill_threshold: float = 0.15,
    horizon_bars: int = 252,
    maxdd_constraint_level: float = 0.25,
    block_size: int = 21,
    n_paths: int = 1000,
    seed: int = 42,
) -> RuinAssessment:
    """Compute single-strategy ruin probability at deployed weight.

    Parameters:
        strategy_returns: Per-bar net SIMPLE (arithmetic) returns, OOS
            preferred -- NOT log returns (drawdown compounds geometrically
            via 1+r; see ``_max_drawdown``, P1-1). Must be sufficiently long
            to bootstrap; recommend at least 2 years.
        deployment_weight: Fraction of full Kelly the strategy is deployed
            at. E.g., 0.30 = 30% of full size.
        gate: Optional ``RuinGate`` (e.g. ``GATE_V38``). When provided it
            OVERRIDES ``portfolio_kill_threshold``, ``horizon_bars`` and
            ``maxdd_constraint_level`` with the gate's values, so the
            assessment is simulated under the doctrine being gated on and
            ``passes_gate(gate)`` can be applied without raising (audit P1-2).
        portfolio_kill_threshold: Portfolio NAV drawdown level that
            trips the kill switch (positive number, e.g., 0.15 = 15%). Ignored
            if ``gate`` is given.
        horizon_bars: Forward horizon to simulate (e.g., 252 daily bars; the
            V3.8 doctrine is 2520 = 10y). Ignored if ``gate`` is given.
        maxdd_constraint_level: DD level for the ``P(MaxDD > X)`` constraint
            (default 0.25 = the V3.8 25% level). Ignored if ``gate`` is given.
        block_size: Block bootstrap size (preserves autocorrelation).
        n_paths: Number of MC paths.
        seed: RNG seed.

    Returns:
        RuinAssessment with key gauges.
    """
    if gate is not None:
        portfolio_kill_threshold = gate.kill_dd
        horizon_bars = gate.horizon_bars
        maxdd_constraint_level = gate.maxdd_constraint_level

    rets = strategy_returns.dropna().to_numpy()
    if len(rets) < block_size * 4:
        raise ValueError(
            f"strategy_returns too short ({len(rets)} bars) for block_size={block_size}"
        )
    if horizon_bars > 4 * len(rets):
        warnings.warn(
            f"Ruin horizon ({horizon_bars}b) >> sample ({len(rets)}b): each block is "
            f"re-used ~{horizon_bars / max(len(rets), 1):.0f}x, so tail estimates over "
            "this horizon are bootstrap-bound (low effective sample coverage). Prefer a "
            "longer return sample or an analytic first-passage cross-check.",
            stacklevel=2,
        )

    rng = np.random.default_rng(seed)
    threshold = -abs(portfolio_kill_threshold)
    constraint = -abs(maxdd_constraint_level)

    kill_trips = 0
    maxdd_gt_constraint = 0
    dd_50_strategy = 0
    maxdds_at_weight: list[float] = []
    recoveries: list[int | None] = []

    for _ in range(n_paths):
        path = _block_bootstrap_path(rets, horizon_bars, block_size, rng)
        scaled = path * deployment_weight
        maxdd_scaled = _max_drawdown(scaled)
        maxdds_at_weight.append(maxdd_scaled)
        if maxdd_scaled <= threshold:
            kill_trips += 1
        if maxdd_scaled <= constraint:
            maxdd_gt_constraint += 1
        # Strategy-only (full size) catastrophic DD
        maxdd_full = _max_drawdown(path)
        if maxdd_full <= -0.50:
            dd_50_strategy += 1
        # Median-recovery: at scaled weight
        # Use the realised MaxDD as threshold so we measure "recover from this DD"
        recoveries.append(_recovery_bars(scaled, threshold_dd=maxdd_scaled * 0.5))

    maxdds_arr = np.array(maxdds_at_weight)
    valid_recoveries = [r for r in recoveries if r is not None and r > 0]

    return RuinAssessment(
        deployment_weight=deployment_weight,
        portfolio_kill_threshold=portfolio_kill_threshold,
        horizon_bars=horizon_bars,
        p_kill_trip=kill_trips / n_paths,
        p_dd_50pct_strategy=dd_50_strategy / n_paths,
        median_maxdd_at_size=float(np.median(maxdds_arr)),
        p95_maxdd_at_size=float(np.percentile(maxdds_arr, 5)),  # 5th percentile = 95% worst
        median_recovery_bars=int(np.median(valid_recoveries)) if valid_recoveries else None,
        n_paths=n_paths,
        block_size=block_size,
        maxdd_constraint_level=maxdd_constraint_level,
        p_maxdd_gt_constraint=maxdd_gt_constraint / n_paths,
    )


def assess_joint_ruin(
    strategy_returns: dict[str, pd.Series],
    *,
    deployment_weights: dict[str, float],
    gate: RuinGate | None = None,
    portfolio_kill_threshold: float = 0.15,
    horizon_bars: int = 252,
    maxdd_constraint_level: float = 0.25,
    block_size: int = 21,
    n_paths: int = 1000,
    seed: int = 42,
) -> RuinAssessment:
    """Compute portfolio-level joint ruin probability.

    Aligns all strategy returns on the common index, applies deployment
    weights, sums for a portfolio return series, then block-bootstraps
    the PORTFOLIO series to preserve empirical cross-correlations.

    NOTE (audit H4): summing-then-bootstrapping a linear fixed-weight portfolio
    preserves only the *historically realised* contemporaneous co-movement; it
    cannot manufacture a crisis correlations->1 regime. For the
    diversification-evaporates-in-the-tail stress (audit H4/H19), use
    :func:`assess_joint_ruin_stressed`, which injects crisis blocks where the
    sleeves are forced to co-move.

    Parameters:
        strategy_returns: Mapping name -> per-bar SIMPLE returns.
        deployment_weights: Mapping name -> deployment weight (must sum
            to <= 1.0 for a valid portfolio).
        gate: Optional ``RuinGate``; overrides kill-DD / horizon /
            MaxDD-constraint level so ``passes_gate(gate)`` applies. See
            ``assess_strategy_ruin``.
        ... (rest same as assess_strategy_ruin).
    """
    common_idx = None
    for name, r in strategy_returns.items():
        r = r.dropna()
        if common_idx is None:
            common_idx = r.index
        else:
            common_idx = common_idx.intersection(r.index)
    if common_idx is None or len(common_idx) == 0:
        raise ValueError("No common index across strategies")

    weight_sum = sum(deployment_weights.values())
    if weight_sum > 1.0 + 1e-9:
        raise ValueError(f"Sum of deployment_weights={weight_sum:.3f} exceeds 1.0")

    portfolio_returns = pd.Series(0.0, index=common_idx)
    for name, ret in strategy_returns.items():
        weight = deployment_weights.get(name, 0.0)
        if weight == 0.0:
            continue
        portfolio_returns = portfolio_returns + (ret.reindex(common_idx).fillna(0.0) * weight)

    # The "deployment_weight" of the portfolio is 1.0 (it's already weighted).
    return assess_strategy_ruin(
        portfolio_returns,
        deployment_weight=1.0,
        gate=gate,
        portfolio_kill_threshold=portfolio_kill_threshold,
        horizon_bars=horizon_bars,
        maxdd_constraint_level=maxdd_constraint_level,
        block_size=block_size,
        n_paths=n_paths,
        seed=seed,
    )


def assess_joint_ruin_stressed(
    strategy_returns: dict[str, pd.Series],
    *,
    deployment_weights: dict[str, float],
    gate: RuinGate | None = None,
    portfolio_kill_threshold: float = 0.15,
    horizon_bars: int = 252,
    maxdd_constraint_level: float = 0.25,
    block_size: int = 21,
    n_paths: int = 1000,
    seed: int = 42,
    crisis_rho: float = 0.9,
    crisis_prob: float = 0.10,
    crisis_vol_mult: float = 1.5,
    crisis_drift_sigmas: float = 1.0,
) -> RuinAssessment:
    """Joint ruin under a STRESSED-correlation scenario (audit P1-3 / H4 / H19).

    The base :func:`assess_joint_ruin` can only reproduce the calm-sample
    correlation structure, so it certifies the book as safe on a
    diversification benefit that *evaporates exactly when it is needed*. This
    function injects crisis blocks: with probability ``crisis_prob`` a block is
    a "crisis" block in which every sleeve is forced to co-move at pairwise
    correlation ``crisis_rho`` via a one-factor model
    ``r_i = mu_i + crisis_vol_mult * sigma_i * (sqrt(rho)*z_c + sqrt(1-rho)*z_i)``
    with a common factor ``z_c`` drawn with a negative mean of
    ``crisis_drift_sigmas`` (a coordinated drawdown). Calm blocks use the
    empirical aligned-index block bootstrap (preserving historical structure).

    The result is directly comparable to :func:`assess_joint_ruin` (same gauges,
    ``deployment_weight=1.0``) but should report a HIGHER P_kill / P(MaxDD>X)
    because the tail correlation is no longer benign. Pass a ``gate`` so
    ``passes_gate(gate)`` can be applied.

    The crisis parameters are STRESS ASSUMPTIONS, not estimates -- they are the
    operator's "what if correlations go to 1 in a crash" scenario. Defaults:
    rho 0.9, 10% of blocks, 1.5x vol, common factor centred at -1 sigma.
    """
    if gate is not None:
        portfolio_kill_threshold = gate.kill_dd
        horizon_bars = gate.horizon_bars
        maxdd_constraint_level = gate.maxdd_constraint_level

    # Align all sleeves on the common index.
    names = [n for n, w in deployment_weights.items() if w != 0.0]
    if not names:
        raise ValueError("No strategies with non-zero weight")
    common_idx = None
    for n in names:
        idx = strategy_returns[n].dropna().index
        common_idx = idx if common_idx is None else common_idx.intersection(idx)
    if common_idx is None or len(common_idx) < block_size * 4:
        raise ValueError("Insufficient common history for stressed joint ruin")

    weight_sum = sum(deployment_weights[n] for n in names)
    if weight_sum > 1.0 + 1e-9:
        raise ValueError(f"Sum of deployment_weights={weight_sum:.3f} exceeds 1.0")

    mat = np.column_stack(
        [strategy_returns[n].reindex(common_idx).fillna(0.0).to_numpy() for n in names]
    )  # (T, K) simple returns
    w = np.array([deployment_weights[n] for n in names], dtype=float)
    mu = mat.mean(axis=0)
    sigma = mat.std(axis=0, ddof=1)
    t_avail = mat.shape[0]
    n_blocks = (horizon_bars + block_size - 1) // block_size
    n_available = t_avail - block_size + 1
    rho = float(min(max(crisis_rho, 0.0), 1.0))
    b = np.sqrt(rho)
    b_idio = np.sqrt(max(0.0, 1.0 - rho))
    rng = np.random.default_rng(seed)

    threshold = -abs(portfolio_kill_threshold)
    constraint = -abs(maxdd_constraint_level)
    kill_trips = 0
    maxdd_gt_constraint = 0
    dd50 = 0
    maxdds: list[float] = []

    for _ in range(n_paths):
        port_blocks: list[np.ndarray] = []
        for _b in range(n_blocks):
            if rng.random() < crisis_prob:
                # Crisis block: one-factor coordinated drawdown across sleeves.
                z_c = rng.normal(-crisis_drift_sigmas, 1.0, size=block_size)
                z_i = rng.normal(0.0, 1.0, size=(block_size, len(names)))
                shocks = b * z_c[:, None] + b_idio * z_i  # (block, K), corr ~ rho
                block = mu[None, :] + crisis_vol_mult * sigma[None, :] * shocks
            else:
                start = int(rng.integers(0, n_available))
                block = mat[start : start + block_size]
            port_blocks.append(block @ w)  # weighted portfolio return per bar
        path = np.concatenate(port_blocks)[:horizon_bars]
        mdd = _max_drawdown(path)
        maxdds.append(mdd)
        if mdd <= threshold:
            kill_trips += 1
        if mdd <= constraint:
            maxdd_gt_constraint += 1
        if mdd <= -0.50:
            dd50 += 1

    maxdds_arr = np.array(maxdds)
    return RuinAssessment(
        deployment_weight=1.0,
        portfolio_kill_threshold=portfolio_kill_threshold,
        horizon_bars=horizon_bars,
        p_kill_trip=kill_trips / n_paths,
        p_dd_50pct_strategy=dd50 / n_paths,
        median_maxdd_at_size=float(np.median(maxdds_arr)),
        p95_maxdd_at_size=float(np.percentile(maxdds_arr, 5)),
        median_recovery_bars=None,
        n_paths=n_paths,
        block_size=block_size,
        maxdd_constraint_level=maxdd_constraint_level,
        p_maxdd_gt_constraint=maxdd_gt_constraint / n_paths,
    )


__all__ = [
    "GATE_V37",
    "GATE_V38",
    "GATE_V38_STRICT",
    "HORIZON_10Y_DAILY",
    "RuinAssessment",
    "RuinGate",
    "assess_joint_ruin",
    "assess_joint_ruin_stressed",
    "assess_strategy_ruin",
]
