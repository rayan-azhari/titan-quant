"""Portfolio Risk Manager -- Live Portfolio-Level Risk Control.

Aggregates per-strategy equity, tracks portfolio-level drawdown, computes an
EWMA realised-vol overlay on a **timestamped, daily-aligned** basis, and
exposes a composite ``scale_factor`` plus a sticky ``halt_all`` kill switch.

Architecture (April 2026 rewrite)
---------------------------------
Each strategy owns a ``StrategyEquityTracker`` (see ``strategy_equity.py``)
that computes a true per-strategy equity curve (seed + realised + MTM). The
strategy calls ``portfolio_risk_manager.update(strategy_id, equity, ts)``
once per bar with that equity and an explicit UTC timestamp.

The risk manager stores each strategy's equity history as a **timestamped
``pd.Series``**, not a raw deque of floats. All variance / correlation math
happens after resampling every strategy onto a shared business-day grid --
no more mixing hourly and daily samples in a ``sqrt(252)`` annualisation.

Wall-clock gating
-----------------
Portfolio-vol recomputation, correlation checks, and allocator rebalances
are triggered by the *calendar date* changing, not by a counter of ticks.
An H1 strategy that fires 24 bars a day and a D1 strategy that fires once
therefore both cause "daily" work to happen once per day.

Halt persistence
----------------
The kill-switch state is persisted to ``.tmp/portfolio_halt.json``. On
startup the manager re-reads that file so a crashed + restarted process
does not silently un-halt. ``reset_halt`` is the only way to clear it and
writes the reset to the same file with an operator timestamp.

Scale factor composition
------------------------
    scale_factor = min(dd_scale, vol_scale, regime_scale, v38_dd_throttle)

    dd_scale       : Drawdown heat -- linear scale-down [0.25, 1.0] on the
                     all-time-HWM drawdown.
    vol_scale      : Vol-targeting -- target_vol / realised_vol, clipped.
    regime_scale   : min(vix_scale, atr_scale) from VIX tiers + ATR percentile.
    v38_dd_throttle: V3.8 envelope DD throttle (P0-2) -- hysteresis step
                     {1.0, 0.5} on the rolling-60d-peak drawdown. Folded in by
                     MIN (never a product) so it shares the dd_scale dimension
                     without de-risking the book twice; 1.0 (inert) whenever
                     the envelope is disabled via V38_ENVELOPE_DISABLE.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from titan.research.framework.covariance import DEFAULT_CRISIS_RHO, shrink_covariance
from titan.research.framework.dd_throttle import (
    DEFAULT_PEAK_WINDOW_BARS,
    DdThrottleState,
    compute_rolling_dd_from_peak,
    initial_throttle_state,
    update_throttle,
)
from titan.research.framework.leverage_envelope import (
    PositionSnapshot,
    evaluate_leverage_envelope,
)
from titan.research.framework.portfolio_heat import (
    PositionHeat,
    evaluate_heat_envelope,
)
from titan.research.metrics import BARS_PER_YEAR, annualize_vol
from titan.risk.drift_monitor import drift_band_decision, realised_rolling_maxdd
from titan.risk.v38_envelope import (
    OnBarDecisions,
    PreTradeDecision,
    PreTradeRejectionReason,
    StrategyV38Config,
    StrategyV38Telemetry,
    V38Telemetry,
)
from titan.strategies.regime_filter.regime import (
    RegimeResult,
    is_crisis_regime,
)
from titan.strategies.v38_position_interface import (
    get_heat_from,
    get_snapshots_from,
)

logger = logging.getLogger(__name__)

# ── Default configuration (overridden by config/risk.toml [portfolio]) ─────────

_DEFAULT_CONFIG: dict = {
    "portfolio_max_dd_pct": 15.0,
    "portfolio_heat_scale_pct": 10.0,
    # P0-4: per-trade R cap -- reject (V3.8) any single candidate whose stop
    # risk exceeds this fraction of total equity. 2% per Objective Reframe.
    "per_trade_r_cap_pct": 2.0,
    "correlation_window_days": 60,
    "correlation_halt_threshold": 0.85,
    "portfolio_max_single_pct": 60.0,
    "vol_target_ann_pct": 12.0,
    "vol_ewma_lambda": 0.94,
    "vol_scale_min": 0.25,
    "vol_scale_max": 2.0,
    "vix_tier_1": 17.8,
    "vix_tier_2": 23.1,
    "vix_tier_3": 30.0,
    "atr_pct_low": 25.0,
    "atr_pct_high": 75.0,
    "atr_pct_extreme": 90.0,
    # Max rows stored per strategy (~4 years of business days).
    "history_max_days": 1000,
}

_HALT_STATE_PATH = Path(__file__).resolve().parents[2] / ".tmp" / "portfolio_halt.json"
# P0-9: the MC-predicted MaxDD band, written daily by scripts/monitor_live_drift.py.
_DD_BAND_PATH = Path(__file__).resolve().parents[2] / ".tmp" / "dd_band.json"

# P0-12: cap the rolling V3.8 NAV history (on_bar ticks ~252/yr) so it can't
# grow unbounded. 512 daily bars (~2y) is ample for the 60-bar rolling-DD peak.
_V38_NAV_HISTORY_CAP = 512

# ── V3.8 envelope: emergency global disable ───────────────────────────────────
# Per `PRM Integration V3.8` §9 + Sleeve B Phase A runbook §8: if a bug in the
# V3.8 wiring causes the PRM to reject every trade (or otherwise misbehave),
# the operator can set `V38_ENVELOPE_DISABLE=1` in the container env and
# restart. Both V3.8 entry points (`check_pre_trade`, `on_bar`) then short-
# circuit to inert "all pass" decisions, restoring V3.7 behaviour without a
# code rollback. Re-evaluated on every call so tests can monkeypatch.
_V38_DISABLE_ENV_VAR = "V38_ENVELOPE_DISABLE"
_V38_DISABLE_TRUTHY = frozenset({"1", "true", "yes", "on"})
_v38_disable_warned: bool = False


def _v38_globally_disabled() -> bool:
    """True iff the V3.8 envelope is bypassed via `V38_ENVELOPE_DISABLE=1`."""
    global _v38_disable_warned
    raw = os.environ.get(_V38_DISABLE_ENV_VAR, "").strip().lower()
    if raw not in _V38_DISABLE_TRUTHY:
        return False
    if not _v38_disable_warned:
        logger.warning(
            "[PortfolioRM] V3.8 envelope DISABLED via %s=%s -- "
            "check_pre_trade + on_bar bypass all V3.8 logic until restart "
            "with env var unset. V3.7 ruin gate (halt_all) is unaffected.",
            _V38_DISABLE_ENV_VAR,
            os.environ.get(_V38_DISABLE_ENV_VAR),
        )
        _v38_disable_warned = True
    return True


# ── Per-strategy state ─────────────────────────────────────────────────────────


@dataclass
class _StrategyState:
    strategy_id: str
    initial_equity: float
    current_equity: float
    equity_hwm: float
    # Tick-level equity samples, indexed by UTC timestamp. Resampled to daily
    # on demand for vol / correlation work -- the raw samples remain here so
    # we can still compute per-tick drawdown.
    samples: "pd.Series" = field(default_factory=lambda: pd.Series(dtype=float))

    @property
    def drawdown_pct(self) -> float:
        if self.equity_hwm <= 0:
            return 0.0
        return (self.current_equity - self.equity_hwm) / self.equity_hwm

    def append(self, ts: pd.Timestamp, equity: float, max_rows: int) -> None:
        # Dedupe same-timestamp overwrites (re-emitted bars).
        self.samples.loc[ts] = equity
        if len(self.samples) > max_rows * 24:  # cap intraday sample count
            self.samples = self.samples.iloc[-(max_rows * 24) :]

    def daily_equity(self, max_days: int) -> pd.Series:
        """Resample to business-day last-observation; returned series is
        right-bounded by today. Cap to ``max_days`` rows.
        """
        if self.samples.empty:
            return pd.Series(dtype=float)
        s = self.samples.copy()
        s.index = pd.DatetimeIndex(s.index).tz_convert("UTC").tz_localize(None)
        daily = s.resample("B").last().dropna()
        return daily.iloc[-max_days:]


# ── Portfolio Risk Manager ─────────────────────────────────────────────────────


class PortfolioRiskManager:
    """Live portfolio-level risk manager (timestamp-aware)."""

    def __init__(self, config: dict | None = None) -> None:
        self._config: dict = {**_DEFAULT_CONFIG, **(config or {})}
        self._strategies: dict[str, _StrategyState] = {}
        self._portfolio_hwm: float | None = None
        self._halt_all: bool = False
        self._halt_reason: str | None = None
        self._scale_factor: float = 1.0

        # Wall-clock gating for expensive work.
        self._last_daily_date: date | None = None
        self._last_corr_date: date | None = None

        # EWMA-vol state (computed from daily portfolio NAV, not ticks).
        self._ewma_var: float | None = None
        self._last_daily_nav: float | None = None

        # Regime inputs.
        self._vix_level: float | None = None
        self._atr_percentiles: dict[str, float] = {}

        # Component scales (for logging / monitoring).
        self._dd_scale: float = 1.0
        self._vol_scale: float = 1.0
        self._regime_scale: float = 1.0

        # P0-9 live-drift de-risk: scale_factor floor when realised MaxDD
        # breaches the MC-predicted p99 band (hysteresis via _drift_triggered).
        self._drift_scale: float = 1.0
        self._drift_triggered: bool = False
        self._dd_band: tuple[float, float] | None = None  # (p95, p99), non-positive
        self._last_drift_date: date | None = None

        # Halt persistence -- read disk state on construction so crash+restart
        # cannot silently un-halt.
        self._load_halt_state()

        # ── V3.8 envelope state (per PRM Integration V3.8 directive) ──────
        # Per-strategy config (default: disabled, V3.7 compatibility).
        self._v38_configs: dict[str, StrategyV38Config] = {}
        # Live strategy instances polled for open positions each bar.
        # Strategies that don't implement the V38PositionVisible Protocol
        # return empty lists via the helper functions.
        self._v38_strategy_instances: dict[str, object] = {}
        # Per-strategy counters for telemetry. Keys: would_have_rejected,
        # actual_rejected, last_rejection_reason.
        self._v38_counters: dict[str, dict] = {}
        # Portfolio NAV history for the rolling-60-bar DD throttle. Indexed
        # by on_bar timestamp.
        self._v38_nav_history: pd.Series = pd.Series(dtype=float)
        # Current DD throttle state (single portfolio-level state).
        self._v38_dd_throttle: DdThrottleState = initial_throttle_state()
        # Latest regime evaluation; None until first on_bar.
        self._v38_regime: RegimeResult | None = None

    # ── Public API ─────────────────────────────────────────────────────────

    def load_config(self, config: dict) -> None:
        self._config = {**_DEFAULT_CONFIG, **config}
        logger.info(
            "[PortfolioRM] Config loaded: max_dd=%.1f%% heat=%.1f%% max_single=%.1f%%",
            self._config["portfolio_max_dd_pct"],
            self._config["portfolio_heat_scale_pct"],
            self._config["portfolio_max_single_pct"],
        )

    def register_strategy(self, strategy_id: str, initial_equity: float) -> None:
        self._strategies[strategy_id] = _StrategyState(
            strategy_id=strategy_id,
            initial_equity=float(initial_equity),
            current_equity=float(initial_equity),
            equity_hwm=float(initial_equity),
        )
        logger.info(
            "[PortfolioRM] Registered '%s' initial_equity=%.2f total=%d",
            strategy_id,
            initial_equity,
            len(self._strategies),
        )

    def update(
        self,
        strategy_id: str,
        current_equity: float,
        ts: pd.Timestamp | datetime | None = None,
    ) -> None:
        """Update a strategy's equity snapshot with explicit timestamp.

        ``ts`` must be UTC. If omitted, wall-clock ``datetime.now(UTC)`` is
        used -- pass an explicit bar timestamp where available.
        """
        if self._halt_all:
            return

        if strategy_id not in self._strategies:
            logger.warning(
                "[PortfolioRM] Unknown strategy '%s' -- register_strategy() first.",
                strategy_id,
            )
            return

        if ts is None:
            ts_utc = pd.Timestamp.now(tz="UTC")
        elif isinstance(ts, (int, float)):
            # Accept nanosecond epoch (NautilusTrader ``bar.ts_event``).
            ts_utc = pd.Timestamp(int(ts), unit="ns", tz="UTC")
        else:
            ts_utc = pd.Timestamp(ts)
            if ts_utc.tzinfo is None:
                ts_utc = ts_utc.tz_localize("UTC")
            else:
                ts_utc = ts_utc.tz_convert("UTC")

        state = self._strategies[strategy_id]
        state.current_equity = float(current_equity)
        state.append(ts_utc, float(current_equity), int(self._config["history_max_days"]))

        if current_equity > state.equity_hwm:
            state.equity_hwm = float(current_equity)

        self._check_portfolio_health(ts_utc)

    def update_vix(self, vix_level: float) -> None:
        self._vix_level = float(vix_level)

    def update_atr_percentile(self, strategy_id: str, atr_pct: float) -> None:
        self._atr_percentiles[strategy_id] = float(atr_pct)

    @property
    def halt_all(self) -> bool:
        return self._halt_all

    @property
    def scale_factor(self) -> float:
        return self._scale_factor

    def get_equity_histories(self) -> dict[str, pd.Series]:
        """Public accessor -- returns each strategy's daily business-day equity.

        Used by ``PortfolioAllocator`` instead of reaching into the private
        ``_strategies`` dict.
        """
        max_days = int(self._config["history_max_days"])
        return {sid: st.daily_equity(max_days) for sid, st in self._strategies.items()}

    def get_summary(self) -> dict:
        total = self._total_equity()
        ann_vol = self._annualized_vol()
        risk_contribs = self._risk_contributions()
        return {
            "total_equity": round(total, 2),
            "portfolio_drawdown_pct": round(self._portfolio_drawdown() * 100, 3),
            "halt_all": self._halt_all,
            "halt_reason": self._halt_reason,
            "scale_factor": round(self._scale_factor, 3),
            "dd_scale": round(self._dd_scale, 3),
            "vol_scale": round(self._vol_scale, 3),
            "regime_scale": round(self._regime_scale, 3),
            "v38_dd_throttle": round(self._effective_v38_dd_throttle_mult(), 3),
            "realized_vol_ann_pct": round(ann_vol * 100, 2) if ann_vol else None,
            "vix_level": self._vix_level,
            "strategy_count": len(self._strategies),
            # Fraction of total portfolio capital held by strategies whose
            # risk contribution was NOT measured (immature, <20d history).
            # 0.0 means every strategy contributes to the rc decomposition.
            "risk_unmeasured_capital_pct": round(risk_contribs.get("__unmeasured__", 0.0) * 100, 2),
            "strategies": {
                sid: {
                    "equity": round(s.current_equity, 2),
                    "drawdown_pct": round(s.drawdown_pct * 100, 3),
                    "weight_pct": round(s.current_equity / total * 100 if total > 0 else 0.0, 2),
                    # V3.7 (L67): risk contribution = fractional share of
                    # portfolio variance contributed by this strategy.
                    # ERC objective: all equal at 1/N. Skewed numbers indicate
                    # risk concentration that capital weights don't reveal.
                    "risk_contrib_pct": round(risk_contribs.get(sid, 0.0) * 100, 2),
                    "atr_pct": round(self._atr_percentiles.get(sid, 50.0), 1),
                    "samples": len(s.samples),
                }
                for sid, s in self._strategies.items()
            },
        }

    def _risk_contributions(self) -> dict[str, float]:
        """V3.7 risk-contribution decomposition per strategy.

        Computes each strategy's fractional contribution to portfolio
        variance using empirical covariance of recent strategy returns
        (60 business days, aligned on common index). Returns a mapping
        ``strategy_id -> risk_contribution`` with values summing to ~1.0
        when valid. Returns empty dict if insufficient history.

        Reference: Maillard, Roncalli & Teiletche 2010, "Properties of
        Equally Weighted Risk Contribution Portfolios."
        """
        try:
            histories = self.get_equity_histories()
        except Exception:  # noqa: BLE001
            return {}
        names: list[str] = []
        rets: list[pd.Series] = []
        for sid, eq in histories.items():
            if len(eq) < 20:
                continue
            r = eq.pct_change().dropna()
            if len(r) < 20:
                continue
            names.append(sid)
            rets.append(r.iloc[-60:])
        if len(rets) < 2:
            return {}
        df = pd.concat(rets, axis=1, keys=names).fillna(0.0)
        # P1-27: Ledoit-Wolf shrunk covariance (robust at small N / short
        # window), with a crisis-correlation overlay when the regime is crisis
        # so the risk decomposition doesn't assume calm-period diversification.
        crisis = self._v38_regime is not None and self._v38_regime.is_crisis
        cov = shrink_covariance(df, crisis_rho=DEFAULT_CRISIS_RHO if crisis else None).cov
        total = self._total_equity()
        if total <= 0:
            return {}
        weights = np.array([self._strategies[n].current_equity / total for n in names])
        mature_capital_frac = float(weights.sum())
        portfolio_var = float(weights @ cov @ weights)
        if portfolio_var <= 1e-18:
            # Degenerate covariance -- equal-share across mature, no signal
            # about the unmeasured slice.
            per = mature_capital_frac / len(names)
            out: dict[str, float] = {n: per for n in names}
        else:
            marginal = cov @ weights
            # Standard Euler decomposition gives rc summing to 1.0 across the
            # mature subset. Multiplying by mature_capital_frac rescales so
            # that the returned values represent each strategy's share of
            # *total portfolio* risk (under the implicit assumption that the
            # excluded immature strategies contribute proportional risk).
            rc = weights * marginal / portfolio_var * mature_capital_frac
            out = {names[i]: float(rc[i]) for i in range(len(names))}

        # Make the unmeasured slice explicit so a viewer cannot read 100%
        # accounted-for when immature strategies were dropped.
        unmeasured = 1.0 - mature_capital_frac
        if unmeasured > 1e-9:
            out["__unmeasured__"] = unmeasured
        return out

    def check_correlation_regime(self) -> None:
        """Compute rolling correlation on a shared business-day grid.

        Uses ``get_equity_histories`` (each strategy's daily last-equity),
        converts to pct-change returns, aligns on the common date index
        (outer-join + fill with zero-return), and correlates. Logs a warning
        when any pair exceeds ``correlation_halt_threshold``.
        """
        threshold = float(self._config["correlation_halt_threshold"])
        window = int(self._config["correlation_window_days"])

        series_map = self.get_equity_histories()
        ret_map: dict[str, pd.Series] = {}
        for sid, eq in series_map.items():
            if len(eq) < 20:
                continue
            r = eq.pct_change().dropna()
            if len(r) < 10:
                continue
            ret_map[sid] = r.iloc[-window:]

        if len(ret_map) < 2:
            return

        df = pd.DataFrame(ret_map)  # auto-aligns on timestamped index
        df = df.fillna(0.0)  # non-trade days contribute zero return
        if len(df) < 20:
            return

        corr = df.corr()
        labels = list(ret_map.keys())
        for i, a in enumerate(labels):
            for b in labels[i + 1 :]:
                if a not in corr.index or b not in corr.columns:
                    continue
                r = float(corr.loc[a, b])
                if abs(r) > threshold:
                    logger.warning(
                        "[PortfolioRM] Correlation alert: '%s' <-> '%s' r=%.3f "
                        "(threshold %.2f). Consider reducing combined allocation.",
                        a,
                        b,
                        r,
                        threshold,
                    )

    def trip_halt(self, reason: str, operator: str = "unknown") -> bool:
        """Externally trip the kill switch (audit P0-10): halt_all + scale 0,
        persisted so a restart stays halted. Used by the reconciliation
        watchdog on a CONFIRMED broker-vs-cache orphan. Idempotent -- if already
        halted the original reason is kept; returns True iff this call tripped
        it. Operator clears via ``reset_halt``.
        """
        if self._halt_all:
            return False
        self._halt_all = True
        self._halt_reason = reason
        self._scale_factor = 0.0
        self._dd_scale = 0.0
        self._persist_halt_state(operator=operator, cleared=False)
        logger.critical(
            "[PortfolioRM] KILL SWITCH TRIPPED externally by %s -- %s", operator, reason
        )
        return True

    def reset_halt(self, operator: str = "unknown") -> None:
        """Manual operator action: clear the halt flag.

        The previous implementation re-anchored ``portfolio_hwm`` to the
        current (drawn-down) total equity, which silently reduced the absolute
        kill-switch distance. The new behaviour keeps the original HWM so a
        second drawdown of the same magnitude still trips the switch from the
        pre-halt peak. Operators who want to re-baseline must explicitly call
        ``reset_hwm`` afterwards.
        """
        old_hwm = self._portfolio_hwm
        self._halt_all = False
        self._halt_reason = None
        self._scale_factor = 1.0
        self._dd_scale = 1.0
        self._vol_scale = 1.0
        self._regime_scale = 1.0
        self._ewma_var = None
        self._last_daily_nav = None
        self._persist_halt_state(operator=operator, cleared=True)
        logger.warning(
            "[PortfolioRM] HALT RESET by operator=%s. HWM preserved at %.2f.",
            operator,
            old_hwm or 0.0,
        )

    def reset_hwm(self, operator: str = "unknown") -> None:
        """Re-anchor the portfolio high-water mark to current total equity."""
        new_hwm = self._total_equity()
        logger.warning(
            "[PortfolioRM] HWM RE-ANCHORED by operator=%s. old=%.2f new=%.2f",
            operator,
            self._portfolio_hwm or 0.0,
            new_hwm,
        )
        self._portfolio_hwm = new_hwm

    # ── V3.8 envelope API (PRM Integration V3.8 directive §2.1) ────────────

    def set_strategy_v38_config(self, strategy_id: str, config: StrategyV38Config) -> None:
        """Set the per-strategy V3.8 envelope config (per integration §3.1).

        Runtime calls this at startup (after register_strategy) and on
        SIGHUP reload. Default (no call) leaves the strategy in
        V3.7-compatibility mode (envelope disabled).
        """
        self._v38_configs[strategy_id] = config
        self._v38_counters.setdefault(
            strategy_id,
            {
                "would_have_rejected": 0,
                "actual_rejected": 0,
                "last_rejection_reason": None,
                "last_heat_contribution": 0.0,
                "last_leverage_contribution": 0.0,
            },
        )
        logger.info(
            "[PortfolioRM] V3.8 config set for '%s': enabled=%s mode=%s",
            strategy_id,
            config.enabled,
            config.mode,
        )

    def set_strategy_instance(self, strategy_id: str, instance: object) -> None:
        """Register a strategy instance so the PRM can poll positions per bar.

        Strategies that implement the V38PositionVisible Protocol expose
        get_open_position_heat / get_open_position_snapshots; others are
        polled trivially via the helper functions which return [] when
        the methods are absent.
        """
        self._v38_strategy_instances[strategy_id] = instance

    def _v38_config_for(self, strategy_id: str) -> StrategyV38Config:
        return self._v38_configs.get(strategy_id, StrategyV38Config())

    def _aggregate_open_positions(
        self,
        exclude_strategy: str | None = None,
    ) -> tuple[list[PositionHeat], list[PositionSnapshot]]:
        """Poll every registered strategy instance for its open positions."""
        heat: list[PositionHeat] = []
        snaps: list[PositionSnapshot] = []
        for sid, inst in self._v38_strategy_instances.items():
            if sid == exclude_strategy:
                continue
            heat.extend(get_heat_from(inst))
            snaps.extend(get_snapshots_from(inst))
        return heat, snaps

    def check_pre_trade(
        self,
        strategy_id: str,
        candidate_heat: PositionHeat,
        candidate_snapshot: PositionSnapshot,
        *,
        now: pd.Timestamp | None = None,  # noqa: ARG002 -- reserved for future telemetry timing
    ) -> PreTradeDecision:
        """Evaluate a candidate trade against the V3.8 envelope.

        Returns PreTradeDecision(accepted=True) unconditionally when the
        strategy's V3.8 envelope is disabled (V3.7-compatibility default).
        When enabled, evaluates heat + leverage + SPAN controls and
        returns accepted=False with a typed reason on breach. In shadow
        mode the rejection is logged (would_have_rejected counter
        increments) but accepted=True is returned with enforced=False.
        """
        if _v38_globally_disabled():
            return PreTradeDecision(accepted=True, enforced=False)
        config = self._v38_config_for(strategy_id)
        if not config.enabled:
            return PreTradeDecision(accepted=True, enforced=False)

        # P0-5: aggregate EVERY strategy's open positions (including the
        # submitting one) so the heat/leverage envelope sees the book's full
        # existing exposure, then add the candidate on top. The week-1 stub
        # excluded the submitting strategy ("candidate replaces its
        # contribution"), which under-counted heat/leverage for a strategy
        # already holding positions. Once a strategy implements the
        # V38PositionVisible protocol its real positions count; one that does
        # not still reports [] (V3.7-compatible). For a reconciling
        # (target-replacing) candidate this conservatively over-counts the same
        # leg -- safe for the shadow telemetry.
        current_heat, current_snaps = self._aggregate_open_positions(
            exclude_strategy=None,
        )
        regime_normal = self._v38_regime is None or not self._v38_regime.is_crisis
        heat_result = evaluate_heat_envelope(
            current_heat,
            self._total_equity(),
            candidate=candidate_heat,
            regime_normal=regime_normal,
        )
        lev_result = evaluate_leverage_envelope(
            current_snaps,
            self._total_equity(),
            candidate=candidate_snapshot,
            regime_normal=regime_normal,
        )

        # Determine rejection (priority: per-trade R -> heat -> leverage -> SPAN).
        reason: PreTradeRejectionReason | None = None
        diagnostic: dict[str, float | str] = {}
        # P0-4: per-trade R cap -- a single candidate risking > r_cap of total
        # equity at its stop is rejected before the portfolio-level checks.
        equity_now = self._total_equity()
        r_cap = float(self._config["per_trade_r_cap_pct"]) / 100.0
        per_trade_r = (candidate_heat.risk_amount / equity_now) if equity_now > 0 else 0.0
        if per_trade_r > r_cap:
            reason = PreTradeRejectionReason.PER_TRADE_R_CAP_HIT
            diagnostic = {
                "per_trade_r_pct": per_trade_r * 100.0,
                "r_cap_pct": r_cap * 100.0,
                "risk_amount": candidate_heat.risk_amount,
            }
        elif not heat_result.heat_pass:
            reason = (
                PreTradeRejectionReason.CRISIS_REGIME_HEAT_REDUCED
                if not regime_normal
                else PreTradeRejectionReason.HEAT_CAP_HIT
            )
            diagnostic = {
                "current_heat_pct": heat_result.portfolio_heat * 100.0,
                "heat_cap_pct": heat_result.heat_cap * 100.0,
                "regime": "crisis" if not regime_normal else "normal",
            }
        elif not lev_result.leverage_pass:
            reason = PreTradeRejectionReason.LEVERAGE_CAP_HIT
            diagnostic = {
                "current_leverage": lev_result.gross_leverage,
                "leverage_cap": lev_result.leverage_cap,
                "regime": "crisis" if not regime_normal else "normal",
            }
        elif not lev_result.span_buffer_pass:
            reason = PreTradeRejectionReason.SPAN_BUFFER_LOW
            diagnostic = {
                "min_buffer_ratio": lev_result.span_buffer_min_ratio,
                "min_required": lev_result.span_buffer_min_required,
                "breaching": ",".join(lev_result.span_breaching_symbols),
            }

        counters = self._v38_counters.setdefault(
            strategy_id,
            {
                "would_have_rejected": 0,
                "actual_rejected": 0,
                "last_rejection_reason": None,
                "last_heat_contribution": 0.0,
                "last_leverage_contribution": 0.0,
            },
        )
        counters["last_heat_contribution"] = candidate_heat.risk_amount
        counters["last_leverage_contribution"] = candidate_snapshot.abs_notional

        if reason is None:
            return PreTradeDecision(
                accepted=True,
                enforced=(config.mode == "live"),
            )

        # Rejection path. Shadow vs live behaviour differ only on
        # `accepted` and the counter that increments.
        if config.mode == "live":
            counters["actual_rejected"] += 1
            counters["last_rejection_reason"] = reason
            logger.warning(
                "[PortfolioRM] V3.8 REJECT strategy=%s reason=%s diag=%s",
                strategy_id,
                reason.value,
                diagnostic,
            )
            return PreTradeDecision(
                accepted=False, reason=reason, diagnostic=diagnostic, enforced=True
            )
        # Shadow mode.
        counters["would_have_rejected"] += 1
        counters["last_rejection_reason"] = reason
        logger.info(
            "[PortfolioRM] V3.8 shadow would_have_rejected strategy=%s reason=%s diag=%s",
            strategy_id,
            reason.value,
            diagnostic,
        )
        return PreTradeDecision(accepted=True, reason=reason, diagnostic=diagnostic, enforced=False)

    def on_bar(
        self,
        *,
        now: pd.Timestamp,
        vix_value: float | None = None,
        regime_vol_percentile: float | None = None,
    ) -> OnBarDecisions:
        """Per-bar V3.8 envelope evaluation: regime + DD throttle + flatten.

        Called once per bar by the live runtime BEFORE any strategy's
        on_bar. Strategies read results via `prm.current_regime()` /
        `prm.current_dd_throttle()`. vix_value falls back to the value
        previously set via `update_vix(...)` if not supplied.

        When `V38_ENVELOPE_DISABLE=1` is set, returns inert decisions
        (normal regime, no throttle, no flatten) without mutating any
        V3.8 state, so the live runtime keeps ticking but every gate
        is open. The V3.7 ruin gate (`halt_all`) is unaffected.
        """
        if _v38_globally_disabled():
            return OnBarDecisions(
                is_crisis_regime=False,
                regime_reasons=(),
                dd_throttle_multiplier=1.0,
                dd_throttle_triggered=False,
                portfolio_dd_from_60d_peak=0.0,
                flatten_recommended=(),
            )
        # Track portfolio NAV history for the rolling DD throttle.
        ts = pd.Timestamp(now)
        if ts.tzinfo is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
        total = self._total_equity()
        if total > 0:
            self._v38_nav_history.loc[ts] = total
            # P0-4/P0-12: on_bar runs once per day (P0-1) so this grows ~252/yr.
            # The rolling-DD only needs the last DEFAULT_PEAK_WINDOW_BARS; cap to
            # a comfortable multiple so the series can't creep unbounded.
            if len(self._v38_nav_history) > _V38_NAV_HISTORY_CAP:
                self._v38_nav_history = self._v38_nav_history.iloc[-_V38_NAV_HISTORY_CAP:]

        # Compute rolling 60-bar DD.
        if len(self._v38_nav_history) >= 2:
            dd_series = compute_rolling_dd_from_peak(
                self._v38_nav_history,
                peak_window_bars=DEFAULT_PEAK_WINDOW_BARS,
            )
            current_dd = float(dd_series.iloc[-1])
        else:
            current_dd = 0.0

        # Update DD throttle state (hysteresis-aware).
        self._v38_dd_throttle = update_throttle(self._v38_dd_throttle, current_dd, timestamp=ts)

        # P0-9 live-drift de-risk: refresh the MC band from disk (written daily
        # by scripts/monitor_live_drift.py), compare the realised trailing-1y
        # MaxDD to it, and set the drift de-risk scale (folded into scale_factor
        # on the next health check). Runs at most once per UTC day.
        if self._last_drift_date != ts.date():
            self._last_drift_date = ts.date()
            self._refresh_dd_band_from_disk()
            if self._dd_band is not None and len(self._v38_nav_history) >= 2:
                realised = realised_rolling_maxdd(self._v38_nav_history, window_bars=252)
                self.update_drift_state(realised)

        # Regime: VIX-or-vol-percentile OR.
        vix_for_eval = vix_value if vix_value is not None else self._vix_level
        self._v38_regime = is_crisis_regime(
            vix_value=vix_for_eval,
            realised_vol_pct=regime_vol_percentile,
        )

        # Pre-emptive flatten recommendations (§4.6 C4, P0-6): below -12%
        # portfolio DD, flag a strategy iff ALL its open positions hitting
        # their stops would push the portfolio past -15% -- a per-position
        # stop-fill projection over the V38PositionVisible heat records (P0-5),
        # replacing the week-1 stub that flagged every strategy with any open
        # position regardless of its actual stop exposure.
        flatten: list[str] = []
        if current_dd <= -0.12:
            strategy_risk = {
                sid: sum(h.risk_amount for h in get_heat_from(inst))
                for sid, inst in self._v38_strategy_instances.items()
            }
            flatten = self._flatten_candidates(current_dd, self._total_equity(), strategy_risk)

        return OnBarDecisions(
            is_crisis_regime=bool(self._v38_regime.is_crisis),
            regime_reasons=tuple(self._v38_regime.reasons),
            dd_throttle_multiplier=self._v38_dd_throttle.multiplier,
            dd_throttle_triggered=self._v38_dd_throttle.triggered,
            portfolio_dd_from_60d_peak=current_dd,
            flatten_recommended=tuple(flatten),
        )

    def current_regime(self) -> RegimeResult | None:
        """Latest regime evaluation from on_bar; None before first call."""
        return self._v38_regime

    def current_dd_throttle(self) -> DdThrottleState:
        """Latest DD throttle state."""
        return self._v38_dd_throttle

    def _effective_v38_dd_throttle_mult(self) -> float:
        """V3.8 DD-throttle multiplier as folded into ``scale_factor`` (P0-2).

        Returns 1.0 (inert) when the envelope is globally disabled. This guard
        matters because ``on_bar`` does NOT mutate ``_v38_dd_throttle`` while
        ``V38_ENVELOPE_DISABLE=1`` -- so a throttle that was engaged (0.5)
        before the operator pulled the switch would otherwise persist and keep
        throttling ``scale_factor`` after the envelope was supposed to be off.
        """
        if _v38_globally_disabled():
            return 1.0
        return float(self._v38_dd_throttle.multiplier)

    @staticmethod
    def _compose_scale_factor(
        dd_scale: float,
        vol_scale: float,
        regime_scale: float,
        v38_dd_throttle: float,
        drift_scale: float = 1.0,
    ) -> float:
        """Combine the throttle components into one scale factor by MIN.

        MIN, never a product (P0-2): the components all respond to portfolio
        drawdown / vol, so multiplying would de-risk the book several times for
        one event. MIN selects the single most-conservative throttle.
        ``drift_scale`` (P0-9) is the live-vs-predicted-MaxDD auto-de-risk floor
        (1.0 normal, 0.5 when realised MaxDD breaches the p99 band).
        """
        return min(dd_scale, vol_scale, regime_scale, v38_dd_throttle, drift_scale)

    def set_predicted_dd_band(self, p95: float, p99: float) -> None:
        """Set the MC-predicted MaxDD band (P0-9). Both non-positive; p99 is the
        deeper (more negative) tail. Fed by scripts/monitor_live_drift.py.
        """
        self._dd_band = (float(p95), float(p99))

    def _refresh_dd_band_from_disk(self) -> None:
        """Load the MaxDD band written by the drift monitor, if present. Absent
        / unreadable -> leave the current band (best-effort, never raises).
        """
        if not _DD_BAND_PATH.exists():
            return
        try:
            data = json.loads(_DD_BAND_PATH.read_text())
            p95, p99 = float(data["p95"]), float(data["p99"])
        except Exception:  # noqa: BLE001
            return
        self.set_predicted_dd_band(p95, p99)

    def update_drift_state(self, realised_maxdd: float) -> None:
        """Compare realised rolling MaxDD to the predicted band and set the
        drift de-risk scale (P0-9): alert at p95, auto-halve at p99, hold until
        recovered inside p95. No band set -> no-op.
        """
        if self._dd_band is None:
            return
        p95, p99 = self._dd_band
        decision = drift_band_decision(
            realised_maxdd, p95, p99, currently_derisked=self._drift_triggered
        )
        was_triggered = self._drift_triggered
        self._drift_scale = decision.drift_scale
        self._drift_triggered = decision.derisk
        if decision.alert and not was_triggered:
            logger.critical("[PortfolioRM] live-drift: %s", decision.reason)
            try:
                from titan.utils.notification import notify_health

                notify_health(
                    "Live drawdown drift vs MC band",
                    severity="critical" if decision.derisk else "warning",
                    detail=(
                        f"{decision.reason}. "
                        + (
                            f"scale_factor auto-de-risked to <= {decision.drift_scale:.2f}."
                            if decision.derisk
                            else "No auto-de-risk (alert only)."
                        )
                    ),
                )
            except Exception:  # noqa: BLE001 -- alerting must never break the tick
                pass
        elif was_triggered and not decision.derisk:
            logger.info("[PortfolioRM] live-drift: recovered inside p95 band -- de-risk released.")

    @staticmethod
    def _flatten_candidates(
        current_dd: float,
        total_equity: float,
        strategy_risk: dict[str, float],
        *,
        trigger_dd: float = -0.12,
        breach_dd: float = -0.15,
    ) -> list[str]:
        """P0-6 pre-emptive flatten (§4.6 C4): per-position stop-fill projection.

        Once portfolio DD breaches ``trigger_dd`` (-12%), flag a strategy iff
        the loss from ALL its open positions hitting their stops would push the
        portfolio past ``breach_dd`` (-15%):

            projected_dd = current_dd - (sum of the strategy's stop losses) / equity

        ``strategy_risk`` maps strategy_id -> the summed ``risk_amount`` (nominal
        loss at stop) of its open positions, from the V38PositionVisible
        protocol (P0-5). Replaces the week-1 stub that flagged EVERY strategy
        with any open position once DD < -12%, regardless of its actual
        stop-loss exposure. Returns [] above the trigger.
        """
        if current_dd > trigger_dd or total_equity <= 0:
            return []
        out: list[str] = []
        for sid, risk in strategy_risk.items():
            if risk <= 0:
                continue  # flat strategy -- no stop-fills to project, nothing to flatten
            projected_dd = current_dd - (risk / total_equity)
            if projected_dd <= breach_dd:
                out.append(sid)
        return out

    def get_v38_telemetry(self) -> V38Telemetry:
        """Snapshot of V3.8 envelope state for daily_summary + watchdogs."""
        per_strategy: list[StrategyV38Telemetry] = []
        for sid, config in self._v38_configs.items():
            counters = self._v38_counters.get(sid, {})
            per_strategy.append(
                StrategyV38Telemetry(
                    strategy_id=sid,
                    enabled=config.enabled,
                    mode=config.mode,
                    would_have_rejected_count=int(counters.get("would_have_rejected", 0)),
                    actual_rejected_count=int(counters.get("actual_rejected", 0)),
                    last_rejection_reason=counters.get("last_rejection_reason"),
                    current_heat_contribution=float(counters.get("last_heat_contribution", 0.0)),
                    current_leverage_contribution=float(
                        counters.get("last_leverage_contribution", 0.0)
                    ),
                )
            )

        all_heat, all_snaps = self._aggregate_open_positions()
        total_equity = self._total_equity()
        heat_result = evaluate_heat_envelope(all_heat, total_equity)
        lev_result = evaluate_leverage_envelope(all_snaps, total_equity)

        if len(self._v38_nav_history) >= 2:
            dd_series = compute_rolling_dd_from_peak(
                self._v38_nav_history,
                peak_window_bars=DEFAULT_PEAK_WINDOW_BARS,
            )
            current_dd = float(dd_series.iloc[-1])
        else:
            current_dd = 0.0

        return V38Telemetry(
            per_strategy=tuple(per_strategy),
            portfolio_heat=heat_result.portfolio_heat,
            gross_leverage=lev_result.gross_leverage,
            is_crisis_regime=bool(self._v38_regime.is_crisis) if self._v38_regime else False,
            dd_throttle_state=(
                self._v38_dd_throttle.multiplier,
                self._v38_dd_throttle.triggered,
            ),
            portfolio_dd_from_60d_peak=current_dd,
        )

    # ── Internal logic ─────────────────────────────────────────────────────

    def _total_equity(self) -> float:
        return sum(s.current_equity for s in self._strategies.values())

    def _portfolio_drawdown(self) -> float:
        total = self._total_equity()
        if self._portfolio_hwm is None:
            self._portfolio_hwm = total
            return 0.0
        if total > self._portfolio_hwm:
            self._portfolio_hwm = total
        if self._portfolio_hwm <= 0:
            return 0.0
        return (total - self._portfolio_hwm) / self._portfolio_hwm

    # ── Vol-targeting: daily NAV, EWMA variance ──────────────────────────

    def _recompute_daily_vol(self) -> None:
        """Recompute EWMA variance from the daily portfolio return series.

        Portfolio return is the previous-day capital-weighted average of
        per-strategy daily returns. A strategy that first reports today
        contributes 0 to today's portfolio return (its weight is 0 because
        it had no equity yesterday), so registering a new strategy never
        injects a spurious one-day return into the NAV series.
        """
        histories = self.get_equity_histories()
        if not histories:
            return

        df = pd.DataFrame(histories)
        if df.empty or len(df) < 11:
            return

        # Per-strategy daily returns. NaN on days a strategy had no prior
        # equity (its debut bar) -- intentionally kept NaN so the strategy
        # contributes zero, not a spurious 0->seed_equity jump.
        # fill_method=None is REQUIRED: the deprecated default would pad NaN
        # before differencing, which is exactly the capital-addition bug.
        per_strat_rets = df.pct_change(fill_method=None)

        # Previous-day capital weights. Strategies absent yesterday have
        # weight 0 (NaN treated as 0 via skipna=True in row-sum).
        eq_prev = df.shift(1)
        row_total = eq_prev.sum(axis=1, skipna=True)
        weights = eq_prev.div(row_total.replace(0.0, np.nan), axis=0)

        port_rets = (weights * per_strat_rets).fillna(0.0).sum(axis=1)
        port_rets = port_rets.iloc[1:]  # drop the leading no-prev-day row
        if len(port_rets) < 10:
            return

        # pandas ewm for consistency with PortfolioAllocator's EWMA convention.
        # adjust=False = infinite-history EWMA; mean of r² = EWMA variance estimator.
        lam = float(self._config["vol_ewma_lambda"])
        ewm_var_series = (port_rets**2).ewm(alpha=1.0 - lam, adjust=False).mean()
        self._ewma_var = float(ewm_var_series.iloc[-1])
        self._last_daily_nav = float(df.iloc[-1].sum(skipna=True))

    def _annualized_vol(self) -> float | None:
        if self._ewma_var is None or self._ewma_var <= 0:
            return None
        # Portfolio NAV is resampled to business-day before EWMA variance is
        # computed (see _recompute_daily_vol), so the series is daily and the
        # 252 factor is correct.
        per_day_std = self._ewma_var**0.5
        return annualize_vol(per_day_std, periods_per_year=BARS_PER_YEAR["D"])

    def _compute_vol_scale(self) -> float:
        ann_vol = self._annualized_vol()
        if ann_vol is None or ann_vol <= 0:
            return 1.0
        target = float(self._config["vol_target_ann_pct"]) / 100.0
        scale_min = float(self._config["vol_scale_min"])
        scale_max = float(self._config["vol_scale_max"])
        return max(scale_min, min(scale_max, target / ann_vol))

    # ── Regime helpers ────────────────────────────────────────────────────

    def _compute_vix_scale(self) -> float:
        if self._vix_level is None:
            return 1.0
        t1 = float(self._config["vix_tier_1"])
        t2 = float(self._config["vix_tier_2"])
        t3 = float(self._config["vix_tier_3"])
        if self._vix_level < t1:
            return 1.0
        if self._vix_level < t2:
            return 0.75
        if self._vix_level < t3:
            return 0.50
        return 0.25

    def _compute_atr_scale(self) -> float:
        if not self._atr_percentiles:
            return 1.0
        max_atr = max(self._atr_percentiles.values())
        low = float(self._config["atr_pct_low"])
        high = float(self._config["atr_pct_high"])
        extreme = float(self._config["atr_pct_extreme"])
        if max_atr < low:
            return 1.25
        if max_atr < high:
            return 1.0
        if max_atr < extreme:
            return 0.50
        return 0.25

    def _compute_regime_scale(self) -> float:
        return min(self._compute_vix_scale(), self._compute_atr_scale())

    # ── Core health check ─────────────────────────────────────────────────

    def _check_portfolio_health(self, now_ts: pd.Timestamp) -> None:
        max_dd = float(self._config["portfolio_max_dd_pct"]) / 100.0
        heat = float(self._config["portfolio_heat_scale_pct"]) / 100.0
        max_single = float(self._config["portfolio_max_single_pct"]) / 100.0

        port_dd = self._portfolio_drawdown()
        total = self._total_equity()

        # Kill switch is always evaluated on every tick.
        if port_dd < -max_dd:
            self._halt_all = True
            self._halt_reason = f"portfolio_dd={port_dd * 100:.2f}% exceeds {max_dd * 100:.1f}%"
            self._scale_factor = 0.0
            self._dd_scale = 0.0
            self._persist_halt_state(operator="auto-kill", cleared=False)
            logger.critical(
                "[PortfolioRM] KILL SWITCH -- portfolio DD %.2f%% > %.1f%%.",
                port_dd * 100,
                max_dd * 100,
            )
            return

        # DD heat always evaluated on every tick.
        if port_dd < -heat:
            heat_fraction = abs(port_dd) / max_dd
            self._dd_scale = max(0.25, 1.0 - heat_fraction)
        else:
            self._dd_scale = 1.0

        # Wall-clock daily gate: recompute vol + regime scales once per date.
        today = now_ts.date()
        if self._last_daily_date != today:
            self._last_daily_date = today
            self._recompute_daily_vol()

        self._vol_scale = self._compute_vol_scale()
        self._regime_scale = self._compute_regime_scale()
        # P0-2 (audit C2/H7/H8): fold the V3.8 rolling-60d DD throttle into the
        # single scale ladder by MIN -- never a product. dd_scale (all-time-HWM
        # heat) and the V3.8 dd_throttle (rolling-60d-peak step) both respond to
        # portfolio drawdown; multiplying them would de-risk the book twice. MIN
        # takes the single most-conservative throttle, consistent with how the
        # other components already compose.
        self._scale_factor = self._compose_scale_factor(
            self._dd_scale,
            self._vol_scale,
            self._regime_scale,
            self._effective_v38_dd_throttle_mult(),
            self._drift_scale,
        )

        # Correlation check once per date too (cheap, but verbose).
        if self._last_corr_date != today:
            self._last_corr_date = today
            try:
                self.check_correlation_regime()
            except Exception as e:
                logger.exception("[PortfolioRM] correlation check failed: %s", e)

        if self._scale_factor < 0.99:
            logger.warning(
                "[PortfolioRM] Scale %.0f%% (DD=%.0f%% Vol=%.0f%% Regime=%.0f%%) "
                "| port_DD=%.2f%% ann_vol=%s vix=%s",
                self._scale_factor * 100,
                self._dd_scale * 100,
                self._vol_scale * 100,
                self._regime_scale * 100,
                port_dd * 100,
                f"{self._annualized_vol() * 100:.1f}%" if self._annualized_vol() else "n/a",
                f"{self._vix_level:.1f}" if self._vix_level else "n/a",
            )

        # Concentration warning.
        if total > 0:
            for sid, state in self._strategies.items():
                weight = state.current_equity / total
                if weight > max_single:
                    logger.warning(
                        "[PortfolioRM] Concentration: '%s' holds %.1f%% (limit %.1f%%).",
                        sid,
                        weight * 100,
                        max_single * 100,
                    )

    # ── Halt persistence ──────────────────────────────────────────────────

    def _load_halt_state(self) -> None:
        if not _HALT_STATE_PATH.exists():
            return  # No file -> normal first run, not halted.
        try:
            data = json.loads(_HALT_STATE_PATH.read_text())
        except Exception as e:  # noqa: BLE001
            # P0-11 fail-safe: a halt file that exists but won't parse is
            # treated as HALTED, never as "proceed". A corrupt kill-switch
            # record must fail closed -- the operator clears it via reset_halt.
            self._halt_all = True
            self._halt_reason = "halt-state-unparseable"
            logger.critical(
                "[PortfolioRM] halt-state file present but UNPARSEABLE (%s) -- "
                "failing CLOSED (halted). Inspect %s and reset_halt() to resume.",
                e,
                _HALT_STATE_PATH,
            )
            return
        if data.get("halted"):
            self._halt_all = True
            self._halt_reason = data.get("reason", "persisted-halt")
            logger.critical(
                "[PortfolioRM] Loaded persisted HALT state -- "
                "reason=%s since=%s. Operator must call reset_halt() "
                "to resume.",
                self._halt_reason,
                data.get("at"),
            )

    def _persist_halt_state(self, operator: str, cleared: bool) -> None:
        try:
            _HALT_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "halted": self._halt_all,
                "reason": self._halt_reason,
                "operator": operator,
                "at": datetime.now(timezone.utc).isoformat(),
                "cleared": cleared,
            }
            # P0-11: atomic write -- serialise to a temp file then os.replace so
            # a crash mid-write can't leave a truncated (unparseable) halt file.
            tmp = _HALT_STATE_PATH.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload, indent=2))
            os.replace(tmp, _HALT_STATE_PATH)
        except Exception as e:
            logger.exception("[PortfolioRM] halt-state persist failed: %s", e)


# ── Module-level singleton ─────────────────────────────────────────────────────


def _load_portfolio_config() -> dict:
    import tomllib

    risk_toml = Path(__file__).resolve().parents[2] / "config" / "risk.toml"
    if not risk_toml.exists():
        return {}
    try:
        with open(risk_toml, "rb") as f:
            cfg = tomllib.load(f)
        return cfg.get("portfolio", {})
    except Exception:
        return {}


portfolio_risk_manager: PortfolioRiskManager = PortfolioRiskManager(config=_load_portfolio_config())
