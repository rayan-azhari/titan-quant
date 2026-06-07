"""Portfolio Allocator -- Inverse-Volatility Capital Allocation.

Computes per-strategy allocation weights using inverse-volatility weighting
on a **daily, timestamp-aligned** equity series. Rebalances by wall-clock
date (not by tick counter, which the previous implementation used and which
caused the "monthly" rebalance to fire every ~1 day when H1 strategies were
in the portfolio).

Reads equity histories via ``portfolio_risk_manager.get_equity_histories()``
(public accessor added in the April 2026 rewrite) instead of reaching into
the private ``_strategies`` dict.

Method
------
    sigma_i = sqrt(EWMA(lambda=0.94) var of daily returns of strategy i) * sqrt(252)
    w_i    = (1 / sigma_i) / SUM(1 / sigma_j)

Constraints (applied post-computation):
    - max_weight: 0.60 (no strategy > 60%)
    - min_weight: 0.05 (every strategy gets at least 5%)
    - correlation_penalty: when |r_ij| > 0.70 on the daily-aligned grid,
      reduce both weights by 10%
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pandas as pd

from titan.research.metrics import BARS_PER_YEAR, annualize_vol
from titan.risk.correlation_dial import correlation_dial_from_config

logger = logging.getLogger(__name__)

_DEFAULT_ALLOC_CONFIG: dict = {
    "rebalance_interval_days": 21,
    "ewma_lambda": 0.94,
    "min_weight": 0.05,
    "max_weight": 0.60,
    "min_history_days": 30,
    "correlation_penalty_threshold": 0.70,
}


class PortfolioAllocator:
    """Inverse-volatility capital allocator (timestamp-aware, wall-clock gated)."""

    def __init__(self, config: dict | None = None) -> None:
        self._config: dict = {**_DEFAULT_ALLOC_CONFIG, **(config or {})}
        self._weights: dict[str, float] = {}
        self._last_rebalance_date: date | None = None
        self._force_next: bool = False
        # Top-of-book leverage governor (correlation dial). Default DISABLED ->
        # leverage 1.0 -> get_weight() identical to inverse-vol behaviour until
        # the operator enables it in config/risk.toml [allocation.correlation_dial].
        self._dial = correlation_dial_from_config(self._config)
        self._leverage: float = 1.0

    def load_config(self, config: dict) -> None:
        self._config = {**_DEFAULT_ALLOC_CONFIG, **config}
        self._dial = correlation_dial_from_config(self._config)
        logger.info(
            "[Allocator] Config loaded: rebal=%dd min=%.0f%% max=%.0f%% corr_dial=%s",
            self._config["rebalance_interval_days"],
            self._config["min_weight"] * 100,
            self._config["max_weight"] * 100,
            "ON" if self._dial._cfg.enabled else "off",
        )

    def tick(self, now: date | None = None) -> None:
        """Wall-clock gated rebalance trigger.

        Safe to call from every bar of every strategy -- the actual
        rebalance only fires when the calendar distance from the last
        rebalance exceeds ``rebalance_interval_days`` business days.
        """
        today = now or date.today()
        interval = int(self._config["rebalance_interval_days"])

        due = (
            self._force_next
            or self._last_rebalance_date is None
            or ((today - self._last_rebalance_date).days >= interval)
        )
        if not due:
            return
        self._rebalance()
        # Refresh the top-of-book leverage governor at each rebalance (1.0 if disabled).
        prev = self._leverage
        self._leverage = float(self._dial.leverage_scalar(today))
        if abs(self._leverage - prev) > 1e-6 and self._dial._cfg.enabled:
            logger.info(
                "[Allocator] Correlation-dial leverage: %.2fx -> %.2fx", prev, self._leverage
            )
        self._last_rebalance_date = today
        self._force_next = False

    def get_weight(self, strategy_id: str) -> float:
        """Per-strategy sleeve allocation, scaled by the top-of-book leverage governor.

        ``leverage`` is 1.0 unless the correlation dial is enabled, so this is byte-identical
        to inverse-vol behaviour by default. When enabled it scales EVERY sleeve's deployed
        notional by the same factor (preserving the relative mix, changing only gross exposure).
        """
        base = (
            1.0
            if not self._weights
            else self._weights.get(strategy_id, 1.0 / max(1, len(self._weights)))
        )
        return base * self._leverage

    def get_all_weights(self) -> dict[str, float]:
        return {sid: w * self._leverage for sid, w in self._weights.items()}

    def get_leverage_scalar(self) -> float:
        """Current correlation-dial leverage scalar (1.0 when disabled). For monitoring."""
        return self._leverage

    def force_rebalance(self) -> None:
        self._force_next = True

    # ── Internal ──────────────────────────────────────────────────────────

    def _rebalance(self) -> None:
        from titan.risk.portfolio_risk_manager import portfolio_risk_manager

        histories = portfolio_risk_manager.get_equity_histories()
        if len(histories) < 2:
            for sid in histories:
                self._weights[sid] = 1.0
            return

        min_hist = int(self._config["min_history_days"])
        lam = float(self._config["ewma_lambda"])
        min_w = float(self._config["min_weight"])
        max_w = float(self._config["max_weight"])
        corr_threshold = float(self._config["correlation_penalty_threshold"])

        # Build aligned daily-return DataFrame once and reuse for vol + corr.
        rets_map: dict[str, pd.Series] = {}
        for sid, eq in histories.items():
            if len(eq) < min_hist:
                continue
            r = eq.pct_change().dropna()
            if len(r) < 10:
                continue
            rets_map[sid] = r

        if len(rets_map) < 2:
            return

        df = pd.DataFrame(rets_map)
        df = df.fillna(0.0)  # non-trade days contribute zero return

        vols: dict[str, float] = {}
        for sid in df.columns:
            ewma_var = df[sid].ewm(alpha=1.0 - lam, adjust=False).var().iloc[-1]
            per_day_std = max(0.0, float(ewma_var)) ** 0.5
            # Per-strategy equity histories are resampled to business-day in
            # PortfolioRiskManager.get_equity_histories, so factor is 252.
            ann_vol = annualize_vol(per_day_std, periods_per_year=BARS_PER_YEAR["D"])
            if ann_vol > 0:
                vols[sid] = ann_vol

        if not vols:
            return

        inv_vols = {sid: 1.0 / v for sid, v in vols.items()}
        total_inv = sum(inv_vols.values())
        raw_weights = {sid: iv / total_inv for sid, iv in inv_vols.items()}

        # Correlation penalty on the aligned grid.
        if len(df) >= 20:
            corr = df.corr()
            sids = list(df.columns)
            for i, a in enumerate(sids):
                for b in sids[i + 1 :]:
                    r = abs(float(corr.loc[a, b]))
                    if r > corr_threshold and a in raw_weights and b in raw_weights:
                        raw_weights[a] *= 0.90
                        raw_weights[b] *= 0.90
                        logger.info(
                            "[Allocator] Correlation penalty: %s <-> %s r=%.2f -- -10%% each",
                            a,
                            b,
                            r,
                        )

        # Two-pass allocation: reserve `min_w` for immature strategies first,
        # then allocate the residual budget among mature strategies subject to
        # [min_w, max_w]. The old single-pass code added `min_w` to every
        # missing strategy *after* the mature weights had already been
        # normalised to 1.0, then divided everyone by the new total -- which
        # dragged every strategy below `min_w` whenever missing > 0.
        all_sids = list(histories.keys())
        mature_sids = list(raw_weights.keys())
        missing = [sid for sid in all_sids if sid not in raw_weights]
        n_total = len(all_sids)

        # Feasibility: every strategy needs at least min_w.
        if n_total * min_w >= 1.0 - 1e-9:
            self._weights = {sid: 1.0 / n_total for sid in all_sids}
            logger.info(
                "[Allocator] Floor infeasible (n=%d, min_w=%.3f) -- equal-weighting.",
                n_total,
                min_w,
            )
            return

        budget_mature = 1.0 - len(missing) * min_w
        # Project raw inverse-vol weights onto the mature budget.
        total_raw = sum(max(0.0, w) for w in raw_weights.values())
        if total_raw <= 0:
            mature_w = {sid: budget_mature / len(raw_weights) for sid in raw_weights}
        else:
            mature_w = {
                sid: max(0.0, w) * budget_mature / total_raw for sid, w in raw_weights.items()
            }

        # Water-fill for [min_w, max_w] subject to sum == budget_mature.
        # Each iteration clamps violators to their bound and redistributes the
        # residual equally among the unsaturated. Converges in at most O(n)
        # iterations since each pass saturates at least one new strategy.
        for _ in range(len(mature_sids) + 2):
            capped = {sid: min(max_w, max(min_w, w)) for sid, w in mature_w.items()}
            diff = budget_mature - sum(capped.values())
            if abs(diff) < 1e-12:
                mature_w = capped
                break
            free = [sid for sid, w in capped.items() if min_w + 1e-12 < w < max_w - 1e-12]
            if not free:
                # Fully saturated -- accept whatever sum we ended at; the
                # caller has chosen bounds that are not simultaneously
                # feasible with budget_mature.
                mature_w = capped
                logger.warning(
                    "[Allocator] Bounds [%.3f, %.3f] not feasible with mature "
                    "budget %.3f -- accepting saturated weights (sum=%.3f).",
                    min_w,
                    max_w,
                    budget_mature,
                    sum(capped.values()),
                )
                break
            per_free = diff / len(free)
            mature_w = {
                sid: (capped[sid] + per_free if sid in free else capped[sid]) for sid in capped
            }
        else:  # noqa: PLW0120 -- iteration cap exhausted without break
            mature_w = capped

        self._weights = {sid: min_w for sid in missing}
        self._weights.update(mature_w)

        logger.info(
            "[Allocator] Rebalanced: %s",
            {sid: f"{w:.1%}" for sid, w in sorted(self._weights.items())},
        )


def _load_alloc_config() -> dict:
    import tomllib

    risk_toml = Path(__file__).resolve().parents[2] / "config" / "risk.toml"
    if not risk_toml.exists():
        return {}
    try:
        with open(risk_toml, "rb") as f:
            cfg = tomllib.load(f)
        return cfg.get("allocation", {})
    except Exception:
        return {}


portfolio_allocator: PortfolioAllocator = PortfolioAllocator(config=_load_alloc_config())
