"""Audit-grade wrapper: ``run_audit`` (Methodology Audit 2026-05-14 §2.10, L3;
External Quant Audit 2026-05-29 P4-2).

§2.10 specified a single ``run_audit`` entrypoint so every audit stops
re-implementing WFO + DSR + MC + sanctuary + decision scaffolding (gaps L1/L3),
and so there is ONE place to enforce pre-registration provenance (P4-1 / C6).
The audit found it was never built; this is it.

``run_audit`` takes a strategy (the :class:`AuditableStrategy` protocol), the
data, and the path to its Pre-Reg directive, then:

  1. Anchors the pre-reg: parses the ``prereg`` manifest, hashes the
     gate-defining fields, and records the git commit (``strict_prereg=True``
     refuses to run against an uncommitted/dirty pre-reg -- a run that cannot be
     anchored to a commit cannot later prove blind selection).
  2. Slices the sanctuary, builds class-standard WFO folds, runs the strategy
     per cell to stitch OOS returns.
  3. Computes the 5 decision axes honestly: bootstrap CI_lo on the canonical
     cell, DSR over the FULL sweep at the REGISTERED N (not survivors), block-MC
     P(MaxDD>X), sanctuary-divergence Sharpe, and the noise axis (defaulted
     CONSERVATIVELY -- see ``noise``).
  4. Runs the L65 ruin gate at the supplied :class:`RuinGate` (default
     ``GATE_V38``).
  5. Returns a :class:`RunAuditResult` carrying the verdict + every input + the
     :class:`PreRegReceipt`, ready to write into the directive's result log.

This wrapper deliberately does NOT execute strategies itself -- the per-fold and
per-MC-path strategy execution is injected via the protocol, so it stays
strategy-agnostic while standardising the gate pipeline.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import pandas as pd

from titan.research.framework.decision import DecisionInputs, DecisionResult, Verdict, decide
from titan.research.framework.dsr import DsrResult, deflated_sharpe, sr_var_from_sweep
from titan.research.framework.mc import McResult, run_block_mc
from titan.research.framework.prereg import (
    PreRegError,
    PreRegReceipt,
    build_receipt,
    parse_prereg_manifest,
)
from titan.research.framework.ruin import GATE_V38, RuinAssessment, RuinGate, assess_strategy_ruin
from titan.research.framework.sanctuary import (
    DivergenceTest,
    sanctuary_divergence_test,
    slice_sanctuary,
)
from titan.research.framework.typology import StrategyClass, defaults_for
from titan.research.framework.wfo import iter_folds
from titan.research.metrics import bootstrap_sharpe_ci, sharpe

logger = logging.getLogger(__name__)


@runtime_checkable
class AuditableStrategy(Protocol):
    """The contract a strategy implements to be audited by ``run_audit``.

    Both methods return per-bar SIMPLE returns (not log returns) so they feed
    the metrics + the geometric ruin engine consistently.
    """

    def fold_returns(self, is_df: pd.DataFrame, oos_df: pd.DataFrame, cell: Mapping) -> pd.Series:
        """OOS per-bar returns for one WFO fold under one parameter ``cell``.
        Train/calibrate on ``is_df``; emit returns indexed to ``oos_df``.
        """
        ...

    def mc_strategy_fn(self, cell: Mapping) -> Callable[[pd.DataFrame], pd.Series]:
        """A closure for ``run_block_mc``: maps a synthetic price DataFrame
        (with at least a 'close' column) to per-bar returns for this ``cell``.
        Bootstraps the UNDERLYING, not strategy returns (lesson A6).
        """
        ...


@dataclass(frozen=True)
class RunAuditResult:
    """Everything ``run_audit`` produces. Embed ``receipt`` + the axis values in
    the directive's result log.
    """

    verdict: Verdict
    decision: DecisionResult
    receipt: PreRegReceipt
    strategy_class: str
    canonical_cell: str
    canonical_sharpe: float
    ci_lo: float
    ci_hi: float
    dsr: DsrResult
    mc: McResult
    divergence: DivergenceTest
    ruin: RuinAssessment | None
    ruin_gate_name: str
    ruin_passes: bool | None
    cell_sharpes: dict[str, float]
    n_folds: int
    n_cells: int
    registered_n_trials: int
    sanctuary_start: pd.Timestamp
    sanctuary_end: pd.Timestamp
    noise_evaluated: bool
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def report(self) -> str:
        lines = [
            f"run_audit verdict: {self.verdict.value}  ({self.strategy_class}, "
            f"canonical={self.canonical_cell})",
            f"  pre-reg: {self.receipt.prereg_path} sha={self.receipt.prereg_sha256[:12]} "
            f"commit={self.receipt.prereg_commit or 'UNCOMMITTED'} clean={self.receipt.is_clean}",
            f"  CI_lo={self.ci_lo:+.3f}  Sharpe={self.canonical_sharpe:+.3f}  "
            f"DSR={self.dsr.dsr_prob:.3f} (N={self.dsr.n_trials}, e_max_SR={self.dsr.e_max_sr})",
            f"  MC P(MaxDD>{self.mc.threshold_pct:.0%})={self.mc.p_maxdd_gt_threshold:.3f} "
            f"(gate {self.mc.pass_threshold_prob:.2f})  "
            f"sanctuary_Sharpe={self.divergence.sanctuary_sharpe:+.3f}"
            f"{' LUCKY' if self.divergence.lucky_flag else ''}",
            f"  ruin[{self.ruin_gate_name}] passes={self.ruin_passes}",
            f"  noise_evaluated={self.noise_evaluated}  folds={self.n_folds}  "
            f"cells={self.n_cells} (registered N={self.registered_n_trials})",
            f"  rationale: {self.decision.rationale}",
        ]
        if self.warnings:
            lines.append("  WARNINGS: " + " | ".join(self.warnings))
        return "\n".join(lines)


def _stitch_cell_oos(
    strategy: AuditableStrategy,
    visible: pd.DataFrame,
    cell: Mapping,
    wfo_cfg,
    bars_per_year: float,
) -> pd.Series:
    """Run the strategy on every fold and concatenate the OOS returns."""
    parts: list[pd.Series] = []
    for _fold, is_df, oos_df in iter_folds(visible, wfo_cfg, bars_per_year=bars_per_year):
        r = strategy.fold_returns(is_df, oos_df, cell)
        if r is not None and len(r) > 0:
            parts.append(r)
    if not parts:
        return pd.Series(dtype=float)
    return pd.concat(parts)


def run_audit(
    strategy: AuditableStrategy,
    data: pd.DataFrame,
    *,
    pre_reg_path: str | Path,
    cells: Mapping[str, Mapping],
    periods_per_year: int,
    price_col: str = "close",
    strategy_class: StrategyClass | None = None,
    ruin_gate: RuinGate = GATE_V38,
    deployment_weight: float = 1.0,
    sanctuary_months: int | None = None,
    noise: tuple[bool, bool] | None = None,
    strict_prereg: bool = True,
    seed: int = 42,
) -> RunAuditResult:
    """Run a framework-grade audit, enforcing pre-reg provenance up front.

    Parameters
    ----------
    strategy: an :class:`AuditableStrategy`.
    data: time-indexed OHLCV DataFrame (must contain ``price_col``).
    pre_reg_path: the Pre-Reg directive carrying the ``prereg`` manifest block.
    cells: mapping of cell-name -> parameter dict for the sweep. Must include
        the manifest's ``canonical_cell``.
    periods_per_year: annualisation factor (also used as bars/year for folds).
    ruin_gate: L65 gate to apply (default ``GATE_V38``).
    sanctuary_months: override the class-default sanctuary window.
    noise: ``(passes_mean, passes_worst)`` from ``run_noise_robustness``. If
        ``None`` the noise axis defaults to ``(False, False)`` = "not evaluated
        -> worst", so an un-noise-tested strategy cannot reach DEPLOY (it caps at
        CONDITIONAL_WATCHPOINT). Pass real results to lift it.
    strict_prereg: if True (default), raise when the pre-reg directive has
        uncommitted changes (cannot be anchored to a commit).

    Raises:
    ------
    PreRegError: no manifest, dirty pre-reg under ``strict_prereg``, or the
        canonical cell is absent from ``cells``.
    ValueError: insufficient data to build any WFO fold.
    """
    pre_reg_path = Path(pre_reg_path)
    if price_col not in data.columns:
        raise ValueError(f"data missing price column {price_col!r}")

    audit_warnings: list[str] = []

    # ── 1. Anchor the pre-registration (P4-1 / C6) ──
    manifest = parse_prereg_manifest(pre_reg_path)
    receipt = build_receipt(pre_reg_path)
    if not receipt.is_clean:
        msg = (
            f"pre-reg {pre_reg_path} has uncommitted changes; commit the manifest "
            "before running so the audit can be anchored to a commit (P4-1)."
        )
        if strict_prereg:
            raise PreRegError(msg)
        audit_warnings.append(msg)

    canonical = manifest.canonical_cell
    if canonical not in cells:
        raise PreRegError(
            f"manifest canonical_cell {canonical!r} is not in the supplied cells "
            f"({sorted(cells)}). The audited sweep must contain the registered canonical cell."
        )
    if manifest.n_trials != len(cells):
        audit_warnings.append(
            f"registered N ({manifest.n_trials}) != actual sweep size ({len(cells)}); "
            "DSR uses the REGISTERED N (honest), but the mismatch is worth investigating."
        )

    # ── 2. Strategy class + sanctuary + folds ──
    if strategy_class is None:
        try:
            strategy_class = StrategyClass[manifest.strategy_class]
        except KeyError as e:
            raise PreRegError(f"unknown strategy_class {manifest.strategy_class!r}") from e
    d = defaults_for(strategy_class)
    months = (
        sanctuary_months if sanctuary_months is not None else getattr(d, "sanctuary_months", 12)
    )
    sanc = slice_sanctuary(data, months=months)
    folds = list(iter_folds(sanc.visible, d.wfo, bars_per_year=periods_per_year))
    if not folds:
        raise ValueError(
            f"insufficient visible data ({len(sanc.visible)} bars) to build any "
            f"{strategy_class.name} WFO fold."
        )

    # ── 3. Per-cell stitched OOS + Sharpe ──
    cell_oos: dict[str, pd.Series] = {}
    cell_sharpes: dict[str, float] = {}
    for name, params in cells.items():
        stitched = _stitch_cell_oos(strategy, sanc.visible, params, d.wfo, periods_per_year)
        cell_oos[name] = stitched
        cell_sharpes[name] = (
            float(sharpe(stitched, periods_per_year=periods_per_year)) if len(stitched) else 0.0
        )

    canonical_oos = cell_oos[canonical]
    if len(canonical_oos) < 30:
        raise ValueError(
            f"canonical cell produced too few OOS bars ({len(canonical_oos)}) to audit."
        )
    canonical_sharpe = cell_sharpes[canonical]
    # P1-7: serially-aware (stationary) bootstrap CI using the class block size,
    # so CI_lo is not optimistically narrow on autocorrelated returns.
    ci_lo, ci_hi = bootstrap_sharpe_ci(
        canonical_oos,
        periods_per_year=periods_per_year,
        seed=seed,
        block_size=d.mc.block_size_bars,
    )

    # ── 4. DSR over the FULL sweep at the REGISTERED N (not survivors) ──
    sr_var = sr_var_from_sweep(list(cell_sharpes.values()))
    dsr = deflated_sharpe(
        canonical_sharpe,
        sr_var_across_trials=sr_var,
        returns=canonical_oos,
        n_trials=manifest.n_trials,
    )

    # ── 5. Block-MC on the underlying (lesson A6) ──
    mc = run_block_mc(
        sanc.visible[price_col],
        d.mc,
        strategy.mc_strategy_fn(cells[canonical]),
        periods_per_year=periods_per_year,
        seed=seed,
    )

    # ── 6. Sanctuary one-shot + divergence ──
    sanctuary_oos = strategy.fold_returns(sanc.visible, sanc.sanctuary, cells[canonical])
    divergence = sanctuary_divergence_test(
        canonical_oos, sanctuary_oos, periods_per_year=periods_per_year
    )

    # ── 7. L65 ruin at the supplied gate ──
    ruin: RuinAssessment | None = None
    ruin_passes: bool | None = None
    try:
        ruin = assess_strategy_ruin(
            canonical_oos, deployment_weight=deployment_weight, gate=ruin_gate
        )
        ruin_passes = ruin.passes_gate(ruin_gate)
    except Exception as e:  # noqa: BLE001 - ruin is a soft input to the verdict; never fatal
        audit_warnings.append(f"ruin assessment skipped: {e}")

    # ── 8. 5-axis decision ──
    if noise is None:
        noise = (False, False)  # not evaluated -> conservative (caps at CONDITIONAL)
        noise_evaluated = False
    else:
        noise_evaluated = True
    decision = decide(
        DecisionInputs(
            ci_lo=ci_lo,
            dsr_prob=dsr.dsr_prob,
            p_maxdd_gt_threshold=mc.p_maxdd_gt_threshold,
            pass_threshold_prob=mc.pass_threshold_prob,
            sanctuary_sharpe=divergence.sanctuary_sharpe,
            noise_passes_mean=noise[0],
            noise_passes_worst=noise[1],
        )
    )

    return RunAuditResult(
        verdict=decision.verdict,
        decision=decision,
        receipt=receipt,
        strategy_class=strategy_class.name,
        canonical_cell=canonical,
        canonical_sharpe=canonical_sharpe,
        ci_lo=float(ci_lo),
        ci_hi=float(ci_hi),
        dsr=dsr,
        mc=mc,
        divergence=divergence,
        ruin=ruin,
        ruin_gate_name=ruin_gate.name,
        ruin_passes=ruin_passes,
        cell_sharpes=cell_sharpes,
        n_folds=len(folds),
        n_cells=len(cells),
        registered_n_trials=manifest.n_trials,
        sanctuary_start=sanc.sanctuary_start,
        sanctuary_end=sanc.sanctuary_end,
        noise_evaluated=noise_evaluated,
        warnings=tuple(audit_warnings),
    )


__all__ = ["AuditableStrategy", "RunAuditResult", "run_audit"]
