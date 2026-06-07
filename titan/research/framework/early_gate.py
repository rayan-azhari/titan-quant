"""Pass-1-gates-Pass-2 — skip MC when headline Sharpe can't clear CI_lo > 0.

Speed-up #1 from the V3.6 speed lever list. Pre-computes a cheap analytic
upper bound on the bootstrap CI_lo of a given headline Sharpe; when the
upper bound is <= 0, MC is wasted compute (the strategy is *obviously*
rejected by the V3.6 deployment gate regardless of the bootstrap details).

Motivating incident (2026-05-16, I1 HMM audit): all 13 cells produced
Sharpe in [-0.31, -0.25]. Pass 2 ran for ~50 minutes on C1 before being
killed; no possible bootstrap CI could lift a -0.28 headline to a positive
lower bound. The gate below would have rejected I1 in milliseconds.

Math
----
Lo (2002) — "The Statistics of Sharpe Ratios". For IID per-period returns
with annualised Sharpe ``SR`` over ``N`` bars per year, evaluated on a
sample of ``T`` bars, the variance of the annualised sample Sharpe is::

    Var(SR_hat) = (1 + 0.5 * SR^2) / T

For block-bootstrap CIs the *effective* sample size is reduced by serial
correlation; a standard conservative inflation factor is the block size
``B``. The Lo SE is therefore inflated::

    SE_inflated = sqrt(Var(SR_hat) * B)

A 95% CI lower bound is approximately ``SR - 1.96 * SE_inflated``. The
strategy "can plausibly clear CI_lo > 0" iff::

    SR > 1.96 * SE_inflated

Below that, Pass 2 (MC + bootstrap) is wasted compute — the gate would
reject regardless.

This is a *conservative* gate: it errs on the side of running Pass 2 when
in doubt. The actual CI from block bootstrap will be wider than the IID-
inflated estimate in adversarial cases, so a strategy that *just barely*
passes this analytic gate may still fail Pass 2. That's fine — the goal
is to skip the obvious losers, not to replace the bootstrap.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from titan.research.metrics import sharpe

# z-score for two-sided confidence levels. Looked up rather than imported
# from scipy to keep this module dep-light (the framework already pulls
# scipy, but the early-gate decision should be cheap).
_Z_TABLE: dict[float, float] = {
    0.90: 1.6449,
    0.95: 1.9600,
    0.99: 2.5758,
}


@dataclass(frozen=True)
class Pass1GateResult:
    """Outcome of the Pass-1-can-clear-CI-gate check.

    Attributes:
        headline_sharpe:
            The Pass-1 OOS Sharpe.
        n_oos_bars:
            Number of OOS bars used (sample size for the SE estimate).
        block_size:
            Block-bootstrap block size used to inflate the IID SE
            (conservative effective-sample-size adjustment).
        confidence:
            Two-sided confidence level for the CI (default 0.95).
        se_inflated:
            Lo-2002 SE inflated by sqrt(block_size).
        approx_ci_lo:
            ``headline_sharpe - z * se_inflated``.
        can_clear:
            ``True`` iff ``approx_ci_lo > 0``. If ``False``, Pass 2 is
            recommended to be SKIPPED.
        reason:
            One-line human-readable summary, for logging.
    """

    headline_sharpe: float
    n_oos_bars: int
    block_size: int
    confidence: float
    se_inflated: float
    approx_ci_lo: float
    can_clear: bool
    reason: str


def pass1_can_clear_ci_gate(
    headline_sharpe: float,
    *,
    n_oos_bars: int,
    block_size: int = 1,
    confidence: float = 0.95,
) -> Pass1GateResult:
    """Cheap analytic check: can the Pass-1 Sharpe clear CI_lo > 0 under MC?

    Parameters:
        headline_sharpe:
            Pass-1 stitched-OOS annualised Sharpe.
        n_oos_bars:
            Total stitched-OOS bar count across all folds. Sample size
            for the Lo (2002) SE.
        block_size:
            Block-bootstrap block size in bars. Inflates the IID SE by
            ``sqrt(block_size)`` (conservative effective-sample-size
            adjustment). Pass ``1`` for IID-equivalent baseline.
        confidence:
            Two-sided confidence level (default 0.95). Must be one of
            0.90 / 0.95 / 0.99.

    Returns:
        ``Pass1GateResult`` with the verdict and the math.
    """
    if confidence not in _Z_TABLE:
        raise ValueError(f"confidence must be one of {sorted(_Z_TABLE)}; got {confidence}")
    if n_oos_bars <= 0:
        raise ValueError(f"n_oos_bars must be > 0; got {n_oos_bars}")
    if block_size <= 0:
        raise ValueError(f"block_size must be > 0; got {block_size}")
    z = _Z_TABLE[confidence]
    if headline_sharpe <= 0:
        # Negative headline can never clear CI_lo > 0 under any MC.
        se = math.sqrt((1.0 + 0.5 * headline_sharpe**2) / n_oos_bars * block_size)
        return Pass1GateResult(
            headline_sharpe=float(headline_sharpe),
            n_oos_bars=int(n_oos_bars),
            block_size=int(block_size),
            confidence=float(confidence),
            se_inflated=float(se),
            approx_ci_lo=float(headline_sharpe - z * se),
            can_clear=False,
            reason=(
                f"headline_sharpe={headline_sharpe:.4f} <= 0 — "
                f"CI_lo > 0 is impossible; skip Pass 2."
            ),
        )
    se = math.sqrt((1.0 + 0.5 * headline_sharpe**2) / n_oos_bars * block_size)
    approx_ci_lo = headline_sharpe - z * se
    can_clear = approx_ci_lo > 0
    if can_clear:
        reason = (
            f"headline_sharpe={headline_sharpe:.4f} > {z:.2f} * SE={se:.4f} "
            f"=> approx CI_lo={approx_ci_lo:.4f} > 0; run Pass 2."
        )
    else:
        reason = (
            f"headline_sharpe={headline_sharpe:.4f} <= {z:.2f} * SE={se:.4f} "
            f"=> approx CI_lo={approx_ci_lo:.4f} <= 0; skip Pass 2 "
            f"(would not clear the V3.6 deployment gate)."
        )
    return Pass1GateResult(
        headline_sharpe=float(headline_sharpe),
        n_oos_bars=int(n_oos_bars),
        block_size=int(block_size),
        confidence=float(confidence),
        se_inflated=float(se),
        approx_ci_lo=float(approx_ci_lo),
        can_clear=bool(can_clear),
        reason=reason,
    )


def pass1_can_clear_from_returns(
    oos_returns: pd.Series,
    *,
    periods_per_year: int,
    block_size: int = 1,
    confidence: float = 0.95,
) -> Pass1GateResult:
    """Same gate, computed directly from the OOS returns series.

    Convenience wrapper that pulls ``headline_sharpe`` and ``n_oos_bars``
    from the series. Use this in the harness right after the stitched-OOS
    returns are assembled, before invoking ``run_block_mc()``.
    """
    clean = oos_returns.dropna()
    if clean.shape[0] == 0:
        return Pass1GateResult(
            headline_sharpe=0.0,
            n_oos_bars=0,
            block_size=int(block_size),
            confidence=float(confidence),
            se_inflated=float("inf"),
            approx_ci_lo=0.0,
            can_clear=False,
            reason="OOS returns are empty/all-NaN; nothing to gate.",
        )
    sr = float(sharpe(clean, periods_per_year=periods_per_year))
    return pass1_can_clear_ci_gate(
        sr,
        n_oos_bars=int(clean.shape[0]),
        block_size=int(block_size),
        confidence=float(confidence),
    )


def pass1_can_clear_any_cell(
    cell_results: dict[str, pd.Series],
    *,
    periods_per_year: int,
    block_size: int = 1,
    confidence: float = 0.95,
) -> tuple[bool, dict[str, Pass1GateResult]]:
    """Sweep-level gate: does ANY cell's Pass-1 Sharpe clear the CI gate?

    Returns ``(any_can_clear, per_cell_results)``. When ``any_can_clear``
    is False, the entire Pass 2 (all cells × MC) can be skipped — no cell
    is promotion-eligible regardless of MC outcome.

    This is the I1-style early termination: 13 negative cells × 200 MC
    paths × 31 HMM-refits each = ~50 minutes of compute that yields a
    foregone-conclusion rejection. Skip it.
    """
    per_cell = {
        name: pass1_can_clear_from_returns(
            rets,
            periods_per_year=periods_per_year,
            block_size=block_size,
            confidence=confidence,
        )
        for name, rets in cell_results.items()
    }
    return (any(r.can_clear for r in per_cell.values()), per_cell)


def format_gate_report(per_cell: dict[str, Pass1GateResult], *, audit_label: str = "") -> str:
    """Markdown summary suitable for paste into the audit result log."""
    label = f" — {audit_label}" if audit_label else ""
    lines = [f"## Pass-1 early-gate check{label}", ""]
    if not per_cell:
        lines.append("No cells supplied.")
        return "\n".join(lines)
    lines.append("| Cell | Headline Sharpe | n_oos | SE | approx CI_lo | Verdict |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for name, r in per_cell.items():
        verdict = "**RUN Pass 2**" if r.can_clear else "skip"
        lines.append(
            f"| {name} | {r.headline_sharpe:.4f} | {r.n_oos_bars} | "
            f"{r.se_inflated:.4f} | {r.approx_ci_lo:.4f} | {verdict} |"
        )
    any_clear = any(r.can_clear for r in per_cell.values())
    lines.append("")
    if not any_clear:
        lines.append(
            "**No cell can plausibly clear `CI_lo > 0`** — full Pass 2 "
            "(MC + bootstrap) skipped. Verdict: RETIRED."
        )
    else:
        clearable = [n for n, r in per_cell.items() if r.can_clear]
        lines.append(
            f"Cells that may clear `CI_lo > 0`: **{', '.join(clearable)}**. "
            "Pass 2 should run on these; others can be skipped."
        )
    return "\n".join(lines)


__all__ = [
    "Pass1GateResult",
    "pass1_can_clear_ci_gate",
    "pass1_can_clear_from_returns",
    "pass1_can_clear_any_cell",
    "format_gate_report",
]
