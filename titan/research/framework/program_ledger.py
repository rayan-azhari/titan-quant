"""Program-wide multiple-testing ledger + cross-program deflation (P1-9).

Specified in directives/Audit Remediation Plan 2026-05-29.md row P1-9
(finding H12): *"Program-wide multiple-testing: a trial ledger of all
distinct strategy hypotheses + a cross-program deflation (BLP `e_max`
over program-N) or PBO before any DEPLOY."*

Why this exists
===============

The Deflated Sharpe Ratio in :mod:`titan.research.framework.dsr` corrects a
single sweep's best cell for the *N cells in that sweep*. It does NOT see the
rest of the research program. But every strategy we have ever audited is a
draw from the same "find an edge" search, so the relevant null is the expected
maximum Sharpe over **all distinct hypotheses the program has tried** -- which
is far larger than any one sweep's N.

A strategy that clears its own sweep's DSR can still be a program-level false
discovery: with ~30+ distinct hypotheses explored (see the Retirement
Registry), the expected null max Sharpe is materially higher, and a borderline
+0.4-Sharpe candidate that looked deployable in isolation may not clear the
program-wide bar. This module makes that correction explicit and mandatory
before any DEPLOY verdict.

Two complementary tools
=======================

1. **Cross-program DSR** (``program_deflated_sharpe`` / ``program_dsr_gate``):
   re-runs the BLP 2014 deflation with ``N = program trial count`` instead of
   the local sweep N. This is the primary gate -- a thin, auditable wrapper
   over the proven :func:`dsr.deflated_sharpe`.

2. **PBO** (``probability_of_backtest_overfitting``): the Bailey-Borwein-Lopez
   de Prado-Zhu (2017) Combinatorially-Symmetric Cross-Validation estimate of
   the probability that the in-sample-best configuration underperforms the
   out-of-sample median. Use it when you have an IS/OOS performance matrix
   across configurations and want a model-free overfitting probability rather
   than a Sharpe deflation.

Wiring into ``decide()``
========================

This module does NOT change :func:`decision.decide`. The intended use is: for
any candidate about to receive a DEPLOY verdict, compute
``program_dsr_gate(...)`` and feed its ``dsr.dsr_prob`` as the ``dsr_prob``
axis input to ``decide()`` (replacing the sweep-local DSR). A program-DSR
below 0.95 collapses the DSR axis off "best", which the count-of-best ladder
then propagates into the verdict.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, replace
from itertools import combinations

import numpy as np
import pandas as pd

from titan.research.framework.dsr import DsrResult, deflated_sharpe

# Deployment DSR gate, shared with the decision matrix (decision.GateThresholds.dsr_best).
DEFAULT_DSR_GATE: float = 0.95

# Valid trial statuses. "rejected" = audited and rejected pre-deployment (never
# live); "retired" = was deployment-eligible and later retired; the rest are
# self-explanatory. Every status still counts as one distinct hypothesis tried.
VALID_STATUSES: frozenset[str] = frozenset(
    {"deployed", "shadow", "paper", "conditional", "retired", "rejected"}
)


@dataclass(frozen=True)
class TrialRecord:
    """One distinct strategy hypothesis explored by the research program.

    A *hypothesis* is a distinct edge/mechanic on a distinct universe -- not a
    parameter cell. ``n_sweep_cells`` records how many parameter cells were
    screened within the hypothesis, so the program trial count can be taken
    either as distinct hypotheses (the default, statistically-honest choice
    since cells within a sweep are correlated) or as total cells (a stricter
    upper bound).

    ``oos_sharpe`` is the realised stitched-OOS (or canonical) Sharpe where the
    registry records one, else ``None``. Only non-None values feed the
    program-wide Sharpe-variance estimate -- we never fabricate a number.
    """

    strategy_id: str
    strategy_class: str
    hypothesis: str
    status: str
    date: str  # ISO date the trial concluded
    oos_sharpe: float | None = None
    n_sweep_cells: int = 1

    def __post_init__(self) -> None:
        """Validate status + cell count at construction."""
        if self.status not in VALID_STATUSES:
            raise ValueError(
                f"TrialRecord {self.strategy_id!r}: status {self.status!r} "
                f"not in {sorted(VALID_STATUSES)}"
            )
        if self.n_sweep_cells < 1:
            raise ValueError(
                f"TrialRecord {self.strategy_id!r}: n_sweep_cells must be >= 1, "
                f"got {self.n_sweep_cells}"
            )


@dataclass(frozen=True)
class ProgramLedgerSummary:
    """Aggregate view of the program ledger, for audit logs + the gate."""

    n_distinct_hypotheses: int
    n_total_cells: int
    n_with_sharpe: int
    sr_var_across_program: float
    deployed: tuple[str, ...]
    shadow: tuple[str, ...]
    retired: tuple[str, ...]
    rejected: tuple[str, ...]


class ProgramLedger:
    """An append-only register of every distinct strategy hypothesis tried.

    The ledger is the denominator of the program-wide multiple-testing
    correction. Maintainers append a :class:`TrialRecord` whenever a new audit
    concludes (mirroring the Retirement Registry); the count then flows into
    :func:`program_deflated_sharpe`.
    """

    def __init__(self, records: Iterable[TrialRecord] = ()) -> None:
        self._records: list[TrialRecord] = list(records)
        self._validate_unique()

    def _validate_unique(self) -> None:
        ids = [r.strategy_id for r in self._records]
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            raise ValueError(f"ProgramLedger: duplicate strategy_id(s): {sorted(dupes)}")

    def add(self, record: TrialRecord) -> None:
        """Append a trial. Raises on a duplicate ``strategy_id``."""
        if any(r.strategy_id == record.strategy_id for r in self._records):
            raise ValueError(f"ProgramLedger: strategy_id {record.strategy_id!r} already present")
        self._records.append(record)

    @property
    def records(self) -> tuple[TrialRecord, ...]:
        return tuple(self._records)

    @property
    def n_distinct_hypotheses(self) -> int:
        return len(self._records)

    def n_program_trials(self, *, count_mode: str = "hypotheses") -> int:
        """The program-wide trial count N.

        ``count_mode="hypotheses"`` (default) returns the number of distinct
        hypotheses -- the statistically-honest effective N, since cells within
        a sweep are correlated and counting them as independent overstates the
        null max. ``count_mode="cells"`` returns the sum of ``n_sweep_cells``,
        a stricter upper bound for sensitivity analysis.
        """
        if count_mode == "hypotheses":
            return self.n_distinct_hypotheses
        if count_mode == "cells":
            return sum(r.n_sweep_cells for r in self._records)
        raise ValueError(f"count_mode must be 'hypotheses' or 'cells', got {count_mode!r}")

    def sr_var_across_program(self) -> float:
        """Sample variance of realised Sharpes across all trials with a recorded
        Sharpe. The program-level analogue of ``dsr.sr_var_from_sweep`` -- the
        dispersion of outcomes the search produces. Returns 0.0 if fewer than
        two trials carry a Sharpe (caller should then supply an explicit
        ``sr_var_across_trials``).
        """
        sharpes = np.array(
            [r.oos_sharpe for r in self._records if r.oos_sharpe is not None], dtype=float
        )
        if len(sharpes) < 2:
            return 0.0
        return float(np.var(sharpes, ddof=1))

    def summary(self) -> ProgramLedgerSummary:
        by_status = lambda s: tuple(r.strategy_id for r in self._records if r.status == s)  # noqa: E731
        # "conditional" rolls up under shadow for the summary's risk view.
        return ProgramLedgerSummary(
            n_distinct_hypotheses=self.n_distinct_hypotheses,
            n_total_cells=self.n_program_trials(count_mode="cells"),
            n_with_sharpe=sum(1 for r in self._records if r.oos_sharpe is not None),
            sr_var_across_program=round(self.sr_var_across_program(), 6),
            deployed=by_status("deployed"),
            shadow=by_status("shadow") + by_status("conditional") + by_status("paper"),
            retired=by_status("retired"),
            rejected=by_status("rejected"),
        )


@dataclass(frozen=True)
class ProgramGateResult:
    """Outcome of the cross-program DSR gate."""

    passes: bool
    dsr: DsrResult
    n_program_trials: int
    count_mode: str
    gate: float
    rationale: str


def program_deflated_sharpe(
    sr_hat: float,
    *,
    returns: pd.Series | np.ndarray,
    ledger: ProgramLedger,
    count_mode: str = "hypotheses",
    extra_trials: int = 0,
    sr_var_across_trials: float | None = None,
) -> DsrResult:
    """Deflate ``sr_hat`` against the *program-wide* trial count.

    Thin wrapper over :func:`dsr.deflated_sharpe` that substitutes
    ``N = ledger.n_program_trials(count_mode) + extra_trials`` for the local
    sweep N, and uses the program-wide Sharpe variance unless an explicit
    ``sr_var_across_trials`` is supplied.

    Parameters
    ----------
    sr_hat:
        The candidate's observed annualised Sharpe.

    Returns:
        The candidate's per-bar return series (skew + kurt come from this).
    ledger:
        The program trial register supplying N and the Sharpe variance.
    count_mode:
        ``"hypotheses"`` (default, honest effective N) or ``"cells"`` (stricter).
    extra_trials:
        Add the candidate's own sweep N on top of the program count (the
        candidate is itself a fresh batch of trials). Default 0.
    sr_var_across_trials:
        Override the program-wide Sharpe variance (e.g. pass the variance from
        the candidate's own sweep family when that is the more relevant null).
    """
    n_program = ledger.n_program_trials(count_mode=count_mode) + int(extra_trials)
    sr_var = (
        ledger.sr_var_across_program()
        if sr_var_across_trials is None
        else float(sr_var_across_trials)
    )
    result = deflated_sharpe(
        sr_hat,
        sr_var_across_trials=sr_var,
        returns=returns,
        n_trials=max(n_program, 2),
    )
    # n_trials in the returned struct should report the program N we actually
    # deflated against, not the max(...,2) guard floor.
    return replace(result, n_trials=n_program)


def program_dsr_gate(
    sr_hat: float,
    *,
    returns: pd.Series | np.ndarray,
    ledger: ProgramLedger,
    count_mode: str = "hypotheses",
    extra_trials: int = 0,
    sr_var_across_trials: float | None = None,
    gate: float = DEFAULT_DSR_GATE,
) -> ProgramGateResult:
    """Run the cross-program DSR and report whether it clears ``gate`` (0.95).

    This is the check the audit mandates *before any DEPLOY*. Feed the
    resulting ``dsr.dsr_prob`` as the DSR axis input to ``decide()``.
    """
    dsr_result = program_deflated_sharpe(
        sr_hat,
        returns=returns,
        ledger=ledger,
        count_mode=count_mode,
        extra_trials=extra_trials,
        sr_var_across_trials=sr_var_across_trials,
    )
    passes = dsr_result.dsr_prob >= gate
    n = dsr_result.n_trials
    rationale = (
        f"program-DSR prob={dsr_result.dsr_prob:.4f} {'>=' if passes else '<'} {gate:.2f} "
        f"| N_program={n} ({count_mode}) | e_max_SR={dsr_result.e_max_sr:.3f} "
        f"| sr_hat={sr_hat:.3f}"
    )
    return ProgramGateResult(
        passes=passes,
        dsr=dsr_result,
        n_program_trials=n,
        count_mode=count_mode,
        gate=gate,
        rationale=rationale,
    )


# --------------------------------------------------------------------------- #
# PBO -- Probability of Backtest Overfitting (CSCV)                           #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PboResult:
    """Combinatorially-Symmetric Cross-Validation overfitting estimate.

    Attributes:
    ----------
    pbo:
        Probability of backtest overfitting in [0, 1] -- the fraction of CSCV
        splits where the in-sample-best configuration landed below the
        out-of-sample median (logit < 0). PBO near 0.5 means the IS-best is no
        better than a coin flip OOS; PBO near 0 means the IS-best reliably
        persists OOS.
    n_combinations:
        Number of IS/OOS splits evaluated, C(n_partitions, n_partitions/2).
    n_partitions:
        Number of disjoint time blocks the sample was split into.
    n_configs:
        Number of competing configurations (columns).
    median_logit:
        Median of the per-split logit relative-rank -- a one-number summary
        (positive = IS-best tends to stay above OOS median).
    """

    pbo: float
    n_combinations: int
    n_partitions: int
    n_configs: int
    median_logit: float


def _block_standardised_score(block: np.ndarray) -> np.ndarray:
    """Per-column standardised mean (mean/std, ddof=1) of a (rows x configs) block.

    This is a *rank-invariant ranking statistic* for CSCV, not an annualised
    Sharpe: annualisation is a positive scalar that does not change any
    argmax or relative rank, so it is intentionally omitted (there is no
    `periods_per_year` to set). Zero-variance columns map to 0.0.
    """
    mean = block.mean(axis=0)
    std = block.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(std > 1e-12, mean / std, 0.0)
    return ratio


def probability_of_backtest_overfitting(
    perf_matrix: np.ndarray | pd.DataFrame,
    *,
    n_partitions: int = 10,
    metric: str = "sharpe",
) -> PboResult:
    """Estimate PBO via CSCV (Bailey, Borwein, López de Prado, Zhu 2017).

    Parameters
    ----------
    perf_matrix:
        A ``(T_observations x N_configs)`` matrix of per-period performance
        (returns by default). Each column is one competing configuration.
    n_partitions:
        Number of disjoint, equal-size, contiguous time blocks S (must be
        even). The number of IS/OOS splits is C(S, S/2): S=10 -> 252,
        S=12 -> 924, S=16 -> 12870. Default 10 balances resolution and cost.
    metric:
        ``"sharpe"`` (default) ranks configs by per-block Sharpe; ``"mean"``
        ranks by mean return.

    Returns:
    -------
    PboResult with the overfitting probability and diagnostics.

    Raises:
    ------
    ValueError
        If ``n_partitions`` is not a positive even integer, if there are fewer
        than 2 configurations, or if the sample has fewer rows than partitions.
    """
    mat = (
        perf_matrix.to_numpy(dtype=float)
        if isinstance(perf_matrix, pd.DataFrame)
        else np.asarray(perf_matrix, dtype=float)
    )
    if mat.ndim != 2:
        raise ValueError("perf_matrix must be 2-D (T_observations x N_configs)")
    t_obs, n_configs = mat.shape
    if n_configs < 2:
        raise ValueError(f"PBO needs >= 2 configurations, got {n_configs}")
    if n_partitions < 2 or n_partitions % 2 != 0:
        raise ValueError(f"n_partitions must be a positive even integer, got {n_partitions}")
    if t_obs < n_partitions:
        raise ValueError(f"need >= n_partitions ({n_partitions}) rows, got {t_obs}")

    if metric == "sharpe":
        score = _block_standardised_score
    elif metric == "mean":
        score = lambda b: b.mean(axis=0)  # noqa: E731
    else:
        raise ValueError(f"metric must be 'sharpe' or 'mean', got {metric!r}")

    # Disjoint contiguous blocks of (near-)equal size. np.array_split tolerates
    # T not divisible by S; the slight size imbalance is immaterial to CSCV.
    blocks = np.array_split(np.arange(t_obs), n_partitions)
    half = n_partitions // 2
    all_block_ids = range(n_partitions)

    logits: list[float] = []
    for is_block_ids in combinations(all_block_ids, half):
        is_set = set(is_block_ids)
        is_rows = np.concatenate([blocks[b] for b in range(n_partitions) if b in is_set])
        oos_rows = np.concatenate([blocks[b] for b in range(n_partitions) if b not in is_set])

        is_score = score(mat[is_rows])
        oos_score = score(mat[oos_rows])

        n_star = int(np.argmax(is_score))
        # Relative rank of the IS-best among OOS scores, in (0, 1). Ties broken
        # by "average" so a config tied with all others sits at the median.
        oos_ranks = pd.Series(oos_score).rank(method="average").to_numpy()
        omega = oos_ranks[n_star] / (n_configs + 1.0)
        omega = min(max(omega, 1e-9), 1.0 - 1e-9)  # keep logit finite
        logits.append(math.log(omega / (1.0 - omega)))

    logit_arr = np.array(logits, dtype=float)
    pbo = float(np.mean(logit_arr < 0.0))
    return PboResult(
        pbo=round(pbo, 4),
        n_combinations=len(logits),
        n_partitions=n_partitions,
        n_configs=n_configs,
        median_logit=round(float(np.median(logit_arr)), 4),
    )


# --------------------------------------------------------------------------- #
# Default program ledger -- the documented research history                   #
# --------------------------------------------------------------------------- #

# One TrialRecord per distinct strategy hypothesis the program has audited.
# Sourced from directives/Retirement Registry.md + the live/shadow/paper
# registries. ``oos_sharpe`` is filled ONLY where the registry records a
# canonical/stitched-OOS Sharpe (no fabrication); ``n_sweep_cells`` is filled
# where the grid size is documented, else defaults to 1.
#
# MAINTENANCE: append a record here when a new audit concludes, exactly as a
# line is added to the Retirement Registry. This tuple is the denominator of
# the program-wide multiple-testing correction (P1-9).
_DEFAULT_TRIALS: tuple[TrialRecord, ...] = (
    # Illustrative ledger for the public demo (fictional; not a real programme).
    TrialRecord(
        "demo_trend", "DAILY_TREND", "Example daily trend (educational)",
        "paper", "2026-01-01", oos_sharpe=0.30,
    ),
    TrialRecord(
        "demo_mean_reversion", "DAILY_MEAN_REVERSION",
        "Example mean-reversion (educational)", "shadow", "2026-01-01",
    ),
    TrialRecord(
        "demo_carry", "CARRY", "Example carry (educational)",
        "retired", "2026-01-01", oos_sharpe=-0.10,
    ),
)


def default_program_ledger() -> ProgramLedger:
    """The program ledger seeded from the documented research history.

    Reflects the Retirement Registry (27 retires/rejections) plus the live,
    shadow, and paper strategies. The distinct-hypothesis count is the
    program-wide multiple-testing denominator; pass ``count_mode="cells"`` to
    :func:`program_deflated_sharpe` for the stricter cell-level bound.
    """
    return ProgramLedger(_DEFAULT_TRIALS)
