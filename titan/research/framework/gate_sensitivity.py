"""Gate-sensitivity harness for the L76 cascade re-test (audit P4-6).

The L76 cascade retired 8-9 strategies; L76 itself warns these are
falsification candidates, not replication targets. P4-6 asks the complementary
anti-confirmation-bias question: is each RETIRE *robust* to defensible gate
choices, or did it hinge on a knife-edge threshold? This harness re-runs
``decide()`` across a grid of plausible gate variations (the program's
questionable knobs: CI_lo_best in {0, 0.1, 0.2}, the DSR worst boundary, the MC
pass ratio) and reports, per strategy, the verdict spread and whether ANY
defensible gate would have deployed it.

A RETIRE with ``robust_retire=True`` (no defensible gate deploys it) is a sound
retirement. One that flips to a deployable tier under a reasonable gate is a
**revive candidate** worth a full re-audit (L40 roll-aware + basket extension,
per the plan).

Scope: this operates at the decision layer (the recorded ``DecisionInputs`` axis
vector per strategy), so it covers the CI_lo / DSR / MC gate knobs. Fold-count /
pre-vs-post-2.4 MC-threshold sensitivity needs a full WFO re-run and is out of
scope for this decision-level harness.
"""

from __future__ import annotations

from dataclasses import dataclass

from titan.research.framework.decision import (
    DecisionInputs,
    GateThresholds,
    Verdict,
    decide,
)

DEPLOYABLE: frozenset[Verdict] = frozenset({Verdict.DEPLOY, Verdict.CONDITIONAL_WATCHPOINT})


@dataclass(frozen=True)
class GateVariation:
    """A named gate-threshold variation to re-decide under."""

    label: str
    thresholds: GateThresholds


@dataclass(frozen=True)
class SensitivityResult:
    """Verdict spread for one strategy across the gate grid."""

    verdicts: dict[str, Verdict]  # variation label -> verdict
    distinct_verdicts: frozenset[Verdict]
    deployable_under: list[str]  # variations that yield a deployable tier
    robust_retire: bool  # no variation deploys it (a sound retirement)
    gate_sensitive: bool  # the verdict changes across the grid at all


def default_l76_variations() -> list[GateVariation]:
    """The program's flagged-questionable decision-layer gates: the CI_lo gate
    (P1-6 suggested raising it to 0.2) and the DSR worst boundary.
    """
    return [
        GateVariation("ci_lo_best=0.0 (lenient/default)", GateThresholds(ci_lo_best=0.0)),
        GateVariation("ci_lo_best=0.1", GateThresholds(ci_lo_best=0.1)),
        GateVariation("ci_lo_best=0.2 (P1-6 strict)", GateThresholds(ci_lo_best=0.2)),
        GateVariation("dsr_worst=0.40 (lenient)", GateThresholds(dsr_worst=0.40)),
        GateVariation("mc_worst_ratio=3.0 (lenient)", GateThresholds(mc_worst_ratio=3.0)),
    ]


def verdict_gate_sensitivity(
    inputs: DecisionInputs,
    variations: list[GateVariation] | None = None,
) -> SensitivityResult:
    """Re-decide ``inputs`` under each gate variation; summarise the spread.

    ``robust_retire`` is True when no variation produces a deployable verdict --
    i.e. the retirement does not depend on the gate choice.
    """
    variations = variations or default_l76_variations()
    verdicts = {v.label: decide(inputs, thresholds=v.thresholds).verdict for v in variations}
    deployable_under = [label for label, v in verdicts.items() if v in DEPLOYABLE]
    distinct = frozenset(verdicts.values())
    return SensitivityResult(
        verdicts=verdicts,
        distinct_verdicts=distinct,
        deployable_under=deployable_under,
        robust_retire=len(deployable_under) == 0,
        gate_sensitive=len(distinct) > 1,
    )


__all__ = [
    "GateVariation",
    "SensitivityResult",
    "DEPLOYABLE",
    "default_l76_variations",
    "verdict_gate_sensitivity",
]
