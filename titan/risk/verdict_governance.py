"""Verdict-staleness deployment governance (audit P4-5).

The audit's "live-on-unconfirmed" finding: CONDITIONAL / unconfirmed strategies
run live with no automatic guardrail, and a strategy can drift past its re-audit
deadline while still holding capital. This module is the enforcement engine that
the daily rollup + allocator consult before a rebalance. Given each live
strategy's verdict, intended deployment weight, re-audit deadline, and V3.8
-envelope status, it returns the GOVERNED weights + an alert payload, enforcing:

  1. CONDITIONAL_WATCHPOINT / TIER_UNCONFIRMED -> hard-capped at <=5% each.
  2. SUSPECT / RETIRE                          -> demoted to paper (weight 0).
  3. Missed re-audit deadline                  -> auto-demoted to paper.
  4. Below the required V3.8 envelope (< shadow) -> demoted to paper.
  5. A daily-summary alert listing every action taken.

It is a PURE function over a list of ``StrategyVerdict`` records -- the records
themselves (currently prose in the config TOMLs) are populated by the caller;
wiring config -> records is a follow-on migration.

Reuses the framework ``Verdict`` (titan.research.framework.decision) so the
governance tiers are exactly the audit verdicts, not a parallel taxonomy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from titan.research.framework.decision import Verdict

# CONDITIONAL / unconfirmed live strategies are hard-capped at 5% each.
DEFAULT_UNCONFIRMED_CAP = 0.05
# Verdicts that may hold capital but only under the cap.
CAPPED_VERDICTS: frozenset[Verdict] = frozenset(
    {Verdict.CONDITIONAL_WATCHPOINT, Verdict.TIER_UNCONFIRMED}
)
# Verdicts that must not hold live capital at all.
PAPER_VERDICTS: frozenset[Verdict] = frozenset({Verdict.SUSPECT, Verdict.RETIRE})
# Acceptable V3.8-envelope states, least-to-most integrated.
ENVELOPE_ORDER: tuple[str, ...] = ("none", "shadow", "live")


@dataclass(frozen=True)
class StrategyVerdict:
    """A live strategy's deployment-governance record."""

    name: str
    verdict: Verdict
    deployment_weight: float  # intended weight before governance
    reaudit_due: date  # the strategy's re-audit deadline
    envelope_status: str = "shadow"  # "none" | "shadow" | "live"


@dataclass(frozen=True)
class GovernanceAction:
    """One governance decision for one strategy."""

    name: str
    action: str  # "ok" | "cap" | "demote_to_paper"
    original_weight: float
    governed_weight: float
    reasons: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class GovernanceReport:
    """Aggregate result of a governance pass."""

    as_of: date
    actions: list[GovernanceAction]
    governed_weights: dict[str, float]

    @property
    def has_violations(self) -> bool:
        return any(a.action != "ok" for a in self.actions)

    def alert_lines(self) -> list[str]:
        """Daily-summary alert lines, one per non-ok action."""
        return [
            f"[verdict-governance {self.as_of.isoformat()}] {a.name}: {a.action} "
            f"{a.original_weight:.3f}->{a.governed_weight:.3f} ({'; '.join(a.reasons)})"
            for a in self.actions
            if a.action != "ok"
        ]


def _envelope_below(status: str, required: str) -> bool:
    """True if ``status`` is below the ``required`` envelope integration level."""
    s = ENVELOPE_ORDER.index(status) if status in ENVELOPE_ORDER else 0
    r = ENVELOPE_ORDER.index(required) if required in ENVELOPE_ORDER else 0
    return s < r


def evaluate_verdict_governance(
    records: list[StrategyVerdict],
    *,
    as_of: date,
    unconfirmed_cap: float = DEFAULT_UNCONFIRMED_CAP,
    require_envelope: str = "shadow",
) -> GovernanceReport:
    """Apply the P4-5 governance cascade and return governed weights + alerts.

    Most-restrictive-wins: a missed deadline, a non-deployable verdict, or a
    below-shadow envelope all demote to paper (weight 0); a CONDITIONAL /
    unconfirmed verdict is capped at ``unconfirmed_cap``. ``require_envelope``
    is the minimum acceptable V3.8-envelope integration ("shadow" by default).
    """
    actions: list[GovernanceAction] = []
    for r in records:
        reasons: list[str] = []
        w = r.deployment_weight
        action = "ok"

        # Demotion conditions (any -> paper). Collect all reasons for the alert.
        if as_of > r.reaudit_due:
            reasons.append(f"re-audit overdue (due {r.reaudit_due.isoformat()})")
        if r.verdict in PAPER_VERDICTS:
            reasons.append(f"verdict {r.verdict.value} not deployable")
        if _envelope_below(r.envelope_status, require_envelope):
            reasons.append(f"envelope '{r.envelope_status}' below required '{require_envelope}'")

        if reasons:
            w, action = 0.0, "demote_to_paper"
        elif r.verdict in CAPPED_VERDICTS and r.deployment_weight > unconfirmed_cap:
            w, action = unconfirmed_cap, "cap"
            reasons.append(f"{r.verdict.value} hard-capped at {unconfirmed_cap:.0%}")

        actions.append(
            GovernanceAction(
                name=r.name,
                action=action,
                original_weight=r.deployment_weight,
                governed_weight=w,
                reasons=reasons,
            )
        )

    return GovernanceReport(
        as_of=as_of,
        actions=actions,
        governed_weights={a.name: a.governed_weight for a in actions},
    )


__all__ = [
    "StrategyVerdict",
    "GovernanceAction",
    "GovernanceReport",
    "evaluate_verdict_governance",
    "DEFAULT_UNCONFIRMED_CAP",
    "CAPPED_VERDICTS",
    "PAPER_VERDICTS",
    "ENVELOPE_ORDER",
]
