"""Per-lesson enforcement registry (audit P4-3, "lessons-as-gates").

The V3.6/V3.7 lessons catalogue is prose. This makes the discipline *auditable*:
each tracked lesson maps to HOW it is enforced -- ``test`` (an automated CI gate),
``reviewer`` (caught only by a human at review), or ``none`` (an open gap). The
companion ``tests/test_lesson_registry.py`` asserts that every ``test``-enforced
lesson's cited evidence file actually exists and that the critical risk lessons
are test-enforced -- so the registry cannot silently claim enforcement it lacks
(the registry is itself a gate).

This is the structured tracking the P4-3 item asked for ("track per-lesson
enforced by: [test|reviewer|none]"); ``reviewer`` / ``none`` rows are the honest
backlog of discipline not yet mechanised.
"""

from __future__ import annotations

from dataclasses import dataclass

TEST = "test"
REVIEWER = "reviewer"
NONE = "none"


@dataclass(frozen=True)
class LessonEnforcement:
    """How a single catalogue lesson is enforced."""

    lesson: str  # catalogue id, e.g. "L65"
    title: str
    enforced_by: str  # TEST | REVIEWER | NONE
    evidence: str  # repo-relative test path (TEST) or a short note


# Evidence paths are repo-relative; the registry test asserts the TEST ones exist.
LESSON_ENFORCEMENT: tuple[LessonEnforcement, ...] = (
    LessonEnforcement(
        "L17", "Same-bar look-ahead / causality", TEST, "tests/test_causality_all.py"
    ),
    LessonEnforcement("L52", "IS->OOS plateau stability", TEST, "tests/test_ic_census_lib.py"),
    LessonEnforcement(
        "L65", "Risk-of-ruin deployment gate", TEST, "tests/test_framework_synthetic.py"
    ),
    LessonEnforcement("L67", "Kelly is a leverage, not a weight", TEST, "tests/test_kelly.py"),
    LessonEnforcement(
        "H2", "Known-edge / known-no-edge calibration", TEST, "tests/test_lessons_h2_synthetic.py"
    ),
    LessonEnforcement("H18", "Pessimistic stop-fill ruin", TEST, "tests/test_pessimistic_fill.py"),
    LessonEnforcement(
        "P1-27", "Covariance shrinkage + crisis overlay", TEST, "tests/test_covariance_shrinkage.py"
    ),
    LessonEnforcement(
        "P1-12_16", "Cost realism + reconciliation", TEST, "tests/test_cost_realism.py"
    ),
    LessonEnforcement(
        "C5", "Survivorship-free point-in-time universe", TEST, "tests/test_universe_pit.py"
    ),
    LessonEnforcement(
        "P4-4",
        "Ratified ruin objective (25%/40%)",
        TEST,
        "tests/test_p4_4_objective_ratification.py",
    ),
    LessonEnforcement(
        "P4-5", "Verdict-staleness deployment governance", TEST, "tests/test_verdict_governance.py"
    ),
    LessonEnforcement(
        "P0-9", "Live-vs-predicted drift de-risk", TEST, "tests/test_p0_9_drift_monitor.py"
    ),
    # Honest backlog -- discipline enforced only by a human reviewer, or not yet.
    LessonEnforcement(
        "L74",
        "Carry-premium math (carry strategies)",
        REVIEWER,
        "carry-not-using-L74 lint not built (P4-3 follow-on)",
    ),
    LessonEnforcement(
        "L75",
        "Hybrid pre-flight before any deploy claim",
        REVIEWER,
        "process discipline; no unit test",
    ),
    LessonEnforcement(
        "L76",
        "Pre-2014 edges are falsification candidates",
        REVIEWER,
        "P4-6 cascade re-test is the harness; per-strategy re-run pending",
    ),
)

CRITICAL_LESSONS: frozenset[str] = frozenset({"L17", "L52", "L65", "L67", "H2", "C5", "P4-4"})


def enforcement_for(lesson: str) -> LessonEnforcement | None:
    return next((e for e in LESSON_ENFORCEMENT if e.lesson == lesson), None)


def lessons_by_mechanism(mechanism: str) -> list[LessonEnforcement]:
    return [e for e in LESSON_ENFORCEMENT if e.enforced_by == mechanism]


def unenforced_lessons() -> list[LessonEnforcement]:
    """Lessons with no enforcement at all -- the open governance gaps."""
    return [e for e in LESSON_ENFORCEMENT if e.enforced_by == NONE]


__all__ = [
    "LessonEnforcement",
    "LESSON_ENFORCEMENT",
    "CRITICAL_LESSONS",
    "TEST",
    "REVIEWER",
    "NONE",
    "enforcement_for",
    "lessons_by_mechanism",
    "unenforced_lessons",
]
