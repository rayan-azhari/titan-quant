"""Unit tests for the portfolio-heat envelope
(`titan/research/framework/portfolio_heat.py`).

Covers V3.8 §4.2 + §4.6 C3:
    1. compute_portfolio_heat identity + edge cases.
    2. Defensive clamp of negative risk_amount to zero.
    3. evaluate_heat_envelope PASS / FAIL + diagnostic reasons.
    4. Normal vs crisis regime cap (8% vs 6%) per §4.6 C3 plumbing.
    5. candidate parameter: "what if I added this trade?" path.
    6. The canonical scenario: 4 trades at 2% R = 8% (boundary) passes,
       4 trades at 2.1% R fails. This is the rationale the directive
       cites for the 8% cap (4 simultaneous correlated trades at 2% R).
    7. would_candidate_breach_heat convenience wrapper.
    8. Public-API contract via __init__.py re-export.
"""

from __future__ import annotations

import pytest

from titan.research.framework.portfolio_heat import (
    DEFAULT_CRISIS_HEAT_CAP,
    DEFAULT_NORMAL_HEAT_CAP,
    HeatCheckResult,
    PositionHeat,
    compute_portfolio_heat,
    evaluate_heat_envelope,
    would_candidate_breach_heat,
)

# ── compute_portfolio_heat ─────────────────────────────────────────────────


def test_heat_empty_portfolio_is_zero():
    assert compute_portfolio_heat([], equity=100_000.0) == 0.0


def test_heat_single_position_matches_ratio():
    pos = PositionHeat(symbol="ES", risk_amount=2_000.0)
    # $2k R / $100k equity = 2.0%
    assert compute_portfolio_heat([pos], equity=100_000.0) == pytest.approx(0.02)


def test_heat_sums_across_positions():
    positions = [
        PositionHeat("ES", 2_000.0),
        PositionHeat("NQ", 2_000.0),
        PositionHeat("ZB", 1_500.0),
    ]
    # Total = $5,500 / $100k = 5.5%
    assert compute_portfolio_heat(positions, equity=100_000.0) == pytest.approx(0.055)


def test_heat_zero_equity_returns_zero():
    """Defensive: division-by-zero is silent; returns 0 not inf."""
    pos = PositionHeat("ES", 2_000.0)
    assert compute_portfolio_heat([pos], equity=0.0) == 0.0
    assert compute_portfolio_heat([pos], equity=-100.0) == 0.0


def test_heat_clamps_negative_risk_to_zero():
    """A negative risk_amount is malformed -- don't credit the calculation."""
    pos = PositionHeat("WEIRD", -500.0)
    assert compute_portfolio_heat([pos], equity=100_000.0) == 0.0


def test_heat_mixed_positive_and_negative_risk():
    """Negative entries clamp to zero; positives still aggregate."""
    positions = [PositionHeat("ES", 2_000.0), PositionHeat("BAD", -1_000.0)]
    assert compute_portfolio_heat(positions, equity=100_000.0) == pytest.approx(0.02)


# ── evaluate_heat_envelope ─────────────────────────────────────────────────


def test_envelope_passes_when_heat_below_cap():
    positions = [PositionHeat("ES", 2_000.0), PositionHeat("ZB", 1_500.0)]
    # 3.5% heat << 8% cap
    res = evaluate_heat_envelope(positions, equity=100_000.0)
    assert res.portfolio_heat == pytest.approx(0.035)
    assert res.heat_cap == DEFAULT_NORMAL_HEAT_CAP
    assert res.heat_pass
    assert res.passes
    assert res.reasons == ()
    assert res.per_position_r == {"ES": 2_000.0, "ZB": 1_500.0}
    assert res.total_r == pytest.approx(3_500.0)


def test_envelope_passes_at_exact_cap():
    """The 8% cap is inclusive: portfolio_heat == 8% exactly is PASS."""
    positions = [PositionHeat("ES", 8_000.0)]  # 8% on $100k
    res = evaluate_heat_envelope(positions, equity=100_000.0)
    assert res.portfolio_heat == pytest.approx(DEFAULT_NORMAL_HEAT_CAP)
    assert res.heat_pass
    assert res.passes


def test_envelope_fails_when_heat_above_cap():
    """6 positions at $1.5k R each on $100k = 9% heat, above 8% cap."""
    positions = [PositionHeat(f"P{i}", 1_500.0) for i in range(6)]
    res = evaluate_heat_envelope(positions, equity=100_000.0)
    assert res.portfolio_heat == pytest.approx(0.09)
    assert not res.heat_pass
    assert not res.passes
    assert any("Portfolio heat" in r for r in res.reasons)


def test_envelope_canonical_scenario_4_trades_at_2pct_R():
    """Directive §4.2 rationale: 4 simultaneous correlated trades at 2% R each
    = 8% total. This sets the cap. Verify the boundary explicitly."""
    positions = [PositionHeat(f"P{i}", 2_000.0) for i in range(4)]
    res = evaluate_heat_envelope(positions, equity=100_000.0)
    # Exactly at cap -> PASS (cap is inclusive).
    assert res.portfolio_heat == pytest.approx(0.08)
    assert res.heat_pass
    # 1 more 2% trade would breach (5 * 2% = 10%).
    candidate = PositionHeat("P5", 2_000.0)
    res2 = evaluate_heat_envelope(positions, equity=100_000.0, candidate=candidate)
    assert res2.portfolio_heat == pytest.approx(0.10)
    assert not res2.heat_pass


def test_envelope_crisis_regime_uses_lower_cap():
    """Crisis-regime cap of 6% trips at lower heat than normal-regime 8%."""
    positions = [PositionHeat(f"P{i}", 1_750.0) for i in range(4)]
    # Total = 7% heat: passes 8% normal, fails 6% crisis.
    normal = evaluate_heat_envelope(positions, equity=100_000.0, regime_normal=True)
    crisis = evaluate_heat_envelope(positions, equity=100_000.0, regime_normal=False)
    assert normal.heat_cap == DEFAULT_NORMAL_HEAT_CAP
    assert normal.heat_pass
    assert crisis.heat_cap == DEFAULT_CRISIS_HEAT_CAP
    assert not crisis.heat_pass
    assert any("crisis" in r for r in crisis.reasons)


def test_envelope_candidate_included_in_heat_sum():
    """Pre-trade hook: candidate's R is folded into the heat sum before
    evaluating the cap."""
    current = [PositionHeat("ES", 3_000.0), PositionHeat("NQ", 3_000.0)]
    candidate = PositionHeat("ZB", 2_500.0)
    # Without candidate: heat = 6_000 / 100_000 = 6% (PASS at 8%)
    # With candidate:    heat = 8_500 / 100_000 = 8.5% (FAIL at 8%)
    res_no = evaluate_heat_envelope(current, equity=100_000.0)
    res_yes = evaluate_heat_envelope(current, equity=100_000.0, candidate=candidate)
    assert res_no.heat_pass
    assert res_yes.portfolio_heat == pytest.approx(0.085)
    assert not res_yes.heat_pass


def test_envelope_custom_caps_respected():
    """Custom normal and crisis caps change the verdict."""
    positions = [PositionHeat("ES", 3_000.0)]
    # Default: 3% passes 8%; custom strict 2% fails.
    default = evaluate_heat_envelope(positions, equity=100_000.0)
    custom = evaluate_heat_envelope(positions, equity=100_000.0, normal_cap=0.02)
    assert default.heat_pass
    assert not custom.heat_pass


def test_envelope_result_dataclass_is_frozen():
    res = evaluate_heat_envelope([], equity=100_000.0)
    assert isinstance(res, HeatCheckResult)
    with pytest.raises(Exception):
        res.heat_pass = False  # type: ignore[misc]


def test_envelope_empty_portfolio_with_no_candidate_passes_trivially():
    res = evaluate_heat_envelope([], equity=100_000.0)
    assert res.portfolio_heat == 0.0
    assert res.heat_pass
    assert res.passes


def test_envelope_zero_equity_yields_zero_heat_passes():
    """If equity is zero, portfolio_heat is 0 (defensive) which is below any
    positive cap -- so the check vacuously passes. Edge case for fresh
    account at startup."""
    res = evaluate_heat_envelope([PositionHeat("ES", 100.0)], equity=0.0)
    assert res.portfolio_heat == 0.0
    assert res.heat_pass


# ── would_candidate_breach_heat convenience wrapper ───────────────────────


def test_would_candidate_breach_heat_true_when_breaches():
    current = [PositionHeat("ES", 7_000.0)]  # 7% heat
    candidate = PositionHeat("NQ", 2_000.0)  # +2% -> 9% heat
    assert would_candidate_breach_heat(current, 100_000.0, candidate) is True


def test_would_candidate_breach_heat_false_when_safe():
    current = [PositionHeat("ES", 3_000.0)]  # 3% heat
    candidate = PositionHeat("NQ", 2_000.0)  # +2% -> 5% heat
    assert would_candidate_breach_heat(current, 100_000.0, candidate) is False


def test_would_candidate_breach_heat_uses_crisis_cap_when_specified():
    current = [PositionHeat("ES", 4_000.0)]  # 4% heat
    candidate = PositionHeat("NQ", 2_500.0)  # +2.5% -> 6.5%
    # 6.5% passes 8% normal, fails 6% crisis
    assert would_candidate_breach_heat(current, 100_000.0, candidate, regime_normal=True) is False
    assert would_candidate_breach_heat(current, 100_000.0, candidate, regime_normal=False) is True


# ── Public-API contract ────────────────────────────────────────────────────


def test_portfolio_heat_symbols_exported_from_framework_init():
    from titan.research.framework import (  # noqa: F401
        DEFAULT_CRISIS_HEAT_CAP,
        DEFAULT_NORMAL_HEAT_CAP,
        HeatCheckResult,
        PositionHeat,
        compute_portfolio_heat,
        evaluate_heat_envelope,
        would_candidate_breach_heat,
    )
