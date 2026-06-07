"""Unit tests for the gross-leverage + SPAN-buffer envelope
(`titan/research/framework/leverage_envelope.py`).

Covers V3.8 §4.6 controls C1 (gross leverage cap) and C2 (SPAN margin buffer):
    1. compute_gross_leverage: sum-of-abs / equity identity + edge cases.
    2. abs notional: short + long positions do NOT net.
    3. compute_span_buffer_ratio: posted / exchange_min with None handling.
    4. evaluate_leverage_envelope four-quadrant gate logic
       (both pass / leverage fails / SPAN fails / both fail).
    5. Normal vs crisis regime cap (8x vs 6x) per §4.6 C3 plumbing.
    6. candidate parameter: "what if I added this trade?" path.
    7. Vacuous SPAN pass when no broker data available.
    8. would_candidate_breach_leverage convenience wrapper.
    9. Public-API contract via __init__.py re-export.
"""

from __future__ import annotations

import pytest

from titan.research.framework.leverage_envelope import (
    DEFAULT_CRISIS_LEVERAGE_CAP,
    DEFAULT_NORMAL_LEVERAGE_CAP,
    LeverageCheckResult,
    PositionSnapshot,
    compute_gross_leverage,
    compute_span_buffer_ratio,
    evaluate_leverage_envelope,
    would_candidate_breach_leverage,
)

# ── compute_gross_leverage ─────────────────────────────────────────────────


def test_gross_leverage_empty_portfolio_is_zero():
    assert compute_gross_leverage([], equity=100_000.0) == 0.0


def test_gross_leverage_single_position_matches_ratio():
    pos = PositionSnapshot(symbol="ES", abs_notional=300_000.0)
    # $300k notional / $100k equity = 3.0x
    assert compute_gross_leverage([pos], equity=100_000.0) == pytest.approx(3.0)


def test_gross_leverage_sums_across_positions():
    positions = [
        PositionSnapshot("ES", 300_000.0),
        PositionSnapshot("NQ", 430_000.0),
        PositionSnapshot("ZB", 120_000.0),
    ]
    # Total = 850_000; equity = 100_000 -> 8.5x
    assert compute_gross_leverage(positions, equity=100_000.0) == pytest.approx(8.5)


def test_gross_leverage_treats_abs_notional_as_positive():
    """V3.8 §4.6: 'abs notional' means short + long do NOT net out.

    We pass abs_notional (the L1 contribution) directly so the function
    treats every position as additive to gross. Caller is responsible for
    converting signed notional to absolute before constructing the
    PositionSnapshot."""
    # Short ES + long ES: same |notional|, both add to gross.
    short_es = PositionSnapshot("ES_short", 300_000.0)
    long_es = PositionSnapshot("ES_long", 300_000.0)
    assert compute_gross_leverage([short_es, long_es], equity=100_000.0) == pytest.approx(6.0)


def test_gross_leverage_zero_equity_returns_zero():
    """Defensive: division-by-zero is silent; returns 0 not inf."""
    pos = PositionSnapshot("ES", 300_000.0)
    assert compute_gross_leverage([pos], equity=0.0) == 0.0
    assert compute_gross_leverage([pos], equity=-100.0) == 0.0


def test_gross_leverage_clamps_negative_abs_notional_to_zero():
    """Defensive: a malformed PositionSnapshot with negative abs_notional
    should not credit the leverage calculation."""
    pos = PositionSnapshot("WEIRD", -50_000.0)
    assert compute_gross_leverage([pos], equity=100_000.0) == 0.0


# ── compute_span_buffer_ratio ─────────────────────────────────────────────


def test_span_buffer_ratio_basic():
    # Broker holding 2x exchange minimum -> ratio = 2.0
    assert compute_span_buffer_ratio(10_000.0, 5_000.0) == pytest.approx(2.0)


def test_span_buffer_ratio_below_minimum():
    # Broker holding only 1.5x exchange minimum -> ratio = 1.5
    assert compute_span_buffer_ratio(7_500.0, 5_000.0) == pytest.approx(1.5)


def test_span_buffer_ratio_none_when_data_unavailable():
    assert compute_span_buffer_ratio(None, 5_000.0) is None
    assert compute_span_buffer_ratio(10_000.0, None) is None
    assert compute_span_buffer_ratio(None, None) is None


def test_span_buffer_ratio_zero_exchange_min_with_posted_returns_inf():
    """Data-quality edge: posted > 0 but exchange_min == 0. Return +inf so the
    PRM sees the position as far above any threshold; logged elsewhere."""
    assert compute_span_buffer_ratio(10_000.0, 0.0) == float("inf")


def test_span_buffer_ratio_zero_exchange_min_with_zero_posted_returns_none():
    """If both are zero (full data gap), return None so the position is
    excluded from the SPAN check entirely."""
    assert compute_span_buffer_ratio(0.0, 0.0) is None


# ── evaluate_leverage_envelope ─────────────────────────────────────────────


def test_envelope_passes_when_leverage_and_span_both_clear():
    positions = [
        PositionSnapshot("ES", 200_000.0, margin_posted=12_000.0, exchange_min_margin=5_000.0),
        PositionSnapshot("ZB", 100_000.0, margin_posted=8_000.0, exchange_min_margin=3_000.0),
    ]
    res = evaluate_leverage_envelope(positions, equity=100_000.0)
    assert res.gross_leverage == pytest.approx(3.0)
    assert res.leverage_cap == DEFAULT_NORMAL_LEVERAGE_CAP
    assert res.leverage_pass
    assert res.span_buffer_pass
    assert res.passes
    assert res.reasons == ()
    assert res.span_breaching_symbols == ()
    # Per-symbol ratios populated
    assert "ES" in res.span_buffer_ratios
    assert "ZB" in res.span_buffer_ratios


def test_envelope_fails_when_leverage_exceeds_cap():
    """3 positions at $400k each on $100k equity = 12x gross, above the 8x cap."""
    positions = [
        PositionSnapshot("ES", 400_000.0),
        PositionSnapshot("NQ", 400_000.0),
        PositionSnapshot("GC", 400_000.0),
    ]
    res = evaluate_leverage_envelope(positions, equity=100_000.0)
    assert res.gross_leverage == pytest.approx(12.0)
    assert not res.leverage_pass
    # No margin data -> SPAN check vacuously passes
    assert res.span_buffer_pass
    assert not res.passes
    assert any("Gross leverage" in r for r in res.reasons)


def test_envelope_fails_when_span_buffer_below_minimum():
    """Single position with broker margin == 1.5x exchange min (below 2x gate)."""
    positions = [
        PositionSnapshot("ES", 200_000.0, margin_posted=7_500.0, exchange_min_margin=5_000.0),
    ]
    res = evaluate_leverage_envelope(positions, equity=100_000.0)
    assert res.leverage_pass  # 2x leverage is fine
    assert not res.span_buffer_pass
    assert "ES" in res.span_breaching_symbols
    assert not res.passes
    assert any("SPAN buffer" in r for r in res.reasons)


def test_envelope_fails_both_controls():
    """High gross leverage AND insufficient SPAN buffer."""
    positions = [
        PositionSnapshot("ES", 500_000.0, margin_posted=6_000.0, exchange_min_margin=5_000.0),
        PositionSnapshot("NQ", 500_000.0, margin_posted=6_000.0, exchange_min_margin=5_000.0),
    ]
    res = evaluate_leverage_envelope(positions, equity=100_000.0)
    assert not res.leverage_pass
    assert not res.span_buffer_pass
    assert not res.passes
    assert len(res.reasons) == 2


def test_envelope_crisis_regime_uses_lower_cap():
    """Crisis-regime cap of 6x trips at lower leverage than normal-regime 8x."""
    positions = [
        PositionSnapshot("ES", 400_000.0),
        PositionSnapshot("NQ", 300_000.0),
    ]
    # Total = 700_000 / 100_000 = 7x: passes 8x normal, fails 6x crisis.
    normal = evaluate_leverage_envelope(positions, equity=100_000.0, regime_normal=True)
    crisis = evaluate_leverage_envelope(positions, equity=100_000.0, regime_normal=False)
    assert normal.leverage_cap == DEFAULT_NORMAL_LEVERAGE_CAP
    assert normal.leverage_pass
    assert crisis.leverage_cap == DEFAULT_CRISIS_LEVERAGE_CAP
    assert not crisis.leverage_pass
    assert any("crisis" in r for r in crisis.reasons)


def test_envelope_candidate_included_in_leverage_sum():
    """Pre-trade hook: candidate's notional is folded into the gross-leverage
    sum before evaluating the cap."""
    current = [
        PositionSnapshot("ES", 300_000.0),
        PositionSnapshot("NQ", 300_000.0),
    ]
    candidate = PositionSnapshot("ZB", 250_000.0)
    # Without candidate: gross = 600_000 / 100_000 = 6x (PASS)
    # With candidate:    gross = 850_000 / 100_000 = 8.5x (FAIL @ 8x)
    res_no = evaluate_leverage_envelope(current, equity=100_000.0)
    res_yes = evaluate_leverage_envelope(current, equity=100_000.0, candidate=candidate)
    assert res_no.leverage_pass
    assert res_yes.gross_leverage == pytest.approx(8.5)
    assert not res_yes.leverage_pass


def test_envelope_vacuous_span_pass_when_no_broker_data():
    """Backtest path: no margin_posted / exchange_min provided -> SPAN check
    vacuously passes; result.span_buffer_pass is True."""
    positions = [PositionSnapshot("ES", 200_000.0), PositionSnapshot("ZB", 100_000.0)]
    res = evaluate_leverage_envelope(positions, equity=100_000.0)
    assert res.span_buffer_pass
    assert res.span_buffer_ratios == {}
    assert res.span_buffer_min_ratio == float("inf")
    assert res.passes  # Leverage 3x passes, SPAN vacuously passes


def test_envelope_partial_span_data_still_evaluates_available_positions():
    """If some positions have margin data and others don't, the SPAN check
    runs on the ones that do; missing-data positions are silently skipped."""
    positions = [
        PositionSnapshot("ES", 200_000.0, margin_posted=12_000.0, exchange_min_margin=5_000.0),
        PositionSnapshot("ZB", 100_000.0),  # No margin data
        PositionSnapshot(
            "NQ", 100_000.0, margin_posted=4_000.0, exchange_min_margin=5_000.0
        ),  # ratio 0.8, FAIL
    ]
    res = evaluate_leverage_envelope(positions, equity=100_000.0)
    assert "ZB" not in res.span_buffer_ratios
    assert res.span_buffer_ratios["ES"] == pytest.approx(2.4)
    assert res.span_buffer_ratios["NQ"] == pytest.approx(0.8)
    assert res.span_breaching_symbols == ("NQ",)
    assert not res.span_buffer_pass


def test_envelope_custom_thresholds_respected():
    """Custom leverage cap + custom span min thresholds change the verdict."""
    positions = [
        PositionSnapshot("ES", 300_000.0, margin_posted=6_000.0, exchange_min_margin=5_000.0),
    ]
    # Default: 3x leverage passes; ratio 1.2 fails 2x default.
    default = evaluate_leverage_envelope(positions, equity=100_000.0)
    assert default.leverage_pass
    assert not default.span_buffer_pass
    # Custom: stricter 2x leverage cap fails; lenient 1.0 SPAN min passes.
    custom = evaluate_leverage_envelope(
        positions, equity=100_000.0, leverage_cap_normal=2.0, span_buffer_min=1.0
    )
    assert not custom.leverage_pass
    assert custom.span_buffer_pass


def test_envelope_result_dataclass_is_frozen():
    """LeverageCheckResult is immutable -- can't mutate after returning."""
    res = evaluate_leverage_envelope([], equity=100_000.0)
    assert isinstance(res, LeverageCheckResult)
    with pytest.raises(Exception):
        res.passes = False  # type: ignore[misc]


def test_envelope_empty_portfolio_with_no_candidate_passes_trivially():
    res = evaluate_leverage_envelope([], equity=100_000.0)
    assert res.gross_leverage == 0.0
    assert res.passes


# ── would_candidate_breach_leverage convenience wrapper ───────────────────


def test_would_candidate_breach_leverage_true_when_breaches():
    current = [PositionSnapshot("ES", 700_000.0)]
    candidate = PositionSnapshot("NQ", 200_000.0)
    # 700 + 200 = 900k / 100k = 9x > 8x cap
    assert would_candidate_breach_leverage(current, 100_000.0, candidate) is True


def test_would_candidate_breach_leverage_false_when_safe():
    current = [PositionSnapshot("ES", 300_000.0)]
    candidate = PositionSnapshot("NQ", 200_000.0)
    # 300 + 200 = 500k / 100k = 5x < 8x cap
    assert would_candidate_breach_leverage(current, 100_000.0, candidate) is False


def test_would_candidate_breach_leverage_uses_crisis_cap_when_specified():
    current = [PositionSnapshot("ES", 400_000.0)]
    candidate = PositionSnapshot("NQ", 250_000.0)
    # 400 + 250 = 650k / 100k = 6.5x -- passes 8x normal, fails 6x crisis
    assert (
        would_candidate_breach_leverage(current, 100_000.0, candidate, regime_normal=True) is False
    )
    assert (
        would_candidate_breach_leverage(current, 100_000.0, candidate, regime_normal=False) is True
    )


# ── Public-API contract ────────────────────────────────────────────────────


def test_leverage_envelope_symbols_exported_from_framework_init():
    from titan.research.framework import (  # noqa: F401
        DEFAULT_CRISIS_LEVERAGE_CAP,
        DEFAULT_NORMAL_LEVERAGE_CAP,
        DEFAULT_SPAN_BUFFER_MIN,
        LeverageCheckResult,
        PositionSnapshot,
        compute_gross_leverage,
        compute_span_buffer_ratio,
        evaluate_leverage_envelope,
        would_candidate_breach_leverage,
    )
