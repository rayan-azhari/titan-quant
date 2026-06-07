"""Unified research framework for the Titan-IBKR-Algo project.

Specified in directives/Methodology Audit & Unified Framework 2026-05-14.md.

Every NEW audit + every RE-AUDIT uses these primitives. The framework
standardises:

    * Strategy-class typology (titan.research.framework.typology)
    * Walk-forward fold construction (titan.research.framework.wfo)
    * Sanctuary discipline (titan.research.framework.sanctuary)
    * Monte Carlo block bootstrap (titan.research.framework.mc)
    * Deflated Sharpe Ratio (titan.research.framework.dsr)
    * 4-axis decision matrix (titan.research.framework.decision)

The existing audit scripts in research/strategies/, research/cross_asset/,
research/ml/, research/orb/, etc. remain as historical records. They
should NOT be mass-refactored; instead they are RE-RUN under the
framework as part of Phase 2 of the methodology directive.
"""

from titan.research.framework.allocator_cdar import (
    DEFAULT_MAX_WEIGHT,
    DEFAULT_TAIL,
    CdarResult,
    compute_cdar_weights,
)
from titan.research.framework.allocator_erc import ErcResult, compute_erc_weights
from titan.research.framework.allocator_hrp import HrpResult, compute_hrp_weights
from titan.research.framework.allocator_nco import (
    NcoResult,
    compute_nco_weights,
    marcenko_pastur_denoise,
)
from titan.research.framework.amortised_mc import (
    InferFn,
    PrefitFn,
    run_block_mc_amortised,
)
from titan.research.framework.calmar import (
    DEFAULT_CALMAR_LIFT_GATE,
    DEFAULT_SHARPE_LIFT_GATE,
    MIN_BARS_FOR_CALMAR,
    CalmarCi,
    CalmarPromotionResult,
    CalmarResult,
    bootstrap_calmar_ci,
    calmar_lift,
    compute_cagr,
    compute_calmar,
    evaluate_promotion,
)
from titan.research.framework.cost_realism import (
    CostReconciliation,
    CostSensitivity,
    apply_cost_model,
    apply_roll_cost,
    apply_sqrt_impact_slippage,
    realistic_cost_gate,
    reconcile_cost,
    reconcile_costs,
)
from titan.research.framework.crisis_stress import (
    NAMED_CRISES,
    CrisisStressResult,
    CrisisWindowResult,
    run_crisis_stress,
)
from titan.research.framework.dashboard import (
    AuditResult,
    CellSummary,
    render_dashboard,
)
from titan.research.framework.dd_throttle import (
    DEFAULT_PEAK_WINDOW_BARS,
    DEFAULT_RESET_DD,
    DEFAULT_THROTTLE_MULTIPLIER,
    DEFAULT_TRIGGER_DD,
    NORMAL_MULTIPLIER,
    DdThrottlePath,
    DdThrottleState,
    compute_rolling_dd_from_peak,
    compute_throttle_multiplier,
    initial_throttle_state,
    simulate_throttle_path,
    update_throttle,
)
from titan.research.framework.decision import (
    DecisionInputs,
    DecisionResult,
    GateThresholds,
    Verdict,
    classify_axis_noise,
    decide,
)
from titan.research.framework.drift_cusum import CusumResult, run_cusum_drift
from titan.research.framework.dsr import DsrResult, deflated_sharpe, sr_var_from_sweep
from titan.research.framework.early_gate import (
    Pass1GateResult,
    format_gate_report,
    pass1_can_clear_any_cell,
    pass1_can_clear_ci_gate,
    pass1_can_clear_from_returns,
)
from titan.research.framework.fdm import (
    DEFAULT_FDM_CAP,
    FdmResult,
    fdm_from_uniform_correlation,
    forecast_diversification_multiplier,
)
from titan.research.framework.gate_sensitivity import (
    GateVariation,
    SensitivityResult,
    default_l76_variations,
    verdict_gate_sensitivity,
)
from titan.research.framework.ic_analysis import (
    IcSummary,
    QuantileResult,
    cross_sectional_ic,
    forward_returns,
    ic_decay,
    quantile_returns,
    rolling_ic,
    summarise_ic,
)
from titan.research.framework.kelly import (
    KellyFraction,
    compute_kelly_fraction,
    normalise_kelly_to_weights,
)
from titan.research.framework.leverage_envelope import (
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
from titan.research.framework.mc import (
    McResult,
    RelativeMcResult,
    run_block_mc,
    run_relative_block_mc,
)
from titan.research.framework.pessimistic_fill import (
    assess_pessimistic_ruin,
    stress_stop_fills,
)
from titan.research.framework.portfolio_heat import (
    DEFAULT_CRISIS_HEAT_CAP,
    DEFAULT_NORMAL_HEAT_CAP,
    HeatCheckResult,
    PositionHeat,
    compute_portfolio_heat,
    evaluate_heat_envelope,
    would_candidate_breach_heat,
)
from titan.research.framework.program_ledger import (
    DEFAULT_DSR_GATE,
    PboResult,
    ProgramGateResult,
    ProgramLedger,
    ProgramLedgerSummary,
    TrialRecord,
    default_program_ledger,
    probability_of_backtest_overfitting,
    program_deflated_sharpe,
    program_dsr_gate,
)
from titan.research.framework.returns_hygiene import (
    mark_pure_missing,
    mask_nonpositive_prices,
    n_vol_observations,
    realised_vol,
    returns_from_prices,
)
from titan.research.framework.robustness import (
    NoiseConfig,
    NoiseLevelResult,
    NoiseRobustnessResult,
    run_noise_robustness,
)
from titan.research.framework.ruin import (
    RuinAssessment,
    assess_joint_ruin,
    assess_strategy_ruin,
)
from titan.research.framework.sanctuary import (
    DivergenceTest,
    MultiSanctuaryResult,
    SanctuarySlice,
    multi_window_sanctuary_test,
    sanctuary_divergence_test,
    slice_multi_sanctuary,
    slice_sanctuary,
)
from titan.research.framework.typology import (
    COST_CME_FUTURES_LIQUID,
    COST_FX_MAJOR,
    COST_IG_DFB_INDEX,
    COST_UCITS_ETF,
    COST_US_EQUITY_LARGE_CAP,
    COST_US_ETF_LIQUID,
    DEFAULTS,
    CostModel,
    McConfig,
    SharpeReporting,
    StrategyClass,
    StrategyClassDefaults,
    WfoConfig,
    defaults_for,
)
from titan.research.framework.universe import (
    Membership,
    load_memberships,
    point_in_time_universe,
    universe_as_of,
    validate_memberships,
)
from titan.research.framework.wfo import (
    CombinatorialFold,
    Fold,
    build_cpcv_folds,
    build_folds,
    cpcv_n_paths,
    iter_folds,
)
from titan.strategies.regime_filter import (
    DEFAULT_VIX_THRESHOLD,
    DEFAULT_VOL_PERCENTILE_THRESHOLD,
    DEFAULT_VOL_PERCENTILE_WINDOW_BARS,
    DEFAULT_VOL_WINDOW_BARS,
    RegimeResult,
    compute_realised_vol_annualised,
    compute_vol_percentile_current,
    is_crisis_regime,
)

__all__ = [
    # V3.8 — Calmar promotion gate (Objective Reframe §2.2)
    "CalmarResult",
    "CalmarCi",
    "CalmarPromotionResult",
    "compute_cagr",
    "compute_calmar",
    "bootstrap_calmar_ci",
    "calmar_lift",
    "evaluate_promotion",
    "DEFAULT_CALMAR_LIFT_GATE",
    "DEFAULT_SHARPE_LIFT_GATE",
    "MIN_BARS_FOR_CALMAR",
    # V3.8 — DD throttle (Objective Reframe §4.3 + §4.6.3 graded ladder)
    "DdThrottleState",
    "DdThrottlePath",
    "compute_rolling_dd_from_peak",
    "compute_throttle_multiplier",
    "initial_throttle_state",
    "update_throttle",
    "simulate_throttle_path",
    "DEFAULT_TRIGGER_DD",
    "DEFAULT_RESET_DD",
    "DEFAULT_PEAK_WINDOW_BARS",
    "DEFAULT_THROTTLE_MULTIPLIER",
    "NORMAL_MULTIPLIER",
    # V3.8 — Gross leverage + SPAN buffer envelope (Objective Reframe §4.6 C1+C2)
    "PositionSnapshot",
    "LeverageCheckResult",
    "compute_gross_leverage",
    "compute_span_buffer_ratio",
    "evaluate_leverage_envelope",
    "would_candidate_breach_leverage",
    "DEFAULT_NORMAL_LEVERAGE_CAP",
    "DEFAULT_CRISIS_LEVERAGE_CAP",
    "DEFAULT_SPAN_BUFFER_MIN",
    # V3.8 — Crisis-regime detector (Objective Reframe §4.6 C3)
    "RegimeResult",
    "compute_realised_vol_annualised",
    "compute_vol_percentile_current",
    "is_crisis_regime",
    "DEFAULT_VIX_THRESHOLD",
    "DEFAULT_VOL_PERCENTILE_THRESHOLD",
    "DEFAULT_VOL_PERCENTILE_WINDOW_BARS",
    "DEFAULT_VOL_WINDOW_BARS",
    # V3.8 — Portfolio heat envelope (Objective Reframe §4.2 + §4.6 C3)
    "PositionHeat",
    "HeatCheckResult",
    "compute_portfolio_heat",
    "evaluate_heat_envelope",
    "would_candidate_breach_heat",
    "DEFAULT_NORMAL_HEAT_CAP",
    "DEFAULT_CRISIS_HEAT_CAP",
    # V3.7 — Risk-of-ruin (L65) + Kelly (L67) + ERC + Crisis + Drift
    "RuinAssessment",
    "assess_strategy_ruin",
    "assess_joint_ruin",
    "stress_stop_fills",
    "assess_pessimistic_ruin",
    "KellyFraction",
    "compute_kelly_fraction",
    "normalise_kelly_to_weights",
    "ErcResult",
    "compute_erc_weights",
    # Cost-realism primitives (audit P1-12/13/14)
    "apply_cost_model",
    "apply_roll_cost",
    "apply_sqrt_impact_slippage",
    "CostSensitivity",
    "realistic_cost_gate",
    "CostReconciliation",
    "reconcile_cost",
    "reconcile_costs",
    # CDaR drawdown-constrained allocator (audit P3-1)
    "CdarResult",
    "compute_cdar_weights",
    "DEFAULT_TAIL",
    "DEFAULT_MAX_WEIGHT",
    # Hierarchical Risk Parity allocator (audit P3-2)
    "HrpResult",
    "compute_hrp_weights",
    # Nested Clustered Optimization allocator (audit P3-2)
    "NcoResult",
    "compute_nco_weights",
    "marcenko_pastur_denoise",
    "CrisisStressResult",
    "CrisisWindowResult",
    "NAMED_CRISES",
    "run_crisis_stress",
    "CusumResult",
    "run_cusum_drift",
    # Typology
    "StrategyClass",
    "StrategyClassDefaults",
    "WfoConfig",
    "McConfig",
    "SharpeReporting",
    "CostModel",
    "DEFAULTS",
    "defaults_for",
    "COST_CME_FUTURES_LIQUID",
    "COST_US_EQUITY_LARGE_CAP",
    "COST_US_ETF_LIQUID",
    "COST_UCITS_ETF",
    "COST_FX_MAJOR",
    "COST_IG_DFB_INDEX",
    # Survivorship-free point-in-time universe (audit data-gate / C5)
    "Membership",
    "universe_as_of",
    "point_in_time_universe",
    "validate_memberships",
    "load_memberships",
    # WFO
    "Fold",
    "build_folds",
    "iter_folds",
    # CPCV — Combinatorial Purged Cross-Validation (P1-10)
    "CombinatorialFold",
    "build_cpcv_folds",
    "cpcv_n_paths",
    # Sanctuary
    "SanctuarySlice",
    "slice_sanctuary",
    "DivergenceTest",
    "sanctuary_divergence_test",
    # Multi-window / bootstrapped-CI sanctuary (P1-10)
    "MultiSanctuaryResult",
    "slice_multi_sanctuary",
    "multi_window_sanctuary_test",
    # MC
    "McResult",
    "run_block_mc",
    "RelativeMcResult",
    "run_relative_block_mc",
    # Amortised MC (IS-frozen model state cached across paths)
    "PrefitFn",
    "InferFn",
    "run_block_mc_amortised",
    # DSR
    "DsrResult",
    "deflated_sharpe",
    "sr_var_from_sweep",
    # Program-wide multiple-testing ledger + cross-program deflation (P1-9)
    "TrialRecord",
    "ProgramLedger",
    "ProgramLedgerSummary",
    "ProgramGateResult",
    "PboResult",
    "program_deflated_sharpe",
    "program_dsr_gate",
    "probability_of_backtest_overfitting",
    "default_program_ledger",
    "DEFAULT_DSR_GATE",
    # FDM (Carver Forecast Diversification Multiplier; backlog J5)
    "FdmResult",
    "forecast_diversification_multiplier",
    "fdm_from_uniform_correlation",
    "DEFAULT_FDM_CAP",
    # IC analysis (Alphalens-style IC decay + quantile spread; annualisation via metrics)
    "IcSummary",
    "QuantileResult",
    "forward_returns",
    "cross_sectional_ic",
    "rolling_ic",
    "summarise_ic",
    "ic_decay",
    "quantile_returns",
    # Early gate (Pass-1-gates-Pass-2 speed-up)
    "Pass1GateResult",
    "pass1_can_clear_ci_gate",
    "pass1_can_clear_from_returns",
    "pass1_can_clear_any_cell",
    "format_gate_report",
    # Decision
    "Verdict",
    "DecisionInputs",
    "DecisionResult",
    "GateThresholds",
    "decide",
    "classify_axis_noise",
    # L76 gate-sensitivity harness (audit P4-6)
    "GateVariation",
    "SensitivityResult",
    "default_l76_variations",
    "verdict_gate_sensitivity",
    # Robustness (noise-injection gate -- Varma)
    "NoiseConfig",
    "NoiseLevelResult",
    "NoiseRobustnessResult",
    "run_noise_robustness",
    # Return + volatility hygiene (P1-22)
    "mask_nonpositive_prices",
    "returns_from_prices",
    "mark_pure_missing",
    "realised_vol",
    "n_vol_observations",
    # Dashboard
    "AuditResult",
    "CellSummary",
    "render_dashboard",
]
