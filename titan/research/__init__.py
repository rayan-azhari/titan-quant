# titan/research — Shared helpers for research and live math.
#
# This module is the single source of truth for Sharpe/vol/z-score calculations
# used across both ``research/`` pipelines and ``titan/strategies/`` live
# sizing. No module outside ``titan/research`` should reimplement these
# primitives — that was the root cause of the April 2026 audit findings
# (frequency-mismatched annualization, filter-then-annualize Sharpe bias,
# duplicated sqrt(252) constants applied to H1/M5 data).
