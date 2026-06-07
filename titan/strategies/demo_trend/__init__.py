"""Educational demo strategy (titan-quant public companion).

A deliberately simple moving-average trend rule with NO expected edge. It exists
to demonstrate the framework's validation pipeline and the live integration
contract, not to make money. See the book, Parts IV-V.
"""

from titan.strategies.demo_trend.strategy import demo_trend_positions, demo_trend_signal

__all__ = ["demo_trend_signal", "demo_trend_positions"]
