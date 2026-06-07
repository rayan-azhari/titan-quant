"""Canonical strategy-audit dashboard.

Builds a self-contained interactive HTML report from a standardised
``AuditResult`` produced by any framework-based audit. The dashboard format
is identical across strategies so they can be compared apples-to-apples.

Sections (per the design spec):
    A. Headline      — equity curve, drawdown, rolling Sharpe, KPI cards.
    B. Distribution  — monthly heatmap, histogram, Q-Q plot.
    C. Robustness    — WFO fold-by-fold, sanctuary divergence, MC fan,
                       relative-MC scatter, noise robustness curve.
    D. Sweep         — per-cell verdict table, plateau bar chart,
                       decision-matrix axis bars.
    E. Strategy-specific overlay (optional, supplied by the caller).

Usage::

    from titan.research.framework.dashboard import AuditResult, render_dashboard

    result = AuditResult(...)  # populated by the audit harness
    render_dashboard(result, output_dir=Path(".tmp/reports/gem/2026-05-14/"))

Output: ``output_dir / dashboard.html``.

Design choices:
    - Plotly for figures so the HTML is interactive (hover, zoom, pan).
    - Jinja2 for layout — keeps Python free of HTML strings.
    - Everything inlined in one HTML file so the report is shareable
      (Slack, email, GitHub gist, git-archive). No external CDN required —
      we use ``include_plotlyjs="cdn"`` by default but flag to embed for
      offline-shareable files.
    - Sections C, D depend on framework primitives the audit already runs;
      Section E is a plug-in callable per StrategyClass.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from jinja2 import Template
from plotly.subplots import make_subplots

# Module-level — tested for output stability via __all__.
__all__ = [
    "AuditResult",
    "CellSummary",
    "render_dashboard",
]


# ── Schema ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CellSummary:
    """One row of the sweep verdict table.

    Captures the 4-axis decision-matrix inputs + the verdict. Field names
    deliberately match the GEM audit's ``CellResult`` so audits can pass
    their existing struct directly.
    """

    cell: str
    sharpe: float
    ci_lo: float
    ci_hi: float
    dsr_prob: float
    # MC axis -- either the absolute P(MaxDD>X) (legacy) or the relative
    # DD ratio (L17). The dashboard renders whichever is populated.
    mc_p_maxdd_gt_threshold: float | None = None
    mc_threshold_pct: float | None = None
    rel_mc_median_ratio: float | None = None
    rel_mc_p_strategy_better: float | None = None
    rel_mc_passes: bool | None = None
    sanctuary_sharpe: float = 0.0
    sanctuary_percentile: float = float("nan")
    sanctuary_lucky_flag: bool = False
    sanctuary_unlucky_flag: bool = False
    verdict: str = "UNKNOWN"
    verdict_rationale: str = ""


@dataclass
class AuditResult:
    """Everything a framework-based audit produces.

    The dashboard's only contract: an audit fills this struct and calls
    ``render_dashboard``. Optional fields ``= None`` are simply omitted
    from the report.
    """

    # Identity
    strategy_name: str
    strategy_class: str  # the StrategyClass enum value as string
    pre_reg_directive: str  # path to the V3.1 pre-reg
    run_date: str  # ISO-8601 e.g. "2026-05-14"
    bars_per_year: int

    # Data window
    data_start: pd.Timestamp
    data_end: pd.Timestamp
    sanctuary_start: pd.Timestamp
    sanctuary_end: pd.Timestamp

    # Canonical cell (the one analysed in detail). Per-cell stitched OOS
    # returns for every cell go in `cell_oos_returns`.
    canonical_cell: str
    cells: list[CellSummary]
    cell_oos_returns: dict[str, pd.Series]  # cell_name -> per-bar OOS returns
    cell_sanctuary_returns: dict[str, pd.Series]  # cell_name -> sanctuary returns

    # Benchmark series for overlay (e.g. buy-and-hold the underlying).
    # Both must be ALIGNED to the same time index as the OOS returns.
    benchmark_oos_returns: pd.Series | None = None
    benchmark_sanctuary_returns: pd.Series | None = None

    # Full strategy returns on the FULL visible window (not just the
    # stitched OOS). Used by the equity panel for a continuous historical
    # view -- the stitched OOS often covers only a fraction of the available
    # history under conservative WFO defaults. The headline metrics still
    # come from the stitched OOS; this is for visual sense-checking only.
    full_strategy_returns: pd.Series | None = None
    full_benchmark_returns: pd.Series | None = None
    # Index marks (start,end) of each OOS fold for the band overlay.
    oos_fold_intervals: list[tuple[pd.Timestamp, pd.Timestamp]] | None = None

    # WFO fold info (per canonical cell). List of dicts with keys
    # 'fold_id', 'oos_start', 'oos_end', 'sharpe'.
    fold_diagnostics: list[dict] | None = None

    # MC paths — both raw and relative. Optional but heavily used by the report.
    mc_strategy_sharpes: list[float] | None = None
    mc_strategy_maxdds: list[float] | None = None
    mc_benchmark_sharpes: list[float] | None = None
    mc_benchmark_maxdds: list[float] | None = None
    mc_threshold_pct: float | None = None
    mc_pass_prob: float | None = None
    rel_mc_median_ratio_gate: float | None = None
    rel_mc_p_strategy_better_gate: float | None = None

    # Noise robustness gate output (Varma).
    noise_levels: list[float] | None = None
    noise_sharpe_means: list[float] | None = None
    noise_sharpe_p5: list[float] | None = None
    noise_base_sharpe: float | None = None
    noise_max_degradation_gate: float | None = None

    # Sanctuary divergence test result.
    sanctuary_historical_window_sharpes: list[float] | None = None
    sanctuary_realised_sharpe: float | None = None
    sanctuary_percentile: float | None = None

    # Strategy-specific overlay panel (Section E). Optional Plotly Figure
    # produced by a per-StrategyClass plug-in.
    extra_overlay_figure: go.Figure | None = None
    extra_overlay_title: str = ""

    # Top-line lessons / notes string for the report header.
    headline_summary: str = ""


# ── Helpers ───────────────────────────────────────────────────────────────


def _cumulative_equity(returns: pd.Series, start_value: float = 1.0) -> pd.Series:
    return start_value * (1.0 + returns.fillna(0.0)).cumprod()


def _drawdown_series(returns: pd.Series) -> pd.Series:
    eq = _cumulative_equity(returns)
    hwm = eq.cummax()
    return eq / hwm - 1.0


def _annualised_metrics(returns: pd.Series, periods_per_year: int) -> dict[str, float]:
    """Quick KPI dict for the headline cards."""
    rets = returns.dropna()
    if len(rets) == 0:
        return {
            "sharpe": 0.0,
            "vol": 0.0,
            "cagr": 0.0,
            "max_dd": 0.0,
            "calmar": 0.0,
            "hit_rate": 0.0,
        }
    mu = float(rets.mean())
    sd = float(rets.std(ddof=1))
    sharpe = (mu / sd) * np.sqrt(periods_per_year) if sd > 0 else 0.0
    vol = sd * np.sqrt(periods_per_year)
    eq_end = float((1.0 + rets).prod())
    years = len(rets) / periods_per_year
    cagr = eq_end ** (1.0 / years) - 1.0 if years > 0 else 0.0
    max_dd = float(_drawdown_series(rets).min())
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0
    hit_rate = float((rets > 0).mean())
    return {
        "sharpe": sharpe,
        "vol": vol,
        "cagr": cagr,
        "max_dd": max_dd,
        "calmar": calmar,
        "hit_rate": hit_rate,
    }


def _monthly_returns_pivot(returns: pd.Series) -> pd.DataFrame:
    """Build the year-by-month percentage-return pivot."""
    rets = returns.dropna()
    if rets.empty:
        return pd.DataFrame()
    df = pd.DataFrame({"r": rets})
    df["year"] = df.index.year
    df["month"] = df.index.month
    monthly = df.groupby(["year", "month"])["r"].apply(lambda s: (1.0 + s).prod() - 1.0)
    return monthly.unstack(level="month")


# ── Section A — Headline figures ──────────────────────────────────────────


def _figure_equity_and_drawdown(
    strategy_oos: pd.Series,
    strategy_sanc: pd.Series | None,
    benchmark_oos: pd.Series | None,
    benchmark_sanc: pd.Series | None,
    sanctuary_start: pd.Timestamp,
    *,
    full_strategy: pd.Series | None = None,
    full_benchmark: pd.Series | None = None,
    oos_fold_intervals: list[tuple[pd.Timestamp, pd.Timestamp]] | None = None,
) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=("Equity (cumulative growth of $1)", "Drawdown"),
        row_heights=[0.65, 0.35],
        vertical_spacing=0.08,
    )

    # Prefer the full historical strategy returns (covers the entire visible
    # window) over the stitched OOS, which under conservative WFO defaults
    # may only span a few years. If full_strategy is provided we plot that
    # plus the sanctuary; the OOS regions are highlighted by fold-band
    # overlays so the reader can see which periods are framework-validated.
    def _concat(oos, sanc):
        if sanc is None or sanc.empty:
            return oos
        return pd.concat([oos, sanc])

    if full_strategy is not None and not full_strategy.empty:
        strat_full = _concat(full_strategy, strategy_sanc)
    else:
        strat_full = _concat(strategy_oos, strategy_sanc)
    strat_eq = _cumulative_equity(strat_full)
    strat_dd = _drawdown_series(strat_full)

    fig.add_trace(
        go.Scatter(
            x=strat_eq.index,
            y=strat_eq.values,
            name="Strategy",
            line=dict(color="#1f77b4", width=1.5),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=strat_dd.index,
            y=strat_dd.values * 100.0,
            name="Strategy DD",
            fill="tozeroy",
            line=dict(color="#1f77b4", width=0.5),
            fillcolor="rgba(31,119,180,0.25)",
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    if benchmark_oos is not None or full_benchmark is not None:
        if full_benchmark is not None and not full_benchmark.empty:
            bench_series = _concat(full_benchmark, benchmark_sanc)
        else:
            bench_series = _concat(benchmark_oos, benchmark_sanc)
        bench_eq = _cumulative_equity(bench_series)
        bench_dd = _drawdown_series(bench_series)
        fig.add_trace(
            go.Scatter(
                x=bench_eq.index,
                y=bench_eq.values,
                name="Benchmark",
                line=dict(color="#888", width=1.0, dash="dot"),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=bench_dd.index,
                y=bench_dd.values * 100.0,
                name="Benchmark DD",
                line=dict(color="#888", width=0.5, dash="dot"),
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    # OOS fold bands -- shade the regions the framework actually validated
    # against. Helps the reader distinguish "this is the WFO OOS coverage"
    # from "this is the full historical strategy run".
    if oos_fold_intervals:
        for s, e in oos_fold_intervals:
            fig.add_shape(
                type="rect",
                x0=s,
                x1=e,
                y0=0,
                y1=1,
                yref="y domain",
                fillcolor="rgba(31,119,180,0.06)",
                line=dict(width=0),
                row=1,
                col=1,
            )

    # Sanctuary boundary line. Use add_shape + add_annotation rather than
    # add_vline because plotly's add_vline annotation positioning is
    # incompatible with pd.Timestamp x-values (TypeError in
    # shapeannotation._mean).
    fig.add_shape(
        type="line",
        x0=sanctuary_start,
        x1=sanctuary_start,
        y0=0,
        y1=1,
        yref="y domain",
        line=dict(color="red", dash="dash", width=1),
        row=1,
        col=1,
    )
    fig.add_annotation(
        x=sanctuary_start,
        y=1.0,
        yref="y domain",
        text="Sanctuary -->",
        showarrow=False,
        xshift=4,
        yshift=-12,
        font=dict(color="red", size=10),
        row=1,
        col=1,
    )

    fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
    fig.update_layout(
        height=620,
        hovermode="x unified",
        margin=dict(l=60, r=20, t=50, b=40),
    )
    return fig


def _figure_rolling_sharpe(
    returns: pd.Series, periods_per_year: int, window: int = 252
) -> go.Figure:
    rets = returns.dropna()
    if len(rets) < window:
        return go.Figure().add_annotation(
            text=f"Need ≥ {window} bars for rolling Sharpe; have {len(rets)}",
            showarrow=False,
        )
    roll_mu = rets.rolling(window).mean()
    roll_sd = rets.rolling(window).std(ddof=1)
    roll_sh = (roll_mu / roll_sd) * np.sqrt(periods_per_year)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=roll_sh.index, y=roll_sh.values, line=dict(color="#1f77b4", width=1.5))
    )
    fig.add_hline(y=0, line_color="grey", line_dash="dot")
    fig.add_hline(y=1, line_color="green", line_dash="dot", annotation_text="Sharpe=1")
    fig.update_layout(
        title=f"Rolling {window}-bar Sharpe",
        height=300,
        yaxis_title="Sharpe",
        margin=dict(l=60, r=20, t=50, b=40),
    )
    return fig


# ── Section B — Distribution figures ──────────────────────────────────────


def _figure_monthly_heatmap(returns: pd.Series) -> go.Figure:
    pivot = _monthly_returns_pivot(returns)
    if pivot.empty:
        return go.Figure().add_annotation(text="No data", showarrow=False)
    z = pivot.values * 100.0
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=[
                "Jan",
                "Feb",
                "Mar",
                "Apr",
                "May",
                "Jun",
                "Jul",
                "Aug",
                "Sep",
                "Oct",
                "Nov",
                "Dec",
            ][: pivot.shape[1]],
            y=pivot.index.astype(str),
            colorscale="RdYlGn",
            zmid=0,
            text=np.round(z, 1),
            texttemplate="%{text}",
            textfont=dict(size=10),
            colorbar=dict(title="%"),
        )
    )
    fig.update_layout(
        title="Monthly returns (%)",
        height=420,
        margin=dict(l=60, r=20, t=50, b=40),
    )
    return fig


def _figure_distribution(returns: pd.Series) -> go.Figure:
    rets = returns.dropna()
    if len(rets) < 30:
        return go.Figure().add_annotation(text="Not enough data for distribution", showarrow=False)
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Daily returns histogram", "Q-Q vs normal"),
    )
    fig.add_trace(
        go.Histogram(x=rets.values * 100.0, nbinsx=60, marker_color="#1f77b4", showlegend=False),
        row=1,
        col=1,
    )
    fig.add_vline(x=0, line_color="grey", line_dash="dot", row=1, col=1)

    # Q-Q
    sorted_r = np.sort(rets.values)
    n = len(sorted_r)
    qs = np.linspace(0.5 / n, 1.0 - 0.5 / n, n)
    from scipy.stats import norm  # noqa: E402

    normal_q = norm.ppf(qs)
    sample_q = (sorted_r - rets.mean()) / rets.std(ddof=1)
    fig.add_trace(
        go.Scatter(
            x=normal_q,
            y=sample_q,
            mode="markers",
            marker=dict(color="#1f77b4", size=3),
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    diag = np.array([normal_q.min(), normal_q.max()])
    fig.add_trace(
        go.Scatter(
            x=diag,
            y=diag,
            mode="lines",
            line=dict(color="red", width=1, dash="dash"),
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    from scipy.stats import kurtosis, skew  # noqa: E402

    sk = float(skew(rets.values))
    kt = float(kurtosis(rets.values, fisher=False))  # Pearson kurt (3=normal)
    fig.add_annotation(
        x=0.05,
        y=0.95,
        xref="x2 domain",
        yref="y2 domain",
        text=f"skew={sk:.2f}<br>kurt={kt:.2f}",
        showarrow=False,
        align="left",
    )
    fig.update_xaxes(title_text="Daily return (%)", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_xaxes(title_text="Theoretical normal quantile", row=1, col=2)
    fig.update_yaxes(title_text="Empirical quantile (std)", row=1, col=2)
    fig.update_layout(height=380, margin=dict(l=60, r=20, t=60, b=40))
    return fig


# ── Section C — Robustness / framework figures ────────────────────────────


def _figure_fold_diagnostics(folds: list[dict] | None) -> go.Figure:
    if not folds:
        return go.Figure().add_annotation(text="No fold diagnostics", showarrow=False)
    fold_ids = [f["fold_id"] for f in folds]
    sharpes = [f["sharpe"] for f in folds]
    colors = ["#2ca02c" if s > 0 else "#d62728" for s in sharpes]
    n_positive = sum(1 for s in sharpes if s > 0)
    quorum_pass = n_positive >= 4  # L13: 4-of-5 quorum
    fig = go.Figure(
        data=go.Bar(
            x=[f"F{fid}" for fid in fold_ids],
            y=sharpes,
            marker_color=colors,
            text=[f"{s:+.2f}" for s in sharpes],
            textposition="outside",
        )
    )
    fig.add_hline(y=0, line_color="grey")
    fig.update_layout(
        title=f"WFO fold-by-fold OOS Sharpe — {n_positive}/{len(folds)} positive "
        f"(L13 quorum {'PASS' if quorum_pass else 'FAIL'})",
        height=320,
        yaxis_title="OOS Sharpe",
        margin=dict(l=60, r=20, t=50, b=40),
    )
    return fig


def _figure_sanctuary_divergence(
    historical_sharpes: list[float] | None,
    sanctuary_sharpe: float | None,
    sanctuary_percentile: float | None,
) -> go.Figure:
    if not historical_sharpes:
        return go.Figure().add_annotation(text="No sanctuary divergence data", showarrow=False)
    arr = np.asarray(historical_sharpes)
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(x=arr, nbinsx=40, marker_color="#888", name="Historical 12mo rolling")
    )
    if sanctuary_sharpe is not None:
        fig.add_vline(
            x=sanctuary_sharpe,
            line_color="red",
            line_width=2,
            annotation_text=f"Sanctuary = {sanctuary_sharpe:+.2f}<br>pct = {sanctuary_percentile:.2f}",
            annotation_position="top right",
        )
    fig.update_layout(
        title="Sanctuary divergence (L15)",
        xaxis_title="12-month rolling Sharpe",
        yaxis_title="count",
        height=320,
        margin=dict(l=60, r=20, t=50, b=40),
    )
    return fig


def _figure_mc_fan(
    strat_maxdds: list[float] | None,
    bench_maxdds: list[float] | None,
    rel_ratio_gate: float | None,
    p_better_gate: float | None,
) -> go.Figure:
    """Relative-MC scatter: strategy vs benchmark MaxDD per path.

    Points below the 45° line are paths where strategy did less damage.
    Shaded region marks the L17 pass zone (strategy_maxdd <= ratio_gate * benchmark_maxdd).
    """
    if not strat_maxdds or not bench_maxdds:
        return go.Figure().add_annotation(text="No relative-MC data", showarrow=False)
    strat = np.asarray(strat_maxdds)
    bench = np.asarray(bench_maxdds)
    # Both are negative numbers.
    fig = go.Figure()
    # Diagonal
    lo = float(min(strat.min(), bench.min()))
    hi = 0.0
    fig.add_trace(
        go.Scatter(
            x=[lo, hi],
            y=[lo, hi],
            mode="lines",
            line=dict(color="black", dash="dash"),
            name="strategy = benchmark",
        ),
    )
    # Pass-zone region: strategy_maxdd >= ratio_gate * benchmark_maxdd
    # (both negative; ratio < 1 means strat is less negative)
    if rel_ratio_gate is not None:
        # Shade the area above the strategy_maxdd == ratio*benchmark_maxdd line
        ratio_xs = np.array([lo, hi])
        ratio_ys = ratio_xs * rel_ratio_gate
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([ratio_xs, [hi, lo]]),
                y=np.concatenate([ratio_ys, [hi, hi]]),
                fill="toself",
                fillcolor="rgba(0,200,0,0.10)",
                line=dict(width=0),
                name=f"L17 pass zone (ratio ≤ {rel_ratio_gate})",
            ),
        )
    fig.add_trace(
        go.Scatter(
            x=bench,
            y=strat,
            mode="markers",
            marker=dict(size=5, color="#1f77b4", opacity=0.7),
            name="MC paths",
            hovertemplate="bench MaxDD: %{x:.1%}<br>strat MaxDD: %{y:.1%}<extra></extra>",
        ),
    )
    p_better = float((strat >= bench).mean())
    fig.update_layout(
        title=f"Relative MC scatter (L17) — strategy reduces MaxDD on {p_better:.1%} of paths"
        + (f"  (gate ≥ {p_better_gate:.0%})" if p_better_gate is not None else ""),
        xaxis_title="Benchmark MaxDD",
        yaxis_title="Strategy MaxDD",
        height=420,
        margin=dict(l=60, r=20, t=50, b=40),
        xaxis=dict(tickformat=".0%"),
        yaxis=dict(tickformat=".0%"),
    )
    return fig


def _figure_noise_robustness(
    noise_levels: list[float] | None,
    sharpe_means: list[float] | None,
    sharpe_p5: list[float] | None,
    base_sharpe: float | None,
    max_degradation_gate: float | None,
) -> go.Figure:
    if not noise_levels:
        return go.Figure().add_annotation(text="No noise robustness data", showarrow=False)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=noise_levels,
            y=sharpe_means,
            mode="lines+markers",
            name="mean Sharpe under noise",
            line=dict(color="#1f77b4"),
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=noise_levels,
            y=sharpe_p5,
            mode="lines+markers",
            name="5th pct (worst-case)",
            line=dict(color="#d62728", dash="dot"),
        ),
    )
    if base_sharpe is not None:
        fig.add_hline(
            y=base_sharpe,
            line_color="green",
            line_dash="dot",
            annotation_text=f"Base Sharpe = {base_sharpe:+.2f}",
        )
        if max_degradation_gate is not None:
            floor = base_sharpe * (1.0 - max_degradation_gate)
            fig.add_hline(
                y=floor,
                line_color="orange",
                line_dash="dot",
                annotation_text=f"Gate floor ({1 - max_degradation_gate:.0%} of base)",
            )
    fig.update_layout(
        title="Noise-injection robustness (Varma)",
        xaxis_title="σ (relative)",
        yaxis_title="Sharpe under perturbation",
        height=320,
        margin=dict(l=60, r=20, t=50, b=40),
    )
    return fig


# ── Section D — Sweep figures ─────────────────────────────────────────────


def _figure_sweep_plateau(cells: list[CellSummary], canonical: str) -> go.Figure:
    if not cells:
        return go.Figure().add_annotation(text="No cells", showarrow=False)
    names = [c.cell for c in cells]
    sharpes = [c.sharpe for c in cells]
    colors = ["#1f77b4" if c.cell == canonical else "#888" for c in cells]
    fig = go.Figure(
        data=go.Bar(
            x=names,
            y=sharpes,
            marker_color=colors,
            text=[f"{s:+.2f}" for s in sharpes],
            textposition="outside",
        )
    )
    spread = (max(sharpes) - min(sharpes)) / abs(max(sharpes)) if max(sharpes) > 0 else float("inf")
    fig.update_layout(
        title=f"Cell sweep plateau (V3.2) — spread {spread:.1%} "
        f"{'PASS' if spread < 0.30 else 'FAIL'}",
        yaxis_title="OOS Sharpe",
        height=320,
        margin=dict(l=60, r=20, t=50, b=40),
    )
    return fig


# ── Top-level renderer ────────────────────────────────────────────────────


_TEMPLATE = Template(
    """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>{{ name }} — Audit Dashboard</title>
<style>
  body { font-family: -apple-system, "Segoe UI", Roboto, sans-serif; margin: 0; padding: 20px; background: #fafafa; color: #1a1a1a; }
  h1 { margin-top: 0; }
  h2 { border-bottom: 1px solid #ccc; padding-bottom: 4px; margin-top: 32px; }
  .header-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 12px; }
  .kpi-grid { display: grid; grid-template-columns: repeat(6, 1fr); gap: 12px; margin: 12px 0; }
  .kpi-card { padding: 12px; background: white; border-radius: 6px; box-shadow: 0 1px 2px rgba(0,0,0,0.06); }
  .kpi-label { font-size: 11px; color: #666; text-transform: uppercase; letter-spacing: 0.05em; }
  .kpi-value { font-size: 20px; font-weight: 600; margin-top: 4px; }
  .verdict-deploy { color: #2ca02c; }
  .verdict-conditional { color: #ff9900; }
  .verdict-tier { color: #888; }
  .verdict-suspect { color: #d62728; }
  .verdict-retire { color: #800000; }
  table { border-collapse: collapse; width: 100%; background: white; margin: 12px 0; }
  th, td { padding: 6px 10px; text-align: left; border-bottom: 1px solid #eee; font-size: 13px; }
  th { background: #f0f0f0; font-weight: 600; }
  .meta { font-size: 13px; color: #555; }
  .figure-wrap { background: white; padding: 8px; border-radius: 6px; box-shadow: 0 1px 2px rgba(0,0,0,0.06); margin: 12px 0; }
  .lit { font-style: italic; color: #444; }
</style>
</head>
<body>
<h1>{{ name }} — Audit Dashboard</h1>
<div class="meta">
  <strong>Strategy class:</strong> {{ strategy_class }} &nbsp;|&nbsp;
  <strong>Pre-reg:</strong> <code>{{ pre_reg_directive }}</code><br/>
  <strong>Run date:</strong> {{ run_date }} &nbsp;|&nbsp;
  <strong>Data range:</strong> {{ data_start }} → {{ data_end }} &nbsp;|&nbsp;
  <strong>Sanctuary:</strong> {{ sanctuary_start }} → {{ sanctuary_end }}<br/>
  {% if headline_summary %}<p class="lit">{{ headline_summary }}</p>{% endif %}
</div>

<h2>A. Headline</h2>
<div class="kpi-grid">
  <div class="kpi-card"><div class="kpi-label">Verdict</div>
    <div class="kpi-value verdict-{{ verdict_class }}">{{ verdict }}</div></div>
  <div class="kpi-card"><div class="kpi-label">OOS Sharpe</div>
    <div class="kpi-value">{{ "%+.2f"|format(sharpe) }}</div></div>
  <div class="kpi-card"><div class="kpi-label">CI95 lo</div>
    <div class="kpi-value">{{ "%+.2f"|format(ci_lo) }}</div></div>
  <div class="kpi-card"><div class="kpi-label">Vol (ann)</div>
    <div class="kpi-value">{{ "%.1f%%"|format(vol*100) }}</div></div>
  <div class="kpi-card"><div class="kpi-label">CAGR</div>
    <div class="kpi-value">{{ "%+.1f%%"|format(cagr*100) }}</div></div>
  <div class="kpi-card"><div class="kpi-label">Max DD</div>
    <div class="kpi-value">{{ "%.1f%%"|format(max_dd*100) }}</div></div>
  <div class="kpi-card"><div class="kpi-label">Calmar</div>
    <div class="kpi-value">{{ "%.2f"|format(calmar) }}</div></div>
  <div class="kpi-card"><div class="kpi-label">Hit rate</div>
    <div class="kpi-value">{{ "%.1f%%"|format(hit_rate*100) }}</div></div>
  <div class="kpi-card"><div class="kpi-label">Bars (OOS)</div>
    <div class="kpi-value">{{ n_bars }}</div></div>
  <div class="kpi-card"><div class="kpi-label">DSR-prob</div>
    <div class="kpi-value">{{ "%.2f"|format(dsr_prob) }}</div></div>
  <div class="kpi-card"><div class="kpi-label">Sanc Sharpe</div>
    <div class="kpi-value">{{ "%+.2f"|format(sanctuary_sharpe) }}</div></div>
  <div class="kpi-card"><div class="kpi-label">Sanc pct</div>
    <div class="kpi-value">{{ "%.0f"|format(sanctuary_percentile*100) }}%</div></div>
</div>
<div class="figure-wrap">{{ fig_equity_dd | safe }}</div>
<div class="figure-wrap">{{ fig_rolling_sharpe | safe }}</div>

<h2>B. Return distribution</h2>
<div class="figure-wrap">{{ fig_monthly_heatmap | safe }}</div>
<div class="figure-wrap">{{ fig_distribution | safe }}</div>

<h2>C. Robustness</h2>
<div class="figure-wrap">{{ fig_fold_diag | safe }}</div>
<div class="figure-wrap">{{ fig_sanctuary_div | safe }}</div>
<div class="figure-wrap">{{ fig_mc_fan | safe }}</div>
<div class="figure-wrap">{{ fig_noise | safe }}</div>

<h2>D. Cell sweep + verdict</h2>
<div class="figure-wrap">{{ fig_sweep | safe }}</div>
<table>
  <thead>
    <tr>
      <th>Cell</th><th>Sharpe</th><th>CI95 lo</th><th>CI95 hi</th>
      <th>DSR-prob</th><th>Rel MC ratio</th><th>Rel MC pass</th>
      <th>Sanc Sharpe</th><th>Sanc pct</th><th>Verdict</th>
    </tr>
  </thead>
  <tbody>
  {% for c in cells %}
    <tr>
      <td>{{ c.cell }}</td>
      <td>{{ "%+.4f"|format(c.sharpe) }}</td>
      <td>{{ "%+.3f"|format(c.ci_lo) }}</td>
      <td>{{ "%+.3f"|format(c.ci_hi) }}</td>
      <td>{{ "%.4f"|format(c.dsr_prob) }}</td>
      <td>{{ "%.4f"|format(c.rel_mc_median_ratio) if c.rel_mc_median_ratio is not none else "—" }}</td>
      <td>{{ "PASS" if c.rel_mc_passes else "FAIL" if c.rel_mc_passes is not none else "—" }}</td>
      <td>{{ "%+.4f"|format(c.sanctuary_sharpe) }}</td>
      <td>{{ "%.2f"|format(c.sanctuary_percentile) if c.sanctuary_percentile == c.sanctuary_percentile else "—" }}</td>
      <td class="verdict-{{ c.verdict|lower|replace('_','-')|truncate(20,True,'') }}">{{ c.verdict }}</td>
    </tr>
  {% endfor %}
  </tbody>
</table>

{% if extra_overlay_figure %}
<h2>E. {{ extra_overlay_title or "Strategy-specific overlay" }}</h2>
<div class="figure-wrap">{{ extra_overlay_figure | safe }}</div>
{% endif %}

</body>
</html>
"""
)


def _verdict_class(verdict: str) -> str:
    return {
        "DEPLOY": "deploy",
        "CONDITIONAL_WATCHPOINT": "conditional",
        "TIER_UNCONFIRMED": "tier",
        "SUSPECT": "suspect",
        "RETIRE": "retire",
    }.get(verdict, "tier")


def render_dashboard(result: AuditResult, output_dir: Path) -> Path:
    """Render the canonical audit dashboard to ``output_dir / dashboard.html``.

    Returns the output Path. Idempotent (overwrites).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "dashboard.html"

    canonical_oos = result.cell_oos_returns.get(result.canonical_cell, pd.Series(dtype=float))
    canonical_sanc = result.cell_sanctuary_returns.get(result.canonical_cell)
    canonical_cell_obj = next(
        (c for c in result.cells if c.cell == result.canonical_cell),
        result.cells[0] if result.cells else None,
    )

    kpis = _annualised_metrics(canonical_oos, result.bars_per_year)

    def _div(fig: go.Figure) -> str:
        return pio.to_html(fig, include_plotlyjs="cdn", full_html=False, div_id=None)

    fig_eq_dd = _figure_equity_and_drawdown(
        canonical_oos,
        canonical_sanc,
        result.benchmark_oos_returns,
        result.benchmark_sanctuary_returns,
        result.sanctuary_start,
        full_strategy=result.full_strategy_returns,
        full_benchmark=result.full_benchmark_returns,
        oos_fold_intervals=result.oos_fold_intervals,
    )
    # Prefer the full strategy series for rolling Sharpe / heatmap / distribution
    # so the visualization spans the entire historical run, not just the
    # WFO-stitched OOS slice. Headline KPIs still come from the OOS slice.
    series_for_visuals = (
        result.full_strategy_returns
        if result.full_strategy_returns is not None and not result.full_strategy_returns.empty
        else canonical_oos
    )
    fig_roll_sh = _figure_rolling_sharpe(series_for_visuals, result.bars_per_year)
    fig_monthly = _figure_monthly_heatmap(series_for_visuals)
    fig_distr = _figure_distribution(series_for_visuals)
    fig_folds = _figure_fold_diagnostics(result.fold_diagnostics)
    fig_sanc = _figure_sanctuary_divergence(
        result.sanctuary_historical_window_sharpes,
        result.sanctuary_realised_sharpe,
        result.sanctuary_percentile,
    )
    fig_mc = _figure_mc_fan(
        result.mc_strategy_maxdds,
        result.mc_benchmark_maxdds,
        result.rel_mc_median_ratio_gate,
        result.rel_mc_p_strategy_better_gate,
    )
    fig_noise = _figure_noise_robustness(
        result.noise_levels,
        result.noise_sharpe_means,
        result.noise_sharpe_p5,
        result.noise_base_sharpe,
        result.noise_max_degradation_gate,
    )
    fig_sweep = _figure_sweep_plateau(result.cells, result.canonical_cell)

    extra_html = ""
    if result.extra_overlay_figure is not None:
        extra_html = _div(result.extra_overlay_figure)

    html = _TEMPLATE.render(
        name=result.strategy_name,
        strategy_class=result.strategy_class,
        pre_reg_directive=result.pre_reg_directive,
        run_date=result.run_date,
        data_start=str(result.data_start.date()),
        data_end=str(result.data_end.date()),
        sanctuary_start=str(result.sanctuary_start.date()),
        sanctuary_end=str(result.sanctuary_end.date()),
        headline_summary=result.headline_summary,
        verdict=canonical_cell_obj.verdict if canonical_cell_obj else "UNKNOWN",
        verdict_class=_verdict_class(
            canonical_cell_obj.verdict if canonical_cell_obj else "TIER_UNCONFIRMED"
        ),
        sharpe=canonical_cell_obj.sharpe if canonical_cell_obj else 0.0,
        ci_lo=canonical_cell_obj.ci_lo if canonical_cell_obj else 0.0,
        dsr_prob=canonical_cell_obj.dsr_prob if canonical_cell_obj else 0.0,
        sanctuary_sharpe=canonical_cell_obj.sanctuary_sharpe if canonical_cell_obj else 0.0,
        sanctuary_percentile=canonical_cell_obj.sanctuary_percentile if canonical_cell_obj else 0.0,
        n_bars=len(canonical_oos),
        vol=kpis["vol"],
        cagr=kpis["cagr"],
        max_dd=kpis["max_dd"],
        calmar=kpis["calmar"],
        hit_rate=kpis["hit_rate"],
        cells=result.cells,
        fig_equity_dd=_div(fig_eq_dd),
        fig_rolling_sharpe=_div(fig_roll_sh),
        fig_monthly_heatmap=_div(fig_monthly),
        fig_distribution=_div(fig_distr),
        fig_fold_diag=_div(fig_folds),
        fig_sanctuary_div=_div(fig_sanc),
        fig_mc_fan=_div(fig_mc),
        fig_noise=_div(fig_noise),
        fig_sweep=_div(fig_sweep),
        extra_overlay_figure=extra_html,
        extra_overlay_title=result.extra_overlay_title,
    )

    out.write_text(html, encoding="utf-8")
    return out
