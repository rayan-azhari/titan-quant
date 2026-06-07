# A. Glossary

Plain-language definitions of the terms used throughout the book. This is a reference, not
a chapter; read it sideways. Each entry says what a term *measures* and points to the
chapter where it is earned, not merely used. Where a term carries a number in live practice,
we describe the quantity, never a deployable value; any figure here is **illustrative** and
labelled as such.

!!! tip "How to read this"
    Definitions are deliberately short: a sentence or two and a pointer. The "Cross-ref"
    column links to the chapter that develops the term. For the formulas, the gates each
    metric enforces, and the bug that bought each rule, follow the link rather than trusting
    the one-liner here.

## Performance & risk metrics

These all interrogate the *same* return series from different angles, and the standing rule
of the book is that no single one is the verdict; see
[Beyond Sharpe: the metric suite](../part2-research/metric-suite.md). Reach for the metric
that answers the question you actually have: *how volatile?* (Sharpe), *how painful the
path?* (Calmar, MaxDD), *how bad the tail?* (CVaR, CDaR), *will it survive?* (risk of ruin).

| Term | Definition | Cross-ref |
|---|---|---|
| **CAGR** | Compound Annual Growth Rate, the constant yearly rate that compounds start equity into end equity. Geometric, so it reflects the *path*, not the arithmetic average of returns. | [Metric suite](../part2-research/metric-suite.md) |
| **Calmar ratio** | CAGR divided by maximum drawdown: return per unit of *worst* pain. Titan promotes a strategy on **Calmar lift**, not Sharpe lift, because drawdown is what gets a strategy switched off mid-trough. | [Metric suite](../part2-research/metric-suite.md) |
| **CDaR** | Conditional Drawdown at Risk, the *average* of the worst drawdowns beyond a chosen quantile (e.g. the mean of the deepest 5% of drawdown days). The drawdown analogue of CVaR: it speaks to how bad the bad path gets, not just how often. | [Tail risk & risk of ruin](../part2-research/tail-risk-and-ruin.md) |
| **CVaR** | Conditional Value at Risk (a.k.a. Expected Shortfall), the average loss *given* you are in the worst tail (e.g. the mean of the worst 5% of returns). Unlike VaR, it sizes the tail rather than just marking its threshold. | [Tail risk & risk of ruin](../part2-research/tail-risk-and-ruin.md) |
| **Drawdown / MaxDD** | The peak-to-trough decline of an equity curve; **maximum drawdown** is the worst such decline over the sample. The path metric a human actually *feels* while holding the strategy, and the one that gets it killed. | [Metric suite](../part2-research/metric-suite.md) |
| **Risk of ruin** | The probability that equity falls below a fatal threshold (a margin call, a mandate limit, or zero) at the *deployed* position size. A survival probability, not a smoothness measure. | [Tail risk & risk of ruin](../part2-research/tail-risk-and-ruin.md) |
| **Sharpe ratio** | Mean excess return divided by its standard deviation, scaled to a year. Return per unit of *total* volatility: the entry point to evaluation, never the verdict, because it is blind to path, tail, and the *sign* of the volatility it penalises. | [A backtest you can trust](../part2-research/backtest-you-can-trust.md) |
| **Sortino ratio** | Like Sharpe, but divides by **downside** deviation only, so an upside spike is not penalised as if it were a loss. The right ratio when the return distribution is asymmetric. | [Metric suite](../part2-research/metric-suite.md) |
| **VaR** | Value at Risk: the loss threshold a given tail quantile sits at (e.g. "5% of days lose at least *X*"). Useful but treacherous alone: it marks where the tail *starts* and says nothing about how deep it goes. Prefer **CVaR**. | [Tail risk & risk of ruin](../part2-research/tail-risk-and-ruin.md) |

!!! warning "Annualisation is a correctness property, not a convention"
    Every ratio above is a per-period quantity scaled by `sqrt(periods_per_year)`. Get the
    bar frequency wrong (treating hourly P&L as daily, say) and the number is off by a
    fixed factor, always in the flattering direction (`sqrt(24)` ≈ 4.9× for that example).
    A single backtest in Titan's own history reported a phantom Sharpe near 4 this way; the
    honest, daily-annualised figure was ordinary. State the timeframe at the top of every
    report. See [the five lies](../part2-research/backtest-you-can-trust.md).

## Validation & research-process terms

The machinery that turns a backtest number into a trustworthy one. The leitmotif is
*suspicion over celebration*: an un-audited number is assumed inflated until proven
otherwise, because the errors that survive your attention are the optimistic ones.

| Term | Definition | Cross-ref |
|---|---|---|
| **Bootstrap CI** | A confidence interval for a metric, built by resampling the return series many times and reading the percentiles. Titan gates on the **95% lower bound** of the Sharpe: `CI_lo ≤ 0` ⇒ `unconfirmed`, regardless of how good the point estimate looks. | [A backtest you can trust](../part2-research/backtest-you-can-trust.md) |
| **Block bootstrap** | A bootstrap that resamples *blocks* of consecutive bars (stationary / Politis-Romano), preserving the autocorrelation that trend and carry strategies live on. The IID bootstrap shuffles bars independently, destroying that structure and biasing the lower bound *upward*: the exact number you gate on. | [A backtest you can trust](../part2-research/backtest-you-can-trust.md) |
| **DSR** | Deflated Sharpe Ratio: a Sharpe corrected for the *number of configurations you explicitly tried*, since the best of many random trials looks good by chance alone. Controls error on the *selection* of a winner; the bootstrap CI controls error on the winner's number. You need both. | [Beating your own optimizer](../part2-research/deflated-sharpe.md) |
| **IS / OOS** | In-sample / out-of-sample. IS is the data a model is fit on; OOS is data held back to test generalisation. The split is meaningless if normalisation statistics (or the researcher's eyes) leak across it. | [Walk-forward](../part2-research/walk-forward.md) |
| **Look-ahead bias** | Using information at bar *t* that was not actually available until later (e.g. same-bar `position * return`, or a full-series z-score). The most expensive and most common backtest lie, and the hardest to spot because it reads as ordinary code. | [Failure-mode catalogue](../part2-research/failure-mode-catalogue.md) |
| **Sanctuary** | A held-out window of recent history the research loop is structurally forbidden to see, spent **once**, last, as final validation. Where DSR penalises *explicit* trials, the sanctuary catches overfitting to the *process*, including the researcher. | [Sanctuary & decision matrix](../part2-research/sanctuary-decision-matrix.md) |
| **Shift discipline** | The rule that a decision and the return it earns must live in disjoint time windows, decision first: `asset_returns * position.shift(1)`. The mechanical fix for look-ahead. | [A backtest you can trust](../part2-research/backtest-you-can-trust.md) |
| **WFO** | Walk-Forward Optimisation: repeatedly fit on an IS window, test on the next OOS window, then roll forward. Approximates how a strategy would actually have been re-tuned and traded through time, rather than fit once with hindsight. | [Walk-forward](../part2-research/walk-forward.md) |

## Sizing terms

How a *confirmed* edge becomes a position size without betting the account. The owning
chapter is [Position sizing: Kelly, fractional Kelly & vol-targeting](../part5-portfolio-risk/position-sizing-kelly.md).

| Term | Definition | Cross-ref |
|---|---|---|
| **Kelly criterion** | The bet fraction that maximises the long-run *geometric* growth rate of capital. Mathematically optimal for growth, but it assumes you know your edge exactly, which you never do, so full Kelly is reckless in practice. | [Position sizing](../part5-portfolio-risk/position-sizing-kelly.md) |
| **Fractional Kelly** | Betting a fixed fraction (a half or a quarter, say) of the full-Kelly stake. Trades a little growth for a large reduction in drawdown and in sensitivity to a mis-estimated edge: the form anyone should actually deploy. | [Position sizing](../part5-portfolio-risk/position-sizing-kelly.md) |
| **Vol targeting** | Sizing a position so its *expected* volatility hits a chosen budget (scale up in calm markets, down in turbulent ones), keeping risk contribution roughly constant across regimes instead of letting it balloon in a crisis. | [Position sizing](../part5-portfolio-risk/position-sizing-kelly.md) |

## Portfolio & allocation terms

Combining strategies so the book is more than the sum of its sleeves. See
[The Allocator & the correlation dial](../part5-portfolio-risk/allocator-correlation-dial.md).

| Term | Definition | Cross-ref |
|---|---|---|
| **Inverse-vol weighting** | Allocating capital in inverse proportion to each strategy's volatility, so quieter strategies carry more notional and each contributes comparable *risk*. Cheap, transparent, hard to break; Titan's default allocator. Its one blind spot: it ignores correlation. | [Allocator & correlation dial](../part5-portfolio-risk/allocator-correlation-dial.md) |
| **ERC** | Equal Risk Contribution (risk parity): weights chosen so every strategy contributes the *same* amount to total portfolio risk, accounting for correlations via the full covariance matrix. Inverse-vol's correlation-aware cousin; the right answer at larger scale. | [Allocator & correlation dial](../part5-portfolio-risk/allocator-correlation-dial.md) |
| **Correlation dial** | A single leverage scalar that de-grosses the *whole* book when its strategies start moving together (diversification evaporating). Fails safe to exactly `1.0` (byte-identical to plain inverse-vol when off), so a broken governor does nothing rather than something wrong. | [Allocator & correlation dial](../part5-portfolio-risk/allocator-correlation-dial.md) |
| **Per-strategy equity** | Each strategy tracks its *own* notional equity rather than sharing one account balance, so sizing, attribution, and halts are isolated and deterministic; one sleeve's loss can't silently resize another. | [Per-strategy equity & currency](../part5-portfolio-risk/per-strategy-equity-fx.md) |

## Live-safety terms

The graded ladder that keeps a live account from following a backtest off a cliff. See
[The Portfolio Risk Manager](../part5-portfolio-risk/portfolio-risk-manager.md) and
[Layered safety](../part5-portfolio-risk/layered-safety.md).

| Term | Definition | Cross-ref |
|---|---|---|
| **HWM** | High-Water Mark: the highest equity reached so far. Drawdown is measured from it, and de-risk thresholds (soft limits, halts) are expressed as a percentage decline below it. | [Portfolio Risk Manager](../part5-portfolio-risk/portfolio-risk-manager.md) |
| **Halt** | A safety state that stops a strategy (or the whole book) from opening new risk once a drawdown or loss threshold is breached. Persisted across restarts, so a crash-and-reboot can't silently un-halt and let the book trade through a breach. | [Portfolio Risk Manager](../part5-portfolio-risk/portfolio-risk-manager.md) |
| **De-risk ladder** | A *graded* response to losses (trim, then halt new entries, then flatten) rather than one all-or-nothing kill switch, so ordinary volatility doesn't trip the most drastic action. | [Layered safety](../part5-portfolio-risk/layered-safety.md) |
| **Kill switch** | The terminal rung: stop everything and flatten. Reserved for the worst breach; the ladder exists precisely so this is rarely the first thing to fire. | [Layered safety](../part5-portfolio-risk/layered-safety.md) |

## Instrument & currency terms

The boring details that mis-size a leg by a third if you get them wrong. See
[Per-strategy equity & deterministic currency](../part5-portfolio-risk/per-strategy-equity-fx.md)
and [Sourcing & storage](../part3-data/sourcing-storage.md).

| Term | Definition | Cross-ref |
|---|---|---|
| **Base vs quote currency** | In a pair like `EUR/USD`, the **base** is the first currency (EUR, the thing being priced) and the **quote** is the second (USD, the price). A P&L or a position size computed in the wrong one is off by the exchange rate, silently. | [Per-strategy equity & currency](../part5-portfolio-risk/per-strategy-equity-fx.md) |
| **Total-return vs price-only** | A *price-only* series ignores dividends, coupons, and distributions; a *total-return* series reinvests them. Backtesting a distributing instrument on price-only data understates the edge; and mixing the two across IS/OOS corrupts the comparison. | [Sourcing & storage](../part3-data/sourcing-storage.md) |
| **UCITS** | An EU-regulated, broadly-marketable collective-fund wrapper (most European ETFs are UCITS). It governs availability, tax treatment, and which ticker line an exposure trades under: e.g. a USD-quoted Treasury UCITS ETF rather than its US-listed cousin. | [Broker realities](../part4-research-to-prod/broker-realities.md) |
| **PRIIPs** | The EU disclosure regime (the standardised **KID** document) for packaged retail products. A practical gatekeeper: a product without a compliant KID may be un-buyable on a given retail account, no matter how good the backtest. | [Broker realities](../part4-research-to-prod/broker-realities.md) |

!!! danger "War-story: the leg sized by a third on a currency assumption"
    The book's most-cited bug: a multi-currency sleeve sized a position as if its P&L were
    in the account base currency when the instrument settled in another, putting the leg
    on at roughly two-thirds of intended size: no exception, just a quietly wrong number.
    It bought the rule that **currency conversion is explicit and deterministic at the
    point of sizing**, never inferred from the account default. Full telling:
    [Per-strategy equity & deterministic currency](../part5-portfolio-risk/per-strategy-equity-fx.md).

## Strategy-class terms

| Term | Definition | Cross-ref |
|---|---|---|
| **Edge / alpha** | A repeatable statistical reason a strategy makes money: a hypothesis about *why* a return should exist, stated *before* it is measured, not a pattern discovered after the fact and rationalised. | [Thinking in edges](../part2-research/typology.md) |
| **Integration contract** | The fixed interface every strategy class implements, so the portfolio, risk, and execution layers can treat any strategy identically. It is what makes "live == research" enforceable rather than aspirational. | [The strategy class & the integration contract](../part4-research-to-prod/strategy-class-contract.md) |
| **`unconfirmed`** | The status of any candidate that fails a gate: `CI_lo ≤ 0`, a blown sanctuary, a collapsed DSR. It cannot enter a default deployment registry, regardless of how good the point estimate looks. The book's single most-used word. | [A backtest you can trust](../part2-research/backtest-you-can-trust.md) |

---

For the interface these terms plug into, see the
[Integration-contract reference card](integration-contract.md) and the
[Research pre-flight checklist](preflight-checklist.md). For where each of these went
wrong in practice (and the rule the bug bought), the
[failure-mode catalogue](../part2-research/failure-mode-catalogue.md) is the deepest
single source.
