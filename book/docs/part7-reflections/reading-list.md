# 27. Reading list & references

Every rule in this book was bought twice: once by a bug in a live or backtested system, and once by someone who had already written down *why* the bug happens and how to avoid it. The war-stories are ours. The theory mostly isn't; it comes from a fairly small canon of papers and books that practitioners keep rediscovering the hard way.

This chapter is that canon, annotated. It is not a bibliography for completeness; it is a *working* reading list. For each entry we say, in a line or two, **why it matters** and **which chapter it backs**, so you can read the source behind any decision the book makes. The principle here is simple and worth stating: a methodology you can't trace to a citation is a methodology you can't defend in an audit. When a reviewer asks "why deflate at this `N`?" or "why a *block* bootstrap?", the honest answer is a reference, not a preference.

A note on how to read these. The book's [decision matrix](../part2-research/sanctuary-decision-matrix.md) is, viewed from one angle, just an engineering assembly of four ideas from this list: overfitting correction (Bailey & López de Prado), serially-aware resampling (Politis & Romano), growth-optimal-but-fractional sizing (Kelly / Thorp / MacLean-Thorp-Ziemba), and the fact that a Sharpe is an estimate with a standard error (Lo). If you read only four things below, read those. Everything else sharpens the edges.

!!! note "On editions, years, and links"
    We cite by author and the year of the canonical version; specific editions and DOIs drift, and this is a public document, not a citation manager. Where a work has a well-known later edition (Carver's books, Thorp's memoir), read the latest. Years are the ones the field uses colloquially (e.g. "Kelly 1956", "Bailey-López de Prado 2014"); verify against the publisher before quoting in your own audit trail.

---

## Theme 1: Systematic trading & portfolio construction

This is the "how to think about a whole book of strategies" layer. It backs most of Part IV, all of Part V, and the diversification logic in the [Allocator](../part5-portfolio-risk/allocator-correlation-dial.md).

- **Robert Carver, *Systematic Trading* (2015).** The single best on-ramp to running a *rules-based* book the way Titan does: forecasts scaled to a volatility target, diversification as the only free lunch, and the discipline of not overriding your own system. Backs the strategy-class contract ([Ch. 15](../part4-research-to-prod/strategy-class-contract.md)) and the vol-targeting half of [position sizing](../part5-portfolio-risk/position-sizing-kelly.md). His insistence that *more uncorrelated markets beats a better signal* is the intellectual root of our [correlation dial](../part5-portfolio-risk/allocator-correlation-dial.md).

- **Robert Carver, *Leveraged Trading* (2019) and *Advanced Futures Trading Strategies* (2023).** The practical companions. *Leveraged Trading* is the gentlest correct treatment of position sizing under leverage we know; *Advanced Futures* is where Carver makes the case, with data, that a trend edge needs **35+ markets across asset classes** to be robust. That single empirical claim is *why* our own FX-only and basket trend experiments were rejected, and it is referenced directly in the [Caveats](caveats-open-problems.md) chapter.

- **Andreas Clenow, *Following the Trend* (2013) and *Stocks on the Move* (2015).** A clean, code-adjacent demonstration that diversified trend-following is mostly *risk management plus patience*, and that a momentum equity book lives or dies on its rebalancing and survivorship handling. Backs the survivorship warnings in the [data-quality gate](../part3-data/data-quality-gate.md) and the "your edge is the exits" framing in the [typology](../part2-research/typology.md) chapter.

- **Antti Ilmanen, *Expected Returns* (2011).** The reference for *where* premia actually come from (value, carry, trend, liquidity, volatility) and why each is compensated. We reach for it whenever we need to justify that a candidate edge is a *known risk premium harvested cleanly* versus a statistical ghost. Backs [Thinking in edges](../part2-research/typology.md): a hypothesis should name its premium before it earns a backtest.

- **Marcos López de Prado, *Advances in Financial Machine Learning* (2018).** Beyond the DSR work below, this is the source for purged/embargoed cross-validation, the dangers of overlapping labels, and why naive k-fold on time series leaks. Backs [walk-forward](../part2-research/walk-forward.md) and the look-ahead discipline throughout [a backtest you can trust](../part2-research/backtest-you-can-trust.md). Read it suspiciously, some methods are heavier than a small book needs, but the diagnosis of *why* time-series ML overfits is exactly right.

---

## Theme 2: Backtest rigour, overfitting & deflation

The Part II core. This is the layer that turns "a Sharpe of 1.3" into "a Sharpe of 1.3 that survived `N` trials, a serially-aware CI, and a held-out year."

- **David Bailey & Marcos López de Prado, "The Deflated Sharpe Ratio" (2014), and "The Probability of Backtest Overfitting" (Bailey, Borwein, López de Prado, Zhu, 2017).** The formal account of the lie at the heart of every parameter sweep: the maximum of `N` noisy Sharpes drifts up even when every cell has zero true edge. The DSR gives the expected null maximum in closed form and converts the gap into `P(true Sharpe > 0)`. This is the entire backbone of [Beating your own optimiser](../part2-research/deflated-sharpe.md), and the source of the rule that **`N` is the candidate pool, not the survivors**.

- **David Bailey, Jonathan Borwein, Marcos López de Prado & Qiji Zhu, "Pseudo-Mathematics and Financial Charlatanism" (2014).** The accessible, polemical version of the same result, aimed squarely at published backtests. The takeaway it cements: *given enough trials, any target Sharpe is reachable on noise alone.* We hand this paper to anyone who thinks a five-decimal in-sample Sharpe is evidence of anything. Backs the suspicion-over-celebration tone of the whole book and the multiple-testing warnings in [the failure-mode catalogue](../part2-research/failure-mode-catalogue.md).

- **Campbell Harvey, Yan Liu & Heqing Zhu, "…and the Cross-Section of Expected Returns" (2016), and Harvey & Liu, "Backtesting" (2015).** The empirical scale of the problem: with hundreds of published "factors," a t-stat of 2 is nowhere near enough; the multiple-testing-adjusted hurdle is closer to 3. This is the macro-justification for our gates being *strict* and for treating any externally-sourced strategy as guilty until re-validated. Backs [walk-forward](../part2-research/walk-forward.md) and the rejection log referenced in [Caveats](caveats-open-problems.md).

- **Andrew Lo, "The Statistics of Sharpe Ratios" (2002).** The often-skipped foundation: a Sharpe ratio is an *estimator* with a standard error, and that error inflates badly under autocorrelation (and so does naive annualisation of a serially-correlated series). This is *why* we never report a Sharpe without an interval and never annualise without stating the frequency. Backs the "a Sharpe is an estimate, not a fact" section of [a backtest you can trust](../part2-research/backtest-you-can-trust.md) and the skew/kurtosis sensitivity used in our DSR implementation.

!!! tip "The two-paper minimum for backtest rigour"
    If a colleague will read exactly two things before reviewing a sweep, make them Bailey-López de Prado (2014) on the DSR and Lo (2002) on the Sharpe's standard error. Together they explain *both* leaks the book gates on: the error from **selecting** a number out of many (DSR), and the error on **the number itself** (the bootstrap CI). The book applies them in that order, never one instead of the other; see the gate diagram in [Ch. 9](../part2-research/deflated-sharpe.md).

---

## Theme 3: Sizing, growth & the cost of ruin

The Part V theory. This layer answers "given an edge I believe in, how *much* do I bet?", and its dominant message is restraint.

- **John Kelly, "A New Interpretation of Information Rate" (1956).** The origin of the growth-optimal bet fraction. The result that matters for us: the bet size that maximises the *long-run geometric growth rate* is finite and computable from the edge and its variance (for a continuous edge, `f* = μ / σ²`). Backs the leverage framing in [position sizing](../part5-portfolio-risk/position-sizing-kelly.md): Kelly is a *leverage*, not a capital weight, a distinction the chapter labours because confusing the two mis-sizes the whole book.

- **Leonard MacLean, Edward Thorp & William Ziemba, *The Kelly Capital Growth Investment Criterion* (2010), and MacLean-Thorp-Ziemba, "Good and Bad Properties of the Kelly Criterion" (2010).** The grown-up treatment: full Kelly is growth-optimal *in the limit* but has brutal interim drawdowns, and any estimation error in `μ` pushes you past optimal into ruinous territory. This is the literature behind our **fractional-Kelly** default (a small fraction of full Kelly; Titan's value is a conservative placeholder here, not the deployed one) and the rule that we size on a *lower bound* of the edge, never the point estimate. The drawdown half of the argument is also why our promotion gate is **Calmar / CDaR**, not Sharpe; see [Beyond Sharpe: the metric suite](../part2-research/metric-suite.md). Backs [position sizing](../part5-portfolio-risk/position-sizing-kelly.md) and the survival logic in [tail risk & risk of ruin](../part2-research/tail-risk-and-ruin.md).

- **Edward Thorp, "The Kelly Criterion in Blackjack, Sports Betting, and the Stock Market" (2006), and the memoir *A Man for All Markets* (2017).** The practitioner's view from someone who actually compounded capital under Kelly for decades. Thorp's lived conclusion, *bet a fraction, because you never know your edge as precisely as your formula pretends*, is the single most important sizing intuition in the book. Backs the fractional-Kelly default and the "estimation error dominates" warning in [position sizing](../part5-portfolio-risk/position-sizing-kelly.md).

- **Sid Browne & William Ziemba (and the broader drawdown-control literature).** For readers who want the bridge from growth-optimality to *drawdown-constrained* growth: maximising growth subject to a cap on the probability of a large loss. This is the formal cousin of our Monte-Carlo `P(MaxDD > x)` gate and the graded de-risk ladder. Backs [tail risk & risk of ruin](../part2-research/tail-risk-and-ruin.md) and [layered safety](../part5-portfolio-risk/layered-safety.md).

!!! danger "Why this theme carries a danger admonition, not a warning"
    Everything in Themes 1 to 2 corrupts a *measurement*. This theme touches *capital*. The recurring lesson across Kelly, Thorp, and MacLean-Thorp-Ziemba is that the failure mode of sizing is not "leaving return on the table"; it is **ruin**, and ruin is irreversible. A backtest you can re-run; a blown-up account you cannot. That asymmetry is why the book sizes on a fraction of a *lower-bounded* edge and why [position sizing](../part5-portfolio-risk/position-sizing-kelly.md) gates on a positive CI lower bound rather than a leverage target.

---

## Theme 4: Resampling, the bootstrap & serial dependence

The narrow-but-critical layer behind every confidence interval in the book.

- **Bradley Efron & Robert Tibshirani, *An Introduction to the Bootstrap* (1993).** The foundation: how to turn one sample into an empirical sampling distribution by resampling. Read this first if "bootstrap CI" is a phrase you've used without deriving. Backs the error-bars discipline in [a backtest you can trust](../part2-research/backtest-you-can-trust.md).

- **Dimitris Politis & Joseph Romano, "The Stationary Bootstrap" (1994).** The fix for the bug that bit us directly. The naive IID bootstrap resamples individual bars, **destroying** the autocorrelation that trend and carry strategies live on, which artificially *narrows* the interval and biases the lower bound *upward*, exactly the optimism the CI is supposed to catch. The stationary block bootstrap resamples *blocks* of consecutive bars (with a geometric-mean block length matched to the strategy's dependence) so serial structure survives. This is *the* citation behind our `block_size` argument and the rule **never accept the IID default for a serially-correlated strategy**. Backs the bootstrap war-story in [a backtest you can trust](../part2-research/backtest-you-can-trust.md).

- **Halbert White, "A Reality Check for Data Snooping" (2000), and Romano & Wolf, "Stepwise Multiple Testing as Formalized Data Snooping" (2005).** The resampling-based approach to the *multiple-comparisons* problem: bootstrapping the distribution of the *best* strategy across a universe to ask whether the winner beats what data-snooping alone would produce. A complementary, distribution-free angle on the same concern the DSR addresses parametrically. Backs the honest-`N` discipline in [Beating your own optimiser](../part2-research/deflated-sharpe.md).

!!! warning "The resampling mistake that survives review"
    Three of the entries above exist on this list because the same error is so easy to ship: applying an **IID** resampling assumption to a **serially dependent** series. It passes code review because the call looks like every textbook bootstrap. The damage is always in the same direction: a too-tight interval, an over-optimistic lower bound, a strategy that clears a gate it should have failed. If you take one operational rule from Theme 4: match the resampling unit to the dependence structure of the data, and if you're unsure, prefer the block bootstrap. Politis & Romano (1994) is the why.

---

## Theme 5: Operations, microstructure & the gap between research and live

The Part IV / Part VI layer: the literature on why a clean backtest still loses money when it meets a real exchange.

- **Larry Harris, *Trading and Exchanges* (2003).** The reference for market microstructure: order types, the bid-ask spread as a real cost, who the liquidity providers are, and why your fill is not the mid. Backs [Broker realities](../part4-research-to-prod/broker-realities.md) and the cost-aware re-validation that decides whether a paper edge survives real financing and spread.

- **Robert Kissell, *The Science of Algorithmic Trading and Portfolio Management* (2013).** The quantitative treatment of *implementation shortfall* and market impact: how the difference between decision price and execution price scales with size. Backs the slippage assumptions in [live-equals-research](../part4-research-to-prod/live-equals-research.md): if your live fills systematically lag your backtest's assumed prices, this is the framework for measuring how badly.

- **Site Reliability Engineering, Beyer et al. (Google, 2016).** Not a trading book, deliberately. The discipline of error budgets, graceful degradation, runbooks, and "every alert must be actionable" maps directly onto operating a live trading stack. Backs the [live runbook](../part6-deploy-ops/live-runbook.md) and the kill-switch and de-risk-ladder thinking in [layered safety](../part5-portfolio-risk/layered-safety.md): a halted strategy that persists its halt state is an SRE idea before it is a trading one.

---

## How to use this list

A reading list that is read once and shelved has failed. Treat this one as an **index into your own audit trail**:

- When you write a gate, cite the paper that justifies its threshold in the code comment. "DSR ≥ 0.95 (Bailey & López de Prado 2014)" ages far better than a bare magic number.
- When you reject a strategy, the rejection note should name the *principle* it violated and the source of that principle. Most of our rejections trace to Theme 2 (overfitting) or Theme 4 (under-estimated CI), and saying which makes the rejection reproducible.
- When a number surprises you, find the entry above that predicts it. A Sharpe that quadruples when you change the bar frequency is Lo (2002). A screener winner that evaporates under deflation is Bailey-López de Prado (2014). A CI that was too tight is Politis & Romano (1994). The literature has usually seen your bug before.

!!! tip "The minimum viable canon"
    If you build nothing else from this book but want the *defensible* core: Carver (*Systematic Trading*) for the architecture, Bailey-López de Prado (2014) for deflation, Lo (2002) for the Sharpe's error bars, Politis & Romano (1994) for the bootstrap, and MacLean-Thorp-Ziemba (2010) for sizing. Those five back the load-bearing decisions in Parts II and V. Everything else in this list makes a specific chapter sharper, not the framework sounder.

## Takeaways

- **A methodology you can't cite is one you can't defend.** Every gate, threshold, and rejection in this book traces to a source above; in an audit, the citation *is* the justification.
- **Read four things first:** Bailey-López de Prado (2014) on deflation, Lo (2002) on the Sharpe's standard error, Politis & Romano (1994) on the block bootstrap, and Kelly/Thorp/MacLean-Thorp-Ziemba on fractional sizing. They back the decision matrix's four binding ideas.
- **The literature is mostly about restraint.** Overfitting correction, serially-aware CIs, and fractional Kelly all push the same direction: your real edge is smaller, noisier, and riskier to size than your first measurement claims. The canon exists to keep you honest about that.
- **Match the tool to the structure.** IID bootstrap for IID data, block bootstrap for serial dependence; per-bar Sharpe for dense strategies, per-trade for sparse ones. Most of the war-stories in this book are a mismatch the relevant source would have caught.

---

This is the last technical chapter. For the honest accounting of what remains *unconfirmed*, contested, or would be rebuilt from scratch, see [Caveats, open problems & what we'd do differently](caveats-open-problems.md). For quick definitions of any term used above, the [glossary](../appendix/glossary.md) collects them; the [integration contract](../appendix/integration-contract.md) and [pre-flight checklist](../appendix/preflight-checklist.md) turn the principles into operational artifacts you can run before any strategy touches capital.
