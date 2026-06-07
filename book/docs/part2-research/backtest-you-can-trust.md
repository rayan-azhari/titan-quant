# 6. A backtest you can trust

A backtest is a measurement. And like any measurement, it is worthless, worse than worthless, because it's *confident*, unless you know its units, whether the instrument was wired up correctly, and how big the error bars are. Most backtests fail all three checks silently, and the failure always points the same direction: it makes the strategy look better than it is.

This chapter is about closing those leaks before any of the heavier machinery (walk-forward, deflation, Monte Carlo) even runs. Get this layer wrong and everything downstream inherits the lie. Get it right and the rest of Part II is just turning up the rigour.

We'll go through the five ways a backtest lies to you, the single discipline that fixes each, why each fix works, and how Titan bakes those disciplines into one shared module so they can't be quietly skipped. Then, because Sharpe is only the *first* number, we'll look at the rest of the suite you actually need.

## The five lies

| # | The lie | What it looks like | The fix |
|---|---|---|---|
| 1 | **Wrong units** | A Sharpe computed on hourly bars, annualised as if daily | An explicit `periods_per_year` on every call |
| 2 | **Survivor math** | Dropping flat bars before annualising | Never filter `returns != 0` before a Sharpe |
| 3 | **Peeking** | The signal at bar *t* uses information from bar *t* | Shift discipline: trade *t*'s signal on *t+1*'s return |
| 4 | **Future-normalised** | A z-score computed over the whole series | Causal (rolling / expanding) or IS-frozen normalisation |
| 5 | **No error bars** | A single Sharpe number, reported to two decimals | A bootstrap confidence interval, gated on the lower bound |

None of these is exotic. Every one of them has shipped to production in real systems, including Titan, and every one is invisible unless you go looking. Let's take them in order, and notice that **each lie flatters the strategy.** That asymmetry is the whole reason for suspicion: errors in a backtest are not random, they are *biased toward optimism*, because the optimistic versions are the ones that survive your attention and get deployed.

## Lie 1, Units: annualisation is a correctness property, not a convention

The Sharpe ratio is a per-period quantity scaled to a year. The scaling factor is the number of bars in a trading year, and it depends entirely on your bar size:

```python
BARS_PER_YEAR = {
    "D":  252,            # daily
    "H4": 252 * 6,        # 6 four-hour bars per 24h FX day
    "H1": 252 * 24,       # = 6048
    "M5": 252 * 24 * 12,  # = 72576
}
```

Why does the factor matter so much? Because Sharpe scales with the *square root* of the number of periods. Annualising per-bar Sharpe means multiplying by `sqrt(periods_per_year)`. Get the period count wrong by a factor of 24 (treating hourly data as daily) and your annualised number is off by `sqrt(24)` ≈ **4.9×**. Nobody makes that error in the direction that makes the strategy look *worse*. The dangerous version is subtle: a pipeline that resamples mid-way and then annualises with the wrong frequency's factor.

!!! warning "War-story: the H1 strategy that was secretly daily"
    A backtest aggregated hourly signals into a once-a-day position, producing a P&L series that was, in substance, **daily**. But the Sharpe helper was still being handed the H1 annualisation factor (`252 * 24`). The strategy reported a Sharpe near 4 and looked like a licence to print money. The real, daily-annualised figure was around 0.8: a fine-but-ordinary strategy. The `sqrt(24)` ≈ 4.9× phantom came entirely from annualising a daily series as if it were hourly. The fix wasn't cleverness; it was making the frequency *impossible to leave implicit*.

The fix is a rule, not vigilance:

!!! tip "Make the annualisation factor a required argument"
    Titan's shared metrics module refuses to compute a Sharpe without being told the frequency. There is no default.

    ```python
    def sharpe(returns, periods_per_year: int) -> float: ...
    ```

    A missing default looks like a papercut. It's actually the cheapest correctness gate in the system: it forces every caller to *state, at the call site,* what frequency the P&L is, which means a reviewer can verify it in one glance. The first line of any backtest output should be the bar timeframe. If you can't state it, stop.

## Lie 2, Survivor math: don't filter the flat bars

A tempting "cleanup" looks innocent:

```python
# WRONG: makes a selective strategy look far better than it is
active = returns[returns != 0.0]
sr = sharpe(active, periods_per_year=252)
```

A strategy that's in the market only a fraction `a` of the time has its Sharpe inflated by `1/sqrt(a)` when you drop the flat bars and then annualise with the full-year factor. The intuition: removing the zeros leaves the *same* mean-per-active-bar but a *smaller* standard deviation per bar (you deleted the calm days), and you still scale by the full year's bar count. A strategy that trades one day in four (`a = 0.25`) gets a free **2×** to its reported Sharpe. The flat days are not noise to be cleaned; they are information about the strategy's *selectivity*, and the annualisation factor already accounts for them.

If you genuinely want per-trade statistics (and for sparse strategies you often should), that's a different measurement with its own helper (`trade_sharpe`), fed a per-trade P&L series, not a zero-filtered bar series. The point is to make the choice explicit, never to silently delete inconvenient zeros.

!!! warning "This is the single most common Sharpe inflation"
    It survives code review because the filter reads as hygiene. Titan's rule: **no Sharpe function filters `returns != 0` internally**, and reviewers reject any caller that does it by hand. If the strategy is sparse, that's a fact the Sharpe should reflect, not one the metric should hide.

## Lie 3, Peeking: shift discipline

This is the one that has cost the most, across the most systems. The signal you compute at the close of bar *t* can only be acted on *after* bar *t*: so it earns the return from *t* to *t+1*, not the return *into* *t*. The mental model: **a decision and the return it earns must live in disjoint time windows, decision first.** Written as code, the only safe pattern is:

```python
# A position decided at the close of t earns t -> t+1:
strat_returns = asset_returns * position.shift(1).fillna(0.0)
```

The bug, "same-bar collect", looks like this, and it is everywhere in regime and cross-sectional code:

```python
# WRONG: `winner` is decided using close[t]; `ret` is also the return into close[t].
winner = momentum.idxmax(axis=1)          # uses information available only AT close t
strat  = ret.where(columns == winner)     # but `ret` already happened by close t
```

The position at *t* was chosen with information that includes bar *t* itself, then "earns" bar *t*'s return. It is pure look-ahead, and because momentum signals are autocorrelated it manufactures a gorgeous, completely fake equity curve: the worst kind, because it looks *plausible*. The fix is mechanical: lag the decision before it touches a return.

```python
winner_lag = winner.shift(1).fillna("CASH")   # decide on yesterday's info
strat = ret.where(columns == winner_lag)      # earn today's return
```

!!! danger "War-story: four leaks in one codebase"
    One audit of one regime-driven codebase found the same-bar pattern in **four separate places**, each one a signal series multiplied by a contemporaneous return. None had been caught in review, because each looked like ordinary pandas. Collectively they turned a flat strategy into a stellar one. The rule Titan now enforces: **any series that multiplies a return must be `.shift(1)`'d first** unless you can *prove* the position was knowable strictly before the return's window opened. Treat same-bar `position * return` as guilty until proven innocent.

## Lie 4, Future-normalised features

Standardising a feature is routine, and the routine version is look-ahead by construction:

```python
# WRONG: mean and std are computed over the ENTIRE series, including the future
z = (x - x.mean()) / x.std()
```

At every historical bar, that z-score "knows" the mean and standard deviation of data that hadn't happened yet. It's a subtle leak because it doesn't touch returns directly; it poisons the *feature*, and the damage flows downstream into whatever signal the feature feeds. The fix is to normalise *causally*, using only data available at each point, or, inside a walk-forward, to **freeze** the normalisation statistics on the in-sample window and apply them unchanged out-of-sample:

```python
z_causal    = rolling_zscore(x, window=252)          # only past data at each bar
z_expanding = expanding_zscore(x, min_periods=252)   # all past data, growing window
z_wfo       = is_frozen_zscore(x, is_end=fold.is_end) # IS stats, applied to OOS
```

Titan's metrics module deliberately **does not offer** a full-series z-score. It can't be called by accident because it doesn't exist; the only z-scores available are the causal and IS-frozen ones. That's a recurring pattern in this book: *the safest API is one where the dangerous operation is simply absent.*

## Lie 5, No error bars: a Sharpe is an estimate, not a fact

A backtest Sharpe of, say, 1.1 is a point estimate from one particular history, one draw from the distribution of histories the world could have produced. Reported alone, to two decimals, it implies a precision it does not have. The honest version is an interval, and the cheapest honest interval is a bootstrap: resample the return series many times, recompute the Sharpe on each resample, and read the 2.5th and 97.5th percentiles.

```python
lo, hi = bootstrap_sharpe_ci(
    returns,
    periods_per_year=252,
    n_resamples=1000,   # more resamples -> tighter percentile estimates
    block_size=block,   # <- see the warning below
    seed=42,            # reproducible: same input, same interval
)
```

Then the deployment rule is brutally simple and applied everywhere in Titan:

> **If the 95% lower bound of the Sharpe is ≤ 0, the strategy is `unconfirmed` and cannot enter a default deployment registry, regardless of how good the point estimate looks.**

A point estimate of 1.1 with a 95% interval of `[-0.2, 2.4]` is not a 1.1-Sharpe strategy; it's a coin flip with a good story. The lower bound is the number that decides capital, because it is the honest answer to *"how bad could this plausibly be?"*

!!! warning "War-story: the bootstrap that lied about itself"
    The naive (IID) bootstrap resamples individual bars independently, which **destroys** the autocorrelation that trend and carry strategies live on. That narrows the interval and biases the *lower* bound **upward**: exactly the optimism you're trying to avoid, applied to the exact number you gate on. An external audit of Titan flagged this: strategies were passing the `CI_lo > 0` gate partly because the CI was artificially tight. The fix is a **stationary block bootstrap** (Politis & Romano, 1994): resample *blocks* of consecutive bars (geometric-mean length matched to the strategy's autocorrelation) so serial dependence survives the resampling, and the lower bound tells the truth. Pass a `block_size`; don't accept the IID default for a serially-correlated strategy.

## Sharpe is necessary, not sufficient

Everything above makes *Sharpe* trustworthy. But Sharpe answers exactly one question, return per unit of *volatility*, and it is blind to the questions that actually decide whether you can live with a strategy:

- It treats upside and downside volatility identically. A strategy that occasionally spikes *up* is penalised the same as one that occasionally craters. **Sortino** fixes this by dividing by downside deviation only.
- It says nothing about the *path*. Two strategies with identical Sharpe can have wildly different worst drawdowns and time-to-recovery. **Calmar** (CAGR over max drawdown) and the **max-drawdown** geometry capture what a human actually experiences holding the thing.
- It is a whole-distribution average, so it under-weights the tail that ends you. **CVaR / CDaR** (the *average* loss in the worst slice, not just a single quantile) and a formal **risk of ruin** speak to survival, not smoothness.

!!! warning "War-story: the 1.4-Sharpe strategy nobody could hold"
    A candidate posted a Sharpe around 1.4, past every gate above, yet its drawdown ran deep into the double digits and took over a year to recover: the kind of trough a strategy gets switched off at the bottom of. That is why **Calmar lift, not Sharpe lift, is the primary promotion metric**; the full telling is in [Beyond Sharpe: the metric suite](metric-suite.md).

Crucially, *all* of these metrics are subject to the same five lies: wrong units, survivor math, peeking, future-normalisation, and no error bars. A Calmar computed on a look-ahead equity curve is exactly as worthless as a Sharpe. So the disciplines in this chapter are not "Sharpe rules"; they are *measurement* rules, and they apply to every number you report.

The full battery, Sortino, Calmar, geometric CAGR, CVaR/CDaR, gets its own treatment in [**Beyond Sharpe: the metric suite**](metric-suite.md), and turning tail risk into a survival probability at deployed size is the subject of [**Tail risk & risk of ruin**](tail-risk-and-ruin.md).

## Why Titan puts all of this in one module

Each fix is a one-liner. The reason they hold over hundreds of research scripts is that **none of them is reimplemented locally.** Every Sharpe, Sortino, Calmar, volatility, z-score, and annualisation in the codebase routes through a single shared metrics module, and the module is written so the wrong thing is hard or impossible:

- `sharpe(...)`, `ewm_vol(...)`, `calmar(...)`, `sortino(...)` **require** `periods_per_year`: no default.
- No Sharpe filters zeros internally.
- Only causal and IS-frozen z-scores exist; the full-series version is absent.
- Edge cases (empty, constant, NaN series) return `0.0`/`NaN` rather than raising, so a guardrail never crashes a batch; it just refuses to flatter you.

The alternative, every researcher writing `def _sharpe(r): return r.mean()/r.std()*np.sqrt(252)` at the top of their notebook, guarantees that the lies reappear, independently, forever. A shared module turns "remember to be careful" into "you literally cannot call it the unsafe way." That trade, a tiny bit of ceremony at the call site for a class of bugs that can't recur, is the whole philosophy of this book in miniature.

!!! example "What 'stating the timeframe' buys you"
    Put the bar frequency at the top of every backtest report. It sounds trivial. But the act of writing `# P&L frequency: H1` forces you to confirm the annualisation factor (`252 * 24`), which is also the moment you'd notice a mid-pipeline resample, a zero-filter, or a daily factor on hourly data. One disciplined comment catches three of the five lies at once.

## Takeaways

- A backtest is a measurement: it needs **units** (`periods_per_year`), **causality** (shift discipline + causal normalisation), and **error bars** (a bootstrap CI you gate on).
- The lies all point the same way: they flatter the strategy. Assume any un-audited number is inflated until you've checked all five.
- **Gate on the lower bound, not the point estimate.** `CI_lo ≤ 0` ⇒ unconfirmed, full stop. And use a *serially-aware* bootstrap, or the lower bound lies too.
- **Sharpe is the entry point, not the verdict.** Drawdown geometry (Calmar), downside risk (Sortino), and the tail (CVaR/CDaR, risk of ruin) decide whether a strategy is *livable*, and they obey the same measurement rules.
- Centralise the metrics so the unsafe version can't be written. The safest API is one where the dangerous operation simply doesn't exist.

---

This chapter fixed the *measurement*. The next chapters harden the *experiment*: [**Beyond Sharpe: the metric suite**](metric-suite.md) builds out the battery of numbers; [**Walk-forward that's actually out-of-sample**](walk-forward.md) makes sure you're not testing on data you trained on; and [**Beating your own optimiser**](deflated-sharpe.md) corrects for the fact that the more parameters you try, the better your best result looks by pure chance.
