# Style & redaction guide

This is the contract every chapter follows. It exists so that 25 chapters written
over time read like one book, and so nothing proprietary leaks into a public document.

## Audience & voice

- **Reader:** a working engineer / quant-curious developer. Assumes Python fluency and
  basic trading literacy (knows what a Sharpe ratio and a limit order are). Explains
  everything project-specific or advanced.
- **Voice:** practical, opinionated, evidence-driven. First-person-plural ("we did X,
  it broke, here's why"). Short sentences. Concrete over abstract.
- **Tone rule, *suspicion over celebration*:** never present a number as impressive.
  Present it as a claim to be stress-tested.

## Chapter shape

Each chapter follows the same skeleton:

1. **H1 title** `# N. Title` (matches the `nav` entry).
2. **Hook** (1 to 2 short paragraphs): the problem this chapter solves, ideally via a
   concrete failure it prevents.
3. **The principle** (generalisable): what's true for *any* quant stack.
4. **The Titan example** (concrete): how Titan does it, with a sanitised code pattern.
5. **War-stories** (`!!! warning`/`!!! danger`): the bugs that bought the rules; see below.
6. **Takeaways** (bulleted) + a forward link to the next relevant chapter.

Keep chapters 1,800 to 3,000 words. Explain *why*, not just *what*: give the reader the
intuition and the mechanism, not a list of rules to memorise. Use tables and one diagram
where they earn their place.

### War-stories are the point (standing rule)

Every chapter carries **at least one, preferably two, war-stories**: a real bug,
mis-assumption, or near-miss, and the specific rule it bought. They are the most
distinctive and memorable part of the book: people remember the strategy that mis-sized a
leg by a third on a currency assumption far longer than they remember the rule.

- Format each as a `!!! warning` (corrupts a result/risk math) or `!!! danger` (touches
  live capital) admonition with a short title, the failure, the consequence, and the fix.
- Sanitise: tell the *shape* of the bug, never the alpha. "A regime strategy decided its
  position using the same bar's close, then earned that bar's return"; not the parameters.
- The deepest repository is the [failure-mode catalogue](../part2-research/failure-mode-catalogue.md);
  other chapters draw the relevant story inline.

### Cover the full metric suite, not just Sharpe

Sharpe is the entry point, never the verdict. Wherever performance is discussed, reach for
the right tool: **Sortino** (downside-only), **Calmar / geometric CAGR & MaxDD**, **CVaR /
CDaR** (tail), **risk of ruin** (survival probability at deployed size), and **Kelly /
fractional Kelly** for sizing. Two chapters own this material:
[Beyond Sharpe: the metric suite](../part2-research/metric-suite.md) and
[Position sizing: Kelly & vol-targeting](../part5-portfolio-risk/position-sizing-kelly.md);
but any chapter quoting a single number should name the others it would check.

## Generalisable-with-example pattern

The book teaches **principles**; Titan is the **running example**. Structure each
technical point as *principle first, then "here's how Titan does it."* A reader on a
different broker / language should still get the lesson. Refer to the system as
**Titan** (a named case study); never imply the reader must use the same tools.

## Markdown conventions

- Admonitions: `!!! note` (context), `!!! tip` (advice), `!!! warning` (can corrupt a
  result or risk math), `!!! danger` (touches live capital), `!!! example` (worked case).
- Code: fenced with a language. Prefer `python`, `bash`, `toml`, `yaml`. Keep snippets
  to the shape of the real thing; annotate with `# (1)!` callouts where helpful.
- Diagrams: Mermaid fenced blocks (version-controlled, render in Material).
- Cross-links: relative, e.g. `[Ch. 18](../part5-portfolio-risk/portfolio-risk-manager.md)`.

## Redaction policy (this is public)

!!! danger "Never publish"
    - Real account IDs, broker usernames/passwords, API keys, webhook/bot tokens.
    - Exact deployable parameters that constitute the edge: precise lookbacks,
      thresholds, tier boundaries, the full instrument shortlist for a live sleeve.
    - Live performance presented as *current* edge (specific live Sharpe / CI / DD).
    - Anything that would let a reader reconstruct the live book of positions.

!!! tip "Always safe to teach"
    - Architecture, module boundaries, the lifecycle, the integration contract.
    - The methodology framework (WFO, sanctuary, DSR, Monte Carlo, decision matrix).
    - Code *patterns* (sanitised), function signatures, type discipline.
    - The failure-mode catalogue and the rules each bug bought.

### How to sanitise numbers

- Replace alpha-bearing parameters with placeholders or illustrative values, and **say
  so**: *"a lookback `L` (Titan uses a value in the low double digits)"* or *"≈ 1.x
  (illustrative)."*
- When a real metric is needed to make a point, label it **illustrative** or frame it as
  a method (*"the bootstrap put the 95% lower bound below zero, so it was rejected"*:
  the *mechanism*, not a publishable track record).
- Account/instrument identifiers: use generic forms (`a USD-quoted Treasury UCITS ETF`,
  `account DUxxxxxxx`).

### Redaction-audit pass

Before a chapter is "done," re-read it once asking only: *could a competitor clone an
edge, or could a secret leak, from this page?* If yes, sanitise. Record the pass in the
chapter PR description.

## Accuracy

- Ground every technical claim in the real source; cite `file:line` where it helps a
  maintainer, but keep prose broker/language-agnostic.
- Mark anything not yet re-validated under the current methodology as **unconfirmed**.
- If you can't verify a claim in under a minute, soften it or cut it.
