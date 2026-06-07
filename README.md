# titan-quant

**The open framework companion to the book _Building a Production Quant Trading Stack_.**

> 📖 **Read the book online:** <https://www.rayanazhari.co.uk/books/building-a-production-quant-trading-stack/>

The book teaches the *process* of building a systematic trading system you can trust; this
repository gives you the actual **framework** (validation, metrics, risk) plus small
**educational strategies**, so you can clone it, run it, and watch the methodology work on real
data. The book itself is hosted as a website (link above); its source is not in this repository.

> [!WARNING]
> **Educational software. Not investment advice. No warranty.**
> Trading involves substantial risk of loss. Nothing here is a recommendation to trade any
> instrument. The example strategies are deliberately simple teaching aids with **no expected edge**.
> The live/paper integration defaults to a **paper** account and you run it entirely at your own
> risk. See the [LICENSE](LICENSE) (Apache-2.0) disclaimer of warranty and limitation of liability.

> [!NOTE]
> **What this repo is _not_.** It contains none of the author's proprietary strategies, parameters,
> instrument selections, account details, or live performance. Those are deliberately excluded. The
> value shared here is the *engineering and methodology*, which is exactly what is safe to make public.

## What's inside

| Path | What it is |
|---|---|
| `titan/research/` | Shared metrics (Sharpe, Sortino, Calmar, CVaR/CDaR, bootstrap CI) and the validation **framework** (typology, walk-forward, sanctuary, DSR, Monte Carlo, risk of ruin, decision matrix, dashboard). |
| `titan/risk/` | The portfolio/risk layer: PortfolioRiskManager, Allocator, per-strategy EquityTracker + FX, correlation-dial leverage governor, drawdown throttle, governance. |
| `titan/strategies/demo_trend/` | A simple educational trend strategy implementing the integration contract. |
| `scripts/` | Sample-data downloader, the data-quality manifest gate, the methodology anti-pattern scanner, and `validate_demo.py` (runs the framework end-to-end). |
| `tests/` | The framework's synthetic-ground-truth safety-net tests. |

## Quickstart

```bash
# 1. Install (uv recommended; pip works too)
uv sync --extra demo            # framework + viz + data + live (NautilusTrader)
#   or a lighter install:  uv sync --extra data --extra viz

# 2. Fetch a little sample data (daily ETFs from Yahoo)
uv run python scripts/download_data_yfinance.py --symbols SPY=SPY GLD=GLD IEF=IEF --interval D --start 2010-01-01

# 3. Watch the framework decide: it runs WFO + bootstrap CI + DSR + Monte Carlo +
#    risk of ruin, then prints a deploy/reject verdict for two demo candidates,
#    one with a plausible edge and one that is market-neutral noise (rejected).
uv run python scripts/validate_demo.py
```

The `validate_demo.py` run is the point: it demonstrates the book's core thesis (*suspicion over
celebration*) by showing the framework **rejecting** a good-looking-but-fake candidate on the same
gates it uses to bless a real one.

## The live/paper demo (optional, advanced)

`titan/strategies/demo_trend/` shows the canonical signal shape behind the live integration
contract (`on_start` register equity, `on_bar` report + check halt + size, `on_position_closed`
update tracker). Wiring it into a live NautilusTrader paper node is documented in the book
(Parts IV & VI). It is **paper-by-default**; never point it at a live account without understanding
every safety gate in Part V.

## Contributing & safety gates

- **Framework code** (Apache-2.0): issues and pull requests welcome. CI runs `ruff`, `pytest`, the
  methodology anti-pattern scanner (`scripts/audit_codebase_methodology.py`), and a **redaction
  gate** (`scripts/check_public_redaction.py`) that fails the build if any account id, secret, or
  proprietary name is introduced. Please keep example strategies edge-free and educational.
- **The book** is all rights reserved and not open to contribution, but **corrections/typos are
  welcome** as issues.

## Licence

This project is **split-licensed**:

- **Code** (this repository: `titan/`, `scripts/`, configs, tests): [Apache-2.0](LICENSE).
  Use it, fork it, build on it.
- **Book** (hosted at <https://www.rayanazhari.co.uk/books/building-a-production-quant-trading-stack/>): **© 2026 Dr. Rayan Azhari,
  all rights reserved**. Free to read; not for redistribution, hosting, resale, modification, or
  model training without permission. The book's source is not included in this repository.

See `NOTICE` for attribution.
