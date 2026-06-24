"""notification.py — Slack / Telegram message dispatch.

Two roles:
  1.  Guardian agent error alerts (the original use case): call
      :func:`send_slack_message` with a free-form string and severity.
  2.  Live trading event notifications: call one of the semantic helpers
      :func:`notify_signal`, :func:`notify_order_event`,
      :func:`notify_position_closed`. Each formats a structured multi-line
      message and dispatches to whichever backends are configured via env.

Backends (auto-detected at send time):
  * Slack      — set ``SLACK_WEBHOOK_URL`` env (incoming-webhook URL).
  * Telegram   — set ``TELEGRAM_BOT_TOKEN`` AND ``TELEGRAM_CHAT_ID`` env.

If both are configured, both receive the message. If neither is configured,
the call is a no-op (with a single one-time stderr warning).

Sends are genuinely fire-and-forget: each backend POST runs on a daemon worker
thread, so a slow/hung webhook (including unbounded DNS resolution, which the
2-second urlopen timeout does NOT bound) can never block the NautilusTrader
event loop — the 2026-06-11 audit found these calls were synchronous on the
trading thread, amplified during rejection storms. The default ``_dispatch``
returns the count of CONFIGURED backends (not successes), so the only meaning
of a 0 return is "no backend configured". The CLI smoke test uses
``block=True`` to wait for the actual delivery result.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import urllib.parse
import urllib.request
from typing import Any

# ── Backend env vars (resolved lazily per call, so dotenv-loaded values work) ──


def _slack_url() -> str | None:
    return os.getenv("SLACK_WEBHOOK_URL")


def _telegram_creds() -> tuple[str | None, str | None]:
    return os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")


_warned_no_backend = False

# Fire-and-forget delivery bookkeeping. A rejection storm (every order across
# every strategy rejected during an account lockout) could otherwise spawn
# unbounded worker threads / queue unbounded backlog, so we cap in-flight sends.
_inflight = 0
_inflight_lock = threading.Lock()
_MAX_INFLIGHT = 200
_warned_dropped = False

# Telegram hard text limit (chars). Over-length messages get HTTP 400.
_TELEGRAM_MAX = 4096

# Order-event de-duplication. The global rejection hook AND a strategy's own
# on_order_rejected both fire for the same OrderRejected, so without this every
# rejection from bond_gold/demo_fxmr produced two near-identical alerts (audit
# 2026-06-11). Collapse repeats of the same (strategy, event, instrument) within
# a short window to a single alert.
_order_event_seen: dict[tuple[str, str, str], float] = {}
_order_event_lock = threading.Lock()
_ORDER_EVENT_TTL = 5.0


def _order_event_is_dup(strategy: str, event_type: str, instrument: str) -> bool:
    key = (strategy, event_type.lower(), instrument)
    now = time.monotonic()
    with _order_event_lock:
        for k, ts in list(_order_event_seen.items()):
            if now - ts > _ORDER_EVENT_TTL:
                del _order_event_seen[k]
        if key in _order_event_seen:
            return True
        _order_event_seen[key] = now
        return False


# ── Low-level senders ─────────────────────────────────────────────────────────


def _post_slack(text: str) -> bool:
    url = _slack_url()
    if not url:
        return False
    payload = json.dumps({"text": text}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            return resp.status == 200
    except Exception as e:
        print(f"  ✗ Slack notify failed: {e}", file=sys.stderr)
        return False


def _strip_slack_markdown(text: str) -> str:
    """Strip Slack-flavoured markup (asterisks for bold, backticks for code).

    Telegram with parse_mode omitted renders those characters literally, so we
    remove them for a clean plain-text render rather than leaking the markup.
    """
    return text.replace("*", "").replace("`", "")


def _post_telegram(text: str) -> bool:
    token, chat_id = _telegram_creds()
    if not token or not chat_id:
        return False
    body = _strip_slack_markdown(text)
    if len(body) > _TELEGRAM_MAX:
        # Truncate with an explicit marker — an over-length body 400s silently.
        body = body[: _TELEGRAM_MAX - 20].rstrip() + "\n…(truncated)"
    api_url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = urllib.parse.urlencode({"chat_id": chat_id, "text": body}).encode("utf-8")
    req = urllib.request.Request(
        api_url,
        data=payload,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    try:
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            return resp.status == 200
    except Exception as e:
        print(f"  ✗ Telegram notify failed: {e}", file=sys.stderr)
        return False


def _safe_send(fn, text: str) -> bool:
    global _inflight
    try:
        return bool(fn(text))
    except Exception as e:  # noqa: BLE001 — a send must never crash a worker
        print(f"  ✗ notify worker failed: {e}", file=sys.stderr)
        return False
    finally:
        with _inflight_lock:
            _inflight -= 1


def _dispatch(text: str, *, block: bool = False) -> int:
    """Dispatch ``text`` to all configured backends.

    Default (``block=False``): fire-and-forget on daemon threads; returns the
    count of CONFIGURED backends (so a 0 return means *no backend configured*,
    never *a send failed* — the prior code conflated the two). ``block=True``
    sends synchronously and returns the count of SUCCESSFUL sends (used by the
    CLI smoke test and any caller that needs the real delivery result).
    """
    global _warned_no_backend, _inflight, _warned_dropped
    backends = []
    if _slack_url():
        backends.append(_post_slack)
    if all(_telegram_creds()):
        backends.append(_post_telegram)

    if not backends:
        if not _warned_no_backend:
            print(
                "  notify: no backend configured (set SLACK_WEBHOOK_URL or "
                "TELEGRAM_BOT_TOKEN+TELEGRAM_CHAT_ID). Notifications skipped.",
                file=sys.stderr,
            )
            _warned_no_backend = True
        return 0

    if block:
        return sum(int(_safe_send_blocking(fn, text)) for fn in backends)

    for fn in backends:
        with _inflight_lock:
            if _inflight >= _MAX_INFLIGHT:
                if not _warned_dropped:
                    print(
                        f"  notify: >{_MAX_INFLIGHT} sends in flight — dropping "
                        "(webhook backlog?).",
                        file=sys.stderr,
                    )
                    _warned_dropped = True
                continue
            _inflight += 1
        threading.Thread(target=_safe_send, args=(fn, text), daemon=True).start()
    return len(backends)


def _safe_send_blocking(fn, text: str) -> bool:
    try:
        return bool(fn(text))
    except Exception as e:  # noqa: BLE001
        print(f"  ✗ notify failed: {e}", file=sys.stderr)
        return False


# ── Backward-compatible Guardian alert API ────────────────────────────────────


def send_slack_message(message: str, severity: str = "warning") -> bool:
    """Original Guardian-agent error-alert API. Kept for backward compat.

    Sends to Slack only (Guardian was Slack-specific). New code should use
    the semantic helpers below — they dispatch to all configured backends.
    """
    emoji_map = {"info": "ℹ️", "warning": "⚠️", "critical": "🚨"}
    emoji = emoji_map.get(severity, "📢")
    body = f"{emoji} *Titan-IBKR-Algo Alert*\n*Severity:* {severity.upper()}\n\n{message}"
    if not _slack_url():
        print("WARNING: SLACK_WEBHOOK_URL not set in .env. Notification skipped.")
        return False
    return _post_slack(body)


# ── Semantic helpers for trading events ───────────────────────────────────────


def _fmt_money(amount: float | None, ccy: str = "USD") -> str:
    if amount is None:
        return "?"
    sign = "+" if amount >= 0 else ""
    return f"{sign}{amount:,.2f} {ccy}"


def _fmt_pct(pct: float | None) -> str:
    if pct is None:
        return "?"
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct * 100:.2f}%"


# P3.6: a BUY whose notional exceeds this multiple of the strategy's own equity
# is the 2026-06-23 CSPX over-size signature (notional 66k on a 6.4k book =
# 10.4x). Flag it loudly in the signal message so a human sees it immediately.
_NOTIONAL_RATIO_WARN = 3.0


def notify_signal(
    strategy: str,
    action: str,
    instrument: str,
    qty: float | int,
    *,
    price: float | None = None,
    notional: float | None = None,
    notional_ccy: str = "USD",
    risk_pct: float | None = None,
    equity: float | None = None,
    equity_ccy: str = "USD",
    account: str | None = None,
    reason: dict[str, Any] | None = None,
) -> int:
    """Send a "signal generated" notification.

    Args:
        strategy: e.g. ``"bond_gold_CSPX"`` (matches the PRM strategy id).
        action: ``"BUY"`` / ``"SELL"`` / ``"SHORT"``.
        instrument: ``"DEMOA.ARCA"``.
        qty: Order quantity (units / shares).
        price: Latest bar close at signal time.
        notional: Position notional in ``notional_ccy``.
        risk_pct: Risk-per-trade as decimal (0.01 = 1%).
        equity: Strategy equity at signal time.
        reason: Free-form dict of indicator values that triggered the signal,
                e.g. ``{"z_score": -0.89, "threshold": 0.5}``.

    Returns the count of backends that accepted the message.
    """
    lines = [f"🎯 *Signal* — `{strategy}`", f"`{action}` {qty} {instrument}"]
    if price is not None:
        line = f"  • Price: {price:,.4f}"
        if notional is not None:
            line += f"   Notional: {_fmt_money(notional, notional_ccy)}"
        lines.append(line)
    if reason:
        body = "  ".join(f"{k}={_fmt_value(v)}" for k, v in reason.items())
        lines.append(f"  • Reason: {body}")
    if risk_pct is not None and equity is not None:
        risk_amt = equity * risk_pct
        lines.append(
            f"  • Risk: {_fmt_pct(risk_pct)} of {_fmt_money(equity, equity_ccy)} "
            f"= {_fmt_money(risk_amt, equity_ccy)}"
        )
    elif equity is not None:
        lines.append(f"  • Strategy equity: {_fmt_money(equity, equity_ccy)}")
    # P3.6: flag an over-sized order -- notional far above the strategy book is
    # the over-size signature the prior controls were blind to.
    if notional is not None and equity:
        ratio = abs(notional) / abs(equity)
        if ratio > _NOTIONAL_RATIO_WARN:
            lines.append(
                f"  • :rotating_light: NOTIONAL {ratio:.1f}x strategy equity "
                f"(> {_NOTIONAL_RATIO_WARN:g}x) — possible OVER-SIZE, check sizing"
            )
    if account:
        lines.append(f"  • Account: `{account}`")
    return _dispatch("\n".join(lines))


def notify_order_event(
    strategy: str,
    event_type: str,
    instrument: str,
    side: str,
    qty: float | int,
    *,
    venue_order_id: str | None = None,
    client_order_id: str | None = None,
    price: float | None = None,
    order_type: str | None = None,
    note: str | None = None,
) -> int:
    """Send an order-lifecycle event notification.

    Args:
        event_type: ``"accepted"`` / ``"rejected"`` / ``"filled"`` /
                    ``"canceled"`` / etc. Determines emoji + tone.
        side: ``"BUY"`` / ``"SELL"``.
        qty: Order quantity.
        venue_order_id, client_order_id: IBKR / NautilusTrader IDs.
        price: Fill price (for filled events).
        order_type: ``"MARKET"`` / ``"LIMIT"`` / ``"STOP"`` etc.
        note: Free-form note (e.g. ``"queued for next session"`` from IBKR
              warning code 399).
    """
    # Collapse the global-hook + per-strategy duplicate for the same event.
    if _order_event_is_dup(strategy, event_type, instrument):
        return 0
    et = event_type.lower()
    emoji_map = {
        "accepted": "📝",
        "submitted": "📝",
        "filled": "💰",
        "rejected": "🚫",
        "canceled": "🚫",
        "expired": "🚫",
    }
    emoji = emoji_map.get(et, "📋")
    headline_map = {
        "accepted": "Order accepted",
        "submitted": "Order submitted",
        "filled": "Order filled",
        "rejected": "Order REJECTED",
        "canceled": "Order canceled",
        "expired": "Order expired",
    }
    headline = headline_map.get(et, f"Order {et}")
    lines = [f"{emoji} *{headline}* — `{strategy}`"]
    detail = f"`{side}` {qty} {instrument}"
    if order_type:
        detail += f" ({order_type})"
    if price is not None:
        detail += f" @ {price:,.4f}"
    lines.append(detail)
    if venue_order_id:
        lines.append(f"  • venue_order_id: `{venue_order_id}`")
    if client_order_id:
        lines.append(f"  • client_order_id: `{client_order_id}`")
    if note:
        lines.append(f"  • Note: {note}")
    return _dispatch("\n".join(lines))


def notify_position_closed(
    strategy: str,
    instrument: str,
    direction: str,
    *,
    realized_pnl: float | None = None,
    realized_pnl_ccy: str = "USD",
    entry_price: float | None = None,
    exit_price: float | None = None,
    held_days: float | None = None,
    held_bars: int | None = None,
    r_multiple: float | None = None,
    equity_after: float | None = None,
    initial_equity: float | None = None,
    equity_ccy: str = "USD",
    exit_reason: str | None = None,
) -> int:
    """Send a position-closed notification.

    Args:
        direction: ``"LONG"`` / ``"SHORT"``.
        realized_pnl: PnL at close in ``realized_pnl_ccy``.
        entry_price, exit_price: First entry and final exit prices.
        held_days: Calendar duration in days (float OK for sub-day).
        held_bars: Bar duration (alternative to days).
        r_multiple: PnL in units of initial risk (signed).
        equity_after: Strategy equity after close.
        initial_equity: Starting equity (for cumulative-pct calculation).
        exit_reason: ``"TP"`` / ``"SL"`` / ``"signal_flip"`` / etc.
    """
    is_win = realized_pnl is not None and realized_pnl > 0
    emoji = "✅" if is_win else "❌" if realized_pnl is not None else "🔚"
    lines = [
        f"{emoji} *Position closed* — `{strategy}`",
        f"`{direction}` {instrument}",
    ]
    if held_days is not None:
        lines.append(f"  • Held: {held_days:.1f}d")
    elif held_bars is not None:
        lines.append(f"  • Held: {held_bars} bars")
    if entry_price is not None and exit_price is not None:
        lines.append(f"  • Entry: {entry_price:,.4f}   Exit: {exit_price:,.4f}")
    if realized_pnl is not None:
        line = f"  • PnL: {_fmt_money(realized_pnl, realized_pnl_ccy)}"
        if equity_after is not None:
            line += f"   Strategy equity: {_fmt_money(equity_after, equity_ccy)}"
        lines.append(line)
    if r_multiple is not None:
        lines.append(f"  • R-multiple: {r_multiple:+.2f}")
    if equity_after is not None and initial_equity is not None and initial_equity > 0:
        cum_pct = (equity_after / initial_equity) - 1.0
        lines.append(
            f"  • Cumulative: {_fmt_pct(cum_pct)} on "
            f"{_fmt_money(initial_equity, equity_ccy)} initial"
        )
    if exit_reason:
        lines.append(f"  • Reason: {exit_reason}")
    return _dispatch("\n".join(lines))


def _fmt_value(v: Any) -> str:
    """Best-effort short formatting for reason-dict values."""
    if isinstance(v, float):
        return f"{v:+.4f}" if abs(v) < 100 else f"{v:,.2f}"
    return str(v)


def notify_health(
    event: str,
    *,
    severity: str = "warning",
    detail: str | None = None,
) -> int:
    """Send an infrastructure-health alert (gateway restart, runner crash,
    connection error, watchdog state change).

    Goes to all configured backends (Slack + Telegram). Use ``severity`` to
    set the emoji: ``info`` (ℹ️), ``warning`` (⚠️), ``critical`` (🚨), or
    ``ok`` (✅) for recovery messages.
    """
    emoji_map = {"info": "ℹ️", "ok": "✅", "warning": "⚠️", "critical": "🚨"}
    emoji = emoji_map.get(severity, "📢")
    lines = [f"{emoji} *Titan health* — {event}"]
    if detail:
        lines.append(detail)
    return _dispatch("\n".join(lines))


def notify_daily_summary(body: str) -> int:
    """Send a free-form daily summary message.

    The strategy assembles the body (account state, positions, per-strategy
    equity, halt status, etc) and passes it here as a pre-formatted string.
    We just prefix the 📊 header and dispatch.
    """
    return _dispatch(f"📊 *Daily Portfolio Summary*\n{body}")


# ── CLI for quick smoke testing ───────────────────────────────────────────────


def main() -> None:
    """``uv run python -m titan.utils.notification [test_type]``

    test_type: ``signal`` | ``order`` | ``position`` | ``alert`` (default)
    """
    test = sys.argv[1] if len(sys.argv) > 1 else "alert"
    if test == "signal":
        n = notify_signal(
            strategy="bond_gold_CSPX",
            action="BUY",
            instrument="DEMOA.ARCA",
            qty=36,
            price=770.50,
            notional=27_738.0,
            risk_pct=0.01,
            equity=10_000.0,
            account="DUxxxxxxx",
            reason={"z_score": -0.89, "threshold": 0.5},
        )
    elif test == "order":
        n = notify_order_event(
            strategy="bond_gold_CSPX",
            event_type="accepted",
            instrument="DEMOA.ARCA",
            side="BUY",
            qty=36,
            order_type="MARKET",
            venue_order_id="101",
            client_order_id="O-20260430-164248-PORTFOLIO-001-1",
            note="queued for next LSE open (IBKR warning 399)",
        )
    elif test == "position":
        n = notify_position_closed(
            strategy="bond_gold_CSPX",
            instrument="DEMOA.ARCA",
            direction="LONG",
            realized_pnl=528.40,
            entry_price=770.50,
            exit_price=785.20,
            held_days=8.0,
            r_multiple=1.4,
            equity_after=10_528.40,
            initial_equity=10_000.0,
            exit_reason="TP",
        )
    elif test == "health":
        n = notify_health(
            "Test health alert — gateway disconnect simulated",
            severity="warning",
            detail="If you see this, watchdog alerts are wired up correctly.",
        )
    else:
        n = send_slack_message("This is a Guardian-style test alert.", severity="info")
        n = 1 if n else 0
        print(f"  notify: dispatched to {n} backend(s)")
        return
    # The semantic helpers above dispatch fire-and-forget; re-send synchronously
    # so the smoke test reports the real delivery result and the daemon worker
    # isn't cut off by process exit.
    delivered = _dispatch("🧪 notification smoke test (synchronous)", block=True)
    print(f"  notify: {n} backend(s) configured; synchronous test delivered to {delivered}")


if __name__ == "__main__":
    main()
