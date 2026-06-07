"""Global event hooks for the live TradingNode.

Subscribes to NautilusTrader's MessageBus at runtime startup so that any
order-lifecycle event (rejection, large fill, etc.) routes through a
single notification path — regardless of whether the originating
strategy's per-instance ``on_order_*`` callback bothered to call
``notify_order_event``.

Tier 1.4 of the operational-robustness framework
(``directives/Operational Robustness Framework 2026-05-12.md``):

  > Right now rejections are logged. Make any IBKR OrderRejected event
  > a Slack alert (not just a log line).

The May 11 audit found only 2 of 14 strategies (bond_gold, demo_fxmr)
had a ``notify_*`` call inside their ``on_order_rejected`` handler. The
other 12 silently logged rejections that — like today's PDT lockout —
could escape detection for hours. Wiring the alert at the *runtime*
level rather than per-strategy makes the alert impossible to forget
when a new strategy is added.

Usage::

    from titan.utils.global_event_hooks import register_global_event_hooks
    node.build()
    register_global_event_hooks(node)
    node.run()
"""

from __future__ import annotations

import logging
from typing import Any

from nautilus_trader.live.node import TradingNode
from nautilus_trader.model.events import OrderCanceled, OrderRejected

from titan.utils.notification import notify_order_event

_log = logging.getLogger(__name__)


def _on_order_event(msg: Any) -> None:
    """MessageBus handler — dispatches on event type.

    Subscribed to a wildcard ``events.order.*`` topic, so this handler
    receives every order lifecycle event. We filter to the two we want
    to amplify into Slack/Telegram (rejections and unexpected
    cancellations) and ignore the rest.
    """
    try:
        if isinstance(msg, OrderRejected):
            strategy_id = str(getattr(msg, "strategy_id", "unknown"))
            instrument = str(getattr(msg, "instrument_id", "?"))
            client_oid = str(getattr(msg, "client_order_id", "") or "")
            venue_oid = str(getattr(msg, "venue_order_id", "") or "")
            reason = str(getattr(msg, "reason", "") or "")
            # IBKR rejection messages are HTML-tinged; flatten for Slack.
            reason_clean = reason.replace("<br>", " ").replace("<BR>", " ").strip()
            notify_order_event(
                strategy=strategy_id,
                event_type="rejected",
                instrument=instrument,
                side="?",  # OrderRejected doesn't carry side; not critical for the alert
                qty=0,
                client_order_id=client_oid,
                venue_order_id=venue_oid,
                note=reason_clean[:500],  # cap to avoid Slack truncation
            )
            _log.warning(
                "Global rejection hook fired: strategy=%s instrument=%s reason=%s",
                strategy_id,
                instrument,
                reason_clean[:200],
            )
        elif isinstance(msg, OrderCanceled):
            # Most cancellations are routine (close-of-day, strategy-initiated
            # cancel) and would be alert spam. Only flag cancellations that
            # match the EXTERNAL pattern (broker-initiated, not strategy).
            # Currently we just log, no Slack — re-evaluate if rate is high.
            client_oid = str(getattr(msg, "client_order_id", "") or "")
            _log.info("Global hook: OrderCanceled client_order_id=%s", client_oid)
    except Exception as e:
        # Never let an event-hook error propagate into the runtime loop.
        _log.warning("Global event hook error: %s", e)


def register_global_event_hooks(node: TradingNode) -> None:
    """Subscribe the global rejection notifier to the node's message bus.

    Must be called *after* ``node.build()`` (the kernel/msgbus is created
    during build). Safe to call before or after ``add_strategy`` calls.

    Subscribes to the broad ``events.order.*`` wildcard topic so any
    NT-internal change to per-event sub-topics still routes through here.
    Filtering happens in the handler.
    """
    try:
        msgbus = node.kernel.msgbus
    except AttributeError as e:
        raise RuntimeError(
            "register_global_event_hooks must be called after node.build(); "
            "the kernel/msgbus is not yet initialised."
        ) from e

    # Subscribe to all order events — handler filters by type.
    msgbus.subscribe(topic="events.order.*", handler=_on_order_event)
    _log.info("Registered global event hook: events.order.* -> notify on rejection")
