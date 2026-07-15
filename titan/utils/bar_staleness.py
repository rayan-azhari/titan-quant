"""bar_staleness.py -- trading-calendar-aware daily-bar staleness helpers.

Motivating incident (2026-07-01): an IBKR market-data farm disconnect left the
daily ``subscribe_historical_bars`` subscriptions frozen at the prior session's
close for 4 trading days. The H1 clock tick self-healed, so the existing
single-tick staleness watchdog reported "recovered" -- while every 1-DAY ETF
bar stayed dead, silently freezing every unrealised mark and stopping all
signal generation. Nothing watched the daily bars.

These pure helpers back three fixes:
  * B1 -- ``daily_summary`` annotates each open position's mark age + STALE banner
  * B2 -- ``subscription_health`` alerts on stale held-position/signal daily bars
  * C1 -- ``watchdog_portfolio`` auto-restarts the node when daily bars go stale

Why *session counting*, not wall-clock age
------------------------------------------
A daily bar arrives once per *trading session* (a weekday, excluding market
holidays) and is silent across weekends. Wall-clock age alone therefore
false-positives every weekend (a Friday bar is ~72h old by Monday's check).
We instead count *missed weekday sessions* between the newest bar's date and
"now". This is robust to weekends. Holidays over-count by at most the number of
consecutive holidays, which is tolerated by:
  * a ``max_missed_sessions`` default of 2 (a single US/UK holiday, which
    differ between the two calendars, only skips one session), and
  * the watchdog's consecutive-restart cap (loop-safety for a rare 2+ holiday
    stretch such as UK Christmas + Boxing Day).

Keeping a full exchange calendar here would be heavier and still incomplete;
coarse weekday counting plus those two guards is the right altitude for an
*alert / self-heal* signal (not a trading gate).
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

# Nanoseconds per second -- Nautilus bar timestamps are ns-since-epoch.
_NS_PER_SEC = 1_000_000_000


def count_missed_sessions(last_bar_date: date, now_date: date) -> int:
    """Number of completed weekday sessions between ``last_bar_date`` and
    ``now_date`` -- i.e. weekdays ``d`` with ``last_bar_date < d < now_date``.

    "Today" (``now_date``) is excluded because its session may not have closed
    yet, so its daily bar legitimately may not exist. Yesterday's bar seen today
    therefore counts as 0 missed sessions, and a Friday bar seen on Monday also
    counts as 0 (Sat/Sun are not weekdays).

    Returns 0 if ``last_bar_date >= now_date`` (fresh, or clock skew).
    """
    if last_bar_date >= now_date:
        return 0
    missed = 0
    d = last_bar_date + timedelta(days=1)
    while d < now_date:
        if d.weekday() < 5:  # Mon=0 .. Fri=4 are trading weekdays
            missed += 1
        d += timedelta(days=1)
    return missed


def evaluate_daily_bar_staleness(
    last_bar_ts_ns: int | None,
    now_ns: int,
    max_missed_sessions: int,
) -> tuple[bool, int | None, float | None]:
    """Decide whether a daily bar is stale.

    Returns ``(is_stale, missed_sessions, age_secs)``:
      * ``last_bar_ts_ns is None`` -> ``(False, None, None)``. "No bar cached"
        is intentionally NOT flagged here -- that ambiguous cold-start /
        never-subscribed case is owned by the one-shot subscription-presence
        check, not the staleness path (mirrors ``evaluate_bar_staleness`` in
        ``subscription_health`` treating None specially).
      * otherwise ``is_stale`` iff ``missed_sessions >= max_missed_sessions``.

    ``age_secs`` is the raw wall-clock age (clamped at 0 for clock skew), kept
    for telemetry / the heartbeat even though the staleness decision uses the
    session count.
    """
    if last_bar_ts_ns is None:
        return False, None, None
    last_date = datetime.fromtimestamp(last_bar_ts_ns / _NS_PER_SEC, tz=timezone.utc).date()
    now_date = datetime.fromtimestamp(now_ns / _NS_PER_SEC, tz=timezone.utc).date()
    missed = count_missed_sessions(last_date, now_date)
    age_secs = max(0.0, (now_ns - last_bar_ts_ns) / _NS_PER_SEC)
    return missed >= max_missed_sessions, missed, age_secs


def evaluate_never_received(
    last_bar_ts_ns: int | None,
    started_date: date | None,
    now_date: date,
    max_missed_sessions: int,
) -> bool:
    """True iff a monitored daily bar has produced NOTHING after a grace window.

    THE HOLE THIS CLOSES. :func:`evaluate_daily_bar_staleness` deliberately maps
    ``last_bar_ts_ns is None`` -> "not stale" (a cold start is ambiguous), and the
    one-shot subscription-presence check only proves the *instrument* is in the
    cache -- not that *bars flow*. Between them, a feed that never delivers a
    single bar is indistinguishable from a healthy one.

    That is not hypothetical: ``demo_alloc`` was deployed 2026-07-11 with two
    brand-new subscriptions (ETFA/ETFB) that never backfilled. It held nothing for
    two days, across multiple market closes, while ``position_bars_stale=False``,
    ``missed_sessions=0`` and ``SubscriptionHealth: PASS`` all reported green.

    A bar type that has NEVER delivered, once ``max_missed_sessions`` weekday
    sessions have elapsed since the strategy started, is dead -- not cold-starting.
    Returns False while still inside the grace window, and False the moment any bar
    has ever arrived (staleness then owns it).
    """
    if last_bar_ts_ns is not None:
        return False  # a bar exists -> the staleness path owns this feed
    if started_date is None:
        return False  # unknown start -> cannot judge; fail quiet, never false-alarm
    return count_missed_sessions(started_date, now_date) >= max_missed_sessions


def instrument_id_from_bar_type(bar_type: str) -> str | None:
    """``'DEMOA.ARCA-1-DAY-LAST-EXTERNAL'`` -> ``'DEMOA.ARCA'``.

    Splits on the literal ``'-1-'`` that terminates the InstrumentId portion of
    a Nautilus bar-type string (step is always 1 in this stack). Returns None
    for a string with no such separator.
    """
    parts = bar_type.split("-1-", 1)
    return parts[0] if len(parts) == 2 and parts[0] else None
