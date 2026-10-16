"""
core/sessions.py
================
Trading sessions, weekday handling and the honest per-session breakdown
(v1.3.0, requested as "add time").

Everything here is UTC-based and defined explicitly, because "the London
session" means nothing until someone writes down the hours.  The panel shows
both UTC and IST, since that is the timezone the operator is in.

Sessions OVERLAP by design - the London/New York overlap is the busiest window
of the crypto day - so a candle can belong to more than one session.  Filtering
therefore means "the candle falls inside at least one allowed session", and the
breakdown attributes a trade to *every* session it touched, with the counting
rule stated on screen so the numbers cannot be misread as a partition.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

#: Session windows in UTC.  `end` is exclusive; windows may wrap past midnight.
SESSIONS: Dict[str, Dict[str, Any]] = {
    "asia": {
        "label": "Asia / Tokyo",
        "start": 0.0, "end": 8.0,
        "description": "The quietest of the three on BTC - thin books, wide spreads.",
    },
    "london": {
        "label": "London",
        "start": 7.0, "end": 16.0,
        "description": "European hours: volume picks up, trends often set the daily range.",
    },
    "newyork": {
        "label": "New York",
        "start": 12.0, "end": 21.0,
        "description": "US hours: the highest-volume window of the day for BTC.",
    },
    "overlap": {
        "label": "London + New York overlap",
        "start": 12.0, "end": 16.0,
        "description": "Where London and New York trade at the same time - usually the "
                       "most liquid four hours of the day.",
    },
    "offhours": {
        "label": "Off-hours (between sessions)",
        "start": 21.0, "end": 24.0,
        "description": "After New York closes and before Asia opens: the thinnest window.",
    },
}

#: Sessions a user can ENABLE as a trading window.  `offhours` is the complement
#: of the others and is offered as a window too (some traders deliberately avoid
#: the busy hours), but it is not part of the "all sessions" default.
SESSION_KEYS: List[str] = ["asia", "london", "newyork", "overlap", "offhours"]

WEEKDAYS: List[str] = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
                       "Saturday", "Sunday"]

#: A separate, explicitly-labelled window covering the whole weekend in UTC.
#: Bitcoin never closes, so "weekend" means "UTC Saturday and Sunday", which is
#: an objective statement rather than a market-hours claim.
WEEKEND_BUCKET = "weekend (UTC Sat+Sun)"


def utc_dt(ts_ms: float) -> datetime:
    return datetime.fromtimestamp(float(ts_ms) / 1000.0, timezone.utc)


def ist_dt(ts_ms: float) -> datetime:
    """The same instant in IST (UTC+05:30, no DST) - the operator's clock."""
    return utc_dt(ts_ms).astimezone(timezone(timedelta(hours=5, minutes=30)))


def hour_utc(ts_ms: float) -> float:
    """Hour of day in UTC as a float (13.5 = 13:30 UTC)."""
    dt = utc_dt(ts_ms)
    return dt.hour + dt.minute / 60.0 + dt.second / 3600.0


def sessions_at(ts_ms: float) -> List[str]:
    """
    Every session a timestamp falls inside, in catalogue order.

    Returns `["offhours"]` when it falls in none of the trading sessions, so the
    result is never empty and a caller never has to special-case it.
    """
    hour = hour_utc(ts_ms)
    found: List[str] = []
    for key in SESSION_KEYS:
        if key == "offhours":
            continue
        window = SESSIONS[key]
        if window["start"] <= hour < window["end"]:
            found.append(key)
    if not found:
        found.append("offhours")
    return found


def primary_session(ts_ms: float) -> str:
    """The most specific session for a timestamp (overlap wins over its parents)."""
    found = sessions_at(ts_ms)
    if "overlap" in found:
        return "overlap"
    for key in ("newyork", "london", "asia"):
        if key in found:
            return key
    return "offhours"


def weekday_utc(ts_ms: float) -> str:
    return WEEKDAYS[utc_dt(ts_ms).weekday()]


def is_weekend_utc(ts_ms: float) -> bool:
    return utc_dt(ts_ms).weekday() >= 5


def in_allowed_sessions(ts_ms: float, allowed: Optional[Iterable[str]]) -> bool:
    """
    Is this timestamp inside one of the allowed sessions?

    An EMPTY (or missing) list means "no filter" and returns True at every hour:
    switching the feature on without picking a window must never quietly stop
    trading.  Names that are not real sessions are ignored rather than treated as
    a block - `StrategyParams.validate()` is what rejects them, loudly, so a typo
    surfaces as a configuration error instead of a silent empty filter.
    """
    """
    True when the timestamp is inside at least one allowed session.

    `allowed` empty/None means NO time filter at all - every hour is allowed.
    That is the default, and it is why enabling this feature changes nothing
    until the user picks windows.
    """
    allowed = [s for s in (allowed or []) if s in SESSIONS]
    if not allowed:
        return True
    return any(s in allowed for s in sessions_at(ts_ms))


def session_label(key: str) -> str:
    return SESSIONS.get(key, {}).get("label", key)


def session_hours(key: str) -> str:
    """'07:00-16:00 UTC (= 12:30-21:30 IST)' for the session, for the UI."""
    window = SESSIONS.get(key)
    if not window:
        # An unknown key is a caller bug, not a session: say so instead of
        # returning a blank that could be rendered as "00:00-00:00".
        return f"{key}: unknown session"
    return f"{_hhmm(window['start'])}-{_hhmm(window['end'])} UTC (= {_ist_range(key)} IST)"


def _hhmm(hour_float: float) -> str:
    hour = int(hour_float) % 24
    minute = int(round((hour_float - int(hour_float)) * 60))
    return f"{hour:02d}:{minute:02d}"


def _ist_range(key: str) -> str:
    window = SESSIONS[key]
    def to_ist(hour: float) -> str:
        total = (hour + 5.5) % 24
        return f"{int(total):02d}:{int(round((total - int(total)) * 60)):02d}"
    return f"{to_ist(window['start'])}-{to_ist(window['end'])}"


# --------------------------------------------------------------------------- #
# Breakdown of realised trades by session and weekday
# --------------------------------------------------------------------------- #

def _trade_ts(trade: Dict[str, Any]) -> Optional[float]:
    for field in ("exit_ts", "entry_ts"):
        value = trade.get(field)
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if number == number and number > 0:          # not NaN
            return number
    return None


def _pnl(trade: Dict[str, Any]) -> Optional[float]:
    value = trade.get("net_pnl")
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number != number else number


def summarise_trades(trades: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Win rate / net P&L / average R for a group of trades."""
    pnls = [p for p in (_pnl(t) for t in trades) if p is not None]
    rs = []
    for t in trades:
        try:
            r = float(t.get("realized_r"))
        except (TypeError, ValueError):
            continue
        if r == r:
            rs.append(r)
    wins = sum(1 for p in pnls if p > 0)
    losses = sum(1 for p in pnls if p < 0)
    scratches = sum(1 for p in pnls if p == 0)
    return {
        "trades": len(pnls),
        "wins": wins,
        "losses": losses,
        "scratches": scratches,
        "win_rate": (wins / len(pnls)) if pnls else None,
        "net_pnl": sum(pnls) if pnls else 0.0,
        "avg_pnl": (sum(pnls) / len(pnls)) if pnls else None,
        "avg_r": (sum(rs) / len(rs)) if rs else None,
    }


def _bucket(trades: Sequence[Dict[str, Any]], key_fn) -> Dict[str, Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for trade in trades:
        ts = _trade_ts(trade)
        if ts is None:
            continue
        key = key_fn(ts)
        if isinstance(key, (list, tuple)):
            for k in key:
                groups.setdefault(k, []).append(trade)
        else:
            groups.setdefault(key, []).append(trade)
    return {k: summarise_trades(v) for k, v in groups.items()}


def by_session(trades: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Per-session statistics.  A trade spanning two sessions counts in BOTH.

    That is deliberate: the question "does this strategy work in the London
    session?" is answered by every trade that traded during London hours.  The
    UI states the counting rule so the columns are never read as a partition.
    """
    return _bucket(trades, sessions_at)


def by_weekday(trades: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return _bucket(trades, weekday_utc)


def by_session_and_weekday(trades: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Grouped the same way as `by_session`, but split weekend out explicitly."""
    def key(ts: float):
        return WEEKEND_BUCKET if is_weekend_utc(ts) else sessions_at(ts)
    return _bucket(trades, key)


def ordered_rows(stats: Dict[str, Dict[str, Any]], order: Optional[Sequence[str]] = None,
                 *, include_empty: bool = True) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Turn a stats dict into display rows in a stable, meaningful order.

    Empty buckets are included by default: "we never traded in the Asia session"
    is a finding, and hiding it would make the sample look bigger than it is.
    """
    keys: List[str] = []
    if order:
        keys.extend(k for k in order if include_empty or k in stats)
        keys.extend(sorted(k for k in stats if k not in keys))
    else:
        keys.extend(sorted(stats))
    return [(k, stats.get(k, summarise_trades([]))) for k in keys]


def sample_warning(groups: Dict[str, Dict[str, Any]], *, min_trades: int = 10) -> Optional[str]:
    """
    Plain-language warning when a breakdown is too small to conclude anything.

    This exists because a 3-trade session with 100% wins is not evidence, and a
    table that shows it without comment invites exactly that mistake.
    """
    sizes = [g["trades"] for g in groups.values() if g["trades"]]
    if not sizes:
        return None
    if len(sizes) > 1 and max(sizes) < min_trades:
        return (f"No group has more than {max(sizes)} trades. These figures describe "
                f"what happened, not what will happen - a handful of trades is noise.")
    if min(sizes) < min_trades:
        return (f"At least one group has fewer than {min_trades} trades "
                f"(smallest: {min(sizes)}). Treat its numbers as anecdote, not evidence.")
    return None


def describe_sessions(allowed: Optional[Sequence[str]]) -> str:
    """One-line summary of the active time filter, for the sidebar and reports."""
    allowed = [s for s in (allowed or []) if s in SESSIONS]
    if not allowed:
        return "Every hour is tradable (no time filter)."
    return "Trading only during: " + ", ".join(
        f"{session_label(s)} ({session_hours(s)})" for s in allowed)


def full_breakdown(trades: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Everything the per-session performance panel needs, from one trade list.

    Returns the overall summary, per-session rows, per-weekday rows, the
    session x weekend split, and a plain-language warning when the sample is too
    small to conclude anything.  Empty buckets are kept: "we never traded in the
    Asia session" is a finding, and dropping it would flatter the sample.

    Counting rule (stated in the UI too): a trade that spans two sessions counts
    in BOTH, because the question being asked is "how did trades that happened
    during London hours do?" - not "partition the P&L".
    """
    trades = [t for t in (trades or []) if t]
    sessions_stats = by_session(trades)
    weekday_stats = by_weekday(trades)
    split_stats = by_session_and_weekday(trades)
    return {
        "overall": summarise_trades(trades),
        "sessions": ordered_rows(sessions_stats, SESSION_KEYS),
        "weekdays": ordered_rows(weekday_stats, WEEKDAYS),
        "session_weekend": ordered_rows(
            split_stats, list(SESSION_KEYS) + [WEEKEND_BUCKET]),
        "warning": sample_warning(sessions_stats),
        "weekday_warning": sample_warning(weekday_stats),
        "counting_rule": ("A trade that spans two sessions is counted in both. The "
                          "columns therefore do not add up to the total."),
        "n_trades": len(trades),
    }


def breakdown_rows(stats: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Format a stats dict as display rows for a table."""
    rows: List[Dict[str, Any]] = []
    for key, g in ordered_rows(stats):
        rows.append({
            "Session": session_label(key) if key in SESSIONS else key,
            "Trades": str(g["trades"]),
            "Wins": str(g["wins"]),
            "Losses": str(g["losses"]),
            "Win %": "-" if g["win_rate"] is None else f"{g['win_rate'] * 100:.1f}%",
            "Net PnL": f"{g['net_pnl']:+,.2f}" if g["trades"] else "-",
            "Avg R": "-" if g["avg_r"] is None else f"{g['avg_r']:+.2f}R",
        })
    return rows


__all__ = [
    "SESSIONS", "SESSION_KEYS", "WEEKDAYS", "WEEKEND_BUCKET",
    "utc_dt", "ist_dt", "hour_utc", "sessions_at", "primary_session",
    "weekday_utc", "is_weekend_utc", "in_allowed_sessions", "session_label",
    "full_breakdown", "breakdown_rows",
    "session_hours", "summarise_trades", "by_session", "by_weekday",
    "by_session_and_weekday", "ordered_rows", "sample_warning", "describe_sessions",
]
