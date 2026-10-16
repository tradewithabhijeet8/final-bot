"""
Public-market dashboard helpers: 24h stats, ranked movers, 30-day performance.

Every number in this module comes from a PUBLIC endpoint of the selected
exchange (no API key, no account, no order).  Three rules are enforced here so
the UI cannot accidentally break them:

1. **Nothing is invented.**  A missing field stays ``None`` and is rendered as
   ``-``; a failed symbol keeps its failure note instead of borrowing another
   symbol's price.
2. **Failures are visible.**  ``fetch_ticker_status`` never raises: it returns a
   status object with ``ok=False`` and a plain-language ``note`` that the UI
   shows next to the affected row.
3. **A market statistic is not a performance claim.**  24h change, funding and
   mark price describe the *market*; the account numbers come from the paper
   ledger (core/metrics.py, core/engine.py) and are never mixed in here.

The ticker snapshot is cached in a module-level TTL cache (like the markets
cache in core/data_feed.py) so a fast auto-refreshing UI still produces at most
one request per symbol per ``TICKER_TTL_S`` unless the user clicks refresh.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .config import (
    MONTHLY_WINDOW_DAYS,
    MOVERS_BASES,
    MOVERS_ROWS,
    TICKER_TTL_S,
)
from .data_feed import MarketDataFeed

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Module-level TTL caches (cleared by tests and by the UI's refresh button)
# --------------------------------------------------------------------------- #

#: key -> (fetched_at_seconds, TickerStatus)
_TICKER_CACHE: Dict[str, Tuple[float, "TickerStatus"]] = {}

#: key -> (fetched_at_seconds, list[TickerStatus])
_MOVERS_CACHE: Dict[str, Tuple[float, List["TickerStatus"]]] = {}

#: Quotes we compare side by side.  USDT and USD are shown separately because
#: they are different instruments - a USDT perpetual is not a USD spot pair.
_QUOTES = ("USDT", "USD")


def clear_caches() -> None:
    """Forget every cached public snapshot (used by "Refresh now" and tests)."""
    _TICKER_CACHE.clear()
    _MOVERS_CACHE.clear()


def cache_age_s() -> Optional[float]:
    """Seconds since the newest cached snapshot, or None when nothing cached."""
    stamps = [t for t, _ in _TICKER_CACHE.values()] + [t for t, _ in _MOVERS_CACHE.values()]
    return (time.time() - max(stamps)) if stamps else None


# --------------------------------------------------------------------------- #
# Number formatting (kept here so the UI and the tests share one definition)
# --------------------------------------------------------------------------- #

def compact_number(value: Optional[float], digits: int = 2) -> str:
    """1234567.0 -> '1.23M'.  Falls back to a plain number below 1,000."""
    if value is None:
        return "-"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "-"
    sign = "-" if v < 0 else ""
    v = abs(v)
    for limit, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if v >= limit:
            return f"{sign}{v / limit:,.{digits}f}{suffix}"
    return f"{sign}{v:,.{digits}f}"


def format_price(value: Optional[float], digits: int = 2) -> str:
    return "-" if value is None else f"{float(value):,.{digits}f}"


def format_pct(value: Optional[float], digits: int = 2, signed: bool = True) -> str:
    """`value` is already in PERCENT units (ccxt's `percentage` field)."""
    if value is None:
        return "-"
    return f"{float(value):{'+' if signed else ''}.{digits}f}%"


# --------------------------------------------------------------------------- #
# 24h ticker snapshot
# --------------------------------------------------------------------------- #

@dataclass
class TickerStatus:
    """One symbol's public 24h snapshot.  Never raises; check `.ok`."""

    exchange_key: str
    symbol: str
    ok: bool = False
    last: Optional[float] = None
    change_abs: Optional[float] = None
    change_pct: Optional[float] = None          # percent units
    high: Optional[float] = None
    low: Optional[float] = None
    quote_volume: Optional[float] = None
    base_volume: Optional[float] = None
    funding_rate: Optional[float] = None        # fraction per funding interval
    mark_price: Optional[float] = None
    fetched_at_ms: float = 0.0
    note: str = ""
    #: "" (unknown), "24h" (the venue's own rolling-24h ticker statistics) or
    #: "day" (derived locally from the venue's daily candles, i.e. change since
    #: 00:00 UTC - a different definition, so the UI labels it differently and
    #: never ranks the two kinds together).
    change_kind: str = ""

    # -- derived ---------------------------------------------------------- #
    @property
    def base(self) -> str:
        return self.symbol.split("/")[0]

    @property
    def is_perp(self) -> bool:
        return ":" in self.symbol

    @property
    def tone(self) -> str:
        """UI tone for the 24h move: 'neu' when it is unknown or exactly flat."""
        if self.change_pct is None or not self.change_pct:
            return "neu"
        return "pos" if self.change_pct > 0 else "neg"

    @property
    def change_label(self) -> str:
        """
        What the change figure actually IS.

        A rolling-24h ticker statistic and a "move since the day opened" figure
        derived from daily candles are different measurements, so they are never
        given the same label.
        """
        return "today (UTC)" if self.change_kind == "day" else "24h change"

    @property
    def arrow(self) -> str:
        if self.change_pct is None or not self.change_pct:
            return "="
        return "▲" if self.change_pct > 0 else "▼"

    def stats_rows(self) -> List[Tuple[str, str, str]]:
        """(label, value, tone) for the live 24h strip above the tabs."""
        change_text = ("-" if self.change_pct is None
                       else f"{self.arrow} {format_pct(self.change_pct)}")
        return [
            (self.change_label, change_text, self.tone),
            ("24h high", format_price(self.high), "neu"),
            ("24h low", format_price(self.low), "neu"),
            ("24h volume", f"{format_price(self.base_volume, 4)} {self.base}" if self.base_volume
             else compact_number(self.quote_volume), "neu"),
            ("Funding (8h)", ("-" if self.funding_rate is None
                              else f"{self.funding_rate * 100:.4f}%"),
             "neu"),
            ("Mark price", format_price(self.mark_price), "neu"),
        ]

    def table_row(self) -> List[Tuple[str, str]]:
        """One row of the movers/watch tables: (text, tone) pairs."""
        vol = (compact_number(self.quote_volume) if self.quote_volume
               else (compact_number(self.base_volume) if self.base_volume else "-"))
        pct = format_pct(self.change_pct) if self.change_pct is not None else "-"
        mark = "\u00b9" if self.change_kind == "day" else ""     # footnote glyph
        return [
            (self.base + mark, "sym"),
            (format_price(self.last), "neu"),
            (f"{self.arrow} {pct}" if self.change_pct is not None else "-", self.tone),
            (vol, "neu"),
        ]


def _to_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if out != out else out          # NaN -> None


def _derive_from_daily_candle(ex, symbol: str) -> Tuple[Dict[str, Any], str]:
    """
    Fallback for venues whose ticker carries only a last price (Coinbase).

    Reads the venue's own DAILY candles - real data, no invented numbers - and
    returns high/low/volume plus the move since the day opened.  That is
    explicitly a *different* definition from a rolling 24h change, so callers
    label it ("today (UTC)") and never rank it beside rolling-24h figures.
    """
    try:
        rows = ex.fetch_ohlcv(symbol, "1d", None, 2) or []
    except Exception as exc:                                          # noqa: BLE001
        log.info("daily-candle fallback failed for %s: %s", symbol, exc)
        return {}, ""
    if not rows:
        return {}, ""
    _ts, open_, high, low, close, volume = (list(rows[-1]) + [None] * 6)[:6]
    out: Dict[str, Any] = {"high": _to_float(high), "low": _to_float(low),
                           "base_volume": _to_float(volume)}
    o, c = _to_float(open_), _to_float(close)
    last = _to_float(close)
    if o and last is not None:
        out["change_pct"] = (last - o) / o * 100.0
        out["change_abs"] = last - o
    return out, "day"


def fetch_ticker_status(feed: MarketDataFeed, exchange_key: str, symbol: str,
                        *, force: bool = False,
                        max_age_s: float = TICKER_TTL_S,
                        derive_from_candles: bool = True) -> TickerStatus:
    """
    Public 24h snapshot for one symbol, cached for `max_age_s` seconds.

    Delta's raw ticker distinguishes `mark_price` (the price used for
    liquidations/funding) from `last` (last traded).  Both are shown, labelled,
    because on a perpetual they genuinely differ.

    When the venue's ticker publishes last price only (`percentage` missing,
    which is the case for Coinbase Exchange), `derive_from_candles` fills
    high/low/volume/change from the venue's daily candles and flags the result
    with `change_kind="day"` so the UI can label it instead of passing it off
    as a rolling 24h figure.  Set `derive_from_candles=False` to skip that
    extra request.
    """
    key = f"{exchange_key}|{symbol}"
    now = time.time()
    if not force:
        hit = _TICKER_CACHE.get(key)
        if hit and (now - hit[0]) <= max_age_s:
            return hit[1]

    status = TickerStatus(exchange_key=exchange_key, symbol=symbol,
                          fetched_at_ms=now * 1000.0)
    try:
        ex = feed.get_exchange(exchange_key)
        ticker = ex.fetch_ticker(symbol) or {}
        status.last = _to_float(ticker.get("last"))
        status.change_abs = _to_float(ticker.get("change"))
        status.change_pct = _to_float(ticker.get("percentage"))
        status.high = _to_float(ticker.get("high"))
        status.low = _to_float(ticker.get("low"))
        status.quote_volume = _to_float(ticker.get("quoteVolume"))
        status.base_volume = _to_float(ticker.get("baseVolume"))
        info = ticker.get("info") or {}
        status.mark_price = _to_float(info.get("mark_price")) or _to_float(
            ticker.get("markPrice"))
        status.ok = status.last is not None
        if status.change_pct is not None:
            status.change_kind = "24h"

        # Venue ticker has no 24h statistics (Coinbase): fill them from the
        # venue's daily candles, clearly flagged as a different definition.
        if (derive_from_candles and status.ok and status.change_pct is None
                and (getattr(ex, "has", {}) or {}).get("fetchOHLCV")):
            derived, kind = _derive_from_daily_candle(ex, symbol)
            if derived:
                status.change_kind = kind
                for field, value in derived.items():
                    if getattr(status, field) is None and value is not None:
                        setattr(status, field, value)
                status.note = (status.note + " ").strip() + (
                    "This venue's public ticker publishes no 24h statistics; the "
                    "high/low/volume/change shown are taken from its daily candle "
                    "(change since 00:00 UTC).")

        # Funding exists on perpetuals only.  It comes from the canonical
        # `fetch_funding_rate` (a FRACTION per interval) rather than the raw
        # ticker field, whose units differ between venues.
        if status.is_perp and (getattr(ex, "has", {}) or {}).get("fetchFundingRate"):
            try:
                fr = ex.fetch_funding_rate(symbol) or {}
                status.funding_rate = _to_float(fr.get("fundingRate"))
                if status.mark_price is None:
                    status.mark_price = _to_float(fr.get("markPrice"))
            except Exception as exc:                                   # noqa: BLE001
                status.note = (f"Funding rate unavailable from {exchange_key} "
                               f"({type(exc).__name__}). Everything else is live.")
                log.info("funding fetch failed for %s: %s", symbol, exc)
        elif status.is_perp:
            status.note = "This venue does not publish a funding rate."

        if not status.ok and not status.note:
            status.note = ("The exchange returned no last price for this symbol "
                           "(it may be delisted or have no trades).")
    except Exception as exc:                                          # noqa: BLE001
        # A market-statistics failure must never break the trading page.
        status.ok = False
        status.note = f"{type(exc).__name__}: {exc}"
        log.info("ticker fetch failed for %s %s: %s", exchange_key, symbol, exc)

    _TICKER_CACHE[key] = (now, status)
    return status


# --------------------------------------------------------------------------- #
# Multi-symbol "Market overview"
# --------------------------------------------------------------------------- #

def _is_derivative(market: Dict[str, Any]) -> bool:
    """
    True for perpetuals and dated futures - and False for OPTIONS.

    This matters on Delta: its option markets also carry base='BTC',
    quote='USDT' and `contract=True`, so a naive check ranks
    "BTC/USDT:USDT-261018-83000-P" (a put with no price) next to the perpetual.
    """
    if market.get("option"):
        return False
    return bool(market.get("swap") or market.get("future"))


def movers_universe(feed: MarketDataFeed, exchange_key: str,
                    *, include: Optional[str] = None,
                    prefer_quote: Optional[str] = None,
                    limit: int = 10) -> List[str]:
    """
    Which symbols the Market-overview panel compares, on THIS venue.

    Only instruments the venue really lists are returned, never a wish-list:

    * derivative markets win over spot for the same base, because on Delta the
      spot BTC/USDT book is dead (no candles, no bid/ask, no volume) - showing it
      beside the perpetual would put a permanently blank row in the table;
    * options are excluded (see `_is_derivative`);
    * ONE quote per base: the quote being traded now wins, then USD, then USDT,
      so the table never lists the same coin twice;
    * `include` (the symbol currently being traded) is always kept, so the
      selected market never disappears from its own comparison;
    * `limit` caps the panel at a readable size - a venue with hundreds of pairs
      still shows a table, not a wall.
    """
    try:
        ex = feed.get_exchange(exchange_key)
        # The registry must be loaded before it can be filtered.  `_load_markets`
        # is the feed's own cached loader (one /products request, TTL-cached), so
        # this costs nothing on subsequent reruns.  Without it the filter would
        # silently return an empty table.
        if not getattr(ex, "markets", None):
            feed._load_markets(ex, exchange_key)                      # noqa: SLF001
        markets = getattr(ex, "markets", {}) or {}
    except Exception as exc:                                          # noqa: BLE001
        log.info("markets unavailable for %s: %s", exchange_key, exc)
        return [include] if include else []

    quotes: List[str] = []
    for q in (prefer_quote, "USD", "USDT"):
        if q in _QUOTES and q not in quotes:
            quotes.append(q)

    per_base: Dict[str, Dict[str, str]] = {}
    for _sym, market in markets.items():
        market = market or {}
        if not market.get("active", True):
            continue
        base, quote = market.get("base"), market.get("quote")
        if base not in MOVERS_BASES or quote not in _QUOTES:
            continue
        slot = per_base.setdefault(base, {})
        rank = "deriv" if _is_derivative(market) else "spot"
        slot[f"{rank}|{quotes.index(quote)}"] = market.get("symbol") or _sym

    symbols: List[str] = []
    # MOVERS_BASES order keeps the table stable between refreshes (it never
    # reshuffles because a price changed), then anything else the venue lists.
    for base in list(MOVERS_BASES) + sorted(b for b in per_base if b not in MOVERS_BASES):
        slot = per_base.get(base)
        if not slot:
            continue
        for key in sorted(slot, key=lambda k: (not k.startswith("deriv"), k)):
            symbols.append(slot[key])
            break

    if include and include not in symbols:
        symbols.insert(0, include)
    if len(symbols) > limit:
        keep = [s for s in symbols if s == include][:1] + [s for s in symbols if s != include][:limit - 1]
        symbols = keep[:limit]
    return symbols


def fetch_movers(feed: MarketDataFeed, exchange_key: str, symbols: Sequence[str],
                 *, force: bool = False) -> List[TickerStatus]:
    """Snapshot every symbol; a symbol that fails is returned with `ok=False`."""
    key = f"{exchange_key}|{','.join(symbols)}"
    now = time.time()
    if not force:
        hit = _MOVERS_CACHE.get(key)
        if hit and (now - hit[0]) <= TICKER_TTL_S:
            return hit[1]
    rows = [fetch_ticker_status(feed, exchange_key, sym, force=force) for sym in symbols]
    _MOVERS_CACHE[key] = (now, rows)
    return rows


def rank_movers(rows: Sequence[TickerStatus],
                limit: int = MOVERS_ROWS) -> Dict[str, List[TickerStatus]]:
    """
    Split a snapshot list into the three ranked columns of the panel.

    Only rows that actually carry the ranking field take part: a symbol whose
    24h volume is unknown is not quietly ranked as zero, and ranking uses the
    last known value of the others.
    """
    volume = [r for r in rows if r.ok and r.quote_volume is not None]
    movers = [r for r in rows if r.ok and r.change_pct is not None]
    return {
        "volume": sorted(volume, key=lambda r: r.quote_volume or 0.0, reverse=True)[:limit],
        "gainers": sorted(movers, key=lambda r: r.change_pct or 0.0, reverse=True)[:limit],
        "losers": sorted(movers, key=lambda r: r.change_pct or 0.0)[:limit],
    }


def panel_mode(rows: Sequence[TickerStatus]) -> str:
    """
    'ranked' when the venue publishes real rolling-24h statistics for at least
    two symbols, otherwise 'watchlist'.

    Ranking a derived "change since 00:00 UTC" figure next to a rolling-24h one
    would compare two different things in one column, so on a venue such as
    Coinbase the panel shows a plainly-labelled watch list instead.
    """
    real = [r for r in rows if r.ok and r.change_kind == "24h" and r.change_pct is not None]
    return "ranked" if len(real) >= 2 else "watchlist"


def watchlist_rows(rows: Sequence[TickerStatus]) -> List[List[Tuple[str, str]]]:
    """Last-price table for venues without published 24h statistics."""
    return [r.table_row() for r in rows if r.ok]


def unavailable_note(rows: Sequence[TickerStatus]) -> Optional[str]:
    """Plain-language note naming symbols this venue could not price."""
    bad = [r for r in rows if not r.ok]
    if not bad:
        return None
    names = ", ".join(r.symbol for r in bad[:4])
    more = "" if len(bad) <= 4 else f" (+{len(bad) - 4} more)"
    return (f"{len(bad)} of {len(rows)} symbol(s) returned no public price on this "
            f"venue and are omitted: {names}{more}.")


def funding_note(status: TickerStatus) -> str:
    """Plain-language funding line, or a reason why it is unavailable."""
    if status.funding_rate is None:
        return status.note or "Funding rate is only published for perpetual markets."
    rate = status.funding_rate * 100.0
    direction = "longs pay shorts" if rate > 0 else ("shorts pay longs" if rate < 0 else "neither side pays")
    return (
        f"Funding {rate:+.4f}% per interval ({direction}). A perpetual has no expiry, so "
        f"funding keeps its price near the index. **Paper fills in this app are price-only: "
        f"funding and margin are not simulated**, so a long position held across a funding "
        f"time does not pay it here."
    )


# --------------------------------------------------------------------------- #
# 30-day realised performance (paper ledger only)
# --------------------------------------------------------------------------- #

@dataclass
class MonthlyStats:
    """Realised statistics over a rolling window of closed paper trades."""

    days: int = MONTHLY_WINDOW_DAYS
    trades: int = 0
    wins: int = 0
    losses: int = 0
    scratches: int = 0
    total_pnl: float = 0.0
    total_return_pct: Optional[float] = None      # % of the starting balance
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    best_trade: Optional[float] = None
    worst_trade: Optional[float] = None
    starting_balance: float = 0.0
    cutoff_ms: float = 0.0

    @property
    def has_trades(self) -> bool:
        return self.trades > 0

    @property
    def win_rate_pct(self) -> Optional[float]:
        if not self.trades:
            return None
        return self.wins / self.trades * 100.0

    @property
    def ring_value(self) -> float:
        """Clamped percentage used by the circular return ring."""
        if self.total_return_pct is None:
            return 0.0
        return max(-100.0, min(100.0, self.total_return_pct))

    def table_rows(self) -> List[Tuple[str, str]]:
        """(label, value) rows for the small table under the ring."""
        n = lambda v: "-" if v is None else f"{v:,.2f}"          # noqa: E731
        return [
            ("Total trades", f"{self.trades}"),
            ("Profitable trades", f"{self.wins}"),
            ("Losing trades", f"{self.losses}"),
            ("Best trade", n(self.best_trade)),
            ("Worst trade", n(self.worst_trade)),
            ("Win rate", "-" if self.win_rate_pct is None else f"{self.win_rate_pct:.1f}%"),
            ("Max drawdown", f"{self.max_drawdown_pct:.2f}%"),
        ]

    def caption(self) -> str:
        if not self.has_trades:
            return (f"No closed paper trades in the last {self.days} days, so the ring "
                    f"shows 0.00% rather than a made-up figure.")
        return (f"{self.trades} closed paper trade(s) in the last {self.days} days. "
                f"Return is realised P/L divided by the starting balance; it excludes open "
                f"positions and is not a prediction.")


def monthly_stats(trades: Sequence[Dict[str, Any]], starting_balance: float,
                  *, days: int = MONTHLY_WINDOW_DAYS,
                  now_ms: Optional[float] = None) -> MonthlyStats:
    """
    Realised statistics for closed trades that exited inside the window.

    Only trades with a resolved `net_pnl` and an `exit_ts` inside the window
    count.  Open trades are excluded (their P/L is an estimate), and the equity
    path used for drawdown starts at the starting balance, so a losing streak
    early in the window is not hidden by profit that came before it.
    """
    now = float(now_ms if now_ms is not None else time.time() * 1000.0)
    cutoff = now - days * 86_400_000.0
    out = MonthlyStats(days=days, starting_balance=float(starting_balance or 0.0),
                       cutoff_ms=cutoff)

    rows: List[Tuple[float, float]] = []
    for t in trades or []:
        if str(t.get("status") or "").lower() != "closed":
            continue
        exit_ts = _to_float(t.get("exit_ts"))
        pnl = _to_float(t.get("net_pnl"))
        if exit_ts is None or pnl is None or exit_ts < cutoff:
            continue
        rows.append((exit_ts, pnl))

    if not rows:
        return out

    rows.sort(key=lambda r: r[0])
    equity = out.starting_balance
    peak = equity
    worst_dd = 0.0
    worst_dd_pct = 0.0
    for _ts, pnl in rows:
        equity += pnl
        peak = max(peak, equity)
        dd = peak - equity
        if dd > worst_dd:
            worst_dd = dd
            worst_dd_pct = (dd / peak * 100.0) if peak else 0.0

    pnls = [p for _t, p in rows]
    out.trades = len(pnls)
    out.wins = sum(1 for p in pnls if p > 0)
    out.losses = sum(1 for p in pnls if p < 0)
    out.scratches = sum(1 for p in pnls if p == 0)
    out.total_pnl = sum(pnls)
    out.best_trade = max(pnls)
    out.worst_trade = min(pnls)
    out.max_drawdown = worst_dd
    out.max_drawdown_pct = worst_dd_pct
    out.total_return_pct = ((out.total_pnl / out.starting_balance) * 100.0
                            if out.starting_balance else None)
    return out


__all__ = [
    "TickerStatus", "MonthlyStats", "panel_mode", "watchlist_rows", "unavailable_note",
    "clear_caches", "cache_age_s",
    "fetch_ticker_status", "fetch_movers", "movers_universe", "rank_movers",
    "funding_note", "monthly_stats",
    "compact_number", "format_price", "format_pct",
]
