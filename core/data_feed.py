"""
core/data_feed.py
=================
Public market-data engine (CCXT) with the defensive behaviour the spec demands.

Responsibilities
----------------
1.  Build/cache CCXT exchange objects (public data only - never any API key).
2.  Verify an exchange is reachable AND actually serves the requested symbol
    and timeframe *before* the dashboard pretends to be live.
3.  Fetch OHLCV with timeouts, bounded retries and exponential backoff + jitter.
4.  Paginate when more candles are requested than one call can return.
5.  Aggregate 1h -> 4h for exchanges with no native 4h (Coinbase), and label it.
6.  Validate every candle (numeric, ordered, high/low sane), de-duplicate and
    sort.  Gaps are *reported*, never filled with invented prices.
7.  Split CLOSED candles from the still-forming candle.  Strategy code only ever
    receives closed candles.
8.  Track staleness, keep the last known-good frame for display, and trip a
    circuit breaker so a hard regional block (HTTP 451) is not hammered.

Nothing in this module imports Streamlit, so it can be unit-tested headlessly.
"""

from __future__ import annotations

import logging
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import errors as E
from .config import (
    EXCHANGES,
    TIMEFRAME_MS,
    FeedConfig,
    STALE_MULTIPLIER,
    get_exchange_spec,
)
from .indicators import OHLCV_COLUMNS, candles_are_complete, find_missing_candles

log = logging.getLogger("btc_feed")

# --------------------------------------------------------------------------- #
# Module-level caches (survive Streamlit reruns; intentionally NOT permanent)
# --------------------------------------------------------------------------- #

_EXCHANGE_CACHE: Dict[str, Any] = {}
_MARKETS_CACHE: Dict[str, Tuple[float, int]] = {}       # key -> (loaded_at, count)
_TIME_OFFSET_CACHE: Dict[str, Tuple[float, float]] = {}  # key -> (offset_ms, cached_at)
_CIRCUIT: Dict[str, Dict[str, float]] = {}
_LAST_GOOD: Dict[str, "CandleSet"] = {}

#: Metadata (markets list) is cached this long.  Price data is NEVER cached here.
MARKETS_TTL_S = 600.0
#: Clock-offset cache lifetime.
TIME_OFFSET_TTL_S = 120.0


def clear_caches() -> None:
    """Drop every cache.  Used by the 'Connection diagnostics' button and tests."""
    _EXCHANGE_CACHE.clear()
    _MARKETS_CACHE.clear()
    _TIME_OFFSET_CACHE.clear()
    _CIRCUIT.clear()
    _LAST_GOOD.clear()


# --------------------------------------------------------------------------- #
# Result containers
# --------------------------------------------------------------------------- #

@dataclass
class CandleSet:
    """Validated candles plus the metadata needed to display them honestly."""

    exchange_key: str
    symbol: str
    timeframe: str
    df: pd.DataFrame                                   # CLOSED candles only
    forming: Optional[Dict[str, float]] = None         # still-forming candle
    fetched_at_ms: float = 0.0
    server_time_ms: float = 0.0
    is_aggregated: bool = False
    source_timeframe: Optional[str] = None
    missing_count: int = 0
    gaps: List[float] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    served_from_cache: bool = False
    candle_count: int = 0
    last_closed_ts: Optional[float] = None
    requests_made: int = 0
    # -- data quality (thin venues such as Delta's BTC perpetual) ----------- #
    #: closed candles that reported zero volume, i.e. no trades at all
    zero_volume_count: int = 0
    zero_volume_share: float = 0.0
    #: closed candles where open == high == low == close (an empty bucket)
    flat_candle_count: int = 0
    flat_candle_share: float = 0.0
    thin_venue: bool = False

    # -- derived ---------------------------------------------------------- #
    @property
    def timeframe_ms(self) -> int:
        return TIMEFRAME_MS.get(self.timeframe, 3_600_000)

    @property
    def reference_now_ms(self) -> float:
        return self.server_time_ms or self.fetched_at_ms or time.time() * 1000.0

    @property
    def age_seconds(self) -> float:
        """Seconds since the most recent closed candle actually closed."""
        if self.last_closed_ts is None:
            return float("inf")
        closed_at = self.last_closed_ts + self.timeframe_ms
        return max(0.0, (self.reference_now_ms - closed_at) / 1000.0)

    @property
    def data_age_seconds(self) -> float:
        """Seconds since we successfully fetched (0 if this frame is fresh)."""
        if not self.fetched_at_ms:
            return float("inf")
        return max(0.0, (time.time() * 1000.0 - self.fetched_at_ms) / 1000.0)

    @property
    def is_stale(self) -> bool:
        return self.age_seconds > STALE_MULTIPLIER * self.timeframe_ms / 1000.0

    @property
    def is_empty(self) -> bool:
        return self.df is None or self.df.empty

    @property
    def data_quality_note(self) -> Optional[str]:
        """
        Plain-English description of how much of this series had no trades.

        Returns None when the data looks normal, so healthy venues (Coinbase,
        Kraken) never show a banner.  Nothing is invented here: the shares are
        measured from the candles the exchange actually returned.
        """
        return describe_candle_quality(
            self.timeframe, self.candle_count, self.zero_volume_count,
            self.zero_volume_share, self.flat_candle_count, self.flat_candle_share,
        )

    def safe_for_trading(self, max_age_seconds: Optional[float] = None) -> Tuple[bool, str]:
        """
        Gatekeeper used before any simulated order is created.

        Returns (ok, reason).  Stale or cached-only data may be *displayed* but
        must never open a trade.
        """
        if self.is_empty:
            return False, "No candle data available."
        limit = max_age_seconds if max_age_seconds is not None else (
            STALE_MULTIPLIER * self.timeframe_ms / 1000.0
        )
        if self.served_from_cache:
            return False, (
                "Data is being served from the last successful fetch "
                f"({self.data_age_seconds:.0f}s old); live feed is down."
            )
        if self.age_seconds > limit:
            return False, (
                f"Last closed candle is {self.age_seconds:.0f}s old "
                f"(limit {limit:.0f}s). Market data looks stale."
            )
        if self.missing_count > 0:
            # Gaps are tolerated for display but flagged; trading continues only
            # if the most recent candle itself is present and fresh.
            log.warning("Candle gaps present: %s missing", self.missing_count)
        return True, "ok"


def measure_candle_quality(df: pd.DataFrame) -> Tuple[int, float, int, float]:
    """
    Measure how much of a CLOSED-candle frame actually contains trades.

    Returns (zero_volume_count, zero_volume_share, flat_count, flat_share).
    A "flat" candle has open == high == low == close, which on a thin venue
    means the exchange emitted a placeholder bucket because nothing traded.
    """
    if df is None or df.empty:
        return 0, 0.0, 0, 0.0
    n = len(df)
    vol = pd.to_numeric(df.get("volume"), errors="coerce").fillna(0.0).to_numpy(dtype="float64")
    o = pd.to_numeric(df["open"], errors="coerce").to_numpy(dtype="float64")
    h = pd.to_numeric(df["high"], errors="coerce").to_numpy(dtype="float64")
    l = pd.to_numeric(df["low"], errors="coerce").to_numpy(dtype="float64")
    c = pd.to_numeric(df["close"], errors="coerce").to_numpy(dtype="float64")
    zv = int((vol <= 0.0).sum())
    flat = int(((o == h) & (h == l) & (l == c)).sum())
    return zv, zv / n, flat, flat / n


def describe_candle_quality(timeframe: str, candle_count: int, zero_volume_count: int,
                            zero_volume_share: float, flat_count: int,
                            flat_share: float) -> Optional[str]:
    """
    Plain-English description of how much of a series had no trades at all.

    Returns None when the data looks normal, so healthy venues (Coinbase,
    Kraken) never raise a banner.  Every number in the text is measured from
    the candles the exchange actually returned - nothing is estimated.
    """
    if candle_count <= 0 or (zero_volume_count == 0 and flat_count == 0):
        return None
    parts = []
    if zero_volume_count:
        parts.append(
            f"{zero_volume_count} of {candle_count} closed {timeframe} candles "
            f"({zero_volume_share * 100:.0f}%) had ZERO volume - no trade printed "
            f"in that bucket at all"
        )
    if flat_count:
        parts.append(
            f"{flat_count} ({flat_share * 100:.0f}%) were completely flat "
            f"(open = high = low = close)"
        )
    return (
        "Thin market data on this venue: " + "; ".join(parts) + ". "
        "Indicators computed over empty buckets are not meaningful, and no signal "
        "is ever generated from one. Use a higher timeframe, or switch to "
        "Coinbase/Kraken for liquid spot BTC."
    )


@dataclass
class FeedStatus:
    """Result of an explicit connectivity / capability probe."""

    exchange_key: str
    ok: bool
    stage: str = "init"
    message: str = ""
    recovery: List[str] = field(default_factory=list)
    category: str = ""
    http_status: Optional[int] = None
    latency_ms: float = 0.0
    markets_loaded: int = 0
    symbol: Optional[str] = None
    symbol_supported: Optional[bool] = None
    timeframe: Optional[str] = None
    timeframe_native: Optional[bool] = None
    server_time_ms: Optional[float] = None
    clock_skew_ms: Optional[float] = None
    sample_price: Optional[float] = None
    detail: str = ""
    checked_at_ms: float = field(default_factory=lambda: time.time() * 1000.0)

    def to_dict(self) -> Dict[str, Any]:
        d = self.__dict__.copy()
        d["recovery"] = list(self.recovery)
        return d


# --------------------------------------------------------------------------- #
# Circuit breaker
# --------------------------------------------------------------------------- #

class _CircuitBreaker:
    """
    Stops us re-hammering an exchange that has already refused us.

    A regional HTTP 451 will not heal on the next rerun, so after
    `threshold` consecutive hard failures we refuse to even attempt a network
    call for `cooldown_s` seconds and return the previous diagnosis instead.
    """

    def __init__(self, threshold: int = 3, cooldown_s: float = 300.0):
        self.threshold = threshold
        self.cooldown_s = cooldown_s

    def _state(self, key: str) -> Dict[str, float]:
        return _CIRCUIT.setdefault(key, {"failures": 0.0, "open_until": 0.0})

    def is_open(self, key: str) -> bool:
        st = self._state(key)
        return time.time() < st["open_until"]

    def seconds_until_closed(self, key: str) -> float:
        st = self._state(key)
        return max(0.0, st["open_until"] - time.time())

    def record_failure(self, key: str, hard: bool) -> None:
        st = self._state(key)
        st["failures"] += 1
        if hard or st["failures"] >= self.threshold:
            st["open_until"] = time.time() + self.cooldown_s
            log.warning("Circuit breaker OPEN for %s for %.0fs", key, self.cooldown_s)

    def record_success(self, key: str) -> None:
        if key in _CIRCUIT:
            _CIRCUIT[key] = {"failures": 0.0, "open_until": 0.0}

    def reset(self, key: Optional[str] = None) -> None:
        if key is None:
            _CIRCUIT.clear()
        else:
            _CIRCUIT.pop(key, None)


# --------------------------------------------------------------------------- #
# The feed
# --------------------------------------------------------------------------- #

class MarketDataFeed:
    """Public market-data engine."""

    def __init__(
        self,
        request_timeout_ms: int = 15_000,
        max_retries: int = 3,
        backoff_base_s: float = 0.8,
        backoff_cap_s: float = 8.0,
        enable_rate_limit: bool = True,
        circuit_breaker_threshold: int = 3,
        circuit_breaker_cooldown_s: float = 300.0,
        sleep_func=time.sleep,
    ):
        self.request_timeout_ms = int(request_timeout_ms)
        self.max_retries = int(max_retries)
        self.backoff_base_s = float(backoff_base_s)
        self.backoff_cap_s = float(backoff_cap_s)
        self.enable_rate_limit = bool(enable_rate_limit)
        self.sleep = sleep_func
        self.breaker = _CircuitBreaker(circuit_breaker_threshold, circuit_breaker_cooldown_s)

    # -- exchange objects ------------------------------------------------- #

    def get_exchange(self, exchange_key: str):
        """Return a cached CCXT instance configured for PUBLIC data only."""
        spec = get_exchange_spec(exchange_key)
        cached = _EXCHANGE_CACHE.get(spec.ccxt_id)
        if cached is not None:
            return cached
        try:
            import ccxt  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise E.DataFeedError(E.classify_exception(exc)) from exc

        cls = getattr(ccxt, spec.ccxt_id, None)
        if cls is None:
            raise E.DataFeedError(E.FeedError(
                category=E.CAT_DEPENDENCY,
                message=f"The installed ccxt version has no '{spec.ccxt_id}' adapter.",
                recovery=["Upgrade ccxt: pip install -U ccxt"],
                retryable=False,
            ))

        # NOTE: no apiKey / secret / password is ever supplied.  Public market
        # data does not require credentials, and this app must never request them.
        ex = cls({
            "enableRateLimit": self.enable_rate_limit,
            "timeout": self.request_timeout_ms,
            "options": {"defaultType": "spot"},
        })
        _EXCHANGE_CACHE[spec.ccxt_id] = ex
        return ex

    def _load_markets(self, ex, exchange_key: str) -> int:
        now = time.time()
        cached = _MARKETS_CACHE.get(exchange_key)
        if cached and (now - cached[0]) < MARKETS_TTL_S:
            return cached[1]
        markets = ex.load_markets(reload=False)
        count = len(markets or {})
        _MARKETS_CACHE[exchange_key] = (now, count)
        return count

    def _server_time_ms(self, ex, exchange_key: str) -> float:
        """
        Exchange server clock, used so a skewed local clock cannot make a
        still-forming candle look closed.  Cached briefly; falls back to the
        local clock if the endpoint is unavailable.
        """
        now = time.time()
        cached = _TIME_OFFSET_CACHE.get(exchange_key)
        if cached and (now - cached[1]) < TIME_OFFSET_TTL_S:
            return now * 1000.0 + cached[0]
        try:
            if getattr(ex, "has", {}).get("fetchTime"):
                server_ms = float(ex.fetch_time())
            else:
                server_ms = float(ex.milliseconds())
        except Exception as exc:
            log.info("fetch_time failed on %s (%s); using local clock", exchange_key, exc)
            server_ms = now * 1000.0
        offset = server_ms - now * 1000.0
        if abs(offset) > 120_000:
            log.warning("Large clock skew vs %s: %.0f ms", exchange_key, offset)
        _TIME_OFFSET_CACHE[exchange_key] = (offset, now)
        return server_ms

    # -- retry wrapper ---------------------------------------------------- #

    def _call_with_retries(self, exchange_key: str, label: str, fn, *args, **kwargs):
        """
        Run `fn` with bounded retries and exponential backoff + jitter.

        Hard failures (geo-block, bad symbol) are NOT retried.
        """
        attempt = 0
        last_feed_error: Optional[E.FeedError] = None
        while True:
            try:
                return fn(*args, **kwargs)
            except E.DataFeedError as exc:
                raise
            except Exception as exc:
                feed_error = E.classify_exception(exc, exchange_key=exchange_key)
                last_feed_error = feed_error
                log.warning("%s failed (attempt %d): %s", label, attempt + 1, feed_error.detail)
                if not feed_error.retryable or attempt >= self.max_retries:
                    self.breaker.record_failure(exchange_key, feed_error.is_hard_failure)
                    raise E.DataFeedError(feed_error) from exc
                delay = min(self.backoff_cap_s, self.backoff_base_s * (2 ** attempt))
                delay += random.uniform(0, delay * 0.25)   # jitter avoids thundering herd
                self.sleep(delay)
                attempt += 1
        # unreachable

    # -- connectivity probe ---------------------------------------------- #

    def check_connection(
        self,
        exchange_key: str,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        force: bool = False,
    ) -> FeedStatus:
        """
        Diagnose reachability and capability step by step.

        Each stage is reported separately so the UI can say exactly where the
        chain broke (import -> construct -> server time -> markets -> symbol ->
        timeframe -> sample candles).
        """
        spec = get_exchange_spec(exchange_key)
        status = FeedStatus(exchange_key=exchange_key, ok=False, symbol=symbol, timeframe=timeframe)
        t0 = time.perf_counter()

        if not force and self.breaker.is_open(exchange_key):
            wait = self.breaker.seconds_until_closed(exchange_key)
            status.stage = "circuit_breaker"
            status.category = E.CAT_EXCHANGE_UNAVAILABLE
            status.message = (
                f"{spec.label} is temporarily blocked after repeated failures. "
                f"Retrying in {wait:.0f}s to avoid hammering the exchange."
            )
            status.recovery = [
                "Wait for the cooldown to expire, or press 'Reset circuit breaker'.",
                "Switch to the alternative market-data source in the sidebar.",
            ]
            status.latency_ms = (time.perf_counter() - t0) * 1000.0
            return status

        # 1. ccxt import + construct
        status.stage = "construct"
        try:
            ex = self.get_exchange(exchange_key)
        except E.DataFeedError as exc:
            return self._fail(status, exc.feed_error, t0)
        except Exception as exc:
            return self._fail(status, E.classify_exception(exc, exchange_key), t0)

        # 2. server time (cheap, proves TCP+TLS+HTTP round trip)
        status.stage = "server_time"
        try:
            server_ms = self._call_with_retries(exchange_key, "fetch_time", self._server_time_ms, ex, exchange_key)
            status.server_time_ms = server_ms
            status.clock_skew_ms = server_ms - time.time() * 1000.0
        except E.DataFeedError as exc:
            return self._fail(status, exc.feed_error, t0)

        # 3. markets metadata
        status.stage = "load_markets"
        try:
            status.markets_loaded = self._call_with_retries(exchange_key, "load_markets", self._load_markets, ex, exchange_key)
        except E.DataFeedError as exc:
            return self._fail(status, exc.feed_error, t0)

        # 4. symbol support
        if symbol:
            status.stage = "symbol"
            try:
                status.symbol_supported = bool(ex.market(symbol) and ex.market(symbol).get("active", True))
            except Exception:
                status.symbol_supported = symbol in (ex.markets or {})
            if not status.symbol_supported:
                status.category = E.CAT_BAD_SYMBOL
                status.message = (
                    f"{spec.label} does not list '{symbol}'. "
                    f"Markets loaded: {status.markets_loaded}."
                )
                status.recovery = [
                    "Pick one of the pairs offered in the sidebar dropdown.",
                    "The dropdown is built from this exchange's live market list.",
                ]
                status.detail = f"symbol={symbol} not in markets"
                status.latency_ms = (time.perf_counter() - t0) * 1000.0
                self.breaker.record_failure(exchange_key, hard=True)
                return status

        # 5. timeframe capability
        if timeframe:
            status.stage = "timeframe"
            native = timeframe in (ex.timeframes or {})
            status.timeframe_native = native
            if not native and timeframe not in spec.native_timeframes:
                if timeframe != "4h":
                    status.category = E.CAT_BAD_TIMEFRAME
                    status.message = (
                        f"{spec.label} does not offer '{timeframe}' candles. "
                        f"Native: {', '.join(ex.timeframes or spec.native_timeframes)}."
                    )
                    status.recovery = ["Select a supported timeframe in the sidebar."]
                    status.latency_ms = (time.perf_counter() - t0) * 1000.0
                    return status

        # 6. sample candles (the real proof)
        if symbol and timeframe:
            status.stage = "fetch_ohlcv"
            probe_tf = timeframe if status.timeframe_native else "1h"
            try:
                sample = self._call_with_retries(
                    exchange_key, "fetch_ohlcv(sample)",
                    ex.fetch_ohlcv, symbol, probe_tf, None, 3,
                )
                if not sample:
                    return self._fail(status, E.empty_response_error(exchange_key, symbol, probe_tf), t0)
                status.sample_price = float(sample[-1][4])
            except E.DataFeedError as exc:
                return self._fail(status, exc.feed_error, t0)

        status.ok = True
        status.stage = "complete"
        status.latency_ms = (time.perf_counter() - t0) * 1000.0
        status.message = f"{spec.label} reachable. {status.markets_loaded} markets loaded."
        if timeframe == "4h" and not status.timeframe_native:
            status.message += " 4h is aggregated locally from 1h candles."
        self.breaker.record_success(exchange_key)
        return status

    @staticmethod
    def _fail(status: FeedStatus, feed_error: E.FeedError, t0: float) -> FeedStatus:
        status.ok = False
        status.category = feed_error.category
        status.message = feed_error.message
        status.recovery = list(feed_error.recovery)
        status.http_status = feed_error.http_status
        status.detail = feed_error.detail
        status.latency_ms = (time.perf_counter() - t0) * 1000.0
        return status

    # -- OHLCV fetching --------------------------------------------------- #

    def fetch_candles(self, cfg: FeedConfig, force: bool = False) -> CandleSet:
        """
        Fetch, validate and split candles for `cfg`.

        Raises DataFeedError on failure *after* storing nothing; the caller can
        fall back to `last_good()` for display purposes.
        """
        cfg.validate()
        spec = get_exchange_spec(cfg.exchange_key)

        if not force and self.breaker.is_open(cfg.exchange_key):
            wait = self.breaker.seconds_until_closed(cfg.exchange_key)
            cached = _LAST_GOOD.get(cfg.feed_key)
            if cached is not None:
                out = self._as_cached(cached)
                out.warnings.append(
                    f"Exchange circuit breaker open ({wait:.0f}s remaining); showing last known-good data."
                )
                return out
            raise E.DataFeedError(E.FeedError(
                category=E.CAT_EXCHANGE_UNAVAILABLE,
                message=(
                    f"{spec.label} is temporarily blocked after repeated failures "
                    f"(cooldown {wait:.0f}s). No cached candles are available to display."
                ),
                recovery=[
                    "Wait for the cooldown, or press 'Reset circuit breaker' in diagnostics.",
                    "Switch to the alternative market-data source.",
                ],
                retryable=False, exchange_key=cfg.exchange_key,
            ))

        ex = self.get_exchange(cfg.exchange_key)
        server_ms = self._call_with_retries(cfg.exchange_key, "fetch_time", self._server_time_ms, ex, cfg.exchange_key)
        fetched_at = time.time() * 1000.0
        warnings: List[str] = []
        requests = 0

        tf_ms = TIMEFRAME_MS[cfg.timeframe]
        native = cfg.timeframe in (getattr(ex, "timeframes", {}) or {}) or cfg.timeframe in spec.native_timeframes
        aggregated_forming: Optional[List[float]] = None

        if native:
            raw, requests = self._fetch_paginated(ex, cfg, cfg.timeframe, cfg.candle_limit)
            source_tf = cfg.timeframe
            is_aggregated = False
        else:
            # Documented aggregation path: build the requested candles from
            # real lower-timeframe candles the venue actually serves.  Nothing
            # is ever interpolated or invented; incomplete buckets are dropped
            # and reported by `_aggregate`.
            source_tf = self.aggregation_source(
                spec, cfg.timeframe, getattr(ex, "timeframes", None))
            if source_tf is None:
                raise E.DataFeedError(E.FeedError(
                    category=E.CAT_BAD_TIMEFRAME,
                    message=f"{spec.label} cannot provide '{cfg.timeframe}' candles.",
                    recovery=["Select a supported timeframe."],
                    retryable=False, exchange_key=cfg.exchange_key,
                ))
            ratio = TIMEFRAME_MS[cfg.timeframe] // TIMEFRAME_MS[source_tf]
            raw, requests = self._fetch_paginated(
                ex, cfg, source_tf, cfg.candle_limit * ratio + 8)
            is_aggregated = True
            raw, aggregated_forming, agg_warnings = self._aggregate(
                raw, from_tf=source_tf, to_tf=cfg.timeframe, now_ms=server_ms
            )
            warnings.extend(agg_warnings)

        if not raw:
            self.breaker.record_failure(cfg.exchange_key, hard=False)
            raise E.DataFeedError(E.empty_response_error(cfg.exchange_key, cfg.symbol, source_tf))

        df = self._to_dataframe(raw)
        df, val_warnings = self._validate(df, tf_ms, cfg)
        warnings.extend(val_warnings)

        if df.empty:
            self.breaker.record_failure(cfg.exchange_key, hard=False)
            raise E.DataFeedError(E.empty_response_error(cfg.exchange_key, cfg.symbol, source_tf))

        # ---- closed vs forming ------------------------------------------ #
        # Every exchange tested returns the still-forming candle as the final
        # row.  It is separated here so strategy code can never see it: signals
        # are generated from fully closed candles only.
        ts = df["timestamp"].to_numpy(dtype="float64")
        complete = candles_are_complete(ts, tf_ms, server_ms)
        forming: Optional[Dict[str, float]] = None
        if not complete.all():
            forming_rows = df[~complete]
            last_forming = forming_rows.iloc[-1].to_dict()
            forming = {k: (None if pd.isna(v) else float(v)) for k, v in last_forming.items()
                       if k in ("timestamp", "open", "high", "low", "close", "volume")}
            df = df[complete].reset_index(drop=True)

        if forming is None and aggregated_forming is not None:
            # Aggregated timeframes carry their partial bucket separately.
            forming = {k: float(v) for k, v in zip(OHLCV_COLUMNS, aggregated_forming)}

        if df.empty:
            raise E.DataFeedError(E.insufficient_history_error(
                cfg.exchange_key, cfg.symbol, cfg.timeframe, 1, 0))

        missing, gaps = find_missing_candles(df["timestamp"].to_numpy(dtype="float64"), tf_ms)
        if missing:
            warnings.append(
                f"{missing} candle gap(s) detected in the {cfg.timeframe} series. "
                f"Gaps are reported, never filled with invented prices."
            )

        # ---- data quality on thin venues -------------------------------- #
        zv, zv_share, flat, flat_share = measure_candle_quality(df)
        thin = bool(spec.thin_data)
        if thin:
            note = describe_candle_quality(cfg.timeframe, len(df), zv, zv_share, flat, flat_share)
            if note:
                warnings.append(note)

        result = CandleSet(
            exchange_key=cfg.exchange_key,
            symbol=cfg.symbol,
            timeframe=cfg.timeframe,
            df=df,
            forming=forming,
            fetched_at_ms=fetched_at,
            server_time_ms=server_ms,
            is_aggregated=is_aggregated,
            source_timeframe=source_tf,
            missing_count=missing,
            gaps=gaps,
            warnings=warnings,
            served_from_cache=False,
            candle_count=len(df),
            last_closed_ts=float(df["timestamp"].iloc[-1]),
            requests_made=requests,
            zero_volume_count=zv,
            zero_volume_share=zv_share,
            flat_candle_count=flat,
            flat_candle_share=flat_share,
            thin_venue=thin,
        )

        _LAST_GOOD[cfg.feed_key] = result
        self.breaker.record_success(cfg.exchange_key)
        log.info("Fetched %d closed %s candles for %s %s (%d request(s))",
                 len(df), cfg.timeframe, cfg.exchange_key, cfg.symbol, requests)
        return result

    #: Candles requested by the cheap live poll.  Delta returns an EMPTY list
    #: for tiny limits (limit=2/3 -> 0 rows), so never ask for fewer than this.
    LIVE_POLL_CANDLES = 12

    def fetch_forming(self, cfg: FeedConfig) -> Dict[str, Any]:
        """
        Fetch ONLY the still-forming candle, as cheaply as the venue allows.

        This exists so the chart can show a candle being built in real time
        without re-downloading the whole history or re-running the strategy.
        It never writes to `_LAST_GOOD` and never changes what the engine sees:
        signals remain closed-candle-only, exactly as before.

        Returns a dict:
            forming        -> OHLCV dict of the in-progress bucket, or None
            last_closed_ts -> open-time (ms) of the newest CLOSED candle seen
            server_time_ms -> exchange clock used to decide what "closed" means
            warnings       -> list[str]
            requests       -> int
        """
        cfg.validate()
        spec = get_exchange_spec(cfg.exchange_key)
        out: Dict[str, Any] = {"forming": None, "last_closed_ts": None,
                               "server_time_ms": 0.0, "warnings": [], "requests": 0}

        if self.breaker.is_open(cfg.exchange_key):
            out["warnings"].append(
                f"Circuit breaker open for {spec.label}; the live candle is paused "
                f"({self.breaker.seconds_until_closed(cfg.exchange_key):.0f}s remaining)."
            )
            return out

        tf_ms = TIMEFRAME_MS[cfg.timeframe]
        try:
            ex = self.get_exchange(cfg.exchange_key)
            server_ms = self._call_with_retries(
                cfg.exchange_key, "fetch_time", self._server_time_ms, ex, cfg.exchange_key)
            out["server_time_ms"] = server_ms
            native = (cfg.timeframe in (getattr(ex, "timeframes", {}) or {})
                      or cfg.timeframe in spec.native_timeframes)

            if native:
                since = server_ms - tf_ms * self.LIVE_POLL_CANDLES
                raw = self._call_with_retries(
                    cfg.exchange_key, "fetch_ohlcv(live)", ex.fetch_ohlcv,
                    cfg.symbol, cfg.timeframe, since, self.LIVE_POLL_CANDLES)
                out["requests"] = 1
                complete_raw, forming_raw, _w = self._aggregate(
                    raw or [], from_tf=cfg.timeframe, to_tf=cfg.timeframe, now_ms=server_ms)
            else:
                # Same documented aggregation as the main path, from whatever
                # lower timeframe this venue really serves.
                source_tf = self.aggregation_source(
                    spec, cfg.timeframe, getattr(ex, "timeframes", None))
                if source_tf is None:
                    out["warnings"].append(
                        f"{spec.label} cannot provide live '{cfg.timeframe}' candles.")
                    return out
                ratio = TIMEFRAME_MS[cfg.timeframe] // TIMEFRAME_MS[source_tf]
                # Enough sub-candles to cover the in-progress bucket plus the
                # previous ones, so `_aggregate` can also report the newest
                # CLOSED bucket at this granularity.
                want = max(self.LIVE_POLL_CANDLES, ratio * 2)
                raw = self._call_with_retries(
                    cfg.exchange_key, f"fetch_ohlcv(live {source_tf})", ex.fetch_ohlcv,
                    cfg.symbol, source_tf,
                    server_ms - TIMEFRAME_MS[source_tf] * want, want)
                out["requests"] = 1
                complete_raw, forming_raw, _w = self._aggregate(
                    raw or [], from_tf=source_tf, to_tf=cfg.timeframe, now_ms=server_ms)

            if complete_raw:
                out["last_closed_ts"] = float(complete_raw[-1][0])
            if forming_raw:
                out["forming"] = {
                    k: (None if pd.isna(v) else float(v))
                    for k, v in zip(OHLCV_COLUMNS, forming_raw)
                }
            elif raw:
                out["warnings"].append(
                    "The exchange returned no in-progress candle for this bucket yet.")
        except Exception as exc:                                   # noqa: BLE001
            # A failed live poll must never break the page: the chart keeps the
            # closed candles it already has and simply says the poll failed.
            self.breaker.record_failure(cfg.exchange_key, hard=False)
            out["warnings"].append(f"Live candle poll failed: {type(exc).__name__}: {exc}")
            log.warning("Live candle poll failed for %s: %s", cfg.feed_key, exc)
        return out

    def last_good(self, cfg: FeedConfig) -> Optional[CandleSet]:
        cached = _LAST_GOOD.get(cfg.feed_key)
        return self._as_cached(cached) if cached else None

    @staticmethod
    def _as_cached(cached: CandleSet) -> CandleSet:
        """Return a copy of a cached frame clearly marked as stale/from-cache."""
        out = CandleSet(**{**cached.__dict__})
        out.df = cached.df.copy()
        out.served_from_cache = True
        out.warnings = list(cached.warnings) + [
            "Showing the last successfully fetched candles. The live feed is NOT connected."
        ]
        return out

    # -- internals -------------------------------------------------------- #

    def _fetch_paginated(self, ex, cfg: FeedConfig, timeframe: str, want: int) -> Tuple[List[List[float]], int]:
        """
        Fetch up to `want` candles.

        Pagination direction matters and is easy to get wrong.  Verified against
        the live APIs: for both `coinbaseexchange` and `kraken`, passing `since`
        returns candles FORWARD from that timestamp (capped per request), while
        `since=None` returns the most RECENT candles.

        So the strategy is:
          * want <= per-request cap  -> one call with since=None (most recent).
          * want  > per-request cap  -> compute the start of the window and page
            FORWARD, advancing the cursor to (last timestamp + one timeframe).

        Paging backwards from the newest candle would re-request almost the same
        window every time and de-duplicate down to a single page.
        """
        cap = max(50, min(want, get_exchange_spec(cfg.exchange_key).max_candles_per_request))
        tf_ms = TIMEFRAME_MS[timeframe]
        collected: List[List[float]] = []
        requests = 0

        # Fast path: everything fits in one request.
        if want <= cap:
            batch = self._call_with_retries(
                cfg.exchange_key, f"fetch_ohlcv({timeframe})",
                ex.fetch_ohlcv, cfg.symbol, timeframe, None, want,
            )
            return (batch or []), 1

        # Slow path: page forward through history.
        # Enough pages for the worst case (a 4h request needs 4x 1h candles),
        # plus headroom for the exchange's off-by-one short pages.
        max_pages = min(24, max(3, math.ceil(want / cap) + 3))
        now_ms = float(ex.milliseconds())
        cursor: Optional[int] = int(now_ms - want * tf_ms)
        newest_seen: Optional[int] = None

        for _ in range(max_pages):
            remaining = want - len({int(r[0]) for r in collected})
            if remaining <= 0:
                break
            limit = min(cap, max(1, remaining))
            batch = self._call_with_retries(
                cfg.exchange_key, f"fetch_ohlcv({timeframe})",
                ex.fetch_ohlcv, cfg.symbol, timeframe, cursor, limit,
            )
            requests += 1
            if not batch:
                break
            collected.extend(batch)
            batch_newest = int(max(float(r[0]) for r in batch))
            if newest_seen is not None and batch_newest <= newest_seen:
                break                      # no forward progress -> stop
            newest_seen = batch_newest
            cursor = batch_newest + tf_ms
            # Termination is driven by *timestamp progress*, not by batch size.
            # Verified quirk: coinbaseexchange returns 299 rows for limit=300
            # (inclusive/exclusive window), so "short batch" does NOT reliably
            # mean the exchange ran out of history.  The cursor strictly
            # increases every iteration and `max_pages` is a hard stop, so this
            # loop cannot spin forever.
            if cursor >= now_ms:
                break                      # reached the present

        # De-duplicate and sort ascending.
        dedup: Dict[int, List[float]] = {}
        for row in collected:
            try:
                dedup[int(row[0])] = row
            except Exception:
                continue
        ordered = [dedup[k] for k in sorted(dedup)]
        # Keep only the most recent `want` candles.
        if len(ordered) > want:
            ordered = ordered[-want:]
        return ordered, requests

    @staticmethod
    def aggregation_source(
        spec, timeframe: str, ex_timeframes: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Which lower timeframe should higher-timeframe `timeframe` be built from?

        An exchange does not have to serve every granularity natively.  When it
        does not, we aggregate from the LARGEST granularity that (a) the venue
        really has - it is in the exchange's own `timeframes` map and/or in the
        registry's `native_timeframes` list - and (b) divides the target
        exactly.  That keeps the number of sub-candles per bucket small (fewer
        requests) and guarantees every bucket is a whole number of candles.

        Returns None when no exact source exists, in which case the timeframe is
        genuinely unsupported on that venue and the caller must say so instead
        of inventing candles.

        Examples (verified against the live APIs):
          * Coinbase 4h  -> 1h   (1 hour candles, 4 per bucket)
          * Coinbase 30m -> 15m  (15 minute candles, 2 per bucket)
          * Delta/Kraken -> never called: every offered timeframe is native.
        """
        target = TIMEFRAME_MS.get(timeframe)
        if not target:
            return None
        native: List[str] = []
        for tf in (spec.native_timeframes or []):
            if tf not in TIMEFRAME_MS:
                continue
            if ex_timeframes and tf not in ex_timeframes:
                continue          # the venue's own map does not know it either
            if timeframe == tf:
                continue
            if TIMEFRAME_MS[tf] > target:
                continue
            if target % TIMEFRAME_MS[tf]:
                continue          # not an exact multiple -> would be lossy
            native.append(tf)
        if not native:
            return None
        return max(native, key=lambda tf: TIMEFRAME_MS[tf])

    @staticmethod
    def _aggregate(
        raw: List[List[float]], from_tf: str, to_tf: str, now_ms: float
    ) -> Tuple[List[List[float]], Optional[List[float]], List[str]]:
        """
        Build higher-timeframe candles from real lower-timeframe candles.

        Returns (complete_buckets, forming_bucket_or_None, warnings).

        A bucket is only emitted as *complete* when its end time is in the past
        AND it contains the full set of sub-candles.  Incomplete historical
        buckets are DROPPED and reported - we never publish a 4h high/low
        computed from a partial set of hourly candles, because that would
        understate the true range and corrupt swing levels.
        """
        warnings: List[str] = []
        if not raw:
            return [], None, warnings
        from_ms = TIMEFRAME_MS[from_tf]
        to_ms = TIMEFRAME_MS[to_tf]
        expected = int(round(to_ms / from_ms))

        buckets: Dict[int, List[List[float]]] = {}
        for row in raw:
            ts = int(row[0])
            key = (ts // to_ms) * to_ms
            buckets.setdefault(key, []).append(row)

        out: List[List[float]] = []
        forming: Optional[List[float]] = None
        dropped = 0
        for key in sorted(buckets):
            rows = sorted(buckets[key], key=lambda r: r[0])
            bucket_end = key + to_ms
            merged = [
                float(key),
                float(rows[0][1]),
                float(max(r[2] for r in rows)),
                float(min(r[3] for r in rows)),
                float(rows[-1][4]),
                float(sum((r[5] or 0.0) for r in rows)),
            ]
            if bucket_end > now_ms:
                forming = merged          # still in progress -> never traded on
                continue
            if len(rows) < expected:
                dropped += 1
                continue
            out.append(merged)

        if dropped:
            warnings.append(
                f"{dropped} aggregated {to_tf} bucket(s) were dropped because they "
                f"contained fewer than {expected} {from_tf} candles. Incomplete "
                f"buckets are never published as real price data."
            )
        warnings.append(
            f"{to_tf} candles are aggregated locally from {from_tf} data because this "
            f"exchange has no native {to_tf} granularity."
        )
        return out, forming, warnings

    @staticmethod
    def _to_dataframe(raw: List[List[float]]) -> pd.DataFrame:
        df = pd.DataFrame(raw, columns=OHLCV_COLUMNS[:len(raw[0])] if raw and len(raw[0]) < 6 else OHLCV_COLUMNS)
        for col in OHLCV_COLUMNS:
            if col not in df.columns:
                df[col] = np.nan
        return df[OHLCV_COLUMNS]

    @staticmethod
    def _validate(df: pd.DataFrame, tf_ms: int, cfg: FeedConfig) -> Tuple[pd.DataFrame, List[str]]:
        """
        Enforce data quality.  Bad rows are dropped and reported - never repaired
        with invented values.
        """
        warnings: List[str] = []
        n0 = len(df)
        out = df.copy()

        # 1. numeric coercion
        for col in OHLCV_COLUMNS:
            out[col] = pd.to_numeric(out[col], errors="coerce")

        # 2. drop rows with a missing/invalid timestamp or OHLC
        bad_ts = out["timestamp"].isna()
        bad_ohlc = out[["open", "high", "low", "close"]].isna().any(axis=1)
        if bad_ts.any():
            warnings.append(f"Dropped {int(bad_ts.sum())} row(s) with an unparseable timestamp.")
        if bad_ohlc.any():
            warnings.append(f"Dropped {int(bad_ohlc.sum())} row(s) with missing OHLC values.")
        out = out[~(bad_ts | bad_ohlc)]

        # 3. drop non-positive prices
        nonpos = (out[["open", "high", "low", "close"]] <= 0).any(axis=1)
        if nonpos.any():
            warnings.append(f"Dropped {int(nonpos.sum())} row(s) with zero or negative prices.")
            out = out[~nonpos]

        # 4. drop impossible high/low envelopes
        bad_env = (out["high"] < out[["open", "close"]].max(axis=1)) | (out["low"] > out[["open", "close"]].min(axis=1))
        bad_env = bad_env | (out["high"] < out["low"])
        if bad_env.any():
            warnings.append(f"Dropped {int(bad_env.sum())} row(s) where high/low did not contain open/close.")
            out = out[~bad_env]

        # 5. negative volume -> NaN (volume is informational only)
        if "volume" in out.columns:
            out.loc[out["volume"] < 0, "volume"] = np.nan

        # 6. duplicate timestamps + ordering
        out["timestamp"] = out["timestamp"].astype("int64")
        dups = int(out["timestamp"].duplicated().sum())
        if dups:
            warnings.append(f"Removed {dups} duplicate timestamp(s).")
        out = out.drop_duplicates(subset="timestamp", keep="last")
        out = out.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

        # 7. timestamps must align to the timeframe grid
        if not out.empty:
            misaligned = int((out["timestamp"] % tf_ms != 0).sum())
            if misaligned:
                warnings.append(
                    f"{misaligned} candle timestamp(s) are not aligned to the {cfg.timeframe} grid."
                )

        if len(out) < n0:
            warnings.append(f"Validation kept {len(out)} of {n0} raw candles.")
        return out, warnings

    # -- diagnostics helpers --------------------------------------------- #

    def list_symbols(self, exchange_key: str) -> List[str]:
        """Live list of BTC pairs the exchange actually serves (for the UI)."""
        try:
            ex = self.get_exchange(exchange_key)
            self._load_markets(ex, exchange_key)
            markets = ex.markets or {}
            pairs = sorted(
                s for s, m in markets.items()
                if s.startswith("BTC/") and ":" not in s and m.get("spot", True) and m.get("active", True)
            )
            return pairs or list(get_exchange_spec(exchange_key).symbols)
        except Exception as exc:
            log.warning("list_symbols(%s) failed: %s", exchange_key, exc)
            return list(get_exchange_spec(exchange_key).symbols)

    def circuit_state(self, exchange_key: str) -> Dict[str, Any]:
        return {
            "open": self.breaker.is_open(exchange_key),
            "seconds_remaining": round(self.breaker.seconds_until_closed(exchange_key), 1),
            "failures": _CIRCUIT.get(exchange_key, {}).get("failures", 0),
        }


__all__ = [
    "CandleSet", "FeedStatus", "MarketDataFeed", "clear_caches",
    "MARKETS_TTL_S", "TIME_OFFSET_TTL_S",
]
