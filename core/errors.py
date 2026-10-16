"""
core/errors.py
==============
Turns raw library exceptions into a small, well-understood set of categories,
each with:

*  a short plain-English message for the dashboard,
*  concrete recovery steps,
*  a flag saying whether retrying is even worth it,
*  a sanitised technical detail string for the log file.

Why this matters
----------------
A raw ``ccxt.ExchangeNotAvailable`` traceback tells a beginner nothing, and a
raw HTTP 451 from an exchange is a *geographic/legal restriction* that no
amount of page-refreshing will fix.  This module makes that distinction
explicit so the UI never suggests a useless remedy.

Security
--------
`sanitize()` scrubs anything that looks like a credential before a message is
logged or displayed.  This app never asks for trading API keys, but defence in
depth is cheap.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

# --------------------------------------------------------------------------- #
# Categories
# --------------------------------------------------------------------------- #

CAT_GEO_RESTRICTED = "geo_restricted"
CAT_EXCHANGE_UNAVAILABLE = "exchange_unavailable"
CAT_RATE_LIMITED = "rate_limited"
CAT_TIMEOUT = "timeout"
CAT_NETWORK = "network"
CAT_BAD_SYMBOL = "bad_symbol"
CAT_BAD_TIMEFRAME = "bad_timeframe"
CAT_AUTH = "auth"
CAT_EMPTY_RESPONSE = "empty_response"
CAT_INSUFFICIENT_HISTORY = "insufficient_history"
CAT_DEPENDENCY = "dependency"
CAT_DATABASE = "database"
CAT_STRATEGY_PARAMS = "strategy_params"
CAT_BACKTEST = "backtest"
CAT_UNKNOWN = "unknown"

#: Categories where retrying the identical request immediately is pointless.
NON_RETRYABLE = {
    CAT_GEO_RESTRICTED,
    CAT_BAD_SYMBOL,
    CAT_BAD_TIMEFRAME,
    CAT_AUTH,
    CAT_DEPENDENCY,
    CAT_STRATEGY_PARAMS,
}

#: Categories that trip the circuit breaker (stop hammering the exchange).
CIRCUIT_BREAKER_CATEGORIES = {
    CAT_GEO_RESTRICTED,
    CAT_EXCHANGE_UNAVAILABLE,
    CAT_AUTH,
    CAT_BAD_SYMBOL,
    CAT_BAD_TIMEFRAME,
}


class ErrorCategory(str):
    """Kept as plain strings above for easy DB/JSON storage."""


@dataclass
class FeedError:
    """A classified, user-presentable error."""

    category: str
    message: str
    recovery: List[str] = field(default_factory=list)
    retryable: bool = True
    detail: str = ""
    http_status: Optional[int] = None
    exchange_key: Optional[str] = None

    @property
    def is_hard_failure(self) -> bool:
        return self.category in CIRCUIT_BREAKER_CATEGORIES

    @property
    def recovery_text(self) -> str:
        if not self.recovery:
            return "No automatic recovery is available for this condition."
        return "\n".join(f"{i}. {step}" for i, step in enumerate(self.recovery, 1))

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "message": self.message,
            "recovery": list(self.recovery),
            "retryable": self.retryable,
            "detail": self.detail,
            "http_status": self.http_status,
            "exchange_key": self.exchange_key,
        }


class DataFeedError(Exception):
    """Raised by the market-data engine.  Always carries a `FeedError`."""

    def __init__(self, feed_error: FeedError):
        self.feed_error = feed_error
        super().__init__(feed_error.message)


class InsufficientHistoryError(DataFeedError):
    pass


# --------------------------------------------------------------------------- #
# Sanitisation
# --------------------------------------------------------------------------- #

_SECRET_PATTERNS = [
    re.compile(r"(?i)(api[_-]?key|apikey|secret|passphrase|password|token|authorization|bearer)\s*[=:]\s*\S+"),
    re.compile(r"\b[A-Za-z0-9_\-]{32,}\b"),   # long opaque strings
]


def sanitize(text: str, max_len: int = 600) -> str:
    """Remove credential-looking material and truncate."""
    if text is None:
        return ""
    out = str(text)
    out = _SECRET_PATTERNS[0].sub("[REDACTED]", out)
    out = _SECRET_PATTERNS[1].sub("[REDACTED]", out)
    out = " ".join(out.split())
    return out[:max_len]


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #

_HTTP_LABEL_RE = re.compile(r"(?i)\bhttp[\s/_-]*(?:status|code|error|response)?[\s:=]*(4\d\d|5\d\d)\b")
_HTTP_PHRASE_RE = re.compile(
    r"(?i)\b(4\d\d|5\d\d)\s+(forbidden|unauthorized|unavailable for legal reasons|"
    r"too many requests|not found|bad gateway|service unavailable|gateway time-?out|"
    r"internal server error|request time-?out|unprocessable)\b"
)
_BARE_451_RE = re.compile(r"\b451\b")


def _extract_http_status(exc: BaseException, text: str) -> Optional[int]:
    """
    Pull an HTTP status code out of an exception, conservatively.

    A bare 3-digit number in an error string is often NOT a status code (it can
    be a candle count, a price or a port), so we only accept:
      * an explicit attribute set by the HTTP layer,
      * "HTTP <code>" style labels,
      * "<code> <reason phrase>" pairs,
      * a bare 451, which is distinctive enough to trust.
    """
    status = getattr(exc, "status_code", None) or getattr(exc, "http_status", None)
    if isinstance(status, int) and 100 <= status <= 599:
        return status
    if not text:
        return None
    for pattern in (_HTTP_LABEL_RE, _HTTP_PHRASE_RE):
        m = pattern.search(text)
        if m:
            return int(m.group(1))
    if _BARE_451_RE.search(text):
        return 451
    return None


def classify_exception(
    exc: BaseException,
    exchange_key: Optional[str] = None,
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None,
) -> FeedError:
    """
    Map any exception raised while talking to an exchange (or importing a
    dependency) onto a `FeedError`.

    The order of checks matters: HTTP 451 / 403 with a geographic hint is
    detected *before* the generic ccxt class hierarchy, because ccxt reports
    those as a plain ExchangeError.
    """
    cls_name = type(exc).__name__
    raw = sanitize(f"{cls_name}: {exc}")
    text = raw.lower()
    http_status = _extract_http_status(exc, raw)

    ctx = ""
    if symbol or timeframe:
        ctx = f" (symbol={symbol or '?'}, timeframe={timeframe or '?'})"

    # ---- missing dependency -------------------------------------------- #
    if cls_name in ("ModuleNotFoundError", "ImportError"):
        module = getattr(exc, "name", None) or "required package"
        return FeedError(
            category=CAT_DEPENDENCY,
            message=f"A required Python package is missing: {module}.",
            recovery=[
                "Activate your virtual environment, then run: pip install -r requirements.txt",
                "Restart Streamlit so the new packages are imported.",
                "On Streamlit Community Cloud, redeploy after committing requirements.txt.",
            ],
            retryable=False,
            detail=raw,
        )

    # ---- geographic restriction (HTTP 451 / 403 with region hints) ------ #
    geo_hints = (
        "451", "unavailable for legal reasons", "restricted jurisdiction",
        "geographic", "geo-restrict", "not available in your", "restricted region",
        "jurisdiction", "compliance restriction",
    )
    if http_status == 451 or any(h in text for h in geo_hints):
        return FeedError(
            category=CAT_GEO_RESTRICTED,
            message=(
                f"The exchange refused this request from the server's region "
                f"(HTTP {http_status or 451}){ctx}. This is a legal/geographic "
                f"restriction, not a temporary outage."
            ),
            recovery=[
                "Refreshing the page will NOT fix this - the restriction is regional.",
                "Select a different market-data source in the sidebar (Kraken is the configured alternative).",
                "If you deploy to Streamlit Community Cloud, its servers may be in a different region than your PC; test both.",
                "Do not attempt to bypass regional restrictions with proxies or VPNs.",
            ],
            retryable=False,
            detail=raw,
            http_status=http_status,
            exchange_key=exchange_key,
        )

    # ---- HTTP 403 on a PUBLIC endpoint ---------------------------------- #
    # This app never sends credentials, so a 403 cannot be "you forgot your API
    # key".  On a public market-data endpoint it means the exchange/WAF blocked
    # the request - typically by region or IP.  Refreshing will not help.
    if http_status == 403:
        return FeedError(
            category=CAT_GEO_RESTRICTED,
            message=(
                f"The exchange blocked this public request (HTTP 403){ctx}. "
                f"Since no credentials are sent, this is an access restriction "
                f"(region, IP or WAF), not an authentication problem."
            ),
            recovery=[
                "Refreshing the page will NOT fix an access block.",
                "Select the alternative market-data source in the sidebar.",
                "If you self-host, the block may apply to your ISP or country; "
                "Streamlit Cloud servers are in a different region and may behave differently.",
                "Do not try to bypass regional restrictions with proxies or VPNs.",
            ],
            retryable=False,
            detail=raw,
            http_status=403,
            exchange_key=exchange_key,
        )

    # ---- ccxt class hierarchy ------------------------------------------ #
    # Imported lazily so the module still works if ccxt is not installed.
    try:
        import ccxt  # type: ignore
    except Exception:  # pragma: no cover - ccxt missing is itself a dependency error
        ccxt = None

    if ccxt is not None:
        if isinstance(exc, getattr(ccxt, "BadSymbol", ())):
            return FeedError(
                category=CAT_BAD_SYMBOL,
                message=f"This exchange does not serve market data for that trading pair{ctx}.",
                recovery=[
                    "Choose a different trading pair from the sidebar dropdown.",
                    "Run 'Connection diagnostics' to list the pairs this exchange actually supports.",
                ],
                retryable=False, detail=raw, exchange_key=exchange_key,
            )
        if isinstance(exc, getattr(ccxt, "AuthenticationError", ())):
            return FeedError(
                category=CAT_AUTH,
                message="The exchange rejected the request as unauthorised.",
                recovery=[
                    "This app uses PUBLIC market data only and must not need an API key.",
                    "If you set an environment variable for this exchange, remove it - public endpoints should stay anonymous.",
                ],
                retryable=False, detail=raw, exchange_key=exchange_key,
            )
        if isinstance(exc, (getattr(ccxt, "RateLimitExceeded", ()), getattr(ccxt, "DDoSProtection", ()))) or http_status == 429:
            return FeedError(
                category=CAT_RATE_LIMITED,
                message="The exchange rate-limited us. We are asking for data too often.",
                recovery=[
                    "Wait a few seconds - the app applies exponential backoff automatically.",
                    "Increase the refresh interval or reduce the candle count.",
                ],
                retryable=True, detail=raw, http_status=http_status, exchange_key=exchange_key,
            )
        if isinstance(exc, getattr(ccxt, "RequestTimeout", ())) or cls_name == "Timeout":
            return FeedError(
                category=CAT_TIMEOUT,
                message="The exchange did not respond in time.",
                recovery=[
                    "The request is retried automatically with backoff.",
                    "If it persists, increase the request timeout in Advanced settings.",
                ],
                retryable=True, detail=raw, exchange_key=exchange_key,
            )
        if isinstance(exc, getattr(ccxt, "ExchangeNotAvailable", ())) or http_status in (500, 502, 503, 504):
            return FeedError(
                category=CAT_EXCHANGE_UNAVAILABLE,
                message="The exchange is unavailable or returning server errors right now.",
                recovery=[
                    "This is usually temporary - the app retries with backoff.",
                    "Check the exchange status page if it lasts more than a few minutes.",
                    "Previously fetched candles stay on screen, clearly labelled with their age.",
                ],
                retryable=True, detail=raw, http_status=http_status, exchange_key=exchange_key,
            )
        if isinstance(exc, getattr(ccxt, "NetworkError", ())):
            return FeedError(
                category=CAT_NETWORK,
                message="A network problem prevented reaching the exchange.",
                recovery=[
                    "Check the internet connection of the machine running Streamlit.",
                    "Corporate/ISP firewalls can block exchange APIs.",
                    "The request is retried automatically.",
                ],
                retryable=True, detail=raw, exchange_key=exchange_key,
            )
        if isinstance(exc, getattr(ccxt, "ExchangeError", ())):
            # NOTE: HTTP 403 is already handled above, before the ccxt hierarchy.
            if "granularity" in text or "timeframe" in text or "interval" in text:
                return FeedError(
                    category=CAT_BAD_TIMEFRAME,
                    message=f"This exchange does not provide {timeframe or 'that'} candles natively{ctx}.",
                    recovery=[
                        "Pick a timeframe from the exchange's native list, or",
                        "let the app aggregate it from a smaller timeframe (shown as 'aggregated').",
                    ],
                    retryable=False, detail=raw, exchange_key=exchange_key,
                )
            return FeedError(
                category=CAT_EXCHANGE_UNAVAILABLE,
                message=f"The exchange returned an error{ctx}.",
                recovery=[
                    "The app retries with backoff; if it persists, switch data source.",
                    "See the diagnostics panel for the sanitised technical detail.",
                ],
                retryable=True, detail=raw, http_status=http_status, exchange_key=exchange_key,
            )

    # ---- sqlite --------------------------------------------------------- #
    if "sqlite" in cls_name.lower() or "database is locked" in text or "no such table" in text:
        return FeedError(
            category=CAT_DATABASE,
            message="The local SQLite database could not be used.",
            recovery=[
                "Check that the app has write permission for its folder.",
                "On Streamlit Community Cloud the filesystem is temporary - history may not survive a restart.",
                "Close any other program that has the .db file open.",
            ],
            retryable=True, detail=raw,
        )

    # ---- fallback ------------------------------------------------------- #
    return FeedError(
        category=CAT_UNKNOWN,
        message=f"Unexpected error while fetching market data{ctx}.",
        recovery=[
            "Open the diagnostics panel for the sanitised detail.",
            "Try a different exchange, symbol or timeframe to isolate the cause.",
        ],
        retryable=True, detail=raw, exchange_key=exchange_key,
    )


def empty_response_error(exchange_key: str, symbol: str, timeframe: str) -> FeedError:
    return FeedError(
        category=CAT_EMPTY_RESPONSE,
        message=(
            f"{exchange_key} returned no candles for {symbol} {timeframe}. "
            f"The pair may be delisted or inactive."
        ),
        recovery=[
            "Confirm the pair trades on that exchange using the diagnostics panel.",
            "Try a more liquid pair such as BTC/USD or BTC/USDT.",
            "Try a shorter timeframe - some exchanges lack deep history for large candles.",
        ],
        retryable=True,
        detail="fetch_ohlcv returned an empty list",
        exchange_key=exchange_key,
    )


def insufficient_history_error(
    exchange_key: str, symbol: str, timeframe: str, needed: int, have: int
) -> FeedError:
    return FeedError(
        category=CAT_INSUFFICIENT_HISTORY,
        message=(
            f"Not enough closed candles to evaluate the strategy on {symbol} {timeframe}: "
            f"have {have}, need at least {needed}."
        ),
        recovery=[
            "Increase the candle count in the sidebar.",
            "Or reduce the EMA slow period / swing lookback so less warm-up is required.",
            "Signals are suppressed (never guessed) until enough history exists.",
        ],
        retryable=False,
        detail=f"needed={needed} have={have}",
        exchange_key=exchange_key,
    )


__all__ = [
    "CAT_GEO_RESTRICTED", "CAT_EXCHANGE_UNAVAILABLE", "CAT_RATE_LIMITED", "CAT_TIMEOUT",
    "CAT_NETWORK", "CAT_BAD_SYMBOL", "CAT_BAD_TIMEFRAME", "CAT_AUTH", "CAT_EMPTY_RESPONSE",
    "CAT_INSUFFICIENT_HISTORY", "CAT_DEPENDENCY", "CAT_DATABASE", "CAT_STRATEGY_PARAMS",
    "CAT_BACKTEST", "CAT_UNKNOWN",
    "NON_RETRYABLE", "CIRCUIT_BREAKER_CATEGORIES",
    "FeedError", "DataFeedError", "InsufficientHistoryError",
    "classify_exception", "empty_response_error", "insufficient_history_error", "sanitize",
]
