"""
core/config.py
==============
Central configuration for the BTC paper-trading terminal.

Everything tunable lives here so that the UI, the strategy, the risk engine and
the backtester all read the *same* defaults.  No secrets are ever stored here:
public market data requires no API keys.

Design notes for beginners
--------------------------
*  A "dataclass" is just a tidy container for related settings with type hints.
*  `validate()` methods raise `InvalidConfigError` with a plain-English message
   so the Streamlit sidebar can show the user exactly what is wrong instead of
   crashing later in a confusing place.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

# --------------------------------------------------------------------------- #
# Application-level constants
# --------------------------------------------------------------------------- #

APP_NAME = "BTC Paper-Trading Terminal"
APP_VERSION = "1.3.0"

#: Authorship line shown in the sidebar and the page footer.
APP_AUTHOR = "Abhijeet"
COPYRIGHT_LINE = "Made by the father of this world Abhijeet"

#: Live (still-forming) candle refresh, in seconds.  This only re-draws the
#: forming candle - it never re-evaluates signals, which stay closed-candle only.
LIVE_REFRESH_CHOICES_S: List[int] = [5, 10, 15, 30, 60]
DEFAULT_LIVE_REFRESH_S = 10

#: ------------------------------------------------------------------------
#: Market-dashboard settings (24h stats strip, movers table, return ring).
#: ------------------------------------------------------------------------
#: How often the 24h stats strip re-reads its cached public ticker.  0 = only
#: when the page reloads.  One small public request per refresh, no key.
DASH_REFRESH_CHOICES_S: List[int] = [0, 15, 30, 60]
DEFAULT_DASH_REFRESH_S = 15

#: Module-level TTL (seconds) for the public ticker snapshot.  A manual
#: "Refresh now" click bypasses it; the auto-refresh never hits the exchange
#: more often than this, whatever the UI interval is set to.
TICKER_TTL_S = 10.0

#: Bases compared side by side in the "Market overview" panel.  Only symbols
#: the selected exchange really lists are shown - nothing is invented and no
#: row is ever filled with a placeholder price.
MOVERS_BASES: List[str] = ["BTC", "ETH", "SOL", "XRP", "DOGE", "BNB", "ADA", "LINK",
                           "AVAX", "LTC", "TRX", "TON"]
#: Rows shown per column of the movers panel.
MOVERS_ROWS = 5

#: Rolling window (days) for the "Strategy performance" ring and its table.
MONTHLY_WINDOW_DAYS = 30

#: Zoom presets for the chart toolbar: label -> candles drawn.  "Auto" draws
#: everything that was fetched (the sidebar's "Candles to fetch" value), so
#: there is no separate "All" - it would mean the same thing.  A VIEW setting:
#: it never changes the frame the strategy, indicators or signals read.
ZOOM_CHOICES: Dict[str, int] = {"Auto": 0, "60": 60, "120": 120, "240": 240}
DEFAULT_ZOOM = "Auto"

#: Hard-coded safety banner.  This application is paper-trading only.
SIMULATION_BANNER = "SIMULATION ONLY - NO REAL ORDERS"

#: Timeframes the UI offers (the chart toolbar mirrors this exact order).
#: Every entry is always offered: on an exchange that does not serve the
#: granularity natively the data feed builds it by aggregating the largest
#: native granularity that divides it exactly, and flags the result as
#: aggregated (see core/data_feed.py `aggregation_source`).  Delta and Kraken
#: serve all seven natively; Coinbase aggregates 30m from 15m and 4h from 1h.
SUPPORTED_TIMEFRAMES: List[str] = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
DEFAULT_TIMEFRAME = "1h"

#: Milliseconds per timeframe.  Used for staleness checks and candle closing.
TIMEFRAME_MS: Dict[str, int] = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "1d": 86_400_000,
}

#: How many candles we keep for charting / indicator warm-up by default.
DEFAULT_CANDLE_LIMIT = 300
MIN_CANDLE_LIMIT = 60
MAX_CANDLE_LIMIT = 1000

#: A closed candle older than this multiple of the timeframe is "stale".
STALE_MULTIPLIER = 3.0

# --------------------------------------------------------------------------- #
# Trading defaults (documented in the README and in the UI)
# --------------------------------------------------------------------------- #

DEFAULT_STARTING_BALANCE = 10_000.0   # virtual USD
DEFAULT_RISK_PER_TRADE_PCT = 1.0      # % of equity risked per trade
DEFAULT_REWARD_TO_RISK = 2.0          # take-profit = 2R
DEFAULT_FEE_PCT_PER_SIDE = 0.05       # 0.05 % taker-style fee, per side
DEFAULT_SLIPPAGE_BPS = 2.0            # 2 basis points of adverse slippage
DEFAULT_DAILY_LOSS_LIMIT_PCT = 3.0    # % of the daily reference balance

#: Stop-loss placement modes.
STOP_MODE_SWING = "swing"   # beyond the prior swing low/high (+ buffer)
STOP_MODE_ATR = "atr"       # ATR multiple
STOP_MODE_PERCENT = "percent"  # fixed % of entry price
STOP_MODES = [STOP_MODE_SWING, STOP_MODE_ATR, STOP_MODE_PERCENT]

#: Fill models.
FILL_SIGNAL_CLOSE = "signal_close"  # fill at the signal candle's close
FILL_NEXT_OPEN = "next_open"        # fill at the next candle's open (default)
FILL_MODES = [FILL_SIGNAL_CLOSE, FILL_NEXT_OPEN]

# Simulated instrument constraints (BTC).  These are conservative defaults for
# a *simulation*; they are not exchange order-book guarantees.
DEFAULT_MIN_QTY = 0.00001            # BTC
DEFAULT_QTY_STEP = 0.000001          # BTC rounding step
DEFAULT_MIN_NOTIONAL = 10.0          # USD
DEFAULT_MAX_NOTIONAL_PCT = 100.0     # max position notional as % of equity (1x, no leverage)


class InvalidConfigError(ValueError):
    """Raised when a user-supplied setting is out of a safe/valid range."""


from .sessions import SESSION_KEYS

# --------------------------------------------------------------------------- #
# Exchange registry
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ExchangeSpec:
    """Static description of a supported public market-data source."""

    key: str                       # internal key, stored in the DB
    label: str                     # what the UI shows
    ccxt_id: str                   # ccxt exchange id
    symbols: List[str]             # supported trading pairs (ccxt unified)
    default_symbol: str
    native_timeframes: List[str]   # granularities the exchange serves directly
    max_candles_per_request: int   # hard cap observed for fetch_ohlcv
    notes: str = ""
    #: "spot" or "perpetual futures" - shown in the UI so nobody mistakes a
    #: derivatives price for a spot price.
    market_kind: str = "spot"
    #: True when the venue is thin enough that many buckets contain no trades at
    #: all.  The feed then measures and reports the zero-volume share, and the
    #: strategy refuses to signal on an empty candle (see `cond_tradeable`).
    thin_data: bool = False

    def supports_timeframe_natively(self, timeframe: str) -> bool:
        return timeframe in self.native_timeframes


# Verified against the live public APIs (no keys required) on 2026-10-15:
#   * delta            -> api.delta.exchange         (1m,3m,5m,15m,30m,1h,2h,4h,6h,1d; <=4000)
#   * coinbaseexchange -> api.exchange.coinbase.com  (1m,5m,15m,1h,6h,1d; <=350)
#   * kraken           -> api.kraken.com             (1m,5m,15m,30m,1h,4h,1d,1w; <=720)
# "4h" is NOT native on Coinbase, so the feed aggregates 1h -> 4h there.
# Delta serves 4h natively, so no aggregation is needed.
#
# DELTA MEASUREMENTS (public endpoints, no key):
#   * BTC/USDT *spot* exists but is dead - zero candles at every timeframe,
#     no bid/ask, no 24h volume.  It is deliberately NOT offered.
#   * BTC/USDT:USDT (perpetual futures) is the only BTC market with data.
#   * Small `limit` values return an EMPTY list (limit=2/3 -> 0 rows); a
#     request must ask for at least ~10 candles.  The live poll asks for 12.
#   * Thin liquidity: of the last ~200 buckets, zero-volume share was 55% (5m),
#     36% (15m), 9% (1h), 1% (4h).  Hence `thin_data=True`.
EXCHANGES: Dict[str, ExchangeSpec] = {
    "delta": ExchangeSpec(
        key="delta",
        label="Delta Exchange (public)",
        ccxt_id="delta",
        symbols=["BTC/USDT:USDT"],
        default_symbol="BTC/USDT:USDT",
        native_timeframes=["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "1d"],
        max_candles_per_request=2000,  # 4000 observed; 2000 keeps a safety margin
        market_kind="perpetual futures",
        thin_data=True,
        notes=(
            "Public Delta Exchange market data. No API key required. "
            "BTC/USDT:USDT is a PERPETUAL FUTURES price, not spot - Delta's spot "
            "BTC/USDT book returns no candle data at all, so it is not offered. "
            "This venue is thin: many low-timeframe buckets contain no trades, "
            "which the app measures and reports, and no signal is ever generated "
            "from a zero-volume candle. 4h is native here (no local aggregation)."
        ),
    ),
    "coinbase": ExchangeSpec(
        key="coinbase",
        label="Coinbase Exchange (public)",
        ccxt_id="coinbaseexchange",
        symbols=["BTC/USD", "BTC/USDC", "BTC/USDT", "BTC/EUR", "BTC/GBP"],
        default_symbol="BTC/USD",
        native_timeframes=["1m", "5m", "15m", "1h", "6h", "1d"],
        max_candles_per_request=300,   # 350 observed; 300 keeps a safety margin
        notes=(
            "Public Coinbase Exchange market data. No API key required. "
            "4h candles are aggregated locally from 1h candles (flagged in the UI)."
        ),
    ),
    "kraken": ExchangeSpec(
        key="kraken",
        label="Kraken (public)",
        ccxt_id="kraken",
        symbols=["BTC/USD", "BTC/USDT", "BTC/EUR", "BTC/USDC"],
        default_symbol="BTC/USD",
        native_timeframes=["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"],
        max_candles_per_request=700,   # 720 observed; 700 keeps a safety margin
        notes="Public Kraken market data. No API key required. Native 4h support.",
    ),
}

DEFAULT_EXCHANGE_KEY = "delta"

#: Default pair for the default exchange (Delta's only BTC market with data).
DEFAULT_SYMBOL = EXCHANGES[DEFAULT_EXCHANGE_KEY].default_symbol

#: Ordered fallback list shown in diagnostics.  The app NEVER switches silently;
#: this list only powers the "recovery steps" hint.
EXCHANGE_ORDER: List[str] = ["delta", "coinbase", "kraken"]


def get_exchange_spec(key: str) -> ExchangeSpec:
    """Return the ExchangeSpec for `key`, or raise a friendly error."""
    if key not in EXCHANGES:
        raise InvalidConfigError(
            f"Unknown exchange '{key}'. Supported: {', '.join(EXCHANGES)}."
        )
    return EXCHANGES[key]


def symbols_for(exchange_key: str) -> List[str]:
    return list(get_exchange_spec(exchange_key).symbols)


# --------------------------------------------------------------------------- #
# Strategy parameters
# --------------------------------------------------------------------------- #

#: ---------------------------------------------------------------------- #
#: Optional entry FILTERS (v1.3.0).  Every one of them is OFF by default, and
#: with all of them off `generate_signals` returns exactly the frame it
#: returned before this feature existed - that is enforced by test.
#:
#: A filter can only ever REMOVE trades.  None of them can invent one, and none
#: of them is a prediction: they are conditions on already-closed candles.
#: ---------------------------------------------------------------------- #
TREND_FILTER_MODES: List[str] = ["off", "above", "below"]

#: ATR-as-%-of-price bounds.  0.0 means "no bound on this side".
DEFAULT_ATR_MIN_PCT = 0.0
DEFAULT_ATR_MAX_PCT = 0.0
DEFAULT_VOLUME_FILTER_RATIO = 0.0
DEFAULT_TREND_EMA_PERIOD = 200
DEFAULT_FILTER_VOLUME_LOOKBACK = 20

@dataclass
class StrategyParams:
    """BOS + RSI + EMA configuration."""

    swing_lookback: int = 10          # candles used for the prior swing level
    swing_shift: int = 1              # >=1 guarantees no self-influence / look-ahead
    rsi_period: int = 14
    rsi_long_min: float = 55.0        # long requires RSI >= this
    rsi_short_max: float = 45.0       # short requires RSI <= this
    ema_fast: int = 20
    ema_slow: int = 50
    require_close_breakout: bool = True   # False => wick breakouts also count
    stop_mode: str = STOP_MODE_SWING
    stop_buffer_pct: float = 0.10     # extra distance beyond the swing/ATR stop
    atr_period: int = 14
    atr_multiplier: float = 1.5
    percent_stop: float = 1.0         # used only when stop_mode == "percent"

    # ------------------- optional entry filters (v1.3.0) ------------------- #
    #: "off" | "above" | "below" - require price on a chosen side of a long
    #: moving average.  A trend filter is the classic way to stop a
    #: mean-reverting window from producing counter-trend entries.
    trend_filter: str = "off"
    trend_ema_period: int = DEFAULT_TREND_EMA_PERIOD
    #: Volatility window: require ATR/price (in %) to sit inside these bounds.
    #: A floor avoids dead, untraded markets; a ceiling avoids blow-off spikes.
    atr_filter_min_pct: float = DEFAULT_ATR_MIN_PCT
    atr_filter_max_pct: float = DEFAULT_ATR_MAX_PCT
    #: Require the candle's volume to be at least this multiple of the average
    #: volume of the PREVIOUS `volume_filter_lookback` candles (0 = off).
    volume_filter_ratio: float = DEFAULT_VOLUME_FILTER_RATIO
    volume_filter_lookback: int = DEFAULT_FILTER_VOLUME_LOOKBACK
    #: Allowed trading sessions (see core/sessions.py).  Empty = every hour.
    session_filter: List[str] = field(default_factory=list)
    #: Trade on Saturdays/Sundays (UTC).  Bitcoin never closes, but weekends are
    #: measurably thinner, so some traders exclude them.
    allow_weekends: bool = True

    def filters_active(self) -> bool:
        """True when at least one filter would change anything."""
        return (self.trend_filter != "off"
                or self.atr_filter_min_pct > 0.0
                or self.atr_filter_max_pct > 0.0
                or self.volume_filter_ratio > 0.0
                or bool(self.session_filter)
                or not self.allow_weekends)

    def validate(self) -> "StrategyParams":
        if not isinstance(self.swing_lookback, int) or self.swing_lookback < 2:
            raise InvalidConfigError("Swing lookback must be an integer >= 2.")
        if self.swing_lookback > 200:
            raise InvalidConfigError("Swing lookback must be <= 200.")
        if not isinstance(self.swing_shift, int) or self.swing_shift < 1:
            raise InvalidConfigError(
                "Swing shift must be >= 1. A shift of 0 would let a candle "
                "influence its own breakout level (look-ahead bias)."
            )
        if not isinstance(self.rsi_period, int) or self.rsi_period < 2:
            raise InvalidConfigError("RSI period must be an integer >= 2.")
        if not 1.0 <= self.rsi_long_min <= 100.0:
            raise InvalidConfigError("RSI long threshold must be between 1 and 100.")
        if not 1.0 <= self.rsi_short_max <= 100.0:
            raise InvalidConfigError("RSI short threshold must be between 1 and 100.")
        if self.rsi_short_max >= self.rsi_long_min:
            raise InvalidConfigError(
                "RSI short threshold must be strictly below the long threshold "
                f"(got short={self.rsi_short_max}, long={self.rsi_long_min})."
            )
        if not isinstance(self.ema_fast, int) or self.ema_fast < 2:
            raise InvalidConfigError("EMA fast period must be an integer >= 2.")
        if not isinstance(self.ema_slow, int) or self.ema_slow < 2:
            raise InvalidConfigError("EMA slow period must be an integer >= 2.")
        if self.ema_slow <= self.ema_fast:
            raise InvalidConfigError(
                f"EMA slow ({self.ema_slow}) must be greater than EMA fast ({self.ema_fast})."
            )
        if self.stop_mode not in STOP_MODES:
            raise InvalidConfigError(f"Stop mode must be one of {STOP_MODES}.")
        if not 0.0 <= self.stop_buffer_pct <= 5.0:
            raise InvalidConfigError("Stop buffer % must be between 0 and 5.")
        if not isinstance(self.atr_period, int) or self.atr_period < 2:
            raise InvalidConfigError("ATR period must be an integer >= 2.")
        if not 0.1 <= self.atr_multiplier <= 20.0:
            raise InvalidConfigError("ATR multiplier must be between 0.1 and 20.")
        if not 0.05 <= self.percent_stop <= 25.0:
            raise InvalidConfigError("Percent stop must be between 0.05 and 25.")
        # ---- v1.3.0 filters ---------------------------------------------- #
        if self.trend_filter not in TREND_FILTER_MODES:
            raise InvalidConfigError(
                f"Trend filter must be one of {TREND_FILTER_MODES}.")
        if not isinstance(self.trend_ema_period, int) or not 5 <= self.trend_ema_period <= 400:
            raise InvalidConfigError("Trend EMA period must be an integer between 5 and 400.")
        for name, value in (("ATR minimum %", self.atr_filter_min_pct),
                            ("ATR maximum %", self.atr_filter_max_pct)):
            if not 0.0 <= float(value) <= 100.0:
                raise InvalidConfigError(f"{name} must be between 0 and 100 (0 = off).")
        if (self.atr_filter_min_pct > 0.0 and self.atr_filter_max_pct > 0.0
                and self.atr_filter_min_pct >= self.atr_filter_max_pct):
            raise InvalidConfigError(
                "ATR minimum % must be below the ATR maximum % "
                f"(got min={self.atr_filter_min_pct}, max={self.atr_filter_max_pct}) - "
                "otherwise no candle can ever qualify."
            )
        if not 0.0 <= float(self.volume_filter_ratio) <= 100.0:
            raise InvalidConfigError("Volume filter ratio must be between 0 (off) and 100.")
        if not isinstance(self.volume_filter_lookback, int) or not 2 <= self.volume_filter_lookback <= 500:
            raise InvalidConfigError("Volume filter lookback must be an integer between 2 and 500.")
        unknown = [s for s in (self.session_filter or []) if s not in SESSION_KEYS]
        if unknown:
            raise InvalidConfigError(
                f"Unknown trading session(s): {', '.join(unknown)}. "
                f"Valid: {', '.join(SESSION_KEYS)}."
            )
        return self

    def min_history(self) -> int:
        """Candles required before any signal can be produced (warm-up)."""
        warmup = max(
            self.ema_slow,
            self.rsi_period + 1,
            self.swing_lookback + self.swing_shift,
            self.atr_period + 1,
        )
        if self.trend_filter != "off":
            warmup = max(warmup, self.trend_ema_period)
        if self.volume_filter_ratio > 0.0:
            warmup = max(warmup, self.volume_filter_lookback + 1)
        return warmup + 2

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Phase-2 price-action parameters (additive; disabled by default)
# --------------------------------------------------------------------------- #

@dataclass
class PriceActionParams:
    """
    Configuration for the advanced price-action layer (core/price_action.py).

    `enabled=False` by default: the Phase-1 four-condition strategy behaves
    EXACTLY as before unless the user opts in.  Every threshold here is an
    objective, testable rule - these concepts are price-based interpretations,
    not proof of hidden institutional order flow.
    """

    enabled: bool = False

    # -- market structure / pivots
    pivot_lookback: int = 3              # fractal k: pivot needs k candles each side
    structure_lookback: int = 10         # reused for BOS when df is not pre-enriched
    choch_confirm_window: int = 20       # candles to confirm an early CHoCH into an MSS
    fail_break_window: int = 5           # candles in which a break can fail

    # -- liquidity
    eq_tolerance_bps: float = 6.0        # pivot highs within this = "equal highs"
    sweep_reclaim_frac: float = 0.5      # close back inside by >= this fraction of range
    liq_expiry_candles: int = 120        # unswept pools expire after this long

    # -- fair value gaps
    fvg_use_atr_filter: bool = True
    fvg_min_atr: float = 0.35            # gap height must be >= this many ATRs
    fvg_min_bps: float = 5.0             # ...and at least this many bps of price
    fvg_mitigation_depth: float = 0.5    # depth (0..1) that counts as mitigation
    fvg_expiry_candles: int = 80

    # -- order blocks
    ob_displacement_atr: float = 1.2     # move that qualifies the block
    ob_displacement_candles: int = 3
    ob_invalidation: str = "close"       # "close" or "wick"
    ob_expiry_candles: int = 80

    # -- supply / demand
    sd_base_candles: int = 3             # consolidation length
    sd_base_range_atr: float = 1.2       # base candles must be this tight (sum of ranges)
    sd_displacement_atr: float = 1.5     # displacement away from the base
    sd_displacement_candles: int = 3
    sd_overlap_pct: float = 50.0         # suppress a new zone overlapping an active one
    sd_expiry_candles: int = 100
    max_active_zones: int = 40

    # -- candlestick patterns
    pattern_body_frac: float = 0.35      # pin bar body <= this fraction of range
    pattern_wick_frac: float = 0.60      # pin bar dominant wick >= this fraction of range
    doji_frac: float = 0.10              # doji body <= this fraction of range
    displacement_atr: float = 1.8        # strong candle range in ATRs
    displacement_body_frac: float = 0.70  # ...of which this much must be body

    # -- breakout / retest
    retest_tol_bps: float = 10.0         # price within this of the level = retest
    breakout_expiry_candles: int = 40

    # -- premium / discount
    equilibrium_band_pct: float = 10.0   # % of range height treated as equilibrium

    # -- multi-timeframe (HTF context)
    mtf_enabled: bool = False            # off by default, like the rest of the add-on
    mtf_timeframe: str = "auto"          # "auto" = next step up (5m->15m->1h->4h->1d)
    mtf_min_candles: int = 30            # closed HTF candles needed before HTF context
    w_htf_bias: float = 1.0
    w_htf_zone: float = 1.0
    w_htf_premium_discount: float = 0.5

    # -- confluence weights (J)
    w_structure: float = 2.0
    w_liquidity: float = 1.5
    w_fvg: float = 1.0
    w_order_block: float = 1.0
    w_supply_demand: float = 1.0
    w_pattern: float = 0.5
    w_breakout: float = 1.5
    w_premium_discount: float = 0.5
    min_confluence_score: float = 3.0    # advisory threshold used by the UI

    def validate(self) -> "PriceActionParams":
        if not isinstance(self.pivot_lookback, int) or self.pivot_lookback < 1:
            raise InvalidConfigError("pivot_lookback must be an integer >= 1.")
        if self.pivot_lookback > 50:
            raise InvalidConfigError("pivot_lookback must be <= 50.")
        if not isinstance(self.structure_lookback, int) or self.structure_lookback < 2:
            raise InvalidConfigError("structure_lookback must be an integer >= 2.")
        if not 1 <= self.choch_confirm_window <= 200:
            raise InvalidConfigError("choch_confirm_window must be between 1 and 200.")
        if not 1 <= self.fail_break_window <= 50:
            raise InvalidConfigError("fail_break_window must be between 1 and 50.")
        if not 0.1 <= self.eq_tolerance_bps <= 100.0:
            raise InvalidConfigError("eq_tolerance_bps must be between 0.1 and 100.")
        if not 0.0 < self.sweep_reclaim_frac <= 1.0:
            raise InvalidConfigError("sweep_reclaim_frac must be in (0, 1].")
        if not 0.0 <= self.fvg_min_atr <= 10.0:
            raise InvalidConfigError("fvg_min_atr must be between 0 and 10.")
        if not 0.0 < self.fvg_mitigation_depth <= 1.0:
            raise InvalidConfigError("fvg_mitigation_depth must be in (0, 1].")
        if self.ob_invalidation not in ("close", "wick"):
            raise InvalidConfigError("ob_invalidation must be 'close' or 'wick'.")
        if not 0.05 <= self.ob_displacement_atr <= 20.0:
            raise InvalidConfigError("ob_displacement_atr must be between 0.05 and 20.")
        if not 2 <= self.sd_base_candles <= 20:
            raise InvalidConfigError("sd_base_candles must be between 2 and 20.")
        if not 0.05 <= self.sd_displacement_atr <= 20.0:
            raise InvalidConfigError("sd_displacement_atr must be between 0.05 and 20.")
        if not 0.0 <= self.sd_overlap_pct <= 100.0:
            raise InvalidConfigError("sd_overlap_pct must be between 0 and 100.")
        if not 1 <= self.max_active_zones <= 500:
            raise InvalidConfigError("max_active_zones must be between 1 and 500.")
        if not 0.0 < self.pattern_body_frac < 1.0:
            raise InvalidConfigError("pattern_body_frac must be in (0, 1).")
        if not 0.0 < self.pattern_wick_frac <= 1.0:
            raise InvalidConfigError("pattern_wick_frac must be in (0, 1].")
        if not 0.0 < self.doji_frac < 0.5:
            raise InvalidConfigError("doji_frac must be in (0, 0.5).")
        if not 0.05 <= self.displacement_atr <= 30.0:
            raise InvalidConfigError("displacement_atr must be between 0.05 and 30.")
        if not 0.0 < self.retest_tol_bps <= 200.0:
            raise InvalidConfigError("retest_tol_bps must be between 0.05 and 200 bps.")
        if not 0.0 <= self.equilibrium_band_pct <= 50.0:
            raise InvalidConfigError("equilibrium_band_pct must be between 0 and 50.")
        if not 0.0 <= self.fvg_min_bps <= 1000.0:
            raise InvalidConfigError("fvg_min_bps must be between 0 and 1000.")
        if not 0.05 <= self.sd_base_range_atr <= 20.0:
            raise InvalidConfigError("sd_base_range_atr must be between 0.05 and 20.")
        if not 0.0 < self.displacement_body_frac <= 1.0:
            raise InvalidConfigError("displacement_body_frac must be in (0, 1].")
        for name in ("liq_expiry_candles", "fvg_expiry_candles", "ob_expiry_candles",
                     "sd_expiry_candles", "breakout_expiry_candles",
                     "ob_displacement_candles", "sd_displacement_candles"):
            v = getattr(self, name)
            if not isinstance(v, int) or not 1 <= v <= 2000:
                raise InvalidConfigError(f"{name} must be an integer between 1 and 2000.")
        for name in ("w_structure", "w_liquidity", "w_fvg", "w_order_block",
                     "w_supply_demand", "w_pattern", "w_breakout", "w_premium_discount",
                     "w_htf_bias", "w_htf_zone", "w_htf_premium_discount"):
            if not 0.0 <= getattr(self, name) <= 10.0:
                raise InvalidConfigError(f"{name} must be between 0 and 10.")
        if self.mtf_timeframe != "auto" and self.mtf_timeframe not in TIMEFRAME_MS:
            raise InvalidConfigError(
                f"mtf_timeframe must be 'auto' or one of {sorted(TIMEFRAME_MS)}.")
        if not isinstance(self.mtf_min_candles, int) or not 5 <= self.mtf_min_candles <= 2000:
            raise InvalidConfigError("mtf_min_candles must be an integer between 5 and 2000.")
        if not 0.0 <= self.min_confluence_score <= 100.0:
            raise InvalidConfigError("min_confluence_score must be between 0 and 100.")
        return self

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Risk parameters
# --------------------------------------------------------------------------- #

@dataclass
class RiskParams:
    """Position sizing, fees and loss limits."""

    starting_balance: float = DEFAULT_STARTING_BALANCE
    risk_per_trade_pct: float = DEFAULT_RISK_PER_TRADE_PCT
    reward_to_risk: float = DEFAULT_REWARD_TO_RISK
    fee_pct_per_side: float = DEFAULT_FEE_PCT_PER_SIDE
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS
    daily_loss_limit_pct: float = DEFAULT_DAILY_LOSS_LIMIT_PCT
    max_notional_pct: float = DEFAULT_MAX_NOTIONAL_PCT
    min_notional: float = DEFAULT_MIN_NOTIONAL
    min_qty: float = DEFAULT_MIN_QTY
    qty_step: float = DEFAULT_QTY_STEP
    max_open_positions: int = 1
    allow_shorts: bool = True

    def validate(self) -> "RiskParams":
        if self.starting_balance <= 0:
            raise InvalidConfigError("Starting balance must be greater than zero.")
        if self.starting_balance > 1_000_000_000:
            raise InvalidConfigError("Starting balance is unrealistically large.")
        if not 0.01 <= self.risk_per_trade_pct <= 25.0:
            raise InvalidConfigError(
                "Risk per trade must be between 0.01% and 25%. "
                f"Got {self.risk_per_trade_pct}%."
            )
        if not 0.1 <= self.reward_to_risk <= 20.0:
            raise InvalidConfigError("Reward-to-risk must be between 0.1 and 20.")
        if not 0.0 <= self.fee_pct_per_side <= 2.0:
            raise InvalidConfigError("Fee per side must be between 0% and 2%.")
        if not 0.0 <= self.slippage_bps <= 200.0:
            raise InvalidConfigError("Slippage must be between 0 and 200 bps.")
        if not 0.1 <= self.daily_loss_limit_pct <= 100.0:
            raise InvalidConfigError("Daily loss limit must be between 0.1% and 100%.")
        if not 1.0 <= self.max_notional_pct <= 100.0:
            raise InvalidConfigError(
                "Max position notional must be between 1% and 100% of equity "
                "(this simulator does not use leverage above 1x)."
            )
        if self.min_notional < 0:
            raise InvalidConfigError("Minimum notional cannot be negative.")
        if self.min_qty <= 0:
            raise InvalidConfigError("Minimum quantity must be greater than zero.")
        if self.qty_step <= 0:
            raise InvalidConfigError("Quantity step must be greater than zero.")
        if not isinstance(self.max_open_positions, int) or self.max_open_positions < 1:
            raise InvalidConfigError("Max open positions must be an integer >= 1.")
        return self

    @property
    def fee_rate(self) -> float:
        """Fee as a decimal fraction (0.05 % -> 0.0005)."""
        return self.fee_pct_per_side / 100.0

    @property
    def slippage_rate(self) -> float:
        """Slippage as a decimal fraction (2 bps -> 0.0002)."""
        return self.slippage_bps / 10_000.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Feed / runtime configuration
# --------------------------------------------------------------------------- #

@dataclass
class FeedConfig:
    """Market-data engine settings."""

    exchange_key: str = DEFAULT_EXCHANGE_KEY
    symbol: str = DEFAULT_SYMBOL
    timeframe: str = DEFAULT_TIMEFRAME
    candle_limit: int = DEFAULT_CANDLE_LIMIT
    request_timeout_ms: int = 15_000
    max_retries: int = 3
    backoff_base_s: float = 0.8
    backoff_cap_s: float = 8.0
    enable_rate_limit: bool = True
    #: Seconds to stop retrying after a persistent hard failure (e.g. HTTP 451).
    circuit_breaker_cooldown_s: float = 300.0
    circuit_breaker_threshold: int = 3

    def validate(self) -> "FeedConfig":
        spec = get_exchange_spec(self.exchange_key)  # raises on unknown key
        if self.symbol not in spec.symbols:
            raise InvalidConfigError(
                f"Symbol '{self.symbol}' is not supported on {spec.label}. "
                f"Supported: {', '.join(spec.symbols)}."
            )
        if self.timeframe not in SUPPORTED_TIMEFRAMES:
            raise InvalidConfigError(
                f"Timeframe '{self.timeframe}' is not supported. "
                f"Use one of {SUPPORTED_TIMEFRAMES}."
            )
        if not MIN_CANDLE_LIMIT <= self.candle_limit <= MAX_CANDLE_LIMIT:
            raise InvalidConfigError(
                f"Candle count must be between {MIN_CANDLE_LIMIT} and {MAX_CANDLE_LIMIT}."
            )
        if not 2_000 <= self.request_timeout_ms <= 120_000:
            raise InvalidConfigError("Request timeout must be between 2 and 120 seconds.")
        if not 0 <= self.max_retries <= 6:
            raise InvalidConfigError("Max retries must be between 0 and 6.")
        return self

    @property
    def feed_key(self) -> str:
        """Stable identity of a data series, used for duplicate-candle guards."""
        return f"{self.exchange_key}:{self.symbol}:{self.timeframe}"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def default_strategy() -> StrategyParams:
    return StrategyParams().validate()


def default_risk() -> RiskParams:
    return RiskParams().validate()


def default_feed(exchange_key: str = DEFAULT_EXCHANGE_KEY) -> FeedConfig:
    spec = get_exchange_spec(exchange_key)
    return FeedConfig(exchange_key=exchange_key, symbol=spec.default_symbol).validate()


__all__ = [
    "APP_NAME", "APP_VERSION", "APP_AUTHOR", "COPYRIGHT_LINE",
    "LIVE_REFRESH_CHOICES_S", "DEFAULT_LIVE_REFRESH_S",
    "DASH_REFRESH_CHOICES_S", "DEFAULT_DASH_REFRESH_S", "TICKER_TTL_S",
    "MOVERS_BASES", "MOVERS_ROWS", "MONTHLY_WINDOW_DAYS",
    "ZOOM_CHOICES", "DEFAULT_ZOOM",
    "SIMULATION_BANNER",
    "SUPPORTED_TIMEFRAMES", "DEFAULT_TIMEFRAME", "TIMEFRAME_MS",
    "DEFAULT_CANDLE_LIMIT", "MIN_CANDLE_LIMIT", "MAX_CANDLE_LIMIT", "STALE_MULTIPLIER",
    "DEFAULT_STARTING_BALANCE", "DEFAULT_RISK_PER_TRADE_PCT", "DEFAULT_REWARD_TO_RISK",
    "DEFAULT_FEE_PCT_PER_SIDE", "DEFAULT_SLIPPAGE_BPS", "DEFAULT_DAILY_LOSS_LIMIT_PCT",
    "STOP_MODES", "STOP_MODE_SWING", "STOP_MODE_ATR", "STOP_MODE_PERCENT",
    "FILL_MODES", "FILL_SIGNAL_CLOSE", "FILL_NEXT_OPEN",
    "DEFAULT_MIN_QTY", "DEFAULT_QTY_STEP", "DEFAULT_MIN_NOTIONAL", "DEFAULT_MAX_NOTIONAL_PCT",
    "InvalidConfigError",
    "ExchangeSpec", "EXCHANGES", "DEFAULT_EXCHANGE_KEY", "DEFAULT_SYMBOL", "EXCHANGE_ORDER",
    "get_exchange_spec", "symbols_for",
    "StrategyParams", "RiskParams", "FeedConfig", "PriceActionParams",
    "default_strategy", "default_risk", "default_feed",
]
