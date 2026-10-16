"""
app.py
======
BTC Paper-Trading Terminal - Streamlit front end.

This file is UI ONLY.  Every piece of trading logic lives in core/ and is
covered by the pytest suite, so the dashboard can be changed without risking
the engine - and the engine can be tested without a browser.

SAFETY: this application is a paper-trading simulator.  It uses PUBLIC market
data only, never asks for exchange API keys, and contains no code path that can
place a real order.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
import copy
from dataclasses import replace
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from core import charts, dashboard as dash, ui_theme as ui
from core.backtest import BacktestConfig, equity_to_frame, result_to_rows, run_backtest
from core.config import (
    APP_NAME,
    APP_VERSION,
    COPYRIGHT_LINE,
    DEFAULT_EXCHANGE_KEY,
    DASH_REFRESH_CHOICES_S,
    DEFAULT_DASH_REFRESH_S,
    DEFAULT_LIVE_REFRESH_S,
    DEFAULT_ZOOM,
    EXCHANGES,
    FILL_MODES,
    FILL_NEXT_OPEN,
    FILL_SIGNAL_CLOSE,
    MAX_CANDLE_LIMIT,
    MIN_CANDLE_LIMIT,
    LIVE_REFRESH_CHOICES_S,
    MONTHLY_WINDOW_DAYS,
    MOVERS_ROWS,
    SIMULATION_BANNER,
    STOP_MODES,
    SUPPORTED_TIMEFRAMES,
    TREND_FILTER_MODES,
    FeedConfig,
    InvalidConfigError,
    PriceActionParams,
    RiskParams,
    StrategyParams,
    TIMEFRAME_MS,
    ZOOM_CHOICES,
    default_feed,
    get_exchange_spec,
)
from core import indicator_catalog as icat
from core import indicator_chart as ichart
from core import sessions as sessions_mod
from core import walkforward as wf
from core.data_feed import CandleSet, MarketDataFeed, clear_caches
from core.db import RESET_CONFIRM_TOKEN, Database
from core.errors import DataFeedError
from core.executor import PaperTradingEngine
from core.metrics import describe
from core.multi_timeframe import describe_mtf, resolve_htf
from core.strategy import explain_latest, generate_signals

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
log = logging.getLogger("btc_app")

# --------------------------------------------------------------------------- #
# Page + theme
# --------------------------------------------------------------------------- #

st.set_page_config(
    page_title=f"{APP_NAME} v{APP_VERSION}",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": f"{APP_NAME} v{APP_VERSION} - educational paper-trading "
                         f"simulator. {COPYRIGHT_LINE}."},
)
ui.inject()

#: Minimum seconds between two live fetches of the same series (rate-limit guard).
FETCH_COOLDOWN_S = 20.0


# --------------------------------------------------------------------------- #
# Cached singletons
# --------------------------------------------------------------------------- #

@st.cache_resource(show_spinner=False)
def get_db() -> Database:
    """
    Open the SQLite ledger.

    The path can be overridden with the BTC_DB_PATH environment variable, which
    is useful for tests and for pointing the app at a mounted volume.  No
    secrets are stored in this database.
    """
    import os
    db = Database(os.environ.get("BTC_DB_PATH", "paper_trading.db"))
    db.init_schema()
    return db


@st.cache_resource(show_spinner=False)
def get_feed() -> MarketDataFeed:
    return MarketDataFeed()


# --------------------------------------------------------------------------- #
# Settings persistence
# --------------------------------------------------------------------------- #

#: Bumped whenever DEFAULTS change in a way that must override settings saved by
#: an OLDER version.  v2 = Delta Exchange default + live candle (app v1.1.0).
#: Without this, a database written by v1.0.0 would keep the old exchange and
#: silently ignore an explicit instruction to switch data source.
SETTINGS_VERSION = 2

DEFAULTS: Dict[str, Any] = {
    "exchange_key": DEFAULT_EXCHANGE_KEY,
    "symbol": EXCHANGES[DEFAULT_EXCHANGE_KEY].default_symbol,
    "timeframe": "1h",
    "candle_limit": 300,
    "swing_lookback": 10,
    "swing_shift": 1,
    "rsi_period": 14,
    "rsi_long_min": 55.0,
    "rsi_short_max": 45.0,
    "ema_fast": 20,
    "ema_slow": 50,
    "require_close_breakout": True,
    "stop_mode": "swing",
    "stop_buffer_pct": 0.10,
    "atr_period": 14,
    "atr_multiplier": 1.5,
    "percent_stop": 1.0,
    "starting_balance": 10_000.0,
    "risk_per_trade_pct": 1.0,
    "reward_to_risk": 2.0,
    "fee_pct_per_side": 0.05,
    "slippage_bps": 2.0,
    "daily_loss_limit_pct": 3.0,
    "max_notional_pct": 100.0,
    "allow_shorts": True,
    "max_open_positions": 1,
    "fill_model": FILL_NEXT_OPEN,
    "show_swings": True,
    "show_signals": True,
    # Live (still-forming) candle.  DISPLAY ONLY: the forming candle is never
    # part of the frame the strategy sees, so signals stay closed-candle-only.
    "live_candle": True,
    "live_refresh_s": DEFAULT_LIVE_REFRESH_S,
    #: How often the 24h public-market strip re-reads its cached ticker
    #: (0 = only when the page reloads).  Market statistics only - never
    #: connected to the paper engine, which stays closed-candle driven.
    "dash_refresh_s": DEFAULT_DASH_REFRESH_S,
    #: Chart "view" preset (candles shown).  A VIEW setting: it never changes
    #: the frame the strategy reads.
    "chart_zoom": DEFAULT_ZOOM,
    # Phase-2 price-action add-on.  OFF by default: nothing about the existing
    # strategy, signals, chart or ledger changes unless the user opts in.
    "pa_enabled": False,
    "pa_show_zones": True,
    "pa_show_events": False,
    "pa_ob_invalidation": "close",
    "pa_min_confluence": 3.0,
    "pa_mtf_enabled": False,
    "pa_mtf_timeframe": "auto",
    # ---- v1.3.0 entry filters.  ALL OFF by default, and with every one of them
    # off `generate_signals` returns exactly the frame it returned before this
    # feature existed (pinned by test).  Rejected range: 0 means "no bound".
    "filter_trend": "off",
    "filter_trend_ema": 200,
    "filter_atr_min": 0.0,
    "filter_atr_max": 0.0,
    "filter_volume_ratio": 0.0,
    "filter_volume_lookback": 20,
    "filter_sessions": [],
    "allow_weekends": True,
    # ---- v1.3.0 Indicators tab.  CHART ONLY: these keys only ever decide what
    # is drawn.  Nothing here is read by the strategy, the engine or a backtest.
    "ind_overlays": ["ema", "bollinger"],
    "ind_panes": ["rsi", "volume"],
    "ind_params": {},
    # ---- v1.3.0 walk-forward defaults (in-sample / out-of-sample split).
    "wf_splits": 4,
    "wf_train_ratio": 0.7,
    "wf_anchored": False,
}


def load_settings() -> None:
    """Load persisted settings into session_state (once per session)."""
    if st.session_state.get("_settings_loaded"):
        return
    db = get_db()
    saved = db.get_setting("ui_settings") or {}
    saved_version = int(saved.get("_settings_version", 1)) if saved else SETTINGS_VERSION
    merged = {**DEFAULTS, **{k: v for k, v in saved.items() if k in DEFAULTS}}
    # One-time migration for databases written by an older version.
    migrated = bool(saved) and saved_version < SETTINGS_VERSION
    if migrated:
        for key in ("exchange_key", "symbol", "live_candle", "live_refresh_s"):
            merged[key] = DEFAULTS[key]
    # Defensive coercion: a stored setting could predate a change to the
    # available exchanges, pairs or timeframes.
    if merged.get("exchange_key") not in EXCHANGES:
        merged["exchange_key"] = DEFAULTS["exchange_key"]
    if merged.get("symbol") not in EXCHANGES[merged["exchange_key"]].symbols:
        merged["symbol"] = EXCHANGES[merged["exchange_key"]].default_symbol
    if merged.get("timeframe") not in SUPPORTED_TIMEFRAMES:
        merged["timeframe"] = DEFAULTS["timeframe"]
    if merged.get("fill_model") not in FILL_MODES:
        merged["fill_model"] = DEFAULTS["fill_model"]
    if merged.get("stop_mode") not in STOP_MODES:
        merged["stop_mode"] = DEFAULTS["stop_mode"]
    if merged.get("live_refresh_s") not in LIVE_REFRESH_CHOICES_S:
        merged["live_refresh_s"] = DEFAULTS["live_refresh_s"]
    if merged.get("dash_refresh_s") not in DASH_REFRESH_CHOICES_S:
        merged["dash_refresh_s"] = DEFAULTS["dash_refresh_s"]
    if merged.get("chart_zoom") not in ZOOM_CHOICES:
        merged["chart_zoom"] = DEFAULTS["chart_zoom"]
    merged["candle_limit"] = int(min(MAX_CANDLE_LIMIT,
                                     max(MIN_CANDLE_LIMIT, merged.get("candle_limit", 300))))
    # v1.3.0: a stored selection could name an indicator that no longer exists,
    # or hold parameters outside the bounds the catalogue declares.
    merged["ind_overlays"] = [k for k in (merged.get("ind_overlays") or [])
                              if k in icat.OVERLAYS]
    merged["ind_panes"] = [k for k in (merged.get("ind_panes") or []) if k in icat.PANES]
    raw_params = merged.get("ind_params") or {}
    merged["ind_params"] = {key: icat.validate_params(key, raw_params.get(key))
                            for key in icat.CATALOG}
    if merged.get("filter_trend") not in TREND_FILTER_MODES:
        merged["filter_trend"] = DEFAULTS["filter_trend"]
    merged["filter_sessions"] = [k for k in (merged.get("filter_sessions") or [])
                                 if k in sessions_mod.SESSION_KEYS]
    merged["filter_trend_ema"] = int(min(400, max(5, int(merged.get("filter_trend_ema", 200)))))
    for lo_key in ("filter_atr_min", "filter_atr_max"):
        merged[lo_key] = float(min(100.0, max(0.0, float(merged.get(lo_key, 0.0)))))
    if merged["filter_atr_min"] and merged["filter_atr_max"] \
            and merged["filter_atr_min"] >= merged["filter_atr_max"]:
        merged["filter_atr_max"] = 0.0          # refuse an impossible range
    merged["filter_volume_ratio"] = float(min(100.0, max(0.0, float(
        merged.get("filter_volume_ratio", 0.0)))))
    merged["filter_volume_lookback"] = int(min(500, max(2, int(
        merged.get("filter_volume_lookback", 20)))))
    merged["allow_weekends"] = bool(merged.get("allow_weekends", True))
    merged["wf_splits"] = int(min(8, max(2, int(merged.get("wf_splits", 4)))))
    merged["wf_train_ratio"] = float(min(0.9, max(0.5, float(
        merged.get("wf_train_ratio", 0.7)))))
    merged["wf_anchored"] = bool(merged.get("wf_anchored", False))
    for k, v in merged.items():
        st.session_state.setdefault(k, v)
    st.session_state["_settings_loaded"] = True
    if migrated and not db.get_setting("migration_v2_note_shown", False):
        st.session_state["_migration_note"] = (
            f"Your saved data source was migrated from the previous version to "
            f"**{EXCHANGES[DEFAULT_EXCHANGE_KEY].label}** "
            f"({EXCHANGES[DEFAULT_EXCHANGE_KEY].market_kind}, `{DEFAULTS['symbol']}`), "
            f"and the live forming candle was switched on. Everything else you saved — "
            f"strategy, risk settings, paper trades — is untouched. Pick a different "
            f"exchange in the sidebar at any time; this message is shown only once."
        )
        db.set_setting("migration_v2_note_shown", True)
    st.session_state.setdefault("data_cache", {})
    st.session_state.setdefault("last_fetch_error", None)
    st.session_state.setdefault("reset_token", "")


def save_settings() -> None:
    payload = {k: st.session_state.get(k, v) for k, v in DEFAULTS.items()}
    payload["_settings_version"] = SETTINGS_VERSION
    get_db().set_setting("ui_settings", payload)


OB_INVALIDATION_MODES = ("close", "wick")


def build_price_action_params() -> PriceActionParams:
    """
    Build the OPTIONAL price-action parameter object from the sidebar.

    Kept separate from `build_params()` so no existing call site changes.
    Returns a disabled object when the add-on is switched off, which makes
    `generate_signals` behave exactly as it did in Phase 1.
    """
    # Defensive fallback only.  Never write to a widget-bound session_state key
    # from here: Streamlit forbids that once the widget exists.
    mode = str(st.session_state.get("pa_ob_invalidation", DEFAULTS["pa_ob_invalidation"]))
    if mode not in OB_INVALIDATION_MODES:
        mode = DEFAULTS["pa_ob_invalidation"]
    return PriceActionParams(
        enabled=bool(st.session_state.get("pa_enabled", False)),
        ob_invalidation=mode,
        min_confluence_score=float(st.session_state.get("pa_min_confluence",
                                                        DEFAULTS["pa_min_confluence"])),
        mtf_enabled=bool(st.session_state.get("pa_mtf_enabled", False)),
        mtf_timeframe=str(st.session_state.get("pa_mtf_timeframe", "auto")),
    ).validate()


def _pa_result(df):
    """The PriceActionResult attached by generate_signals, if the add-on ran."""
    try:
        return df.attrs.get("pa_result")
    except AttributeError:
        return None


def _mtf_result(df):
    """The MtfResult attached by generate_signals, if higher-timeframe is on."""
    try:
        return df.attrs.get("mtf_result")
    except AttributeError:
        return None


def _chart_zones(df):
    """Lower-timeframe zones plus (dashed) higher-timeframe zones for the chart."""
    res = _pa_result(df)
    if res is None:
        return None
    zones = list(res.active_zones())
    mtf = _mtf_result(df)
    if mtf is not None and mtf.available:
        zones.extend(mtf.zones())
    return zones or None


def _htf_options() -> list:
    """Higher timeframes strictly above the current chart timeframe."""
    ltf_ms = TIMEFRAME_MS.get(str(st.session_state.get("timeframe")), 0)
    return ["auto"] + [t for t in sorted(TIMEFRAME_MS, key=lambda k: TIMEFRAME_MS[k])
                       if TIMEFRAME_MS[t] > ltf_ms]


def build_params() -> tuple:
    """Turn sidebar values into validated parameter objects."""
    strategy = StrategyParams(
        swing_lookback=int(st.session_state.swing_lookback),
        swing_shift=int(st.session_state.swing_shift),
        rsi_period=int(st.session_state.rsi_period),
        rsi_long_min=float(st.session_state.rsi_long_min),
        rsi_short_max=float(st.session_state.rsi_short_max),
        ema_fast=int(st.session_state.ema_fast),
        ema_slow=int(st.session_state.ema_slow),
        require_close_breakout=bool(st.session_state.require_close_breakout),
        stop_mode=st.session_state.stop_mode,
        stop_buffer_pct=float(st.session_state.stop_buffer_pct),
        atr_period=int(st.session_state.atr_period),
        atr_multiplier=float(st.session_state.atr_multiplier),
        percent_stop=float(st.session_state.percent_stop),
        # v1.3.0 entry filters.  All default to "off"/empty, so an untouched
        # install builds a StrategyParams identical to the previous version.
        trend_filter=str(st.session_state.get("filter_trend", "off")),
        trend_ema_period=int(st.session_state.get("filter_trend_ema", 200)),
        atr_filter_min_pct=float(st.session_state.get("filter_atr_min", 0.0)),
        atr_filter_max_pct=float(st.session_state.get("filter_atr_max", 0.0)),
        volume_filter_ratio=float(st.session_state.get("filter_volume_ratio", 0.0)),
        volume_filter_lookback=int(st.session_state.get("filter_volume_lookback", 20)),
        session_filter=list(st.session_state.get("filter_sessions", []) or []),
        allow_weekends=bool(st.session_state.get("allow_weekends", True)),
    ).validate()

    risk = RiskParams(
        starting_balance=float(st.session_state.starting_balance),
        risk_per_trade_pct=float(st.session_state.risk_per_trade_pct),
        reward_to_risk=float(st.session_state.reward_to_risk),
        fee_pct_per_side=float(st.session_state.fee_pct_per_side),
        slippage_bps=float(st.session_state.slippage_bps),
        daily_loss_limit_pct=float(st.session_state.daily_loss_limit_pct),
        max_notional_pct=float(st.session_state.max_notional_pct),
        allow_shorts=bool(st.session_state.allow_shorts),
        max_open_positions=int(st.session_state.max_open_positions),
    ).validate()

    feed = FeedConfig(
        exchange_key=st.session_state.exchange_key,
        symbol=st.session_state.symbol,
        timeframe=st.session_state.timeframe,
        candle_limit=int(st.session_state.candle_limit),
    ).validate()
    return strategy, risk, feed


# --------------------------------------------------------------------------- #
# Data fetching (session cache + explicit refresh)
# --------------------------------------------------------------------------- #

def fetch_candles(feed: FeedConfig, force: bool = False):
    """
    Fetch candles, respecting a short cooldown so refreshes cannot hammer the
    exchange.  Returns (CandleSet | None, FeedError | None).

    On failure the last known-good frame is returned for DISPLAY, clearly marked
    `served_from_cache`, and the engine refuses to trade on it.
    """
    cache = st.session_state.setdefault("data_cache", {})
    key = f"{feed.feed_key}:{feed.candle_limit}"
    now = time.time()

    entry = cache.get(key)
    if entry and not force and (now - entry["at"]) < FETCH_COOLDOWN_S:
        return entry["cs"], None

    fd = get_feed()
    try:
        cs = fd.fetch_candles(feed, force=force)
        cache[key] = {"cs": cs, "at": now}
        st.session_state["last_fetch_error"] = None
        return cs, None
    except DataFeedError as exc:
        st.session_state["last_fetch_error"] = exc.feed_error
        cached = fd.last_good(feed)
        if cached is not None:
            cache[key] = {"cs": cached, "at": entry["at"] if entry else 0.0}
            return cached, exc.feed_error
        return None, exc.feed_error
    except Exception as exc:                      # never crash the dashboard
        log.exception("Unexpected data-feed failure")
        from core.errors import classify_exception
        fe = classify_exception(exc, feed.exchange_key, feed.symbol, feed.timeframe)
        st.session_state["last_fetch_error"] = fe
        cached = get_feed().last_good(feed)
        return (cached, fe) if cached else (None, fe)


def cache_age_seconds(feed: FeedConfig) -> Optional[float]:
    entry = st.session_state.get("data_cache", {}).get(f"{feed.feed_key}:{feed.candle_limit}")
    return None if not entry else max(0.0, time.time() - entry["at"])


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #

def render_sidebar() -> Dict[str, Any]:
    db = get_db()
    with st.sidebar:
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">'
            f'<div style="width:34px;height:34px;border-radius:9px;flex:0 0 auto;display:flex;'
            f'align-items:center;justify-content:center;font-weight:800;color:#fff;font-size:1.05rem;'
            f'background:linear-gradient(145deg,{ui.ACCENT},{ui.ACCENT_DEEP});'
            f'box-shadow:0 4px 14px rgba(225,29,46,0.35);">₿</div>'
            f'<div><div style="font-size:1.0rem;font-weight:700;letter-spacing:-0.02em;'
            f'color:{ui.TEXT};">{APP_NAME}</div>'
            f'<div style="font-size:0.66rem;color:{ui.MUTED};letter-spacing:0.10em;'
            f'text-transform:uppercase;margin-top:2px;">v{APP_VERSION} · educational simulator</div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )
        ui.sim_banner(SIMULATION_BANNER)

        # ---- market data ------------------------------------------------ #
        # Widgets are bound DIRECTLY to the persisted session-state keys (no
        # parallel "_ui_*" copies).  Double-managed state silently reverts user
        # and programmatic changes, which is a real bug class in Streamlit apps.
        st.markdown("#### Market data")
        st.selectbox(
            "Exchange", list(EXCHANGES.keys()),
            format_func=lambda k: f"{EXCHANGES[k].label} · {EXCHANGES[k].market_kind}",
            help="Public market-data source only. No API key is ever requested. "
                 "The app never switches exchanges silently.",
            key="exchange_key", on_change=_on_exchange_change,
        )
        spec = EXCHANGES[st.session_state.exchange_key]
        st.selectbox(
            "Trading pair", spec.symbols,
            help="Pairs this exchange actually lists. On Delta the only BTC market "
                 "with candle data is the BTC/USDT perpetual - its spot book returns "
                 "nothing at all, so it is not offered.",
            key="symbol",
        )
        if spec.market_kind != "spot":
            st.caption(f"⚠ These are **{spec.market_kind}** prices, not spot. Paper trades "
                       f"simulate buying the instrument itself; funding and margin are not modelled.")
        if spec.thin_data:
            st.caption("⚠ Thin venue: many low-timeframe buckets contain no trades at all. "
                       "The measured zero-volume share is shown with the data, and no signal "
                       "is ever generated from an empty candle.")
        st.selectbox(
            "Timeframe", SUPPORTED_TIMEFRAMES,
            help=("Signals are generated from FULLY CLOSED candles only. Every "
                  "timeframe here is always offered: where the exchange does not "
                  "serve it natively the feed aggregates it from the largest native "
                  "granularity that divides it exactly (Coinbase: 30m from 15m, 4h "
                  "from 1h) and the UI says so. Delta and Kraken serve all of them."),
            key="timeframe",
        )
        st.slider(
            "Candles to fetch", MIN_CANDLE_LIMIT, MAX_CANDLE_LIMIT, value=None, step=10,
            help="More candles = deeper history for the chart and indicators.",
            key="candle_limit",
        )

        st.checkbox(
            "Live candle (show it forming)", key="live_candle",
            help=("Re-polls ONLY the in-progress candle and redraws it. The forming candle "
                  "is display-only: it never enters the frame the strategy reads, so every "
                  "signal still comes from a fully closed candle."),
        )
        if st.session_state.live_candle:
            st.selectbox(
                "Live refresh interval", LIVE_REFRESH_CHOICES_S,
                format_func=lambda v: f"every {int(v)}s",
                key="live_refresh_s",
                help="How often the forming candle is re-polled. One small request per poll.",
            )
        st.selectbox(
            "Market panel refresh", DASH_REFRESH_CHOICES_S,
            format_func=lambda v: "on page load only" if int(v) == 0 else f"every {int(v)}s",
            key="dash_refresh_s",
            help=("How often the 24h statistics strip re-reads its cached public "
                  "ticker. Statistics are cached for "
                  f"{dash.TICKER_TTL_S:.0f}s, so this never floods the exchange. "
                  "It changes NOTHING about signals or trades."),
        )

        c1, c2 = st.columns(2)
        if c1.button("↻ Refresh data", width="stretch",
                     help=f"At most one live fetch per {int(FETCH_COOLDOWN_S)}s per series."):
            try:
                strategy, risk, feed = build_params()
                fetch_candles(feed, force=True)
            except InvalidConfigError as exc:
                st.error(str(exc))
        if c2.button("Run diagnostics", width="stretch"):
            st.session_state["_show_diagnostics"] = True

        st.divider()

        # ---- bot control ------------------------------------------------ #
        st.markdown("#### Paper bot")
        running = bool(db.get_setting(f"bot_running:{_feed_key()}", False))
        st.markdown(
            ui.pill("BOT RUNNING" if running else "BOT STOPPED", "ok" if running else "warn"),
            unsafe_allow_html=True,
        )
        b1, b2 = st.columns(2)
        if b1.button("▶ Start", width="stretch", type="primary"):
            _engine().set_running(True)
            st.rerun()
        if b2.button("■ Stop", width="stretch"):
            _engine().set_running(False)
            st.rerun()

        st.selectbox(
            "Fill model", FILL_MODES,
            key="fill_model",
            help=(
                "next_open (default, realistic): a signal on a closed candle fills at the "
                "NEXT candle's open. signal_close: fills at the signal candle's close, which "
                "is slightly optimistic because you can only observe that price as it closes."
            ),
        )
        st.caption("Slippage and fees are always applied adversely.")

        st.divider()

        # ---- strategy --------------------------------------------------- #
        with st.expander("Strategy parameters", expanded=False):
            st.number_input(
                "Swing lookback (candles)", 2, 200, step=1, key="swing_lookback",
                help="How many past candles define the prior swing high/low.",
            )
            st.number_input(
                "Swing shift (>=1)", 1, 20, step=1, key="swing_shift",
                help=("Shifts the swing level into the past so a candle cannot influence its "
                      "own breakout level. 1 = strictly causal. 0 is rejected as look-ahead bias."),
            )
            st.checkbox(
                "Require candle-CLOSE breakout", key="require_close_breakout",
                help="Off = a wick through the level also counts as a break of structure.",
            )
            c1, c2 = st.columns(2)
            c1.number_input("RSI period", 2, 100, step=1, key="rsi_period")
            c2.number_input("EMA fast", 2, 200, step=1, key="ema_fast",
                            help="Condition 1 of the entry (EMA fast > EMA slow). "
                                 "The lines are no longer drawn on the chart.")
            st.number_input("EMA slow (must exceed fast)", 3, 400, step=1, key="ema_slow",
                            help="Still evaluated by the strategy; not drawn on the chart.")
            st.caption("EMA lines and the volume panel were removed from the chart. "
                       "The EMA crossover is still condition 1 of the four-condition entry, "
                       "so signals and backtest results are unchanged.")
            c1, c2 = st.columns(2)
            c1.number_input("RSI long >=", 1.0, 100.0, value=None, step=1.0, key="rsi_long_min")
            c2.number_input("RSI short <=", 1.0, 100.0, value=None, step=1.0, key="rsi_short_max")

            st.selectbox(
                "Stop-loss mode", STOP_MODES, key="stop_mode",
                help=("swing: beyond the prior swing low/high. atr: a multiple of ATR. "
                      "percent: a fixed % of the entry price."),
            )
            st.slider(
                "Stop buffer %", 0.0, 3.0, value=None, step=0.05, key="stop_buffer_pct",
                help="Extra distance beyond the reference level, to avoid obvious stop hunts.",
            )
            if st.session_state.stop_mode == "atr":
                c1, c2 = st.columns(2)
                c1.number_input("ATR period", 2, 100, step=1, key="atr_period")
                c2.number_input("ATR multiplier", 0.1, 20.0, value=None, step=0.1, key="atr_multiplier")
            if st.session_state.stop_mode == "percent":
                st.number_input("Percent stop", 0.05, 25.0, value=None, step=0.05, key="percent_stop")

            # ---- v1.3.0 entry filters ---------------------------------- #
            st.markdown("**Entry filters** · every one of them is off by default")
            st.caption(
                "A filter can only ever REMOVE a trade — none of them can invent one, and "
                "each is evaluated on closed candles only. With them all off, signals are "
                "byte-for-byte what they were before this feature existed."
            )
            st.selectbox(
                "Trend filter", TREND_FILTER_MODES, key="filter_trend",
                format_func=lambda m: {"off": "off (no trend filter)",
                                       "above": "only above the trend EMA",
                                       "below": "only below the trend EMA"}.get(m, m),
                help=("Requires price to be on one side of a long moving average before an "
                      "entry is allowed. While the average is still warming up the filter "
                      "does not block anything, and the Signals tab says so."),
            )
            if st.session_state.filter_trend != "off":
                st.number_input("Trend EMA period", 5, 400, step=5, key="filter_trend_ema",
                                help="200 is the usual 'is this a bull or bear market' line.")
            c1, c2 = st.columns(2)
            c1.number_input("Min ATR % of price (0 = off)", 0.0, 100.0, value=None,
                            step=0.05, key="filter_atr_min",
                            help="Skips dead, barely-traded stretches.")
            c2.number_input("Max ATR % of price (0 = off)", 0.0, 100.0, value=None,
                            step=0.05, key="filter_atr_max",
                            help="Skips blow-off volatility spikes.")
            c3, c4 = st.columns(2)
            c3.number_input("Volume ≥ x average (0 = off)", 0.0, 100.0, value=None,
                            step=0.1, key="filter_volume_ratio",
                            help="Compares the signal candle's volume with the average of the "
                                 "PREVIOUS candles only — a bar is never part of its own average.")
            c4.number_input("Volume average over", 2, 500, step=1,
                            key="filter_volume_lookback", help="Candles used for that average.")
            st.multiselect(
                "Trade only during", sessions_mod.SESSION_KEYS, key="filter_sessions",
                format_func=sessions_mod.session_label,
                help=("Empty (the default) means every hour is tradable. Selecting windows "
                      "restricts entries to those hours — the Signals tab shows the exact "
                      "IST clock time it decided on."),
            )
            st.checkbox("Allow weekend entries (UTC)", key="allow_weekends",
                        help="Bitcoin never closes, but weekends are measurably thinner. "
                             "Untick to skip Saturday and Sunday.")
            _filters_note = sessions_mod.describe_sessions(st.session_state.filter_sessions)
            if not st.session_state.allow_weekends:
                _filters_note += " Weekends are excluded."
            if st.session_state.filter_trend != "off":
                _filters_note += (f" Price must be {st.session_state.filter_trend}"
                                  f" EMA{st.session_state.filter_trend_ema}.")
            st.caption(_filters_note)
            if any([st.session_state.filter_trend != "off",
                    float(st.session_state.filter_atr_min) > 0,
                    float(st.session_state.filter_atr_max) > 0,
                    float(st.session_state.filter_volume_ratio) > 0,
                    bool(st.session_state.filter_sessions),
                    not bool(st.session_state.allow_weekends)]):
                st.caption(
                    "Filters are ON. The Backtest tab has a walk-forward test that splits "
                    "history into in-sample / out-of-sample windows, so you can see whether "
                    "this actually holds up on candles it was not chosen from."
                )

        # ---- price action (add-on) -------------------------------------- #
        with st.expander("Price action add-on", expanded=False):
            st.checkbox(
                "Enable price-action analysis", key="pa_enabled",
                help=(
                    "Adds CHoCH/MSS, liquidity sweeps, fair value gaps, order blocks, "
                    "supply/demand zones, candlestick patterns, breakout-retest tracking "
                    "and premium/discount. ANALYSIS ONLY: it never changes the four "
                    "conditions that open a simulated trade."
                ),
            )
            if bool(st.session_state.pa_enabled):
                st.checkbox("Draw zones on the chart", key="pa_show_zones")
                st.checkbox("Draw event markers on the chart", key="pa_show_events",
                            help="Sweeps, CHoCH/MSS, gap and breakout events.")
                st.selectbox(
                    "Order-block invalidation", list(OB_INVALIDATION_MODES),
                    key="pa_ob_invalidation",
                    help=("close: the block dies when a candle CLOSES through it. "
                          "wick: it dies as soon as a wick trades through it (stricter)."),
                )
                st.slider(
                    "Confluence threshold", 0.0, 10.0, value=None, step=0.5,
                    key="pa_min_confluence",
                    help=("A confluence read is only called 'actionable' when the agreeing "
                          "weight reaches this score. Advisory - it gates nothing."),
                )
                st.checkbox(
                    "Add higher-timeframe context", key="pa_mtf_enabled",
                    help=(
                        "Aggregates your closed candles up to a higher timeframe, runs the "
                        "same price-action rules on it, and shows that bias / zones / "
                        "premium-discount next to the lower-timeframe read. Only FULLY "
                        "CLOSED higher-timeframe candles are used - the one still forming "
                        "is excluded, so it cannot look ahead."
                    ),
                )
                if bool(st.session_state.pa_mtf_enabled):
                    # Repair the persisted choice BEFORE the widget is created: a
                    # value that is no longer above the chart timeframe would make
                    # Streamlit raise (e.g. chart moved from 1h to 4h).
                    opts = _htf_options()
                    if st.session_state.get("pa_mtf_timeframe") not in opts:
                        st.session_state["pa_mtf_timeframe"] = "auto"
                    st.selectbox(
                        "Higher timeframe", opts, key="pa_mtf_timeframe",
                        help=("auto = one step up the chain "
                              "(5m→15m→30m→1h→4h→1d). Needs enough history: at least "
                              "30 closed higher-timeframe candles, otherwise no "
                              "higher-timeframe context is published."),
                    )
                    st.caption(
                        f"Chart timeframe {st.session_state.timeframe} → higher timeframe "
                        f"{resolve_htf(str(st.session_state.timeframe), str(st.session_state.pa_mtf_timeframe)) or 'n/a'}."
                    )
                st.caption(
                    "Zones are price-based interpretations of candles. They are not proof "
                    "of hidden institutional orders and they guarantee nothing."
                )

        # ---- risk ------------------------------------------------------- #
        with st.expander("Risk management", expanded=False):
            st.number_input(
                "Starting virtual balance (USD)", 100.0, 10_000_000.0,
                value=None, step=1000.0, key="starting_balance",
                help="Only used when the account is reset; it never edits the live ledger.",
            )
            st.slider(
                "Risk per trade %", 0.05, 10.0, value=None, step=0.05, key="risk_per_trade_pct",
                help="Percentage of equity lost if the stop is hit, fees included.",
            )
            st.slider(
                "Reward : risk target", 0.5, 10.0, value=None, step=0.5, key="reward_to_risk",
                help="Take-profit distance as a multiple of the stop distance.",
            )
            c1, c2 = st.columns(2)
            c1.number_input(
                "Fee % per side", 0.0, 1.0, value=None, step=0.01, key="fee_pct_per_side",
                help="Simulated taker-style fee, applied on entry AND exit.",
            )
            c2.number_input(
                "Slippage (bps)", 0.0, 100.0, value=None, step=1.0, key="slippage_bps",
                help="Adverse slippage in basis points, applied to every fill.",
            )
            st.slider(
                "Daily loss limit %", 0.5, 20.0, value=None, step=0.5, key="daily_loss_limit_pct",
                help=("Percentage of the balance at 00:00 UTC. When breached, no new entries "
                      "until the next UTC day. Open positions keep their stops."),
            )
            st.slider(
                "Max position notional % of equity", 5.0, 100.0, value=None, step=5.0,
                key="max_notional_pct",
                help="Caps position size. 100% = full cash, no leverage.",
            )
            st.number_input("Max open positions", 1, 5, step=1, key="max_open_positions")
            st.checkbox("Allow simulated SHORT positions", key="allow_shorts")

        st.divider()

        # ---- chart options ---------------------------------------------- #
        with st.expander("Chart options", expanded=False):
            st.checkbox("Show prior swing levels", key="show_swings")
            st.checkbox("Show BUY / SELL markers", key="show_signals")
            st.caption("Removed from the chart: EMA 20 / EMA 50 overlays and the volume panel.")

        if st.button("💾 Save settings", width="stretch"):
            try:
                build_params()          # validate before persisting
                save_settings()
                st.success("Settings saved.")
            except InvalidConfigError as exc:
                st.error(str(exc))

        # ---- danger zone ------------------------------------------------ #
        with st.expander("⚠ Reset virtual account", expanded=False):
            st.warning(
                "This permanently deletes ALL paper trades, bot events and the "
                "processed-candle log. Backtest history is kept. This cannot be undone.",
                icon="⚠️",
            )
            st.text_input(f"Type {RESET_CONFIRM_TOKEN} to confirm",
                          placeholder=RESET_CONFIRM_TOKEN, key="reset_token")
            if st.button("Delete paper-trading history", width="stretch",
                         disabled=st.session_state.reset_token != RESET_CONFIRM_TOKEN):
                counts = db.reset_paper_account(RESET_CONFIRM_TOKEN)
                db.set_setting(f"bot_running:{_feed_key()}", False)
                st.session_state.reset_token = ""
                st.session_state.pop("data_cache", None)
                st.success(
                    f"Reset complete. Deleted {counts.get('paper_trades', 0)} trade(s), "
                    f"{counts.get('bot_events', 0)} event(s), "
                    f"{counts.get('processed_candles', 0)} candle marker(s)."
                )
                st.rerun()

        st.caption(
            "Educational simulator. Not financial advice. No real orders are ever sent."
        )
        ui.sidebar_credit(COPYRIGHT_LINE, f"{APP_NAME} v{APP_VERSION}")

    save_settings()
    return {}


def _on_exchange_change() -> None:
    """
    Runs BEFORE the rerun when the exchange changes.

    Different exchanges list different pairs, so the currently selected symbol
    may no longer be valid.  Resetting it here (rather than after the widget is
    drawn) is the sanctioned Streamlit pattern and avoids a widget-state error.
    """
    spec = EXCHANGES[st.session_state.exchange_key]
    if st.session_state.get("symbol") not in spec.symbols:
        st.session_state.symbol = spec.default_symbol
    if st.session_state.get("timeframe") not in SUPPORTED_TIMEFRAMES:
        st.session_state.timeframe = "1h"


def _feed_key() -> str:
    return f"{st.session_state.exchange_key}:{st.session_state.symbol}:{st.session_state.timeframe}"


def _engine() -> PaperTradingEngine:
    strategy, risk, feed = build_params()
    return PaperTradingEngine(
        get_db(), feed, strategy, risk,
        fill_model=st.session_state.fill_model,
        bot_running=bool(get_db().get_setting(f"bot_running:{feed.feed_key}", False)),
        starting_balance=float(st.session_state.starting_balance),
    )


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #

def render_header(cs: Optional[CandleSet], feed_err, running: bool, age: Optional[float]) -> None:
    spec = EXCHANGES[st.session_state.exchange_key]
    if cs is not None and feed_err is None:
        status = ("LIVE", "ok")
    elif cs is not None and feed_err is not None:
        status = ("STALE / CACHED", "warn")
    else:
        status = ("DISCONNECTED", "bad")

    live_on = bool(st.session_state.get("live_candle", False))
    ui.masthead(
        "BTC Market Terminal",
        f"{spec.label} · {spec.market_kind} · {st.session_state.symbol} · "
        f"{st.session_state.timeframe}"
        + (f" · aggregated from {cs.source_timeframe}" if (cs and cs.is_aggregated) else ""),
        [
            (f"Feed: {status[0]}", status[1]),
            ("LIVE CANDLE" if live_on else "CLOSED-CANDLE VIEW", "live" if live_on else "info"),
            ("PAPER MODE", "info"),
            ("BOT RUNNING" if running else "BOT STOPPED", "ok" if running else "warn"),
        ],
    )
    ui.sim_banner(SIMULATION_BANNER)

    cols = st.columns(4)
    last_update = "-"
    if cs is not None and cs.fetched_at_ms:
        last_update = datetime.fromtimestamp(
            cs.fetched_at_ms / 1000.0, timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    cols[0].markdown(ui.card("Last successful update (UTC)", last_update,
                             f"cache age {age:.0f}s" if age is not None else "no cached fetch",
                             "neu" if feed_err is None else "warn"))
    cols[1].markdown(ui.card("Last closed candle",
                             _ts_str(cs.last_closed_ts) if cs else "-",
                             f"candle age {cs.age_seconds:.0f}s" if cs else "",
                             "warn" if (cs and cs.is_stale) else "neu"))
    cols[2].markdown(ui.card("Closed candles", f"{cs.candle_count:,}" if cs else "0",
                             f"{cs.missing_count} gap(s)" if cs else "", "neu"))
    cols[3].markdown(ui.card("Forming candle",
                             f"{cs.forming['close']:,.2f}" if (cs and cs.forming) else "-",
                             "excluded from all signals", "neu"))

    if feed_err is not None:
        kind = "error" if cs is None else "warning"
        getattr(st, kind)(
            f"**{feed_err.category.replace('_', ' ').upper()}** — {feed_err.message}\n\n"
            f"**Recovery steps**\n{feed_err.recovery_text}",
            icon="🚫" if cs is None else "⚠️",
        )
    if cs is not None:
        # Data quality gets a proper warning box (it changes how the numbers
        # should be read); everything else stays a quiet caption.
        note = cs.data_quality_note
        if note:
            st.warning(note, icon="📉")
        for w in cs.warnings[:4]:
            if w.startswith("Thin market data"):
                continue          # already shown above, once
            st.caption(f"⚠ {w}")


def _ts_str(ms: Optional[float]) -> str:
    """
    Format a millisecond timestamp, or "-" when there is not one.

    NaN-safe on purpose: an OPEN trade has no exit timestamp, and pandas turns
    that SQL NULL into NaN as soon as the ledger frame mixes open and closed
    rows.  `float(NaN)` is truthy, so a naive check would reach
    `datetime.fromtimestamp(nan)` and take the whole page down with
    "Invalid value NaN (not a number)".
    """
    if ms is None:
        return "-"
    try:
        value = float(ms)
    except (TypeError, ValueError):
        return "-"
    if value != value or value <= 0:          # NaN / never set
        return "-"
    return datetime.fromtimestamp(value / 1000.0, timezone.utc).strftime("%m-%d %H:%M UTC")


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def render_metrics(eng: PaperTradingEngine, cs: Optional[CandleSet]) -> None:
    db = get_db()
    mark = float(cs.df["close"].iloc[-1]) if (cs is not None and not cs.is_empty) else None
    if cs is not None and cs.forming and cs.forming.get("close"):
        mark = float(cs.forming["close"])

    balance = eng.realized_balance()
    equity = eng.equity(mark)
    unrealized = eng.unrealized(mark)
    net_total = equity - eng.starting_balance

    closed = db.get_trades(status="closed", limit=5000)
    from core.metrics import compute_metrics
    m = compute_metrics(closed, eng.starting_balance)

    midnight = None
    from core.db import utc_midnight_ms
    midnight = utc_midnight_ms()
    daily_realized = db.realized_pnl_since(midnight)
    daily = eng.daily_state()
    open_trades = db.get_open_trades()

    st.markdown("#### Account performance")
    ui.card_row([
        ui.card("Starting balance", ui.money(eng.starting_balance), "virtual USD", "neu"),
        ui.card("Realized equity", ui.money(balance),
                f"{len(closed)} closed trade(s)", "neu"),
        ui.card("Unrealized P/L", ui.money(unrealized, signed=True),
                "marked at the latest price" if mark else "no live price",
                ui.tone_for(unrealized)),
        ui.card("Total net P/L", ui.money(net_total, signed=True),
                f"{(net_total / eng.starting_balance * 100):+.2f}% of start",
                ui.tone_for(net_total)),
    ])
    ui.card_row([
        ui.card("Daily realized P/L", ui.money(daily_realized, signed=True),
                f"since 00:00 UTC ({daily.utc_date})", ui.tone_for(daily_realized)),
        ui.card("Win rate",
                "n/a" if m.win_rate is None else f"{m.win_rate * 100:.1f}%",
                f"{m.wins}W / {m.losses}L / {m.scratches}BE",
                "neu" if m.win_rate is None else ("pos" if m.win_rate >= 0.5 else "neg")),
        ui.card("Profit factor",
                "n/a" if m.profit_factor is None else f"{m.profit_factor:.2f}",
                "gross profit / gross loss" if m.profit_factor is not None else "no losing trades yet",
                "neu"),
        ui.card("Max drawdown",
                f"-{m.max_drawdown:,.2f}", f"{m.max_drawdown_pct:.2f}% from peak",
                "neg" if m.max_drawdown > 0 else "neu"),
    ])
    ui.card_row([
        ui.card("Closed trades", f"{m.closed_trades}",
                f"{len(open_trades)} open", "neu"),
        ui.card("Avg win / avg loss",
                (f"{m.avg_win:,.0f} / {m.avg_loss:,.0f}"
                 if (m.avg_win is not None and m.avg_loss is not None) else "n/a"),
                f"avg R {m.avg_r:+.2f}" if m.avg_r is not None else "insufficient trades",
                "neu"),
        ui.card("Open position",
                (f"{open_trades[0]['direction'].upper()} {open_trades[0]['quantity']:.6f}"
                 if open_trades else "FLAT"),
                (f"entry {open_trades[0]['entry_price']:,.2f}" if open_trades else "no exposure"),
                "pos" if open_trades else "neu"),
        ui.card("Risk per trade",
                f"{eng.risk.risk_per_trade_pct:.2f}%",
                f"daily limit {eng.risk.daily_loss_limit_pct:.1f}% · "
                f"{'LOCKED' if daily.locked else f'{daily.used_pct:.0f}% used'}",
                "neg" if daily.locked else "neu"),
    ])

    if daily.locked:
        st.error(daily.message, icon="🔒")
    else:
        st.caption(daily.message)

    with st.expander("How these metrics are calculated"):
        for line in describe(m):
            st.markdown(f"- {line}")
        st.markdown(
            "**Definitions.** Win rate = wins / all closed trades (break-even trades count as "
            "scratches, not wins). Profit factor = gross profit / |gross loss|, undefined when "
            "there are no losing trades. Max drawdown = largest peak-to-trough fall of the "
            "*realised* equity curve. Unrealized P/L is an estimate marked at the latest price "
            "and does not include the exit fee."
        )
        st.info(
            "A stop-loss does not guarantee an exact realised loss. Fills, fees and slippage "
            "in this simulator are estimates; real execution can be worse.", icon="ℹ️")


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #

def _live_status_row(cs: Optional[CandleSet], forming: Optional[Dict[str, float]],
                     poll_s: Optional[float], warnings: List[str],
                     live_on: bool = True) -> None:
    """One row of pills describing the live candle, honestly."""
    if not live_on:
        # The forming candle still exists - we are simply not polling or drawing
        # it.  Saying "no forming candle available" here would be a lie.
        last = _ts_str(cs.last_closed_ts) if cs is not None else "-"
        ui.pills([("live candle off", "info"), (f"newest closed candle {last}", "info"),
                  ("signals unchanged", "ok")])
        for w in warnings:
            st.caption(f"⚠ {w}")
        return

    items: List[tuple] = []
    tf_ms = TIMEFRAME_MS.get(str(st.session_state.timeframe), 3_600_000)
    if forming and forming.get("timestamp"):
        now_ms = (cs.reference_now_ms if cs is not None else time.time() * 1000.0)
        closes_in = max(0.0, (float(forming["timestamp"]) + tf_ms - now_ms) / 1000.0)
        close_txt = (f"closes in {int(closes_in // 60):02d}:{int(closes_in % 60):02d}"
                     if closes_in > 0 else "closing now")
        close = forming.get("close")
        if close:
            items.append((f"LIVE {close:,.2f}", "live"))
        items.append((close_txt, "info"))
        vol = float(forming.get("volume") or 0.0)
        if vol <= 0.0:
            items.append(("no trades in this candle yet", "warn"))
        else:
            items.append((f"volume {vol:,.2f}", "info"))
    else:
        items.append(("no forming candle available", "warn"))
    if poll_s is not None:
        items.append((f"polled {poll_s * 1000:.0f} ms ago", "info"))
    ui.pills(items)
    for w in warnings:
        st.caption(f"⚠ {w}")


def _render_live_chart(cs: Optional[CandleSet], cache: Dict[str, Any],
                       live_on: bool, open_trade: Optional[Dict[str, Any]]) -> None:
    """
    Fragment body: draw the cached closed-candle chart plus the live candle.

    Runs on its own timer (`run_every`) so the forming candle updates without a
    full page rerun.  It ONLY re-polls the in-progress bucket; the closed
    candles, the indicators and every signal come from the outer run and remain
    closed-candle-only.
    """
    base = cache.get("fig")
    if base is None:
        st.info("Chart is not ready yet.")
        return

    forming = cs.forming if cs is not None else None
    warnings: List[str] = []
    poll_s: Optional[float] = None
    if live_on:
        t0 = time.time()
        try:
            _strategy, _risk, feed_cfg = build_params()
            live = get_feed().fetch_forming(feed_cfg)
            if live.get("forming"):
                forming = live["forming"]
            warnings = list(live.get("warnings") or [])
            poll_s = time.time() - t0
        except InvalidConfigError as exc:
            warnings = [f"Live candle paused - invalid settings: {exc}"]
        except Exception as exc:                                  # noqa: BLE001
            warnings = [f"Live candle poll failed ({type(exc).__name__}); showing the last known forming candle."]
            log.warning("Live candle poll failed: %s", exc)

    if live_on and forming:
        fig = copy.deepcopy(base)
        charts.overlay_live_candle(fig, forming, str(st.session_state.timeframe))
    else:
        fig = base

    st.plotly_chart(fig, width="stretch", key="main_chart",
                    config={"displaylogo": False, "scrollZoom": True})
    if live_on and forming:
        # Read-out for the candle that is still being built.  Labelled, and kept
        # inside the fragment so it always matches the candle on the chart.
        previous = (cs.df.iloc[-1] if (cs is not None and not cs.is_empty) else None)
        rows = []
        if previous is not None:
            rows.append({k: previous[k] for k in ("open", "high", "low", "close")})
        rows.append({k: forming.get(k) for k in ("open", "high", "low", "close")})
        render_ohlc_readout(pd.DataFrame(rows), tag="FORMING candle · excluded from every signal")
    _live_status_row(cs, forming, poll_s, warnings, live_on=live_on)
    if live_on:
        st.caption(
            "The hollow dashed candle is still forming. It is drawn for context only: "
            "it never enters the indicator or signal calculations, which use fully "
            "closed candles exclusively."
        )


# --------------------------------------------------------------------------- #
# v1.2.0: public-market strip, quick actions, chart toolbar, position rail
# --------------------------------------------------------------------------- #

def _utc_clock(ms: float) -> str:
    return datetime.fromtimestamp(float(ms) / 1000.0, timezone.utc).strftime("%H:%M:%S UTC")


def _price_str(value: Optional[float]) -> str:
    """Prices keep more decimals when they are small (DOGE 0.0805 vs BTC 82,512)."""
    if value is None:
        return "-"
    return dash.format_price(value, 5 if abs(value) < 10 else 2)


def _render_market_strip_body(feed_cfg: FeedConfig, cs: Optional[CandleSet]) -> None:
    """
    Fragment body: the 24h public statistics strip.

    Runs on its own timer (sidebar "Market panel refresh") so the strip updates
    without a page-wide rerun.  It reads ONLY the public ticker snapshot - it
    cannot influence signals, the ledger or the paper engine.
    """
    spec = EXCHANGES[feed_cfg.exchange_key]
    venue = spec.label.split(" (")[0]
    status = dash.fetch_ticker_status(get_feed(), feed_cfg.exchange_key, feed_cfg.symbol)
    badge = f"{'PERP' if status.is_perp else spec.market_kind.upper()} · {venue}"
    pair = status.symbol.split(":")[0]

    if status.ok and status.last is not None:
        if status.change_abs is None or status.change_pct is None:
            change = status.change_label
        else:
            change = (f"{status.arrow} {_price_str(status.change_abs)} "
                      f"({dash.format_pct(status.change_pct)})")
        cells = list(status.stats_rows())
        cells.append(("Snapshot (UTC)", _utc_clock(status.fetched_at_ms), "neu"))
        age = dash.cache_age_s()
        note_bits = [f"snapshot {age:.0f}s old" if (age is not None and age >= 2) else "fresh snapshot",
                     "public endpoint · no API key · market statistics only, never an input to signals"]
        if status.note:
            note_bits.append(status.note)
        ui.ticker_strip(pair, _price_str(status.last), change, status.tone, cells,
                        badge=badge, note=" · ".join(note_bits))
        return

    # Honest failure state: keep the strip's shape, say what failed, fall back to
    # the last CLOSED candle close (labelled as such) instead of showing a zero.
    fallback = None
    if cs is not None and not cs.is_empty:
        fallback = float(cs.df["close"].iloc[-1])
    blanks = [(name, "-", "neu") for name in
              ("24h change", "24h high", "24h low", "24h volume", "Funding", "Mark price")]
    ui.ticker_strip(
        pair or st.session_state.symbol, _price_str(fallback), "live statistics unavailable", "neu",
        blanks, badge=badge,
        note=(f"The public ticker request failed ({status.note or 'unknown error'}). "
              + (f"The price shown is the close of the newest CLOSED candle ({_ts_str(cs.last_closed_ts)}), "
                 f"not a live tick. " if fallback is not None else "")
              + "Press 'Refresh market data' to retry; candles, signals and trades are unaffected."),
    )


def render_market_strip(feed_cfg: FeedConfig, cs: Optional[CandleSet]) -> None:
    """Create the auto-refreshing statistics strip (one small request per tick)."""
    interval = int(st.session_state.get("dash_refresh_s", DEFAULT_DASH_REFRESH_S) or 0)
    fragment = st.fragment(run_every=(interval if interval > 0 else None))
    fragment(_render_market_strip_body)(feed_cfg, cs)


def _flash(text: str, kind: str = "success") -> None:
    """
    Queue a one-shot message for the NEXT script run.

    Needed because a button that clears caches also calls `st.rerun()`, which
    throws away anything rendered in the current run - the user would click and
    see nothing at all.
    """
    st.session_state["_flash"] = (kind, text)


def _drain_flash() -> None:
    pending = st.session_state.pop("_flash", None)
    if not pending:
        return
    kind, text = pending
    getattr(st, kind, st.info)(text)


def render_quick_actions(feed_cfg: FeedConfig) -> None:
    """
    Shortcut row above the tabs.

    Every button calls an EXISTING code path (the feed cache, the connection
    probe, the trade ledger) - nothing is duplicated, and these shortcuts hold
    no state of their own.  Starting/stopping the bot deliberately stays in the
    sidebar only: that state is owned in exactly one place.
    """
    db = get_db()
    _drain_flash()
    with st.container(key="qbar"):
        cols = st.columns([1.2, 1.2, 1.2, 2.6])
        if cols[0].button("↻ Refresh market data", width="stretch",
                          help=("Drops the cached candles and the cached ticker snapshot, then "
                                "re-reads both from the public API. The 20s cooldown still applies.")):
            clear_caches()
            dash.clear_caches()
            st.session_state["data_cache"] = {}
            st.session_state.pop("diag_status", None)
            _flash("Caches cleared - re-reading the public feed at the next rerun.")
            st.rerun()
        if cols[1].button("🔌 Connection check", width="stretch",
                          help=("Runs the same probe as Diagnostics → 'Run full connection check' "
                                "(server clock, markets, symbol, candles). The result is also "
                                "stored for the Diagnostics tab.")):
            with st.spinner("Probing the exchange…"):
                check = get_feed().check_connection(
                    feed_cfg.exchange_key, feed_cfg.symbol, feed_cfg.timeframe, force=True)
            st.session_state["diag_status"] = check
            ui.pills([
                ("CONNECTION OK" if check.ok else "CONNECTION FAILED", "ok" if check.ok else "bad"),
                (f"stage {check.stage}", "info"),
                (f"{check.latency_ms:.0f} ms", "info"),
                (f"sample {check.sample_price:,.2f}" if check.sample_price else "no sample price", "info"),
                (f"native {feed_cfg.timeframe}: {check.timeframe_native}", "info"),
            ])
            if not check.ok:
                st.error(f"**{check.category or 'failure'}** — {check.message}\n\n"
                         f"**Recovery steps**\n" +
                         "\n".join(f"{i}. {step}" for i, step in enumerate(check.recovery, 1)))
        trades = db.get_trades(limit=5000)
        csv = (pd.DataFrame(trades).to_csv(index=False).encode() if trades
               else b"trade_id,direction,entry_ts,entry_price,exit_ts,exit_price,net_pnl\n")
        cols[2].download_button(
            "⬇ Trade history CSV", csv, file_name="paper_trades.csv", mime="text/csv",
            width="stretch", disabled=not trades,
            help="The same ledger the Trades tab shows, as CSV. Empty file until the first trade.")
        cols[3].caption(
            "Shortcuts, not new machinery: they call the same feed, probe and ledger the "
            "sidebar and Diagnostics tabs use. Start/Stop stays in the sidebar, which owns it."
        )


def _set_timeframe(tf: str) -> None:
    """
    Toolbar button callback: switch the timeframe.

    A callback runs BEFORE the script re-executes, which is the only safe moment
    to change a widget-bound key - mutating it later raises, and copying it into
    a second key would create two sources of truth for one setting.
    """
    st.session_state["timeframe"] = tf


def _toggle_live_candle() -> None:
    """Toolbar button callback: flip the forming-candle switch (sidebar owns it)."""
    st.session_state["live_candle"] = not bool(st.session_state.get("live_candle", True))


def render_chart_toolbar(cs: Optional[CandleSet]) -> str:
    """Timeframe buttons, view zoom and the live toggle.  Returns the zoom label."""
    zoom = st.session_state.get("chart_zoom", DEFAULT_ZOOM)
    if zoom not in ZOOM_CHOICES:
        zoom = DEFAULT_ZOOM
    current_tf = str(st.session_state.get("timeframe", ""))
    live_on = bool(st.session_state.get("live_candle", False))
    with st.container(key="charttoolbar"):
        cols = st.columns([0.62] * len(SUPPORTED_TIMEFRAMES) + [1.15, 2.5])
        for i, tf in enumerate(SUPPORTED_TIMEFRAMES):
            spec = EXCHANGES[st.session_state.exchange_key]
            native = spec.supports_timeframe_natively(tf)
            cols[i].button(
                tf, key=f"tfbtn_{tf}", width="stretch",
                type="primary" if tf == current_tf else "secondary",
                on_click=_set_timeframe, args=(tf,),
                help=(f"{tf} candles are displayed now." if tf == current_tf else
                      (f"Switch to {tf} candles." if native else
                       f"Switch to {tf} candles - {spec.label} does not serve {tf} "
                       f"natively, so the feed aggregates it from real lower candles.")),
            )
        cols[-2].button("● LIVE" if live_on else "○ live off", key="livebtn", width="stretch",
                        type="primary" if live_on else "secondary",
                        on_click=_toggle_live_candle,
                        help=("The forming candle is drawn hollow and dashed. Toggling it changes "
                              "the DISPLAY only - the sidebar switch and every signal are unchanged."))
        cols[-1].radio("View", list(ZOOM_CHOICES), key="chart_zoom", horizontal=True,
                       label_visibility="collapsed",
                       help=("How many candles the chart draws. A view-only setting: it never "
                             "changes the frame the strategy, indicators or signals read."))
    return zoom


def render_ohlc_readout(df: pd.DataFrame, *, tag: str) -> None:
    """One-line O/H/L/C read-out for the newest row of `df`."""
    if df is None or df.empty:
        return
    row = df.iloc[-1]
    change = None
    prev_pct = None
    try:
        if len(df) > 1:
            prev = float(df["close"].iloc[-2])
            change = float(row["close"]) - prev
            prev_pct = (change / prev * 100.0) if prev else None
    except (TypeError, ValueError):
        # A field the venue has not published yet (an empty forming bucket has
        # no close) must not break the read-out - it shows what exists.
        change = None
        prev_pct = None
    items = [
        ("O", _price_str(row["open"]), "neu"),
        ("H", _price_str(row["high"]), "neu"),
        ("L", _price_str(row["low"]), "neu"),
        ("C", _price_str(row["close"]), "neu"),
    ]
    if change is not None:
        sign = "+" if change > 0 else ""
        items.append(("Δ", f"{sign}{_price_str(change)} ({sign}{prev_pct:.2f}%)"
                      if prev_pct is not None else f"{sign}{_price_str(change)}",
                      ui.tone_for(change)))
    ui.readout(items, tag=tag)


def render_position_rail(eng: PaperTradingEngine, cs: Optional[CandleSet],
                         open_trade: Optional[Dict[str, Any]]) -> None:
    """Right-hand rail: position, pending orders, recent closed trades."""
    db = get_db()
    mark = None
    if cs is not None and not cs.is_empty:
        mark = float(cs.df["close"].iloc[-1])
        if cs.forming and cs.forming.get("close"):
            mark = float(cs.forming["close"])

    from core.simulation import mark_to_market
    if open_trade and mark:
        qty = float(open_trade["quantity"])
        upnl = mark_to_market(open_trade["direction"], qty, float(open_trade["entry_price"]), mark)
        notional = qty * mark
        equity = eng.equity(mark)
        stop = float(open_trade["stop_loss"] or 0.0)
        tp = float(open_trade["take_profit"] or 0.0)
        to_stop = (mark - stop) if open_trade["direction"] == "long" else (stop - mark)
        to_tp = (tp - mark) if open_trade["direction"] == "long" else (mark - tp)
        ui.rail_card(
            f"Position · {open_trade['direction'].upper()}", [
                ("Size", f"{qty:.6f}", "neu"),
                ("Entry", f"{float(open_trade['entry_price']):,.2f}", "neu"),
                ("Mark", f"{mark:,.2f}", "neu"),
                ("Notional", ui.money(notional), "neu"),
                ("Stop-loss", f"{stop:,.2f}", "neg"),
                ("Distance to stop", f"{to_stop:,.2f}", "neg" if to_stop > 0 else "warn"),
                ("Take-profit", f"{tp:,.2f}", "pos"),
                ("Distance to target", f"{to_tp:,.2f}", "pos" if to_tp > 0 else "warn"),
                ("Free balance", ui.money(max(equity - notional, 0.0)), "neu"),
            ],
            badge=f"risk {float(open_trade.get('initial_risk') or 0):,.2f}",
            big=(f"{ui.money(upnl, signed=True)}", ui.tone_for(upnl)),
        )
        st.caption("1x notional, no leverage - so there is no liquidation price to show, and "
                   "no margin call exists in this simulator.")
    else:
        equity = eng.equity(mark)
        realized = db.realized_pnl_total()
        rows = [("Wallet balance", ui.money(eng.realized_balance()), "neu"),
                ("Equity", ui.money(equity), "neu"),
                ("Realized P/L (all time)", ui.money(realized, signed=True),
                 ui.tone_for(realized))]
        ui.rail_card("Position", rows, badge="FLAT", big=("FLAT", "neu"),
                     empty="No open position. The engine opens one when a fully closed candle "
                           "satisfies all four conditions and risk allows it.")

    pending = db.get_pending_orders(feed_key=getattr(eng, "feed_key", None))
    if pending:
        rows = [(f"{p['direction'].upper()}", f"{float(p.get('quantity') or 0):.6f} @ market-open",
                 "pos" if p["direction"] == "long" else "neg") for p in pending[:6]]
        ui.rail_card("Open orders", rows, badge=f"{len(pending)} pending")
    else:
        ui.rail_card("Open orders", [], badge="none",
                     empty=("No pending orders. With the 'next candle open' fill model a pending "
                            "row only exists between the signal candle and the next open."))

    closed = db.get_trades(status="closed", limit=6)
    if closed:
        rows = []
        for t in reversed(closed):
            ts = _ts_str(t.get("exit_ts"))
            pnl = float(t.get("net_pnl") or 0.0)
            rows.append((f"{ts} · {str(t.get('direction', '')).upper()}",
                         ui.money(pnl, signed=True), ui.tone_for(pnl)))
        ui.rail_card("Trade history", rows, badge=f"last {len(rows)}")
    else:
        ui.rail_card("Trade history", [], badge="empty",
                     empty="No closed trades yet. The ledger records every simulated fill, "
                           "including fees and slippage.")


def tab_market(feed_cfg: FeedConfig, cs: Optional[CandleSet]) -> None:
    """
    Market overview: public 24h statistics for the venue's comparable markets.

    Two honest shapes: a RANKED panel when the venue publishes real rolling-24h
    statistics, and a plainly-labelled WATCH LIST when it does not (Coinbase's
    public ticker carries a last price only).  The two are never mixed in one
    ranking column.
    """
    spec = EXCHANGES[feed_cfg.exchange_key]
    st.markdown("#### Market overview")
    st.caption(
        f"Public 24h statistics from {spec.label} - no API key, no account, no order. "
        f"These numbers describe the market; your paper account's numbers come from the ledger below."
    )
    if st.button("↻ Refresh these statistics", help="Bypasses the 10s cache and re-reads the public tickers."):
        dash.clear_caches()
        st.rerun()

    try:
        symbols = dash.movers_universe(get_feed(), feed_cfg.exchange_key,
                                       include=feed_cfg.symbol, prefer_quote=feed_cfg.symbol.split("/")[-1].split(":")[0])
        rows = dash.fetch_movers(get_feed(), feed_cfg.exchange_key, symbols)
    except Exception as exc:                                          # noqa: BLE001
        st.warning(f"Could not read the public ticker set for this venue "
                   f"({type(exc).__name__}). Candles, signals and trades are unaffected.")
        rows = []

    if not rows:
        st.info(
            "No comparable markets could be read on this venue right now. This panel never "
            "invents a price to fill the gap - check the Diagnostics tab for the connection state."
        )
    elif dash.panel_mode(rows) == "ranked":
        ranked = dash.rank_movers(rows, MOVERS_ROWS)
        cols = st.columns(3)
        for col, (key, title) in zip(cols, (("volume", "Most traded · 24h volume"),
                                            ("gainers", "Top gainers · 24h"),
                                            ("losers", "Top losers · 24h"))):
            with col:
                st.markdown(f"**{title}**")
                ui.mini_table(["Pair", "Last", "24h", "Volume"],
                              [r.table_row() for r in ranked[key]])
        st.caption("Ranked from each venue's own rolling-24h ticker statistics. A symbol whose "
                   "statistics this venue does not publish is left out of a column rather than "
                   "ranked as zero.")
    else:
        ui.mini_table(["Pair", "Last", "Today (UTC)\u00b9", "Volume\u00b9"],
                      dash.watchlist_rows(rows))
        st.caption(f"\u00b9 {spec.label} publishes a last price only in its public ticker, so the "
                   f"change and volume here come from that venue's own DAILY candle - that is the "
                   f"move since 00:00 UTC, not a rolling 24 hours, which is why this panel shows a "
                   f"watch list instead of a 'top movers' ranking. Delta Exchange and Kraken publish "
                   f"real 24h statistics.")

    note = dash.unavailable_note(rows)
    if note:
        st.caption(f"⚠ {note}")

    st.markdown("#### Selected market")
    status = dash.fetch_ticker_status(get_feed(), feed_cfg.exchange_key, feed_cfg.symbol)
    left, right = st.columns([1.15, 1])
    with left:
        ui.mini_table(["Statistic", "Value"],
                      [[(label, "lbl"), (value, tone)] for label, value, tone in status.stats_rows()])
    with right:
        st.markdown(dash.funding_note(status))
        if spec.thin_data:
            st.caption(
                "⚠ This venue is thin: many low-timeframe buckets contain no trades at all. The "
                "feed measures the zero-volume share, shows it with the data, and refuses to "
                "generate a signal from an empty candle."
            )
        if cs is not None and not cs.is_empty:
            st.caption(
                f"Data quality of the {cs.timeframe} series in use: "
                f"{cs.zero_volume_share * 100:.1f}% zero-volume, "
                f"{cs.flat_candle_share * 100:.1f}% flat (O=H=L=C), {cs.missing_count} gap(s)."
            )


def tab_chart(cs: Optional[CandleSet], strategy: StrategyParams, eng: PaperTradingEngine) -> None:
    if cs is None or cs.is_empty:
        st.info("No candle data available yet. Check the connection diagnostics.")
        return

    zoom_label = render_chart_toolbar(cs)
    pa_params = build_price_action_params()
    sig_df = generate_signals(cs.df, strategy, price_action=pa_params,
                              timeframe=st.session_state.timeframe)
    open_trades = get_db().get_open_trades(eng.feed_key)
    open_trade = open_trades[0] if open_trades else None

    # "View" zoom: how many candles the CHART draws.  The indicators, signals
    # and every risk decision above were computed on the full series; this only
    # slices what is plotted, and the read-out says so.
    view_limit = int(ZOOM_CHOICES.get(zoom_label, 0) or 0)
    view_df = sig_df if (not view_limit or view_limit >= len(sig_df)) else \
        sig_df.tail(view_limit).reset_index(drop=True)

    # Reuse the identical figure object when nothing relevant changed, so the
    # chart is not needlessly rebuilt on every rerun.
    # Rebuild the figure only when something that actually affects it changed.
    signature = (
        eng.feed_key, cs.candle_count, cs.last_closed_ts,
        repr(strategy.to_dict()),
        bool(st.session_state.show_swings), bool(st.session_state.show_signals),
        repr(pa_params) if pa_params.enabled else "pa-off",
        st.session_state.timeframe,
        bool(st.session_state.pa_show_zones), bool(st.session_state.pa_show_events),
        open_trade["trade_id"] if open_trade else None,
        (open_trade or {}).get("entry_price"), (open_trade or {}).get("stop_loss"),
        (open_trade or {}).get("take_profit"),
        tuple((sig_df["signal"] != 0).tolist()[-50:]),
        zoom_label,
        len(view_df),
        bool(st.session_state.live_candle),
    )
    cache = st.session_state.setdefault("fig_cache", {})
    if cache.get("sig") != signature:
        cache["sig"] = signature
        # The cached figure holds CLOSED candles only.  The live candle is drawn
        # on a copy of it by the fragment below, so polling never rebuilds the
        # whole chart (~20 ms per poll instead of ~220 ms).
        built = charts.build_chart(
            view_df,
            symbol=st.session_state.symbol,
            timeframe=st.session_state.timeframe,
            exchange_label=EXCHANGES[st.session_state.exchange_key].label,
            show_swings=bool(st.session_state.show_swings),
            show_signals=bool(st.session_state.show_signals),
            open_trade=open_trade,
            forming=None,
            zones=(_chart_zones(sig_df)),
            pa_events=(_pa_result(sig_df).candle_events if _pa_result(sig_df) else None),
            show_zones=bool(st.session_state.pa_show_zones),
            show_pa_events=bool(st.session_state.pa_show_events),
        )
        # Last-price line + tag ("where price is" at a glance, like the price
        # label in every charting terminal).  Drawn on the cached closed-candle
        # figure; the live poll adds a second, brighter tag for the forming
        # candle on its own copy, so the chart never needs rebuilding.
        charts.overlay_price_tag(
            built, float(sig_df["close"].iloc[-1]), color=charts.ACCENT,
            dim=bool(st.session_state.live_candle),
        )
        cache["fig"] = built

    live_on = bool(st.session_state.live_candle)
    interval = int(st.session_state.live_refresh_s) if live_on else None

    chart_col, rail_col = st.columns([3.15, 1.0], gap="medium")
    with chart_col:
        render_ohlc_readout(
            sig_df,
            tag=(f"last closed candle · {st.session_state.timeframe} · "
                 f"view {len(view_df)} of {len(sig_df)} candles"),
        )
        # `run_every` must be known when the fragment is created, so it is
        # applied dynamically rather than via a decorator.  None = no auto-refresh.
        fragment = st.fragment(run_every=interval)
        fragment(_render_live_chart)(cs, cache, live_on, open_trade)
    with rail_col:
        render_position_rail(eng, cs, open_trade)

    if open_trade:
        st.markdown("#### Open position detail")
        mark = float(cs.forming["close"]) if (cs.forming and cs.forming.get("close")) else float(cs.df["close"].iloc[-1])
        from core.simulation import mark_to_market
        upnl = mark_to_market(open_trade["direction"], float(open_trade["quantity"]),
                              float(open_trade["entry_price"]), mark)
        cols = st.columns(6)
        cols[0].markdown(ui.card("Direction", open_trade["direction"].upper(), open_trade["trade_id"],
                                 "pos" if open_trade["direction"] == "long" else "neg"))
        cols[1].markdown(ui.card("Entry", f"{open_trade['entry_price']:,.2f}",
                                 f"qty {open_trade['quantity']:.6f}", "neu"))
        cols[2].markdown(ui.card("Stop-loss", f"{open_trade['stop_loss']:,.2f}",
                                 f"{open_trade['stop_mode'] or 'n/a'} stop", "neg"))
        cols[3].markdown(ui.card("Take-profit", f"{open_trade['take_profit']:,.2f}",
                                 f"{open_trade['fill_model'] or 'n/a'} fill", "pos"))
        cols[4].markdown(ui.card("Unrealized P/L", ui.money(upnl, signed=True),
                                 f"mark {mark:,.2f}", ui.tone_for(upnl)))
        cols[5].markdown(ui.card("Risk at stop", f"{(open_trade['initial_risk'] or 0):,.2f}",
                                 "fees included", "neu"))
        if st.button("Flatten position now (safe stop)",
                     help="Closes the open simulated position at the latest price and records the reason as safe_stop_flatten."):
            ids = eng.flatten_open_positions(mark)
            st.success(f"Flattened {len(ids)} position(s).")
            st.rerun()
    else:
        st.caption("No open simulated position.")


# --------------------------------------------------------------------------- #
# Indicators tab (v1.3.0) - CHART ONLY
# --------------------------------------------------------------------------- #

def _indicator_param_widgets(key: str, spec: Any) -> Dict[str, float]:
    """
    Draw one indicator's parameter widgets and return its current values.

    Values are seeded from the persisted `ind_params` block on first render and
    clamped to the bounds the catalogue declares, so a stale saved value can
    never make Streamlit raise for being out of range.
    """
    stored = (st.session_state.get("ind_params") or {}).get(key, {})
    values: Dict[str, float] = {}
    for param in spec.params:
        wkey = f"indp_{key}_{param.name}"
        if wkey not in st.session_state:
            seed = stored.get(param.name, param.default)
            if param.integer:
                seed = int(max(int(param.minimum),
                               min(int(param.maximum), int(float(seed)))))
            else:
                seed = float(max(float(param.minimum),
                                 min(float(param.maximum), float(seed))))
            st.session_state[wkey] = seed
        if param.integer:
            val = st.number_input(param.label, int(param.minimum), int(param.maximum),
                                  step=int(param.step), key=wkey)
        else:
            val = st.number_input(param.label, float(param.minimum), float(param.maximum),
                                  step=float(param.step), format="%.3f", key=wkey)
        values[param.name] = int(val) if param.integer else float(val)
    return values


def _persist_indicator_params(params: Dict[str, Dict[str, float]]) -> None:
    """
    Persist the indicator PERIODS.

    The tick boxes themselves are `st.multiselect` widgets, so Streamlit owns
    their session-state keys and refuses to let anything assign to them after
    the widget exists (that raises StreamlitAPIException).  They therefore save
    themselves through their own `on_change` callback - see `tab_indicators`.
    `ind_params` is NOT widget-bound, so it is written directly here.
    """
    if (st.session_state.get("ind_params") or {}) != params:
        st.session_state.ind_params = params
        save_settings()


def tab_indicators(cs: Optional[CandleSet]) -> None:
    st.markdown("#### Indicators")
    st.caption(
        "**Chart only.** Tick what you want to see and set its periods. Nothing on this tab "
        "changes a signal, a trade, the paper ledger or a backtest - the trading engine reads "
        "none of it."
    )
    if cs is None or cs.is_empty:
        st.info("No candle data available yet. Check the connection diagnostics.")
        return

    df = cs.df
    params_state: Dict[str, Dict[str, float]] = {}

    controls, chart_col = st.columns([1.0, 2.8], gap="medium")

    with controls:
        overlays = st.multiselect(
            "Overlays on the price chart", icat.OVERLAYS, key="ind_overlays",
            format_func=lambda k: icat.get(k).label, on_change=save_settings,
            help="Drawn on top of the candles, sharing the price axis.",
        )
        panes = st.multiselect(
            f"Panes below price (max {icat.MAX_PANES} drawn)", icat.PANES, key="ind_panes",
            format_func=lambda k: icat.get(k).label, on_change=save_settings,
            help=("Each pane gets its own subplot and its own scale. You can tick more than "
                  f"{icat.MAX_PANES}; the extras are listed as 'not drawn' rather than "
                  "silently dropped."),
        )
        selected = list(overlays) + list(panes)
        if selected:
            st.markdown("**Periods**")
            for key in selected:
                spec = icat.get(key)
                if not spec.params:
                    continue
                with st.expander(spec.label, expanded=False):
                    params_state[key] = _indicator_param_widgets(key, spec)
                    if spec.description:
                        st.caption(spec.description)
                    if spec.note:
                        st.caption(spec.note)
        for key in selected:
            params_state.setdefault(key, {})
        _persist_indicator_params(params_state)

    drawn_panes = list(panes)[:icat.MAX_PANES]
    dropped = list(panes)[icat.MAX_PANES:]

    with chart_col:
        if not overlays and not panes:
            st.plotly_chart(ichart.empty_indicator_figure(height=520), width="stretch",
                            key="ind_chart_empty", config={"displaylogo": False})
            return

        computed = icat.compute_many(list(overlays) + drawn_panes, df, params_state)
        overlay_lines = [ln for key in overlays for ln in computed.get(key, [])]
        pane_lines = {key: computed.get(key, []) for key in drawn_panes}
        pane_meta = {key: {"label": icat.get(key).label, "y_title": icat.get(key).y_title,
                           "warmup": icat.get(key).warmup} for key in drawn_panes}

        warm = icat.warmup_for(list(overlays) + list(panes))
        live_on = bool(st.session_state.get("live_candle"))
        fig = ichart.build_indicator_figure(
            df, overlay_lines=overlay_lines, pane_lines=pane_lines, pane_meta=pane_meta,
            symbol=st.session_state.symbol, timeframe=st.session_state.timeframe,
            exchange_label=EXCHANGES[st.session_state.exchange_key].label,
            height=520 + 130 * len(drawn_panes),
            forming=(cs.forming if live_on else None),
        )
        st.plotly_chart(fig, width="stretch", key="ind_chart",
                        config={"displaylogo": False, "scrollZoom": True})

        for line in ichart.selection_summary(overlays, drawn_panes, dropped):
            st.caption(line)
        st.caption(
            f"Drawn on {len(df)} closed candles. An indicator needing more history than "
            f"that ({warm} candles for the longest one selected) is simply blank at the "
            f"left - never guessed and never zero-filled. Timestamps are UTC."
        )
        if live_on and cs.forming:
            st.caption(
                "The translucent candle at the right edge is still forming and is drawn for "
                "context only: no indicator on this chart is computed from it."
            )
        missing = [icat.get(k).label for k in list(overlays) + drawn_panes
                   if not computed.get(k)]
        if missing:
            st.warning("Not enough usable candles to compute: " + ", ".join(missing)
                       + ". Load more history and try again.", icon="⚠️")

        notes = [f"**{icat.get(k).label}** - {icat.get(k).note}"
                 for k in list(overlays) + drawn_panes
                 if icat.get(k).note and computed.get(k)]
        if notes:
            with st.expander("Reading notes for what is on the chart", expanded=False):
                for note in notes:
                    st.markdown(note)


def tab_signals(cs: Optional[CandleSet], strategy: StrategyParams) -> None:
    if cs is None or cs.is_empty:
        st.info("No candle data available.")
        return
    pa_params = build_price_action_params()
    sig = generate_signals(cs.df, strategy, price_action=pa_params,
                           timeframe=st.session_state.timeframe)
    exp = explain_latest(sig, strategy)

    st.markdown("#### Latest closed candle - signal explanation")
    if exp is None:
        st.info("Nothing to explain yet.")
        return

    tone = {"valid_long": "ok", "valid_short": "bad", "awaiting_confirmation": "warn",
            "invalid": "warn", "insufficient_history": "warn"}.get(exp.status, "info")
    label = {"valid_long": "VALID LONG SETUP", "valid_short": "VALID SHORT SETUP",
             "awaiting_confirmation": "AWAITING CONFIRMATION", "invalid": "NO SETUP",
             "insufficient_history": "INSUFFICIENT HISTORY"}.get(exp.status, exp.status.upper())
    pill_items = [(label, tone), (f"Candle {_ts_str(exp.timestamp)}", "info")]
    if not getattr(exp, "tradeable", True):
        pill_items.append(("No trades in this candle", "bad"))
    ui.pills(pill_items)
    st.markdown(f"**{exp.summary}**")

    if exp.checks:
        rows = []
        for chk in exp.checks:
            rows.append({
                "Condition": chk.name,
                "Result": "PASS" if chk.passed else "FAIL",
                "Detail": chk.detail,
            })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    else:
        st.warning(
            f"Warm-up: at least {strategy.min_history()} closed candles are needed before "
            f"the indicators are defined. Currently {cs.candle_count} are available. "
            f"No signal is guessed during warm-up.")

    # The EMA 20/50 read-outs were removed from this panel on request.  The
    # strategy still evaluates the EMA crossover as condition 1 of the entry, so
    # nothing about signal generation changed - only what is displayed.
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Close", f"{exp.close:,.2f}" if exp.close else "n/a")
    c2.metric("RSI(14)", f"{exp.rsi:.2f}" if exp.rsi is not None else "n/a",
              f"long ≥ {strategy.rsi_long_min:g} · short ≤ {strategy.rsi_short_max:g}",
              delta_color="off")
    c3.metric("Prior swing high", f"{exp.prior_swing_high:,.2f}" if exp.prior_swing_high else "n/a",
              "breakout level for longs", delta_color="off")
    c4.metric("Prior swing low", f"{exp.prior_swing_low:,.2f}" if exp.prior_swing_low else "n/a",
              "breakout level for shorts", delta_color="off")
    if not (exp.prior_swing_high and exp.prior_swing_low):
        st.caption("Swing levels not yet defined - not enough closed candles.")
    if exp.bos_bull_wick or exp.bos_bear_wick:
        st.info(
            "A wick broke a swing level but the candle did not CLOSE through it. "
            "With 'require candle-close breakout' enabled this is not treated as a "
            "confirmed break of structure.", icon="ℹ️")

    with st.expander("Recent candle-by-candle conditions"):
        cols = ["timestamp", "close", "rsi", "prior_swing_high", "prior_swing_low",
                "bos_bull", "bos_bear", "long_score", "short_score", "cond_tradeable", "signal"]
        view = sig[[c for c in cols if c in sig.columns]].tail(40).iloc[::-1].copy()
        # The price-action result rides along in df.attrs; Streamlit tries to
        # JSON-serialise attrs for every displayed frame and warns when it
        # cannot.  This is a display-only copy, so drop them here.
        view.attrs.clear()
        if "timestamp" in view.columns:
            view["timestamp"] = view["timestamp"].map(_ts_str)
        view = view.rename(columns={"signal": "signal (1=buy,-1=sell)",
                                    "cond_tradeable": "had trades"})
        st.dataframe(view, width="stretch", hide_index=True)

    render_price_action_panel(sig, exp, pa_params)

    st.info(
        "This strategy makes no claim about future profitability or any particular win "
        "rate. It is a transparent rule set: a signal appears only when all four "
        "conditions pass on a fully closed candle.", icon="ℹ️")


def render_price_action_panel(sig: pd.DataFrame, exp, pa_params: PriceActionParams) -> None:
    """
    Phase-2 (J): show the price-action confluence for the latest CLOSED candle.

    Everything here is explanatory.  It never opens, blocks or modifies a trade,
    and absent evidence is listed as absent rather than assumed.
    """
    st.divider()
    st.markdown("#### Price-action confluence")
    conf = getattr(exp, "confluence", None) if exp is not None else None
    if conf is None:
        st.caption(
            "The price-action add-on is switched off. Enable it under "
            "Sidebar → *Price action add-on* to see CHoCH/MSS, liquidity sweeps, "
            "fair value gaps, order blocks, supply/demand, candlestick patterns, "
            "breakout-retest state, premium/discount and their confluence score."
        )
        return

    res = _pa_result(sig)
    bias = int(conf.get("bias", 0))
    score = float(conf.get("score", 0.0))
    opposing = float(conf.get("opposing", 0.0))
    net = float(conf.get("net", 0.0))
    meets = bool(conf.get("meets_threshold", False))
    label = {1: "BULLISH EVIDENCE", -1: "BEARISH EVIDENCE", 0: "NO DIRECTIONAL EVIDENCE"}[bias]
    tone = {1: "ok", -1: "bad", 0: "warn"}[bias]
    ui.pills([
        (label, tone),
        (f"score {score:.1f} vs {opposing:.1f} opposing", "info"),
        (f"net {net:+.1f}", "info"),
        (f"threshold {pa_params.min_confluence_score:.1f} · "
         + ("MET" if meets else "not met"), "ok" if meets else "warn"),
    ])
    st.caption(conf.get("summary", ""))

    agreeing = conf.get("agreeing", [])
    disagreeing = conf.get("disagreeing", [])
    missing = conf.get("missing", [])
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"**Agreeing ({len(agreeing)})**")
    c1.markdown("\n".join(f"- ✅ **{f['name']}** (w {f['weight']:.1f}) — {f['detail']}"
                           for f in agreeing) or "- _none_")
    c2.markdown(f"**Disagreeing ({len(disagreeing)})**")
    c2.markdown("\n".join(f"- ❌ **{f['name']}** (w {f['weight']:.1f}) — {f['detail']}"
                           for f in disagreeing) or "- _none_")
    c3.markdown(f"**No evidence from ({len(missing)})**")
    c3.markdown("\n".join(f"- ⬜ {m.replace('_', ' ')}" for m in missing) or "- _all concepts reported_")

    suppressed = conf.get("suppressed", [])
    if suppressed:
        st.caption("Counted once (weaker duplicates suppressed): " + "; ".join(suppressed))

    mtf = _mtf_result(sig)
    if mtf is not None:
        st.markdown(f"**Higher timeframe ({mtf.htf_timeframe or 'n/a'})**")
        if mtf.available:
            for line in describe_mtf(mtf):
                st.markdown(f"- {line}")
            with st.expander(f"Higher-timeframe zones ({len(mtf.zones())})"):
                hz = mtf.zones()
                if not hz:
                    st.info("No live higher-timeframe zone right now.")
                else:
                    rows = [{
                        "Zone": z.kind.replace("htf_", "").replace("_", " "),
                        "Side": {1: "support / bullish", -1: "resistance / bearish",
                                 0: "neutral"}.get(z.direction, "?"),
                        "Top": f"{z.top:,.2f}", "Bottom": f"{z.bottom:,.2f}",
                        "Formed": _ts_str(z.created_ts), "Status": z.status,
                        "Retests": z.retests, "Strength": f"{z.strength:.2f}",
                    } for z in sorted(hz, key=lambda z: z.created_ts, reverse=True)]
                    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
                    st.caption(
                        f"Built from CLOSED {mtf.htf_timeframe} candles aggregated from your "
                        f"{mtf.ltf_timeframe} series. Drawn with dashed outlines on the chart."
                    )
        else:
            for w in mtf.warnings:
                st.warning(w, icon="ℹ️")
            st.caption(
                "No higher-timeframe read is shown rather than a partly warmed-up or "
                "guessed one. Fetch more candles or choose a lower higher-timeframe."
            )

    if res is not None:
        with st.expander(f"Active zones ({len(res.active_zones())})"):
            zones = res.active_zones()
            if not zones:
                st.info("No live price-action zone right now.")
            else:
                rows = [{
                    "Zone": z.kind.replace("_", " "),
                    "Side": {1: "support / bullish", -1: "resistance / bearish",
                             0: "neutral"}.get(z.direction, "?"),
                    "Top": f"{z.top:,.2f}",
                    "Bottom": f"{z.bottom:,.2f}",
                    "Formed": _ts_str(z.created_ts),
                    "Confirmed": _ts_str(z.confirmed_ts) if z.confirmed_ts else "n/a",
                    "Status": z.status,
                    "Retests": z.retests,
                    "Strength": f"{z.strength:.2f}",
                } for z in sorted(zones, key=lambda z: z.created_ts, reverse=True)]
                st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
                st.caption(
                    "Strength is the qualifying move measured in ATRs. "
                    "A zone is retired when it is filled, mitigated, invalidated or expired."
                )

        with st.expander("Recent price-action events"):
            events = list(res.candle_events)[-40:][::-1]
            if not events:
                st.info("No structural or liquidity events recorded yet.")
            else:
                rows = [{"Time (UTC)": _ts_str(e.get("ts")),
                         "Event": str(e.get("event", "")).replace("_", " "),
                         "Level": (f"{float(e['level']):,.2f}"
                                   if e.get("level") is not None else
                                   (f"{float(e['top']):,.2f}" if e.get("top") is not None else "n/a")),
                         "Note": e.get("note", "")} for e in events]
                st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

        with st.expander("How this confluence score is built"):
            st.markdown(
                f"""
- Every detected concept becomes a **named factor** with a direction and a weight
  (structure {pa_params.w_structure:g}, liquidity {pa_params.w_liquidity:g},
  breakout {pa_params.w_breakout:g}, FVG {pa_params.w_fvg:g},
  order block {pa_params.w_order_block:g}, supply/demand {pa_params.w_supply_demand:g},
  pattern {pa_params.w_pattern:g}, premium/discount {pa_params.w_premium_discount:g}).
- **Nothing is counted twice.** Within a group only the strongest present factor
  survives: MSS beats an early CHoCH, a sweep-and-reclaim beats a bare sweep, a
  retest beats a formation, a held retest beats the breakout itself. Candlestick
  shapes contribute at most one bullish and one bearish factor per candle.
- Your existing four Phase-1 conditions are **listed but never scored** (weight 0),
  so this panel can never quietly change what the bot trades.
- Score = weight agreeing with the bias; opposing = weight against it; the read is
  only called *actionable* at or above the {pa_params.min_confluence_score:g} threshold.
- Missing concepts are shown as missing. Absence of evidence is never treated as
  agreement, and no factor claims to know what institutions did.
"""
            )
    st.info(
        "Price-action concepts are interpretations of past candles. They describe what "
        "already happened; they do not predict the next move and they are not trading advice.",
        icon="ℹ️")


def tab_trades(eng: PaperTradingEngine) -> None:
    db = get_db()
    st.markdown("#### Strategy performance (paper ledger)")
    closed_all = db.get_trades(status="closed", limit=5000)
    window = dash.monthly_stats(closed_all, eng.starting_balance, days=MONTHLY_WINDOW_DAYS)
    ring_col, table_col = st.columns([1.0, 1.5], gap="medium")
    with ring_col:
        st.plotly_chart(
            charts.build_return_ring(
                window.total_return_pct, has_trades=window.has_trades,
                centre=f"Total Return · {MONTHLY_WINDOW_DAYS}d",
            ),
            width="stretch", key="return_ring",
            config={"displaylogo": False, "displayModeBar": False},
        )
    with table_col:
        st.markdown(f"**Last {MONTHLY_WINDOW_DAYS} days** — realised results only")
        ui.mini_table(["Metric", "Value"],
                      [[(k, "lbl"), (v, "neu")] for k, v in window.table_rows()])
        st.caption(window.caption())

    st.markdown("#### Simulated trade ledger")
    counts = db.count_trades()
    ui.pills([
        (f"{counts['open']} open", "info"),
        (f"{counts['closed']} closed", "ok"),
        (f"{counts['total']} total", "info"),
    ])

    trades = db.get_trades(limit=1000)
    if not trades:
        st.info(
            "No simulated trades yet. Start the bot and wait for a fully closed candle "
            "that satisfies all four conditions.")
    else:
        df = pd.DataFrame(trades)
        for c in ["signal_ts", "entry_ts", "exit_ts"]:
            if c in df.columns:
                df[c] = df[c].map(_ts_str)
        cols = ["trade_id", "direction", "signal_ts", "entry_ts", "entry_price", "quantity",
                "stop_loss", "take_profit", "exit_ts", "exit_price", "exit_reason",
                "gross_pnl", "net_pnl", "realized_r", "status"]
        df = df[[c for c in cols if c in df.columns]]
        st.dataframe(df, width="stretch", hide_index=True, height=360)
        st.download_button(
            "⬇ Download trade history (CSV)", df.to_csv(index=False).encode(),
            file_name="paper_trades.csv", mime="text/csv")

    st.markdown("#### Bot event log")
    events = db.get_events(limit=200)
    if not events:
        st.caption("No events recorded yet.")
    else:
        edf = pd.DataFrame(events)
        edf["ts"] = edf["ts"].map(_ts_str)
        edf = edf[["ts", "level", "event_type", "message"]]
        st.dataframe(edf, width="stretch", hide_index=True, height=300)
    if st.button("Prune old events (keep newest 2,000)"):
        removed = db.prune_events(2000)
        st.success(f"Removed {removed} old event row(s).")


def tab_backtest(cs: Optional[CandleSet], strategy: StrategyParams, risk: RiskParams) -> None:
    st.markdown("#### Historical backtest")
    st.caption(
        "Runs the SAME rules, position sizing, fees, slippage and exit logic as the live "
        "paper bot, over historical closed candles. It reads no live trades and cannot "
        "modify or delete your saved paper-trade history.")

    if cs is None or cs.is_empty:
        st.warning("Market data is required before a backtest can run.")
        return

    c1, c2, c3 = st.columns(3)
    bt_balance = c1.number_input("Starting balance", 100.0, 1_000_000.0,
                                 value=float(st.session_state.starting_balance),
                                 step=1000.0, key="bt_bal")
    bt_fill = c2.selectbox("Fill model", FILL_MODES,
                           index=FILL_MODES.index(st.session_state.fill_model), key="bt_fill")
    bt_risk_pct = c3.slider("Risk per trade %", 0.05, 10.0,
                            value=float(st.session_state.risk_per_trade_pct),
                            step=0.05, key="bt_risk")

    c4, c5, c6 = st.columns(3)
    bt_rr = c4.slider("Reward : risk", 0.5, 10.0,
                      value=float(st.session_state.reward_to_risk), step=0.5, key="bt_rr")
    bt_fee = c5.number_input("Fee % per side", 0.0, 1.0,
                             value=float(st.session_state.fee_pct_per_side),
                             step=0.01, key="bt_fee")
    bt_slip = c6.number_input("Slippage bps", 0.0, 100.0,
                              value=float(st.session_state.slippage_bps),
                              step=1.0, key="bt_slip")

    # ---- v1.3.0: the walk-forward test.  Placed BEFORE the backtest result so
    # it stays reachable even when no backtest has been run yet - it is the more
    # important of the two, because it is the one that can say "this does not
    # hold up".
    render_walk_forward(cs, strategy)

    st.divider()
    st.markdown("#### Historical backtest (single pass over the loaded candles)")
    run = st.button("▶ Run backtest on the loaded candles", type="primary")
    if not run:
        prev = st.session_state.get("bt_result")
        if prev is None:
            st.info("Configure the parameters above and run the backtest.")
            return
        res = prev
    else:
        bt_risk = RiskParams(**{**risk.to_dict(), "starting_balance": bt_balance,
                                "risk_per_trade_pct": bt_risk_pct, "reward_to_risk": bt_rr,
                                "fee_pct_per_side": bt_fee, "slippage_bps": bt_slip})
        cfg = BacktestConfig(
            exchange=st.session_state.exchange_key, symbol=st.session_state.symbol,
            timeframe=st.session_state.timeframe, candles=len(cs.df),
            starting_balance=bt_balance, fill_model=bt_fill,
            strategy=strategy, risk=bt_risk.validate(),
        )
        try:
            with st.spinner("Running backtest…"):
                res = run_backtest(cs.df, cfg)
            st.session_state["bt_result"] = res
        except Exception as exc:
            log.exception("Backtest failure")
            st.error(f"Backtest failed: {exc}")
            return

    if not res.ok:
        st.error(res.message)
        for w in res.warnings:
            st.caption(f"⚠ {w}")
        return

    m = res.metrics
    st.success(f"{res.message} Completed in {res.duration_ms:.0f} ms.")
    for w in res.warnings:
        st.warning(w, icon="⚠️")

    if res.n_trades == 0:
        st.info(
            "No completed trades. With so few (or no) qualifying setups, no performance "
            "statistic can be reported honestly. Try a longer history, a different "
            "timeframe, or looser RSI thresholds - but do not read edge into an empty sample.")
        return

    ui.card_row([
        ui.card("Trades", f"{res.n_trades}", f"{res.signals_seen} signal(s) seen", "neu"),
        ui.card("Win rate", "n/a" if m.win_rate is None else f"{m.win_rate * 100:.1f}%",
                f"{m.wins}W / {m.losses}L", "neu"),
        ui.card("Net P/L", ui.money(m.net_pnl, signed=True),
                f"gross +{m.gross_profit:,.2f} / {m.gross_loss:,.2f}", ui.tone_for(m.net_pnl)),
        ui.card("Profit factor", "n/a" if m.profit_factor is None else f"{m.profit_factor:.2f}",
                "undefined without losses", "neu"),
    ])
    ui.card_row([
        ui.card("Max drawdown", f"-{m.max_drawdown:,.2f}", f"{m.max_drawdown_pct:.2f}%",
                "neg" if m.max_drawdown else "neu"),
        ui.card("Avg win / loss",
                (f"{m.avg_win:,.0f} / {m.avg_loss:,.0f}" if m.avg_win is not None and m.avg_loss is not None else "n/a"),
                f"largest loss {m.largest_loss:,.2f}" if m.largest_loss is not None else "", "neu"),
        ui.card("Avg realized R", "n/a" if m.avg_r is None else f"{m.avg_r:+.2f}",
                "1R = the amount risked", ui.tone_for(m.avg_r)),
        ui.card("Max consecutive losses", f"{m.max_consecutive_losses}",
                f"max wins {m.max_consecutive_wins}", "neg" if m.max_consecutive_losses > 3 else "neu"),
    ])

    eq = equity_to_frame(res)
    if not eq.empty:
        from core.metrics import equity_curve
        realised = equity_curve([t["net_pnl"] for t in res.trades], res.config.starting_balance)
        st.plotly_chart(charts.build_equity_curve(eq, starting_balance=res.config.starting_balance,
                                                  title="Backtest equity (mark-to-market)"),
                        width="stretch", key="bt_equity")
        st.plotly_chart(charts.build_drawdown_chart(realised, starting_balance=res.config.starting_balance),
                        width="stretch", key="bt_dd")
    st.plotly_chart(charts.build_exit_reason_chart(m.exit_reason_breakdown),
                    width="stretch", key="bt_exits")

    st.markdown("#### Trade-by-trade")
    rows = result_to_rows(res)
    if not rows.empty:
        view = rows.copy()
        for c in ["signal_ts", "entry_ts", "exit_ts"]:
            view[c] = view[c].map(_ts_str)
        st.dataframe(view, width="stretch", hide_index=True, height=340)
        st.download_button("⬇ Download backtest trades (CSV)", view.to_csv(index=False).encode(),
                           file_name="backtest_trades.csv", mime="text/csv", key="bt_csv")

    if st.button("💾 Save this backtest run"):
        run_id = get_db().save_backtest(
            {"exchange": res.config.exchange, "symbol": res.config.symbol,
             "timeframe": res.config.timeframe, "candles": res.candles_available,
             "config": res.config.to_dict(), "metrics": m.to_dict(),
             "n_trades": res.n_trades, "net_pnl": res.net_pnl, "warnings": res.warnings},
            res.trades)
        st.success(f"Saved as {run_id}. Live paper trades were not touched.")

    with st.expander("Backtest limitations & assumptions"):
        for lim in res.limitations:
            st.markdown(f"- {lim}")
        st.markdown(
            f"- Warm-up: {res.warmup_skipped} candle(s) had undefined indicators and were skipped.\n"
            f"- Daily loss lockouts triggered: {res.daily_lockouts}.\n"
            f"- Entries blocked by risk rules: {res.blocked_entries}.\n"
            f"- Fill model: `{res.config.fill_model}` — signal and fill timestamps are stored separately."
        )
        st.info(
            "This is a HISTORICAL SIMULATION on past data, not a forecast and not actual "
            "market performance.", icon="ℹ️")

    # Only shown when the walk-forward section above has no result to break down:
    # one session table, sourced from the strongest evidence available.
    wf_prev = st.session_state.get("wf_result")
    if not (wf_prev is not None and wf_prev.ok):
        render_session_breakdown(res.trades, source_label="the backtest run above")


# --------------------------------------------------------------------------- #
# Walk-forward test + session breakdown (v1.3.0)
# --------------------------------------------------------------------------- #

def _filters_off_copy(strategy: StrategyParams) -> StrategyParams:
    """The same settings with every entry filter disabled, for an A/B comparison."""
    return replace(strategy, trend_filter="off", atr_filter_min_pct=0.0,
                   atr_filter_max_pct=0.0, volume_filter_ratio=0.0,
                   session_filter=[], allow_weekends=True)


def _variant_list(strategy: StrategyParams) -> List[tuple]:
    """
    What to walk forward: the settings as configured, plus - only when a filter is
    actually on - the same settings with filters off.  Comparing the two is the
    entire point: it answers 'did the filter help on candles it did not pick?'.
    """
    if strategy.filters_active():
        return [("filters OFF (baseline)", _filters_off_copy(strategy)),
                ("filters ON (your settings)", strategy)]
    return [("your settings", strategy)]


def render_walk_forward(cs: CandleSet, strategy: StrategyParams) -> None:
    st.divider()
    st.markdown("#### Walk-forward test · in-sample vs out-of-sample")
    st.caption(
        "History is cut into consecutive blocks. In each block the first portion is the "
        "IN-SAMPLE window - the part you are allowed to look at - and the rest is the "
        "OUT-OF-SAMPLE window, tested with settings that are never re-fitted from it. "
        "Out-of-sample results are the only ones that say anything about the future, and "
        "they are still only evidence from the past, not a promise."
    )

    c1, c2, c3 = st.columns([1, 1, 1])
    splits = c1.number_input("Folds", 2, 8, step=1, key="wf_splits",
                             help="How many in-sample / out-of-sample blocks to cut.")
    train_ratio = c2.slider("In-sample share of each block", 0.5, 0.9, step=0.05,
                            key="wf_train_ratio")
    anchored = c3.checkbox("Anchored (expanding) training window", key="wf_anchored",
                           help="Off: the in-sample window rolls forward with each block. "
                                "On: every block trains from the very first candle.")
    c4, c5, c6 = st.columns([1, 1, 2])
    wf_balance = c4.number_input("Starting balance", 100.0, 1_000_000.0, step=1000.0,
                                 key="wf_bal", value=float(st.session_state.starting_balance))
    wf_risk_pct = c5.slider("Risk per trade %", 0.05, 10.0, step=0.05, key="wf_risk",
                            value=float(st.session_state.risk_per_trade_pct))
    with c6:
        st.caption(
            "The engine, fees, slippage and exit logic are identical to the live paper bot; "
            "only the history window changes. Each window starts flat, with indicators "
            "warmed on earlier candles but no position carried in."
        )

    run = st.button("🧪 Run walk-forward test", type="primary", key="wf_run")
    if not run:
        prev = st.session_state.get("wf_result")
        if prev is None:
            st.info("Set the folds above and run the test. Nothing is computed until you ask.")
            return
        report = prev
    else:
        try:
            risk = RiskParams(**{**build_params()[1].to_dict(),
                                 "starting_balance": wf_balance,
                                 "risk_per_trade_pct": wf_risk_pct}).validate()
        except InvalidConfigError as exc:
            st.error(f"Invalid risk settings: {exc}")
            return
        cfg = BacktestConfig(
            exchange=st.session_state.exchange_key, symbol=st.session_state.symbol,
            timeframe=st.session_state.timeframe, candles=len(cs.df),
            starting_balance=wf_balance, fill_model=st.session_state.fill_model,
            strategy=strategy, risk=risk,
        )
        try:
            with st.spinner("Walking forward through history…"):
                report = wf.walk_forward(
                    cs.df, cfg, _variant_list(strategy),
                    wf.WalkForwardConfig(n_splits=int(splits),
                                         train_ratio=float(train_ratio),
                                         anchored=bool(anchored)),
                )
            st.session_state["wf_result"] = report
        except Exception as exc:                                   # noqa: BLE001
            log.exception("Walk-forward failure")
            st.error(f"Walk-forward failed: {exc}")
            return

    if not report.ok:
        st.warning(report.message)
        for w in report.warnings:
            st.caption(f"⚠ {w}")
        return

    st.success(report.message)
    for line in report.describe():
        st.markdown(f"- {line}")
    for w in report.warnings:
        st.warning(w, icon="⚠️")

    st.dataframe(pd.DataFrame(report.table_rows()), width="stretch", hide_index=True)
    st.caption(
        "Win rate is shown next to expectancy (average R per trade), net P&L and the number "
        "of profitable windows on purpose: a high win rate on its own is meaningless - a "
        "small target against a wide stop wins often and still loses money. Net P&L and "
        "drawdown across folds are sums of separately-sized windows, so the drawdown figure "
        "is approximate. '-' means undefined, never zero."
    )

    with st.expander("Fold-by-fold detail", expanded=False):
        for v in report.variants:
            if not v.folds:
                continue
            st.markdown(f"**{v.name}**")
            rows = []
            for f in v.folds:
                ins, oos = f.in_sample, f.out_of_sample
                rows.append({
                    "Fold": f.label(),
                    "In-sample candles": str(f.window.n_train),
                    "Out-of-sample candles": str(f.window.n_test),
                    "In: trades": str(ins.n_trades),
                    "In: expectancy R": wf._r(wf._exp_r(ins.metrics)),
                    "Out: trades": str(oos.n_trades),
                    "Out: win %": wf._pct(oos.metrics.win_rate),
                    "Out: expectancy R": wf._r(wf._exp_r(oos.metrics)),
                    "Out: net P&L": wf._money(oos.metrics.net_pnl) if oos.n_trades else "-",
                })
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    st.caption(
        "Deliberately absent from this page: an optimiser that searches settings until the "
        "past looks good. That is how a backtest is fitted to noise. Settings that only win "
        "in-sample are flagged as such instead of being celebrated."
    )

    oos_trades = [t for v in report.variants if v.out_of_sample
                  for t in v.out_of_sample.trades]
    if oos_trades:
        render_session_breakdown(
            oos_trades,
            source_label=("every out-of-sample window of the walk-forward test - the only "
                          "trades on this page that were never used to choose a setting"))


def render_session_breakdown(trades: List[Dict[str, Any]], *, source_label: str) -> None:
    """Honest per-session / per-weekday performance for a list of closed trades."""
    st.divider()
    st.markdown("#### Performance by session and weekday")
    st.caption(
        f"Closed trades from: {source_label}. Session hours are fixed UTC windows, so the "
        f"columns stay comparable across months; the IST clock time is shown alongside."
    )

    data = sessions_mod.full_breakdown(trades)
    overall = data["overall"]
    ui.card_row([
        ui.card("Trades", f"{overall['trades']}", "that could be grouped by time", "neu"),
        ui.card("Win rate", "n/a" if overall["win_rate"] is None
                else f"{overall['win_rate'] * 100:.1f}%", "wins / all closed", "neu"),
        ui.card("Net P&L", f"{overall['net_pnl']:+,.2f}", "fees included",
                ui.tone_for(overall["net_pnl"])),
        ui.card("Avg R", "n/a" if overall["avg_r"] is None else f"{overall['avg_r']:+.2f}R",
                "per closed trade", ui.tone_for(overall["avg_r"])),
    ])
    if data["warning"]:
        st.warning(data["warning"], icon="⚠️")

    left, right = st.columns(2)
    with left:
        st.markdown("**By session (UTC windows)**")
        st.dataframe(pd.DataFrame(sessions_mod.breakdown_rows(
            {k: dict(v) for k, v in data["sessions"]})), width="stretch", hide_index=True)
        st.caption(data["counting_rule"])
        st.caption(sessions_mod.describe_sessions(list(st.session_state.filter_sessions)))
    with right:
        st.markdown("**By weekday (UTC)**")
        st.dataframe(pd.DataFrame(sessions_mod.breakdown_rows(
            {k: dict(v) for k, v in data["weekdays"]})), width="stretch", hide_index=True)
        st.caption("Weekend hours are thinner on every crypto venue; the rows above show "
                   "whether that mattered for these trades.")

    with st.expander("Sessions, with the IST clock times", expanded=False):
        rows = [{"Session": sessions_mod.session_label(k), "UTC window": sessions_mod.session_hours(k)}
                for k in sessions_mod.SESSION_KEYS]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.caption("A trade is counted in every session its entry-to-exit span touched, so "
                   "these are overlapping windows rather than a partition of the day.")


def tab_diagnostics(cs: Optional[CandleSet], feed: FeedConfig) -> None:
    st.markdown("#### Market-data connection diagnostics")
    fd = get_feed()
    db = get_db()

    if st.button("Run full connection check", type="primary"):
        with st.spinner("Probing the exchange…"):
            status = fd.check_connection(feed.exchange_key, feed.symbol, feed.timeframe, force=True)
        st.session_state["diag_status"] = status

    status = st.session_state.get("diag_status")
    if status is not None:
        ok = status.ok
        ui.pills([
            ("CONNECTED" if ok else "FAILED", "ok" if ok else "bad"),
            (f"stage: {status.stage}", "info"),
            (f"latency {status.latency_ms:.0f} ms", "info"),
        ])
        rows = {
            "Exchange": status.exchange_key,
            "Result": "OK" if ok else "FAILED",
            "Stage reached": status.stage,
            "Category": status.category or "-",
            "HTTP status": status.http_status if status.http_status else "-",
            "Latency (ms)": round(status.latency_ms, 1),
            "Markets loaded": status.markets_loaded,
            "Symbol": status.symbol or "-",
            "Symbol supported": status.symbol_supported,
            "Timeframe": status.timeframe or "-",
            "Timeframe native": status.timeframe_native,
            "Sample price": f"{status.sample_price:,.2f}" if status.sample_price else "-",
            "Server time (UTC)": _ts_str(status.server_time_ms) if status.server_time_ms else "-",
            "Clock skew (ms)": round(status.clock_skew_ms) if status.clock_skew_ms is not None else "-",
            "Market kind": (get_exchange_spec(status.exchange_key).market_kind
                            if status.exchange_key in EXCHANGES else "-"),
            "Message": status.message,
        }
        st.table(pd.DataFrame({"Item": list(rows.keys()), "Value": [str(v) for v in rows.values()]}))
        skew = status.clock_skew_ms
        if skew is not None and abs(skew) > 30_000:
            st.caption(
                f"ℹ Clock skew is {skew / 1000:+.0f}s. Delta's time endpoint is served through a "
                f"CDN that caches the response, so the 'server time' it returns can be tens of "
                f"seconds old. The app uses that value as its clock, which means a candle is "
                f"treated as closed slightly LATE - the safe direction. It can never make a "
                f"still-forming candle look closed, so no look-ahead is introduced."
            )
        if not ok:
            st.error(f"**Recovery steps**\n{chr(10).join(f'{i}. {s}' for i, s in enumerate(status.recovery, 1))}")
            if status.detail:
                st.code(status.detail[:800], language="text")

    st.markdown("#### Data quality of the current series")
    if cs is None or cs.is_empty:
        st.caption("No candles fetched yet.")
    else:
        spec_q = EXCHANGES[st.session_state.exchange_key]
        q_rows = {
            "Exchange / market": f"{spec_q.label} · {spec_q.market_kind}",
            "Closed candles": f"{cs.candle_count:,}",
            "Zero-volume candles": f"{cs.zero_volume_count:,} ({cs.zero_volume_share * 100:.1f}%)",
            "Flat candles (O=H=L=C)": f"{cs.flat_candle_count:,} ({cs.flat_candle_share * 100:.1f}%)",
            "Missing candles (gaps)": f"{cs.missing_count:,}",
            "Thin venue flag": str(spec_q.thin_data),
            "Signals from empty candles": "impossible - blocked by cond_tradeable",
        }
        st.table(pd.DataFrame({"Item": list(q_rows.keys()),
                               "Value": [str(v) for v in q_rows.values()]}))
        if cs.data_quality_note:
            st.caption(cs.data_quality_note)

    st.markdown("#### Public ticker snapshot (powers the stats strip)")
    snap = dash.fetch_ticker_status(get_feed(), feed.exchange_key, feed.symbol,
                                    derive_from_candles=False)
    ui.pills([
        ("TICKER OK" if snap.ok else "TICKER FAILED", "ok" if snap.ok else "bad"),
        (f"last {snap.last:,.2f}" if snap.last else "no last price", "info"),
        (f"change source: {snap.change_kind or 'none'}", "info"),
        ("funding published" if snap.funding_rate is not None else "no funding rate", "info"),
    ])
    st.caption(snap.note or
               "The 24h strip reads this same snapshot. It is cached for "
               f"{dash.TICKER_TTL_S:.0f}s, is public, and never feeds a signal.")

    st.markdown("#### Circuit breaker")
    for key in EXCHANGES:
        stt = fd.circuit_state(key)
        ui.pills([
            (key, "info"),
            ("OPEN" if stt["open"] else "closed", "bad" if stt["open"] else "ok"),
            (f"failures: {int(stt['failures'])}", "warn" if stt["failures"] else "info"),
            (f"{stt['seconds_remaining']:.0f}s remaining", "warn") if stt["open"] else ("ready", "ok"),
        ])
    if st.button("Reset circuit breaker (allow immediate retry)"):
        clear_caches()
        st.session_state.pop("diag_status", None)
        st.success("Caches and circuit breakers cleared.")
        st.rerun()

    st.markdown("#### Environment")
    import ccxt
    import plotly
    import numpy
    import sys as _sys
    _env_items = [
        ("App version", APP_VERSION),
        ("Python", _sys.version.split()[0]),
        ("Streamlit", st.__version__),
        ("ccxt", ccxt.__version__),
        ("pandas", pd.__version__),
        ("numpy", numpy.__version__),
        ("plotly", plotly.__version__),
        ("Feed key", feed.feed_key),
        ("Last fetch error", (st.session_state.get("last_fetch_error").category
                              if st.session_state.get("last_fetch_error") else "none")),
        ("Data cache entries", len(st.session_state.get("data_cache", {}))),
        ("Database path", db.path),
    ]
    # Every value is stringified: a mixed-type column breaks Arrow serialisation.
    st.table(pd.DataFrame({"Item": [i for i, _ in _env_items],
                           "Value": [str(v) for _, v in _env_items]}))

    st.markdown("#### Database")
    stats = db.stats()
    st.table(pd.DataFrame({
        "Item": ["Path", "Size (bytes)"] + [f"rows: {k}" for k in stats["rows"]],
        "Value": [stats["path"], f"{stats['bytes']:,}"] + [f"{v:,}" for v in stats["rows"].values()],
    }))
    st.caption(
        "The processed-candle count is what makes refreshes idempotent: each closed candle "
        "is recorded once per feed key, so it can never open two trades."
    )
    st.warning(
        "**Streamlit Community Cloud storage is temporary.** The filesystem (and therefore this "
        "SQLite file) may be wiped when the app restarts, redeploys or sleeps. Run locally for "
        "durable history, or wire an external managed database.", icon="⚠️")

    if st.session_state.pop("_show_diagnostics", False):
        st.info("Diagnostics refreshed below.")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    load_settings()
    render_sidebar()

    try:
        strategy, risk, feed = build_params()
    except InvalidConfigError as exc:
        st.error(f"**Invalid settings:** {exc}")
        st.stop()

    db = get_db()
    if db.get_setting("starting_balance") is None:
        db.set_setting("starting_balance", float(st.session_state.starting_balance))

    eng = _engine()
    running = eng.is_running()

    with st.spinner("Fetching market data…"):
        cs, feed_err = fetch_candles(feed)
    age = cache_age_seconds(feed)

    if st.session_state.get("_migration_note"):
        st.info(st.session_state.pop("_migration_note"), icon="🔁")

    render_header(cs, feed_err, running, age)
    render_market_strip(feed, cs)

    # Let the engine act on any newly closed candle.  It is fully idempotent:
    # repeated refreshes over the same data cannot create duplicate trades.
    process_result = None
    if cs is not None:
        try:
            process_result = eng.process(cs)
        except Exception as exc:
            log.exception("Paper engine failure")
            st.error(f"The paper engine hit an error and skipped this cycle: {exc}")

    render_metrics(eng, cs)

    if process_result is not None and process_result.actions:
        interesting = [a for a in process_result.actions
                       if a.kind not in ("no_new_candles", "no_signal", "bot_stopped")]
        if interesting:
            with st.expander(f"Latest engine actions ({len(interesting)})", expanded=bool(
                    any(a.kind in ("entry", "exit") for a in interesting))):
                for a in interesting[-12:]:
                    icon = {"entry": "🟢", "exit": "🔴", "pending_created": "🕓",
                            "blocked_risk": "⛔", "blocked_levels": "⛔",
                            "duplicate_signal": "🔁", "daily_loss_lockout": "🔒",
                            "gate_blocked": "⛔", "bootstrap": "🚀"}.get(a.kind, "•")
                    st.markdown(f"{icon} **{a.kind}** — {a.message}")

    render_quick_actions(feed)

    tabs = st.tabs(["Chart", "Indicators", "Signals", "Market", "Trades", "Backtest",
                    "Diagnostics"])
    with tabs[0]:
        tab_chart(cs, strategy, eng)
    with tabs[1]:
        tab_indicators(cs)
    with tabs[2]:
        tab_signals(cs, strategy)
    with tabs[3]:
        tab_market(feed, cs)
    with tabs[4]:
        tab_trades(eng)
    with tabs[5]:
        tab_backtest(cs, strategy, risk)
    with tabs[6]:
        tab_diagnostics(cs, feed)

    ui.footer(
        COPYRIGHT_LINE,
        app_line=f"{APP_NAME} v{APP_VERSION} · {SIMULATION_BANNER}",
        fine_print=(
            "Educational simulator, not financial advice. Public market data only; "
            "no exchange API keys are used and no real order is ever sent."
        ),
    )


# Streamlit's script runner executes this file with __name__ == "__main__" on
# every rerun.  Guarding the entry point keeps the module importable by the test
# suite (and by `streamlit.testing.v1.AppTest`) without side effects.
if __name__ == "__main__":
    main()
