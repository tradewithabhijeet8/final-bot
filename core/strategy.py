"""
core/strategy.py
================
Break-of-Structure + RSI + EMA trend filter.

The strategy is deliberately boring and fully transparent.  Every candle gets a
list of named checks with a pass/fail result, so the UI can explain *why* a
signal did or did not fire.  Nothing is inferred, smoothed over or guessed.

Long (all must pass)
    1. bullish BOS      : close > prior swing high
    2. RSI confirmation : rsi  >= rsi_long_min   (default 55)
    3. momentum         : close > EMA fast       (default 20)
    4. trend            : EMA fast > EMA slow    (default 50)

Short (all must pass)
    1. bearish BOS      : close < prior swing low
    2. RSI confirmation : rsi  <= rsi_short_max  (default 45)
    3. momentum         : close < EMA fast
    4. trend            : EMA fast < EMA slow

Look-ahead safety
-----------------
`generate_signals` only uses indicator columns that are themselves strictly
causal (see core/indicators.py).  It should be called with CLOSED candles only;
it raises if it is handed a frame that still contains the forming candle.

This strategy does not claim any particular win rate.  It is an educational
rule set whose behaviour can be inspected candle by candle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from . import sessions as sessions_mod
from .config import PriceActionParams, StrategyParams
from .confluence import build_confluence
from .indicators import ema, enrich
from .multi_timeframe import analyse_mtf
from .price_action import analyse as analyse_price_action

# Signal codes
NO_SIGNAL = 0
LONG = 1
SHORT = -1

# Statuses
STATUS_VALID_LONG = "valid_long"
STATUS_VALID_SHORT = "valid_short"
STATUS_INVALID = "invalid"
STATUS_AWAITING = "awaiting_confirmation"
STATUS_INSUFFICIENT = "insufficient_history"

CONDITION_COLUMNS = [
    "cond_bos", "cond_rsi", "cond_momentum", "cond_trend",
]

#: Data-quality gate, kept SEPARATE from the four strategy conditions so the
#: four-condition table is byte-for-byte unchanged.  A bucket in which nothing
#: traded carries no information, so no signal may come from it.
TRADEABLE_COLUMN = "cond_tradeable"
#: True when every ENABLED entry filter passed.  All-True when no filter is on.
FILTERS_COLUMN = "filters_ok"


@dataclass
class Check:
    """One named condition and whether it passed."""

    name: str
    passed: bool
    detail: str
    required: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "passed": self.passed,
                "detail": self.detail, "required": self.required}


@dataclass
class SignalExplanation:
    """Human-readable breakdown for a single candle."""

    timestamp: Optional[float]
    direction: int
    status: str
    close: Optional[float] = None
    rsi: Optional[float] = None
    ema_fast: Optional[float] = None
    ema_slow: Optional[float] = None
    prior_swing_high: Optional[float] = None
    prior_swing_low: Optional[float] = None
    bos_bull: bool = False
    bos_bear: bool = False
    bos_bull_wick: bool = False
    bos_bear_wick: bool = False
    checks: List[Check] = field(default_factory=list)
    summary: str = ""
    #: False when the candle had zero volume (nothing traded in that bucket).
    #: Such a candle can never produce a signal, whatever the conditions say.
    tradeable: bool = True
    # Phase-2 additions (optional; empty unless price-action analysis was run).
    # They are kept SEPARATE from `checks` so the original four-condition
    # explanation is byte-for-byte unchanged when price action is disabled.
    confluence: Optional[Dict[str, Any]] = None
    confluence_checks: List[Check] = field(default_factory=list)
    # v1.3.0: the optional entry filters, kept in their OWN list for the same
    # reason as the confluence checks - the four-condition table must stay
    # byte-for-byte identical when no filter is enabled.
    filter_checks: List[Check] = field(default_factory=list)
    #: True when at least one ENABLED filter blocked this candle.
    filtered_out: bool = False

    @property
    def passed_names(self) -> List[str]:
        return [c.name for c in self.checks if c.passed]

    @property
    def failed_names(self) -> List[str]:
        return [c.name for c in self.checks if not c.passed]

    def to_dict(self) -> Dict[str, Any]:
        d = self.__dict__.copy()
        d["checks"] = [c.to_dict() for c in self.checks]
        d["confluence_checks"] = [c.to_dict() for c in self.confluence_checks]
        d["filter_checks"] = [c.to_dict() for c in self.filter_checks]
        d["passed"] = self.passed_names
        d["failed"] = self.failed_names
        return d


# --------------------------------------------------------------------------- #
# Vectorised signal generation
# --------------------------------------------------------------------------- #

def generate_signals(df: pd.DataFrame, params: StrategyParams,
                     price_action: Optional[PriceActionParams] = None,
                     timeframe: Optional[str] = None) -> pd.DataFrame:
    """
    Attach indicator, condition and signal columns to a copy of `df`.

    Input must be CLOSED candles sorted ascending by timestamp.

    Phase-2: `price_action` is OPTIONAL and additive.  Passing None (the
    default) or a disabled PriceActionParams leaves the returned frame exactly
    as before - same columns, same signals, same order.  When enabled, the
    `pa_*` columns from core/price_action.py are appended and the full
    PriceActionResult is attached as ``out.attrs["pa_result"]`` so the
    explanation layer and the UI can consume it without recomputing.
    """
    params.validate()

    if "is_forming" in df.columns and bool(df["is_forming"].any()):
        raise ValueError(
            "generate_signals received a frame containing the still-forming "
            "candle. Signals must be computed from fully closed candles only."
        )

    out = enrich(
        df,
        swing_lookback=params.swing_lookback,
        swing_shift=params.swing_shift,
        rsi_period=params.rsi_period,
        ema_fast=params.ema_fast,
        ema_slow=params.ema_slow,
        atr_period=params.atr_period,
        require_close_breakout=params.require_close_breakout,
    )

    close = out["close"].to_numpy(dtype="float64")
    rsi_vals = out["rsi"].to_numpy(dtype="float64")
    ef = out["ema_fast"].to_numpy(dtype="float64")
    es = out["ema_slow"].to_numpy(dtype="float64")

    # Warm-up mask: any NaN means "we do not know yet", never "condition failed".
    ready = ~(np.isnan(rsi_vals) | np.isnan(ef) | np.isnan(es)
              | np.isnan(out["prior_swing_high"].to_numpy(dtype="float64"))
              | np.isnan(out["prior_swing_low"].to_numpy(dtype="float64")))
    out["ready"] = ready

    with np.errstate(invalid="ignore"):
        # ---- long conditions
        long_bos = out["bos_bull"].to_numpy(dtype=bool)
        long_rsi = rsi_vals >= params.rsi_long_min
        long_mom = close > ef
        long_trend = ef > es

        # ---- short conditions
        short_bos = out["bos_bear"].to_numpy(dtype=bool)
        short_rsi = rsi_vals <= params.rsi_short_max
        short_mom = close < ef
        short_trend = ef < es

    # ---- data-quality gate ------------------------------------------------ #
    # Thin venues (e.g. Delta's BTC perpetual) emit placeholder buckets when
    # nothing traded: zero volume, often open == high == low == close.  RSI,
    # ATR and break-of-structure computed over those rows are meaningless, so
    # a signal is never allowed to come from one.  On liquid venues this mask
    # is all-True and every signal is exactly what it was before.
    if "volume" in out.columns:
        vol = pd.to_numeric(out["volume"], errors="coerce").fillna(0.0).to_numpy(dtype="float64")
    else:
        vol = np.ones(len(out), dtype="float64")
    tradeable = vol > 0.0
    out[TRADEABLE_COLUMN] = tradeable

    # ---- optional entry filters (v1.3.0) ---------------------------------- #
    # Every filter is OFF by default.  With all of them off, `filters_ok` is
    # all-True and the four-condition signals below are byte-for-byte what they
    # were before this feature existed (pinned by test).  A filter can only
    # REMOVE a trade: none of them can create one, and each one is evaluated on
    # closed candles only - the volume average deliberately excludes the bar it
    # is judging.
    filters_ok = np.ones(len(out), dtype=bool)
    filter_report: Dict[str, Any] = {}

    if params.trend_filter != "off":
        trend_ema = ema(close, params.trend_ema_period)
        out["filter_trend_ema"] = trend_ema
        with np.errstate(invalid="ignore"):
            if params.trend_filter == "above":
                trend_pass = close > trend_ema
            else:
                trend_pass = close < trend_ema
        # During warm-up the EMA is NaN, so `close > NaN` is False - which would
        # silently block every early candle.  Warm-up belongs to `ready`, so the
        # filter is treated as passing while the average does not exist yet.
        trend_pass = np.where(np.isnan(trend_ema), True, trend_pass)
        filters_ok &= trend_pass
        filter_report["trend"] = trend_pass

    if params.atr_filter_min_pct > 0.0 or params.atr_filter_max_pct > 0.0:
        atr_vals = out["atr"].to_numpy(dtype="float64")
        with np.errstate(invalid="ignore", divide="ignore"):
            atr_pct = np.where(close > 0, 100.0 * atr_vals / close, np.nan)
        out["filter_atr_pct"] = atr_pct
        atr_pass = np.ones(len(out), dtype=bool)
        unknown = np.isnan(atr_pct)          # warm-up / no trades: not a failure
        if params.atr_filter_min_pct > 0.0:
            atr_pass &= (atr_pct >= params.atr_filter_min_pct) | unknown
        if params.atr_filter_max_pct > 0.0:
            atr_pass &= (atr_pct <= params.atr_filter_max_pct) | unknown
        filters_ok &= atr_pass
        filter_report["atr"] = atr_pass

    if params.volume_filter_ratio > 0.0:
        lookback = int(params.volume_filter_lookback)
        # mean of the PREVIOUS `lookback` bars - the current bar is excluded so a
        # single huge candle cannot justify itself.
        prior_mean = pd.Series(vol).shift(1).rolling(lookback, min_periods=lookback).mean()
        prior_mean = prior_mean.to_numpy(dtype="float64")
        out["filter_volume_avg"] = prior_mean
        with np.errstate(invalid="ignore"):
            volume_pass = vol >= params.volume_filter_ratio * prior_mean
        volume_pass = np.where(np.isnan(prior_mean), True, volume_pass)
        filters_ok &= volume_pass
        filter_report["volume"] = volume_pass

    if params.session_filter or not params.allow_weekends:
        allowed = list(params.session_filter or [])
        stamps = out["timestamp"].to_numpy(dtype="float64")
        session_pass = np.array(
            [sessions_mod.in_allowed_sessions(ts, allowed) for ts in stamps], dtype=bool)
        if not params.allow_weekends:
            weekend_pass = np.array([not sessions_mod.is_weekend_utc(ts) for ts in stamps],
                                    dtype=bool)
            session_pass &= weekend_pass
            filter_report["weekend"] = weekend_pass
        filters_ok &= session_pass
        filter_report["session"] = session_pass

    out[FILTERS_COLUMN] = filters_ok

    out["cond_bos_long"] = long_bos
    out["cond_rsi_long"] = long_rsi
    out["cond_momentum_long"] = long_mom
    out["cond_trend_long"] = long_trend
    out["cond_bos_short"] = short_bos
    out["cond_rsi_short"] = short_rsi
    out["cond_momentum_short"] = short_mom
    out["cond_trend_short"] = short_trend

    out["long_score"] = (long_bos.astype(int) + long_rsi.astype(int)
                         + long_mom.astype(int) + long_trend.astype(int))
    out["short_score"] = (short_bos.astype(int) + short_rsi.astype(int)
                          + short_mom.astype(int) + short_trend.astype(int))

    long_ok = ready & tradeable & filters_ok & long_bos & long_rsi & long_mom & long_trend
    short_ok = ready & tradeable & filters_ok & short_bos & short_rsi & short_mom & short_trend

    # A candle can theoretically satisfy both sides in a degenerate series;
    # resolve deterministically and flag it rather than silently picking one.
    both = long_ok & short_ok
    if both.any():
        out.loc[both, "conflicting_signal"] = True
        long_ok = long_ok & ~both
        short_ok = short_ok & ~both
    else:
        out["conflicting_signal"] = False

    signal = np.where(long_ok, LONG, np.where(short_ok, SHORT, NO_SIGNAL))
    out["signal"] = signal
    out["long_ok"] = long_ok
    out["short_ok"] = short_ok

    # ---- optional price-action layer (Phase 2) ----------------------------- #
    pa_on = price_action is not None and getattr(price_action, "enabled", False)
    if pa_on:
        pa_params = price_action.validate()
        pa_result = analyse_price_action(out, pa_params)
        pa_cols = [c for c in pa_result.df.columns
                   if c.startswith("pa_") and c not in out.columns]
        # One concat, not one insertion per column: assigning ~65 columns
        # individually fragments the frame and triggers a PerformanceWarning.
        block = pa_result.df[pa_cols].copy()
        block.index = out.index
        out = pd.concat([out, block], axis=1)

        # ---- optional higher-timeframe context --------------------------- #
        # Requires the price-action add-on: HTF context is part of it.  Runs the
        # same analyse() pass on aggregated CLOSED higher-timeframe candles and
        # aligns them so an HTF candle can only inform an LTF candle that closed
        # at or after it (see core/multi_timeframe.py).
        mtf = None
        if pa_params.mtf_enabled:
            mtf = analyse_mtf(out, timeframe, pa_params)
            block = mtf.df.copy()
            block.index = out.index
            out = pd.concat([out, block], axis=1)

        # Attach attrs LAST: pd.concat returns a new object and does not carry
        # attrs over, so setting them before the concat would silently lose them.
        out.attrs["pa_result"] = pa_result
        out.attrs["pa_params"] = pa_params
        if mtf is not None:
            out.attrs["mtf_result"] = mtf
    return out


# --------------------------------------------------------------------------- #
# Per-candle explanation
# --------------------------------------------------------------------------- #

def _fmt(value: Optional[float], digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    return f"{value:,.{digits}f}"


def _filter_checks(row: pd.Series, params: StrategyParams) -> List[Check]:
    """
    Explain every ENABLED entry filter in plain language.

    Returns [] when no filter is on, so the Signals panel shows exactly what it
    showed before this feature existed.  A filter that passes during warm-up
    (its input is still NaN) says so rather than claiming a pass it did not earn.
    """
    checks: List[Check] = []
    if not params.filters_active():
        return checks

    if params.trend_filter != "off":
        trend_ema = _opt(row.get("filter_trend_ema"))
        side = "above" if params.trend_filter == "above" else "below"
        if trend_ema is None:
            passed, detail = True, ("Not enough history for the trend EMA yet, so this "
                                    "filter is not blocking anything.")
        else:
            close = _opt(row.get("close"))
            passed = (close > trend_ema) if params.trend_filter == "above" else (close < trend_ema)
            detail = (f"Close {_fmt(close)} must be {side} EMA{params.trend_ema_period} "
                      f"{_fmt(trend_ema)}.")
        checks.append(Check(f"Trend filter: price {side} EMA{params.trend_ema_period}",
                            bool(passed), detail))

    if params.atr_filter_min_pct > 0.0 or params.atr_filter_max_pct > 0.0:
        atr_pct = _opt(row.get("filter_atr_pct"))
        bounds = []
        if params.atr_filter_min_pct > 0.0:
            bounds.append(f">= {params.atr_filter_min_pct:g}%")
        if params.atr_filter_max_pct > 0.0:
            bounds.append(f"<= {params.atr_filter_max_pct:g}%")
        want = " and ".join(bounds)
        if atr_pct is None:
            passed, detail = True, "ATR is still warming up, so this filter is not blocking anything."
        else:
            passed = True
            if params.atr_filter_min_pct > 0.0:
                passed &= atr_pct >= params.atr_filter_min_pct
            if params.atr_filter_max_pct > 0.0:
                passed &= atr_pct <= params.atr_filter_max_pct
            detail = f"ATR is {atr_pct:.3f}% of price; filter wants {want}."
        checks.append(Check("Volatility filter (ATR % of price)", bool(passed), detail))

    if params.volume_filter_ratio > 0.0:
        avg = _opt(row.get("filter_volume_avg"))
        vol = _opt(row.get("volume"))
        want = params.volume_filter_ratio
        if avg is None:
            passed, detail = True, "Not enough history for the volume average yet."
        else:
            passed = vol is not None and vol >= want * avg
            detail = (f"Volume {_fmt(vol)} vs {want:g}x the previous "
                      f"{params.volume_filter_lookback}-candle average {_fmt(avg)}.")
        checks.append(Check(f"Volume filter ({want:g}x average)", bool(passed), detail))

    if params.session_filter:
        from . import sessions as _sessions
        ts = _opt(row.get("timestamp"), 0)
        allowed = list(params.session_filter)
        names = ", ".join(_sessions.session_label(a) for a in allowed)
        if ts is None:
            passed, detail = False, "No timestamp on this candle."
        else:
            passed = _sessions.in_allowed_sessions(ts, allowed)
            detail = (f"{_sessions.ist_dt(ts):%H:%M IST} on {_sessions.weekday_utc(ts)}; "
                      f"allowed window(s): {names}.")
        checks.append(Check("Time-of-day filter", bool(passed), detail))

    if not params.allow_weekends:
        ts = _opt(row.get("timestamp"), 0)
        from . import sessions as _sessions
        if ts is None:
            passed, detail = False, "No timestamp on this candle."
        else:
            weekend = _sessions.is_weekend_utc(ts)
            passed = not weekend
            detail = (f"{_sessions.weekday_utc(ts)} - weekends are excluded "
                      f"by the weekend filter." if weekend else
                      f"{_sessions.weekday_utc(ts)} - a weekday, so this filter passes.")
        checks.append(Check("Weekend filter", bool(passed), detail))

    return checks


def explain_row(row: pd.Series, params: StrategyParams,
                pa_params: Optional[PriceActionParams] = None) -> SignalExplanation:
    """
    Build a plain-language explanation for one candle of an already-enriched
    frame (i.e. the output of `generate_signals`).

    Phase-2: when `pa_params` is supplied AND the row carries `pa_*` columns, a
    confluence report (agreeing / disagreeing / missing evidence) is attached.
    Without it, the explanation is identical to the Phase-1 behaviour.
    """
    ready = bool(row.get("ready", False))
    close = _opt(row.get("close"))
    rsi_val = _opt(row.get("rsi"))
    ef = _opt(row.get("ema_fast"))
    es = _opt(row.get("ema_slow"))
    psh = _opt(row.get("prior_swing_high"))
    psl = _opt(row.get("prior_swing_low"))

    bos_bull = bool(row.get("bos_bull", False))
    bos_bear = bool(row.get("bos_bear", False))
    bos_bull_wick = bool(row.get("bos_bull_wick", False))
    bos_bear_wick = bool(row.get("bos_bear_wick", False))

    confluence, conf_checks = _confluence_for_row(row, pa_params)
    filter_checks = _filter_checks(row, params)
    filtered_out = any(not c.passed for c in filter_checks)

    if not ready:
        return SignalExplanation(
            timestamp=_opt(row.get("timestamp"), 0),
            direction=NO_SIGNAL,
            status=STATUS_INSUFFICIENT,
            close=close, rsi=rsi_val, ema_fast=ef, ema_slow=es,
            prior_swing_high=psh, prior_swing_low=psl,
            bos_bull=bos_bull, bos_bear=bos_bear,
            bos_bull_wick=bos_bull_wick, bos_bear_wick=bos_bear_wick,
            checks=[],
            summary=(
                "Not enough history yet for this candle - one or more indicators "
                "are still in warm-up. No signal is produced (never guessed)."
            ),
            confluence=confluence,
            confluence_checks=conf_checks,
            filter_checks=filter_checks,
            filtered_out=filtered_out,
        )

    # Decide which side to narrate.  Prefer the side with the higher score so
    # the explanation is still useful when nothing fired.
    long_score = int(row.get("long_score", 0))
    short_score = int(row.get("short_score", 0))
    side = "long" if long_score >= short_score else "short"

    if side == "long":
        checks = [
            Check(
                "Bullish BOS",
                bos_bull,
                f"Close {_fmt(close)} vs prior swing high {_fmt(psh)} -> "
                + ("closed ABOVE (structure broke up)." if bos_bull else
                   ("wick poked above but did NOT close above." if bos_bull_wick
                    else "did not break the level.")),
            ),
            Check(
                f"RSI >= {params.rsi_long_min:g}",
                bool(row.get("cond_rsi_long", False)),
                f"RSI(14) = {_fmt(rsi_val)}; needs >= {params.rsi_long_min:g}.",
            ),
            Check(
                f"Close > EMA{params.ema_fast}",
                bool(row.get("cond_momentum_long", False)),
                f"Close {_fmt(close)} vs EMA{params.ema_fast} {_fmt(ef)}.",
            ),
            Check(
                f"EMA{params.ema_fast} > EMA{params.ema_slow}",
                bool(row.get("cond_trend_long", False)),
                f"EMA{params.ema_fast} {_fmt(ef)} vs EMA{params.ema_slow} {_fmt(es)} -> "
                + ("uptrend." if bool(row.get("cond_trend_long", False)) else "not in an uptrend."),
            ),
        ]
    else:
        checks = [
            Check(
                "Bearish BOS",
                bos_bear,
                f"Close {_fmt(close)} vs prior swing low {_fmt(psl)} -> "
                + ("closed BELOW (structure broke down)." if bos_bear else
                   ("wick poked below but did NOT close below." if bos_bear_wick
                    else "did not break the level.")),
            ),
            Check(
                f"RSI <= {params.rsi_short_max:g}",
                bool(row.get("cond_rsi_short", False)),
                f"RSI(14) = {_fmt(rsi_val)}; needs <= {params.rsi_short_max:g}.",
            ),
            Check(
                f"Close < EMA{params.ema_fast}",
                bool(row.get("cond_momentum_short", False)),
                f"Close {_fmt(close)} vs EMA{params.ema_fast} {_fmt(ef)}.",
            ),
            Check(
                f"EMA{params.ema_fast} < EMA{params.ema_slow}",
                bool(row.get("cond_trend_short", False)),
                f"EMA{params.ema_fast} {_fmt(ef)} vs EMA{params.ema_slow} {_fmt(es)} -> "
                + ("downtrend." if bool(row.get("cond_trend_short", False)) else "not in a downtrend."),
            ),
        ]

    passed = all(c.passed for c in checks)
    n_passed = sum(1 for c in checks if c.passed)

    if passed:
        direction = LONG if side == "long" else SHORT
        status = STATUS_VALID_LONG if side == "long" else STATUS_VALID_SHORT
        summary = (
            f"VALID {side.upper()} SETUP - all 4 conditions passed. "
            + ("Bullish structure break confirmed with RSI and EMA trend agreement."
               if side == "long" else
               "Bearish structure break confirmed with RSI and EMA trend agreement.")
        )
    elif n_passed == 0:
        direction, status = NO_SIGNAL, STATUS_INVALID
        summary = "No setup - none of the 4 conditions passed on this candle."
    else:
        direction, status = NO_SIGNAL, STATUS_AWAITING
        failed = [c.name for c in checks if not c.passed]
        summary = (
            f"Partial {side} setup ({n_passed}/4 passed). "
            f"Still failing: {', '.join(failed)}. No trade is opened."
        )

    tradeable = bool(row.get(TRADEABLE_COLUMN, True))
    if not tradeable:
        status = STATUS_INVALID
        direction = NO_SIGNAL
        summary = (
            f"No trades printed in this candle (zero volume), so it carries no information. "
            f"{n_passed}/4 conditions appeared to pass, but a signal is never generated from "
            f"an empty bucket - the price did not move because nothing traded, not because "
            f"the market agreed on a level."
        )

    if bool(row.get("conflicting_signal", False)):
        status = STATUS_INVALID
        direction = NO_SIGNAL
        summary = (
            "Conflicting conditions: long and short checks both passed on this "
            "candle. Treated as invalid rather than guessing a direction."
        )

    return SignalExplanation(
        timestamp=_opt(row.get("timestamp"), 0),
        direction=direction,
        status=status,
        close=close, rsi=rsi_val, ema_fast=ef, ema_slow=es,
        prior_swing_high=psh, prior_swing_low=psl,
        bos_bull=bos_bull, bos_bear=bos_bear,
        bos_bull_wick=bos_bull_wick, bos_bear_wick=bos_bear_wick,
        checks=checks,
        summary=summary,
        tradeable=tradeable,
        confluence=confluence,
        confluence_checks=conf_checks,
        filter_checks=filter_checks,
        filtered_out=filtered_out,
    )


def _confluence_for_row(row: pd.Series,
                        pa_params: Optional[PriceActionParams]):
    """Return (confluence_dict, [Check,...]) or (None, []) when PA is off."""
    if pa_params is None or not getattr(pa_params, "enabled", False):
        return None, []
    if not any(str(c).startswith("pa_") for c in getattr(row, "index", [])):
        return None, []

    # Show the Phase-1 conditions alongside (weight 0: listed, never scored).
    baseline = {str(c): bool(row.get(c)) for c in row.index
                if str(c).startswith("cond_") and str(c).endswith(("_long", "_short"))}
    report = build_confluence(row, pa_params, baseline=baseline)

    checks = [
        Check(f"PA: {f.name}", True, f.detail, required=False)
        for f in report.agreeing
    ] + [
        Check(f"PA against: {f.name}", False, f.detail, required=False)
        for f in report.disagreeing
    ] + [
        Check(f"PA: {f.name}", False, f.detail, required=False)
        for f in report.factors if f.direction == 0 and not f.baseline
    ]
    if report.missing:
        checks.append(Check(
            "PA missing evidence",
            False,
            "No price-action evidence from: " + ", ".join(report.missing)
            + ". Absent evidence is reported, never assumed to be true.",
            required=False,
        ))
    if report.suppressed:
        checks.append(Check(
            "PA de-duplicated",
            True,
            "Counted once, weaker duplicate(s) suppressed: "
            + "; ".join(report.suppressed),
            required=False,
        ))
    return report.to_dict(), checks


def _opt(value: Any, digits: Optional[int] = None) -> Optional[float]:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(f):
        return None
    return f


def explain_latest(df: pd.DataFrame, params: StrategyParams) -> Optional[SignalExplanation]:
    """Explain the most recent CLOSED candle of an enriched frame."""
    if df is None or df.empty:
        return None
    return explain_row(df.iloc[-1], params, pa_params=df.attrs.get("pa_params"))


def signal_counts(df: pd.DataFrame) -> Dict[str, int]:
    """How many long/short signals the frame contains (diagnostics + tests)."""
    if df is None or df.empty or "signal" not in df.columns:
        return {"long": 0, "short": 0, "none": 0}
    s = df["signal"].to_numpy()
    return {
        "long": int((s == LONG).sum()),
        "short": int((s == SHORT).sum()),
        "none": int((s == NO_SIGNAL).sum()),
    }


__all__ = [
    "NO_SIGNAL", "LONG", "SHORT",
    "STATUS_VALID_LONG", "STATUS_VALID_SHORT", "STATUS_INVALID",
    "STATUS_AWAITING", "STATUS_INSUFFICIENT",
    "CONDITION_COLUMNS", "Check", "SignalExplanation",
    "generate_signals", "explain_row", "explain_latest", "signal_counts",
]
