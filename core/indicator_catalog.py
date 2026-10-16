"""
core/indicator_catalog.py
=========================
The declarative registry behind the **Indicators tab** (v1.3.0).

Why a registry instead of hand-written UI code per indicator:

* the tab, the chart layer and the tests all read the SAME definition, so a new
  indicator is one entry here rather than edits in three places;
* parameter validation, defaults, persistence keys and labels can never drift
  apart from what is actually computed;
* and it keeps the promise that these indicators are **display-only**: nothing
  in this module writes to the frame the strategy reads.

Each entry declares where it draws:

``overlay``
    drawn on top of price (same y-axis): moving averages, bands, channels.
``pane``
    drawn in its own subplot below price (oscillators, volume, ADX).

`compute()` returns an ordered list of `Line`s, each with a stable `key`, a
human label, and the array.  The chart layer turns them into Plotly traces; the
tests check the values against hand-computed references.

Colours here are deliberately NOT green/red: those two are reserved for market
direction (see `core/charts.py`).  `directional=True` marks the FEW lines whose
colour legitimately encodes up/down - currently only Supertrend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import indicators as ind

# --------------------------------------------------------------------------- #
# Colours (no green/red: those mean direction and nothing else)
# --------------------------------------------------------------------------- #

BLUE = "#60a5fa"
CYAN = "#22d3ee"
AMBER = "#fbbf24"
ORANGE = "#fb923c"
VIOLET = "#a78bfa"
PURPLE = "#c084fc"
PINK = "#f472b6"
SLATE = "#94a3b8"
TEAL = "#2dd4bf"
GOLD = "#e0b252"
ICE = "#9ec3e0"


@dataclass(frozen=True)
class Line:
    """One drawable output of an indicator."""

    key: str
    label: str
    values: np.ndarray
    colour: str = SLATE
    dash: str = "solid"
    width: float = 1.4
    opacity: float = 0.95
    #: True only when the colour itself carries direction (green up / red down).
    directional: bool = False
    #: Bars to shift the line forward on the x-axis (Ichimoku convention).
    shift_forward: int = 0
    fill: Optional[str] = None          # optional fill group, e.g. "band"
    shape: str = "lines"                # "lines" | "markers"


@dataclass(frozen=True)
class ParamSpec:
    """One configurable number of an indicator."""

    name: str
    label: str
    default: float
    minimum: float
    maximum: float
    step: float = 1.0
    integer: bool = True


@dataclass(frozen=True)
class IndicatorSpec:
    """Everything the UI, the chart and the tests need to know."""

    key: str
    label: str
    kind: str                           # "overlay" | "pane"
    compute: Callable[..., List[Line]]
    params: Tuple[ParamSpec, ...] = ()
    #: Roughly how many bars of history the first value needs, used for the
    #: "not enough candles" caption.
    warmup: int = 20
    description: str = ""
    #: Extra caption shown under the indicator in the tab (honesty notes).
    note: str = ""
    y_title: str = ""

    def defaults(self) -> Dict[str, float]:
        return {p.name: p.default for p in self.params}

    def param_keys(self) -> List[str]:
        return [p.name for p in self.params]


# --------------------------------------------------------------------------- #
# Compute functions - each returns a list of Line
# --------------------------------------------------------------------------- #

def _ma(values, period, fn, colour, label, **kw) -> List[Line]:
    return [Line(key=f"ma{period}", label=label, values=fn(values, period),
                 colour=colour, **kw)]


def _compute_ema(df, period=20, **_):
    return _ma(df["close"], int(period), ind.ema, BLUE, f"EMA {int(period)}")


def _compute_sma(df, period=50, **_):
    return _ma(df["close"], int(period), ind.sma, AMBER, f"SMA {int(period)}")


def _compute_wma(df, period=20, **_):
    return _ma(df["close"], int(period), ind.wma, PURPLE, f"WMA {int(period)}")


def _compute_hma(df, period=16, **_):
    return _ma(df["close"], int(period), ind.hma, CYAN, f"HMA {int(period)}")


def _compute_bollinger(df, period=20, mult=2.0, **_):
    lower, mid, upper = ind.bollinger_bands(df["close"], int(period), float(mult))
    return [
        Line("bb_upper", f"BB upper {int(period)}", upper, colour=SLATE, width=1.2,
             fill="band"),
        Line("bb_mid", f"BB mid {int(period)}", mid, colour=SLATE, width=1.0,
             dash="dot"),
        Line("bb_lower", f"BB lower {int(period)}", lower, colour=SLATE, width=1.2,
             fill="band"),
    ]


def _compute_vwap(df, period=20, **_):
    return [Line("vwap", f"Rolling VWAP {int(period)}",
                 ind.vwap(df["high"], df["low"], df["close"], df["volume"], int(period)),
                 colour=CYAN, width=1.6)]


def _compute_donchian(df, period=20, **_):
    lower, mid, upper = ind.donchian(df["high"], df["low"], int(period))
    return [
        Line("dc_upper", f"Donchian upper {int(period)}", upper, colour=ORANGE,
             width=1.2, fill="band"),
        Line("dc_mid", f"Donchian mid {int(period)}", mid, colour=ORANGE, width=1.0,
             dash="dot"),
        Line("dc_lower", f"Donchian lower {int(period)}", lower, colour=ORANGE,
             width=1.2, fill="band"),
    ]


def _compute_keltner(df, ema_period=20, atr_period=10, mult=2.0, **_):
    lower, mid, upper = ind.keltner(df["high"], df["low"], df["close"],
                                    int(ema_period), int(atr_period), float(mult))
    return [
        Line("kc_upper", f"Keltner upper {int(ema_period)}", upper, colour=VIOLET,
             width=1.2, fill="band"),
        Line("kc_mid", f"Keltner mid {int(ema_period)}", mid, colour=VIOLET, width=1.0,
             dash="dot"),
        Line("kc_lower", f"Keltner lower {int(ema_period)}", lower, colour=VIOLET,
             width=1.2, fill="band"),
    ]


def _compute_supertrend(df, period=10, mult=3.0, **_):
    line, direction = ind.supertrend(df["high"], df["low"], df["close"],
                                     int(period), float(mult))
    return [Line("supertrend", f"Supertrend {int(period)}x{mult:g}", line,
                 colour=AMBER, width=1.8, directional=True)]


def _compute_ichimoku(df, conversion=9, base=26, span_b=52, displacement=26, **_):
    tenkan, kijun, span_a, span_b_line = ind.ichimoku(
        df["high"], df["low"], int(conversion), int(base), int(span_b))
    shift = int(displacement)
    return [
        Line("ich_tenkan", f"Tenkan {int(conversion)}", tenkan, colour=BLUE, width=1.2),
        Line("ich_kijun", f"Kijun {int(base)}", kijun, colour=ORANGE, width=1.2),
        Line("ich_span_a", "Senkou A (shifted)", span_a, colour=CYAN, width=1.0,
             dash="dot", shift_forward=shift, fill="band"),
        Line("ich_span_b", "Senkou B (shifted)", span_b_line, colour=PINK, width=1.0,
             dash="dot", shift_forward=shift, fill="band"),
    ]


def _compute_psar(df, step=0.02, max_step=0.2, **_):
    return [Line("psar", "Parabolic SAR", ind.psar(df["high"], df["low"],
                                                   float(step), float(max_step)),
                 colour=GOLD, width=0.0, shape="markers", opacity=0.85)]


def _compute_rsi(df, period=14, **_):
    return [Line("rsi", f"RSI {int(period)}", ind.rsi(df["close"], int(period)),
                 colour=AMBER, width=1.6)]


def _compute_macd(df, fast=12, slow=26, signal=9, **_):
    macd_line, signal_line, hist = ind.macd(df["close"], int(fast), int(slow), int(signal))
    return [
        Line("macd", f"MACD {int(fast)}/{int(slow)}", macd_line, colour=BLUE, width=1.5),
        Line("macd_signal", f"Signal {int(signal)}", signal_line, colour=ORANGE, width=1.3),
        Line("macd_hist", "Histogram", hist, colour=SLATE, width=0.0, shape="bar"),
    ]


def _compute_stoch(df, k_period=14, d_period=3, smooth_k=3, **_):
    k, d = ind.stochastic(df["high"], df["low"], df["close"],
                          int(k_period), int(d_period), int(smooth_k))
    return [
        Line("stoch_k", f"%K {int(k_period)}", k, colour=CYAN, width=1.5),
        Line("stoch_d", f"%D {int(d_period)}", d, colour=AMBER, width=1.2),
    ]


def _compute_adx(df, period=14, **_):
    adx_line, plus_di, minus_di = ind.adx(df["high"], df["low"], df["close"], int(period))
    return [
        Line("adx", f"ADX {int(period)}", adx_line, colour=GOLD, width=1.8),
        Line("plus_di", "+DI", plus_di, colour=ICE, width=1.1),
        Line("minus_di", "-DI", minus_di, colour=PINK, width=1.1),
    ]


def _compute_atr(df, period=14, **_):
    return [Line("atr", f"ATR {int(period)}",
                 ind.atr(df["high"], df["low"], df["close"], int(period)),
                 colour=ORANGE, width=1.5)]


def _compute_atr_pct(df, period=14, **_):
    atr_arr = ind.atr(df["high"], df["low"], df["close"], int(period))
    close = df["close"].to_numpy(dtype="float64")
    with np.errstate(invalid="ignore", divide="ignore"):
        pct = np.where(close > 0, 100.0 * atr_arr / close, np.nan)
    return [Line("atr_pct", f"ATR {int(period)} % of price", pct, colour=AMBER, width=1.5)]


def _compute_obv(df, **_):
    return [Line("obv", "On-Balance Volume", ind.obv(df["close"], df["volume"]),
                 colour=VIOLET, width=1.5)]


def _compute_volume(df, period=20, **_):
    vol = df["volume"].to_numpy(dtype="float64")
    avg = ind.sma(vol, int(period)) if int(period) >= 2 else None
    lines = [Line("volume", "Volume", vol, colour=SLATE, width=0.0, shape="bar")]
    if avg is not None:
        lines.append(Line("volume_ma", f"Volume SMA {int(period)}", avg,
                          colour=AMBER, width=1.3))
    return lines


def _compute_rvol(df, period=20, **_):
    """Relative volume: this bar's volume / the average of the PRIOR `period` bars."""
    vol = df["volume"].to_numpy(dtype="float64")
    period = int(period)
    avg = np.full(vol.size, np.nan, dtype="float64")
    for i in range(period, vol.size):
        window = vol[i - period:i]          # strictly previous bars, never i
        if np.isnan(window).any():
            continue
        mean = float(window.mean())
        if mean > 0:
            avg[i] = vol[i] / mean
    return [Line("rvol", f"Relative volume ({period})", avg, colour=TEAL, width=1.5)]


# --------------------------------------------------------------------------- #
# The catalogue
# --------------------------------------------------------------------------- #

def _specs() -> List[IndicatorSpec]:
    P = ParamSpec
    return [
        # ---- overlays on price ------------------------------------------- #
        IndicatorSpec(
            "ema", "EMA (exponential MA)", "overlay", _compute_ema,
            (P("period", "Period", 20, 2, 400),), warmup=400,
            description="Weighted average that reacts faster than an SMA. "
                        "Common choices: 9 / 21 / 50 / 200.",
            note="The trading strategy already uses EMA 20 and EMA 50 internally; "
                 "drawing them here does not change a single signal."),
        IndicatorSpec(
            "sma", "SMA (simple MA)", "overlay", _compute_sma,
            (P("period", "Period", 50, 2, 400),), warmup=400,
            description="Plain average of the last N closes - the classic trend filter."),
        IndicatorSpec(
            "wma", "WMA (weighted MA)", "overlay", _compute_wma,
            (P("period", "Period", 20, 2, 400),), warmup=400,
            description="Linear weights: the newest bar counts most."),
        IndicatorSpec(
            "hma", "HMA (Hull MA)", "overlay", _compute_hma,
            (P("period", "Period", 16, 4, 400),), warmup=400,
            description="Smoother and less laggy than an SMA of the same period.",
            note="Needs a longer warm-up: both inner averages must be ready."),
        IndicatorSpec(
            "bollinger", "Bollinger Bands", "overlay", _compute_bollinger,
            (P("period", "Period", 20, 5, 200), P("mult", "Std-dev multiple", 2.0, 0.5, 5.0, 0.1, False)),
            warmup=200,
            description="Average +/- N standard deviations. Width shows volatility; "
                        "a squeeze often precedes a range break.",
            note="A flat, untraded stretch makes the bands collapse onto the average - "
                 "that is the real reading, not a bug."),
        IndicatorSpec(
            "vwap", "VWAP (rolling)", "overlay", _compute_vwap,
            (P("period", "Rolling bars", 20, 5, 500),), warmup=500,
            description="Volume-weighted average price over a rolling window.",
            note="This is a ROLLING VWAP, not an exchange session VWAP. Bars with no "
                 "trades contribute no weight, and if the whole window has no volume "
                 "the line is left blank rather than drawn at zero."),
        IndicatorSpec(
            "donchian", "Donchian Channel", "overlay", _compute_donchian,
            (P("period", "Lookback", 20, 5, 200),), warmup=200,
            description="Highest high / lowest low of the last N bars - the classic "
                        "breakout channel."),
        IndicatorSpec(
            "keltner", "Keltner Channel", "overlay", _compute_keltner,
            (P("ema_period", "EMA period", 20, 5, 200), P("atr_period", "ATR period", 10, 2, 100),
             P("mult", "ATR multiple", 2.0, 0.5, 5.0, 0.1, False)), warmup=200,
            description="EMA +/- N ATRs: a volatility envelope built from true range."),
        IndicatorSpec(
            "supertrend", "Supertrend", "overlay", _compute_supertrend,
            (P("period", "ATR period", 10, 2, 100), P("mult", "ATR multiple", 3.0, 0.5, 10.0, 0.5, False)),
            warmup=100,
            description="ATR trailing stop that flips with the trend. Coloured by "
                        "direction - one of the few lines allowed to use green/red.",
            note="Flips are evaluated on the bar's close, so a reading can change "
                 "until that candle actually closes."),
        IndicatorSpec(
            "ichimoku", "Ichimoku Cloud", "overlay", _compute_ichimoku,
            (P("conversion", "Tenkan", 9, 2, 100), P("base", "Kijun", 26, 2, 200),
             P("span_b", "Senkou B", 52, 5, 300), P("displacement", "Displacement", 26, 0, 100)),
            warmup=300,
            description="Equilibrium lines plus two forward spans (the 'cloud').",
            note="The spans are drawn 26 bars to the RIGHT, which is the Ichimoku "
                 "convention - a projection of values already known, never of future "
                 "prices. Chikou (the past-shifted line) is deliberately not drawn: it "
                 "would paint current information over closed history."),
        IndicatorSpec(
            "psar", "Parabolic SAR", "overlay", _compute_psar,
            (P("step", "Step", 0.02, 0.005, 0.2, 0.005, False),
             P("max_step", "Max step", 0.2, 0.05, 1.0, 0.05, False)), warmup=20,
            description="Dots that trail price: below it in an uptrend, above it in a "
                        "downtrend."),
        # ---- own subplot -------------------------------------------------- #
        IndicatorSpec(
            "rsi", "RSI", "pane", _compute_rsi,
            (P("period", "Period", 14, 2, 100),), warmup=100, y_title="RSI",
            description="Momentum 0-100. The strategy uses RSI 14 as one of its four "
                        "conditions; this is the same calculation, drawn again."),
        IndicatorSpec(
            "macd", "MACD", "pane", _compute_macd,
            (P("fast", "Fast EMA", 12, 2, 100), P("slow", "Slow EMA", 26, 3, 300),
             P("signal", "Signal EMA", 9, 2, 100)), warmup=300, y_title="MACD",
            description="Difference of two EMAs plus a signal line and a histogram."),
        IndicatorSpec(
            "stochastic", "Stochastic", "pane", _compute_stoch,
            (P("k_period", "%K period", 14, 2, 100), P("d_period", "%D period", 3, 1, 50),
             P("smooth_k", "%K smoothing", 3, 1, 20)), warmup=100, y_title="%K / %D",
            description="Where the close sits inside the recent range, 0-100."),
        IndicatorSpec(
            "adx", "ADX / DMI", "pane", _compute_adx,
            (P("period", "Period", 14, 2, 100),), warmup=200, y_title="ADX",
            description="Trend STRENGTH (not direction): above ~25 is trending, below "
                        "~20 is ranging. +DI and -DI show which side is winning."),
        IndicatorSpec(
            "atr", "ATR (points)", "pane", _compute_atr,
            (P("period", "Period", 14, 2, 100),), warmup=100, y_title="ATR",
            description="Average true range in price units - the same ATR the "
                        "strategy uses for its stop-loss option."),
        IndicatorSpec(
            "atr_pct", "ATR (% of price)", "pane", _compute_atr_pct,
            (P("period", "Period", 14, 2, 100),), warmup=100, y_title="ATR %",
            description="ATR as a percentage of price, so volatility is comparable "
                        "across symbols and across time."),
        IndicatorSpec(
            "volume", "Volume", "pane", _compute_volume,
            (P("period", "Average period", 20, 2, 200),), warmup=200, y_title="Volume",
            description="Traded volume per candle with an optional average.",
            note="On a thin venue many bars show zero volume: that is a real reading "
                 "(no trades printed), and the app never signals from such a candle."),
        IndicatorSpec(
            "rvol", "Relative volume", "pane", _compute_rvol,
            (P("period", "Average period", 20, 2, 200),), warmup=200, y_title="x average",
            description="This candle's volume divided by the average of the PREVIOUS N "
                        "candles (1.0 = average). The current bar is excluded from its "
                        "own average, so the line cannot flatter itself."),
        IndicatorSpec(
            "obv", "On-Balance Volume", "pane", _compute_obv,
            (), warmup=2, y_title="OBV",
            description="Running total of volume, added on up-closes and subtracted on "
                        "down-closes. Read the DIRECTION, not the level."),
    ]


CATALOG: Dict[str, IndicatorSpec] = {spec.key: spec for spec in _specs()}

OVERLAYS: List[str] = [s.key for s in CATALOG.values() if s.kind == "overlay"]
PANES: List[str] = [s.key for s in CATALOG.values() if s.kind == "pane"]

#: How many extra panes the chart will draw before it stops adding rows.  Each
#: pane costs vertical space, and a chart split into eight strips is unreadable.
MAX_PANES = 3


def get(key: str) -> IndicatorSpec:
    """Look up a spec, with a clear error for an unknown key."""
    try:
        return CATALOG[key]
    except KeyError:
        raise KeyError(f"Unknown indicator '{key}'. Known: {', '.join(sorted(CATALOG))}.")


def validate_params(key: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    """
    Coerce and bounds-check a parameter dict for one indicator.

    Unknown keys are ignored and out-of-range values are CLAMPED to the declared
    bounds rather than silently producing a broken indicator - and never
    accepted as-is, which would let a stored setting crash the chart.
    """
    spec = get(key)
    merged = spec.defaults()
    for name, value in (params or {}).items():
        if name not in merged:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if number != number:                     # NaN
            continue
        p = next(p for p in spec.params if p.name == name)
        number = min(max(number, p.minimum), p.maximum)
        merged[name] = float(int(round(number))) if p.integer else number
    return merged


def compute(key: str, df: pd.DataFrame,
            params: Optional[Dict[str, Any]] = None) -> List[Line]:
    """
    Run one indicator over an OHLCV frame.

    Returns [] (never raises) when the frame is unusable - an indicator that
    cannot be drawn must show nothing rather than take the page down.
    """
    if df is None or df.empty:
        return []
    spec = get(key)
    try:
        lines = spec.compute(df, **validate_params(key, params))
    except Exception:                                                 # noqa: BLE001
        return []
    length = len(df)
    return [ln for ln in lines if ln.values is not None and len(ln.values) == length]


def compute_many(active: Sequence[str],
                 df: pd.DataFrame,
                 params: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, List[Line]]:
    """Compute several indicators at once, preserving the caller's order."""
    params = params or {}
    out: Dict[str, List[Line]] = {}
    for key in active:
        if key not in CATALOG:
            continue
        out[key] = compute(key, df, params.get(key))
    return out


def default_param_state() -> Dict[str, Dict[str, float]]:
    """Every indicator's default parameters - used to seed persisted settings."""
    return {key: spec.defaults() for key, spec in CATALOG.items()}


def warmup_for(keys: Sequence[str]) -> int:
    """Largest warm-up among the selected indicators (drives the caption)."""
    values = [CATALOG[k].warmup for k in keys if k in CATALOG]
    return max(values) if values else 0


__all__ = [
    "Line", "ParamSpec", "IndicatorSpec", "CATALOG", "OVERLAYS", "PANES", "MAX_PANES",
    "get", "validate_params", "compute", "compute_many", "default_param_state",
    "warmup_for",
]
