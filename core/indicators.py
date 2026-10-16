"""
core/indicators.py
==================
Transparent, dependency-light technical indicators.

Everything here is written with plain NumPy/Pandas loops or vectorised rolling
windows so the maths can be read and unit-tested.  No third-party TA library is
used - fewer moving parts, no version surprises on Streamlit Cloud.

Causality guarantee (important)
------------------------------
Every indicator at index ``i`` may only use data from indices ``<= i``.
Nothing here looks ahead.  `tests/test_indicators.py` proves this by
recomputing indicators on a truncated series and asserting the overlapping
values are identical.

Conventions
-----------
EMA
    Seeded with the simple moving average of the first `period` values, then the
    standard recursion ``ema[i] = a*x[i] + (1-a)*ema[i-1]`` with
    ``a = 2/(period+1)``.  The first ``period-1`` values are NaN (unknown),
    which prevents fake signals during warm-up.

RSI
    Wilder's original smoothing (also called RMA):
        avg_gain[period] = mean(gains[1..period])
        avg_gain[i]      = (avg_gain[i-1]*(period-1) + gain[i]) / period
    Edge cases, documented deliberately:
        avg_loss == 0 and avg_gain > 0  -> RSI = 100
        avg_gain == 0 and avg_loss > 0  -> RSI = 0
        both == 0 (perfectly flat)      -> RSI = 50 (neutral)
    RSI is NaN for the first `period` bars.

Swing levels
    `prior_swing_high[i]` = highest high of the `lookback` candles ending at
    ``i - shift``.  With the default ``shift = 1`` candle ``i`` can never
    contribute to the level it is being compared against, which removes the
    classic BOS look-ahead bug.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def _as_float_array(values) -> np.ndarray:
    """Convert a Series/array/list to a float64 NumPy array, preserving NaNs."""
    if isinstance(values, pd.Series):
        arr = values.to_numpy(dtype="float64", na_value=np.nan, copy=True)
    else:
        arr = np.asarray(values, dtype="float64")
    return arr


# --------------------------------------------------------------------------- #
# EMA
# --------------------------------------------------------------------------- #

def ema(values, period: int) -> np.ndarray:
    """
    Exponential moving average, SMA-seeded.

    Returns an array the same length as `values`; the first `period-1` entries
    are NaN because there is not enough information yet.
    """
    if period is None or int(period) < 1:
        raise ValueError("EMA period must be an integer >= 1.")
    period = int(period)
    x = _as_float_array(values)
    n = x.size
    out = np.full(n, np.nan, dtype="float64")
    if n == 0 or n < period:
        return out

    alpha = 2.0 / (period + 1.0)

    # Walk forward, skipping NaNs at the head (warm-up) but never looking back
    # beyond the current index.
    seed_end: Optional[int] = None
    for i in range(n):
        if np.isnan(x[i]):
            continue
        if seed_end is None:
            # need `period` consecutive non-NaN values to seed
            if i + period <= n and not np.isnan(x[i:i + period]).any():
                out[i + period - 1] = float(np.mean(x[i:i + period]))
                seed_end = i + period - 1
            else:
                break
        else:
            break

    if seed_end is None:
        return out

    prev = out[seed_end]
    for i in range(seed_end + 1, n):
        xi = x[i]
        if np.isnan(xi):
            out[i] = prev          # carry forward; do not invent a price
        else:
            prev = alpha * xi + (1.0 - alpha) * prev
            out[i] = prev
    return out


def ema_series(values: pd.Series, period: int) -> pd.Series:
    """Pandas-friendly wrapper around :func:`ema`."""
    return pd.Series(ema(values, period), index=values.index, name=f"ema{period}")


# --------------------------------------------------------------------------- #
# RSI (Wilder)
# --------------------------------------------------------------------------- #

def rsi(values, period: int = 14) -> np.ndarray:
    """
    Relative Strength Index using Wilder's smoothing.

    Returns an array the same length as `values`; the first `period` entries are
    NaN (RSI needs `period` *changes*, i.e. `period+1` prices).
    """
    if period is None or int(period) < 1:
        raise ValueError("RSI period must be an integer >= 1.")
    period = int(period)
    x = _as_float_array(values)
    n = x.size
    out = np.full(n, np.nan, dtype="float64")
    if n < period + 1:
        return out

    delta = np.diff(x)                     # length n-1, delta[i] = x[i+1]-x[i]
    gains = np.where(np.isnan(delta), 0.0, np.clip(delta, 0.0, None))
    losses = np.where(np.isnan(delta), 0.0, np.clip(-delta, 0.0, None))

    avg_gain = float(np.mean(gains[:period]))
    avg_loss = float(np.mean(losses[:period]))
    out[period] = _rsi_value(avg_gain, avg_loss)

    for i in range(period + 1, n):
        g = gains[i - 1]
        l = losses[i - 1]
        avg_gain = (avg_gain * (period - 1) + g) / period
        avg_loss = (avg_loss * (period - 1) + l) / period
        out[i] = _rsi_value(avg_gain, avg_loss)
    return out


def _rsi_value(avg_gain: float, avg_loss: float) -> float:
    """RSI from smoothed averages, with the documented flat/zero edge cases."""
    if avg_loss == 0.0 and avg_gain == 0.0:
        return 50.0          # perfectly flat market -> neutral
    if avg_loss == 0.0:
        return 100.0         # only gains
    if avg_gain == 0.0:
        return 0.0           # only losses
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def rsi_series(values: pd.Series, period: int = 14) -> pd.Series:
    return pd.Series(rsi(values, period), index=values.index, name=f"rsi{period}")


# --------------------------------------------------------------------------- #
# ATR (Wilder) - used for the optional ATR stop mode
# --------------------------------------------------------------------------- #

def atr(high, low, close, period: int = 14) -> np.ndarray:
    """Average True Range with Wilder smoothing."""
    h = _as_float_array(high)
    l = _as_float_array(low)
    c = _as_float_array(close)
    n = c.size
    out = np.full(n, np.nan, dtype="float64")
    if n < 2 or period < 1 or n < period + 1:
        return out

    prev_close = np.concatenate(([np.nan], c[:-1]))
    tr = np.nanmax(
        np.vstack([h - l, np.abs(h - prev_close), np.abs(l - prev_close)]),
        axis=0,
    )
    tr[0] = np.nan  # no previous close for the first bar

    valid = tr[1:period + 1]
    if np.isnan(valid).any() or valid.size < period:
        return out
    prev = float(np.mean(valid))
    out[period] = prev
    for i in range(period + 1, n):
        t = tr[i]
        if np.isnan(t):
            out[i] = prev
        else:
            prev = (prev * (period - 1) + t) / period
            out[i] = prev
    return out


# --------------------------------------------------------------------------- #
# Swing highs / lows and Break of Structure
# --------------------------------------------------------------------------- #

def prior_swing_high(high, lookback: int, shift: int = 1) -> np.ndarray:
    """
    Highest high of the `lookback` candles ending `shift` candles before `i`.

    ``prior_swing_high[i] = max(high[i-shift-lookback+1 : i-shift+1])``

    With shift >= 1 candle `i` cannot influence its own reference level.
    Entries without a full window are NaN.
    """
    return _rolling_extreme(_as_float_array(high), lookback, shift, mode="max")


def prior_swing_low(low, lookback: int, shift: int = 1) -> np.ndarray:
    """Lowest low of the `lookback` candles ending `shift` candles before `i`."""
    return _rolling_extreme(_as_float_array(low), lookback, shift, mode="min")


def _rolling_extreme(x: np.ndarray, lookback: int, shift: int, mode: str) -> np.ndarray:
    if lookback < 1:
        raise ValueError("lookback must be >= 1")
    if shift < 0:
        raise ValueError("shift must be >= 0")
    n = x.size
    out = np.full(n, np.nan, dtype="float64")
    op = np.nanmax if mode == "max" else np.nanmin
    for i in range(n):
        end = i - shift + 1              # exclusive
        start = end - lookback
        if start < 0 or end <= 0 or end > n:
            continue
        window = x[start:end]
        if np.isnan(window).all():
            continue
        val = op(window)
        out[i] = float(val) if not np.isnan(val) else np.nan
    return out


# --------------------------------------------------------------------------- #
# v1.3.0: indicator library for the Indicators tab
#
# Every function here obeys the same three rules, because they are the whole
# point of this project:
#
#   1. CAUSAL.  The value at index i is computed from data at indices <= i only.
#      Nothing is centred, nothing is shifted backwards, and no function ever
#      touches a future row.
#   2. NaN-HONEST.  Warm-up is NaN, never a guessed seed.  A caller can always
#      tell "not enough history yet" from "the value is zero".
#   3. NO SIDE EFFECTS.  Inputs are never mutated; output is a fresh float64
#      array (or a tuple of them) the same length as the input.
#
# These indicators are DISPLAY-ONLY in the app: they are drawn on the chart and
# they never enter the trading strategy unless the user wires one in explicitly.
# --------------------------------------------------------------------------- #

def _rolling_mean(x: np.ndarray, period: int) -> np.ndarray:
    """Simple moving average with an honest NaN warm-up (no partial windows)."""
    period = int(period)
    out = np.full(x.size, np.nan, dtype="float64")
    if period < 1 or x.size < period:
        return out
    for i in range(period - 1, x.size):
        window = x[i - period + 1: i + 1]
        if np.isnan(window).any():
            continue
        out[i] = float(window.mean())
    return out


def _rolling_std(x: np.ndarray, period: int, ddof: int = 0) -> np.ndarray:
    period = int(period)
    out = np.full(x.size, np.nan, dtype="float64")
    if period < 2 or x.size < period:
        return out
    for i in range(period - 1, x.size):
        window = x[i - period + 1: i + 1]
        if np.isnan(window).any():
            continue
        out[i] = float(window.std(ddof=ddof))
    return out


def _rolling_sum(x: np.ndarray, period: int) -> np.ndarray:
    period = int(period)
    out = np.full(x.size, np.nan, dtype="float64")
    if period < 1 or x.size < period:
        return out
    for i in range(period - 1, x.size):
        window = x[i - period + 1: i + 1]
        if np.isnan(window).any():
            continue
        out[i] = float(window.sum())
    return out


def _rolling_extreme_series(x: np.ndarray, period: int, mode: str) -> np.ndarray:
    """Rolling max/min over the CURRENT bar and the previous `period-1` bars."""
    period = int(period)
    out = np.full(x.size, np.nan, dtype="float64")
    if period < 1 or x.size < period:
        return out
    fn = np.max if mode == "max" else np.min
    for i in range(period - 1, x.size):
        window = x[i - period + 1: i + 1]
        if np.isnan(window).any():
            continue
        out[i] = float(fn(window))
    return out


def sma(values, period: int) -> np.ndarray:
    """Simple moving average.  First ``period-1`` values are NaN."""
    if period is None or int(period) < 1:
        raise ValueError("SMA period must be an integer >= 1.")
    return _rolling_mean(_as_float_array(values), int(period))


def sma_series(values: pd.Series, period: int) -> pd.Series:
    return pd.Series(sma(values, period), index=values.index, name=f"sma{period}")


def wma(values, period: int) -> np.ndarray:
    """Linearly weighted moving average (most recent bar weighted highest)."""
    period = int(period)
    x = _as_float_array(values)
    out = np.full(x.size, np.nan, dtype="float64")
    if period < 1 or x.size < period:
        return out
    weights = np.arange(1, period + 1, dtype="float64")
    weight_sum = float(weights.sum())
    for i in range(period - 1, x.size):
        window = x[i - period + 1: i + 1]
        if np.isnan(window).any():
            continue
        out[i] = float(np.dot(window, weights) / weight_sum)
    return out


def hma(values, period: int) -> np.ndarray:
    """
    Hull moving average: WMA(2*WMA(n/2) - WMA(n), sqrt(n)).

    Smoother and less laggy than an SMA of the same period, at the cost of a
    longer effective warm-up (both inner WMAs must be ready).
    """
    period = int(period)
    if period < 2:
        raise ValueError("HMA period must be an integer >= 2.")
    x = _as_float_array(values)
    half = max(1, period // 2)
    root = max(1, int(round(period ** 0.5)))
    inner = 2.0 * wma(x, half) - wma(x, period)
    return wma(inner, root)


def bollinger_bands(values, period: int = 20, mult: float = 2.0,
                    ddof: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bollinger Bands -> (lower, middle, upper).

    The width is a rolling standard deviation of the SAME window that produced
    the middle band, so a band is never drawn from partial data.
    """
    period = int(period)
    x = _as_float_array(values)
    mid = _rolling_mean(x, period)
    sd = _rolling_std(x, period, ddof=ddof)
    upper = mid + float(mult) * sd
    lower = mid - float(mult) * sd
    # A window with zero variance puts all three lines on top of each other;
    # that is real (a dead market), not a bug, so it is left alone.
    return lower, mid, upper


def vwap(high, low, close, volume, period: int = 20) -> np.ndarray:
    """
    Rolling volume-weighted average price over `period` bars.

    This is a ROLLING VWAP, not an exchange-anchored session VWAP: the app has
    no session concept in the data feed, and computing a "daily" VWAP from a
    300-candle window would silently mix days.  The label in the UI says
    "rolling" for exactly that reason.
    """
    period = int(period)
    h, l, c, v = (_as_float_array(a) for a in (high, low, close, volume))
    n = c.size
    out = np.full(n, np.nan, dtype="float64")
    if period < 1 or n < period:
        return out
    typical = (h + l + c) / 3.0
    pv = typical * v
    for i in range(period - 1, n):
        vol_window = v[i - period + 1: i + 1]
        if np.isnan(vol_window).any():
            continue
        total_vol = float(vol_window.sum())
        if total_vol <= 0:                     # no trades -> no VWAP. Never 0.
            continue
        window_pv = pv[i - period + 1: i + 1]
        if np.isnan(window_pv).any():
            continue
        out[i] = float(window_pv.sum()) / total_vol
    return out


def donchian(high, low, period: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Donchian channel -> (lower, middle, upper).  Includes the current bar."""
    period = int(period)
    h, l = _as_float_array(high), _as_float_array(low)
    upper = _rolling_extreme_series(h, period, "max")
    lower = _rolling_extreme_series(l, period, "min")
    mid = (upper + lower) / 2.0
    return lower, mid, upper


def true_range(high, low, close) -> np.ndarray:
    """Wilder's true range; the first bar uses high-low (no previous close)."""
    h, l, c = _as_float_array(high), _as_float_array(low), _as_float_array(close)
    n = h.size
    out = np.full(n, np.nan, dtype="float64")
    if n == 0:
        return out
    out[0] = h[0] - l[0] if not (np.isnan(h[0]) or np.isnan(l[0])) else np.nan
    prev_close = c[0]
    for i in range(1, n):
        if np.isnan(prev_close) or np.isnan(h[i]) or np.isnan(l[i]):
            out[i] = np.nan
        else:
            out[i] = max(h[i] - l[i], abs(h[i] - prev_close), abs(l[i] - prev_close))
        prev_close = c[i]
    return out


def adx(high, low, close, period: int = 14) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Wilder's ADX with +DI / -DI -> (adx, plus_di, minus_di).

    ADX measures TREND STRENGTH (not direction): >25 means a trending market,
    <20 a ranging one.  It is honest about its own warm-up - Wilder smoothing
    needs roughly 2x the period before the first value appears.
    """
    period = int(period)
    h, l, c = _as_float_array(high), _as_float_array(low), _as_float_array(close)
    n = h.size
    nan = np.full(n, np.nan, dtype="float64")
    if n < period * 2 + 1 or period < 1:
        return nan.copy(), nan.copy(), nan.copy()

    up_move = np.full(n, np.nan, dtype="float64")
    down_move = np.full(n, np.nan, dtype="float64")
    for i in range(1, n):
        if any(np.isnan(v) for v in (h[i], h[i - 1], l[i], l[i - 1])):
            continue
        up_move[i] = h[i] - h[i - 1]
        down_move[i] = l[i - 1] - l[i]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm[0] = np.nan
    minus_dm[0] = np.nan
    tr = true_range(h, l, c)
    tr[0] = np.nan

    alpha = 1.0 / period
    def _wilder(x: np.ndarray) -> np.ndarray:
        out = np.full(n, np.nan, dtype="float64")
        start = period                     # first `period` bars seed the average
        if n <= start:
            return out
        seed = x[1:start + 1]
        if np.isnan(seed).any():
            return out
        out[start] = float(seed.sum())
        for i in range(start + 1, n):
            if np.isnan(x[i]) or np.isnan(out[i - 1]):
                out[i] = out[i - 1]
                continue
            out[i] = out[i - 1] - (out[i - 1] / period) + x[i]
        return out

    tr_s = _wilder(tr)
    plus_s = _wilder(plus_dm)
    minus_s = _wilder(minus_dm)
    with np.errstate(invalid="ignore", divide="ignore"):
        plus_di = 100.0 * plus_s / tr_s
        minus_di = 100.0 * minus_s / tr_s
        dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_out = np.full(n, np.nan, dtype="float64")
    first = period * 2 - 1
    if first < n and not np.isnan(dx[period:first + 1]).any():
        adx_out[first] = float(np.nanmean(dx[period:first + 1]))
        for i in range(first + 1, n):
            if np.isnan(dx[i]) or np.isnan(adx_out[i - 1]):
                adx_out[i] = adx_out[i - 1]
                continue
            adx_out[i] = (adx_out[i - 1] * (period - 1) + dx[i]) / period
    return adx_out, plus_di, minus_di


def stochastic(high, low, close, k_period: int = 14, d_period: int = 3,
               smooth_k: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stochastic oscillator -> (%K, %D), both 0-100.

    %K is smoothed `smooth_k` times (the "slow" convention) and %D is the SMA of
    %K.  When the high-low range is zero the reading is undefined and stays NaN -
    a flat, untraded candle must not report "50" as if it were information.
    """
    h, l, c = _as_float_array(high), _as_float_array(low), _as_float_array(close)
    n = c.size
    raw = np.full(n, np.nan, dtype="float64")
    hh = _rolling_extreme_series(h, int(k_period), "max")
    ll = _rolling_extreme_series(l, int(k_period), "min")
    with np.errstate(invalid="ignore", divide="ignore"):
        span = hh - ll
        raw = np.where(span > 0, 100.0 * (c - ll) / span, np.nan)
        raw = np.where(np.isnan(span), np.nan, raw)
    k = _rolling_mean(raw, int(smooth_k)) if smooth_k and int(smooth_k) > 1 else raw
    d = _rolling_mean(k, int(d_period))
    return k, d


def macd(values, fast: int = 12, slow: int = 26,
         signal: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    MACD -> (macd_line, signal_line, histogram).

    Both EMAs are SMA-seeded and strictly causal, so the histogram at bar i uses
    bars 0..i only.
    """
    fast, slow, signal = int(fast), int(slow), int(signal)
    if fast >= slow:
        raise ValueError("MACD fast period must be smaller than the slow period.")
    x = _as_float_array(values)
    macd_line = ema(x, fast) - ema(x, slow)
    # The signal EMA must start where the MACD line first exists, otherwise its
    # seed would be contaminated by the MACD warm-up NaNs.
    first = np.argmax(~np.isnan(macd_line)) if (~np.isnan(macd_line)).any() else None
    sig_line = np.full(x.size, np.nan, dtype="float64")
    if first is not None:
        tail = macd_line[first:]
        sig_line[first:] = ema(tail, signal)
    hist = macd_line - sig_line
    return macd_line, sig_line, hist


def obv(close, volume) -> np.ndarray:
    """
    On-Balance Volume: cumulative volume, added on up-closes, subtracted on
    down-closes.  Starts at 0 on the first bar, so it shows DIRECTION and
    divergence, not an absolute level.
    """
    c, v = _as_float_array(close), _as_float_array(volume)
    n = c.size
    out = np.full(n, np.nan, dtype="float64")
    if n == 0:
        return out
    running = 0.0
    out[0] = 0.0 if not np.isnan(v[0]) else np.nan
    for i in range(1, n):
        if np.isnan(c[i]) or np.isnan(c[i - 1]) or np.isnan(v[i]):
            out[i] = out[i - 1]
            continue
        if c[i] > c[i - 1]:
            running += v[i]
        elif c[i] < c[i - 1]:
            running -= v[i]
        out[i] = running
    return out


def psar(high, low, step: float = 0.02, max_step: float = 0.2) -> np.ndarray:
    """
    Parabolic SAR (Wilder).  Dots below price = uptrend, above = downtrend.

    The first value is seeded from the first two bars (a documented, causal
    convention) and the acceleration factor is capped at `max_step`.
    """
    h, l = _as_float_array(high), _as_float_array(low)
    n = h.size
    out = np.full(n, np.nan, dtype="float64")
    if n < 3:
        return out
    step, max_step = float(step), float(max_step)
    uptrend = True
    af = step
    extreme = h[0]
    sar = l[0]
    out[0] = sar
    for i in range(1, n):
        prior = sar
        sar = prior + af * (extreme - prior)
        if uptrend:
            sar = min(sar, l[i - 1], l[max(0, i - 2)])
            if l[i] < sar:                     # flip to downtrend
                uptrend = False
                sar = extreme
                extreme = l[i]
                af = step
            elif h[i] > extreme:
                extreme = h[i]
                af = min(af + step, max_step)
        else:
            sar = max(sar, h[i - 1], h[max(0, i - 2)])
            if h[i] > sar:                     # flip to uptrend
                uptrend = True
                sar = extreme
                extreme = h[i]
                af = step
            elif l[i] < extreme:
                extreme = l[i]
                af = min(af + step, max_step)
        out[i] = sar
    return out


def supertrend(high, low, close, period: int = 10,
               multiplier: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Supertrend -> (line, direction) where direction is +1 (up) or -1 (down).

    Built on ATR (the same ATR the strategy uses), and strictly causal: the band
    at bar i is only allowed to tighten while the trend continues, and the flip
    is evaluated against the bar's close.  Direction is NaN during warm-up.
    """
    h, l, c = _as_float_array(high), _as_float_array(low), _as_float_array(close)
    n = c.size
    line = np.full(n, np.nan, dtype="float64")
    direction = np.full(n, np.nan, dtype="float64")
    atr_arr = atr(h, l, c, int(period))
    if n == 0:
        return line, direction
    upper_basic = (h + l) / 2.0 + float(multiplier) * atr_arr
    lower_basic = (h + l) / 2.0 - float(multiplier) * atr_arr
    final_upper = np.full(n, np.nan, dtype="float64")
    final_lower = np.full(n, np.nan, dtype="float64")
    started = False
    for i in range(n):
        if any(np.isnan(v) for v in (upper_basic[i], lower_basic[i])):
            continue
        if not started:
            final_upper[i], final_lower[i] = upper_basic[i], lower_basic[i]
            direction[i] = 1.0
            line[i] = final_lower[i]
            started = True
            continue
        prev_u, prev_l = final_upper[i - 1], final_lower[i - 1]
        final_upper[i] = (upper_basic[i] if (upper_basic[i] < prev_u or c[i - 1] > prev_u)
                          else prev_u)
        final_lower[i] = (lower_basic[i] if (lower_basic[i] > prev_l or c[i - 1] < prev_l)
                          else prev_l)
        prev_dir = direction[i - 1]
        if prev_dir == 1.0:
            direction[i] = -1.0 if c[i] < final_lower[i] else 1.0
        else:
            direction[i] = 1.0 if c[i] > final_upper[i] else -1.0
        line[i] = final_lower[i] if direction[i] == 1.0 else final_upper[i]
    return line, direction


def ichimoku(high, low, conversion: int = 9, base: int = 26, span_b: int = 52,
             displacement: int = 26) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Ichimoku Kinko Hyo -> (tenkan, kijun, senkou_a, senkou_b).

    Returns the two spans UNSHIFTED, i.e. the value belonging to bar i is
    computed from bars up to i.  The DISPLACEMENT (drawing the spans 26 bars
    ahead, and Chikou 26 bars behind) is applied by the chart layer, not here -
    this function stays a pure, testable, causal transform.  Chikou is omitted
    on purpose: shifting a line into the past would draw price-derived
    information over candles that had already closed.
    """
    h, l = _as_float_array(high), _as_float_array(low)
    tenkan = (_rolling_extreme_series(h, int(conversion), "max")
              + _rolling_extreme_series(l, int(conversion), "min")) / 2.0
    kijun = (_rolling_extreme_series(h, int(base), "max")
             + _rolling_extreme_series(l, int(base), "min")) / 2.0
    span_a = (tenkan + kijun) / 2.0
    span_b_line = (_rolling_extreme_series(h, int(span_b), "max")
                   + _rolling_extreme_series(l, int(span_b), "min")) / 2.0
    return tenkan, kijun, span_a, span_b_line


def keltner(high, low, close, ema_period: int = 20, atr_period: int = 10,
            mult: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keltner Channel -> (lower, middle, upper) around an EMA, using ATR."""
    mid = ema(close, int(ema_period))
    band = float(mult) * atr(high, low, close, int(atr_period))
    return mid - band, mid, mid + band


def detect_bos(
    df: pd.DataFrame,
    lookback: int = 10,
    shift: int = 1,
    require_close_breakout: bool = True,
) -> pd.DataFrame:
    """
    Add Break-of-Structure columns to a copy of `df`.

    Columns added
    -------------
    prior_swing_high, prior_swing_low : causal reference levels
    bos_bull, bos_bear               : confirmed close-through breakouts
    bos_bull_wick, bos_bear_wick     : wick broke the level but the close did not

    `require_close_breakout=True` (default) means a wick-only poke does NOT
    count as a BOS.  The wick columns are still produced so the UI can explain
    "a wick broke the level but the candle did not close through it".
    """
    for col in ("high", "low", "close"):
        if col not in df.columns:
            raise ValueError(f"detect_bos needs a '{col}' column")

    out = df.copy()
    psh = prior_swing_high(out["high"].to_numpy(), lookback, shift)
    psl = prior_swing_low(out["low"].to_numpy(), lookback, shift)
    close = out["close"].to_numpy(dtype="float64")
    high = out["high"].to_numpy(dtype="float64")
    low = out["low"].to_numpy(dtype="float64")

    out["prior_swing_high"] = psh
    out["prior_swing_low"] = psl

    with np.errstate(invalid="ignore"):
        bull_close = np.where(np.isnan(psh), False, close > psh)
        bear_close = np.where(np.isnan(psl), False, close < psl)
        bull_wick_raw = np.where(np.isnan(psh), False, high > psh)
        bear_wick_raw = np.where(np.isnan(psl), False, low < psl)

    # A "wick-only" break is a break of the level that the close did not confirm.
    out["bos_bull_wick"] = bull_wick_raw & ~bull_close
    out["bos_bear_wick"] = bear_wick_raw & ~bear_close

    if require_close_breakout:
        out["bos_bull"] = bull_close
        out["bos_bear"] = bear_close
    else:
        # Wick breakouts also qualify (less strict, more signals, more noise).
        out["bos_bull"] = bull_wick_raw
        out["bos_bear"] = bear_wick_raw

    return out


# --------------------------------------------------------------------------- #
# Convenience: compute every indicator the strategy needs
# --------------------------------------------------------------------------- #

def enrich(
    df: pd.DataFrame,
    swing_lookback: int = 10,
    swing_shift: int = 1,
    rsi_period: int = 14,
    ema_fast: int = 20,
    ema_slow: int = 50,
    atr_period: int = 14,
    require_close_breakout: bool = True,
) -> pd.DataFrame:
    """
    Return a copy of an OHLCV frame with all indicator columns attached.

    Expected input columns: timestamp, open, high, low, close, volume.
    The frame must be sorted ascending by timestamp and contain CLOSED candles
    only (the data feed is responsible for that split).
    """
    missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"OHLCV frame is missing columns: {missing}")

    out = df.copy()
    if not out.empty:
        out = out.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

    close = out["close"]
    out[f"ema{ema_fast}"] = ema(close, ema_fast)
    out[f"ema{ema_slow}"] = ema(close, ema_slow)
    out["ema_fast"] = out[f"ema{ema_fast}"]
    out["ema_slow"] = out[f"ema{ema_slow}"]
    out["rsi"] = rsi(close, rsi_period)
    out["atr"] = atr(out["high"], out["low"], close, atr_period)

    out = detect_bos(
        out,
        lookback=swing_lookback,
        shift=swing_shift,
        require_close_breakout=require_close_breakout,
    )
    return out


def candles_are_complete(timestamps, timeframe_ms: int, now_ms: int) -> np.ndarray:
    """
    Boolean mask: True where the candle has fully closed.

    A candle opened at `t` closes at `t + timeframe_ms`.  It is complete only
    when the current time is at or beyond that boundary.
    """
    ts = _as_float_array(timestamps)
    return (ts + timeframe_ms) <= now_ms


def find_missing_candles(timestamps, timeframe_ms: int) -> Tuple[int, list]:
    """
    Count gaps in a candle series without inventing prices.

    Returns (missing_count, list_of_gap_start_timestamps).  Gaps are reported so
    the UI can be honest about thin data; they are never filled.
    """
    ts = _as_float_array(timestamps)
    gaps = []
    if ts.size < 2:
        return 0, gaps
    diffs = np.diff(ts)
    for i, d in enumerate(diffs):
        steps = int(round(d / timeframe_ms))
        if steps > 1:
            gaps.append(float(ts[i]))
    missing = sum(int(round(d / timeframe_ms)) - 1 for d in diffs if round(d / timeframe_ms) > 1)
    return int(missing), gaps


__all__ = [
    "OHLCV_COLUMNS", "ema", "ema_series", "rsi", "rsi_series", "atr",
    "prior_swing_high", "prior_swing_low", "detect_bos", "enrich",
    "candles_are_complete", "find_missing_candles",
    # v1.3.0 display-indicator library
    "sma", "sma_series", "wma", "hma", "bollinger_bands", "vwap", "donchian",
    "true_range", "adx", "stochastic", "macd", "obv", "psar", "supertrend",
    "ichimoku", "keltner",
]
