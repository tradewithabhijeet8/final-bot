"""
core/multi_timeframe.py
=======================
Higher-timeframe (HTF) context for the price-action add-on.

WHY THIS MODULE IS MOSTLY ABOUT BOOK-KEEPING
--------------------------------------------
Reading a higher timeframe is easy to get subtly wrong, and the subtle version
is a look-ahead bug: while your 1h chart is mid-way through the 13:00 hour, the
4h candle that started at 12:00 HAS NOT CLOSED.  Its high, low and close can
still change, so its structure bias, its fair value gaps and its dealing range
are not knowable yet.  Publishing them would let the dashboard "see" the future.

Rule used here, with no exceptions:

    An HTF candle may only inform an LTF candle whose own close happens at or
    after the HTF candle's close.

        available(i)  <=>  T_htf + htf_ms  <=  t_i + ltf_ms

Nothing is re-implemented:
  * aggregation reuses `MarketDataFeed._aggregate` (the exact code path that
    already builds 4h candles from 1h for the live feed), which emits a bucket
    only when it is complete AND its end is in the past;
  * HTF analysis reuses `price_action.analyse` on the aggregated closed candles;
  * alignment is a `searchsorted` on HTF close times, so it cannot drift.

If there is not enough higher-timeframe history, this module says so and returns
NaN columns.  It never pads, extrapolates or guesses an HTF state.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .config import TIMEFRAME_MS, PriceActionParams
from .price_action import PriceActionResult, Zone, analyse

#: "auto" walks one step up this chain.
HTF_CHAIN = {"1m": "5m", "5m": "15m", "15m": "30m", "30m": "1h", "1h": "4h",
             "4h": "1d", "2h": "4h", "6h": "1d"}

#: HTF zone kinds that count as a location worth reporting.
ZONE_GROUPS = (("fvg_bull", "fvg_bear"), ("ob_bull", "ob_bear"),
               ("sd_demand", "sd_supply"))

HTF_COLUMNS = [
    "htf_ts", "htf_closed_ts", "htf_index", "htf_bias", "htf_pdz",
    "htf_range_top", "htf_range_bottom", "htf_range_mid",
    "htf_close", "htf_high", "htf_low", "htf_zone_dir", "htf_zone_strength",
]
HTF_BOOL_COLUMNS = [
    "htf_in_fvg_bull", "htf_in_fvg_bear", "htf_in_ob_bull", "htf_in_ob_bear",
    "htf_in_sd_demand", "htf_in_sd_supply", "htf_available",
]


def resolve_htf(ltf_timeframe: str, requested: str = "auto") -> Optional[str]:
    """Higher timeframe for `ltf_timeframe`, or None when there isn't one."""
    if requested and requested != "auto":
        return requested if requested in TIMEFRAME_MS else None
    return HTF_CHAIN.get(ltf_timeframe)


def infer_timeframe(df: pd.DataFrame) -> Optional[str]:
    """
    Identify a frame's timeframe from its timestamp spacing.

    Used only when the caller cannot say (e.g. a backtest frame).  Returns None
    rather than guessing when the spacing is irregular or unknown.
    """
    ts = pd.Series(df["timestamp"].to_numpy(dtype="float64"))
    if len(ts) < 3:
        return None
    step = float(np.median(np.diff(ts.to_numpy())))
    if not np.isfinite(step) or step <= 0:
        return None
    best = min(TIMEFRAME_MS.items(), key=lambda kv: abs(kv[1] - step))
    # Only accept an exact match: a 1h series with gaps must not be relabelled.
    return best[0] if abs(best[1] - step) < 1.0 else None


def _empty(df: pd.DataFrame, ltf: str, htf: Optional[str],
           warnings: List[str]) -> "MtfResult":
    """Honest 'no HTF context' result: columns exist, every value is NaN/False."""
    out = pd.DataFrame(index=df.index)
    for c in HTF_COLUMNS:
        out[c] = np.nan
    for c in HTF_BOOL_COLUMNS:
        out[c] = False
    return MtfResult(available=False, ltf_timeframe=ltf, htf_timeframe=htf,
                     htf_ms=TIMEFRAME_MS.get(htf or "", 0), df=out,
                     htf_df=None, htf_result=None, warnings=warnings)


@dataclass
class MtfResult:
    available: bool
    ltf_timeframe: str
    htf_timeframe: Optional[str]
    htf_ms: int
    df: pd.DataFrame                                  # LTF-length htf_* columns
    htf_df: Optional[pd.DataFrame] = None             # analysed CLOSED HTF candles
    htf_result: Optional[PriceActionResult] = None
    warnings: List[str] = field(default_factory=list)

    def zones(self, live_only: bool = True) -> List[Zone]:
        """HTF zones, with kinds prefixed `htf_` so they never collide with LTF ones."""
        if self.htf_result is None:
            return []
        src = (self.htf_result.active_zones() if live_only else self.htf_result.zones)
        return [replace(z, kind=f"htf_{z.kind}") for z in src]

    def latest(self) -> Dict[str, Any]:
        """HTF context for the most recent closed LTF candle (UI convenience)."""
        if self.df.empty or not bool(self.df["htf_available"].iloc[-1]):
            return {"available": False, "htf_timeframe": self.htf_timeframe,
                    "warnings": list(self.warnings)}
        row = self.df.iloc[-1]
        kinds = [c[8:] for c in HTF_BOOL_COLUMNS
                 if c.startswith("htf_in_") and bool(row[c])]
        return {
            "available": True,
            "htf_timeframe": self.htf_timeframe,
            "bias": int(row["htf_bias"]) if np.isfinite(row["htf_bias"]) else 0,
            "pdz": None if not np.isfinite(row["htf_pdz"]) else float(row["htf_pdz"]),
            "range_top": None if not np.isfinite(row["htf_range_top"]) else float(row["htf_range_top"]),
            "range_bottom": None if not np.isfinite(row["htf_range_bottom"]) else float(row["htf_range_bottom"]),
            "range_mid": None if not np.isfinite(row["htf_range_mid"]) else float(row["htf_range_mid"]),
            "close": None if not np.isfinite(row["htf_close"]) else float(row["htf_close"]),
            "closed_ts": None if not np.isfinite(row["htf_closed_ts"]) else float(row["htf_closed_ts"]),
            "inside_zones": kinds,
            "zone_dir": int(row["htf_zone_dir"]) if np.isfinite(row["htf_zone_dir"]) else 0,
            "zone_strength": float(row["htf_zone_strength"]) if np.isfinite(row["htf_zone_strength"]) else 0.0,
            "warnings": list(self.warnings),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {"available": self.available, "ltf_timeframe": self.ltf_timeframe,
                "htf_timeframe": self.htf_timeframe, "htf_ms": self.htf_ms,
                "warnings": list(self.warnings),
                "htf_candles": 0 if self.htf_df is None else len(self.htf_df),
                "latest": self.latest()}


def analyse_mtf(df: pd.DataFrame, ltf_timeframe: Optional[str],
                params: PriceActionParams,
                htf_timeframe: Optional[str] = None) -> MtfResult:
    """
    Build strictly causal higher-timeframe context for every LTF candle in `df`.

    `df` must contain CLOSED LTF candles sorted ascending by timestamp.
    """
    warnings: List[str] = []
    ltf = ltf_timeframe or infer_timeframe(df)
    if ltf is None or ltf not in TIMEFRAME_MS:
        warnings.append(
            "Cannot determine the chart timeframe from the candle spacing, so no "
            "higher-timeframe context is published (nothing is guessed).")
        return _empty(df, str(ltf_timeframe), None, warnings)
    if df.empty:
        return _empty(df, ltf, None, ["No closed candles to aggregate."])

    wanted = htf_timeframe or params.mtf_timeframe or "auto"
    htf = resolve_htf(ltf, wanted)
    if htf is None:
        warnings.append(f"Unknown higher timeframe '{wanted}'; no HTF context published.")
        return _empty(df, ltf, None, warnings)
    if TIMEFRAME_MS[htf] <= TIMEFRAME_MS[ltf]:
        warnings.append(
            f"Higher timeframe '{htf}' is not above the chart timeframe '{ltf}', "
            f"so it was refused rather than silently swapped.")
        return _empty(df, ltf, htf, warnings)

    ltf_ms = TIMEFRAME_MS[ltf]
    htf_ms = TIMEFRAME_MS[htf]

    # ---- aggregate with the SAME code the live feed uses -------------------- #
    from .data_feed import MarketDataFeed          # lazy: keeps ccxt out of the
    rows = df[["timestamp", "open", "high", "low", "close", "volume"]] \
        .to_numpy(dtype="float64").tolist()        # analysis import chain
    now_ms = float(df["timestamp"].iloc[-1]) + ltf_ms      # end of the last CLOSED candle
    buckets, forming, agg_warnings = MarketDataFeed._aggregate(rows, ltf, htf, now_ms)
    warnings.extend(agg_warnings)
    if forming is not None:
        # Deliberately ignored: the forming HTF bucket is not knowable yet.
        warnings.append(
            f"The in-progress {htf} candle was EXCLUDED (it has not closed).")

    if len(buckets) < params.mtf_min_candles:
        warnings.append(
            f"Not enough higher-timeframe history: {len(buckets)} closed {htf} "
            f"candles available, {params.mtf_min_candles} needed. No HTF context "
            f"is published rather than publishing a half-warmed-up read. "
            f"Fetch more {ltf} candles or pick a lower HTF.")
        return _empty(df, ltf, htf, warnings)

    htf_df = pd.DataFrame(buckets, columns=["timestamp", "open", "high", "low",
                                            "close", "volume"])
    htf_res = analyse(htf_df, params)
    ha = htf_res.df

    # ---- causal alignment --------------------------------------------------- #
    htf_close_times = ha["timestamp"].to_numpy(dtype="float64") + htf_ms
    ltf_close_times = df["timestamp"].to_numpy(dtype="float64") + ltf_ms
    idx = np.searchsorted(htf_close_times, ltf_close_times, side="right") - 1

    out = pd.DataFrame(index=df.index)
    for c in HTF_COLUMNS:
        out[c] = np.nan
    for c in HTF_BOOL_COLUMNS:
        out[c] = False

    ok = idx >= 0
    if not ok.any():
        warnings.append(
            f"No {htf} candle had closed by the end of this {ltf} series.")
        return _empty(df, ltf, htf, warnings)

    def take(col: str) -> np.ndarray:
        src = ha[col].to_numpy(dtype="float64") if col in ha.columns \
            else np.full(len(ha), np.nan)
        vals = np.full(len(df), np.nan)
        vals[ok] = src[idx[ok]]
        return vals

    out["htf_index"] = np.where(ok, idx, np.nan)
    out["htf_ts"] = take("timestamp")
    out["htf_closed_ts"] = out["htf_ts"] + htf_ms
    out["htf_bias"] = take("pa_bias")
    out["htf_pdz"] = take("pa_pdz")
    out["htf_range_top"] = take("pa_range_top")
    out["htf_range_bottom"] = take("pa_range_bottom")
    out["htf_range_mid"] = take("pa_range_mid")
    out["htf_close"] = take("close")
    out["htf_high"] = take("high")
    out["htf_low"] = take("low")
    out["htf_available"] = ok

    # ---- which HTF zones was price inside, as at each LTF close? ------------ #
    close = df["close"].to_numpy(dtype="float64")
    in_flags = {c: np.zeros(len(df), dtype=bool) for c in HTF_BOOL_COLUMNS
                if c.startswith("htf_in_")}
    zone_dir = np.zeros(len(df), dtype="float64")
    zone_strength = np.zeros(len(df), dtype="float64")

    # Only zones that were alive at that HTF candle's close can be used.
    for i in np.flatnonzero(ok):
        ts_htf = float(out["htf_ts"].iloc[i])
        price = float(close[i])
        if not np.isfinite(price):
            continue
        net_dir = 0.0
        best_strength = 0.0
        # zones_at() is the causal liveness test: it already excludes any zone
        # created after `ts` or killed (filled / invalidated / expired) at or
        # before it.  Do NOT also test z.status here - Zone objects are mutated
        # in place during the pass, so `status` is the zone's FINAL state and
        # filtering on it would leak the future into a past candle.
        for z in htf_res.zones_at(ts_htf + htf_ms - 1):
            if not z.contains(price):
                continue
            col = f"htf_in_{z.kind}"
            if col in in_flags:
                in_flags[col][i] = True
            net_dir += float(z.direction)
            best_strength = max(best_strength, float(z.strength or 0.0))
        zone_dir[i] = np.sign(net_dir)
        zone_strength[i] = best_strength

    for c, arr in in_flags.items():
        out[c] = arr
    out["htf_zone_dir"] = np.where(ok, zone_dir, np.nan)
    out["htf_zone_strength"] = np.where(ok, zone_strength, np.nan)

    return MtfResult(available=True, ltf_timeframe=ltf, htf_timeframe=htf,
                     htf_ms=htf_ms, df=out, htf_df=ha, htf_result=htf_res,
                     warnings=warnings)


def describe_mtf(res: MtfResult) -> List[str]:
    """Beginner-friendly lines describing the HTF context (used by the UI)."""
    if not res.available:
        return [f"No higher-timeframe context. {' '.join(res.warnings)}".strip()]
    info = res.latest()
    tf = res.htf_timeframe
    n_htf = 0 if res.htf_df is None else len(res.htf_df)
    lines = [f"Higher timeframe: **{tf}** ({n_htf} closed candles)"]
    bias = info["bias"]
    lines.append("HTF structure bias: " + {1: "bullish (higher highs holding)",
                                           -1: "bearish (lower lows holding)",
                                           0: "neutral / not yet established"}[bias])
    if info["range_top"] and info["range_bottom"]:
        pdz = info["pdz"]
        if pdz is None:
            # NaN means the HTF dealing range has been invalidated (or is not
            # established yet).  Claiming "equilibrium" here would be a lie.
            where = ("undefined - the HTF range has been invalidated or is not "
                     "established yet")
        else:
            where = {1: "discount half", -1: "premium half",
                     0: "equilibrium band"}[int(pdz)]
        lines.append(f"HTF dealing range {info['range_bottom']:,.2f} - {info['range_top']:,.2f} "
                     f"(mid {info['range_mid']:,.2f}); premium/discount is {where}.")
    if info["inside_zones"]:
        lines.append("Price is inside higher-timeframe "
                     + ", ".join(z.replace("_", " ") for z in info["inside_zones"]) + ".")
    else:
        lines.append("Price is not inside any live higher-timeframe zone.")
    for w in res.warnings:
        lines.append(f"Note: {w}")
    return lines


__all__ = ["MtfResult", "analyse_mtf", "resolve_htf", "infer_timeframe",
           "describe_mtf", "HTF_CHAIN", "HTF_COLUMNS", "HTF_BOOL_COLUMNS"]
