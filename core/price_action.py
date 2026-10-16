"""
core/price_action.py
====================
Phase-2 advanced price-action layer: CHoCH/MSS, liquidity, fair value gaps,
order blocks, supply & demand zones, candlestick patterns, breakout-retest
lifecycle, premium/discount and protected structure.

IMPORTANT FRAMING
-----------------
Every concept here is a PRICE-BASED INTERPRETATION of publicly visible candles.
Nothing in this module claims to observe hidden institutional order flow, and
nothing here is a guarantee of future direction.  Zones and levels are
descriptions of where price previously reacted, expressed as objective,
testable rules.

CAUSALITY
---------
The whole analysis is ONE forward pass over the candles.  At candle ``i`` the
engine may only use rows ``<= i``.  Fractal pivots are the only exception and
they are handled explicitly: a pivot at index ``i`` is not *visible* until
index ``i + pivot_lookback``, which is when its right-hand confirmation candles
have closed.  `tests/test_price_action.py` proves truncation-invariance.

REUSE, NOT DUPLICATION
----------------------
Swing levels, BOS and ATR come from core/indicators.py.  If the frame handed to
:func:`analyse` is already enriched by core/strategy.py (it has ``bos_bull`` /
``atr`` columns) those columns are reused; otherwise they are computed once with
the same functions.  No indicator is reimplemented here.

DOCUMENTED CONVENTIONS
----------------------
pivot high/low      fractal: high[i] is the max of high[i-k..i+k]; visible at i+k
protected high      a pivot high that price has since CLOSED above (it became a
                    defended level); protected low is the mirror image
CHoCH (early)       while the structure bias is bearish, the first CLOSE above
                    the most recent visible pivot high (and mirror).  This is an
                    *early* warning, not a confirmed reversal.
CHoCH confirmed/MSS an early CHoCH followed, within `choch_confirm_window`, by a
                    new pivot extreme in the same direction.  This is the
                    market-structure shift.
failed break        a BOS whose broken level is closed back through within
                    `fail_break_window` candles
continuation        a new pivot extreme in the direction of the current bias
                    with no CHoCH against it
equal highs/lows    two visible pivots within `eq_tolerance_bps`; the pool level
                    is the outer extreme (where resting stops sit)
sweep               a wick beyond a pool/level with the CLOSE back inside
sweep-and-reclaim   a sweep whose close is back inside by at least
                    `sweep_reclaim_frac` of that candle's range
bullish FVG         low[i] > high[i-2];  zone = [high[i-2], low[i]]
bearish FVG         high[i] < low[i-2];  zone = [high[i], low[i-2]]
FVG lifecycle       retest = wick enters; mitigated = wick reaches
                    `fvg_mitigation_depth`; filled = wick traverses the whole
                    zone; invalidated = a CLOSE beyond the far edge;
                    expired = no retest within `fvg_expiry_candles`
bullish order block the last DOWN candle immediately before a qualifying
                    displacement up (>= `ob_displacement_atr` ATR within
                    `ob_displacement_candles`); zone = that candle's range
OB invalidation     `close` mode: a CLOSE beyond the far edge; `wick` mode: any
                    wick beyond it
demand zone         a base of `sd_base_candles` tight candles followed by
                    displacement up; supply is the mirror.  A new zone that
                    overlaps an ACTIVE same-direction zone by more than
                    `sd_overlap_pct` is SUPPRESSED to avoid zone spam.
breakout            a close beyond the prior swing level (reuses BOS)
retest / hold / fail / expired
                    price returns within `retest_tol_bps` of the level; a close
                    back on the breakout side is a successful hold; a close back
                    on the original side is a failed breakout; no retest within
                    `breakout_expiry_candles` expires the setup
dealing range       last visible pivot low .. last visible pivot high
premium/discount    above/below the range midpoint, outside an equilibrium band
                    of `equilibrium_band_pct` % of the range height
range invalidation  a CLOSE beyond a range edge invalidates that range; a new
                    range forms as new pivots confirm
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .config import PriceActionParams
from .indicators import atr as compute_atr
from .indicators import detect_bos

DAY_MS = 86_400_000
WEEK_MS = 7 * DAY_MS

# Zone statuses
ST_ACTIVE = "active"
ST_RETESTED = "retested"
ST_MITIGATED = "mitigated"
ST_FILLED = "filled"
ST_INVALIDATED = "invalidated"
ST_EXPIRED = "expired"


# --------------------------------------------------------------------------- #
# Containers
# --------------------------------------------------------------------------- #

@dataclass
class Zone:
    """One tracked price zone (FVG / OB / S-D / liquidity pool / S-R level)."""

    zone_id: int
    kind: str
    direction: int                 # +1 = bullish/support, -1 = bearish/resistance
    top: float
    bottom: float
    created_ts: float
    confirmed_ts: Optional[float] = None
    status: str = ST_ACTIVE
    retests: int = 0
    strength: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def height(self) -> float:
        return max(0.0, self.top - self.bottom)

    def contains(self, price: float) -> bool:
        return self.bottom <= price <= self.top

    def overlap_pct(self, other: "Zone") -> float:
        """Percentage of THIS zone's height that overlaps `other`."""
        if self.height <= 0:
            return 0.0
        lo = max(self.bottom, other.bottom)
        hi = min(self.top, other.top)
        return max(0.0, hi - lo) / self.height * 100.0

    def to_dict(self) -> Dict[str, Any]:
        d = self.__dict__.copy()
        return d


@dataclass
class PriceActionResult:
    """Output of one causal analysis pass."""

    df: pd.DataFrame
    zones: List[Zone] = field(default_factory=list)
    zone_events: List[Dict[str, Any]] = field(default_factory=list)
    candle_events: List[Dict[str, Any]] = field(default_factory=list)
    pivots_high: List[Dict[str, Any]] = field(default_factory=list)
    pivots_low: List[Dict[str, Any]] = field(default_factory=list)
    params: Optional[PriceActionParams] = None

    def active_zones(self, kind: Optional[str] = None) -> List[Zone]:
        out = [z for z in self.zones if z.status in (ST_ACTIVE, ST_RETESTED, ST_MITIGATED)]
        return [z for z in out if kind is None or z.kind == kind]

    def zones_at(self, ts: float, kind: Optional[str] = None) -> List[Zone]:
        """Zones that were live at a given timestamp (causal lookup)."""
        out = []
        for z in self.zones:
            if z.created_ts > ts:
                continue
            dead = z.meta.get("dead_ts")
            if dead is not None and dead <= ts:
                continue
            if kind is None or z.kind == kind:
                out.append(z)
        return out


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _week_bucket(ts_ms: float) -> int:
    """Monday-aligned UTC week bucket (documented convention)."""
    days = int(ts_ms // DAY_MS)
    weekday = (days + 3) % 7          # 1970-01-01 was a Thursday -> weekday 3 (Mon=0)
    return (days - weekday) // 7


def _find_pivots(arr: np.ndarray, k: int, mode: str) -> List[int]:
    """Indices of fractal pivots.  A tie inside the window does not qualify."""
    out: List[int] = []
    n = arr.size
    if n < 2 * k + 1:
        return out
    for i in range(k, n - k):
        window = arr[i - k:i + k + 1]
        centre = arr[i]
        if mode == "high":
            if centre >= window.max() and np.sum(window == centre) == 1:
                out.append(i)
        else:
            if centre <= window.min() and np.sum(window == centre) == 1:
                out.append(i)
    return out


def _col(df: pd.DataFrame, name: str, default=np.nan) -> np.ndarray:
    if name in df.columns:
        return df[name].to_numpy(dtype="float64")
    return np.full(len(df), default, dtype="float64")


# --------------------------------------------------------------------------- #
# Vectorised candlestick patterns (F)
# --------------------------------------------------------------------------- #

def candle_patterns(df: pd.DataFrame, p: PriceActionParams, atr_arr: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Objective, ratio-based pattern detection.  Returns boolean arrays.

    These are CONTEXT-FREE shapes; whether they matter is decided by the
    confluence engine, never by the pattern alone.
    """
    o = df["open"].to_numpy(dtype="float64")
    h = df["high"].to_numpy(dtype="float64")
    l = df["low"].to_numpy(dtype="float64")
    c = df["close"].to_numpy(dtype="float64")
    n = c.size

    rng = np.where((h - l) > 0, h - l, np.nan)
    body = np.abs(c - o)
    upper = h - np.maximum(o, c)
    lower = np.minimum(o, c) - l
    body_frac = body / rng
    safe_atr = np.where(np.isnan(atr_arr) | (atr_arr <= 0), np.nan, atr_arr)

    bull = c > o
    bear = c < o

    pin_bull = (body_frac <= p.pattern_body_frac) & (lower / rng >= p.pattern_wick_frac) & (lower > 2 * np.maximum(body, 1e-12))
    pin_bear = (body_frac <= p.pattern_body_frac) & (upper / rng >= p.pattern_wick_frac) & (upper > 2 * np.maximum(body, 1e-12))

    engulf_bull = np.zeros(n, dtype=bool)
    engulf_bear = np.zeros(n, dtype=bool)
    if n > 1:
        prev_body_hi = np.maximum(o[:-1], c[:-1])
        prev_body_lo = np.minimum(o[:-1], c[:-1])
        cur_body_hi = np.maximum(o[1:], c[1:])
        cur_body_lo = np.minimum(o[1:], c[1:])
        engulf_bull[1:] = (c[1:] > o[1:]) & (o[1:] < c[:-1]) & (cur_body_hi > prev_body_hi) & (cur_body_lo < prev_body_lo) & (c[:-1] < o[:-1])
        engulf_bear[1:] = (c[1:] < o[1:]) & (o[1:] > c[:-1]) & (cur_body_hi > prev_body_hi) & (cur_body_lo < prev_body_lo) & (c[:-1] > o[:-1])

    doji = body_frac <= p.doji_frac

    inside = np.zeros(n, dtype=bool)
    outside = np.zeros(n, dtype=bool)
    if n > 1:
        inside[1:] = (h[1:] <= h[:-1]) & (l[1:] >= l[:-1])
        outside[1:] = (h[1:] >= h[:-1]) & (l[1:] <= l[:-1])

    morning = np.zeros(n, dtype=bool)
    evening = np.zeros(n, dtype=bool)
    if n > 2:
        a_b, b_b, c_b = body[:-2], body[1:-1], body[2:]
        a_rng, c_rng = rng[:-2], rng[2:]
        first_bear = c[:-2] < o[:-2]
        first_bull = c[:-2] > o[:-2]
        mid_small = b_b <= 0.35 * np.where(np.isnan(a_rng), np.nan, a_rng)
        third_bull = c[2:] > o[2:]
        third_bear = c[2:] < o[2:]
        third_strong = c_b >= 0.5 * np.where(np.isnan(a_rng), np.nan, a_rng)
        closes_above_mid = c[2:] > (np.maximum(o[:-2], c[:-2]) + np.minimum(o[:-2], c[:-2])) / 2
        closes_below_mid = c[2:] < (np.maximum(o[:-2], c[:-2]) + np.minimum(o[:-2], c[:-2])) / 2
        morning[2:] = first_bear & mid_small & third_bull & third_strong & closes_above_mid
        evening[2:] = first_bull & mid_small & third_bear & third_strong & closes_below_mid

    displacement = (rng / safe_atr >= p.displacement_atr) & (body_frac >= p.displacement_body_frac)

    # A candlestick pattern is a shape made BY TRADING.  On thin venues the
    # exchange emits placeholder buckets for periods with no trades at all; on
    # Delta's 5m series 65% of buckets were empty.  Without this mask every one
    # of them would be reported as an inside bar / doji, which is noise dressed
    # up as evidence.  Zero volume => no pattern, no exceptions.
    if "volume" in df.columns:
        vol = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0).to_numpy(dtype="float64")
        traded = vol > 0.0
    else:
        traded = np.ones(n, dtype=bool)

    patterns = {
        "pa_pin_bull": pin_bull, "pa_pin_bear": pin_bear,
        "pa_engulf_bull": engulf_bull, "pa_engulf_bear": engulf_bear,
        "pa_doji": doji, "pa_inside_bar": inside, "pa_outside_bar": outside,
        "pa_morning_star": morning, "pa_evening_star": evening,
        "pa_displacement": displacement,
        "pa_hammer": pin_bull,          # hammer == bullish pin bar by definition
        "pa_shooting_star": pin_bear,   # shooting star == bearish pin bar
    }
    return {k: (v & traded) for k, v in patterns.items()}


# --------------------------------------------------------------------------- #
# The main causal pass
# --------------------------------------------------------------------------- #

def analyse(df: pd.DataFrame, params: PriceActionParams) -> PriceActionResult:
    """
    Run the full price-action analysis over CLOSED candles.

    Returns a PriceActionResult whose `df` carries `pa_*` columns and whose
    `zones` / `candle_events` give the zone lifecycle and named events.
    """
    params.validate()
    if df is None or df.empty:
        return PriceActionResult(df=df if df is not None else pd.DataFrame(), params=params)

    work = df.copy().sort_values("timestamp", kind="mergesort").reset_index(drop=True)

    # Reuse existing enrichment where present (no duplicate indicators).
    if "atr" not in work.columns:
        work["atr"] = compute_atr(work["high"], work["low"], work["close"], 14)
    if "bos_bull" not in work.columns:
        work = detect_bos(work, lookback=params.structure_lookback, shift=1)

    n = len(work)
    ts = work["timestamp"].to_numpy(dtype="float64")
    o = work["open"].to_numpy(dtype="float64")
    h = work["high"].to_numpy(dtype="float64")
    l = work["low"].to_numpy(dtype="float64")
    c = work["close"].to_numpy(dtype="float64")
    atr_arr = work["atr"].to_numpy(dtype="float64")
    bos_bull = work["bos_bull"].to_numpy(dtype=bool)
    bos_bear = work["bos_bear"].to_numpy(dtype=bool)
    psh = work["prior_swing_high"].to_numpy(dtype="float64")
    psl = work["prior_swing_low"].to_numpy(dtype="float64")

    # ---- pattern columns (vectorised) ----------------------------------- #
    patterns = candle_patterns(work, params, atr_arr)
    for name, arr in patterns.items():
        work[name] = arr

    # ---- output column scaffolding -------------------------------------- #
    pa_cols = [
        "pa_pdh", "pa_pdl", "pa_pwh", "pa_pwl",
        "pa_eqh_level", "pa_eql_level",
        "pa_sweep_bsl", "pa_sweep_ssl", "pa_sweep_reclaim_bull", "pa_sweep_reclaim_bear",
        "pa_choch_bull_early", "pa_choch_bear_early",
        "pa_choch_bull_conf", "pa_choch_bear_conf", "pa_mss_bull", "pa_mss_bear",
        "pa_protected_high", "pa_protected_low", "pa_cont_up", "pa_cont_down",
        "pa_failed_break_bull", "pa_failed_break_bear",
        "pa_breakout_bull", "pa_breakout_bear",
        "pa_retest_bull", "pa_retest_bear", "pa_retest_hold_bull", "pa_retest_hold_bear",
        "pa_breakout_failed_bull", "pa_breakout_failed_bear",
        "pa_fvg_bull_form", "pa_fvg_bear_form", "pa_fvg_retrace_bull", "pa_fvg_retrace_bear",
        "pa_fvg_filled", "pa_fvg_invalidated",
        "pa_ob_bull_form", "pa_ob_bear_form", "pa_ob_retest_bull", "pa_ob_retest_bear",
        "pa_ob_invalidated",
        "pa_sd_demand_form", "pa_sd_supply_form", "pa_sd_retest_demand", "pa_sd_retest_supply",
        "pa_sd_invalidated",
        "pa_range_top", "pa_range_bottom", "pa_range_mid", "pa_pdz", "pa_range_invalidated",
        "pa_bias",          # running structural bias after this candle (+1/-1/0)
    ]
    # Build every output column in TWO insertions.  Assigning ~65 columns one at
    # a time fragments the block manager and makes pandas emit a
    # PerformanceWarning (and re-copy the frame repeatedly).
    bool_cols = [
        "pa_sweep_bsl", "pa_sweep_ssl", "pa_sweep_reclaim_bull", "pa_sweep_reclaim_bear",
        "pa_choch_bull_early", "pa_choch_bear_early", "pa_choch_bull_conf",
        "pa_choch_bear_conf", "pa_mss_bull", "pa_mss_bear", "pa_protected_high",
        "pa_protected_low", "pa_cont_up", "pa_cont_down", "pa_failed_break_bull",
        "pa_failed_break_bear", "pa_breakout_bull", "pa_breakout_bear",
        "pa_retest_bull", "pa_retest_bear", "pa_retest_hold_bull", "pa_retest_hold_bear",
        "pa_breakout_failed_bull", "pa_breakout_failed_bear", "pa_fvg_bull_form",
        "pa_fvg_bear_form", "pa_fvg_retrace_bull", "pa_fvg_retrace_bear",
        "pa_fvg_filled", "pa_fvg_invalidated", "pa_ob_bull_form", "pa_ob_bear_form",
        "pa_ob_retest_bull", "pa_ob_retest_bear", "pa_ob_invalidated",
        "pa_sd_demand_form", "pa_sd_supply_form", "pa_sd_retest_demand",
        "pa_sd_retest_supply", "pa_sd_invalidated", "pa_range_invalidated",
    ]
    float_cols = [c for c in pa_cols if c not in bool_cols]
    scaffold = pd.DataFrame(index=work.index)
    scaffold[float_cols] = np.nan
    scaffold[bool_cols] = False
    work = pd.concat([work, scaffold], axis=1)

    # ---- pivots (visible with a lag) ------------------------------------ #
    k = params.pivot_lookback
    ph_idx = _find_pivots(h, k, "high")
    pl_idx = _find_pivots(l, k, "low")
    pivots_high = [{"idx": i, "ts": ts[i], "price": h[i], "visible_at": i + k,
                    "protected": False, "protected_ts": None} for i in ph_idx]
    pivots_low = [{"idx": i, "ts": ts[i], "price": l[i], "visible_at": i + k,
                   "protected": False, "protected_ts": None} for i in pl_idx]
    ph_visible_at = {p_["idx"]: p_["visible_at"] for p_ in pivots_high}
    pl_visible_at = {p_["idx"]: p_["visible_at"] for p_ in pivots_low}

    # ---- previous day / week levels (causal) ---------------------------- #
    day_buckets = (ts // DAY_MS).astype("int64")
    week_buckets = np.array([_week_bucket(t) for t in ts], dtype="int64")
    pdh = pdl = pwh = pwl = np.nan
    cur_day = day_buckets[0]
    cur_week = week_buckets[0]
    day_hi = day_lo = c[0]
    week_hi = week_lo = c[0]
    prev_day_hi = prev_day_lo = np.nan
    prev_week_hi = prev_week_lo = np.nan

    # ---- zone bookkeeping ----------------------------------------------- #
    zones: List[Zone] = []
    zone_events: List[Dict[str, Any]] = []
    candle_events: List[Dict[str, Any]] = []
    next_zone_id = 1
    # A given source candle may only ever define ONE order block / S-D zone,
    # even though the displacement test can stay true for several candles.
    ob_src_used: set = set()
    sd_src_used: set = set()

    def add_zone(kind: str, direction: int, top: float, bottom: float,
                 created: float, confirmed: Optional[float] = None,
                 strength: float = 0.0, **meta) -> Zone:
        nonlocal next_zone_id
        # Call sites pass `confirmed_ts=`; without this it would silently land
        # in **meta and leave Zone.confirmed_ts empty.
        kw_confirmed = meta.pop("confirmed_ts", None)
        if confirmed is None:
            confirmed = kw_confirmed
        z = Zone(zone_id=next_zone_id, kind=kind, direction=direction, top=top,
                 bottom=bottom, created_ts=created, confirmed_ts=confirmed,
                 strength=strength, meta=meta)
        next_zone_id += 1
        zones.append(z)
        return z

    def mark(z: Zone, status: str, at: float, note: str = "") -> None:
        z.status = status
        z.meta["dead_ts"] = at
        zone_events.append({"ts": at, "zone_id": z.zone_id, "kind": z.kind,
                            "event": status, "note": note})

    def live(kind: Optional[str] = None) -> List[Zone]:
        out = [z for z in zones
               if z.status in (ST_ACTIVE, ST_RETESTED, ST_MITIGATED)
               and z.meta.get("dead_ts") is None]
        if kind is not None:
            out = [z for z in out if z.kind == kind]
        return out

    def retire_if_needed() -> None:
        act = live()
        if len(act) <= params.max_active_zones:
            return
        for z in sorted(act, key=lambda z: z.created_ts)[:len(act) - params.max_active_zones]:
            mark(z, ST_EXPIRED, ts_i, "active-zone cap reached; oldest zone retired")

    # ---- structure state ------------------------------------------------- #
    bias = 0                       # +1 up, -1 down (from BOS / MSS)
    last_ph: Optional[Dict[str, Any]] = None
    last_pl: Optional[Dict[str, Any]] = None
    choch_pending: Optional[Dict[str, Any]] = None     # {"dir":±1, "idx":i, "level":x}
    break_pending: Optional[Dict[str, Any]] = None     # failed-break watcher
    brk: Dict[str, Any] = {}                           # breakout-retest watcher

    ts_i = 0.0

    # ===================================================================== #
    # SINGLE CAUSAL FORWARD PASS
    # ===================================================================== #
    for i in range(n):
        ts_i = ts[i]
        a = atr_arr[i] if not np.isnan(atr_arr[i]) else np.nan

        # ---------- previous day / week roll ---------------------------- #
        # ORDER MATTERS: roll the day/week buckets FIRST, then publish the
        # previous-day/week levels for this candle, then fold this candle into
        # the new running aggregate.  Any other order either leaks the current
        # candle into "previous day" or publishes one day late.
        if day_buckets[i] != cur_day:
            prev_day_hi, prev_day_lo = day_hi, day_lo
            day_hi = day_lo = np.nan
            cur_day = day_buckets[i]
        if week_buckets[i] != cur_week:
            prev_week_hi, prev_week_lo = week_hi, week_lo
            week_hi = week_lo = np.nan
            cur_week = week_buckets[i]
        work.at[i, "pa_pdh"] = prev_day_hi
        work.at[i, "pa_pdl"] = prev_day_lo
        work.at[i, "pa_pwh"] = prev_week_hi
        work.at[i, "pa_pwl"] = prev_week_lo
        day_hi = h[i] if np.isnan(day_hi) else max(day_hi, h[i])
        day_lo = l[i] if np.isnan(day_lo) else min(day_lo, l[i])
        week_hi = h[i] if np.isnan(week_hi) else max(week_hi, h[i])
        week_lo = l[i] if np.isnan(week_lo) else min(week_lo, l[i])

        # ---------- pivots become visible ------------------------------- #
        for p_ in pivots_high:
            if p_["visible_at"] == i:
                last_ph = p_
        for p_ in pivots_low:
            if p_["visible_at"] == i:
                last_pl = p_

        # ---------- protected highs / lows (I) -------------------------- #
        for p_ in pivots_high:
            if not p_["protected"] and p_["visible_at"] <= i and c[i] > p_["price"]:
                p_["protected"] = True
                p_["protected_ts"] = ts_i
                work.at[i, "pa_protected_high"] = True
        for p_ in pivots_low:
            if not p_["protected"] and p_["visible_at"] <= i and c[i] < p_["price"]:
                p_["protected"] = True
                p_["protected_ts"] = ts_i
                work.at[i, "pa_protected_low"] = True

        # ---------- equal highs / lows -> liquidity pools (B) ----------- #
        tol = params.eq_tolerance_bps / 10_000.0
        eqh_level = eql_level = np.nan
        vis_h = [p_ for p_ in pivots_high if p_["visible_at"] <= i and not p_.get("swept")]
        vis_l = [p_ for p_ in pivots_low if p_["visible_at"] <= i and not p_.get("swept")]
        if len(vis_h) >= 2:
            recent = vis_h[-6:]
            for x in range(len(recent)):
                for y in range(x + 1, len(recent)):
                    hi_a, hi_b = recent[x]["price"], recent[y]["price"]
                    if abs(hi_a - hi_b) <= max(hi_a, hi_b) * tol:
                        eqh_level = max(hi_a, hi_b)
        if len(vis_l) >= 2:
            recent = vis_l[-6:]
            for x in range(len(recent)):
                for y in range(x + 1, len(recent)):
                    lo_a, lo_b = recent[x]["price"], recent[y]["price"]
                    if abs(lo_a - lo_b) <= max(lo_a, lo_b) * tol:
                        eql_level = min(lo_a, lo_b)
        work.at[i, "pa_eqh_level"] = eqh_level
        work.at[i, "pa_eql_level"] = eql_level

        # liquidity pool zones (created once per level)
        if not np.isnan(eqh_level):
            if not any(z.kind == "liq_bsl" and z.status == ST_ACTIVE and
                       abs(z.top - eqh_level) <= eqh_level * tol for z in zones):
                add_zone("liq_bsl", -1, eqh_level, eqh_level, ts_i, confirmed_ts=ts_i,
                         note="equal highs: buy-side liquidity above")
        if not np.isnan(eql_level):
            if not any(z.kind == "liq_ssl" and z.status == ST_ACTIVE and
                       abs(z.bottom - eql_level) <= eql_level * tol for z in zones):
                add_zone("liq_ssl", +1, eql_level, eql_level, ts_i, confirmed_ts=ts_i,
                         note="equal lows: sell-side liquidity below")

        # ---------- sweeps & reclaim (B) -------------------------------- #
        swept_bsl = swept_ssl = False
        for z in live("liq_bsl"):
            if h[i] > z.top and c[i] < z.top:
                swept_bsl = True
                work.at[i, "pa_sweep_bsl"] = True
                rng_i = (h[i] - l[i]) or np.nan
                if not np.isnan(rng_i) and (z.top - c[i]) >= params.sweep_reclaim_frac * rng_i:
                    work.at[i, "pa_sweep_reclaim_bull"] = True
                    candle_events.append({"ts": ts_i, "event": "sweep_and_reclaim_bull",
                                          "level": z.top,
                                          "note": "buy-side liquidity swept then reclaimed"})
                mark(z, ST_FILLED, ts_i, "pool swept")
                for p_ in pivots_high:
                    if abs(p_["price"] - z.top) <= z.top * tol:
                        p_["swept"] = True
        for z in live("liq_ssl"):
            if l[i] < z.bottom and c[i] > z.bottom:
                swept_ssl = True
                work.at[i, "pa_sweep_ssl"] = True
                rng_i = (h[i] - l[i]) or np.nan
                if not np.isnan(rng_i) and (c[i] - z.bottom) >= params.sweep_reclaim_frac * rng_i:
                    work.at[i, "pa_sweep_reclaim_bear"] = True
                    candle_events.append({"ts": ts_i, "event": "sweep_and_reclaim_bear",
                                          "level": z.bottom,
                                          "note": "sell-side liquidity swept then reclaimed"})
                mark(z, ST_FILLED, ts_i, "pool swept")
                for p_ in pivots_low:
                    if abs(p_["price"] - z.bottom) <= z.bottom * tol:
                        p_["swept"] = True
        # previous-day levels also act as liquidity
        if not np.isnan(prev_day_hi) and h[i] > prev_day_hi and c[i] < prev_day_hi:
            work.at[i, "pa_sweep_bsl"] = True
        if not np.isnan(prev_day_lo) and l[i] < prev_day_lo and c[i] > prev_day_lo:
            work.at[i, "pa_sweep_ssl"] = True

        # ---------- BOS-driven structure state (I) ---------------------- #
        work.at[i, "pa_breakout_bull"] = bool(bos_bull[i])
        work.at[i, "pa_breakout_bear"] = bool(bos_bear[i])

        # ---------- breakout / retest lifecycle (G) --------------------- #
        if brk:
            lvl = brk["level"]
            tol_abs = lvl * params.retest_tol_bps / 10_000.0
            age = i - brk["idx"]
            if brk["state"] == "awaiting_retest" and age > 0:
                touched = (l[i] <= lvl + tol_abs) if brk["dir"] == +1 else (h[i] >= lvl - tol_abs)
                if touched:
                    brk["state"] = "retested"
                    if brk["dir"] == +1:
                        work.at[i, "pa_retest_bull"] = True
                    else:
                        work.at[i, "pa_retest_bear"] = True
                    candle_events.append({"ts": ts_i, "event": f"retest_{'bull' if brk['dir']==1 else 'bear'}",
                                          "level": lvl})
            elif brk["state"] == "retested":
                if brk["dir"] == +1:
                    if c[i] > lvl:
                        brk["state"] = "done"
                        work.at[i, "pa_retest_hold_bull"] = True
                        candle_events.append({"ts": ts_i, "event": "retest_hold_bull", "level": lvl})
                    elif c[i] < lvl - tol_abs:
                        brk["state"] = "done"
                        work.at[i, "pa_breakout_failed_bull"] = True
                        candle_events.append({"ts": ts_i, "event": "breakout_failed_bull", "level": lvl})
                else:
                    if c[i] < lvl:
                        brk["state"] = "done"
                        work.at[i, "pa_retest_hold_bear"] = True
                        candle_events.append({"ts": ts_i, "event": "retest_hold_bear", "level": lvl})
                    elif c[i] > lvl + tol_abs:
                        brk["state"] = "done"
                        work.at[i, "pa_breakout_failed_bear"] = True
                        candle_events.append({"ts": ts_i, "event": "breakout_failed_bear", "level": lvl})
            if brk and brk["state"] == "awaiting_retest" and age > params.breakout_expiry_candles:
                brk["state"] = "expired"
                candle_events.append({"ts": ts_i, "event": "breakout_expired", "level": lvl})
                brk = {}
            if brk and brk["state"] == "done":
                brk = {}


        # A swing level stays "broken" for several candles while the swing
        # window catches up, so BOS can fire repeatedly at the SAME level.
        # Only a genuinely NEW level starts a fresh breakout/retest watcher;
        # otherwise the watcher would be reset forever and no retest could
        # ever be recognised.
        if bos_bull[i] and brk.get("dir") == -1 and brk.get("state") in ("awaiting_retest", "retested"):
            work.at[i, "pa_breakout_failed_bear"] = True
            candle_events.append({"ts": ts_i, "event": "breakout_failed_bear",
                                  "level": brk.get("level"),
                                  "note": "opposite structural break closed the setup"})
            brk = {}
        if bos_bear[i] and brk.get("dir") == +1 and brk.get("state") in ("awaiting_retest", "retested"):
            work.at[i, "pa_breakout_failed_bull"] = True
            candle_events.append({"ts": ts_i, "event": "breakout_failed_bull",
                                  "level": brk.get("level"),
                                  "note": "opposite structural break closed the setup"})
            brk = {}

        if bos_bull[i]:
            bias = +1
            lvl_new = psh[i]
            break_pending = {"dir": +1, "idx": i, "level": lvl_new}
            same = (brk.get("dir") == +1 and np.isfinite(brk.get("level", np.nan))
                    and abs(brk["level"] - lvl_new) <= max(lvl_new, 1e-9) * 1e-9)
            if not same:
                brk = {"dir": +1, "idx": i, "level": lvl_new, "state": "awaiting_retest"}
                candle_events.append({"ts": ts_i, "event": "breakout_bull", "level": lvl_new})
        if bos_bear[i]:
            bias = -1
            lvl_new = psl[i]
            break_pending = {"dir": -1, "idx": i, "level": lvl_new}
            same = (brk.get("dir") == -1 and np.isfinite(brk.get("level", np.nan))
                    and abs(brk["level"] - lvl_new) <= max(lvl_new, 1e-9) * 1e-9)
            if not same:
                brk = {"dir": -1, "idx": i, "level": lvl_new, "state": "awaiting_retest"}
                candle_events.append({"ts": ts_i, "event": "breakout_bear", "level": lvl_new})

        # failed structure break (I)
        if break_pending is not None:
            age = i - break_pending["idx"]
            lvl = break_pending["level"]
            if 0 < age <= params.fail_break_window and not np.isnan(lvl):
                if break_pending["dir"] == +1 and c[i] < lvl:
                    work.at[i, "pa_failed_break_bull"] = True
                    candle_events.append({"ts": ts_i, "event": "failed_break_bull", "level": lvl})
                    bias = -1
                    break_pending = None
                elif break_pending["dir"] == -1 and c[i] > lvl:
                    work.at[i, "pa_failed_break_bear"] = True
                    candle_events.append({"ts": ts_i, "event": "failed_break_bear", "level": lvl})
                    bias = +1
                    break_pending = None
            elif age > params.fail_break_window:
                break_pending = None

        # CHoCH early / confirmed / MSS (A + I)
        if bias == -1 and last_ph is not None and c[i] > last_ph["price"] and not bos_bull[i]:
            work.at[i, "pa_choch_bull_early"] = True
            choch_pending = {"dir": +1, "idx": i, "level": last_ph["price"], "pivot_idx": last_ph["idx"]}
            candle_events.append({"ts": ts_i, "event": "choch_bull_early", "level": last_ph["price"]})
        if bias == +1 and last_pl is not None and c[i] < last_pl["price"] and not bos_bear[i]:
            work.at[i, "pa_choch_bear_early"] = True
            choch_pending = {"dir": -1, "idx": i, "level": last_pl["price"], "pivot_idx": last_pl["idx"]}
            candle_events.append({"ts": ts_i, "event": "choch_bear_early", "level": last_pl["price"]})

        if choch_pending is not None:
            age = i - choch_pending["idx"]
            if 0 < age <= params.choch_confirm_window:
                confirmed = False
                if choch_pending["dir"] == +1:
                    for p_ in pivots_high:
                        if p_["visible_at"] == i and p_["idx"] > choch_pending["pivot_idx"] \
                                and p_["price"] > choch_pending["level"]:
                            confirmed = True
                else:
                    for p_ in pivots_low:
                        if p_["visible_at"] == i and p_["idx"] > choch_pending["pivot_idx"] \
                                and p_["price"] < choch_pending["level"]:
                            confirmed = True
                if confirmed:
                    if choch_pending["dir"] == +1:
                        work.at[i, "pa_choch_bull_conf"] = True
                        work.at[i, "pa_mss_bull"] = True
                        bias = +1
                        candle_events.append({"ts": ts_i, "event": "mss_bull",
                                              "level": choch_pending["level"]})
                    else:
                        work.at[i, "pa_choch_bear_conf"] = True
                        work.at[i, "pa_mss_bear"] = True
                        bias = -1
                        candle_events.append({"ts": ts_i, "event": "mss_bear",
                                              "level": choch_pending["level"]})
                    choch_pending = None
            elif age > params.choch_confirm_window:
                choch_pending = None

        # structural continuation (I)
        if last_ph is not None and last_ph["visible_at"] == i and bias == +1:
            prevs = [q for q in pivots_high if q["visible_at"] < i]
            if prevs and last_ph["price"] > prevs[-1]["price"] and choch_pending is None:
                work.at[i, "pa_cont_up"] = True
                candle_events.append({"ts": ts_i, "event": "continuation_up",
                                      "level": last_ph["price"]})
        if last_pl is not None and last_pl["visible_at"] == i and bias == -1:
            prevs = [q for q in pivots_low if q["visible_at"] < i]
            if prevs and last_pl["price"] < prevs[-1]["price"] and choch_pending is None:
                work.at[i, "pa_cont_down"] = True
                candle_events.append({"ts": ts_i, "event": "continuation_down",
                                      "level": last_pl["price"]})

        # ---------- fair value gaps (C) ---------------------------------- #
        if i >= 2 and not np.isnan(a):
            gap_bull = l[i] - h[i - 2]
            gap_bear = l[i - 2] - h[i]
            min_h = max(params.fvg_min_bps / 10_000.0 * c[i],
                        params.fvg_min_atr * a if params.fvg_use_atr_filter else 0.0)
            if gap_bull > 0 and gap_bull >= min_h:
                work.at[i, "pa_fvg_bull_form"] = True
                add_zone("fvg_bull", +1, l[i], h[i - 2], ts_i, confirmed_ts=ts[i - 2],
                         strength=gap_bull / a if a > 0 else 0.0, form_idx=i)
                candle_events.append({"ts": ts_i, "event": "fvg_bull_form",
                                      "top": l[i], "bottom": h[i - 2]})
            if gap_bear > 0 and gap_bear >= min_h:
                work.at[i, "pa_fvg_bear_form"] = True
                add_zone("fvg_bear", -1, l[i - 2], h[i], ts_i, confirmed_ts=ts[i - 2],
                         strength=gap_bear / a if a > 0 else 0.0, form_idx=i)
                candle_events.append({"ts": ts_i, "event": "fvg_bear_form",
                                      "top": l[i - 2], "bottom": h[i]})

        # ---------- order blocks (D) ------------------------------------ #
        if i >= params.ob_displacement_candles and not np.isnan(a):
            win = slice(i - params.ob_displacement_candles + 1, i + 1)
            move_up = c[i] - c[win.start - 1] if win.start - 1 >= 0 else np.nan
            move_dn = c[win.start - 1] - c[i] if win.start - 1 >= 0 else np.nan
            disp_up = not np.isnan(move_up) and move_up >= params.ob_displacement_atr * a
            disp_dn = not np.isnan(move_dn) and move_dn >= params.ob_displacement_atr * a
            if disp_up:
                j = i
                while j > win.start - 1 and not (c[j] < o[j]):
                    j -= 1
                if j > win.start - 1 and c[j] < o[j] and ("ob_bull", j) not in ob_src_used:
                    ob_src_used.add(("ob_bull", j))
                    work.at[i, "pa_ob_bull_form"] = True
                    add_zone("ob_bull", +1, h[j], l[j], ts_i, confirmed_ts=ts_i,
                             strength=move_up / a if a > 0 else 0.0, src_idx=j)
                    candle_events.append({"ts": ts_i, "event": "ob_bull_form",
                                          "top": h[j], "bottom": l[j]})
            if disp_dn:
                j = i
                while j > win.start - 1 and not (c[j] > o[j]):
                    j -= 1
                if j > win.start - 1 and c[j] > o[j] and ("ob_bear", j) not in ob_src_used:
                    ob_src_used.add(("ob_bear", j))
                    work.at[i, "pa_ob_bear_form"] = True
                    add_zone("ob_bear", -1, h[j], l[j], ts_i, confirmed_ts=ts_i,
                             strength=move_dn / a if a > 0 else 0.0, src_idx=j)
                    candle_events.append({"ts": ts_i, "event": "ob_bear_form",
                                          "top": h[j], "bottom": l[j]})

        # ---------- supply / demand zones (E) --------------------------- #
        bc = params.sd_base_candles
        if i >= bc + params.sd_displacement_candles and not np.isnan(a):
            base = slice(i - bc - params.sd_displacement_candles + 1,
                         i - params.sd_displacement_candles + 1)
            base_rows = range(base.start, base.stop)
            base_range = float(np.sum(h[list(base_rows)] - l[list(base_rows)]))
            tight = base_range <= params.sd_base_range_atr * a * bc
            if tight:
                win = slice(i - params.sd_displacement_candles + 1, i + 1)
                start_ref = base.stop - 1
                move_up = c[i] - c[start_ref]
                move_dn = c[start_ref] - c[i]
                b_top = float(np.max(h[list(base_rows)]))
                b_bot = float(np.min(l[list(base_rows)]))
                cand = None
                if move_up >= params.sd_displacement_atr * a:
                    cand = ("sd_demand", +1, b_top, b_bot, move_up)
                elif move_dn >= params.sd_displacement_atr * a:
                    cand = ("sd_supply", -1, b_top, b_bot, move_dn)
                if cand:
                    kind, dirn, top, bot, mv = cand
                    overlap = False
                    for z in live(kind):
                        probe = Zone(zone_id=-1, kind=kind, direction=dirn, top=top,
                                     bottom=bot, created_ts=ts_i)
                        if probe.overlap_pct(z) > params.sd_overlap_pct or z.overlap_pct(probe) > params.sd_overlap_pct:
                            overlap = True
                            break
                    if not overlap and (kind, base.start) not in sd_src_used:
                        sd_src_used.add((kind, base.start))
                        work.at[i, "pa_sd_demand_form" if dirn == 1 else "pa_sd_supply_form"] = True
                        add_zone(kind, dirn, top, bot, ts_i, confirmed_ts=ts_i,
                                 strength=mv / a if a > 0 else 0.0,
                                 base_candles=bc, src_idx=base.start)
                        candle_events.append({"ts": ts_i, "event": kind + "_form",
                                              "top": top, "bottom": bot})

        # ---------- zone lifecycle updates ------------------------------ #
        for z in list(live()):
            if z.created_ts >= ts_i:
                continue
            kind = z.kind
            if kind == "fvg_bull":
                if l[i] <= z.top:
                    z.retests += 1
                    depth = (z.top - l[i]) / z.height if z.height > 0 else 1.0
                    work.at[i, "pa_fvg_retrace_bull"] = True
                    if c[i] < z.bottom:
                        mark(z, ST_INVALIDATED, ts_i, "closed through the far edge")
                        work.at[i, "pa_fvg_invalidated"] = True
                    elif l[i] <= z.bottom:
                        mark(z, ST_FILLED, ts_i, "fully filled")
                        work.at[i, "pa_fvg_filled"] = True
                    elif depth >= params.fvg_mitigation_depth:
                        if z.status != ST_MITIGATED:
                            z.status = ST_MITIGATED
                            zone_events.append({"ts": ts_i, "zone_id": z.zone_id,
                                                "kind": z.kind, "event": ST_MITIGATED,
                                                "note": f"mitigated {depth:.0%}"})
                    else:
                        z.status = ST_RETESTED
                elif z.status in (ST_ACTIVE, ST_RETESTED) and i - z.meta.get("form_idx", i) > params.fvg_expiry_candles:
                    mark(z, ST_EXPIRED, ts_i, "no revisit within expiry window")
            elif kind == "fvg_bear":
                if h[i] >= z.bottom:
                    z.retests += 1
                    depth = (h[i] - z.bottom) / z.height if z.height > 0 else 1.0
                    work.at[i, "pa_fvg_retrace_bear"] = True
                    if c[i] > z.top:
                        mark(z, ST_INVALIDATED, ts_i, "closed through the far edge")
                        work.at[i, "pa_fvg_invalidated"] = True
                    elif h[i] >= z.top:
                        mark(z, ST_FILLED, ts_i, "fully filled")
                        work.at[i, "pa_fvg_filled"] = True
                    elif depth >= params.fvg_mitigation_depth:
                        if z.status != ST_MITIGATED:
                            z.status = ST_MITIGATED
                            zone_events.append({"ts": ts_i, "zone_id": z.zone_id,
                                                "kind": z.kind, "event": ST_MITIGATED,
                                                "note": f"mitigated {depth:.0%}"})
                    else:
                        z.status = ST_RETESTED
                elif z.status in (ST_ACTIVE, ST_RETESTED) and i - z.meta.get("form_idx", i) > params.fvg_expiry_candles:
                    mark(z, ST_EXPIRED, ts_i, "no revisit within expiry window")
            elif kind in ("ob_bull", "sd_demand"):
                touch = l[i] <= z.top
                if touch:
                    z.retests += 1
                    col = "pa_ob_retest_bull" if kind == "ob_bull" else "pa_sd_retest_demand"
                    work.at[i, col] = True
                    inval = (c[i] < z.bottom) if params.ob_invalidation == "close" else (l[i] < z.bottom)
                    if inval:
                        mark(z, ST_INVALIDATED, ts_i, f"{params.ob_invalidation} beyond far edge")
                        if kind == "ob_bull":
                            work.at[i, "pa_ob_invalidated"] = True
                        else:
                            work.at[i, "pa_sd_invalidated"] = True
                    elif z.status == ST_ACTIVE:
                        z.status = ST_RETESTED
                elif z.status in (ST_ACTIVE, ST_RETESTED):
                    age_c = i - int(z.meta.get("src_idx", 0) or 0)
                    limit = params.ob_expiry_candles if kind == "ob_bull" else params.sd_expiry_candles
                    if age_c > limit:
                        mark(z, ST_EXPIRED, ts_i, "expiry window elapsed")
            elif kind in ("ob_bear", "sd_supply"):
                touch = h[i] >= z.bottom
                if touch:
                    z.retests += 1
                    col = "pa_ob_retest_bear" if kind == "ob_bear" else "pa_sd_retest_supply"
                    work.at[i, col] = True
                    inval = (c[i] > z.top) if params.ob_invalidation == "close" else (h[i] > z.top)
                    if inval:
                        mark(z, ST_INVALIDATED, ts_i, f"{params.ob_invalidation} beyond far edge")
                        if kind == "ob_bear":
                            work.at[i, "pa_ob_invalidated"] = True
                        else:
                            work.at[i, "pa_sd_invalidated"] = True
                    elif z.status == ST_ACTIVE:
                        z.status = ST_RETESTED
                elif z.status in (ST_ACTIVE, ST_RETESTED):
                    age_c = i - int(z.meta.get("src_idx", 0) or 0)
                    limit = params.ob_expiry_candles if kind == "ob_bear" else params.sd_expiry_candles
                    if age_c > limit:
                        mark(z, ST_EXPIRED, ts_i, "expiry window elapsed")

        retire_if_needed()

        # ---------- premium / discount dealing range (H) ---------------- #
        if last_ph is not None and last_pl is not None:
            top_r = max(last_ph["price"], last_pl["price"])
            bot_r = min(last_ph["price"], last_pl["price"])
            mid = (top_r + bot_r) / 2.0
            height = top_r - bot_r
            work.at[i, "pa_range_top"] = top_r
            work.at[i, "pa_range_bottom"] = bot_r
            work.at[i, "pa_range_mid"] = mid
            if height > 0:
                # A CLOSE beyond either edge invalidates the range: from that
                # candle on, premium/discount is undefined until new pivots
                # confirm a fresh range.  Reporting a stale zone would be
                # misleading, so pa_pdz is left NaN on invalidated candles.
                if c[i] > top_r or c[i] < bot_r:
                    work.at[i, "pa_range_invalidated"] = True
                    candle_events.append({"ts": ts_i, "event": "dealing_range_invalidated",
                                          "top": top_r, "bottom": bot_r})
                else:
                    band = height * params.equilibrium_band_pct / 100.0
                    if c[i] > mid + band:
                        work.at[i, "pa_pdz"] = -1.0      # premium
                    elif c[i] < mid - band:
                        work.at[i, "pa_pdz"] = +1.0      # discount
                    else:
                        work.at[i, "pa_pdz"] = 0.0       # equilibrium

        # Running structural bias AFTER every update for this candle.  Exposed
        # so the multi-timeframe layer can read a higher timeframe's bias
        # without re-deriving it, and so tests can assert bias transitions.
        work.at[i, "pa_bias"] = bias

    # ---- assemble -------------------------------------------------------- #
    work["pa_pattern_names"] = [
        ";".join(name[3:] for name, arr in patterns.items()
                 if name not in ("pa_hammer", "pa_shooting_star") and bool(arr[i]))
        for i in range(n)
    ]
    return PriceActionResult(df=work, zones=zones, zone_events=zone_events,
                             candle_events=candle_events, pivots_high=pivots_high,
                             pivots_low=pivots_low, params=params)


# --------------------------------------------------------------------------- #
# Convenience lookups for the confluence engine and the UI
# --------------------------------------------------------------------------- #

def events_at(result: PriceActionResult, ts: float) -> List[Dict[str, Any]]:
    return [e for e in result.candle_events if e["ts"] == ts]


def summarise(result: PriceActionResult) -> Dict[str, Any]:
    """Compact overview for the diagnostics / price-action UI panel."""
    by_status: Dict[str, int] = {}
    for z in result.zones:
        by_status[z.status] = by_status.get(z.status, 0) + 1
    by_kind: Dict[str, int] = {}
    for z in result.zones:
        by_kind[z.kind] = by_kind.get(z.kind, 0) + 1
    return {
        "zones_total": len(result.zones),
        "zones_by_status": by_status,
        "zones_by_kind": by_kind,
        "events": len(result.candle_events),
        "pivots_high": len(result.pivots_high),
        "pivots_low": len(result.pivots_low),
    }


__all__ = [
    "Zone", "PriceActionResult", "analyse", "candle_patterns", "events_at",
    "summarise", "ST_ACTIVE", "ST_RETESTED", "ST_MITIGATED", "ST_FILLED",
    "ST_INVALIDATED", "ST_EXPIRED",
]
