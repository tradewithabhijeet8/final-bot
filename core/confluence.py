"""
core/confluence.py
==================
Phase-2 (J): an explainable price-action confluence engine.

It turns the `pa_*` columns produced by core/price_action.py into a list of
named, weighted FACTORS, resolves which of them agree / disagree / are absent,
and reduces them to a single directional score.

ANTI-DOUBLE-COUNTING RULES (the important part)
-----------------------------------------------
1. Every factor has a unique id and belongs to a `group`.
2. Within a group only the STRONGEST present factor survives, using an explicit
   precedence list.  Examples:
       structure : mss  >  choch_early            (a confirmed shift supersedes
                                                    the early warning)
       liquidity : sweep_reclaim > sweep          (the reclaim contains the sweep)
       fvg       : retrace > formation
       ob        : retest  > formation
       sd        : retest  > formation
       breakout  : retest_hold > breakout, and a FAILED breakout is scored in
                   the OPPOSITE direction instead of the original one
3. Candlestick patterns are mutually reinforcing descriptions of ONE candle, so
   the pattern group contributes AT MOST one factor per direction (the
   highest-weighted shape present).  Three bullish shapes on one candle are one
   piece of evidence, not three.
4. The existing Phase-1 conditions can be passed in as `baseline` factors with
   weight 0: they are LISTED (so you can see whether the old and new layers
   agree) but they never change the confluence score.

The engine never trades by itself.  It produces an explanation and a score;
whether anything gates an entry is a separate, opt-in decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .config import PriceActionParams
from .multi_timeframe import HTF_BOOL_COLUMNS

# Group precedence: earlier entries win within the same group.
PRECEDENCE: Dict[str, List[str]] = {
    "structure": ["mss", "choch_early", "continuation", "failed_break"],
    "liquidity": ["sweep_reclaim", "sweep"],
    "fvg": ["retrace", "formation"],
    "order_block": ["retest", "formation"],
    "supply_demand": ["retest", "formation"],
    "breakout": ["retest_hold", "failed", "breakout"],
    "pattern": [],                     # handled by the one-per-direction rule
    "premium_discount": [],
    "multi_timeframe": [],             # bias / zone / premium-discount are
                                       # three DIFFERENT facts, not duplicates
    "baseline": [],                    # listed only, never scored
}

#: Concept groups the engine knows about; used to report "missing" evidence.
KNOWN_GROUPS = ["structure", "liquidity", "fvg", "order_block", "supply_demand",
                "pattern", "breakout", "premium_discount", "multi_timeframe"]


@dataclass
class Factor:
    fid: str
    group: str
    name: str
    direction: int            # +1 bullish, -1 bearish, 0 neutral/context
    weight: float
    detail: str
    baseline: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class ConfluenceReport:
    ts: Optional[float] = None
    bias: int = 0
    score: float = 0.0          # weight agreeing with `bias`
    opposing: float = 0.0       # weight disagreeing with `bias`
    net: float = 0.0            # score - opposing
    meets_threshold: bool = False  # score >= params.min_confluence_score
    agreeing: List[Factor] = field(default_factory=list)
    disagreeing: List[Factor] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)
    suppressed: List[str] = field(default_factory=list)
    factors: List[Factor] = field(default_factory=list)
    summary: str = ""

    @property
    def actionable_long(self) -> bool:
        """Bias + threshold.  Advisory only: nothing here opens a trade."""
        return self.bias == 1 and self.meets_threshold

    @property
    def actionable_short(self) -> bool:
        return self.bias == -1 and self.meets_threshold

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ts": self.ts, "bias": self.bias, "score": self.score,
            "meets_threshold": self.meets_threshold,
            "opposing": self.opposing, "net": self.net,
            "agreeing": [f.to_dict() for f in self.agreeing],
            "disagreeing": [f.to_dict() for f in self.disagreeing],
            "missing": list(self.missing),
            "suppressed": list(self.suppressed),
            "factors": [f.to_dict() for f in self.factors],
            "summary": self.summary,
        }


def _b(row: pd.Series, col: str) -> bool:
    v = row.get(col)
    if v is None:
        return False
    try:
        return bool(v)
    except (TypeError, ValueError):
        return False


def _f(row: pd.Series, col: str) -> Optional[float]:
    v = row.get(col)
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


# --------------------------------------------------------------------------- #
# Factor extraction
# --------------------------------------------------------------------------- #

def extract_factors(row: pd.Series, p: PriceActionParams,
                    baseline: Optional[Dict[str, bool]] = None) -> List[Factor]:
    """Build the raw (pre-dedup) factor list for one candle."""
    out: List[Factor] = []
    add = out.append

    # ---- structure ------------------------------------------------------ #
    # NOTE: pa_choch_bull_conf / pa_choch_bear_conf are deliberately NOT read
    # here.  The engine sets them on the very same candle as pa_mss_bull /
    # pa_mss_bear (a confirmed CHoCH *is* the market-structure shift), so
    # scoring both would count one event twice.
    if _b(row, "pa_mss_bull"):
        add(Factor("mss", "structure", "Market-structure shift (bullish)", +1,
                   p.w_structure, "Early CHoCH confirmed by a new higher pivot: structure has shifted up."))
    if _b(row, "pa_mss_bear"):
        add(Factor("mss", "structure", "Market-structure shift (bearish)", -1,
                   p.w_structure, "Early CHoCH confirmed by a new lower pivot: structure has shifted down."))
    if _b(row, "pa_choch_bull_early"):
        add(Factor("choch_early", "structure", "CHoCH early (bullish)", +1,
                   p.w_structure * 0.5, "First close above the last pivot high against a bearish bias (unconfirmed)."))
    if _b(row, "pa_choch_bear_early"):
        add(Factor("choch_early", "structure", "CHoCH early (bearish)", -1,
                   p.w_structure * 0.5, "First close below the last pivot low against a bullish bias (unconfirmed)."))
    if _b(row, "pa_cont_up"):
        add(Factor("continuation", "structure", "Structural continuation (up)", +1,
                   p.w_structure * 0.5, "A new higher high with no CHoCH against it: the up-leg continues."))
    if _b(row, "pa_cont_down"):
        add(Factor("continuation", "structure", "Structural continuation (down)", -1,
                   p.w_structure * 0.5, "A new lower low with no CHoCH against it: the down-leg continues."))
    if _b(row, "pa_failed_break_bull"):
        add(Factor("failed_break", "structure", "Failed bullish break", -1,
                   p.w_structure * 0.5, "A bullish BOS was closed back through: the break failed and now acts as resistance."))
    if _b(row, "pa_failed_break_bear"):
        add(Factor("failed_break", "structure", "Failed bearish break", +1,
                   p.w_structure * 0.5, "A bearish BOS was closed back through: the break failed and now acts as support."))

    # ---- liquidity ------------------------------------------------------ #
    if _b(row, "pa_sweep_reclaim_bull"):
        add(Factor("sweep_reclaim", "liquidity", "Buy-side sweep + reclaim", +1,
                   p.w_liquidity, "Price took out equal highs / prior-day high and closed back inside: classic stop-hunt rejection."))
    if _b(row, "pa_sweep_reclaim_bear"):
        add(Factor("sweep_reclaim", "liquidity", "Sell-side sweep + reclaim", -1,
                   p.w_liquidity, "Price took out equal lows / prior-day low and closed back inside: classic stop-hunt rejection."))
    if _b(row, "pa_sweep_bsl"):
        add(Factor("sweep", "liquidity", "Buy-side liquidity sweep", +1,
                   p.w_liquidity * 0.5, "A wick beyond resting buy-side liquidity with the close back inside."))
    if _b(row, "pa_sweep_ssl"):
        add(Factor("sweep", "liquidity", "Sell-side liquidity sweep", -1,
                   p.w_liquidity * 0.5, "A wick beyond resting sell-side liquidity with the close back inside."))

    # ---- fair value gaps -------------------------------------------------- #
    if _b(row, "pa_fvg_retrace_bull"):
        add(Factor("retrace", "fvg", "Retracement into bullish FVG", +1,
                   p.w_fvg, "Price pulled back into an active bullish fair value gap."))
    if _b(row, "pa_fvg_retrace_bear"):
        add(Factor("retrace", "fvg", "Retracement into bearish FVG", -1,
                   p.w_fvg, "Price pulled back into an active bearish fair value gap."))
    if _b(row, "pa_fvg_bull_form"):
        add(Factor("formation", "fvg", "Bullish FVG formed", +1,
                   p.w_fvg * 0.5, "A three-candle bullish imbalance opened on this candle."))
    if _b(row, "pa_fvg_bear_form"):
        add(Factor("formation", "fvg", "Bearish FVG formed", -1,
                   p.w_fvg * 0.5, "A three-candle bearish imbalance opened on this candle."))

    # ---- order blocks ---------------------------------------------------- #
    if _b(row, "pa_ob_retest_bull"):
        add(Factor("retest", "order_block", "Retest of bullish order block", +1,
                   p.w_order_block, "Price returned into an active bullish order block."))
    if _b(row, "pa_ob_retest_bear"):
        add(Factor("retest", "order_block", "Retest of bearish order block", -1,
                   p.w_order_block, "Price returned into an active bearish order block."))
    if _b(row, "pa_ob_bull_form"):
        add(Factor("formation", "order_block", "Bullish order block formed", +1,
                   p.w_order_block * 0.5, "Displacement up qualified the last down candle as a bullish OB."))
    if _b(row, "pa_ob_bear_form"):
        add(Factor("formation", "order_block", "Bearish order block formed", -1,
                   p.w_order_block * 0.5, "Displacement down qualified the last up candle as a bearish OB."))

    # ---- supply / demand -------------------------------------------------- #
    if _b(row, "pa_sd_retest_demand"):
        add(Factor("retest", "supply_demand", "Retest of demand zone", +1,
                   p.w_supply_demand, "Price returned into an active demand zone."))
    if _b(row, "pa_sd_retest_supply"):
        add(Factor("retest", "supply_demand", "Retest of supply zone", -1,
                   p.w_supply_demand, "Price returned into an active supply zone."))
    if _b(row, "pa_sd_demand_form"):
        add(Factor("formation", "supply_demand", "Demand zone formed", +1,
                   p.w_supply_demand * 0.5, "A tight base followed by displacement up created a demand zone."))
    if _b(row, "pa_sd_supply_form"):
        add(Factor("formation", "supply_demand", "Supply zone formed", -1,
                   p.w_supply_demand * 0.5, "A tight base followed by displacement down created a supply zone."))

    # ---- breakout / retest ------------------------------------------------ #
    if _b(row, "pa_retest_hold_bull"):
        add(Factor("retest_hold", "breakout", "Bullish breakout retest held", +1,
                   p.w_breakout, "Price retested the broken level and closed back above it: breakout confirmed."))
    if _b(row, "pa_retest_hold_bear"):
        add(Factor("retest_hold", "breakout", "Bearish breakout retest held", -1,
                   p.w_breakout, "Price retested the broken level and closed back below it: breakout confirmed."))
    if _b(row, "pa_breakout_failed_bull"):
        add(Factor("failed", "breakout", "Bullish breakout failed", -1,
                   p.w_breakout, "After the retest, price closed back below the level: the breakout failed."))
    if _b(row, "pa_breakout_failed_bear"):
        add(Factor("failed", "breakout", "Bearish breakout failed", +1,
                   p.w_breakout, "After the retest, price closed back above the level: the breakout failed."))
    if _b(row, "pa_breakout_bull"):
        add(Factor("breakout", "breakout", "Bullish structure breakout", +1,
                   p.w_breakout * 0.5, "A closed candle broke above the prior swing high."))
    if _b(row, "pa_breakout_bear"):
        add(Factor("breakout", "breakout", "Bearish structure breakout", -1,
                   p.w_breakout * 0.5, "A closed candle broke below the prior swing low."))

    # ---- candlestick patterns: ONE factor per direction ------------------- #
    # pa_hammer / pa_shooting_star are ALIASES of the pin bars (same shape), so
    # they sit after the primary names: the `break` below means one candle can
    # only ever contribute a single bullish and a single bearish shape.
    bull_shapes = [
        ("pa_engulf_bull", "Bullish engulfing", 1.0),
        ("pa_morning_star", "Morning star", 1.0),
        ("pa_pin_bull", "Bullish pin bar / hammer", 0.8),
        ("pa_hammer", "Hammer (bullish pin bar)", 0.8),
        ("pa_displacement", "Strong displacement candle", 0.6),
    ]
    bear_shapes = [
        ("pa_engulf_bear", "Bearish engulfing", 1.0),
        ("pa_evening_star", "Evening star", 1.0),
        ("pa_pin_bear", "Bearish pin bar / shooting star", 0.8),
        ("pa_shooting_star", "Shooting star (bearish pin bar)", 0.8),
    ]
    for col, name, mult in bull_shapes:
        if _b(row, col):
            add(Factor("pattern", "pattern", name, +1, p.w_pattern * mult,
                       f"{name} printed on this candle (contextual evidence only)."))
            break                       # one bullish shape is enough
    for col, name, mult in bear_shapes:
        if _b(row, col):
            add(Factor("pattern", "pattern", name, -1, p.w_pattern * mult,
                       f"{name} printed on this candle (contextual evidence only)."))
            break
    if _b(row, "pa_doji"):
        add(Factor("pattern_neutral", "pattern", "Doji", 0, 0.0,
                   "Doji: indecision. Listed as context; contributes no directional weight."))

    # ---- premium / discount ----------------------------------------------- #
    pdz = _f(row, "pa_pdz")
    if pdz is not None:
        if pdz > 0:
            add(Factor("premium_discount", "premium_discount", "Price in discount", +1,
                       p.w_premium_discount, "Close is in the discount half of the confirmed dealing range."))
        elif pdz < 0:
            add(Factor("premium_discount", "premium_discount", "Price in premium", -1,
                       p.w_premium_discount, "Close is in the premium half of the confirmed dealing range."))
        else:
            add(Factor("premium_discount", "premium_discount", "Price at equilibrium", 0, 0.0,
                       "Close sits in the equilibrium band: neither premium nor discount."))

    # ---- higher timeframe (only ever CLOSED HTF candles) ------------------- #
    if _b(row, "htf_available"):
        htf_bias = _f(row, "htf_bias")
        if htf_bias is not None and htf_bias != 0:
            direction = 1 if htf_bias > 0 else -1
            add(Factor("htf_bias", "multi_timeframe",
                       "Higher-timeframe structure bias", direction, p.w_htf_bias,
                       f"The last CLOSED higher-timeframe candle's structure is "
                       f"{'bullish' if direction > 0 else 'bearish'}. The in-progress "
                       f"higher-timeframe candle is excluded, so this cannot look ahead."))

        inside = [c[len("htf_in_"):].replace("_", " ")
                  for c in HTF_BOOL_COLUMNS if c.startswith("htf_in_") and _b(row, c)]
        if inside:
            zd = _f(row, "htf_zone_dir")
            direction = 0 if zd is None else int(zd)
            add(Factor("htf_zone", "multi_timeframe",
                       "Inside a higher-timeframe zone", direction,
                       p.w_htf_zone if direction != 0 else 0.0,
                       f"Price sits inside higher-timeframe {', '.join(inside)}."
                       + ("" if direction != 0 else
                          " Opposing higher-timeframe zones cancel out, so this is "
                          "reported as context with no directional weight.")))

        htf_pdz = _f(row, "htf_pdz")
        if htf_pdz is not None and htf_pdz != 0:
            add(Factor("htf_premium_discount", "multi_timeframe",
                       "Higher-timeframe premium/discount",
                       1 if htf_pdz > 0 else -1, p.w_htf_premium_discount,
                       "On the higher timeframe, price is in the "
                       + ("discount half" if htf_pdz > 0 else "premium half")
                       + " of its confirmed dealing range."))

    # ---- baseline (Phase-1) conditions: listed, never scored -------------- #
    if baseline:
        for name, passed in baseline.items():
            direction = 0
            if passed and name.endswith("_long"):
                direction = +1
            elif passed and name.endswith("_short"):
                direction = -1
            add(Factor(f"baseline_{name}", "baseline", f"Phase-1: {name}",
                       direction, 0.0,
                       "Existing four-condition check, shown for comparison only.",
                       baseline=True))
    return out


# --------------------------------------------------------------------------- #
# Dedup + scoring
# --------------------------------------------------------------------------- #

def _dedup(factors: List[Factor]) -> tuple:
    """Apply precedence within groups and the one-pattern-per-direction rule."""
    kept: List[Factor] = []
    suppressed: List[str] = []
    by_group: Dict[str, List[Factor]] = {}
    for f in factors:
        by_group.setdefault(f.group, []).append(f)

    for group, items in by_group.items():
        if group in ("pattern", "baseline", "premium_discount", "multi_timeframe"):
            kept.extend(items)
            continue
        order = PRECEDENCE.get(group, [])
        if not order:
            kept.extend(items)
            continue
        # highest-precedence fid present wins the group
        present = {f.fid for f in items}
        winner_fid = next((fid for fid in order if fid in present), None)
        for f in items:
            if f.fid == winner_fid:
                kept.append(f)
            else:
                suppressed.append(f"{group}:{f.fid} (superseded by {winner_fid})")
    return kept, suppressed


def build_confluence(row: pd.Series, p: PriceActionParams,
                     baseline: Optional[Dict[str, bool]] = None) -> ConfluenceReport:
    """Reduce one candle's price-action state to an explainable report."""
    raw = extract_factors(row, p, baseline)
    scored, suppressed = _dedup(raw)
    directional = [f for f in scored if f.direction != 0 and not f.baseline]

    pos = sum(f.weight for f in directional if f.direction == +1)
    neg = sum(f.weight for f in directional if f.direction == -1)

    if pos > neg:
        bias = 1
    elif neg > pos:
        bias = -1
    else:
        bias = 0

    report = ConfluenceReport(
        ts=_f(row, "timestamp"),
        meets_threshold=bool(bias != 0 and max(pos, neg) >= p.min_confluence_score),
        bias=bias,
        score=max(pos, neg),
        opposing=min(pos, neg),
        net=pos - neg,
        agreeing=[f for f in directional if f.direction == bias] if bias else [],
        disagreeing=[f for f in directional if f.direction == -bias] if bias else directional,
        suppressed=suppressed,
        factors=scored,
    )

    present_groups = {f.group for f in directional}
    report.missing = [g for g in KNOWN_GROUPS if g not in present_groups]

    if bias == 0:
        report.summary = (
            "No directional price-action confluence on this candle. "
            + (f"Context only: {', '.join(f.name for f in scored if f.direction == 0)}."
               if any(f.direction == 0 for f in scored) else "No evidence present.")
        )
    else:
        side = "BULLISH" if bias == 1 else "BEARISH"
        bar = ("" if report.meets_threshold else
               f" [below the {p.min_confluence_score:.1f} confluence threshold]")
        report.summary = (
            f"{side} confluence {report.score:.1f} vs {report.opposing:.1f} opposing "
            f"(net {report.net:+.1f}){bar}. Agreeing: "
            + ", ".join(f.name for f in report.agreeing)
            + (". Disagreeing: " + ", ".join(f.name for f in report.disagreeing)
               if report.disagreeing else "")
            + f". No evidence from: {', '.join(report.missing) if report.missing else 'nothing'}."
        )
    return report


def confluence_frame(result_df: pd.DataFrame, p: PriceActionParams) -> pd.DataFrame:
    """Convenience: a per-candle summary table for the UI / tests."""
    rows = []
    for _, row in result_df.iterrows():
        r = build_confluence(row, p)
        rows.append({"timestamp": row["timestamp"], "bias": r.bias, "score": r.score,
                     "opposing": r.opposing, "net": r.net, "summary": r.summary})
    return pd.DataFrame(rows)


__all__ = ["Factor", "ConfluenceReport", "extract_factors", "build_confluence",
           "confluence_frame", "PRECEDENCE", "KNOWN_GROUPS"]
