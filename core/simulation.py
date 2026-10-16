"""
core/simulation.py
==================
Pure, shared trade-simulation logic used by BOTH the live paper engine
(core/executor.py) and the historical backtester (core/backtest.py).

Keeping this in one place is deliberate: if the two engines implemented exits
separately they would drift apart, and the backtest would stop being a fair
preview of what the paper bot does.

Fill model
----------
Signals are produced from a CLOSED candle.  Two execution models are supported:

``next_open`` (default)
    The signal on candle *i* becomes a pending order that fills at the OPEN of
    candle *i+1*.  This is the realistic model and matches the backtest, because
    in live trading you observe the close and can only act afterwards.

``signal_close``
    Fills at the close of candle *i* itself.  Simpler and more immediate, but
    slightly optimistic: it assumes you transacted at a price you only learned
    at the instant the candle closed.

Slippage is always applied ADVERSELY (you buy higher and sell lower), so the
simulated results are biased pessimistic rather than flattering.

Intrabar ambiguity
------------------
If a single candle contains BOTH the stop-loss and the take-profit, the true
order of events is unknown from OHLC data.  We assume the STOP-LOSS executed
first.  This is the conservative convention required by the spec, and it makes
reported results a lower bound rather than an inflated estimate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

LONG = "long"
SHORT = "short"

EXIT_STOP_LOSS = "stop_loss"
EXIT_STOP_GAP = "stop_loss_gap"
EXIT_TAKE_PROFIT = "take_profit"
EXIT_TAKE_PROFIT_GAP = "take_profit_gap"
EXIT_SAFE_STOP = "safe_stop_flatten"
EXIT_END_OF_DATA = "end_of_data"

EXIT_REASONS = {
    EXIT_STOP_LOSS, EXIT_STOP_GAP, EXIT_TAKE_PROFIT,
    EXIT_TAKE_PROFIT_GAP, EXIT_SAFE_STOP, EXIT_END_OF_DATA,
}


@dataclass
class ExitDecision:
    """Result of testing one candle against an open position's levels."""

    exit_price: float
    reason: str
    hit_stop: bool
    hit_take_profit: bool
    gapped: bool
    note: str

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


# --------------------------------------------------------------------------- #
# Slippage
# --------------------------------------------------------------------------- #

def entry_fill_price(price: float, direction: str, slippage_rate: float) -> float:
    """Adverse slippage on entry: longs pay more, shorts receive less."""
    if slippage_rate <= 0:
        return float(price)
    if direction == LONG:
        return float(price) * (1.0 + slippage_rate)
    return float(price) * (1.0 - slippage_rate)


def exit_fill_price(price: float, direction: str, slippage_rate: float) -> float:
    """Adverse slippage on exit: longs receive less, shorts pay more."""
    if slippage_rate <= 0:
        return float(price)
    if direction == LONG:
        return float(price) * (1.0 - slippage_rate)
    return float(price) * (1.0 + slippage_rate)


# --------------------------------------------------------------------------- #
# Exit evaluation
# --------------------------------------------------------------------------- #

def check_exit(
    direction: str,
    entry_price: float,
    stop: float,
    take_profit: float,
    open_: float,
    high: float,
    low: float,
    close: float,
    slippage_rate: float = 0.0,
) -> Optional[ExitDecision]:
    """
    Test a CLOSED candle against an open position.

    Returns an ExitDecision when the stop or the target was touched, else None.

    Order of evaluation (documented and conservative):
      1. Gap check   - if the candle OPENED beyond a level, the fill is the open,
                       which is realistically worse (stop) or better (target).
      2. Both levels inside one candle - STOP-LOSS FIRST.
      3. Single level touched - fill at that level.

    `slippage_rate` is applied adversely to the raw fill price.
    """
    if direction not in (LONG, SHORT):
        raise ValueError(f"Unknown direction '{direction}'")
    for name, value in (("entry", entry_price), ("stop", stop), ("take_profit", take_profit),
                        ("open", open_), ("high", high), ("low", low), ("close", close)):
        if value is None or not math.isfinite(float(value)):
            raise ValueError(f"check_exit received a non-finite {name}: {value!r}")

    if direction == LONG:
        hit_stop = low <= stop
        hit_tp = high >= take_profit
        gap_stop = open_ <= stop
        gap_tp = open_ >= take_profit
    else:
        hit_stop = high >= stop
        hit_tp = low <= take_profit
        gap_stop = open_ >= stop
        gap_tp = open_ <= take_profit

    raw: Optional[float] = None
    reason = ""
    gapped = False
    note = ""

    if gap_stop:
        raw = open_
        reason = EXIT_STOP_GAP
        gapped = True
        note = (
            "The candle opened beyond the stop-loss, so the simulated fill is the "
            "open price, not the stop price. Real gaps can be worse than this."
        )
    elif gap_tp:
        raw = open_
        reason = EXIT_TAKE_PROFIT_GAP
        gapped = True
        note = "The candle opened beyond the take-profit; filled at the open."
    elif hit_stop and hit_tp:
        raw = stop
        reason = EXIT_STOP_LOSS
        note = (
            "Both the stop-loss and the take-profit fall inside this candle and the "
            "intrabar order is unknown. The STOP-LOSS is assumed to fill first "
            "(conservative convention)."
        )
    elif hit_stop:
        raw = stop
        reason = EXIT_STOP_LOSS
        note = "Price reached the stop-loss inside this candle."
    elif hit_tp:
        raw = take_profit
        reason = EXIT_TAKE_PROFIT
        note = "Price reached the take-profit inside this candle."

    if raw is None:
        return None

    fill = exit_fill_price(raw, direction, slippage_rate)
    return ExitDecision(
        exit_price=fill,
        reason=reason,
        hit_stop=bool(hit_stop or gap_stop),
        hit_take_profit=bool(hit_tp or gap_tp),
        gapped=gapped,
        note=note,
    )


def mark_to_market(direction: str, quantity: float, entry_price: float, mark_price: float) -> float:
    """Unrealised P/L before fees at `mark_price`."""
    if quantity <= 0:
        return 0.0
    if direction == LONG:
        return quantity * (mark_price - entry_price)
    return quantity * (entry_price - mark_price)


def unrealized_fees(quantity: float, entry_price: float, mark_price: float, fee_rate: float) -> float:
    """Fees already paid on entry plus the estimated fee to exit at `mark_price`."""
    return quantity * (entry_price + mark_price) * fee_rate


__all__ = [
    "LONG", "SHORT",
    "EXIT_STOP_LOSS", "EXIT_STOP_GAP", "EXIT_TAKE_PROFIT", "EXIT_TAKE_PROFIT_GAP",
    "EXIT_SAFE_STOP", "EXIT_END_OF_DATA", "EXIT_REASONS",
    "ExitDecision", "entry_fill_price", "exit_fill_price", "check_exit",
    "mark_to_market", "unrealized_fees",
]
