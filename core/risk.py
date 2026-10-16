"""
core/risk.py
============
Position sizing, stop-loss / take-profit placement and loss limits.

Position-sizing formula (documented, deterministic, testable)
-------------------------------------------------------------
    equity            = realised balance + unrealised P/L of open positions
    risk_amount       = equity * risk_per_trade_pct / 100
    stop_distance     = |entry - stop|
    fee_entry_per_unit = fee_rate * entry
    fee_exit_per_unit  = fee_rate * stop        (conservative: assume we exit at the stop)
    per_unit_risk     = stop_distance + fee_entry_per_unit + fee_exit_per_unit
    quantity          = risk_amount / per_unit_risk

The quantity is then capped by, in order:
    1. maximum position notional   -> qty <= equity * max_notional_pct/100 / entry
    2. available simulated capital -> qty <= available_balance / entry   (1x, no leverage)
    3. rounding down to `qty_step`

Finally the trade is REJECTED (never silently shrunk into nonsense) when:
    quantity < min_qty, or notional < min_notional.

Because fees are inside `per_unit_risk`, the loss realised if the stop is hit is
approximately `risk_amount` *including* both fees.  That is the point: without
it, a 1 % risk setting actually loses slightly more than 1 %.

Important honesty note
----------------------
A stop-loss does NOT guarantee the exact realised loss.  Simulated fills use the
candle's stop price (or the open if the candle gapped through it), and fees and
slippage are estimates.  Real execution can be worse.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .config import (
    RiskParams,
    STOP_MODE_ATR,
    STOP_MODE_PERCENT,
    STOP_MODE_SWING,
    STOP_MODES,
    StrategyParams,
)

LONG = "long"
SHORT = "short"

#: A stop closer than this fraction of the entry price is rejected: the fee
#: would dominate the risk budget and the quantity would explode.
MIN_STOP_DISTANCE_FRAC = 1e-5


class RiskValidationError(ValueError):
    """Raised when a proposed trade would violate a risk rule."""


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #

@dataclass
class Levels:
    """Entry / stop / take-profit triple plus the distances between them."""

    direction: str
    entry: float
    stop: float
    take_profit: float
    stop_distance: float
    tp_distance: float
    reward_to_risk: float
    stop_mode: str
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class PositionSize:
    """Outcome of a sizing attempt.  `approved=False` means do not trade."""

    approved: bool
    quantity: float = 0.0
    notional: float = 0.0
    risk_amount: float = 0.0
    effective_risk_amount: float = 0.0
    effective_risk_pct: float = 0.0
    entry_fee: float = 0.0
    exit_fee_at_stop: float = 0.0
    per_unit_risk: float = 0.0
    reasons: List[str] = field(default_factory=list)
    capped_by: Optional[str] = None

    @property
    def reason_text(self) -> str:
        return "; ".join(self.reasons) if self.reasons else "ok"

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class DailyLossState:
    """Daily loss-limit evaluation."""

    locked: bool
    reference_balance: float
    realized_today: float
    limit_amount: float
    remaining: float
    used_pct: float
    message: str
    utc_date: str

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


# --------------------------------------------------------------------------- #
# Stop-loss / take-profit
# --------------------------------------------------------------------------- #

def compute_levels(
    direction: str,
    entry: float,
    params: StrategyParams,
    risk: RiskParams,
    prior_swing_high: Optional[float] = None,
    prior_swing_low: Optional[float] = None,
    atr_value: Optional[float] = None,
) -> Levels:
    """
    Place the stop-loss and take-profit for a simulated trade.

    LONG : stop below entry, take-profit above entry.
    SHORT: stop above entry, take-profit below entry.

    Stop modes
        swing   : beyond the prior swing low/high, plus a buffer
        atr     : `atr_multiplier` ATRs away, plus a buffer
        percent : a fixed percentage of the entry price

    Raises RiskValidationError when the resulting stop is on the wrong side of
    the entry or is too close to it to be meaningful.
    """
    if direction not in (LONG, SHORT):
        raise RiskValidationError(f"Direction must be '{LONG}' or '{SHORT}', got '{direction}'.")
    if not math.isfinite(entry) or entry <= 0:
        raise RiskValidationError(f"Entry price must be a positive number, got {entry}.")
    if params.stop_mode not in STOP_MODES:
        raise RiskValidationError(f"Unknown stop mode '{params.stop_mode}'.")

    notes: List[str] = []
    buffer_frac = params.stop_buffer_pct / 100.0
    raw_stop: Optional[float] = None

    if params.stop_mode == STOP_MODE_SWING:
        if direction == LONG:
            if prior_swing_low is None or not math.isfinite(prior_swing_low) or prior_swing_low <= 0:
                raise RiskValidationError(
                    "Swing stop mode needs a valid prior swing low, which is not "
                    "available yet (insufficient history)."
                )
            raw_stop = prior_swing_low * (1.0 - buffer_frac)
            notes.append(
                f"Stop placed below the prior swing low ({prior_swing_low:,.2f}) "
                f"with a {params.stop_buffer_pct:g}% buffer."
            )
        else:
            if prior_swing_high is None or not math.isfinite(prior_swing_high) or prior_swing_high <= 0:
                raise RiskValidationError(
                    "Swing stop mode needs a valid prior swing high, which is not "
                    "available yet (insufficient history)."
                )
            raw_stop = prior_swing_high * (1.0 + buffer_frac)
            notes.append(
                f"Stop placed above the prior swing high ({prior_swing_high:,.2f}) "
                f"with a {params.stop_buffer_pct:g}% buffer."
            )

    elif params.stop_mode == STOP_MODE_ATR:
        if atr_value is None or not math.isfinite(atr_value) or atr_value <= 0:
            raise RiskValidationError("ATR stop mode needs a valid ATR value.")
        distance = atr_value * params.atr_multiplier
        raw_stop = entry - distance if direction == LONG else entry + distance
        raw_stop = raw_stop * (1.0 - buffer_frac) if direction == LONG else raw_stop * (1.0 + buffer_frac)
        notes.append(
            f"Stop placed {params.atr_multiplier:g} x ATR ({atr_value:,.2f}) from entry, "
            f"plus a {params.stop_buffer_pct:g}% buffer."
        )

    else:  # percent
        frac = params.percent_stop / 100.0
        raw_stop = entry * (1.0 - frac) if direction == LONG else entry * (1.0 + frac)
        notes.append(f"Stop placed {params.percent_stop:g}% from entry.")

    stop_distance = abs(entry - raw_stop)

    # --- side validation ------------------------------------------------- #
    if direction == LONG and raw_stop >= entry:
        raise RiskValidationError(
            f"LONG stop ({raw_stop:,.2f}) must be BELOW entry ({entry:,.2f}). "
            f"The swing/ATR reference is on the wrong side - no trade is opened."
        )
    if direction == SHORT and raw_stop <= entry:
        raise RiskValidationError(
            f"SHORT stop ({raw_stop:,.2f}) must be ABOVE entry ({entry:,.2f}). "
            f"The swing/ATR reference is on the wrong side - no trade is opened."
        )
    if stop_distance < entry * MIN_STOP_DISTANCE_FRAC:
        raise RiskValidationError(
            f"Stop distance ({stop_distance:,.6f}) is too small relative to entry "
            f"({entry:,.2f}); fees would dominate the risk budget."
        )

    tp_distance = stop_distance * risk.reward_to_risk
    take_profit = entry + tp_distance if direction == LONG else entry - tp_distance

    if direction == LONG and take_profit <= entry:
        raise RiskValidationError("LONG take-profit must be above entry.")
    if direction == SHORT and take_profit <= 0:
        raise RiskValidationError(
            f"SHORT take-profit ({take_profit:,.2f}) would be at or below zero. "
            f"Reduce the reward-to-risk target."
        )

    return Levels(
        direction=direction,
        entry=entry,
        stop=raw_stop,
        take_profit=take_profit,
        stop_distance=stop_distance,
        tp_distance=tp_distance,
        reward_to_risk=risk.reward_to_risk,
        stop_mode=params.stop_mode,
        notes=notes,
    )


def validate_levels(levels: Levels) -> None:
    """Re-check an existing Levels object (used before persisting a trade)."""
    if levels.direction == LONG:
        if not levels.stop < levels.entry < levels.take_profit:
            raise RiskValidationError(
                f"Invalid LONG levels: stop={levels.stop}, entry={levels.entry}, tp={levels.take_profit}"
            )
    else:
        if not levels.take_profit < levels.entry < levels.stop:
            raise RiskValidationError(
                f"Invalid SHORT levels: tp={levels.take_profit}, entry={levels.entry}, stop={levels.stop}"
            )
    if levels.stop <= 0 or levels.take_profit <= 0 or levels.entry <= 0:
        raise RiskValidationError("All prices must be positive.")


# --------------------------------------------------------------------------- #
# Position sizing
# --------------------------------------------------------------------------- #

def compute_position_size(
    equity: float,
    available_balance: float,
    entry: float,
    stop: float,
    risk: RiskParams,
) -> PositionSize:
    """
    Size a simulated position so that hitting the stop loses ~risk_per_trade_pct
    of equity *after* both fees.  See the module docstring for the formula.
    """
    risk.validate()
    reasons: List[str] = []

    if not math.isfinite(equity) or equity <= 0:
        return PositionSize(False, reasons=[f"Equity must be positive (got {equity})."])
    if not math.isfinite(entry) or entry <= 0:
        return PositionSize(False, reasons=[f"Entry price must be positive (got {entry})."])
    if not math.isfinite(stop) or stop <= 0:
        return PositionSize(False, reasons=[f"Stop price must be positive (got {stop})."])

    stop_distance = abs(entry - stop)
    if stop_distance <= entry * MIN_STOP_DISTANCE_FRAC:
        return PositionSize(False, reasons=[
            f"Stop distance {stop_distance:.8f} is too small relative to entry {entry:.2f}."
        ])

    fee_rate = risk.fee_rate
    per_unit_risk = stop_distance + fee_rate * entry + fee_rate * stop
    if per_unit_risk <= 0:
        return PositionSize(False, reasons=["Computed per-unit risk is not positive."])

    risk_amount = equity * risk.risk_per_trade_pct / 100.0
    if risk_amount <= 0:
        return PositionSize(False, reasons=["Risk amount is zero."])

    qty = risk_amount / per_unit_risk
    capped_by: Optional[str] = None

    # Cap 1: maximum position notional.
    max_notional = equity * risk.max_notional_pct / 100.0
    qty_by_notional = max_notional / entry
    if qty > qty_by_notional:
        qty = qty_by_notional
        capped_by = f"max notional ({risk.max_notional_pct:g}% of equity)"
        reasons.append(
            f"Quantity reduced from {risk_amount / per_unit_risk:.8f} to {qty:.8f} BTC "
            f"by the max-notional cap of {max_notional:,.2f}."
        )

    # Cap 2: available simulated capital (no leverage in this simulator).
    if available_balance is not None and math.isfinite(available_balance):
        qty_by_balance = max(0.0, available_balance) / entry
        if qty > qty_by_balance:
            qty = qty_by_balance
            capped_by = "available balance"
            reasons.append(
                f"Quantity reduced to {qty:.8f} BTC by the available balance "
                f"({available_balance:,.2f})."
            )

    # Round DOWN to the instrument step - never round up into extra risk.
    if risk.qty_step > 0:
        qty = math.floor(qty / risk.qty_step + 1e-12) * risk.qty_step
    qty = float(qty)

    if qty <= 0:
        return PositionSize(False, reasons=reasons + [
            "Computed quantity rounds to zero. Increase risk %, widen equity, or tighten the stop."
        ], risk_amount=risk_amount, per_unit_risk=per_unit_risk)

    if qty < risk.min_qty:
        return PositionSize(False, reasons=reasons + [
            f"Quantity {qty:.8f} is below the simulated minimum {risk.min_qty:.8f} BTC."
        ], quantity=qty, risk_amount=risk_amount, per_unit_risk=per_unit_risk)

    notional = qty * entry
    if notional < risk.min_notional:
        return PositionSize(False, reasons=reasons + [
            f"Notional {notional:,.2f} is below the simulated minimum {risk.min_notional:,.2f}."
        ], quantity=qty, notional=notional, risk_amount=risk_amount, per_unit_risk=per_unit_risk)

    entry_fee = qty * entry * fee_rate
    exit_fee_at_stop = qty * stop * fee_rate
    effective_risk = qty * per_unit_risk

    return PositionSize(
        approved=True,
        quantity=qty,
        notional=notional,
        risk_amount=risk_amount,
        effective_risk_amount=effective_risk,
        effective_risk_pct=(effective_risk / equity * 100.0) if equity > 0 else 0.0,
        entry_fee=entry_fee,
        exit_fee_at_stop=exit_fee_at_stop,
        per_unit_risk=per_unit_risk,
        reasons=reasons or ["ok"],
        capped_by=capped_by,
    )


# --------------------------------------------------------------------------- #
# P/L maths
# --------------------------------------------------------------------------- #

def gross_pnl(direction: str, quantity: float, entry: float, exit_price: float) -> float:
    """Price-based P/L before fees."""
    if direction == LONG:
        return quantity * (exit_price - entry)
    if direction == SHORT:
        return quantity * (entry - exit_price)
    raise RiskValidationError(f"Unknown direction '{direction}'.")


def fees_for(quantity: float, entry: float, exit_price: float, fee_rate: float) -> Tuple[float, float]:
    """(entry_fee, exit_fee) for a round trip."""
    return quantity * entry * fee_rate, quantity * exit_price * fee_rate


def net_pnl(direction: str, quantity: float, entry: float, exit_price: float, fee_rate: float) -> Tuple[float, float, float]:
    """Return (gross, total_fees, net)."""
    g = gross_pnl(direction, quantity, entry, exit_price)
    ef, xf = fees_for(quantity, entry, exit_price, fee_rate)
    return g, ef + xf, g - ef - xf


def realized_r_multiple(direction: str, quantity: float, entry: float, exit_price: float,
                        stop: float, fee_rate: float) -> Optional[float]:
    """
    Net P/L expressed as a multiple of the amount originally risked.

    The denominator is the loss that would have occurred at the stop, fees
    included, so 1R means "exactly the planned risk".
    """
    if quantity <= 0:
        return None
    risk_at_stop = abs(gross_pnl(direction, quantity, entry, stop))
    _, fees_at_stop, net_at_stop = net_pnl(direction, quantity, entry, stop, fee_rate)
    denominator = abs(net_at_stop) if abs(net_at_stop) > 1e-12 else risk_at_stop
    if denominator <= 1e-12:
        return None
    _, _, net = net_pnl(direction, quantity, entry, exit_price, fee_rate)
    return net / denominator


# --------------------------------------------------------------------------- #
# Daily loss limit
# --------------------------------------------------------------------------- #

def evaluate_daily_loss(
    equity: float,
    realized_since_utc_midnight: float,
    risk: RiskParams,
    utc_date: str,
) -> DailyLossState:
    """
    Daily loss lockout.

    The *documented daily reference balance* is the account equity as it stood
    at 00:00 UTC today.  It is reconstructed from the ledger rather than stored
    separately, so it can never drift out of sync:

        reference = current_equity - realised_P/L_since_00:00_UTC

    The lockout triggers when today's realised P/L is at or below
    `-daily_loss_limit_pct %` of that reference.  While locked, NO new entries
    are opened.  Existing open positions keep their stops and are still managed
    (see the safe-stop note in core/executor.py).
    """
    reference = equity - realized_since_utc_midnight
    if reference <= 0:
        reference = max(equity, 0.0)
    limit_amount = reference * risk.daily_loss_limit_pct / 100.0
    # Only LOSSES consume the budget; a profitable day uses none of it.
    consumed = max(0.0, -realized_since_utc_midnight)
    locked = bool(limit_amount > 0 and consumed >= limit_amount)
    remaining = limit_amount - consumed          # negative => breached
    used_pct = (consumed / limit_amount * 100.0) if limit_amount > 0 else 0.0

    if locked:
        message = (
            f"DAILY LOSS LOCKOUT ACTIVE for {utc_date}: realised {realized_since_utc_midnight:,.2f} "
            f"has reached the {risk.daily_loss_limit_pct:g}% daily limit "
            f"({limit_amount:,.2f}) of the {reference:,.2f} reference balance. "
            f"No new entries until 00:00 UTC. Open positions are still managed."
        )
    else:
        message = (
            f"Daily loss budget {utc_date}: {remaining:,.2f} of {limit_amount:,.2f} remaining "
            f"({used_pct:.1f}% used)."
        )

    return DailyLossState(
        locked=bool(locked),
        reference_balance=reference,
        realized_today=realized_since_utc_midnight,
        limit_amount=limit_amount,
        remaining=remaining,
        used_pct=used_pct,
        message=message,
        utc_date=utc_date,
    )


__all__ = [
    "LONG", "SHORT", "MIN_STOP_DISTANCE_FRAC", "RiskValidationError",
    "Levels", "PositionSize", "DailyLossState",
    "compute_levels", "validate_levels", "compute_position_size",
    "gross_pnl", "fees_for", "net_pnl", "realized_r_multiple",
    "evaluate_daily_loss",
]
