"""
core/executor.py
================
The live paper-trading engine.  It converts closed-candle signals into simulated
trades, stores them in SQLite, and manages exits.

IT NEVER SENDS A REAL ORDER.  There is no code path in this project that calls a
private/authenticated exchange endpoint, and no API key is ever requested.

Per-candle processing order (deliberate)
----------------------------------------
For each newly closed candle, in chronological order:

  1. Fill pending orders    (a `next_open` order queued on the previous candle
                             fills at THIS candle's open)
  2. Manage open positions  (test this candle against stop-loss / take-profit)
  3. Evaluate a new signal  (only if the bot is running, no position is open,
                             the daily loss limit is not breached and the data is fresh)
  4. Mark the candle processed

Step 4 is what makes dashboard refreshes idempotent: the UNIQUE(feed_key,
candle_ts) row means the same closed candle can never open two trades.

Safe stop
---------
`bot_running = False` prevents NEW entries.  Open positions keep being managed
(their stops still fire) because abandoning a position is not "safe".  An
explicit `flatten_open_positions()` closes everything at the last available
price and records `safe_stop_flatten` as the reason.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import risk as R
from . import simulation as S
from .config import (
    FILL_NEXT_OPEN,
    FILL_SIGNAL_CLOSE,
    FILL_MODES,
    TIMEFRAME_MS,
    FeedConfig,
    RiskParams,
    StrategyParams,
)
from .data_feed import CandleSet
from .db import (
    STATUS_CLOSED,
    STATUS_OPEN,
    Database,
    utc_date_str,
    utc_midnight_ms,
)
from .strategy import LONG as SIG_LONG
from .strategy import SHORT as SIG_SHORT
from .strategy import generate_signals

log = logging.getLogger("btc_executor")

STRATEGY_ID = "BOS+RSI+EMA-v1"

#: A pending next-open order older than this many candles is expired rather than
#: filled, because acting on a several-hours-old signal is not realistic.
MAX_PENDING_CANDLES = 2


@dataclass
class Action:
    """One thing the engine did (or refused to do) during a refresh."""

    kind: str
    message: str
    candle_ts: Optional[int] = None
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessResult:
    actions: List[Action] = field(default_factory=list)
    opened: List[str] = field(default_factory=list)
    closed: List[str] = field(default_factory=list)
    processed_candles: int = 0
    trading_allowed: bool = True
    gate_reason: str = "ok"
    daily_locked: bool = False
    daily_message: str = ""

    def add(self, kind: str, message: str, candle_ts: Optional[int] = None, **data: Any) -> None:
        self.actions.append(Action(kind, message, candle_ts, data))


class PaperTradingEngine:
    """Simulated order-management engine."""

    def __init__(
        self,
        db: Database,
        feed_cfg: FeedConfig,
        strategy: StrategyParams,
        risk_params: RiskParams,
        fill_model: str = FILL_NEXT_OPEN,
        bot_running: bool = False,
        starting_balance: Optional[float] = None,
    ):
        if fill_model not in FILL_MODES:
            raise ValueError(f"fill_model must be one of {FILL_MODES}, got '{fill_model}'")
        strategy.validate()
        risk_params.validate()
        feed_cfg.validate()

        self.db = db
        self.cfg = feed_cfg
        self.strategy = strategy
        self.risk = risk_params
        self.fill_model = fill_model
        self.bot_running = bool(bot_running)
        self.starting_balance = float(
            starting_balance if starting_balance is not None
            else (db.get_setting("starting_balance") or risk_params.starting_balance)
        )

    # ------------------------------------------------------------------ #
    # Account state
    # ------------------------------------------------------------------ #

    @property
    def feed_key(self) -> str:
        return self.cfg.feed_key

    def realized_balance(self) -> float:
        """Starting balance plus every closed trade's net P/L (ledger-consistent)."""
        return self.starting_balance + self.db.realized_pnl_total()

    def open_positions(self) -> List[Dict[str, Any]]:
        return self.db.get_open_trades(self.feed_key)

    def equity(self, mark_price: Optional[float] = None) -> float:
        """Realised balance plus unrealised P/L of open positions."""
        eq = self.realized_balance()
        if mark_price is None or not np.isfinite(mark_price):
            return eq
        for t in self.open_positions():
            if not t.get("entry_price") or not t.get("quantity"):
                continue
            eq += S.mark_to_market(t["direction"], float(t["quantity"]),
                                   float(t["entry_price"]), float(mark_price))
        return eq

    def unrealized(self, mark_price: Optional[float]) -> float:
        return self.equity(mark_price) - self.realized_balance()

    def committed_notional(self) -> float:
        total = 0.0
        for t in self.open_positions():
            if t.get("entry_price") and t.get("quantity"):
                total += float(t["entry_price"]) * float(t["quantity"])
        return total

    def daily_state(self, now_ms: Optional[float] = None) -> R.DailyLossState:
        now_ms = now_ms if now_ms is not None else time.time() * 1000.0
        midnight = utc_midnight_ms(now_ms)
        realized_today = self.db.realized_pnl_since(midnight)
        eq = self.realized_balance()
        return R.evaluate_daily_loss(eq, realized_today, self.risk, utc_date_str(now_ms))

    # ------------------------------------------------------------------ #
    # Gating
    # ------------------------------------------------------------------ #

    def trading_gate(self, candles: CandleSet) -> Tuple[bool, str]:
        """
        Decide whether simulated trading may happen at all on this refresh.

        Display is always allowed; TRADING is not.
        """
        ok, why = candles.safe_for_trading()
        if not ok:
            return False, f"Market data gate: {why}"
        if candles.candle_count < self.strategy.min_history():
            return False, (
                f"Need at least {self.strategy.min_history()} closed candles for the "
                f"indicator warm-up; have {candles.candle_count}. Signals are "
                f"suppressed, never guessed."
            )
        return True, "ok"

    # ------------------------------------------------------------------ #
    # Main entry point
    # ------------------------------------------------------------------ #

    def process(self, candles: CandleSet) -> ProcessResult:
        """Evaluate every newly closed candle and act on it."""
        result = ProcessResult()

        allowed, reason = self.trading_gate(candles)
        result.trading_allowed = allowed
        result.gate_reason = reason
        if not allowed:
            result.add("gate_blocked", reason)
            self.db.log_event("gate_blocked", reason, level="warning", feed_key=self.feed_key)
            return result

        df = generate_signals(candles.df, self.strategy)
        last_processed = self.db.last_processed_candle(self.feed_key)

        if last_processed is None:
            # First run on this series: do NOT replay history into live trades.
            # Every candle currently on screen - INCLUDING the most recent closed
            # one - is marked as already seen.  The bot therefore starts acting
            # from the next candle that closes after it was started, and a
            # repeated refresh of the same data can never process anything twice.
            bootstrap = df["timestamp"].to_numpy(dtype="int64")
            with self.db.transaction() as conn:
                for ts in bootstrap:
                    self.db.mark_candle_processed(self.feed_key, int(ts), "bootstrap", conn)
                self.db.log_event(
                    "bootstrap",
                    f"First run on {self.feed_key}: {len(bootstrap)} candle(s) already on "
                    f"screen were marked as seen, so the bot does not replay history. "
                    f"It will act on the next candle that closes.",
                    level="info", feed_key=self.feed_key, conn=conn,
                )
            result.add("bootstrap",
                       f"Baselined {len(bootstrap)} candle(s); no trades replayed. "
                       f"The bot acts from the next closed candle onwards.")
            return result

        # The daily loss state is evaluated on EVERY refresh, even when no new
        # candle has closed, so the dashboard always shows the true lockout
        # status rather than a stale default.
        daily = self.daily_state(candles.reference_now_ms)
        result.daily_locked = daily.locked
        result.daily_message = daily.message

        new_rows = df[df["timestamp"] > last_processed]
        if new_rows.empty:
            result.add("no_new_candles", "No newly closed candles since the last refresh.")
            return result

        self._expire_stale_pending(new_rows, result)

        for _, row in new_rows.iterrows():
            self._process_candle(row, candles, daily, result)
            result.processed_candles += 1

        if daily.locked:
            result.add("daily_loss_lockout", daily.message)
            self.db.log_event("daily_loss_lockout", daily.message, level="warning",
                              feed_key=self.feed_key)
        return result

    # ------------------------------------------------------------------ #
    # Per-candle work
    # ------------------------------------------------------------------ #

    def _process_candle(self, row: pd.Series, candles: CandleSet,
                        daily: R.DailyLossState, result: ProcessResult) -> None:
        ts = int(row["timestamp"])
        outcome = "evaluated"

        # 1. fill pending next-open orders -------------------------------- #
        if self.fill_model == FILL_NEXT_OPEN:
            for order in self.db.get_pending_orders(self.feed_key):
                if int(order["signal_ts"]) < ts:
                    filled = self._fill_pending(order, float(row["open"]), ts, daily, result)
                    if filled:
                        outcome = "filled_pending"

        # 2. manage open positions ---------------------------------------- #
        for trade in self.open_positions():
            decision = self._maybe_close(trade, row, result)
            if decision is not None:
                outcome = "exit"

        # 3. evaluate a new signal ---------------------------------------- #
        if self.bot_running and not daily.locked:
            signal = int(row.get("signal", 0) or 0)
            if signal != 0:
                direction = R.LONG if signal == SIG_LONG else R.SHORT
                outcome = self._handle_signal(direction, row, ts, candles, daily, result) or outcome
            else:
                result.add("no_signal", f"No valid setup on the candle closed at {ts}.", ts)
        elif not self.bot_running:
            result.add("bot_stopped", "Bot is stopped: no new entries are opened.", ts)

        # 4. idempotency guard ------------------------------------------- #
        self.db.mark_candle_processed(self.feed_key, ts, outcome)

    def _expire_stale_pending(self, new_rows: pd.DataFrame, result: ProcessResult) -> None:
        """Cancel pending orders that are too old to fill realistically."""
        if self.fill_model != FILL_NEXT_OPEN or new_rows.empty:
            return
        tf_ms = TIMEFRAME_MS[self.cfg.timeframe]
        newest = int(new_rows["timestamp"].max())
        for order in self.db.get_pending_orders(self.feed_key):
            age_candles = (newest - int(order["signal_ts"])) // tf_ms
            if age_candles > MAX_PENDING_CANDLES:
                self.db.set_pending_status(order["order_id"], "expired")
                msg = (f"Pending order from signal {order['signal_ts']} expired "
                       f"({age_candles} candles old).")
                result.add("pending_expired", msg, int(order["signal_ts"]))
                self.db.log_event("pending_expired", msg, level="warning", feed_key=self.feed_key)

    def _handle_signal(self, direction: str, row: pd.Series, ts: int,
                       candles: CandleSet, daily: R.DailyLossState,
                       result: ProcessResult) -> Optional[str]:
        """React to a valid signal on a closed candle."""
        open_now = self.open_positions()
        if len(open_now) >= self.risk.max_open_positions:
            msg = (f"Signal {direction.upper()} on candle {ts} ignored: already at the "
                   f"maximum of {self.risk.max_open_positions} open position(s).")
            result.add("max_positions", msg, ts)
            self.db.log_event("max_positions", msg, level="info", feed_key=self.feed_key)
            return "blocked_max_positions"

        if direction == R.SHORT and not self.risk.allow_shorts:
            msg = f"Short signal on candle {ts} ignored: shorts are disabled in settings."
            result.add("shorts_disabled", msg, ts)
            self.db.log_event("shorts_disabled", msg, level="info", feed_key=self.feed_key)
            return "blocked_shorts_disabled"

        if self.fill_model == FILL_SIGNAL_CLOSE:
            ok = self._open_position(direction, float(row["close"]), ts, ts, row, candles, daily, result)
            return "entry" if ok else "blocked_risk"

        # next-open model: queue it, uniquely
        params = {
            "prior_swing_high": _num(row.get("prior_swing_high")),
            "prior_swing_low": _num(row.get("prior_swing_low")),
            "atr": _num(row.get("atr")),
            "rsi": _num(row.get("rsi")),
            "signal_close": _num(row.get("close")),
        }
        order_id = self.db.create_pending_order(self.feed_key, ts, direction, params)
        if order_id is None:
            msg = (f"Duplicate signal on candle {ts} ignored - a pending order for this "
                   f"candle already exists.")
            result.add("duplicate_signal", msg, ts)
            self.db.log_event("duplicate_signal", msg, level="info", feed_key=self.feed_key)
            return "blocked_duplicate"
        msg = (f"{direction.upper()} signal on the candle closed at {ts}. Order queued to "
               f"fill at the next candle's open ({self.fill_model} fill model).")
        result.add("pending_created", msg, ts, order_id=order_id)
        self.db.log_event("pending_created", msg, level="info", feed_key=self.feed_key,
                          data={"order_id": order_id, "direction": direction, **params})
        return "pending"

    def _fill_pending(self, order: Dict[str, Any], open_price: float, ts: int,
                      daily: R.DailyLossState, result: ProcessResult) -> bool:
        """Fill a queued next-open order at this candle's open."""
        direction = order["direction"]
        params = order.get("params") or {}
        signal_ts = int(order["signal_ts"])

        if len(self.open_positions()) >= self.risk.max_open_positions:
            self.db.set_pending_status(order["order_id"], "expired")
            result.add("pending_expired", f"Pending order {order['order_id']} expired: position limit reached.", ts)
            return False

        row = pd.Series({
            "prior_swing_high": params.get("prior_swing_high"),
            "prior_swing_low": params.get("prior_swing_low"),
            "atr": params.get("atr"),
            "rsi": params.get("rsi"),
        })
        ok = self._open_position(direction, open_price, signal_ts, ts, row, None, daily, result,
                                 order_id=order["order_id"])
        self.db.set_pending_status(order["order_id"], "filled" if ok else "expired")
        return ok

    # ------------------------------------------------------------------ #
    # Opening
    # ------------------------------------------------------------------ #

    def _open_position(self, direction: str, raw_price: float, signal_ts: int, entry_ts: int,
                       row: pd.Series, candles: Optional[CandleSet], daily: R.DailyLossState,
                       result: ProcessResult, order_id: Optional[str] = None) -> bool:
        """
        Compute levels, size the position and (if every check passes) record the
        trade and its audit event in ONE transaction.
        """
        entry = S.entry_fill_price(raw_price, direction, self.risk.slippage_rate)
        mark = float(candles.df["close"].iloc[-1]) if candles is not None else entry
        equity = self.equity(mark)
        available = max(0.0, equity - self.committed_notional())

        try:
            levels = R.compute_levels(
                direction=direction,
                entry=entry,
                params=self.strategy,
                risk=self.risk,
                prior_swing_high=_num(row.get("prior_swing_high")),
                prior_swing_low=_num(row.get("prior_swing_low")),
                atr_value=_num(row.get("atr")),
            )
            R.validate_levels(levels)
        except R.RiskValidationError as exc:
            msg = f"Trade blocked (invalid stop placement): {exc}"
            result.add("blocked_levels", msg, signal_ts)
            self.db.log_event("blocked_levels", msg, level="warning", feed_key=self.feed_key)
            return False

        sizing = R.compute_position_size(
            equity=equity, available_balance=available,
            entry=levels.entry, stop=levels.stop, risk=self.risk,
        )
        if not sizing.approved:
            msg = f"Trade blocked by risk rules: {sizing.reason_text}"
            result.add("blocked_risk", msg, signal_ts)
            self.db.log_event("blocked_risk", msg, level="warning", feed_key=self.feed_key,
                              data=sizing.to_dict())
            return False

        trade = {
            "exchange": self.cfg.exchange_key,
            "symbol": self.cfg.symbol,
            "strategy": STRATEGY_ID,
            "direction": direction,
            "signal_ts": int(signal_ts),
            "entry_ts": int(entry_ts),
            "entry_price": levels.entry,
            "quantity": sizing.quantity,
            "stop_loss": levels.stop,
            "take_profit": levels.take_profit,
            "initial_risk": sizing.effective_risk_amount,
            "entry_fee": sizing.entry_fee,
            "status": STATUS_OPEN,
            "fill_model": self.fill_model,
            "slippage_bps": self.risk.slippage_bps,
            "stop_mode": self.strategy.stop_mode,
            "notes": " | ".join(levels.notes + sizing.reasons),
        }

        msg = (
            f"Opened simulated {direction.upper()} {sizing.quantity:.8f} {self.cfg.symbol} "
            f"@ {levels.entry:,.2f} | SL {levels.stop:,.2f} | TP {levels.take_profit:,.2f} | "
            f"risk {sizing.effective_risk_amount:,.2f} ({sizing.effective_risk_pct:.2f}%)"
        )
        try:
            with self.db.transaction() as conn:
                trade_id = self.db.insert_trade(trade, conn)
                self.db.log_event("entry", msg, level="info", feed_key=self.feed_key,
                                  data={"order_id": order_id, "sizing": sizing.to_dict(),
                                        "levels": levels.to_dict()}, conn=conn)
        except Exception as exc:
            log.exception("Failed to persist trade")
            result.add("db_error", f"Could not save the trade: {exc}", signal_ts)
            return False

        result.opened.append(trade_id)
        result.add("entry", msg, signal_ts, trade_id=trade_id)
        return True

    # ------------------------------------------------------------------ #
    # Closing
    # ------------------------------------------------------------------ #

    def _maybe_close(self, trade: Dict[str, Any], row: pd.Series,
                     result: ProcessResult) -> Optional[S.ExitDecision]:
        entry = trade.get("entry_price")
        stop = trade.get("stop_loss")
        tp = trade.get("take_profit")
        qty = trade.get("quantity")
        if None in (entry, stop, tp, qty):
            result.add("data_error", f"Trade {trade.get('trade_id')} has missing level data; skipped.",
                       int(row["timestamp"]))
            return None

        decision = S.check_exit(
            direction=trade["direction"], entry_price=float(entry), stop=float(stop),
            take_profit=float(tp),
            open_=float(row["open"]), high=float(row["high"]),
            low=float(row["low"]), close=float(row["close"]),
            slippage_rate=self.risk.slippage_rate,
        )
        if decision is None:
            return None

        self._close(trade, decision.exit_price, int(row["timestamp"]), decision.reason,
                    decision.note, result)
        return decision

    def _close(self, trade: Dict[str, Any], exit_price: float, exit_ts: int,
               reason: str, note: str, result: ProcessResult) -> bool:
        direction = trade["direction"]
        qty = float(trade["quantity"])
        entry = float(trade["entry_price"])
        gross, fees_total, net = R.net_pnl(direction, qty, entry, exit_price, self.risk.fee_rate)
        exit_fee = qty * exit_price * self.risk.fee_rate
        r_multiple = R.realized_r_multiple(direction, qty, entry, exit_price,
                                           float(trade["stop_loss"]), self.risk.fee_rate)

        msg = (
            f"Closed {direction.upper()} {qty:.8f} @ {exit_price:,.2f} ({reason}) | "
            f"net {net:+,.2f} | {r_multiple:+.2f}R"
        )
        updates = {
            "exit_ts": int(exit_ts), "exit_price": exit_price, "exit_fee": exit_fee,
            "gross_pnl": gross, "net_pnl": net, "realized_r": r_multiple,
            "exit_reason": reason, "status": STATUS_CLOSED,
            "notes": ((trade.get("notes") or "") + " || " + note).strip(" |"),
        }
        try:
            with self.db.transaction() as conn:
                self.db.close_trade(trade["trade_id"], updates, conn)
                self.db.log_event("exit", msg, level="info", feed_key=self.feed_key,
                                  data={"reason": reason, "note": note, "net_pnl": net,
                                        "realized_r": r_multiple}, conn=conn)
        except Exception as exc:
            log.exception("Failed to close trade")
            result.add("db_error", f"Could not persist the exit: {exc}", exit_ts)
            return False

        result.closed.append(trade["trade_id"])
        result.add("exit", msg, exit_ts, trade_id=trade["trade_id"], reason=reason, net_pnl=net)
        return True

    def flatten_open_positions(self, mark_price: float, reason: str = S.EXIT_SAFE_STOP) -> List[str]:
        """Close every open position at `mark_price` (explicit user action)."""
        closed: List[str] = []
        result = ProcessResult()
        for trade in self.open_positions():
            price = S.exit_fill_price(mark_price, trade["direction"], self.risk.slippage_rate)
            if self._close(trade, price, int(time.time() * 1000), reason,
                           f"Position flattened by the user at mark price {mark_price:,.2f}.",
                           result):
                closed.append(trade["trade_id"])
        return closed

    # ------------------------------------------------------------------ #
    # Lifecycle helpers used by the UI
    # ------------------------------------------------------------------ #

    def set_running(self, running: bool) -> None:
        self.bot_running = bool(running)
        self.db.set_setting(f"bot_running:{self.feed_key}", self.bot_running)
        self.db.log_event(
            "bot_started" if running else "bot_stopped",
            ("Paper bot STARTED. New entries are allowed on valid signals."
             if running else
             "Paper bot STOPPED. No new entries will be opened. Existing open "
             "positions keep their stops and are still managed."),
            level="info", feed_key=self.feed_key,
        )

    def is_running(self) -> bool:
        stored = self.db.get_setting(f"bot_running:{self.feed_key}")
        return bool(stored) if stored is not None else self.bot_running


def _num(value: Any) -> Optional[float]:
    """Coerce a possibly-NaN numpy scalar to float or None."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(f):
        return None
    return f


__all__ = [
    "PaperTradingEngine", "ProcessResult", "Action", "STRATEGY_ID", "MAX_PENDING_CANDLES",
]
