"""
core/backtest.py
================
Historical backtesting, kept strictly separate from live paper execution.

Guarantees
----------
*  It reuses core/simulation.py and core/risk.py - the SAME exit, slippage,
   fee and position-sizing code as the live paper engine - so a backtest is a
   fair preview of the bot's rules rather than a different model.
*  It never reads or writes the `paper_trades` table.  Saved live paper history
   cannot be altered or deleted by running a backtest.
*  No look-ahead: at candle `i` the engine may only see rows `0..i`.  Indicators
   are causal by construction and `tests/test_backtest.py` proves that extending
   the series never changes an earlier decision.
*  Signals are timestamped separately from fills.  With the default `next_open`
   model a signal on candle `i` fills at the open of candle `i+1`.
*  When a candle contains both the stop and the target, the stop fills first.

Everything a backtest cannot know (real order-book depth, funding, partial fills,
exchange outages, latency) is listed in `limitations` so the report never
overstates itself.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from . import risk as R
from . import simulation as S
from .config import (
    FILL_NEXT_OPEN,
    FILL_SIGNAL_CLOSE,
    RiskParams,
    StrategyParams,
)
from .db import utc_date_str, utc_midnight_ms
from .metrics import PerformanceMetrics, compute_metrics
from .strategy import LONG as SIG_LONG
from .strategy import SHORT as SIG_SHORT
from .strategy import generate_signals

log = logging.getLogger("btc_backtest")

LIMITATIONS = [
    "OHLC data cannot reveal the order of events inside a candle; the stop-loss is assumed to fill first.",
    "Fills are simulated at the candle open/close plus a fixed slippage assumption, not against real order-book depth.",
    "Fees are a flat percentage per side; real fees depend on volume tier, maker/taker role and the exchange.",
    "No partial fills, requotes, exchange outages, network latency or funding costs are modelled.",
    "Past simulated performance does not predict future results and is not financial advice.",
]


@dataclass
class BacktestConfig:
    """Everything the user can configure, echoed back in the report."""

    exchange: str
    symbol: str
    timeframe: str
    candles: int
    starting_balance: float
    fill_model: str = FILL_NEXT_OPEN
    strategy: StrategyParams = field(default_factory=StrategyParams)
    risk: RiskParams = field(default_factory=RiskParams)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "exchange": self.exchange, "symbol": self.symbol, "timeframe": self.timeframe,
            "candles": self.candles, "starting_balance": self.starting_balance,
            "fill_model": self.fill_model,
            "strategy": self.strategy.to_dict(), "risk": self.risk.to_dict(),
        }


@dataclass
class BacktestResult:
    ok: bool
    config: BacktestConfig
    trades: List[Dict[str, Any]] = field(default_factory=list)
    equity_curve: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Optional[PerformanceMetrics] = None
    warnings: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=lambda: list(LIMITATIONS))
    candles_available: int = 0
    candles_evaluated: int = 0
    warmup_skipped: int = 0
    signals_seen: int = 0
    blocked_entries: int = 0
    daily_lockouts: int = 0
    ended_with_open_position: bool = False
    message: str = ""
    duration_ms: float = 0.0

    @property
    def n_trades(self) -> int:
        return len(self.trades)

    @property
    def net_pnl(self) -> float:
        return float(self.metrics.net_pnl) if self.metrics else 0.0

    def summary_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok, "n_trades": self.n_trades, "net_pnl": self.net_pnl,
            "win_rate": self.metrics.win_rate if self.metrics else None,
            "profit_factor": self.metrics.profit_factor if self.metrics else None,
            "max_drawdown": self.metrics.max_drawdown if self.metrics else None,
            "candles_available": self.candles_available,
            "candles_evaluated": self.candles_evaluated,
            "signals_seen": self.signals_seen,
            "warnings": list(self.warnings),
        }


def run_backtest(df: pd.DataFrame, config: BacktestConfig) -> BacktestResult:
    """
    Run the strategy over historical CLOSED candles.

    `df` must be an OHLCV frame of closed candles sorted ascending by timestamp
    (exactly what `CandleSet.df` provides).
    """
    t0 = time.perf_counter()
    config.strategy.validate()
    config.risk.validate()
    if config.fill_model not in (FILL_NEXT_OPEN, FILL_SIGNAL_CLOSE):
        raise ValueError(f"Unknown fill model '{config.fill_model}'")

    result = BacktestResult(ok=False, config=config)

    if df is None or df.empty:
        result.message = "No candles were supplied to the backtest."
        result.warnings.append(result.message)
        return result

    result.candles_available = len(df)
    min_history = config.strategy.min_history()
    if len(df) < min_history + 5:
        result.message = (
            f"Not enough candles: the backtest needs at least {min_history + 5} closed "
            f"candles (indicator warm-up plus a few to trade), but only {len(df)} were "
            f"supplied. Increase the candle count or reduce the EMA slow period."
        )
        result.warnings.append(result.message)
        return result

    # Indicators are causal, so computing them once over the whole frame is
    # equivalent to recomputing on every bar - but far faster.
    sig = generate_signals(df, config.strategy)
    result.warmup_skipped = int((~sig["ready"].to_numpy(dtype=bool)).sum())

    balance = float(config.starting_balance)
    fee_rate = config.risk.fee_rate
    slippage = config.risk.slippage_rate

    position: Optional[Dict[str, Any]] = None
    pending: Optional[Dict[str, Any]] = None
    day_key: Optional[str] = None
    day_start_balance = balance
    trades: List[Dict[str, Any]] = []

    rows = sig.to_dict("records")

    for i, row in enumerate(rows):
        ts = int(row["timestamp"])
        o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
        utc_day = utc_date_str(ts)

        # new UTC day -> reset the daily loss reference
        if utc_day != day_key:
            day_key = utc_day
            day_start_balance = balance

        # ---- 1. fill a pending next-open order -------------------------- #
        if pending is not None and position is None and config.fill_model == FILL_NEXT_OPEN:
            if ts > int(pending["signal_ts"]):
                entry_raw = o
                position = _open(
                    pending["direction"], entry_raw, int(pending["signal_ts"]), ts,
                    pending["row"], balance, config, slippage,
                )
                if position is None:
                    result.blocked_entries += 1
                pending = None

        # ---- 2. manage the open position -------------------------------- #
        if position is not None:
            decision = S.check_exit(
                direction=position["direction"], entry_price=position["entry_price"],
                stop=position["stop_loss"], take_profit=position["take_profit"],
                open_=o, high=h, low=l, close=c, slippage_rate=slippage,
            )
            if decision is not None:
                balance = _close(position, decision.exit_price, ts, decision.reason,
                                 fee_rate, balance, trades)
                position = None

        # ---- 3. look for a new signal ----------------------------------- #
        if position is None and pending is None and bool(row.get("ready", False)):
            signal = int(row.get("signal", 0) or 0)
            if signal != 0:
                result.signals_seen += 1
                direction = R.LONG if signal == SIG_LONG else R.SHORT

                realized_today = balance - day_start_balance
                daily = R.evaluate_daily_loss(
                    balance, realized_today, config.risk, utc_day)
                if daily.locked:
                    result.daily_lockouts += 1
                    result.warnings.append(
                        f"Entry on {utc_day} skipped: daily loss limit reached "
                        f"({realized_today:,.2f} vs limit {daily.limit_amount:,.2f})."
                    )
                elif direction == R.SHORT and not config.risk.allow_shorts:
                    result.blocked_entries += 1
                elif config.fill_model == FILL_SIGNAL_CLOSE:
                    position = _open(direction, c, ts, ts, row, balance, config, slippage)
                    if position is None:
                        result.blocked_entries += 1
                else:
                    pending = {"direction": direction, "signal_ts": ts, "row": row}

        # ---- 4. equity snapshot ----------------------------------------- #
        unrealized = 0.0
        if position is not None:
            unrealized = S.mark_to_market(position["direction"], position["quantity"],
                                          position["entry_price"], c)
        result.equity_curve.append({
            "timestamp": ts,
            "balance": balance,
            "equity": balance + unrealized,
            "in_position": position is not None,
            "close": c,
        })

    result.candles_evaluated = len(rows)

    # ---- close any position still open at the end of the data ---------- #
    if position is not None:
        last = rows[-1]
        price = S.exit_fill_price(float(last["close"]), position["direction"], slippage)
        balance = _close(position, price, int(last["timestamp"]), S.EXIT_END_OF_DATA,
                         fee_rate, balance, trades)
        position = None
        result.ended_with_open_position = True
        result.warnings.append(
            "The backtest ended with an open position. It was closed at the final "
            "close price and recorded as 'end_of_data'; that trade is not a real exit signal."
        )
    if pending is not None:
        result.warnings.append(
            "A signal on the final candle could not be filled because there is no "
            "next candle in the requested history window."
        )

    result.trades = trades
    result.metrics = compute_metrics(trades, config.starting_balance)

    if result.signals_seen == 0:
        result.warnings.append(
            "No signals were generated at all. The BOS + RSI + EMA combination is "
            "deliberately strict; with these parameters this period simply had no "
            "qualifying setup. No statistics should be read into an empty sample."
        )
    elif not trades:
        result.warnings.append(
            f"{result.signals_seen} signal(s) appeared but every entry was blocked "
            f"({result.blocked_entries} blocked). Check the risk limits and stop mode."
        )

    if result.candles_available and result.warmup_skipped / max(1, result.candles_available) > 0.5:
        result.warnings.append(
            f"{result.warmup_skipped} of {result.candles_available} candles were still in "
            f"indicator warm-up. Request more history for a meaningful test."
        )

    result.ok = True
    result.message = (
        f"Backtested {result.candles_evaluated} closed candle(s); "
        f"{len(trades)} completed trade(s)."
    )
    result.duration_ms = (time.perf_counter() - t0) * 1000.0
    log.info("Backtest finished: %s", result.message)
    return result


# --------------------------------------------------------------------------- #
# Internal helpers (mirror the live engine exactly)
# --------------------------------------------------------------------------- #

def _open(direction: str, raw_price: float, signal_ts: int, entry_ts: int,
          row: Dict[str, Any], balance: float, config: BacktestConfig,
          slippage: float) -> Optional[Dict[str, Any]]:
    """Size and open a simulated position.  Returns None when blocked."""
    entry = S.entry_fill_price(raw_price, direction, slippage)
    try:
        levels = R.compute_levels(
            direction=direction, entry=entry, params=config.strategy, risk=config.risk,
            prior_swing_high=_num(row.get("prior_swing_high")),
            prior_swing_low=_num(row.get("prior_swing_low")),
            atr_value=_num(row.get("atr")),
        )
        R.validate_levels(levels)
    except R.RiskValidationError:
        return None

    sizing = R.compute_position_size(
        equity=balance, available_balance=balance,
        entry=levels.entry, stop=levels.stop, risk=config.risk,
    )
    if not sizing.approved:
        return None

    return {
        "direction": direction,
        "signal_ts": int(signal_ts),
        "entry_ts": int(entry_ts),
        # Equity at the moment of sizing.  Recorded so the max-notional cap can
        # be audited per trade (the balance compounds, so a single starting
        # balance is not a valid reference for later trades).
        "equity_at_entry": float(balance),
        "entry_price": levels.entry,
        "quantity": sizing.quantity,
        "stop_loss": levels.stop,
        "take_profit": levels.take_profit,
        "initial_risk": sizing.effective_risk_amount,
        "entry_fee": sizing.entry_fee,
    }


def _close(position: Dict[str, Any], exit_price: float, exit_ts: int, reason: str,
           fee_rate: float, balance: float, trades: List[Dict[str, Any]]) -> float:
    """Realise a trade and return the new balance."""
    direction = position["direction"]
    qty = float(position["quantity"])
    entry = float(position["entry_price"])
    gross, fees, net = R.net_pnl(direction, qty, entry, exit_price, fee_rate)
    r_mult = R.realized_r_multiple(direction, qty, entry, exit_price,
                                   float(position["stop_loss"]), fee_rate)
    record = dict(position)
    record.update({
        "exit_ts": int(exit_ts), "exit_price": exit_price,
        "entry_fee": qty * entry * fee_rate, "exit_fee": qty * exit_price * fee_rate,
        "gross_pnl": gross, "net_pnl": net, "realized_r": r_mult,
        "exit_reason": reason, "status": "closed",
        "balance": balance + net,
    })
    trades.append(record)
    return balance + net


def _num(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def result_to_rows(result: BacktestResult) -> pd.DataFrame:
    """Trade-by-trade table for the UI."""
    if not result.trades:
        return pd.DataFrame()
    cols = ["signal_ts", "entry_ts", "direction", "entry_price", "quantity",
            "stop_loss", "take_profit", "exit_ts", "exit_price", "exit_reason",
            "gross_pnl", "net_pnl", "realized_r", "balance"]
    df = pd.DataFrame(result.trades)
    for c in cols:
        if c not in df.columns:
            df[c] = None
    return df[cols]


def equity_to_frame(result: BacktestResult) -> pd.DataFrame:
    if not result.equity_curve:
        return pd.DataFrame()
    return pd.DataFrame(result.equity_curve)


__all__ = [
    "BacktestConfig", "BacktestResult", "run_backtest", "result_to_rows",
    "equity_to_frame", "LIMITATIONS",
]
