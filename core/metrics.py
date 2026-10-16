"""
core/metrics.py
===============
Performance analytics with mathematically explicit definitions.

Every metric here states what it means and returns None (with an explanatory
note) when it cannot be computed honestly - for example a profit factor with no
losing trades, or a win rate with zero trades.  Reporting 0% or 100% in those
cases would be misleading.

Definitions used
----------------
win / loss        : a CLOSED trade with net P/L > 0 / < 0.  net P/L == 0 is a
                    scratch and counted separately.
win rate          : wins / (wins + losses + scratches)
profit factor     : gross profit / |gross loss|.  Undefined (None) when there is
                    no losing trade; reported as "no losses yet" rather than inf.
max drawdown      : largest peak-to-trough decline of the realised equity curve,
                    expressed both in currency and as a % of the peak.
avg R             : mean realised P/L divided by the amount risked on that trade.
expectancy        : mean net P/L per closed trade.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

EPS = 1e-9


@dataclass
class PerformanceMetrics:
    """All headline numbers, plus `notes` explaining anything unavailable."""

    closed_trades: int = 0
    wins: int = 0
    losses: int = 0
    scratches: int = 0
    win_rate: Optional[float] = None          # 0..1
    gross_profit: float = 0.0
    gross_loss: float = 0.0                   # negative or zero
    net_pnl: float = 0.0
    profit_factor: Optional[float] = None
    max_drawdown: float = 0.0                 # positive currency amount
    max_drawdown_pct: float = 0.0             # positive percentage
    peak_equity: float = 0.0
    trough_equity: float = 0.0
    avg_win: Optional[float] = None
    avg_loss: Optional[float] = None
    avg_net: Optional[float] = None
    avg_r: Optional[float] = None
    largest_win: Optional[float] = None
    largest_loss: Optional[float] = None
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    total_fees: float = 0.0
    exit_reason_breakdown: Dict[str, int] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


def compute_metrics(
    closed_trades: Sequence[Dict[str, Any]],
    starting_balance: float,
) -> PerformanceMetrics:
    """
    Compute metrics from a sequence of CLOSED trade dicts.

    Each trade needs at least `net_pnl`; `gross_pnl`, `entry_fee`, `exit_fee`,
    `realized_r`, `exit_reason` and `exit_ts` are used when present.
    Trades are sorted by exit timestamp so consecutive-streak and drawdown
    maths follow the real chronological order.
    """
    m = PerformanceMetrics()

    trades = [t for t in (closed_trades or []) if t is not None]
    if not trades:
        m.notes.append(
            "No closed trades yet. Win rate, profit factor, drawdown and averages "
            "are undefined until at least one simulated trade completes."
        )
        m.peak_equity = starting_balance
        m.trough_equity = starting_balance
        return m

    def _ts(t: Dict[str, Any]) -> float:
        v = t.get("exit_ts") or t.get("entry_ts") or t.get("created_at") or 0
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.0

    trades = sorted(trades, key=_ts)

    nets: List[float] = []
    rs: List[float] = []
    for t in trades:
        net = t.get("net_pnl")
        if net is None:
            m.notes.append(f"Trade {t.get('trade_id')} has no net P/L and was excluded.")
            continue
        try:
            nets.append(float(net))
        except (TypeError, ValueError):
            m.notes.append(f"Trade {t.get('trade_id')} has a non-numeric net P/L and was excluded.")
            continue
        r = t.get("realized_r")
        if r is not None:
            try:
                if float(r) == float(r):  # not NaN
                    rs.append(float(r))
            except (TypeError, ValueError):
                pass

        for fee_key in ("entry_fee", "exit_fee"):
            v = t.get(fee_key)
            if v is not None:
                try:
                    m.total_fees += float(v)
                except (TypeError, ValueError):
                    pass

        reason = t.get("exit_reason") or "unknown"
        m.exit_reason_breakdown[reason] = m.exit_reason_breakdown.get(reason, 0) + 1

    m.closed_trades = len(nets)
    if m.closed_trades == 0:
        m.notes.append("No closed trades contained a usable net P/L.")
        m.peak_equity = starting_balance
        m.trough_equity = starting_balance
        return m

    wins = [x for x in nets if x > EPS]
    losses = [x for x in nets if x < -EPS]
    scratches = [x for x in nets if -EPS <= x <= EPS]

    m.wins, m.losses, m.scratches = len(wins), len(losses), len(scratches)
    m.win_rate = len(wins) / len(nets)
    m.gross_profit = float(sum(wins))
    m.gross_loss = float(sum(losses))          # negative
    m.net_pnl = float(sum(nets))
    m.avg_win = (m.gross_profit / len(wins)) if wins else None
    m.avg_loss = (m.gross_loss / len(losses)) if losses else None
    m.avg_net = m.net_pnl / len(nets)
    m.largest_win = max(wins) if wins else None
    m.largest_loss = min(losses) if losses else None
    m.avg_r = (sum(rs) / len(rs)) if rs else None

    if not losses:
        m.profit_factor = None
        m.notes.append(
            "Profit factor is undefined: there are no losing trades yet. "
            "It is shown as 'n/a' rather than infinity."
        )
    elif abs(m.gross_loss) < EPS:
        m.profit_factor = None
        m.notes.append("Profit factor is undefined because gross loss is effectively zero.")
    else:
        m.profit_factor = m.gross_profit / abs(m.gross_loss)

    if scratches:
        m.notes.append(f"{len(scratches)} trade(s) closed at exactly break-even and count as scratches, not wins.")
    if len(nets) < 10:
        m.notes.append(
            f"Only {len(nets)} closed trade(s). Statistics from such a small sample "
            f"are not meaningful evidence of edge."
        )

    # streaks
    m.max_consecutive_wins, m.max_consecutive_losses = _streaks(nets)

    # equity curve / drawdown (realised basis)
    equity, peak, max_dd, max_dd_pct = drawdown_from_pnl(nets, starting_balance)
    m.max_drawdown = max_dd
    m.max_drawdown_pct = max_dd_pct
    m.peak_equity = peak
    m.trough_equity = peak - max_dd if max_dd > 0 else peak
    m.__dict__["_equity_curve"] = equity
    return m


def equity_curve(nets: Sequence[float], starting_balance: float) -> List[float]:
    """Realised equity after each closed trade."""
    out = [float(starting_balance)]
    running = float(starting_balance)
    for n in nets:
        running += float(n)
        out.append(running)
    return out


def drawdown_from_pnl(
    nets: Sequence[float], starting_balance: float
) -> tuple:
    """Return (equity_curve, peak_equity, max_drawdown, max_drawdown_pct)."""
    curve = equity_curve(nets, starting_balance)
    peak = curve[0]
    max_dd = 0.0
    max_dd_pct = 0.0
    for value in curve:
        peak = max(peak, value)
        dd = peak - value
        if dd > max_dd:
            max_dd = dd
            max_dd_pct = (dd / peak * 100.0) if peak > EPS else 0.0
    return curve, peak, max_dd, max_dd_pct


def _streaks(nets: Sequence[float]) -> tuple:
    """(max consecutive wins, max consecutive losses)."""
    best_w = best_l = cur_w = cur_l = 0
    for n in nets:
        if n > EPS:
            cur_w += 1
            cur_l = 0
            best_w = max(best_w, cur_w)
        elif n < -EPS:
            cur_l += 1
            cur_w = 0
            best_l = max(best_l, cur_l)
        else:
            cur_w = cur_l = 0
    return best_w, best_l


def describe(m: PerformanceMetrics) -> List[str]:
    """Plain-language lines for the UI's 'how to read this' panel."""
    out = []
    if m.closed_trades == 0:
        return ["No closed trades yet, so every performance statistic is unavailable."]
    out.append(
        f"{m.closed_trades} closed trade(s): {m.wins} win(s), {m.losses} loss(es), "
        f"{m.scratches} break-even."
    )
    if m.win_rate is not None:
        out.append(f"Win rate {m.win_rate * 100:.1f}% - wins divided by all closed trades.")
    if m.profit_factor is None:
        out.append("Profit factor: unavailable (no losing trades yet).")
    else:
        out.append(
            f"Profit factor {m.profit_factor:.2f} - gross profit {m.gross_profit:,.2f} "
            f"divided by gross loss {abs(m.gross_loss):,.2f}."
        )
    out.append(
        f"Max drawdown {m.max_drawdown:,.2f} ({m.max_drawdown_pct:.2f}%) - the largest "
        f"peak-to-trough fall in realised equity."
    )
    if m.avg_win is not None and m.avg_loss is not None:
        out.append(f"Average win {m.avg_win:,.2f} vs average loss {m.avg_loss:,.2f}.")
    if m.avg_r is not None:
        out.append(f"Average realised R {m.avg_r:+.2f} (1R = the amount risked on a trade).")
    out.append(f"Total simulated fees paid: {m.total_fees:,.2f}.")
    out.extend(m.notes)
    return out


__all__ = ["PerformanceMetrics", "compute_metrics", "equity_curve",
           "drawdown_from_pnl", "describe"]
