"""
Walk-forward evaluation (v1.3.0).

Why this module exists
----------------------
A strategy setting can always be made to look good on the candles it was
chosen from.  The only cheap defence is to split the history in time: pick and
look at one part (the **in-sample** window), then run the SAME settings,
untouched, on a later part the settings never saw (the **out-of-sample**
window).  If a setting only works in-sample, it was fitted to that period and
says nothing about the future.

Deliberate design choices, all of them on the side of honest numbers
--------------------------------------------------------------------
* **Windows are time-ordered and disjoint.**  Rolling folds, no shuffling, no
  random splits - candles are a time series and shuffling them is meaningless.
* **Each window is run independently, starting flat.**  A window is sliced as
  ``history[warm-up : window_end]`` so indicators arrive warm, but no position
  is carried in from before the window.  That removes the "a trade opened just
  before the boundary is quietly counted in both halves" ambiguity.
* **Out-of-sample settings are never re-fitted.**  There is no optimiser here
  at all.  Compute learns nothing from the test window; it only counts what
  would have happened.
* **Small samples are reported as small.**  A window with a handful of trades
  gets an explicit "this is noise" note instead of a headline win rate.
* **No target is promised.**  This module reports what happened.  It cannot and
  does not claim a win rate, and it never will.

Everything here is display / research only: it reads closed candles and calls
the existing backtest engine, and it changes no live behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from .backtest import BacktestConfig, BacktestResult, run_backtest
from .config import StrategyParams
from .indicators import enrich
from .metrics import PerformanceMetrics, compute_metrics

#: Fewer closed trades than this in a window and the numbers are noise.
MIN_MEANINGFUL_TRADES = 10
#: Extra candles added on top of the strategy's own warm-up when slicing, so
#: every indicator is genuinely warm at the first candle of the window.
WARMUP_MARGIN = 10
#: A setting is called "held up" only if at least this share of out-of-sample
#: windows were profitable.  Chosen to be strict, not flattering.
CONSISTENCY_THRESHOLD = 0.6


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

@dataclass
class WalkForwardConfig:
    """How to cut history into in-sample / out-of-sample folds."""

    n_splits: int = 4
    #: Share of each fold used to LOOK at the strategy.  The rest is tested.
    train_ratio: float = 0.7
    #: True => each fold's train window starts at the beginning of history
    #: (anchored / expanding).  False => train rolls forward (default).
    anchored: bool = False

    def validate(self) -> "WalkForwardConfig":
        if not isinstance(self.n_splits, int) or not 1 <= self.n_splits <= 20:
            raise ValueError("n_splits must be an integer between 1 and 20.")
        if not 0.2 <= float(self.train_ratio) <= 0.95:
            raise ValueError("train_ratio must be between 0.2 and 0.95.")
        return self

    @property
    def test_ratio(self) -> float:
        return 1.0 - float(self.train_ratio)

    def to_dict(self) -> Dict[str, Any]:
        return {"n_splits": self.n_splits, "train_ratio": self.train_ratio,
                "anchored": self.anchored}


@dataclass
class Window:
    """One in-sample / out-of-sample pair of candle ranges (index positions)."""

    index: int
    train_start: int
    train_end: int          # exclusive
    test_start: int
    test_end: int           # exclusive

    @property
    def n_train(self) -> int:
        return max(0, self.train_end - self.train_start)

    @property
    def n_test(self) -> int:
        return max(0, self.test_end - self.test_start)

    def label(self) -> str:
        return f"Fold {self.index + 1}"


def build_windows(n_candles: int, cfg: WalkForwardConfig) -> List[Window]:
    """
    Cut `n_candles` history into rolling (or anchored) walk-forward windows.

    Folds are consecutive blocks: fold k covers segment k of history, the first
    `train_ratio` of the segment is in-sample and the remainder is tested.
    Blocks are disjoint, so no candle is counted twice across folds.
    """
    cfg.validate()
    n = int(n_candles)
    if n < 40:
        return []
    seg = n // cfg.n_splits
    if seg < 20:
        return []
    windows: List[Window] = []
    for k in range(cfg.n_splits):
        start = k * seg
        end = n if k == cfg.n_splits - 1 else (k + 1) * seg
        train_end = start + int(round((end - start) * float(cfg.train_ratio)))
        # At least one candle must be left to test on.
        if train_end >= end:
            train_end = end - 1
        if train_end <= start:
            continue
        train_start = 0 if cfg.anchored else start
        windows.append(Window(index=k, train_start=train_start,
                              train_end=train_end, test_start=train_end, test_end=end))
    return windows


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #

@dataclass
class HalfResult:
    """Metrics plus the trades behind them for one half of one fold."""

    metrics: PerformanceMetrics
    trades: List[Dict[str, Any]] = field(default_factory=list)
    candles: int = 0
    start_ts: Optional[int] = None
    end_ts: Optional[int] = None
    signals: int = 0

    @property
    def n_trades(self) -> int:
        return len(self.trades)

    @property
    def thin(self) -> bool:
        return self.n_trades < MIN_MEANINGFUL_TRADES

    def to_dict(self) -> Dict[str, Any]:
        d = self.metrics.to_dict()
        d.update({"n_trades": self.n_trades, "candles": self.candles,
                  "start_ts": self.start_ts, "end_ts": self.end_ts,
                  "signals": self.signals, "thin": self.thin})
        return d


@dataclass
class FoldResult:
    """One walk-forward fold: how it looked, then how it actually went."""

    window: Window
    in_sample: HalfResult
    out_of_sample: HalfResult

    def label(self) -> str:
        return self.window.label()

    def degradation_r(self) -> Optional[float]:
        """Out-of-sample expectancy minus in-sample, in R. None if unknown."""
        a = _exp_r(self.in_sample.metrics)
        b = _exp_r(self.out_of_sample.metrics)
        if a is None or b is None:
            return None
        return float(b) - float(a)

    def to_dict(self) -> Dict[str, Any]:
        return {"fold": self.window.index + 1,
                "train_start": self.window.train_start, "train_end": self.window.train_end,
                "test_start": self.window.test_start, "test_end": self.window.test_end,
                "in_sample": self.in_sample.to_dict(),
                "out_of_sample": self.out_of_sample.to_dict(),
                "degradation_r": self.degradation_r()}


@dataclass
class VariantReport:
    """
    One named set of strategy settings, walked forward through every fold.

    `in_sample` and `out_of_sample` are the concatenated trades of all folds.
    The sum of net PnL, win rate, profit factor and expectancy are exact for
    those trades; the drawdown figure is a concatenation of separately-sized
    windows and is therefore approximate - it is labelled as such.
    """

    name: str
    strategy: StrategyParams
    folds: List[FoldResult] = field(default_factory=list)
    in_sample: Optional[HalfResult] = None
    out_of_sample: Optional[HalfResult] = None
    warnings: List[str] = field(default_factory=list)

    # ---------------- derived, honest summaries ---------------- #
    @property
    def oos_profitable_folds(self) -> int:
        return sum(1 for f in self.folds if (f.out_of_sample.metrics.net_pnl or 0.0) > 0)

    @property
    def n_folds_with_trades(self) -> int:
        return sum(1 for f in self.folds if f.out_of_sample.n_trades > 0)

    @property
    def consistency(self) -> Optional[float]:
        """Share of out-of-sample folds that made money. None if none traded."""
        if not self.n_folds_with_trades:
            return None
        return self.oos_profitable_folds / self.n_folds_with_trades

    @property
    def expectancy_delta_r(self) -> Optional[float]:
        a = _exp_r(self.in_sample.metrics) if self.in_sample else None
        b = _exp_r(self.out_of_sample.metrics) if self.out_of_sample else None
        if a is None or b is None:
            return None
        return float(b) - float(a)

    def verdict(self) -> str:
        """
        A deliberately unglamorous one-liner.  It never says "profitable
        strategy" and never mentions a target win rate: it says whether this
        setting held up on candles it did not see, or whether it only worked
        where it was chosen.
        """
        oos = self.out_of_sample
        ins = self.in_sample
        if oos is None or ins is None or not self.folds:
            return "Not enough history to walk forward."
        if oos.n_trades == 0:
            return ("No out-of-sample trades at all, so there is nothing to judge. "
                    "This is a 'we do not know', not a pass.")
        if oos.n_trades < MIN_MEANINGFUL_TRADES:
            return (f"Only {oos.n_trades} out-of-sample trade(s) in total - far too few to "
                    "judge. Treat every number below as noise.")
        iexp = _exp_r(ins.metrics)
        oexp = _exp_r(oos.metrics)
        cons = self.consistency or 0.0
        if oexp is None:
            return "Out-of-sample expectancy could not be computed."
        if oexp > 0 and cons >= CONSISTENCY_THRESHOLD:
            return (f"Held up out-of-sample: positive expectancy on candles it had not "
                    f"seen, and {self.oos_profitable_folds}/{self.n_folds_with_trades} "
                    f"windows profitable. That is evidence, not a promise - the next "
                    f"{oos.n_trades} trades can still lose.")
        if oexp > 0:
            return (f"Positive out-of-sample expectancy, but only "
                    f"{self.oos_profitable_folds}/{self.n_folds_with_trades} windows were "
                    f"profitable - one good window is carrying the average.")
        if iexp is not None and iexp > 0 >= oexp:
            return (f"Looked better in-sample ({iexp:+.2f}R) than out ({oexp:+.2f}R). That is "
                    f"the classic signature of settings fitted to one period; do not trust "
                    f"the in-sample number.")
        return (f"Lost money out-of-sample ({oexp:+.2f}R per trade over {oos.n_trades} "
                f"trades). On this history these settings did not work.")

    def to_dict(self) -> Dict[str, Any]:
        oos = self.out_of_sample
        return {
            "name": self.name,
            "strategy": self.strategy.to_dict(),
            "folds": [f.to_dict() for f in self.folds],
            "in_sample": self.in_sample.to_dict() if self.in_sample else None,
            "out_of_sample": oos.to_dict() if oos else None,
            "oos_profitable_folds": self.oos_profitable_folds,
            "n_folds_with_trades": self.n_folds_with_trades,
            "consistency": self.consistency,
            "expectancy_delta_r": self.expectancy_delta_r,
            "verdict": self.verdict(),
            "warnings": list(self.warnings),
        }


@dataclass
class WalkForwardReport:
    """The whole run: every variant, every fold, plus the caveats."""

    ok: bool = False
    message: str = ""
    timeframe: str = ""
    candles: int = 0
    windows: List[Window] = field(default_factory=list)
    variants: List[VariantReport] = field(default_factory=list)
    config: Optional[WalkForwardConfig] = None
    warnings: List[str] = field(default_factory=list)

    # ---------------- presentation helpers ---------------- #
    def table_rows(self) -> List[Dict[str, Any]]:
        """
        One row per variant, in-sample beside out-of-sample.

        Formatted as strings with "-" for anything undefined: the UI must never
        print 0.00 where the honest answer is "we do not know".
        """
        rows: List[Dict[str, Any]] = []
        for v in self.variants:
            ins, oos = v.in_sample, v.out_of_sample
            rows.append({
                "Setting": v.name,
                "Trades (in)": str(ins.n_trades) if ins else "-",
                "Win % (in)": _pct(ins.metrics.win_rate) if ins else "-",
                "Expectancy R (in)": _r(_exp_r(ins.metrics)) if ins else "-",
                "Trades (out)": str(oos.n_trades) if oos else "-",
                "Win % (out)": _pct(oos.metrics.win_rate) if oos else "-",
                "Expectancy R (out)": _r(_exp_r(oos.metrics)) if oos else "-",
                "Net PnL (out)": _money(oos.metrics.net_pnl) if oos else "-",
                "Profitable windows": (f"{v.oos_profitable_folds}/{v.n_folds_with_trades}"
                                       if v.n_folds_with_trades else "-"),
            })
        return rows

    def describe(self) -> List[str]:
        """Plain-language lines; the UI shows these above the table."""
        out: List[str] = []
        if self.message:
            out.append(self.message)
        for v in self.variants:
            out.append(f"{v.name}: {v.verdict()}")
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {"ok": self.ok, "message": self.message, "timeframe": self.timeframe,
                "candles": self.candles, "config": self.config.to_dict() if self.config else None,
                "variants": [v.to_dict() for v in self.variants],
                "warnings": list(self.warnings)}


def _pct(value: Optional[float]) -> str:
    """`win_rate` is a 0..1 fraction, so it must be scaled for display."""
    return "-" if value is None else f"{float(value) * 100.0:.1f}%"


def _r(value: Optional[float]) -> str:
    return "-" if value is None else f"{float(value):+.2f}R"


def _money(value: Optional[float]) -> str:
    return "-" if value is None else f"{float(value):+,.2f}"


def _exp_r(metrics: Optional[PerformanceMetrics]) -> Optional[float]:
    """
    Expectancy expressed in R: the average R multiple per closed trade.

    That is exactly what `PerformanceMetrics.avg_r` already is, so this reads
    it rather than recomputing anything - one definition, one number.
    """
    if metrics is None:
        return None
    value = getattr(metrics, "avg_r", None)
    return None if value is None else float(value)


# --------------------------------------------------------------------------- #
# Running the walk-forward
# --------------------------------------------------------------------------- #

def _slice_for(df: pd.DataFrame, start: int, end: int, warmup: int) -> Tuple[pd.DataFrame, int]:
    """
    Return the frame to hand the engine for window [start, end) plus the index
    of the window's first candle inside that frame.

    The slice reaches BACK `warmup` candles so indicators are warm, but no
    position is carried in: the engine starts flat at the window boundary.
    """
    lo = max(0, int(start) - int(warmup))
    return df.iloc[lo:int(end)], int(start) - lo


def _run_half(df: pd.DataFrame, cfg: BacktestConfig, start: int, end: int,
              warmup: int) -> HalfResult:
    """Run the engine over one window and return its metrics and trades."""
    frame, offset = _slice_for(df, start, end, warmup)
    if len(frame) < 2:
        empty = compute_metrics([], cfg.starting_balance)
        return HalfResult(metrics=empty)
    sub_cfg = BacktestConfig(
        exchange=cfg.exchange, symbol=cfg.symbol, timeframe=cfg.timeframe,
        candles=len(frame), starting_balance=cfg.starting_balance,
        fill_model=cfg.fill_model, strategy=cfg.strategy, risk=cfg.risk,
    )
    result = run_backtest(frame, sub_cfg)
    trades = list(result.trades or [])
    # Only trades ENTERED inside the window count.  With the warm-up slice this
    # should already be true; the check keeps that guarantee explicit.
    if offset > 0 and trades:
        first_ts = frame.iloc[offset]["timestamp"] if "timestamp" in frame.columns else None
        if first_ts is not None:
            trades = [t for t in trades if float(t.get("entry_ts", 0)) >= float(first_ts)]
    metrics = compute_metrics(trades, cfg.starting_balance)
    stamps = _stamps(frame.iloc[offset:])
    return HalfResult(metrics=metrics, trades=trades, candles=max(0, end - start),
                      start_ts=stamps[0] if stamps else None,
                      end_ts=stamps[1] if stamps else None,
                      signals=int(getattr(result, "signals_seen", 0) or 0))


def _stamps(frame: pd.DataFrame) -> Optional[Tuple[int, int]]:
    if frame is None or len(frame) == 0 or "timestamp" not in frame.columns:
        return None
    try:
        lo = int(float(frame["timestamp"].iloc[0]))
        hi = int(float(frame["timestamp"].iloc[-1]))
    except (TypeError, ValueError):
        return None
    return lo, hi


def walk_forward(
    df: pd.DataFrame,
    base_config: BacktestConfig,
    variants: Sequence[Tuple[str, StrategyParams]],
    cfg: Optional[WalkForwardConfig] = None,
    *,
    enrich_first: bool = True,
) -> WalkForwardReport:
    """
    Walk every variant forward through time and report in-sample vs out-of-sample.

    `variants` is a sequence of ``(label, StrategyParams)`` pairs - typically the
    current settings and the same settings with filters enabled, so the user can
    see whether the filters actually help on candles they were not chosen from.

    The engine is called through `run_backtest`, unmodified.  Nothing in this
    module alters live strategy behaviour.
    """
    cfg = (cfg or WalkForwardConfig()).validate()
    report = WalkForwardReport(config=cfg, timeframe=base_config.timeframe)

    if df is None or len(df) == 0:
        report.message = "No candles to walk forward over."
        return report
    if "timestamp" not in df.columns:
        report.message = "Candles have no timestamp column; time ordering cannot be trusted."
        return report
    if not df["timestamp"].is_monotonic_increasing:
        df = df.sort_values("timestamp").reset_index(drop=True)
        report.warnings.append("Candles were not in ascending time order; sorted before use.")

    frame = enrich(df) if enrich_first else df
    report.candles = len(frame)

    windows = build_windows(len(frame), cfg)
    if len(windows) < 2:
        report.message = (f"Not enough closed candles ({len(frame)}) for {cfg.n_splits} "
                          f"walk-forward folds. Request more history.")
        return report
    report.windows = windows

    if not variants:
        report.message = "No settings were supplied to evaluate."
        return report

    for label, params in variants:
        try:
            params.validate()
        except Exception as exc:                        # noqa: BLE001 - reported, not raised
            report.warnings.append(f"'{label}' has invalid settings and was skipped: {exc}")
            continue

        warmup = int(params.min_history()) + WARMUP_MARGIN
        variant = VariantReport(name=label, strategy=params)

        is_trades: List[Dict[str, Any]] = []
        oos_trades: List[Dict[str, Any]] = []
        for w in windows:
            fold_cfg = BacktestConfig(
                exchange=base_config.exchange, symbol=base_config.symbol,
                timeframe=base_config.timeframe, candles=len(frame),
                starting_balance=base_config.starting_balance,
                fill_model=base_config.fill_model, strategy=params, risk=base_config.risk,
            )
            try:
                ins = _run_half(frame, fold_cfg, w.train_start, w.train_end, warmup)
                oos = _run_half(frame, fold_cfg, w.test_start, w.test_end, warmup)
            except Exception as exc:                    # noqa: BLE001
                variant.warnings.append(f"{w.label()} could not be evaluated: {exc}")
                continue
            variant.folds.append(FoldResult(window=w, in_sample=ins, out_of_sample=oos))
            is_trades.extend(ins.trades)
            oos_trades.extend(oos.trades)

        if not variant.folds:
            report.warnings.append(f"'{label}' produced no evaluable folds.")
            continue

        variant.in_sample = HalfResult(
            metrics=compute_metrics(is_trades, base_config.starting_balance), trades=is_trades,
            candles=sum(f.window.n_train for f in variant.folds))
        variant.out_of_sample = HalfResult(
            metrics=compute_metrics(oos_trades, base_config.starting_balance), trades=oos_trades,
            candles=sum(f.window.n_test for f in variant.folds))

        if variant.out_of_sample.thin:
            variant.warnings.append(
                f"'{label}' has only {variant.out_of_sample.n_trades} out-of-sample "
                f"trade(s) across {len(variant.folds)} window(s). The strategy is "
                f"deliberately selective; more history is needed before any of this means "
                f"anything.")
        report.variants.append(variant)

    if not report.variants:
        report.message = "No setting could be evaluated."
        return report

    report.ok = True
    report.message = (
        f"Walk-forward over {len(frame):,} candles in {len(windows)} rolling folds "
        f"(first {cfg.train_ratio:.0%} of each block in-sample, the rest tested). "
        f"Each window runs flat from its own start, and out-of-sample settings are "
        f"never re-fitted."
    )
    return report


def session_breakdown(trades: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Session / weekday performance for a list of closed trades.

    Delegates to core/sessions.py so the live, backtest and walk-forward views
    cannot drift apart.  Kept here as the single entry point for the UI.
    """
    from . import sessions as sessions_mod
    return sessions_mod.full_breakdown(trades)
