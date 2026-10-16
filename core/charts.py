"""
core/charts.py
==============
Plotly chart builder for the trading terminal.

Two stacked panels sharing one x-axis:
    1. price  - candlesticks, the live (still-forming) candle, prior swing
                levels, trade markers, and the entry / stop-loss / take-profit
                lines of the open trade
    2. RSI    - with 30 / 50 / 70 reference levels

The EMA 20/50 overlays and the volume panel were REMOVED from the chart on
request (v1.1.0).  The strategy still computes both internally - the EMA
crossover is condition 1 of the four-condition entry - so signals and backtest
results are unchanged.  Nothing is hidden from the strategy, only from the plot.

Colours: candles use the conventional **green-up / red-down** scheme (v1.3.0,
on request), while the surrounding terminal chrome stays dark red.  Green and
red are the two colours humans read fastest on a price chart, so they are
reserved for direction and never used for anything else in the figure.

Optional add-on overlays (Phase 2): price-action zones and event markers are
drawn ONLY when the caller passes `zones` / `pa_events`.  With those arguments
omitted the figure is exactly what it always was.

Timestamps are rendered in UTC and labelled as such, because mixing local time
into a trading chart is a classic source of confusion.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .config import TIMEFRAME_MS

# Palette (matches ui_theme.py) - professional dark RED terminal theme.
BG = "#0b0708"          # page / plot background, near-black with a red cast
PANEL = "#150e10"       # panel background
GRID = "#2b1a1e"        # gridlines
TEXT = "#f4e9e9"        # primary text
MUTED = "#a4898d"       # secondary text
UP = "#22c55e"          # rising candle / profit / long: green
DOWN = "#dc2626"        # falling candle / loss / short: red
UP_FILL = "rgba(34,197,94,0.20)"      # translucent body fill for the LIVE candle
DOWN_FILL = "rgba(220,38,38,0.20)"
ACCENT = "#e11d2e"      # primary brand red
SLATE = "#94a3b8"       # neutral accent (entry line, info)
AMBER = "#f59e0b"       # warnings, RSI line
BLUE = SLATE            # back-compat alias: "info" accent is now slate
#: Swing levels are NOT direction-coded: gold for the high, soft blue for the
#: low.  Green and red are reserved for candles / outcomes, so a glance at a
#: green line can only ever mean "up" (see `test_green_and_red_are_reserved...`).
SWING_HIGH_COLOR = "#e0b252"
SWING_LOW_COLOR = "#9ec3e0"
LIVE_CANDLE_COLOR = ACCENT

# Phase-2 price-action zone palette (add-on overlays; drawn only when asked for)
ZONE_COLORS = {
    # Warm/red family throughout: bullish zones read pale, bearish zones read
    # crimson, so direction is legible at a glance on the dark-red background.
    "fvg_bull": "#efd9d0", "fvg_bear": "#e11d2e",
    "ob_bull": "#d9b8ae", "ob_bear": "#ff5c6c",
    "sd_demand": "#f5e6dc", "sd_supply": "#a3121f",
    "liq_bsl": "#ff9aa6", "liq_ssl": "#e0b252",
}
ZONE_LABELS = {
    "fvg_bull": "FVG \u2191", "fvg_bear": "FVG \u2193",
    "ob_bull": "OB \u2191", "ob_bear": "OB \u2193",
    "sd_demand": "DEMAND", "sd_supply": "SUPPLY",
    "liq_bsl": "BSL", "liq_ssl": "SSL",
}
# Higher-timeframe zones reuse the same hues with a dashed outline, so you can
# tell at a glance which timeframe a box came from.
HTF_ZONE_COLOR = "#f0e2e2"
HTF_ZONE_LABEL_PREFIX = "HTF "

EVENT_SYMBOLS = {
    "sweep_and_reclaim_bull": "x-thin", "sweep_and_reclaim_bear": "x-thin",
    "choch_bull_early": "diamond-open", "choch_bear_early": "diamond-open",
    "mss_bull": "diamond", "mss_bear": "diamond",
    "fvg_bull_form": "square-open", "fvg_bear_form": "square-open",
    "ob_bull_form": "circle-open", "ob_bear_form": "circle-open",
    "breakout_bull": "triangle-up-open", "breakout_bear": "triangle-down-open",
    "retest_hold_bull": "triangle-up", "retest_hold_bear": "triangle-down",
    "failed_break_bull": "x", "failed_break_bear": "x",
}


def _utc(ts) -> pd.Series:
    return pd.to_datetime(pd.Series(np.asarray(ts, dtype="float64")), unit="ms", utc=True)


def build_chart(
    df: pd.DataFrame,
    *,
    symbol: str = "BTC/USD",
    timeframe: str = "1h",
    exchange_label: str = "",
    show_swings: bool = True,
    show_signals: bool = True,
    open_trade: Optional[Dict[str, Any]] = None,
    forming: Optional[Dict[str, float]] = None,
    show_live_candle: bool = True,
    height: int = 760,
    title_suffix: str = "",
    zones: Optional[List[Any]] = None,
    pa_events: Optional[List[Dict[str, Any]]] = None,
    show_zones: bool = True,
    show_pa_events: bool = True,
    max_zones: int = 25,
) -> go.Figure:
    """
    Build the terminal chart from an enriched OHLCV frame.

    `df` is expected to be the output of `strategy.generate_signals` (so it may
    contain rsi / ema / signal columns); plain OHLCV also works.
    """
    if df is None or df.empty:
        fig = go.Figure()
        fig.update_layout(
            template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG,
            title="No candle data available", height=height,
        )
        return fig

    d = df.copy().sort_values("timestamp").reset_index(drop=True)
    x = _utc(d["timestamp"])

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.045,
        row_heights=[0.74, 0.26],
        specs=[[{"secondary_y": False}], [{"secondary_y": False}]],
        subplot_titles=(
            f"{symbol} · {timeframe}{title_suffix}",
            "RSI (14)",
        ),
    )

    # ---- 1. candlesticks ------------------------------------------------- #
    fig.add_trace(go.Candlestick(
        x=x, open=d["open"], high=d["high"], low=d["low"], close=d["close"],
        name="OHLC",
        increasing_line_color=UP, increasing_fillcolor=UP,
        decreasing_line_color=DOWN, decreasing_fillcolor=DOWN,
        opacity=0.95, whiskerwidth=0.4,
        hoverlabel=dict(namelength=0),
    ), row=1, col=1)

    # ---- prior swing levels --------------------------------------------- #
    if show_swings and "prior_swing_high" in d.columns:
        fig.add_trace(go.Scatter(
            x=x, y=d["prior_swing_high"], name="Prior swing high", mode="lines",
            line=dict(color=SWING_HIGH_COLOR, width=1, dash="dot"), opacity=0.75,
            hovertemplate="Prior swing high<br>%{y:,.2f}<extra></extra>",
        ), row=1, col=1)
    if show_swings and "prior_swing_low" in d.columns:
        fig.add_trace(go.Scatter(
            x=x, y=d["prior_swing_low"], name="Prior swing low", mode="lines",
            line=dict(color=SWING_LOW_COLOR, width=1, dash="dot"), opacity=0.75,
            hovertemplate="Prior swing low<br>%{y:,.2f}<extra></extra>",
        ), row=1, col=1)

    # ---- price-action zones (optional add-on) --------------------------- #
    if show_zones and zones:
        _add_pa_zones(fig, d, zones, max_zones=max_zones)
    if show_pa_events and pa_events:
        _add_pa_events(fig, d, pa_events)

    # ---- signal markers -------------------------------------------------- #
    if show_signals and "signal" in d.columns:
        longs = d[d["signal"] == 1]
        shorts = d[d["signal"] == -1]
        if not longs.empty:
            fig.add_trace(go.Scatter(
                x=_utc(longs["timestamp"]), y=longs["low"] * 0.9985,
                name="BUY signal", mode="markers",
                marker=dict(symbol="triangle-up", size=13, color=UP,
                            line=dict(width=1, color="#3a2226")),
                customdata=np.stack([longs["close"], longs.get("rsi", pd.Series([np.nan] * len(longs)))], axis=-1),
                hovertemplate=(
                    "<b>BUY</b> (closed candle)<br>close %{customdata[0]:,.2f}"
                    "<br>RSI %{customdata[1]:.1f}<extra></extra>"
                ),
            ), row=1, col=1)
        if not shorts.empty:
            fig.add_trace(go.Scatter(
                x=_utc(shorts["timestamp"]), y=shorts["high"] * 1.0015,
                name="SELL signal", mode="markers",
                marker=dict(symbol="triangle-down", size=13, color=DOWN,
                            line=dict(width=1, color="#2b0a0d")),
                customdata=np.stack([shorts["close"], shorts.get("rsi", pd.Series([np.nan] * len(shorts)))], axis=-1),
                hovertemplate=(
                    "<b>SELL</b> (closed candle)<br>close %{customdata[0]:,.2f}"
                    "<br>RSI %{customdata[1]:.1f}<extra></extra>"
                ),
            ), row=1, col=1)

    # ---- open trade levels ---------------------------------------------- #
    if open_trade:
        _add_level(fig, x, open_trade.get("entry_price"), "Entry", SLATE, "solid")
        _add_level(fig, x, open_trade.get("stop_loss"), "Stop-loss", DOWN, "dash")
        _add_level(fig, x, open_trade.get("take_profit"), "Take-profit", UP, "dash")

    # ---- live (still-forming) candle ------------------------------------- #
    # Drawn from the exchange's in-progress bucket and clearly marked as live.
    # It is DISPLAY ONLY: it never enters `df`, so no indicator, condition or
    # signal can see a candle that has not closed.
    if forming and forming.get("timestamp"):
        if show_live_candle:
            _add_live_candle(fig, forming, TIMEFRAME_MS.get(timeframe, 3_600_000))
        else:
            fx = _utc([forming["timestamp"]])
            fig.add_trace(go.Scatter(
                x=fx, y=[forming.get("close")], name="Forming candle (not traded)",
                mode="markers",
                marker=dict(symbol="circle-open", size=9, color=AMBER),
                hovertemplate=("Forming candle<br>open %{customdata[0]:,.2f}<br>"
                               "close %{y:,.2f}<br>NOT used for signals<extra></extra>"),
                customdata=[[forming.get("open", float("nan"))]],
            ), row=1, col=1)

    # ---- 2. RSI ---------------------------------------------------------- #
    if "rsi" in d.columns:
        fig.add_trace(go.Scatter(
            x=x, y=d["rsi"], name="RSI", mode="lines",
            line=dict(color=AMBER, width=1.5),
            hovertemplate="RSI %{y:.2f}<extra></extra>",
        ), row=2, col=1)
        for level, colour in ((70, DOWN), (50, MUTED), (30, SWING_HIGH_COLOR)):
            fig.add_hline(y=level, line=dict(color=colour, width=1, dash="dot"),
                          opacity=0.55, row=2, col=1)
        fig.update_yaxes(range=[0, 100], row=2, col=1)

    # ---- layout ---------------------------------------------------------- #
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG,
        plot_bgcolor=PANEL,
        height=height,
        margin=dict(l=58, r=18, t=54, b=38),
        font=dict(family="Inter, Segoe UI, Arial, sans-serif", size=12, color=TEXT),
        bargap=0.12,
        hovermode="x unified",
        dragmode="zoom",
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.015, xanchor="left", x=0,
            font=dict(size=11, color=MUTED), bgcolor="rgba(0,0,0,0)",
        ),
        xaxis_rangeslider_visible=False,
        title=dict(
            text=(f"{exchange_label} · {symbol} · {timeframe} · UTC"
                  if exchange_label else f"{symbol} · {timeframe} · UTC"),
            font=dict(size=14, color=TEXT), x=0.01, xanchor="left",
        ),
    )
    # Candlestick panels must not show the built-in rangeslider.
    fig.update_xaxes(rangeslider_visible=False)
    fig.update_xaxes(showgrid=True, gridcolor=GRID, zeroline=False, color=MUTED)
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False, color=MUTED)
    fig.update_yaxes(title_text="Price (USDT)", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    # Plotly's Layout is an attribute object, not a dict: `.get()` does not exist.
    for ann in (fig.layout.annotations or ()):
        ann.font = dict(size=12, color=MUTED)
    return fig


def overlay_live_candle(fig: go.Figure, forming: Optional[Dict[str, float]],
                        timeframe: str = "1h") -> go.Figure:
    """
    Draw the still-forming candle onto an EXISTING figure, in place.

    The app caches the (expensive) closed-candle figure and, on each live poll,
    copies it and calls this - about 13 ms instead of a ~220 ms full rebuild.
    Passing None leaves the figure untouched.
    """
    if not forming or not forming.get("timestamp"):
        return fig
    _add_live_candle(fig, forming, TIMEFRAME_MS.get(timeframe, 3_600_000))
    return fig


def _add_live_candle(fig: go.Figure, forming: Dict[str, float], tf_ms: int) -> None:
    """
    Draw the in-progress candle as a hollow, dashed-outline bar plus a LIVE tag.

    Shapes (not a Candlestick trace) are used deliberately: a one-point
    candlestick gets an unpredictable body width in Plotly, and the hollow
    styling makes "this candle is not closed yet" obvious at a glance.
    """
    ts = float(forming["timestamp"])
    x_dt = pd.to_datetime(ts, unit="ms", utc=True)
    half = pd.Timedelta(milliseconds=max(tf_ms * 0.30, 1.0))
    o = float(forming.get("open") or forming.get("close") or float("nan"))
    h = float(forming.get("high") or o)
    l = float(forming.get("low") or o)
    c = float(forming.get("close") or o)
    if any(v != v for v in (o, h, l, c)):
        return
    rising = c >= o
    colour = UP if rising else DOWN
    fill = UP_FILL if rising else DOWN_FILL

    # wick
    fig.add_shape(type="line", x0=x_dt, x1=x_dt, y0=l, y1=h, layer="above",
                  line=dict(color=colour, width=1.6, dash="dot"), opacity=0.95,
                  row=1, col=1)
    # body (hollow: transparent-ish fill + dashed outline)
    fig.add_shape(type="rect", x0=x_dt - half, x1=x_dt + half,
                  y0=min(o, c), y1=max(o, c), layer="above",
                  fillcolor=fill, line=dict(color=colour, width=1.6, dash="dash"),
                  row=1, col=1)
    # hoverable marker so the numbers are still readable on hover
    fig.add_trace(go.Scatter(
        x=[x_dt], y=[c], name="LIVE candle (forming)", mode="markers",
        marker=dict(symbol="circle-open", size=8, color=LIVE_CANDLE_COLOR,
                    line=dict(width=1.6, color=LIVE_CANDLE_COLOR)),
        hovertemplate=(
            "<b>LIVE - candle still forming</b><br>open %{customdata[0]:,.2f}"
            "<br>high %{customdata[1]:,.2f}<br>low %{customdata[2]:,.2f}"
            "<br>last %{y:,.2f}<br>vol %{customdata[3]:,.2f}"
            "<br><i>excluded from every signal</i><extra></extra>"
        ),
        customdata=[[o, h, l, float(forming.get("volume") or 0.0)]],
    ), row=1, col=1)
    fig.add_annotation(x=x_dt, y=h, text="LIVE", showarrow=False, yshift=8,
                       xanchor="center", yanchor="bottom",
                       font=dict(size=9, color=LIVE_CANDLE_COLOR, family="Inter, sans-serif"),
                       bgcolor="rgba(11,7,8,0.75)", bordercolor=LIVE_CANDLE_COLOR,
                       borderwidth=1, borderpad=2, row=1, col=1)


def _zone_dict(z: Any) -> Dict[str, Any]:
    return z.to_dict() if hasattr(z, "to_dict") else dict(z)


def _add_pa_zones(fig: go.Figure, d: pd.DataFrame, zones: List[Any],
                  max_zones: int = 25) -> None:
    """
    Draw tracked price-action zones as rectangles on the price panel.

    A zone that is still alive is drawn through to the most recent closed
    candle; a finished zone stops where it was invalidated / filled / expired.
    Shapes (not traces) are used so the unified hover stays clean.
    """
    if d.empty:
        return
    x_all = _utc(d["timestamp"])
    last_x = x_all.iloc[-1]
    first_x = x_all.iloc[0]

    norm = []
    for z in zones:
        zd = _zone_dict(z)
        if zd.get("top") is None or zd.get("bottom") is None:
            continue
        top, bottom = float(zd["top"]), float(zd["bottom"])
        if not (np.isfinite(top) and np.isfinite(bottom)):
            continue
        created = zd.get("created_ts")
        if created is None or not np.isfinite(float(created)):
            continue
        x0 = _utc([float(created)])[0]
        if x0 > last_x:
            continue
        dead = (zd.get("meta") or {}).get("dead_ts")
        x1 = _utc([float(dead)])[0] if dead and np.isfinite(float(dead)) else last_x
        if x1 < first_x:
            continue
        norm.append((zd, x0, min(x1, last_x), top, bottom))

    # Most recent zones first; cap the count so the chart stays readable.
    norm.sort(key=lambda t: t[1], reverse=True)
    for zd, x0, x1, top, bottom in norm[:max(1, int(max_zones))]:
        kind = str(zd.get("kind", ""))
        is_htf = kind.startswith("htf_")
        base_kind = kind[len("htf_"):] if is_htf else kind
        colour = HTF_ZONE_COLOR if is_htf else ZONE_COLORS.get(kind, MUTED)
        label = (HTF_ZONE_LABEL_PREFIX + ZONE_LABELS.get(base_kind, base_kind)
                 if is_htf else ZONE_LABELS.get(kind, kind))
        dash = "dash" if is_htf else "solid"
        if top - bottom <= 0:
            # liquidity pools are single price levels -> dashed line segment
            fig.add_shape(type="line", xref="x", yref="y", x0=x0, x1=x1, y0=top, y1=top,
                          line=dict(color=colour, width=1.2, dash="dash"),
                          opacity=0.85, row=1, col=1)
        else:
            fig.add_shape(type="rect", xref="x", yref="y", x0=x0, x1=x1, y0=bottom, y1=top,
                          fillcolor=colour, opacity=0.08 if is_htf else 0.14,
                          line=dict(color=colour, width=1, dash=dash), row=1, col=1)
        fig.add_annotation(x=x1, y=top, text=label,
                           showarrow=False, xanchor="left", yanchor="bottom",
                           font=dict(size=9, color=colour), row=1, col=1)


def _add_pa_events(fig: go.Figure, d: pd.DataFrame, events: List[Dict[str, Any]],
                   limit: int = 400) -> None:
    """Mark structural / liquidity events on the price panel."""
    if d.empty:
        return
    lo = d["low"].to_numpy(dtype="float64")
    hi = d["high"].to_numpy(dtype="float64")
    ts_to_pos = {float(t): i for i, t in enumerate(d["timestamp"].to_numpy(dtype="float64"))}

    xs, ys, texts, syms, cols = [], [], [], [], []
    for e in list(events)[-limit:]:
        ts = e.get("ts")
        if ts is None or float(ts) not in ts_to_pos:
            continue
        i = ts_to_pos[float(ts)]
        name = str(e.get("event", ""))
        bullish = name.endswith(("bull", "_up")) or name.startswith("choch_bull")
        level = e.get("level", e.get("top"))
        y = float(level) if level is not None and np.isfinite(float(level)) else \
            (float(lo[i]) if bullish else float(hi[i]))
        xs.append(_utc([float(ts)])[0])
        ys.append(y * (0.998 if bullish else 1.002))
        texts.append(name.replace("_", " "))
        syms.append(EVENT_SYMBOLS.get(name, "circle-open"))
        cols.append(UP if bullish else DOWN)

    if not xs:
        return
    fig.add_trace(go.Scatter(
        x=xs, y=ys, name="Price-action events", mode="markers",
        marker=dict(symbol=syms, size=8, color=cols, line=dict(width=1, color=BG)),
        customdata=[[t] for t in texts],
        hovertemplate="PA event: %{customdata[0]}<br>%{y:,.2f}<extra></extra>",
    ), row=1, col=1)


def _add_level(fig: go.Figure, x: pd.Series, price: Optional[float], label: str,
               color: str, dash: str) -> None:
    if price is None:
        return
    try:
        p = float(price)
    except (TypeError, ValueError):
        return
    if not np.isfinite(p):
        return
    fig.add_trace(go.Scatter(
        x=x, y=[p] * len(x), name=label, mode="lines",
        line=dict(color=color, width=1.4, dash=dash), opacity=0.9,
        hovertemplate=f"{label} {p:,.2f}<extra></extra>",
    ), row=1, col=1)


def build_equity_curve(equity: List[float] | pd.DataFrame, *, starting_balance: float,
                       height: int = 300, title: str = "Equity curve") -> go.Figure:
    """Realised equity over closed trades (or a backtest equity frame)."""
    fig = go.Figure()
    if isinstance(equity, pd.DataFrame):
        if equity.empty:
            return _empty_fig(title, height)
        x = _utc(equity["timestamp"]) if "timestamp" in equity.columns else list(range(len(equity)))
        y = equity["equity"] if "equity" in equity.columns else equity["balance"]
    else:
        if not equity:
            return _empty_fig(title, height)
        x = list(range(len(equity)))
        y = list(equity)

    fig.add_trace(go.Scatter(
        x=x, y=y, mode="lines", name="Equity",
        line=dict(color=BLUE, width=2), fill="tozeroy",
        fillcolor="rgba(59,130,246,0.08)",
        hovertemplate="%{y:,.2f}<extra></extra>",
    ))
    fig.add_hline(y=starting_balance, line=dict(color=MUTED, width=1, dash="dot"),
                  annotation_text=f"Start {starting_balance:,.0f}")
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=PANEL, height=height,
        margin=dict(l=58, r=18, t=42, b=34), title=dict(text=title, font=dict(size=13, color=TEXT)),
        font=dict(size=11, color=TEXT), showlegend=False,
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID, color=MUTED)
    fig.update_yaxes(showgrid=True, gridcolor=GRID, color=MUTED)
    return fig


def build_drawdown_chart(equity: List[float], *, starting_balance: float,
                         height: int = 220) -> go.Figure:
    """Underwater plot: how far below the running peak the equity has been."""
    if not equity:
        return _empty_fig("Drawdown", height)
    arr = np.asarray(equity, dtype="float64")
    peak = np.maximum.accumulate(arr)
    dd = np.where(peak > 0, (arr - peak) / peak * 100.0, 0.0)
    fig = go.Figure(go.Scatter(
        x=list(range(len(dd))), y=dd, mode="lines", name="Drawdown %",
        line=dict(color=DOWN, width=1.5), fill="tozeroy",
        fillcolor="rgba(239,68,68,0.12)",
        hovertemplate="%{y:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=PANEL, height=height,
        margin=dict(l=58, r=18, t=42, b=34),
        title=dict(text="Drawdown from peak (%)", font=dict(size=13, color=TEXT)),
        font=dict(size=11, color=TEXT), showlegend=False,
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID, color=MUTED)
    fig.update_yaxes(showgrid=True, gridcolor=GRID, color=MUTED)
    return fig


def build_exit_reason_chart(breakdown: Dict[str, int], height: int = 260) -> go.Figure:
    if not breakdown:
        return _empty_fig("Exit reasons", height)
    colours = {
        "take_profit": UP, "take_profit_gap": "#16a34a",
        "stop_loss": DOWN, "stop_loss_gap": "#b91c1c",
        "end_of_data": MUTED, "safe_stop_flatten": AMBER,
    }
    labels = list(breakdown.keys())
    values = [breakdown[k] for k in labels]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color=[colours.get(l, BLUE) for l in labels],
        hovertemplate="%{y}: %{x} trade(s)<extra></extra>",
    ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=PANEL, height=height,
        margin=dict(l=120, r=18, t=42, b=34),
        title=dict(text="Exit reason breakdown", font=dict(size=13, color=TEXT)),
        font=dict(size=11, color=TEXT), showlegend=False,
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID, color=MUTED)
    fig.update_yaxes(showgrid=False, color=MUTED)
    return fig


def overlay_price_tag(fig: go.Figure, price: Optional[float], *,
                      label: Optional[str] = None, color: str = ACCENT,
                      digits: int = 2, dim: bool = False,
                      row: int = 1) -> go.Figure:
    """
    Dotted last-price line plus a price tag in the right gutter (in place).

    Same idea as the price label every charting terminal shows: one glance tells
    you where price is against the levels, without reading the axis.  The right
    margin is widened to hold the tag; calling it twice (closed price on the
    cached figure, forming price on the live copy) is idempotent, so the chart
    never jumps when the live candle appears.

    `price=None` or a non-finite value leaves the figure untouched.
    """
    try:
        p = float(price)                                   # type: ignore[arg-type]
    except (TypeError, ValueError):
        return fig
    if not np.isfinite(p):
        return fig

    yref = "y" if row == 1 else f"y{row}"
    fig.add_shape(type="line", xref="paper", x0=0.0, x1=1.0, yref=yref, y0=p, y1=p,
                  layer="above", line=dict(color=color, width=1.0, dash="dot"),
                  opacity=0.45 if dim else 0.85)
    fig.add_annotation(
        xref="paper", x=1.0, xanchor="left", xshift=3, yref=yref, y=p, yanchor="middle",
        text=(label if label is not None else f"{p:,.{digits}f}"), showarrow=False,
        bgcolor=color, bordercolor=color, borderwidth=1, borderpad=3,
        font=dict(size=10, color=BG, family="Inter, sans-serif"),
    )
    current = fig.layout.margin.r if (fig.layout.margin and fig.layout.margin.r) else 0
    fig.update_layout(margin=dict(r=max(74, int(current))))
    return fig


def build_return_ring(value_pct: Optional[float], *, height: int = 232,
                      centre: str = "Total Return", caption: str = "",
                      has_trades: bool = True) -> go.Figure:
    """
    Circular return ring for the Strategy-performance panel.

    The arc length is proportional to the realised return in percentage points
    (clamped at +/-100), so a +0.75% result shows a deliberately SHORT arc: a
    decorative full ring would make a small number look like a large one.  Red
    for a loss, green for a profit, muted outline when there is nothing to
    report yet.
    """
    value = float(value_pct) if value_pct is not None else 0.0
    value = max(-100.0, min(100.0, value))
    colour = (UP if value > 0 else DOWN if value < 0 else MUTED) if has_trades else MUTED
    filled = abs(value) / 100.0
    if not has_trades:
        filled = 0.0
    filled = max(filled, 0.004)          # a hairline arc, never a filled ring at 0
    rest = max(1.0 - filled, 0.004)

    text = f"<b>{value:+.2f}%</b>" if has_trades else "<b>-</b>"
    sub = centre if has_trades else "no closed trades"
    fig = go.Figure(
        go.Pie(
            values=[filled, rest], sort=False, direction="clockwise", rotation=0,
            hole=0.76, textinfo="none", hoverinfo="skip",
            marker=dict(colors=[colour, "#241417"], line=dict(color=PANEL, width=2)),
        )
    )
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=PANEL, plot_bgcolor=PANEL, height=height,
        margin=dict(l=6, r=6, t=10, b=6), showlegend=False,
        annotations=[dict(
            text=f"{text}<br><span style='font-size:10px;color:{MUTED}'>{sub}</span>",
            showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper",
            font=dict(size=23, color=colour, family="Inter, sans-serif"),
        )],
    )
    if caption:
        fig.add_annotation(x=0.5, y=-0.06, xref="paper", yref="paper", showarrow=False,
                           text=caption, font=dict(size=9, color=MUTED, family="Inter, sans-serif"))
    return fig


def _empty_fig(title: str, height: int) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=PANEL, height=height,
        title=dict(text=f"{title} - no data yet", font=dict(size=13, color=MUTED)),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
    )
    return fig


__all__ = ["build_chart", "overlay_live_candle", "overlay_price_tag", "build_return_ring",
           "build_equity_curve", "build_drawdown_chart", "build_exit_reason_chart"]
