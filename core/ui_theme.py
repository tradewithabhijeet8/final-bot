"""
core/ui_theme.py
================
Dark RED "trading terminal" styling and small layout helpers.

All CSS is injected inline so the dashboard looks identical when run locally and
on Streamlit Community Cloud, with no external stylesheet to fail.

Colour system (v1.1.0, red theme)
---------------------------------
The palette is deliberately small so the screen reads as one product instead of
a bag of widgets:

    BG / PANEL / PANEL_ALT   near-black surfaces with a red cast
    BORDER                   hairline separators
    ACCENT                   crimson - the single brand colour
    UP   (green)             rising candle, profit, "ok"
    DOWN (red)               falling candle, loss, "bad"
    AMBER                    warnings only
    SLATE                    neutral information

Candles use the conventional green-up / red-down scheme (v1.3.0, on request):
green and red are reserved for direction and outcome, and nothing else in the
interface uses them, so a green candle can only ever mean "up" and a red number
can only ever mean "loss".
"""

from __future__ import annotations

from typing import Optional

import streamlit as st

BG = "#0b0708"
PANEL = "#150e10"
PANEL_ALT = "#100b0c"
BORDER = "#2b1a1e"
BORDER_SOFT = "#221418"
TEXT = "#f4e9e9"
MUTED = "#a4898d"
UP = "#22c55e"          # rising / profit / ok (green)
DOWN = "#dc2626"        # falling / loss / bad
ACCENT = "#e11d2e"      # brand crimson
ACCENT_DEEP = "#8f1220"
SLATE = "#94a3b8"       # neutral info
AMBER = "#f59e0b"       # warnings
BLUE = SLATE            # back-compat alias: the old blue accent is now slate

MONO = "'JetBrains Mono', 'SFMono-Regular', Consolas, 'Liberation Mono', monospace"

CSS = f"""
<style>
/* ---------- page chrome ---------- */
html, body, [class*="css"] {{
    font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
}}
.stApp {{
    background-color: {BG};
    background-image:
        radial-gradient(1100px 420px at 12% -8%, rgba(225,29,46,0.13), transparent 62%),
        radial-gradient(900px 380px at 96% 0%, rgba(143,18,32,0.12), transparent 60%);
    background-attachment: fixed;
    color: {TEXT};
}}
#MainMenu, footer {{ visibility: hidden; }}
.block-container {{ padding-top: 0.9rem; padding-bottom: 2.2rem; max-width: 1680px; }}

/* ---------- typography ---------- */
h1, h2, h3, h4 {{ color: {TEXT}; font-weight: 600; letter-spacing: -0.01em; }}
h4 {{ text-transform: uppercase; letter-spacing: 0.10em; font-size: 0.82rem;
      color: {MUTED}; margin-top: 0.4rem; }}

/* ---------- masthead ---------- */
.masthead {{
    display: flex; align-items: center; justify-content: space-between;
    gap: 14px; flex-wrap: wrap;
    padding: 14px 18px; margin-bottom: 12px;
    background: linear-gradient(100deg, rgba(225,29,46,0.20) 0%, rgba(21,14,16,0.92) 46%, {PANEL} 100%);
    border: 1px solid {BORDER};
    border-left: 3px solid {ACCENT};
    border-radius: 12px;
    box-shadow: 0 10px 34px rgba(0,0,0,0.45);
}}
.masthead-brand {{ display: flex; align-items: center; gap: 12px; }}
.masthead-mark {{
    width: 38px; height: 38px; border-radius: 10px; flex: 0 0 auto;
    display: flex; align-items: center; justify-content: center;
    background: linear-gradient(145deg, {ACCENT}, {ACCENT_DEEP});
    color: #fff; font-weight: 800; font-size: 1.15rem;
    box-shadow: 0 0 0 1px rgba(255,255,255,0.10) inset, 0 6px 18px rgba(225,29,46,0.32);
}}
.masthead-title {{ font-size: 1.22rem; font-weight: 700; letter-spacing: -0.02em; line-height: 1.15; }}
.masthead-sub {{ font-size: 0.74rem; color: {MUTED}; letter-spacing: 0.04em; margin-top: 2px; }}

/* ---------- cards ---------- */
.card {{
    background: linear-gradient(180deg, {PANEL} 0%, {PANEL_ALT} 100%);
    border: 1px solid {BORDER};
    border-left: 3px solid {ACCENT};
    border-radius: 11px;
    padding: 13px 15px;
    margin-bottom: 12px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.32);
}}
.card-title {{
    font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.11em;
    color: {MUTED}; margin-bottom: 6px; font-weight: 600;
}}
.card-value {{
    font-size: 1.42rem; font-weight: 650; line-height: 1.15;
    font-family: {MONO}; letter-spacing: -0.02em;
}}
.card-sub {{ font-size: 0.73rem; color: {MUTED}; margin-top: 5px; }}
.pos {{ color: {UP}; }}
.neg {{ color: {DOWN}; }}
.neu {{ color: {TEXT}; }}
.warn {{ color: {AMBER}; }}

/* ---------- status pills ---------- */
.pill {{
    display: inline-flex; align-items: center; gap: 6px;
    padding: 3px 10px; border-radius: 999px; font-size: 0.71rem; font-weight: 650;
    letter-spacing: 0.05em; text-transform: uppercase;
    border: 1px solid {BORDER}; background: {PANEL}; color: {TEXT};
    font-family: {MONO};
}}
.pill-ok   {{ color: {UP};    border-color: rgba(34,197,94,0.40);  background: rgba(34,197,94,0.10); }}
.pill-bad  {{ color: #ff8b8b; border-color: rgba(220,38,38,0.45);   background: rgba(220,38,38,0.12); }}
.pill-warn {{ color: {AMBER}; border-color: rgba(245,158,11,0.40);  background: rgba(245,158,11,0.10); }}
.pill-info {{ color: {SLATE}; border-color: rgba(148,163,184,0.32); background: rgba(148,163,184,0.08); }}
.dot {{ width: 7px; height: 7px; border-radius: 50%; background: currentColor; }}
.pill-live {{
    color: #fff; border-color: {ACCENT};
    background: linear-gradient(90deg, rgba(225,29,46,0.85), rgba(143,18,32,0.85));
}}
.pill-live .dot {{ background: #fff; animation: pulse 1.6s ease-in-out infinite; }}
@keyframes pulse {{
    0%, 100% {{ opacity: 1; transform: scale(1); }}
    50%      {{ opacity: 0.35; transform: scale(0.78); }}
}}

/* ---------- simulation banner ---------- */
.sim-banner {{
    background: linear-gradient(90deg, rgba(245,158,11,0.16), rgba(245,158,11,0.04));
    border: 1px solid rgba(245,158,11,0.42);
    border-left: 4px solid {AMBER};
    color: #ffd48a;
    padding: 8px 14px; border-radius: 8px;
    font-weight: 700; letter-spacing: 0.08em; font-size: 0.8rem;
    text-align: center; margin-bottom: 12px;
}}

/* ---------- legacy header row (kept for compatibility) ---------- */
.term-header {{
    display: flex; flex-wrap: wrap; align-items: center; gap: 10px;
    justify-content: space-between; margin-bottom: 8px;
}}
.term-title {{ font-size: 1.35rem; font-weight: 700; letter-spacing: -0.02em; }}
.term-sub {{ font-size: 0.78rem; color: {MUTED}; }}
.pill-row {{ display: flex; flex-wrap: wrap; gap: 7px; align-items: center; }}

/* ---------- copyright footer ---------- */
.term-footer {{
    margin-top: 22px; padding-top: 12px;
    border-top: 1px solid {BORDER};
    text-align: center;
}}
.term-footer-mark {{
    font-size: 0.86rem; font-weight: 700; letter-spacing: 0.06em;
    background: linear-gradient(90deg, #ff6b6b, {ACCENT}, #ff9aa6);
    -webkit-background-clip: text; background-clip: text; color: transparent;
}}
.term-footer-line {{ font-size: 0.72rem; color: {MUTED}; margin-top: 4px; letter-spacing: 0.03em; }}
.term-footer-fine {{ font-size: 0.68rem; color: #6f5b5e; margin-top: 6px; }}

/* ---------- streamlit widget harmonisation ---------- */
div[data-testid="stMetric"] {{
    background: linear-gradient(180deg, {PANEL} 0%, {PANEL_ALT} 100%);
    border: 1px solid {BORDER}; border-left: 3px solid {ACCENT};
    border-radius: 11px; padding: 12px 14px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.28);
}}
div[data-testid="stMetricLabel"] p {{
    font-size: 0.68rem !important; text-transform: uppercase;
    letter-spacing: 0.11em; color: {MUTED} !important; font-weight: 600;
}}
div[data-testid="stMetricValue"] {{
    font-size: 1.32rem; font-weight: 650; color: {TEXT}; font-family: {MONO};
}}
div[data-testid="stMetricDelta"] {{ font-size: 0.74rem; font-family: {MONO}; }}

.stButton > button, .stDownloadButton > button {{
    background: linear-gradient(180deg, {PANEL} 0%, {PANEL_ALT} 100%);
    color: {TEXT}; border: 1px solid {BORDER};
    border-radius: 9px; font-weight: 600; letter-spacing: 0.02em;
    transition: all .16s ease;
}}
.stButton > button:hover, .stDownloadButton > button:hover {{
    border-color: {ACCENT}; color: #fff;
    background: linear-gradient(180deg, #20141a 0%, #170f12 100%);
    box-shadow: 0 0 0 1px rgba(225,29,46,0.25), 0 6px 18px rgba(225,29,46,0.18);
}}
.stButton > button[kind="primary"] {{
    background: linear-gradient(180deg, {ACCENT} 0%, {ACCENT_DEEP} 100%);
    border-color: rgba(255,120,120,0.45); color: #fff;
}}
.stButton > button[kind="primary"]:hover {{
    background: linear-gradient(180deg, #f0243a 0%, {ACCENT_DEEP} 100%);
}}
div[data-testid="stExpander"] {{
    border: 1px solid {BORDER}; border-radius: 10px; background: {PANEL_ALT};
}}
div[data-testid="stExpander"] summary {{ font-weight: 600; letter-spacing: 0.02em; }}
div[data-testid="stSidebar"] {{
    background-color: #0d0809;
    border-right: 1px solid {BORDER};
    box-shadow: 12px 0 34px rgba(0,0,0,0.35);
}}
div[data-testid="stSidebar"] * {{ font-size: 0.85rem; }}
div[data-testid="stSidebar"] h4 {{ color: {ACCENT}; }}
hr {{ border-color: {BORDER}; }}

/* tabs: crisp, uppercase, crimson underline on the active tab */
.stTabs [data-baseweb="tab-list"] {{ gap: 4px; border-bottom: 1px solid {BORDER}; }}
.stTabs [data-baseweb="tab"] {{
    background: transparent; color: {MUTED}; border-radius: 8px 8px 0 0;
    padding: 9px 16px; font-size: 0.78rem; font-weight: 650;
    text-transform: uppercase; letter-spacing: 0.09em;
}}
.stTabs [data-baseweb="tab"]:hover {{ color: {TEXT}; background: rgba(225,29,46,0.07); }}
.stTabs [aria-selected="true"] {{
    color: #fff; background: linear-gradient(180deg, rgba(225,29,46,0.22), rgba(21,14,16,0.6));
    box-shadow: inset 0 -2px 0 0 {ACCENT};
}}

div[data-testid="stDataFrame"] {{
    border: 1px solid {BORDER}; border-radius: 10px; background: {PANEL_ALT};
}}
.stAlert {{ border-radius: 10px; }}
.stAlert code {{ background: #1d1215; color: #ffb3b3; }}
code {{ background: #1d1215; color: #ffb3b3; padding: 1px 5px; border-radius: 4px; }}
div[data-testid="stSidebar"] .stRadio label, div[data-testid="stCheckbox"] label {{ letter-spacing: 0.01em; }}
a {{ color: #ff8b8b; }}

/* ---------- v1.2.0: 24h stats strip ---------- */
.tstrip {{
    display: grid; grid-template-columns: minmax(190px, 1.35fr) repeat(auto-fit, minmax(118px, 1fr));
    gap: 1px; background: {BORDER}; border: 1px solid {BORDER}; border-radius: 12px;
    overflow: hidden; margin-bottom: 12px; box-shadow: 0 8px 26px rgba(0,0,0,0.34);
}}
.tstrip > div {{ background: linear-gradient(180deg, {PANEL} 0%, {PANEL_ALT} 100%); padding: 11px 14px; }}
.tstrip-sym {{ border-left: 3px solid {ACCENT}; }}
.tstrip-base {{ font-size: 0.80rem; font-weight: 700; letter-spacing: 0.04em; color: {TEXT}; }}
.tstrip-last {{ font-family: {MONO}; font-size: 1.30rem; font-weight: 650; line-height: 1.2;
                letter-spacing: -0.02em; margin-top: 2px; }}
.tstrip-chg {{ font-family: {MONO}; font-size: 0.76rem; margin-top: 3px; }}
.tstrip-label {{ font-size: 0.63rem; text-transform: uppercase; letter-spacing: 0.11em;
                 color: {MUTED}; font-weight: 600; }}
.tstrip-value {{ font-family: {MONO}; font-size: 0.95rem; margin-top: 5px; }}
.tstrip-note {{ font-size: 0.70rem; color: {MUTED}; margin: -6px 0 12px 2px; }}

/* ---------- v1.2.0: quick-action row ---------- */
.st-key-qbar .stButton > button, .st-key-qbar .stDownloadButton > button {{
    padding: 5px 10px; font-size: 0.74rem; letter-spacing: 0.03em; min-height: 0;
    border-radius: 9px; background: {PANEL}; border: 1px solid {BORDER};
}}
.st-key-qbar .stButton > button:hover, .st-key-qbar .stDownloadButton > button:hover {{
    border-color: {ACCENT}; color: {TEXT};
}}

/* ---------- v1.2.0: chart toolbar ---------- */
.st-key-charttoolbar .stButton > button {{
    padding: 2px 4px; font-size: 0.71rem; font-family: {MONO}; min-height: 0;
    letter-spacing: 0.02em;
}}
.st-key-charttoolbar .stButton > button[kind="primary"] {{ background: {ACCENT}; color: #fff; }}
.toolbar-hint {{ font-size: 0.68rem; color: {MUTED}; text-align: right; padding-top: 6px; }}

/* ---------- v1.2.0: OHLC read-out ---------- */
.readout {{
    font-family: {MONO}; font-size: 0.79rem; color: {TEXT};
    background: {PANEL_ALT}; border: 1px solid {BORDER}; border-radius: 9px;
    padding: 6px 12px; margin: 2px 0 8px 0; display: flex; gap: 14px; flex-wrap: wrap;
    align-items: center;
}}
.readout .rd-k {{ color: {MUTED}; margin-right: 4px; font-size: 0.69rem; letter-spacing: 0.08em; }}
.readout .rd-tag {{ font-size: 0.63rem; text-transform: uppercase; letter-spacing: 0.11em;
                    color: {MUTED}; border-left: 1px solid {BORDER}; padding-left: 12px; }}

/* ---------- v1.2.0: right-hand rail ---------- */
.railfit .stTabs {{ margin-top: 2px; }}
.railcard {{
    background: linear-gradient(180deg, {PANEL} 0%, {PANEL_ALT} 100%);
    border: 1px solid {BORDER}; border-radius: 11px; padding: 11px 13px; margin-bottom: 11px;
}}
.railcard-title {{ font-size: 0.66rem; text-transform: uppercase; letter-spacing: 0.12em;
                   color: {MUTED}; font-weight: 700; margin-bottom: 8px;
                   display: flex; justify-content: space-between; gap: 8px; }}
.railrow {{ display: flex; justify-content: space-between; gap: 10px; font-size: 0.76rem;
            padding: 3px 0; border-bottom: 1px dashed {BORDER_SOFT}; }}
.railrow:last-child {{ border-bottom: none; }}
.railrow .rl {{ color: {MUTED}; }}
.railrow .rv {{ font-family: {MONO}; text-align: right; }}
.rail-empty {{ font-size: 0.74rem; color: {MUTED}; padding: 6px 0 2px 0; }}
.rail-big {{ font-family: {MONO}; font-size: 1.05rem; font-weight: 650; }}

/* ---------- v1.2.0: compact tables (movers / watchlist) ---------- */
.mini {{ width: 100%; border-collapse: collapse; font-size: 0.74rem; }}
.mini th {{ font-size: 0.62rem; text-transform: uppercase; letter-spacing: 0.09em;
            color: {MUTED}; text-align: right; padding: 4px 6px; border-bottom: 1px solid {BORDER};
            font-weight: 600; }}
.mini th:first-child, .mini td:first-child {{ text-align: left; }}
.mini td {{ padding: 4px 6px; text-align: right; font-family: {MONO};
            border-bottom: 1px dashed {BORDER_SOFT}; }}
.mini tr:last-child td {{ border-bottom: none; }}
.mini .sym {{ font-family: 'Inter', sans-serif; font-weight: 650; letter-spacing: 0.03em; }}
.mini .lbl {{ font-family: 'Inter', sans-serif; font-weight: 500; letter-spacing: 0.02em; }}
.mini-note {{ font-size: 0.68rem; color: {MUTED}; margin-top: 6px; }}
</style>
"""


def inject() -> None:
    """Inject the terminal theme once per session."""
    st.markdown(CSS, unsafe_allow_html=True)


def pill(text: str, kind: str = "info", dot: bool = True) -> str:
    """Return HTML for a status pill.  kind: ok | bad | warn | info | live."""
    d = '<span class="dot"></span>' if dot else ""
    return f'<span class="pill pill-{kind}">{d}{text}</span>'


def pills(items) -> None:
    """Render a row of pills from (text, kind) tuples."""
    html = '<div class="pill-row">' + "".join(pill(t, k) for t, k in items) + "</div>"
    st.markdown(html, unsafe_allow_html=True)


def sim_banner(text: str) -> None:
    st.markdown(f'<div class="sim-banner">{text}</div>', unsafe_allow_html=True)


def masthead(title: str, subtitle: str, items=(), mark: str = "₿") -> None:
    """
    Terminal masthead: brand mark + title on the left, status pills on the right.

    Replaces the older `header()` layout with a single bordered plate so the top
    of the page reads as a product header rather than loose text.
    """
    pill_html = "".join(pill(t, k) for t, k in items)
    st.markdown(
        f"""
        <div class="masthead">
          <div class="masthead-brand">
            <div class="masthead-mark">{mark}</div>
            <div>
              <div class="masthead-title">{title}</div>
              <div class="masthead-sub">{subtitle}</div>
            </div>
          </div>
          <div class="pill-row">{pill_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def card(label: str, value: str, sub: Optional[str] = None, tone: str = "neu") -> str:
    """Return HTML for a metric card."""
    sub_html = f'<div class="card-sub">{sub}</div>' if sub else ""
    return (
        f'<div class="card"><div class="card-title">{label}</div>'
        f'<div class="card-value {tone}">{value}</div>{sub_html}</div>'
    )


def card_row(cards) -> None:
    """Render a responsive row of card() HTML strings."""
    st.markdown(
        '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));'
        f'gap:10px;margin-bottom:6px;">{"".join(cards)}</div>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# v1.2.0 components: ticker strip, compact tables, rail cards, read-out
# --------------------------------------------------------------------------- #

def _esc(text: str) -> str:
    """Escape text that reaches the HTML helper output."""
    return (str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def ticker_strip(base: str, last: str, change: str, tone: str, cells,
                 *, badge: str = "", note: str = "") -> None:
    """
    24h statistics strip above the tabs.

    `cells` is a list of (label, value, tone).  Values are strings on purpose:
    the caller decides what "-" means (a field this venue does not publish),
    and the strip never renders a 0 where data is simply missing.
    """
    badge_html = (f'<span class="pill pill-info" style="margin-left:8px;">{_esc(badge)}</span>'
                  if badge else "")
    body = "".join(
        f'<div><div class="tstrip-label">{_esc(label)}</div>'
        f'<div class="tstrip-value {tone_i}">{_esc(value)}</div></div>'
        for label, value, tone_i in cells
    )
    st.markdown(
        f'<div class="tstrip">'
        f'<div class="tstrip-sym">'
        f'<div class="tstrip-base">{_esc(base)}{badge_html}</div>'
        f'<div class="tstrip-last">{_esc(last)}</div>'
        f'<div class="tstrip-chg {tone}">{_esc(change)}</div>'
        f'</div>{body}</div>'
        + (f'<div class="tstrip-note">{note}</div>' if note else ""),
        unsafe_allow_html=True,
    )


def mini_table(headers, rows) -> None:
    """
    Compact HTML table.  `rows` is a list of lists of (text, tone) pairs, so a
    single cell can be red without colouring the whole row.
    """
    head = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    body = ""
    for row in rows:
        tds = "".join(
            f'<td class="{tone}">{_esc(text)}</td>' for text, tone in row
        )
        body += f"<tr>{tds}</tr>"
    if not body:
        body = f'<tr><td colspan="{len(headers)}" style="text-align:left;">No rows.</td></tr>'
    st.markdown(f'<table class="mini"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>',
                unsafe_allow_html=True)


def readout(items, *, tag: str = "") -> None:
    """One-line OHLC read-out: items = (letter, value, tone) triples."""
    cells = "".join(
        f'<span><span class="rd-k">{_esc(k)}</span><span class="{tone}">{_esc(v)}</span></span>'
        for k, v, tone in items
    )
    tag_html = f'<span class="rd-tag">{_esc(tag)}</span>' if tag else ""
    st.markdown(f'<div class="readout">{cells}{tag_html}</div>', unsafe_allow_html=True)


def rail_card(title: str, rows, *, badge: str = "", empty: str = "",
              big: Optional[tuple] = None) -> None:
    """
    Small right-hand rail card: a title plus label/value rows.

    `rows` is (label, value, tone); `big` is an optional (value, tone) shown
    large at the top (used for the position's P/L).  `empty` replaces the body
    when there is nothing to show, so a rail card never displays invented zeroes.
    """
    badge_html = f'<span>{_esc(badge)}</span>' if badge else ""
    body = ""
    if big is not None:
        value, tone = big
        body += f'<div class="rail-big {tone}">{_esc(value)}</div>'
    for label, value, tone in rows:
        body += (f'<div class="railrow"><span class="rl">{_esc(label)}</span>'
                 f'<span class="rv {tone}">{_esc(value)}</span></div>')
    if empty:
        # The explanation is shown even when there are rows: a flat account
        # still has a balance to report, and the reader deserves the "why".
        body += f'<div class="rail-empty">{_esc(empty)}</div>'
    elif not body:
        body = '<div class="rail-empty">Nothing to report.</div>' 
    st.markdown(
        f'<div class="railcard"><div class="railcard-title"><span>{_esc(title)}</span>'
        f'{badge_html}</div>{body}</div>',
        unsafe_allow_html=True,
    )


def tone_for_pct(value: Optional[float]) -> str:
    """Tone for a percentage: 'neu' when it is missing or exactly flat."""
    if value is None or not value:
        return "neu"
    return "pos" if value > 0 else "neg"


def header(title: str, subtitle: str, items) -> None:
    """Terminal header: title on the left, status pills on the right."""
    st.markdown(
        f"""
        <div class="term-header">
          <div>
            <div class="term-title">{title}</div>
            <div class="term-sub">{subtitle}</div>
          </div>
          <div class="pill-row">{''.join(pill(t, k) for t, k in items)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def footer(copyright_line: str, app_line: str = "", fine_print: str = "") -> None:
    """Page footer carrying the author/copyright line."""
    parts = [f'<div class="term-footer-mark">{copyright_line}</div>']
    if app_line:
        parts.append(f'<div class="term-footer-line">{app_line}</div>')
    if fine_print:
        parts.append(f'<div class="term-footer-fine">{fine_print}</div>')
    st.markdown(f'<div class="term-footer">{"".join(parts)}</div>', unsafe_allow_html=True)


def sidebar_credit(copyright_line: str, version: str = "") -> None:
    """Compact credit block pinned to the bottom of the sidebar."""
    st.markdown(
        f"""
        <div style="margin-top:18px;padding-top:12px;border-top:1px solid {BORDER};
                    text-align:center;">
          <div style="font-size:0.76rem;font-weight:700;letter-spacing:0.05em;color:{ACCENT};">
            {copyright_line}
          </div>
          <div style="font-size:0.66rem;color:{MUTED};margin-top:4px;">
            {version}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def money(value: Optional[float], digits: int = 2, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if v != v:  # NaN
        return "n/a"
    sign = "+" if (signed and v > 0) else ("" if not signed else "")
    return f"{sign}${v:,.{digits}f}"


def tone_for(value: Optional[float], invert: bool = False) -> str:
    if value is None:
        return "neu"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "neu"
    if v != v:
        return "neu"
    if abs(v) < 1e-12:
        return "neu"
    positive_is_good = not invert
    return ("pos" if v > 0 else "neg") if positive_is_good else ("neg" if v > 0 else "pos")


def pct(value: Optional[float], digits: int = 2, signed: bool = True) -> str:
    if value is None:
        return "n/a"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if v != v:
        return "n/a"
    return f"{v:+.{digits}f}%" if signed else f"{v:.{digits}f}%"


__all__ = ["inject", "pill", "pills", "sim_banner", "masthead", "card", "card_row",
           "header", "footer", "sidebar_credit", "money", "tone_for", "pct", "CSS",
           "BG", "PANEL", "PANEL_ALT", "BORDER", "BORDER_SOFT", "TEXT", "MUTED",
           "UP", "DOWN", "ACCENT", "ACCENT_DEEP", "SLATE", "AMBER", "BLUE", "MONO"]
