"""
STOCKSENSE UI Helpers
Reusable component factories for the enterprise-grade dashboard.
"""

from __future__ import annotations
from typing import Optional
import math


# ─────────────────────────── Design tokens ────────────────────────────────
COLORS = {
    "primary":   "#2563EB",
    "bg":        "#F8FAFC",
    "card":      "#FFFFFF",
    "text":      "#0F172A",
    "muted":     "#64748B",
    "border":    "#E2E8F0",
    "high":      "#DC2626",
    "high_bg":   "#FEF2F2",
    "high_bd":   "#FECACA",
    "med":       "#D97706",
    "med_bg":    "#FFFBEB",
    "med_bd":    "#FDE68A",
    "low":       "#16A34A",
    "low_bg":    "#F0FDF4",
    "low_bd":    "#BBF7D0",
    "neutral_bg":"#F1F5F9",
    "neutral_bd":"#CBD5E1",
    "neutral":   "#475569",
}

FONT = "'DM Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"


def pluralize(n: int, singular: str, plural: str | None = None) -> str:
    """Return '1 product' or '4 products'."""
    return f"{n:,} {singular if n == 1 else (plural or singular + 's')}"


def risk_colors(tier: str) -> tuple[str, str, str]:
    """Returns (text_color, bg_color, border_color) for a tier string."""
    t = tier.upper()
    if t == "ACT_NOW":
        return COLORS["high"], COLORS["high_bg"], COLORS["high_bd"]
    if t == "WATCH":
        return COLORS["med"], COLORS["med_bg"], COLORS["med_bd"]
    return COLORS["low"], COLORS["low_bg"], COLORS["low_bd"]


def risk_label(tier: str) -> str:
    t = tier.upper()
    if t == "ACT_NOW": return "High Risk"
    if t == "WATCH":   return "Watch"
    return "Safe"


# ─────────────────────────── CSS ──────────────────────────────────────────
GLOBAL_CSS = f"""
<style>
/* ── Google Font ─────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

/* ── Reset Streamlit chrome ─────────────────── */
#MainMenu, footer, header {{ display: none !important; }}
[data-testid="stDecoration"] {{ display: none !important; }}
[data-testid="stToolbar"] {{ display: none !important; }}

/* ── Page base ───────────────────────────────── */
.stApp, [data-testid="stAppViewContainer"], .main {{
    background-color: #F4F7FB !important;
    font-family: {FONT} !important;
    color: {COLORS['text']};
}}
.block-container {{
    padding: 1.35rem clamp(1rem, 2.2vw, 2.25rem) 2.5rem !important;
    max-width: 1680px !important;
}}

/* ── Shared widget treatment ─────────────────── */
[data-testid="stButton"] button {{
    min-height: 2.25rem;
    border-radius: 8px;
    font-family: {FONT} !important;
    font-size: .82rem;
    font-weight: 600;
    transition: background .15s ease, border-color .15s ease, color .15s ease;
}}
[data-testid="stPlotlyChart"] {{
    background: #FFFFFF;
    border: 1px solid #E5EAF1;
    border-radius: 12px;
    padding: .35rem .5rem .1rem;
    box-shadow: 0 2px 8px rgba(15, 23, 42, .035);
}}
[data-testid="stTextInput"] input,
[data-testid="stSelectbox"] div[data-baseweb="select"] {{
    border-radius: 8px !important;
}}

/* ── Top navigation bar ──────────────────────── */
.ss-topbar {{
    background: {COLORS['card']};
    border-bottom: 1px solid {COLORS['border']};
    padding: 0.85rem 2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: sticky;
    top: 0;
    z-index: 999;
    box-shadow: 0 1px 3px rgba(15,23,42,.06);
}}
.ss-logo {{
    font-family: {FONT};
    font-size: 1.3rem;
    font-weight: 800;
    color: {COLORS['text']};
    letter-spacing: -0.5px;
    line-height: 1;
}}
.ss-logo-accent {{ color: {COLORS['primary']}; }}
.ss-logo-sub {{
    font-size: 0.78rem;
    color: {COLORS['muted']};
    font-weight: 400;
    margin-top: 2px;
}}
.ss-topbar-right {{
    display: flex;
    align-items: center;
    gap: 1rem;
}}
.ss-ts {{
    font-size: 0.78rem;
    color: {COLORS['muted']};
}}

/* ── Main content wrapper ────────────────────── */
.ss-main {{
    padding: 1.5rem 2rem 3rem;
}}

/* ── Store chip row ──────────────────────────── */
.ss-chip-row {{
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-bottom: 1.5rem;
}}
.ss-chip {{
    background: {COLORS['card']};
    border: 1px solid {COLORS['border']};
    border-radius: 999px;
    padding: 3px 10px;
    font-size: 0.76rem;
    color: {COLORS['muted']};
    font-weight: 500;
    white-space: nowrap;
}}
.ss-chip b {{ color: {COLORS['text']}; }}

/* ── Forecast calibration banner ─────────────── */
.ss-banner {{
    background: #EFF6FF;
    border: 1px solid #CFE0FF;
    border-radius: 10px;
    padding: 0.75rem 1rem;
    font-size: 0.82rem;
    color: #355070;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 1.25rem;
}}

/* ── KPI card ────────────────────────────────── */
.ss-kpi {{
    background: {COLORS['card']};
    border: 1px solid #E5EAF1;
    border-radius: 10px;
    padding: 1rem 1.1rem .9rem;
    box-shadow: 0 2px 8px rgba(15,23,42,.035);
    border-left: 3px solid transparent;
    cursor: pointer;
    transition: box-shadow .15s, transform .12s;
}}
.ss-kpi:hover {{
    box-shadow: 0 5px 16px rgba(15,23,42,.08);
    transform: translateY(-2px);
}}
.ss-kpi-label {{
    font-size: 0.73rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: .6px;
    color: {COLORS['muted']};
    margin-bottom: .35rem;
}}
.ss-kpi-num {{
    font-size: 2.15rem;
    font-weight: 800;
    line-height: 1;
    font-variant-numeric: tabular-nums;
    margin-bottom: .2rem;
}}
.ss-kpi-desc {{
    font-size: 0.78rem;
    color: {COLORS['muted']};
    font-weight: 500;
}}
.ss-kpi-active {{ box-shadow: 0 0 0 2px {COLORS['primary']}; }}

/* ── Risk pill badge ─────────────────────────── */
.ss-badge {{
    display: inline-flex;
    align-items: center;
    gap: 4px;
    border-radius: 999px;
    padding: 2px 9px;
    font-size: 0.76rem;
    font-weight: 600;
    white-space: nowrap;
    border: 1px solid;
}}

/* ── Section header ──────────────────────────── */
.ss-section-hd {{
    font-size: 1rem;
    font-weight: 700;
    color: {COLORS['text']};
    margin: 1.5rem 0 .6rem;
    display: flex;
    align-items: center;
    gap: .4rem;
}}
.ss-section-hd .ss-sub {{
    font-size: 0.8rem;
    font-weight: 400;
    color: {COLORS['muted']};
}}

/* ── Insight card (WHY / WHAT) ───────────────── */
.ss-insight {{
    background: {COLORS['card']};
    border: 1px solid {COLORS['border']};
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin-bottom: 1rem;
}}
.ss-insight-title {{
    font-size: 0.82rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .5px;
    color: {COLORS['muted']};
    margin-bottom: .5rem;
}}
.ss-insight-body {{
    font-size: 0.88rem;
    color: {COLORS['text']};
    line-height: 1.55;
}}

/* ── Progress bar (days of stock) ────────────── */
.ss-bar-wrap {{
    background: {COLORS['neutral_bg']};
    border-radius: 999px;
    height: 6px;
    width: 100%;
    overflow: hidden;
    margin-top: 3px;
}}
.ss-bar-fill {{
    height: 6px;
    border-radius: 999px;
}}

/* ── Metric row inside Product page ─────────── */
.ss-metric {{
    background: {COLORS['card']};
    border: 1px solid {COLORS['border']};
    border-radius: 10px;
    padding: 1rem 1.25rem;
}}
.ss-metric-lbl {{
    font-size: 0.74rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: .5px;
    color: {COLORS['muted']};
    margin-bottom: .3rem;
}}
.ss-metric-val {{
    font-size: 1.75rem;
    font-weight: 800;
    line-height: 1;
    font-variant-numeric: tabular-nums;
    color: {COLORS['text']};
}}
.ss-metric-sub {{
    font-size: 0.76rem;
    color: {COLORS['muted']};
    margin-top: .25rem;
}}

/* ── CTA button override ─────────────────────── */
.ss-btn {{
    display: inline-block;
    background: {COLORS['primary']};
    color: #fff !important;
    font-weight: 600;
    font-size: 0.85rem;
    padding: 0.5rem 1.2rem;
    border-radius: 8px;
    border: none;
    cursor: pointer;
    text-decoration: none;
}}

/* ── Tab style override ──────────────────────── */
[data-testid="stTabs"] [role="tablist"] {{
    gap: 0;
    border-bottom: 1px solid {COLORS['border']};
}}
[data-testid="stTabs"] button[role="tab"] {{
    font-family: {FONT} !important;
    font-size: 0.86rem !important;
    font-weight: 500 !important;
    color: {COLORS['muted']} !important;
    padding: .55rem 1rem !important;
    border-radius: 0 !important;
    border-bottom: 2px solid transparent !important;
}}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {{
    color: {COLORS['primary']} !important;
    border-bottom: 2px solid {COLORS['primary']} !important;
    font-weight: 600 !important;
}}

/* ── DataFrame table tweaks ──────────────────── */
[data-testid="stDataFrame"] {{
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid {COLORS['border']};
}}

/* ── Streamlit selectbox, text_input ─────────── */
[data-testid="stSelectbox"] > label,
[data-testid="stTextInput"] > label {{
    font-size: 0.8rem !important;
    color: {COLORS['muted']} !important;
    font-weight: 600 !important;
}}

/* ── Sidebar ─────────────────────────────────── */
section[data-testid="stSidebar"] {{
    background: #111B2E !important;
    border-right: 1px solid #243149 !important;
    width: 260px !important;
    min-width: 260px !important;
    padding: 0 !important;
}}
section[data-testid="stSidebar"] > div:first-child {{
    padding: 0 !important;
}}
section[data-testid="stSidebar"] [data-testid="stButton"] {{
    padding: 0 .65rem;
    margin: .18rem 0;
}}
section[data-testid="stSidebar"] [data-testid="stButton"] button {{
    justify-content: flex-start;
    min-height: 2.55rem;
    padding: .55rem .8rem;
    border: 1px solid transparent;
    background: transparent;
    color: #C3CDDC;
    text-align: left;
}}
section[data-testid="stSidebar"] [data-testid="stButton"] button:hover {{
    background: #1A2940;
    border-color: #2B3B55;
    color: #FFFFFF;
}}
section[data-testid="stSidebar"] [data-testid="stButton"] button[kind="primary"] {{
    background: #203A5D;
    border-color: #2B527F;
    color: #F3F8FF;
}}
/* Hide default Streamlit sidebar collapse button chrome */
[data-testid="collapsedControl"] {{
    color: #64748B !important;
    top: 1rem !important;
}}

/* Sidebar selectbox label */
section[data-testid="stSidebar"] [data-testid="stSelectbox"] > label {{
    color: #94A3B8 !important;
    font-size: 0.7rem !important;
    text-transform: uppercase;
    letter-spacing: .6px;
}}
/* Sidebar selectbox input */
section[data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] {{
    background: #1A2940 !important;
    border-color: #34445F !important;
    border-radius: 8px !important;
    color: #F8FAFC !important;
}}
section[data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] * {{
    color: #F8FAFC !important;
    background: #1A2940 !important;
}}
/* Sidebar nav buttons */
section[data-testid="stSidebar"] .ss-nav-btn {{
    display: flex;
    align-items: center;
    gap: .65rem;
    width: 100%;
    padding: .6rem .85rem;
    border-radius: 8px;
    font-size: .875rem;
    font-weight: 500;
    color: #94A3B8;
    cursor: pointer;
    transition: background .12s, color .12s;
    text-decoration: none;
    border: none;
    background: transparent;
    margin-bottom: 2px;
}}
section[data-testid="stSidebar"] .ss-nav-btn:hover {{
    background: #1E293B;
    color: #F8FAFC;
}}
section[data-testid="stSidebar"] .ss-nav-btn.active {{
    background: #1E3A5F;
    color: #93C5FD;
    font-weight: 600;
}}
/* Sidebar divider */
.ss-sb-divider {{
    border: none;
    border-top: 1px solid #1E293B;
    margin: .75rem 0;
}}
/* Sidebar section label */
.ss-sb-lbl {{
    font-size: .68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .8px;
    color: #475569;
    padding: 0 .85rem;
    margin: .5rem 0 .3rem;
}}
/* Sidebar stat mini card */
.ss-sb-stat {{
    background: #1E293B;
    border-radius: 8px;
    padding: .6rem .85rem;
    margin-bottom: .4rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}}
.ss-sb-stat-lbl {{
    font-size: .78rem;
    color: #94A3B8;
    font-weight: 500;
}}
.ss-sb-stat-val {{
    font-size: .9rem;
    font-weight: 700;
    font-variant-numeric: tabular-nums;
}}

/* scrollbar thin */
::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: {COLORS['bg']}; }}
::-webkit-scrollbar-thumb {{ background: {COLORS['border']}; border-radius: 3px; }}
section[data-testid="stSidebar"]::-webkit-scrollbar-track {{ background: #0F172A; }}
section[data-testid="stSidebar"]::-webkit-scrollbar-thumb {{ background: #1E293B; }}

@media (max-width: 720px) {{
    .block-container {{ padding: 1rem .85rem 2rem !important; }}
    .ss-kpi {{ padding: .85rem .9rem; }}
    .ss-kpi-num {{ font-size: 1.8rem; }}
    .ss-banner {{ align-items: flex-start; line-height: 1.5; }}
}}
</style>
"""


# ─────────────────────────── Component functions ───────────────────────────

def kpi_card(label: str, value: int, description: str,
             color: str, active: bool = False) -> str:
    active_cls = "ss-kpi-active" if active else ""
    return f"""
<div class="ss-kpi {active_cls}" style="border-left-color:{color}">
  <div class="ss-kpi-label">{label}</div>
  <div class="ss-kpi-num" style="color:{color}">{value:,}</div>
  <div class="ss-kpi-desc">{description}</div>
</div>"""


def status_badge(tier: str, label: Optional[str] = None) -> str:
    tc, bg, bd = risk_colors(tier)
    dot = "●"
    text = label or risk_label(tier)
    return (f'<span class="ss-badge" style="color:{tc};background:{bg};border-color:{bd}">'
            f'<span style="font-size:.6rem">{dot}</span>{text}</span>')


def section_header(title: str, subtitle: str = "") -> str:
    sub = f'<span class="ss-sub">— {subtitle}</span>' if subtitle else ""
    return f'<div class="ss-section-hd">{title} {sub}</div>'


def insight_box(title: str, body: str) -> str:
    return f"""
<div class="ss-insight">
  <div class="ss-insight-title">{title}</div>
  <div class="ss-insight-body">{body}</div>
</div>"""


def days_bar(days: float, max_days: float = 14) -> str:
    pct = min(max(days / max_days, 0), 1) * 100
    color = COLORS["high"] if days < 3 else (COLORS["med"] if days < 7 else COLORS["low"])
    days_str = f"{days:.1f}d" if days < 99 else "14d+"
    return f"""
<div style="font-size:.8rem;font-weight:600;color:{color};">{days_str}</div>
<div class="ss-bar-wrap">
  <div class="ss-bar-fill" style="width:{pct:.0f}%;background:{color};"></div>
</div>"""
