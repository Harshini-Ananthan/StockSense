"""
STOCKSENSE Chart Factory
All Plotly charts with a consistent enterprise theme.
No function renames — all existing call-sites still work.
"""

from __future__ import annotations
from typing import Optional
import pandas as pd
import plotly.graph_objects as go

# ── Shared theme ────────────────────────────────────────────────────────────
FONT = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"
BG   = "rgba(0,0,0,0)"

C = {
    "high":    "#DC2626",
    "med":     "#D97706",
    "low":     "#16A34A",
    "primary": "#2563EB",
    "neutral": "#94A3B8",
    "grid":    "#F1F5F9",
    "text":    "#0F172A",
    "muted":   "#64748B",
}

def _base_layout(title: str = "", height: int = 320) -> dict:
    return dict(
        title=dict(text=f"<b>{title}</b>" if title else "",
                   font=dict(family=FONT, size=14, color=C["text"]),
                   x=0, xanchor="left", pad=dict(b=4)),
        font=dict(family=FONT, color=C["text"], size=11),
        paper_bgcolor=BG, plot_bgcolor=BG,
        margin=dict(l=12, r=12, t=44 if title else 20, b=32),
        height=height,
        legend=dict(font=dict(size=11), bgcolor="rgba(0,0,0,0)",
                    orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        hovermode="closest",
    )

def _style_axes(fig: go.Figure) -> go.Figure:
    fig.update_xaxes(showgrid=True, gridcolor=C["grid"], linecolor=C["grid"],
                     tickfont=dict(size=10, color=C["muted"]))
    fig.update_yaxes(showgrid=True, gridcolor=C["grid"], linecolor=C["grid"],
                     tickfont=dict(size=10, color=C["muted"]))
    return fig


# ── 1. Stock vs Demand grouped horizontal bar chart ─────────────────────────
def plot_stock_vs_demand(items: list[dict], using_reorder: bool = False) -> go.Figure:
    """
    Grouped horizontal bar: Current Stock (blue) vs Expected Demand/Reorder Level (orange).
    Highlight products where stock < demand in red.
    items: list of dicts with keys: name, closing, demand
    """
    if not items:
        fig = go.Figure()
        fig.update_layout(**_base_layout("Stock vs Expected Demand"), height=220)
        fig.add_annotation(text="No products to display.", showarrow=False,
                           font=dict(color=C["muted"], size=13), yref="paper", y=0.5)
        return fig

    names   = [it["name"]    for it in items]
    stock   = [it["closing"] for it in items]
    demand  = [it["demand"]  for it in items]
    colors  = [C["high"] if s < d else C["primary"] for s, d in zip(stock, demand)]

    demand_label = "Reorder Level (forecast pending)" if using_reorder else "7-Day Expected Demand"
    title_suffix = "<br><span style='font-size:10px;font-weight:400;color:#64748B'>"
    title_suffix += " Using reorder level — forecast being calibrated</span>" if using_reorder else ""

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Current Stock", y=names, x=stock, orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        hovertemplate="<b>%{y}</b><br>Stock: %{x:,} units<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name=demand_label, y=names, x=demand, orientation="h",
        marker=dict(color="rgba(251,146,60,.65)", line=dict(width=0)),
        hovertemplate="<b>%{y}</b><br>Demand: %{x:,} units<extra></extra>",
    ))
    fig.update_layout(
        **_base_layout(f"Current Stock vs Expected Demand{title_suffix}", height=max(280, len(items)*52+80)),
        barmode="group",
    )
    _style_axes(fig)
    fig.update_yaxes(autorange="reversed")
    return fig


# ── 2. Risk distribution donut ───────────────────────────────────────────────
def plot_risk_donut(high: int, med: int, low_: int) -> go.Figure:
    total = high + med + low_
    if total == 0:
        fig = go.Figure()
        fig.update_layout(**_base_layout("Stock-out Risk", height=260))
        fig.add_annotation(text="No data.", showarrow=False,
                           font=dict(color=C["muted"]))
        return fig

    fig = go.Figure(go.Pie(
        labels=["High Risk", "Watch", "Safe"],
        values=[high, med, low_],
        hole=0.6,
        marker=dict(colors=[C["high"], C["med"], C["low"]],
                    line=dict(color="#fff", width=2)),
        textinfo="label+percent",
        textfont=dict(size=11),
        hovertemplate="%{label}: %{value:,} products (%{percent})<extra></extra>",
    ))
    fig.add_annotation(text=f"<b>{total}</b><br><span style='font-size:10px'>products</span>",
                       showarrow=False, font=dict(size=16, color=C["text"]))
    fig.update_layout(**_base_layout("Stock-out Risk", height=260))
    return fig


# ── 3. Top sellers bar chart ─────────────────────────────────────────────────
def plot_top_sellers(items: list[dict]) -> go.Figure:
    """items: list of {name, demand}"""
    if not items:
        fig = go.Figure()
        fig.update_layout(**_base_layout("Expected Top Sellers — Next 7 Days", height=260))
        fig.add_annotation(
            text="Forecast data will appear here once calibration is complete.",
            showarrow=False, font=dict(color=C["muted"], size=12),
            yref="paper", y=0.5)
        return fig

    names  = [it["name"]   for it in items]
    demand = [it["demand"] for it in items]

    fig = go.Figure(go.Bar(
        y=names, x=demand, orientation="h",
        marker=dict(color=C["primary"], line=dict(width=0)),
        hovertemplate="<b>%{y}</b><br>Expected demand: %{x:,} units<extra></extra>",
    ))
    fig.update_layout(**_base_layout("Expected Top Sellers — Next 7 Days", height=260))
    _style_axes(fig)
    fig.update_yaxes(autorange="reversed")
    return fig


# ── 4. Sales trend line (Product page) ──────────────────────────────────────
def plot_sales_trend(df: pd.DataFrame, title: str = "Sales Trend — Last 30 Days") -> go.Figure:
    if df.empty or "date" not in df.columns or "units_sold" not in df.columns:
        fig = go.Figure()
        fig.update_layout(**_base_layout(title, height=280))
        fig.add_annotation(text="No sales history available.",
                           showarrow=False, font=dict(color=C["muted"]))
        return fig

    daily = df.groupby("date", as_index=False)["units_sold"].sum().sort_values("date").tail(30)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily["date"], y=daily["units_sold"],
        mode="lines", name="Units sold",
        line=dict(color=C["primary"], width=2.5),
        fill="tozeroy", fillcolor="rgba(37,99,235,.07)",
        hovertemplate="<b>%{x|%d %b}</b><br>%{y:,} units<extra></extra>",
    ))
    fig.update_layout(**_base_layout(title, height=280))
    _style_axes(fig)
    return fig


# ── 5. Category health stacked bar ───────────────────────────────────────────
def plot_category_health_breakdown(df_store_items: pd.DataFrame) -> go.Figure:
    """Stacked horizontal bar: High/Watch/Safe per product category."""
    if df_store_items.empty or "category" not in df_store_items.columns or "tier" not in df_store_items.columns:
        fig = go.Figure()
        fig.update_layout(**_base_layout("Category Inventory Status", height=280))
        fig.add_annotation(text="No data available.",
                           showarrow=False, font=dict(color=C["muted"]))
        return fig

    grouped = (
        df_store_items.groupby(["category", "tier"]).size()
        .unstack(fill_value=0).reset_index()
    )

    fig = go.Figure()
    for col, color, label in [
        ("ACT_NOW", C["high"], "High Risk"),
        ("WATCH",   C["med"],  "Watch"),
        ("SAFE",    C["low"],  "Safe"),
    ]:
        if col in grouped.columns:
            fig.add_trace(go.Bar(
                name=label, y=grouped["category"], x=grouped[col],
                orientation="h",
                marker=dict(color=color, line=dict(width=0)),
                hovertemplate=f"<b>%{{y}}</b><br>{label}: %{{x:,}}<extra></extra>",
            ))

    fig.update_layout(
        **_base_layout("Category Inventory Status", height=max(240, len(grouped)*42+80)),
        barmode="stack",
    )
    _style_axes(fig)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return fig


def plot_store_attention_comparison(df_store_items: pd.DataFrame) -> go.Figure:
    """Compare high-risk and watch product counts across stores."""
    required = {"store", "tier"}
    if df_store_items.empty or not required.issubset(df_store_items.columns):
        fig = go.Figure()
        fig.update_layout(**_base_layout("Products Needing Attention by Store", height=260))
        fig.add_annotation(text="No store comparison data available.", showarrow=False,
                           font=dict(color=C["muted"]))
        return fig

    grouped = df_store_items.groupby(["store", "tier"]).size().unstack(fill_value=0)
    for tier in ("ACT_NOW", "WATCH"):
        if tier not in grouped.columns:
            grouped[tier] = 0

    fig = go.Figure()
    for tier, color, label in (
        ("ACT_NOW", C["high"], "Act Now"),
        ("WATCH", C["med"], "Watch"),
    ):
        fig.add_trace(go.Bar(
            name=label,
            y=grouped.index,
            x=grouped[tier],
            orientation="h",
            marker=dict(color=color, line=dict(width=0)),
            hovertemplate=f"<b>%{{y}}</b><br>{label}: %{{x:,}} products<extra></extra>",
        ))
    fig.update_layout(
        **_base_layout("Products Needing Attention by Store", height=max(230, len(grouped) * 48 + 80)),
        barmode="stack",
    )
    _style_axes(fig)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return fig


# ── Legacy alias (keeps old app.py calls working) ───────────────────────────
def plot_executive_sales_trend(df: pd.DataFrame,
                               title: str = "Sales Trend") -> go.Figure:
    return plot_sales_trend(df, title=title)
