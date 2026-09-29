"""
StockSense.ai — Enterprise Inventory Decision Intelligence
Professional dashboard with sidebar navigation for store managers.
Run:  streamlit run app.py
"""
from __future__ import annotations

import importlib
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

# ── Project root on path ─────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ── Hot-reload submodules so Streamlit reruns pick up disk changes ────────────
import src.dashboard.data_loader   as _dl; importlib.reload(_dl)
import src.dashboard.decision_engine as _de; importlib.reload(_de)
import src.dashboard.visualization  as _viz; importlib.reload(_viz)
import src.dashboard.ui_helpers     as _ui; importlib.reload(_ui)

from src.dashboard.data_loader import (
    load_datasets, load_predictions, get_latest_inventory_snapshot, merge_predictions
)
from src.dashboard.decision_engine import (
    assess_inventory_risk,
    explain_inventory_decision,
    recommend_inventory_action,
    get_formal_sales_trajectory,
)
from src.dashboard.visualization import (
    plot_stock_vs_demand,
    plot_risk_donut,
    plot_top_sellers,
    plot_sales_trend,
    plot_category_health_breakdown,
    plot_store_attention_comparison,
)
from src.dashboard.ui_helpers import (
    GLOBAL_CSS, COLORS,
    kpi_card, status_badge, section_header, insight_box,
    days_bar, pluralize, risk_label, risk_colors,
)

# ════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="StockSense.ai",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def _load_all():
    return load_datasets()

with st.spinner("Loading store data…"):
    datasets    = _load_all()
    df_master   = datasets.get("master",   pd.DataFrame())
    df_products = datasets.get("products", pd.DataFrame())
    df_stores   = datasets.get("stores",   pd.DataFrame())

if df_master.empty:
    st.error("⚠️  Could not load master data. Check that `data/processed/master_daily.csv` exists.")
    st.stop()

df_predictions, is_ml_ready, _ml_msg = load_predictions()

# ── Build store options ───────────────────────────────────────────────────────
store_opts, store_meta_map = [], {}
for _, s in df_stores.iterrows():
    label = f"{s['city']} — {s['store_type']}"
    store_opts.append(label)
    store_meta_map[label] = s.to_dict()

if not store_opts:
    for _sid in df_master["store_id"].unique():
        label = f"Store {_sid}"
        store_opts.append(label)
        store_meta_map[label] = {
            "store_id": _sid, "city": "–", "store_type": "Retail",
            "floor_area_sqft": 0, "avg_daily_customers": 0, "region": "–",
        }

date_values = []
if "date" in df_master.columns:
    date_values = sorted(
        pd.to_datetime(df_master["date"], errors="coerce").dropna().unique(),
        reverse=True,
    )
date_labels = {pd.Timestamp(value).strftime("%d %b %Y"): pd.Timestamp(value)
               for value in date_values}
default_date_index = 0
if (df_predictions is not None and "date" in df_predictions.columns
        and "predicted_7d_demand" in df_predictions.columns):
    demand_dates = pd.to_datetime(
        df_predictions.loc[df_predictions["predicted_7d_demand"].notna(), "date"],
        errors="coerce",
    ).dropna()
    if not demand_dates.empty:
        latest_demand_date = demand_dates.max()
        latest_demand_label = latest_demand_date.strftime("%d %b %Y")
        if latest_demand_label in date_labels:
            default_date_index = list(date_labels).index(latest_demand_label)

product_labels = {"All products": None}
if "product_id" in df_products.columns:
    product_rows = df_products.drop_duplicates("product_id")
    for _, product_row in product_rows.iterrows():
        product_id = str(product_row["product_id"])
        subcategory = str(product_row.get("sub_category", "")).strip()
        brand = str(product_row.get("brand", "")).strip()
        product_name = f"{subcategory} ({brand})" if subcategory and brand else subcategory
        product_labels[f"{product_name or product_id} · {product_id}"] = product_id
else:
    for product_id in sorted(df_master.get("product_id", pd.Series(dtype=str)).dropna().astype(str).unique()):
        product_labels[product_id] = product_id

# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:

    # ── Brand ────────────────────────────────────────────────────────────────
    st.markdown("""
<div style="padding:1.4rem 1rem 1rem">
  <div style="font-size:1.25rem;font-weight:800;color:#F8FAFC;letter-spacing:-.4px;line-height:1">
    StockSense<span style="color:#60A5FA">.ai</span>
  </div>
  <div style="font-size:.72rem;color:#475569;margin-top:3px;font-weight:500">
    Inventory Decision Intelligence
  </div>
</div>
<hr class="ss-sb-divider" style="margin:0 0 .75rem">
""", unsafe_allow_html=True)

    # ── Store Selector ───────────────────────────────────────────────────────
    st.markdown('<div class="ss-sb-lbl">Active Store</div>', unsafe_allow_html=True)
    selected_label = st.selectbox(
        "Store", store_opts, label_visibility="collapsed",
        key="sidebar_store_sel"
    )

    st.markdown('<div class="ss-sb-lbl">Product</div>', unsafe_allow_html=True)
    selected_product_label = st.selectbox(
        "Product", list(product_labels), label_visibility="collapsed",
        key="sidebar_product_sel",
    )
    selected_product_id = product_labels[selected_product_label]

    st.markdown('<div class="ss-sb-lbl">Inventory date</div>', unsafe_allow_html=True)
    if date_labels:
        selected_date_label = st.selectbox(
            "Inventory date", list(date_labels), label_visibility="collapsed",
            index=default_date_index,
            key="sidebar_date_sel_v2",
        )
        selected_as_of_date = date_labels[selected_date_label]
    else:
        selected_as_of_date = None
        st.caption("No valid inventory dates available.")

    st.markdown('<hr class="ss-sb-divider">', unsafe_allow_html=True)

    # ── Navigation ───────────────────────────────────────────────────────────
    st.markdown('<div class="ss-sb-lbl">Navigation</div>', unsafe_allow_html=True)

    nav_pages = [
        ("📋", "Action Board",     "action"),
        ("🔍", "Product Insights", "product"),
        ("🏬", "Store Overview",   "store"),
    ]
    if "active_page" not in st.session_state:
        st.session_state.active_page = "action"

    for icon, label, key in nav_pages:
        is_active = st.session_state.active_page == key
        clicked = st.button(
            f"{icon}  {label}",
            key=f"nav_{key}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
        )
        if clicked:
            st.session_state.active_page = key
            st.rerun()

    st.markdown('<hr class="ss-sb-divider">', unsafe_allow_html=True)

    # ── Quick Stats (populated after scoring — placeholder until scoring runs) ─
    # We store these in session state so sidebar shows them after first load
    if "sb_stats" not in st.session_state:
        st.session_state.sb_stats = None

    if st.session_state.sb_stats:
        sbs = st.session_state.sb_stats
        st.markdown('<div class="ss-sb-lbl">Store Snapshot</div>', unsafe_allow_html=True)
        for lbl, val, color in sbs:
            st.markdown(f"""
<div class="ss-sb-stat">
  <span class="ss-sb-stat-lbl">{lbl}</span>
  <span class="ss-sb-stat-val" style="color:{color}">{val}</span>
</div>""", unsafe_allow_html=True)
        st.markdown('<hr class="ss-sb-divider">', unsafe_allow_html=True)

    # ── Forecast Status ───────────────────────────────────────────────────────
    st.markdown('<div class="ss-sb-lbl">Forecast Engine</div>', unsafe_allow_html=True)
    if is_ml_ready:
        st.markdown("""
<div style="background:#052E16;border:1px solid #166534;border-radius:8px;
            padding:.6rem .85rem;margin-bottom:.4rem">
  <div style="font-size:.78rem;color:#4ADE80;font-weight:600">✓ Forecast Active</div>
  <div style="font-size:.7rem;color:#166534;margin-top:2px">7-day ML predictions loaded</div>
</div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
<div style="background:#1C1917;border:1px solid #292524;border-radius:8px;
            padding:.6rem .85rem;margin-bottom:.4rem">
  <div style="font-size:.78rem;color:#A8A29E;font-weight:600">⏳ Calibrating</div>
    <div style="font-size:.7rem;color:#A8A29E;margin-top:2px">Forecast file not present</div>
</div>""", unsafe_allow_html=True)

    st.markdown('<hr class="ss-sb-divider">', unsafe_allow_html=True)

    # ── Data Freshness ────────────────────────────────────────────────────────
    last_date = df_master["date"].max() if not df_master.empty else "–"
    if hasattr(last_date, "strftime"):
        last_date_str = last_date.strftime("%d %b %Y")
        today = pd.Timestamp(datetime.now().date())
        staleness_days = (today - last_date).days
        fresh_color = "#4ADE80" if staleness_days <= 1 else ("#FBBF24" if staleness_days <= 3 else "#F87171")
    else:
        last_date_str = str(last_date)
        fresh_color = "#94A3B8"
        staleness_days = 0

    st.markdown(f"""
<div style="padding:0 .85rem">
  <div style="font-size:.68rem;font-weight:700;text-transform:uppercase;
              letter-spacing:.8px;color:#475569;margin-bottom:.4rem">Data Freshness</div>
  <div style="font-size:.78rem;color:{fresh_color};font-weight:600">● {last_date_str}</div>
  <div style="font-size:.7rem;color:#334155;margin-top:1px">
    {f'{staleness_days}d ago' if staleness_days > 0 else 'Today'}
  </div>
</div>""", unsafe_allow_html=True)

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown(f"""
<div style="position:absolute;bottom:0;left:0;right:0;
            padding:.85rem 1rem;border-top:1px solid #1E293B">
  <div style="font-size:.7rem;color:#334155;font-weight:500">NovaMart Operations</div>
  <div style="font-size:.65rem;color:#1E293B;margin-top:1px">
    v2.0 · {datetime.now().strftime('%H:%M')} IST
  </div>
</div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# RESOLVE SELECTED STORE
# ════════════════════════════════════════════════════════════════════════════
meta = store_meta_map[selected_label]
sid  = meta["store_id"]
df_latest_inv = get_latest_inventory_snapshot(df_master, selected_as_of_date)

# ════════════════════════════════════════════════════════════════════════════
# TOPBAR (main area header — single row)
# ════════════════════════════════════════════════════════════════════════════
hcol_l, hcol_r = st.columns([5, 2])
with hcol_l:
    page_titles = {
        "action":  ("📋  Action Board",     "Inventory status and reorder alerts"),
        "product": ("🔍  Product Insights", "Drill into any product's diagnostics"),
        "store":   ("🏬  Store Overview",   "Category health and store summary"),
    }
    ptitle, psub = page_titles.get(st.session_state.active_page, ("Dashboard", ""))
    st.markdown(f"""
<div style="padding:.5rem 0 .25rem">
  <div style="font-size:1.15rem;font-weight:800;color:#0F172A">{ptitle}</div>
  <div style="font-size:.78rem;color:#64748B">{psub}</div>
</div>""", unsafe_allow_html=True)

with hcol_r:
    st.markdown(
        f'<div style="text-align:right;padding:.5rem 0;font-size:.75rem;color:#94A3B8">'
        f'📍 <b style="color:#0F172A">{meta.get("city","–")}</b> · '
        f'{meta.get("store_type","–")} · '
        f'<b style="color:#2563EB">ID {sid}</b></div>',
        unsafe_allow_html=True,
    )

st.markdown("<hr style='margin:.35rem 0 1rem;border:none;border-top:1px solid #E2E8F0'>",
            unsafe_allow_html=True)

# ── Calibration banner ────────────────────────────────────────────────────────
if not is_ml_ready:
    st.markdown("""
<div class="ss-banner">
    <b>Forecast data is not available yet.</b>
    Risk categories below use the inventory/reorder rule; demand and probability are not estimated.
</div>""", unsafe_allow_html=True)
    if "found" in _ml_msg.lower() or "error" in _ml_msg.lower():
        st.warning(_ml_msg)
elif not _ml_msg.startswith("Forecast output loaded"):
        st.warning(_ml_msg)

# ════════════════════════════════════════════════════════════════════════════
# BUILD SCORED ITEM LIST
# ════════════════════════════════════════════════════════════════════════════
store_master = df_master[df_master["store_id"].astype(str) == str(sid)].copy()
if selected_as_of_date is not None and "date" in store_master.columns:
    store_master = store_master[
        pd.to_datetime(store_master["date"], errors="coerce") <= selected_as_of_date
    ]
store_inv = df_latest_inv[df_latest_inv["store_id"].astype(str) == str(sid)].copy()
store_inv = merge_predictions(store_inv, df_predictions if is_ml_ready else None)
if selected_product_id is not None:
    store_inv = store_inv[store_inv["product_id"].astype(str) == str(selected_product_id)]

def _sku_name(row):
    sub   = str(row.get("sub_category", "")) if pd.notna(row.get("sub_category")) else ""
    brand = str(row.get("brand", ""))         if pd.notna(row.get("brand"))         else ""
    pid   = str(row.get("product_id", ""))
    if sub and brand: return f"{sub} ({brand})"
    return sub or pid

store_inv["sku_name"] = store_inv.apply(_sku_name, axis=1)
all_items: list[dict] = []

for _, row in store_inv.iterrows():
    stock_value = row.get("closing")
    if pd.isna(stock_value):
        stock_value = row.get("current_inventory")
    closing = pd.to_numeric(pd.Series([stock_value]), errors="coerce").iloc[0]
    reorder = pd.to_numeric(pd.Series([row.get("reorder_lvl")]), errors="coerce").iloc[0]
    pred_7d = pd.to_numeric(pd.Series([row.get("predicted_7d_demand")]), errors="coerce").iloc[0]
    stockout = pd.to_numeric(pd.Series([row.get("stockout_probability")]), errors="coerce").iloc[0]
    supplied_risk = row.get("risk_level")
    lead = pd.to_numeric(pd.Series([row.get("lead_days")]), errors="coerce").iloc[0]
    closing = float(closing) if pd.notna(closing) else np.nan
    reorder = float(reorder) if pd.notna(reorder) else np.nan
    pred_7d = float(pred_7d) if pd.notna(pred_7d) else np.nan
    stockout = float(stockout) if pd.notna(stockout) else np.nan
    lead = float(lead) if pd.notna(lead) else np.nan

    hist = store_master[store_master["product_id"] == row["product_id"]]\
               .groupby("date")["units_sold"].sum().sort_index()
    hist = pd.to_numeric(hist, errors="coerce").dropna()
    avg_daily = hist.tail(14).mean() if len(hist) >= 7 else hist.mean()
    cover_days = min(closing / avg_daily, 99.0) if pd.notna(closing) and avg_daily > 0 else np.nan

    tier, risk_method = assess_inventory_risk(
        closing, reorder, pred_7d, stockout, supplied_risk
    )
    diagnostics = explain_inventory_decision(
        closing, reorder, pred_7d, stockout, hist
    )
    recommendation_text = recommend_inventory_action(tier, hist)
    recommendation = {
        "headline": recommendation_text,
        "action": recommendation_text,
        "rationale": f"Risk category is based on {risk_method.lower()}.",
    }
    display_demand = pred_7d
    using_reorder = pd.isna(pred_7d)

    all_items.append({
        "product_id":   row["product_id"],
        "name":         row["sku_name"],
        "category":     str(row.get("category", "General")),
        "closing":      closing,
        "reorder":      reorder,
        "lead_days":    lead,
        "pred_7d":      pred_7d,
        "stockout":     stockout,
        "cover_days":   cover_days,
        "tier":         tier,
        "risk_method":  risk_method,
        "diagnostics":  diagnostics,
        "recommendation": recommendation,
        "display_demand": display_demand,
        "using_reorder":  using_reorder,
        "unit_cost":    pd.to_numeric(pd.Series([row.get("cost_price")]), errors="coerce").iloc[0],
        "mrp":          pd.to_numeric(pd.Series([row.get("mrp")]), errors="coerce").iloc[0],
        "recent_sales": hist,
    })

critical = sorted([it for it in all_items if it["tier"] == "ACT_NOW"], key=lambda x: x["cover_days"])
watchlist = sorted([it for it in all_items if it["tier"] == "WATCH"],   key=lambda x: x["cover_days"])
safe_list = [it for it in all_items if it["tier"] == "SAFE"]
total     = len(all_items)

# ── Update sidebar Quick Stats after scoring ──────────────────────────────────
st.session_state.sb_stats = [
    ("High Risk",    str(len(critical)),  "#F87171"),
    ("Watch",        str(len(watchlist)), "#FBBF24"),
    ("Safe",         str(len(safe_list)), "#4ADE80"),
    ("Total SKUs",   str(total),          "#93C5FD"),
]

# ════════════════════════════════════════════════════════════════════════════
# KPI CARDS ROW
# ════════════════════════════════════════════════════════════════════════════
if "kpi_filter" not in st.session_state:
    st.session_state.kpi_filter = "ALL"

k1, k2, k3, k4 = st.columns(4)

def _kpi_btn(col, tier_key, label, count, desc, color):
    active = st.session_state.kpi_filter == tier_key
    with col:
        st.markdown(kpi_card(label, count, desc, color, active), unsafe_allow_html=True)
        if st.button("Clear filter" if active else "Filter",
                     key=f"kpi_{tier_key}"):
            st.session_state.kpi_filter = "ALL" if active else tier_key
            st.rerun()

_kpi_btn(k1, "ACT_NOW", "🔴 Act Now",
         len(critical),
         pluralize(len(critical), "product") + " need reordering",
         COLORS["high"])

_kpi_btn(k2, "WATCH", "🟡 Watch",
         len(watchlist),
         pluralize(len(watchlist), "product") + " approaching threshold",
         COLORS["med"])

_kpi_btn(k3, "SAFE", "🟢 Safe",
         len(safe_list),
         pluralize(len(safe_list), "product") + " with healthy stock",
         COLORS["low"])

with k4:
    st.markdown(kpi_card("Total Products", total,
                          pluralize(total, "product") + " tracked in store",
                          COLORS["primary"]), unsafe_allow_html=True)

st.markdown("<div style='margin-bottom:1.4rem'></div>", unsafe_allow_html=True)

# ── Apply KPI filter ──────────────────────────────────────────────────────────
_flt = st.session_state.kpi_filter
filtered_items = (
    critical  if _flt == "ACT_NOW" else
    watchlist if _flt == "WATCH"   else
    safe_list if _flt == "SAFE"    else
    all_items
)

# ════════════════════════════════════════════════════════════════════════════
# PAGE ROUTING
# ════════════════════════════════════════════════════════════════════════════
active_page = st.session_state.active_page


# ══════════════════════════════════════════════════════════════
# PAGE 1 ─ ACTION BOARD
# ══════════════════════════════════════════════════════════════
if active_page == "action":

    # Row 1: Stock vs Demand chart
    st.markdown(section_header("Stock vs Expected Demand",
                               "top products needing attention"), unsafe_allow_html=True)

    forecasted_items = [item for item in all_items if pd.notna(item["pred_7d"])]
    urgent_items = [item for item in critical + watchlist if pd.notna(item["pred_7d"])]
    chart_items = urgent_items[:10] if urgent_items else sorted(
        forecasted_items, key=lambda item: item["pred_7d"], reverse=True
    )[:10]
    chart_data = [{"name": item["name"][:30], "closing": item["closing"],
                   "demand": item["pred_7d"]}
                  for item in chart_items if pd.notna(item["closing"])]
    using_reorder_global = all(item["using_reorder"] for item in chart_items)

    if chart_data:
        st.plotly_chart(plot_stock_vs_demand(chart_data, using_reorder=using_reorder_global),
                        width="stretch")
    elif forecasted_items:
        st.info("No forecasted-demand items are available for this selection.")
    else:
        st.info("Forecast data is not available yet. No predicted demand is shown as a substitute.")

    # Row 2: Donut + Top sellers
    ch_l, ch_r = st.columns(2)
    with ch_l:
        st.markdown(section_header("Risk Category Distribution"), unsafe_allow_html=True)
        st.plotly_chart(plot_risk_donut(len(critical), len(watchlist), len(safe_list)),
                        width="stretch")
        risk_methods = sorted({item["risk_method"] for item in all_items})
        if risk_methods:
            st.caption("Assessment: " + ", ".join(risk_methods) + ". Categories are not probabilities unless supplied by the forecast.")
    with ch_r:
        st.markdown(section_header("Expected Top Sellers", "next 7 days"), unsafe_allow_html=True)
        if is_ml_ready:
            top10 = sorted(all_items,
                           key=lambda x: x["pred_7d"] if pd.notna(x["pred_7d"]) else 0,
                           reverse=True)[:10]
            sellers_data = [{"name": it["name"][:30], "demand": int(it["pred_7d"])}
                            for it in top10 if pd.notna(it["pred_7d"])]
        else:
            sellers_data = []
        if sellers_data:
            st.plotly_chart(plot_top_sellers(sellers_data), width="stretch")
        else:
            st.info(
                "No forecasted demand is available for this selection."
                if is_ml_ready else "Forecast data is not available yet."
            )

    # Row 3: Products Table
    st.markdown(section_header("Products Needing Attention"), unsafe_allow_html=True)

    f1, f2 = st.columns([4, 2])
    with f1:
        search_q = st.text_input("Search", placeholder="Search by product name…",
                                 label_visibility="collapsed")
    with f2:
        risk_filter_opts = ["All risks", "High Risk", "Watch", "Safe"]
        risk_filter_sel  = st.selectbox("Risk", risk_filter_opts,
                                        label_visibility="collapsed")

    display_items = filtered_items
    if search_q:
        sq = search_q.lower()
        display_items = [it for it in display_items if sq in it["name"].lower()]
    if risk_filter_sel != "All risks":
        tier_map = {"High Risk": "ACT_NOW", "Watch": "WATCH", "Safe": "SAFE"}
        rtier = tier_map.get(risk_filter_sel)
        if rtier:
            display_items = [it for it in display_items if it["tier"] == rtier]

    if display_items:
        rows = []
        for it in display_items:
            demand_str = f"{it['pred_7d']:,.0f} units" if pd.notna(it["pred_7d"]) else "Not available"
            rows.append({
                "Product":          it["name"],
                "Current Stock":    f"{it['closing']:,.0f} units" if pd.notna(it["closing"]) else "Not available",
                "Expected 7-Day Demand": demand_str,
                "Risk":             risk_label(it["tier"]),
                "Recommended Action": it["recommendation"]["headline"],
            })
        df_table = pd.DataFrame(rows)

        def _color_risk(val):
            if "High" in val: return "color:#DC2626;font-weight:600"
            if "Watch" in val: return "color:#D97706;font-weight:600"
            return "color:#16A34A;font-weight:600"

        styled = df_table.style.map(_color_risk, subset=["Risk"])
        st.dataframe(styled, width="stretch", hide_index=True,
                     column_config={
                         "Product":           st.column_config.TextColumn("Product",  width="large"),
                         "Current Stock":     st.column_config.TextColumn("Current Stock"),
                         "Expected 7-Day Demand": st.column_config.TextColumn("Expected 7-Day Demand"),
                         "Recommended Action": st.column_config.TextColumn("Recommended Action", width="large"),
                     })
        methods = sorted({item["risk_method"] for item in display_items})
        st.caption("Risk assessment: " + ", ".join(methods) + ". No derived category is presented as a probability.")
        csv = df_table.to_csv(index=False).encode()
        st.download_button("⬇  Download as CSV", csv,
                           file_name=f"stocksense_{sid}_{datetime.now().strftime('%Y%m%d')}.csv",
                           mime="text/csv")
    else:
        st.info("No products match the current filter.")


# ══════════════════════════════════════════════════════════════
# PAGE 2 ─ PRODUCT INSIGHTS
# ══════════════════════════════════════════════════════════════
elif active_page == "product":

    product_options = {
        f"{item['name']} · {item['product_id']}": item
        for item in all_items
    }
    if product_options:
        selected_product = st.selectbox(
            "Select a product to inspect", list(product_options),
            label_visibility="collapsed",
        )
        sel_item = product_options[selected_product]
    else:
        sel_item = None

    if not sel_item:
        st.warning("Product not found.")
    else:
        st.markdown("<div style='margin:.75rem 0'></div>", unsafe_allow_html=True)

        tc, bg, bd = risk_colors(sel_item["tier"])
        st.markdown(f"""
<div style="display:flex;align-items:center;gap:1rem;flex-wrap:wrap;margin-bottom:.75rem">
  <span style="font-size:1.3rem;font-weight:800;color:#0F172A">{sel_item['name']}</span>
  <span style="font-size:.78rem;color:#64748B;background:#F1F5F9;
        border-radius:999px;padding:2px 10px">{sel_item['product_id']}</span>
  <span style="font-size:.78rem;color:#64748B;background:#F1F5F9;
        border-radius:999px;padding:2px 10px">{sel_item['category']}</span>
  {status_badge(sel_item['tier'])}
</div>""", unsafe_allow_html=True)

        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            delta = sel_item["closing"] - sel_item["reorder"] if pd.notna(sel_item["closing"]) and pd.notna(sel_item["reorder"]) else np.nan
            dc    = "color:#16A34A" if delta >= 0 else "color:#DC2626"
            sign  = "+" if delta >= 0 else ""
            st.markdown(f"""
<div class="ss-metric">
  <div class="ss-metric-lbl">Current Stock</div>
    <div class="ss-metric-val">{f"{sel_item['closing']:,.0f}" if pd.notna(sel_item['closing']) else "Not available"}</div>
    <div class="ss-metric-sub" style="{dc}">{f"{sign}{delta:,.0f} vs reorder level" if pd.notna(delta) else "Reorder comparison unavailable"}</div>
</div>""", unsafe_allow_html=True)

        with mc2:
            if pd.notna(sel_item["pred_7d"]):
                dval, dsub = f"{sel_item['pred_7d']:,.0f}", "7-day demand from forecast file"
            else:
                dval, dsub = "Not available", "Forecast data is not available yet"
            st.markdown(f"""
<div class="ss-metric">
  <div class="ss-metric-lbl">Expected Demand</div>
  <div class="ss-metric-val">{dval}</div>
  <div class="ss-metric-sub">{dsub}</div>
</div>""", unsafe_allow_html=True)

        with mc3:
            if pd.notna(sel_item["stockout"]):
                spv, sps = f"{sel_item['stockout']*100:.0f}%", "stock-out probability"
            else:
                spv, sps = "Not provided", f"Risk: {risk_label(sel_item['tier'])} · {sel_item['risk_method']}"
            st.markdown(f"""
<div class="ss-metric">
  <div class="ss-metric-lbl">Stock-out Risk</div>
  <div class="ss-metric-val" style="color:{tc}">{spv}</div>
  <div class="ss-metric-sub">{sps}</div>
</div>""", unsafe_allow_html=True)

        st.markdown("<div style='margin:.9rem 0'></div>", unsafe_allow_html=True)

        prod_history = store_master[store_master["product_id"] == sel_item["product_id"]]
        st.plotly_chart(plot_sales_trend(prod_history), width="stretch")

        hist_s = prod_history.groupby("date")["units_sold"].sum()
        traj, traj_detail, traj_color, traj_sym = get_formal_sales_trajectory(hist_s)
        st.markdown(
            f'<div style="font-size:.83rem;color:{traj_color};font-weight:600;'
            f'margin-top:-.4rem;margin-bottom:1rem">{traj_sym} {traj_detail}</div>',
            unsafe_allow_html=True)

        why_col, what_col = st.columns(2)
        with why_col:
            bullets = "".join(
                f"<div style='margin-bottom:.35rem'>• {reason}</div>"
                for reason in sel_item["diagnostics"][:4]
            )
            st.markdown(insight_box("Why is this product flagged?", bullets),
                        unsafe_allow_html=True)

        with what_col:
            rec = sel_item["recommendation"]
            body = f"""
<div style='font-size:1rem;font-weight:700;color:#0F172A;margin-bottom:.4rem'>
    {rec['headline']}
</div>
<div style='font-size:.85rem;color:#475569;margin-bottom:.5rem'>{rec['rationale']}</div>"""
            st.markdown(insight_box("What should you do?", body), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE 3 ─ STORE OVERVIEW
# ══════════════════════════════════════════════════════════════
elif active_page == "store":

    # Summary row
    total_on_hand  = sum(it["closing"]              for it in all_items)
    total_val_cost = sum(it["closing"] * it["unit_cost"] for it in all_items)
    safe_rate      = (len(safe_list) / total * 100) if total > 0 else 100

    s1, s2, s3, s4 = st.columns(4)
    for col, lbl, val, sub in [
        (s1, "Stock Availability",  f"{safe_rate:.0f}%",          "products with healthy stock"),
        (s2, "Total Products",      str(total),                    "tracked at this store"),
        (s3, "Units on Hand",       f"{total_on_hand:,}",          "across all products"),
        (s4, "Inventory Value",     f"₹{total_val_cost:,.0f}",     "at cost price"),
    ]:
        col.markdown(f"""
<div class="ss-metric" style="margin-bottom:.75rem">
  <div class="ss-metric-lbl">{lbl}</div>
  <div class="ss-metric-val">{val}</div>
  <div class="ss-metric-sub">{sub}</div>
</div>""", unsafe_allow_html=True)

    # Store detail card
    st.markdown("<div style='margin:.25rem 0 .75rem'></div>", unsafe_allow_html=True)
    st.markdown(f"""
<div style="background:#fff;border:1px solid #E2E8F0;border-radius:12px;
            padding:1rem 1.5rem;margin-bottom:1.25rem;
            display:flex;flex-wrap:wrap;gap:1.5rem;align-items:center">
  <div>
    <div style="font-size:.7rem;font-weight:700;text-transform:uppercase;
                letter-spacing:.6px;color:#94A3B8">Facility ID</div>
    <div style="font-size:1rem;font-weight:700;color:#0F172A">{sid}</div>
  </div>
  <div>
    <div style="font-size:.7rem;font-weight:700;text-transform:uppercase;
                letter-spacing:.6px;color:#94A3B8">Location</div>
    <div style="font-size:1rem;font-weight:700;color:#0F172A">
      {meta.get('city','–')}, {meta.get('region','–')}
    </div>
  </div>
  <div>
    <div style="font-size:.7rem;font-weight:700;text-transform:uppercase;
                letter-spacing:.6px;color:#94A3B8">Format</div>
    <div style="font-size:1rem;font-weight:700;color:#0F172A">{meta.get('store_type','–')}</div>
  </div>
  <div>
    <div style="font-size:.7rem;font-weight:700;text-transform:uppercase;
                letter-spacing:.6px;color:#94A3B8">Floor Area</div>
    <div style="font-size:1rem;font-weight:700;color:#0F172A">
      {meta.get('floor_area_sqft',0):,} sq ft
    </div>
  </div>
  <div>
    <div style="font-size:.7rem;font-weight:700;text-transform:uppercase;
                letter-spacing:.6px;color:#94A3B8">Daily Footfall</div>
    <div style="font-size:1rem;font-weight:700;color:#0F172A">
      ~{meta.get('avg_daily_customers',0):,} shoppers
    </div>
  </div>
</div>""", unsafe_allow_html=True)

    # Category chart
    st.markdown(section_header("Category Inventory Status"), unsafe_allow_html=True)
    df_scored = pd.DataFrame([{"category": it["category"], "tier": it["tier"]}
                               for it in all_items])
    st.plotly_chart(plot_category_health_breakdown(df_scored), width="stretch")

    if len(store_meta_map) > 1:
        comparison_items = merge_predictions(
            df_latest_inv,
            df_predictions if is_ml_ready else None,
        )
        if selected_product_id is not None:
            comparison_items = comparison_items[
                comparison_items["product_id"].astype(str) == str(selected_product_id)
            ]
        store_names = {
            str(store_meta["store_id"]): f"{store_meta.get('city', 'Store')} · {store_meta.get('store_type', '')}"
            for store_meta in store_meta_map.values()
        }
        comparison_rows = []
        for _, comparison_row in comparison_items.iterrows():
            stock = pd.to_numeric(pd.Series([comparison_row.get("closing")]), errors="coerce").iloc[0]
            reorder_level = pd.to_numeric(pd.Series([comparison_row.get("reorder_lvl")]), errors="coerce").iloc[0]
            predicted_demand = pd.to_numeric(pd.Series([comparison_row.get("predicted_7d_demand")]), errors="coerce").iloc[0]
            stockout_probability = pd.to_numeric(pd.Series([comparison_row.get("stockout_probability")]), errors="coerce").iloc[0]
            tier, _ = assess_inventory_risk(
                stock, reorder_level, predicted_demand, stockout_probability,
                comparison_row.get("risk_level"),
            )
            store_id = str(comparison_row.get("store_id"))
            comparison_rows.append({
                "store": store_names.get(store_id, store_id),
                "tier": tier,
            })
        st.markdown(section_header("Store Comparison", "products needing attention"), unsafe_allow_html=True)
        st.plotly_chart(
            plot_store_attention_comparison(pd.DataFrame(comparison_rows)),
            width="stretch",
        )

    # Shift checklist
    st.markdown(section_header("Shift Manager Checklist"), unsafe_allow_html=True)
    checklist_items = [
        ("08:00 — Morning",  COLORS["high"],
         f"Review <b>{len(critical)}</b> Act Now products. Verify physical stock. Raise purchase orders."),
        ("13:00 — Midday",   COLORS["med"],
         f"Check <b>{len(watchlist)}</b> Watch products. Confirm supplier deliveries for today."),
        ("18:00 — Evening",  COLORS["low"],
         "Log received stock. Spot-check cold-chain (Dairy, Frozen). Update system counts."),
    ]
    for time_lbl, color, detail in checklist_items:
        st.markdown(f"""
<div style="background:#fff;border:1px solid #E2E8F0;border-left:4px solid {color};
            border-radius:0 10px 10px 0;padding:.75rem 1.25rem;margin-bottom:.6rem">
  <div style="font-size:.75rem;font-weight:700;text-transform:uppercase;
              letter-spacing:.5px;color:{color};margin-bottom:.25rem">{time_lbl}</div>
  <div style="font-size:.88rem;color:#334155">{detail}</div>
</div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div style="border-top:1px solid #E2E8F0;margin-top:2.5rem;padding:1rem 0;
            display:flex;justify-content:space-between;font-size:.73rem;color:#94A3B8">
  <span>StockSense.ai · NovaMart Retail Intelligence · Internal Use Only</span>
  <span>Data snapshot: {df_master['date'].max().strftime('%d %b %Y') if not df_master.empty else '–'}
        · Generated {datetime.now().strftime('%d %b %Y, %H:%M')}</span>
</div>""", unsafe_allow_html=True)
