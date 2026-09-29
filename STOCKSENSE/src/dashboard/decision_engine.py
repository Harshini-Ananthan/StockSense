"""
Enterprise Decision Intelligence Engine for STOCKSENSE
Provides formal retail operations decision logic, stock cover metrics,
operational advisories, and causal diagnostic factors.
"""

from typing import List, Optional, Tuple, Dict, Any
import pandas as pd
import numpy as np


def assess_inventory_risk(
    closing: float,
    reorder_lvl: float,
    predicted_7d_demand: Optional[float] = None,
    stockout_probability: Optional[float] = None,
    supplied_risk_level: Optional[str] = None,
) -> Tuple[str, str]:
    """Return (tier, method) while keeping model probabilities distinct from rules."""
    probability_tier = None
    if pd.notna(stockout_probability):
        probability = float(stockout_probability)
        if probability >= 0.70:
            probability_tier = "ACT_NOW"
        elif probability >= 0.40:
            probability_tier = "WATCH"
        else:
            probability_tier = "SAFE"

    risk_map = {
        "HIGH": "ACT_NOW", "HIGH RISK": "ACT_NOW", "ACT_NOW": "ACT_NOW",
        "MEDIUM": "WATCH", "MEDIUM RISK": "WATCH", "WATCH": "WATCH",
        "LOW": "SAFE", "LOW RISK": "SAFE", "SAFE": "SAFE",
    }
    if probability_tier is None and pd.notna(supplied_risk_level):
        probability_tier = risk_map.get(str(supplied_risk_level).upper())

    demand_tier = None
    if pd.notna(predicted_7d_demand) and pd.notna(closing):
        demand = max(float(predicted_7d_demand), 0.0)
        stock = max(float(closing), 0.0)
        if stock <= 0 or demand >= stock:
            demand_tier = "ACT_NOW"
        elif probability_tier is None and (
                demand >= stock * 0.70
                or (pd.notna(reorder_lvl) and stock < float(reorder_lvl))):
            demand_tier = "WATCH"
        elif probability_tier is None:
            demand_tier = "SAFE"

    tier_order = {"SAFE": 0, "WATCH": 1, "ACT_NOW": 2}
    available_tiers = [tier for tier in (probability_tier, demand_tier) if tier is not None]
    if available_tiers:
        tier = max(available_tiers, key=tier_order.__getitem__)
        if probability_tier is not None and demand_tier is not None:
            if tier != probability_tier:
                return tier, "Derived stock-versus-demand rule (more urgent than model probability)"
            return tier, "ML stock-out probability and stock-versus-demand rule"
        if probability_tier is not None:
            return tier, "ML stock-out probability"
        return tier, "Derived stock-versus-demand rule"

    if pd.isna(closing):
        return "WATCH", "Insufficient inventory data"
    if closing <= 0 or (pd.notna(reorder_lvl) and closing < reorder_lvl):
        return "ACT_NOW", "Inventory/reorder rule"
    if pd.notna(reorder_lvl) and closing <= reorder_lvl * 1.25:
        return "WATCH", "Inventory/reorder rule"
    return "SAFE", "Inventory/reorder rule"


def explain_inventory_decision(
    closing: float,
    reorder_lvl: float,
    predicted_7d_demand: Optional[float],
    stockout_probability: Optional[float],
    recent_sales: pd.Series,
) -> List[str]:
    """Build short explanations only from values present in the source data."""
    reasons: List[str] = []
    if pd.isna(closing):
        reasons.append("Current inventory data is not available.")
    else:
        if closing <= 0:
            reasons.append("No stock is currently available.")
        elif pd.notna(reorder_lvl) and closing < reorder_lvl:
            reasons.append(f"Current stock is {int(reorder_lvl - closing)} units below the reorder level.")

    if pd.notna(predicted_7d_demand) and pd.notna(closing):
        if predicted_7d_demand > closing:
            reasons.append("Current stock is below expected 7-day demand.")
        else:
            reasons.append("Inventory appears sufficient for expected 7-day demand.")
    elif pd.notna(stockout_probability):
        reasons.append("A stock-out likelihood is available from the forecast output.")

    if len(recent_sales) >= 14:
        recent = float(recent_sales.tail(7).mean())
        previous = float(recent_sales.iloc[-14:-7].mean())
        if previous > 0 and recent >= previous * 1.10:
            reasons.append("Recent sales are increasing.")
        elif previous > 0 and recent <= previous * 0.90:
            reasons.append("Recent sales are decreasing.")

    if not reasons and pd.notna(closing) and pd.notna(reorder_lvl):
        reasons.append("Current stock is at or above the reorder level.")
    return reasons[:4]


def recommend_inventory_action(tier: str, recent_sales: pd.Series) -> str:
    """Return a plain-language decision-support action."""
    if tier == "ACT_NOW":
        action = "Consider replenishing inventory."
    elif tier == "WATCH":
        action = "Monitor inventory closely."
    else:
        action = "No immediate action."

    if len(recent_sales) >= 14:
        recent = float(recent_sales.tail(7).mean())
        previous = float(recent_sales.iloc[-14:-7].mean())
        if previous > 0 and recent >= previous * 1.10:
            return f"{action} Consider preparing additional stock as recent sales are increasing."
        if previous > 0 and recent <= previous * 0.90 and tier == "SAFE":
            return "Review inventory levels; recent sales are decreasing."
    return action


def get_product_status(
    closing: float,
    reorder_lvl: float,
    stockout_prob: Optional[float] = None,
    lead_days: Optional[float] = None
) -> Tuple[str, str, str]:
    """
    Returns (status_tier, formal_label, badge_html).
    Tiers:
      - 'ACT_NOW': CRITICAL ACTION / HIGH RISK
      - 'WATCH':   ACTIVE WATCHLIST / MONITORING
      - 'SAFE':    OPTIMAL STOCK COVER / SAFE
    """
    if pd.notna(stockout_prob):
        # ML Model stock-out probability threshold
        if stockout_prob >= 0.70:
            label = "CRITICAL ACTION"
            badge = '<span style="background-color:#FEE2E2; color:#991B1B; border:1px solid #FCA5A5; padding:3px 9px; border-radius:6px; font-weight:700; font-size:0.8rem; letter-spacing:0.5px;">🔴 CRITICAL (≥70%)</span>'
            return "ACT_NOW", label, badge
        elif stockout_prob >= 0.40:
            label = "ACTIVE WATCHLIST"
            badge = '<span style="background-color:#FEF3C7; color:#92400E; border:1px solid #FCD34D; padding:3px 9px; border-radius:6px; font-weight:700; font-size:0.8rem; letter-spacing:0.5px;">🟡 WATCHLIST (40–69%)</span>'
            return "WATCH", label, badge
        else:
            label = "OPTIMAL COVER"
            badge = '<span style="background-color:#ECFDF5; color:#065F46; border:1px solid #6EE7B7; padding:3px 9px; border-radius:6px; font-weight:700; font-size:0.8rem; letter-spacing:0.5px;">🟢 OPTIMAL (<40%)</span>'
            return "SAFE", label, badge
    else:
        # Factual inventory safety thresholds when ML is pending
        if pd.isna(closing):
            return "SAFE", "OPTIMAL COVER", '<span style="background-color:#ECFDF5; color:#065F46; border:1px solid #6EE7B7; padding:3px 9px; border-radius:6px; font-weight:700; font-size:0.8rem;">🟢 OPTIMAL</span>'

        if closing <= 0:
            label = "DEPLETED / STOCKOUT"
            badge = '<span style="background-color:#FEE2E2; color:#991B1B; border:1px solid #FCA5A5; padding:3px 9px; border-radius:6px; font-weight:700; font-size:0.8rem;">🔴 STOCKOUT (0 Units)</span>'
            return "ACT_NOW", label, badge
        elif pd.notna(reorder_lvl) and closing < reorder_lvl:
            label = "CRITICAL DEFICIT"
            badge = '<span style="background-color:#FEE2E2; color:#991B1B; border:1px solid #FCA5A5; padding:3px 9px; border-radius:6px; font-weight:700; font-size:0.8rem;">🔴 BELOW REORDER</span>'
            return "ACT_NOW", label, badge
        elif pd.notna(reorder_lvl) and closing <= (reorder_lvl * 1.25):
            label = "ACTIVE WATCHLIST"
            badge = '<span style="background-color:#FEF3C7; color:#92400E; border:1px solid #FCD34D; padding:3px 9px; border-radius:6px; font-weight:700; font-size:0.8rem;">🟡 NEAR THRESHOLD</span>'
            return "WATCH", label, badge
        else:
            label = "OPTIMAL COVER"
            badge = '<span style="background-color:#ECFDF5; color:#065F46; border:1px solid #6EE7B7; padding:3px 9px; border-radius:6px; font-weight:700; font-size:0.8rem;">🟢 OPTIMAL</span>'
            return "SAFE", label, badge


def get_formal_diagnostic_reasons(
    row: pd.Series,
    hist_sales_series: Optional[pd.Series] = None
) -> List[Dict[str, str]]:
    """
    Returns structured, formal causal drivers answering:
    'Why is this item at risk?' in clear, executive retail language.
    """
    drivers: List[Dict[str, str]] = []

    closing = row.get("closing", np.nan)
    reorder_lvl = row.get("reorder_lvl", np.nan)
    lead_days = row.get("lead_days", np.nan)
    pred_7d = row.get("predicted_7d_demand", np.nan)
    shelf_life = row.get("shelf_life_days", np.nan)
    holiday = row.get("holiday", 0)
    weekend = row.get("weekend", 0)

    # 1. On-Hand Inventory Buffer Status
    if pd.notna(closing) and closing <= 0:
        drivers.append({
            "factor": "Inventory Depletion",
            "observation": f"On-hand inventory is fully exhausted (0 units available vs minimum safety target of {int(reorder_lvl)} units)."
        })
    elif pd.notna(closing) and pd.notna(reorder_lvl) and closing < reorder_lvl:
        deficit = int(reorder_lvl - closing)
        drivers.append({
            "factor": "Safety Buffer Deficit",
            "observation": f"Current stock ({int(closing)} units) is operating {deficit} units below configured replenishment reorder threshold ({int(reorder_lvl)} units)."
        })
    elif pd.notna(closing) and pd.notna(reorder_lvl) and closing <= (reorder_lvl * 1.25):
        drivers.append({
            "factor": "Buffer Compression",
            "observation": f"Stock level ({int(closing)} units) is hovering within 25% of reorder trigger point ({int(reorder_lvl)} units)."
        })

    # 2. Demand Forecast Pressure
    if pd.notna(pred_7d) and pd.notna(closing):
        if pred_7d > closing:
            shortfall = int(pred_7d - closing)
            drivers.append({
                "factor": "Demand-Supply Imbalance",
                "observation": f"Projected 7-day customer demand ({int(pred_7d)} units) outpaces current inventory ({int(closing)} units) by {shortfall} units."
            })
        elif closing > 0 and (closing / (pred_7d / 7.0 if pred_7d > 0 else 1.0)) < (lead_days if pd.notna(lead_days) else 2):
            cover_days = closing / (pred_7d / 7.0 if pred_7d > 0 else 1.0)
            drivers.append({
                "factor": "Lead Time Exposure",
                "observation": f"Estimated stock coverage ({cover_days:.1f} days) is less than vendor replenishment cycle ({int(lead_days)} days)."
            })

    # 3. Sales Velocity Trends
    if hist_sales_series is not None and len(hist_sales_series) >= 7:
        recent_avg = hist_sales_series.tail(7).mean()
        older_avg = hist_sales_series.mean()
        if older_avg > 0 and recent_avg > (older_avg * 1.2):
            growth_pct = ((recent_avg - older_avg) / older_avg) * 100
            drivers.append({
                "factor": "Consumption Acceleration",
                "observation": f"Recent 7-day sales velocity ({recent_avg:.1f} units/day) is running +{growth_pct:.0f}% ahead of standard baseline ({older_avg:.1f} units/day)."
            })
        elif older_avg > 0 and recent_avg < (older_avg * 0.75):
            drivers.append({
                "factor": "Demand Contraction",
                "observation": f"Recent consumption velocity has tapered down by {((older_avg - recent_avg) / older_avg) * 100:.0f}% below baseline."
            })

    # 4. Shelf Life & Perishability Constraints
    if pd.notna(shelf_life) and shelf_life <= 3:
        drivers.append({
            "factor": "Perishability Lifecycle",
            "observation": f"Strict shelf life constraint ({int(shelf_life)} days) precludes excessive safety buffering; requires synchronized JIT intake."
        })

    # 5. External Footfall Dynamics
    if weekend == 1 or holiday == 1:
        drivers.append({
            "factor": "Calendar Traffic Surge",
            "observation": "Elevated customer footfall index active due to scheduled weekend / commercial holiday schedule."
        })

    if not drivers:
        drivers.append({
            "factor": "Nominal Parameters",
            "observation": "Current stock buffers and consumption rates remain aligned with standard replenishment guidelines."
        })

    return drivers


def get_formal_recommendation(
    status_tier: str,
    closing: float,
    reorder_lvl: float,
    pred_7d: Optional[float] = None,
    lead_days: Optional[float] = None
) -> Dict[str, str]:
    """
    Returns formal, industry-grade operational recommendations framed as decision support.
    """
    tier = status_tier.upper()

    if tier == "ACT_NOW":
        if pd.notna(pred_7d) and pd.notna(closing) and pred_7d > closing:
            return {
                "headline": "Generate Expedited Replenishment Order",
                "action": "Consider issuing an immediate purchase order to increase allocation prior to the upcoming 7-day consumption cycle.",
                "rationale": f"Current stock ({int(closing)} units) is insufficient to service projected demand ({int(pred_7d)} units). Expedited intake is recommended.",
                "urgency": "HIGH PRIORITY"
            }
        else:
            return {
                "headline": "Issue Replenishment Purchase Order",
                "action": "Consider generating a replenishment purchase order to restore safety stock above minimum threshold.",
                "rationale": f"Inventory ({int(closing)} units) has breached safety threshold ({int(reorder_lvl)} units). Immediate vendor dispatch advised.",
                "urgency": "HIGH PRIORITY"
            }

    elif tier == "WATCH":
        return {
            "headline": "Initiate Active Buffer Surveillance",
            "action": "Maintain daily surveillance on point-of-sale velocity and verify supplier delivery readiness.",
            "rationale": "Inventory is approaching trigger threshold. Prepare purchase requisition if daily sell-through accelerates.",
            "urgency": "MEDIUM PRIORITY"
        }

    elif tier == "SAFE":
        if pd.notna(closing) and pd.notna(reorder_lvl) and closing > (reorder_lvl * 3):
            return {
                "headline": "Evaluate Working Capital & Holding Balance",
                "action": "Consider reviewing excess inventory allocation to optimize store holding costs and floor utilization.",
                "rationale": "Stock depth exceeds normal cycle coverage. Assess opportunities for cross-store rebalancing or promotional support.",
                "urgency": "ROUTINE REVIEW"
            }
        else:
            return {
                "headline": "Maintain Standard Replenishment Cycle",
                "action": "No immediate intervention indicated; maintain scheduled review in standard operational cycle.",
                "rationale": "Current inventory levels comfortably satisfy replenishment buffer guidelines.",
                "urgency": "NORMAL OPERATION"
            }

    return {
        "headline": "Awaiting Machine Learning Predictive Forecast",
        "action": "Continue baseline stock surveillance; automated replenishment models will activate upon receipt of forecast output.",
        "rationale": "Standard historical monitoring currently active pending ML model deployment.",
        "urgency": "EXPLORATORY MODE"
    }


def get_formal_sales_trajectory(hist_sales_series: Optional[pd.Series]) -> Tuple[str, str, str, str]:
    """
    Analyzes historical sales consumption and returns:
    (formal_status, detail_str, color_hex, badge_symbol)
    """
    if hist_sales_series is None or len(hist_sales_series) < 14:
            return "Limited Sales History", "Not enough recent sales history to compare two full weeks.", "#64748B", "•"

    recent = hist_sales_series.tail(7).mean()
    previous = hist_sales_series.iloc[-14:-7].mean()

    if previous <= 0:
        return "Stable Consumption Rate", "Sales velocity is tracking within expected variance.", "#2563EB", "■"

    pct_change = ((recent - previous) / previous) * 100

    if pct_change >= 10.0:
        return (
            "Accelerating Velocity",
            f"Sales volume is expanding (+{pct_change:.1f}% vs prior 7-day cycle).",
            "#16A34A",
            "▲"
        )
    elif pct_change <= -10.0:
        return (
            "Decelerating Velocity",
            f"Sales volume is contracting ({pct_change:.1f}% vs prior 7-day cycle).",
            "#D97706",
            "▼"
        )
    else:
        return (
            "Stable Consumption Rate",
            f"Sales velocity remains consistent ({pct_change:+.1f}% variance).",
            "#2563EB",
            "■"
        )


# Backward Compatibility Aliases
get_shopkeeper_reasons = get_formal_diagnostic_reasons
generate_explanation = get_formal_diagnostic_reasons
get_shopkeeper_recommendation = get_formal_recommendation
generate_recommendation = get_formal_recommendation
get_sales_direction = get_formal_sales_trajectory
classify_risk = get_product_status

