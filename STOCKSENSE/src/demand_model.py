"""
demand_model.py
===============
StockSense Round 2 - Demand Forecasting (Model 1)
Person 2 (ML Engineer)

Target: next_7_day_demand (regression)
Baseline: previous 7-day rolling mean demand

Models compared:
  1. Baseline (rolling-mean heuristic)
  2. Linear Regression
  3. Random Forest Regressor
  4. XGBoost Regressor

Evaluation: MAE, RMSE, MAPE, R2
Time-aware split: Train / Validation / Test (no random shuffling)

Outputs:
    models/demand_model.pkl
    models/demand_scaler.pkl
    data/processed/demand_predictions.csv
    reports/demand_model_comparison.csv
    reports/demand_model_comparison.md
    reports/demand_feature_importance.csv
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model   import LinearRegression
from sklearn.ensemble       import RandomForestRegressor
from sklearn.preprocessing  import StandardScaler
from sklearn.metrics        import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

warnings.filterwarnings("ignore")

RANDOM_SEED = 42

BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_PATH   = os.path.join(BASE_DIR, "data", "processed", "ml_features.csv")
MODELS_DIR      = os.path.join(BASE_DIR, "models")
REPORTS_DIR     = os.path.join(BASE_DIR, "reports")
PRED_PATH       = os.path.join(BASE_DIR, "data", "processed", "demand_predictions.csv")

# Temporal split thresholds
TRAIN_END = "2026-07-25"
VAL_END   = "2026-08-12"
# Test = 2026-08-13 to 2026-08-31

# ---------------------------------------------------------------------------
# Feature columns used for Model 1 (demand forecasting)
# NOTE: next_7_day_demand is the TARGET — not a feature.
# NOTE: stockout_flag is the Model 2 target — excluded here to avoid circularity.
# NOTE: revenue / sold / opening excluded (post-hoc info or duplicates).
# ---------------------------------------------------------------------------
DEMAND_FEATURES = [
    # time
    "day_of_week", "weekend_flag", "month", "week_no",
    "day_of_month", "quarter", "festival_flag", "holiday_or_festival",
    "holiday", "local_event",
    # lag
    "lag_1", "lag_7", "lag_14",
    # rolling
    "rolling_mean_7", "rolling_mean_14", "rolling_std_7",
    # inventory (current-day, no leakage)
    "days_of_inventory", "inventory_to_demand_ratio",
    "reorder_gap", "stock_coverage_ratio",
    "lag_closing_1", "lag_closing_7",
    # price/promo
    "discount_pct", "price_change", "mrp_to_cost_ratio",
    "promotion_flag", "promo_weekend",
    # store/product
    "store_type_enc", "category_enc", "brand_enc", "region_enc",
    "store_id_enc", "product_id_enc",
    "shelf_life_days", "lead_days",
    "avg_daily_customers", "floor_area_sqft",
    "customer_to_floor_ratio", "is_cold_start",
    "mrp", "cost_price",
    # weather
    "temp_c", "rain_mm",
]
TARGET = "next_7_day_demand"


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def df_to_md(df):
    """Convert DataFrame to markdown table without tabulate dependency."""
    cols   = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep    = "|" + "|".join(["---"] * len(cols)) + "|"
    rows   = ["| " + " | ".join(str(v) for v in row.values) + " |"
              for _, row in df.iterrows()]
    return "\n".join([header, sep] + rows)


def mape(y_true, y_pred, eps=1e-6):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true > eps
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def compute_metrics(y_true, y_pred, label=""):
    mae_  = mean_absolute_error(y_true, y_pred)
    rmse_ = np.sqrt(mean_squared_error(y_true, y_pred))
    mape_ = mape(y_true, y_pred)
    r2_   = r2_score(y_true, y_pred)
    if label:
        print(f"  [{label}]  MAE={mae_:.3f}  RMSE={rmse_:.3f}  "
              f"MAPE={mape_:.2f}%  R2={r2_:.4f}")
    return {"MAE": mae_, "RMSE": rmse_, "MAPE": mape_, "R2": r2_}


# ---------------------------------------------------------------------------
# Load & split
# ---------------------------------------------------------------------------

def load_and_split():
    df = pd.read_csv(FEATURES_PATH)
    df["date"] = pd.to_datetime(df["date"])

    # Drop rows where target is NaN (incomplete 7-day window)
    df = df.dropna(subset=[TARGET]).copy()
    print(f"[INFO] Usable rows after dropping NaN target: {len(df)}")

    train = df[df["date"] <= TRAIN_END].copy()
    val   = df[(df["date"] > TRAIN_END) & (df["date"] <= VAL_END)].copy()
    test  = df[df["date"] > VAL_END].copy()

    print(f"[INFO] Train: {len(train)} rows  ({train['date'].min().date()} to {train['date'].max().date()})")
    print(f"[INFO] Val:   {len(val)} rows  ({val['date'].min().date()} to {val['date'].max().date()})")
    print(f"[INFO] Test:  {len(test)} rows  ({test['date'].min().date()} to {test['date'].max().date()})")

    return df, train, val, test


def prepare_xy(split_df, feature_cols, scaler=None, fit_scaler=False):
    X = split_df[feature_cols].copy()
    y = split_df[TARGET].copy()

    # Fill any residual NaN (e.g., P999 cold-start rows) with 0
    X = X.fillna(0)

    if fit_scaler:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    elif scaler is not None:
        X_scaled = scaler.transform(X)
    else:
        X_scaled = X.values

    return X_scaled, y.values, scaler


# ---------------------------------------------------------------------------
# Baseline: rolling-mean of previous 7 days per group
# ---------------------------------------------------------------------------

def evaluate_baseline(train, val, test):
    """
    Baseline prediction = rolling_mean_7 (already computed in features).
    This is the 7-day historical mean, approximating next-7-day demand.
    NaN rows (incomplete target window or cold-start lag) are excluded.
    """
    print("\n[INFO] --- Baseline: rolling_mean_7 ---")
    results = {}

    for split_name, split_df in [("Validation", val), ("Test", test)]:
        # Filter to rows where both target and predictor are non-NaN
        mask = split_df[TARGET].notna() & split_df["rolling_mean_7"].notna()
        sub = split_df[mask].copy()
        y_true = sub[TARGET].values
        y_pred = sub["rolling_mean_7"].fillna(0).values * 7  # scale to 7-day window
        m = compute_metrics(y_true, y_pred, label=f"Baseline/{split_name}")
        results[split_name] = m

    return results


# ---------------------------------------------------------------------------
# Train models
# ---------------------------------------------------------------------------

def train_and_evaluate(train, val, test, feature_cols):
    scaler = None
    X_tr, y_tr, scaler = prepare_xy(train, feature_cols, fit_scaler=True)
    X_va, y_va, _      = prepare_xy(val,   feature_cols, scaler=scaler)
    X_te, y_te, _      = prepare_xy(test,  feature_cols, scaler=scaler)

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest":     RandomForestRegressor(
            n_estimators=200, max_depth=12, min_samples_leaf=3,
            random_state=RANDOM_SEED, n_jobs=-1),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_SEED,
            eval_metric="rmse", verbosity=0)

    results_val  = {}
    results_test = {}
    trained      = {}

    for name, model in models.items():
        print(f"\n[INFO] Training {name} ...")
        if name == "XGBoost":
            model.fit(X_tr, y_tr,
                      eval_set=[(X_va, y_va)],
                      verbose=False)
        else:
            model.fit(X_tr, y_tr)

        va_pred = model.predict(X_va)
        te_pred = model.predict(X_te)

        results_val[name]  = compute_metrics(y_va, va_pred, label=f"{name}/Val")
        results_test[name] = compute_metrics(y_te, te_pred, label=f"{name}/Test")
        trained[name]      = model

    return trained, scaler, results_val, results_test, (X_te, y_te)


# ---------------------------------------------------------------------------
# Select best model using validation MAE
# ---------------------------------------------------------------------------

def select_best(results_val):
    best_name = min(results_val, key=lambda k: results_val[k]["MAE"])
    print(f"\n[INFO] Best model (lowest Val MAE): {best_name}")
    return best_name


# ---------------------------------------------------------------------------
# Save model and scaler
# ---------------------------------------------------------------------------

def save_artifacts(best_model, scaler, best_name):
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path  = os.path.join(MODELS_DIR, "demand_model.pkl")
    scaler_path = os.path.join(MODELS_DIR, "demand_scaler.pkl")

    with open(model_path,  "wb") as f:
        pickle.dump(best_model, f)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    # Save metadata
    meta = {
        "model_name": best_name,
        "feature_cols": DEMAND_FEATURES,
        "target": TARGET,
        "train_end": TRAIN_END,
        "val_end": VAL_END,
    }
    meta_path = os.path.join(MODELS_DIR, "demand_model_meta.pkl")
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)

    print(f"[INFO] Saved demand model:  {model_path}")
    print(f"[INFO] Saved demand scaler: {scaler_path}")
    print(f"[INFO] Saved model meta:    {meta_path}")


# ---------------------------------------------------------------------------
# Demand predictions CSV
# ---------------------------------------------------------------------------

def save_predictions(test, best_model, scaler):
    X_te, y_te, _ = prepare_xy(test, DEMAND_FEATURES, scaler=scaler)
    preds = best_model.predict(X_te)
    preds = np.maximum(preds, 0)  # clip to non-negative

    out = test[["date", "store_id", "product_id"]].copy()
    out["actual_next_7_day_demand"]    = y_te
    out["predicted_next_7_day_demand"] = preds

    os.makedirs(os.path.dirname(PRED_PATH), exist_ok=True)
    out.to_csv(PRED_PATH, index=False)
    print(f"[INFO] Saved demand predictions: {PRED_PATH}")
    return out


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def save_feature_importance(best_model, best_name, feature_cols):
    fi_path = os.path.join(REPORTS_DIR, "demand_feature_importance.csv")
    os.makedirs(REPORTS_DIR, exist_ok=True)

    if hasattr(best_model, "feature_importances_"):
        imp = best_model.feature_importances_
        fi_df = pd.DataFrame({
            "feature":    feature_cols,
            "importance": imp,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
    elif hasattr(best_model, "coef_"):
        imp = np.abs(best_model.coef_)
        fi_df = pd.DataFrame({
            "feature":    feature_cols,
            "importance": imp,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
    else:
        fi_df = pd.DataFrame({"feature": feature_cols, "importance": np.nan})

    fi_df.to_csv(fi_path, index=False)
    print(f"[INFO] Saved demand feature importance: {fi_path}")
    print(f"[INFO] Top 10 features ({best_name}):")
    print(fi_df.head(10).to_string(index=False))

    # Chart
    chart_path = os.path.join(REPORTS_DIR, "demand_feature_importance.png")
    fig, ax = plt.subplots(figsize=(10, 8))
    top = fi_df.head(20)
    ax.barh(top["feature"][::-1], top["importance"][::-1], color="#2196F3")
    ax.set_xlabel("Importance")
    ax.set_title(f"Top 20 Features — {best_name} (Demand Model)")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(chart_path, dpi=120)
    plt.close(fig)
    print(f"[INFO] Saved chart: {chart_path}")

    return fi_df


# ---------------------------------------------------------------------------
# Save comparison reports
# ---------------------------------------------------------------------------

def save_comparison_reports(baseline_results, results_val, results_test, best_name):
    os.makedirs(REPORTS_DIR, exist_ok=True)

    rows = []

    # Baseline row (use test-split metrics)
    bm = baseline_results.get("Test", {})
    rows.append({
        "model": "Baseline (rolling_mean_7 x7)",
        "split": "Test",
        "MAE":  round(bm.get("MAE",  np.nan), 4),
        "RMSE": round(bm.get("RMSE", np.nan), 4),
        "MAPE": round(bm.get("MAPE", np.nan), 4),
        "R2":   round(bm.get("R2",   np.nan), 4),
    })

    # Validation metrics for each trained model
    for name, m in results_val.items():
        rows.append({
            "model": name,
            "split": "Validation",
            "MAE":  round(m["MAE"],  4),
            "RMSE": round(m["RMSE"], 4),
            "MAPE": round(m["MAPE"], 4),
            "R2":   round(m["R2"],   4),
        })

    # Test metrics for each trained model
    for name, m in results_test.items():
        rows.append({
            "model": name,
            "split": "Test",
            "MAE":  round(m["MAE"],  4),
            "RMSE": round(m["RMSE"], 4),
            "MAPE": round(m["MAPE"], 4),
            "R2":   round(m["R2"],   4),
        })

    comp_df = pd.DataFrame(rows)
    csv_path = os.path.join(REPORTS_DIR, "demand_model_comparison.csv")
    comp_df.to_csv(csv_path, index=False)
    print(f"[INFO] Saved: {csv_path}")

    # Markdown report
    md_path = os.path.join(REPORTS_DIR, "demand_model_comparison.md")
    bv = results_val.get(best_name, {})
    bt = results_test.get(best_name, {})

    xgb_note = (
        "XGBoost was available (v3.0.5) and included in comparison."
        if XGBOOST_AVAILABLE
        else "XGBoost was not available; only LinearRegression and RandomForest compared."
    )

    md = [
        "# Demand Model Comparison Report",
        "",
        "## Temporal Split",
        f"| Split | Dates |",
        "|---|---|",
        f"| Train | 2026-05-01 to {TRAIN_END} |",
        f"| Validation | {TRAIN_END[:7]}-26 to {VAL_END} |",
        f"| Test | 2026-08-13 to 2026-08-31 |",
        "",
        "## Baseline",
        "**Strategy**: For each row, multiply `rolling_mean_7` (previous 7-day mean) by 7.",
        "This gives a naive projection of next-7-day demand based purely on recent history.",
        "",
        "| Metric | Baseline (Test) |",
        "|---|---|",
        f"| MAE  | {bm.get('MAE',  np.nan):.3f} |",
        f"| RMSE | {bm.get('RMSE', np.nan):.3f} |",
        f"| MAPE | {bm.get('MAPE', np.nan):.2f}% |",
        f"| R2   | {bm.get('R2',   np.nan):.4f} |",
        "",
        "## Model Comparison Table",
        "",
        df_to_md(comp_df),
        "",
        f"*{xgb_note}*",
        "",
        "## Model Selection",
        f"**Selected model**: `{best_name}`",
        "",
        "**Selection rationale**: The model with the lowest Validation MAE was selected.",
        "MAE is the primary metric because it directly measures average units-sold error,",
        "which maps to real-world inventory planning errors (e.g., over/under-ordering).",
        "RMSE penalises large errors more heavily; R2 shows variance explained.",
        "The selected model was evaluated once on the held-out Test set.",
        "",
        "## Selected Model — Test Performance",
        f"| Metric | {best_name} (Test) |",
        "|---|---|",
        f"| MAE  | {bt.get('MAE',  np.nan):.3f} |",
        f"| RMSE | {bt.get('RMSE', np.nan):.3f} |",
        f"| MAPE | {bt.get('MAPE', np.nan):.2f}% |",
        f"| R2   | {bt.get('R2',   np.nan):.4f} |",
        "",
        "## Saved Artefacts",
        "- `models/demand_model.pkl` — trained model object",
        "- `models/demand_scaler.pkl` — StandardScaler fitted on training data",
        "- `models/demand_model_meta.pkl` — feature list and metadata",
        "- `data/processed/demand_predictions.csv` — test-set predictions",
        "- `reports/demand_feature_importance.csv` — feature importances",
    ]

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"[INFO] Saved: {md_path}")

    return comp_df


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def run_demand_pipeline():
    print("=" * 60)
    print("DEMAND FORECASTING PIPELINE")
    print("=" * 60)

    df, train, val, test = load_and_split()
    feature_cols = [c for c in DEMAND_FEATURES if c in df.columns]
    print(f"[INFO] Using {len(feature_cols)} features.")

    # Baseline
    baseline_results = evaluate_baseline(train, val, test)

    # Train & evaluate models
    trained, scaler, results_val, results_test, _ = train_and_evaluate(
        train, val, test, feature_cols
    )

    # Select best
    best_name  = select_best(results_val)
    best_model = trained[best_name]

    # Save
    save_artifacts(best_model, scaler, best_name)
    save_predictions(test, best_model, scaler)
    save_feature_importance(best_model, best_name, feature_cols)
    save_comparison_reports(baseline_results, results_val, results_test, best_name)

    print("\n[DONE] Demand modelling pipeline complete.")
    return best_model, scaler, best_name, results_val, results_test


if __name__ == "__main__":
    run_demand_pipeline()
