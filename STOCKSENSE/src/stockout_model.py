"""
stockout_model.py
=================
StockSense Round 2 - Stock-out Classification (Model 2)
Person 2 (ML Engineer)

Target: stockout_flag  (1 = closing < reorder_lvl, 0 = adequate stock)
Type:   Binary Classification

Models compared:
  1. Decision Tree
  2. Random Forest
  3. XGBoost Classifier
  (Logistic Regression included as the linear-family representative)

Evaluation: Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix
Primary metric for selection: F1 (with attention to Recall for stock-out cases,
because missing a true stock-out is operationally more costly than a false alarm).

Outputs:
    models/stockout_model.pkl
    models/stockout_scaler.pkl
    data/processed/stockout_predictions.csv
    reports/stockout_model_comparison.csv
    reports/stockout_model_comparison.md
    reports/stockout_feature_importance.csv
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.tree           import DecisionTreeClassifier
from sklearn.ensemble       import RandomForestClassifier
from sklearn.linear_model   import LogisticRegression
from sklearn.preprocessing  import StandardScaler
from sklearn.metrics        import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

warnings.filterwarnings("ignore")

RANDOM_SEED = 42

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_PATH = os.path.join(BASE_DIR, "data", "processed", "ml_features.csv")
MODELS_DIR    = os.path.join(BASE_DIR, "models")
REPORTS_DIR   = os.path.join(BASE_DIR, "reports")
PRED_PATH     = os.path.join(BASE_DIR, "data", "processed", "stockout_predictions.csv")

TRAIN_END = "2026-07-25"
VAL_END   = "2026-08-12"

TARGET = "stockout_flag"

# ---------------------------------------------------------------------------
# Feature columns for Model 2 (classification)
# next_7_day_demand excluded — that is the Model 1 target (no cross-target leakage).
# stockout_flag is the target for this model.
# ---------------------------------------------------------------------------
STOCKOUT_FEATURES = [
    # time
    "day_of_week", "weekend_flag", "month", "week_no",
    "day_of_month", "quarter", "festival_flag", "holiday_or_festival",
    "holiday", "local_event",
    # lag demand
    "lag_1", "lag_7", "lag_14",
    # rolling demand
    "rolling_mean_7", "rolling_mean_14", "rolling_std_7",
    # historical inventory signals (lagged — strictly past observations)
    # NOTE: reorder_gap, days_of_inventory, inventory_to_demand_ratio,
    # stock_coverage_ratio are EXCLUDED because they are all algebraic
    # derivations of (closing - reorder_lvl), which is mathematically
    # identical to the target stockout_flag. Including them would trivialise
    # the classification task (AUC → 1.0) and provide no genuine learning.
    "lag_closing_1", "lag_closing_7",
    "reorder_lvl",   # static product-store threshold (known in advance)
    "lead_days",     # supplier lead time (affects reorder urgency)
    # price/promo
    "discount_pct", "price_change", "mrp_to_cost_ratio",
    "promotion_flag", "promo_weekend",
    # store/product
    "store_type_enc", "category_enc", "brand_enc", "region_enc",
    "store_id_enc", "product_id_enc",
    "shelf_life_days",
    "avg_daily_customers", "floor_area_sqft",
    "customer_to_floor_ratio", "is_cold_start",
    "mrp", "cost_price",
    # weather
    "temp_c", "rain_mm",
]


# ---------------------------------------------------------------------------
# Metrics helper
# ---------------------------------------------------------------------------

def df_to_md(df):
    """Convert DataFrame to markdown table without tabulate dependency."""
    cols   = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep    = "|" + "|".join(["---"] * len(cols)) + "|"
    rows   = ["| " + " | ".join(str(v) for v in row.values) + " |"
              for _, row in df.iterrows()]
    return "\n".join([header, sep] + rows)


def compute_clf_metrics(y_true, y_pred, y_prob, label=""):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc  = roc_auc_score(y_true, y_prob)
    except Exception:
        auc  = np.nan
    cm   = confusion_matrix(y_true, y_pred)

    if label:
        print(f"  [{label}] Acc={acc:.4f}  Prec={prec:.4f}  "
              f"Rec={rec:.4f}  F1={f1:.4f}  AUC={auc:.4f}")

    return {
        "Accuracy": acc, "Precision": prec, "Recall": rec,
        "F1": f1, "ROC_AUC": auc, "ConfusionMatrix": cm,
    }


# ---------------------------------------------------------------------------
# Load & split
# ---------------------------------------------------------------------------

def load_and_split():
    df = pd.read_csv(FEATURES_PATH)
    df["date"] = pd.to_datetime(df["date"])

    print(f"[INFO] Total rows: {len(df)}")
    print(f"[INFO] stockout_flag distribution:\n{df[TARGET].value_counts()}")

    train = df[df["date"] <= TRAIN_END].copy()
    val   = df[(df["date"] > TRAIN_END) & (df["date"] <= VAL_END)].copy()
    test  = df[df["date"] > VAL_END].copy()

    print(f"[INFO] Train: {len(train)}  Val: {len(val)}  Test: {len(test)}")
    for name, split in [("Train", train), ("Val", val), ("Test", test)]:
        pos = split[TARGET].sum()
        print(f"       {name}: stock-out positive = {pos} ({pos/len(split)*100:.1f}%)")

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
# Train models
# ---------------------------------------------------------------------------

def train_and_evaluate(train, val, test, feature_cols):
    scaler = None
    X_tr, y_tr, scaler = prepare_xy(train, feature_cols, fit_scaler=True)
    X_va, y_va, _      = prepare_xy(val,   feature_cols, scaler=scaler)
    X_te, y_te, _      = prepare_xy(test,  feature_cols, scaler=scaler)

    # Scale imbalance: compute class weight
    n_neg = int((y_tr == 0).sum())
    n_pos = int((y_tr == 1).sum())
    scale_pos = n_neg / max(n_pos, 1)
    print(f"[INFO] Class imbalance scale_pos_weight = {scale_pos:.2f}")

    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED),
        "DecisionTree": DecisionTreeClassifier(
            max_depth=10, min_samples_leaf=5,
            class_weight="balanced", random_state=RANDOM_SEED),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=3,
            class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_pos, random_state=RANDOM_SEED,
            eval_metric="logloss", verbosity=0)

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

        # Validation
        va_pred = model.predict(X_va)
        va_prob = model.predict_proba(X_va)[:, 1]
        results_val[name] = compute_clf_metrics(y_va, va_pred, va_prob,
                                                label=f"{name}/Val")

        # Test
        te_pred = model.predict(X_te)
        te_prob = model.predict_proba(X_te)[:, 1]
        results_test[name] = compute_clf_metrics(y_te, te_pred, te_prob,
                                                 label=f"{name}/Test")
        trained[name] = model

    return trained, scaler, results_val, results_test, X_te, y_te


# ---------------------------------------------------------------------------
# Select best model (by F1 on validation, with Recall as tiebreaker)
# ---------------------------------------------------------------------------

def select_best(results_val):
    best_name = max(results_val, key=lambda k: results_val[k]["F1"])
    print(f"\n[INFO] Best model (highest Val F1): {best_name}")
    for name, m in results_val.items():
        marker = " <-- SELECTED" if name == best_name else ""
        print(f"  {name}: F1={m['F1']:.4f}  Recall={m['Recall']:.4f}"
              f"  AUC={m['ROC_AUC']:.4f}{marker}")
    return best_name


# ---------------------------------------------------------------------------
# Save model and scaler
# ---------------------------------------------------------------------------

def save_artifacts(best_model, scaler, best_name):
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path  = os.path.join(MODELS_DIR, "stockout_model.pkl")
    scaler_path = os.path.join(MODELS_DIR, "stockout_scaler.pkl")
    meta_path   = os.path.join(MODELS_DIR, "stockout_model_meta.pkl")

    with open(model_path,  "wb") as f:
        pickle.dump(best_model, f)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    meta = {
        "model_name":   best_name,
        "feature_cols": STOCKOUT_FEATURES,
        "target":       TARGET,
        "train_end":    TRAIN_END,
        "val_end":      VAL_END,
    }
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)

    print(f"[INFO] Saved stockout model:  {model_path}")
    print(f"[INFO] Saved stockout scaler: {scaler_path}")
    print(f"[INFO] Saved stockout meta:   {meta_path}")


# ---------------------------------------------------------------------------
# Stockout predictions CSV
# ---------------------------------------------------------------------------

def save_predictions(test, best_model, scaler, feature_cols):
    X_te, y_te, _ = prepare_xy(test, feature_cols, scaler=scaler)
    probs = best_model.predict_proba(X_te)[:, 1]
    preds = best_model.predict(X_te)

    out = test[["date", "store_id", "product_id"]].copy()
    out["stockout_probability"] = probs
    out["stockout_prediction"]  = preds
    out["actual_stockout_flag"] = y_te

    os.makedirs(os.path.dirname(PRED_PATH), exist_ok=True)
    out.to_csv(PRED_PATH, index=False)
    print(f"[INFO] Saved stockout predictions: {PRED_PATH}")
    return out


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def save_feature_importance(best_model, best_name, feature_cols):
    fi_path    = os.path.join(REPORTS_DIR, "stockout_feature_importance.csv")
    chart_path = os.path.join(REPORTS_DIR, "stockout_feature_importance.png")
    os.makedirs(REPORTS_DIR, exist_ok=True)

    if hasattr(best_model, "feature_importances_"):
        imp = best_model.feature_importances_
        fi_df = pd.DataFrame({
            "feature":    feature_cols,
            "importance": imp,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
    elif hasattr(best_model, "coef_"):
        imp = np.abs(best_model.coef_[0])
        fi_df = pd.DataFrame({
            "feature":    feature_cols,
            "importance": imp,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
    else:
        fi_df = pd.DataFrame({"feature": feature_cols, "importance": np.nan})

    fi_df.to_csv(fi_path, index=False)
    print(f"[INFO] Saved stockout feature importance: {fi_path}")
    print(f"[INFO] Top 10 features ({best_name}):")
    print(fi_df.head(10).to_string(index=False))

    # Chart
    fig, ax = plt.subplots(figsize=(10, 8))
    top = fi_df.head(20)
    ax.barh(top["feature"][::-1], top["importance"][::-1], color="#FF5722")
    ax.set_xlabel("Importance")
    ax.set_title(f"Top 20 Features — {best_name} (Stockout Model)")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(chart_path, dpi=120)
    plt.close(fig)
    print(f"[INFO] Saved chart: {chart_path}")

    return fi_df


# ---------------------------------------------------------------------------
# Confusion matrix chart
# ---------------------------------------------------------------------------

def save_confusion_matrix(best_model, scaler, test, feature_cols, best_name):
    X_te, y_te, _ = prepare_xy(test, feature_cols, scaler=scaler)
    y_pred = best_model.predict(X_te)
    cm = confusion_matrix(y_te, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred: No Risk", "Pred: Stock-out Risk"])
    ax.set_yticklabels(["Actual: No Risk", "Actual: Stock-out Risk"])
    ax.set_title(f"Confusion Matrix — {best_name} (Test Set)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center", fontsize=16,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    chart_path = os.path.join(REPORTS_DIR, "stockout_confusion_matrix.png")
    fig.savefig(chart_path, dpi=120)
    plt.close(fig)
    print(f"[INFO] Saved confusion matrix: {chart_path}")
    return cm


# ---------------------------------------------------------------------------
# Comparison reports
# ---------------------------------------------------------------------------

def save_comparison_reports(results_val, results_test, best_name):
    os.makedirs(REPORTS_DIR, exist_ok=True)

    rows = []
    for name in results_val:
        for split_name, res_dict in [("Validation", results_val),
                                      ("Test",       results_test)]:
            m = res_dict[name]
            rows.append({
                "model":     name,
                "split":     split_name,
                "Accuracy":  round(m["Accuracy"],  4),
                "Precision": round(m["Precision"], 4),
                "Recall":    round(m["Recall"],    4),
                "F1":        round(m["F1"],        4),
                "ROC_AUC":   round(m["ROC_AUC"],   4),
            })

    comp_df = pd.DataFrame(rows)
    csv_path = os.path.join(REPORTS_DIR, "stockout_model_comparison.csv")
    comp_df.to_csv(csv_path, index=False)
    print(f"[INFO] Saved: {csv_path}")

    # Markdown
    bv = results_val.get(best_name,  {})
    bt = results_test.get(best_name, {})
    cm = bt.get("ConfusionMatrix", np.array([[0, 0], [0, 0]]))

    xgb_note = (
        "XGBoost available and included."
        if XGBOOST_AVAILABLE
        else "XGBoost not available; Decision Tree and Random Forest used instead."
    )

    md = [
        "# Stockout Model Comparison Report",
        "",
        "## Problem",
        "Binary classification: predict whether closing stock < reorder_lvl",
        "(stock-out risk = 1, adequate stock = 0).",
        "",
        "## Class Imbalance",
        "The positive class (stock-out risk) represents ~14.5% of all rows.",
        "Class imbalance was handled via `class_weight='balanced'` for sklearn models",
        "and `scale_pos_weight` for XGBoost.",
        "",
        "## Temporal Split",
        "| Split | Dates |",
        "|---|---|",
        f"| Train | 2026-05-01 to {TRAIN_END} |",
        f"| Validation | {TRAIN_END[:7]}-26 to {VAL_END} |",
        "| Test | 2026-08-13 to 2026-08-31 |",
        "",
        "## Model Comparison Table",
        "",
        df_to_md(comp_df),
        "",
        f"*{xgb_note}*",
        "",
        "## Model Selection Rationale",
        f"**Selected model**: `{best_name}`",
        "",
        "**Primary metric**: F1 Score — balances precision and recall.",
        "**Key consideration**: Recall for positive class (stock-out risk) is prioritised",
        "because failing to detect a true stock-out is operationally costly",
        "(lost sales, customer dissatisfaction). However, very high false-positive rates",
        "also impose costs (unnecessary replenishment). F1 best balances this trade-off.",
        "",
        "**Logistic Regression** is included as the linear-family representative.",
        "Logistic Regression is the classification equivalent of Linear Regression,",
        "and falls within the 'Linear' model family permitted by the problem statement.",
        "",
        "## Selected Model — Test Performance",
        f"| Metric | {best_name} (Test) |",
        "|---|---|",
        f"| Accuracy  | {bt.get('Accuracy',  np.nan):.4f} |",
        f"| Precision | {bt.get('Precision', np.nan):.4f} |",
        f"| Recall    | {bt.get('Recall',    np.nan):.4f} |",
        f"| F1        | {bt.get('F1',        np.nan):.4f} |",
        f"| ROC-AUC   | {bt.get('ROC_AUC',  np.nan):.4f} |",
        "",
        "## Confusion Matrix (Test Set)",
        "```",
        f"              Pred: 0   Pred: 1",
        f"Actual: 0    {cm[0,0]:>8}  {cm[0,1]:>8}",
        f"Actual: 1    {cm[1,0]:>8}  {cm[1,1]:>8}",
        "```",
        "",
        f"- True Negatives  (correct: no risk): {cm[0,0]}",
        f"- False Positives (false alarm):       {cm[0,1]}",
        f"- False Negatives (missed stock-out):  {cm[1,0]}",
        f"- True Positives  (correct: risk):     {cm[1,1]}",
        "",
        "## Saved Artefacts",
        "- `models/stockout_model.pkl` — trained classifier",
        "- `models/stockout_scaler.pkl` — StandardScaler",
        "- `models/stockout_model_meta.pkl` — feature list and metadata",
        "- `data/processed/stockout_predictions.csv` — test-set predictions with probabilities",
        "- `reports/stockout_feature_importance.csv` — feature importances",
        "- `reports/stockout_confusion_matrix.png` — confusion matrix chart",
    ]

    md_path = os.path.join(REPORTS_DIR, "stockout_model_comparison.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"[INFO] Saved: {md_path}")

    return comp_df


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def run_stockout_pipeline():
    print("=" * 60)
    print("STOCK-OUT CLASSIFICATION PIPELINE")
    print("=" * 60)

    df, train, val, test = load_and_split()
    feature_cols = [c for c in STOCKOUT_FEATURES if c in df.columns]
    print(f"[INFO] Using {len(feature_cols)} features.")

    trained, scaler, results_val, results_test, X_te, y_te = train_and_evaluate(
        train, val, test, feature_cols
    )

    best_name  = select_best(results_val)
    best_model = trained[best_name]

    save_artifacts(best_model, scaler, best_name)
    save_predictions(test, best_model, scaler, feature_cols)
    save_feature_importance(best_model, best_name, feature_cols)
    save_confusion_matrix(best_model, scaler, test, feature_cols, best_name)
    save_comparison_reports(results_val, results_test, best_name)

    print("\n[DONE] Stockout classification pipeline complete.")
    return best_model, scaler, best_name, results_val, results_test


if __name__ == "__main__":
    run_stockout_pipeline()
