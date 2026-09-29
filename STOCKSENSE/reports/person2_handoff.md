# Person 2 → Person 3 Handoff Report
**StockSense Round 2 — ML Engineer Handoff**
**Date**: 2026-09-29

---

## 1. Final Dataset Dimensions

| Metric | Value |
|---|---|
| Source file | `data/processed/master_daily.csv` |
| Total rows | 6,440 |
| Columns (raw) | 34 |
| Date range | 2026-05-01 to 2026-08-31 |
| Stores | 4 (S01, S02, S03, S04) |
| Products | 14 (including P999 cold-start) |
| Grain | One row per Date × Store × Product |
| Missing values | 0 (fully clean) |
| Duplicate keys | 0 |

---

## 2. Usable Modelling Rows

| Purpose | Rows |
|---|---|
| Feature matrix (all rows) | 6,440 |
| Demand model (target not NaN) | 6,048 |
| Stockout model (all rows) | 6,440 |
| Rows excluded (last 7 days per group) | 392 |

---

## 3. Feature List

The feature matrix (`data/processed/ml_features.csv`) contains 68 columns including:

### Time Features
`day_of_week`, `weekend_flag`, `month`, `week_no`, `day_of_month`, `quarter`,
`festival_flag`, `holiday_or_festival`

### Lag Features (per Store × Product group)
`lag_1`, `lag_7`, `lag_14`, `lag_closing_1`, `lag_closing_7`

### Rolling Features (shift(1).rolling, no current-day data)
`rolling_mean_7`, `rolling_mean_14`, `rolling_std_7`

### Inventory Features
`days_of_inventory`, `inventory_to_demand_ratio`, `reorder_gap`, `stock_coverage_ratio`

> **Note for Person 3**: `reorder_gap`, `days_of_inventory`, `inventory_to_demand_ratio`,
> and `stock_coverage_ratio` are available for dashboard use (they are current-day observations),
> but were **excluded from the stockout model** because they algebraically encode the target.
> Use them directly for threshold-based risk display.

### Price / Promotion Features
`discount_pct`, `price_change`, `mrp_to_cost_ratio`, `promotion_flag`, `promo_weekend`

### Store / Product Encoded Features
`store_type_enc`, `category_enc`, `brand_enc`, `region_enc`,
`store_id_enc`, `product_id_enc`, `is_cold_start`

### Numeric Store/Product Attributes
`shelf_life_days`, `lead_days`, `avg_daily_customers`, `floor_area_sqft`,
`customer_to_floor_ratio`, `mrp`, `cost_price`, `reorder_lvl`

### Weather
`temp_c`, `rain_mm`

---

## 4. Demand Target Definition

**Column**: `next_7_day_demand`

**Formula**: For each (store_id, product_id, date=t):
```
next_7_day_demand(t) = units_sold(t+1) + units_sold(t+2) + ... + units_sold(t+7)
```

**Implementation**: Within each (store_id, product_id) group sorted by date,
reverse-order rolling sum of window=7 applied with shift(1). The last 7 rows
per group receive NaN and are excluded from supervised training.

**No leakage**: Only future `units_sold` constitutes the target. No features use future data.

---

## 5. Stock-out Target Definition

**Column**: `stockout_flag`

**Formula**: `stockout_flag = 1 if closing < reorder_lvl else 0`

**Rationale**:
- `closing` = end-of-day stock after all sales
- `reorder_lvl` = business-defined threshold for placing a replenishment order
- When closing stock falls below the reorder level, the store is in an
  operationally high-risk state and may run out before the next delivery arrives

**No leakage**: Both `closing` and `reorder_lvl` are same-day observable values.

---

## 6. Train / Validation / Test Date Ranges

| Split | Start Date | End Date | Rows (demand) | Rows (stockout) |
|---|---|---|---|---|
| **Train** | 2026-05-01 | 2026-07-25 | 4,472 | 4,472 |
| **Validation** | 2026-07-26 | 2026-08-12 | 936 | 936 |
| **Test** | 2026-08-13 | 2026-08-24 | 640 | 1,032 |

> **Note**: Demand test ends 2026-08-24 because the last 7 days (2026-08-25 to 2026-08-31)
> have no valid `next_7_day_demand` target. Stockout test uses all rows through 2026-08-31.

**Temporal integrity**: No random shuffling. Split strictly by date threshold.

---

## 7. Demand Baseline Metrics

Baseline strategy: `predicted = rolling_mean_7 × 7`

| Split | MAE | RMSE | MAPE | R² |
|---|---|---|---|---|
| Validation | 30.71 | 46.43 | 9.95% | 0.9812 |
| Test | 55.50 | 89.68 | 13.69% | 0.9477 |

---

## 8. Demand Model Comparison

| Model | Split | MAE | RMSE | MAPE | R² |
|---|---|---|---|---|---|
| Baseline (rolling_mean × 7) | Test | 55.50 | 89.68 | 13.69% | 0.9477 |
| Linear Regression | Validation | 29.05 | 45.65 | 9.77% | 0.9818 |
| Linear Regression | Test | 79.93 | 135.52 | 26.46% | 0.8799 |
| Random Forest | Validation | 26.03 | 42.80 | 8.26% | 0.9840 |
| Random Forest | Test | 51.87 | 89.84 | 12.48% | 0.9472 |
| XGBoost | Validation | 26.13 | 39.91 | 8.29% | 0.9861 |
| XGBoost | Test | 51.92 | 90.20 | 12.68% | 0.9468 |

---

## 9. Selected Demand Model

**Model**: `RandomForestRegressor`

**Selection reason**: Lowest Validation MAE (26.03). XGBoost is essentially tied
(26.13 MAE) and has slightly better RMSE on validation, but RandomForest is selected
by the primary MAE criterion. Both generalize similarly on the test set.

**Performance summary**: ~15% improvement over naive baseline on validation MAE.

---

## 10. Final Demand Test Metrics

| Metric | Value |
|---|---|
| MAE | 51.87 units |
| RMSE | 89.84 units |
| MAPE | 12.48% |
| R² | 0.9472 |
| Mean Error (bias) | +37.59 units (slight under-prediction) |

---

## 11. Stockout Model Comparison

| Model | Split | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | Validation | 0.8077 | 0.4318 | 0.9638 | 0.5964 | 0.9298 |
| Logistic Regression | Test | 0.7248 | 0.3555 | 0.9259 | 0.5137 | 0.8617 |
| Decision Tree | Validation | 0.8280 | 0.4569 | 0.8841 | 0.6025 | 0.8766 |
| Decision Tree | Test | 0.7713 | 0.3879 | 0.7901 | 0.5203 | 0.8150 |
| Random Forest | Validation | 0.8889 | 0.6250 | 0.6159 | 0.6204 | 0.9420 |
| Random Forest | Test | 0.8798 | 0.6532 | 0.5000 | 0.5664 | 0.9047 |
| **XGBoost** | **Validation** | **0.9391** | **0.7485** | **0.8841** | **0.8106** | **0.9789** |
| **XGBoost** | **Test** | **0.9012** | **0.6705** | **0.7284** | **0.6982** | **0.9503** |

---

## 12. Selected Stockout Model

**Model**: `XGBClassifier`

**Selection reason**: Highest Validation F1 = 0.8106. XGBoost balances both
Precision (0.75) and Recall (0.88) on validation — the best combination for
operational stock-out management. Logistic Regression achieves higher Recall
but at unacceptably low Precision (too many false alarms).

---

## 13. Final Stockout Test Metrics

| Metric | Value |
|---|---|
| Accuracy | 90.12% |
| Precision | 67.05% |
| Recall | 72.84% |
| F1 | 69.82% |
| ROC-AUC | 0.9503 |
| True Negatives | 812 |
| False Positives | 58 |
| False Negatives | 44 |
| True Positives | 118 |

---

## 14. Class Balance

| Class | Count | Percentage |
|---|---|---|
| 0 — Adequate stock | 5,509 | 85.5% |
| 1 — Stock-out risk | 931 | 14.5% |
| **Total** | **6,440** | |

Imbalance handled via `class_weight='balanced'` (sklearn) and `scale_pos_weight=6.09` (XGBoost).

---

## 15. Top Demand Model Features (Random Forest)

| Rank | Feature | Importance |
|---|---|---|
| 1 | `rolling_mean_14` | 0.887 |
| 2 | `rolling_mean_7` | 0.060 |
| 3 | `lag_14` | 0.014 |
| 4 | `reorder_gap` | 0.006 |
| 5 | `brand_enc` | 0.005 |
| 6 | `lag_7` | 0.005 |
| 7 | `mrp_to_cost_ratio` | 0.004 |
| 8 | `lag_1` | 0.003 |
| 9 | `product_id_enc` | 0.003 |
| 10 | `cost_price` | 0.002 |

**Key insight**: Historical rolling average (`rolling_mean_14`) dominates demand forecasting,
capturing product-specific demand velocity. Promotion and time features contribute less
but are important for edge cases.

---

## 16. Top Stockout Model Features (XGBoost)

| Rank | Feature | Importance |
|---|---|---|
| 1 | `lag_closing_1` | 0.089 |
| 2 | `reorder_lvl` | 0.084 |
| 3 | `rolling_mean_7` | 0.068 |
| 4 | `lag_closing_7` | 0.067 |
| 5 | `rolling_mean_14` | 0.066 |
| 6 | `promotion_flag` | 0.044 |
| 7 | `lag_7` | 0.041 |
| 8 | `lag_14` | 0.034 |
| 9 | `lag_1` | 0.032 |
| 10 | `discount_pct` | 0.025 |

**Key insight**: Yesterday's closing stock (`lag_closing_1`) and the reorder threshold
(`reorder_lvl`) are the strongest predictors of stockout risk — the model learns the
structural relationship between demand velocity and inventory trajectory.
Promotion flag is the 6th most important feature — confirming that promotions
drive stock-out risk (as shown in the statistical analysis in Round 1).

---

## 17. Important Error Analysis Findings

### Demand Model
1. **Mean Error = +37.59 units** — slight positive bias (model tends to under-predict).
   For Person 3: add a small safety buffer (e.g., ×1.05) to predicted demand when
   computing recommended order quantities to compensate for systematic under-prediction.
2. **P999 is the highest-error product** — cold-start product with only 11 days of history.
   Use category-mean forecast (Beverages/Energy Drink) as fallback for P999 predictions.
3. **Hypermarket (S02) shows highest absolute errors** due to higher volume.

### Stockout Model
1. **44 False Negatives** (missed stock-outs out of 162 positives on test set).
   These are the most dangerous cases — consider lowering the classification threshold
   from 0.5 to 0.4 or 0.35 to reduce FN at the cost of more FP.
2. **58 False Positives** — manageable false alarms that trigger unnecessary early replenishment.
3. **Promotion-driven stock-outs** are harder to predict — the model underperforms on
   days immediately following promotions.
4. The `stockout_probability` column enables threshold tuning without retraining.

---

## 18. Exact Model File Paths

| Artefact | Path |
|---|---|
| Demand model | `models/demand_model.pkl` |
| Demand scaler | `models/demand_scaler.pkl` |
| Demand metadata | `models/demand_model_meta.pkl` |
| Stockout model | `models/stockout_model.pkl` |
| Stockout scaler | `models/stockout_scaler.pkl` |
| Stockout metadata | `models/stockout_model_meta.pkl` |

**Loading example**:
```python
import pickle

with open('models/demand_model.pkl', 'rb') as f:
    demand_model = pickle.load(f)
with open('models/demand_scaler.pkl', 'rb') as f:
    demand_scaler = pickle.load(f)

# To predict: scale features first, then call model.predict()
X_scaled = demand_scaler.transform(X_new[demand_feature_cols])
predictions = demand_model.predict(X_scaled)
```

**Feature lists are stored in metadata files**:
```python
with open('models/demand_model_meta.pkl', 'rb') as f:
    meta = pickle.load(f)
feature_cols = meta['feature_cols']
```

---

## 19. Exact Prediction File Paths

| File | Path | Columns |
|---|---|---|
| Demand predictions | `data/processed/demand_predictions.csv` | date, store_id, product_id, actual_next_7_day_demand, predicted_next_7_day_demand |
| Stockout predictions | `data/processed/stockout_predictions.csv` | date, store_id, product_id, stockout_probability, stockout_prediction, actual_stockout_flag |
| Feature matrix | `data/processed/ml_features.csv` | All 68 columns including all features and targets |

---

## 20. Limitations & Warnings for Person 3

> [!WARNING]
> The following limitations MUST be communicated to stakeholders.

1. **Short history (123 days)**: Only May–August 2026. No annual seasonality (e.g., winter
   holidays, Diwali) is captured. Model performance in seasonal peaks is unknown.

2. **Cold-start product (P999)**: Only 11 days of history. The demand model has no
   meaningful lag features for this product. Use a fallback strategy:
   - Use category-level (Energy Drink / Beverages) average as the demand forecast
   - Stockout risk = `lag_closing_1 < reorder_lvl` as a direct rule

3. **Synthetic data**: The dataset is synthetically generated. Real-world deployment
   requires retraining on actual NovaMart transaction data.

4. **Demand under-prediction bias**: Mean error = +37.59 units. The model systematically
   under-predicts. Apply a safety buffer when computing order quantities.

5. **Stockout threshold**: Default classification threshold = 0.5. Recommend Person 3
   tune this using `stockout_probability` column based on business cost of FN vs FP.

6. **No supplier failure modelling**: Lead times are simplified (1–4 days). The model
   does not account for supply chain disruptions.

7. **Feature encoding**: Categorical features are label-encoded. If new stores or
   products are added, the encoding maps in the metadata files must be updated before
   running inference.

---

## Summary of All Deliverables

| Deliverable | Status |
|---|---|
| `data/processed/ml_features.csv` | ✅ Complete |
| `data/processed/demand_predictions.csv` | ✅ Complete |
| `data/processed/stockout_predictions.csv` | ✅ Complete |
| `models/demand_model.pkl` | ✅ Complete |
| `models/demand_scaler.pkl` | ✅ Complete |
| `models/stockout_model.pkl` | ✅ Complete |
| `models/stockout_scaler.pkl` | ✅ Complete |
| `reports/ml_leakage_check.md` | ✅ Complete |
| `reports/demand_model_comparison.csv` | ✅ Complete |
| `reports/demand_model_comparison.md` | ✅ Complete |
| `reports/stockout_model_comparison.csv` | ✅ Complete |
| `reports/stockout_model_comparison.md` | ✅ Complete |
| `reports/demand_feature_importance.csv` | ✅ Complete |
| `reports/stockout_feature_importance.csv` | ✅ Complete |
| `reports/ml_error_analysis.md` | ✅ Complete |
| `notebooks/02_round2_ml.ipynb` | ✅ Complete |
| `src/feature_engineering.py` | ✅ Complete |
| `src/demand_model.py` | ✅ Complete |
| `src/stockout_model.py` | ✅ Complete |
| `src/train_models.py` | ✅ Complete |

---

## PERSON 3 CAN NOW START.

Person 3 should use the following to build the recommendation engine and dashboard:

- **Stockout risk categories**: Use `stockout_probability` from `stockout_predictions.csv`
  to create 3 tiers: Green (< 0.3), Amber (0.3–0.6), Red (> 0.6)
- **Recommended stock**: `predicted_next_7_day_demand × lead_days / 7 + safety_buffer`
- **Reorder quantity**: `recommended_stock - current_closing + reorder_lvl`
- **Action list**: Products where `stockout_prediction == 1` AND `closing < reorder_lvl`
- **Explanations**: Use feature importance rankings above + error analysis findings
- **Dashboard**: Use `ml_features.csv` for all computed signals; load saved models
  via the model files above for live scoring

Person 3 does NOT need to retrain models or modify the feature engineering pipeline.
