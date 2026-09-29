# STOCKSENSE: IntelliData 2026 Data Science Hackathon

## Project Overview
This repository contains the solution for the "STOCKSENSE" challenge presented by NovaMart Retail Pvt. Ltd. The goal is to build a data-driven decision support solution to forecast demand, predict stock-out risk, and recommend inventory actions to store managers.

**Team Workflow:**
- **Person 1**: Data Generation, Cleaning, Integration, EDA, and Statistical Analysis (Round 1)
- **Person 2**: Feature Engineering and Machine Learning (Round 2)
- **Person 3**: Explainability, Visualizations, and Prototype (Round 3)

## Dataset Description
Due to the constraints of the hackathon, realistic synthetic data was generated matching the provided schema. The dataset simulates a 122-day history (May - August 2026) across 4 stores and 14 products.

- `transactions.csv`: Transaction-level sales data.
- `products.csv`: Product metadata (category, MRP, shelf life).
- `stores.csv`: Store locations and characteristics.
- `inventory.csv`: Daily opening/closing stock and received quantities.
- `external_factors.csv`: Daily weather and event flags for each city.

## Folder Structure
```
STOCKSENSE/
│
├── data/
│   ├── raw/               # Original synthetic datasets (contains intentional data traps)
│   └── processed/         # Cleaned files and final master_daily.csv
│
├── notebooks/
│   └── 01_round1_eda.ipynb # EDA addressing business questions
│
├── src/
│   ├── generate_synthetic_data.py # Script to generate raw data
│   ├── clean_and_build_master.py  # Script for data cleaning and aggregation
│   ├── perform_statistics.py      # Script to run statistical tests
│   └── generate_notebook.py       # Script to generate the EDA notebook
│
├── reports/
│   ├── dataset_overview.md        # Raw data dimensions and metrics
│   ├── data_quality_report.csv    # Log of all cleaning actions
│   ├── data_quality_report.md     # Markdown version of data quality log
│   ├── master_data_validation.md  # Final grain and sanity checks
│   ├── statistical_analysis.md    # Results of hypothesis testing
│   └── round1_summary.md          # Comprehensive summary of Round 1
│
└── README.md
```

## How to Reproduce Round 1

1. **Regenerate Synthetic Data**
   Run the data generation script. This creates 5 CSVs in `data/raw/` with a fixed random seed.
   ```bash
   python src/generate_synthetic_data.py
   ```

2. **Run Cleaning and Build Master Data**
   This script loads the raw data, applies cleaning rules, merges all tables, and generates `data/processed/master_daily.csv`.
   ```bash
   python src/clean_and_build_master.py
   ```

3. **Run Statistical Analysis**
   This script runs hypothesis tests on the master data and outputs to `reports/statistical_analysis.md`.
   ```bash
   python src/perform_statistics.py
   ```

4. **View EDA**
   The Exploratory Data Analysis is available in `notebooks/01_round1_eda.ipynb`.

## Round 1 Outputs
- **Clean Master Data**: `data/processed/master_daily.csv` (Ready for ML!)
- **Data Quality Report**: `reports/data_quality_report.md`
- **Statistical Analysis**: `reports/statistical_analysis.md`
- **Round 1 Summary**: `reports/round1_summary.md`

## Important Assumptions
- All monetary values are in INR (₹).
- Inventory flows follow: `Closing Stock = Opening Stock + Received Quantity - Units Sold`.
- The dataset intentionally includes a cold-start item (`P999`) to simulate sparse history challenges in Machine Learning.

---

## Round 2: Feature Engineering & Machine Learning

**Person 2 (ML Engineer)**

### Features Created

| Category | Features |
|---|---|
| Time | `day_of_week`, `weekend_flag`, `month`, `week_no`, `festival_flag`, `holiday_or_festival` |
| Lag | `lag_1`, `lag_7`, `lag_14`, `lag_closing_1`, `lag_closing_7` |
| Rolling | `rolling_mean_7`, `rolling_mean_14`, `rolling_std_7` |
| Inventory | `days_of_inventory`, `inventory_to_demand_ratio`, `reorder_gap`, `stock_coverage_ratio` |
| Price/Promo | `discount_pct`, `price_change`, `mrp_to_cost_ratio`, `promotion_flag`, `promo_weekend` |
| Store/Product | `store_type_enc`, `category_enc`, `brand_enc`, `region_enc`, `store_id_enc`, `product_id_enc`, `is_cold_start` |

All lag and rolling features are computed within each (store_id, product_id) group sorted by date. No future data is used.

### Demand Target Definition

`next_7_day_demand(t)` = sum of `units_sold` for days t+1 through t+7 per (store_id, product_id). Rows with incomplete 7-day future window are excluded from training.

### Stock-out Target Definition

`stockout_flag = 1` if `closing < reorder_lvl`, else `0`. Represents end-of-day stock falling below the replenishment threshold — operationally equivalent to stock-out risk.

### Temporal Split (No Random Shuffling)

| Split | Dates |
|---|---|
| Train | 2026-05-01 to 2026-07-25 |
| Validation | 2026-07-26 to 2026-08-12 |
| Test | 2026-08-13 onwards |

### Models Tested

**Demand Forecasting**: Baseline (rolling mean), Linear Regression, Random Forest (selected), XGBoost

**Stock-out Classification**: Logistic Regression, Decision Tree, Random Forest, XGBoost (selected)

### Test Performance

| Task | Model | Key Metrics |
|---|---|---|
| Demand | RandomForest | MAE=51.87, RMSE=89.84, MAPE=12.48%, R2=0.947 |
| Stockout | XGBoost | Acc=90.1%, Recall=72.8%, F1=69.8%, AUC=0.950 |

### Leakage Prevention

- Lag/rolling features use only historical rows (shift >= 1)
- next_7_day_demand never used as a predictor
- Inventory-derived features that encode the target excluded from the stockout model
- Temporal split: strictly by date, no random shuffle

### Saved Model Locations

- `models/demand_model.pkl` + `models/demand_scaler.pkl`
- `models/stockout_model.pkl` + `models/stockout_scaler.pkl`

### How to Reproduce Round 2

```bash
python src/train_models.py
```

### Round 2 Outputs

- `data/processed/ml_features.csv` — engineered feature matrix
- `data/processed/demand_predictions.csv` — test-set demand forecasts
- `data/processed/stockout_predictions.csv` — test-set stockout probabilities
- `reports/ml_leakage_check.md` — data leakage audit
- `reports/demand_model_comparison.md` — demand model comparison
- `reports/stockout_model_comparison.md` — stockout model comparison
- `reports/demand_feature_importance.csv` — feature importances
- `reports/stockout_feature_importance.csv` — feature importances
- `reports/ml_error_analysis.md` — error analysis
- `reports/person2_handoff.md` — complete handoff to Person 3
- `notebooks/02_round2_ml.ipynb` — ML documentation notebook

## Round 3: Decision Intelligence & Dashboard Prototype

### Running the Dashboard
To launch the interactive STOCKSENSE decision support dashboard:

```bash
streamlit run app.py
```

### Dashboard Capabilities
1. **Interactive Sidebar Filters**: Store selection, Product catalog, Date range filtering, and Risk level filtering.
2. **Top Executive KPI Cards**: Total Stores, Total Products, Current Closing Inventory, Units Sold, 7-Day Predicted Demand, and High Risk Items.
3. **Inventory Overview**:
   - Total Current Inventory by Store
   - Inventory by Product with category tags
   - Daily Sales Trend & 7-Day Moving Average velocity
   - Top Products by Total Volume Sold
   - Low Stock Alert (Closing stock vs. Reorder levels)
4. **Stock-Out Risk & Decision Intelligence**:
   - Operates in **Data-Exploration Mode** when ML predictions are pending.
   - Loads the real test-set forecasts from `data/processed/demand_predictions.csv` and `data/processed/stockout_predictions.csv`.
   - Categorizes risk:
     - 🔴 **HIGH**: Stock-out probability &ge; 70%
     - 🟡 **MEDIUM**: Stock-out probability 40% – 69%
     - 🟢 **LOW**: Stock-out probability &lt; 40%
   - Visualizes risk distribution, top vulnerable items, store-level risk breakdown, and inventory depth vs. vulnerability scatter plots.
5. **Explainability Engine ("Why is this item at risk?")**:
   - Synthesizes factual drivers: low inventory buffers, projected demand surges, sales velocity, short shelf-life perishability, high customer footfall, and external factors (weather, holidays, weekend peaks).
6. **Actionable Recommendations ("Recommended Action")**:
   - Provides decision-support guidance (e.g., immediate stock replenishment, proactive restock, active watchlist monitoring, inventory rebalancing).
7. **Deep Dive Views & Data Explorer**:
   - Product-level intelligence (MRP, cost, margins, shelf life, demand trends).
   - Store-level intelligence (format, city, square footage, footfall).
   - Interactive data table with CSV export.

### ML Forecast Integration
The dashboard reads these real ML outputs without modifying them:
- `data/processed/demand_predictions.csv` — keyed by `date`, `store_id`, and `product_id`; demand output is `predicted_next_7_day_demand` (normalized internally to `predicted_7d_demand`).
- `data/processed/stockout_predictions.csv` — keyed by `date`, `store_id`, and `product_id`; includes `stockout_probability`.

The published demand test forecasts cover 2026-08-13 through 2026-08-24. Dates without a matching demand row remain unavailable; the dashboard does not fabricate demand. Model probability thresholds are HIGH at 70%+, WATCH at 40%+, and SAFE below 40%. When forecast demand exceeds current inventory, a separate stock-versus-demand rule can escalate the action category; that rule does not alter or masquerade as the model probability.

