# Demand Model Comparison Report

## Temporal Split
| Split | Dates |
|---|---|
| Train | 2026-05-01 to 2026-07-25 |
| Validation | 2026-07-26 to 2026-08-12 |
| Test | 2026-08-13 to 2026-08-31 |

## Baseline
**Strategy**: For each row, multiply `rolling_mean_7` (previous 7-day mean) by 7.
This gives a naive projection of next-7-day demand based purely on recent history.

| Metric | Baseline (Test) |
|---|---|
| MAE  | 55.499 |
| RMSE | 89.680 |
| MAPE | 13.69% |
| R2   | 0.9477 |

## Model Comparison Table

| model | split | MAE | RMSE | MAPE | R2 |
|---|---|---|---|---|---|
| Baseline (rolling_mean_7 x7) | Test | 55.4995 | 89.6796 | 13.687 | 0.9477 |
| LinearRegression | Validation | 29.0448 | 45.6501 | 9.774 | 0.9818 |
| RandomForest | Validation | 26.032 | 42.7945 | 8.258 | 0.984 |
| XGBoost | Validation | 26.134 | 39.9134 | 8.2881 | 0.9861 |
| LinearRegression | Test | 79.9338 | 135.5238 | 26.4614 | 0.8799 |
| RandomForest | Test | 51.8688 | 89.8411 | 12.4817 | 0.9472 |
| XGBoost | Test | 51.9219 | 90.2022 | 12.6799 | 0.9468 |

*XGBoost was available (v3.0.5) and included in comparison.*

## Model Selection
**Selected model**: `RandomForest`

**Selection rationale**: The model with the lowest Validation MAE was selected.
MAE is the primary metric because it directly measures average units-sold error,
which maps to real-world inventory planning errors (e.g., over/under-ordering).
RMSE penalises large errors more heavily; R2 shows variance explained.
The selected model was evaluated once on the held-out Test set.

## Selected Model — Test Performance
| Metric | RandomForest (Test) |
|---|---|
| MAE  | 51.869 |
| RMSE | 89.841 |
| MAPE | 12.48% |
| R2   | 0.9472 |

## Saved Artefacts
- `models/demand_model.pkl` — trained model object
- `models/demand_scaler.pkl` — StandardScaler fitted on training data
- `models/demand_model_meta.pkl` — feature list and metadata
- `data/processed/demand_predictions.csv` — test-set predictions
- `reports/demand_feature_importance.csv` — feature importances