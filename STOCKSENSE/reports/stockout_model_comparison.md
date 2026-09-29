# Stockout Model Comparison Report

## Problem
Binary classification: predict whether closing stock < reorder_lvl
(stock-out risk = 1, adequate stock = 0).

## Class Imbalance
The positive class (stock-out risk) represents ~14.5% of all rows.
Class imbalance was handled via `class_weight='balanced'` for sklearn models
and `scale_pos_weight` for XGBoost.

## Temporal Split
| Split | Dates |
|---|---|
| Train | 2026-05-01 to 2026-07-25 |
| Validation | 2026-07-26 to 2026-08-12 |
| Test | 2026-08-13 to 2026-08-31 |

## Model Comparison Table

| model | split | Accuracy | Precision | Recall | F1 | ROC_AUC |
|---|---|---|---|---|---|---|
| LogisticRegression | Validation | 0.8077 | 0.4318 | 0.9638 | 0.5964 | 0.9298 |
| LogisticRegression | Test | 0.7248 | 0.3555 | 0.9259 | 0.5137 | 0.8617 |
| DecisionTree | Validation | 0.828 | 0.4569 | 0.8841 | 0.6025 | 0.8766 |
| DecisionTree | Test | 0.7713 | 0.3879 | 0.7901 | 0.5203 | 0.815 |
| RandomForest | Validation | 0.8889 | 0.625 | 0.6159 | 0.6204 | 0.942 |
| RandomForest | Test | 0.8798 | 0.6532 | 0.5 | 0.5664 | 0.9047 |
| XGBoost | Validation | 0.9391 | 0.7485 | 0.8841 | 0.8106 | 0.9789 |
| XGBoost | Test | 0.9012 | 0.6705 | 0.7284 | 0.6982 | 0.9503 |

*XGBoost available and included.*

## Model Selection Rationale
**Selected model**: `XGBoost`

**Primary metric**: F1 Score — balances precision and recall.
**Key consideration**: Recall for positive class (stock-out risk) is prioritised
because failing to detect a true stock-out is operationally costly
(lost sales, customer dissatisfaction). However, very high false-positive rates
also impose costs (unnecessary replenishment). F1 best balances this trade-off.

**Logistic Regression** is included as the linear-family representative.
Logistic Regression is the classification equivalent of Linear Regression,
and falls within the 'Linear' model family permitted by the problem statement.

## Selected Model — Test Performance
| Metric | XGBoost (Test) |
|---|---|
| Accuracy  | 0.9012 |
| Precision | 0.6705 |
| Recall    | 0.7284 |
| F1        | 0.6982 |
| ROC-AUC   | 0.9503 |

## Confusion Matrix (Test Set)
```
              Pred: 0   Pred: 1
Actual: 0         812        58
Actual: 1          44       118
```

- True Negatives  (correct: no risk): 812
- False Positives (false alarm):       58
- False Negatives (missed stock-out):  44
- True Positives  (correct: risk):     118

## Saved Artefacts
- `models/stockout_model.pkl` — trained classifier
- `models/stockout_scaler.pkl` — StandardScaler
- `models/stockout_model_meta.pkl` — feature list and metadata
- `data/processed/stockout_predictions.csv` — test-set predictions with probabilities
- `reports/stockout_feature_importance.csv` — feature importances
- `reports/stockout_confusion_matrix.png` — confusion matrix chart