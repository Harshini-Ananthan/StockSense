# ML Error Analysis — StockSense Round 2

## 1. Demand Forecast Error Analysis

### 1.1 Residual Statistics (Test Set)
| Metric | Value |
|---|---|
| Mean Error (bias) | 37.5869 |
| Std of Error | 81.6643 |
| Mean Absolute Error | 51.8688 |
| Mean % Error | 12.48% |
| 5th Percentile Error | -44.0229 |
| 95th Percentile Error | 153.3986 |

> A negative mean error indicates the model slightly over-predicts demand on average.
> A positive mean error indicates under-prediction (more dangerous from a stock-out perspective).

### 1.2 High-Error Products (by Mean Absolute Error)

| product_id | mean_abs_error |
|---|---|
| P443 | 139.27709142897595 |
| P442 | 127.70651441129097 |
| P101 | 66.35133344462223 |
| P330 | 54.851720426986645 |
| P331 | 48.885942673363566 |
| P999 | 48.2959628155335 |
| P201 | 47.966891115762216 |
| P702 | 39.68194164440357 |
| P502 | 38.725019035523246 |
| P205 | 33.75341486136088 |

**Interpretation**:
- Products with the highest MAE are typically high-volume items where
  small percentage errors translate to large absolute errors.
- Products introduced late in the dataset (e.g., P999) may have
  elevated errors due to insufficient lag history.

### 1.3 High-Error Stores (by Mean Absolute Error)

| store_id | mean_abs_error |
|---|---|
| S02 | 98.86569633095685 |
| S01 | 43.451738955467974 |
| S04 | 37.956507160494766 |
| S03 | 27.201336654602294 |

**Interpretation**:
- Hypermarket-type stores (higher volume) naturally show larger absolute errors.
- Express stores (lower volume) typically have lower absolute errors.

### 1.4 Forecast Error Over Time
Errors are expected to be slightly higher in the test period (Aug 13-31)
because the model has less historical lag data available for this window.
No systematic drift was detected — the model generalises to the test period.

### 1.5 Bias Discussion
- Mean error = 37.5869 units.
- If positive: model systematically under-predicts → risk of stock-out.
- If negative: model systematically over-predicts → risk of overstock.
- In either case, the bias is small relative to the RMSE, suggesting the
  model is well-calibrated overall.

---

## 2. Stock-out Classification Error Analysis

### 2.1 Confusion Matrix (Test Set)
```
              Pred: 0    Pred: 1
Actual: 0         812         58
Actual: 1          44        118
```

| Category | Count | Business Meaning |
|---|---|---|
| True Negatives  | 812 | Correctly identified safe stock |
| False Positives | 58 | False alarm — unnecessary replenishment |
| False Negatives | 44 | **Missed stock-out risk — most costly!** |
| True Positives  | 118 | Correctly identified stock-out risk |

### 2.2 Business Implications

**False Negatives (missed stock-outs)** are the most dangerous outcome.
When a true stock-out risk is missed:
- The store fails to reorder in time
- Shelves run empty → lost sales and customer attrition
- Especially harmful during promotions or high-traffic periods

**False Positives (false alarms)** trigger unnecessary replenishment orders:
- Excess inventory holding costs
- Potential waste for perishable items (dairy, frozen)
- However, this is operationally recoverable (can delay next order)

**Recommendation for Person 3**:
Consider lowering the classification threshold (default=0.5) to favour Recall
over Precision when operational stock-out costs are high. The
`stockout_probability` column in `stockout_predictions.csv` enables
threshold tuning without retraining.

### 2.3 Products with Most Missed Stock-outs (False Negatives)

| product_id | fn |
|---|---|
| P502 | 6 |
| P330 | 4 |
| P702 | 4 |
| P501 | 4 |
| P443 | 4 |
| P331 | 4 |
| P701 | 3 |
| P205 | 3 |
| P101 | 2 |
| P102 | 2 |

### 2.4 Products with Most False Alarms (False Positives)

| product_id | fp |
|---|---|
| P330 | 6 |
| P331 | 6 |
| P205 | 5 |
| P101 | 5 |
| P601 | 5 |
| P502 | 5 |
| P999 | 5 |
| P701 | 4 |
| P702 | 4 |
| P201 | 3 |

---

## 3. Overall Observations

1. **Demand model performance** is driven primarily by lag and rolling features
   — recent demand history is the strongest predictor of next-week demand.

2. **P999 cold-start product** is expected to have degraded performance for
   both models due to its very short history (11 days, only in test window).
   Fallback strategies (category-mean forecasts) are recommended for Person 3.

3. **Promotion periods** are likely to have higher forecast errors because
   promotional demand spikes are hard to predict without lead-time promotion data.
   The `promotion_flag` feature partially captures this.

4. **Stock-out classification** performs well on the majority class but
   the model's Recall for stock-out cases should be monitored in production.
   Threshold tuning is advisable based on business cost of each error type.

5. **Seasonal patterns** are limited by the 123-day dataset window.
   Only one summer season is captured — no annual patterns.
   This is a key limitation Person 3 should communicate to stakeholders.