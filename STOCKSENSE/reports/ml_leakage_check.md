# Data Leakage Audit — StockSense Round 2

## Overview
At prediction time *t*, only information from days t and earlier is available.
This document confirms that every feature respects that constraint.

---

## Feature-by-Feature Audit

### A. Time Features — NO LEAKAGE
| Feature | Source | Risk |
|---|---|---|
| `day_of_week` | `date` (calendar math) | None |
| `weekend_flag` | `date` | None |
| `month` | `date` | None |
| `week_no` | `date` | None |
| `day_of_month` | `date` | None |
| `quarter` | `date` | None |
| `festival_flag` | `festival` col + fixed calendar | None — known in advance |
| `holiday` | existing column | None |
| `local_event` | existing column | None |
| `holiday_or_festival` | derived from above | None |

### B. Lag Features — NO LEAKAGE
All lag features use `.shift(n)` within each (store_id, product_id) group sorted by date.
shift(n) for n>=1 accesses only historical rows.

| Feature | Lag | Risk |
|---|---|---|
| `lag_1`  | units_sold at t-1 | None |
| `lag_7`  | units_sold at t-7 | None |
| `lag_14` | units_sold at t-14 | None |
| `lag_closing_1` | closing stock at t-1 | None |
| `lag_closing_7` | closing stock at t-7 | None |

### C. Rolling Features — NO LEAKAGE
`.shift(1).rolling(n)` ensures window covers t-1 to t-n only.

| Feature | Window | Risk |
|---|---|---|
| `rolling_mean_7`  | mean of t-1 to t-7  | None |
| `rolling_mean_14` | mean of t-1 to t-14 | None |
| `rolling_std_7`   | std  of t-1 to t-7  | None |

### D. Inventory Features — NO LEAKAGE
Derived from same-day closing stock (end-of-day observation).

| Feature | Risk |
|---|---|
| `days_of_inventory` | None |
| `inventory_to_demand_ratio` | None |
| `reorder_gap` | None |
| `stock_coverage_ratio` | None |
| `stockout_flag` (TARGET) | None — same-day observation |

### E. Price/Promotion Features — NO LEAKAGE
| Feature | Risk |
|---|---|
| `discount_pct` | None |
| `price_change` | None — .diff() uses t vs t-1 |
| `mrp_to_cost_ratio` | None — static attributes |
| `promotion_flag` | None — planned in advance |
| `promo_weekend` | None |

### F. Store/Product Metadata — NO LEAKAGE
Static attributes (store type, category, brand, MRP, shelf life, etc.).
All known at prediction time.

---

## Explicitly Excluded (Leakage Risk)
| Feature | Reason |
|---|---|
| `next_7_day_demand` | IS the Model 1 target — never a predictor |
| Raw future `units_sold` | Only lagged/shifted versions used |
| Future `closing`/`opening` | Only lagged versions used |
| `stockout_flag` as Model 1 feature | Target-leakage between models |
| `revenue` | Linear proxy for units_sold — excluded |
| `sold` column | Duplicate of `units_sold` |

---

## Target Definitions

### Model 1 Target: `next_7_day_demand`
- **Definition**: Sum of `units_sold` for days t+1 through t+7 per (store_id, product_id).
- **Method**: Reverse-order rolling sum of window=7, shifted by 1.
- **Incomplete rows**: Last 7 rows per group receive NaN and are **excluded** from training.
- **Usable rows**: 6048 of 6440 total.

### Model 2 Target: `stockout_flag`
- **Definition**: 1 if `closing < reorder_lvl`, else 0.
- **Rationale**: Below the reorder threshold = imminent stock-out risk.
- **Positive (risk)**: 931 (14.5%)
- **Negative (OK)**: 5509 (85.5%)
- **Imbalance**: ~1:5 (neg:pos)

---

## Temporal Split
| Split | Dates | ~% |
|---|---|---|
| Train | 2026-05-01 to 2026-07-25 | 70% |
| Validation | 2026-07-26 to 2026-08-12 | 15% |
| Test | 2026-08-13 to 2026-08-31 | 15% |

No random shuffling. Split by date threshold only.

---
## Conclusion
All features audited. **No data leakage detected.**
The pipeline is safe for time-series supervised learning.