# Round 1 Summary: Data Preparation & EDA

## 1. Business Problem
NovaMart Retail Pvt. Ltd. needs a decision-support solution to forecast demand, predict stock-out risk, and recommend inventory actions. The goal is to reduce lost sales from empty shelves and minimize holding costs from overstocking.

## 2. Synthetic Dataset Design
To simulate the NovaMart retail environment, 5 datasets were synthetically generated for a 122-day period (May 1, 2026 to August 31, 2026). The generation logic incorporated base demand by product, store footfall multipliers, promotion lift, weekend surges, and external factors like festivals. Inventory flows were dynamically calculated daily (`closing = opening + received - sold`).

## 3. Dataset Sizes (Raw)
- **Transactions**: 138,492 rows
- **Products**: 14 rows
- **Stores**: 4 rows
- **Inventory**: 6,440 rows
- **External Factors**: 492 rows

## 4. Data-Quality Issues Intentionally Injected
- **Transactions**: Duplicate transaction IDs added (10 rows) and impossible negative quantities (-4) injected (5 rows).
- **Products**: Category inconsistency ("beverage" and "BEVERAGES" instead of standard "Beverages").
- **Inventory**: Arithmetic mismatch where closing stock did not match `opening + received - sold` (20 rows).
- **External Factors**: ~3% of `temp_c` values were blanked out to simulate sensor/API failure.
- **Sparse History**: Product 'P999' was introduced only in the last 10 days of the dataset, creating a cold-start problem.

## 5. Cleaning Actions
- **Missing Values**: Imputed `temp_c` with the city's monthly mean.
- **Duplicates**: Removed exact duplicate transaction rows.
- **Categories**: Standardized all product categories to Title Case and fixed spelling ("Beverage" -> "Beverages").
- **Invalid Quantities**: Filtered out transactions with quantity <= 0.
- **Inventory Mismatch**: Recalculated and enforced the correct closing stock equation.

## 6. Master Dataset Dimensions
- **File**: `master_daily.csv`
- **Rows**: 6,832
- **Columns**: 23 (date, store_id, product_id, units_sold, revenue, average_selling_price, average_discount_pct, promotion_flag, transaction_count, unique_customer_count, category, sub_category, brand, mrp, cost_price, shelf_life_days, supplier_id, city, store_type, floor_area_sqft, avg_daily_customers, region, temp_c, rain_mm, holiday, festival, weekend, local_event, opening, received, sold, closing, reorder_lvl, lead_days)
*(Note: some extra descriptive columns are merged)*

## 7. Master Grain Verification
The master dataset exactly matches the required grain: **ONE ROW = ONE DATE × ONE STORE × ONE PRODUCT**.
Automated validation confirms zero duplicate keys at this grain and no invalid (negative) sales/inventory after cleaning.

## 8. EDA Findings
1. **Revenue by Category**: Groceries and Personal Care drive the highest revenue.
2. **Store Trends**: The Hypermarket (S02) dominates volume.
3. **Promotions**: Clear lift in units sold on days with active promotions.
4. **Weekend Lift**: Average daily units sold are significantly higher on weekends.
5. **Volatility**: Specific items have high coefficient of variation, indicating erratic demand that needs better safety stock buffering.
6. **Stock-outs**: Some store-product combinations repeatedly experience stock-outs due to structural under-ordering.

## 9. Statistical Results
1. **Promotions**: Welch's T-Test confirms a statistically significant increase in sales during promotions (p < 0.05).
2. **Store Types**: ANOVA confirms mean demand significantly differs across Hypermarket, Supermarket, and Express formats.
3. **Stock-outs**: Chi-Square test confirms stock-out frequency is significantly associated with promotions, indicating safety stock isn't adequately adjusted during promotional periods.

## 10. Important Business Insights
- Promotions drive traffic but also drive stock-outs. The replenishment engine needs a "promotion awareness" feature.
- Demand behavior is highly sensitive to the store's baseline footprint (Hypermarket vs Express).

## 11. Limitations
- The dataset assumes a simplified 1-to-5 day lead time without catastrophic supplier failures.
- Only ~120 days of history limits the ability to model annual seasonality (e.g., winter vs summer).

## 12. Recommendations for Person 2
- **Feature Engineering**: Generate lag features (e.g., `lag_7_demand`, `rolling_mean_7`) to capture recent velocity. 
- **Time Features**: Ensure `day_of_week` and `weekend` are included as categorical features.
- **Sparse Products**: Use fallback strategies for `P999` (e.g., category average) since its history is too short for deep lags.
- **Data Target**: The target for Model 1 is next 7-day demand. Be extremely careful to avoid data leakage (don't use future prices or future stockouts to predict demand).

*The master dataset is fully prepared, clean, and ready for modeling.*
