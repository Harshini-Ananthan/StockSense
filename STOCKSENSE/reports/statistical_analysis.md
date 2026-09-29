# Statistical Analysis

## 1. Do promotions significantly increase sales?
- **Business Question**: Does running a promotion lead to a higher number of units sold on average?
- **H0 (Null Hypothesis)**: The mean units sold during promotions is equal to the mean units sold without promotions.
- **H1 (Alternative Hypothesis)**: The mean units sold during promotions is different (greater) than without promotions.
- **Test Used**: Welch's Independent T-Test
- **Test Statistic**: 8.7986
- **p-value**: 1.0751e-17
- **Significance Level**: 0.05
- **Decision**: Reject H0
- **Business Interpretation**: Promotions have a statistically significant positive impact on the average number of units sold. The company should strategically use promotions to drive volume.

## 2. Does mean demand differ across store types?
- **Business Question**: Are sales volumes fundamentally different depending on whether a store is a Hypermarket, Supermarket, or Express?
- **H0**: The mean units sold is the same across all store types.
- **H1**: At least one store type has a significantly different mean demand.
- **Test Used**: One-Way ANOVA
- **Test Statistic**: 1471.0606
- **p-value**: 0.0000e+00
- **Significance Level**: 0.05
- **Decision**: Reject H0
- **Business Interpretation**: Store type significantly affects average demand. Hypermarkets clearly drive different volumes than Express stores. Inventory and replenishment logic should be customized by store format.

## 3. Is stock-out frequency associated with promotion status?
- **Business Question**: Do stock-outs happen more often when a product is on promotion?
- **H0**: Stock-out frequency and promotion status are independent.
- **H1**: There is a significant association between promotion status and stock-out frequency.
- **Test Used**: Chi-Square Test of Independence
- **Test Statistic**: 0.0000
- **p-value**: 1.0000e+00
- **Significance Level**: 0.05
- **Decision**: Fail to reject H0
- **Business Interpretation**: Stock-outs are not significantly more likely during promotions. Replenishment logic seems capable of handling the current promotion lift.

