import pandas as pd
import numpy as np
import scipy.stats as stats
import os

def main():
    print("Performing Statistical Analysis...")
    base_dir = os.path.dirname(os.path.dirname(__file__))
    master_path = os.path.join(base_dir, 'data', 'processed', 'master_daily.csv')
    report_path = os.path.join(base_dir, 'reports', 'statistical_analysis.md')
    
    df = pd.read_csv(master_path)
    
    md_content = "# Statistical Analysis\n\n"
    
    # 1. Do promotions significantly increase sales?
    # Test: Independent T-test
    promo_sales = df[df['promotion_flag'] == 1]['units_sold']
    no_promo_sales = df[df['promotion_flag'] == 0]['units_sold']
    
    t_stat, p_val = stats.ttest_ind(promo_sales, no_promo_sales, equal_var=False)
    
    md_content += "## 1. Do promotions significantly increase sales?\n"
    md_content += "- **Business Question**: Does running a promotion lead to a higher number of units sold on average?\n"
    md_content += "- **H0 (Null Hypothesis)**: The mean units sold during promotions is equal to the mean units sold without promotions.\n"
    md_content += "- **H1 (Alternative Hypothesis)**: The mean units sold during promotions is different (greater) than without promotions.\n"
    md_content += "- **Test Used**: Welch's Independent T-Test\n"
    md_content += f"- **Test Statistic**: {t_stat:.4f}\n"
    md_content += f"- **p-value**: {p_val:.4e}\n"
    md_content += "- **Significance Level**: 0.05\n"
    if p_val < 0.05:
        md_content += "- **Decision**: Reject H0\n"
        md_content += "- **Business Interpretation**: Promotions have a statistically significant positive impact on the average number of units sold. The company should strategically use promotions to drive volume.\n\n"
    else:
        md_content += "- **Decision**: Fail to reject H0\n"
        md_content += "- **Business Interpretation**: Promotions do not show a statistically significant effect on the average units sold. The current promotional strategy needs to be evaluated for effectiveness.\n\n"
        
    # 2. Does mean demand differ across store types?
    # Test: One-way ANOVA
    store_types = df['store_type'].unique()
    store_groups = [df[df['store_type'] == st]['units_sold'] for st in store_types]
    
    f_stat, p_val_anova = stats.f_oneway(*store_groups)
    
    md_content += "## 2. Does mean demand differ across store types?\n"
    md_content += "- **Business Question**: Are sales volumes fundamentally different depending on whether a store is a Hypermarket, Supermarket, or Express?\n"
    md_content += "- **H0**: The mean units sold is the same across all store types.\n"
    md_content += "- **H1**: At least one store type has a significantly different mean demand.\n"
    md_content += "- **Test Used**: One-Way ANOVA\n"
    md_content += f"- **Test Statistic**: {f_stat:.4f}\n"
    md_content += f"- **p-value**: {p_val_anova:.4e}\n"
    md_content += "- **Significance Level**: 0.05\n"
    if p_val_anova < 0.05:
        md_content += "- **Decision**: Reject H0\n"
        md_content += "- **Business Interpretation**: Store type significantly affects average demand. Hypermarkets clearly drive different volumes than Express stores. Inventory and replenishment logic should be customized by store format.\n\n"
    else:
        md_content += "- **Decision**: Fail to reject H0\n"
        md_content += "- **Business Interpretation**: Store type does not significantly explain the variation in average demand.\n\n"
        
    # 3. Is stock-out frequency associated with promotion status?
    # Test: Chi-Square Test of Independence
    df['stockout_flag'] = (df['closing'] == 0).astype(int)
    contingency_table = pd.crosstab(df['promotion_flag'], df['stockout_flag'])
    chi2_stat, p_val_chi2, dof, expected = stats.chi2_contingency(contingency_table)
    
    md_content += "## 3. Is stock-out frequency associated with promotion status?\n"
    md_content += "- **Business Question**: Do stock-outs happen more often when a product is on promotion?\n"
    md_content += "- **H0**: Stock-out frequency and promotion status are independent.\n"
    md_content += "- **H1**: There is a significant association between promotion status and stock-out frequency.\n"
    md_content += "- **Test Used**: Chi-Square Test of Independence\n"
    md_content += f"- **Test Statistic**: {chi2_stat:.4f}\n"
    md_content += f"- **p-value**: {p_val_chi2:.4e}\n"
    md_content += "- **Significance Level**: 0.05\n"
    if p_val_chi2 < 0.05:
        md_content += "- **Decision**: Reject H0\n"
        md_content += "- **Business Interpretation**: There is a statistically significant association between running promotions and experiencing stock-outs. The company must improve safety stock logic specifically for promoted items.\n\n"
    else:
        md_content += "- **Decision**: Fail to reject H0\n"
        md_content += "- **Business Interpretation**: Stock-outs are not significantly more likely during promotions. Replenishment logic seems capable of handling the current promotion lift.\n\n"
        
    with open(report_path, 'w') as f:
        f.write(md_content)
        
    print("Statistical Analysis Complete.")

if __name__ == "__main__":
    main()
