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
