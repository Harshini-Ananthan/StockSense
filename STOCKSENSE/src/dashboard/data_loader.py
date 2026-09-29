"""
Data Loader for STOCKSENSE Dashboard
Loads cleaned datasets and checks for future ML predictions file safely.
"""

import os
from typing import Tuple, Optional, Dict
import pandas as pd
import streamlit as st


def get_data_dir() -> str:
    """Returns the absolute path to the data/processed directory."""
    # Try relative to this file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # src/dashboard -> src -> project_root -> data/processed
    candidate_1 = os.path.abspath(os.path.join(current_dir, "..", "..", "data", "processed"))
    if os.path.exists(candidate_1):
        return candidate_1

    # Try working directory
    candidate_2 = os.path.abspath(os.path.join("data", "processed"))
    if os.path.exists(candidate_2):
        return candidate_2

    # Fallback to StockSense/STOCKSENSE/data/processed
    candidate_3 = os.path.abspath(os.path.join("STOCKSENSE", "data", "processed"))
    if os.path.exists(candidate_3):
        return candidate_3

    return candidate_1


def _read_csv_with_dates(path: str) -> pd.DataFrame:
    """Read a project CSV without failing the full dashboard on bad dates."""
    frame = pd.read_csv(path)
    for column in ("date", "forecast_date", "prediction_date"):
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    return frame


def _normalize_prediction_columns(frame: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "store_id": ("store_id", "store", "store_code"),
        "product_id": ("product_id", "product", "sku", "sku_id"),
        "date": ("date", "forecast_date", "prediction_date", "as_of_date"),
        "current_inventory": ("current_inventory", "current_stock", "inventory", "closing"),
        "predicted_7d_demand": (
            "predicted_7d_demand", "predicted_next_7_day_demand",
            "predicted_demand_7d", "forecast_7d_demand", "demand_7d", "next_7d_demand",
        ),
        "stockout_probability": (
            "stockout_probability", "stock_out_probability", "stockout_prob",
        ),
        "risk_level": ("risk_level", "risk_category", "risk"),
    }
    normalized_names = {str(column).strip().lower(): column for column in frame.columns}
    rename_map = {}
    for canonical, candidates in aliases.items():
        if canonical in frame.columns:
            continue
        for candidate in candidates:
            original = normalized_names.get(candidate)
            if original is not None:
                rename_map[original] = canonical
                break
    frame = frame.rename(columns=rename_map)
    for column in ("store_id", "product_id"):
        if column in frame.columns:
            frame[column] = frame[column].astype("string").str.strip()
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for column in ("predicted_7d_demand", "stockout_probability", "current_inventory"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


@st.cache_data(show_spinner=False)
def load_datasets() -> Dict[str, pd.DataFrame]:
    """
    Loads all primary processed datasets required for the dashboard.
    Returns a dictionary of DataFrames:
        - 'master': master_daily.csv
        - 'products': cleaned_products.csv
        - 'stores': cleaned_stores.csv
        - 'inventory': cleaned_inventory.csv
        - 'external': cleaned_external_factors.csv
    """
    proc_dir = get_data_dir()

    master_path = os.path.join(proc_dir, "master_daily.csv")
    products_path = os.path.join(proc_dir, "cleaned_products.csv")
    stores_path = os.path.join(proc_dir, "cleaned_stores.csv")
    inventory_path = os.path.join(proc_dir, "cleaned_inventory.csv")
    external_path = os.path.join(proc_dir, "cleaned_external_factors.csv")

    datasets: Dict[str, pd.DataFrame] = {}

    if os.path.exists(master_path):
        df_master = _read_csv_with_dates(master_path)
        datasets["master"] = df_master
    else:
        datasets["master"] = pd.DataFrame()

    if os.path.exists(products_path):
        datasets["products"] = _read_csv_with_dates(products_path)
    else:
        datasets["products"] = pd.DataFrame()

    if os.path.exists(stores_path):
        datasets["stores"] = _read_csv_with_dates(stores_path)
    else:
        datasets["stores"] = pd.DataFrame()

    if os.path.exists(inventory_path):
        datasets["inventory"] = _read_csv_with_dates(inventory_path)
    else:
        datasets["inventory"] = pd.DataFrame()

    if os.path.exists(external_path):
        datasets["external"] = _read_csv_with_dates(external_path)
    else:
        datasets["external"] = pd.DataFrame()

    return datasets


def load_predictions(predictions_file_path: Optional[str] = None) -> Tuple[Optional[pd.DataFrame], bool, str]:
    """
    Checks for the ML Engineer's predictions file (e.g. data/processed/predictions.csv).

    Returns:
        tuple: (df_predictions, is_available, message)
    """
    proc_dir = get_data_dir()
    configured_path = predictions_file_path or os.environ.get("STOCKSENSE_PREDICTIONS_FILE")
    demand_path = os.path.join(proc_dir, "demand_predictions.csv")
    stockout_path = os.path.join(proc_dir, "stockout_predictions.csv")

    try:
        source_paths = [configured_path] if configured_path else []
        if not source_paths:
            combined_path = os.path.join(proc_dir, "predictions.csv")
            if os.path.exists(combined_path):
                source_paths = [combined_path]
            else:
                source_paths = [path for path in (demand_path, stockout_path) if os.path.exists(path)]
        if not source_paths:
            expected_paths = f"'{demand_path}' and/or '{stockout_path}'"
            return None, False, f"Forecast files not found: {expected_paths}."

        prediction_frames = [
            _normalize_prediction_columns(_read_csv_with_dates(path))
            for path in source_paths
        ]
        df_pred = prediction_frames[0]
        merge_keys = [column for column in ("date", "store_id", "product_id")
                      if column in df_pred.columns and all(column in frame.columns for frame in prediction_frames)]
        for frame in prediction_frames[1:]:
            if not merge_keys:
                continue
            overlap = [column for column in frame.columns if column in df_pred.columns and column not in merge_keys]
            df_pred = df_pred.merge(
                frame.drop(columns=overlap),
                on=merge_keys,
                how="outer",
                validate="one_to_one",
            )
        df_pred = _normalize_prediction_columns(df_pred)

        missing_keys = [column for column in ("store_id", "product_id") if column not in df_pred.columns]
        has_decision_output = any(
            column in df_pred.columns
            for column in ("predicted_7d_demand", "stockout_probability", "risk_level")
        )
        if missing_keys or not has_decision_output:
            missing = missing_keys + ([] if has_decision_output else ["forecast demand, probability, or risk"])
            return (
                None,
                False,
                f"Prediction file found but missing usable columns: {', '.join(missing)}",
            )

        for column in ("store_id", "product_id"):
            df_pred[column] = df_pred[column].astype("string").str.strip()
        if "date" in df_pred.columns:
            df_pred["date"] = pd.to_datetime(df_pred["date"], errors="coerce")
        for column in ("predicted_7d_demand", "stockout_probability", "current_inventory"):
            if column in df_pred.columns:
                df_pred[column] = pd.to_numeric(df_pred[column], errors="coerce")

        if "stockout_probability" in df_pred.columns:
            probability = df_pred["stockout_probability"]
            if probability.max(skipna=True) > 1.0:
                probability = probability / 100.0
            df_pred["stockout_probability"] = probability.where(probability.between(0, 1))

        if "risk_level" in df_pred.columns:
            risk_map = {
                "HIGH": "HIGH", "HIGH RISK": "HIGH", "ACT_NOW": "HIGH",
                "MEDIUM": "MEDIUM", "MEDIUM RISK": "MEDIUM", "WATCH": "MEDIUM",
                "LOW": "LOW", "LOW RISK": "LOW", "SAFE": "LOW",
            }
            df_pred["risk_level"] = (
                df_pred["risk_level"].astype("string").str.strip().str.upper().map(risk_map)
            )

        usable_columns = [
            column for column in ("predicted_7d_demand", "stockout_probability", "risk_level")
            if column in df_pred.columns
        ]
        if not df_pred[usable_columns].notna().any(axis=None):
            return None, False, "Prediction file contains no usable forecast or risk values."

        dedupe_columns = ["store_id", "product_id"]
        if "date" in df_pred.columns and df_pred["date"].notna().any():
            dedupe_columns.append("date")
        df_pred = df_pred.dropna(subset=["store_id", "product_id"])
        df_pred = df_pred.drop_duplicates(subset=dedupe_columns, keep="last")

        paths_label = ", ".join(os.path.basename(path) for path in source_paths)
        return df_pred, True, f"Forecast output loaded from {paths_label}."

    except Exception as e:
        return (
            None,
            False,
            f"Error loading predictions: {str(e)}"
        )


def merge_predictions(
    inventory: pd.DataFrame,
    predictions: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Join one forecast per store/product, preferring the inventory's exact date."""
    result = inventory.copy()
    output_columns = (
        "predicted_7d_demand", "stockout_probability", "risk_level", "current_inventory"
    )
    for column in output_columns:
        if column not in result.columns:
            result[column] = pd.NA
    if result.empty or predictions is None or predictions.empty:
        return result

    keys = ["store_id", "product_id"]
    if any(column not in result.columns or column not in predictions.columns for column in keys):
        return result

    left = result.copy()
    right = predictions.copy()
    for frame in (left, right):
        for column in keys:
            frame[column] = frame[column].astype("string").str.strip()

    value_columns = [column for column in output_columns if column in right.columns]
    has_forecast_dates = "date" in right.columns and right["date"].notna().any()
    if has_forecast_dates and "date" in left.columns:
        left["date"] = pd.to_datetime(left["date"], errors="coerce")
        right["date"] = pd.to_datetime(right["date"], errors="coerce")
        join_keys = keys + ["date"]
        right = right.dropna(subset=["date"])
    else:
        join_keys = keys
        if has_forecast_dates:
            right = right.sort_values("date", na_position="first")
        right = right.drop_duplicates(join_keys, keep="last")

    forecast = right[join_keys + value_columns].drop_duplicates(join_keys, keep="last")
    merged = left.merge(
        forecast,
        on=join_keys,
        how="left",
        suffixes=("", "_forecast"),
        validate="many_to_one",
    )
    for column in value_columns:
        forecast_column = f"{column}_forecast"
        if forecast_column in merged.columns:
            merged[column] = merged[forecast_column].combine_first(merged[column])
            merged = merged.drop(columns=forecast_column)
    return merged


def get_latest_inventory_snapshot(
    df_master: pd.DataFrame,
    as_of_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Extracts the latest available inventory snapshot for each store-product pair
    from the master dataset.
    """
    if df_master.empty:
        return pd.DataFrame()

    if "date" not in df_master.columns:
        return pd.DataFrame()
    dates = pd.to_datetime(df_master["date"], errors="coerce")
    valid_dates = dates.dropna()
    if valid_dates.empty:
        return pd.DataFrame()
    chosen_date = pd.Timestamp(as_of_date) if as_of_date is not None else valid_dates.max()
    snapshot = df_master.loc[dates == chosen_date].copy()
    return snapshot.drop_duplicates(subset=[c for c in ("store_id", "product_id") if c in snapshot], keep="last")
