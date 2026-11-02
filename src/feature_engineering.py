"""
feature_engineering.py
Creates input features for the ML models from the cleaned mandi price data.

Features built:
  - Price lag features (yesterday's price, 2 days ago, 3 days ago, 7 days ago)
  - Arrival lag features (yesterday's arrivals, 3 days ago)
  - Rolling averages (7-day moving average of price and arrivals)
  - Price rate of change (how much price changed in last 7 days)
  - Day of week (encoded as sin/cos so Monday is close to Tuesday)

Target variable: Price 7 days from now (next week's price)
"""

import numpy as np
import pandas as pd
from typing import List

PRICE_LAGS = [1, 2, 3, 7]
ARRIVAL_LAGS = [1, 3]
HORIZON = 7  # Predict price 7 days ahead

def construct_features_for_mandi(
    df_mandi: pd.DataFrame,
    horizon: int = HORIZON
) -> pd.DataFrame:
    """
    Builds features from a single mandi's daily time series.
    These features help the ML model learn price patterns.
    """
    df = df_mandi.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Target: What will the price be 7 days from now?
    df["target_price"] = df["modal_price"].shift(-horizon)
    df["target_price_change_pct"] = (df["target_price"] - df["modal_price"]) / df["modal_price"]

    # Alert labels: flag if price jumps or drops by more than 15% in a week
    df["target_spike_alert"] = (df["target_price_change_pct"] >= 0.15).astype(int)
    df["target_crash_alert"] = (df["target_price_change_pct"] <= -0.15).astype(int)

    # Today's price
    df["price_t"] = df["modal_price"]

    # Past price features (price 1 day ago, 2 days ago, etc.)
    for lag in PRICE_LAGS:
        df[f"price_lag_{lag}"] = df["modal_price"].shift(lag)

    # Today's arrivals
    df["arrivals_t"] = df["arrivals_tonnes"]

    # Past arrival features
    for lag in ARRIVAL_LAGS:
        df[f"arrivals_lag_{lag}"] = df["arrivals_tonnes"].shift(lag)

    # 7-day moving average of price (smooths out daily noise)
    df["price_rolling_mean_7"] = df["modal_price"].rolling(window=7, min_periods=3).mean()

    # 7-day moving average of arrivals
    df["arrivals_rolling_mean_7"] = df["arrivals_tonnes"].rolling(window=7, min_periods=3).mean()

    # How much has the price changed in the last 7 days? (rate of change)
    df["price_pct_change_7d"] = (df["modal_price"] - df["price_lag_7"]) / (df["price_lag_7"] + 1e-5)

    # Day of week encoded as sin/cos (so the model knows weekday patterns)
    day_of_week = df["date"].dt.dayofweek
    df["sin_dow"] = np.sin(2 * np.pi * day_of_week / 7.0)
    df["cos_dow"] = np.cos(2 * np.pi * day_of_week / 7.0)

    # Drop rows where we don't have enough past data (first 7 days)
    # and rows where we don't have the target (last 7 days)
    df_clean = df.dropna().reset_index(drop=True)
    return df_clean

def get_feature_columns() -> List[str]:
    """
    Returns the list of feature names used as input to ML models.
    """
    feature_cols = [
        "price_t",                      # Today's price
        *[f"price_lag_{l}" for l in PRICE_LAGS],   # Past prices
        "arrivals_t",                   # Today's arrivals
        *[f"arrivals_lag_{l}" for l in ARRIVAL_LAGS], # Past arrivals
        "price_rolling_mean_7",         # 7-day average price
        "arrivals_rolling_mean_7",      # 7-day average arrivals
        "price_pct_change_7d",          # Weekly price change rate
        "sin_dow",                      # Day of week (sin)
        "cos_dow",                      # Day of week (cos)
    ]
    return feature_cols

def build_features_dataset(df_cleaned: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    """
    Builds features for all mandis in the dataset.
    """
    results = []
    for mandi, group in df_cleaned.groupby("mandi"):
        mandi_feats = construct_features_for_mandi(group, horizon=horizon)
        results.append(mandi_feats)
    return pd.concat(results, ignore_index=True)

if __name__ == "__main__":
    from synthetic_generator import generate_benchmark_dataset
    from preprocessor import MandiPreprocessor
    files = generate_benchmark_dataset()
    df_raw = pd.read_csv(files["Onion"])
    prep = MandiPreprocessor(commodity="Onion")
    df_clean, _ = prep.clean_dataset(df_raw)
    df_feats = build_features_dataset(df_clean)
    print("Feature dataset shape:", df_feats.shape)
    print("Features used:", get_feature_columns())
