"""
synthetic_generator.py
Generates sample Agmarknet-style mandi data for Onion and Tomato.
Creates 45 days of daily price and arrival records that look like
real data downloaded from https://agmarknet.gov.in

Price ranges:
  - Onion: Rs 3000 to Rs 4500 per Quintal
  - Tomato: Rs 2000 to Rs 2800 per Quintal
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
from typing import Dict

# Mandi configuration with realistic Agmarknet price ranges
MANDI_CONFIGS = {
    "Onion": {
        "Lasalgaon": {
            "state": "Maharashtra",
            "base_price": 3600.0,       # Rs per Quintal (middle of 3000-4500)
            "base_arrivals": 2500.0,    # Tonnes per day
            "daily_noise_std": 80.0,    # Small daily price fluctuation
            "arrival_noise_std": 200.0, # Daily arrival fluctuation
        },
        "Azadpur": {
            "state": "Delhi",
            "base_price": 3900.0,       # Slightly higher (transport cost to Delhi)
            "base_arrivals": 1400.0,
            "daily_noise_std": 90.0,
            "arrival_noise_std": 150.0,
        }
    },
    "Tomato": {
        "Kolar": {
            "state": "Karnataka",
            "base_price": 2350.0,       # Rs per Quintal (middle of 2000-2800)
            "base_arrivals": 1600.0,
            "daily_noise_std": 60.0,
            "arrival_noise_std": 180.0,
        },
        "Azadpur": {
            "state": "Delhi",
            "base_price": 2550.0,       # Higher in Delhi due to transport
            "base_arrivals": 1200.0,
            "daily_noise_std": 65.0,
            "arrival_noise_std": 130.0,
        }
    }
}

def generate_mandi_series(
    commodity: str,
    mandi: str,
    num_days: int = 45,
    start_date: str = "2026-09-15",
    random_state: int = 42,
    outlier_prob: float = 0.02
) -> pd.DataFrame:
    """
    Generates daily price and arrival data for one mandi.
    Data looks like a real Agmarknet download with columns:
    date, mandi, state, commodity, variety, arrivals_tonnes,
    min_price, max_price, modal_price

    Prices stay within realistic ranges:
      - Onion:  Rs 3000 - 4500 per quintal
      - Tomato: Rs 2000 - 2800 per quintal
    """
    np.random.seed(random_state)
    cfg = MANDI_CONFIGS[commodity][mandi]

    # Set price boundaries based on commodity
    if commodity == "Onion":
        price_min, price_max = 3000.0, 4500.0
    else:
        price_min, price_max = 2000.0, 2800.0

    date_range = pd.date_range(start=start_date, periods=num_days, freq="D")
    n = len(date_range)
    days_of_week = date_range.dayofweek.values

    # --- Generate Daily Modal Prices ---
    # Start from base price and do a random walk with small daily changes
    # This creates a gentle upward/downward trend like real market data
    daily_changes = np.random.normal(0, cfg["daily_noise_std"], size=n)

    # Add a small upward drift (gentle price increase over the month)
    drift = np.linspace(0, (price_max - price_min) * 0.08, n)

    modal_price = np.zeros(n)
    modal_price[0] = cfg["base_price"]
    for t in range(1, n):
        modal_price[t] = modal_price[t-1] + daily_changes[t] + drift[t] - drift[t-1]

    # Keep prices within the specified range
    modal_price = np.clip(modal_price, price_min, price_max)

    # --- Generate Min and Max Prices ---
    # In Agmarknet, min_price < modal_price < max_price
    spread = modal_price * np.random.uniform(0.05, 0.12, size=n)
    min_price = np.maximum(price_min * 0.9, modal_price - spread)
    max_price = modal_price + spread * np.random.uniform(0.8, 1.3, size=n)

    # --- Generate Daily Arrivals (Tonnes) ---
    arrival_noise = np.random.normal(0, cfg["arrival_noise_std"], size=n)
    arrivals = cfg["base_arrivals"] + arrival_noise

    # Sunday: mandi is closed, very low arrivals
    sunday_mask = (days_of_week == 6)
    arrivals[sunday_mask] = arrivals[sunday_mask] * 0.15

    arrivals = np.clip(arrivals, 10.0, None)

    # --- Inject a few data-entry errors (like real Agmarknet typos) ---
    raw_modal = modal_price.copy()
    is_outlier = np.zeros(n, dtype=bool)
    outlier_types = ["none"] * n

    num_outliers = max(1, int(n * outlier_prob))
    outlier_indices = np.random.choice(n, size=num_outliers, replace=False)

    for idx in outlier_indices:
        err_type = np.random.choice(["extra_zero", "dropped_zero"])
        is_outlier[idx] = True
        outlier_types[idx] = err_type

        if err_type == "extra_zero":
            # Clerk typed 35000 instead of 3500
            raw_modal[idx] = min(65000.0, raw_modal[idx] * 10.0)
        elif err_type == "dropped_zero":
            # Clerk typed 350 instead of 3500
            raw_modal[idx] = max(50.0, raw_modal[idx] * 0.1)

    # Build the dataframe (same columns as Agmarknet download)
    df = pd.DataFrame({
        "date": date_range.strftime("%Y-%m-%d"),
        "mandi": mandi,
        "state": cfg["state"],
        "commodity": commodity,
        "variety": "Local / Hybrid" if commodity == "Tomato" else "Red / Nasik",
        "arrivals_tonnes": np.round(arrivals, 2),
        "min_price": np.round(min_price, 2),
        "max_price": np.round(max_price, 2),
        "modal_price": np.round(raw_modal, 2),
        "true_modal_price": np.round(modal_price, 2),
        "is_outlier": is_outlier,
        "outlier_type": outlier_types
    })

    return df

def generate_benchmark_dataset(output_dir: str = "data/raw") -> Dict[str, str]:
    """
    Generates and saves sample Agmarknet dataset for Onion and Tomato.
    Creates CSV files that look like real agmarknet.gov.in downloads.
    """
    os.makedirs(output_dir, exist_ok=True)
    generated_files = {}

    for commodity, mandis in MANDI_CONFIGS.items():
        all_dfs = []
        for mandi in mandis.keys():
            df_mandi = generate_mandi_series(commodity=commodity, mandi=mandi)
            all_dfs.append(df_mandi)

        combined_df = pd.concat(all_dfs, ignore_index=True)
        fname = f"{commodity.lower()}_mandi_data.csv"
        fpath = os.path.join(output_dir, fname)
        combined_df.to_csv(fpath, index=False)
        generated_files[commodity] = fpath
        print(f"Generated {commodity} dataset: {len(combined_df)} records across {len(mandis)} mandis -> {fpath}")

    return generated_files

if __name__ == "__main__":
    generate_benchmark_dataset()
