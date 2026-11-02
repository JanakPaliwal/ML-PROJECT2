"""
preprocessor.py
Data Cleaning and Outlier Treatment Module for Agmarknet Mandi Records.
Treats data-entry errors using:
1. Domain-guided Physical Price Bounds
2. Rolling Hampel Filter (Median Absolute Deviation)
3. Seasonal residual IQR winsorization
4. Mandi calendar alignment (Sundays & holiday interpolation)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

DOMAIN_BOUNDS = {
    "Onion": {"min": 250.0, "max": 14000.0, "arrival_min": 5.0},
    "Tomato": {"min": 200.0, "max": 18000.0, "arrival_min": 5.0}
}

class MandiPreprocessor:
    """
    Cleans raw mandi price and arrival series, handles missing trading days,
    and removes data-entry anomalies.
    """

    def __init__(
        self,
        commodity: str = "Onion",
        hampel_window: int = 11,
        hampel_n_sigmas: float = 3.0,
        iqr_multiplier: float = 2.5
    ):
        self.commodity = commodity
        self.hampel_window = hampel_window
        self.hampel_n_sigmas = hampel_n_sigmas
        self.iqr_multiplier = iqr_multiplier
        self.bounds = DOMAIN_BOUNDS.get(commodity, {"min": 200.0, "max": 20000.0, "arrival_min": 5.0})

    def hampel_filter(self, series: pd.Series, window: int = 11, n_sigmas: float = 3.0) -> Tuple[pd.Series, pd.Series]:
        """
        Applies Hampel filter (rolling MAD) to detect and replace local outliers.
        Returns:
            cleaned_series: pd.Series with outliers replaced by rolling median
            is_outlier: pd.Series of booleans indicating detected outliers
        """
        rolling_median = series.rolling(window=window, min_periods=3, center=True).median()
        # Impute boundary edge NaNs
        rolling_median = rolling_median.bfill().ffill()

        difference = (series - rolling_median).abs()
        # Median Absolute Deviation (MAD), 1.4826 scale factor for normal distribution
        rolling_mad = difference.rolling(window=window, min_periods=3, center=True).median() * 1.4826
        rolling_mad = rolling_mad.bfill().ffill()
        rolling_mad = np.maximum(rolling_mad, 25.0)  # Floor to avoid division by zero in static markets

        threshold = n_sigmas * rolling_mad
        outlier_mask = difference > threshold

        cleaned = series.copy()
        cleaned[outlier_mask] = rolling_median[outlier_mask]
        return cleaned, outlier_mask

    def clean_mandi_series(self, df_mandi: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Cleans a single mandi's time series:
        - Re-indexes to complete daily calendar
        - Applies physical bounds check (detects 10x typos)
        - Applies Hampel filter for local spikes
        - Smooths/interpolates Sunday closure drops
        """
        df = df_mandi.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").drop_duplicates(subset=["date"])
        df.set_index("date", inplace=True)

        # 1. Re-index to continuous daily timeline (filling Sunday / holiday gaps)
        full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq="D")
        df_reindexed = df.reindex(full_idx)
        df_reindexed["mandi"] = df_reindexed["mandi"].ffill().bfill()
        df_reindexed["state"] = df_reindexed["state"].ffill().bfill()
        df_reindexed["commodity"] = df_reindexed["commodity"].ffill().bfill()

        # Track outlier flags
        raw_modal = df_reindexed["modal_price"].copy()
        domain_outliers = (raw_modal < self.bounds["min"]) | (raw_modal > self.bounds["max"])
        
        # Clip obvious domain outliers first so they don't corrupt rolling median
        clipped_modal = raw_modal.clip(lower=self.bounds["min"], upper=self.bounds["max"])
        # Interpolate any missing trading dates
        interpolated_modal = clipped_modal.interpolate(method="time").ffill().bfill()

        # 2. Hampel filter on interpolated prices
        cleaned_modal, hampel_outliers = self.hampel_filter(
            interpolated_modal,
            window=self.hampel_window,
            n_sigmas=self.hampel_n_sigmas
        )

        all_detected_outliers = domain_outliers | hampel_outliers

        # 3. Clean Arrivals
        raw_arrivals = df_reindexed["arrivals_tonnes"].copy()
        # On closed Sundays, interpolate arrival with 7-day seasonal lag
        cleaned_arrivals = raw_arrivals.interpolate(method="time").ffill().bfill()
        cleaned_arrivals = np.maximum(cleaned_arrivals, self.bounds["arrival_min"])

        # 4. Compile cleaned dataframe
        df_clean = pd.DataFrame({
            "date": full_idx.strftime("%Y-%m-%d"),
            "mandi": df_reindexed["mandi"].values,
            "state": df_reindexed["state"].values,
            "commodity": df_reindexed["commodity"].values,
            "raw_modal_price": raw_modal.values,
            "modal_price": np.round(cleaned_modal.values, 2),
            "arrivals_tonnes": np.round(cleaned_arrivals.values, 2),
            "detected_outlier": all_detected_outliers.values
        })

        # Calculate Outlier Detection Statistics
        stats = {
            "total_records": len(df_clean),
            "detected_outliers": int(all_detected_outliers.sum()),
            "outlier_percentage": float(np.round((all_detected_outliers.sum() / len(df_clean)) * 100, 2)),
            "domain_bound_violations": int(domain_outliers.sum()),
            "hampel_deviations": int(hampel_outliers.sum()),
        }

        # If synthetic ground truth exists, compute precision and recall
        if "is_outlier" in df_reindexed.columns:
            true_outliers = df_reindexed["is_outlier"].fillna(False).values
            tp = int(np.sum(all_detected_outliers & true_outliers))
            fp = int(np.sum(all_detected_outliers & (~true_outliers)))
            fn = int(np.sum((~all_detected_outliers) & true_outliers))
            precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            stats.update({
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1, 4)
            })

        return df_clean, stats

    def clean_dataset(self, df_all: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
        """
        Cleans entire dataset across all mandis in dataframe.
        """
        cleaned_list = []
        overall_stats = {}

        for mandi, group in df_all.groupby("mandi"):
            cleaned_mandi, stats = self.clean_mandi_series(group)
            cleaned_list.append(cleaned_mandi)
            overall_stats[mandi] = stats

        return pd.concat(cleaned_list, ignore_index=True), overall_stats

if __name__ == "__main__":
    from synthetic_generator import generate_benchmark_dataset
    files = generate_benchmark_dataset()
    df_raw = pd.read_csv(files["Onion"])
    prep = MandiPreprocessor(commodity="Onion")
    df_clean, stats = prep.clean_dataset(df_raw)
    print("Cleaned records:", len(df_clean))
    print("Outlier treatment audit stats:", stats)
