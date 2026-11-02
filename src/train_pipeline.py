"""
train_pipeline.py
Main pipeline that runs the entire mandi price forecasting system.

Steps:
1. Generate sample mandi data (like Agmarknet download)
2. Clean the data (remove typing errors, fill missing days)
3. Create features (past prices, moving averages, etc.)
4. Train ML models (Linear Regression, Ridge, k-NN, MLP Neural Network)
5. Test how accurate each model is on unseen data
6. Generate 7-day price forecast
7. Save results for the web dashboard
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any

import sys
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from synthetic_generator import generate_benchmark_dataset, MANDI_CONFIGS
from preprocessor import MandiPreprocessor
from feature_engineering import build_features_dataset, get_feature_columns
from models import get_model_registry, save_trained_model
from evaluator import compute_regression_metrics, evaluate_price_alerts, walk_forward_cv

OUTPUT_DIR = "data/processed"
MODELS_DIR = "models"
WEB_DATA_PATH = "web/forecast_data.json"

def run_pipeline() -> Dict[str, Any]:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(WEB_DATA_PATH), exist_ok=True)

    print("=" * 70)
    print("STEP 1: LOADING MANDI PRICE DATA")
    print("=" * 70)
    raw_files = generate_benchmark_dataset()

    commodities = ["Onion", "Tomato"]
    dashboard_payload = {
        "metadata": {
            "title": "Mandi Price Forecasting for Onion & Tomato",
            "sdgs": ["SDG 2: Zero Hunger", "SDG 12: Responsible Consumption and Production"],
            "commodities": commodities,
            "horizon_days": 7,
            "generated_at": pd.Timestamp.now().isoformat()
        },
        "commodities_data": {},
        "benchmark_summary": {}
    }

    feature_cols = get_feature_columns()

    for commodity in commodities:
        print(f"\n" + "=" * 50)
        print(f"PROCESSING: {commodity.upper()}")
        print("=" * 50)

        raw_csv_path = raw_files[commodity]
        df_raw = pd.read_csv(raw_csv_path)

        # Step 2: Clean the data (remove outliers/typos)
        preprocessor = MandiPreprocessor(commodity=commodity)
        df_clean, outlier_stats = preprocessor.clean_dataset(df_raw)

        cleaned_csv_path = os.path.join(OUTPUT_DIR, f"{commodity.lower()}_cleaned.csv")
        df_clean.to_csv(cleaned_csv_path, index=False)
        print(f"Data cleaned. Saved to {cleaned_csv_path}")

        # Step 3: Create features for ML models
        df_feats = build_features_dataset(df_clean, horizon=7)
        feats_csv_path = os.path.join(OUTPUT_DIR, f"{commodity.lower()}_features.csv")
        df_feats.to_csv(feats_csv_path, index=False)
        print(f"Features created. Shape: {df_feats.shape}")

        commodity_dict = {
            "mandis": list(df_clean["mandi"].unique()),
            "outlier_audit": outlier_stats,
            "mandi_details": {}
        }

        # Step 4 & 5: Train and test models for each mandi
        for mandi in commodity_dict["mandis"]:
            print(f"\n--- Mandi: {mandi} ({commodity}) ---")
            df_m = df_feats[df_feats["mandi"] == mandi].copy().reset_index(drop=True)

            if len(df_m) < 10:
                print(f"  Not enough data for {mandi} (only {len(df_m)} rows). Skipping.")
                continue

            X = df_m[feature_cols].values
            y = df_m["target_price"].values
            p_current = df_m["price_t"].values
            dates = df_m["date"].values

            # Split data: first 70% for training, last 30% for testing
            test_size = max(5, int(len(df_m) * 0.3))
            train_idx = np.arange(len(df_m) - test_size)
            test_idx = np.arange(len(df_m) - test_size, len(df_m))

            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]
            p_curr_test = p_current[test_idx]
            test_dates = dates[test_idx]

            mandi_models = get_model_registry()
            mandi_results = {}
            predictions_store = {}

            best_model_name = None
            best_rmse = float("inf")

            for model_name, model in mandi_models.items():
                # Cross-validation on training data
                cv_splits = min(3, max(2, len(X_train) // 5))
                cv_metrics, _ = walk_forward_cv(
                    model, X_train, y_train, p_current[train_idx], n_splits=cv_splits
                )

                # Train on all training data, then test on unseen test data
                model.fit(X_train, y_train)
                y_pred_test = model.predict(X_test)

                # Calculate accuracy metrics
                holdout_metrics = compute_regression_metrics(y_test, y_pred_test)
                alert_metrics = evaluate_price_alerts(
                    p_curr_test, y_test, y_pred_test, spike_threshold=0.15
                )

                mandi_results[model_name] = {
                    "cv_metrics": cv_metrics,
                    "holdout_metrics": holdout_metrics,
                    "volatility_early_warning": alert_metrics
                }

                predictions_store[model_name] = np.round(y_pred_test, 2).tolist()

                # Track best model (excluding baseline)
                if holdout_metrics["RMSE"] < best_rmse and model_name != "Naive Baseline":
                    best_rmse = holdout_metrics["RMSE"]
                    best_model_name = model_name

                # Save trained model
                model_save_path = os.path.join(
                    MODELS_DIR, f"{commodity.lower()}_{mandi.lower()}_{model_name.replace(' ', '_').lower()}.joblib"
                )
                save_trained_model(model, model_save_path)

            print(f"Best ML Model for {mandi}: {best_model_name} (Test RMSE: Rs {best_rmse})")

            # Step 6: Predict next 7 days using latest data
            latest_vector = X[-1:].copy()
            latest_price = float(p_current[-1])
            live_forecasts = {}

            for m_name, model in mandi_models.items():
                pred_val = float(model.predict(latest_vector)[0])
                pct_change = (pred_val - latest_price) / latest_price * 100.0
                live_forecasts[m_name] = {
                    "predicted_price_t_plus_7": round(pred_val, 2),
                    "expected_change_pct": round(pct_change, 2)
                }

            # Set alert level based on predicted price change
            best_live_change = live_forecasts[best_model_name]["expected_change_pct"]
            if best_live_change >= 15.0:
                alert_level = "PRICE_RISE_WARNING"
                alert_color = "#ef4444"
                recommendation = "Prices may rise significantly. Consider buying in advance or releasing stored stock."
            elif best_live_change >= 5.0:
                alert_level = "SLIGHT_INCREASE_EXPECTED"
                alert_color = "#f59e0b"
                recommendation = "A small price increase is expected. Normal market fluctuation."
            elif best_live_change <= -15.0:
                alert_level = "PRICE_DROP_WARNING"
                alert_color = "#dc2626"
                recommendation = "Prices may drop sharply. Farmers should consider holding stock or finding alternative buyers."
            elif best_live_change <= -5.0:
                alert_level = "SLIGHT_DECREASE_EXPECTED"
                alert_color = "#f59e0b"
                recommendation = "A small price decrease is expected. Good time for consumers to buy."
            else:
                alert_level = "STABLE"
                alert_color = "#10b981"
                recommendation = "Prices are expected to remain stable. Normal market conditions."

            # Save chart data for the web dashboard
            commodity_dict["mandi_details"][mandi] = {
                "latest_date": str(dates[-1]),
                "latest_price": latest_price,
                "latest_arrival": float(df_m["arrivals_t"].values[-1]),
                "best_model": best_model_name,
                "alert": {
                    "level": alert_level,
                    "color": alert_color,
                    "expected_change_pct": round(best_live_change, 2),
                    "predicted_price": round(live_forecasts[best_model_name]["predicted_price_t_plus_7"], 2),
                    "recommendation": recommendation
                },
                "live_forecasts": live_forecasts,
                "model_benchmarks": mandi_results,
                "chart_data": {
                    "dates": [str(d)[:10] for d in test_dates],
                    "actual_prices": [round(float(v), 2) for v in y_test],
                    "current_prices": [round(float(v), 2) for v in p_curr_test],
                    "arrivals": [round(float(v), 2) for v in df_m["arrivals_t"].values[test_idx]],
                    "predictions": {
                        m_name: preds for m_name, preds in predictions_store.items()
                    }
                }
            }

        dashboard_payload["commodities_data"][commodity] = commodity_dict

    # Build summary table for all models across all mandis
    summary_table = []
    for comm in commodities:
        for mandi, m_info in dashboard_payload["commodities_data"][comm]["mandi_details"].items():
            for m_name, m_res in m_info["model_benchmarks"].items():
                summary_table.append({
                    "Commodity": comm,
                    "Mandi": mandi,
                    "Model": m_name,
                    "Test_RMSE": float(m_res["holdout_metrics"]["RMSE"]),
                    "Test_MAE": float(m_res["holdout_metrics"]["MAE"]),
                    "Test_MAPE": float(m_res["holdout_metrics"]["MAPE"]),
                    "Test_R2": float(m_res["holdout_metrics"]["R2"]),
                    "CV_RMSE_Avg": float(m_res["cv_metrics"]["RMSE_mean"]),
                    "CV_MAPE_Avg": float(m_res["cv_metrics"]["MAPE_mean"]),
                })

    dashboard_payload["benchmark_summary"] = summary_table

    # Save results
    with open(WEB_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(dashboard_payload, f, indent=2, default=str)

    summary_df = pd.DataFrame(summary_table)
    summary_csv_path = os.path.join(OUTPUT_DIR, "model_comparison.csv")
    summary_df.to_csv(summary_csv_path, index=False)

    print("\n" + "=" * 70)
    print("ALL DONE!")
    print(f"Model comparison saved to: {summary_csv_path}")
    print(f"Dashboard data saved to: {WEB_DATA_PATH}")
    print("=" * 70)

    return dashboard_payload

if __name__ == "__main__":
    run_pipeline()
