"""
evaluator.py
Measures how good our ML models are at predicting mandi prices.

Metrics used (from DSC-17 Unit 3 - Regression Evaluation Metrics):
  - RMSE (Root Mean Squared Error): Average prediction error in Rupees
  - MAE (Mean Absolute Error): Average absolute difference between predicted and actual
  - MAPE (Mean Absolute Percentage Error): Error as a percentage
  - R2 Score: How much of the price variation is explained by the model (1.0 = perfect)
"""

import numpy as np
from typing import Dict, Any, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculates how accurate the model's predictions are.

    Returns:
      RMSE: Root Mean Squared Error (lower = better)
      MAE:  Mean Absolute Error (lower = better)
      MAPE: Mean Absolute Percentage Error (lower = better)
      R2:   R-squared score (higher = better, max 1.0)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # Avoid division by zero in MAPE
    mask = y_true > 1e-3
    y_t_safe = y_true[mask]
    y_p_safe = y_pred[mask]

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    mape = float(np.mean(np.abs((y_t_safe - y_p_safe) / y_t_safe)) * 100.0)
    r2 = float(r2_score(y_true, y_pred))

    return {
        "RMSE": round(rmse, 2),
        "MAE": round(mae, 2),
        "MAPE": round(mape, 2),
        "R2": round(r2, 4)
    }

def evaluate_price_alerts(
    p_current: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    spike_threshold: float = 0.15
) -> Dict[str, Any]:
    """
    Checks if the model correctly predicts price increases (spikes).

    A 'spike' = price goes up by 15% or more in 7 days.
    We check:
      - Precision: When model says 'spike', how often is it right?
      - Recall: Out of all real spikes, how many did the model catch?
      - F1 Score: Combined score of precision and recall
    """
    p_curr = np.asarray(p_current, dtype=float)
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)

    actual_pct_change = (y_t - p_curr) / (p_curr + 1e-5)
    pred_pct_change = (y_p - p_curr) / (p_curr + 1e-5)

    true_spike = actual_pct_change >= spike_threshold
    pred_spike = pred_pct_change >= spike_threshold

    tp = int(np.sum(pred_spike & true_spike))       # True Positives
    fp = int(np.sum(pred_spike & (~true_spike)))     # False Positives
    fn = int(np.sum((~pred_spike) & true_spike))     # False Negatives
    tn = int(np.sum((~pred_spike) & (~true_spike)))  # True Negatives

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "spike_events_actual": int(np.sum(true_spike)),
        "spike_alerts_triggered": int(np.sum(pred_spike)),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
    }

def walk_forward_cv(
    model,
    X: np.ndarray,
    y: np.ndarray,
    p_current: np.ndarray,
    n_splits: int = 3
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Time-series cross-validation (walk-forward testing).

    Unlike regular cross-validation, this respects time order:
    - Train on past data, test on future data
    - Slides forward in time for each fold
    - This prevents 'data leakage' (using future data to predict the past)
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rmse_scores, mae_scores, mape_scores, r2_scores = [], [], [], []

    all_y_true, all_y_pred, all_p_curr = [], [], []

    for train_idx, test_idx in tscv.split(X):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te, y_te = X[test_idx], y[test_idx]

        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)

        metrics = compute_regression_metrics(y_te, preds)
        rmse_scores.append(metrics["RMSE"])
        mae_scores.append(metrics["MAE"])
        mape_scores.append(metrics["MAPE"])
        r2_scores.append(metrics["R2"])

        all_y_true.extend(y_te)
        all_y_pred.extend(preds)
        all_p_curr.extend(p_current[test_idx])

    cv_summary = {
        "RMSE_mean": round(float(np.mean(rmse_scores)), 2),
        "RMSE_std": round(float(np.std(rmse_scores)), 2),
        "MAE_mean": round(float(np.mean(mae_scores)), 2),
        "MAE_std": round(float(np.std(mae_scores)), 2),
        "MAPE_mean": round(float(np.mean(mape_scores)), 2),
        "MAPE_std": round(float(np.std(mape_scores)), 2),
        "R2_mean": round(float(np.mean(r2_scores)), 4),
        "R2_std": round(float(np.std(r2_scores)), 4),
    }

    warning_summary = evaluate_price_alerts(
        np.array(all_p_curr),
        np.array(all_y_true),
        np.array(all_y_pred)
    )

    return cv_summary, warning_summary

if __name__ == "__main__":
    y_true = np.array([3500, 3600, 3800, 4000, 3900])
    y_pred = np.array([3480, 3550, 3750, 3950, 3880])
    metrics = compute_regression_metrics(y_true, y_pred)
    print("Sample Metrics:", metrics)
