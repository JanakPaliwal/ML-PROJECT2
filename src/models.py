"""
models.py
ML models used for predicting next-week mandi prices.

Models used (all from DSC-17 syllabus):
1. Naive Baseline  - Just predicts today's price as next week's price (no learning)
2. Linear Regression - Finds a straight-line relationship between features and price
3. Ridge Regression  - Linear Regression + Regularization to prevent overfitting
4. k-NN Regressor   - Finds k most similar past days and averages their prices
5. MLP (Neural Network) - Multi-layer perceptron with backpropagation
"""

import numpy as np
from typing import Dict
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
import joblib

class NaiveBaseline(BaseEstimator, RegressorMixin):
    """
    Simplest possible prediction: next week's price = today's price.
    This is our baseline to compare ML models against.
    If an ML model can't beat this, it's not useful.
    """
    def __init__(self, price_col_idx: int = 0):
        self.price_col_idx = price_col_idx

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        X_arr = np.asarray(X)
        return X_arr[:, self.price_col_idx]

def get_model_registry() -> Dict[str, BaseEstimator]:
    """
    Returns all ML models we want to train and compare.

    Each model uses StandardScaler first (Feature Scaling from Unit 2)
    to normalize features to mean=0, std=1 before training.

    Models used:
    - Linear Regression (Unit 3): y = w1*x1 + w2*x2 + ... + b
    - Ridge Regression (Unit 3): Linear Regression + L2 regularization to reduce overfitting
    - k-NN Regressor (Unit 4): Predicts using average of k nearest neighbors
    - MLP Neural Network (Unit 4): Multi-layer perceptron trained with backpropagation
    """
    models = {
        "Naive Baseline": NaiveBaseline(price_col_idx=0),

        "Linear Regression": Pipeline([
            ("scaler", StandardScaler()),        # Feature Scaling (Unit 2)
            ("regressor", LinearRegression())    # Linear Regression (Unit 3)
        ]),

        "Ridge Regression": Pipeline([
            ("scaler", StandardScaler()),        # Feature Scaling (Unit 2)
            ("regressor", Ridge(alpha=10.0))     # Regularization to prevent overfitting (Unit 3)
        ]),

        "k-NN Regressor": Pipeline([
            ("scaler", StandardScaler()),        # Feature Scaling (Unit 2)
            ("regressor", KNeighborsRegressor(   # k-Nearest Neighbors (Unit 4)
                n_neighbors=5,
                weights="distance",
                metric="minkowski"
            ))
        ]),

        "MLP Neural Network": Pipeline([
            ("scaler", StandardScaler()),        # Feature Scaling (Unit 2)
            ("regressor", MLPRegressor(          # Neural Network with Backpropagation (Unit 4)
                hidden_layer_sizes=(32, 16),     # 2 hidden layers
                activation="relu",               # ReLU activation function
                solver="adam",                   # Gradient descent optimizer
                alpha=0.01,                      # Regularization strength
                max_iter=500,
                early_stopping=True,
                n_iter_no_change=20,
                random_state=42
            ))
        ])
    }
    return models

def save_trained_model(model: BaseEstimator, filepath: str) -> None:
    joblib.dump(model, filepath)

def load_trained_model(filepath: str) -> BaseEstimator:
    return joblib.load(filepath)

if __name__ == "__main__":
    registry = get_model_registry()
    print("ML Models registered:")
    for name in registry.keys():
        print(f" - {name}")
