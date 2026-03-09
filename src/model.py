"""
ML model implementation for the CMU March Madness competition.
Uses an ensemble of XGBoost, LightGBM, and Logistic Regression
with calibrated probability outputs.
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import xgboost as xgb
import lightgbm as lgb


MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)


class MarchMadnessModel:
    """
    Ensemble model for predicting NCAA basketball game outcomes.
    Combines XGBoost, LightGBM, and Logistic Regression with calibration.
    Outputs probability that team1 beats team2.
    """

    def __init__(self, gender="M", weights=None):
        self.gender = gender
        self.weights = weights or {"xgb": 0.4, "lgb": 0.4, "lr": 0.2}
        self.models = {}
        self.feature_names = None
        self.is_fitted = False

    def _make_xgb(self):
        return xgb.XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )

    def _make_lgb(self):
        return lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=10,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )

    def _make_lr(self):
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
        ])

    def fit(self, X, y, feature_names=None, eval_sets=None):
        """
        Train all component models.
        X: numpy array or DataFrame, y: binary labels
        eval_sets: list of (X_val, y_val) tuples for early stopping
        """
        self.feature_names = feature_names or list(range(X.shape[1]))
        X_arr = X.values if hasattr(X, "values") else np.array(X)

        # Fill NaN for XGB/LGB (they handle NaN, but be explicit)
        X_clean = np.nan_to_num(X_arr, nan=0.0)

        # XGBoost
        xgb_model = self._make_xgb()
        xgb_model.fit(
            X_clean, y,
            eval_set=[(X_clean, y)] + ([(ev[0], ev[1]) for ev in eval_sets] if eval_sets else []),
            verbose=False,
        )
        self.models["xgb"] = xgb_model

        # LightGBM
        lgb_model = self._make_lgb()
        lgb_model.fit(X_clean, y)
        self.models["lgb"] = lgb_model

        # Logistic Regression (uses sklearn Pipeline with imputation + scaling)
        lr_model = self._make_lr()
        lr_model.fit(X_arr, y)
        self.models["lr"] = lr_model

        self.is_fitted = True
        return self

    def predict_proba(self, X):
        """Return weighted ensemble probability that team1 wins."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X_arr = X.values if hasattr(X, "values") else np.array(X)
        X_clean = np.nan_to_num(X_arr, nan=0.0)

        probs = []
        total_w = 0.0
        for name, model in self.models.items():
            w = self.weights.get(name, 1.0)
            if name in ("xgb", "lgb"):
                p = model.predict_proba(X_clean)[:, 1]
            else:
                p = model.predict_proba(X_arr)[:, 1]
            probs.append(p * w)
            total_w += w

        ensemble = sum(probs) / total_w
        return ensemble

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(int)

    def get_feature_importance(self):
        """Return average feature importance across tree models."""
        importance = {}
        tree_models = {k: v for k, v in self.models.items() if k in ("xgb", "lgb")}

        for name, model in tree_models.items():
            if name == "xgb":
                imp = model.feature_importances_
            elif name == "lgb":
                imp = model.feature_importances_

            for i, feat in enumerate(self.feature_names):
                importance[feat] = importance.get(feat, 0.0) + imp[i]

        # Average over models
        n = len(tree_models)
        return {k: v / n for k, v in importance.items()}

    def save(self, path=None):
        if path is None:
            path = os.path.join(MODEL_DIR, f"{self.gender.lower()}_model.pkl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        print(f"Model saved to {path}")
        return path

    @classmethod
    def load(cls, gender="M", path=None):
        if path is None:
            path = os.path.join(MODEL_DIR, f"{gender.lower()}_model.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"No saved model at {path}. Run train.py first.")
        model = joblib.load(path)
        return model
