"""
ML model implementation for the CMU March Madness competition.
Uses an ensemble of XGBoost, LightGBM, and Logistic Regression
with calibrated probability outputs.

Auto-improvement is provided via ``MarchMadnessModel.tune()``, which uses
Optuna to search for better hyperparameters automatically.
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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
import xgboost as xgb
import lightgbm as lgb


MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)


class MarchMadnessModel:
    """
    Ensemble model for predicting NCAA basketball game outcomes.
    Combines XGBoost, LightGBM, and Logistic Regression with calibration.
    Outputs probability that team1 beats team2.

    Auto-improvement
    ----------------
    Call ``model.tune(X, y, n_trials=50)`` before (or instead of) ``fit()``
    to let Optuna search for optimal hyperparameters.  After tuning the best
    params are stored in ``self.best_params`` and ``fit()`` will use them
    automatically.
    """

    def __init__(self, gender="M", weights=None, best_params=None):
        self.gender = gender
        self.weights = weights or {"xgb": 0.4, "lgb": 0.4, "lr": 0.2}
        self.models = {}
        self.feature_names = None
        self.is_fitted = False
        # Hyperparameters discovered by tune(); None means use defaults
        self.best_params = best_params or {}

    # ------------------------------------------------------------------
    # Model factories
    # ------------------------------------------------------------------

    def _make_xgb(self, params=None):
        p = {
            "n_estimators": 300,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "gamma": 0.1,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "eval_metric": "logloss",
            "random_state": 42,
            "n_jobs": -1,
        }
        if params:
            p.update({k.replace("xgb_", ""): v for k, v in params.items() if k.startswith("xgb_")})
        return xgb.XGBClassifier(**p)

    def _make_lgb(self, params=None):
        p = {
            "n_estimators": 300,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_samples": 10,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }
        if params:
            p.update({k.replace("lgb_", ""): v for k, v in params.items() if k.startswith("lgb_")})
        return lgb.LGBMClassifier(**p)

    def _make_lr(self, params=None):
        C = 1.0
        if params and "lr_C" in params:
            C = params["lr_C"]
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=C, max_iter=1000, random_state=42)),
        ])

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

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

        params = self.best_params

        # XGBoost
        xgb_model = self._make_xgb(params)
        xgb_model.fit(
            X_clean, y,
            eval_set=[(X_clean, y)] + ([(ev[0], ev[1]) for ev in eval_sets] if eval_sets else []),
            verbose=False,
        )
        self.models["xgb"] = xgb_model

        # LightGBM
        lgb_model = self._make_lgb(params)
        lgb_model.fit(X_clean, y)
        self.models["lgb"] = lgb_model

        # Logistic Regression (uses sklearn Pipeline with imputation + scaling)
        lr_model = self._make_lr(params)
        lr_model.fit(X_arr, y)
        self.models["lr"] = lr_model

        self.is_fitted = True
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Auto-improvement via Optuna
    # ------------------------------------------------------------------

    def tune(self, X, y, feature_names=None, n_trials=50, cv_folds=5,
             timeout=None, show_progress=True):
        """
        Automatically search for better hyperparameters using Optuna.

        Runs ``n_trials`` of Bayesian optimisation (TPE sampler) using
        stratified k-fold cross-validation on log-loss.  The best found
        parameters are stored in ``self.best_params`` and will be used
        automatically by subsequent calls to ``fit()``.

        Parameters
        ----------
        X : array-like
        y : array-like
        feature_names : list or None
        n_trials : int
            Number of Optuna trials.
        cv_folds : int
            Number of cross-validation folds.
        timeout : float or None
            Time limit in seconds (passed to Optuna).
        show_progress : bool
            Whether to print Optuna progress.

        Returns
        -------
        dict
            The best hyperparameters found.
        """
        try:
            import optuna
        except ImportError:
            print("  [tune] optuna not installed. Run: pip install optuna")
            return {}

        optuna.logging.set_verbosity(
            optuna.logging.INFO if show_progress else optuna.logging.WARNING
        )

        X_arr = X.values if hasattr(X, "values") else np.array(X)
        X_clean = np.nan_to_num(X_arr, nan=0.0)
        y_arr = np.array(y)

        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

        def objective(trial):
            params = {
                # XGBoost params
                "xgb_n_estimators": trial.suggest_int("xgb_n_estimators", 100, 600),
                "xgb_max_depth": trial.suggest_int("xgb_max_depth", 3, 7),
                "xgb_learning_rate": trial.suggest_float("xgb_learning_rate", 0.01, 0.2, log=True),
                "xgb_subsample": trial.suggest_float("xgb_subsample", 0.6, 1.0),
                "xgb_colsample_bytree": trial.suggest_float("xgb_colsample_bytree", 0.5, 1.0),
                "xgb_min_child_weight": trial.suggest_int("xgb_min_child_weight", 1, 10),
                "xgb_gamma": trial.suggest_float("xgb_gamma", 0.0, 1.0),
                "xgb_reg_alpha": trial.suggest_float("xgb_reg_alpha", 1e-4, 10.0, log=True),
                "xgb_reg_lambda": trial.suggest_float("xgb_reg_lambda", 1e-4, 10.0, log=True),
                # LightGBM params
                "lgb_n_estimators": trial.suggest_int("lgb_n_estimators", 100, 600),
                "lgb_max_depth": trial.suggest_int("lgb_max_depth", 3, 7),
                "lgb_learning_rate": trial.suggest_float("lgb_learning_rate", 0.01, 0.2, log=True),
                "lgb_subsample": trial.suggest_float("lgb_subsample", 0.6, 1.0),
                "lgb_colsample_bytree": trial.suggest_float("lgb_colsample_bytree", 0.5, 1.0),
                "lgb_min_child_samples": trial.suggest_int("lgb_min_child_samples", 5, 50),
                "lgb_reg_alpha": trial.suggest_float("lgb_reg_alpha", 1e-4, 10.0, log=True),
                "lgb_reg_lambda": trial.suggest_float("lgb_reg_lambda", 1e-4, 10.0, log=True),
                # Logistic Regression
                "lr_C": trial.suggest_float("lr_C", 0.01, 10.0, log=True),
                # Ensemble weights
                "w_xgb": trial.suggest_float("w_xgb", 0.1, 0.8),
                "w_lgb": trial.suggest_float("w_lgb", 0.1, 0.8),
                "w_lr": trial.suggest_float("w_lr", 0.05, 0.4),
            }

            fold_losses = []
            for train_idx, val_idx in skf.split(X_clean, y_arr):
                Xtr, Xval = X_clean[train_idx], X_clean[val_idx]
                Xtr_raw = X_arr[train_idx]
                Xval_raw = X_arr[val_idx]
                ytr, yval = y_arr[train_idx], y_arr[val_idx]

                # XGBoost
                xgb_m = self._make_xgb(params)
                xgb_m.fit(Xtr, ytr, eval_set=[(Xtr, ytr)], verbose=False)
                p_xgb = xgb_m.predict_proba(Xval)[:, 1]

                # LightGBM
                lgb_m = self._make_lgb(params)
                lgb_m.fit(Xtr, ytr)
                p_lgb = lgb_m.predict_proba(Xval)[:, 1]

                # Logistic Regression
                lr_m = self._make_lr(params)
                lr_m.fit(Xtr_raw, ytr)
                p_lr = lr_m.predict_proba(Xval_raw)[:, 1]

                # Weighted ensemble
                w_sum = params["w_xgb"] + params["w_lgb"] + params["w_lr"]
                p_ens = (
                    params["w_xgb"] * p_xgb +
                    params["w_lgb"] * p_lgb +
                    params["w_lr"] * p_lr
                ) / w_sum
                p_ens = np.clip(p_ens, 1e-7, 1 - 1e-7)

                fold_losses.append(log_loss(yval, p_ens))

            return float(np.mean(fold_losses))

        study = optuna.create_study(direction="minimize",
                                    sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, timeout=timeout,
                       show_progress_bar=show_progress)

        best = study.best_params
        print(f"\n  [tune] Best log-loss: {study.best_value:.4f} "
              f"(after {len(study.trials)} trials)")
        print(f"  [tune] Best params: {best}")

        # Store best params and update weights
        self.best_params = best
        self.weights = {
            "xgb": best.get("w_xgb", 0.4),
            "lgb": best.get("w_lgb", 0.4),
            "lr": best.get("w_lr", 0.2),
        }
        return best

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def get_feature_importance(self):
        """Return average feature importance across tree models."""
        importance = {}
        tree_models = {k: v for k, v in self.models.items() if k in ("xgb", "lgb")}

        for name, model in tree_models.items():
            imp = model.feature_importances_
            for i, feat in enumerate(self.feature_names):
                importance[feat] = importance.get(feat, 0.0) + imp[i]

        # Average over models
        n = len(tree_models)
        return {k: v / n for k, v in importance.items()}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

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

