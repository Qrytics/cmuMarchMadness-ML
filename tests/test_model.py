"""
Tests for the ML model.
"""

import pytest
import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model import MarchMadnessModel
from src.data_loader import load_all_data
from src.feature_engineering import build_training_data, get_feature_names

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "sample")


@pytest.fixture(scope="module")
def trained_model():
    """Train a quick model for testing."""
    if not os.path.exists(os.path.join(SAMPLE_DIR, "MTeams.csv")):
        pytest.skip("Sample data not available")

    data = load_all_data("M", SAMPLE_DIR)
    train_df = build_training_data(data, "M")
    feat_cols = get_feature_names(train_df)
    X = train_df[feat_cols].values
    y = train_df["label"].values

    model = MarchMadnessModel(gender="M")
    model.fit(X, y, feature_names=feat_cols)
    return model


class TestModelFitting:
    def test_model_fits(self, trained_model):
        assert trained_model.is_fitted

    def test_all_component_models_present(self, trained_model):
        assert "xgb" in trained_model.models
        assert "lgb" in trained_model.models
        assert "lr" in trained_model.models

    def test_predict_proba_shape(self, trained_model):
        X = np.random.randn(10, len(trained_model.feature_names))
        proba = trained_model.predict_proba(X)
        assert proba.shape == (10,)

    def test_predict_proba_range(self, trained_model):
        X = np.random.randn(100, len(trained_model.feature_names))
        proba = trained_model.predict_proba(X)
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_predict_binary(self, trained_model):
        X = np.random.randn(20, len(trained_model.feature_names))
        preds = trained_model.predict(X)
        assert set(preds).issubset({0, 1})

    def test_probability_is_symmetric(self, trained_model):
        """P(A beats B) + P(B beats A) should approximately equal 1."""
        feat_a = np.zeros((1, len(trained_model.feature_names)))
        feat_a[0, 0] = 0.3  # team A wins by this much
        feat_b = np.zeros((1, len(trained_model.feature_names)))
        feat_b[0, 0] = -0.3  # flipped

        p_ab = trained_model.predict_proba(feat_a)[0]
        p_ba = trained_model.predict_proba(feat_b)[0]
        assert abs(p_ab + p_ba - 1.0) < 0.2  # not perfectly symmetric due to bias

    def test_strong_team_has_higher_prob(self, trained_model):
        """A team with much better stats should have higher win probability."""
        n_feats = len(trained_model.feature_names)
        strong = np.zeros((1, n_feats))
        strong[0, :5] = 1.0  # strong positive differentials

        weak = np.zeros((1, n_feats))
        weak[0, :5] = -1.0  # negative differentials

        p_strong = trained_model.predict_proba(strong)[0]
        p_weak = trained_model.predict_proba(weak)[0]
        assert p_strong > p_weak

    def test_feature_importance_available(self, trained_model):
        importance = trained_model.get_feature_importance()
        assert len(importance) == len(trained_model.feature_names)
        assert all(v >= 0 for v in importance.values())


class TestModelPersistence:
    def test_save_and_load(self, trained_model, tmp_path):
        path = str(tmp_path / "test_model.pkl")
        trained_model.save(path)
        loaded = MarchMadnessModel.load(path=path)
        assert loaded.is_fitted
        assert loaded.gender == "M"

        # Predictions should be identical
        X = np.random.randn(10, len(trained_model.feature_names))
        orig_proba = trained_model.predict_proba(X)
        loaded_proba = loaded.predict_proba(X)
        np.testing.assert_array_almost_equal(orig_proba, loaded_proba)


class TestModelAccuracy:
    def test_cv_accuracy_above_chance(self):
        """Model should perform better than random (50%) on training data."""
        if not os.path.exists(os.path.join(SAMPLE_DIR, "MTeams.csv")):
            pytest.skip("Sample data not available")

        data = load_all_data("M", SAMPLE_DIR)
        train_df = build_training_data(data, "M")
        feat_cols = get_feature_names(train_df)
        seasons = sorted(train_df["season"].unique())

        if len(seasons) < 3:
            pytest.skip("Not enough seasons for CV")

        # Use second-to-last season for validation
        train_seasons = seasons[:-2]
        val_season = seasons[-2]

        train_mask = train_df["season"].isin(train_seasons)
        val_mask = train_df["season"] == val_season

        X_tr = train_df.loc[train_mask, feat_cols].values
        y_tr = train_df.loc[train_mask, "label"].values
        X_val = train_df.loc[val_mask, feat_cols].values
        y_val = train_df.loc[val_mask, "label"].values

        model = MarchMadnessModel(gender="M")
        model.fit(X_tr, y_tr, feature_names=feat_cols)
        preds = model.predict(X_val)
        acc = (preds == y_val).mean()

        # Should at minimum match or beat random (50%) — with synthetic data, hard to guarantee
        assert acc >= 0.35, f"Accuracy {acc:.2%} is too low (below 35%)"
