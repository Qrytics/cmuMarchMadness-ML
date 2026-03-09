"""
Training script for the CMU March Madness ML Competition.
Trains separate models for men's and women's tournaments.

Usage:
    python -m src.train          # train both M and W models using sample data
    python -m src.train --gender M --data-dir data/raw
    python -m src.train --gender W --data-dir data/raw
"""

import argparse
import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score

from src.data_loader import load_all_data
from src.feature_engineering import build_training_data, get_feature_names
from src.model import MarchMadnessModel

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(RESULTS_DIR, exist_ok=True)


def train_gender(gender="M", data_dir=None, cv_folds=5, save=True):
    """
    Train and evaluate a model for the given gender.
    Returns the trained model and metrics dict.
    """
    print(f"\n{'='*60}")
    print(f"Training {gender} model...")
    print(f"{'='*60}")

    # Load data
    data = load_all_data(gender, data_dir)
    print(f"  Seasons: {sorted(data['regular']['Season'].unique())}")
    print(f"  Teams: {data['teams'].shape[0]}")
    print(f"  Regular season games: {data['regular'].shape[0]}")
    print(f"  Tournament games: {data['tourney'].shape[0]}")

    # Build training data
    train_df = build_training_data(data, gender)
    feature_cols = get_feature_names(train_df)
    print(f"  Training samples: {len(train_df)} | Features: {len(feature_cols)}")

    X = train_df[feature_cols].values
    y = train_df["label"].values

    # Cross-validation using season as fold to simulate real validation
    seasons = train_df["season"].values
    unique_seasons = sorted(train_df["season"].unique())

    # Walk-forward CV: train on past seasons, validate on next season
    wf_scores = {"accuracy": [], "log_loss": [], "auc": []}
    n_wf = min(5, len(unique_seasons) - 2)  # use last n seasons for validation

    for val_season in unique_seasons[-n_wf:]:
        train_mask = seasons < val_season
        val_mask = seasons == val_season

        if train_mask.sum() < 10 or val_mask.sum() < 5:
            continue

        X_tr, y_tr = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]

        m = MarchMadnessModel(gender=gender)
        m.fit(X_tr, y_tr, feature_names=feature_cols)
        proba = m.predict_proba(X_val)
        pred = (proba >= 0.5).astype(int)

        wf_scores["accuracy"].append(accuracy_score(y_val, pred))
        wf_scores["log_loss"].append(log_loss(y_val, proba))
        wf_scores["auc"].append(roc_auc_score(y_val, proba))
        print(f"  Season {val_season}: acc={wf_scores['accuracy'][-1]:.3f} "
              f"logloss={wf_scores['log_loss'][-1]:.3f} "
              f"auc={wf_scores['auc'][-1]:.3f}")

    metrics = {
        "gender": gender,
        "n_samples": len(train_df),
        "n_features": len(feature_cols),
        "cv_accuracy_mean": float(np.mean(wf_scores["accuracy"])),
        "cv_accuracy_std": float(np.std(wf_scores["accuracy"])),
        "cv_log_loss_mean": float(np.mean(wf_scores["log_loss"])),
        "cv_log_loss_std": float(np.std(wf_scores["log_loss"])),
        "cv_auc_mean": float(np.mean(wf_scores["auc"])),
        "cv_auc_std": float(np.std(wf_scores["auc"])),
    }

    print(f"\n  Walk-forward CV Results:")
    print(f"  Accuracy: {metrics['cv_accuracy_mean']:.3f} ± {metrics['cv_accuracy_std']:.3f}")
    print(f"  Log Loss: {metrics['cv_log_loss_mean']:.3f} ± {metrics['cv_log_loss_std']:.3f}")
    print(f"  AUC:      {metrics['cv_auc_mean']:.3f} ± {metrics['cv_auc_std']:.3f}")

    # Train final model on ALL data
    print(f"\n  Training final model on all {len(train_df)} samples...")
    final_model = MarchMadnessModel(gender=gender)
    final_model.fit(X, y, feature_names=feature_cols)

    if save:
        model_path = final_model.save()
        metrics_path = os.path.join(RESULTS_DIR, f"{gender.lower()}_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"  Metrics saved to {metrics_path}")

        # Save feature importances
        importance = final_model.get_feature_importance()
        imp_df = pd.DataFrame(list(importance.items()), columns=["Feature", "Importance"])
        imp_df = imp_df.sort_values("Importance", ascending=False)
        imp_path = os.path.join(RESULTS_DIR, f"{gender.lower()}_feature_importance.csv")
        imp_df.to_csv(imp_path, index=False)
        print(f"  Feature importance saved to {imp_path}")

    return final_model, metrics


def main():
    parser = argparse.ArgumentParser(description="Train March Madness ML models")
    parser.add_argument("--gender", choices=["M", "W", "both"], default="both",
                        help="Which tournament to train for")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Directory containing Kaggle data files")
    parser.add_argument("--no-save", action="store_true",
                        help="Do not save the model after training")
    args = parser.parse_args()

    genders = ["M", "W"] if args.gender == "both" else [args.gender]
    all_metrics = {}

    for g in genders:
        model, metrics = train_gender(g, data_dir=args.data_dir, save=not args.no_save)
        all_metrics[g] = metrics

    # Save combined metrics
    combined_path = os.path.join(RESULTS_DIR, "training_metrics.json")
    with open(combined_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nAll metrics saved to {combined_path}")
    return all_metrics


if __name__ == "__main__":
    main()
