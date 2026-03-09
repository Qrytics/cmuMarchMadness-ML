"""
Prediction generation for the CMU March Madness competition.
Generates all-pairs predictions for men's and women's tournaments.

Submission format: CSV with columns 'WTeamID' and 'LTeamID'
- Men's:   MTourneyPredictions.csv  (C(381,2) = 72,390 rows)
- Women's: WTourneyPredictions.csv  (C(379,2) = 71,631 rows)

Usage:
    python -m src.predict          # generate both M and W predictions
    python -m src.predict --gender M
    python -m src.predict --gender W
"""

import argparse
import os
import itertools
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data_loader import load_all_data, load_2026_teams
from src.feature_engineering import build_team_features, build_matchup_features, get_feature_names
from src.model import MarchMadnessModel

PRED_DIR = os.path.join(os.path.dirname(__file__), "..", "predictions")
os.makedirs(PRED_DIR, exist_ok=True)

# Prediction year for the 2026 tournament
PRED_SEASON = 2025  # Use 2024-25 regular season stats for 2026 tournament


def generate_predictions(gender="M", data_dir=None, save=True):
    """
    Generate predictions for all possible team matchups.
    Returns a DataFrame with columns WTeamID, LTeamID.
    """
    print(f"\n{'='*60}")
    print(f"Generating {gender} predictions...")
    print(f"{'='*60}")

    # Load model
    model = MarchMadnessModel.load(gender=gender)
    print(f"  Loaded model from disk.")

    # Load data for the most recent season's stats
    data = load_all_data(gender, data_dir)
    rankings = data["rankings"]
    season_stats = data["season_stats"]

    # Get the most recent season available
    available_seasons = sorted(season_stats["Season"].unique())
    use_season = available_seasons[-1]
    print(f"  Using season {use_season} stats for 2026 predictions.")

    # Build team features for the most recent season
    team_feats = build_team_features(season_stats, rankings, use_season)

    # Get 2026 team list
    teams_df = load_2026_teams(gender, data_dir)
    team_ids = sorted(teams_df["TeamID"].tolist())
    print(f"  Teams: {len(team_ids)}")

    # Expected pairs
    expected = len(team_ids) * (len(team_ids) - 1) // 2
    print(f"  Expected matchups: {expected:,}")

    # Generate all pairs
    pairs = list(itertools.combinations(team_ids, 2))
    print(f"  Building feature matrix for {len(pairs):,} matchups...")

    rows = []
    batch_size = 10000
    predictions = []

    for i in tqdm(range(0, len(pairs), batch_size), desc=f"Predicting {gender}"):
        batch = pairs[i: i + batch_size]
        feat_rows = []
        for t1, t2 in batch:
            f1 = team_feats.get(t1, {})
            f2 = team_feats.get(t2, {})
            feat = build_matchup_features(f1, f2)
            feat_rows.append(feat)

        feat_df = pd.DataFrame(feat_rows)
        # Align columns with trained model's feature names
        for col in model.feature_names:
            if col not in feat_df.columns:
                feat_df[col] = 0.0
        feat_df = feat_df[model.feature_names]

        proba = model.predict_proba(feat_df)

        for j, (t1, t2) in enumerate(batch):
            p = proba[j]
            if p >= 0.5:
                predictions.append({"WTeamID": t1, "LTeamID": t2})
            else:
                predictions.append({"WTeamID": t2, "LTeamID": t1})

    pred_df = pd.DataFrame(predictions)
    print(f"  Generated {len(pred_df):,} predictions.")

    if save:
        filename = f"{gender}TourneyPredictions.csv"
        path = os.path.join(PRED_DIR, filename)
        pred_df.to_csv(path, index=False)
        print(f"  Saved to {path}")

    return pred_df


def main():
    parser = argparse.ArgumentParser(description="Generate March Madness predictions")
    parser.add_argument("--gender", choices=["M", "W", "both"], default="both",
                        help="Which tournament to predict")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Directory containing data files")
    parser.add_argument("--no-save", action="store_true",
                        help="Do not save predictions")
    args = parser.parse_args()

    genders = ["M", "W"] if args.gender == "both" else [args.gender]
    for g in genders:
        generate_predictions(g, data_dir=args.data_dir, save=not args.no_save)


if __name__ == "__main__":
    main()
