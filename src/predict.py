"""
Prediction generation for the CMU March Madness competition.
Generates all-pairs predictions for men's and women's tournaments.

Two output formats are produced:
1. WTeamID / LTeamID CSV  (bracket-style; WTeamID is the predicted winner)
2. Kaggle submission CSV   (ID = "Season_TeamID1_TeamID2", Pred = win probability)

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

# Kaggle competition year
COMPETITION_SEASON = 2026


def generate_predictions(gender="M", data_dir=None, save=True):
    """
    Generate predictions for all possible team matchups.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (bracket_df, kaggle_df) where bracket_df has columns WTeamID/LTeamID
        and kaggle_df has columns ID/Pred in the official Kaggle format.
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
    print(f"  Using season {use_season} stats for {COMPETITION_SEASON} predictions.")

    # Build team features for the most recent season
    team_feats = build_team_features(season_stats, rankings, use_season)

    # Get 2026 team list
    teams_df = load_2026_teams(gender, data_dir)
    team_ids = sorted(teams_df["TeamID"].tolist())
    print(f"  Teams: {len(team_ids)}")

    # Expected pairs
    expected = len(team_ids) * (len(team_ids) - 1) // 2
    print(f"  Expected matchups: {expected:,}")

    # Generate all pairs (always smaller ID first to match Kaggle format)
    pairs = list(itertools.combinations(team_ids, 2))
    print(f"  Building feature matrix for {len(pairs):,} matchups...")

    bracket_predictions = []
    kaggle_predictions = []

    batch_size = 10000

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
            p = float(proba[j])  # prob that t1 beats t2
            # t1 < t2 always (from combinations), so ID is always consistent
            kaggle_id = f"{COMPETITION_SEASON}_{t1}_{t2}"
            kaggle_predictions.append({"ID": kaggle_id, "Pred": round(p, 6)})

            # Bracket format: predicted winner first
            if p >= 0.5:
                bracket_predictions.append({"WTeamID": t1, "LTeamID": t2})
            else:
                bracket_predictions.append({"WTeamID": t2, "LTeamID": t1})

    bracket_df = pd.DataFrame(bracket_predictions)
    kaggle_df = pd.DataFrame(kaggle_predictions)
    print(f"  Generated {len(bracket_df):,} predictions.")

    if save:
        # Bracket-style CSV (required filename for CMU MMML competition)
        bracket_filename = f"{gender}NCAATourneyPredictions.csv"
        bracket_path = os.path.join(PRED_DIR, bracket_filename)
        bracket_df.to_csv(bracket_path, index=False)
        print(f"  Bracket predictions saved to {bracket_path}")

        # Kaggle submission CSV
        kaggle_filename = f"{gender}KaggleSubmission.csv"
        kaggle_path = os.path.join(PRED_DIR, kaggle_filename)
        kaggle_df.to_csv(kaggle_path, index=False)
        print(f"  Kaggle submission saved to {kaggle_path}")

    return bracket_df, kaggle_df


def generate_combined_kaggle_submission(data_dir=None):
    """
    Combine M and W predictions into a single Kaggle submission file.

    If a SampleSubmissionStage2.csv exists (Kaggle format), the output is
    filtered/aligned to exactly the IDs required by the competition.
    The combined file contains all rows from both genders, sorted by ID.
    """
    dfs = []
    for g in ["M", "W"]:
        path = os.path.join(PRED_DIR, f"{g}KaggleSubmission.csv")
        if os.path.exists(path):
            dfs.append(pd.read_csv(path))

    if not dfs:
        print("No Kaggle submission files found. Run generate_predictions first.")
        return None

    combined = pd.concat(dfs, ignore_index=True).sort_values("ID").reset_index(drop=True)

    # Try to align with Stage 2 sample submission if it exists
    from src.data_loader import _load, KAGGLE_DIR, SAMPLE_DIR
    try:
        stage2 = _load("SampleSubmissionStage2.csv", data_dir)
        # Filter our predictions to only the IDs Kaggle needs
        stage2_ids = set(stage2["ID"])
        aligned = combined[combined["ID"].isin(stage2_ids)].copy()
        missing = stage2_ids - set(aligned["ID"])
        if missing:
            print(f"  Warning: {len(missing)} Stage-2 IDs not found in our predictions "
                  "— filling with 0.5")
            missing_df = pd.DataFrame({"ID": sorted(missing), "Pred": 0.5})
            aligned = pd.concat([aligned, missing_df], ignore_index=True).sort_values("ID")
        combined = aligned
        print(f"  Aligned to Stage-2 format ({len(combined):,} rows)")
    except FileNotFoundError:
        print(f"  SampleSubmissionStage2.csv not found; keeping all {len(combined):,} rows")

    out_path = os.path.join(PRED_DIR, "submission.csv")
    combined.to_csv(out_path, index=False)
    print(f"  Combined Kaggle submission saved to {out_path} ({len(combined):,} rows)")
    return combined


def main():
    parser = argparse.ArgumentParser(description="Generate March Madness predictions")
    parser.add_argument("--gender", choices=["M", "W", "both"], default="both",
                        help="Which tournament to predict")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Directory containing data files")
    parser.add_argument("--no-save", action="store_true",
                        help="Do not save predictions")
    parser.add_argument("--combine", action="store_true",
                        help="After predicting, merge M and W into one submission.csv")
    args = parser.parse_args()

    genders = ["M", "W"] if args.gender == "both" else [args.gender]
    for g in genders:
        generate_predictions(g, data_dir=args.data_dir, save=not args.no_save)

    if args.combine or args.gender == "both":
        generate_combined_kaggle_submission(data_dir=args.data_dir)


if __name__ == "__main__":
    main()

