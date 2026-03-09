"""
Data loading utilities for the CMU March Madness ML Competition.
Loads NCAA basketball data in Kaggle format (sample or real Kaggle data).
"""

import os
import pandas as pd
import numpy as np

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "sample")
KAGGLE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def _load(filename, data_dir=None):
    """Load a CSV, preferring kaggle data if available, else sample data."""
    if data_dir is not None:
        path = os.path.join(data_dir, filename)
        if os.path.exists(path):
            return pd.read_csv(path)

    kaggle_path = os.path.join(KAGGLE_DIR, filename)
    if os.path.exists(kaggle_path):
        return pd.read_csv(kaggle_path)

    sample_path = os.path.join(SAMPLE_DIR, filename)
    if os.path.exists(sample_path):
        return pd.read_csv(sample_path)

    raise FileNotFoundError(
        f"Could not find {filename} in {KAGGLE_DIR} or {SAMPLE_DIR}. "
        "Run scripts/download_data.py to fetch Kaggle data, or "
        "scripts/generate_sample_data.py to generate sample data."
    )


def load_teams(gender="M", data_dir=None):
    return _load(f"{gender}Teams.csv", data_dir)


def load_regular_season(gender="M", data_dir=None):
    return _load(f"{gender}RegularSeasonDetailedResults.csv", data_dir)


def load_tourney_results(gender="M", data_dir=None):
    return _load(f"{gender}NCAATourneyDetailedResults.csv", data_dir)


def load_seeds(gender="M", data_dir=None):
    return _load(f"{gender}Seeds.csv", data_dir)


def load_rankings(gender="M", data_dir=None):
    try:
        return _load(f"{gender}Rankings.csv", data_dir)
    except FileNotFoundError:
        return pd.DataFrame(columns=["Season", "RankingDayNum", "SystemName", "TeamID", "OrdinalRank"])


def load_2026_teams(gender="M", data_dir=None):
    """Load the 2026 team list for prediction generation."""
    try:
        return _load(f"{gender}Teams2026.csv", data_dir)
    except FileNotFoundError:
        # Fall back to the general team list
        return load_teams(gender, data_dir)


def get_seed_number(seed_str):
    """Extract numeric seed from seed string like 'W01', 'X11a'."""
    import re
    m = re.search(r"\d+", str(seed_str))
    return int(m.group()) if m else 16


def compute_season_stats(season_df):
    """
    Compute per-team season averages from a detailed results dataframe.
    Returns a DataFrame with one row per (Season, TeamID) with aggregated stats.
    """
    cols = ["FGM", "FGA", "FGM3", "FGA3", "FTM", "FTA",
            "OR", "DR", "Ast", "TO", "Stl", "Blk", "PF"]

    records = []

    for season, sdf in season_df.groupby("Season"):
        team_stats = {}

        def _accumulate(row, prefix, suffix):
            tid = row[f"{prefix}TeamID"]
            score = row[f"{prefix}Score"]
            opp_score = row[f"{'L' if prefix == 'W' else 'W'}Score"]
            win = 1 if prefix == "W" else 0

            if tid not in team_stats:
                team_stats[tid] = {
                    "Wins": 0, "Games": 0,
                    "PointsFor": 0, "PointsAgainst": 0,
                    **{f"Avg{c}": 0.0 for c in cols},
                }

            s = team_stats[tid]
            s["Wins"] += win
            s["Games"] += 1
            s["PointsFor"] += score
            s["PointsAgainst"] += opp_score
            for c in cols:
                s[f"Avg{c}"] += row.get(f"{prefix}{c}", 0.0)

        for _, row in sdf.iterrows():
            _accumulate(row, "W", "L")
            _accumulate(row, "L", "W")

        for tid, s in team_stats.items():
            g = max(s["Games"], 1)
            record = {
                "Season": season,
                "TeamID": tid,
                "WinPct": s["Wins"] / g,
                "AvgPointsFor": s["PointsFor"] / g,
                "AvgPointsAgainst": s["PointsAgainst"] / g,
                "PointDiff": (s["PointsFor"] - s["PointsAgainst"]) / g,
                "NumGames": g,
            }
            for c in cols:
                record[f"Avg{c}"] = s[f"Avg{c}"] / g
            records.append(record)

    return pd.DataFrame(records)


def load_all_data(gender="M", data_dir=None):
    """Load and return all relevant datasets for a gender."""
    teams = load_teams(gender, data_dir)
    regular = load_regular_season(gender, data_dir)
    tourney = load_tourney_results(gender, data_dir)
    seeds = load_seeds(gender, data_dir)
    rankings = load_rankings(gender, data_dir)

    season_stats = compute_season_stats(regular)

    return {
        "teams": teams,
        "regular": regular,
        "tourney": tourney,
        "seeds": seeds,
        "rankings": rankings,
        "season_stats": season_stats,
    }
