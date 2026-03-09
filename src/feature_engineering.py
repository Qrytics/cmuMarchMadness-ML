"""
Feature engineering for the CMU March Madness ML Competition.
Creates matchup-level features by diffing team season statistics.
"""

import numpy as np
import pandas as pd
from src.data_loader import get_seed_number


FEATURE_COLS = None  # set after first call to build_features

_EMPTY_RANK = {"AvgRank": np.nan, "BestRank": np.nan, "NumRankSystems": 0}


def get_team_ranking_features(rankings_agg_or_df, season, team_id, day_limit=133):
    """Look up pre-aggregated ranking features for a team.

    ``rankings_agg_or_df`` can be either:
    * A pre-aggregated dict (keyed by (season, team_id)) – fast O(1) path.
    * A raw rankings DataFrame – slower legacy path, kept for backwards compat.
    """
    if rankings_agg_or_df is None:
        return _EMPTY_RANK.copy()

    # Fast path: pre-aggregated dict
    if isinstance(rankings_agg_or_df, dict):
        return rankings_agg_or_df.get((season, team_id), _EMPTY_RANK).copy()

    # Legacy path: raw DataFrame
    rankings_df = rankings_agg_or_df
    if rankings_df.empty:
        return _EMPTY_RANK.copy()

    sub = rankings_df[
        (rankings_df["Season"] == season) &
        (rankings_df["TeamID"] == team_id) &
        (rankings_df["RankingDayNum"] <= day_limit)
    ]
    if sub.empty:
        return _EMPTY_RANK.copy()

    latest_day = sub["RankingDayNum"].max()
    latest = sub[sub["RankingDayNum"] == latest_day]
    return {
        "AvgRank": latest["OrdinalRank"].mean(),
        "BestRank": latest["OrdinalRank"].min(),
        "NumRankSystems": latest["SystemName"].nunique(),
    }


def build_team_features(season_stats_df, rankings, season):
    """Build a per-team feature dict for a given season.

    ``rankings`` can be a pre-aggregated dict or a raw DataFrame.
    """
    sub = season_stats_df[season_stats_df["Season"] == season].copy()
    features = {}
    for _, row in sub.iterrows():
        tid = row["TeamID"]
        rank_feats = get_team_ranking_features(rankings, season, tid)
        feats = row.to_dict()
        feats.update(rank_feats)
        features[tid] = feats
    return features


def build_matchup_features(team1_feats, team2_feats, seed1=None, seed2=None):
    """
    Build a feature vector for a matchup between team1 and team2.

    Features are differences: team1 - team2 (or derived ratios).
    Includes basic box-score differentials, advanced efficiency metrics,
    shooting quality, and seed information.
    """
    # ------------------------------------------------------------------
    # Basic stats
    # ------------------------------------------------------------------
    basic_stat_cols = [
        "WinPct", "AvgPointsFor", "AvgPointsAgainst", "PointDiff",
        "AvgFGM", "AvgFGA", "AvgFGM3", "AvgFGA3",
        "AvgFTM", "AvgFTA", "AvgOR", "AvgDR",
        "AvgAst", "AvgTO", "AvgStl", "AvgBlk", "AvgPF",
        "AvgRank", "BestRank",
    ]

    # Advanced metrics introduced by the improved compute_season_stats()
    adv_stat_cols = [
        "PythExpect", "Pace", "OffEff", "DefEff", "NetEff",
        "eFGPct", "TSPct", "FTPct", "FTRate", "TORatePerPoss",
    ]

    feat = {}

    for col in basic_stat_cols + adv_stat_cols:
        v1 = team1_feats.get(col, np.nan)
        v2 = team2_feats.get(col, np.nan)
        try:
            v1 = float(v1)
        except (TypeError, ValueError):
            v1 = np.nan
        try:
            v2 = float(v2)
        except (TypeError, ValueError):
            v2 = np.nan
        feat[f"diff_{col}"] = (v1 - v2) if (not np.isnan(v1) and not np.isnan(v2)) else 0.0

    # Shooting percentage differentials (derived from raw averages)
    for (fgm_col, fga_col) in [("AvgFGM", "AvgFGA"), ("AvgFGM3", "AvgFGA3"), ("AvgFTM", "AvgFTA")]:
        pct_col = fgm_col.replace("M", "Pct")
        v1a = team1_feats.get(fga_col, 1.0) or 1.0
        v2a = team2_feats.get(fga_col, 1.0) or 1.0
        v1m = team1_feats.get(fgm_col, 0.0) or 0.0
        v2m = team2_feats.get(fgm_col, 0.0) or 0.0
        pct1 = v1m / v1a
        pct2 = v2m / v2a
        feat[f"diff_{pct_col}"] = pct1 - pct2

    # ------------------------------------------------------------------
    # Derived composite features
    # ------------------------------------------------------------------
    feat["diff_OffEff_simple"] = (
        team1_feats.get("AvgPointsFor", 0) - team2_feats.get("AvgPointsFor", 0)
    )
    feat["diff_DefEff_simple"] = (
        team2_feats.get("AvgPointsAgainst", 0) - team1_feats.get("AvgPointsAgainst", 0)
    )
    feat["diff_NetEff_simple"] = feat["diff_OffEff_simple"] + feat["diff_DefEff_simple"]

    # Number of ranking systems available (proxy for data quality / prestige)
    feat["diff_NumRankSystems"] = (
        team1_feats.get("NumRankSystems", 0) - team2_feats.get("NumRankSystems", 0)
    )

    # ------------------------------------------------------------------
    # Seed features
    # ------------------------------------------------------------------
    if seed1 is not None and seed2 is not None:
        s1 = get_seed_number(seed1) if isinstance(seed1, str) else int(seed1)
        s2 = get_seed_number(seed2) if isinstance(seed2, str) else int(seed2)
        feat["SeedDiff"] = s1 - s2
        feat["HigherSeed"] = 1 if s1 < s2 else 0
        feat["SeedSum"] = s1 + s2
    else:
        feat["SeedDiff"] = 0
        feat["HigherSeed"] = 0
        feat["SeedSum"] = 16

    return feat


def build_training_data(data, gender="M"):
    """
    Build training dataset from historical tournament results.
    Each row is a matchup with features and label (1 = team1 wins).
    """
    tourney = data["tourney"]
    seeds = data["seeds"]
    season_stats = data["season_stats"]
    # Use pre-aggregated rankings dict if available, else fall back to raw DataFrame
    rankings = data.get("rankings_agg") or data.get("rankings")

    rows = []

    for season, season_tourney in tourney.groupby("Season"):
        team_feats = build_team_features(season_stats, rankings, season)
        season_seeds = seeds[seeds["Season"] == season]
        seed_map = dict(zip(season_seeds["TeamID"], season_seeds["Seed"]))

        for _, game in season_tourney.iterrows():
            w_id = game["WTeamID"]
            l_id = game["LTeamID"]

            # Skip if we don't have stats for either team
            if w_id not in team_feats or l_id not in team_feats:
                continue

            w_seed = seed_map.get(w_id)
            l_seed = seed_map.get(l_id)

            # Build features (winner as team1, loser as team2) → label = 1
            feats_w1 = build_matchup_features(
                team_feats[w_id], team_feats[l_id], w_seed, l_seed
            )
            feats_w1["label"] = 1
            feats_w1["season"] = season
            rows.append(feats_w1)

            # Mirror: loser as team1, winner as team2 → label = 0
            feats_l1 = build_matchup_features(
                team_feats[l_id], team_feats[w_id], l_seed, w_seed
            )
            feats_l1["label"] = 0
            feats_l1["season"] = season
            rows.append(feats_l1)

    df = pd.DataFrame(rows)
    return df


def get_feature_names(df):
    """Get the feature column names from a training dataframe."""
    return [c for c in df.columns if c not in ("label", "season")]


