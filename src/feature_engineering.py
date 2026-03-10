"""
Feature engineering for the CMU March Madness ML Competition.
Creates matchup-level features by diffing team season statistics.

Features fall into several groups
----------------------------------
1. Basic box-score differentials (win %, scoring, shooting, rebounding)
2. Advanced efficiency metrics (Pythagorean, pace, OffEff/DefEff/NetEff,
   eFG%, TS%, FT rate, turnover rate)
3. Rankings (average rank, best rank across 40+ Massey systems)
4. Dean Oliver's 4th factor: rebound rates (ORpct, DRpct)
5. Strength of Schedule (SoS, SoS_Wins)
6. Recent form in last ~28 regular-season days (RecentWinPct, RecentPointDiff)
7. Tournament history (TourneyApps, TourneyWins, TourneyWinRate, AvgTourneySeed)
8. Coaching tenure (CoachSeasons, CoachTourneyApps)
9. Conference strength (ConfStrength, ConfTourneyApps)
10. Conference tournament form (ConfTourneyWinPct, ConfTourneyPointDiff)
11. Home/away/neutral splits (HomeWinPct, AwayWinPct, NeutralWinPct)
12. Seed features (SeedDiff, HigherSeed, SeedSum, HistoricUpsetProb)
13. Interaction features (SeedDiff × NetEff, SeedDiff × SoS, etc.)
14. KenPom: AdjEM, AdjO, AdjD, AdjT, Luck, SOS metrics
15. Barttorvik: AdjOE, AdjDE, Barthag, Tempo, shooting/rebounding rates
16. NCAA NET ranking
17. Recruiting composite score
18. Player stats: top PRPG!, avg TS%, usage, star-player count
19. NBA Draft prospects: top pick number, round-1 picks
"""

import numpy as np
import pandas as pd
from src.data_loader import get_seed_number


FEATURE_COLS = None  # set after first call to build_features

_EMPTY_RANK = {"AvgRank": np.nan, "BestRank": np.nan, "NumRankSystems": 0}

# ---------------------------------------------------------------------------
# Historical upset probabilities by seed matchup (lower seed = favoured team).
# Computed from tournament game results in MNCAATourneyCompactResults.csv
# across 1985-2024 (39 seasons, ~2,340 Round-1 through Round-6 games).
# For seeds not present (Round 2+), we fall back to 0.5 (no prior info).
# ---------------------------------------------------------------------------
# fmt: off
_SEED_WIN_PROB: dict = {
    (1, 16): 0.993, (16, 1): 0.007,
    (2, 15): 0.941, (15, 2): 0.059,
    (3, 14): 0.853, (14, 3): 0.147,
    (4, 13): 0.793, (13, 4): 0.207,
    (5, 12): 0.647, (12, 5): 0.353,
    (6, 11): 0.621, (11, 6): 0.379,
    (7, 10): 0.607, (10, 7): 0.393,
    (8,  9): 0.506, ( 9, 8): 0.494,
}
# fmt: on


def _historic_win_prob(s1: int, s2: int) -> float:
    """Return historical seed-matchup win probability for seed s1 vs s2."""
    return _SEED_WIN_PROB.get((s1, s2), 0.5)


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
    shooting quality, rebound rates, strength of schedule, recent form,
    tournament history, coaching tenure, conference strength, and seed info.
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

    # Advanced metrics
    adv_stat_cols = [
        "PythExpect", "Pace", "OffEff", "DefEff", "NetEff",
        "eFGPct", "TSPct", "FTPct", "FTRate", "TORatePerPoss",
        # 4th factor rebound rates
        "ORpct", "DRpct",
        # Location splits
        "HomeWinPct", "AwayWinPct", "NeutralWinPct",
    ]

    # New enriched features (all default gracefully to 0 if missing)
    extra_stat_cols = [
        # Strength of schedule
        "SoS", "SoS_Wins",
        # Recent form
        "RecentWinPct", "RecentPointDiff",
        # Tournament history
        "TourneyApps", "TourneyWins", "TourneyWinRate", "AvgTourneySeed",
        # Coaching
        "CoachSeasons", "CoachTourneyApps",
        # Conference strength
        "ConfStrength", "ConfTourneyApps",
        # Conference tournament form
        "ConfTourneyWinPct", "ConfTourneyPointDiff",
        # KenPom
        "KP_AdjEM", "KP_AdjO", "KP_AdjD", "KP_AdjT", "KP_Luck",
        "KP_SOS_AdjEM", "KP_OppO", "KP_OppD", "KP_NCSOS_AdjEM",
        # Barttorvik
        "BT_AdjOE", "BT_AdjDE", "BT_Barthag", "BT_AdjT",
        "BT_EFG_O", "BT_EFG_D", "BT_OR_Pct", "BT_DR_Pct",
        "BT_FTR", "BT_FTRD", "BT_AdjNetEff",
        # NCAA NET
        "NET_Rank", "NET_Rating",
        # Recruiting
        "REC_Composite", "REC_NumCommits", "REC_Rank",
        # Player stats
        "PS_TopPRPG", "PS_AvgTS", "PS_TopUsg", "PS_StarCount",
        # NBA Draft
        "DRAFT_TopPick", "DRAFT_NumPicks", "DRAFT_NumRound1",
    ]

    feat = {}

    for col in basic_stat_cols + adv_stat_cols + extra_stat_cols:
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

    # SoS-adjusted efficiency: NetEff weighted by SoS
    net1 = float(team1_feats.get("NetEff", 0) or 0)
    net2 = float(team2_feats.get("NetEff", 0) or 0)
    sos1 = float(team1_feats.get("SoS", 0.5) or 0.5)
    sos2 = float(team2_feats.get("SoS", 0.5) or 0.5)
    feat["SoS_AdjNetEff1"] = net1 * sos1
    feat["SoS_AdjNetEff2"] = net2 * sos2
    feat["diff_SoS_AdjNetEff"] = feat["SoS_AdjNetEff1"] - feat["SoS_AdjNetEff2"]

    # ------------------------------------------------------------------
    # Seed features
    # ------------------------------------------------------------------
    if seed1 is not None and seed2 is not None:
        s1 = get_seed_number(seed1) if isinstance(seed1, str) else int(seed1)
        s2 = get_seed_number(seed2) if isinstance(seed2, str) else int(seed2)
        feat["SeedDiff"] = s1 - s2
        feat["HigherSeed"] = 1 if s1 < s2 else 0
        feat["SeedSum"] = s1 + s2
        feat["HistoricUpsetProb"] = _historic_win_prob(s1, s2)
        feat["SeedProduct"] = s1 * s2
    else:
        feat["SeedDiff"] = 0
        feat["HigherSeed"] = 0
        feat["SeedSum"] = 16
        feat["HistoricUpsetProb"] = 0.5
        feat["SeedProduct"] = 64

    # ------------------------------------------------------------------
    # Interaction features (capture non-linear relationships)
    # ------------------------------------------------------------------
    seed_diff = feat["SeedDiff"]
    net_eff_diff = feat["diff_NetEff"]
    sos_diff = feat["diff_SoS"]

    # Seed × efficiency: a high-seeded team with great efficiency is very dangerous
    feat["interact_SeedDiff_NetEff"] = seed_diff * net_eff_diff
    # Seed × SoS: upset likelihood adjusts when the underdog played a tough schedule
    feat["interact_SeedDiff_SoS"] = seed_diff * sos_diff
    # Recent form × tournament experience: hot team with tourney know-how
    recent_diff = feat["diff_RecentWinPct"]
    tourney_diff = feat["diff_TourneyWinRate"]
    feat["interact_RecentForm_TourneyExp"] = recent_diff * tourney_diff
    # NetEff × SoS: strength-of-schedule-adjusted net efficiency signal
    feat["interact_NetEff_SoS"] = net_eff_diff * sos_diff

    # KenPom × Barttorvik consensus: when two elite systems agree, signal is stronger
    kp_em_diff = feat.get("diff_KP_AdjEM", 0.0)
    bt_net_diff = feat.get("diff_BT_AdjNetEff", 0.0)
    feat["interact_KP_BT_AdjEM"] = kp_em_diff * bt_net_diff

    # Seed × KenPom AdjEM: upset detection using gold-standard efficiency
    feat["interact_SeedDiff_KP_AdjEM"] = seed_diff * kp_em_diff

    # Star-player advantage: top PRPG! differential × seed differential
    ps_top_diff = feat.get("diff_PS_TopPRPG", 0.0)
    feat["interact_StarPlayer_SeedDiff"] = ps_top_diff * seed_diff

    # Recruiting pipeline vs. recent form: elite recruiting class that also plays well
    rec_diff = feat.get("diff_REC_Composite", 0.0)
    feat["interact_Recruiting_RecentForm"] = rec_diff * recent_diff

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


