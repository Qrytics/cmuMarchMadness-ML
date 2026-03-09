"""
Data loading utilities for the CMU March Madness ML Competition.
Loads NCAA basketball data in Kaggle format (sample or real Kaggle data).
"""

import os
import pandas as pd
import numpy as np

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "sample")
KAGGLE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def _load(filename, data_dir=None, fallbacks=None):
    """Load a CSV, preferring kaggle data if available, else sample data.

    Parameters
    ----------
    filename : str
        Primary filename to load.
    data_dir : str or None
        Optional explicit directory to search first.  When provided, **only**
        this directory is searched for the primary and fallback filenames
        (i.e. KAGGLE_DIR / SAMPLE_DIR are not consulted).  This matches the
        behaviour expected by tests that pass an explicit ``data_dir``.
    fallbacks : list of str or None
        Alternative filenames to try if the primary is not found.
    """
    all_names = [filename] + (fallbacks or [])

    if data_dir is not None:
        # Explicit data dir: only look there
        for name in all_names:
            path = os.path.join(data_dir, name)
            if os.path.exists(path):
                return pd.read_csv(path)
        raise FileNotFoundError(
            f"Could not find {filename} (or fallbacks {fallbacks}) in {data_dir}."
        )

    # No explicit dir: check KAGGLE_DIR, then SAMPLE_DIR
    for name in all_names:
        for d in [KAGGLE_DIR, SAMPLE_DIR]:
            path = os.path.join(d, name)
            if os.path.exists(path):
                return pd.read_csv(path)

    raise FileNotFoundError(
        f"Could not find {filename} (or fallbacks {fallbacks}) in "
        f"{KAGGLE_DIR} or {SAMPLE_DIR}. "
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
    # Kaggle 2026 uses MNCAATourneySeeds.csv; older samples use MSeeds.csv
    return _load(
        f"{gender}NCAATourneySeeds.csv",
        data_dir,
        fallbacks=[f"{gender}Seeds.csv"],
    )


def load_rankings(gender="M", data_dir=None):
    """Load ordinal rankings.  The Kaggle 2026 dataset ships the comprehensive
    Massey Ordinals file (MMasseyOrdinals.csv); older sample data uses a
    smaller MRankings.csv with the same schema.  Both share the columns
    Season, RankingDayNum, SystemName, TeamID, OrdinalRank.
    """
    empty = pd.DataFrame(
        columns=["Season", "RankingDayNum", "SystemName", "TeamID", "OrdinalRank"]
    )
    try:
        return _load(
            f"{gender}MasseyOrdinals.csv",
            data_dir,
            fallbacks=[f"{gender}Rankings.csv"],
        )
    except FileNotFoundError:
        return empty


def load_2026_teams(gender="M", data_dir=None):
    """Load the 2026 team list for prediction generation.

    Priority order:
    1. SampleSubmissionStage2.csv — the authoritative Kaggle source for which
       team pairs must be predicted; derives the exact team ID set.
    2. {gender}Teams2026.csv (hand-curated override if Stage2 unavailable)
    3. {gender}Teams.csv (all D1 teams as a last resort)
    """
    # Primary: derive from Kaggle Stage-2 submission file (authoritative)
    try:
        sub = _load("SampleSubmissionStage2.csv", data_dir)
        ids_split = sub["ID"].str.split("_", expand=True)
        all_ids = (
            set(ids_split[1].astype(int)) | set(ids_split[2].astype(int))
        )
        # Men's IDs start with 1xxx, Women's with 3xxx
        if gender == "M":
            team_ids = sorted(t for t in all_ids if t < 3000)
        else:
            team_ids = sorted(t for t in all_ids if t >= 3000)
        if team_ids:
            return pd.DataFrame({"TeamID": team_ids})
    except FileNotFoundError:
        pass

    # Fallback: hand-curated team list
    try:
        return _load(f"{gender}Teams2026.csv", data_dir)
    except FileNotFoundError:
        pass

    # Final fallback: full team list
    return load_teams(gender, data_dir)


def get_seed_number(seed_str):
    """Extract numeric seed from seed string like 'W01', 'X11a'."""
    import re
    m = re.search(r"\d+", str(seed_str))
    return int(m.group()) if m else 16


def preaggregate_rankings(rankings_df, day_limit=133):
    """
    Pre-aggregate ordinal rankings for all (Season, TeamID) pairs near the
    end of the regular season.

    Returns a dict mapping (season, team_id) → {"AvgRank": float, "BestRank": float,
    "NumRankSystems": int}.  This is O(n) once and lets ``get_team_ranking_features``
    do O(1) lookups instead of re-filtering the 5M-row Massey file each time.
    """
    if rankings_df is None or rankings_df.empty:
        return {}

    sub = rankings_df[rankings_df["RankingDayNum"] <= day_limit].copy()
    if sub.empty:
        return {}

    # For each (Season, TeamID) keep only the latest day's rankings
    latest_day = sub.groupby(["Season", "TeamID"])["RankingDayNum"].transform("max")
    sub = sub[sub["RankingDayNum"] == latest_day]

    agg = sub.groupby(["Season", "TeamID"])["OrdinalRank"].agg(
        AvgRank="mean", BestRank="min"
    ).reset_index()

    n_systems = (
        sub.groupby(["Season", "TeamID"])["SystemName"]
        .nunique()
        .reset_index(name="NumRankSystems")
    )
    agg = agg.merge(n_systems, on=["Season", "TeamID"])

    result = {}
    for row in agg.itertuples(index=False):
        result[(row.Season, row.TeamID)] = {
            "AvgRank": row.AvgRank,
            "BestRank": row.BestRank,
            "NumRankSystems": row.NumRankSystems,
        }
    return result


def compute_season_stats(season_df):
    """
    Compute per-team season averages from a detailed results dataframe.

    Returns a DataFrame with one row per (Season, TeamID) containing:
    - Basic box-score averages (scoring, shooting, rebounding, etc.)
    - Advanced metrics: Pythagorean expectation, pace, offensive/defensive
      efficiency per 100 possessions, effective FG%, true shooting %, turnover
      rate, free-throw rate, and rebound rates.
    """
    box_cols = ["FGM", "FGA", "FGM3", "FGA3", "FTM", "FTA",
                "OR", "DR", "Ast", "TO", "Stl", "Blk", "PF"]

    records = []

    for season, sdf in season_df.groupby("Season"):
        team_stats = {}

        def _init(tid):
            if tid not in team_stats:
                team_stats[tid] = {
                    "Wins": 0, "Games": 0,
                    "PointsFor": 0, "PointsAgainst": 0,
                    **{f"Sum{c}": 0.0 for c in box_cols},
                    # opponent box-score totals (for pace / efficiency)
                    **{f"OppSum{c}": 0.0 for c in box_cols},
                }

        def _accumulate(row, prefix):
            opp_prefix = "L" if prefix == "W" else "W"
            tid = row[f"{prefix}TeamID"]
            score = row[f"{prefix}Score"]
            opp_score = row[f"{opp_prefix}Score"]
            win = 1 if prefix == "W" else 0
            _init(tid)
            s = team_stats[tid]
            s["Wins"] += win
            s["Games"] += 1
            s["PointsFor"] += score
            s["PointsAgainst"] += opp_score
            for c in box_cols:
                s[f"Sum{c}"] += row.get(f"{prefix}{c}", 0.0)
                s[f"OppSum{c}"] += row.get(f"{opp_prefix}{c}", 0.0)

        for _, row in sdf.iterrows():
            _accumulate(row, "W")
            _accumulate(row, "L")

        for tid, s in team_stats.items():
            g = max(s["Games"], 1)

            pf = s["PointsFor"]
            pa = s["PointsAgainst"]

            # Pythagorean expectation (exponent 10.25 is well-calibrated for CBB)
            exp = 10.25
            pyth = (pf ** exp) / ((pf ** exp) + (pa ** exp)) if (pf + pa) > 0 else 0.5

            # Possession estimate per game: FGA - OR + TO + 0.44*FTA
            own_poss = (
                s["SumFGA"]
                - s["SumOR"]
                + s["SumTO"]
                + 0.44 * s["SumFTA"]
            ) / g
            opp_poss = (
                s["OppSumFGA"]
                - s["OppSumOR"]
                + s["OppSumTO"]
                + 0.44 * s["OppSumFTA"]
            ) / g
            # Pace = average possessions per game (team + opponent) / 2
            pace = (own_poss + opp_poss) / 2.0 if (own_poss + opp_poss) > 0 else 65.0

            # Offensive / defensive efficiency per 100 possessions
            off_eff = (pf / g) / max(own_poss, 1.0) * 100
            def_eff = (pa / g) / max(opp_poss, 1.0) * 100

            fga = max(s["SumFGA"] / g, 1.0)
            fga3 = s["SumFGA3"] / g
            fta = max(s["SumFTA"] / g, 1.0)
            fgm = s["SumFGM"] / g
            fgm3 = s["SumFGM3"] / g
            ftm = s["SumFTM"] / g

            # Effective FG% = (FGM + 0.5*FGM3) / FGA
            efg = (fgm + 0.5 * fgm3) / fga
            # True Shooting % = PTS / (2 * (FGA + 0.44*FTA))
            ts_denom = 2.0 * (fga + 0.44 * fta)
            ts_pct = (pf / g) / ts_denom if ts_denom > 0 else 0.0
            # Turnover rate (per 100 possessions)
            to_rate = (s["SumTO"] / g) / max(own_poss, 1.0) * 100
            # Free-throw rate
            ft_rate = fta / fga

            record = {
                "Season": season,
                "TeamID": tid,
                "NumGames": g,
                "WinPct": s["Wins"] / g,
                "AvgPointsFor": pf / g,
                "AvgPointsAgainst": pa / g,
                "PointDiff": (pf - pa) / g,
                # Pythagorean
                "PythExpect": pyth,
                # Pace & efficiency
                "Pace": pace,
                "OffEff": off_eff,
                "DefEff": def_eff,
                "NetEff": off_eff - def_eff,
                # Shooting
                "eFGPct": efg,
                "TSPct": ts_pct,
                "FTPct": ftm / max(fta, 1.0),
                "FTRate": ft_rate,
                # Turnover
                "TORatePerPoss": to_rate,
            }
            for c in box_cols:
                record[f"Avg{c}"] = s[f"Sum{c}"] / g
            records.append(record)

    return pd.DataFrame(records)


def load_all_data(gender="M", data_dir=None):
    """Load and return all relevant datasets for a gender.

    Returns a dict with keys:
      teams, regular, tourney, seeds, rankings, season_stats,
      rankings_agg  (pre-aggregated rankings lookup dict for fast access)
    """
    teams = load_teams(gender, data_dir)
    regular = load_regular_season(gender, data_dir)
    tourney = load_tourney_results(gender, data_dir)
    seeds = load_seeds(gender, data_dir)
    rankings = load_rankings(gender, data_dir)

    season_stats = compute_season_stats(regular)
    rankings_agg = preaggregate_rankings(rankings)

    return {
        "teams": teams,
        "regular": regular,
        "tourney": tourney,
        "seeds": seeds,
        "rankings": rankings,
        "rankings_agg": rankings_agg,
        "season_stats": season_stats,
    }
