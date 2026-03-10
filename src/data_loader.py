"""
Data loading utilities for the CMU March Madness ML Competition.
Loads NCAA basketball data in Kaggle format (sample or real Kaggle data).

Data sources used
-----------------
Kaggle competition files (data/raw/ after running scripts/download_data.py):
  MRegularSeasonDetailedResults.csv / WRegularSeasonDetailedResults.csv
  MNCAATourneyDetailedResults.csv   / WNCAATourneyDetailedResults.csv
  MNCAATourneySeeds.csv             / WNCAATourneySeeds.csv
  MMasseyOrdinals.csv               (men's only – 40+ rating systems)
  MTeamCoaches.csv                  (men's only – head-coach tenure)
  MTeamConferences.csv / WTeamConferences.csv
  MConferenceTourneyGames.csv / WConferenceTourneyGames.csv
  MGameCities.csv / WGameCities.csv

External data (see scripts/fetch_external_data.py for download instructions):
  data/external/barttorvik_{season}.csv – T-Rank adjusted efficiency (free)
  data/external/net_rankings_{season}.csv – NCAA NET rankings (free)
  data/external/kenpom_{season}.csv – KenPom ratings (subscription ~$20/yr)
  data/external/player_stats_{season}.csv – Player-level stats (PER/TS%/Usg)
  data/external/recruiting_{season}.csv – 247Sports recruiting composite
  data/external/draft_{season}.csv – NBA Draft prospect rankings
"""

import os
import pandas as pd
import numpy as np

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "sample")
KAGGLE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
EXTERNAL_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "external")

# Default value used when a team has no tournament history (midpoint of seeds 1-16)
_DEFAULT_AVG_SEED = 8.5


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


def _load_or_empty(filename, data_dir, fallbacks, columns):
    """Load a CSV or return an empty DataFrame with given columns on failure."""
    try:
        return _load(filename, data_dir, fallbacks)
    except FileNotFoundError:
        return pd.DataFrame(columns=columns)


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


def load_coaches(gender="M", data_dir=None):
    """Load head-coach roster from MTeamCoaches.csv (men only in Kaggle data).

    Columns: Season, TeamID, FirstDayNum, LastDayNum, CoachName.
    Returns an empty DataFrame when the file is not available.
    """
    return _load_or_empty(
        "MTeamCoaches.csv",
        data_dir,
        fallbacks=[f"{gender}TeamCoaches.csv"],
        columns=["Season", "TeamID", "FirstDayNum", "LastDayNum", "CoachName"],
    )


def load_conference_affiliations(gender="M", data_dir=None):
    """Load team conference affiliations (MTeamConferences.csv / WTeamConferences.csv).

    Columns: Season, TeamID, ConfAbbrev.
    Returns an empty DataFrame when the file is not available.
    """
    return _load_or_empty(
        f"{gender}TeamConferences.csv",
        data_dir,
        fallbacks=[],
        columns=["Season", "TeamID", "ConfAbbrev"],
    )


def load_conference_tourney_games(gender="M", data_dir=None):
    """Load conference tournament game list (MConferenceTourneyGames.csv / W…).

    Columns: ConfAbbrev, Season, DayNum, WTeamID, LTeamID.
    Returns an empty DataFrame when the file is not available.
    """
    return _load_or_empty(
        f"{gender}ConferenceTourneyGames.csv",
        data_dir,
        fallbacks=[],
        columns=["ConfAbbrev", "Season", "DayNum", "WTeamID", "LTeamID"],
    )


def load_game_cities(gender="M", data_dir=None):
    """Load per-game city information (MGameCities.csv / WGameCities.csv).

    Columns: Season, DayNum, WTeamID, LTeamID, CRType, CityID.
    Returns an empty DataFrame when the file is not available.
    """
    return _load_or_empty(
        f"{gender}GameCities.csv",
        data_dir,
        fallbacks=[],
        columns=["Season", "DayNum", "WTeamID", "LTeamID", "CRType", "CityID"],
    )


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


# ---------------------------------------------------------------------------
# New feature computation helpers
# ---------------------------------------------------------------------------

def compute_strength_of_schedule(regular_df, season_stats_df):
    """Compute Strength of Schedule (SoS) for each team per season.

    SoS = average win percentage of opponents faced during the regular season.
    This is a recognised predictor of tournament success: teams from brutal
    schedules are better prepared than their record alone suggests.

    Returns
    -------
    pd.DataFrame
        Columns: Season, TeamID, SoS, SoS_Wins (avg opp win pct for wins only).
    """
    if regular_df.empty or season_stats_df.empty:
        return pd.DataFrame(columns=["Season", "TeamID", "SoS", "SoS_Wins"])

    # Build quick lookup: (Season, TeamID) -> WinPct
    win_pct_map = {
        (int(r.Season), int(r.TeamID)): float(r.WinPct)
        for r in season_stats_df[["Season", "TeamID", "WinPct"]].itertuples()
    }

    rows = []
    for _, row in regular_df.iterrows():
        season = int(row["Season"])
        w_id = int(row["WTeamID"])
        l_id = int(row["LTeamID"])
        w_opp_pct = win_pct_map.get((season, l_id), 0.5)
        l_opp_pct = win_pct_map.get((season, w_id), 0.5)
        rows.append((season, w_id, w_opp_pct, 1))
        rows.append((season, l_id, l_opp_pct, 0))

    tmp = pd.DataFrame(rows, columns=["Season", "TeamID", "OppWinPct", "IsWin"])
    sos_all = tmp.groupby(["Season", "TeamID"])["OppWinPct"].mean().reset_index(name="SoS")
    sos_wins = (
        tmp[tmp["IsWin"] == 1]
        .groupby(["Season", "TeamID"])["OppWinPct"]
        .mean()
        .reset_index(name="SoS_Wins")
    )
    sos = sos_all.merge(sos_wins, on=["Season", "TeamID"], how="left")
    sos["SoS_Wins"] = sos["SoS_Wins"].fillna(sos["SoS"])
    return sos


def compute_recent_form(regular_df, last_n_days=28):
    """Compute recent-form statistics for the tail end of the regular season.

    Only the last ``last_n_days`` DayNums of each season are included.
    This captures hot/cold streaks entering the tournament.

    Returns
    -------
    pd.DataFrame
        Columns: Season, TeamID, RecentWinPct, RecentPointDiff, RecentGames.
    """
    if regular_df.empty:
        return pd.DataFrame(
            columns=["Season", "TeamID", "RecentWinPct", "RecentPointDiff", "RecentGames"]
        )

    records = []
    for season, sdf in regular_df.groupby("Season"):
        max_day = sdf["DayNum"].max()
        cutoff = max_day - last_n_days
        recent = sdf[sdf["DayNum"] >= cutoff]

        team_stats: dict = {}

        def _update(tid, scored, allowed, won):
            if tid not in team_stats:
                team_stats[tid] = {"wins": 0, "games": 0, "pf": 0, "pa": 0}
            s = team_stats[tid]
            s["wins"] += int(won)
            s["games"] += 1
            s["pf"] += scored
            s["pa"] += allowed

        for _, row in recent.iterrows():
            _update(int(row["WTeamID"]), row["WScore"], row["LScore"], True)
            _update(int(row["LTeamID"]), row["LScore"], row["WScore"], False)

        for tid, s in team_stats.items():
            g = max(s["games"], 1)
            records.append({
                "Season": season,
                "TeamID": tid,
                "RecentWinPct": s["wins"] / g,
                "RecentPointDiff": (s["pf"] - s["pa"]) / g,
                "RecentGames": s["games"],
            })

    return pd.DataFrame(records) if records else pd.DataFrame(
        columns=["Season", "TeamID", "RecentWinPct", "RecentPointDiff", "RecentGames"]
    )


def compute_tournament_history(tourney_df, seeds_df):
    """Compute cumulative prior-season tournament statistics per team.

    For each (Season S, TeamID T), we aggregate tournament data from all
    seasons *before* S so there is zero leakage into the model.

    Returns
    -------
    pd.DataFrame
        Columns: Season, TeamID, TourneyApps, TourneyWins, TourneyWinRate,
                 AvgTourneySeed, BestTourneyRound.
    """
    if tourney_df.empty or seeds_df.empty:
        return pd.DataFrame(columns=[
            "Season", "TeamID", "TourneyApps", "TourneyWins",
            "TourneyWinRate", "AvgTourneySeed", "BestTourneyRound",
        ])

    # Wins per (Season, TeamID)
    wins_per_season = (
        tourney_df.groupby(["Season", "WTeamID"])
        .size().reset_index(name="wins")
        .rename(columns={"WTeamID": "TeamID"})
    )

    # Seed per (Season, TeamID) – take minimum seed number per appearance
    seeds_work = seeds_df[["Season", "TeamID", "Seed"]].copy()
    seeds_work["seed_num"] = seeds_work["Seed"].apply(get_seed_number)
    appearances = (
        seeds_work.groupby(["Season", "TeamID"])
        .agg(seed_num=("seed_num", "min"))
        .reset_index()
    )
    appearances = appearances.merge(wins_per_season, on=["Season", "TeamID"], how="left")
    appearances["wins"] = appearances["wins"].fillna(0).astype(int)

    # Build lookup: (TeamID, Season) -> {wins, seed_num}
    lookup: dict = {}
    for row in appearances.itertuples(index=False):
        lookup[(row.TeamID, row.Season)] = {"wins": row.wins, "seed": row.seed_num}

    all_teams = sorted(seeds_df["TeamID"].unique())
    all_seasons = sorted(seeds_df["Season"].unique())

    records = []
    for tid in all_teams:
        cum_apps = 0
        cum_wins = 0
        seed_list: list = []

        for season in all_seasons:
            # Record BEFORE updating with this season (no leakage)
            total_games = cum_wins + cum_apps  # each appearance = 1 loss
            records.append({
                "Season": season,
                "TeamID": tid,
                "TourneyApps": cum_apps,
                "TourneyWins": cum_wins,
                "TourneyWinRate": cum_wins / max(total_games, 1) if total_games > 0 else 0.0,
                "AvgTourneySeed": float(np.mean(seed_list)) if seed_list else _DEFAULT_AVG_SEED,
                "BestTourneyRound": cum_wins if cum_apps > 0 else 0,
            })
            # Now update with current season
            if (tid, season) in lookup:
                d = lookup[(tid, season)]
                cum_apps += 1
                cum_wins += d["wins"]
                seed_list.append(d["seed"])

    return pd.DataFrame(records)


def compute_coaching_features(coaches_df, seeds_df):
    """Compute coaching-experience features per (Season, TeamID).

    For each team, we identify the head coach at the end of the season
    (last DayNum >= 132) and calculate:
    - CoachSeasons: number of seasons this coach has been at this school.
    - CoachTourneyApps: number of prior NCAA tournaments this coach has led
      *any* team to.

    Returns an empty DataFrame when coaches_df is empty.
    """
    if coaches_df.empty:
        return pd.DataFrame(
            columns=["Season", "TeamID", "CoachSeasons", "CoachTourneyApps"]
        )

    # End-of-season coach: latest active coach at DayNum >= 132
    end = (
        coaches_df[coaches_df["LastDayNum"] >= 132]
        .sort_values("FirstDayNum")
        .groupby(["Season", "TeamID"])
        .last()
        .reset_index()
    )[["Season", "TeamID", "CoachName"]]

    # Build (coach, team) → sorted list of seasons
    coach_team_seasons: dict = {}
    for row in end.itertuples(index=False):
        key = (row.CoachName, row.TeamID)
        coach_team_seasons.setdefault(key, []).append(row.Season)

    # Tournament appearances per coach (from seeds_df)
    # (coach, season) → appeared in tourney?
    tourney_seasons: set = set(zip(seeds_df["Season"], seeds_df["TeamID"]))
    # Map (season, TeamID) → coach
    coach_lookup = {(r.Season, r.TeamID): r.CoachName for r in end.itertuples(index=False)}

    records = []
    for row in end.itertuples(index=False):
        season = row.Season
        tid = row.TeamID
        coach = row.CoachName

        # Seasons this coach has been at this school up to and including now
        key = (coach, tid)
        seasons_at_school = sorted(s for s in coach_team_seasons.get(key, []) if s <= season)
        coach_seasons = len(seasons_at_school)

        # Prior tourney appearances by this coach (any team, any school)
        prior_apps = sum(
            1
            for (s, t), c in coach_lookup.items()
            if c == coach and s < season and (s, t) in tourney_seasons
        )

        records.append({
            "Season": season,
            "TeamID": tid,
            "CoachSeasons": coach_seasons,
            "CoachTourneyApps": prior_apps,
        })

    return pd.DataFrame(records) if records else pd.DataFrame(
        columns=["Season", "TeamID", "CoachSeasons", "CoachTourneyApps"]
    )


def compute_conference_features(conf_df, tourney_df, seeds_df):
    """Compute conference-strength features per (Season, TeamID).

    Conference strength = fraction of that conference's tournament games won
    in all *prior* seasons.  Teams in elite conferences (e.g. ACC, Big Ten)
    tend to be battle-hardened entering the tournament.

    Returns
    -------
    pd.DataFrame
        Columns: Season, TeamID, ConfStrength, ConfTourneyApps.
    """
    if conf_df.empty or tourney_df.empty or seeds_df.empty:
        return pd.DataFrame(
            columns=["Season", "TeamID", "ConfStrength", "ConfTourneyApps"]
        )

    # Build (Season, TeamID) -> ConfAbbrev
    conf_map = {
        (int(r.Season), int(r.TeamID)): r.ConfAbbrev
        for r in conf_df.itertuples(index=False)
    }

    # Tournament wins/losses per (Season, ConfAbbrev)
    wins_records = []
    for _, row in tourney_df.iterrows():
        season = int(row["Season"])
        w_conf = conf_map.get((season, int(row["WTeamID"])))
        l_conf = conf_map.get((season, int(row["LTeamID"])))
        if w_conf:
            wins_records.append({"Season": season, "ConfAbbrev": w_conf, "Result": 1})
        if l_conf:
            wins_records.append({"Season": season, "ConfAbbrev": l_conf, "Result": 0})

    if not wins_records:
        return pd.DataFrame(
            columns=["Season", "TeamID", "ConfStrength", "ConfTourneyApps"]
        )

    conf_games_df = pd.DataFrame(wins_records)
    conf_agg = (
        conf_games_df.groupby(["Season", "ConfAbbrev"])
        .agg(ConfWins=("Result", "sum"), ConfGames=("Result", "count"))
        .reset_index()
    )

    # Cumulative prior-season conference strength
    all_confs = conf_agg["ConfAbbrev"].unique()
    all_seasons = sorted(conf_agg["Season"].unique())
    cum_conf: dict = {c: {"wins": 0, "games": 0} for c in all_confs}

    conf_strength_map: dict = {}
    for season in all_seasons:
        for conf in all_confs:
            g = cum_conf[conf]["games"]
            w = cum_conf[conf]["wins"]
            conf_strength_map[(season, conf)] = w / g if g > 0 else 0.5

        # Update cumulative with this season
        season_rows = conf_agg[conf_agg["Season"] == season]
        for row in season_rows.itertuples(index=False):
            cum_conf[row.ConfAbbrev]["wins"] += row.ConfWins
            cum_conf[row.ConfAbbrev]["games"] += row.ConfGames

    # Tournament appearances per conference per prior season
    seed_conf = seeds_df.copy()
    seed_conf["ConfAbbrev"] = seed_conf.apply(
        lambda r: conf_map.get((int(r["Season"]), int(r["TeamID"]))), axis=1
    )
    conf_apps = (
        seed_conf.groupby(["Season", "ConfAbbrev"])
        .size().reset_index(name="Apps")
    )
    # cumulative apps per conf
    cum_apps: dict = {c: 0 for c in all_confs}
    conf_apps_map: dict = {}
    for season in all_seasons:
        for conf in all_confs:
            conf_apps_map[(season, conf)] = cum_apps.get(conf, 0)
        season_rows = conf_apps[conf_apps["Season"] == season]
        for row in season_rows.itertuples(index=False):
            if row.ConfAbbrev:
                cum_apps[row.ConfAbbrev] = cum_apps.get(row.ConfAbbrev, 0) + row.Apps

    # Assign to teams
    records = []
    for row in conf_df.itertuples(index=False):
        season = int(row.Season)
        tid = int(row.TeamID)
        conf = row.ConfAbbrev
        records.append({
            "Season": season,
            "TeamID": tid,
            "ConfStrength": conf_strength_map.get((season, conf), 0.5),
            "ConfTourneyApps": conf_apps_map.get((season, conf), 0),
        })

    return pd.DataFrame(records) if records else pd.DataFrame(
        columns=["Season", "TeamID", "ConfStrength", "ConfTourneyApps"]
    )


def compute_conference_tourney_form(regular_df, conf_tourney_games_df):
    """Compute per-team performance specifically in conference tournament games.

    Conference tournament results (just before Selection Sunday) are a
    strong indicator of current team form.

    Returns
    -------
    pd.DataFrame
        Columns: Season, TeamID, ConfTourneyWinPct, ConfTourneyPointDiff.
    """
    if conf_tourney_games_df.empty or regular_df.empty:
        return pd.DataFrame(
            columns=["Season", "TeamID", "ConfTourneyWinPct", "ConfTourneyPointDiff"]
        )

    # The conf tourney games file has WTeamID/LTeamID but no scores.
    # Scores are in the regular season detailed results (they share the same
    # Season/DayNum/WTeamID/LTeamID key).
    score_map = {
        (int(r.Season), int(r.DayNum), int(r.WTeamID), int(r.LTeamID)):
            (r.WScore, r.LScore)
        for r in regular_df.itertuples(index=False)
        if hasattr(r, "WScore")
    }

    records_map: dict = {}

    for row in conf_tourney_games_df.itertuples(index=False):
        season = int(row.Season)
        day = int(row.DayNum)
        w_id = int(row.WTeamID)
        l_id = int(row.LTeamID)
        scores = score_map.get((season, day, w_id, l_id))
        w_score = scores[0] if scores else 0
        l_score = scores[1] if scores else 0

        for tid, scored, allowed, won in [
            (w_id, w_score, l_score, 1),
            (l_id, l_score, w_score, 0),
        ]:
            key = (season, tid)
            if key not in records_map:
                records_map[key] = {"wins": 0, "games": 0, "pf": 0, "pa": 0}
            s = records_map[key]
            s["wins"] += won
            s["games"] += 1
            s["pf"] += scored
            s["pa"] += allowed

    rows = []
    for (season, tid), s in records_map.items():
        g = max(s["games"], 1)
        rows.append({
            "Season": season,
            "TeamID": tid,
            "ConfTourneyWinPct": s["wins"] / g,
            "ConfTourneyPointDiff": (s["pf"] - s["pa"]) / g,
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["Season", "TeamID", "ConfTourneyWinPct", "ConfTourneyPointDiff"]
    )


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
      rate, free-throw rate, rebound rates, and home/neutral/away splits.
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
                    "HomeWins": 0, "HomeGames": 0,
                    "AwayWins": 0, "AwayGames": 0,
                    "NeutralWins": 0, "NeutralGames": 0,
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
            loc_col = row.get("WLoc", "N")  # H/A/N from the winner's perspective
            _init(tid)
            s = team_stats[tid]
            s["Wins"] += win
            s["Games"] += 1
            s["PointsFor"] += score
            s["PointsAgainst"] += opp_score
            for c in box_cols:
                s[f"Sum{c}"] += row.get(f"{prefix}{c}", 0.0)
                s[f"OppSum{c}"] += row.get(f"{opp_prefix}{c}", 0.0)

            # Track home/away/neutral splits
            if prefix == "W":
                # winner's location is WLoc directly
                loc = loc_col
            else:
                # loser's location is opposite
                if loc_col == "H":
                    loc = "A"
                elif loc_col == "A":
                    loc = "H"
                else:
                    loc = "N"

            if loc == "H":
                s["HomeWins"] += win
                s["HomeGames"] += 1
            elif loc == "A":
                s["AwayWins"] += win
                s["AwayGames"] += 1
            else:
                s["NeutralWins"] += win
                s["NeutralGames"] += 1

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

            # --- Dean Oliver's 4th Factor: Rebound Rates ---
            # Offensive rebound rate: OR / (OR + opp_DR)
            own_or = s["SumOR"] / g
            opp_dr = s["OppSumDR"] / g
            or_rate = own_or / max(own_or + opp_dr, 1.0)

            # Defensive rebound rate: DR / (DR + opp_OR)
            own_dr = s["SumDR"] / g
            opp_or = s["OppSumOR"] / g
            dr_rate = own_dr / max(own_dr + opp_or, 1.0)

            # --- Home / Away / Neutral splits ---
            hg = max(s["HomeGames"], 1)
            ag = max(s["AwayGames"], 1)
            ng = max(s["NeutralGames"], 1)
            home_win_pct = s["HomeWins"] / hg if s["HomeGames"] > 0 else np.nan
            away_win_pct = s["AwayWins"] / ag if s["AwayGames"] > 0 else np.nan
            neutral_win_pct = s["NeutralWins"] / ng if s["NeutralGames"] > 0 else np.nan

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
                # Rebound rates (Dean Oliver 4th factor)
                "ORpct": or_rate,
                "DRpct": dr_rate,
                # Location splits
                "HomeWinPct": home_win_pct,
                "AwayWinPct": away_win_pct,
                "NeutralWinPct": neutral_win_pct,
            }
            for c in box_cols:
                record[f"Avg{c}"] = s[f"Sum{c}"] / g
            records.append(record)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# External data loaders (KenPom, Barttorvik, NET, player stats,
# recruiting, NBA draft)
# ---------------------------------------------------------------------------

def _load_external(filename, season):
    """Load an external CSV from data/external/, returning empty DataFrame on miss."""
    path = os.path.join(EXTERNAL_DIR, filename.format(season=season))
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


def load_kenpom(season):
    """Load KenPom efficiency ratings for a season from data/external/.

    Expected columns (after staging via scripts/fetch_external_data.py):
      TeamID, Season, KP_AdjEM, KP_AdjO, KP_AdjD, KP_AdjT, KP_Luck,
      KP_SOS_AdjEM, KP_OppO, KP_OppD, KP_NCSOS_AdjEM
    """
    df = _load_external("kenpom_{season}.csv", season)
    if df.empty or "TeamID" not in df.columns:
        return pd.DataFrame()
    kp_cols = ["TeamID", "Season"] + [c for c in df.columns
                                       if c.startswith("KP_") and c in df.columns]
    available = [c for c in kp_cols if c in df.columns]
    df = df[available].dropna(subset=["TeamID"])
    df["TeamID"] = df["TeamID"].astype(int)
    return df


def load_barttorvik(season):
    """Load Barttorvik T-Rank data for a season from data/external/.

    Expected columns (after staging):
      TeamID, Season, BT_AdjOE, BT_AdjDE, BT_Barthag, BT_AdjT,
      BT_EFG_O, BT_EFG_D, BT_OR_Pct, BT_DR_Pct, BT_FTR, BT_FTRD,
      BT_AdjNetEff
    """
    df = _load_external("barttorvik_{season}.csv", season)
    if df.empty or "TeamID" not in df.columns:
        return pd.DataFrame()
    bt_cols = ["TeamID", "Season"] + [c for c in df.columns
                                       if c.startswith("BT_") and c in df.columns]
    available = [c for c in bt_cols if c in df.columns]
    df = df[available].dropna(subset=["TeamID"])
    df["TeamID"] = df["TeamID"].astype(int)
    return df


def load_net_rankings(season):
    """Load NCAA NET rankings for a season from data/external/.

    Expected columns: TeamID, Season, NET_Rank, NET_Rating
    """
    df = _load_external("net_rankings_{season}.csv", season)
    if df.empty or "TeamID" not in df.columns:
        return pd.DataFrame()
    net_cols = [c for c in ["TeamID", "Season", "NET_Rank", "NET_Rating"] if c in df.columns]
    df = df[net_cols].dropna(subset=["TeamID"])
    df["TeamID"] = df["TeamID"].astype(int)
    return df


def load_player_stats(season):
    """Load player-level statistics for a season from data/external/.

    Expected columns: Season, Player, Team, TeamID, Min%, PRPG!, Usg, TS
    """
    df = _load_external("player_stats_{season}.csv", season)
    if df.empty or "TeamID" not in df.columns:
        return pd.DataFrame()
    df = df.dropna(subset=["TeamID"])
    df["TeamID"] = df["TeamID"].astype(int)
    return df


def load_recruiting(season):
    """Load recruiting composite rankings for a season from data/external/.

    Expected columns: Season, TeamName, TeamID, Composite, NumCommits, Rank
    """
    df = _load_external("recruiting_{season}.csv", season)
    if df.empty or "TeamID" not in df.columns:
        return pd.DataFrame()
    rec_cols = [c for c in ["TeamID", "Season", "Composite", "NumCommits", "Rank"]
                if c in df.columns]
    df = df[rec_cols].dropna(subset=["TeamID"])
    df["TeamID"] = df["TeamID"].astype(int)
    return df.rename(columns={"Composite": "REC_Composite",
                               "NumCommits": "REC_NumCommits",
                               "Rank": "REC_Rank"})


def load_draft_prospects(season):
    """Load NBA Draft prospect data for a season from data/external/.

    Expected columns: Season, PlayerName, School, DraftRound, DraftPick, TeamID
    """
    df = _load_external("draft_{season}.csv", season)
    if df.empty or "TeamID" not in df.columns:
        return pd.DataFrame()
    df = df.dropna(subset=["TeamID"])
    df["TeamID"] = df["TeamID"].astype(int)
    return df


# ---------------------------------------------------------------------------
# Feature computation helpers for external data
# ---------------------------------------------------------------------------

def compute_external_features(season_stats_df, seasons):
    """Merge all available external features into season_stats_df.

    For each season present in ``seasons``, loads the matching external
    CSV files and merges them on (Season, TeamID).  Only seasons that have
    corresponding files in data/external/ are enriched; others are left
    with NaN for the new columns (which are filled to 0 later by
    build_matchup_features).

    Parameters
    ----------
    season_stats_df : pd.DataFrame
        Per-team per-season features from ``compute_season_stats()``.
    seasons : iterable of int
        All seasons present in the training data.

    Returns
    -------
    pd.DataFrame
        ``season_stats_df`` with new external-feature columns appended.
    """
    # Collect external rows across all seasons
    kp_rows, bt_rows, net_rows, rec_rows = [], [], [], []
    ps_rows, draft_rows = [], []

    for season in seasons:
        kp = load_kenpom(season)
        if not kp.empty:
            kp_rows.append(kp)

        bt = load_barttorvik(season)
        if not bt.empty:
            bt_rows.append(bt)

        net = load_net_rankings(season)
        if not net.empty:
            net_rows.append(net)

        rec = load_recruiting(season)
        if not rec.empty:
            rec_rows.append(rec)

        ps = load_player_stats(season)
        if not ps.empty:
            ps_rows.append(ps)

        draft = load_draft_prospects(season)
        if not draft.empty:
            draft_rows.append(draft)

    # --- KenPom ---
    if kp_rows:
        kp_all = pd.concat(kp_rows, ignore_index=True)
        kp_all = kp_all.drop_duplicates(subset=["Season", "TeamID"])
        season_stats_df = season_stats_df.merge(kp_all, on=["Season", "TeamID"], how="left")

    # --- Barttorvik ---
    if bt_rows:
        bt_all = pd.concat(bt_rows, ignore_index=True)
        bt_all = bt_all.drop_duplicates(subset=["Season", "TeamID"])
        season_stats_df = season_stats_df.merge(bt_all, on=["Season", "TeamID"], how="left")

    # --- NET ---
    if net_rows:
        net_all = pd.concat(net_rows, ignore_index=True)
        net_all = net_all.drop_duplicates(subset=["Season", "TeamID"])
        season_stats_df = season_stats_df.merge(net_all, on=["Season", "TeamID"], how="left")

    # --- Recruiting ---
    if rec_rows:
        rec_all = pd.concat(rec_rows, ignore_index=True)
        rec_all = rec_all.drop_duplicates(subset=["Season", "TeamID"])
        season_stats_df = season_stats_df.merge(rec_all, on=["Season", "TeamID"], how="left")

    # --- Player stats (aggregate per team: top player PRPG!, avg TS%, star count) ---
    if ps_rows:
        ps_all = pd.concat(ps_rows, ignore_index=True)
        ps_all = ps_all.rename(columns={"PRPG!": "PRPG", "Min%": "MinPct", "TS": "TS_Pct"})
        for col in ["PRPG", "MinPct", "Usg", "TS_Pct"]:
            if col in ps_all.columns:
                ps_all[col] = pd.to_numeric(ps_all[col], errors="coerce")
        team_ps = ps_all.groupby(["Season", "TeamID"]).agg(
            PS_TopPRPG=("PRPG", "max"),
            PS_AvgTS=("TS_Pct", "mean"),
            PS_TopUsg=("Usg", "max"),
            PS_StarCount=("PRPG", lambda x: (x >= 5.0).sum()),
        ).reset_index()
        season_stats_df = season_stats_df.merge(team_ps, on=["Season", "TeamID"], how="left")

    # --- NBA Draft prospects (per team: top pick number, total picks) ---
    if draft_rows:
        draft_all = pd.concat(draft_rows, ignore_index=True)
        for col in ["DraftPick", "DraftRound"]:
            if col in draft_all.columns:
                draft_all[col] = pd.to_numeric(draft_all[col], errors="coerce")
        team_draft = draft_all.groupby(["Season", "TeamID"]).agg(
            DRAFT_TopPick=("DraftPick", "min"),
            DRAFT_NumPicks=("DraftPick", "count"),
            DRAFT_NumRound1=("DraftRound", lambda x: (x == 1).sum()),
        ).reset_index()
        season_stats_df = season_stats_df.merge(team_draft, on=["Season", "TeamID"], how="left")

    return season_stats_df


def load_all_data(gender="M", data_dir=None):
    """Load and return all relevant datasets for a gender.

    Returns a dict with keys:
      teams, regular, tourney, seeds, rankings, season_stats,
      rankings_agg  (pre-aggregated rankings lookup dict for fast access)

    ``season_stats`` is enriched with the following additional columns when
    the corresponding data files are available:
      SoS, SoS_Wins       – Strength of Schedule
      RecentWinPct, RecentPointDiff, RecentGames  – recent form (last 28 days)
      TourneyApps, TourneyWins, TourneyWinRate,
        AvgTourneySeed, BestTourneyRound           – tournament history
      CoachSeasons, CoachTourneyApps               – coaching tenure
      ConfStrength, ConfTourneyApps                – conference strength
      ConfTourneyWinPct, ConfTourneyPointDiff       – conf tourney form
      ORpct, DRpct                                 – rebound rates (4th factor)
      HomeWinPct, AwayWinPct, NeutralWinPct        – location splits

    External enrichment (when data/external/ files are present):
      KP_AdjEM, KP_AdjO, KP_AdjD, KP_AdjT, KP_Luck,
        KP_SOS_AdjEM, KP_OppO, KP_OppD, KP_NCSOS_AdjEM  – KenPom
      BT_AdjOE, BT_AdjDE, BT_Barthag, BT_AdjT,
        BT_EFG_O, BT_EFG_D, BT_OR_Pct, BT_DR_Pct,
        BT_FTR, BT_FTRD, BT_AdjNetEff                    – Barttorvik
      NET_Rank, NET_Rating                               – NCAA NET
      REC_Composite, REC_NumCommits, REC_Rank            – recruiting
      PS_TopPRPG, PS_AvgTS, PS_TopUsg, PS_StarCount      – player stats
      DRAFT_TopPick, DRAFT_NumPicks, DRAFT_NumRound1     – NBA Draft
    """
    teams = load_teams(gender, data_dir)
    regular = load_regular_season(gender, data_dir)
    tourney = load_tourney_results(gender, data_dir)
    seeds = load_seeds(gender, data_dir)
    rankings = load_rankings(gender, data_dir)

    season_stats = compute_season_stats(regular)
    rankings_agg = preaggregate_rankings(rankings)

    # ------------------------------------------------------------------
    # Enrich season_stats with new features (all merged by Season+TeamID)
    # ------------------------------------------------------------------

    # 1. Strength of Schedule
    sos_df = compute_strength_of_schedule(regular, season_stats)
    if not sos_df.empty:
        season_stats = season_stats.merge(sos_df, on=["Season", "TeamID"], how="left")

    # 2. Recent form
    recent_df = compute_recent_form(regular)
    if not recent_df.empty:
        season_stats = season_stats.merge(recent_df, on=["Season", "TeamID"], how="left")

    # 3. Tournament history (cumulative, no leakage)
    th_df = compute_tournament_history(tourney, seeds)
    if not th_df.empty:
        season_stats = season_stats.merge(th_df, on=["Season", "TeamID"], how="left")

    # 4. Coaching features (men only; graceful no-op for women)
    coaches_df = load_coaches(gender, data_dir)
    coach_feats = compute_coaching_features(coaches_df, seeds)
    if not coach_feats.empty:
        season_stats = season_stats.merge(coach_feats, on=["Season", "TeamID"], how="left")

    # 5. Conference strength
    conf_df = load_conference_affiliations(gender, data_dir)
    conf_feats = compute_conference_features(conf_df, tourney, seeds)
    if not conf_feats.empty:
        season_stats = season_stats.merge(conf_feats, on=["Season", "TeamID"], how="left")

    # 6. Conference tournament form
    conf_tourney_df = load_conference_tourney_games(gender, data_dir)
    conf_tourney_form = compute_conference_tourney_form(regular, conf_tourney_df)
    if not conf_tourney_form.empty:
        season_stats = season_stats.merge(conf_tourney_form, on=["Season", "TeamID"], how="left")

    # 7. External data enrichment (KenPom, Barttorvik, NET, player stats,
    #    recruiting, NBA Draft) – only for men's tournament (external files
    #    currently cover men's D1 only).  Skipped gracefully if files are absent.
    if gender == "M":
        seasons = season_stats["Season"].unique()
        season_stats = compute_external_features(season_stats, seasons)
        ext_cols = [c for c in season_stats.columns
                    if c.startswith(("KP_", "BT_", "NET_", "REC_", "PS_", "DRAFT_"))]
        if ext_cols:
            print(f"  External features loaded: {ext_cols}")

    return {
        "teams": teams,
        "regular": regular,
        "tourney": tourney,
        "seeds": seeds,
        "rankings": rankings,
        "rankings_agg": rankings_agg,
        "season_stats": season_stats,
    }
