"""
Generate synthetic NCAA basketball data in Kaggle format for training and testing.
Produces realistic data for men's and women's tournaments from 2010-2024.
"""

import numpy as np
import pandas as pd
import os
import random

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "sample")
os.makedirs(OUT_DIR, exist_ok=True)

# --- Configuration ---
SEASONS = list(range(2010, 2026))  # 2010–2025 (2026 is the current year being predicted)
N_MEN_TEAMS = 381
N_WOMEN_TEAMS = 379

MEN_TEAM_IDS = list(range(1101, 1101 + N_MEN_TEAMS))
WOMEN_TEAM_IDS = list(range(3101, 3101 + N_WOMEN_TEAMS))

# Real-ish team names (padded with generic names)
MEN_TEAM_NAMES_BASE = [
    "Duke", "Kentucky", "Kansas", "North Carolina", "UCLA", "Villanova",
    "Gonzaga", "Michigan", "Arizona", "Virginia", "Baylor", "Houston",
    "Connecticut", "Auburn", "Tennessee", "Purdue", "Texas", "Alabama",
    "Creighton", "Marquette", "Xavier", "Illinois", "Indiana", "Michigan St",
    "Ohio St", "Florida", "Wisconsin", "Missouri", "Arkansas", "Iowa St",
    "San Diego St", "Utah St", "Nevada", "Loyola Chicago", "Wichita St",
    "Memphis", "Cincinnati", "Temple", "SMU", "Tulsa", "UCF", "East Carolina",
    "South Florida", "Tulane", "Navy", "Army", "Air Force", "Colgate",
    "Belmont", "Murray St", "Morehead St", "Eastern Kentucky", "Lipscomb",
    "Jacksonville St", "Austin Peay", "Tennessee St", "Tennessee Tech",
    "Samford", "Furman", "VMI", "Wofford", "Citadel", "Chattanooga",
    "ETSU", "UNC Greensboro", "Western Carolina", "Mercer", "Gardner-Webb",
    "High Point", "Longwood", "Presbyterian", "SC Upstate", "NC A&T",
    "NC Central", "Howard", "Morgan St", "Coppin St", "UMES", "Hampton",
    "Norfolk St", "Delaware St", "Md-Eastern Shore", "Bowie St",
    "VCU", "Richmond", "George Mason", "Old Dominion", "Charlotte",
    "UNCW", "James Madison", "William & Mary", "Drexel", "Hofstra",
    "Towson", "Northeastern", "Elon", "Delaware", "Stony Brook",
    "Albany", "Hartford", "Binghamton", "Vermont", "New Hampshire",
    "Maine", "UMBC", "Maryland", "Georgetown", "St John's", "Providence",
    "Seton Hall", "Butler", "DePaul", "St Joseph's", "La Salle",
]


def _gen_team_names(base, total):
    names = base[:]
    for i in range(len(base), total):
        names.append(f"Team_{i + 1}")
    return names[:total]


# -----------------------------------------------------------------------
# 1. Teams
# -----------------------------------------------------------------------

def gen_teams(ids, names):
    return pd.DataFrame({"TeamID": ids, "TeamName": names})


def save_teams():
    m_names = _gen_team_names(MEN_TEAM_NAMES_BASE, N_MEN_TEAMS)
    w_names = [f"W{n}" if not n.startswith("W") else n for n in _gen_team_names(MEN_TEAM_NAMES_BASE, N_WOMEN_TEAMS)]

    gen_teams(MEN_TEAM_IDS, m_names).to_csv(os.path.join(OUT_DIR, "MTeams.csv"), index=False)
    gen_teams(WOMEN_TEAM_IDS, w_names).to_csv(os.path.join(OUT_DIR, "WTeams.csv"), index=False)
    print("Saved MTeams.csv and WTeams.csv")


# -----------------------------------------------------------------------
# 2. Team latent strength (used to generate realistic game results)
# -----------------------------------------------------------------------

def gen_team_strengths(team_ids, seed=42):
    """Assign each team a latent 'strength' value drawn from a log-normal distribution."""
    rng = np.random.default_rng(seed)
    strengths = rng.lognormal(mean=0.0, sigma=0.5, size=len(team_ids))
    return dict(zip(team_ids, strengths))


MEN_STRENGTHS = gen_team_strengths(MEN_TEAM_IDS, seed=42)
WOMEN_STRENGTHS = gen_team_strengths(WOMEN_TEAM_IDS, seed=43)


# -----------------------------------------------------------------------
# 3. Regular season games
# -----------------------------------------------------------------------

def _game_stats(strength_w, strength_l, rng):
    """Generate realistic game stats for a winner and loser."""
    base_score = 65 + rng.normal(0, 5)
    w_bonus = rng.normal(0, 3)
    l_bonus = rng.normal(0, 3)

    w_score = max(50, int(base_score + 5 * (strength_w - strength_l) / 2 + w_bonus))
    l_score = max(40, int(base_score - 5 * (strength_w - strength_l) / 2 + l_bonus))
    if l_score >= w_score:
        l_score = w_score - rng.integers(1, 10)

    def _stats(score, strength):
        fga = int(rng.normal(60, 6))
        fgm = max(1, int(score * 0.4 + rng.normal(0, 2)))
        fga3 = int(rng.normal(20, 4))
        fgm3 = max(0, int(fga3 * (0.3 + strength * 0.05) + rng.normal(0, 1)))
        fta = int(rng.normal(18, 4))
        ftm = max(0, int(fta * (0.7 + strength * 0.02) + rng.normal(0, 1)))
        or_ = max(0, int(rng.normal(10, 3)))
        dr = max(0, int(rng.normal(25, 4)))
        ast = max(0, int(fgm * 0.55 + rng.normal(0, 2)))
        to = max(0, int(rng.normal(13, 3)))
        stl = max(0, int(rng.normal(6, 2)))
        blk = max(0, int(rng.normal(4, 2)))
        pf = max(0, int(rng.normal(18, 3)))
        return fgm, fga, fgm3, fga3, ftm, fta, or_, dr, ast, to, stl, blk, pf

    w_stats = _stats(w_score, strength_w)
    l_stats = _stats(l_score, strength_l)
    return w_score, l_score, w_stats, l_stats


def gen_regular_season(team_ids, strengths, season, n_games_per_team=30, prefix="M", rng=None):
    if rng is None:
        rng = np.random.default_rng(season)

    records = []
    team_list = list(team_ids)
    day_range = range(10, 130)

    games_played = {t: 0 for t in team_list}

    # Generate matchups
    matchups = []
    for i, t1 in enumerate(team_list):
        opponents = rng.choice([t for t in team_list if t != t1],
                               size=min(n_games_per_team, len(team_list) - 1),
                               replace=False)
        for t2 in opponents:
            if games_played[t1] < n_games_per_team and games_played[t2] < n_games_per_team:
                matchups.append((t1, t2))
                games_played[t1] += 1
                games_played[t2] += 1

    for t1, t2 in matchups:
        s1, s2 = strengths[t1], strengths[t2]
        prob_win = s1 / (s1 + s2)
        winner = t1 if rng.random() < prob_win else t2
        loser = t2 if winner == t1 else t1
        sw, sl = strengths[winner], strengths[loser]

        day = int(rng.choice(day_range))
        loc = rng.choice(["H", "A", "N"], p=[0.4, 0.4, 0.2])
        ws, ls, wst, lst = _game_stats(sw, sl, rng)

        records.append({
            "Season": season,
            "DayNum": day,
            "WTeamID": winner,
            "WScore": ws,
            "LTeamID": loser,
            "LScore": ls,
            "WLoc": loc,
            "NumOT": 0,
            "WFGM": wst[0], "WFGA": wst[1], "WFGM3": wst[2], "WFGA3": wst[3],
            "WFTM": wst[4], "WFTA": wst[5], "WOR": wst[6], "WDR": wst[7],
            "WAst": wst[8], "WTO": wst[9], "WStl": wst[10], "WBlk": wst[11], "WPF": wst[12],
            "LFGM": lst[0], "LFGA": lst[1], "LFGM3": lst[2], "LFGA3": lst[3],
            "LFTM": lst[4], "LFTA": lst[5], "LOR": lst[6], "LDR": lst[7],
            "LAst": lst[8], "LTO": lst[9], "LStl": lst[10], "LBlk": lst[11], "LPF": lst[12],
        })

    return pd.DataFrame(records)


def save_regular_seasons():
    m_dfs, w_dfs = [], []
    for season in SEASONS:
        rng_m = np.random.default_rng(season * 100)
        rng_w = np.random.default_rng(season * 100 + 1)
        m_dfs.append(gen_regular_season(MEN_TEAM_IDS, MEN_STRENGTHS, season, rng=rng_m))
        w_dfs.append(gen_regular_season(WOMEN_TEAM_IDS, WOMEN_STRENGTHS, season, rng=rng_w))

    pd.concat(m_dfs).to_csv(os.path.join(OUT_DIR, "MRegularSeasonDetailedResults.csv"), index=False)
    pd.concat(w_dfs).to_csv(os.path.join(OUT_DIR, "WRegularSeasonDetailedResults.csv"), index=False)
    print("Saved regular season files")


# -----------------------------------------------------------------------
# 4. Tournament seeds and results
# -----------------------------------------------------------------------

SEED_SLOTS = (
    [f"W{i:02d}" for i in range(1, 17)] +
    [f"X{i:02d}" for i in range(1, 17)] +
    [f"Y{i:02d}" for i in range(1, 17)] +
    [f"Z{i:02d}" for i in range(1, 17)]
)  # 64 slots

PLAY_IN_SLOTS = ["W16a", "W16b", "X16a", "X16b"]  # 4 play-in games → 8 extra teams

ALL_SEED_SLOTS = SEED_SLOTS + PLAY_IN_SLOTS


def get_seed_number(seed_str):
    """Extract numeric seed from seed string like 'W01', 'X11a'."""
    import re
    m = re.search(r"\d+", seed_str)
    return int(m.group()) if m else 16


def gen_tournament_seeds(team_ids, strengths, season, rng=None):
    if rng is None:
        rng = np.random.default_rng(season)

    # Sort teams by strength (descending) with some noise for realism
    noisy_strengths = {t: strengths[t] * rng.lognormal(0, 0.15) for t in team_ids}
    sorted_teams = sorted(noisy_strengths.keys(), key=lambda t: noisy_strengths[t], reverse=True)

    # Take top 68 teams
    tourney_teams = sorted_teams[:68]
    rng.shuffle(tourney_teams)

    records = []
    for i, slot in enumerate(ALL_SEED_SLOTS[:68]):
        records.append({"Season": season, "Seed": slot, "TeamID": tourney_teams[i]})

    return pd.DataFrame(records), tourney_teams[:64]  # return 64 main bracket teams


def simulate_tournament(seeds_df, strengths, season, rng):
    """Simulate single-elimination tournament, return game records."""
    # Build bracket from seeds
    seed_to_team = dict(zip(seeds_df["Seed"], seeds_df["TeamID"]))

    regions = ["W", "X", "Y", "Z"]
    bracket_teams = {}
    for region in regions:
        for seed_num in range(1, 17):
            slot = f"{region}{seed_num:02d}"
            if slot in seed_to_team:
                bracket_teams[(region, seed_num)] = seed_to_team[slot]

    # Handle play-in: slot W16 gets filled by winner of W16a vs W16b etc.
    play_in_pairs = [("W", 16), ("X", 16)]
    for region, seed_num in play_in_pairs:
        slot_a = f"{region}{seed_num:02d}a"
        slot_b = f"{region}{seed_num:02d}b"
        if slot_a in seed_to_team and slot_b in seed_to_team:
            t_a = seed_to_team[slot_a]
            t_b = seed_to_team[slot_b]
            sa, sb = strengths.get(t_a, 1.0), strengths.get(t_b, 1.0)
            winner = t_a if rng.random() < sa / (sa + sb) else t_b
            bracket_teams[(region, seed_num)] = winner

    # First round matchups: 1v16, 2v15, 3v14, 4v13, 5v12, 6v11, 7v10, 8v9
    first_round_pairs = [(1, 16), (2, 15), (3, 14), (4, 13), (5, 12), (6, 11), (7, 10), (8, 9)]
    records = []
    day_start = 136  # tournament starts around day 136

    def play_game(t1, t2, day):
        s1, s2 = strengths.get(t1, 1.0), strengths.get(t2, 1.0)
        prob = s1 / (s1 + s2)
        winner = t1 if rng.random() < prob else t2
        loser = t2 if winner == t1 else t1
        sw, sl = strengths.get(winner, 1.0), strengths.get(loser, 1.0)
        ws, ls, wst, lst = _game_stats(sw * 1.5, sl * 1.5, rng)  # higher scoring in tourney

        records.append({
            "Season": season, "DayNum": day,
            "WTeamID": winner, "WScore": ws,
            "LTeamID": loser, "LScore": ls,
            "WLoc": "N", "NumOT": 0,
            "WFGM": wst[0], "WFGA": wst[1], "WFGM3": wst[2], "WFGA3": wst[3],
            "WFTM": wst[4], "WFTA": wst[5], "WOR": wst[6], "WDR": wst[7],
            "WAst": wst[8], "WTO": wst[9], "WStl": wst[10], "WBlk": wst[11], "WPF": wst[12],
            "LFGM": lst[0], "LFGA": lst[1], "LFGM3": lst[2], "LFGA3": lst[3],
            "LFTM": lst[4], "LFTA": lst[5], "LOR": lst[6], "LDR": lst[7],
            "LAst": lst[8], "LTO": lst[9], "LStl": lst[10], "LBlk": lst[11], "LPF": lst[12],
        })
        return winner

    # Simulate 6 rounds
    current_round = {}
    for region in regions:
        region_bracket = {}
        for s1, s2 in first_round_pairs:
            t1 = bracket_teams.get((region, s1))
            t2 = bracket_teams.get((region, s2))
            if t1 and t2:
                winner = play_game(t1, t2, day_start)
                region_bracket[s1] = winner

        # Round 2
        r2_pairs = [(1, 8), (2, 7), (3, 6), (4, 5)]  # winners of (1v16) vs (8v9) etc.
        r2_winners = {}
        for s1, s2 in r2_pairs:
            t1 = region_bracket.get(s1)
            t2 = region_bracket.get(s2)
            if t1 and t2:
                winner = play_game(t1, t2, day_start + 2)
                r2_winners[s1] = winner

        # Sweet 16
        r3_winners = {}
        pairs_r3 = [(1, 4), (2, 3)]
        for s1, s2 in pairs_r3:
            t1 = r2_winners.get(s1)
            t2 = r2_winners.get(s2)
            if t1 and t2:
                winner = play_game(t1, t2, day_start + 4)
                r3_winners[s1] = winner

        # Elite 8
        t1 = r3_winners.get(1)
        t2 = r3_winners.get(2)
        if t1 and t2:
            current_round[region] = play_game(t1, t2, day_start + 6)

    # Final Four
    regions_list = list(regions)
    sf_winners = []
    for i in range(0, len(regions_list), 2):
        r1, r2 = regions_list[i], regions_list[i + 1]
        t1 = current_round.get(r1)
        t2 = current_round.get(r2)
        if t1 and t2:
            sf_winners.append(play_game(t1, t2, day_start + 10))

    # Championship
    if len(sf_winners) >= 2:
        play_game(sf_winners[0], sf_winners[1], day_start + 12)

    return pd.DataFrame(records)


def save_tournaments():
    m_seed_dfs, m_game_dfs = [], []
    w_seed_dfs, w_game_dfs = [], []

    for season in SEASONS:
        rng_m = np.random.default_rng(season * 200)
        rng_w = np.random.default_rng(season * 200 + 1)

        m_seeds, _ = gen_tournament_seeds(MEN_TEAM_IDS, MEN_STRENGTHS, season, rng=rng_m)
        m_games = simulate_tournament(m_seeds, MEN_STRENGTHS, season, rng=rng_m)
        m_seed_dfs.append(m_seeds)
        m_game_dfs.append(m_games)

        w_seeds, _ = gen_tournament_seeds(WOMEN_TEAM_IDS, WOMEN_STRENGTHS, season, rng=rng_w)
        w_games = simulate_tournament(w_seeds, WOMEN_STRENGTHS, season, rng=rng_w)
        w_seed_dfs.append(w_seeds)
        w_game_dfs.append(w_games)

    pd.concat(m_seed_dfs).to_csv(os.path.join(OUT_DIR, "MSeeds.csv"), index=False)
    pd.concat(m_game_dfs).to_csv(os.path.join(OUT_DIR, "MNCAATourneyDetailedResults.csv"), index=False)
    pd.concat(w_seed_dfs).to_csv(os.path.join(OUT_DIR, "WSeeds.csv"), index=False)
    pd.concat(w_game_dfs).to_csv(os.path.join(OUT_DIR, "WNCAATourneyDetailedResults.csv"), index=False)
    print("Saved tournament seeds and results")


# -----------------------------------------------------------------------
# 5. Rankings (Massey-style)
# -----------------------------------------------------------------------

def gen_rankings(team_ids, strengths, seasons, rng_seed=0):
    """Generate ordinal rankings for each season at key dates."""
    rng = np.random.default_rng(rng_seed)
    records = []
    systems = ["MAS", "KPI", "NET", "KenPom", "SAG"]
    ranking_days = [65, 80, 95, 110, 128, 133]  # snapshot days during season

    for season in seasons:
        # Sort by strength + noise to get ranking
        noisy = {t: strengths[t] * rng.lognormal(0, 0.2) for t in team_ids}
        sorted_teams = sorted(noisy.keys(), key=lambda t: noisy[t], reverse=True)

        for day in ranking_days:
            # Add slight noise per day to simulate evolving rankings
            day_noisy = {t: noisy[t] * rng.lognormal(0, 0.05) for t in team_ids}
            day_sorted = sorted(day_noisy.keys(), key=lambda t: day_noisy[t], reverse=True)
            rank_map = {t: i + 1 for i, t in enumerate(day_sorted)}

            for system in systems:
                sys_noisy = {t: day_noisy[t] * rng.lognormal(0, 0.1) for t in team_ids}
                sys_sorted = sorted(sys_noisy.keys(), key=lambda t: sys_noisy[t], reverse=True)
                for i, t in enumerate(sys_sorted):
                    records.append({
                        "Season": season,
                        "RankingDayNum": day,
                        "SystemName": system,
                        "TeamID": t,
                        "OrdinalRank": i + 1,
                    })

    return pd.DataFrame(records)


def save_rankings():
    m_rank = gen_rankings(MEN_TEAM_IDS, MEN_STRENGTHS, SEASONS, rng_seed=10)
    w_rank = gen_rankings(WOMEN_TEAM_IDS, WOMEN_STRENGTHS, SEASONS, rng_seed=11)

    m_rank.to_csv(os.path.join(OUT_DIR, "MRankings.csv"), index=False)
    w_rank.to_csv(os.path.join(OUT_DIR, "WRankings.csv"), index=False)
    print("Saved MRankings.csv and WRankings.csv")


# -----------------------------------------------------------------------
# 6. 2026 team list (for prediction submission)
# -----------------------------------------------------------------------

def save_2026_teams():
    """Save the team lists for 2026 (used for generating all-pairs predictions)."""
    m_names = _gen_team_names(MEN_TEAM_NAMES_BASE, N_MEN_TEAMS)
    w_names = [f"W{n}" if not n.startswith("W") else n for n in _gen_team_names(MEN_TEAM_NAMES_BASE, N_WOMEN_TEAMS)]

    pd.DataFrame({"TeamID": MEN_TEAM_IDS, "TeamName": m_names}).to_csv(
        os.path.join(OUT_DIR, "MTeams2026.csv"), index=False)
    pd.DataFrame({"TeamID": WOMEN_TEAM_IDS, "TeamName": w_names}).to_csv(
        os.path.join(OUT_DIR, "WTeams2026.csv"), index=False)
    print("Saved 2026 team lists")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating synthetic NCAA basketball data...")
    save_teams()
    save_regular_seasons()
    save_tournaments()
    save_rankings()
    save_2026_teams()
    print("Done! Sample data saved to:", OUT_DIR)
