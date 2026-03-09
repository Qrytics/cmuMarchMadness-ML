"""
Bracket simulation and evaluation for the CMU March Madness competition.
Implements both Regular and Progressive bracket scoring.

Scoring:
  Round 1 (play-in + first round): 1 pt
  Round 2: 2 pts
  Round 3 (Elite 8): 4 pts
  Round 4 (quarterfinals): 8 pts
  Round 5 (semifinals): 16 pts
  Round 6 (championship): 32 pts
  Perfect bracket: 196 pts
"""

import re
import pandas as pd
import numpy as np


ROUND_POINTS = {1: 1, 2: 2, 3: 4, 4: 8, 5: 16, 6: 32}

# First round matchups: seed1 vs seed2 in each region
ROUND1_PAIRS = [(1, 16), (8, 9), (5, 12), (4, 13), (6, 11), (3, 14), (7, 10), (2, 15)]
REGIONS = ["W", "X", "Y", "Z"]


def get_seed_number(seed_str):
    """Extract numeric seed from seed string like 'W01', 'X11a'."""
    m = re.search(r"\d+", str(seed_str))
    return int(m.group()) if m else 16


def get_region(seed_str):
    return str(seed_str)[0] if seed_str else "W"


class Bracket:
    """
    Represents a single tournament bracket with predictions and results.
    """

    def __init__(self, seeds_df, gender="M"):
        """
        seeds_df: DataFrame with columns [Season, Seed, TeamID]
        """
        self.gender = gender
        self.seeds_df = seeds_df
        self.seed_to_team = dict(zip(seeds_df["Seed"], seeds_df["TeamID"]))
        self.team_to_seed = dict(zip(seeds_df["TeamID"], seeds_df["Seed"]))
        self.predictions = {}  # (team1_id, team2_id) -> predicted_winner_id
        self.results = {}      # same, but actual results

    def add_predictions(self, pred_df):
        """
        Load predictions from a DataFrame with columns WTeamID, LTeamID.
        The 'winner' for each pair is WTeamID.
        """
        for _, row in pred_df.iterrows():
            w, l = row["WTeamID"], row["LTeamID"]
            key = (min(w, l), max(w, l))
            self.predictions[key] = w

    def predict_matchup(self, t1, t2):
        """Get predicted winner of a matchup between t1 and t2."""
        key = (min(t1, t2), max(t1, t2))
        return self.predictions.get(key, t1)  # default to lower ID if unknown

    def simulate_regular_bracket(self):
        """
        Simulate a regular bracket:
        - All predictions made before tournament
        - Work through rounds using predicted winners
        Returns ordered list of (round, team1, team2, predicted_winner)
        """
        bracket = {}

        # Handle play-in for seeds 16
        for region in REGIONS:
            for play_in_letter in ["a", "b"]:
                slot_a = f"{region}16{play_in_letter}"
                t_a = self.seed_to_team.get(slot_a)
                slot_b_partner = f"{region}16{'b' if play_in_letter == 'a' else 'a'}"
                t_b = self.seed_to_team.get(slot_b_partner)
                if t_a and t_b and play_in_letter == "a":
                    winner = self.predict_matchup(t_a, t_b)
                    bracket[(region, 16)] = winner
                    break
            else:
                slot = f"{region}16"
                t = self.seed_to_team.get(slot)
                if t:
                    bracket[(region, 16)] = t

        # Initialize bracket with seeded teams
        for region in REGIONS:
            for seed_num in range(1, 16):
                slot = f"{region}{seed_num:02d}"
                t = self.seed_to_team.get(slot)
                if t:
                    bracket[(region, seed_num)] = t

        games = []

        # Round 1: 32 games across 4 regions
        r1_winners = {}
        for region in REGIONS:
            for s1, s2 in ROUND1_PAIRS:
                t1 = bracket.get((region, s1))
                t2 = bracket.get((region, s2))
                if t1 and t2:
                    winner = self.predict_matchup(t1, t2)
                    r1_winners[(region, s1)] = winner
                    games.append((1, t1, t2, winner))

        # Round 2: winners of (1v16) vs (8v9), etc.
        r2_pairs = [(1, 8), (5, 4), (6, 3), (7, 2)]
        r2_winners = {}
        for region in REGIONS:
            for s1, s2 in r2_pairs:
                t1 = r1_winners.get((region, s1))
                t2 = r1_winners.get((region, s2))
                if t1 and t2:
                    winner = self.predict_matchup(t1, t2)
                    r2_winners[(region, s1)] = winner
                    games.append((2, t1, t2, winner))

        # Round 3 (Sweet 16): (1 side vs 4 side), (2 side vs 3 side)
        r3_pairs = [(1, 5), (6, 7)]
        r3_winners = {}
        for region in REGIONS:
            for s1, s2 in r3_pairs:
                t1 = r2_winners.get((region, s1))
                t2 = r2_winners.get((region, s2))
                if t1 and t2:
                    winner = self.predict_matchup(t1, t2)
                    r3_winners[(region, s1)] = winner
                    games.append((3, t1, t2, winner))

        # Round 4 (Elite 8): Regional finals
        r4_winners = {}
        for region in REGIONS:
            t1 = r3_winners.get((region, 1))
            t2 = r3_winners.get((region, 6))
            if t1 and t2:
                winner = self.predict_matchup(t1, t2)
                r4_winners[region] = winner
                games.append((4, t1, t2, winner))

        # Round 5 (Final Four): W vs X, Y vs Z
        final_four_pairs = [("W", "X"), ("Y", "Z")]
        r5_winners = []
        for r1_, r2_ in final_four_pairs:
            t1 = r4_winners.get(r1_)
            t2 = r4_winners.get(r2_)
            if t1 and t2:
                winner = self.predict_matchup(t1, t2)
                r5_winners.append(winner)
                games.append((5, t1, t2, winner))

        # Round 6 (Championship)
        if len(r5_winners) == 2:
            winner = self.predict_matchup(r5_winners[0], r5_winners[1])
            games.append((6, r5_winners[0], r5_winners[1], winner))

        return games

    def score_bracket(self, predicted_games, actual_results_df):
        """
        Score a regular bracket against actual results.
        predicted_games: list of (round, t1, t2, predicted_winner)
        actual_results_df: DataFrame with WTeamID, LTeamID columns

        Returns total score and per-round breakdown.
        """
        # Build actual results lookup
        actual = {}
        for _, row in actual_results_df.iterrows():
            w, l = row["WTeamID"], row["LTeamID"]
            actual[(min(w, l), max(w, l))] = w

        total = 0
        per_round = {}
        correct = 0
        total_games = 0

        for round_num, t1, t2, pred_winner in predicted_games:
            key = (min(t1, t2), max(t1, t2))
            actual_winner = actual.get(key)
            if actual_winner is None:
                continue

            pts = ROUND_POINTS.get(round_num, 1)
            total_games += 1
            if pred_winner == actual_winner:
                total += pts
                correct += 1
                per_round[round_num] = per_round.get(round_num, 0) + pts

        return {
            "total_score": total,
            "correct_predictions": correct,
            "total_games": total_games,
            "accuracy": correct / total_games if total_games > 0 else 0,
            "per_round": per_round,
            "max_possible": 196,
        }


def backtest_season(pred_df, seeds_df, actual_results_df, gender="M"):
    """
    Backtest predictions against a historical season's results.
    Returns scoring metrics.
    """
    bracket = Bracket(seeds_df, gender=gender)
    bracket.add_predictions(pred_df)
    games = bracket.simulate_regular_bracket()
    score = bracket.score_bracket(games, actual_results_df)
    return score
