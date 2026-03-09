"""
Tests for bracket simulation and scoring.
"""

import pytest
import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.bracket import (
    Bracket, backtest_season, ROUND_POINTS, get_seed_number
)

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "sample")


def _make_seeds(season=2024, n_teams=64, base_id=1101):
    """Create a simple seeds DataFrame for testing."""
    regions = ["W", "X", "Y", "Z"]
    records = []
    tid = base_id
    for region in regions:
        for seed in range(1, 17):
            records.append({"Season": season, "Seed": f"{region}{seed:02d}", "TeamID": tid})
            tid += 1
    return pd.DataFrame(records)


def _make_predictions(seeds_df):
    """Create random predictions DataFrame."""
    team_ids = sorted(seeds_df["TeamID"].unique())
    records = []
    for i in range(len(team_ids)):
        for j in range(i + 1, len(team_ids)):
            t1, t2 = team_ids[i], team_ids[j]
            records.append({"WTeamID": t1, "LTeamID": t2})
    return pd.DataFrame(records)


class TestBracketSimulation:
    @pytest.fixture
    def seeds(self):
        return _make_seeds(2024)

    @pytest.fixture
    def preds(self, seeds):
        return _make_predictions(seeds)

    def test_bracket_creation(self, seeds):
        bracket = Bracket(seeds)
        assert bracket.gender == "M"
        assert len(bracket.seed_to_team) == 64

    def test_add_predictions(self, seeds, preds):
        bracket = Bracket(seeds)
        bracket.add_predictions(preds)
        assert len(bracket.predictions) > 0

    def test_simulate_regular_bracket(self, seeds, preds):
        bracket = Bracket(seeds)
        bracket.add_predictions(preds)
        games = bracket.simulate_regular_bracket()
        assert len(games) > 0
        # Should have at most 63 games (63 in a 64-team single-elim)
        assert len(games) <= 63

    def test_rounds_are_correct(self, seeds, preds):
        bracket = Bracket(seeds)
        bracket.add_predictions(preds)
        games = bracket.simulate_regular_bracket()
        rounds = [g[0] for g in games]
        # Only valid round numbers
        for r in rounds:
            assert r in ROUND_POINTS, f"Invalid round: {r}"

    def test_score_bracket(self, seeds, preds):
        bracket = Bracket(seeds)
        bracket.add_predictions(preds)
        games = bracket.simulate_regular_bracket()

        # Make all predictions correct (use same games as actual results)
        actual_df = pd.DataFrame([{"WTeamID": g[3], "LTeamID": g[2] if g[3] == g[1] else g[1]}
                                   for g in games if g[3] is not None])
        score = bracket.score_bracket(games, actual_df)

        assert score["total_score"] >= 0
        assert score["total_score"] <= 196
        assert "accuracy" in score
        assert "per_round" in score

    def test_perfect_predictions_max_score(self, seeds, preds):
        """If all predictions are correct, score should be maximized."""
        bracket = Bracket(seeds)
        bracket.add_predictions(preds)
        games = bracket.simulate_regular_bracket()

        # Make "actual" results match predictions perfectly
        actual_results = []
        for round_num, t1, t2, pred_winner in games:
            loser = t2 if pred_winner == t1 else t1
            actual_results.append({"WTeamID": pred_winner, "LTeamID": loser})
        actual_df = pd.DataFrame(actual_results)

        score = bracket.score_bracket(games, actual_df)
        assert score["accuracy"] == pytest.approx(1.0, abs=0.01)


class TestSeedParsing:
    def test_seed_number_extraction(self):
        assert get_seed_number("W01") == 1
        assert get_seed_number("X16") == 16
        assert get_seed_number("Y08") == 8
        assert get_seed_number("Z11") == 11
        assert get_seed_number("W16a") == 16
        assert get_seed_number(5) == 5


class TestBacktest:
    def test_backtest_returns_score(self):
        if not os.path.exists(os.path.join(SAMPLE_DIR, "MSeeds.csv")):
            pytest.skip("Sample data not available")

        from src.data_loader import load_seeds, load_tourney_results
        seeds_df = load_seeds("M", SAMPLE_DIR)
        tourney = load_tourney_results("M", SAMPLE_DIR)

        # Use most recent season
        season = seeds_df["Season"].max()
        season_seeds = seeds_df[seeds_df["Season"] == season]
        season_tourney = tourney[tourney["Season"] == season]

        # Use actual results as predictions (should score high)
        pred_df = season_tourney[["WTeamID", "LTeamID"]].copy()

        score = backtest_season(pred_df, season_seeds, season_tourney)
        assert score["total_score"] >= 0
        assert score["max_possible"] == 196

    def test_scoring_weights(self):
        """Verify round point values are correct."""
        assert ROUND_POINTS[1] == 1   # Round 1
        assert ROUND_POINTS[2] == 2   # Round 2
        assert ROUND_POINTS[3] == 4   # Elite 8
        assert ROUND_POINTS[4] == 8   # Quarterfinals
        assert ROUND_POINTS[5] == 16  # Semifinals
        assert ROUND_POINTS[6] == 32  # Championship

    def test_max_score_is_196(self):
        """Play-in (4×1) + Round1 (32×1) + Round2 (16×2) + R3 (8×4) + R4 (4×8) + R5 (2×16) + R6 (1×32) = 196."""
        # Per the competition rules: play-in games are worth 1 pt (same as first round)
        play_in_score = 4 * 1   # 4 play-in games
        regular_rounds = {1: 32, 2: 16, 3: 8, 4: 4, 5: 2, 6: 1}
        total = play_in_score + sum(ROUND_POINTS[r] * n for r, n in regular_rounds.items())
        assert total == 196


class TestPredictionFormat:
    def test_submission_format(self):
        """Validate the submission CSV format."""
        pred_path = os.path.join(
            os.path.dirname(__file__), "..", "predictions", "MTourneyPredictions.csv"
        )
        if not os.path.exists(pred_path):
            pytest.skip("Predictions not generated yet")

        df = pd.read_csv(pred_path)
        assert list(df.columns) == ["WTeamID", "LTeamID"], f"Wrong columns: {list(df.columns)}"
        # Row count = C(n_teams, 2).  With Stage-2 Kaggle data: 365 teams → 66,430.
        # With sample-only data: 381 teams → 72,390.  Accept either.
        n = len(df)
        assert n in {66430, 72390}, f"Unexpected row count: {n}"
        assert df["WTeamID"].dtype in [int, np.int64]
        assert df["LTeamID"].dtype in [int, np.int64]
        # Ensure no duplicate pairs
        pairs = set(zip(df["WTeamID"].tolist(), df["LTeamID"].tolist()))
        assert len(pairs) == len(df), "Duplicate pairs found"

    def test_women_submission_format(self):
        pred_path = os.path.join(
            os.path.dirname(__file__), "..", "predictions", "WTourneyPredictions.csv"
        )
        if not os.path.exists(pred_path):
            pytest.skip("Predictions not generated yet")

        df = pd.read_csv(pred_path)
        # Row count = C(n_teams, 2).  Stage-2: 363 teams → 65,703.
        # Sample-only: 379 teams → 71,631.  Accept either.
        n = len(df)
        assert n in {65703, 71631}, f"Unexpected row count: {n}"
