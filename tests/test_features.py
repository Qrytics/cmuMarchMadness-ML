"""
Tests for data loading and feature engineering.
"""

import pytest
import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_loader import (
    load_teams, load_regular_season, load_tourney_results,
    load_seeds, load_rankings, compute_season_stats, get_seed_number
)
from src.feature_engineering import (
    build_team_features, build_matchup_features, build_training_data,
    get_feature_names
)

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "sample")


@pytest.fixture(scope="module")
def sample_data_dir():
    """Return path to sample data directory."""
    if not os.path.exists(os.path.join(SAMPLE_DIR, "MTeams.csv")):
        pytest.skip("Sample data not generated — run scripts/generate_sample_data.py first")
    return SAMPLE_DIR


class TestDataLoading:
    def test_load_men_teams(self, sample_data_dir):
        df = load_teams("M", sample_data_dir)
        assert "TeamID" in df.columns
        assert "TeamName" in df.columns
        assert len(df) == 381
        assert df["TeamID"].between(1000, 1999).all()

    def test_load_women_teams(self, sample_data_dir):
        df = load_teams("W", sample_data_dir)
        assert len(df) == 379
        assert df["TeamID"].between(3000, 3999).all()

    def test_load_regular_season(self, sample_data_dir):
        df = load_regular_season("M", sample_data_dir)
        required_cols = ["Season", "DayNum", "WTeamID", "WScore", "LTeamID", "LScore"]
        for col in required_cols:
            assert col in df.columns, f"Missing column: {col}"
        assert len(df) > 0

    def test_load_tourney_results(self, sample_data_dir):
        df = load_tourney_results("M", sample_data_dir)
        assert "WTeamID" in df.columns
        assert "LTeamID" in df.columns
        assert len(df) > 0

    def test_load_seeds(self, sample_data_dir):
        df = load_seeds("M", sample_data_dir)
        assert "Season" in df.columns
        assert "Seed" in df.columns
        assert "TeamID" in df.columns

    def test_compute_season_stats(self, sample_data_dir):
        regular = load_regular_season("M", sample_data_dir)
        stats = compute_season_stats(regular)
        assert "TeamID" in stats.columns
        assert "WinPct" in stats.columns
        assert "PointDiff" in stats.columns
        assert stats["WinPct"].between(0, 1).all()

    def test_get_seed_number(self):
        assert get_seed_number("W01") == 1
        assert get_seed_number("X16") == 16
        assert get_seed_number("Y03") == 3
        assert get_seed_number("W16a") == 16


class TestFeatureEngineering:
    @pytest.fixture(scope="class")
    def m_data(self, sample_data_dir):
        from src.data_loader import load_all_data
        return load_all_data("M", sample_data_dir)

    def test_build_team_features(self, m_data):
        season_stats = m_data["season_stats"]
        rankings = m_data["rankings"]
        season = season_stats["Season"].max()
        feats = build_team_features(season_stats, rankings, season)
        assert len(feats) > 0
        # Check a feature exists for a team
        some_team = list(feats.keys())[0]
        assert "WinPct" in feats[some_team]
        assert "PointDiff" in feats[some_team]

    def test_build_matchup_features(self):
        f1 = {"WinPct": 0.8, "AvgPointsFor": 75, "AvgPointsAgainst": 65, "PointDiff": 10,
               "AvgFGM": 27, "AvgFGA": 60, "AvgFGM3": 8, "AvgFGA3": 20,
               "AvgFTM": 15, "AvgFTA": 20, "AvgOR": 10, "AvgDR": 25,
               "AvgAst": 15, "AvgTO": 10, "AvgStl": 6, "AvgBlk": 4, "AvgPF": 18,
               "AvgRank": 5.0, "BestRank": 3}
        f2 = {"WinPct": 0.5, "AvgPointsFor": 68, "AvgPointsAgainst": 68, "PointDiff": 0,
               "AvgFGM": 24, "AvgFGA": 60, "AvgFGM3": 6, "AvgFGA3": 20,
               "AvgFTM": 12, "AvgFTA": 18, "AvgOR": 9, "AvgDR": 22,
               "AvgAst": 12, "AvgTO": 13, "AvgStl": 5, "AvgBlk": 3, "AvgPF": 20,
               "AvgRank": 50.0, "BestRank": 40}

        feats = build_matchup_features(f1, f2, seed1="W01", seed2="W16")
        assert "diff_WinPct" in feats
        assert "SeedDiff" in feats
        assert feats["SeedDiff"] == 1 - 16  # seed1 - seed2
        assert feats["diff_WinPct"] == pytest.approx(0.3, abs=0.01)

    def test_build_training_data(self, m_data):
        df = build_training_data(m_data, "M")
        assert "label" in df.columns
        assert "season" in df.columns
        assert set(df["label"].unique()) == {0, 1}
        # Should be balanced (each game appears twice with flipped labels)
        assert abs(df["label"].mean() - 0.5) < 0.01

    def test_training_data_no_missing(self, m_data):
        df = build_training_data(m_data, "M")
        feat_cols = get_feature_names(df)
        # Features should be mostly finite (some NaN ok for ranking)
        assert df[feat_cols].isnull().mean().mean() < 0.5


EXTERNAL_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "external")
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


class TestExternalDataLoaders:
    """Tests for the new external data loaders (KenPom, Barttorvik, NET, etc.)."""

    @pytest.fixture(scope="class")
    def external_available(self):
        """Skip class if external data has not been staged."""
        if not os.path.exists(os.path.join(EXTERNAL_DIR, "kenpom_2026.csv")):
            pytest.skip("External data not staged — run scripts/fetch_external_data.py first")
        return True

    def test_kenpom_loader(self, external_available):
        from src.data_loader import load_kenpom
        df = load_kenpom(2026)
        assert not df.empty, "KenPom data should load for season 2026"
        assert "TeamID" in df.columns
        assert "KP_AdjEM" in df.columns
        assert (df["TeamID"] > 0).all()
        assert len(df) > 200

    def test_barttorvik_loader(self, external_available):
        from src.data_loader import load_barttorvik
        df = load_barttorvik(2026)
        assert not df.empty, "Barttorvik data should load for season 2026"
        assert "TeamID" in df.columns
        assert "BT_AdjOE" in df.columns
        assert "BT_AdjDE" in df.columns
        assert len(df) > 200

    def test_net_rankings_loader(self, external_available):
        from src.data_loader import load_net_rankings
        df = load_net_rankings(2026)
        assert not df.empty, "NET rankings should load for season 2026"
        assert "TeamID" in df.columns
        assert "NET_Rank" in df.columns
        # Top teams should be ranked low (rank 1 = best)
        assert df["NET_Rank"].min() == 1

    def test_player_stats_loader(self, external_available):
        from src.data_loader import load_player_stats
        df = load_player_stats(2026)
        assert not df.empty, "Player stats should load for season 2026"
        assert "TeamID" in df.columns
        assert len(df) > 100

    def test_recruiting_loader(self, external_available):
        from src.data_loader import load_recruiting
        df = load_recruiting(2026)
        assert not df.empty, "Recruiting data should load for season 2026"
        assert "TeamID" in df.columns
        assert "REC_Composite" in df.columns

    def test_draft_loader(self, external_available):
        from src.data_loader import load_draft_prospects
        df = load_draft_prospects(2026)
        assert not df.empty, "Draft data should load for season 2026"
        assert "TeamID" in df.columns

    def test_external_features_in_season_stats(self, external_available):
        """External features are merged into season_stats for 2026."""
        if not os.path.exists(os.path.join(RAW_DIR, "MTeams.csv")):
            pytest.skip("Kaggle raw data not extracted")
        from src.data_loader import load_all_data
        data = load_all_data("M", RAW_DIR)
        ss = data["season_stats"]
        ext_cols = [c for c in ss.columns if c.startswith("KP_")]
        assert len(ext_cols) > 0, "KenPom features should be present in season_stats"
        # Spot-check: 2026 season should have KenPom coverage for most teams
        s2026 = ss[ss["Season"] == 2026]
        coverage = s2026["KP_AdjEM"].notna().mean()
        assert coverage > 0.8, f"KenPom 2026 coverage too low: {coverage:.1%}"

    def test_new_matchup_features_present(self):
        """build_matchup_features includes new external feature diffs."""
        f1 = {
            "WinPct": 0.8, "AvgPointsFor": 75, "AvgPointsAgainst": 65,
            "PointDiff": 10, "AvgFGM": 27, "AvgFGA": 60, "AvgFGM3": 8,
            "AvgFGA3": 20, "AvgFTM": 15, "AvgFTA": 20, "AvgOR": 10, "AvgDR": 25,
            "AvgAst": 15, "AvgTO": 10, "AvgStl": 6, "AvgBlk": 4, "AvgPF": 18,
            "AvgRank": 5.0, "BestRank": 3,
            "KP_AdjEM": 25.0, "BT_AdjOE": 120.0, "BT_AdjDE": 95.0,
            "NET_Rank": 3, "REC_Composite": 68.0, "PS_TopPRPG": 7.5,
            "DRAFT_TopPick": 3, "DRAFT_NumRound1": 2,
        }
        f2 = {
            "WinPct": 0.5, "AvgPointsFor": 68, "AvgPointsAgainst": 68,
            "PointDiff": 0, "AvgFGM": 24, "AvgFGA": 60, "AvgFGM3": 6,
            "AvgFGA3": 20, "AvgFTM": 12, "AvgFTA": 18, "AvgOR": 9, "AvgDR": 22,
            "AvgAst": 12, "AvgTO": 13, "AvgStl": 5, "AvgBlk": 3, "AvgPF": 20,
            "AvgRank": 50.0, "BestRank": 40,
            "KP_AdjEM": 5.0, "BT_AdjOE": 108.0, "BT_AdjDE": 103.0,
            "NET_Rank": 30, "REC_Composite": 50.0, "PS_TopPRPG": 3.0,
            "DRAFT_TopPick": float("nan"), "DRAFT_NumRound1": 0,
        }
        feats = build_matchup_features(f1, f2, seed1="W01", seed2="W16")
        assert "diff_KP_AdjEM" in feats
        assert "diff_BT_AdjOE" in feats
        assert "diff_NET_Rank" in feats
        assert "diff_REC_Composite" in feats
        assert "diff_PS_TopPRPG" in feats
        assert "diff_DRAFT_TopPick" in feats
        assert "interact_KP_BT_AdjEM" in feats
        assert "interact_StarPlayer_SeedDiff" in feats
        # Better team should have positive KP_AdjEM diff
        assert feats["diff_KP_AdjEM"] == pytest.approx(20.0)
