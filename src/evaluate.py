"""
Model evaluation and backtesting for the CMU March Madness competition.
Evaluates models against historical tournament seasons to estimate real performance.

Usage:
    python -m src.evaluate          # evaluate both M and W models
    python -m src.evaluate --gender M --seasons 2022 2023 2024
"""

import argparse
import os
import json
import numpy as np
import pandas as pd


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.data_loader import load_all_data, get_seed_number
from src.feature_engineering import build_team_features, build_matchup_features
from src.model import MarchMadnessModel
from src.bracket import backtest_season, ROUND_POINTS

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
DOCS_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "assets")
os.makedirs(DOCS_DIR, exist_ok=True)


def evaluate_gender(gender="M", data_dir=None, seasons=None, save_plots=True):
    """
    Evaluate the model against historical tournament seasons.
    Returns a dict of season-by-season results.
    """
    print(f"\n{'='*60}")
    print(f"Evaluating {gender} model...")
    print(f"{'='*60}")

    model = MarchMadnessModel.load(gender=gender)
    data = load_all_data(gender, data_dir)

    all_seasons = sorted(data["tourney"]["Season"].unique())
    if seasons:
        eval_seasons = [s for s in seasons if s in all_seasons]
    else:
        # Evaluate on the most recent 5 seasons
        eval_seasons = all_seasons[-min(5, len(all_seasons)):]

    print(f"  Evaluating seasons: {eval_seasons}")

    all_results = []
    bracket_scores = []

    # Use pre-aggregated rankings if available (fast path for Massey ordinals)
    rankings_agg = data.get("rankings_agg", {})

    for season in eval_seasons:
        # Build team features using only prior seasons' data (no leakage)
        prior_stats = data["season_stats"][data["season_stats"]["Season"] < season]

        # For the rankings dict, filter to entries before the eval season
        if rankings_agg:
            prior_rankings = {
                k: v for k, v in rankings_agg.items() if k[0] < season
            }
        else:
            prior_rankings_df = data["rankings"]
            prior_rankings = (
                prior_rankings_df[prior_rankings_df["Season"] < season]
                if not prior_rankings_df.empty
                else prior_rankings_df
            )

        if prior_stats.empty:
            print(f"  Season {season}: No prior data, skipping.")
            continue

        use_season = prior_stats["Season"].max()
        team_feats = build_team_features(prior_stats, prior_rankings, use_season)

        # Get tournament games and seeds for this season
        tourney_games = data["tourney"][data["tourney"]["Season"] == season]
        season_seeds = data["seeds"][data["seeds"]["Season"] == season]
        seed_map = dict(zip(season_seeds["TeamID"], season_seeds["Seed"]))

        # Predict each game
        correct = 0
        total = 0
        game_results = []

        for _, game in tourney_games.iterrows():
            w_id = game["WTeamID"]
            l_id = game["LTeamID"]

            if w_id not in team_feats or l_id not in team_feats:
                continue

            w_seed = seed_map.get(w_id)
            l_seed = seed_map.get(l_id)

            feats = build_matchup_features(team_feats[w_id], team_feats[l_id], w_seed, l_seed)
            feat_df = pd.DataFrame([feats])

            # Align features
            for col in model.feature_names:
                if col not in feat_df.columns:
                    feat_df[col] = 0.0
            feat_df = feat_df[model.feature_names]

            proba = model.predict_proba(feat_df)[0]
            pred_winner = w_id if proba >= 0.5 else l_id

            is_correct = (pred_winner == w_id)
            correct += int(is_correct)
            total += 1

            game_results.append({
                "season": season,
                "WTeamID": w_id,
                "LTeamID": l_id,
                "pred_winner": pred_winner,
                "correct": is_correct,
                "confidence": proba if proba >= 0.5 else 1 - proba,
            })

        if total > 0:
            acc = correct / total
            print(f"  Season {season}: {correct}/{total} correct ({acc:.1%})")

            # Bracket simulation score
            pred_df = pd.DataFrame([
                {"WTeamID": r["pred_winner"],
                 "LTeamID": r["LTeamID"] if r["pred_winner"] == r["WTeamID"] else r["WTeamID"]}
                for r in game_results
            ])
            score = backtest_season(pred_df, season_seeds, tourney_games, gender)
            bracket_scores.append({"season": season, **score})
            print(f"  Season {season}: Bracket score = {score['total_score']}/196 "
                  f"(accuracy: {score['accuracy']:.1%})")

            all_results.extend(game_results)

    results_df = pd.DataFrame(all_results) if all_results else pd.DataFrame()

    if save_plots and not results_df.empty:
        _plot_evaluation(gender, results_df, bracket_scores)

    # Save bracket scores
    scores_path = os.path.join(RESULTS_DIR, f"{gender.lower()}_backtest_scores.json")
    with open(scores_path, "w") as f:
        json.dump(bracket_scores, f, indent=2, cls=_NumpyEncoder)

    return {"game_results": results_df.to_dict("records"), "bracket_scores": bracket_scores}


def _plot_evaluation(gender, results_df, bracket_scores):
    """Generate evaluation plots and save to docs/assets."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"{gender} Tournament Model Evaluation", fontsize=14, fontweight="bold")

    # Plot 1: Accuracy per season
    ax1 = axes[0]
    seasons = [s["season"] for s in bracket_scores]
    accs = [s["accuracy"] * 100 for s in bracket_scores]
    bars = ax1.bar(seasons, accs, color="#1f77b4", alpha=0.8)
    ax1.axhline(50, color="red", linestyle="--", label="Random baseline (50%)")
    ax1.set_xlabel("Season")
    ax1.set_ylabel("Game Accuracy (%)")
    ax1.set_title("Prediction Accuracy by Season")
    ax1.set_ylim(0, 100)
    ax1.legend()
    for bar, acc in zip(bars, accs):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"{acc:.0f}%", ha="center", va="bottom", fontsize=9)

    # Plot 2: Bracket score per season
    ax2 = axes[1]
    scores = [s["total_score"] for s in bracket_scores]
    bars2 = ax2.bar(seasons, scores, color="#2ca02c", alpha=0.8)
    ax2.axhline(196, color="gold", linestyle="--", label="Perfect (196)")
    ax2.axhline(np.mean(scores) if scores else 0, color="orange",
                linestyle=":", label=f"Avg ({np.mean(scores):.0f})")
    ax2.set_xlabel("Season")
    ax2.set_ylabel("Bracket Score")
    ax2.set_title("Bracket Score by Season (max 196)")
    ax2.set_ylim(0, 210)
    ax2.legend()
    for bar, sc in zip(bars2, scores):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                 str(sc), ha="center", va="bottom", fontsize=9)

    # Plot 3: Confidence distribution
    ax3 = axes[2]
    conf = results_df["confidence"].values
    correct = results_df["correct"].values
    ax3.hist(conf[correct == 1], bins=20, alpha=0.7, color="green", label="Correct")
    ax3.hist(conf[correct == 0], bins=20, alpha=0.7, color="red", label="Incorrect")
    ax3.set_xlabel("Model Confidence")
    ax3.set_ylabel("Count")
    ax3.set_title("Confidence Distribution")
    ax3.legend()

    plt.tight_layout()
    path = os.path.join(DOCS_DIR, f"{gender.lower()}_evaluation.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved to {path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate March Madness models")
    parser.add_argument("--gender", choices=["M", "W", "both"], default="both")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--seasons", type=int, nargs="+", default=None,
                        help="Specific seasons to evaluate")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    genders = ["M", "W"] if args.gender == "both" else [args.gender]
    for g in genders:
        evaluate_gender(g, data_dir=args.data_dir, seasons=args.seasons,
                        save_plots=not args.no_plots)


if __name__ == "__main__":
    main()
