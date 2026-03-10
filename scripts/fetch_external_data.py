"""
Fetch and stage external data sources that can significantly improve
prediction accuracy beyond what the Kaggle competition dataset alone provides.

Run this script once per season to refresh the external data:
    python scripts/fetch_external_data.py [--season 2026]

Data sources
============
The following are ranked by expected accuracy impact and ease of access.

TIER 1 – Highest impact, free or very low cost
-----------------------------------------------
1. Barttorvik / T-Rank  (https://barttorvik.com)
   - Adjusted offensive / defensive efficiency, adjusted tempo, luck rating,
     SOS-adjusted record.  Very similar to KenPom but FREE.
   - Download: https://barttorvik.com/trank.php  (CSV export available)
   - Files written: data/external/barttorvik_{season}.csv

2. NCAA NET Rankings  (https://www.ncaa.com/rankings/basketball-men/d1/ncaa-net)
   - Official NCAA ranking metric used by the selection committee since 2019.
   - Highly predictive of selection / seeding and therefore of outcomes.
   - Files written: data/external/net_rankings_{season}.csv

3. Massey Composite  (https://masseyratings.com/ranks)
   - Composite of 60+ rating systems (superset of the Kaggle Massey Ordinals).
   - Already partially included in MMasseyOrdinals.csv but the composite
     consensus is an additional signal.
   - Files written: data/external/massey_composite_{season}.csv

TIER 2 – High impact, require a paid subscription or scraping
-------------------------------------------------------------
4. KenPom  (https://kenpom.com)
   - ~$20/year subscription.  Adjusted efficiency, adjusted tempo,
     experience index, luck rating, SOS metrics.  The gold standard for
     NCAA basketball analytics.
   - Expected accuracy lift vs. no KenPom: +1-2% game accuracy.
   - Files written: data/external/kenpom_{season}.csv
   - Schema expected: Team, AdjEM, AdjO, AdjD, AdjT, Luck, SOS_AdjEM,
                      OppO, OppD, NCSOS_AdjEM

5. Historical Vegas point spreads & totals
   - Point spreads from oddsportal.com or covers.com encode market consensus.
   - Expected accuracy lift: +2-3% (comparable to seed alone).
   - Files written: data/external/vegas_lines_{season}.csv
   - Schema expected: Season, Team1, Team2, Spread, Total, Date

TIER 3 – Medium impact, harder to acquire
------------------------------------------
6. Player-level statistics  (https://www.sports-reference.com/cbb/)
   - Individual player PER, usage rate, True Shooting %
   - Enables detecting star-player presence / key-player absence.
   - Note: Sports-Reference has a robots.txt; check their data licensing.
   - Files written: data/external/player_stats_{season}.csv

7. Recruiting rankings  (https://247sports.com/Season/2025-Basketball/CompositeTeamRankings/)
   - Team recruiting class composite score.  Strong talent pipeline proxy.
   - Files written: data/external/recruiting_{season}.csv
   - Schema expected: Season, TeamName, Composite, NumCommits, Rank

8. NBA Draft prospect rankings  (https://nbadraft.net / ESPN)
   - Presence of a projected top-10 NBA pick is a huge tournament predictor.
   - Files written: data/external/nba_draft_prospects_{season}.csv

TIER 4 – Novel, lower proven impact
-------------------------------------
9. Social media sentiment (Twitter/Reddit)
   - Team momentum / public perception signals.
   - Requires Twitter API (Basic tier, ~$100/month) or Reddit API (free).
   - Files written: data/external/sentiment_{season}.csv

10. Travel / geography features
    - Distance each team must travel to tournament venue.
    - Time-zone shift, which affects player performance.
    - Computable from Cities.csv (already in Kaggle data) + team home-city
      lookup.

11. Player height / physical attributes from photos
    - Computer-vision estimation of average team height from player photos.
    - Could proxy for rebounding ability / inside scoring.
    - Practical limitation: requires scraping team roster photos
      (e.g. from ESPN or the school's athletics site) and running a
      height-estimation CNN.  This is a research-level effort; not
      straightforward to operationalise.

12. Video / highlight analysis
    - Characterise play style (pace, 3-point vs. inside focus) from video.
    - Requires large compute (GPU) and data-scraping infrastructure.
    - Not practical for this competition without a significant engineering lift.

HOW TO USE THE FILES ONCE DOWNLOADED
======================================
Place the CSVs in data/external/.  The model pipeline reads them
automatically via load_all_data() once the corresponding loader
functions detect them.  Currently the loaders for Barttorvik, NET,
and KenPom are stubbed as examples below; fill in the actual paths
or URLs when you have access.

Schema conventions used by the rest of the pipeline
-----------------------------------------------------
All external files must contain at least:
  - Season (int): the year the season ends
  - TeamID (int): Kaggle-format team ID

For KenPom / Barttorvik the suggested mapping workflow is:
  1. Load MTeamSpellings.csv (Kaggle) to get TeamNameSpelling → TeamID.
  2. Left-join the external file on the lower-cased team name.
  3. Write the enriched file with a TeamID column.

A helper function (``map_team_names``) is provided at the bottom of this
script to assist with this mapping step.
"""

import os
import sys
import argparse
import re

import pandas as pd

EXTERNAL_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "external")
KAGGLE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "sample")

os.makedirs(EXTERNAL_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Team-name mapping helper
# ---------------------------------------------------------------------------

def load_team_spellings(gender="M"):
    """Load MTeamSpellings.csv or WTeamSpellings.csv for name normalisation."""
    for d in [KAGGLE_DIR, SAMPLE_DIR]:
        path = os.path.join(d, f"{gender}TeamSpellings.csv")
        if os.path.exists(path):
            return pd.read_csv(path)
    return pd.DataFrame(columns=["TeamNameSpelling", "TeamID"])


def map_team_names(external_df, name_col, gender="M"):
    """Map an external team-name column to Kaggle TeamIDs.

    Parameters
    ----------
    external_df : pd.DataFrame
        The external data frame.  Must have a column ``name_col`` with team
        names as strings.
    name_col : str
        Name of the column containing team names (e.g. "Team").
    gender : str
        "M" or "W".

    Returns
    -------
    pd.DataFrame
        A copy of ``external_df`` with a new ``TeamID`` column.  Rows whose
        team name could not be matched are assigned TeamID = -1.
    """
    spellings = load_team_spellings(gender)
    if spellings.empty:
        print(f"  WARNING: Could not load {gender}TeamSpellings.csv – TeamID mapping skipped.")
        external_df = external_df.copy()
        external_df["TeamID"] = pd.NA
        return external_df

    # Normalise: lowercase, strip punctuation
    def _normalise(s):
        return re.sub(r"[^a-z0-9]", "", str(s).lower())

    spell_map = {_normalise(r.TeamNameSpelling): r.TeamID
                 for r in spellings.itertuples(index=False)}

    result = external_df.copy()
    result["TeamID"] = result[name_col].apply(
        lambda n: spell_map.get(_normalise(n), pd.NA)
    )
    unmatched = result["TeamID"].isna().sum()
    if unmatched:
        print(f"  WARNING: {unmatched} team names could not be mapped to a TeamID.")
    return result


# ---------------------------------------------------------------------------
# Barttorvik (T-Rank) – free, public
# ---------------------------------------------------------------------------

def fetch_barttorvik(season: int):
    """Download T-Rank efficiency data for a given season.

    Barttorvik publishes historical tables at:
        https://barttorvik.com/trank.php?year={season}&...

    The table has 40+ columns.  We keep the most useful ones:
        Rk, Team, Conf, Rec, AdjOE, AdjDE, Barthag (win probability),
        EFG_O, EFG_D, TOR, TORD, ORB, DRB, FTR, FTRD, 2P_O, 2P_D,
        3P_O, 3P_D, Adj_T (tempo)

    Availability: freely available for all seasons back to 2008.

    This function attempts to fetch the data.  If the URL is unreachable
    (e.g. in a sandboxed environment), it prints instructions for manual
    download instead.
    """
    out_path = os.path.join(EXTERNAL_DIR, f"barttorvik_{season}.csv")
    if os.path.exists(out_path):
        print(f"  Barttorvik {season}: already exists at {out_path}")
        return

    url = (
        f"https://barttorvik.com/trank.php?year={season}"
        f"&sort=&top=0&conlimit=All&venue=All&type=All"
        f"&begin={season - 1}1101&end={season}0401&npcts=&csv=1"
    )
    print(f"  Fetching Barttorvik {season} from: {url}")
    try:
        df = pd.read_csv(url, header=0)
        # Rename columns to standard names
        rename = {
            "Team": "Team",
            "AdjOE": "BT_AdjOE",
            "AdjDE": "BT_AdjDE",
            "Barthag": "BT_Barthag",
            "EFG%": "BT_EFG_O",
            "EFGD%": "BT_EFG_D",
            "TOR": "BT_TOR",
            "TORD": "BT_TORD",
            "ORB": "BT_ORB",
            "DRB": "BT_DRB",
            "FTR": "BT_FTR",
            "FTRD": "BT_FTRD",
            "2P%": "BT_2P_O",
            "2PD%": "BT_2P_D",
            "3P%": "BT_3P_O",
            "3PD%": "BT_3P_D",
            "Adj. T": "BT_AdjT",
        }
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
        df["Season"] = season
        df = map_team_names(df, "Team", gender="M")
        df.to_csv(out_path, index=False)
        print(f"  Saved to {out_path} ({len(df)} rows)")
    except Exception as exc:
        print(f"  Could not fetch Barttorvik data automatically: {exc}")
        print(f"\n  MANUAL DOWNLOAD INSTRUCTIONS (Barttorvik {season}):")
        print(f"    1. Visit: https://barttorvik.com/trank.php?year={season}")
        print(f"    2. Scroll to the bottom → click 'Download CSV'")
        print(f"    3. Save as: {out_path}")
        print(f"    4. Run this script again to apply team-name mapping.\n")


# ---------------------------------------------------------------------------
# NCAA NET Rankings – free, public
# ---------------------------------------------------------------------------

def fetch_net_rankings(season: int):
    """Fetch NCAA NET rankings for a given season.

    The NCAA publishes NET rankings at:
        https://www.ncaa.com/rankings/basketball-men/d1/ncaa-net

    Historical archives (back to 2019) are available from:
        https://www.warrennolan.com/basketball/2026/net-nitty-gritty

    This function prints instructions since the NCAA site requires
    JavaScript rendering that prevents simple CSV fetching.
    """
    out_path = os.path.join(EXTERNAL_DIR, f"net_rankings_{season}.csv")
    if os.path.exists(out_path):
        print(f"  NET rankings {season}: already exists at {out_path}")
        return

    print(f"\n  MANUAL DOWNLOAD INSTRUCTIONS (NCAA NET rankings {season}):")
    print(f"    Option A (Warren Nolan):")
    print(f"      1. Visit: https://www.warrennolan.com/basketball/{season}/net-nitty-gritty")
    print(f"      2. Copy the table (or use a browser extension to export as CSV).")
    print(f"      3. Save as: {out_path}")
    print(f"    Option B (official NCAA page):")
    print(f"      1. Visit: https://www.ncaa.com/rankings/basketball-men/d1/ncaa-net")
    print(f"      2. The table is rendered in JS; use a tool like Selenium or")
    print(f"         https://parsehub.com to scrape it to CSV.")
    print(f"    Expected columns: Rank, Team, NET, AdjEM (optional)")
    print(f"    Then run: python scripts/fetch_external_data.py --map-only --file {out_path}")
    print()


# ---------------------------------------------------------------------------
# KenPom – subscription required (~$20/year)
# ---------------------------------------------------------------------------

def fetch_kenpom(season: int):
    """Provide instructions for obtaining KenPom data.

    KenPom (https://kenpom.com) is the gold standard for NCAA basketball
    analytics.  It provides:
      - AdjEM  (adjusted efficiency margin – best single predictor)
      - AdjO   (adjusted offensive efficiency per 100 possessions)
      - AdjD   (adjusted defensive efficiency per 100 possessions)
      - AdjT   (adjusted tempo – possessions per 40 minutes)
      - Luck   (deviation of actual record from expected)
      - SOS_AdjEM, OppO, OppD, NCSOS_AdjEM  (strength of schedule variants)

    Expected accuracy lift (vs. model without KenPom): ~1-2% game accuracy
    and ~0.01-0.02 log-loss reduction, based on past competition results.
    """
    out_path = os.path.join(EXTERNAL_DIR, f"kenpom_{season}.csv")
    if os.path.exists(out_path):
        print(f"  KenPom {season}: already exists at {out_path}")
        return

    print(f"\n  KENPOM DATA INSTRUCTIONS (season {season}):")
    print(f"    1. Subscribe at https://kenpom.com (~$20/year).")
    print(f"    2. Log in → go to 'Summary' page.")
    print(f"    3. Select season {season} from the dropdown.")
    print(f"    4. Click 'Export' (CSV) – file is named e.g. summary{season}.csv.")
    print(f"    5. Save as: {out_path}")
    print(f"    6. Expected columns: Team, AdjEM, AdjO, AdjD, AdjT, Luck,")
    print(f"       SOS_AdjEM, OppO, OppD, NCSOS_AdjEM")
    print(f"    7. Run: python scripts/fetch_external_data.py --map-only --file {out_path}")
    print()


# ---------------------------------------------------------------------------
# Vegas spreads – historical lines
# ---------------------------------------------------------------------------

def fetch_vegas_lines(season: int):
    """Provide instructions for obtaining historical betting lines.

    Betting market consensus (point spreads and totals) is one of the most
    predictive signals available for March Madness.  Spreads implicitly
    encode adjusted strength-of-schedule, injuries, travel, and market
    sentiment – all at once.

    Free sources for historical lines:
      - oddsportal.com  (manual scraping needed)
      - covers.com      (manual scraping needed)
      - https://www.sportsbookreviewsonline.com/scoresoddsarchives/ (CSV)
    """
    out_path = os.path.join(EXTERNAL_DIR, f"vegas_lines_{season}.csv")
    if os.path.exists(out_path):
        print(f"  Vegas lines {season}: already exists at {out_path}")
        return

    print(f"\n  VEGAS SPREADS INSTRUCTIONS (season {season}):")
    print(f"    Option A – Sportsbook Reviews Online (free CSV):")
    print(f"      1. Visit: https://www.sportsbookreviewsonline.com/scoresoddsarchives/")
    print(f"      2. Download the NCAA Basketball archive for season {season}.")
    print(f"      3. Save as: {out_path}")
    print(f"    Option B – oddsportal.com (scraping):")
    print(f"      Use Selenium or playwright to scrape tournament game lines.")
    print(f"    Expected columns: Date, Team1, Team2, Spread, Total, Season")
    print()


# ---------------------------------------------------------------------------
# Player statistics (Sports-Reference)
# ---------------------------------------------------------------------------

def fetch_player_stats(season: int):
    """Print instructions for player-level statistics."""
    print(f"\n  PLAYER STATISTICS INSTRUCTIONS (season {season}):")
    print(f"    Sports-Reference / College Basketball Reference:")
    print(f"      https://www.sports-reference.com/cbb/friv/standings.fcgi")
    print(f"    Individual player stats (PER, TS%, usage):")
    print(f"      https://www.sports-reference.com/cbb/play-index/psl_finder.fcgi")
    print(f"    NOTE: Respect robots.txt and rate-limit scraping.")
    print(f"    Expected columns: Season, TeamID, PlayerName, PER, USG, TS_pct, MPG")
    print()


# ---------------------------------------------------------------------------
# Recruiting rankings
# ---------------------------------------------------------------------------

def fetch_recruiting(season: int):
    """Print instructions for recruiting composite rankings."""
    print(f"\n  RECRUITING RANKINGS INSTRUCTIONS (season {season}):")
    print(f"    247Sports composite team rankings:")
    print(f"      https://247sports.com/Season/{season - 1}-Basketball/CompositeTeamRankings/")
    print(f"    Expected columns: Season, TeamName, Composite, NumCommits, Rank")
    print(f"    Save as: {os.path.join(EXTERNAL_DIR, f'recruiting_{season}.csv')}")
    print()


# ---------------------------------------------------------------------------
# NBA Draft prospect rankings
# ---------------------------------------------------------------------------

def fetch_nba_draft_prospects(season: int):
    """Print instructions for NBA Draft prospect rankings."""
    print(f"\n  NBA DRAFT PROSPECTS INSTRUCTIONS (season {season}):")
    print(f"    ESPN draft board: https://www.espn.com/nba/draft/tracker")
    print(f"    Consensus prospect rankings encode star-player presence,")
    print(f"    which is a powerful predictor of tournament deep runs.")
    print(f"    Expected columns: Season, PlayerName, School, DraftRound, DraftPick")
    print(f"    Save as: {os.path.join(EXTERNAL_DIR, f'nba_prospects_{season}.csv')}")
    print()


# ---------------------------------------------------------------------------
# Map-only mode (for files downloaded manually)
# ---------------------------------------------------------------------------

def map_only(file_path: str, name_col: str = "Team", gender: str = "M"):
    """Read an external CSV, add TeamID column via name mapping, and overwrite."""
    if not os.path.exists(file_path):
        print(f"ERROR: file not found: {file_path}")
        return
    df = pd.read_csv(file_path)
    if name_col not in df.columns:
        print(f"ERROR: column '{name_col}' not found in {file_path}")
        print(f"Available columns: {list(df.columns)}")
        return
    df = map_team_names(df, name_col, gender=gender)
    df.to_csv(file_path, index=False)
    print(f"  Mapped TeamIDs saved to {file_path} ({df['TeamID'].notna().sum()} matched).")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fetch / stage external data for NCAA tournament prediction."
    )
    parser.add_argument("--season", type=int, default=2026,
                        help="Season year (default: 2026)")
    parser.add_argument("--map-only", action="store_true",
                        help="Only apply team-name mapping to an existing file")
    parser.add_argument("--file", type=str, default=None,
                        help="Path to the file to map (used with --map-only)")
    parser.add_argument("--name-col", type=str, default="Team",
                        help="Column name containing team names (default: 'Team')")
    parser.add_argument("--gender", choices=["M", "W"], default="M")
    parser.add_argument("--source", choices=[
        "all", "barttorvik", "net", "kenpom", "vegas",
        "players", "recruiting", "draft"
    ], default="all", help="Which data source to fetch (default: all)")
    args = parser.parse_args()

    if args.map_only:
        if not args.file:
            print("ERROR: --file is required with --map-only")
            sys.exit(1)
        map_only(args.file, args.name_col, args.gender)
        return

    season = args.season
    print(f"\n{'='*65}")
    print(f" External Data Fetcher – Season {season}")
    print(f"{'='*65}\n")
    print(f" External files will be written to: {EXTERNAL_DIR}\n")

    sources = {
        "barttorvik": fetch_barttorvik,
        "net": fetch_net_rankings,
        "kenpom": fetch_kenpom,
        "vegas": fetch_vegas_lines,
        "players": fetch_player_stats,
        "recruiting": fetch_recruiting,
        "draft": fetch_nba_draft_prospects,
    }

    if args.source == "all":
        for name, fn in sources.items():
            print(f"--- {name.upper()} ---")
            fn(season)
    else:
        sources[args.source](season)

    print("\nDone.  Place any manually downloaded files in:")
    print(f"  {EXTERNAL_DIR}")
    print("Then re-run training:  python -m src.train\n")


if __name__ == "__main__":
    main()
