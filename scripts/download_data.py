"""
Download NCAA basketball data from Kaggle for the March Madness ML competition.
Requires a Kaggle API token (kaggle.json) at ~/.kaggle/kaggle.json.

Usage:
    python scripts/download_data.py

Data source: Kaggle "March Machine Learning Mania" competitions.
"""

import os
import zipfile
import shutil

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

# Kaggle competition slugs (updated annually)
COMPETITION_MEN = "march-machine-learning-mania-2026"
COMPETITION_WOMEN = "march-machine-learning-mania-2026"


def check_kaggle():
    """Verify Kaggle API credentials are available."""
    kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")
    env_user = os.environ.get("KAGGLE_USERNAME")
    env_key = os.environ.get("KAGGLE_KEY")

    if not os.path.exists(kaggle_json) and not (env_user and env_key):
        print("ERROR: Kaggle API credentials not found.")
        print("Please either:")
        print("  1. Place your kaggle.json at ~/.kaggle/kaggle.json")
        print("  2. Set KAGGLE_USERNAME and KAGGLE_KEY environment variables")
        print("\nGet your API token from: https://www.kaggle.com/settings/account")
        return False
    return True


def download_competition_data():
    """Download the Kaggle competition data."""
    if not check_kaggle():
        print("\nFalling back to sample data. Run scripts/generate_sample_data.py")
        return False

    try:
        import kaggle
    except ImportError:
        print("ERROR: kaggle package not installed. Run: pip install kaggle")
        return False

    print("Downloading March Machine Learning Mania data from Kaggle...")
    try:
        # Download both competitions (they share data)
        kaggle.api.competition_download_files(
            COMPETITION_MEN,
            path=RAW_DIR,
            quiet=False
        )

        # Extract zip files
        for f in os.listdir(RAW_DIR):
            if f.endswith(".zip"):
                zip_path = os.path.join(RAW_DIR, f)
                print(f"Extracting {f}...")
                with zipfile.ZipFile(zip_path, "r") as z:
                    z.extractall(RAW_DIR)
                os.remove(zip_path)

        print(f"\nData downloaded to: {RAW_DIR}")
        print("Files available:")
        for f in sorted(os.listdir(RAW_DIR)):
            size = os.path.getsize(os.path.join(RAW_DIR, f))
            print(f"  {f}: {size:,} bytes")
        return True

    except Exception as e:
        print(f"ERROR downloading data: {e}")
        print("Falling back to sample data.")
        return False


def verify_data_files():
    """Check that required data files exist."""
    required = [
        "MTeams.csv", "MRegularSeasonDetailedResults.csv",
        "MNCAATourneyDetailedResults.csv", "MSeeds.csv",
        "WTeams.csv", "WRegularSeasonDetailedResults.csv",
        "WNCAATourneyDetailedResults.csv", "WSeeds.csv",
    ]
    optional = ["MRankings.csv", "WRankings.csv", "MMasseyOrdinals.csv"]

    # Check raw dir first, then sample dir
    sample_dir = os.path.join(os.path.dirname(__file__), "..", "data", "sample")
    missing = []

    for f in required:
        raw_path = os.path.join(RAW_DIR, f)
        sample_path = os.path.join(sample_dir, f)
        if not os.path.exists(raw_path) and not os.path.exists(sample_path):
            missing.append(f)

    if missing:
        print(f"Missing required files: {missing}")
        print("Run scripts/generate_sample_data.py to create sample data.")
        return False

    print("All required data files found.")
    return True


if __name__ == "__main__":
    success = download_competition_data()
    verify_data_files()
