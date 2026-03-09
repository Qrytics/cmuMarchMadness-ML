# cmuMarchMadness-ML

[![CI](https://github.com/Qrytics/cmuMarchMadness-ML/actions/workflows/ci.yml/badge.svg)](https://github.com/Qrytics/cmuMarchMadness-ML/actions/workflows/ci.yml)
[![GitHub Pages](https://github.com/Qrytics/cmuMarchMadness-ML/actions/workflows/pages.yml/badge.svg)](https://qrytics.github.io/cmuMarchMadness-ML/)

A machine learning system for predicting NCAA basketball tournament winners, built for the **CMU Second Annual March Madness Machine Learning (MMML) Competition**.

> **Competition Deadline:** March 17, 2026 at Noon EDT

## 🏀 Project Overview

This project implements ensemble ML models that predict the winner of any matchup between two NCAA Division I basketball teams. Predictions are used to fill tournament brackets for both the **Men's** and **Women's** NCAA tournaments.

### Competition Structure
- **4 competitions**: Men's regular bracket, Men's progressive bracket, Women's regular bracket, Women's progressive bracket
- **Scoring**: 1 pt (R1) → 2 pts (R2) → 4 pts (Elite 8) → 8 pts (QF) → 16 pts (SF) → 32 pts (Championship)
- **Perfect bracket**: 196 points (4 play-in + 32+16+8+4+2+1 rounds)
- **Submission format**: `MTourneyPredictions.csv` and `WTourneyPredictions.csv`

## 🌐 Live Dashboard

**[View the ML Dashboard & Bracket Simulator](https://qrytics.github.io/cmuMarchMadness-ML/)**

The GitHub Pages site includes:
- 📊 Model performance metrics and walk-forward backtesting results
- 🏆 Interactive bracket simulator for the 2026 tournament
- 🔬 Feature importance charts and model architecture details
- 📥 Download links for prediction CSV files

## 🤖 Model Architecture

The prediction system uses an **ensemble of three models**:

| Model | Weight | Description |
|-------|--------|-------------|
| XGBoost | 40% | Gradient-boosted trees; captures non-linear interactions |
| LightGBM | 40% | Leaf-wise growth; efficient on tabular data |
| Logistic Regression | 20% | Calibrated linear baseline; prevents overfitting |

### Features (28 differential features per matchup)
- **Win percentage** and **point differential** (season averages)
- **Shooting efficiency**: FG%, 3P%, FT%
- **Rebounding**: offensive + defensive rebounds
- **Playmaking**: assists, turnovers, steals, blocks, fouls
- **Rankings**: Massey ordinal rankings (NET, KPI, KenPom, SAG)
- **Tournament context**: seed difference, higher-seed flag

### Training Methodology
- Walk-forward cross-validation (train on seasons 2010–N, validate on season N+1)
- Both men's and women's models trained separately
- No data leakage — only historical stats used for each prediction year

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate Sample Data (or use real Kaggle data)

```bash
# Generate synthetic training data
python scripts/generate_sample_data.py

# OR download real NCAA data from Kaggle (requires API credentials)
# Get token from: https://www.kaggle.com/settings/account
python scripts/download_data.py
```

### 3. Train Models

```bash
# Train both Men's and Women's models
python -m src.train

# Train only Men's
python -m src.train --gender M

# Train with real Kaggle data
python -m src.train --data-dir data/raw
```

### 4. Evaluate Historical Performance

```bash
# Backtest on historical tournament seasons
python -m src.evaluate

# Evaluate specific seasons
python -m src.evaluate --gender M --seasons 2022 2023 2024
```

### 5. Generate Predictions

```bash
# Generate all 72,390 men's + 71,631 women's matchup predictions
python -m src.predict

# Output: predictions/MTourneyPredictions.csv
#         predictions/WTourneyPredictions.csv
```

### 6. Update the Dashboard

```bash
# Export model data to the GitHub Pages site
python scripts/export_site_data.py
```

## 📁 Project Structure

```
cmuMarchMadness-ML/
├── .github/
│   └── workflows/
│       ├── ci.yml           # Run tests on every push/PR
│       ├── train.yml        # Train models (manual/weekly)
│       └── pages.yml        # Deploy GitHub Pages
├── data/
│   ├── raw/                 # Real Kaggle data (not committed)
│   └── sample/              # Synthetic sample data for testing
├── docs/                    # GitHub Pages site
│   ├── index.html           # Dashboard
│   ├── bracket.html         # Bracket simulator
│   ├── model.html           # Model details
│   ├── data/                # JSON/CSV data for site
│   └── assets/              # CSS, JS, evaluation plots
├── models/                  # Trained model files (.pkl)
├── predictions/             # Competition submission files
│   ├── MTourneyPredictions.csv   # 72,390 men's predictions
│   └── WTourneyPredictions.csv   # 71,631 women's predictions
├── scripts/
│   ├── generate_sample_data.py   # Generate synthetic training data
│   ├── download_data.py          # Download from Kaggle
│   └── export_site_data.py       # Copy metrics to docs/
├── src/
│   ├── data_loader.py       # Data loading and preprocessing
│   ├── feature_engineering.py    # ML feature computation
│   ├── model.py             # Ensemble model implementation
│   ├── train.py             # Training orchestration
│   ├── predict.py           # Prediction generation
│   ├── evaluate.py          # Backtesting and evaluation
│   └── bracket.py           # Bracket simulation and scoring
├── tests/
│   ├── test_features.py     # Feature engineering tests
│   ├── test_model.py        # Model training/inference tests
│   └── test_bracket.py      # Bracket simulation tests
└── requirements.txt
```

## 🔬 Using Real Kaggle Data

For production-quality predictions, use real NCAA basketball data:

1. **Get Kaggle API credentials**: Go to [kaggle.com/settings](https://www.kaggle.com/settings/account) → Create New Token
2. **Place credentials**: Save `kaggle.json` to `~/.kaggle/kaggle.json`
3. **Download data**: `python scripts/download_data.py`

Key files from the Kaggle "March Machine Learning Mania" competition:
- `MRegularSeasonDetailedResults.csv` — Full box scores for every regular season game
- `MNCAATourneyDetailedResults.csv` — Tournament game results (for training labels)
- `MSeeds.csv` — Tournament bracket seedings by year
- `MMasseyOrdinals.csv` — Team rankings from 30+ rating systems
- (Corresponding `W*` files for women's tournament)

## ⚙️ CI/CD Pipeline

Three GitHub Actions workflows handle automation:

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| `ci.yml` | Every push/PR | Run tests, validate data, check prediction format |
| `train.yml` | Manual / Weekly | Retrain models, generate predictions, commit results |
| `pages.yml` | Push to main | Deploy dashboard to GitHub Pages |

### Setting Up Kaggle Secrets (for CI)

Add these secrets to your GitHub repository settings:
- `KAGGLE_USERNAME` — Your Kaggle username
- `KAGGLE_KEY` — Your Kaggle API key

## 🧪 Running Tests

```bash
python -m pytest tests/ -v
```

The test suite covers:
- Data loading and schema validation
- Feature engineering correctness
- Model training and inference
- Bracket simulation and scoring
- Prediction file format validation (72,390 M / 71,631 W rows)

## �� Submission Format

The competition requires CSV files with columns `WTeamID` and `LTeamID`:

```csv
WTeamID,LTeamID
1101,1102
1101,1103
...
```

- **Men's**: `MTourneyPredictions.csv` — 72,390 rows (all C(381,2) pairs)
- **Women's**: `WTourneyPredictions.csv` — 71,631 rows (all C(379,2) pairs)
- Team IDs: Men's = 1000–1999, Women's = 3000–3999

## 🏆 Competition Resources

- [CMU MMML Competition Page](https://www.cs.cmu.edu/~reids/mmml/)
- [Kaggle March Machine Learning Mania](https://www.kaggle.com/c/march-machine-learning-mania-2024)
- Deadline: March 17, 2026 at 12:00 PM EDT
