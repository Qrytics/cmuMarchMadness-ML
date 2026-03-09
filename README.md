# cmuMarchMadness-ML

[![CI](https://github.com/Qrytics/cmuMarchMadness-ML/actions/workflows/ci.yml/badge.svg)](https://github.com/Qrytics/cmuMarchMadness-ML/actions/workflows/ci.yml)
[![GitHub Pages](https://github.com/Qrytics/cmuMarchMadness-ML/actions/workflows/pages.yml/badge.svg)](https://qrytics.github.io/cmuMarchMadness-ML/)

A machine learning system for predicting NCAA basketball tournament winners, built for the **CMU Second Annual March Madness Machine Learning (MMML) Competition**.

> **⏰ Competition Deadline: March 17, 2026 at Noon EDT**
>
> Brackets must be submitted to your team captain **before** the tournament bracket is revealed on Selection Sunday (March 15, 2026).

---

## 🏀 Project Overview

This project implements an ensemble ML model that predicts the winner of **any matchup** between two NCAA Division I basketball teams. Our predictions fill out brackets for **four competitions**:

| Competition | Bracket Type | Teams |
|-------------|-------------|-------|
| Men's Regular | Pre-tournament, all rounds locked in | 68 |
| Men's Progressive | Updated after each round with real results | 68 |
| Women's Regular | Pre-tournament, all rounds locked in | 68 |
| Women's Progressive | Updated after each round with real results | 68 |

### Scoring System

| Round | Points per correct pick | Games |
|-------|------------------------|-------|
| Round of 64 | 1 | 32 |
| Round of 32 | 2 | 16 |
| Sweet 16 | 4 | 8 |
| Elite 8 | 8 | 4 |
| Final Four | 16 | 2 |
| Championship | 32 | 1 |

**Perfect bracket = 196 points.** Later rounds are worth exponentially more — getting the champion right (32 pts) equals all 32 first-round picks combined.

---

## ❓ Is the Model Already Running?

**Yes — the model is already trained and predictions have already been generated.** Here is the current state:

| Component | Status |
|-----------|--------|
| Men's model | ✅ Trained (see `models/`) |
| Women's model | ✅ Trained (see `models/`) |
| Men's predictions | ✅ Generated — `predictions/MTourneyPredictions.csv` (72,390 rows) |
| Women's predictions | ✅ Generated — `predictions/WTourneyPredictions.csv` (71,631 rows) |
| Live dashboard | ✅ Running at [qrytics.github.io/cmuMarchMadness-ML](https://qrytics.github.io/cmuMarchMadness-ML/) |

**⚠️ Current Limitation:** The model was trained on **synthetic data** (not real NCAA statistics). This means predictions are valid in format but not optimized for accuracy. See [Using Real Kaggle Data](#-using-real-kaggle-data) to dramatically improve prediction quality.

---

## 🎯 What Do YOU Need to Do?

### Option A — Use predictions as-is (quick, minimal effort)

The prediction files are already committed and valid for submission:

1. Find `predictions/MTourneyPredictions.csv` and `predictions/WTourneyPredictions.csv`
2. Submit to your team captain before March 17, 2026

The CI pipeline has already validated the format (72,390 M rows, 71,631 W rows with correct team ID ranges).

### Option B — Improve the model with real data (recommended for better scores)

Follow the [Pre-Competition Preparation Checklist](#-pre-competition-preparation-checklist) to retrain with real NCAA data and potentially reach 65–75% accuracy.

---

## ✅ Pre-Competition Preparation Checklist

Complete these steps **before March 17, 2026** to maximize your bracket score:

### Step 1 — Get Kaggle API credentials

1. Create an account at [kaggle.com](https://kaggle.com) (free)
2. Go to [kaggle.com/settings/account](https://www.kaggle.com/settings/account)
3. Scroll to **"API"** → click **"Create New Token"**
4. A file `kaggle.json` downloads automatically:
   ```json
   {"username": "yourusername", "key": "your-api-key-here"}
   ```
5. Save it to your home directory:
   ```bash
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
   chmod 600 ~/.kaggle/kaggle.json   # required on Linux/Mac
   ```

### Step 2 — Install dependencies

```bash
git clone https://github.com/Qrytics/cmuMarchMadness-ML
cd cmuMarchMadness-ML
pip install -r requirements.txt
```

### Step 3 — Download real NCAA data

```bash
python scripts/download_data.py
```

This downloads the **Kaggle March Machine Learning Mania** dataset into `data/raw/`. Key files:

| File | Contents |
|------|----------|
| `MRegularSeasonDetailedResults.csv` | Full box scores for every regular season game (men's) |
| `MNCAATourneyDetailedResults.csv` | Tournament results used as training labels |
| `MSeeds.csv` | Tournament bracket seedings by year |
| `MMasseyOrdinals.csv` | Team rankings from 30+ systems (NET, KPI, KenPom, SAG, ...) |
| `WRegularSeasonDetailedResults.csv` | Women's regular season games |
| `WNCAATourneyDetailedResults.csv` | Women's tournament results |
| `WSeeds.csv` | Women's seedings |

> If you do not have Kaggle credentials yet, use synthetic data for now:
> `python scripts/generate_sample_data.py`

### Step 4 — Retrain models with real data

```bash
# Train both men's and women's models (takes 2–5 minutes)
python -m src.train --data-dir data/raw

# Or train each gender separately:
python -m src.train --gender M --data-dir data/raw
python -m src.train --gender W --data-dir data/raw
```

With real data you should see **65–75% CV accuracy** (up from ~55% on synthetic data).

### Step 5 — Evaluate performance (optional but informative)

```bash
# Backtest on historical tournament seasons to see how the model performs
python -m src.evaluate --data-dir data/raw

# Evaluate specific seasons only:
python -m src.evaluate --gender M --seasons 2022 2023 2024 --data-dir data/raw
```

This generates plots in `docs/assets/` and bracket score data in `models/`.

### Step 6 — Generate final predictions

```bash
# Generate all possible matchup predictions
python -m src.predict --data-dir data/raw

# Output files:
#   predictions/MTourneyPredictions.csv  (72,390 rows)
#   predictions/WTourneyPredictions.csv  (71,631 rows)
```

### Step 7 — Update the live dashboard

```bash
python scripts/export_site_data.py
git add -A && git commit -m "Update predictions and model metrics" && git push
```

GitHub Actions deploys the updated dashboard automatically within ~1 minute.

### Step 8 — Submit predictions

Submit both files to your team captain before **March 17, 2026 at Noon EDT**:
- `predictions/MTourneyPredictions.csv`
- `predictions/WTourneyPredictions.csv`

---

## 🌐 Live Dashboard

**[View the ML Dashboard & Bracket Simulator](https://qrytics.github.io/cmuMarchMadness-ML/)**

| Page | What you will find |
|------|--------------------|
| [Dashboard](https://qrytics.github.io/cmuMarchMadness-ML/) | Model accuracy, backtest scores, feature importance, scoring system |
| [Bracket Simulator](https://qrytics.github.io/cmuMarchMadness-ML/bracket.html) | Interactive 2026 bracket with predicted winners, download links |
| [Model Details](https://qrytics.github.io/cmuMarchMadness-ML/model.html) | Hyperparameters, feature engineering docs, CI/CD info |
| [Getting Started](https://qrytics.github.io/cmuMarchMadness-ML/getting-started.html) | Visual step-by-step setup guide |

---

## 🤖 Model Architecture

The prediction system uses an **ensemble of three models**:

| Model | Weight | Why it helps |
|-------|--------|-------------|
| XGBoost | 40% | Captures non-linear feature interactions; handles missing rankings gracefully |
| LightGBM | 40% | Efficient leaf-wise growth; especially strong on ordinal ranking features |
| Logistic Regression | 20% | Calibrated linear baseline; stabilizes probability estimates |

Each model outputs a win probability (0–1). The three probabilities are combined via weighted average. The team with probability ≥ 50% is predicted the winner.

### Features (28 differential features per matchup)

All features are computed as **(Team A stat) − (Team B stat)** so positive values always favor Team A:

| Category | Features |
|----------|---------|
| Record | Win percentage, point differential |
| Scoring | Points for, points against |
| Shooting | FG%, 3-point%, free throw% |
| Rebounding | Offensive rebounds, defensive rebounds |
| Playmaking | Assists, turnovers, steals, blocks, fouls |
| Rankings | Average and best Massey ordinal rank (NET, KPI, KenPom, SAG) |
| Tournament context | Seed difference, higher-seed indicator |

### Training Methodology

- **Walk-forward cross-validation**: train on all seasons through year N, validate on year N+1
- Models trained **separately** for men's and women's tournaments
- **No data leakage**: predictions for year N only use data from years before N

---

## 📁 Project Structure

```
cmuMarchMadness-ML/
├── .github/
│   └── workflows/
│       ├── ci.yml                    # Tests run on every push/PR
│       ├── train.yml                 # Retrain models (manual or weekly)
│       └── pages.yml                 # Deploy GitHub Pages dashboard
├── data/
│   ├── raw/                          # Real Kaggle data (gitignored — download locally)
│   └── sample/                       # Synthetic sample data (committed for CI)
├── docs/                             # GitHub Pages website
│   ├── index.html                    # Main dashboard
│   ├── bracket.html                  # Interactive bracket simulator
│   ├── model.html                    # Model details & hyperparameters
│   ├── getting-started.html          # Step-by-step setup guide
│   ├── data/                         # JSON metrics + prediction CSVs for site
│   └── assets/                       # CSS, evaluation plots
├── models/                           # Trained model .pkl files + metric JSONs
├── predictions/                      # Competition submission files
│   ├── MTourneyPredictions.csv       # 72,390 men's matchup predictions ← SUBMIT THIS
│   └── WTourneyPredictions.csv       # 71,631 women's matchup predictions ← AND THIS
├── scripts/
│   ├── generate_sample_data.py       # Generate synthetic training data
│   ├── download_data.py              # Download real data from Kaggle
│   └── export_site_data.py           # Copy model outputs to docs/data/
├── src/
│   ├── data_loader.py                # Data loading and preprocessing
│   ├── feature_engineering.py        # ML feature computation
│   ├── model.py                      # Ensemble model definition
│   ├── train.py                      # Training orchestration
│   ├── predict.py                    # Prediction generation
│   ├── evaluate.py                   # Backtesting and evaluation
│   └── bracket.py                    # Bracket simulation and scoring
├── tests/
│   ├── test_features.py              # Feature engineering tests
│   ├── test_model.py                 # Model training/inference tests
│   └── test_bracket.py              # Bracket simulation tests
└── requirements.txt
```

---

## 🔬 Using Real Kaggle Data

Real NCAA data dramatically improves prediction accuracy:

| Data source | Expected CV accuracy |
|-------------|---------------------|
| Synthetic sample data | ~55% (essentially guessing) |
| Real Kaggle NCAA data | **65–75%** |

### One-time setup

```bash
# 1. Get Kaggle token: kaggle.com/settings/account → Create New Token
mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

# 2. Download ~500 MB of NCAA data:
python scripts/download_data.py

# 3. Train with real data (~5 minutes):
python -m src.train --data-dir data/raw

# 4. Generate final competition predictions:
python -m src.predict --data-dir data/raw
```

---

## ⚙️ CI/CD Pipeline

Three GitHub Actions workflows automate the pipeline:

| Workflow | Trigger | What it does |
|----------|---------|-------------|
| `ci.yml` | Every push / PR | Runs tests, validates data schemas, checks prediction file format |
| `train.yml` | Manual / Weekly schedule | Downloads data, retrains models, generates predictions, commits results |
| `pages.yml` | Push to `main` | Deploys `docs/` to GitHub Pages |

### Setting Up Kaggle Secrets (to enable automated retraining)

In your GitHub repository: **Settings → Secrets and variables → Actions → New repository secret**

- `KAGGLE_USERNAME` — Your Kaggle username
- `KAGGLE_KEY` — Your Kaggle API key (the string from `kaggle.json`)

Once set, go to **Actions → train.yml → Run workflow** to retrain in the cloud.

---

## 🧪 Running Tests

```bash
python -m pytest tests/ -v
```

Tests cover:
- Data loading and schema validation
- Feature engineering correctness
- Model training and inference
- Bracket simulation and scoring
- Prediction file format (72,390 M rows / 71,631 W rows with correct columns)

---

## 📋 Submission Format

The competition requires two CSV files with exactly these columns:

```csv
WTeamID,LTeamID
1101,1102
1101,1103
...
```

- `WTeamID` — team predicted to **win**
- `LTeamID` — team predicted to **lose**
- Every possible pair must appear (not just bracket matchups)

| File | Rows | Team ID range |
|------|------|---------------|
| `MTourneyPredictions.csv` | 72,390 — all C(381,2) pairs | 1000–1999 |
| `WTourneyPredictions.csv` | 71,631 — all C(379,2) pairs | 3000–3999 |

---

## 🏆 Competition Resources

- **[CMU MMML Competition Page](https://www.cs.cmu.edu/~reids/mmml/)** — Official rules and submission instructions
- **[Kaggle March Machine Learning Mania](https://www.kaggle.com/c/march-machine-learning-mania-2024)** — Training data source
- **[Live Dashboard](https://qrytics.github.io/cmuMarchMadness-ML/)** — Model metrics and bracket simulator
- **Deadline: March 17, 2026 at 12:00 PM EDT**
