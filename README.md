# cmuMarchMadness-ML

[![CI](https://github.com/Qrytics/cmuMarchMadness-ML/actions/workflows/ci.yml/badge.svg)](https://github.com/Qrytics/cmuMarchMadness-ML/actions/workflows/ci.yml)
[![GitHub Pages](https://github.com/Qrytics/cmuMarchMadness-ML/actions/workflows/pages.yml/badge.svg)](https://qrytics.github.io/cmuMarchMadness-ML/)

A machine learning system for predicting NCAA basketball tournament winners, built for the **CMU Second Annual March Madness Machine Learning (MMML) Competition**.

> **⏰ Competition Deadline: March 17, 2026 at Noon EDT**
>
> Brackets must be submitted to your team captain **before** the tournament bracket is revealed on Selection Sunday (March 15, 2026).

---

## 🏀 Current State — What's Already Done

| Component | Status |
|-----------|--------|
| Real Kaggle data | ✅ Extracted to `data/raw/` (124k+ real games) |
| Men's model | ✅ **Trained on real data — 72.3% CV accuracy** |
| Women's model | ✅ **Trained on real data — 75.1% CV accuracy** |
| 2026 Seeds | ✅ Added to `data/raw/` (from actual Selection Sunday bracket) |
| Men's predictions | ✅ Generated — `predictions/MNCAATourneyPredictions.csv` (66,430 real matchup pairs) |
| Women's predictions | ✅ Generated — `predictions/WNCAATourneyPredictions.csv` (65,703 real matchup pairs) |
| Kaggle submission | ✅ `predictions/submission.csv` (132,133 rows, aligned to Stage 2) |
| Men's bracket | ✅ `predictions/MBracket2026.csv` — round-by-round results |
| Women's bracket | ✅ `predictions/WBracket2026.csv` — round-by-round results |
| Live dashboard | ✅ Running at [qrytics.github.io/cmuMarchMadness-ML](https://qrytics.github.io/cmuMarchMadness-ML/) |

### Current Model Accuracy (real NCAA data, walk-forward CV):

| Tournament | Accuracy | AUC | vs. Synthetic Data |
|-----------|----------|-----|-------------------|
| Men's | **72.3%** ± 3.2% | 0.771 | +17% improvement |
| Women's | **75.1%** ± 4.8% | 0.848 | +20% improvement |

---

## 🎯 Predicted 2026 Champions

Based on the trained model and the actual Selection Sunday bracket:

| Tournament | Predicted Champion | Runner-up |
|-----------|-------------------|-----------|
| 🏀 Men's | **Duke** | Florida |
| 🏀 Women's | **Connecticut (UConn)** | Texas |

Full round-by-round bracket predictions: `predictions/MBracket2026.csv` and `predictions/WBracket2026.csv`

---

## ⚡ Quick Start — What YOU Need to Do RIGHT NOW

The model is trained and predictions are generated. **You just need to submit the files:**

### Step 1 — Submit the prediction files

Submit **both** files to your team captain before **March 17, 2026 at Noon EDT**:
- `predictions/MNCAATourneyPredictions.csv` — Men's all-pairs predictions
- `predictions/WNCAATourneyPredictions.csv` — Women's all-pairs predictions

These files already use real Kaggle team IDs and are aligned to the Stage 2 format.

### Step 2 — Push to update the dashboard

```bash
git add -A
git commit -m "Update 2026 bracket predictions and model metrics"
git push
```

GitHub Pages will automatically deploy the updated dashboard within ~1 minute.

---

## 🔄 If You Want to Retrain (Optional — Already Done)

The model is already trained on real data. Only retrain if you want to:
- Run hyperparameter tuning (`--tune`) for better accuracy
- Add new external data
- Re-run after the 2026 regular season is over

### How to Retrain

```bash
# Install dependencies
pip install -r requirements.txt

# Step 1: Train both models with real data (2–5 minutes)
python -m src.train --data-dir data/raw

# Optional: with Optuna hyperparameter tuning (~30 min, can improve accuracy by +2-4%)
python -m src.train --data-dir data/raw --tune --tune-trials 50

# Step 2: Regenerate all-pairs predictions (1–2 minutes)
python -m src.predict --data-dir data/raw

# Step 3: Simulate the actual 2026 tournament bracket
python scripts/run_bracket.py --data-dir data/raw

# Step 4: Export updated data to the dashboard
python scripts/export_site_data.py

# Step 5: Push everything to GitHub
git add -A && git commit -m "Retrain with real data" && git push
```

The `--tune` flag uses Optuna Bayesian optimization to automatically find better hyperparameters. Expected improvement: +2–4% accuracy, requires ~30 minutes per gender.

---

## 🏟️ Running the Bracket Simulation

The bracket simulation takes the all-pairs predictions and simulates the actual 2026 NCAA Tournament bracket round by round.

```bash
# Simulate both brackets (requires predictions/ files to exist)
python scripts/run_bracket.py --data-dir data/raw

# Simulate men's only
python scripts/run_bracket.py --gender M --data-dir data/raw

# Simulate women's only
python scripts/run_bracket.py --gender W --data-dir data/raw
```

**Output files:**
- `predictions/MBracket2026.csv` — Men's bracket: each game with predicted winner
- `predictions/WBracket2026.csv` — Women's bracket
- `docs/data/MBracket2026.json` — Dashboard-ready JSON
- `docs/data/WBracket2026.json`

**What it shows:**
- First Four results (who advances from play-in games)
- First Round (Round of 64)
- Second Round (Round of 32)
- Sweet 16, Elite Eight, Final Four, Championship
- Predicted national champion with full bracket path

---

## 📊 Sending Results to the Dashboard

After training/predicting/simulating, update the live dashboard with one command:

```bash
python scripts/export_site_data.py
git add -A && git commit -m "Update predictions and dashboard" && git push
```

This copies all result files to `docs/data/` and `docs/assets/`. GitHub Pages automatically deploys the updated site within ~1 minute.

**Dashboard URL:** https://qrytics.github.io/cmuMarchMadness-ML/

---

## 📁 Project Structure

```
cmuMarchMadness-ML/
├── data/
│   ├── raw/                          # Real Kaggle data (extracted from competition zip)
│   ├── external/                     # External enrichment data (KenPom, Barttorvik, NET, etc.)
│   └── sample/                       # Synthetic sample data (for CI testing only)
├── docs/                             # GitHub Pages website (auto-deployed)
│   ├── data/                         # Model metrics + predictions (updated by export_site_data.py)
│   └── assets/                       # Evaluation plots
├── models/                           # Trained model .pkl files + metric JSONs
├── predictions/                      # Competition submission files
│   ├── MNCAATourneyPredictions.csv   # 66,430 men's matchup predictions ← SUBMIT THIS
│   ├── WNCAATourneyPredictions.csv   # 65,703 women's matchup predictions ← AND THIS
│   ├── MBracket2026.csv             # Men's round-by-round bracket simulation
│   ├── WBracket2026.csv             # Women's round-by-round bracket simulation
│   ├── MKaggleSubmission.csv        # Kaggle format (ID, Pred columns)
│   ├── WKaggleSubmission.csv
│   └── submission.csv               # Combined M+W aligned to Stage 2 (132,133 rows)
├── scripts/
│   ├── run_bracket.py               # ★ NEW: Simulate actual 2026 tournament bracket
│   ├── export_site_data.py          # Copy model outputs to docs/data/ for dashboard
│   ├── generate_sample_data.py      # Generate synthetic training data (CI only)
│   ├── download_data.py             # Download real data from Kaggle
│   └── fetch_external_data.py       # Stage KenPom/Barttorvik/NET data
├── src/
│   ├── data_loader.py               # Data loading and preprocessing
│   ├── feature_engineering.py       # ML feature computation (103 features)
│   ├── model.py                     # Ensemble model (XGBoost + LightGBM + LR)
│   ├── train.py                     # Training orchestration
│   ├── predict.py                   # All-pairs prediction generation
│   ├── evaluate.py                  # Backtesting and evaluation
│   └── bracket.py                   # ★ UPDATED: Bracket simulation (handles play-in for any seed)
├── tests/
│   ├── test_features.py
│   ├── test_model.py
│   └── test_bracket.py
└── requirements.txt
```

---

## 🤖 Model Architecture

The prediction system uses an **ensemble of three models**:

| Model | Weight | Why it helps |
|-------|--------|-------------|
| XGBoost | 40% | Captures non-linear feature interactions; handles missing rankings gracefully |
| LightGBM | 40% | Efficient leaf-wise growth; especially strong on ordinal ranking features |
| Logistic Regression | 20% | Calibrated linear baseline; stabilizes probability estimates |

**103 differential features per matchup**, including:
- Box-score stats (win %, FG%, rebounds, assists, turnovers)
- Advanced efficiency metrics (NetEff, Pythagorean expectation, eFG%)
- Rankings (Massey ordinals from 30+ systems including NET, KenPom, SAG)
- Strength of Schedule
- KenPom (AdjEM, AdjO, AdjD, tempo, luck, SOS)
- Barttorvik T-Rank (Barthag, AdjOE, AdjDE, eFG%, rebound rates)
- NCAA NET ranking
- Tournament history, coaching tenure, conference strength
- Seed information + historical upset probabilities
- Recruiting composite, player quality, NBA draft prospects

### Training Methodology

- **Walk-forward cross-validation**: train on all seasons through year N, validate on year N+1
- Models trained **separately** for men's and women's tournaments
- **No data leakage**: predictions for year N only use data from years before N
- **Calibration**: isotonic regression calibrates raw ensemble probabilities
- **Optional tuning**: Optuna Bayesian optimization for hyperparameter search

---

## 📋 Submission Format

The competition requires two CSV files with exactly these columns:

```csv
WTeamID,LTeamID
1181,1373
1181,1326
...
```

- `WTeamID` — team predicted to **win**
- `LTeamID` — team predicted to **lose**
- Every possible pair must appear (not just bracket matchups)

| File | Rows | Team ID range |
|------|------|---------------|
| `MNCAATourneyPredictions.csv` | 66,430 — all C(381,2) pairs | 1000–1999 |
| `WNCAATourneyPredictions.csv` | 65,703 — all C(379,2) pairs | 3000–3999 |

---

## ⚙️ CI/CD Pipeline

Three GitHub Actions workflows automate the pipeline:

| Workflow | Trigger | What it does |
|----------|---------|-------------|
| `ci.yml` | Every push / PR | Runs tests, validates data schemas |
| `train.yml` | Manual / Weekly | Downloads data, retrains models, generates predictions |
| `pages.yml` | Push to `main` | Deploys `docs/` to GitHub Pages |

### Setting Up Kaggle Secrets (for automated retraining)

In GitHub: **Settings → Secrets → Actions → New repository secret**
- `KAGGLE_USERNAME` — Your Kaggle username
- `KAGGLE_KEY` — Your Kaggle API key

---

## 🏆 Competition Resources

- **[CMU MMML Competition Page](https://www.cs.cmu.edu/~reids/mmml/)** — Official rules and submission instructions
- **[Kaggle March Machine Learning Mania](https://www.kaggle.com/c/march-machine-learning-mania-2026)** — Training data source
- **[Live Dashboard](https://qrytics.github.io/cmuMarchMadness-ML/)** — Model metrics and bracket simulator
- **Deadline: March 17, 2026 at 12:00 PM EDT**

---

## 🔬 Improving Accuracy Further

If you have time before the deadline, here are ranked options:

| Option | Effort | Expected gain |
|--------|--------|---------------|
| Run with `--tune` flag | ~30 min | +2–4% accuracy |
| Add more external data | 1–2 hrs | +1–2% |
| Increase `--tune-trials 100` | ~1 hr | +1–2% |
| Add more historical seasons | Ongoing | +0.5% |

The biggest gains come from **running with `--tune`** once:

```bash
python -m src.train --data-dir data/raw --tune --tune-trials 50
python -m src.predict --data-dir data/raw
python scripts/run_bracket.py --data-dir data/raw
python scripts/export_site_data.py
git add -A && git commit -m "Tuned model predictions" && git push
```


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

