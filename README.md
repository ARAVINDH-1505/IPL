# 🏏 IPL Match Outcome & Player Performance Prediction

This project builds a hybrid machine learning system to predict individual player performance (runs and wickets) and estimate the outcome of IPL cricket matches using an ensemble of **PyTorch** and **XGBoost** models.

## 📁 Files

- `IPL.ipynb` – Complete notebook with preprocessing, modeling, ensemble prediction, and visualization.
- `final_merged_match_player_data.csv` – Main dataset containing match and player-level statistics.

## 🔍 Project Overview

This project tackles the problem of predicting:
- Total runs and wickets for each player in a match.
- Match win probability based on team compositions and match conditions.

It includes:
- Role classification (batsman, bowler, allrounder) based on historical behavior.
- Encoding and normalization of teams, players, venues, and seasons.
- Two-stage model training (PyTorch MLP + XGBoost).
- Ensemble prediction using average output of both models.
- Full match simulation for both teams.
- Interactive dashboard using Plotly.

---

## ⚙️ Workflow

### 1. 📊 Data Preprocessing
- Handle missing values
- Encode categorical features using LabelEncoder
- Scale numerical fields like season
- Classify player role using heuristics from `balls_faced` and `legal_deliveries`

### 2. 🧠 Model Training

#### PyTorch Regressors:
- A simple multi-layer perceptron is trained for predicting:
  - `total_runs` for batters
  - `wickets` for bowlers

#### XGBoost Regressors:
- Tree-based ensemble models trained on the same features
- Provide robustness and interpretability

#### Ensemble:
- Final prediction = average of PyTorch + XGBoost outputs

---

## 🎯 Match Simulation
The function `predict_match_ensemble()` simulates a match:
- Takes in team player names, venue, toss decision
- Predicts each player’s expected runs and wickets
- Aggregates total team scores and displays win probabilities

---

## 📈 Visualization Dashboard

Interactive dashboard using **Plotly**:
- Bar plots for player runs & wickets
- Gauge charts comparing team scores
- Launched automatically at the end of prediction

---

