### Two-Stage Uncertainty Modeling with Coverage-Constrained Optimization

**Leaderboard Rank:** 40th (Individual)

**Participants:** 691 Teams

**Task:** Prediction Interval Estimation

**Metric:** Winkler Score with Coverage Constraint

---

## Overview

This project presents a **robust uncertainty-aware regression pipeline** designed to produce calibrated prediction intervals for housing prices.

Instead of predicting only a point estimate, the model generates **lower and upper bounds** for each property’s sale price while ensuring:

* high predictive accuracy
* calibrated uncertainty estimates
* minimum coverage guarantees

The solution achieved **40th place as an individual competitor** among 691 teams.

---

## 🎯 Problem Objective

Given structured real estate transaction data, predict:

* **lower bound** of sale price
* **upper bound** of sale price

while minimizing the **Winkler Score** under a **coverage constraint**.

### Evaluation Metric

The competition uses the **Winkler Score**, which penalizes:

* wide prediction intervals
* missed coverage (true value outside interval)

Lower scores indicate better calibrated and tighter intervals.

---

## Solution Strategy

The solution uses a **Two-Stage Uncertainty Modeling approach**:

### Stage 1 — Point Prediction

Predict sale price using gradient boosting.

### Stage 2 — Uncertainty Estimation

Model residual variance to estimate prediction uncertainty.

### Stage 3 — Interval Construction

Prediction intervals are constructed using:

```
Lower = ŷ − γ₀ √σ̂²
Upper = ŷ + γ₁ √σ̂²
```

### Stage 4 — Coverage-Constrained Optimization

Gamma parameters are optimized to:

* minimize Winkler Score
* maintain ≥ 90% coverage

---

## Key Techniques

### ✔ Feature Engineering

Extensive domain-informed feature construction:

#### Time Features

* month, quarter, weekday
* cyclical encoding (sin/cos)
* start/end of month flags

#### Property Value Features

* total value & ratios
* value per square foot
* land/building ratios
* log transformations

#### Spatial Context Features

* latitude & longitude
* KNN neighborhood price statistics

---

### ✔ KNN Spatial Aggregations

Local neighborhood information improves calibration:

* mean nearby sale price
* neighborhood price variance
* average spatial distance

Computed using multiple neighborhood sizes (k = 5, 10, 15).

---

### ✔ Two-Stage Uncertainty Modeling

Residual variance is modeled explicitly:

* Stage-1 predicts price
* Stage-2 predicts squared residuals
* ensures heteroscedastic uncertainty modeling

This captures real-world price volatility patterns.

---

### ✔ Coverage-Constrained Optimization

Gamma scaling parameters are tuned to satisfy:

* minimum coverage threshold (90%)
* optimal Winkler score

This prevents overly narrow intervals that fail coverage.

---

### ✔ Cross-Validation Stability

* 7-fold cross validation
* out-of-fold residual modeling
* robust gamma optimization

Ensures generalization and leaderboard stability.

---

## Why This Works

Real estate prices exhibit:

* heteroscedastic noise
* spatial clustering
* temporal patterns
* structural non-linearity

This pipeline addresses all four simultaneously.

---

## Model Architecture

**Stage 1 Model:** XGBoost Regressor
**Stage 2 Model:** XGBoost Gamma Regression

Key regularization choices:

* shallow trees for generalization
* subsampling & column sampling
* L1/L2 regularization
* conservative learning rates

---

## Performance Highlights

✔ High coverage reliability
✔ Tight prediction intervals
✔ Strong generalization
✔ Robust leaderboard stability

**Final Ranking:**
🏆 **40th / 691 teams (Individual)**

---

## 📁 Project Structure

```
├── dataset.csv
├── test.csv
├── kaggle_solution.ipynb
└── README.md
```

---

## ▶️ How to Run

### 1️⃣ Install dependencies

```bash
pip install pandas numpy scikit-learn xgboost
```

### 2️⃣ Place datasets

```
dataset.csv
test.csv
```

### 3️⃣ Run

```bash
python kaggle_solution.ipynb
```

### 4️⃣ Output

```
submission.csv
```

---

## 🔍 Prediction Interval Example

| Property | Lower | Upper |
| -------- | ----- | ----- |
| House A  | 420k  | 510k  |
| House B  | 310k  | 385k  |

The interval reflects both prediction and uncertainty.
