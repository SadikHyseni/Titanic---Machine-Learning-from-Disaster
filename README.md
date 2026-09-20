# Titanic - Machine Learning from Disaster

A complete machine learning pipeline for [Kaggle's Titanic competition](https://www.kaggle.com/competitions/titanic), predicting passenger survival from the RMS Titanic sinking.

## Overview

This project takes the classic Titanic dataset and builds an end-to-end pipeline: data exploration, feature engineering, model comparison, and final prediction.

## Approach

**1. Data Analysis**
Checked missing values and survival rates by key variables. Sex and passenger class turned out to be the strongest raw predictors (74% survival for women vs. 19% for men; 63% for 1st class vs. 24% for 3rd class).

**2. Feature Engineering**
- Extracted `Title` (Mr/Mrs/Miss/Master/Rare) from the `Name` field — turned out to be the single most predictive feature, since it encodes gender, age bracket, and social status together
- Combined `SibSp` + `Parch` into `FamilySize` and `IsAlone`
- Filled missing `Age` using group-aware medians (by Title + Pclass) rather than a single global value
- Converted sparse `Cabin` data into a simple `HasCabin` flag
- Bucketed `Age` and `Fare` into bands to help tree-based models split more cleanly
- One-hot encoded categorical variables (`Sex`, `Embarked`, `Title`)

**3. Model Comparison**
Compared several models using 5-fold stratified cross-validation:

| Model | CV Accuracy |
|---|---|
| Decision Tree | 81.25% |
| Logistic Regression | 83.39% |
| Random Forest | 83.72% |
| Gradient Boosting | 83.95% |
| **HistGradientBoosting** | **84.74%** |

Also tested KNN, SVM, Naive Bayes, LDA/QDA, AdaBoost, Extra Trees, a neural network (MLP), and Voting/Stacking ensembles — `HistGradientBoostingClassifier` came out on top.

**4. Final Model**
`HistGradientBoostingClassifier` (scikit-learn), tuned via `GridSearchCV`, trained on the full training set.

## Results

- Cross-validated accuracy: ~84.7%
- Kaggle public leaderboard score: 0.75358

## Files

- `submission.py` — the full pipeline
- `train.csv` / `test.csv` — competition data from Kaggle
- `submission.csv` — final predictions in Kaggle's required format

## Requirements

```bash
pip install pandas numpy scikit-learn
```

## Usage

```bash
python submission.py
```

Outputs `submission.csv`, ready to upload to Kaggle.
