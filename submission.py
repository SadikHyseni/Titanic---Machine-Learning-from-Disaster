
"""
=====================================================================
TITANIC - MACHINE LEARNING FROM DISASTER
A complete, heavily-commented pipeline for learning how this works.
=====================================================================
 
THE GOAL, IN ONE SENTENCE:
Learn a pattern from 891 passengers whose fate we KNOW (train.csv),
then apply that pattern to predict the fate of 418 passengers whose
fate we DON'T know (test.csv).
 
Run with:  python submission.py
Requires: train.csv and test.csv sitting in the SAME FOLDER as this
          script (this script uses relative paths, e.g. 'train.csv',
          not full paths).
 
WHAT YOU NEED INSTALLED (only 3 libraries):
    pip install pandas numpy scikit-learn
"""
 


import pandas as pd
import numpy as np
import warnings

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)

warnings.filterwarnings('ignore')

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print("=" * 60)
print("Step 1: Load Data")
print("=" * 60)

print(f"Train shape: {train.shape} Test shape: {test.shape}")


"Explore the data"

print("\n" + "=" * 60)
print("STEP 2: EXPLORE")
print("=" * 60)

print("Overall survival rate:", round(train['Survived'].mean(), 3))

print("Survival rate by Sex:\n", train.groupby('Sex')['Survived'].mean())

print("Survival rate by Pclass: \n", train.groupby('Pclass')['Survived'].mean())

missing = train.isnull().sum()
print("\nMissing values in train: \n", missing[missing > 0])

print("\n" + "=" * 60)
print("STEP 3: CLEAN")
print("=" * 60)

# Flag each row's origin so train/test can be separated again after preprocessing
train['is_train'] = 1
test['is_train'] = 0

test['Survived'] = np.nan # placeholder so both frames share the same columns

# Combine train and test to apply identical preprocessing to both
full = pd.concat([train, test], sort=False).reset_index(drop=True)

# Extract title (Mr/Mrs/Miss/etc.) from the Name field, e.g. "Braund, Mr. Owen Harris" -> Mr
full['Title'] = full['Name'].str.extract(r',\s*([^\.]+)\.')

# Consolidate rare/alternate titles into standard categories
title_map = {
    'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs',     
    'Lady': 'Rare', 'Countess': 'Rare', 'Capt': 'Rare', 'Col': 'Rare',
    'Don': 'Rare', 'Dr': 'Rare', 'Major': 'Rare', 'Rev': 'Rare',
    'Sir': 'Rare', 'Jonkheer': 'Rare', 'Dona': 'Rare'  
}

full['Title'] = full['Title'].replace(title_map) 
full.loc[~full['Title'].isin(['Mr', 'Mrs', 'Miss', 'Master', 'Rare']), 'Title'] = 'Rare'

# Derive family size and solo-traveler flag from SibSp/Parch
full['FamilySize'] = full['SibSp'] + full['Parch'] + 1 
full['IsAlone'] = (full['FamilySize'] == 1).astype(int)

# Impute Age using the median within the same Title/Pclass group
full['Age'] = full.groupby(['Title', 'Pclass'])['Age'].transform(lambda x: x.fillna(x.median()))
full['Age'] = full['Age'].fillna(full['Age'].median())  # fallback for any group with no data

# Impute Fare using the median for that passenger's class
full['Fare'] = full.groupby('Pclass')['Fare'].transform(lambda x: x.fillna(x.median()))
# Impute Embarked with the most frequent value
full['Embarked'] = full['Embarked'].fillna(full['Embarked'].mode()[0])

# Cabin has too many missing values to use directly; reduce to a presence flag
full['HasCabin'] = full['Cabin'].notnull().astype(int)

# Bucket Age and Fare into bands to help tree-based models split more effectively
full['AgeBand'] = pd.cut(full['Age'], bins=[0, 12, 18, 35, 60, 100], labels=[0, 1, 2, 3, 4]).astype(int)
full['FareBand'] = pd.qcut(full['Fare'], 4, labels=[0, 1, 2, 3]).astype(int)

# Encode categorical variables
full['Sex'] = full['Sex'].map({'male': 0, 'female': 1})
full = pd.get_dummies(full, columns=['Embarked', 'Title'], drop_first=True)

# Final feature set used for training
feature_cols = [
    'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize', 'IsAlone', 'HasCabin', 'AgeBand', 'FareBand'] + [c for c in full.columns if c.startswith('Embarked_') or c.startswith('Title_')]

print (f"Final feature set ({len(feature_cols)} features):" , feature_cols)

print("Remaining missing values: \n", full[feature_cols].isnull().sum().sum())

# Split back into train/test now that preprocessing is complete
train_processed = full[full['is_train'] == 1]
test_processed = full[full['is_train'] == 0]

X = train_processed[feature_cols]
y = train_processed['Survived'].astype(int)
X_test = test_processed[feature_cols]

print("\n" + "=" * 60)
print("STEP 4: COMPARE MODELS")
print("=" * 60)

# 5-fold stratified cross-validation for an unbiased accuracy estimate
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=300, max_depth=4, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42), 
    'Hist Gradient Boosting': HistGradientBoostingClassifier(max_iter=100, max_depth=None, learning_rate=0.05, l2_regularization=0.1, random_state=42),
}

results = {}
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    results[name] = scores.mean()
    print(f"{name:<25}{scores.mean():.4f}")

best_model_name = max(results, key=results.get)
print(f"\nBest model: {best_model_name} ({results[best_model_name]:.4f})")

print("\n" + "=" * 60)
print("STEP 5: TRAIN FINAL MODEL + PREDICT")
print("=" * 60)

final_model = models[best_model_name]

final_model.fit(X, y) # retrain on full training set for the final model

predictions = final_model.predict(X_test).astype(int)

submission = pd.DataFrame({
    'PassengerId': test_processed['PassengerId'].astype(int),
    'Survived': predictions
})

submission.to_csv('submission.csv', index=False)
 
print(f"submission.csv written: {submission.shape[0]} rows")
print(submission.head())  
print(f"\nPredicted survival rate: {predictions.mean():.3f}")