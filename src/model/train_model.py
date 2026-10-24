import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score, classification_report

# --- Load engineered features ---
df = pd.read_csv("data/processed/model_features.csv")

# Keep only repos with a defined target label (needs >=3 days tracked — all 10 should qualify now)
df = df.dropna(subset=["still_trending"])
print(f"Repos with a usable label: {len(df)} / 10")
print(df[["full_name", "n_days_tracked", "buzz_trend_slope", "still_trending"]])

if len(df) < 4:
    print("[NOTICE] Not enough labeled repos yet to train meaningfully. Exiting.")
    exit()

# --- Feature selection (buzz_trend_slope EXCLUDED on purpose — it directly determines the label: data leakage) ---
FEATURES = ["latest_stars", "latest_buzz_score", "avg_buzz_score",
            "avg_hn_mentions", "star_growth_rate", "buzz_score_delta"]
X = df[FEATURES].fillna(0).values
y = df["still_trending"].astype(int).values

print(f"\nClass balance -> trending(1): {sum(y)}, not trending(0): {len(y)-sum(y)}")

from sklearn.preprocessing import StandardScaler

# --- Leave-One-Out Cross-Validation, with per-fold scaling + a majority-class baseline ---
loo = LeaveOneOut()
preds, actuals, baseline_preds, skipped = [], [], [], []

for train_idx, test_idx in loo.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    if len(set(y_train)) < 2:
        skipped.append(df.iloc[test_idx[0]]["full_name"])
        continue

    # Scale using ONLY the training fold's statistics — test repo must stay unseen
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)
    pred = model.predict(X_test_scaled)[0]

    # Majority-class baseline: what would "always guess the common label" score?
    majority_label = pd.Series(y_train).mode()[0]

    preds.append(pred)
    actuals.append(y_test[0])
    baseline_preds.append(majority_label)

if skipped:
    print(f"\n[NOTICE] {len(skipped)} fold(s) skipped — training set had only one class "
          f"after holding out: {skipped}")

model_acc = accuracy_score(actuals, preds)
baseline_acc = accuracy_score(actuals, baseline_preds)
print(f"\nLeave-One-Out CV Accuracy (over {len(preds)}/{len(y)} evaluable folds): {model_acc:.2%}")
print(f"Majority-class baseline accuracy (same folds):                        {baseline_acc:.2%}")
if model_acc <= baseline_acc:
    print("[HONEST FINDING] The trained model does NOT beat the naive baseline on this data.")

print("\nPer-repo prediction vs actual (evaluable folds only):")
evaluated_names = [n for n in df["full_name"] if n not in skipped]
for name, p, a in zip(evaluated_names, preds, actuals):
    mark = "✅" if p == a else "❌"
    print(f"  {mark} {name:30s} predicted={p} actual={a}")

# --- Final model trained on ALL data, WITH scaling, for coefficient inspection ---
final_scaler = StandardScaler()
X_scaled_full = final_scaler.fit_transform(X)
final_model = LogisticRegression(max_iter=1000)
final_model.fit(X_scaled_full, y)
print("\nFeature coefficients (final model, standardized features, trained on all repos):")
for feat, coef in zip(FEATURES, final_model.coef_[0]):
    direction = "higher -> more likely trending" if coef > 0 else "higher -> less likely trending"
    print(f"  {feat:20s} {coef:+.4f}  ({direction})")