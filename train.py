import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm
import time

os.makedirs("model", exist_ok=True)

df = pd.read_csv("data/features.csv")

FEATURES = [
    "QualiGapToPole",
    "GridPosition",
    "DriverFormLast3",
    "TeamFormLast3",
    "DNF_last3",
    "CircuitHistory",
    "StartingTyreEncoded",
    "IsWet",
    "SprintPosition",
    "TrackType",
]

X = df[FEATURES]
y = df["FinalPosition"]
groups = df["Year"].astype(str) + "_" + df["RoundNumber"].astype(str)

# Train on ALL available data for the live API
train_mask = df["Year"] <= df["Year"].max()
test_mask = df["Year"] == df["Year"].max() # Just used to print the final evaluation metrics

X_train = X[train_mask]
X_test = X[test_mask]
y_train = y[train_mask]
y_test = y[test_mask]

print(f"\nDataset ready")
print(f"Train: {len(X_train)} rows ({df[train_mask]['EventName'].nunique()} races, 2022-2024)")
print(f"Test:  {len(X_test)} rows ({df[test_mask]['EventName'].nunique()} races, 2025 only)\n")

# ── PROGRESS BAR ──────────────────────────────────────────────
N_ESTIMATORS = 2000
pbar = tqdm(
    total=N_ESTIMATORS,
    desc="Training",
    unit="tree",
    ncols=80,
    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} trees [{elapsed}<{remaining}, {rate_fmt}]"
)

class ProgressCallback(xgb.callback.TrainingCallback):
    def after_iteration(self, model, epoch, evals_log):
        pbar.update(1)
        if (epoch + 1) % 50 == 0 and evals_log:
            try:
                mae = list(evals_log["validation_0"]["mae"])[-1]
                pbar.set_postfix({"MAE": f"{mae:.3f}"})
            except Exception:
                pass
        return False

start = time.time()

model = xgb.XGBRegressor(
    n_estimators=2000,
    learning_rate=0.01,
    max_depth=4,
    subsample=0.7,
    colsample_bytree=0.7,
    min_child_weight=3,
    gamma=0.1,
    early_stopping_rounds=50,    # moved here
    random_state=42,
    n_jobs=-1,
    eval_metric="mae",
    callbacks=[ProgressCallback()]
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

pbar.close()
print(f"\nTraining complete in {time.time() - start:.1f}s")

# ── EVALUATE ──────────────────────────────────────────────────
print("\nEvaluating...")

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

test_df = df[test_mask].copy()
test_df["PredictedPosition"] = y_pred
test_df["PredictedRank"] = test_df.groupby(["Year", "RoundNumber"])["PredictedPosition"].rank()

podium_correct = 0
winner_correct = 0
total_races = 0

race_groups = list(test_df.groupby(["Year", "RoundNumber"]))

for (year, rnd), group in tqdm(race_groups, desc="Scoring races", ncols=80):
    actual_podium = set(group[group["FinalPosition"] <= 3]["Abbreviation"])
    predicted_podium = set(group[group["PredictedRank"] <= 3]["Abbreviation"])
    podium_correct += len(actual_podium & predicted_podium)

    actual_winner = group[group["FinalPosition"] == 1]["Abbreviation"].values
    predicted_winner = group[group["PredictedRank"] == 1]["Abbreviation"].values
    if len(actual_winner) > 0 and len(predicted_winner) > 0:
        if actual_winner[0] == predicted_winner[0]:
            winner_correct += 1
    total_races += 1

podium_accuracy = podium_correct / (total_races * 3) * 100
winner_accuracy = winner_correct / total_races * 100

print(f"\n{'='*40}")
print(f"  MAE:              {mae:.2f} positions")
print(f"  Podium accuracy:  {podium_accuracy:.1f}%")
print(f"  Winner accuracy:  {winner_accuracy:.1f}%")
print(f"  Races evaluated:  {total_races}")
print(f"{'='*40}\n")

# ── FEATURE IMPORTANCE ────────────────────────────────────────
importance = pd.Series(model.feature_importances_, index=FEATURES)
importance = importance.sort_values(ascending=False)
print("Feature importance:")
for feat, score in importance.items():
    bar = "█" * int(score * 200)
    print(f"  {feat:<22} {bar} {score:.4f}")

# ── SAVE ──────────────────────────────────────────────────────
model.save_model("model/model.json")
print("\nModel saved to model/model.json")