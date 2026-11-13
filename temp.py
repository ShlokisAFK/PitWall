import joblib
import pandas as pd
import xgboost as xgb

model = xgb.XGBRegressor()
model.load_model("model/model.json")

df = pd.read_csv("data/features.csv")
test = df[df["Year"] == 2026].copy()

FEATURES = [
    "QualiGapToPole", "GridPosition", "DriverFormLast3",
    "TeamFormLast3", "DNF_last3", "CircuitHistory",
    "StartingTyreEncoded", "IsWet", "SprintPosition", "TrackType"
]

test["PredictedScore"] = model.predict(test[FEATURES])
test["PredictedRank"] = test.groupby(["Year","RoundNumber"])["PredictedScore"].rank()

for event, group in test.groupby("EventName"):
    print(f"\n{event}")
    group = group.sort_values("PredictedRank")
    for _, row in group.head(5).iterrows():
        marker = "✓" if row["FinalPosition"] <= 3 and row["PredictedRank"] <= 3 else " "
        print(f"  {marker} Predicted {int(row['PredictedRank'])}: {row['Abbreviation']} (Actual: {int(row['FinalPosition'])})")