import pandas as pd
import numpy as np
import os

os.makedirs("data", exist_ok=True)

df = pd.read_csv("data/raw_races.csv")

# ── CLEAN UP ──────────────────────────────────────────────────
df = df.dropna(subset=["FinalPosition"])
df["GridPosition"] = df["GridPosition"].fillna(20)
df["FinalPosition"] = df["FinalPosition"].astype(int)
df = df.sort_values(["Year", "RoundNumber"]).reset_index(drop=True)
df["DNF"] = df["Status"].apply(lambda x: 0 if x == "Finished" else 1)

# ── TRACK TYPE ────────────────────────────────────────────────
# Each circuit gets a type encoding
# 0 = power/low downforce (Monza, Spa)
# 1 = balanced (most circuits)
# 2 = high downforce (Hungary, Monaco)
# 3 = street circuit (Monaco, Baku, Singapore, Miami, Las Vegas, Jeddah)
# 4 = high deg (Bahrain, Abu Dhabi, COTA)

track_type_map = {
    "Bahrain Grand Prix": 4,
    "Saudi Arabian Grand Prix": 3,
    "Australian Grand Prix": 1,
    "Japanese Grand Prix": 2,
    "Chinese Grand Prix": 1,
    "Miami Grand Prix": 3,
    "Emilia Romagna Grand Prix": 1,
    "Monaco Grand Prix": 3,
    "Spanish Grand Prix": 1,
    "Canadian Grand Prix": 1,
    "Austrian Grand Prix": 0,
    "British Grand Prix": 1,
    "Hungarian Grand Prix": 2,
    "Belgian Grand Prix": 0,
    "Dutch Grand Prix": 2,
    "Italian Grand Prix": 0,
    "Azerbaijan Grand Prix": 3,
    "Singapore Grand Prix": 3,
    "United States Grand Prix": 4,
    "Mexico City Grand Prix": 2,
    "São Paulo Grand Prix": 1,
    "Las Vegas Grand Prix": 3,
    "Qatar Grand Prix": 4,
    "Abu Dhabi Grand Prix": 4,
}

df["TrackType"] = df["EventName"].map(track_type_map).fillna(1)

# ── CHAMPIONSHIP STANDING ─────────────────────────────────────
# Compute each driver's championship position BEFORE each race
# so we never leak current race points into the feature

def compute_championship_standing(df):
    results = []

    for year in df["Year"].unique():
        year_df = df[df["Year"] == year].copy()
        rounds = sorted(year_df["RoundNumber"].unique())

        # Cumulative points per driver up to but not including current round
        for rnd in rounds:
            past = year_df[year_df["RoundNumber"] < rnd]

            if len(past) == 0:
                # First race of season — everyone starts equal
                current_round = year_df[year_df["RoundNumber"] == rnd].copy()
                current_round["DriverChampPosition"] = 10
                current_round["TeamChampPosition"] = 5
                results.append(current_round)
                continue

            # Driver standings
            driver_points = past.groupby("Abbreviation")["Points"].sum().reset_index()
            driver_points = driver_points.sort_values("Points", ascending=False)
            driver_points["DriverChampPosition"] = range(1, len(driver_points) + 1)

            # Team standings
            team_points = past.groupby("TeamName")["Points"].sum().reset_index()
            team_points = team_points.sort_values("Points", ascending=False)
            team_points["TeamChampPosition"] = range(1, len(team_points) + 1)

            # Merge into current round
            current_round = year_df[year_df["RoundNumber"] == rnd].copy()
            current_round = current_round.merge(
                driver_points[["Abbreviation", "DriverChampPosition"]],
                on="Abbreviation", how="left"
            )
            current_round = current_round.merge(
                team_points[["TeamName", "TeamChampPosition"]],
                on="TeamName", how="left"
            )

            # New drivers mid season won't have points history
            current_round["DriverChampPosition"] = current_round["DriverChampPosition"].fillna(15)
            current_round["TeamChampPosition"] = current_round["TeamChampPosition"].fillna(8)

            results.append(current_round)

    return pd.concat(results).sort_index()

print("Computing championship standings...")
df = compute_championship_standing(df)

# ── ROLLING FEATURES ──────────────────────────────────────────
def rolling_driver_form(df, n=3):
    results = []
    for driver in df["Abbreviation"].unique():
        mask = df["Abbreviation"] == driver
        driver_df = df[mask].copy()
        driver_df["DriverFormLast3"] = (
            driver_df["FinalPosition"]
            .shift(1)
            .rolling(window=n, min_periods=1)
            .mean()
        )
        results.append(driver_df)
    return pd.concat(results).sort_index()

def rolling_team_form(df, n=3):
    # 1. Get the team's average finish position for each specific race
    team_avg = df.groupby(['TeamName', 'Year', 'RoundNumber'])['FinalPosition'].mean().reset_index()
    team_avg = team_avg.sort_values(['Year', 'RoundNumber'])
    
    # 2. Shift by 1 RACE (not 1 row) and calculate the rolling mean
    team_avg['TeamFormLast3'] = team_avg.groupby('TeamName')['FinalPosition'].transform(
        lambda x: x.shift(1).rolling(window=n, min_periods=1).mean()
    )
    
    # 3. Merge this clean, leak-free feature back into the main dataframe
    df = df.merge(
        team_avg[['TeamName', 'Year', 'RoundNumber', 'TeamFormLast3']], 
        on=['TeamName', 'Year', 'RoundNumber'], 
        how='left'
    )
    return df

def rolling_dnf_rate(df, n=3):
    results = []
    for driver in df["Abbreviation"].unique():
        mask = df["Abbreviation"] == driver
        driver_df = df[mask].copy()
        driver_df["DNF_last3"] = (
            driver_df["DNF"]
            .shift(1)
            .rolling(window=n, min_periods=1)
            .sum()
        )
        results.append(driver_df)
    return pd.concat(results).sort_index()

def circuit_history(df):
    results = []
    for driver in df["Abbreviation"].unique():
        for event in df["EventName"].unique():
            mask = (df["Abbreviation"] == driver) & (df["EventName"] == event)
            sub = df[mask].copy()
            if len(sub) == 0:
                continue
            sub["CircuitHistory"] = (
                sub["FinalPosition"]
                .shift(1)
                .expanding()
                .mean()
            )
            results.append(sub)
    return pd.concat(results).sort_index()

print("Computing driver form...")
df = rolling_driver_form(df)
print("Computing team form...")
df = rolling_team_form(df)
print("Computing DNF rate...")
df = rolling_dnf_rate(df)
print("Computing circuit history...")
df = circuit_history(df)

# ── ENCODE TYRE ───────────────────────────────────────────────
tyre_map = {"SOFT": 0, "MEDIUM": 1, "HARD": 2, "INTERMEDIATE": 3, "WET": 4}
df["StartingTyreEncoded"] = df["StartingTyre"].map(tyre_map).fillna(1)

# ── FINAL FEATURE SET ─────────────────────────────────────────
features = [
    "QualiGapToPole",
    "GridPosition",
    "DriverFormLast3",
    "TeamFormLast3",
    "DNF_last3",
    "CircuitHistory",
    "StartingTyreEncoded",
    "IsWet",
    "SprintPosition",
    "DriverChampPosition",
    "TeamChampPosition",
    "TrackType",
]

meta = ["Abbreviation", "TeamName", "Year", "RoundNumber", "EventName", "FinalPosition"]

df_out = df[meta + features].copy()

# ── FILL NULLS ────────────────────────────────────────────────
df_out["QualiGapToPole"] = df_out["QualiGapToPole"].fillna(df_out["QualiGapToPole"].median())
df_out["CircuitHistory"] = df_out["CircuitHistory"].fillna(10)
df_out["SprintPosition"] = df_out["SprintPosition"].fillna(10)
df_out["DriverFormLast3"] = df_out["DriverFormLast3"].fillna(10)
df_out["TeamFormLast3"] = df_out["TeamFormLast3"].fillna(10)
df_out["DNF_last3"] = df_out["DNF_last3"].fillna(0)
df_out["DriverChampPosition"] = df_out["DriverChampPosition"].fillna(10)
df_out["TeamChampPosition"] = df_out["TeamChampPosition"].fillna(5)

print(f"\nFinal dataset: {len(df_out)} rows")
print(f"Null counts:\n{df_out.isnull().sum()}")

df_out.to_csv("data/features.csv", index=False)
print("\nSaved to data/features.csv")