import fastf1
import pandas as pd
import time

fastf1.Cache.enable_cache("data/cache")

df = pd.read_csv("data/raw_races.csv")

print(f"Sprint quali nulls before: {df['SprintQualiTime'].isna().sum()}")

races = df[["Year", "RoundNumber", "EventName"]].drop_duplicates()

for _, race in races.iterrows():
    year = int(race["Year"])
    event = race["EventName"]
    rnd = int(race["RoundNumber"])

    sq_results = None
    for sq_name in ["SQ", "SS", "Sprint Qualifying", "Sprint Shootout"]:
        try:
            session_sq = fastf1.get_session(year, event, sq_name)
            session_sq.load(telemetry=False, weather=False, messages=True)
            time.sleep(1)

            if "Q3" in session_sq.results.columns and session_sq.results["Q3"].notna().any():
                sq = session_sq.results[["Abbreviation", "Q3"]].copy()
                sq = sq.rename(columns={"Q3": "SprintQualiTime"})
            elif "Q1" in session_sq.results.columns and session_sq.results["Q1"].notna().any():
                sq = session_sq.results[["Abbreviation", "Q1"]].copy()
                sq = sq.rename(columns={"Q1": "SprintQualiTime"})
            else:
                print(f"  [{year}] {event} — Sprint Quali found ({sq_name}) but no times")
                continue

            sq_results = sq
            print(f"  [{year}] {event} — Sprint Quali found ({sq_name})")
            break
        except Exception:
            continue

    if sq_results is None:
        continue

    for _, row in sq_results.iterrows():
        mask = (
            (df["Year"] == year) &
            (df["RoundNumber"] == rnd) &
            (df["Abbreviation"] == row["Abbreviation"])
        )
        df.loc[mask, "SprintQualiTime"] = row["SprintQualiTime"]

df.to_csv("data/raw_races.csv", index=False)
print(f"Sprint quali nulls after: {df['SprintQualiTime'].isna().sum()}")
print("Done.")