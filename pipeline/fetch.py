# pipeline/fetch.py  — replace your entire existing file with this

import fastf1
import pandas as pd
import os
import time

fastf1.Cache.enable_cache("data/cache")
os.makedirs("data/cache", exist_ok=True)
os.makedirs("data", exist_ok=True)

# ── Load what we already have ──────────────────────────────────
RAW_CSV = "data/raw_races.csv"

if os.path.exists(RAW_CSV):
    existing = pd.read_csv(RAW_CSV)
    # build a set of (year, round) already fetched
    fetched = set(
        zip(existing["Year"].dropna().astype(int), existing["RoundNumber"].dropna().astype(int))
    )
    print(f"Already have {len(fetched)} rounds in CSV")
else:
    existing = pd.DataFrame()
    fetched  = set()
    print("No existing CSV — full fetch")

# ── Determine years to scan ────────────────────────────────────
from datetime import datetime
current_year = datetime.now().year
years = list(range(2022, current_year + 1))

new_rows = []

for year in years:
    try:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
    except Exception as e:
        print(f"Failed schedule {year}: {e}")
        continue

    for _, race in schedule.iterrows():
        rnd   = int(race["RoundNumber"])
        event = race["EventName"]
        date  = pd.to_datetime(race["EventDate"], utc=True)

        if rnd == 0:
            continue

        # skip if already fetched
        if (year, rnd) in fetched:
            print(f"[{year}] R{rnd} {event} — already in CSV, skipping")
            continue

        # skip if race hasn't happened yet (future rounds)
        if date > pd.Timestamp.now(tz="UTC"):
            print(f"[{year}] R{rnd} {event} — future race, skipping")
            continue

        print(f"\n[{year}] R{rnd}: {event}")

        # ── Qualifying ─────────────────────────────────────────
        try:
            sq = fastf1.get_session(year, rnd, "Q")
            sq.load(telemetry=False, weather=True, messages=False)
            time.sleep(1)

            q = sq.results[["Abbreviation", "TeamName", "Q1", "Q2", "Q3", "Position"]].copy()
            q = q.rename(columns={"Position": "GridPosition"})

            def best_time(row):
                for col in ["Q3", "Q2", "Q1"]:
                    if pd.notna(row[col]):
                        return row[col]
                return pd.NaT

            q["BestQualiTime"] = q.apply(best_time, axis=1)
            pole_time = q[q["GridPosition"] == 1]["BestQualiTime"].values
            pole_time = pole_time[0] if len(pole_time) else pd.NaT

            def gap_to_pole(row):
                t = row["BestQualiTime"]
                if pd.isna(t) or pd.isna(pole_time):
                    return None
                return (t - pole_time).total_seconds()

            q["QualiGapToPole"] = q.apply(gap_to_pole, axis=1)

        except Exception as e:
            print(f"  Quali failed: {e} — skipping round")
            continue

        # ── Sprint (optional) ──────────────────────────────────
        sprint_pos = {}
        try:
            ss = fastf1.get_session(year, rnd, "S")
            ss.load(telemetry=False, weather=False, messages=False)
            time.sleep(1)
            for _, row in ss.results.iterrows():
                sprint_pos[row["Abbreviation"]] = int(row["Position"]) if pd.notna(row["Position"]) else 0
            print(f"  Sprint data: {len(sprint_pos)} drivers")
        except Exception:
            pass  # no sprint this round

        # ── Race ───────────────────────────────────────────────
        try:
            sr = fastf1.get_session(year, rnd, "R")
            sr.load(telemetry=False, weather=True, messages=False)
            time.sleep(1)

            weather   = sr.weather_data
            is_wet    = int(weather["Rainfall"].mean() > 0.1) if weather is not None and not weather.empty else 0

            for _, row in sr.results.iterrows():
                abbr = row["Abbreviation"]
                team = row["TeamName"]

                # grid position from quali if not in race results
                grid = q[q["Abbreviation"] == abbr]["GridPosition"].values
                grid = int(grid[0]) if len(grid) else 20

                gap = q[q["Abbreviation"] == abbr]["QualiGapToPole"].values
                gap = float(gap[0]) if len(gap) and pd.notna(gap[0]) else 0.0

                status = str(row.get("Status", ""))
                dnf    = 0 if status == "Finished" or status.startswith("+") else 1

                try:
                    pos = int(row["Position"])
                except (ValueError, TypeError):
                    pos = 20

                new_rows.append({
                    "Year":            year,
                    "RoundNumber":     rnd,
                    "EventName":       event,
                    "Abbreviation":    abbr,
                    "TeamName":        team,
                    "GridPosition":    grid,
                    "QualiGapToPole":  gap,
                    "FinalPosition":   pos,
                    "DNF":             dnf,
                    "IsWet":           is_wet,
                    "SprintPosition":  sprint_pos.get(abbr, 0),
                })
                
            print(f"  Done: {len(sr.results)} drivers")

        except Exception as e:
            print(f"  Race failed: {e}")
            continue

# ── Append only new rows ───────────────────────────────────────
if new_rows:
    new_df = pd.DataFrame(new_rows)
    combined = pd.concat([existing, new_df], ignore_index=True) if not existing.empty else new_df
    combined.to_csv(RAW_CSV, index=False)
    print(f"\nAppended {len(new_rows)} new rows — total {len(combined)}")
else:
    print("\nNothing new to fetch")