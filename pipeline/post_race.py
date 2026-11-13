# F1/pipeline/post_race.py
# Triggered automatically after race end is detected by the scheduler.
# 1. Fetches full race results from FastF1
# 2. Builds feature rows (re-uses features.py logic)
# 3. Appends to raw_races.csv and features.csv
# 4. Runs model against new data and logs accuracy
# 5. Optionally retrains (manual trigger only by default)

import os
import logging
import json
from datetime import datetime, timezone
from typing import Optional

import fastf1
import numpy as np
import pandas as pd
import xgboost as xgb

log = logging.getLogger("f1_api.post_race")

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_CSV       = os.path.join(BASE_DIR, "data", "raw_races.csv")
FEATURES_CSV  = os.path.join(BASE_DIR, "data", "features.csv")
MODEL_PATH    = os.path.join(BASE_DIR, "model", "model.json")
ACCURACY_LOG  = os.path.join(BASE_DIR, "data", "accuracy_log.json")
CACHE_DIR     = os.path.join(BASE_DIR, "data", "cache")

TYRE_MAP  = {"SOFT": 0, "MEDIUM": 1, "HARD": 2, "INTER": 3, "WET": 4}
TRACK_MAP = {"street": 0, "permanent": 1, "mixed": 2}

FEATURE_COLS = [
    "QualiGapToPole", "GridPosition", "DriverFormLast3", "TeamFormLast3",
    "DNF_last3", "CircuitHistory", "StartingTyreEncoded", "IsWet",
    "SprintPosition", "TrackType",
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_accuracy_log() -> list:
    if os.path.exists(ACCURACY_LOG):
        with open(ACCURACY_LOG) as f:
            return json.load(f)
    return []


def _save_accuracy_log(log_data: list) -> None:
    with open(ACCURACY_LOG, "w") as f:
        json.dump(log_data, f, indent=2)


def _track_type_for_circuit(circuit: str) -> str:
    # Extend this as needed
    street_circuits = {"monaco", "baku", "singapore", "jeddah", "las vegas", "miami"}
    c = circuit.lower()
    for s in street_circuits:
        if s in c:
            return "street"
    return "permanent"


def _compute_driver_form(raw_df: pd.DataFrame, driver: str, race_idx: int, n: int = 3) -> float:
    prior = raw_df[(raw_df["Driver"] == driver) & (raw_df.index < race_idx)]
    if prior.empty:
        return 10.0  # fallback neutral value
    return prior["Position"].tail(n).mean()


def _compute_team_form(raw_df: pd.DataFrame, team: str, race_idx: int, n: int = 3) -> float:
    prior = raw_df[(raw_df["Team"] == team) & (raw_df.index < race_idx)]
    if prior.empty:
        return 10.0
    return prior["Position"].tail(n).mean()


def _compute_dnf_last3(raw_df: pd.DataFrame, driver: str, race_idx: int) -> int:
    prior = raw_df[(raw_df["Driver"] == driver) & (raw_df.index < race_idx)]
    if prior.empty:
        return 0
    return int(prior["DNF"].tail(3).sum())


def _compute_circuit_history(raw_df: pd.DataFrame, driver: str, circuit: str) -> float:
    prior = raw_df[(raw_df["Driver"] == driver) & (raw_df["Circuit"] == circuit)]
    if prior.empty:
        return 10.0
    return prior["Position"].mean()


# ── Main post-race function ────────────────────────────────────────────────────

def run_post_race(
    year: int,
    round_num: int,
    event_name: str,
    prior_predict_request: Optional[dict] = None,
    retrain: bool = False,
) -> dict:
    """
    Full post-race pipeline. Returns a summary dict with accuracy metrics.
    prior_predict_request: the raw /predict request dict stored in state,
    used to recover quali gap, tyre, sprint pos, etc. that FastF1 doesn't store.
    """
    log.info("Post-race pipeline starting: %s %d round %d", event_name, year, round_num)

    # ── 1. Fetch session ───────────────────────────────────────────────────────
    session = fastf1.get_session(year, round_num, "R")
    session.load(laps=True, telemetry=False, weather=True, messages=False)

    results   = session.results   # DataFrame: Abbreviation, TeamName, GridPosition, Position, Status
    laps      = session.laps
    weather   = session.weather_data

    if results is None or results.empty:
        log.error("No results available for %s", event_name)
        return {"error": "No results available"}

    is_wet = 0
    if weather is not None and not weather.empty:
        is_wet = int(weather["Rainfall"].mean() > 0.1)

    circuit   = event_name
    track_type = _track_type_for_circuit(circuit)

    # ── 2. Load existing raw data ──────────────────────────────────────────────
    raw_df = pd.read_csv(RAW_CSV) if os.path.exists(RAW_CSV) else pd.DataFrame()

    # ── 3. Build new rows for raw_races.csv ───────────────────────────────────
    # Recover quali data from the stored predict request if available
    quali_map = {}   # driver abbr -> DriverInput dict
    if prior_predict_request:
        for d in prior_predict_request.get("drivers", []):
            quali_map[d["driver"]] = d

    new_raw_rows = []
    for _, row in results.iterrows():
        abbr   = str(row.get("Abbreviation", ""))
        team   = str(row.get("TeamName", ""))
        status = str(row.get("Status", ""))
        dnf    = 0 if status == "Finished" or status.startswith("+") else 1

        # Final position: FastF1 gives float, cast to int; DNFs get position 20+
        try:
            pos = int(row["Position"])
        except (ValueError, TypeError):
            pos = 20

        # Starting tyre: from stored quali data or default MEDIUM
        starting_tyre = "MEDIUM"
        if abbr in quali_map:
            starting_tyre = quali_map[abbr].get("starting_tyre", "MEDIUM")

        # Sprint position
        sprint_pos = 0
        if abbr in quali_map:
            sprint_pos = quali_map[abbr].get("sprint_position", 0)

        # Quali gap to pole
        quali_gap = 0.0
        if abbr in quali_map:
            quali_gap = quali_map[abbr].get("quali_gap_to_pole", 0.0)

        new_raw_rows.append({
            "Season":        year,
            "Round":         round_num,
            "Circuit":       circuit,
            "Driver":        abbr,
            "Team":          team,
            "GridPosition":  int(row.get("GridPosition", 20)),
            "Position":      pos,
            "DNF":           dnf,
            "IsWet":         is_wet,
            "TrackType":     track_type,
            "StartingTyre":  starting_tyre,
            "SprintPosition": sprint_pos,
            "QualiGapToPole": quali_gap,
        })

    new_raw_df = pd.DataFrame(new_raw_rows)

    # Append to raw_races.csv
    if raw_df.empty:
        combined_raw = new_raw_df
    else:
        combined_raw = pd.concat([raw_df, new_raw_df], ignore_index=True)

    combined_raw.to_csv(RAW_CSV, index=False)
    log.info("Appended %d rows to raw_races.csv (total %d)", len(new_raw_rows), len(combined_raw))

    # ── 4. Build feature rows ──────────────────────────────────────────────────
    # We use the combined raw df (which now includes historical rows)
    # to compute rolling features for each driver in this new race.
    # Index the new rows by their position in combined_raw.

    new_feature_rows = []
    new_race_start_idx = len(combined_raw) - len(new_raw_rows)

    for local_i, raw_row in new_raw_df.iterrows():
        global_idx = new_race_start_idx + local_i
        driver     = raw_row["Driver"]
        team       = raw_row["Team"]

        feat = {
            "Season":              year,
            "Round":               round_num,
            "Circuit":             circuit,
            "Driver":              driver,
            "Team":                team,
            "FinalPosition":       raw_row["Position"],   # label
            "QualiGapToPole":      raw_row["QualiGapToPole"],
            "GridPosition":        raw_row["GridPosition"],
            "DriverFormLast3":     _compute_driver_form(combined_raw, driver, global_idx),
            "TeamFormLast3":       _compute_team_form(combined_raw, team, global_idx),
            "DNF_last3":           _compute_dnf_last3(combined_raw, driver, global_idx),
            "CircuitHistory":      _compute_circuit_history(combined_raw, driver, circuit),
            "StartingTyreEncoded": TYRE_MAP.get(raw_row["StartingTyre"].upper(), 1),
            "IsWet":               is_wet,
            "SprintPosition":      raw_row["SprintPosition"],
            "TrackType":           TRACK_MAP.get(track_type, 1),
        }
        new_feature_rows.append(feat)

    new_feat_df = pd.DataFrame(new_feature_rows)

    # Append to features.csv
    feat_df = pd.read_csv(FEATURES_CSV) if os.path.exists(FEATURES_CSV) else pd.DataFrame()
    combined_feat = pd.concat([feat_df, new_feat_df], ignore_index=True) if not feat_df.empty else new_feat_df
    combined_feat.to_csv(FEATURES_CSV, index=False)
    log.info("Appended %d rows to features.csv (total %d)", len(new_feature_rows), len(combined_feat))

    # ── 5. Evaluate model on this new race ────────────────────────────────────
    model = xgb.XGBRegressor()
    model.load_model(MODEL_PATH)

    X_new = new_feat_df[FEATURE_COLS].values
    y_true = new_feat_df["FinalPosition"].values

    raw_scores  = model.predict(X_new)
    pred_ranks  = np.argsort(np.argsort(raw_scores)) + 1   # predicted positions

    mae         = float(np.mean(np.abs(pred_ranks - y_true)))

    # Winner accuracy: did P1 predicted = P1 actual?
    pred_winner  = new_feat_df["Driver"].iloc[int(np.argmin(raw_scores))]
    actual_winner = new_feat_df[new_feat_df["FinalPosition"] == 1]["Driver"]
    actual_winner = actual_winner.iloc[0] if not actual_winner.empty else ""
    winner_correct = pred_winner == actual_winner

    # Podium accuracy: intersection of predicted top 3 and actual top 3
    pred_podium   = set(new_feat_df["Driver"].iloc[np.argsort(raw_scores)[:3]])
    actual_podium = set(new_feat_df[new_feat_df["FinalPosition"] <= 3]["Driver"])
    podium_hits   = len(pred_podium & actual_podium)
    podium_acc    = podium_hits / 3.0

    log.info(
        "%s accuracy — MAE: %.2f | Winner: %s | Podium: %d/3",
        event_name, mae, winner_correct, podium_hits
    )

    # ── 6. Append to accuracy log ──────────────────────────────────────────────
    acc_entry = {
        "race":           event_name,
        "year":           year,
        "round":          round_num,
        "logged_at":      datetime.now(timezone.utc).isoformat(),
        "mae":            round(mae, 3),
        "winner_correct": winner_correct,
        "podium_hits":    podium_hits,
        "podium_acc":     round(podium_acc, 3),
        "pred_winner":    pred_winner,
        "actual_winner":  actual_winner,
        "pred_podium":    list(pred_podium),
        "actual_podium":  list(actual_podium),
    }

    acc_log = _load_accuracy_log()
    acc_log.append(acc_entry)
    _save_accuracy_log(acc_log)
    log.info("Accuracy logged to %s", ACCURACY_LOG)

    # ── 7. Optional retrain ────────────────────────────────────────────────────
    retrain_status = "skipped"
    if retrain:
        log.info("Retraining on full features.csv (%d rows)", len(combined_feat))
        X_all = combined_feat[FEATURE_COLS].values
        y_all = combined_feat["FinalPosition"].values

        new_model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )
        new_model.fit(X_all, y_all)
        new_model.save_model(MODEL_PATH)
        log.info("Model retrained and saved to %s", MODEL_PATH)
        retrain_status = "done"

    return {
        "race":           event_name,
        "mae":            round(mae, 3),
        "winner_correct": winner_correct,
        "podium_hits":    podium_hits,
        "podium_acc":     round(podium_acc, 3),
        "rows_appended":  len(new_raw_rows),
        "retrain":        retrain_status,
    }
