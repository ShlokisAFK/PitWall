# pipeline/post_race_auto.py
# Full post-race automation: fetch -> features -> test model -> log accuracy
# Run this via cron after every race. No manual steps needed.

import os
import sys
import json
import logging
import subprocess
from datetime import datetime, timezone

import fastf1
import pandas as pd
import numpy as np
import xgboost as xgb

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger("post_race_auto")

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_CSV = os.path.join(BASE_DIR, "data", "features.csv")
MODEL_PATH   = os.path.join(BASE_DIR, "model", "model.json")
ACC_LOG      = os.path.join(BASE_DIR, "data", "accuracy_log.json")

FEATURE_COLS = [
    "QualiGapToPole", "GridPosition", "DriverFormLast3", "TeamFormLast3",
    "DNF_last3", "CircuitHistory", "StartingTyreEncoded", "IsWet",
    "SprintPosition", "TrackType",
]

def run_step(script: str) -> bool:
    log.info("Running %s ...", script)
    result = subprocess.run(
        [sys.executable, os.path.join(BASE_DIR, "pipeline", script)],
        cwd=BASE_DIR,
        capture_output=True, text=True
    )
    if result.stdout: log.info(result.stdout.strip())
    if result.stderr: log.warning(result.stderr.strip())
    if result.returncode != 0:
        log.error("%s failed with code %d", script, result.returncode)
        return False
    return True

def load_accuracy_log():
    if os.path.exists(ACC_LOG):
        with open(ACC_LOG) as f:
            return json.load(f)
    return []

def save_accuracy_log(data):
    with open(ACC_LOG, "w") as f:
        json.dump(data, f, indent=2)

def test_model_on_new_races():
    df = pd.read_csv(FEATURES_CSV)
    acc_log = load_accuracy_log()
    already_logged = {(e["year"], e["round"]) for e in acc_log}

    model = xgb.XGBRegressor()
    model.load_model(MODEL_PATH)

    current_year = datetime.now().year
    season = df[df["Year"] == current_year].copy()

    if season.empty:
        log.info("No current season data found")
        return

    new_entries = []

    for (rnd, event), group in season.groupby(["RoundNumber", "EventName"]):
        rnd = int(rnd)
        if (current_year, rnd) in already_logged:
            log.info("R%d %s already logged, skipping", rnd, event)
            continue

        if len(group) < 5:
            log.warning("R%d %s has only %d rows, skipping", rnd, event, len(group))
            continue

        X      = group[FEATURE_COLS].fillna(10).values
        y_true = group["FinalPosition"].values
        scores = model.predict(X)

        # predicted rank
        pred_ranks = np.argsort(np.argsort(scores)) + 1
        mae        = float(np.mean(np.abs(pred_ranks - y_true)))

        # winner
        pred_winner_idx  = int(np.argmin(scores))
        pred_winner      = group["Abbreviation"].iloc[pred_winner_idx]
        actual_winner_df = group[group["FinalPosition"] == 1]["Abbreviation"]
        actual_winner    = actual_winner_df.iloc[0] if not actual_winner_df.empty else "?"
        winner_correct   = pred_winner == actual_winner

        # podium
        pred_podium   = set(group["Abbreviation"].iloc[np.argsort(scores)[:3]])
        actual_podium = set(group[group["FinalPosition"] <= 3]["Abbreviation"])
        podium_hits   = len(pred_podium & actual_podium)
        podium_acc    = podium_hits / 3.0

        entry = {
            "race":           event,
            "year":           current_year,
            "round":          rnd,
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

        log.info(
            "R%d %s — MAE: %.2f | Winner: %s (%s vs %s) | Podium: %d/3",
            rnd, event, mae,
            "✓" if winner_correct else "✗",
            pred_winner, actual_winner,
            podium_hits
        )

        new_entries.append(entry)

    if new_entries:
        acc_log.extend(new_entries)
        save_accuracy_log(acc_log)
        log.info("Logged %d new race(s) to accuracy_log.json", len(new_entries))
    else:
        log.info("No new races to log")

if __name__ == "__main__":
    log.info("=== Post-race automation started ===")

    if not run_step("fetch.py"):
        sys.exit(1)

    if not run_step("features.py"):
        sys.exit(1)

    test_model_on_new_races()

    log.info("=== Done ===")
